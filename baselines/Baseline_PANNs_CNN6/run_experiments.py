#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""비교모델 실험 실행기 — 학습률 선정(9회 중 이 모델 몫 3회) -> 본실험(sup·pl·ceil × seed 1~5 = 15회).

[2026-09-06] 천장 조건(ceil)이 추가되어 본실험이 모델당 10회에서 15회가 됐다(조건표 표 0-2, 비교모델 합계 54회).

[신설 2026-09-05] Mark5_EXP model/run_experiments.py(2026-09-03 판)의 재개 규칙을 그대로 따른다.
조건의 근거는 Downloads/Mark5_비교실험_조건표_20260905.pdf 5·6절.

[변경 2026-10-25 / A안] 2026-10-22~10-25 의 Prodigy(자동 학습률, 태그 _pd)를 걷어내고 Adam 으로 되돌렸다. 학습률은 다시 고르지 않고
  Adam 시절 선정 기록(lr_selected.json, 2026-09-24 선정 1e-4)의 값을 고정해 쓴다(common/naming.adam_lr).
  --phase all 은 main 만 돌고, --phase sweep 과 --lr 은 멈춘다. run_tag 는 _lr<lr> 대신 _adam(common/naming.py).
  아래 sweep 설명은 선정 기록을 만든 때(2026-10-21까지)의 절차다.

단계
  sweep : cond=sup, **seed 0** 으로 학습률 후보 {1e-4, 5e-5, 2e-5} 를 하나씩 학습 -> val 평가.
          val Macro F1 이 가장 높은 학습률을 plots/lr_selected.json 에 적는다. test 는 보지 않는다.
  main  : 선정된 학습률로 cond ∈ {sup, pl, ceil} × seed ∈ {1..5} 를 학습 -> val 평가 -> test 평가.
          ceil = labeled 2,000 + unlabeled 2,000 에 진짜 라벨(ceiling_labels.csv) — 상한 기준선(Cances et al. 2022).
          선정에 쓴 회차(seed 0)는 본실험에 들어가지 않는다(조건표 5절 '재사용' — seed 를 아예 분리했다.
          cudnn deterministic 이라 같은 seed 를 다시 돌리면 같은 값이 나와 '새로 돈다'가 의미 없기 때문).

재개 규칙 (단계 단위)
  학습  : 체크포인트(model/<tag>.pth) 와 완주 표식(plots/loss_history_<tag>.csv) 이 둘 다 있어야 끝난 것.
  평가  : performance_summary CSV 가 존재하고 지표 6개가 전부 읽혀야 끝난 것.
  끝난 단계는 건너뛴다. 세션이 끊기면 같은 명령을 다시 실행하면 된다.

예)  python run_experiments.py --dry_run
     python run_experiments.py                       # main 전부([2026-10-25] 학습률은 선정값 고정 — --phase sweep·--lr 은 멈춘다)
     python run_experiments.py --phase main --seeds 9 --conds sup --extra "--max_clips 16 --epochs 1"
         # 동작 확인용 — train.py 가 태그에 _smoke 를 붙이므로 본실험 산출물과 겹치지 않는다
"""
import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model_info import MODEL_TAG, DISPLAY_NAME   # noqa: E402  (torch 를 물지 않는 상수)
from common.naming import run_tag as _run_tag, artifact_paths as _artifact_paths, parse_tag as _parse_tag   # noqa: E402
from common.naming import UNLABELED_SUBSAMPLE_RULE                                                          # noqa: E402
from common.naming import OPTIMIZER_NAME, ADAM_LR, adam_lr                                                  # noqa: E402  [2026-10-25 / A안]
from common.paths import index_fingerprint   # noqa: E402  (torch 를 물지 않는다)
from common.others_rule import rule_json_is_current   # [2026-10-26] 지금 체크포인트 + 판정 규칙 argmax_v1

PLOT_DIR = os.path.join(ROOT, "plots")
LOG_DIR = os.path.join(ROOT, "logFiles")
METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "ROC AUC (Macro, OvR)", "Others FPR"]
DEFAULT_LRS = [1e-4, 5e-5, 2e-5]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_CONDS = ["sup", "pl", "ceil"]     # [2026-09-06] ceil(천장) 추가. 순서대로 돈다.
SELECTED_JSON = os.path.join(PLOT_DIR, "lr_selected.json")
SMOKE = False          # --extra 에 --max_clips 가 있으면 True. 태그·선정 파일이 전부 _smoke 로 갈린다(본실험 산출물과 분리).


def selected_json():
    return os.path.join(PLOT_DIR, "lr_selected_smoke.json" if SMOKE else "lr_selected.json")


def _smoke_from_extra(extra):
    for i, a in enumerate(extra):
        if a == "--max_clips" and i + 1 < len(extra):
            try:
                return int(extra[i + 1]) > 0
            except ValueError:
                return False
    return False


DEFAULT_EPOCHS = 80


def run_tag(cond, lr, seed, labeled_n=None, unlabeled_n=None):
    return _run_tag(MODEL_TAG, cond, lr, seed, smoke=SMOKE, labeled_n=labeled_n, unlabeled_n=unlabeled_n)   # 규칙은 common/naming.py


# [신설 2026-09-17 / 9절] 논문 회차 계획 — Mark5_EXP2 의 PAPER_PLAN 과 같은 모양(labeled, cond, unlabeled|None).
#   labeled 전량 : sup · pl(+합의) · ceil(+진짜)          labeled 200·500·1,000 : sup · pl
#   labeled 200 의 데이터 효율 곡선 : pl 에 라벨 없는 클립 500 · 1,000 개 안의 합의 라벨만(2,000 은 위 l200 pl 과 같은 회차라 중복 제거)
#     [2026-10-01] 전에는 '합의 라벨 500 · 1,000 개' 였다 — common/naming.py UNLABELED_SUBSAMPLE_RULE
#   -> 조합 11개 x seed 5 = 55회 (+ 학습률 선정 3회는 phase sweep). 체크리스트 9절의 48+15=63 은 중복을 두 번 센 값.
PAPER_PLAN = [
    (None, "sup", None), (None, "pl", None), (None, "ceil", None),
    (200, "sup", None), (200, "pl", None), (500, "sup", None), (500, "pl", None), (1000, "sup", None), (1000, "pl", None),
    (200, "pl", 500), (200, "pl", 1000),
]


def artifacts(tag):
    return _artifact_paths(ROOT, tag)


# [2026-09-30 / 전문가 4차(클립 입력) 재학습 대응] 재개 판정 때 대조하는 CSV 를 조건별로 좁힌다.
#   train.py 는 전처럼 CSV 5개의 sha1 을 전부 train_meta 에 적고, 여기서는 그 조건의 학습 재료가 되는 파일만 비교한다.
#   전에는 조건과 상관없이 5개를 다 비교해서, 합의 라벨(consensus_labels.csv)만 새로 만들어도 합의 라벨을 읽지 않는
#   sup·ceil 회차까지 "인덱스가 바뀜" 으로 다시 돌았다(모델당 55회). 이제 합의 라벨만 바뀌면 pl 30회만 다시 돈다.
#     sup  : labeled·val                      (train.py 가 읽는 것)
#     pl   : labeled·val·합의 라벨 + unlabeled (합의 라벨은 unlabeled 목록으로 만든 파일 — 재료의 재료라 같이 본다)
#     ceil : labeled·val·ceiling 라벨·unlabeled (train.py 가 unlabeled 로 ceiling 라벨이 빠짐없이 덮였는지 검사한다)
#   ⚠ 조건이 읽는 CSV 가 바뀌도록 train.py 를 고치면 이 표도 같이 고친다(안 고치면 바뀐 재료로 만든 옛 결과를 완료로 본다).
# [신설 2026-10-22 / 검증 결합기] --preset cmb: 설계 문서 '비교모델 조건' — 지도만(sup)·결합기 한 라벨(plc) × 정답 전량·1,000·500·200,
#   조건마다 1회(seed 1). plc 는 Mark5.3 의 pseudo_labels_cmbone_l<n|all>_s<seed>.csv 를 직접 읽는다(common/mark53_labels.py).
CMB_PLAN = [(None, "sup", None), (1000, "sup", None), (500, "sup", None), (200, "sup", None),
            (None, "plc", None), (1000, "plc", None), (500, "plc", None), (200, "plc", None)]

_FP_FILES_BY_COND = {
    "sup": ("dataset_labeled.csv", "dataset_val.csv"),
    "pl": ("dataset_labeled.csv", "dataset_val.csv", "dataset_unlabeled.csv", "consensus_labels.csv"),
    "ceil": ("dataset_labeled.csv", "dataset_val.csv", "dataset_unlabeled.csv", "ceiling_labels.csv"),
    "plc": ("dataset_labeled.csv", "dataset_val.csv", "dataset_unlabeled.csv"),   # [2026-10-22] 결합기 파일은 아래 _train_meta_ok 가 해시로 따로 대조
}


def _train_meta_ok(tag):
    """train_meta_<tag>.json 이 있으면 조건이 본실험과 같은지 본다(축소 실행·epoch 축소·증강 off 는 미완료로).
    [2026-09-05 저녁, code-reviewer 지적] 태그가 같은 다른 조건의 산출물을 완료로 착각하지 않게."""
    meta_path = artifacts(tag)["meta"]
    if not os.path.isfile(meta_path):
        return True, ""
    try:
        with open(meta_path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return False, "train_meta 를 읽을 수 없음"
    reasons = []
    if m.get("smoke") or int(m.get("max_clips", 0) or 0) > 0:
        reasons.append("축소 실행(max_clips)")
    if int(m.get("epochs_max", DEFAULT_EPOCHS)) != DEFAULT_EPOCHS:
        reasons.append(f"epochs_max={m.get('epochs_max')}")
    if m.get("spec_augment") is False:
        reasons.append("SpecAugment off")
    # [변경 2026-10-25 / A안] _adam 태그 회차는 train_meta 의 옵티마이저·학습률 기록까지 지금 절차(Adam · adam_lr(조건))여야 완료다.
    _p_opt = _parse_tag(tag)
    if _p_opt and _p_opt["lr"] is None:
        if m.get("optimizer") != OPTIMIZER_NAME:
            reasons.append(f"옵티마이저 기록 {m.get('optimizer')!r} ≠ {OPTIMIZER_NAME}")
        elif not isinstance(m.get("lr"), (int, float)) or abs(float(m["lr"]) - adam_lr(_p_opt["cond"])) > 1e-12:
            reasons.append(f"학습률 기록 {m.get('lr')!r} ≠ {adam_lr(_p_opt['cond']):g}")
    # [2026-09-17 / 2차 검수 C-3] 데이터 판 대조. 09-06 옛 데이터 산출물(train_meta 에 index_sha1 없음)이나 인덱스가 바뀐 뒤의 산출물은
    # 완료로 치지 않는다 — 안 그러면 --force 없이 이어 돌릴 때 옛 결과가 [건너뜀] 으로 새 aggregate_raw 에 섞인다.
    saved_fp = m.get("index_sha1")
    if not isinstance(saved_fp, dict):
        reasons.append("데이터 판 기록 없음(옛 산출물)")
    else:
        # [2026-09-30] 조건별 대조(위 _FP_FILES_BY_COND). 태그에서 조건을 못 읽으면 옛 동작 그대로 기록된 파일 전부를 대조한다.
        _p = _parse_tag(tag)
        keys = _FP_FILES_BY_COND.get(_p["cond"]) if _p else None
        if keys is None:
            keys = tuple(saved_fp.keys())
        missing = [k for k in keys if k not in saved_fp]
        if missing:
            reasons.append(f"데이터 판 기록에 {', '.join(missing)} 없음")
        now_fp = index_fingerprint(tuple(k for k in keys if k in saved_fp))
        changed = [k for k in keys if k in saved_fp and saved_fp[k] != now_fp.get(k)]
        if changed:
            reasons.append(f"인덱스가 바뀜({', '.join(changed)})")
    # [2026-10-01] --unlabeled_n 회차(_u<n>)는 추출 규칙의 판까지 맞아야 완료다. 옛 규칙(합의 라벨에서 n 개)으로 만든 산출물은
    # 태그가 같아도 재료가 다르므로 다시 돈다(common/naming.py UNLABELED_SUBSAMPLE_RULE).
    _p2 = _parse_tag(tag)
    if _p2 and _p2["cond"] == "pl" and _p2["unlabeled"] is not None and m.get("unlabeled_subsample_rule") != UNLABELED_SUBSAMPLE_RULE:
        reasons.append("라벨 없는 클립 추출 규칙이 옛 판(합의 라벨에서 n 개)")
    # [2026-10-22 / plc] 읽은 결합기 파일이 지금 Mark5.3 파일과 같은 판이어야 완료다 — 교사·결합기를 다시 돌리면 옛 결과가 [건너뜀] 되지 않고 다시 돈다.
    _p3 = _parse_tag(tag)
    if _p3 and _p3["cond"] == "plc":
        from common.mark53_labels import pseudo_csv_path, sha256_file
        _f = pseudo_csv_path(_p3["labeled"], _p3["seed"])
        if not os.path.isfile(_f) or m.get("pseudo_csv_sha256") != sha256_file(_f):
            reasons.append("결합기 가짜 정답 파일이 학습 때 판과 다름(또는 없음)")
    return (not reasons), ", ".join(reasons)


def read_summary(path):
    """performance_summary CSV 앞 블록(Metric,Score)을 dict 로. 없거나 깨졌으면 None."""
    if not os.path.isfile(path):
        return None
    out = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            in_block = False
            for row in csv.reader(f):
                if not row or (row[0].startswith("#")):
                    if in_block and not row:
                        break
                    continue
                if row[0] == "Metric":
                    in_block = True
                    continue
                if in_block and len(row) >= 2:
                    # pandas 는 NaN 을 빈칸으로 쓴다(예: 평가셋에 없는 클래스가 있어 ROC AUC 가 정의되지 않을 때).
                    # 빈칸은 NaN 으로 읽고, 숫자도 빈칸도 아닌 값이 있을 때만 깨진 파일로 본다.
                    if row[1].strip() == "":
                        out[row[0]] = float("nan")
                        continue
                    try:
                        out[row[0]] = float(row[1])
                    except ValueError:
                        # [2026-09-07] reselect_others_threshold.py 가 붙이는 "Others Rule"(direct/argmax) 같은
                        # 문자열 행은 지표가 아니다. 그 행만 건너뛰고, 그 밖의 문자열은 깨진 파일로 본다.
                        # [수정 2026-09-22 / 검수] common/metrics.py 가 A(클래스 편향) 뒤로 "Class Bias" 문자열 행도 쓴다.
                        # 이 행을 몰라서 None 을 돌려주면 모든 새 요약이 '불완전'이 되어 val·test 를 매번 다시 돌리고
                        # select_lr·aggregate_baseline 이 회차를 전부 빠뜨린다(검수에서 잡힘).
                        if row[0].strip() in ("Others Rule", "Class Bias"):
                            continue
                        return None
    except Exception:
        return None
    return out


def summary_is_complete(path):
    s = read_summary(path)
    return s is not None and all(m in s for m in METRICS)


def needs(tag, force, skip_test=False):
    a = artifacts(tag)
    trained = os.path.exists(a["ckpt"]) and os.path.exists(a["loss"])
    meta_ok, why = _train_meta_ok(tag)
    if trained and not meta_ok:
        print(f"  [WARN] {tag}: 기존 산출물이 본실험 조건이 아닙니다({why}) — 학습부터 다시 돕니다.")
        trained = False
    # [2026-09-17 / 검수 P1] val 은 요약 CSV 뿐 아니라 others 규칙 json(a["rule"])도 있어야 완료다 — 옛 산출물(json 없음)에서
    # 이어 돌릴 때 test 가 "json 이 없습니다" 로 멈추지 않게 val 을 다시 돈다.
    # [2차 검수 P-1] val 을 다시 돌리면(json 없음 등) test 도 같이 다시 돈다 — 한 회차 안에서 val 규칙과 test 판정이 갈리지 않게.
    # [2026-09-18 / 3차 검수 C-2] val 완료 = 요약 완전 + others 규칙 json 이 **지금 체크포인트**로 고른 것(run_tag·bytes·mtime 대조).
    # 학습만 새로 끝나고 val 이 안 돈 회차에서 옛 val·test 요약(+ checkpoint null 인 옛 json)이 완료로 통과하던 구멍을 막는다.
    need_train = force or not trained
    rule_ok = os.path.exists(a.get("rule", "")) and rule_json_is_current(a["rule"], tag, a["ckpt"])
    need_val = need_train or not summary_is_complete(a["val"]) or not rule_ok
    need_test = (not skip_test) and (need_val or not summary_is_complete(a["test"]))
    return need_train, need_val, need_test


def _clear_eval_outputs(tag):
    """[2026-09-18 / 3차 검수 C-1·C-2] 학습을 새로 시작하기 전에 그 태그의 val·test·규칙 json·예측 CSV·그림을 지운다.
    지우지 않으면 학습·val 이 실패했을 때 옛 요약이 남아 select_lr 이 옛 val F1 로 학습률을 고르거나 집계에 옛 test 숫자가 들어간다.
    (train_loop.fit 은 ckpt·loss CSV 만 지운다.)"""
    a = artifacts(tag)
    victims = [a["val"], a["test"], a["rule"]]
    victims += glob.glob(os.path.join(PLOT_DIR, f"prediction_details_{tag}*.csv"))
    victims += glob.glob(os.path.join(PLOT_DIR, f"performance_summary_{tag}_eval_*.csv"))
    victims += glob.glob(os.path.join(PLOT_DIR, f"confusion_matrix_{tag}*.*")) + glob.glob(os.path.join(PLOT_DIR, f"roc_curve_{tag}*.png"))
    n = 0
    for v in victims:
        if os.path.isfile(v):
            os.remove(v); n += 1
    if n:
        print(f"  [INFO] {tag}: 옛 평가 산출물 {n}개 삭제(학습부터 다시 도는 회차)")


def _warn_nan(path):
    s = read_summary(path)
    if s:
        nan_keys = [k for k in METRICS if k in s and s[k] != s[k]]
        if nan_keys:
            print(f"  [WARN] {os.path.basename(path)}: 값이 NaN 인 지표 {nan_keys} — 평가셋에 없는 클래스가 있었는지 확인")


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def _sh(cmd, log_path=None, timeout_sec=None):
    """한 단계 실행. 자식 출력을 한 줄씩 받아 화면·마스터 로그·단계 로그에 같이 남긴다."""
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    t0 = time.time()
    child_log = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else None
    if child_log:
        child_log.write("\n" + "=" * 70 + f"\n# {' '.join(cmd)}\n# 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    proc, killer, timed_out = None, None, []
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1, encoding="utf-8", errors="replace")
        if timeout_sec:
            def _kill():
                timed_out.append(True)
                try:
                    proc.kill()
                except Exception:
                    pass
            killer = threading.Timer(timeout_sec, _kill)
            killer.daemon = True
            killer.start()
        for line in proc.stdout:
            sys.stdout.write(line)
            if child_log:
                child_log.write(line)
        rc = proc.wait()
    except KeyboardInterrupt:
        if proc is not None:
            proc.kill()
        print("\n[INTERRUPT] 사용자 중단 — 자식 프로세스를 정리했습니다.")
        raise
    finally:
        if killer:
            killer.cancel()
        if child_log:
            child_log.close()
    dt = time.time() - t0
    if rc == 0:
        print(f"[OK] {dt / 60:.1f}분")
        return True
    if timed_out:
        print(f"[FAIL] 시간 제한({timeout_sec / 60:.0f}분) 초과로 중단")
    else:
        print(f"[FAIL] 종료코드 {rc} ({dt / 60:.1f}분)")
    return False


def _py(script, *args):
    return [sys.executable, "-u", os.path.join(ROOT, script), *[str(a) for a in args]]


EVAL_ONLY = False   # [2026-10-26] --eval_only 이면 main 이 True 로 둔다


def run_combo(cond, lr, seed, *, force, skip_test, extra, data_root, timeout, dry_run, labeled_n=None, unlabeled_n=None):
    tag = run_tag(cond, lr, seed, labeled_n, unlabeled_n)
    need_train, need_val, need_test = needs(tag, force, skip_test)
    steps = [s for s, n in (("학습", need_train), ("val", need_val), ("test", need_test)) if n]
    _ln = f"l{labeled_n:<5}" if labeled_n is not None else "lall  "
    _un = f"u{unlabeled_n:<5}" if unlabeled_n is not None else "      "
    _lr_txt = f"{adam_lr(cond):g}" if lr is None else f"{lr:g}"   # [2026-10-25 / A안] lr=None = 지금 절차 회차(선정값 고정)
    if not steps:
        print(f"  [건너뜀] {cond:<3} lr {_lr_txt:<7} {_ln} {_un} seed {seed}   {tag}")
        return True
    print(f"  [실행]   {cond:<3} lr {_lr_txt:<7} {_ln} {_un} seed {seed}   {tag}   ({'·'.join(steps)})")
    if need_train and EVAL_ONLY:
        # [2026-10-26] --eval_only: 평가만 다시 하는 실행(노트북 5)에서 학습이 필요하다고 판정되면 학습하지 않고 이 회차를 실패로 둔다.
        print(f"  [FAIL] --eval_only 인데 학습이 필요하다고 판정됐습니다(체크포인트·loss_history·train_meta 확인) — 학습하지 않습니다: {tag}")
        return False
    if dry_run:
        return True
    common = ["--cond", cond, "--seed", seed] + ([] if lr is None else ["--lr", f"{lr:g}"])
    if labeled_n is not None:
        common += ["--labeled_n", labeled_n]
    if unlabeled_n is not None:
        common += ["--unlabeled_n", unlabeled_n]
    if data_root:
        common += ["--data_root", data_root]
    ok = True
    if need_train:
        _clear_eval_outputs(tag)
        ok = _sh(_py("train.py", *common, *extra), os.path.join(LOG_DIR, f"{tag}__train.log"), timeout)
        if ok:
            a = artifacts(tag)
            ok = os.path.exists(a["ckpt"]) and os.path.exists(a["loss"])
            if not ok:
                print("[FAIL] 학습 프로세스는 끝났지만 체크포인트·완주 표식이 없습니다.")
    if ok and need_val:
        ok = _sh(_py("eval.py", *common, "--split", "val", *_eval_extra(extra)),
                 os.path.join(LOG_DIR, f"{tag}__eval_val.log"), timeout)
        if ok:
            _warn_nan(artifacts(tag)["val"])
    if ok and need_test:
        ok = _sh(_py("eval.py", *common, "--split", "test", *_eval_extra(extra)),
                 os.path.join(LOG_DIR, f"{tag}__eval_test.log"), timeout)
        if ok:
            _warn_nan(artifacts(tag)["test"])
    return ok


def _eval_extra(extra):
    """train 용 추가 인자 중 eval 에도 뜻이 있는 것만 넘긴다(--max_clips N, --no_preload)."""
    out, i = [], 0
    while i < len(extra):
        if extra[i] == "--max_clips" and i + 1 < len(extra):
            out += [extra[i], extra[i + 1]]
            i += 2
            continue
        if extra[i] in ("--no_preload", "--allow_missing_audio"):
            out.append(extra[i])
        i += 1
    return out


def select_lr(lrs, seed, exclude=()):   # [2026-10-25 / A안] 선정 기록을 만들던 함수. 지금은 그 기록을 고정값(adam_lr)으로 쓰고 호출하지 않는다
    """[2026-09-18 / 3차 검수 C-1] exclude 에 든 태그(이번 실행에서 실패한 회차)는 옛 요약이 남아 있어도 후보에서 뺀다.
    val 요약도 json 이 지금 체크포인트·지금 판정 규칙(argmax_v1) 것일 때만 인정한다(rule_json_is_current, 2026-10-26)."""
    scores = {}
    for lr in lrs:
        tag = run_tag("sup", lr, seed)
        a = artifacts(tag)
        if tag in exclude or not (os.path.exists(a["rule"]) and rule_json_is_current(a["rule"], tag, a["ckpt"])):
            continue
        s = read_summary(a["val"])
        if s and "Macro F1" in s:
            scores[f"{lr:g}"] = s["Macro F1"]
    if len(scores) < len(lrs):
        missing = [f"{lr:g}" for lr in lrs if f"{lr:g}" not in scores]
        print(f"[WARN] 학습률 후보 중 val 결과가 없는 것: {missing} — 선정을 보류합니다.")
        return None
    best = max(scores, key=lambda k: scores[k])
    payload = {"model": MODEL_TAG, "lr": float(best), "criterion": "val Macro F1 (cond=sup)",
               "sweep_seed": seed, "candidates_val_macro_f1": scores,
               "selected_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(selected_json(), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 학습률 선정: {best} (val Macro F1 {scores[best]:.4f}) — 후보 {scores} -> {selected_json()}")
    return float(best)


def main():
    ap = argparse.ArgumentParser(description=f"{DISPLAY_NAME} 비교실험 실행기")
    ap.add_argument("--phase", choices=["all", "sweep", "main"], default="all")
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS, help="학습률 후보(sweep)")
    ap.add_argument("--sweep_seed", type=int, default=0, help="학습률 선정용 seed(본실험 1~5 와 겹치지 않게 0)")
    ap.add_argument("--lr", type=float, default=None, help="쓰지 않음(2026-10-25~ 학습률은 선정값 고정, common/naming.adam_lr). 주면 멈춘다")
    ap.add_argument("--conds", nargs="+", default=DEFAULT_CONDS, choices=["sup", "pl", "ceil"])
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--force", action="store_true", help="끝난 단계도 다시 돈다")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="계획만 출력")
    ap.add_argument("--eval_only", action="store_true",
                    help="[2026-10-26] 평가만(val·test) 다시 한다. 학습이 필요하다고 판정된 회차는 학습하지 않고 실패로 둔다")
    ap.add_argument("--step_timeout_min", type=float, default=None, help="한 단계 시간 제한(분)")
    ap.add_argument("--no_log", action="store_true")
    ap.add_argument("--extra", type=str, default="", help='train.py 에 그대로 넘길 인자. 예: "--max_clips 16 --epochs 1"')
    # [신설 2026-09-17 / 9절] labeled 축·데이터 효율 곡선·논문 회차 preset.
    ap.add_argument("--labeled_ns", type=int, nargs="+", default=[0], help="labeled 개수 목록(0=전량). conds x seeds 와 곱한다")
    ap.add_argument("--unlabeled_ns", type=int, nargs="+", default=[0],
                    help="pl 에서 쓸 라벨 없는 클립 개수 목록(0=전량). 그 클립들 안의 합의 라벨만 쓴다(Mark5_EXP2 와 같은 클립). pl 에만 적용")
    ap.add_argument("--preset", type=str, default=None, choices=["paper", "cmb"],   # [2026-10-22] cmb = CMB_PLAN(지도만·결합기 한 라벨 × 정답 4종, seed 1)
                    help="paper: PAPER_PLAN(조합 11개 x seed) — --conds/--labeled_ns/--unlabeled_ns 를 무시한다")
    args = ap.parse_args()
    global EVAL_ONLY
    EVAL_ONLY = bool(args.eval_only)
    if args.preset == "cmb" and args.seeds == DEFAULT_SEEDS:
        args.seeds = [1]   # [2026-10-22 결정] 비교모델은 조건마다 1회(seed 1). --seeds 를 주면 그 값을 쓴다
    extra = shlex.split(args.extra) if args.extra else []
    timeout = args.step_timeout_min * 60 if args.step_timeout_min else None
    global SMOKE
    SMOKE = _smoke_from_extra(extra)
    # [변경 2026-10-25 / A안] 학습률은 Adam 시절 선정값 고정이라 학습률 선정(sweep)과 --lr 을 받지 않는다. --phase all 은 main 만 돈다.
    if args.phase == "sweep" or args.lr is not None:
        raise SystemExit("[ERROR] 학습률은 Adam 시절 선정값으로 고정입니다(common/naming.adam_lr) — 학습률 선정(--phase sweep)과 --lr 을 "
                         "쓰지 않습니다. 인자를 빼고 다시 실행하십시오(run_tag _adam).")

    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    tee = None
    if not args.no_log and not args.dry_run:
        tee = _Tee(sys.stdout, os.path.join(LOG_DIR, f"run_experiments_{MODEL_TAG}_{time.strftime('%Y%m%d_%H%M%S')}.log"))
        sys.stdout = tee
    orig_stdout = sys.stdout if tee is None else tee.stream
    ok_all, fails = 0, []
    try:
        print("=" * 70)
        print(f" {DISPLAY_NAME} 비교실험  phase={args.phase}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" 옵티마이저=Adam(학습률 = Adam 시절 선정값 고정, 다시 고르지 않음)  conds={args.conds}  seeds={args.seeds}  extra={extra}")
        if SMOKE:
            print(" ★ 동작 확인(smoke) 모드 — 모든 태그에 _smoke 가 붙습니다(학습률 선정 없음). 본실험 산출물과 겹치지 않습니다.")
        print("=" * 70)

        selected = args.lr
        if args.phase in ("all", "main"):
            selected = None   # [변경 2026-10-25 / A안] lr_selected.json 을 읽지 않고 고정값(adam_lr)을 쓴다. None = 지금 절차 회차(_adam)
            # [2026-09-17 / 9절] 조합 = (labeled_n, cond, unlabeled_n). preset paper 면 PAPER_PLAN, 아니면 격자.
            if args.preset == "paper":
                plan = list(PAPER_PLAN)
            elif args.preset == "cmb":
                plan = list(CMB_PLAN)
                # [2026-10-22 / 최신 판 확인] 학습 전에 plc 회차가 읽을 Mark5.3 결합기 파일을 전부 대조한다(--dry_run 은 알리기만).
                from common.mark53_labels import pseudo_csv_path, check_fresh
                for _l in sorted({l for l, c, u in plan if c == "plc"}, key=lambda x: -1 if x is None else x):
                    for _s in args.seeds:
                        try:
                            _sha = check_fresh(pseudo_csv_path(_l, _s))
                            print(f" [FRESH] {os.path.basename(pseudo_csv_path(_l, _s))} ({_sha[:12]})")
                        except SystemExit as e:
                            print(f" {e}")   # 메시지가 [STALE] 로 시작한다
                            if not args.dry_run:
                                raise
            else:
                plan = []
                for l in args.labeled_ns:
                    for cond in args.conds:
                        for u in (args.unlabeled_ns if cond == "pl" else [0]):
                            plan.append((None if l == 0 else int(l), cond, None if u == 0 else int(u)))
            print(f"\n[2] 본실험 — Adam lr {ADAM_LR:g}(선정값 고정), 조합 {len(plan)}개 x seeds={args.seeds} = {len(plan) * len(args.seeds)}회"
                  + (f"  [preset {args.preset}]" if args.preset else f", conds={args.conds}, labeled_ns={args.labeled_ns}, unlabeled_ns={args.unlabeled_ns}"))
            for l, cond, u in plan:
                for seed in args.seeds:
                    if seed == args.sweep_seed:
                        print(f"[WARN] seed {seed} 는 학습률 선정용 seed 와 같습니다 — 본실험 seed 는 1~5 를 권합니다.")
                    ok = run_combo(cond, selected, seed, force=args.force, skip_test=args.skip_test, extra=extra,
                                   data_root=args.data_root, timeout=timeout, dry_run=args.dry_run,
                                   labeled_n=l, unlabeled_n=u)
                    if ok:
                        ok_all += 1
                    else:
                        fails.append(run_tag(cond, selected, seed, l, u))
        print("\n" + "=" * 70)
        print(f" 끝. 성공 {ok_all}개 / 실패 {len(fails)}개" + (f"  실패: {fails}" if fails else ""))
        print("=" * 70)
    finally:
        if tee is not None:
            sys.stdout = orig_stdout
            tee.close()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
