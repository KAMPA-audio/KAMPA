# -*- coding: utf-8 -*-
"""others 판정 규칙(운영점) — val 에서 고르고 json 으로 남겨 test·평가 세트·제로샷이 같은 규칙을 읽는다.

[신설 2026-09-17 / 검수 P1, B안] 네 비교모델 공용. 제안모델(Mark5.2 / Mark5_EXP2 eval.py + postprocess_utils.select_others_threshold)과
같은 후보·같은 선택 규칙이다. 전에는 reselect_others_threshold.py 가 전량 회차의 결과 파일만 사후에 덮어썼고 eval.py 는 언제나
argmax 였다. 이제 eval.py 가 --split val 에서 직접 고르고, 그 밖의 split 은 json 을 읽는다(없거나 체크포인트 지문이 다르면 멈춤).

후보와 규칙
  argmax        9클래스 softmax argmax(others 도 한 클래스, others 강제 없음).
  direct(thr)   Prob_others >= thr 이면 others, 아니면 others 를 뺀 8종 argmax. 격자 0.02~0.95/0.01.
  선택: val Macro F1 최대. 동률 격자점은 가운데, argmax 가 최댓값과 동률이면 argmax.
  ROC AUC 는 규칙과 무관한 softmax 확률로 잰다.

json (plots/others_threshold_<run_tag>.json)
  rule · threshold · val_macro_f1_at_threshold · criterion · grid · n_val_clips · candidates · curve · run_tag
  · checkpoint{file, sha256, bytes, mtime, mtime_epoch} · selected_at
  · [추가 2026-09-22] class_bias{클래스: 로짓 편향} · class_bias_grid · val_macro_f1_before_bias

[추가 2026-09-22 / A 클래스별 로짓 편향 — 제안모델(Mark5.2 eval.py _select_class_bias)과 같은 절차, 공정성 때문에 비교모델에도 건다]
  others 를 뺀 8칸에 로짓 편향 b 를 더한다(= 확률에 exp(b) 를 곱하고 다시 정규화). val 에서 좌표 상승(격자 −1.5~1.5 step 0.25, 3회,
  동률이면 |b| 작은 쪽)으로 고르고, 목적함수는 "그 편향에서 others 규칙 후보(argmax + direct 격자)의 최고 val Macro F1".
  편향을 고른 뒤 select_rule 이 같은 확률 위에서 임계값을 확정한다. test·eval_*·gen 라벨은 json 의 class_bias 를 그대로 쓴다.
  옛 json(class_bias 없음)은 편향 0 = 옛 동작. 제로샷(15앵커)에는 걸지 않는다 — 9클래스 val 로 고른 사전확률 보정을 새 앵커 6종에
  적용할 근거가 없고, 제안모델 zeroshot_eval.py 도 안 건다(양쪽 동일).
  근거(2026-09-21 Mark5.2 2차): 편향 {water_shower −1.5, construction −1.0, water_toilet +1.25 …} 로 val 0.8907→0.9345, test 0.9233→0.9558.
"""
import hashlib
import json
import os
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

GRID = (0.02, 0.95, 0.01)
BIAS_GRID = (-1.5, 1.5, 0.25)     # [2026-09-22] 클래스 편향 격자(제안모델 vild_config.class_bias_grid 와 같은 값)
BIAS_ROUNDS = 3


def predict_rule(probs, class_names, rule, thr):
    """probs [N, C] softmax → 판정 인덱스 [N]."""
    probs = np.asarray(probs, dtype=float)
    oi = class_names.index("others")
    if rule == "argmax":
        return probs.argmax(axis=1)
    if rule != "direct":
        raise ValueError(f"[ERROR] 모르는 others 규칙 {rule!r} ('argmax' 또는 'direct')")
    if thr is None:
        raise ValueError("[ERROR] direct 규칙에는 임계값이 있어야 합니다.")
    forced = probs[:, oi] >= float(thr)
    masked = probs.copy()
    masked[:, oi] = -1.0
    preds = masked.argmax(axis=1)
    preds[forced] = oi
    return preds


def _macro_f1(labels, preds, C):
    _, _, f, _ = precision_recall_fscore_support(labels, preds, average="macro", labels=list(range(C)), zero_division=0)
    return float(f)


# ---------------------------------------------------------------------------------------------------------
# [추가 2026-09-22 / A] 클래스별 로짓 편향 — Mark5.2 model/eval.py 의 _apply_class_bias/_macro_f1_np/_best_rule_f1/_select_class_bias 와 같은 계산
# ---------------------------------------------------------------------------------------------------------
def apply_class_bias(probs, bias):
    """확률 [N, C] 에 exp(편향) 을 곱하고 행마다 다시 정규화한다(로짓에 편향을 더한 것과 같다). 편향이 전부 0 이면 그대로."""
    p = np.asarray(probs, dtype=np.float64)
    if bias is None or not np.any(np.asarray(bias, dtype=np.float64)):
        return p
    q = p * np.exp(np.asarray(bias, dtype=np.float64))[None, :]
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)


def _macro_f1_np(labels, preds, C):
    """sklearn f1_score(average='macro', labels=range(C), zero_division=0) 과 같은 값(numpy, 격자 탐색용)."""
    labels = np.asarray(labels, dtype=int); preds = np.asarray(preds, dtype=int)
    cm = np.bincount(labels * C + preds, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(np.float64); fp = cm.sum(axis=0) - tp; fn = cm.sum(axis=1) - tp
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return float(f1.mean())


def best_rule_f1(probs, labels, class_names, grid=GRID):
    """주어진 확률에서 others 규칙 후보(argmax + direct 격자) 중 최고 val Macro F1 — 편향 탐색의 목적함수."""
    C = len(class_names); oi = class_names.index("others")
    start, stop, step = grid
    pts = np.round(np.arange(start, stop + 1e-9, step), 4)
    p = np.asarray(probs, dtype=np.float64); labels = np.asarray(labels, dtype=int)
    best = _macro_f1_np(labels, p.argmax(axis=1), C)
    masked = p.copy(); masked[:, oi] = -1.0
    am8 = masked.argmax(axis=1); po = p[:, oi]
    for t in pts:
        best = max(best, _macro_f1_np(labels, np.where(po >= t, oi, am8), C))
    return best


def select_class_bias(probs, labels, class_names, grid=BIAS_GRID, rounds=BIAS_ROUNDS):
    """(bias[C], f1_before, f1_after). others 칸은 0 으로 둔다. 좌표 상승, 동률이면 |편향| 작은 쪽."""
    C = len(class_names); oi = class_names.index("others")
    bias = np.zeros(C)
    f0 = best_rule_f1(probs, labels, class_names)
    if not grid:
        return bias, f0, f0
    pts = np.round(np.arange(float(grid[0]), float(grid[1]) + 1e-9, float(grid[2])), 4)
    best = f0
    for _ in range(int(rounds)):
        improved = False
        for i in range(C):
            if i == oi:
                continue
            cur = bias[i]; cand_best, cand_val = best, cur
            for g in pts:
                b = bias.copy(); b[i] = g
                f = best_rule_f1(apply_class_bias(probs, b), labels, class_names)
                if f > cand_best + 1e-12 or (abs(f - cand_best) <= 1e-12 and abs(g) < abs(cand_val)):
                    cand_best, cand_val = f, g
            if cand_val != cur:
                bias[i] = cand_val; best = cand_best; improved = True
        if not improved:
            break
    return bias, f0, best


def bias_from_saved(saved, class_names):
    """json 의 class_bias(dict) → [C] 배열. 없으면 0(옛 json = 옛 동작)."""
    cb = (saved or {}).get("class_bias") or {}
    return np.asarray([float(cb.get(n, 0.0)) for n in class_names], dtype=np.float64)


def describe_bias(bias, class_names):
    items = [f"{n} {float(b):+g}" for n, b in zip(class_names, bias) if float(b) != 0.0]
    return "편향 {" + ", ".join(items) + "}" if items else "편향 없음"


def select_rule(probs, labels, class_names, grid=GRID):
    """(rule, thr, best_f1, curve). Mark5_EXP2 postprocess_utils.select_others_threshold 와 같은 선택 규칙."""
    labels = np.asarray(labels, dtype=int)
    C = len(class_names)
    start, stop, step = grid
    pts = [float(t) for t in np.round(np.arange(start, stop + 1e-9, step), 4)]
    if not pts:
        raise ValueError("[ERROR] others 임계값 격자가 비어 있습니다.")
    f1_arg = _macro_f1(labels, predict_rule(probs, class_names, "argmax", None), C)
    f1s = [_macro_f1(labels, predict_rule(probs, class_names, "direct", t), C) for t in pts]
    curve = [("argmax", None, f1_arg)] + [("direct", t, f) for t, f in zip(pts, f1s)]
    best = max(f1s)
    if f1_arg >= best - 1e-12:
        return "argmax", None, f1_arg, curve
    top = [i for i, f in enumerate(f1s) if f >= best - 1e-12]
    mid = top[len(top) // 2]
    return "direct", pts[mid], f1s[mid], curve


def describe(rule, thr):
    return f"direct(thr {float(thr):.2f})" if rule == "direct" else "argmax"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rule_json_path(plot_dir, tag):
    return os.path.join(plot_dir, f"others_threshold_{tag}.json")


def save_rule_json(path, tag, rule, thr, best_f1, curve, n_val, ckpt_path, grid=GRID, class_bias=None, class_names=None,
                   f1_before_bias=None):
    info = {"mode": "val_select", "rule": rule, "threshold": thr, "selected_on": "val", "criterion": "macro_f1",
            "grid": [float(g) for g in grid], "candidates": "argmax + direct grid",
            "val_macro_f1_at_threshold": float(best_f1), "n_val_clips": int(n_val),
            "curve": [{"rule": r, "threshold": t, "val_macro_f1": f} for r, t, f in curve],
            "run_tag": tag,
            "checkpoint": {"file": os.path.basename(ckpt_path), "sha256": sha256_of(ckpt_path), "bytes": os.path.getsize(ckpt_path),
                           "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt_path))),
                           "mtime_epoch": float(os.path.getmtime(ckpt_path))},
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "eval.py --split val 이 고른 값. test·eval_*·zeroshot_eval 은 이 파일을 읽는다(체크포인트 sha256 대조)."}
    if class_bias is not None and class_names is not None:
        # [2026-09-22 / A] 편향은 임계값보다 먼저 고른 값. test·eval_*·gen 라벨은 이 편향을 확률에 건 뒤 rule 을 적용한다.
        info["class_bias"] = {n: float(b) for n, b in zip(class_names, class_bias)}
        info["class_bias_grid"] = [float(g) for g in BIAS_GRID]; info["class_bias_rounds"] = int(BIAS_ROUNDS)
        info["val_macro_f1_before_bias"] = None if f1_before_bias is None else float(f1_before_bias)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def load_rule_json(path, tag, ckpt_path):
    """(rule, thr, saved). 없으면 FileNotFoundError, run_tag·체크포인트 지문이 다르면 RuntimeError — 조용한 기본값은 없다."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[ERROR] {path} 이 없습니다. others 규칙은 val 에서 고른 것을 써야 하므로 eval.py --split val 을 먼저 돌리십시오"
            f"(run_experiments.py 는 이 순서로 돕니다). 고정 규칙으로 보려면 --others_threshold argmax 또는 --others_threshold 0.09.")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    fp = saved.get("checkpoint") or {}
    now = sha256_of(ckpt_path)
    if saved.get("run_tag") != tag or fp.get("sha256") != now:
        raise RuntimeError(
            f"[ERROR] {os.path.basename(path)} 은 지금 체크포인트로 고른 값이 아닙니다"
            f"(json run_tag={saved.get('run_tag')} sha256={str(fp.get('sha256'))[:12]} / 지금 {tag} sha256={now[:12]}). "
            f"재학습했거나 옛 reselect 산출물이면 eval.py --split val 을 다시 돌리십시오.")
    rule = saved.get("rule", "argmax")
    thr = float(saved["threshold"]) if rule == "direct" else None
    return rule, thr, saved


_LOST_WARNED = set()


def _ckpt_lost_after_eval(fp, ckpt_path):
    """[2026-10-02] 평가가 끝난 뒤 체크포인트 파일만 비어 버린 회차인가.
    파일은 0바이트인데 운영점 json 에 평가 때 잰 크기(>0)·sha256 이 있고, 파일 시각이 그때 기록한 시각(mtime_epoch)과
    120초 안에서 같으면(= 같은 파일) True. 2026-09-24 CLAP sup 3회(l500 s3 · l1000 s3 · l1000 s4)가 드라이브에 이 상태로 남았다 —
    평가는 온전한 파일로 끝났고 val·test 요약·예측 CSV 는 그대로 있다. 이런 회차는 결과를 그대로 쓰되 다시 평가할 수는 없다."""
    if os.path.getsize(ckpt_path) != 0 or not fp.get("sha256") or not fp.get("bytes") or fp.get("mtime_epoch") is None:
        return False
    return abs(float(fp["mtime_epoch"]) - os.path.getmtime(ckpt_path)) <= 120.0


def rule_json_matches_ckpt(path, tag, ckpt_path):
    """[2026-09-18 / 3차 검수 C-2] json 이 지금 체크포인트로 고른 것인지 **가볍게**(해시 없이) 본다 — run_experiments 의 재개 판정·집계용.
    run_tag 가 같고 checkpoint.bytes·mtime(초 단위)이 지금 파일과 같으면 True. 옛 reselect 형식(checkpoint null)은 False.
    엄밀한 대조(sha256)는 eval.py 가 load_rule_json 으로 한다.
    [2026-10-02] 평가 뒤 체크포인트만 비어 버린 회차(_ckpt_lost_after_eval)도 True — 그 평가 결과를 그대로 쓰고 경고를 한 번 찍는다."""
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return False
    fp = saved.get("checkpoint") or {}
    if saved.get("run_tag") != tag or not fp.get("sha256") or not os.path.isfile(ckpt_path):
        return False
    if _ckpt_lost_after_eval(fp, ckpt_path):
        if tag not in _LOST_WARNED:
            _LOST_WARNED.add(tag)
            print(f"[WARN] {tag}: 체크포인트 파일이 비어 있다(0 bytes) — 평가 때 기록한 크기 {int(fp['bytes']):,} · sha256 "
                  f"{str(fp['sha256'])[:12]} 의 val·test 결과를 그대로 쓴다(평가 뒤 파일만 유실, 다시 평가할 수는 없음)", flush=True)
        return True
    if fp.get("bytes") != os.path.getsize(ckpt_path):
        return False
    if fp.get("mtime_epoch") is not None:
        # 시간대와 무관한 초 단위 비교. 코랩 드라이브 마운트(FUSE)가 다시 마운트되면 서버 modifiedTime 이 몇 초 다를 수 있어 120초를
        # 허용한다 — 재학습은 분 단위로 떨어지니 구분에 충분하고, 엄밀한 대조(sha256)는 eval.py 가 따로 한다.
        return abs(float(fp["mtime_epoch"]) - os.path.getmtime(ckpt_path)) <= 120.0
    now_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt_path)))
    return fp.get("mtime") == now_mtime

# [2026-10-26] 판정 규칙 통일 — 모든 모델(Mark5.3 포함)의 답 = 9클래스 확률의 argmax(보정 없음).
# 2026-09-22~10-26 의 val 선택(클래스 편향 + others 규칙)은 Mark5.3 이 10-22 에 보정을 뺀 뒤로 판정 규칙이 달라져 기본 경로에서 뺐다.
# val 평가가 이 판정 규칙을 json 에 적고(편향 0 · argmax), test 는 그 json 이 이 판정 규칙일 때만 읽는다. 옛 val 선택 json 은
# 실행기가 '안 끝남' 으로 보고 val·test 를 다시 평가한다(학습은 다시 하지 않는다).
DECISION_POLICY = "argmax_v1"


def save_argmax_json(path, tag, n_val, ckpt_path, class_names):
    """[2026-10-26] 판정 규칙 기록(argmax · 클래스 편향 0). 체크포인트 지문은 save_rule_json 과 같은 형식이라 load_rule_json·
    rule_json_matches_ckpt 가 그대로 읽는다."""
    info = {"mode": "argmax", "decision_policy": DECISION_POLICY, "rule": "argmax", "threshold": None,
            "selected_on": None, "n_val_clips": int(n_val), "run_tag": tag,
            "class_bias": {n: 0.0 for n in class_names},
            "checkpoint": {"file": os.path.basename(ckpt_path), "sha256": sha256_of(ckpt_path), "bytes": os.path.getsize(ckpt_path),
                           "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt_path))),
                           "mtime_epoch": float(os.path.getmtime(ckpt_path))},
            "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "보정 없는 argmax(Mark5.3 과 같은 판정 규칙). eval.py --split val 이 적고 test 가 읽는다(체크포인트 sha256 대조)."}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def rule_json_is_current(path, tag, ckpt_path):
    """[2026-10-26] 지금 체크포인트의 json(rule_json_matches_ckpt) 이면서 지금 판정 규칙(DECISION_POLICY)으로 적힌 것인가.
    run_experiments 의 완료 판정·집계가 쓴다 — 옛 val 선택 json 이면 False 라 val·test 를 다시 평가한다."""
    if not rule_json_matches_ckpt(path, tag, ckpt_path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("decision_policy") == DECISION_POLICY
    except (OSError, ValueError):
        return False
