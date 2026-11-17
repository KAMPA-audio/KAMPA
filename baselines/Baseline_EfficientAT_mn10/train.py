#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""비교모델 파인튜닝 1회 — 조건(sup | pl | ceil) · seed · 학습률 하나로 학습해 체크포인트와 완주 표식을 남긴다.

[신설 2026-09-05 / 개정 2026-09-05 저녁, code-reviewer 지적 반영 / 2026-09-06 천장 조건 추가]
조건표: Downloads/Mark5와비교모델_비교실험조건표.pdf (2026-09-06 개정판)

    --cond sup  : labeled 2,000 만 (조건표 표 0-1 의 A 블록 = 바닥)
    --cond pl   : labeled 2,000 + 합의 라벨(consensus_labels.csv, 확신도 >= 0.5) (B 블록 = 반지도)
                  [2026-10-01] 합의 라벨 CSV 에 전원 거절 → others 행(origin=reject_others)이 있으면 같이 쓴다(Mark5.2 의 OR 과 같은 재료).
                  --unlabeled_n 은 라벨 없는 클립 n 개를 먼저 뽑고 그 안의 합의 라벨만 쓴다(Mark5_EXP2 와 같은 클립).
    --cond ceil : labeled 2,000 + unlabeled 2,000 에 **진짜 라벨**(ceiling_labels.csv) (C 블록 = 천장)
                  Cances et al. (2022) 의 "Supervised, 라벨 100%" 기준선. 제안 방법이 아니라 상한을 재는 조건이다.
                  CSV 는 Mark5_EXP/model/make_ceiling_labels.py 가 data_provenance.xlsx 에서 만든다.
    --cond pseudolabel | meanteacher | fixmatch | freematch
                : [2026-10-07 / 논문 표 1] 표준 반지도 기준선 — labeled + 라벨 없는 클립 2,000(dataset_unlabeled.csv)을 섞은 배치로
                  학습한다(_run_ssl → common/ssl_loop.fit_ssl, 손실은 common/ssl.py). 숨긴 정답 ceiling_labels.csv 는 가짜 정답
                  품질 기록(plots/pl_quality_<run_tag>.csv · train_meta 의 pl_quality_final)에만 쓴다.
                  [2026-10-15 / 라벨 없는 양 곡선] --unlabeled_n 을 주면 라벨 없는 클립을 pl 과 같은 규칙(seed 기준 누적,
                  common/data.py subsample_nested)으로 n 개만 쓴다 — Mark5.3 의 --unlabeled_n 과 같은 클립. 태그 _u<n>.

산출물 (run_tag = <model>_<cond>_adam[_l<n>][_u<n>]_s<seed>[_smoke], 규칙은 common/naming.py 한 곳 · 2026-10-21까지는 _lr<lr>)
    model/<run_tag>.pth                 best-val 체크포인트 (에폭 루프 안에서 덮어씀)
    plots/loss_history_<run_tag>.csv    완주 표식 (루프가 끝난 뒤에만 씀)
    plots/loss_curve_<run_tag>.png
    plots/train_meta_<run_tag>.json     조건·데이터 개수·파라미터 수·소요 시간 (실행기가 재개 판정에 읽는다)

--max_clips 를 주면 태그에 _smoke 가 붙어 본실험 산출물과 절대 겹치지 않는다(전에는 같은 태그라 실행기가
축소 실행을 완료로 착각할 수 있었다).

예)  python train.py --cond sup --seed 1
     python train.py --cond pl  --seed 3 --data_root /content/drive/MyDrive/NewProject/codes/experiments/KAMPA   # [2026-11-17] 데이터를 Mark5.2 에서 KAMPA 로 옮김
     python train.py --cond sup --seed 1 --max_clips 16 --epochs 1      # 동작 확인용 -> *_smoke
     python train.py --cond fixmatch --seed 1 --labeled_n 200           # [2026-10-07] 표준 반지도 기준선(표 1)
     [2026-10-25 / A안] 옵티마이저는 Adam, 학습률은 Adam 시절 선정값 고정(common/naming.adam_lr)이라 --lr 을 받지 않는다(주면 멈춤).
     run_tag 는 _lr<lr> 대신 _adam(2026-10-22~10-25 Prodigy 시절은 _pd).
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.paths import CLASS_NAMES, LABEL_MAP, default_data_root, resolve_csv, resolve_audio_path, index_fingerprint   # noqa: E402
from common.data import (ClipDataset, clip_key, collate, read_index, load_consensus_rows, select_labeled_subset,   # noqa: E402
                         restrict_to_unlabeled_subset, subsample_nested)
from common.naming import run_tag, artifact_paths, UNLABELED_SUBSAMPLE_RULE, SSL_METHODS, ssl_impl_version, OPTIMIZER_NAME, adam_lr   # noqa: E402
from common.seed import set_seed                                                         # noqa: E402
from common.train_loop import fit                                                        # noqa: E402
from common.ssl_loop import fit_ssl                                                      # noqa: E402  [2026-10-07 / 표 1]
from model_def import MODEL_TAG, DISPLAY_NAME, build_model                               # noqa: E402

DEFAULT_EPOCHS = 80


def _subsample(rows, n, seed):
    """동작 확인용. seed 고정으로 섞어서 앞에서 n 개."""
    if not n or n >= len(rows):
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in sorted(idx[:n])]


def build_parser():
    ap = argparse.ArgumentParser(description=f"{DISPLAY_NAME} 파인튜닝 1회")
    ap.add_argument("--cond", choices=["sup", "pl", "plc", "ceil", *SSL_METHODS], required=True,   # [2026-10-22] plc = 결합기 한 라벨
                    help="sup=labeled 2000만, pl=+합의 라벨, ceil=+진짜 라벨(천장), "
                         "pseudolabel·meanteacher·fixmatch·freematch=표준 반지도 기준선(+라벨 없는 클립 2,000, 표 1)")
    ap.add_argument("--seed", type=int, required=True)
    # [변경 2026-10-25 / A안] 학습률은 선정값 고정(common/naming.adam_lr)이라 받지 않는다. 옛 명령을 그대로 넣으면 조용히 무시하지 않고 main 이 멈춘다.
    ap.add_argument("--lr", type=float, default=None, help="쓰지 않음(학습률은 선정값 고정, common/naming.adam_lr). 주면 멈춘다")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--data_root", type=str, default=None, help="오디오 기준 폴더(기본: 옆의 Mark5_EXP)")
    ap.add_argument("--consensus_csv", type=str, default="consensus_labels.csv")
    ap.add_argument("--ceiling_csv", type=str, default="ceiling_labels.csv",
                    help="--cond ceil 이 읽는 진짜 라벨 CSV(Mark5_EXP make_ceiling_labels.py 산출)")
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--no_spec_augment", action="store_true")
    ap.add_argument("--no_deterministic", action="store_true", help="cudnn deterministic 끔(기본 켬)")
    ap.add_argument("--no_preload", action="store_true", help="파형을 메모리에 미리 올리지 않음")
    ap.add_argument("--allow_missing_audio", action="store_true", help="오디오 누락을 오류로 보지 않음(비상용)")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_clips", type=int, default=0, help="동작 확인용: labeled/val 각각 이 개수만. 태그에 _smoke 가 붙는다")
    # [신설 2026-09-17 / 9절] 라벨 효율 표·데이터 효율 곡선(Mark5_EXP2 와 같은 규칙·같은 태그 뜻).
    ap.add_argument("--labeled_n", type=int, default=None,
                    help="labeled 를 몇 클립만 쓸지(200/500/1000). 클래스별 고르게, seed 기준 누적. 태그 _l<n>. 미지정 시 전량.")
    ap.add_argument("--unlabeled_n", type=int, default=None,
                    help="--cond pl 에서 라벨 없는 클립을 몇 개만 쓸지(500/1000/2000). seed 기준 누적으로 뽑고 그 안의 합의 라벨만 쓴다"
                         "(Mark5_EXP2 와 같은 클립). [2026-10-15] 표준 반지도 4방법에서도 같은 규칙으로 라벨 없는 클립을 n 개만 쓴다"
                         "(라벨 없는 양 곡선). 태그 _u<n>. 미지정 시 전량.")
    return ap


def _run_ssl(args, tag, paths, data_root, device, labeled, smoke):
    """[신설 2026-10-07 / 논문 표 1] 표준 반지도 기준선 1회 — 정답 클립(labeled 부분집합) + 라벨 없는 클립 전부(dataset_unlabeled.csv).

    두 집합을 한 데이터셋으로 섞어 배치 16(Mark5.3 과 같은 구성), 손실·평가 모델은 common/ssl.py, 루프는 common/ssl_loop.fit_ssl.
    숨긴 정답(ceiling_labels.csv)은 가짜 정답 품질 기록에만 쓴다 — 라벨 없는 클립은 정답 칸을 비워(-1) 데이터셋에 넣는다."""
    if args.no_spec_augment:
        raise SystemExit("[ERROR] 표준 반지도 기준선의 강한 증강이 SpecAugment 라서 --no_spec_augment 와 함께 쓸 수 없습니다.")
    unl_csv = os.path.join(ROOT, "dataset_unlabeled.csv")
    if not os.path.isfile(unl_csv):
        raise SystemExit(f"[ERROR] --cond {args.cond} 에는 dataset_unlabeled.csv 가 필요합니다(라벨 없는 클립 목록): {unl_csv}")
    unl_rows = [{"path": r["path"], "label": "", "source": "unlabeled"} for r in read_index(unl_csv)]   # 정답 칸을 비운다
    if args.unlabeled_n is not None:
        # [신설 2026-10-15 / 라벨 없는 양 곡선] pl 의 restrict_to_unlabeled_subset 과 같은 함수·같은 행 순서(dataset_unlabeled.csv 그대로)
        # ·같은 seed 라 Mark5.3 --unlabeled_n 과 같은 클립을 쓴다. 0 개면 이 방법은 지도만(--cond sup)과 같으므로 받지 않는다.
        if args.unlabeled_n <= 0:
            raise SystemExit("[ERROR] 표준 반지도에서 --unlabeled_n 은 1 이상이어야 합니다 — 라벨 없는 클립 0개는 지도만(--cond sup)과 같습니다.")
        _all = len(unl_rows)
        unl_rows = subsample_nested(unl_rows, args.unlabeled_n, args.seed)
        print(f"[INFO] 라벨 없는 클립 서브샘플: {_all}개 중 {len(unl_rows)}개 (seed={args.seed} 기준 누적, {UNLABELED_SUBSAMPLE_RULE})")
    if smoke:
        unl_rows = _subsample(unl_rows, args.max_clips, args.seed)
    lab_paths = {p for p in (resolve_audio_path(r["path"], data_root) for r in labeled) if p is not None}
    dup = [r["path"] for r in unl_rows if (resolve_audio_path(r["path"], data_root) or object()) in lab_paths]
    if dup:
        raise SystemExit(f"[ERROR] 라벨 없는 클립 {len(dup)}개가 labeled 와 같은 클립입니다(예: {dup[0]}).")

    # 숨긴 정답 — 가짜 정답 품질(표 3) 기록 전용. 파일 이름(clip_key)으로 맞춘다(경로 앞부분은 판마다 다르다).
    ceil_path = args.ceiling_csv if os.path.isabs(args.ceiling_csv) else os.path.join(ROOT, args.ceiling_csv)
    gt_by_key = {}
    if os.path.isfile(ceil_path):
        for r in read_index(ceil_path):
            if r.get("label") in LABEL_MAP:
                gt_by_key[clip_key(r["path"])] = LABEL_MAP[r["label"]]
    unl_keys = {clip_key(r["path"]) for r in unl_rows}
    covered = len(unl_keys & set(gt_by_key))
    print(f"[INFO] 숨긴 정답(가짜 정답 품질 기록 전용): 라벨 없는 클립 {len(unl_keys)}개 중 {covered}개 대조 가능 ({ceil_path})")
    if covered < len(unl_keys) and not smoke:
        raise SystemExit(f"[ERROR] ceiling_labels.csv 가 라벨 없는 클립 {len(unl_keys) - covered}개를 덮지 못합니다 — 표 3 이 불완전해집니다.")

    val_rows = read_index(resolve_csv("dataset_val.csv"))
    if smoke:
        val_rows = _subsample(val_rows, args.max_clips, args.seed)
    strict = not args.allow_missing_audio
    lab_ds = ClipDataset(labeled, data_root, "train(정답)", preload=not args.no_preload, strict=strict)
    unl_ds = ClipDataset(unl_rows, data_root, "train(라벨 없음)", preload=not args.no_preload, require_labels=False, strict=strict)
    if any(l != -1 for _, l, _ in unl_ds.items):
        raise SystemExit("[ERROR] 라벨 없는 클립에 정답 인덱스가 들어갔습니다 — 정답 누출이라 멈춥니다.")
    if len(unl_ds) == 0:
        raise SystemExit("[ERROR] 라벨 없는 클립이 0개입니다.")
    val_ds = ClipDataset(val_rows, data_root, "val", preload=not args.no_preload, strict=strict)
    print(f"[INFO] train 정답 클래스 분포: {lab_ds.label_counts()} · 라벨 없는 클립 {len(unl_ds)}개")
    print(f"[INFO] val   클래스 분포: {val_ds.label_counts()}")
    pin = device.type == "cuda"
    train_loader = DataLoader(ConcatDataset([lab_ds, unl_ds]), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.num_workers, pin_memory=pin)
    unl_eval_loader = DataLoader(unl_ds, batch_size=32, shuffle=False, collate_fn=collate,
                                 num_workers=args.num_workers, pin_memory=pin)

    model = build_model(num_classes=len(CLASS_NAMES), load_pretrained=True).to(device)
    if not getattr(model, "pretrained_loaded", False):
        raise SystemExit("[ERROR] 사전학습 가중치가 로드되지 않았습니다.")
    n_total, n_train = model.count_parameters(), model.count_parameters(True)
    print(f"[INFO] {DISPLAY_NAME}: 학습 대상 파라미터 {n_train:,} (모듈 전체 {n_total:,}) — 전 층 파인튜닝 · 표준 반지도 {args.cond}")

    t0 = time.time()
    result = fit_ssl(model, train_loader, val_loader, device, method=args.cond,
                     lr=adam_lr(args.cond), weight_decay=args.weight_decay, epochs=args.epochs, patience=args.patience,
                     num_classes=len(CLASS_NAMES), ckpt_path=paths["ckpt"], loss_csv_path=paths["loss"],
                     curve_png_path=paths["curve"], pl_csv_path=paths["pl"], tag=tag,
                     extra_ckpt_meta={"model_tag": MODEL_TAG, "cond": args.cond, "seed": args.seed,
                                      "num_classes": len(CLASS_NAMES), "class_names": CLASS_NAMES},
                     gt_by_key=gt_by_key, unl_eval_loader=unl_eval_loader)
    elapsed = time.time() - t0
    meta = {
        "run_tag": tag, "model": MODEL_TAG, "display_name": DISPLAY_NAME, "cond": args.cond,
        "seed": args.seed, "lr": adam_lr(args.cond), "optimizer": OPTIMIZER_NAME, "epochs_max": args.epochs, "batch_size": args.batch_size,
        "patience": args.patience, "weight_decay": args.weight_decay,
        "spec_augment": True, "deterministic": not args.no_deterministic,
        "smoke": smoke, "max_clips": args.max_clips, "labeled_n": args.labeled_n, "unlabeled_n": args.unlabeled_n,
        # [2026-10-15] --unlabeled_n 회차의 추출 규칙 판(실행기 재개 판정이 대조한다). 전량이면 빈칸.
        "unlabeled_subsample_rule": (UNLABELED_SUBSAMPLE_RULE if args.unlabeled_n is not None else ""),
        "n_train_labeled": len(lab_ds), "n_train_unlabeled": len(unl_ds), "n_train_consensus": 0, "n_val": len(val_ds),
        "extra_label_source": f"ssl:{args.cond}", "extra_label_csv": "",
        "ssl_method": args.cond, "ssl_impl_version": ssl_impl_version(args.cond), "ssl_hparams": result["ssl_hparams"],
        "eval_model": result["ssl_hparams"]["eval_model"],
        "pl_quality_final": result["pl_final"], "pl_quality_best_epoch_train": result["pl_best_epoch_train"],
        "pl_ground_truth_csv": os.path.basename(ceil_path), "pl_ground_truth_covered": covered,
        "parameters_total": n_total, "parameters_trainable": n_train,
        "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
        "pretrained_source": str(getattr(model, "source", "")),
        "best_epoch": result["best_epoch"], "best_val_loss": result["best_val_loss"],
        "epochs_run": result["epochs_run"], "stopped_early": result["stopped_early"],
        "train_seconds": round(elapsed, 1), "device": str(device),
        "index_sha1": index_fingerprint(),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(paths["meta"]), exist_ok=True)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 학습 끝: {elapsed / 60:.1f}분, best epoch {result['best_epoch']}, meta -> {paths['meta']}")


def main():
    args = build_parser().parse_args()
    smoke = args.max_clips > 0
    if args.lr is not None:
        raise SystemExit(f"[ERROR] 학습률은 Adam 시절 선정값으로 고정입니다(common/naming.adam_lr, 받은 값 {args.lr:g}). --lr 을 빼고 다시 실행하십시오.")
    if args.unlabeled_n is not None and args.cond not in ("pl", "plc", *SSL_METHODS):
        raise SystemExit("[ERROR] --unlabeled_n 은 --cond pl 과 표준 반지도 4방법에서만 뜻이 있습니다(라벨 없는 클립 개수).")
    if args.labeled_n is not None and args.cond == "ceil":
        raise SystemExit("[ERROR] --labeled_n 과 --cond ceil 은 함께 쓰지 않습니다(상한 행은 labeled 전량 + 진짜 라벨 전량).")
    tag = run_tag(MODEL_TAG, args.cond, args.lr, args.seed, smoke=smoke, labeled_n=args.labeled_n, unlabeled_n=args.unlabeled_n)
    paths = artifact_paths(ROOT, tag)
    data_root = args.data_root or default_data_root()
    set_seed(args.seed, deterministic=not args.no_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 실험 조건: model={DISPLAY_NAME} ({MODEL_TAG}), cond={args.cond}, seed={args.seed}, "
          f"옵티마이저=Adam(lr {adam_lr(args.cond):g}, 선정값 고정), run_tag={tag}, device={device}")
    print(f"[INFO] data_root={data_root}")
    if smoke:
        print(f"[WARN] --max_clips {args.max_clips}: 동작 확인용 축소 실행입니다. 산출물 이름에 _smoke 가 붙고 표에 쓰지 않습니다.")

    labeled = read_index(resolve_csv("dataset_labeled.csv"))
    for r in labeled:
        r["source"] = "labeled"
    if args.labeled_n is not None:
        # [2026-09-17 / 9절] Mark5_EXP2 와 같은 규칙으로 같은 클립을 뽑는다(같은 dataset_labeled.csv·같은 seed 전제).
        labeled, _rest = select_labeled_subset(labeled, args.labeled_n, args.seed, CLASS_NAMES)
        _cnt = {}
        for r in labeled:
            _cnt[r["label"]] = _cnt.get(r["label"], 0) + 1
        print(f"[INFO] labeled 서브샘플: {len(labeled)}개 (seed={args.seed}, 클래스별 {_cnt})")
    if smoke:
        labeled = _subsample(labeled, args.max_clips, args.seed)   # 합의 라벨은 줄이지 않는다(병합 경로가 실제로 돌게)
    if args.cond in SSL_METHODS:
        # [신설 2026-10-07 / 논문 표 1] 표준 반지도 기준선은 따로 돈다(아래 sup·pl·ceil 경로는 그대로).
        _run_ssl(args, tag, paths, data_root, device, labeled, smoke)
        return
    train_rows = list(labeled)
    consensus = []           # pl: 합의 라벨 행 / ceil: 진짜 라벨 행 (변수 이름은 옛 것을 유지 — meta 키와 맞춤)
    pseudo_csv_sha256 = ""
    if args.cond == "plc":
        # [신설 2026-10-22 / 검증 결합기] Mark5.3 의 결합기 한 라벨 파일을 옆 저장소에서 직접 읽는다(복사본 없음). 정답 수·seed 마다 파일이 다르다.
        # 읽기 전에 그 파일이 지금 Mark5.3 교사·점수 캐시·결합기의 최신 판인지 해시로 대조한다(어긋나면 멈춤).
        from common.mark53_labels import pseudo_csv_path, check_fresh
        args.consensus_csv = pseudo_csv_path(args.labeled_n, args.seed)
        if not os.path.isfile(args.consensus_csv):
            raise SystemExit(f"[ERROR] --cond plc 인데 결합기 가짜 정답 파일이 없습니다: {args.consensus_csv}\n"
                             "        Mark5.3 에서 model/make_combiner_labels.py 를 먼저 돌리십시오.")
        pseudo_csv_sha256 = check_fresh(args.consensus_csv)
        print(f"[FRESH] {os.path.basename(args.consensus_csv)} — Mark5.3 최신 판({pseudo_csv_sha256[:12]})")
    if args.cond in ("pl", "plc", "ceil"):
        if args.cond in ("pl", "plc"):
            csv_arg, what, src, min_conf = args.consensus_csv, "합의 라벨", "consensus", args.min_conf
            hint = "Mark5_EXP 에서 model/make_consensus_labels.py 를 돌려 만든 파일을 이 폴더에 복사하십시오."
        else:
            # 천장: 진짜 라벨이라 확신도 임계값을 걸지 않는다(CSV 의 confidence 는 전부 1.0).
            csv_arg, what, src, min_conf = args.ceiling_csv, "천장 라벨", "ceiling", 0.0
            hint = "Mark5_EXP 에서 model/make_ceiling_labels.py 를 돌려 만든 파일을 이 폴더에 복사하십시오."
        cpath = csv_arg if os.path.isabs(csv_arg) else os.path.join(ROOT, csv_arg)
        if not os.path.isfile(cpath):
            raise SystemExit(f"[ERROR] --cond {args.cond} 인데 {what} CSV 가 없습니다: {cpath}\n        {hint}")
        consensus = load_consensus_rows(cpath, min_conf, source=src, what=what)   # 0행이면 안에서 멈춘다
        if args.cond in ("pl", "plc") and args.unlabeled_n is not None:
            # [변경 2026-10-01] 데이터 효율 곡선: **라벨 없는 클립**(dataset_unlabeled.csv) 에서 seed 기준 누적으로 n 개를 뽑고 그 안에 있는
            # 합의 라벨만 쓴다 — Mark5_EXP2 의 unlabeled 서브샘플과 같은 규칙·같은 클립(common/data.py restrict_to_unlabeled_subset).
            # 전(2026-09-17~10-01)에는 이미 채택된 합의 라벨에서 n 개를 뽑아 Mark5.2 보다 가짜 정답을 약 1.35배 더 받았다.
            unl_csv = os.path.join(ROOT, "dataset_unlabeled.csv")
            if not os.path.isfile(unl_csv):
                raise SystemExit(f"[ERROR] --unlabeled_n 에는 dataset_unlabeled.csv 가 필요합니다(라벨 없는 클립 목록): {unl_csv}")
            unl_rows = read_index(unl_csv)
            _all = len(consensus)
            consensus = restrict_to_unlabeled_subset(consensus, unl_rows, args.unlabeled_n, args.seed)
            _n_rej = sum(1 for r in consensus if r.get("origin") == "reject_others")
            print(f"[INFO] 라벨 없는 클립 서브샘플: {len(unl_rows)}개 중 {args.unlabeled_n}개(seed={args.seed} 기준 누적) → 그 안의 합의 라벨 "
                  f"{len(consensus)}개 사용 (전체 {_all}개 중" + (f", 전원 거절→others {_n_rej}개 포함" if _n_rej else "") + ")")
        # labeled 와 같은 클립이 있으면 라벨 충돌·가중치 2배가 되므로 멈춘다.
        # [2026-09-06 리뷰 반영] 해석 실패(None)는 집합에서 빼야 한다 — None 끼리 같다고 거짓 중복이 잡힌다.
        lab_paths = {p for p in (resolve_audio_path(r["path"], data_root) for r in labeled) if p is not None}
        dup = [r["path"] for r in consensus
               if (resolve_audio_path(r["path"], data_root) or object()) in lab_paths]
        if dup:
            raise SystemExit(f"[ERROR] {what} {len(dup)}개가 labeled 와 같은 클립입니다(예: {dup[0]}). "
                             f"{what}은 dataset_unlabeled.csv 로 만들어야 합니다.")
        if args.cond == "ceil":
            # 천장은 unlabeled **전부**에 라벨을 되돌려 준 조건이어야 한다. 일부만 들어오면 천장이 아니다.
            unl_csv = os.path.join(ROOT, "dataset_unlabeled.csv")
            if os.path.isfile(unl_csv):
                unl_keys = {os.path.basename(str(r["path"]).replace("\\", "/")) for r in read_index(unl_csv)}
                ceil_keys = {os.path.basename(str(r["path"]).replace("\\", "/")) for r in consensus}
                if ceil_keys != unl_keys and not smoke:
                    raise SystemExit(f"[ERROR] 천장 라벨 CSV({len(ceil_keys)}개)가 dataset_unlabeled.csv({len(unl_keys)}개)와 "
                                     f"같은 클립 집합이 아닙니다(없음 {len(unl_keys - ceil_keys)}, 남음 {len(ceil_keys - unl_keys)}).")
            else:
                print("[WARN] dataset_unlabeled.csv 가 없어 천장 라벨이 unlabeled 전부를 덮는지 확인하지 못했습니다.")
        train_rows += consensus
    val_rows = read_index(resolve_csv("dataset_val.csv"))
    if smoke:
        val_rows = _subsample(val_rows, args.max_clips, args.seed)

    strict = not args.allow_missing_audio
    train_ds = ClipDataset(train_rows, data_root, "train", preload=not args.no_preload, strict=strict)
    val_ds = ClipDataset(val_rows, data_root, "val", preload=not args.no_preload, strict=strict)
    n_consensus_used = sum(1 for _, _, m in train_ds.items if m["source"] in ("consensus", "ceiling"))
    if args.cond in ("pl", "plc", "ceil") and n_consensus_used == 0:
        raise SystemExit(f"[ERROR] --cond {args.cond} 인데 학습에 들어간 추가 라벨 클립이 0개입니다.")
    print(f"[INFO] train 클래스 분포: {train_ds.label_counts()}")
    print(f"[INFO] val   클래스 분포: {val_ds.label_counts()}")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.num_workers, pin_memory=pin)

    model = build_model(num_classes=len(CLASS_NAMES), load_pretrained=True).to(device)
    if not getattr(model, "pretrained_loaded", False):
        raise SystemExit("[ERROR] 사전학습 가중치가 로드되지 않았습니다.")
    n_total, n_train = model.count_parameters(), model.count_parameters(True)
    print(f"[INFO] {DISPLAY_NAME}: 학습 대상 파라미터 {n_train:,} (모듈 전체 {n_total:,}, 고정 커널 포함) — 전 층 파인튜닝")

    t0 = time.time()
    result = fit(model, train_loader, val_loader, device,
                 lr=adam_lr(args.cond), weight_decay=args.weight_decay, epochs=args.epochs, patience=args.patience,
                 use_spec_augment=not args.no_spec_augment,
                 ckpt_path=paths["ckpt"], loss_csv_path=paths["loss"], curve_png_path=paths["curve"], tag=tag,
                 extra_ckpt_meta={"model_tag": MODEL_TAG, "cond": args.cond, "seed": args.seed,
                                  "num_classes": len(CLASS_NAMES), "class_names": CLASS_NAMES})
    elapsed = time.time() - t0
    meta = {
        "run_tag": tag, "model": MODEL_TAG, "display_name": DISPLAY_NAME, "cond": args.cond,
        "seed": args.seed, "lr": adam_lr(args.cond), "optimizer": OPTIMIZER_NAME, "epochs_max": args.epochs, "batch_size": args.batch_size,
        "patience": args.patience, "weight_decay": args.weight_decay, "min_conf": args.min_conf,
        "spec_augment": not args.no_spec_augment, "deterministic": not args.no_deterministic,
        "smoke": smoke, "max_clips": args.max_clips,
        "labeled_n": args.labeled_n, "unlabeled_n": args.unlabeled_n,      # [2026-09-17 / 9절]
        "n_train_labeled": sum(1 for _, _, m in train_ds.items if m["source"] == "labeled"),
        "n_train_consensus": n_consensus_used, "n_consensus_csv_rows_used": len(consensus), "n_val": len(val_ds),
        # [2026-10-01] 합의 라벨 중 전원 거절 → others 행 수(origin=reject_others, 없는 옛 CSV 는 0) · --unlabeled_n 추출 규칙의 판 이름
        "n_train_reject_others": sum(1 for r in consensus if r.get("origin") == "reject_others"),
        "unlabeled_subsample_rule": (UNLABELED_SUBSAMPLE_RULE if (args.cond in ("pl", "plc") and args.unlabeled_n is not None) else ""),
        "consensus_generated_at": (consensus[0].get("generated_at", "") if consensus else ""),
        # [2026-09-06] 어느 라벨을 얹었는지 이름으로도 남긴다(pl=합의 라벨, ceil=진짜 라벨, sup=없음).
        "extra_label_source": ("consensus" if args.cond == "pl" else "combiner" if args.cond == "plc" else "ceiling" if args.cond == "ceil" else ""),
        "extra_label_csv": (args.consensus_csv if args.cond in ("pl", "plc") else args.ceiling_csv if args.cond == "ceil" else ""),
        "pseudo_csv_sha256": pseudo_csv_sha256,     # [2026-10-22] plc: 읽은 결합기 파일의 해시(실행기가 지금 파일과 대조해 옛 판 결과를 다시 돌린다)
        "parameters_total": n_total, "parameters_trainable": n_train,
        "pretrained_loaded": bool(getattr(model, "pretrained_loaded", False)),
        "pretrained_source": str(getattr(model, "source", "")),
        "best_epoch": result["best_epoch"], "best_val_loss": result["best_val_loss"],
        "epochs_run": result["epochs_run"], "stopped_early": result["stopped_early"],
        "train_seconds": round(elapsed, 1), "device": str(device),
        "index_sha1": index_fingerprint(),          # [2026-09-17 / 2차 검수 C-3] 어느 데이터 판으로 학습했는지(재개 판정용)
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(paths["meta"]), exist_ok=True)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 학습 끝: {elapsed / 60:.1f}분, best epoch {result['best_epoch']}, meta -> {paths['meta']}")


if __name__ == "__main__":
    main()
