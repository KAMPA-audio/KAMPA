#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""비교모델 평가 — 체크포인트 하나를 val 또는 test 에 돌려 Mark5_EXP 와 같은 형식의 지표 파일을 남긴다.

[신설 2026-09-05 / 개정 2026-09-05 저녁]

    plots/performance_summary_<run_tag>.csv        (test)
    plots/performance_summary_<run_tag>_val.csv    (val)
    plots/performance_summary_<run_tag>_eval_far_heldout.csv 등  (평가 전용 세트, 2026-09-17 10절 — open-set ①)
    plots/prediction_details_<...>.csv · confusion_matrix_<...>.png · roc_curve_<...>.png

예)  python eval.py --cond sup --seed 1 --lr 5e-5 --split val
     python eval.py --cond sup --seed 1 --lr 5e-5 --split test
     python eval.py --cond sup --seed 1 --lr 5e-5 --split eval_far_heldout    # open-set ① (Others FPR = 1 − ①)
     python eval.py --cond fixmatch --seed 1 --lr 1e-4 --labeled_n 200 --split test   # [2026-10-07] 표준 반지도(표 1). MT 는 EMA 선생이 평가된다
     python eval.py --cond sup --seed 1 --split test    # [2026-10-25 / A안] --lr 없이 = 지금 절차 회차(run_tag _adam). 위 --lr 예시는 2026-10-21까지의 산출물용

[변경 2026-10-26] 판정 규칙 = 보정 없는 argmax(Mark5.3 과 같음, common/others_rule.DECISION_POLICY). 2026-09-17~10-26 에는
  val 에서 클래스 편향과 others 규칙을 골라 test 에 걸었으나, Mark5.3 이 10-22 에 보정을 뺀 뒤로 두 쪽 판정이 달라져 기본 경로에서 뺐다.
  --split val   : plots/others_threshold_<run_tag>.json 에 판정 규칙(argmax · 편향 0)과 체크포인트 지문을 적는다.
  그 밖의 split : 그 json 을 읽는다. json 이 없거나 체크포인트 sha256 이 다르거나 옛 val 선택 규칙이면 멈춘다(조용한 기본값 없음).
  --others_threshold argmax|<실수> : 고정 규칙으로 보고 싶을 때(json 을 쓰지 않고 요약 CSV 의 Others Rule 에 fixed 로 표시).
  요약 CSV 끝에 "Others Threshold"·"Others Rule" 행이 붙는다(aggregate_baseline·openset_report 가 읽는 이름).
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.paths import CLASS_NAMES, default_data_root, resolve_csv   # noqa: E402
from common.data import ClipDataset, collate, read_index                # noqa: E402
from common.naming import run_tag, artifact_paths, SSL_METHODS          # noqa: E402
from common.seed import set_seed                                        # noqa: E402
from common.metrics import predict, save_evaluation                     # noqa: E402
from common.others_rule import (describe, load_rule_json, predict_rule,  # noqa: E402
                                rule_json_path, apply_class_bias, bias_from_saved, describe_bias,
                                DECISION_POLICY, save_argmax_json)   # [2026-10-26] 판정 규칙 = 보정 없는 argmax
from model_def import MODEL_TAG, DISPLAY_NAME, build_model              # noqa: E402
from train import _subsample                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=f"{DISPLAY_NAME} 평가")
    # [2026-10-07 / 표 1] 표준 반지도 기준선도 같은 방식으로 평가한다 — 체크포인트 model_state 가 평가 모델(MT = EMA 선생)이다.
    ap.add_argument("--cond", choices=["sup", "pl", "plc", "ceil", *SSL_METHODS], required=True)   # [2026-10-22] plc = 결합기 한 라벨
    ap.add_argument("--seed", type=int, required=True)
    # [변경 2026-10-25 / A안] 안 주면 지금 절차 회차(_adam)를 찾는다. 숫자는 2026-10-21까지의 산출물(_lr<lr>)을 다시 평가할 때만 준다.
    ap.add_argument("--lr", type=float, default=None, help="없으면 지금 절차 회차(_adam). 숫자 = 2026-10-21까지의 회차(_lr<lr>)")
    # [변경 2026-09-17 / 10절] val·test 외에 평가 전용 세트도 받는다: eval_far_heldout · eval_boundary_B
    # (dataset_eval_*.csv 는 KAMPA/preprocessing/generate_eval_index.py 가 만들고(2026-11-17 Mark5.2 에서 옮김), 네 인덱스처럼 이 폴더 루트에 복사해 둔다).
    # 정답이 전부 others 인 세트라 Others FPR = 1 − "학습에 없던 소리를 others 로 보낸 비율(open-set ①)" 이다.
    ap.add_argument("--split", type=str, default="test",
                    help="test | val | eval_far_heldout | eval_boundary_B (dataset_<split>.csv 를 읽는다)")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_preload", action="store_true")
    ap.add_argument("--allow_missing_audio", action="store_true")
    ap.add_argument("--labeled_n", type=int, default=None, help="[2026-09-17] 학습 때 준 값과 같게(_l<n> 태그)")
    ap.add_argument("--unlabeled_n", type=int, default=None, help="[2026-09-17] 학습 때 준 값과 같게(_u<n> 태그)")
    ap.add_argument("--max_clips", type=int, default=0, help="동작 확인용 (train.py 와 같이 주면 _smoke 태그를 읽는다)")
    ap.add_argument("--others_threshold", type=str, default=None,
                    help="[2026-09-17] 고정 규칙: 'argmax' 또는 direct 임계값(예 0.09). 주지 않으면 val 선택 json 을 쓴다.")
    args = ap.parse_args()

    smoke = args.max_clips > 0
    tag = run_tag(MODEL_TAG, args.cond, args.lr, args.seed, smoke=smoke, labeled_n=args.labeled_n, unlabeled_n=args.unlabeled_n)
    paths = artifact_paths(ROOT, tag)
    out_tag = tag if args.split == "test" else f"{tag}_{args.split}"        # test → <tag>, val → <tag>_val, 평가 세트 → <tag>_eval_<세트>
    data_root = args.data_root or default_data_root()
    set_seed(args.seed, deterministic=True)          # 증강·dropout 이 없어 논리적으로는 무관하지만 GPU 끝자리까지 고정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(paths["ckpt"]):
        raise SystemExit(f"[ERROR] 체크포인트가 없습니다: {paths['ckpt']} — train.py 를 먼저 돌리십시오.")
    print(f"[INFO] 평가: {DISPLAY_NAME}, run_tag={tag}, split={args.split}, device={device}")

    rows = read_index(resolve_csv(f"dataset_{args.split}.csv"))
    if smoke:
        rows = _subsample(rows, args.max_clips, args.seed)
        print(f"[WARN] --max_clips {args.max_clips}: 동작 확인용 축소 실행입니다.")
    ds = ClipDataset(rows, data_root, args.split, preload=not args.no_preload, strict=not args.allow_missing_audio)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = build_model(num_classes=len(CLASS_NAMES), load_pretrained=False)
    ckpt = torch.load(paths["ckpt"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"[INFO] 체크포인트: epoch {ckpt.get('epoch')} (val loss {ckpt.get('val_loss', float('nan')):.6f})")

    labels, _argmax_preds, probs, names = predict(model, loader, device)
    print(f"[INFO] 유효 {args.split} 샘플: {len(labels)}개")

    # ---- others 규칙(운영점) + [2026-09-22 / A] 클래스별 로짓 편향 ----
    # 편향은 제안모델(Mark5.2 eval.py)과 같은 절차로 val 이 고르고 json 에 적는다. 확률에 편향을 건 뒤 others 규칙을 적용한다.
    # AUC 와 PreBias_Prob_* 열은 편향 전 확률(모델 자체의 순위)로 남긴다(제안모델과 같음). 고정 규칙(--others_threshold)은 편향도 0.
    plot_dir = os.path.join(ROOT, "plots")
    rule_json = rule_json_path(plot_dir, tag)
    probs_raw = np.asarray(probs, dtype=np.float64)
    bias = np.zeros(len(CLASS_NAMES))
    if args.others_threshold is not None:
        if args.others_threshold.strip().lower() == "argmax":
            rule, thr = "argmax", None
        else:
            rule, thr = "direct", float(args.others_threshold)
        rule_info = {"rule": rule, "threshold": thr, "mode": "fixed"}
        print(f"[INFO] others 규칙: 고정 {describe(rule, thr)} (--others_threshold, json 을 쓰지 않음, 클래스 편향 0)")
    elif args.split == "val":
        # [2026-10-26] 판정 규칙 = 보정 없는 argmax(Mark5.3 과 같음). val 은 고르는 것 없이 이 규칙을 json 에 적기만 한다.
        rule, thr = "argmax", None
        # [2026-10-26] 이 회차의 옛 val·test 결과(성능 요약·예측 CSV)를 json 을 쓰기 전에 지운다. 학습 없이 다시 평가할 때 val 이 json 을
        # 지금 규칙으로 바꾼 뒤 test 가 끝나기 전에 끊기면, 옛 규칙의 test 요약이 남아 실행기가 '끝남' 으로 잡던 구멍을 막는다.
        for _old in (paths["val"], paths["test"], os.path.join(plot_dir, f"prediction_details_{tag}_val.csv"),
                     os.path.join(plot_dir, f"prediction_details_{tag}.csv")):
            if os.path.exists(_old):
                os.remove(_old)
                print(f"[INFO] 옛 평가 결과 삭제(다시 평가): {os.path.basename(_old)}")
        save_argmax_json(rule_json, tag, len(labels), paths["ckpt"], CLASS_NAMES)
        rule_info = {"rule": rule, "threshold": thr, "mode": DECISION_POLICY}
        print(f"[INFO] 판정 규칙: 보정 없는 argmax({DECISION_POLICY}, 클래스 편향 0) -> {rule_json}")
    else:
        rule, thr, _saved = load_rule_json(rule_json, tag, paths["ckpt"])
        if _saved.get("decision_policy") != DECISION_POLICY or rule != "argmax":
            raise SystemExit(f"[ERROR] {rule_json} 은 옛 val 선택 규칙(클래스 편향·others 규칙)입니다 — eval.py --split val 을 먼저 다시 "
                             f"돌리십시오(판정 규칙 {DECISION_POLICY}, run_experiments.py 는 이 순서로 다시 평가합니다).")
        bias = bias_from_saved(_saved, CLASS_NAMES)
        rule_info = {"rule": rule, "threshold": thr, "mode": DECISION_POLICY, "from": rule_json}
        print(f"[INFO] 판정 규칙: 보정 없는 argmax({DECISION_POLICY}) — {rule_json}, 체크포인트 지문 일치")
    probs_b = apply_class_bias(probs_raw, bias)
    preds = predict_rule(probs_b, CLASS_NAMES, rule, thr)
    save_evaluation(labels, preds, probs_b, names, plot_dir, out_tag, rule_info=rule_info, probs_prebias=probs_raw)


if __name__ == "__main__":
    main()
