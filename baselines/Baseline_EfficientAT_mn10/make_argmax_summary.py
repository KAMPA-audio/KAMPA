#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""비교모델 test 4지표를 보정 없는 argmax 로 다시 계산한 표 — 논문 결과물(Mark5.3 make_four_results.py)과 같은 판정 규칙.

[신설 2026-10-26] 이 저장소 eval.py 는 val 에서 고른 클래스 편향·others 규칙을 test 에 걸어 성능 요약(performance_summary_<tag>.csv)에
남긴다. Mark5.3 은 보정 없는 argmax 라서, 논문 비교에는 test 예측 CSV 의 보정 전 확률(PreBias_Prob_*)로 argmax 를 다시 계산한 값을
쓴다(재학습·재평가 없음). 같은 회차의 파일인지는 예측 CSV 의 Predicted Label 로 다시 낸 4지표가 성능 요약과 같은지로 확인하고,
다르면 멈춘다. 지표 식은 eval.py(common/metrics.py)와 같다: sklearn macro, 9클래스, zero_division=0.

    python make_argmax_summary.py                 # plots/ 의 정답 라벨만(sup) Adam 회차 20개(정답 200·500·1,000·전량 × seed 1~5)

읽는 것  plots/performance_summary_<tag>.csv · plots/prediction_details_<tag>.csv   (<tag> = <MODEL_TAG>_sup_adam[_l<n>]_s<seed>)
쓰는 것  plots/argmax_summary_<MODEL_TAG>_sup_adam_test.csv
         회차마다 정답 라벨 수·seed·클립 수, argmax 4지표, 지금 성능 요약의 4지표, 차이(argmax − 성능 요약)
         (2026-10-26 판정 규칙 argmax_v1 로 다시 평가하기 전의 요약은 val 에서 고른 보정 값, 다시 평가한 뒤에는 차이 0)
"""
import argparse
import csv
import os
import sys

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from model_info import MODEL_TAG          # noqa: E402
from common.naming import parse_tag       # noqa: E402
from common.paths import CLASS_NAMES, canonical_row      # noqa: E402

METRICS = ("Accuracy", "Macro Precision", "Macro Recall", "Macro F1")
LABELED = (200, 500, 1000, None)          # None = 정답 전량(2,000)
SEEDS = (1, 2, 3, 4, 5)


def four_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", labels=list(CLASS_NAMES), zero_division=0)
    return dict(zip(METRICS, (float(accuracy_score(y_true, y_pred)), float(p), float(r), float(f1))))


def read_summary(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] in METRICS:
                out[row[0]] = float(row[1])
    if len(out) != len(METRICS):
        raise SystemExit(f"[STOP] {path}: 4지표가 다 없습니다")
    return out


def main():
    ap = argparse.ArgumentParser(description="비교모델 test 4지표 — 보정 없는 argmax(예측 CSV 의 PreBias_Prob_*)")
    ap.add_argument("--plots", default=os.path.join(ROOT, "plots"), help="성능 요약·예측 CSV 가 있는 폴더(기본: 이 저장소 plots)")
    ap.add_argument("--out", default=None, help="출력 CSV 경로(기본: --plots/argmax_summary_<MODEL_TAG>_sup_adam_test.csv)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.plots, f"argmax_summary_{MODEL_TAG}_sup_adam_test.csv")
    cols = [f"PreBias_Prob_{c}" for c in CLASS_NAMES]
    rows, missing = [], []
    for l in LABELED:
        for s in SEEDS:
            tag = f"{MODEL_TAG}_sup_adam" + ("" if l is None else f"_l{l}") + f"_s{s}"
            parse_tag(tag)                                   # 이름 규칙(common/naming)과 맞는지
            summ_p = os.path.join(args.plots, f"performance_summary_{tag}.csv")
            pred_p = os.path.join(args.plots, f"prediction_details_{tag}.csv")
            if not (os.path.isfile(summ_p) and os.path.isfile(pred_p)):
                missing.append(tag)
                continue
            with open(pred_p, newline="", encoding="utf-8") as f:
                pred = [canonical_row(r) for r in csv.DictReader(f)]   # [2026-11-18] 끝난 회차의 예측 CSV 는 옛 이름 talking
            if not pred or any(c not in pred[0] for c in cols + ["True Label", "Predicted Label"]):
                raise SystemExit(f"[STOP] {pred_p}: 보정 전 확률 열(PreBias_Prob_*)·정답·예측 열이 없습니다")
            y = [r["True Label"] for r in pred]
            reported = read_summary(summ_p)
            again = four_metrics(y, [r["Predicted Label"] for r in pred])
            if any(abs(again[m] - reported[m]) > 1e-6 for m in METRICS):
                raise SystemExit(f"[STOP] {tag}: 예측 CSV 로 다시 낸 값 {again} 이 성능 요약 {reported} 와 다릅니다 — 같은 회차의 파일이 아닙니다")
            arg = four_metrics(y, [max(CLASS_NAMES, key=lambda c, r=r: float(r[f"PreBias_Prob_{c}"])) for r in pred])   # 동점이면 앞 클래스(np.argmax)
            rows.append([tag, 2000 if l is None else l, s, len(pred)]
                        + [f"{arg[m]:.6f}" for m in METRICS] + [f"{reported[m]:.6f}" for m in METRICS]
                        + [f"{arg[m] - reported[m]:+.6f}" for m in METRICS])
    if missing:
        print("\n".join(missing))
        raise SystemExit(f"[STOP] 성능 요약·예측 CSV 가 없는 회차 {len(missing)}개 — 아무것도 쓰지 않았습니다")
    header = (["run_tag", "labeled", "seed", "n_clips"] + [f"argmax {m}" for m in METRICS]
              + [f"reported(summary) {m}" for m in METRICS] + [f"argmax-reported {m}" for m in METRICS])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[OK] {out} — {len(rows)}회 (보정 없는 argmax · 지금 성능 요약 · 차이)")


if __name__ == "__main__":
    main()
