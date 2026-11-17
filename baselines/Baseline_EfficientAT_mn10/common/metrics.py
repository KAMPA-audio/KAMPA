# -*- coding: utf-8 -*-
"""평가 지표 — Mark5_EXP model/eval.py(2026-08-30 판)의 지표 계산과 저장 형식을 그대로 옮긴 것.

[신설 2026-09-05] 조건표 2절 '지표'. 계산 코드가 같아야 표에서 같은 지표라고 말할 수 있다.

저장 형식(performance_summary_<tag>.csv)은 Mark5_EXP 의 aggregate_results.py 가 그대로 읽을 수
있게 행 이름·순서를 바꾸지 않는다:
    Accuracy · Macro Precision · Macro Recall · Macro F1 · ROC AUC (Macro, OvR) · Others FPR
others 규칙: [2026-09-17 / 검수 P1, B안] eval.py 가 val 에서 고른 규칙(argmax 또는 direct 임계값, common/others_rule.py)으로
판정한 preds 를 받는다. predict() 자체는 softmax argmax 를 돌려주고, 규칙 적용은 eval.py 가 한다.
요약 CSV 끝의 "Others Threshold"·"Others Rule" 행이 어느 규칙이었는지 적는다(rule_info).
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .paths import CLASS_NAMES


@torch.no_grad()
def predict(model, loader, device):
    """loader 전체를 돌려 (정답 인덱스, 예측 인덱스, 확률 [N, C], 파일명) 을 돌려준다."""
    model.eval()
    labels, probs, names = [], [], []
    for waves, y, metas in loader:
        logits = model(waves.to(device), augment=False)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.extend(int(v) for v in y)
        names.extend(os.path.basename(m["path"]) for m in metas)
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(CLASS_NAMES)), dtype=np.float32)
    preds = probs.argmax(axis=1).tolist() if len(probs) else []
    return labels, preds, probs, names


def save_evaluation(all_labels, all_preds, all_probs, filenames, plot_dir, out_tag, class_names=CLASS_NAMES, rule_info=None,
                    probs_prebias=None):
    """[2026-09-22 / A] all_probs 는 클래스 편향을 건 확률(판정에 쓴 것). probs_prebias 를 주면 ROC AUC 는 그것(편향 전, 모델 자체의
    순위)으로 재고 CSV 에 PreBias_Prob_* 열을 같이 남긴다 — 제안모델 eval.py 와 같은 규칙. 안 주면 옛 동작(all_probs 로 AUC)."""
    os.makedirs(plot_dir, exist_ok=True)
    n = len(class_names)
    labels_idx = list(range(n))
    if len(all_labels) == 0:
        raise ValueError("[ERROR] 평가 샘플이 0개입니다.")

    print(classification_report(all_labels, all_preds, labels=labels_idx, target_names=class_names,
                                digits=4, zero_division=0))
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds, labels=labels_idx)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 12})
    plt.title(f"Confusion Matrix - {out_tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{out_tag}.png"))
    cm_df.to_csv(os.path.join(plot_dir, f"confusion_matrix_{out_tag}.csv"), encoding="utf-8")   # [2026-09-17 / 7절] 혼동 칸을 숫자로(행 = 정답, 열 = 판정)
    plt.close()

    all_probs_arr = np.asarray(all_probs)
    auc_probs_arr = np.asarray(probs_prebias) if probs_prebias is not None else all_probs_arr   # [2026-09-22] AUC 는 편향 전 확률
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=labels_idx, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", labels=labels_idx, zero_division=0)
    try:
        roc_auc_macro = roc_auc_score(all_labels, auc_probs_arr, multi_class="ovr", average="macro",
                                      labels=labels_idx)
        print(f"ROC AUC (macro, one-vs-rest): {roc_auc_macro:.4f}")
    except ValueError as e:
        roc_auc_macro = float("nan")
        print(f"[WARN] ROC AUC 계산 불가(평가셋에 없는 클래스가 있을 수 있음): {e}")

    # others FPR = 실제 others 인데 타겟 8종으로 오분류된 비율 (= 1 - Recall(others)). eval.py 와 같은 정의.
    others_idx = class_names.index("others")
    others_row = cm[others_idx]
    others_row_sum = int(others_row.sum())
    others_fpr = float((others_row_sum - others_row[others_idx]) / others_row_sum) if others_row_sum > 0 else float("nan")
    print(f"[others FPR] 실제 others 인데 타겟 클래스로 오분류된 비율: {others_fpr:.4f}")

    plt.figure(figsize=(7, 6))
    all_labels_arr = np.asarray(all_labels)
    for i, cname in enumerate(class_names):
        fpr_c, tpr_c, _ = roc_curve((all_labels_arr == i).astype(int), auc_probs_arr[:, i])   # [2026-09-22] ROC 도 편향 전 확률
        plt.plot(fpr_c, tpr_c, label=f"{cname} AUC={auc(fpr_c, tpr_c):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({out_tag})")
    plt.legend(loc="lower right", fontsize=8); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"roc_curve_{out_tag}.png"))
    plt.close()

    # [2026-10-26] 예측 CSV 를 성능 요약보다 먼저 쓴다 — 실행기는 성능 요약을 완료 표식으로 보므로, 요약만 새것이고 CSV 는 옛것인
    # 상태(쓰는 도중 끊김)가 '끝남' 으로 잡히지 않게 한다.
    # 샘플별 확률 — 나중에 운영점·규칙을 바꿔 볼 때 재실행 없이 CSV 만으로 계산하기 위한 것.
    rows = []
    for k, (name, y, p, pr) in enumerate(zip(filenames, all_labels, all_preds, all_probs_arr)):
        row = {"Filename": name, "True Label": class_names[y], "Predicted Label": class_names[p],
               "Top Confidence": float(pr.max()),
               **{f"Prob_{c}": float(pr[i]) for i, c in enumerate(class_names)}}
        if probs_prebias is not None:      # [2026-09-22 / A] 편향 전 확률(제안모델 CSV 의 PreBias_Prob_* 와 같은 열)
            row.update({f"PreBias_Prob_{c}": float(auc_probs_arr[k][i]) for i, c in enumerate(class_names)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(plot_dir, f"prediction_details_{out_tag}.csv"),
                              index=False, encoding="utf-8")
    # ★ 행 이름·순서 고정 — Mark5_EXP aggregate_results.py 와 make_paper_tables.py 가 이 이름으로 읽는다.
    summary_csv = os.path.join(plot_dir, f"performance_summary_{out_tag}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Performance Summary for {out_tag}\n\n")
        metric_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "ROC AUC (Macro, OvR)", "Others FPR"]
        scores = [accuracy, macro_precision, macro_recall, macro_f1, roc_auc_macro, others_fpr]
        if rule_info is not None:
            # [2026-09-17] reselect_others_threshold.py 가 쓰던 것과 같은 두 행. aggregate_baseline 은 "Others Rule" 문자열 행을 건너뛰고,
            # Mark5_EXP2/model/openset_report.py 는 이 두 행으로 표 7 의 규칙 열을 채운다.
            _rule = rule_info.get("rule", "argmax")
            _thr = rule_info.get("threshold")
            metric_names += ["Others Threshold", "Others Rule"]
            scores += [float(_thr) if (_rule == "direct" and _thr is not None) else float("nan"),
                       _rule if rule_info.get("mode", "val_select") != "fixed" else f"{_rule} (fixed)"]
            if rule_info.get("class_bias") is not None:
                # [2026-09-22 / A] 문자열 행. 읽는 쪽: run_experiments.read_summary(aggregate_baseline 도 이것을 씀)는 "Others Rule"·"Class Bias"
                # 두 이름만 건너뛰고 그 밖의 문자열은 깨진 파일로 보므로(검수로 잡혀 2026-09-22 에 "Class Bias" 를 추가), 문자열 행을 더
                # 붙일 때는 그쪽 목록도 같이 고친다. Mark5_EXP2 aggregate_results._read_summary 는 float 변환 실패 행을 전부 건너뛴다.
                metric_names += ["Class Bias"]
                scores += [str(rule_info["class_bias"])]
        pd.DataFrame({"Metric": metric_names, "Score": scores}).to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        pd.DataFrame({"Class": class_names, "Precision": precision, "Recall": recall, "F1-Score": f1}).to_csv(f, index=False)
    print(f"[INFO] 성능 요약 CSV 저장: {summary_csv}")

    return {"Accuracy": accuracy, "Macro Precision": macro_precision, "Macro Recall": macro_recall,
            "Macro F1": macro_f1, "ROC AUC (Macro, OvR)": roc_auc_macro, "Others FPR": others_fpr}
