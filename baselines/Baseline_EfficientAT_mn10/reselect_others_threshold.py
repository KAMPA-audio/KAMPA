#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""비교모델의 others 판정 규칙(운영점)을 회차마다 val 에서 고르고, 재학습·평가 재실행 없이 결과 파일을 다시 쓴다.

★ [2026-09-17 / 검수 P1, B안 이후 미사용] 이제 eval.py 가 --split val 에서 직접 규칙을 고르고(common/others_rule.py, 같은 후보·같은 선택),
  test·eval_*·zeroshot_eval.py 가 그 json 을 읽는다(체크포인트 sha256 대조). 이 스크립트는 전량 회차(_lr<lr>_s<k>)의 09-07 옛 산출물을
  사후 재처리하던 것으로, 새 흐름(코랩 실행순서)에서는 돌리지 않는다. 09-07 결과 재현용으로만 남긴다.
  (이 스크립트가 만든 옛 others_threshold_*.json 은 checkpoint 가 null 이라 새 eval.py 가 읽으면 지문 불일치로 멈춘다 — val 을 다시 돌리면 덮어쓴다.)

[신설 2026-09-07, code-reviewer 지적 1 반영] 왜 필요했나.
  비교모델은 softmax argmax(조건표 2절)로만 판정하고, 제안모델(Mark5_EXP)은 others direct 규칙의 임계값을
  회차마다 val 에서 고르게 바꿨다. 같은 표에 "규칙을 val 로 고른 모델"과 "고정 규칙 모델"이 나란히 실리면 기울어진다
  (u2000 에서 val 튜닝이 argmax 대비 Macro F1 +0.046 을 벌어 준 실측). 네 모델 모두 **같은 후보·같은 절차**로
  운영점을 val 에서 고른다.

후보와 규칙 — Mark5_EXP2/vild/postprocess_utils.select_others_threshold 와 같은 정의(여기서는 vild 를 물지 않으려고
같은 논리를 다시 적었다. 한쪽을 고치면 다른 쪽도 고칠 것).
  argmax        9클래스 argmax(others 강제 없음). 지금까지의 비교모델 규칙.
  direct(thr)   Prob_others >= thr 이면 others, 아니면 others 를 빼고 8종 argmax. 격자 0.02~0.95/0.01.
  선택: val Macro F1 최대. 동률 격자점은 가운데, argmax 가 동률이면 argmax.
  ROC AUC 는 규칙과 무관한 softmax 확률로 잰다(원래 그랬음).

다시 쓰는 것 (본실험 회차 = <model>_<cond>_lr<lr>_s<1~5>, val·test 둘 다. 학습률 선정 seed 0 은 val 만 있어 제외)
  performance_summary_<tag>[_val].csv   요약 블록 끝에 "Others Threshold"·"Others Rule" 행 추가
  prediction_details_<tag>[_val].csv    Predicted Label 갱신(Prob_*·Top Confidence 는 그대로)
  confusion_matrix / roc_curve PNG      다시 그림
  others_threshold_<tag>.json           고른 규칙·임계값·val F1
  others_threshold_selected_<model>.csv 회차별 표(기존 표와 run_tag 기준 병합)
  원본은 plots/backup_rule_argmax/ 에 첫 실행 때만 보관.

검증: --check 는 파일에 적힌 규칙(행이 없으면 argmax)으로 다시 계산해 파일과 같은지만 본다(쓰기 없음).
사용:
  python reselect_others_threshold.py --check
  python reselect_others_threshold.py --dry_run
  python reselect_others_threshold.py
  이어서: python aggregate_baseline.py --split test / --split val
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model_info import MODEL_TAG  # noqa: E402

PLOT_DIR = os.path.join(ROOT, "plots")
PRED_PREFIX = "prediction_details_"
SUMMARY_METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
                   "ROC AUC (Macro, OvR)", "Others FPR"]
GRID = (0.02, 0.95, 0.01)
TAG_RE = re.compile(rf"^{re.escape(MODEL_TAG)}_(?P<cond>sup|pl|ceil)_lr(?P<lr>[0-9.e+-]+)_s(?P<seed>\d+)$")


# ── 규칙 (Mark5 apply_others_calibration direct/argmax 와 같은 정의) ─────────────────────────
def predict_rule(probs, class_names, rule, thr):
    oi = class_names.index("others")
    if rule == "argmax":
        return probs.argmax(axis=1), np.zeros(len(probs), dtype=bool)
    if rule != "direct":
        raise ValueError(f"모르는 규칙 {rule!r}")
    forced = probs[:, oi] >= float(thr)
    masked = probs.copy()
    masked[:, oi] = -1.0
    preds = masked.argmax(axis=1)
    preds[forced] = oi
    return preds, forced


def macro_f1(labels, preds, C):
    _, _, f, _ = precision_recall_fscore_support(labels, preds, average="macro", labels=list(range(C)), zero_division=0)
    return float(f)


def select_rule(probs, labels, class_names):
    """(rule, thr, best_f1, curve). select_others_threshold 와 같은 선택 규칙."""
    C = len(class_names)
    start, stop, step = GRID
    grid = [float(t) for t in np.round(np.arange(start, stop + 1e-9, step), 4)]
    f1_arg = macro_f1(labels, predict_rule(probs, class_names, "argmax", None)[0], C)
    f1s = [macro_f1(labels, predict_rule(probs, class_names, "direct", t)[0], C) for t in grid]
    curve = [("argmax", None, f1_arg)] + [("direct", t, f) for t, f in zip(grid, f1s)]
    best = max(f1s)
    if f1_arg >= best - 1e-12:
        return "argmax", None, f1_arg, curve
    top = [i for i, f in enumerate(f1s) if f >= best - 1e-12]
    mid = top[len(top) // 2]
    return "direct", grid[mid], f1s[mid], curve


def compute_metrics(labels, preds, probs, class_names, auc_probs=None):
    C = len(class_names)
    idx = list(range(C))
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=idx, zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(labels, preds, average="macro", labels=idx, zero_division=0)
    try:
        # [2026-09-22] auc_probs(편향 전 확률)가 있으면 그것으로 — common/metrics.py 의 요약 AUC 와 같은 자.
        auc_m = roc_auc_score(labels, probs if auc_probs is None else auc_probs, multi_class="ovr", average="macro", labels=idx)
    except ValueError:
        auc_m = float("nan")
    cm = confusion_matrix(labels, preds, labels=idx)
    oi = class_names.index("others")
    s = int(cm[oi].sum())
    fpr = float((s - cm[oi][oi]) / s) if s > 0 else float("nan")
    return {"Accuracy": acc, "Macro Precision": mp, "Macro Recall": mr, "Macro F1": mf1,
            "ROC AUC (Macro, OvR)": auc_m, "Others FPR": fpr, "per_class": (p, r, f1), "cm": cm}


# ── 파일 ─────────────────────────────────────────────────────────────────────
def find_tags():
    tags = []
    for p in sorted(glob.glob(os.path.join(PLOT_DIR, f"{PRED_PREFIX}{MODEL_TAG}_*_val.csv"))):
        tag = os.path.basename(p)[len(PRED_PREFIX):-len("_val.csv")]
        m = TAG_RE.match(tag)
        if not m or "_smoke" in tag or int(m.group("seed")) == 0:
            continue
        if os.path.exists(os.path.join(PLOT_DIR, f"{PRED_PREFIX}{tag}.csv")):
            tags.append(tag)
        else:
            print(f"[WARN] {tag}: val 은 있는데 test 예측 CSV 가 없어 건너뜁니다.")
    return tags


def load_pred(tag, split, allow_bias=False):
    path = os.path.join(PLOT_DIR, f"{PRED_PREFIX}{tag}{'' if split == 'test' else '_' + split}.csv")
    df = pd.read_csv(path, encoding="utf-8")
    from common.paths import LEGACY_CLASS_NAMES, canonical_column   # [2026-11-18] 끝난 회차의 예측 CSV 는 옛 이름 talking
    df = df.rename(columns=canonical_column)
    for col in ("True Label", "Predicted Label"):
        if col in df.columns:
            df[col] = df[col].replace(LEGACY_CLASS_NAMES)
    class_names = [c[len("Prob_"):] for c in df.columns if c.startswith("Prob_")]
    probs = df[[f"Prob_{c}" for c in class_names]].to_numpy(dtype=np.float64)
    # [수정 2026-09-22 / 검수] 2026-09-22 부터 eval.py 는 val 이 고른 클래스 편향을 건 확률을 Prob_* 에, 편향 전 확률을
    # PreBias_Prob_* 에 쓴다. 이 도구는 편향 개념이 없어(json 에 class_bias 를 안 쓰고 AUC 도 Prob_* 로 잼) 새 산출물에
    # 쓰기 모드로 돌리면 편향이 조용히 0 으로 되돌아간다 → 쓰기·dry_run 은 막고(--split val 로 다시 고르라고 안내),
    # --check 만 편향 전 확률로 AUC 를 재계산해 요약과 대조한다.
    pre_cols = [f"PreBias_Prob_{c}" for c in class_names]
    has_bias = all(c in df.columns for c in pre_cols)
    if has_bias and not allow_bias:
        raise SystemExit(f"[ERROR] {os.path.basename(path)} 은 클래스 편향(A)이 걸린 새 산출물(PreBias_Prob_* 열)입니다. "
                         f"이 도구(2026-09-07 옛 산출물 이식용)로 규칙을 다시 고르면 편향이 사라집니다 — "
                         f"eval.py --split val 로 다시 고르십시오(--check 는 됩니다).")
    probs_prebias = df[pre_cols].to_numpy(dtype=np.float64) if has_bias else None
    unknown = sorted(set(df["True Label"]) - set(class_names))
    if unknown:
        raise SystemExit(f"[ERROR] {path}: 클래스 밖 라벨 {unknown}")
    labels = np.array([class_names.index(l) for l in df["True Label"]], dtype=int)
    return path, df, class_names, probs, labels, probs_prebias


def read_existing_summary(path):
    out, per_class, rule, thr = {}, {}, None, None
    in_class = False
    with open(path, encoding="utf-8", newline="") as f:
        for parts in csv.reader(f):
            if not parts or parts[0].startswith("#") or not parts[0].strip():
                continue
            if parts[0] == "Class":
                in_class = True
                continue
            if parts[0] == "Metric":
                continue
            if in_class:
                per_class[parts[0]] = tuple(float(x) for x in parts[1:4])
            elif parts[0] == "Others Rule":
                rule = parts[1].strip()
            elif parts[0] == "Others Threshold":
                try:
                    thr = float(parts[1])
                except ValueError:
                    thr = None
            else:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    if rule is None:
        rule = "direct" if thr is not None else "argmax"
    return out, per_class, rule, thr


def write_summary(path, out_tag, m, class_names, rule, thr):
    p, r, f1 = m["per_class"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Performance Summary for {out_tag}\n\n")
        pd.DataFrame({
            "Metric": SUMMARY_METRICS + ["Others Threshold", "Others Rule"],
            "Score": [m[k] for k in SUMMARY_METRICS] + [float(thr) if rule == "direct" else float("nan"), rule],
        }).to_csv(f, index=False)
        f.write("\n# Class-wise Metrics\n\n")
        pd.DataFrame({"Class": class_names, "Precision": p, "Recall": r, "F1-Score": f1}).to_csv(f, index=False)


def write_confusion_png(path, out_tag, cm, class_names):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 12})
    plt.title(f"Confusion Matrix - {out_tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_roc_png(path, out_tag, labels, probs, class_names):
    plt.figure(figsize=(7, 6))
    labels = np.asarray(labels)
    for i, cname in enumerate(class_names):
        fpr_c, tpr_c, _ = roc_curve((labels == i).astype(int), probs[:, i])
        plt.plot(fpr_c, tpr_c, label=f"{cname} AUC={auc(fpr_c, tpr_c):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({out_tag})")
    plt.legend(loc="lower right", fontsize=8); plt.grid(True); plt.tight_layout()
    plt.savefig(path)
    plt.close()


def backup(paths, backup_dir):
    os.makedirs(backup_dir, exist_ok=True)
    for p in paths:
        if os.path.exists(p):
            dst = os.path.join(backup_dir, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--fixed", type=str, default=None, help="val 선택 대신 이 규칙: 실수(direct 임계값) 또는 argmax")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    tags = args.tags or find_tags()
    if not tags:
        raise SystemExit(f"[ERROR] {PLOT_DIR} 에 본실험 예측 CSV 가 없습니다.")
    mode = "check" if args.check else ("dry_run" if args.dry_run else "write")
    print(f"[INFO] {MODEL_TAG} 회차 {len(tags)}개 · 모드 {mode} · 후보 argmax + direct {GRID[0]:g}~{GRID[1]:g}/{GRID[2]:g}")
    backup_dir = os.path.join(PLOT_DIR, "backup_rule_argmax")
    rows, n_fail = [], 0
    for tag in tags:
        if args.check:
            ok = True
            for split in ("val", "test"):
                out_tag = tag if split == "test" else f"{tag}_{split}"
                spath = os.path.join(PLOT_DIR, f"performance_summary_{out_tag}.csv")
                old, old_pc, rule, thr = read_existing_summary(spath)
                _, df, class_names, probs, labels, probs_prebias = load_pred(tag, split, allow_bias=True)
                preds, _ = predict_rule(probs, class_names, rule, thr)
                m = compute_metrics(labels, preds, probs, class_names, auc_probs=probs_prebias)
                diffs = {k: abs(old.get(k, np.nan) - m[k]) for k in SUMMARY_METRICS
                         if not (np.isnan(old.get(k, np.nan)) and np.isnan(m[k]))}
                pc_diff = max(abs(old_pc[c][j] - m["per_class"][j][i]) for i, c in enumerate(class_names) for j in range(3))
                lab_mism = int((np.array([class_names[i] for i in preds]) != df["Predicted Label"].to_numpy()).sum())
                worst = max(diffs.values()) if diffs else 0.0
                good = worst < 1e-9 and pc_diff < 1e-9 and lab_mism == 0
                ok &= good
                print(f"  {'PASS' if good else 'FAIL'}  {out_tag:<36} 규칙 {rule}{'' if thr is None else f' {thr:g}'} · "
                      f"지표 최대차 {worst:.2e} · 클래스별 {pc_diff:.2e} · 예측 불일치 {lab_mism}")
            n_fail += 0 if ok else 1
            continue

        vpath, vdf, class_names, vprobs, vlabels, _ = load_pred(tag, "val")
        tpath, tdf, cn2, tprobs, tlabels, _ = load_pred(tag, "test")
        if cn2 != class_names:
            raise SystemExit(f"[ERROR] {tag}: val/test 클래스 열 순서가 다릅니다.")
        if args.fixed is not None:
            rule, thr, vf1 = ("argmax", None, None) if args.fixed.strip().lower() == "argmax" else ("direct", float(args.fixed), None)
            selected_on = "fixed"
        else:
            rule, thr, vf1, _ = select_rule(vprobs, vlabels, class_names)
            selected_on = "val"
        out = {}
        for split, (path, df, probs, labels) in (("val", (vpath, vdf, vprobs, vlabels)), ("test", (tpath, tdf, tprobs, tlabels))):
            preds, forced = predict_rule(probs, class_names, rule, thr)
            out[split] = (path, df, preds, compute_metrics(labels, preds, probs, class_names), probs, labels)
        mt, mv = out["test"][3], out["val"][3]
        shown = f"direct {thr:.2f}" if rule == "direct" else "argmax"
        rows.append({"run_tag": tag, "rule": rule, "threshold": thr, "selected_on": selected_on, "val_macro_f1_at_threshold": vf1,
                     **{f"val {k}": mv[k] for k in SUMMARY_METRICS}, **{f"test {k}": mt[k] for k in SUMMARY_METRICS}})
        print(f"  {tag:<32} argmax -> {shown:<12} | test Acc {mt['Accuracy']:.4f} P {mt['Macro Precision']:.4f} "
              f"R {mt['Macro Recall']:.4f} F1 {mt['Macro F1']:.4f} AUC {mt['ROC AUC (Macro, OvR)']:.4f} othersFPR {mt['Others FPR']:.4f}")
        if args.dry_run:
            continue
        for split, (path, df, preds, m, probs, labels) in out.items():
            out_tag = tag if split == "test" else f"{tag}_{split}"
            spath = os.path.join(PLOT_DIR, f"performance_summary_{out_tag}.csv")
            cpath = os.path.join(PLOT_DIR, f"confusion_matrix_{out_tag}.png")
            rpath = os.path.join(PLOT_DIR, f"roc_curve_{out_tag}.png")
            backup([spath, path, cpath, rpath], backup_dir)
            df = df.copy()
            df["Predicted Label"] = [class_names[i] for i in preds]
            df.to_csv(path, index=False, encoding="utf-8")
            write_summary(spath, out_tag, m, class_names, rule, thr)
            write_confusion_png(cpath, out_tag, m["cm"], class_names)
            write_roc_png(rpath, out_tag, labels, probs, class_names)
        with open(os.path.join(PLOT_DIR, f"others_threshold_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": "val_select" if selected_on == "val" else "fixed", "rule": rule, "threshold": thr,
                       "selected_on": selected_on, "criterion": "macro_f1", "grid": list(GRID),
                       "candidates": "argmax + direct grid", "val_macro_f1_at_threshold": vf1,
                       "n_val_clips": int(len(vlabels)), "previous_rule": "argmax", "run_tag": tag,
                       "checkpoint": None, "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "note": "재학습·평가 재실행 없이 prediction_details CSV 로 재계산"}, f, ensure_ascii=False, indent=2)

    if args.check:
        print(f"\n[CHECK] 회차 {len(tags)}개 중 불일치 {n_fail}개 -> {'전부 재현됨' if n_fail == 0 else '재현 실패 있음'}")
        sys.exit(0 if n_fail == 0 else 1)
    if rows and not args.dry_run:
        table = os.path.join(PLOT_DIR, f"others_threshold_selected_{MODEL_TAG}.csv")
        new = pd.DataFrame(rows)
        if os.path.exists(table):
            old = pd.read_csv(table, encoding="utf-8-sig")
            new = pd.concat([old[~old["run_tag"].isin(new["run_tag"])], new], ignore_index=True).sort_values("run_tag").reset_index(drop=True)
        new.to_csv(table, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] 회차 {len(rows)}개 다시 씀 · 선택표 {table}({len(new)}행) · 원본 백업 {backup_dir}")
        print("[INFO] 이어서: python aggregate_baseline.py --split test / --split val")


if __name__ == "__main__":
    main()
