#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_expert_runs.py — 전문가 8개(Mark4.1~4.8)의 두 재학습 회차 성능을 나란히 놓는다 (2026-09-21 신설).

왜: 09-21 others 재분할 뒤 2차 재학습 결과를 1차(09-20, 각 저장소 plots/_run1_20260920/)와 대조해야 한다.
    회차마다 손으로 CSV 를 여는 대신 한 번에 표(md)로 뽑는다. 네 지표(Accuracy·Macro P·Macro R·Macro F1)는 항상 같이 낸다.
무엇을: 저장소마다 plots/<run1>/performance_summary_mark4.N{,_val}.csv 와 plots/performance_summary_mark4.N{,_val}.csv(현재 = 2차)를 읽어
    val·test 각각 Accuracy·Macro P·Macro R·Macro F1·ROC AUC·Others FPR·운영점(threshold)·플래토 폭(target_threshold json)을 표로 만들고,
    test 오분류를 prediction_details 로 세어 FP/FN 개수도 붙인다(출처표는 --provenance 를 주면 FP 파일의 출처를 provenance 에서 찾아 붙인다).
사용: python preprocessing/compare_expert_runs.py [--run1 _run1_20260920] [--experts ../Mark4_series] [--provenance ../data_provenance.xlsx]
      [--out preprocessing/rebuild_cache/expert_run_compare_YYYYMMDD.md]
    2차 결과가 아직 없는 저장소는 "(2차 없음)" 으로 표시하고 건너뛴다.
"""
import os, sys, json, argparse, time, csv
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
METRICS = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "ROC AUC", "Others FPR", "Target Threshold"]
SHORT = {"Accuracy": "Acc", "Macro Precision": "MacP", "Macro Recall": "MacR", "Macro F1": "MacF1",
         "ROC AUC": "AUC", "Others FPR": "OthFPR", "Target Threshold": "thr"}


def read_summary(path):
    """performance_summary CSV 의 Metric,Score 구간을 dict 로. 없으면 None."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] in METRICS:
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


def plateau_width(json_path):
    if not os.path.exists(json_path):
        return None
    try:
        j = json.load(open(json_path, encoding="utf-8"))
        return float(j.get("plateau", {}).get("width", float("nan")))
    except Exception:
        return None


def errors(pred_csv):
    """prediction_details 에서 FP(others→타겟)·FN(타겟→others) 파일명 목록."""
    if not os.path.exists(pred_csv):
        return None, None
    fp, fn = [], []
    with open(pred_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t, p = r.get("True Label"), r.get("Predicted Label")
            if t == p:
                continue
            (fp if t == "others" else fn).append(r["Filename"])
    return fp, fn


def fmt(v, nd=4):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", default="_run1_20260920", help="1차 보관 폴더 이름(plots/ 아래)")
    ap.add_argument("--experts", default=os.path.join(ROOT, "..", "Mark4_series"))
    ap.add_argument("--provenance", default=None, help="주면 test FP 파일의 출처(aihub_src_file 의 source 폴더)를 세어 붙인다")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    prov = None
    if a.provenance:
        import pandas as pd
        p = pd.read_excel(a.provenance)
        p = p[p["removed_20260715"].astype(str) == "active"]
        prov = p

    lines = [f"# 전문가 재학습 대조 — 1차({a.run1}) vs 현재 plots/ ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    lines.append("| 전문가 | split | 회차 | Acc | MacP | MacR | MacF1 | AUC | OthFPR | thr | 플래토폭 | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    deltas = []
    for n in range(1, 9):
        mk = f"mark4.{n}"
        repo = os.path.join(a.experts, f"Mark4.{n}")
        for split, suf in (("val", "_val"), ("test", "")):
            rows = {}
            run1_csv = os.path.join(repo, "plots", a.run1, f"performance_summary_{mk}{suf}.csv")
            for tag, sub in (("1차", a.run1), ("2차", "")):
                pd_ = os.path.join(repo, "plots", sub) if sub else os.path.join(repo, "plots")
                csv_path = os.path.join(pd_, f"performance_summary_{mk}{suf}.csv")
                # [주의] plots/ 바로 아래 summary 가 1차 보관본보다 오래됐으면(예: 08-02 옛 결과) 2차가 아직 없는 것이다.
                if tag == "2차" and os.path.exists(csv_path) and os.path.exists(run1_csv) \
                        and os.path.getmtime(csv_path) <= os.path.getmtime(run1_csv):
                    rows[tag] = None
                    continue
                s = read_summary(csv_path)
                if s is None:
                    rows[tag] = None
                    continue
                w = plateau_width(os.path.join(pd_, f"target_threshold_{mk}.json"))
                fp, fn = errors(os.path.join(pd_, f"prediction_details_{mk}{suf}.csv"))
                rows[tag] = (s, w, fp, fn)
                lines.append(f"| 4.{n} | {split} | {tag} | " + " | ".join(fmt(s.get(m)) for m in METRICS[:6])
                             + f" | {fmt(s.get('Target Threshold'), 2)} | {fmt(w, 3)} | {'-' if fp is None else len(fp)} | {'-' if fn is None else len(fn)} |")
            if rows.get("1차") and rows.get("2차"):
                d = {m: (rows["2차"][0][m] - rows["1차"][0][m]) if (m in rows["2차"][0] and m in rows["1차"][0]) else None for m in METRICS[:6]}
                lines.append(f"| 4.{n} | {split} | Δ(2차−1차) | " + " | ".join("-" if d[m] is None else f"{d[m]:+.4f}" for m in METRICS[:6]) + " | | | | |")
                deltas.append((n, split, d))
            elif rows.get("1차") and not rows.get("2차"):
                lines.append(f"| 4.{n} | {split} | 2차 | (2차 없음 — plots/ 의 summary 가 1차 보관본보다 오래됨) | | | | | | | | | |")
        # test FP 출처표
        if prov is not None:
            for tag, sub in (("1차", a.run1), ("2차", "")):
                pd_ = os.path.join(repo, "plots", sub) if sub else os.path.join(repo, "plots")
                pred = os.path.join(pd_, f"prediction_details_{mk}.csv")
                r1 = os.path.join(repo, "plots", a.run1, f"performance_summary_{mk}.csv")
                if tag == "2차" and os.path.exists(pred) and os.path.exists(r1) and os.path.getmtime(pred) <= os.path.getmtime(r1):
                    continue
                fp, fn = errors(pred)
                if not fp:
                    continue
                sel = prov[(prov["mark_version"] == mk) & (prov["assigned_split"] == "test") & (prov["local_filename"].isin(fp))]
                src = Counter(str(x).split("/")[0].split("\\")[0] for x in sel["aihub_src_file"].astype(str))
                lines.append(f"  - 4.{n} test {tag} FP {len(fp)}개 출처: " + ", ".join(f"{k} {v}" for k, v in src.most_common()))
    lines.append("")
    lines.append("표기: FP = others 를 타겟으로 본 개수, FN = 타겟을 others 로 본 개수. thr = val 에서 자동 선택된 운영점, 플래토폭 = accuracy_plateau 구간 폭.")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[저장] {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
