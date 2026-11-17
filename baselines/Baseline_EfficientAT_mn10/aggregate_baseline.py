#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""비교모델 집계 — seed 5회 결과를 평균 ± 표준편차로 모아 Mark5_EXP 표 1 에 붙일 행을 만든다.

[신설 2026-09-05]

읽는 것  plots/performance_summary_<model>_<cond>_lr<lr>_s<seed>[_val].csv  (cond ∈ sup, pl, ceil)
         [2026-09-06] ceil(천장) 조건이 있으면 메운 비율 (pl − sup) ÷ (ceil − sup) 도 낸다 -> plots/gap_closed_<model>_<split>.csv
쓰는 것  plots/aggregate_raw_<model>_<split>.csv      회차별 원자료 (cond, seed, lr, 지표 6개)
         plots/aggregate_summary_<model>_<split>.csv  조건별 mean · sd · n · 95% CI(t 분포)
         plots/paper_rows_<model>_<split>.csv         Mark5_EXP make_paper_tables.py --extra_csv 용
                                                      (열 이름이 지표명과 같고 값은 "mean ± sd")

Mark5 와의 검정(짝지은 t · Holm)은 Mark5_EXP 쪽에서 원자료 두 벌을 읽어 하는 것이 맞아
여기서는 하지 않는다. 원자료 CSV 가 그 입력이다.

예)  python aggregate_baseline.py --split test                 # [2026-10-25 / A안] --lr 없이 = 지금 절차 회차(_adam, Adam 선정값 고정)
     python aggregate_baseline.py --split val --lr 5e-5          # 2026-10-21까지의 회차(_lr5e-05) 다시 집계
"""
import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model_info import MODEL_TAG, DISPLAY_NAME                 # noqa: E402
from run_experiments import (METRICS, PLOT_DIR, artifacts,   # noqa: E402
                             read_summary, run_tag)   # run_tag/artifacts 의 규칙은 common/naming.py (본실험 태그만 집계)
from common.naming import parse_tag, adam_lr                        # noqa: E402  [2026-09-17] _l<n>·_u<n> 태그 읽기 · [2026-10-25] 고정 학습률
from common.others_rule import rule_json_is_current   # [2026-10-26] 지금 체크포인트 + 판정 규칙 argmax_v1

COND_LABEL = {"sup": f"{DISPLAY_NAME}", "pl": f"{DISPLAY_NAME} + consensus labels",
              "plc": f"{DISPLAY_NAME} + combiner pseudo labels",                 # [2026-10-22] Mark5.3 검증 결합기 한 라벨
              "ceil": f"{DISPLAY_NAME} + true labels (upper bound)"}     # [2026-09-06] 천장
# df=n-1 의 양측 95% t 임계값 (scipy 없이도 돌게). n=5 -> df 4 -> 2.776
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_sd(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return m, sd, n


def ci95(m, sd, n):
    if n < 2 or math.isnan(sd):
        return float("nan"), float("nan")
    t = T_CRIT.get(n - 1, 1.96)
    h = t * sd / math.sqrt(n)
    return m - h, m + h


def fmt(m, sd, n, nd=4):
    if n == 0 or math.isnan(m):
        return "-"
    if n == 1 or math.isnan(sd):
        return f"{m:.{nd}f} (n=1)"
    return f"{m:.{nd}f} ± {sd:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=f"{DISPLAY_NAME} 결과 집계")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    # [변경 2026-10-25 / A안] 없으면 지금 절차 회차(_adam)를 모은다(lr_selected.json 은 읽지 않는다). 숫자 = 2026-10-21까지의 회차.
    ap.add_argument("--lr", type=float, default=None, help="없으면 지금 절차 회차(_adam). 숫자 = 2026-10-21까지의 회차(_lr<lr>)")
    ap.add_argument("--conds", nargs="+", default=["sup", "pl", "plc", "ceil"])   # [2026-10-22] plc = 결합기 한 라벨
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    lr = args.lr
    _lr_txt = "Adam 선정값 고정(_adam)" if lr is None else f"{lr:g}"
    print(f"[INFO] {DISPLAY_NAME} 집계 — split={args.split}, lr={_lr_txt}, conds={args.conds}, seeds={args.seeds}")

    # [변경 2026-09-17 / 9절] 조건 x seed 를 열거하지 않고 plots/ 의 요약 CSV 를 전부 읽어 태그로 가른다.
    # labeled 전량 회차(옛 태그)와 _l<n>·_u<n> 회차(라벨 효율 표·데이터 효율 곡선)가 같이 들어오며,
    # 원자료 CSV 에 labeled·unlabeled 열이 생긴다(전량 = dataset_labeled.csv 행 수 / 합의 CSV 전량은 빈칸).
    # Mark5_EXP2/model/aggregate_results.py --baseline_raw 가 이 열로 seed 짝지은 검정을 한다.
    import glob as _glob
    _full = 2000
    _lab_csv = os.path.join(ROOT, "dataset_labeled.csv")
    if os.path.isfile(_lab_csv):
        with open(_lab_csv, newline="", encoding="utf-8-sig") as f:
            _full = max(0, sum(1 for _ in f) - 1)
    suffix = "" if args.split == "test" else "_val"
    raw, missing, skipped, stale = [], [], [], []
    seen_tags = set()
    for path in sorted(_glob.glob(os.path.join(PLOT_DIR, f"performance_summary_{MODEL_TAG}_*{suffix}.csv"))):
        name = os.path.basename(path)
        tag = name[len("performance_summary_"):-len(".csv")]
        if suffix:
            if not tag.endswith(suffix):
                continue
            tag = tag[:-len(suffix)]
        elif tag.endswith("_val"):
            continue
        info = parse_tag(tag)
        if info is None or info["smoke"] or info["cond"] not in args.conds or info["seed"] not in args.seeds:
            skipped.append(name); continue
        if (info["lr"] is None) != (lr is None) or (lr is not None and abs(info["lr"] - lr) > 1e-12):   # [2026-10-25] 지금 절차(_adam)·옛 회차(_lr)를 섞지 않는다
            skipped.append(name); continue
        s = read_summary(path)
        if not s or any(m not in s for m in METRICS):
            missing.append(name); continue
        # [2026-09-18 / 3차 검수 C-2] 요약이 지금 체크포인트의 결과인지(others 규칙 json 의 run_tag·bytes·mtime 이 ckpt 와 일치) 본다.
        # 학습만 새로 끝나고 val·test 가 안 돈 회차의 옛 요약, 09-06 옛 데이터 요약(json checkpoint null)은 집계에 넣지 않는다.
        a = artifacts(tag)
        if not (os.path.exists(a["rule"]) and rule_json_is_current(a["rule"], tag, a["ckpt"])):
            stale.append(name); continue
        seen_tags.add(tag)
        raw.append({"model": MODEL_TAG, "cond": info["cond"], "seed": info["seed"], "lr": adam_lr(info["cond"]) if lr is None else lr, "run_tag": tag,
                    "labeled": info["labeled"] if info["labeled"] is not None else _full,
                    "unlabeled": info["unlabeled"] if info["unlabeled"] is not None else "",
                    **{m: s[m] for m in METRICS}})
    # 전량 회차 중 빠진 것은 예전처럼 이름을 찍어 준다.
    for cond in args.conds:
        for seed in args.seeds:
            tag = run_tag(cond, lr, seed)
            if tag not in seen_tags:
                missing.append(os.path.basename(artifacts(tag)[args.split]))
    if missing:
        print(f"[WARN] 결과가 없거나 불완전한 회차 {len(missing)}개: {missing}")
    if stale:
        print(f"[WARN] 지금 체크포인트의 결과가 아니라 제외한 요약 {len(stale)}개(others_threshold json 없음·옛 형식·체크포인트와 불일치 — "
              f"run_experiments.py 로 val·test 를 다시 돌리십시오): {stale[:8]}{' …' if len(stale) > 8 else ''}")
    if skipped:
        print(f"[INFO] 집계에서 제외(다른 lr·smoke·조건 밖): {len(skipped)}개")
    if not raw:
        raise SystemExit("[ERROR] 집계할 결과가 하나도 없습니다.")

    os.makedirs(PLOT_DIR, exist_ok=True)
    raw_csv = os.path.join(PLOT_DIR, f"aggregate_raw_{MODEL_TAG}_{args.split}.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "cond", "seed", "lr", "labeled", "unlabeled", "run_tag"] + METRICS)
        w.writeheader()
        w.writerows(raw)

    summary_rows, paper_rows = [], []
    print(f"\n{'조건':<34} {'n':>2}  " + "  ".join(f"{m[:12]:>16}" for m in METRICS))
    # [2026-09-17 / 9절] 그룹 = (cond, labeled, unlabeled). 전량 그룹이 먼저(옛 표와 같은 순서), 그 뒤 labeled 축·곡선 그룹.
    def _gkey(r):
        return (r["cond"], int(r["labeled"]), int(r["unlabeled"]) if r["unlabeled"] != "" else -1)
    groups = []
    for cond in args.conds:
        for key in sorted({_gkey(r) for r in raw if r["cond"] == cond}, key=lambda k: (-k[1], k[2])):
            groups.append(key)
    for cond, lab, unl in groups:
        rows = [r for r in raw if _gkey(r) == (cond, lab, unl)]
        if not rows:
            continue
        # [2026-10-01] _u<n> 은 "라벨 없는 클립 n 개 안의 합의 라벨"(전에는 "합의 라벨 n 개") — 이름도 [unlabeled n] 으로 바꿨다.
        gname = COND_LABEL[cond] + ("" if lab == _full else f" [labeled {lab}]") + ("" if unl < 0 else f" [unlabeled {unl}]")
        line = {"model": gname, "cond": cond, "labeled": lab, "unlabeled": ("" if unl < 0 else unl), "lr": adam_lr(cond) if lr is None else lr,
                "n_runs": len(rows)}
        paper = {"model": gname}
        cells = []
        for m in METRICS:
            mu, sd, n = mean_sd([r[m] for r in rows])
            lo, hi = ci95(mu, sd, n)
            # 지표별 n 을 따로 적는다 — NaN(예: 평가셋에 없는 클래스로 ROC AUC 미정의)이 조용히 빠지면 n 이 회차 수보다 작아진다.
            line.update({f"{m} mean": mu, f"{m} sd": sd, f"{m} n": n, f"{m} ci_low": lo, f"{m} ci_high": hi})
            if n < len(rows):
                print(f"[WARN] {gname} / {m}: {len(rows)}회 중 {len(rows) - n}회가 NaN 이라 n={n} 으로 평균냈습니다.")
            if n < 5:
                print(f"[WARN] {gname} / {m}: n={n} (<5). 논문 표에 넣을 값이 아닙니다.")
            paper[m] = fmt(mu, sd, n)
            cells.append(f"{fmt(mu, sd, n):>16}")
        summary_rows.append(line)
        paper_rows.append(paper)
        print(f"{gname:<34} {len(rows):>2}  " + "  ".join(cells))

    sum_csv = os.path.join(PLOT_DIR, f"aggregate_summary_{MODEL_TAG}_{args.split}.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    paper_csv = os.path.join(PLOT_DIR, f"paper_rows_{MODEL_TAG}_{args.split}.csv")
    with open(paper_csv, "w", newline="", encoding="utf-8-sig") as f:
        # 열 이름에 쉼표가 든 "ROC AUC (Macro, OvR)" 는 csv 모듈이 따옴표로 감싼다(make_paper_tables 요구사항).
        w = csv.DictWriter(f, fieldnames=["model"] + METRICS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(paper_rows)
    print(f"\n[INFO] 저장: {raw_csv}\n       {sum_csv}\n       {paper_csv}")
    print("       표 1 에 붙이려면: Mark5_EXP 에서  python model/make_paper_tables.py --extra_csv <paper_rows 경로>")

    # ---- [2026-09-06] 메운 비율 = (pl − sup) ÷ (ceil − sup) — 세 조건이 다 있을 때만 ----
    # [2026-09-17] 메운 비율은 labeled 전량·합의 전량 회차로만 센다(옛 정의 그대로).
    _rawf = [r for r in raw if int(r["labeled"]) == _full and r["unlabeled"] == ""]
    have = {c for c in ("sup", "pl", "ceil") if any(r["cond"] == c for r in _rawf)}
    if have == {"sup", "pl", "ceil"}:
        by = {c: [r for r in _rawf if r["cond"] == c] for c in have}
        gap_rows = []
        print(f"\n{'지표':<24} {'sup(바닥)':>10} {'pl(합의)':>10} {'ceil(천장)':>10} {'메운 비율':>10} {'seed 짝':>18}")
        for m in METRICS:
            mu_s, _, _ = mean_sd([r[m] for r in by["sup"]])
            mu_p, _, _ = mean_sd([r[m] for r in by["pl"]])
            mu_c, _, _ = mean_sd([r[m] for r in by["ceil"]])
            denom = mu_c - mu_s
            ratio = (mu_p - mu_s) / denom if denom > 1e-9 else float("nan")
            # seed 짝: 같은 seed 의 세 값으로 seed 마다 계산
            per = []
            for seed in args.seeds:
                trio = [next((r[m] for r in by[c] if r["seed"] == seed), None) for c in ("sup", "pl", "ceil")]
                if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in trio):
                    continue
                d = trio[2] - trio[0]
                if d > 1e-9:
                    per.append((trio[1] - trio[0]) / d)
            pm, psd, pn = mean_sd(per)
            gap_rows.append({"metric": m, "sup_mean": mu_s, "pl_mean": mu_p, "ceil_mean": mu_c,
                             "gap_closed_mean_based": ratio, "gap_closed_per_seed_mean": pm,
                             "gap_closed_per_seed_sd": psd, "n_seed_pairs": pn})
            print(f"{m:<24} {mu_s:>10.4f} {mu_p:>10.4f} {mu_c:>10.4f} {ratio:>10.4f} {fmt(pm, psd, pn):>18}")
        gap_csv = os.path.join(PLOT_DIR, f"gap_closed_{MODEL_TAG}_{args.split}.csv")
        with open(gap_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()))
            w.writeheader()
            w.writerows(gap_rows)
        print(f"[INFO] 메운 비율: {gap_csv}  (1.0 = 라벨이 완벽했을 때만큼 오름)")
    elif "ceil" in have:
        print(f"[INFO] 메운 비율은 sup·pl·ceil 세 조건이 다 있을 때 냅니다(지금: {sorted(have)}).")


if __name__ == "__main__":
    main()
