#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""논문 표 1(같은 mn10 에서 표준 반지도 방법 비교)과 표 3(가짜 정답 품질·양)을 만든다.

[신설 2026-10-07] paper-wiki wiki/concepts/ssl-comparison-protocol.md 의 표 1·3 설계(수진님 결정).

읽는 것
  plots/performance_summary_<tag>.csv (test)
      [2026-10-25 / A안] 지금은 모든 행이 Adam 선정값 고정 회차(_adam) — 아래 _lr<…json> 은 2026-10-21까지의 이름이다.
      지도만(하한)  = efficientat_mn10_sup_lr<lr_selected.json>[_l<n>]_s<1~5>
      상한          = efficientat_mn10_ceil_lr<lr_selected.json>_s<1~5>      (labeled 전량 + 라벨 없는 클립의 정답)
      표준 반지도   = efficientat_mn10_<방법>_lr<lr_selected_<방법>.json>[_l<n>]_s<1~5>
  --mark53_dir/performance_summary_<Mark5.3 tag>.csv  (제안 모델, 같은 test 430·같은 seed 1~5)
  plots/train_meta_<표준 반지도 tag>.json 의 pl_quality_final      (표 3)
  consensus_labels.csv · ceiling_labels.csv                        (표 3 의 전문가 합의 행)

쓰는 것
  plots/ssl_table1_test.csv          라벨 수 x 행(방법) x 지표 평균·표준편차 + Mark5.3 대비 짝지은 t(p, Holm)
  plots/ssl_table3_pl_quality.csv    방법 x 라벨 수의 가짜 정답 품질·양(평균·표준편차)

검정: Mark5.3 − 기준선을 seed 끼리 짝지은 양측 t(scipy ttest_rel). Holm 보정은 라벨 수·지표마다 기준선 5개(하한 + 4방법) 안에서,
**보고하는 지표 4개(Accuracy·Macro P·Macro R·Macro F1) 모두**에 한다(2026-10-07: Accuracy 에만 보정했던 실수 재발 방지).
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model_info import MODEL_TAG                                  # noqa: E402
from common.naming import run_tag, SSL_METHODS, ssl_impl_version, CUR, adam_lr  # noqa: E402
from run_experiments import read_summary, PLOT_DIR                # noqa: E402  (torch 를 물지 않는다)

METRICS4 = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
COLS = METRICS4 + ["Others FPR"]
LABELED = [(None, "2,000"), (1000, "1,000"), (500, "500"), (200, "200")]
SEEDS = [1, 2, 3, 4, 5]
NAMES = {"sup": "지도만 (하한)", "pseudolabel": "Pseudo-Label", "meanteacher": "Mean Teacher", "fixmatch": "FixMatch",
         "freematch": "FreeMatch", "mark53": "Mark5.3 (제안)", "ceil": "라벨 전부 (상한)"}
# [변경 2026-10-25 / A안] 제안모델 = 부품 6개 제거 + JF + 단일 분류기 + Adam 선정값 고정(_ADAM). 옛 기본값: …_RW_M53_elr0.0001(10-07 판) ·
# …_SINGLE_PD(2026-10-22~10-25 Prodigy 판)
MARK53_PREFIX = "mark5.2_PL_SA_G04_AC_LT_noDB_noSEG_LIN_PH_HW1_OR02_NEG_SM_CT_BG0_RW_JF_SINGLE_ADAM"


def _lr_of(cond):
    """[변경 2026-10-25 / A안] 모든 조건이 Adam 선정값 고정 회차(_adam)라 학습률 기록(lr_selected*.json)을 읽지 않고 CUR 을 돌려준다
    (실제 학습률은 common/naming.adam_lr(cond)). 학습률 기록을 읽던 판은 git a4da6a8 에 있다."""
    return CUR


def _clip_key(path):
    return os.path.basename(str(path).replace("\\", "/"))


def _holm(pvals):
    """Holm 보정 p (입력 순서대로). NaN 은 그대로 둔다."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    m = len(idx)
    out = list(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: pvals[k])):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out


def _meta_version_ok(tag, method):
    """[2026-10-08] 표준 반지도 회차의 train_meta 구현 판이 지금 판과 같은가(다르면 표·표 3 에 넣지 않는다)."""
    p = os.path.join(PLOT_DIR, f"train_meta_{tag}.json")
    if not os.path.isfile(p):
        return False
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("ssl_impl_version") == ssl_impl_version(method)


def _collect(tag_of_seed, plots_dir, ssl_method=None):
    """seed -> 요약 dict (test). 없는 seed 는 빠진다. ssl_method 를 주면 구현 판이 지금 판인 회차만 넣는다."""
    out = {}
    for s in SEEDS:
        if ssl_method is not None and not _meta_version_ok(tag_of_seed(s), ssl_method):
            print(f"[WARN] {tag_of_seed(s)}: train_meta 가 없거나 구현 판이 {ssl_impl_version(ssl_method)} 가 아니라 표에서 뺍니다.")
            continue
        summ = read_summary(os.path.join(plots_dir, f"performance_summary_{tag_of_seed(s)}.csv"))
        if summ and all(c in summ for c in COLS):
            out[s] = summ
    return out


def main():
    ap = argparse.ArgumentParser(description="논문 표 1·3 (표준 반지도 비교)")
    ap.add_argument("--mark53_dir", type=str, required=True, help="Mark5.3 plots 폴더(드라이브 …/KAMPA/plots)")
    ap.add_argument("--mark53_prefix", type=str, default=MARK53_PREFIX)
    # [2026-10-08] 출력 파일 이름 꼬리 — Mark5.3 판을 바꿔 다시 만들 때(예: --mark53_prefix …_M53_JF… --out_tag _M53JF) 기본 표를 덮어쓰지 않게.
    ap.add_argument("--out_tag", type=str, default="", help="출력 이름 꼬리: plots/ssl_table1_test<꼬리>.csv · ssl_table3_pl_quality<꼬리>.csv")
    args = ap.parse_args()
    try:
        from scipy import stats
    except Exception:
        stats = None
        print("[WARN] scipy 가 없어 t 검정을 건너뜁니다.")

    sup_lr = _lr_of("sup")
    ssl_lr = {m: _lr_of(m) for m in SSL_METHODS}
    print(f"[INFO] 학습률(Adam 선정값 고정, 태그 _{sup_lr}): sup·ceil {adam_lr('sup'):g} · " + " · ".join(f"{m} {adam_lr(m):g}" for m in SSL_METHODS))

    rows_out = []
    print("\n=== 표 1 — 같은 mn10, test, 5회 평균 ± 표준편차 (Mark5.3 − 기준선: 짝지은 t, p / Holm p) ===")
    for lab, lab_name in LABELED:
        groups = {}
        if sup_lr is not None:
            groups["sup"] = _collect(lambda s: run_tag(MODEL_TAG, "sup", sup_lr, s, labeled_n=lab), PLOT_DIR)
        for m in SSL_METHODS:
            if ssl_lr[m] is not None:
                groups[m] = _collect(lambda s, m=m: run_tag(MODEL_TAG, m, ssl_lr[m], s, labeled_n=lab), PLOT_DIR, ssl_method=m)
        sfx = "" if lab is None else f"_l{lab}"
        groups["mark53"] = _collect(lambda s: f"{args.mark53_prefix}{sfx}_u2000_s{s}", args.mark53_dir)
        if lab is None and sup_lr is not None:
            groups["ceil"] = _collect(lambda s: run_tag(MODEL_TAG, "ceil", sup_lr, s), PLOT_DIR)
        ours = groups["mark53"]
        bases = [g for g in ("sup", *SSL_METHODS) if g in groups]
        # 지표마다 기준선 5개의 p 를 모아 Holm
        tests = {}
        for metric in METRICS4:
            raw = []
            for g in bases:
                ks = sorted(set(ours) & set(groups[g]))
                if stats is not None and len(ks) >= 2:
                    a = [ours[k][metric] for k in ks]
                    b = [groups[g][k][metric] for k in ks]
                    d = [x - y for x, y in zip(a, b)]
                    p = float(stats.ttest_rel(a, b).pvalue) if any(abs(x) > 0 for x in d) else 1.0
                    raw.append((g, st.mean(d), p, len(ks)))
                else:
                    raw.append((g, float("nan"), float("nan"), len(ks)))
            adj = _holm([r[2] for r in raw])
            for (g, dm, p, n), pa in zip(raw, adj):
                tests[(g, metric)] = (dm, p, pa, n)
        print(f"\n--- 정답 라벨 {lab_name} ---")
        print(f"{'행':<18}{'n':>3} " + " ".join(f"{c:>17}" for c in COLS) + "   Mark5.3 − 이 행: Acc / F1  (p, Holm p)")
        for g in ("sup", *SSL_METHODS, "mark53", "ceil"):
            if g not in groups:
                continue
            r = groups[g]
            if not r:
                print(f"{NAMES[g]:<18}{0:>3}  (결과 없음)")
                continue
            cells = []
            line = {"labeled": lab_name, "row": NAMES[g], "cond": g, "n_seeds": len(r)}
            for c in COLS:
                v = [r[k][c] for k in sorted(r)]
                mu, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
                cells.append(f"{mu:.4f}±{sd:.4f}")
                line[f"{c} mean"] = round(mu, 4)
                line[f"{c} sd"] = round(sd, 4)
            cmp_txt = ""
            if g in bases:
                parts = []
                for metric in METRICS4:
                    dm, p, pa, n = tests[(g, metric)]
                    line[f"{metric} diff(Mark5.3-row)"] = round(dm, 4) if dm == dm else ""
                    line[f"{metric} p"] = round(p, 4) if p == p else ""
                    line[f"{metric} p_holm"] = round(pa, 4) if pa == pa else ""
                for metric in ("Accuracy", "Macro F1"):
                    dm, p, pa, n = tests[(g, metric)]
                    parts.append(f"{dm * 100:+.2f}%p (p {p:.3f}, Holm {pa:.3f})" if dm == dm else "검정 불가")
                cmp_txt = "   " + " / ".join(parts)
            print(f"{NAMES[g]:<18}{len(r):>3} " + " ".join(f"{x:>17}" for x in cells) + cmp_txt)
            rows_out.append(line)
    out1 = os.path.join(PLOT_DIR, f"ssl_table1_test{args.out_tag}.csv")
    keys = []
    for r in rows_out:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n저장: {out1}")

    # ---- 표 3: 가짜 정답 품질·양 ----
    print("\n=== 표 3 — 가짜 정답 품질(숨긴 정답과 같은 비율)·양(쓰인 비율), 라벨 없는 클립 2,000개 ===")
    t3 = []
    gt = {}
    ceil_csv = os.path.join(ROOT, "ceiling_labels.csv")
    if os.path.isfile(ceil_csv):
        with open(ceil_csv, newline="", encoding="utf-8-sig") as f:
            gt = {_clip_key(r["path"]): r["label"] for r in csv.DictReader(f)}
    cons_csv = os.path.join(ROOT, "consensus_labels.csv")
    if gt and os.path.isfile(cons_csv):
        with open(cons_csv, newline="", encoding="utf-8-sig") as f:
            cons = list(csv.DictReader(f))
        for origin, label in ((None, "전문가 합의 (전체)"), ("consensus", "전문가 합의 (합의 채택)"), ("reject_others", "전문가 합의 (전원 거절→others)")):
            sub = [r for r in cons if origin is None or r.get("origin", "consensus") == origin]
            known = [r for r in sub if _clip_key(r["path"]) in gt]
            ok = sum(1 for r in known if gt[_clip_key(r["path"])] == r["label"])
            q = ok / len(known) if known else float("nan")
            line = {"row": label, "labeled": "모든 라벨 수(전문가는 고정)", "quantity mean": round(len(sub) / len(gt), 4), "quantity sd": 0.0,
                    "quality mean": round(q, 4), "quality sd": 0.0, "n_runs": 1, "n_pseudo": len(sub)}
            t3.append(line)
            print(f"{label:<28} 양 {len(sub)}/{len(gt)} = {len(sub) / len(gt):.4f} · 품질 {ok}/{len(known)} = {q:.4f}")
    for m in SSL_METHODS:
        if ssl_lr[m] is None:
            continue
        for lab, lab_name in LABELED:
            qs, ns, n_none = [], [], 0
            for s in SEEDS:
                p = os.path.join(PLOT_DIR, f"train_meta_{run_tag(MODEL_TAG, m, ssl_lr[m], s, labeled_n=lab)}.json")
                if not os.path.isfile(p) or not _meta_version_ok(run_tag(MODEL_TAG, m, ssl_lr[m], s, labeled_n=lab), m):
                    continue
                with open(p, encoding="utf-8") as f:
                    pf = json.load(f).get("pl_quality_final")
                if pf:
                    if pf.get("quantity") is not None:
                        ns.append(pf["quantity"])
                    # [2026-10-07 검수 반영] 문턱 통과 0개인 회차는 품질이 없다(None) — 평균에서 빼고 개수를 알린다.
                    if pf.get("quality") is None or pf["quality"] != pf["quality"]:
                        n_none += 1
                    else:
                        qs.append(pf["quality"])
            if not ns:
                continue
            if n_none:
                print(f"[WARN] {NAMES[m]} 라벨 {lab_name}: 문턱 통과 가짜 정답이 0개라 품질이 없는 회차 {n_none}개 — 품질 평균에서 뺐다")
            if not qs:
                qs = [float("nan")]
            line = {"row": NAMES[m], "labeled": lab_name, "n_runs": len(ns), "n_quality_missing": n_none,
                    "quantity mean": round(st.mean(ns), 4), "quantity sd": round(st.stdev(ns), 4) if len(ns) > 1 else 0.0,
                    "quality mean": round(st.mean(qs), 4), "quality sd": round(st.stdev(qs), 4) if len(qs) > 1 else 0.0}
            t3.append(line)
            print(f"{NAMES[m]:<14} 라벨 {lab_name:>5}  회차 {len(ns)}(품질 {len(ns) - n_none})  양 {line['quantity mean']:.4f}±{line['quantity sd']:.4f} · "
                  f"품질 {line['quality mean']:.4f}±{line['quality sd']:.4f}")
    out3 = os.path.join(PLOT_DIR, f"ssl_table3_pl_quality{args.out_tag}.csv")
    with open(out3, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row", "labeled", "n_runs", "n_quality_missing", "n_pseudo",
                                          "quantity mean", "quantity sd", "quality mean", "quality sd"])
        w.writeheader()
        w.writerows(t3)
    print(f"저장: {out3}")


if __name__ == "__main__":
    main()
