#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""라벨 없는 양 곡선(정답 라벨 200 고정) — Mark5.3 과 표준 반지도 4방법을 같은 곡선에 그린다.

[신설 2026-10-15] 수진님 결정(B안): "이득이 라벨 없는 데이터에서 나온다"를 기여점으로 보이려고,
Roy & Etemad(AAAI 2024, UnMixMatch 4.2절)처럼 자기 방법과 다른 방법을 같은 곡선에 그려 "같은 라벨 없는 데이터를
더 잘 활용한다"를 기울기(이득)로 보인다. 실험 설계(정답 고정·라벨 없는 양만 늘림·지도만 기준선)는 Oliver 외(NeurIPS 2018) 4.5절.

읽는 것 (test, seed 1~5, 정답 라벨 200)
  Mark5.3      --mark53_dir/performance_summary_<prefix>_l200_u{0,500,1000,2000}_s<k>.csv
               u0 = 같은 Mark5.3 을 라벨 없는 클립 없이(정답 클립에 작용하는 부품 OVA·SC 는 그대로) — "방법 자체의 몫"
  표준 반지도  plots/performance_summary_efficientat_mn10_<방법>_adam_l200[_u<n>]_s<k>.csv   ([2026-10-25] 그 전 판은 _lr<선정>)
               u2000 = 표 1 의 회차(태그에 _u 없음), u500·u1000 = --ssl_unlabeled_ns 회차,
               u0 = 지도만(하한, efficientat_mn10_sup_adam_l200_s<k>) — 라벨 없는 클립이 없으면 표준 방법은 지도만과 같다.
  두 쪽의 라벨 없는 클립은 같은 seed 면 같은 클립이다(common/data.py subsample_nested = Mark5.3 train_mark5.py 893~906행, 2026-10-15 대조).

쓰는 것
  plots/unlabeled_curve_l200_test.csv        모델 x 라벨 없는 수 x 지표 평균·표준편차(Accuracy·Macro P·Macro R·Macro F1·Others FPR)
  plots/unlabeled_curve_l200_gain_test.csv   이득 = 그 모델의 라벨 없는 0개 대비 차이(짝지은 t) ·
                                             이득 차이 = Mark5.3 이득 − 방법 이득(짝지은 t, 라벨 없는 수·지표마다 4방법 Holm)
  plots/unlabeled_curve_l200_test.png        지표 4개 작은 그림(평균 ± 표준편차)
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
from model_info import MODEL_TAG                                                   # noqa: E402
from common.naming import run_tag, SSL_METHODS, ssl_impl_version, UNLABELED_SUBSAMPLE_RULE, adam_lr   # noqa: E402
from run_experiments import read_summary, PLOT_DIR                                 # noqa: E402  (torch 를 물지 않는다)
from make_ssl_table import _lr_of, _holm                                           # noqa: E402

METRICS4 = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
COLS = METRICS4 + ["Others FPR"]
SEEDS = [1, 2, 3, 4, 5]
LABELED_N = 200
UNL = [0, 500, 1000, 2000]
NAMES = {"mark53": "Mark5.3 (ours)", "pseudolabel": "Pseudo-Label", "meanteacher": "Mean Teacher",
         "fixmatch": "FixMatch", "freematch": "FreeMatch"}
ORDER = ["mark53", *SSL_METHODS]
# (2026-10-09~10-21) 최종 모델 = Mark5.3 10-08 수정판(--m53 all --joint_forward, 학습률 1e-4) — make_ssl_table.MARK53_PREFIX(10-07 판, JF 없음)와 따로 두었다.
# [변경 2026-10-25 / A안] 제안모델 = 부품 6개 제거 + JF + 단일 분류기 + Adam 선정값 고정(_ADAM). 옛 기본값: …_RW_M53_JF_elr0.0001(10-08 판) ·
# …_SINGLE_PD(2026-10-22~10-25 Prodigy 판)
MARK53_FINAL_PREFIX = "mark5.2_PL_SA_G04_AC_LT_noDB_noSEG_LIN_PH_HW1_OR02_NEG_SM_CT_BG0_RW_JF_SINGLE_ADAM"
# dataviz 기본 범주 색 1~5번(흰 바탕, validate_palette.js 통과 2026-10-15). 대비가 낮은 색이 있어 표식 모양·범례로 함께 구분한다.
COLORS = {"mark53": "#2a78d6", "pseudolabel": "#eb6834", "meanteacher": "#1baf7a", "fixmatch": "#eda100", "freematch": "#e87ba4"}
MARKERS = {"mark53": "o", "pseudolabel": "s", "meanteacher": "^", "fixmatch": "D", "freematch": "v"}


def _meta(tag):
    p = os.path.join(PLOT_DIR, f"train_meta_{tag}.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _ssl_ok(tag, method, unlabeled_n):
    """표준 반지도 회차가 지금 구현 판이고, _u<n> 회차면 추출 규칙 판까지 맞는가."""
    m = _meta(tag)
    if m is None or m.get("ssl_impl_version") != ssl_impl_version(method):
        return False
    if unlabeled_n is not None and m.get("unlabeled_subsample_rule") != UNLABELED_SUBSAMPLE_RULE:
        return False
    return True


def _collect(tag_of_seed, plots_dir, check=None):
    """seed -> 요약 dict(test). check(tag) 가 False 인 회차와 지표가 모자란 회차는 뺀다."""
    out = {}
    for s in SEEDS:
        tag = tag_of_seed(s)
        if check is not None and not check(tag):
            print(f"[WARN] {tag}: train_meta 가 없거나 판(구현·추출 규칙)이 지금과 달라 뺍니다.")
            continue
        summ = read_summary(os.path.join(plots_dir, f"performance_summary_{tag}.csv"))
        if summ and all(c in summ for c in COLS):
            out[s] = summ
    return out


def _paired(a_runs, b_runs, metric, stats):
    """같은 seed 끼리 a − b. (평균 차이, p, 짝 수)."""
    ks = sorted(set(a_runs) & set(b_runs))
    if len(ks) < 2 or stats is None:
        return float("nan"), float("nan"), len(ks)
    a = [a_runs[k][metric] for k in ks]
    b = [b_runs[k][metric] for k in ks]
    d = [x - y for x, y in zip(a, b)]
    p = float(stats.ttest_rel(a, b).pvalue) if any(abs(x) > 0 for x in d) else 1.0
    return st.mean(d), p, len(ks)


def _fmt_p(p):
    """보고용 p 표기 — 0.001 보다 작으면 <0.001, 빈칸이면 -."""
    if p == "" or p != p:
        return "-"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _gain_runs(runs_n, runs_0):
    """seed 마다 (n 개 − 0 개) 차이를 요약 dict 모양으로. 두 쪽에 다 있는 seed 만."""
    return {k: {m: runs_n[k][m] - runs_0[k][m] for m in METRICS4} for k in sorted(set(runs_n) & set(runs_0))}


def _plot(curve, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 축척(2026-10-15 계산): 그림 10 x 7.4 인치 · 2 x 2 칸이라 칸 하나 약 4.4 x 2.9 인치. 눈금·축 글자 10pt(칸 높이의 약 5%),
    # 칸 제목 11pt. 범례는 칸 안에 두면 선을 가리므로 맨 위 한 줄(5개)로 뺀다 — 5개 x 약 1.6 인치 = 8 인치 < 10 인치.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    for ax, metric in zip(axes.ravel(), METRICS4):
        for key in ORDER:
            pts = [(u, curve[key][u]) for u in UNL if curve[key].get(u) and len(curve[key][u]) > 0]
            if not pts:
                continue
            xs = [u for u, _ in pts]
            ms = [st.mean(r[s][metric] for s in r) for _, r in pts]
            sds = [st.stdev([r[s][metric] for s in r]) if len(r) > 1 else 0.0 for _, r in pts]
            lw = 2.4 if key == "mark53" else 1.6
            ax.errorbar(xs, ms, yerr=sds, color=COLORS[key], marker=MARKERS[key], markersize=7 if key == "mark53" else 6,
                        linewidth=lw, capsize=3, elinewidth=1.0, label=NAMES[key], zorder=3 if key == "mark53" else 2)
        ax.set_title(metric, fontsize=11)
        ax.set_xticks(UNL)
        ax.tick_params(labelsize=10)
        ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Number of unlabeled clips (labeled = 200)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score (test, mean ± SD, n = 5)", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "At 0 unlabeled clips the four SSL methods reduce to supervised-only training (same point); "
             "Mark5.3 at 0 keeps its labeled-side parts.", ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_png, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="라벨 없는 양 곡선(정답 200) — Mark5.3 대 표준 반지도 4방법")
    ap.add_argument("--mark53_dir", type=str, required=True, help="Mark5.3 plots 폴더(드라이브 …/KAMPA/plots)")
    ap.add_argument("--mark53_prefix", type=str, default=MARK53_FINAL_PREFIX, help="Mark5.3 태그 앞부분(…_ADAM 까지). 기본 = 제안모델(2026-10-25 Adam 판)")
    ap.add_argument("--allow_missing", action="store_true", help="seed 가 5개보다 적어도 멈추지 않는다(동작 확인용)")
    args = ap.parse_args()
    try:
        from scipy import stats
    except Exception:
        stats = None
        print("[WARN] scipy 가 없어 t 검정을 건너뜁니다.")

    sup_lr = _lr_of("sup")
    ssl_lr = {m: _lr_of(m) for m in SSL_METHODS}
    print(f"[INFO] 학습률(Adam 선정값 고정, 태그 _{sup_lr}): sup {adam_lr('sup'):g} · " + " · ".join(f"{m} {adam_lr(m):g}" for m in SSL_METHODS))

    curve = {"mark53": {}}
    for u in UNL:
        curve["mark53"][u] = _collect(lambda s, u=u: f"{args.mark53_prefix}_l{LABELED_N}_u{u}_s{s}", args.mark53_dir)
    sup = _collect(lambda s: run_tag(MODEL_TAG, "sup", sup_lr, s, labeled_n=LABELED_N), PLOT_DIR) if sup_lr is not None else {}
    for m in SSL_METHODS:
        curve[m] = {0: sup}
        if ssl_lr[m] is None:
            continue
        for u in UNL[1:]:
            un = None if u == 2000 else u          # 2,000 = 전량 = 표 1 회차(태그에 _u 없음)
            curve[m][u] = _collect(lambda s, m=m, un=un: run_tag(MODEL_TAG, m, ssl_lr[m], s, labeled_n=LABELED_N, unlabeled_n=un),
                                   PLOT_DIR, check=lambda tag, m=m, un=un: _ssl_ok(tag, m, un))

    short = [(k, u, len(curve[k].get(u) or {})) for k in ORDER for u in UNL if len(curve[k].get(u) or {}) < len(SEEDS)]
    if short:
        print("[WARN] seed 가 5개보다 적은 점: " + ", ".join(f"{NAMES[k]} u{u}={n}" for k, u, n in short))
        if not args.allow_missing:
            raise SystemExit("[ERROR] 곡선의 점이 모자랍니다 — 두 노트북의 학습이 다 끝났는지 확인하십시오(--allow_missing 은 동작 확인용).")

    # ---- 곡선 값 ----
    rows = []
    print(f"\n=== 라벨 없는 양 곡선 — 정답 라벨 {LABELED_N}, test, 5회 평균 ± 표준편차 (Acc / P / R / F1 · Others FPR) ===")
    for k in ORDER:
        for u in UNL:
            r = curve[k].get(u) or {}
            if not r:
                continue
            line = {"model": NAMES[k], "key": k, "unlabeled": u, "n_seeds": len(r)}
            for c in COLS:
                v = [r[s][c] for s in r]
                line[f"{c} mean"] = round(st.mean(v), 4)
                line[f"{c} sd"] = round(st.stdev(v), 4) if len(v) > 1 else ""
            rows.append(line)
            print(f"{NAMES[k]:<16} u{u:<5} " + " / ".join(f"{line[f'{c} mean']:.4f}" for c in METRICS4)
                  + f" · FPR {line['Others FPR mean']:.4f}  (n={len(r)})")
    out_csv = os.path.join(PLOT_DIR, f"unlabeled_curve_l{LABELED_N}_test.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 이득과 이득 차이 ----
    grows = []
    print("\n=== 이득 = 라벨 없는 n 개 − 0 개 (같은 seed 짝지은 t) · 이득 차이 = Mark5.3 이득 − 방법 이득 (4방법 Holm) ===")
    for u in UNL[1:]:
        g53 = _gain_runs(curve["mark53"].get(u) or {}, curve["mark53"].get(0) or {})
        gm = {m: _gain_runs(curve[m].get(u) or {}, curve[m].get(0) or {}) for m in SSL_METHODS}
        for metric in METRICS4:
            # 각 모델의 이득 자체(0 개 대비)
            own = {"mark53": _paired(curve["mark53"].get(u) or {}, curve["mark53"].get(0) or {}, metric, stats)}
            for m in SSL_METHODS:
                own[m] = _paired(curve[m].get(u) or {}, curve[m].get(0) or {}, metric, stats)
            # Mark5.3 이득 − 방법 이득, 4방법 안에서 Holm
            diffs = [(m, *_paired(g53, gm[m], metric, stats)) for m in SSL_METHODS]
            adj = _holm([d[2] for d in diffs])
            for k in ORDER:
                dm, p, n = own[k]
                line = {"unlabeled": u, "metric": metric, "model": NAMES[k], "gain_mean": round(dm, 4) if dm == dm else "",
                        "gain_p": round(p, 6) if p == p else "", "n_pairs": n,
                        "mark53_gain_minus_this": "", "p": "", "p_holm": ""}
                if k != "mark53":
                    i = SSL_METHODS.index(k)
                    _, ddm, dp, _ = diffs[i]
                    line.update({"mark53_gain_minus_this": round(ddm, 4) if ddm == ddm else "",
                                 "p": round(dp, 6) if dp == dp else "", "p_holm": round(adj[i], 6) if adj[i] == adj[i] else ""})
                grows.append(line)
        print(f"\n--- 라벨 없는 {u}개 ---")
        for k in ORDER:
            parts = []
            for metric in METRICS4:
                ln = next(g for g in grows if g["unlabeled"] == u and g["metric"] == metric and g["model"] == NAMES[k])
                txt = f"{ln['gain_mean'] * 100:+.2f}%p" if ln["gain_mean"] != "" else "-"
                if k != "mark53" and ln["mark53_gain_minus_this"] != "":
                    txt += f" [Δ {ln['mark53_gain_minus_this'] * 100:+.2f}, Holm {_fmt_p(ln['p_holm'])}]"
                parts.append(txt)
            print(f"{NAMES[k]:<16} 이득 Acc / P / R / F1: " + " / ".join(parts))
    out_gain = os.path.join(PLOT_DIR, f"unlabeled_curve_l{LABELED_N}_gain_test.csv")
    with open(out_gain, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0].keys()))
        w.writeheader()
        w.writerows(grows)

    out_png = os.path.join(PLOT_DIR, f"unlabeled_curve_l{LABELED_N}_test.png")
    _plot(curve, out_png)
    print(f"\n저장: {out_csv}\n      {out_gain}\n      {out_png}")


if __name__ == "__main__":
    main()
