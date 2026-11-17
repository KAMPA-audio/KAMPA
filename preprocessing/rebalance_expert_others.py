#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebalance_expert_others.py — 전문가 8개 공통 others 의 "바깥 소리" 절반을 AI Hub 우선으로 다시 뽑는다 (2026-09-15, B안).

왜: 09-07 결정으로 경계 소리(문·런닝머신·골프·악기·고양이·학원·골프연습장·에어컨)를 others 에서 빼자
    전문가 others 의 AI Hub 몫이 교통·확성기 5종 55개(train)만 남았고, audit ① 에서 "AI Hub 녹음이면 타겟"
    이 90% 로 나왔다(이진 전문가의 출처 지름길). Mark5.2(9클래스)는 AI Hub 가 7종에 고루 걸쳐 문제없다.
무엇을: 전문가 others 중 바깥 소리(role=far) 파일만 같은 자리(파일명 유지)에 AI Hub 교통·확성기 우선 →
    남는 자리는 ESC-50 19종으로 다시 채운다. 집 녹음 절반·talking·Mark5.2 는 손대지 않는다.
    후보는 게이트 + 정적 기준(active >= 0.5s, 빈 구간 < 2.0s)을 통과한 것만. val·test 는 Mark5.2 val·test
    원본을 제외, train 은 Mark5.2 train 이 안 쓴 원본을 먼저 쓴다.
기대: AI Hub→타겟 단서 train 500/(500+175)=74%(AI Hub 상한 70%), val·test 약 78% (AI Hub far 풀 train 352 / val 65 / test 66).
사용: python preprocessing/rebalance_expert_others.py --dry_run | (실행)
"""
import os, sys, json, glob, time, shutil, hashlib, argparse
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                      # noqa: E402
from repair_after_audit import sparse, sha, expert_paths, STAGE, MIN_ACTIVE, MAX_GAP  # noqa: E402

CACHE = R.CACHE_DIR
TODAY = R.TODAY


def pick(items, want, used, prefer_fresh=None):
    """draw 순서(group 균등·세션 라운드로빈)대로 훑어 정적 기준을 통과하는 것만 want 개 고른다."""
    out = []
    for it in R.draw(items, min(len(items), want * 3 + 50), used, prefer_fresh):
        if len(out) >= want:
            break
        seg, st = R.load_segment(it)
        if seg is None:
            continue
        act, gap = sparse(seg)
        if act >= MIN_ACTIVE and gap < MAX_GAP:
            out.append((it, seg, st)); used.add(it["src"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cache = json.load(open(os.path.join(CACHE, "gate_cache.json")))
    plan = json.load(open(os.path.join(CACHE, "plan.json")))
    repair = {(r["ds"], r["split"], r["name"]): r["new"]
              for r in json.load(open(os.path.join(CACHE, f"repair_{TODAY}.json")))}
    pools = R.gate_pool(R.build_chime() + R.build_sins() + R.build_esc() + R.build_aihub(), cache)
    by_src = {it["src"]: it for it in pools}
    P = defaultdict(list)
    for it in pools:
        P[(it["role"], it["split"])].append(it)

    # 현재 전문가 others 파일 -> 원본 (repair 반영)
    cur = {}
    for sp in ("train", "val", "test"):
        for i, s in enumerate(plan[f"expert|{sp}|others"], 1):
            name = f"others_{sp}_{i:03d}.wav"
            cur[(sp, name)] = repair.get(("expert", sp, name), s)
    # Mark5.2 가 쓴 원본(split 별) — repair 반영
    m52_used = {"train": set(), "val": set(), "test": set()}
    for key, lst in plan.items():
        ds, sp, cls = key.split("|")
        if ds != "m52":
            continue
        tgt = "train" if sp == "unlabeled" else sp
        for i, s in enumerate(lst, 1):
            name = f"unlabeled_{1556 + i:04d}.wav" if sp == "unlabeled" else f"{cls}_{sp}_{i:03d}.wav"
            m52_used[tgt].add(repair.get(("m52", sp, name), s))
    # 전문가가 계속 쓰는 원본(집 녹음 others + talking)
    keep = set(s for (sp, n), s in cur.items() if by_src.get(s, {}).get("role") != "far")
    for sp in ("train", "val", "test"):
        keep |= set(repair.get(("expert", sp, f"talking_{sp}_{i:03d}.wav"), s)
                    for i, s in enumerate(plan[f"expert|{sp}|talking"], 1))

    print("[현재 전문가 others 구성]")
    plan_new = {}
    for sp in ("train", "val", "test"):
        far_slots = [n for (s, n), src in cur.items() if s == sp and by_src.get(src, {}).get("role") == "far"]
        far_slots.sort()
        n_ai_now = sum(1 for n in far_slots if by_src[cur[(sp, n)]]["source"] == "aihub")
        used = set(keep) | m52_used[sp]                     # val·test: Mark5.2 같은 split 원본 제외
        prefer = m52_used["train"] if sp == "train" else None
        ai = [it for it in P[("far", sp)] if it["source"] == "aihub"]
        esc = [it for it in P[("far", sp)] if it["source"] == "esc50"]
        # train 은 AI Hub 상한 70% — 전부 AI Hub 로 채우면 ESC 소리가 val·test 와 unlabeled 에만 나타나
        # train 에서 못 본 소리가 평가에 등장하는 어긋남이 생긴다. val·test 는 풀이 작아 있는 만큼 쓴다.
        cap = int(round(len(far_slots) * 0.7)) if sp == "train" else len(far_slots)
        got = pick(ai, cap, used, prefer)
        got += pick(esc, len(far_slots) - len(got), used, prefer)
        n_ai = sum(1 for it, _, _ in got if it["source"] == "aihub")
        n_t = {"train": 500, "val": 215, "test": 215}[sp]
        print(f"  {sp:5s} 바깥 소리 {len(far_slots)}자리: AI Hub {n_ai_now} → {n_ai}, ESC {len(far_slots)-n_ai_now} → {len(got)-n_ai}"
              f"   AI Hub→타겟 단서 {n_t/(n_t+n_ai_now)*100:.0f}% → {n_t/(n_t+n_ai)*100:.0f}%"
              f"   {'부족 ' + str(len(far_slots)-len(got)) if len(got) < len(far_slots) else 'OK'}")
        plan_new[sp] = list(zip(far_slots, got))
    if a.dry_run:
        print(f"\n[DRY RUN] 파일·provenance 는 바꾸지 않았습니다. ({time.time()-t0:.0f}s)")
        return

    import pandas as pd
    prov = pd.read_excel(R.PROV_PATH)
    backup = R.PROV_PATH + f".bak_before_rebalance_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(R.PROV_PATH, backup)
    print(f"[provenance] {len(prov)}행 읽음, 백업 {os.path.basename(backup)}  ({time.time()-t0:.0f}s)")
    active = prov["removed_20260715"] == "active"
    n_rows, n_files, log = 0, 0, []
    for sp, pairs in plan_new.items():
        for name, (it, seg, st) in pairs:
            paths = [os.path.join(STAGE, sp, name)] + expert_paths(sp, name)
            for p in paths:
                sf.write(p, seg, R.TARGET_SR, subtype="FLOAT")
            n_files += 1
            m = active & prov["mark_version"].isin(R.EXPERT_VERSIONS) & (prov["local_filename"] == name) & \
                (prov["assigned_split"] == sp) & (prov["target_class"] == "others")
            idx = prov.index[m]
            prev = cur[(sp, name)]
            prov.loc[idx, "original_labels"] = it["category"]; prov.loc[idx, "source"] = it["source"]
            prov.loc[idx, "aihub_src_file"] = it["src"]; prov.loc[idx, "aihub_category"] = it["category"]
            prov.loc[idx, "aihub_volume"] = it["volume"]
            prov.loc[idx, "seg_start_sec"] = round(float(st), 3); prov.loc[idx, "seg_end_sec"] = round(float(st) + R.SEG_SEC, 3)
            prov.loc[idx, "sha256"] = sha(paths[-1]); prov.loc[idx, "size_bytes"] = os.path.getsize(paths[-1])
            prov.loc[idx, "mark52_note"] = f"rebalanced_aihub_{TODAY}(prev={prev})"
            n_rows += len(idx)
            log.append(dict(split=sp, name=name, prev=prev, new=it["src"], group=it["group"]))
        print(f"  {sp:5s} {len(pairs)}개 교체  ({time.time()-t0:.0f}s)", flush=True)
    prov.to_excel(R.PROV_PATH, index=False)
    json.dump(log, open(os.path.join(CACHE, f"rebalance_{TODAY}.json"), "w"), ensure_ascii=False, indent=0)
    print(f"[완료] 파일 {n_files}개(x9 사본), provenance 행 {n_rows}개, 저장 {len(prov)}행  ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
