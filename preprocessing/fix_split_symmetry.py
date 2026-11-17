#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_split_symmetry.py — audit ⑧(val·test 출처 대칭)이 잡은 나머지 비대칭을 같은 자리 교체로 고친다 (2026-09-21, resplit 뒤 단계).

resplit_chime_others.py(CHiME 생활소음 b·o → train 전용) 뒤에 audit ⑧ 을 돌리자 세 가지가 더 나왔다.
  G1. 전문가 train others 의 AI Hub 바깥 소리가 트럭·이륜차·신호기 3종뿐(59/55/61)이고 확성기·자동차가 0 —
      val·test 에는 확성기 10/14, 자동차 15/11 이 있다. 09-15 rebalance 가 미사용 우선으로 뽑다가 두 종을 못 채웠다.
      1차 재학습에서 4.2(9)·4.4(10)·4.5(5)·4.6(3) 이 test 확성기를 타겟으로 잘못 본 것이 이것이다.
      → 트럭·이륜차·신호기 자리 70개를 확성기 35 + 자동차 35 로 교체(train-split AI Hub 풀, 미사용 59/59).
  G2. 전문가 train others 의 ESC-50 이 8종(75개)뿐이라 val·test 에 있는 crackling_fire(val 7/test 4)·crow(5/1)가 train 에 0
      → train ESC 자리 8개를 crackling_fire 4 + crow 4 로 교체(train-split 미사용). test 는 그대로.
  G4. talking: CHiME 말소리 청크 중 'b'(광대역 소음) 가 든 것 21개가 test 세션 두 곳(200110_1711·230110_1036)에 19개 —
      Mark4.5 test 11·Mark5.2 test 1 이 이것이고 train·val 에는 0. others 의 b 와 같은 구조라 같은 규칙: 'b' 든 CHiME 청크는
      역할과 무관하게 train 전용.
      → 4.5 test 11·Mark5.2 test 1 을 CHiME 말소리(비-b) test 미사용 청크로 교체(4출처 25% 유지),
        4.5 train 의 CHiME 자리 21개를 b 청크 21개(옮긴 12 + 미사용 9)로, Mark5.2 unlabeled talking 자리 1개를 옮긴 1개로.
  G0. (정합) 전문가 공통 others 930 파일의 8곳 사본이 libsndfile PEAK 청크의 기록 시각(초) 때문에 351개가 바이트 단위로 달랐다
      (오디오 샘플은 동일 — cmp 로 1바이트, sf.read 차이 0). staging 사본을 기준으로 8곳을 바이트 동일하게 맞추고
      provenance sha256·size_bytes 를 staging 값으로 통일한다. (원인: 9곳에 sf.write 를 따로 했기 때문. 이 스크립트와 resplit 은 staging 에 한 번 쓰고 copyfile.)

  ★ 사후(03:00): G4 의 train·unlabeled 쪽(b 청크 투입)은 audit ② 세션 겹침 때문에 revert_b_test_sessions.py 로 되돌렸다. test 교체·G0~G2 는 유효.

사용: python preprocessing/fix_split_symmetry.py --dry_run | (실행)
"""
import os, sys, json, time, shutil, hashlib, argparse
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402
from repair_after_audit import sparse, sha, expert_paths, MIN_ACTIVE, MAX_GAP  # noqa: E402
from resplit_chime_others import pick, even_slots, m52_path, STAGE, TODAY, TAG, M52_DIR  # noqa: E402

CACHE = R.CACHE_DIR
AIHUB_ADD = {"C.사업장_1.상가건물_a.옥외설치확성기의소음": 35, "D.교통소음_1.자동차_a.심야에자동차가빠르게주행하는소리": 35}
ESC_BAD = {"crackling_fire", "crow"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    import pandas as pd

    cache = json.load(open(os.path.join(CACHE, "gate_cache.json")))
    pools = R.gate_pool(R.build_chime() + R.build_sins() + R.build_esc() + R.build_aihub(), cache)
    by_src = {it["src"]: it for it in pools}
    prov = pd.read_excel(R.PROV_PATH)
    if "mark52_note" not in prov.columns:
        prov["mark52_note"] = ""
    active = prov["removed_20260715"].astype(str) == "active"
    vers = set(R.EXPERT_VERSIONS) | {"mark5.2", "mark5.2_eval"}
    used = set(prov.loc[active & prov["mark_version"].isin(vers), "aihub_src_file"].astype(str))
    print(f"[풀] {len(pools)} / active 원본 {len(used)}  ({time.time()-t0:.0f}s)")

    def rows_of(ver, split, cls):
        m = active & (prov["mark_version"] == ver) & (prov["assigned_split"] == split) & \
            (prov["target_class"] == cls) & (prov["source_type"] == "original")
        return prov[m]

    # ── G1: 전문가 train others AI Hub ──
    tr = rows_of("mark4.1", "train", "others")
    ai_rows = tr[(tr["source"] == "aihub") & ~tr["aihub_category"].isin(AIHUB_ADD)]
    g1_slots = even_slots([r for _, r in ai_rows.iterrows()], sum(AIHUB_ADD.values()))
    g1_new = []
    for cat, n in AIHUB_ADD.items():
        cand = [it for it in pools if it["role"] == "far" and it["source"] == "aihub" and it["split"] == "train" and it["category"] == cat]
        got = pick(cand, n, used)
        g1_new += got
        print(f"[G1] {cat[-14:]}: 후보 {len(cand)} → {len(got)}개")
    g1 = list(zip(g1_slots, g1_new))
    print(f"[G1] 전문가 train AI Hub 자리 {len(g1)}개 교체 (빼는 카테고리: {Counter(str(r['aihub_category'])[-10:] for r, _ in g1)})")

    # ── G2: 전문가 train others 에 ESC crackling_fire·crow 를 넣는다 ──
    # resplit 이 test 에 넣은 crackling_fire 4·crow 1 은 val 에도 7·5 가 있는 카테고리라 test 쪽을 바꾸지 않고 train 에 그 카테고리를
    # 넣는 쪽이 맞다(ESC test 미사용은 정적 기준에서 전부 탈락, AI Hub test-split 은 미사용 0). train ESC 자리 8개(큰 카테고리 4종에서 2개씩)
    # → crackling_fire 4 + crow 4 (train-split 미사용 14/14, 정적 통과 12/13).
    esc_tr = tr[tr["source"] == "esc50"]
    big = [c for c, _ in Counter(esc_tr["aihub_category"].astype(str)).most_common(4)]
    g2_slots = even_slots([r for _, r in esc_tr[esc_tr["aihub_category"].astype(str).isin(big)].iterrows()], 8)
    got2 = []
    for cat in ("crackling_fire", "crow"):
        cand = [it for it in pools if it["role"] == "far" and it["source"] == "esc50" and it["split"] == "train" and it["category"] == cat]
        got2 += pick(cand, 4, used)
    g2 = list(zip(g2_slots, got2))
    print(f"[G2] 전문가 train ESC 자리 {len(g2)}개(빼는 {Counter(str(r['aihub_category']) for r, _ in g2)}) → {Counter(it['category'] for _, (it, _, _) in g2)}")

    # ── G4: talking 'b' ──
    speech_chime = lambda sp: [it for it in pools if it["role"] == "speech" and it["source"] == "chime_home" and it["split"] == sp and "b" not in it["category"]]
    t45 = {sp: rows_of("mark4.5", sp, "talking") for sp in ("train", "val", "test")}
    t52 = {sp: rows_of("mark5.2", sp, "talking") for sp in ("train", "val", "test")}
    t52["unlabeled"] = rows_of("mark5.2", "unlabeled", "unlabeled:talking")
    isb = lambda rows: rows[(rows["source"] == "chime_home") & rows["aihub_category"].astype(str).str.contains("b")]
    g4 = []                                  # (ds, split, row, (it,seg,st))
    moved = {}
    for ds, sp, rows in (("m52", "test", t52["test"]), ("m52", "val", t52["val"]), ("expert", "test", t45["test"]), ("expert", "val", t45["val"])):
        b = isb(rows).sort_values("local_filename")
        if b.empty:
            continue
        got = pick(speech_chime(sp), len(b), used)
        assert len(got) == len(b), (ds, sp, len(b), len(got))
        for (_, r), new in zip(b.iterrows(), got):
            g4.append((ds, sp, r, new)); moved[str(r["aihub_src_file"])] = by_src[str(r["aihub_src_file"])]
        print(f"[G4] {ds} {sp} talking b {len(b)}개 → CHiME 비-b {len(got)}개")
    b_all = sorted(set(it["src"] for it in pools if it["role"] == "speech" and it["source"] == "chime_home" and "b" in it["category"]))
    b_items = [by_src[s] for s in b_all]                    # 21개 전부(옮긴 것 + 미사용)
    ch_tr = t45["train"][t45["train"]["source"] == "chime_home"]
    g4_tr_slots = even_slots([r for _, r in ch_tr.iterrows()], len(b_items))
    g4_tr = []
    for r, it in zip(g4_tr_slots, b_items):
        seg, st = R.load_segment(it)
        if seg is None:
            print(f"   [G4] 게이트 탈락 {it['src']} {st}"); continue
        g4_tr.append((r, (it, seg, st)))
    m52_b = [moved[s] for s in moved if s in set(str(x) for x in t52["test"]["aihub_src_file"])]
    ch_un = t52["unlabeled"][t52["unlabeled"]["source"] == "chime_home"]
    g4_un = []
    for r, it in zip(even_slots([r for _, r in ch_un.iterrows()], len(m52_b)), m52_b):
        seg, st = R.load_segment(it)
        g4_un.append((r, (it, seg, st)))
    print(f"[G4] 4.5 train CHiME 자리 {len(g4_tr)}개 → b 말소리 {len(b_items)}개 중 {len(g4_tr)}  |  Mark5.2 unlabeled talking 자리 {len(g4_un)}개 → 옮긴 b")

    # ── G0: 8곳 사본 정합 (staging 기준) ──
    g0 = []
    for sp in ("train", "val", "test"):
        for f in sorted(os.listdir(os.path.join(STAGE, sp))):
            if not f.endswith(".wav"):
                continue
            s0 = sha(os.path.join(STAGE, sp, f))
            bad = [p for p in expert_paths(sp, f) if sha(p) != s0]
            if bad:
                g0.append((sp, f, bad))
    print(f"[G0] staging 과 바이트가 다른 사본: 파일 {len(g0)}개, 사본 {sum(len(b) for _, _, b in g0)}개  ({time.time()-t0:.0f}s)")

    if a.dry_run:
        print(f"\n[DRY RUN] 바꾸지 않았습니다. ({time.time()-t0:.0f}s)")
        return

    backup = R.PROV_PATH + f".bak_before_fix_symmetry_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)
    print(f"[provenance] 백업 {os.path.basename(backup)}")
    new_rows, n_old, log = [], 0, []

    def retire(ver_list, name, split, cls):
        m = (prov["removed_20260715"].astype(str) == "active") & prov["mark_version"].isin(ver_list) & \
            (prov["local_filename"] == name) & (prov["assigned_split"] == split) & (prov["target_class"] == cls)
        prov.loc[prov.index[m], "removed_20260715"] = TAG
        return int(m.sum())

    def archive(src_path, dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)

    def write_expert(split, name, seg, cls):
        """staging 에 한 번 쓰고 8곳에 copyfile — 바이트 동일 보장."""
        sub = "" if cls == "others" else "talking"
        st_path = os.path.join(STAGE, sub, split, name) if sub else os.path.join(STAGE, split, name)
        archive(st_path, os.path.join(STAGE, f"_replaced_{TODAY}", sub, split, name) if sub else os.path.join(STAGE, f"_replaced_{TODAY}", split, name))
        sf.write(st_path, seg, R.TARGET_SR, subtype="FLOAT")
        targets = expert_paths(split, name) if cls == "others" else [os.path.join(R.EXPERT_ROOT, "Mark4.5", "data", split, name)]
        for p in targets:
            root = os.path.dirname(os.path.dirname(p))
            archive(p, os.path.join(root, f"_replaced_{TODAY}", split, name))
            shutil.copyfile(st_path, p)
        return st_path

    def do_expert_others(split, r, it, seg, st, note):
        nonlocal n_old
        p = write_expert(split, r["local_filename"], seg, "others")
        n_old += retire(R.EXPERT_VERSIONS, r["local_filename"], split, "others")
        for v in R.EXPERT_VERSIONS:
            new_rows.append(R.prov_row(r["local_filename"], "others", split, it, st, p, version=v, extra={"mark52_note": note}))

    def do_45_talking(split, r, it, seg, st, note):
        nonlocal n_old
        p = write_expert(split, r["local_filename"], seg, "talking")
        n_old += retire(["mark4.5"], r["local_filename"], split, "talking")
        new_rows.append(R.prov_row(r["local_filename"], "talking", split, it, st, p, version="mark4.5", extra={"mark52_note": note}))

    def do_m52(split, r, it, seg, st, note, cls):
        nonlocal n_old
        p = m52_path(split, r["local_filename"]) if cls != "talking" else os.path.join(R.PROJECT_ROOT, M52_DIR[split], "talking", r["local_filename"])
        if split == "unlabeled":
            p = os.path.join(R.PROJECT_ROOT, M52_DIR["unlabeled"], r["local_filename"])
        archive(p, os.path.join(R.PROJECT_ROOT, f"_replaced_{TODAY}", M52_DIR[split], r["local_filename"]))
        sf.write(p, seg, R.TARGET_SR, subtype="FLOAT")
        pc = f"unlabeled:{cls}" if split == "unlabeled" else cls
        n_old += retire(["mark5.2"], r["local_filename"], split, pc)
        new_rows.append(R.prov_row(r["local_filename"], pc, split, it, st, p, version="mark5.2", extra={"mark52_note": note}))

    for r, (it, seg, st) in g1:
        do_expert_others("train", r, it, seg, st, f"aihub_fill_{TODAY}(prev={r['aihub_src_file']})")
        log.append(dict(step="G1", split="train", name=r["local_filename"], prev=str(r["aihub_src_file"]), new=it["src"], group=it["group"]))
    print(f"[G1] {len(g1)}개 교체  ({time.time()-t0:.0f}s)", flush=True)
    for r, (it, seg, st) in g2:
        do_expert_others("train", r, it, seg, st, f"esc_fill_{TODAY}(prev={r['aihub_src_file']})")
        log.append(dict(step="G2", split="train", name=r["local_filename"], prev=str(r["aihub_src_file"]), new=it["src"], group=it["group"]))
    print(f"[G2] {len(g2)}개 교체", flush=True)
    for ds, sp, r, (it, seg, st) in g4:
        note = f"chime_b_resplit_{TODAY}(prev={r['aihub_src_file']})"
        (do_45_talking(sp, r, it, seg, st, note) if ds == "expert" else do_m52(sp, r, it, seg, st, note, "talking"))
        log.append(dict(step="G4", ds=ds, split=sp, name=r["local_filename"], prev=str(r["aihub_src_file"]), new=it["src"], group=it["group"]))
    for r, (it, seg, st) in g4_tr:
        do_45_talking("train", r, it, seg, st, f"chime_b_train_only_{TODAY}(prev={r['aihub_src_file']})")
        log.append(dict(step="G4tr", ds="expert", split="train", name=r["local_filename"], prev=str(r["aihub_src_file"]), new=it["src"], group=it["group"]))
    for r, (it, seg, st) in g4_un:
        do_m52("unlabeled", r, it, seg, st, f"chime_b_train_only_{TODAY}(prev={r['aihub_src_file']})", "talking")
        log.append(dict(step="G4un", ds="m52", split="unlabeled", name=r["local_filename"], prev=str(r["aihub_src_file"]), new=it["src"], group=it["group"]))
    print(f"[G4] test {len(g4)} · 4.5 train {len(g4_tr)} · m52 unlabeled {len(g4_un)} 교체  ({time.time()-t0:.0f}s)", flush=True)

    # G0: 사본 정합 + sha 통일 (교체가 끝난 뒤 다시 계산)
    n_copy = 0
    fixed_sha = {}
    for sp in ("train", "val", "test"):
        for f in sorted(os.listdir(os.path.join(STAGE, sp))):
            if not f.endswith(".wav"):
                continue
            src = os.path.join(STAGE, sp, f); s0 = sha(src)
            for p in expert_paths(sp, f):
                if sha(p) != s0:
                    shutil.copyfile(src, p); n_copy += 1
            fixed_sha[(sp, f)] = (s0, os.path.getsize(src))
    act = (prov["removed_20260715"].astype(str) == "active") & prov["mark_version"].isin(R.EXPERT_VERSIONS) & (prov["target_class"] == "others") & (prov["source_type"] == "original")
    n_sha = 0
    for (sp, f), (s0, sz) in fixed_sha.items():
        m = act & (prov["assigned_split"] == sp) & (prov["local_filename"] == f)
        idx = prov.index[m & (prov["sha256"] != s0)]
        if len(idx):
            prov.loc[idx, "sha256"] = s0; prov.loc[idx, "size_bytes"] = sz; n_sha += len(idx)
    for row in new_rows:                                    # 오늘 새 행도 staging 값으로
        k = (row["assigned_split"], row["local_filename"])
        if row["target_class"] == "others" and k in fixed_sha:
            row["sha256"], row["size_bytes"] = fixed_sha[k]
    print(f"[G0] 사본 copyfile {n_copy}개, provenance sha256 통일 {n_sha}행  ({time.time()-t0:.0f}s)", flush=True)

    merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
    merged.to_excel(R.PROV_PATH, index=False)
    json.dump(log, open(os.path.join(CACHE, f"fix_symmetry_{TODAY}.json"), "w"), ensure_ascii=False, indent=0)
    print(f"[완료] 옛 행 {n_old} → {TAG}, 새 행 {len(new_rows)}, 저장 {len(merged)}행  ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
