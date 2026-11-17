#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resplit_chime_others.py — CHiME-Home 생활소음(b·o) others 를 train 전용으로 옮긴다 (2026-09-21, B-2).

■ 왜
  09-15 재구성에서 CHiME-Home 'b'(광대역 생활소음) 청크가 전문가 8개 공통 others 의 train 0 / val 0 / test 29,
  Mark5.2 others 의 train 0 / val 0 / test 6 으로 test 에만 들어갔다(순수 b 청크 35개 중 34개가 한 녹음 세션
  CR_lounge_230110_1036 이라 세션 단위 분할에서 한 split 으로 통째로 갔다). construction 전문가(Mark4.3)가
  val 에서 못 본 이 소리를 test 에서 29개 중 25개 construction 으로 판정해 Others FPR 0.93%→13.49%,
  test Accuracy 0.9674→0.8953. 1차 재학습 8개 전부 같은 test 를 썼다(다른 7개는 자기 클래스와 안 닮아 others 로 넘김).

■ 무엇을 (같은 자리 교체 — 파일명은 그대로, 내용·provenance 만 바뀐다. repair_after_audit·rebalance 와 같은 방식)
  A. Mark5.2 test others 의 CHiME 6개   → SINS(요리·작업·식사) test 풀에서 미사용 원본
  B. Mark5.2 val  others 의 CHiME 1개   → SINS val 풀
  C. 전문가 test others 의 CHiME 30개   → SINS test 풀 (A 와 겹치지 않게; "Mark5.2 val·test 원본은 전문가 val·test 에서 제외")
  D. 전문가 val  others 의 CHiME (0개 예상)
  E. 전문가 train others: SINS 자리 N개 → A~D 에서 빠진 CHiME 원본 전부(37개 예상). 집:바깥 250:250 유지.
  F. Mark5.2 unlabeled others: SINS 자리 M개 → 그중 'b' 청크 전부(35~36개 예상). 수진님 결정 (b) 09-21.
     (전문가 train 과 Mark5.2 train/unlabeled 는 같은 원본을 써도 된다 — 09-07 방침)
  후보는 게이트(check_waveform) + 정적 기준(active >= 0.5s, 빈 구간 < 2.0s)을 통과한 것만.
  빠지는 파일은 지우지 않고 <저장소>/data/_replaced_20260921/<split>/ (Mark5.2 는 _replaced_20260921/<폴더>/) 로 복사해 둔다.
  provenance: 빠지는 행은 removed_20260715="replaced_20260921" 로 표시(행 유지), 들어오는 행은 신규(mark52_note 에 prev=).
  재발 방지는 audit_dataset.py ⑧(val·test 출처 대칭) 과 rebuild_mark52_data.py build_chime() 의 house→train 고정.

  ★ 사후(03:00): E·F 로 train 에 넣은 test 세션 b 청크는 audit ② 세션 겹침 때문에 revert_b_test_sessions.py 로 되돌렸다(폐기). A~D 는 유효.

■ 사용
  python preprocessing/resplit_chime_others.py --dry_run     # 계획·후보·검사만
  python preprocessing/resplit_chime_others.py               # 실행 (provenance 백업 뒤 한 번 쓰기)
"""
import os, sys, json, time, shutil, hashlib, argparse
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402
from repair_after_audit import sparse, sha, expert_paths, MIN_ACTIVE, MAX_GAP  # noqa: E402

CACHE = R.CACHE_DIR
STAGE = os.path.join(CACHE, "expert_common")
TODAY = "20260921"
TAG = f"replaced_{TODAY}"
M52_DIR = {"train": "data_source_labeled", "val": "data_source_val", "test": "data_source_test",
           "unlabeled": "data_source_unlabeled"}


def m52_path(split, name):
    d = os.path.join(R.PROJECT_ROOT, M52_DIR[split])
    return os.path.join(d, name) if split == "unlabeled" else os.path.join(d, "others", name)


def pick(items, want, used, prefer=None):
    """draw 순서(group 균등·세션 라운드로빈)로 훑어 게이트+정적 기준을 통과한 것만 want 개."""
    out = []
    for it in R.draw(items, min(len(items), want * 3 + 50), used, prefer):
        if len(out) >= want:
            break
        seg, st = R.load_segment(it)
        if seg is None:
            continue
        act, gap = sparse(seg)
        if act >= MIN_ACTIVE and gap < MAX_GAP:
            out.append((it, seg, st)); used.add(it["src"])
    return out


def even_slots(rows, want):
    """SINS 자리를 활동(cooking/working/eating)별로 고르게, 파일명 순으로 뽑는다(재현 가능)."""
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["local_filename"]):
        by[str(r["aihub_category"])].append(r)
    keys = sorted(by)
    out = []
    while len(out) < want and any(by.values()):
        for k in keys:
            if by[k] and len(out) < want:
                out.append(by[k].pop(0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--provenance_path", type=str, default=R.PROV_PATH)
    a = ap.parse_args()
    t0 = time.time()
    import pandas as pd

    # ── 풀 ──
    cache = json.load(open(os.path.join(CACHE, "gate_cache.json")))
    pools = R.gate_pool(R.build_chime() + R.build_sins() + R.build_esc() + R.build_aihub(), cache)
    json.dump(cache, open(os.path.join(CACHE, "gate_cache.json"), "w"))
    by_src = {it["src"]: it for it in pools}
    P = defaultdict(list)
    for it in pools:
        P[(it["role"], it["split"])].append(it)
    sins_house = {sp: [it for it in P[("house", sp)] if it["source"] == "sins"] for sp in ("train", "val", "test")}
    far = {sp: P[("far", sp)] for sp in ("train", "val", "test")}
    print(f"[풀] 게이트 통과 {len(pools)}  SINS 집 녹음 train/val/test {len(sins_house['train'])}/{len(sins_house['val'])}/{len(sins_house['test'])}")

    # ── provenance (한 번 읽기) ──
    prov = pd.read_excel(a.provenance_path)
    if "mark52_note" not in prov.columns:
        prov["mark52_note"] = ""
    active = prov["removed_20260715"].astype(str) == "active"
    vers = set(R.EXPERT_VERSIONS) | {"mark5.2", "mark5.2_eval"}
    used = set(prov.loc[active & prov["mark_version"].isin(vers), "aihub_src_file"].astype(str))
    print(f"[provenance] {len(prov)}행, active 원본 사용 중 {len(used)}개  ({time.time()-t0:.0f}s)")

    def rows_of(ver, split, cls):
        m = active & (prov["mark_version"] == ver) & (prov["assigned_split"] == split) & \
            (prov["target_class"] == cls) & (prov["source_type"] == "original")
        return prov[m]

    exp = {sp: rows_of("mark4.1", sp, "others") for sp in ("train", "val", "test")}
    m52 = {sp: rows_of("mark5.2", sp, "others") for sp in ("train", "val", "test")}
    m52["unlabeled"] = rows_of("mark5.2", "unlabeled", "unlabeled:others")
    for sp in ("train", "val", "test"):
        assert len(exp[sp]) == R.EXPERT_OTHERS[sp], (sp, len(exp[sp]))
    print("[현재] 전문가 others  " + " ".join(f"{sp} {len(exp[sp])}(chime {int((exp[sp]['source']=='chime_home').sum())})" for sp in exp))
    print("[현재] Mark5.2 others " + " ".join(f"{sp} {len(m52[sp])}(chime {int((m52[sp]['source']=='chime_home').sum())})" for sp in m52))

    # ── A~D 교체 대상(CHiME → SINS 같은 split) ──
    jobs = []            # dict(ds, split, name, old_src, new=(it,seg,st))
    moved = []           # A~D 에서 빠지는 CHiME item (E·F 로 간다)
    short = []
    for ds, sp, rows in (("m52", "test", m52["test"]), ("m52", "val", m52["val"]),
                         ("expert", "test", exp["test"]), ("expert", "val", exp["val"])):
        ch = rows[rows["source"] == "chime_home"].sort_values("local_filename")
        if ch.empty:
            continue
        got = pick(sins_house[sp], len(ch), used)
        if len(got) < len(ch):
            extra = pick(far[sp], len(ch) - len(got), used)
            short.append((ds, sp, len(ch), len(got), len(extra)))
            got += extra
        for (_, r), new in zip(ch.iterrows(), got):
            jobs.append(dict(ds=ds, split=sp, name=r["local_filename"], old_src=str(r["aihub_src_file"]), new=new))
            if str(r["aihub_src_file"]) in by_src:
                moved.append(by_src[str(r["aihub_src_file"])])
            else:
                short.append((ds, sp, r["local_filename"], "풀에 없는 CHiME 원본", 0))
        n_s = sum(1 for it, _, _ in got if it["source"] == "sins")
        print(f"  {ds:6s} {sp:4s} CHiME {len(ch)}개 → SINS {n_s} / 바깥 {len(got) - n_s}")
    moved_src = {it["src"]: it for it in moved}
    moved_b = [it for it in moved_src.values() if "b" in it["category"]]
    print(f"[이동할 CHiME] {len(moved_src)}개 (b 포함 {len(moved_b)}, 세션 {sorted(set(it['sess'] for it in moved_src.values()))})")

    # ── E. 전문가 train: SINS 자리 → CHiME 전부 ──
    e_slots = even_slots([r for _, r in exp["train"][exp["train"]["source"] == "sins"].iterrows()], len(moved_src))
    e_items = [moved_src[k] for k in sorted(moved_src)]
    e_pairs = []
    for r, it in zip(e_slots, e_items):
        seg, st = R.load_segment(it)
        if seg is None:
            short.append(("expert", "train", it["src"], f"게이트 탈락 {st}", 0)); continue
        e_pairs.append((r, it, seg, st))
    # ── F. Mark5.2 unlabeled: SINS 자리 → CHiME 'b' ──
    f_slots = even_slots([r for _, r in m52["unlabeled"][m52["unlabeled"]["source"] == "sins"].iterrows()], len(moved_b))
    f_pairs = []
    for r, it in zip(f_slots, sorted(moved_b, key=lambda x: x["src"])):
        seg, st = R.load_segment(it)
        if seg is None:
            short.append(("m52", "unlabeled", it["src"], f"게이트 탈락 {st}", 0)); continue
        f_pairs.append((r, it, seg, st))
    print(f"[E] 전문가 train SINS {len(e_pairs)}자리 → CHiME  [F] Mark5.2 unlabeled SINS {len(f_pairs)}자리 → CHiME b  ({time.time()-t0:.0f}s)")

    # ── 계획 요약(전후 출처 구성) ──
    def comp(rows, changes):
        c = Counter(str(s) for s in rows["source"])
        for old_src, new_src in changes:
            c[old_src] -= 1; c[new_src] += 1
        return dict(sorted(c.items()))
    print("\n[출처 구성 전 → 후]")
    for ds, sp, rows in (("expert", "train", exp["train"]), ("expert", "val", exp["val"]), ("expert", "test", exp["test"]),
                         ("m52", "train", m52["train"]), ("m52", "val", m52["val"]), ("m52", "test", m52["test"]),
                         ("m52", "unlabeled", m52["unlabeled"])):
        ch = [("chime_home", j["new"][0]["source"]) for j in jobs if (j["ds"], j["split"]) == (ds, sp)]
        if (ds, sp) == ("expert", "train"):
            ch += [("sins", "chime_home")] * len(e_pairs)
        if (ds, sp) == ("m52", "unlabeled"):
            ch += [("sins", "chime_home")] * len(f_pairs)
        print(f"  {ds:6s} {sp:9s} {comp(rows, [])}  →  {comp(rows, ch)}")
    # 검사: 새 원본 중복·sha256 중복·Mark5.2 val/test 와 전문가 val/test 겹침
    new_srcs = [j["new"][0]["src"] for j in jobs]
    dup = [s for s, n in Counter(new_srcs).items() if n > 1]
    m52_vt = set(j["new"][0]["src"] for j in jobs if j["ds"] == "m52" and j["split"] in ("val", "test"))
    exp_vt = set(j["new"][0]["src"] for j in jobs if j["ds"] == "expert" and j["split"] in ("val", "test"))
    print(f"[검사] 새 원본 중복 {len(dup)}  Mark5.2 val·test ∩ 전문가 val·test {len(m52_vt & exp_vt)}  "
          f"새 원본이 기존 active 에 있음 {sum(1 for s in new_srcs if s in used and s not in set(new_srcs))}")
    if short:
        print("[주의] 부족·탈락:")
        for s in short:
            print("   ", s)
    if a.dry_run:
        print(f"\n[DRY RUN] 파일·provenance 는 바꾸지 않았습니다. ({time.time()-t0:.0f}s)")
        return
    if dup or (m52_vt & exp_vt):
        raise SystemExit("[ERROR] 검사 실패 — 중단")

    # ── 실행 ──
    backup = a.provenance_path + f".bak_before_chime_resplit_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(a.provenance_path, backup)
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

    def do_expert(split, name, it, seg, st, note):
        nonlocal n_old
        for v in R.EXPERT_VERSIONS:
            root = os.path.join(R.EXPERT_ROOT, "Mark" + v[4:], "data")
            archive(os.path.join(root, split, name), os.path.join(root, f"_replaced_{TODAY}", split, name))
        archive(os.path.join(STAGE, split, name), os.path.join(STAGE, f"_replaced_{TODAY}", split, name))
        paths = [os.path.join(STAGE, split, name)] + expert_paths(split, name)
        # staging 에 한 번만 쓰고 8곳은 copyfile — sf.write 를 9번 하면 PEAK 청크의 기록 시각(초) 때문에 바이트가 달라진다(09-21 실측 351파일).
        sf.write(paths[0], seg, R.TARGET_SR, subtype="FLOAT")
        for p in paths[1:]:
            shutil.copyfile(paths[0], p)
        n_old += retire(R.EXPERT_VERSIONS, name, split, "others")
        for v in R.EXPERT_VERSIONS:
            new_rows.append(R.prov_row(name, "others", split, it, st, paths[-1], version=v, extra={"mark52_note": note}))

    def do_m52(split, name, it, seg, st, note):
        nonlocal n_old
        p = m52_path(split, name)
        archive(p, os.path.join(R.PROJECT_ROOT, f"_replaced_{TODAY}", M52_DIR[split], name))
        sf.write(p, seg, R.TARGET_SR, subtype="FLOAT")
        cls = "unlabeled:others" if split == "unlabeled" else "others"
        n_old += retire(["mark5.2"], name, split, cls)
        new_rows.append(R.prov_row(name, cls, split, it, st, p, version="mark5.2", extra={"mark52_note": note}))

    for j in jobs:                                              # A~D
        it, seg, st = j["new"]
        note = f"chime_resplit_{TODAY}(prev={j['old_src']})"
        (do_expert if j["ds"] == "expert" else do_m52)(j["split"], j["name"], it, seg, st, note)
        log.append(dict(step="A-D", ds=j["ds"], split=j["split"], name=j["name"], prev=j["old_src"], new=it["src"], group=it["group"]))
    print(f"[A~D] {len(jobs)}개 교체  ({time.time()-t0:.0f}s)", flush=True)
    for r, it, seg, st in e_pairs:                              # E
        note = f"chime_train_only_{TODAY}(prev={r['aihub_src_file']})"
        do_expert("train", r["local_filename"], it, seg, st, note)
        log.append(dict(step="E", ds="expert", split="train", name=r["local_filename"], prev=str(r["aihub_src_file"]), new=it["src"], group=it["group"]))
    print(f"[E] 전문가 train {len(e_pairs)}개 교체  ({time.time()-t0:.0f}s)", flush=True)
    for r, it, seg, st in f_pairs:                              # F
        note = f"chime_train_only_{TODAY}(prev={r['aihub_src_file']})"
        do_m52("unlabeled", r["local_filename"], it, seg, st, note)
        log.append(dict(step="F", ds="m52", split="unlabeled", name=r["local_filename"], prev=str(r["aihub_src_file"]), new=it["src"], group=it["group"]))
    print(f"[F] Mark5.2 unlabeled {len(f_pairs)}개 교체  ({time.time()-t0:.0f}s)", flush=True)

    merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
    merged.to_excel(a.provenance_path, index=False)
    json.dump(log, open(os.path.join(CACHE, f"resplit_chime_{TODAY}.json"), "w"), ensure_ascii=False, indent=0)
    print(f"[완료] 옛 행 {n_old}개 → {TAG}, 새 행 {len(new_rows)}개, 저장 {len(merged)}행  ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
