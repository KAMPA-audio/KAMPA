#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
revert_b_test_sessions.py — test·val 세션의 CHiME 'b' 청크를 train 에 넣었던 것을 되돌린다 (2026-09-21 03:00, B-2 수정).

왜: resplit(E·F)·fix_split_symmetry(G4tr·G4un) 가 test 세션(230110_1036·200110_1711)·val 세션(210110_0739)의 'b' 청크 90개를
    전문가 train others 37 / Mark5.2 unlabeled others 35 / Mark4.5 train talking 17 / Mark5.2 unlabeled talking 1 에 넣었다.
    그러자 audit ② 세션 겹침이 FAIL — train 과 test 가 같은 CHiME 세션(다른 청크)을 공유하게 됐다. 청크는 다르지만 같은 방·같은 날·
    같은 마이크라 세션 특성이 새는 통로가 되고, 09-15 까지 지켜 온 "세션은 한 split 에만" 규칙이 깨진다.
    그래서 규칙을 고친다: **val·test 세션의 'b' 청크는 어느 split 에도 쓰지 않는다(폐기).** train 세션의 b·o 는 train 에 그대로.
    (수진님 결정 (b) "unlabeled 에 b 35" 는 이 통로 때문에 실행하지 못했다 — 아침 보고에 명시.)
무엇을: 90개 자리에 _replaced_20260921/ 의 옛 파일(SINS 등)을 그대로 되돌려 놓고, provenance 는
    옛 행 replaced_20260921 → active, CHiME b 행 active → dropped_b_testsess_20260921. 전문가 8곳은 staging 사본에서 copyfile.
사용: python preprocessing/revert_b_test_sessions.py --dry_run | (실행)
"""
import os, sys, json, time, shutil, argparse
from collections import Counter
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402
from repair_after_audit import sha, expert_paths                  # noqa: E402
from resplit_chime_others import STAGE, TODAY, TAG, M52_DIR        # noqa: E402

CACHE = R.CACHE_DIR
DROP = f"dropped_b_testsess_{TODAY}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rs = json.load(open(os.path.join(CACHE, f"resplit_chime_{TODAY}.json"), encoding="utf-8"))
    fx = json.load(open(os.path.join(CACHE, f"fix_symmetry_{TODAY}.json"), encoding="utf-8"))
    jobs = []   # (kind, split, name, chime_src, prev_src)
    for r in rs:
        if r["step"] == "E":
            jobs.append(("exp_others", "train", r["name"], r["new"], r["prev"]))
        elif r["step"] == "F":
            jobs.append(("m52_unl_others", "unlabeled", r["name"], r["new"], r["prev"]))
    for r in fx:
        if r["step"] == "G4tr":
            jobs.append(("m45_talking", "train", r["name"], r["new"], r["prev"]))
        elif r["step"] == "G4un":
            jobs.append(("m52_unl_talking", "unlabeled", r["name"], r["new"], r["prev"]))
    print("[대상]", Counter(k for k, *_ in jobs), f"합 {len(jobs)}")
    assert all("CR_lounge_" in j[3] for j in jobs), "b 청크가 아닌 것이 섞임"

    def src_archive(kind, split, name):
        if kind == "exp_others":
            return os.path.join(STAGE, f"_replaced_{TODAY}", split, name)
        if kind == "m45_talking":
            return os.path.join(STAGE, f"_replaced_{TODAY}", "talking", split, name)
        return os.path.join(R.PROJECT_ROOT, f"_replaced_{TODAY}", M52_DIR["unlabeled"], name)

    def targets(kind, split, name):
        if kind == "exp_others":
            return [os.path.join(STAGE, split, name)] + expert_paths(split, name)
        if kind == "m45_talking":
            return [os.path.join(STAGE, "talking", split, name), os.path.join(R.EXPERT_ROOT, "Mark4.5", "data", split, name)]
        return [os.path.join(R.PROJECT_ROOT, M52_DIR["unlabeled"], name)]

    missing = [j for j in jobs if not os.path.exists(src_archive(*j[:3]))]
    print(f"[보관본 확인] 없음 {len(missing)}")
    for j in missing[:5]:
        print("   ", j)
    if a.dry_run:
        print(f"[DRY RUN] ({time.time()-t0:.0f}s)")
        return
    assert not missing

    prov = pd.read_excel(R.PROV_PATH)
    backup = R.PROV_PATH + f".bak_before_revert_b_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)
    print(f"[provenance] {len(prov)}행, 백업 {os.path.basename(backup)}  ({time.time()-t0:.0f}s)")
    st = prov["removed_20260715"].astype(str)
    n_drop = n_back = n_files = 0
    for kind, split, name, chime_src, prev_src in jobs:
        arc = src_archive(kind, split, name)
        tg = targets(kind, split, name)
        for p in tg:
            shutil.copyfile(arc, p); n_files += 1
        s0, sz = sha(tg[0]), os.path.getsize(tg[0])
        if kind == "exp_others":
            vers, cls = R.EXPERT_VERSIONS, "others"
        elif kind == "m45_talking":
            vers, cls = ["mark4.5"], "talking"
        elif kind == "m52_unl_others":
            vers, cls = ["mark5.2"], "unlabeled:others"
        else:
            vers, cls = ["mark5.2"], "unlabeled:talking"
        base = prov["mark_version"].isin(vers) & (prov["local_filename"] == name) & (prov["assigned_split"] == split) & (prov["target_class"] == cls)
        cur = base & (st == "active") & (prov["aihub_src_file"].astype(str) == chime_src)
        old = base & (st == TAG) & (prov["aihub_src_file"].astype(str) == prev_src)
        assert cur.sum() == len(vers) and old.sum() == len(vers), (kind, name, int(cur.sum()), int(old.sum()))
        prov.loc[prov.index[cur], "removed_20260715"] = DROP
        prov.loc[prov.index[old], "removed_20260715"] = "active"
        prov.loc[prov.index[old], "sha256"] = s0
        prov.loc[prov.index[old], "size_bytes"] = sz
        prov.loc[prov.index[old], "mark52_note"] = f"restored_{TODAY}(b_test_session_dropped)"
        n_drop += int(cur.sum()); n_back += int(old.sum())
    prov.to_excel(R.PROV_PATH, index=False)
    json.dump([dict(kind=k, split=s, name=n, dropped=c, restored=p) for k, s, n, c, p in jobs],
              open(os.path.join(CACHE, f"revert_b_{TODAY}.json"), "w"), ensure_ascii=False, indent=0)
    print(f"[완료] 파일 복원 {n_files}개, provenance b 행 {n_drop} → {DROP}, 옛 행 {n_back} → active  ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
