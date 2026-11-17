#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
replace_house_background.py — 교사 8개가 함께 쓰는 others 의 '집 녹음' 배경을 Mark5.3 과 녹음이 겹치지 않는 것으로 바꾼다 (2026-10-23).

왜: 배경 others 중 집 녹음(SINS cooking·eating·working, CHiME 'o')은 train 125 전부가 Mark5.3 학습 쪽과 같은 세션이고(audit ⑨R5),
    val 53 · test 51 중 49 · 39 가 Mark5.3 평가 쪽 세션이었다(⑨R6). 세션이 녹음 단위라 클립이 달라도 같은 녹음이다.
무엇을 (수진님 결정 3번 — SINS 남는 세션 + FSD50K 집 생활음, 10-23):
    자리: 지금 8곳 공통 배경 중 source 가 sins·chime_home 인 원본 행(train 125 · val 53 · test 51)의 파일명을 그대로 쓴다(8곳 같은 이름).
    채움: 활동(cooking·eating·working) 개수는 지금 배경과 맞춘다(train 의 CHiME 'o' 7 은 활동 비율대로 나눈다).
      ① Mark5.3 이 어느 split 에도 쓰지 않은 SINS 집 세션 — 매니페스트 split 그대로(남는 세션은 val·test 쪽뿐), 기존 게이트 캐시 통과분.
      ② FSD50K 집 생활음(gate_external_candidates.py 의 others_house 통과분 — 타겟 닮은 라벨 제외 + 내용 검사).
         업로더 하나는 한 split 에만 넣는다(같은 사람·같은 부엌 녹음이 train 과 val/test 에 나뉘지 않게). 업로더 순서는 시드 42.
    기록: 옛 행 removed_20260715="replaced_bghouse_20261023", 새 행 mark52_note="bghouse_20261023(from=…)". 실행 전 provenance 백업.
기대값: 개수 일치 · Mark5.3 녹음과 겹침 0 · split 사이 녹음(세션·FSD id) 겹침 0 · 업로더 split 사이 겹침 0 · test 하위라벨 ⊆ val 하위라벨.
뒤 단계: resplit_teacher_positives.py · reallocate_hard_negatives.py 와 함께 → run_augment_all.sh → run_expert_index_all.sh → verify_after_rebuild.sh
사용: python preprocessing/replace_house_background.py --dry_run | (실행)
"""
import os, re, sys, json, time, shutil, argparse, random
from collections import Counter, defaultdict

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402
from cross_reject_experts import rec_key, sub_key, data_path, sha, SEED   # noqa: E402

TODAY = "20261023"
TAG_OLD, NOTE = f"replaced_bghouse_{TODAY}", f"bghouse_{TODAY}"
GATE_REPORT = "/mnt/c/Users/user/journal_external_raw/gated/gate_report.csv"
STAGE = os.path.join(R.CACHE_DIR, "expert_common_bghouse")
ACTS = ("cooking", "eating", "working")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    log = []
    def say(msg):
        print(msg, flush=True); log.append(msg)

    prov = pd.read_excel(R.PROV_PATH)
    st = prov["removed_20260715"].astype(str)
    act = prov[st == "active"].copy()
    act["_rk"] = act.apply(rec_key, axis=1)
    orig = act[act["source_type"] == "original"]
    m52 = orig[orig["mark_version"].isin(["mark5.2", "mark5.2_eval"])]
    M_all = set(k for k in m52["_rk"] if k is not None)
    bg = orig[(orig["target_class"] == "others") & orig["source"].isin(["sins", "chime_home"])
              & ~orig["mark52_note"].astype(str).str.startswith(("cross_target", "hardneg"))]
    slots = {s: sorted(bg[(bg["mark_version"] == "mark4.1") & (bg["assigned_split"] == s)]["local_filename"]) for s in ("train", "val", "test")}
    for v in R.EXPERT_VERSIONS:
        for s in slots:
            if sorted(bg[(bg["mark_version"] == v) & (bg["assigned_split"] == s)]["local_filename"]) != slots[s]:
                raise SystemExit(f"[STOP] {v} {s}: 집 배경 자리 이름이 mark4.1 과 다름")
    ref = bg[bg["mark_version"] == "mark4.1"]
    # 목표 개수: split 별 활동 개수(CHiME 'o' 는 활동 비율대로 나눔)
    want = {}
    for s in slots:
        c = Counter(ref[ref["assigned_split"] == s]["aihub_category"])
        extra = sum(v for k, v in c.items() if k not in ACTS)
        base = {k: c.get(k, 0) for k in ACTS}
        tot = sum(base.values())
        add = {k: int(extra * base[k] / tot) for k in ACTS}
        for k in sorted(ACTS, key=lambda k: -(extra * base[k] / tot - add[k]))[:extra - sum(add.values())]:
            add[k] += 1
        want[s] = {k: base[k] + add[k] for k in ACTS}
        say(f"[목표] {s}: " + " · ".join(f"{k} {want[s][k]}" for k in ACTS) + f" = {sum(want[s].values())} (자리 {len(slots[s])})")

    # ① 남는 SINS 집 세션
    cache = json.load(open(os.path.join(R.CACHE_DIR, "gate_cache.json")))
    sins = [it for it in R.build_sins() if it["role"] == "house"]
    def sess_key(it):
        m = re.search(r"ex(\d+)", it["src"]); return ("S", "sins:ex" + m.group(1)) if m else None
    free = [it for it in sins if sess_key(it) not in M_all and cache.get(it["src"], {}).get("pass")]
    picked = {s: [] for s in slots}
    for s in slots:
        for k in ACTS:
            pool = sorted([it for it in free if it["split"] == s and it["category"] == k], key=lambda it: it["src"])
            take = pool[:want[s][k]]
            picked[s] += [dict(kind="sins", act=k, key=sess_key(it), item=it, group=f"sins:{sess_key(it)[1]}") for it in take]
    say("[①] 남는 SINS 세션: " + " · ".join(f"{s} {len(picked[s])}" for s in slots)
        + f" (세션 {sorted({sess_key(it)[1] for it in free})})")

    # ② FSD50K 집 생활음 — 업로더 단위로 split 배정
    gate = pd.read_csv(GATE_REPORT, encoding="utf-8-sig", dtype=str).fillna("")
    g = gate[(gate["cls"] == "others_house") & (gate["status"] == "통과")].to_dict("records")
    acts = {r["ext_id"]: r.get("activity", "") for r in g}
    by_up = defaultdict(list)
    for r in sorted(g, key=lambda r: int(r["ext_id"])):
        if ("F", r["ext_id"]) in M_all:
            continue
        by_up[r["group"]].append(r)
    ups = sorted(by_up)
    random.Random(SEED).shuffle(ups)
    need = {s: {k: want[s][k] - sum(1 for p in picked[s] if p["act"] == k) for k in ACTS} for s in slots}
    up_split = {}
    for u in ups:
        clips = by_up[u]
        gain = {s: sum(min(need[s][k], sum(1 for r in clips if acts[r["ext_id"]] == k)) for k in ACTS) for s in slots}
        s_best = max(("val", "test", "train"), key=lambda s: gain[s])      # 같으면 val·test 먼저(작아서 먼저 채운다)
        if gain[s_best] == 0:
            continue
        up_split[u] = s_best
        for r in clips:
            k = acts[r["ext_id"]]
            if need[s_best][k] > 0:
                picked[s_best].append(dict(kind="fsd50k", act=k, key=("F", r["ext_id"]), ext=r, group=f"up:{u}"))
                need[s_best][k] -= 1
    short = {s: {k: v for k, v in need[s].items() if v > 0} for s in slots}
    say("[②] FSD50K 집 생활음: " + " · ".join(f"{s} {sum(1 for p in picked[s] if p['kind']=='fsd50k')}" for s in slots)
        + f" · 업로더 {len(up_split)} · 부족 {short}")

    # ── 검사 ──
    keys = {s: [p["key"] for p in picked[s]] for s in slots}
    groups = {s: {p["group"] for p in picked[s]} for s in slots}
    sk = {s: {(p["kind"], p["act"]) for p in picked[s]} for s in slots}
    chk = dict(
        size=all(len(picked[s]) == len(slots[s]) for s in slots),
        vs_m53=all(not (set(keys[s]) & M_all) for s in slots),
        split_disjoint=not (set(keys["train"]) & (set(keys["val"]) | set(keys["test"]))) and not (set(keys["val"]) & set(keys["test"])),
        group_disjoint=not (groups["train"] & (groups["val"] | groups["test"])) and not (groups["val"] & groups["test"]),
        no_dup=all(len(set(k for k in keys[s] if k[0] == "F")) == sum(1 for k in keys[s] if k[0] == "F") for s in slots),
        sub_ok=sk["test"] <= sk["val"],
    )
    ok = all(chk.values())
    say(f"[검사] " + " · ".join(f"{k}={v}" for k, v in chk.items()) + f" → {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
    for s in slots:
        say(f"   {s}: " + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(Counter((p['kind'], p['act']) for p in picked[s]).items())))
    json.dump({s: [dict(kind=p["kind"], act=p["act"], key=list(p["key"])) for p in picked[s]] for s in slots},
              open(os.path.join(R.CACHE_DIR, f"bghouse_{TODAY}_{'dryrun' if a.dry_run else 'apply'}.json"), "w"), ensure_ascii=False, indent=0)
    if a.dry_run:
        open(os.path.join(R.CACHE_DIR, f"bghouse_{TODAY}_dryrun.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
        print("[DRY RUN] 파일·provenance 는 바꾸지 않았습니다.")
        return
    if not ok:
        raise SystemExit("[STOP] 기대값 불일치 — 실행하지 않음")

    # ── 실행: 공통 스테이징 → 8곳 같은 자리에 복사 ──
    from datetime import date
    import soundfile as sf
    rows_tmpl = {}
    for s in slots:
        os.makedirs(os.path.join(STAGE, s), exist_ok=True)
        rows_tmpl[s] = []
        for name, p in zip(slots[s], picked[s]):
            dst = os.path.join(STAGE, s, name)
            if p["kind"] == "sins":
                seg, start = R.load_segment(p["item"])
                sf.write(dst, seg, R.TARGET_SR, subtype="FLOAT")
                row = R.prov_row(name, "others", s, p["item"], start, dst, version="__EXPERT__")
                row["mark52_note"] = f"{NOTE}(from=sins:{p['item']['src']})"
            else:
                x = p["ext"]
                shutil.copyfile(x["out_path"], dst)
                row = {"local_filename": name, "original_labels": x["labels"], "target_class": "others", "assigned_split": s,
                       "source_type": "original", "removed_20260715": "active", "source": "fsd50k", "fsd50k_fname": int(x["ext_id"]),
                       "aihub_category": p["act"], "aihub_volume": x["group"], "seg_start_sec": float(x["seg_start_sec"]),
                       "seg_end_sec": float(x["seg_start_sec"]) + R.SEG_SEC, "download_date": date.today().isoformat(),
                       "sha256": sha(dst), "size_bytes": os.path.getsize(dst),
                       "mark52_note": f"{NOTE}(from=fsd50k:{x['ext_id']};license={x['license']};url={x['url']})"}
            rows_tmpl[s].append(row)
    backup = R.PROV_PATH + f".bak_before_bghouse_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)
    new_rows = []
    for v in R.EXPERT_VERSIONS:
        for s in slots:
            for name, row in zip(slots[s], rows_tmpl[s]):
                shutil.copyfile(os.path.join(STAGE, s, name), data_path(v, s, name))
                m = ((prov["mark_version"] == v) & (prov["assigned_split"] == s) & (prov["local_filename"] == name)
                     & (prov["target_class"] == "others") & (prov["removed_20260715"].astype(str) == "active") & (prov["source_type"] == "original"))
                assert m.sum() == 1, (v, s, name, int(m.sum()))
                prov.loc[prov.index[m][0], "removed_20260715"] = TAG_OLD
                d = dict(row); d["mark_version"] = v
                new_rows.append(d)
        say(f"[실행] {v}: 집 배경 {sum(len(slots[s]) for s in slots)}자리 교체 ({time.time()-t0:.0f}s)")
    prov = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
    prov.to_excel(R.PROV_PATH, index=False)
    say(f"[provenance] 옛 행 → {TAG_OLD}, 새 행 {len(new_rows)}, 백업 {os.path.basename(backup)} ({time.time()-t0:.0f}s)")
    open(os.path.join(R.CACHE_DIR, f"bghouse_{TODAY}.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    print(f"[완료] ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
