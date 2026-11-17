#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_talking_m45.py — Mark4.5(talking 교사) train 을 집 녹음 쪽으로 다시 짠다 (2026-10-24, 수진님 결정 나안).

왜: 10-23 talking train(FSD50K 말소리 250 + VOiCES TV 238)으로 학습한 mn10 교사가 1에폭 뒤로 val 이 계속 떨어졌다
    (train 0.85→0.9955, val 0.9605→0.92 · 저장 에폭 1 · 학습 후 판정 '학습 부족'). talking val/test 는 CHiME-Home·SINS 집 녹음이라
    train 을 집 녹음으로 바꾸고, others 쪽 오답(B0 val: ESC-50 숨소리 등 40% 오답)을 줄이려고 사람 비발화음을 넣는다.
무엇을 (Mark4.5 train 만 — val/test 와 다른 교사 7개는 그대로):
    talking 500 = SINS 평가 세트 TV 250 + SINS 평가 세트 대화 85 + CHiME-6 말소리 165   (지금 488 → 500, 이름 talking_train_001~500)
    others 집 배경 자리 125(bghouse_*) = SINS 평가 세트 cooking 48 · eating 34 · working 43   (출처 쏠림 방지: SINS 가 talking 에만 있지 않게)
    others 일반 자리 125 중 60(aihub_fill → esc_fill → rebalanced_aihub 순) = FSD50K 사람 비발화음
    후보는 gate_external_candidates.py --tag 20261024 통과분. 세션·업로더·라벨을 돌아가며 고른다(시드 42).
검사(기대값): 개수 · Mark5.3 녹음 겹침 0 · Mark4.5 val/test 녹음 겹침 0 · 새 클립 중복 0 · SINS 출처의 talking 몫 ≤ 0.75(감사 ① OK 선).
기록: 옛 행 removed_20260715="replaced_talk_20261024", 새 행 mark52_note="talk_20261024(from=…)". 옛 파일은 data/_replaced_20261024/train/ 로.
뒤 단계: augment_density(mark4.5, --reset) → run_expert_index_all.sh(4.5) → audit(⑨R5·R6) → make_data_release.py(4.5)
사용: python preprocessing/rebuild_talking_m45.py --dry_run | (실행)
"""
import os, sys, json, time, shutil, random, argparse
from collections import Counter, defaultdict
from datetime import date

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402
from cross_reject_experts import rec_key, data_path, sha, SEED    # noqa: E402

TODAY = "20261024"
MV = "mark4.5"
TAG_OLD, NOTE = f"replaced_talk_{TODAY}", f"talk_{TODAY}"
GATE_REPORT = f"/mnt/c/Users/user/journal_external_raw/gated/gate_report_{TODAY}.csv"
TALK_PLAN = {("sins_eval", "tv"): 250, ("sins_eval", "social"): 85, ("chime6", "speech"): 165}
HOUSE_PLAN = {"cooking": 48, "eating": 34, "working": 43}
N_HUMAN = 60
HUMAN_SLOT_ORDER = ("aihub_fill", "esc_fill", "rebalanced_aihub")
SINS_SHARE_MAX = 0.75


def round_robin(rows, n, key):
    """key 묶음(세션·업로더)을 돌아가며 n 개. 묶음 안 순서는 시드 42 로 섞는다."""
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    rng = random.Random(SEED)
    ks = sorted(g)
    for k in ks:
        rng.shuffle(g[k])
    got = []
    while len(got) < n and any(g[k] for k in ks):
        for k in ks:
            if g[k] and len(got) < n:
                got.append(g[k].pop(0))
    return got


def ext_key(r):
    if r["source"] == "sins_eval":
        return rec_key(pd.Series(dict(source="sins_eval", aihub_src_file=r["title"])))
    if r["source"] == "chime6":
        return rec_key(pd.Series(dict(source="chime6", aihub_src_file=r["group"])))
    return rec_key(pd.Series(dict(source="fsd50k", fsd50k_fname=float(r["ext_id"]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    log = []
    def say(msg):
        print(msg, flush=True); log.append(msg)

    prov = pd.read_excel(R.PROV_PATH)
    act = prov[prov["removed_20260715"].astype(str) == "active"].copy()
    act["_rk"] = act.apply(rec_key, axis=1)
    orig = act[act["source_type"] == "original"]
    M_all = set(k for k in orig[orig["mark_version"].isin(["mark5.2", "mark5.2_eval"])]["_rk"] if k is not None)
    m45 = orig[orig["mark_version"] == MV]
    vt_keys = set(k for k in m45[m45["assigned_split"].isin(["val", "test"])]["_rk"] if k is not None)
    tr = m45[m45["assigned_split"] == "train"]
    talk_old = sorted(tr[tr["target_class"] == "talking"]["local_filename"])
    oth = tr[tr["target_class"] == "others"]
    note = oth["mark52_note"].astype(str)
    house_slots = sorted(oth[note.str.startswith("bghouse")]["local_filename"])
    human_slots = []
    for p in HUMAN_SLOT_ORDER:
        human_slots += sorted(oth[note.str.startswith(p)]["local_filename"])
    human_slots = human_slots[:N_HUMAN]
    n_talk = sum(TALK_PLAN.values())
    talk_names = [f"talking_train_{i:03d}.wav" for i in range(1, n_talk + 1)]
    say(f"[자리] talking 옛 {len(talk_old)} → 새 {n_talk} · 집 배경 {len(house_slots)} · 사람 소리 {len(human_slots)}"
        f" (일반 자리 {Counter(oth[oth['local_filename'].isin(human_slots)]['mark52_note'].astype(str).str.extract(r'^([a-z_]+)')[0])})")

    gate = pd.read_csv(GATE_REPORT, encoding="utf-8-sig", dtype=str).fillna("")
    ok = gate[gate["status"] == "통과"].to_dict("records")
    for r in ok:
        r["_rk"] = ext_key(r)
    ok = [r for r in ok if r["_rk"] not in M_all and r["_rk"] not in vt_keys]
    picked = {"talking": [], "house": [], "human": []}
    for (src, actv), n in TALK_PLAN.items():
        pool = [r for r in ok if r["cls"] == "talking" and r["source"] == src and r["activity"] == actv]
        got = round_robin(pool, n, key=lambda r: r["group"])
        picked["talking"] += got
        say(f"[talking] {src}/{actv}: {len(got)}/{n} (후보 {len(pool)}, 묶음 {len({r['group'] for r in got})})")
    for actv, n in HOUSE_PLAN.items():
        pool = [r for r in ok if r["cls"] == "others_house_sins" and r["activity"] == actv]
        got = round_robin(pool, n, key=lambda r: r["group"])
        picked["house"] += got
        say(f"[집 배경] {actv}: {len(got)}/{n} (후보 {len(pool)}, 세션 {len({r['group'] for r in got})})")
    pool = [r for r in ok if r["cls"] == "others_human"]
    picked["human"] = round_robin(pool, len(human_slots), key=lambda r: r["activity"])
    say(f"[사람 소리] {len(picked['human'])}/{len(human_slots)} (후보 {len(pool)}) {dict(Counter(r['activity'] for r in picked['human']))}")

    # ── 검사 ──
    new_keys = [r["_rk"] for v in picked.values() for r in v]
    paths = [r["out_path"] for v in picked.values() for r in v]
    after = tr[~tr["local_filename"].isin(talk_old + house_slots + human_slots)]
    src_cls = Counter((s, c) for s, c in zip(after["source"], after["target_class"]))
    for r in picked["talking"]:
        src_cls[("sins" if r["source"] == "sins_eval" else r["source"], "talking")] += 1
    for r in picked["house"]:
        src_cls[("sins", "others")] += 1
    sins_t, sins_o = src_cls[("sins", "talking")], src_cls[("sins", "others")]
    share = sins_t / max(1, sins_t + sins_o)
    chk = dict(
        size=len(picked["talking"]) == n_talk and len(picked["house"]) == len(house_slots) == sum(HOUSE_PLAN.values())
             and len(picked["human"]) == len(human_slots) == N_HUMAN,
        vs_m53=not (set(new_keys) & M_all),
        vs_m45_valtest=not (set(new_keys) & vt_keys),
        no_dup_clip=len(set(paths)) == len(paths),
        no_dup_fsd=len({k for k in new_keys if k[0] == "F"}) == sum(1 for k in new_keys if k[0] == "F"),
        sins_share=share <= SINS_SHARE_MAX,
    )
    good = all(chk.values())
    say("[검사] " + " · ".join(f"{k}={v}" for k, v in chk.items()) + f" · SINS talking 몫 {sins_t}/{sins_t + sins_o}={share:.3f}"
        + f" → {'OK' if good else 'FAIL'} ({time.time()-t0:.0f}s)")
    json.dump({k: [dict(source=r["source"], ext_id=r["ext_id"], activity=r["activity"], key=list(r["_rk"])) for r in v] for k, v in picked.items()},
              open(os.path.join(R.CACHE_DIR, f"talk_m45_{TODAY}_{'dryrun' if a.dry_run else 'apply'}.json"), "w"), ensure_ascii=False, indent=0)
    if a.dry_run:
        open(os.path.join(R.CACHE_DIR, f"talk_m45_{TODAY}_dryrun.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
        print("[DRY RUN] 파일·provenance 는 바꾸지 않았습니다.")
        return
    if not good:
        raise SystemExit("[STOP] 기대값 불일치 — 실행하지 않음")

    # ── 실행 ──
    repo_train = os.path.dirname(data_path(MV, "train", "x.wav"))
    keep = os.path.join(os.path.dirname(repo_train), f"_replaced_{TODAY}", "train")
    os.makedirs(keep, exist_ok=True)
    backup = R.PROV_PATH + f".bak_before_talk_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)

    def new_row(name, cls, r):
        src = r["source"]
        row = {"local_filename": name, "mark_version": MV, "target_class": cls, "assigned_split": "train", "source_type": "original",
               "removed_20260715": "active", "source": src, "aihub_category": r["activity"],
               "seg_start_sec": float(r["seg_start_sec"]), "seg_end_sec": float(r["seg_start_sec"]) + R.SEG_SEC,
               "duration_sec": R.SEG_SEC, "download_date": date.today().isoformat(),
               "sha256": sha(data_path(MV, "train", name)), "size_bytes": os.path.getsize(data_path(MV, "train", name)),
               "mark52_note": f"{NOTE}(from={src}:{r['ext_id']};license={r['license']};url={r['url']})"}
        if src == "fsd50k":
            row.update(fsd50k_fname=int(r["ext_id"]), original_labels=r["labels"], aihub_volume=r["group"])
        elif src == "sins_eval":
            row.update(aihub_src_file=r["title"], aihub_volume=r["group"],
                       original_labels={"tv": "watching_tv", "social": "social_activity"}.get(r["activity"], r["activity"]))
        else:
            row.update(aihub_src_file=r["group"], aihub_volume=r["ext_id"], original_labels="speech")
        return row

    jobs = [(n, "talking", r) for n, r in zip(talk_names, picked["talking"])] \
         + [(n, "others", r) for n, r in zip(house_slots, picked["house"])] \
         + [(n, "others", r) for n, r in zip(human_slots, picked["human"])]
    new_rows = []
    for name, cls, r in jobs:
        dst = data_path(MV, "train", name)
        if os.path.exists(dst):
            shutil.move(dst, os.path.join(keep, name))
        shutil.copyfile(r["out_path"], dst)
        new_rows.append(new_row(name, cls, r))
    old = (prov["mark_version"] == MV) & (prov["assigned_split"] == "train") & (prov["source_type"] == "original") \
        & (prov["removed_20260715"].astype(str) == "active") & prov["local_filename"].isin(talk_old + house_slots + human_slots)
    assert old.sum() == len(talk_old) + len(house_slots) + len(human_slots), int(old.sum())
    prov.loc[old, "removed_20260715"] = TAG_OLD
    prov = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
    prov.to_excel(R.PROV_PATH, index=False)
    say(f"[실행] 파일 {len(jobs)}개 교체(옛 파일 → {os.path.relpath(keep, os.path.dirname(os.path.dirname(repo_train)))}) · "
        f"provenance 옛 행 {int(old.sum())} → {TAG_OLD}, 새 행 {len(new_rows)}, 백업 {os.path.basename(backup)}")
    open(os.path.join(R.CACHE_DIR, f"talk_m45_{TODAY}.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    print(f"[완료] ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
