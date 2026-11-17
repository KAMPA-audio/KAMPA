#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
remove_child_speech.py — talking 에서 아이 말소리(CHiME-Home 'c')를 빼고, 교사 Mark4.5 val/test 를 고르게 다시 나눈다 (2026-10-25, 수진님 결정 B안).

왜: Mark4.5(talking 교사)가 두 번 연속 학습 후 판정에서 떨어졌다. 원인 세 가지가 겹쳤다.
    ① CHiME-Home 은 세션이 7개뿐이고 모두 Mark5.3 이 써서, 녹음 분리 원칙상 교사 train 에 CHiME-Home(아이 말소리)을 넣을 수 없다.
    ② 10-23 교사 val/test 세션 분할이 TV 비율만 보고 아이 말소리 세션 3개를 모두 val 로 보냈다(val 아이 100/215=47%, test 5/215,
       Mark5.3 talking 전체는 129/540=24%).
    ③ 10-24 others 에 사람 몸 소리(숨·기침 등)를 넣어 아이 옹알이·소리 지르기가 others 쪽으로 밀렸다(새 B0 아이 정확도 0.83→0.58).
    talking 정의를 "어른 말소리 + TV" 로 정하고 아이 말소리를 Mark5.3 과 교사에서 모두 뺀다(others 로 옮기지 않는다).
무엇을:
  1단계 Mark5.3(데이터 창고 Mark5.2/data_source_*) — 아이 말소리 talking 클립(정답 라벨·라벨 없는·val·test)을 **같은 CHiME-Home 세션**의
         아이 없는 청크(어른 말소리 f·m 먼저, 모자라면 TV v)로 바꾼다. 같은 세션에 모자라면 같은 split 쪽(Mark5.3 학습 쪽 세션끼리,
         val 은 val 세션끼리, test 는 test 세션끼리)의 다른 세션에서 채운다 — split 별 세션 구성은 그대로다.
         파일 이름이 같아 데이터 목록(dataset_*.csv)은 그대로다. 청크는 rebuild_mark52_data 의 build_chime·게이트·3초 창 규칙 그대로이고,
         지금 살아 있는 행(어느 저장소든)이 쓰지 않는 청크만 쓴다(교체돼 아무도 안 쓰는 옛 3차 교사 청크는 쓸 수 있다).
  2단계 교사 Mark4.5 val/test — 1단계를 반영한 Mark5.3 학습 쪽 talking(정답 라벨 222 + 라벨 없는 222)을 세션 단위로 나눈다.
         점수 = Σ|TV 비율 − 목표| + |CHiME 비율 차| + |SINS 대화 비율 차| (val·test 사이). 세션 분리 · 각 215 · test 하위라벨 ⊆ val.
         TV 목표 = 아이를 뺀 학습 쪽 talking 전체의 TV 비율(약 0.7 — 말소리가 아이 말소리만큼 줄었다), 각 split 은 목표 ±0.20.
         (±0.10 은 숫자상 불가능하다: 학습 쪽 말소리 약 130개 중 89개가 SINS 세션 하나(ex202)에 있어, 세션을 쪼개지 않으면
          그 세션을 받는 쪽은 말소리가 많고 다른 쪽은 TV 가 80%를 넘는다 — 10-25 dry-run 실측)
         교사 train 과 다른 교사 7개는 바꾸지 않는다(그 교사들 others 에 든 말소리 클립은 새 정의에서도 맞는 음성 예제다).
  3단계(실행 시) 드라이브용 — 바뀐 Mark5.3 파일의 data_labeled·data_unlabeled·data_val·data_test 판을 fix_audio_length 와 같은 방식으로
         스테이징 폴더에 만든다(드라이브의 data_* 는 data_source_* 를 이 함수로 만든 것이다).
기대값: 1단계 교체 수 = 아이 클립 수(10-25 실측 53·54·10·12) · 같은 split 쪽 세션 · 새 클립 아이 0 · 청크 중복 0 · split 별 개수 그대로.
        2단계 val/test 각 215 · 세션 겹침 0 · Mark5.3 평가 쪽 세션 0 · 아이 0.
기록: Mark5.3 옛 행 removed_20260715="replaced_nochild_20261025", 새 행 mark52_note="nochild_20261025(prev=…)".
      Mark4.5 옛 val/test talking 행 "replaced_talkvt_20261025", 새 행 "talkvt_20261025(from=…)". 옛 파일은 각 _replaced_20261025/ 로.
뒤 단계: Mark4.5 인덱스(파일 이름 그대로) → 감사(Mark5.2 · Mark4.5) → make_data_release.py --only 5 → 커밋 → 드라이브(바뀐 파일만)
사용: python preprocessing/remove_child_speech.py --dry_run | (실행)
"""
import os, sys, json, time, shutil, random, argparse
from collections import Counter, defaultdict

import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                        # noqa: E402
from cross_reject_experts import rec_key, sub_key, sha, SEED           # noqa: E402
import resplit_teacher_positives as RS                                 # noqa: E402
from fix_audio_length import process_audio_file                        # noqa: E402

TODAY = "20261025"
TAG_M53, NOTE_M53 = f"replaced_nochild_{TODAY}", f"nochild_{TODAY}"
TAG_M45, NOTE_M45 = f"replaced_talkvt_{TODAY}", f"talkvt_{TODAY}"
M52_SRC = {"train": "data_source_labeled", "unlabeled": "data_source_unlabeled", "val": "data_source_val", "test": "data_source_test"}
M52_OUT = {"train": "data_labeled", "unlabeled": "data_unlabeled", "val": "data_val", "test": "data_test"}
TALK = ("talking", "unlabeled:talking")
STAGE_DRIVE = os.path.expanduser(f"~/staging_m52_nochild_{TODAY}")


def m52_file(split, name, root=R.PROJECT_ROOT, dirs=M52_SRC):
    return os.path.join(root, dirs[split], name) if split == "unlabeled" else os.path.join(root, dirs[split], "talking", name)


def is_child(cat):
    return "c" in str(cat)


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
    m52 = orig[orig["mark_version"] == "mark5.2"]
    M_eval = set(k for k in orig[orig["mark_version"].isin(["mark5.2", "mark5.2_eval"]) & orig["assigned_split"].isin(
        ["val", "test", "eval_far_heldout", "eval_boundary_B", "eval_zeroshot6"])]["_rk"] if k is not None)

    # ── 1단계: Mark5.3 아이 말소리 교체 ─────────────────────────────────────
    tgt = m52[(m52["source"] == "chime_home") & m52["target_class"].isin(TALK) & m52["aihub_category"].map(is_child)]
    say("[1단계] 아이 말소리 talking: " + " · ".join(f"{s} {n}" for s, n in tgt.groupby("assigned_split").size().items()) + f" = {len(tgt)}")
    used = set(act[act["source"] == "chime_home"]["aihub_src_file"].dropna().astype(str))   # 지금 살아 있는 행이 쓰는 청크
    sess_side = {}                                                       # 세션 → Mark5.3 쪽(train·val·test)
    pool = []
    for it in R.build_chime():
        sess_side[it["volume"]] = it["split"]
        if it["role"] in ("tv", "speech") and not is_child(it["category"]) and it["src"] not in used:
            pool.append(it)
    cache_path = os.path.join(R.CACHE_DIR, "gate_cache.json")
    cache = json.load(open(cache_path))
    passed = R.gate_pool(pool, cache)
    json.dump(cache, open(cache_path, "w"))
    by_sess = defaultdict(lambda: {"speech": [], "tv": []})
    for it in sorted(passed, key=lambda x: x["src"]):
        by_sess[it["volume"]][it["role"]].append(it)
    rng = random.Random(SEED)
    for s in by_sess.values():
        rng.shuffle(s["speech"]); rng.shuffle(s["tv"])
    side_of = lambda split: "train" if split in ("train", "unlabeled") else split
    taken = set()
    def take(sess):
        for role in ("speech", "tv"):
            for it in by_sess[sess][role]:
                if it["src"] not in taken:
                    taken.add(it["src"]); return it
        return None
    plan1 = []                                   # (옛 행, 새 item)
    moved, short = Counter(), Counter()
    rows_sorted = sorted(tgt.iterrows(), key=lambda x: (str(x[1]["aihub_volume"]), x[1]["assigned_split"], x[1]["local_filename"]))
    pending = []
    for _, r in rows_sorted:                     # ① 같은 세션
        it = take(str(r["aihub_volume"]))
        (plan1.append((r, it)) if it else pending.append(r))
    others = defaultdict(list)                   # ② 같은 split 쪽의 다른 세션(세션을 돌아가며)
    for r in pending:
        sides = sorted(s for s in by_sess if sess_side.get(s) == side_of(r["assigned_split"]) and s != str(r["aihub_volume"]))
        it, k = None, len(others[side_of(r["assigned_split"])])
        for j in range(len(sides)):
            it = take(sides[(k + j) % len(sides)])
            if it:
                break
        others[side_of(r["assigned_split"])].append(1)
        if it:
            plan1.append((r, it)); moved[f"{r['aihub_volume']}→{it['volume']}"] += 1
        else:
            short[str(r["aihub_volume"])] += 1
    for sess, rows in sorted(tgt.groupby(tgt["aihub_volume"].astype(str))):
        say(f"   {sess}({sess_side.get(sess)}): 교체 {len(rows)} · 후보 어른 말소리 {len(by_sess[sess]['speech'])} · TV {len(by_sess[sess]['tv'])}")
    new_src = [it["src"] for _, it in plan1]
    chk1 = dict(
        count=len(plan1) == len(tgt) and not short,
        same_side=all(sess_side.get(it["volume"]) == side_of(r["assigned_split"]) for r, it in plan1),
        no_child=not any(is_child(it["category"]) for _, it in plan1),
        no_dup=len(set(new_src)) == len(new_src),
        not_in_use=not (set(new_src) & used),
    )
    mix = Counter((r["assigned_split"], it["role"]) for r, it in plan1)
    say("[1단계 검사] " + " · ".join(f"{k}={v}" for k, v in chk1.items()) + f" · 부족 {dict(short)} · 다른 세션에서 {dict(moved)} · 교체 구성 {dict(mix)}")

    # ── 2단계: 교사 Mark4.5 talking val/test 다시 나누기 ─────────────────────
    repl = {(r["assigned_split"], r["local_filename"]): it for r, it in plan1}
    rows2 = m52[m52["assigned_split"].isin(["train", "unlabeled"]) & m52["target_class"].isin(TALK)]
    items = []
    for r in rows2.to_dict("records"):
        if r["_rk"] in M_eval:
            continue
        row = dict(r)
        it_new = repl.get((r["assigned_split"], r["local_filename"]))
        key = r["_rk"]
        if it_new is not None:                   # 1단계 뒤의 내용으로 본다(파일은 같은 이름, 세션은 새 청크의 세션)
            row.update(aihub_category=it_new["category"], original_labels=it_new["category"], aihub_src_file=it_new["src"],
                       aihub_volume=it_new["volume"])
            key = rec_key(pd.Series(row))
        items.append(dict(key=key, row=row, tv=RS.talk_is_tv(row), chime=row["source"] == "chime_home",
                          social=row["source"] == "sins" and row["aihub_category"] == "social_activity",
                          src_path=m52_file(r["assigned_split"], r["local_filename"]),
                          from_=f"mark5.2:{r['assigned_split']}:{r['local_filename']}"))
    sess = defaultdict(list)
    for it in sorted(items, key=lambda x: x["from_"]):
        sess[it["key"]].append(it)
    names = sorted(sess)
    V = RS.VT_SIZE
    tv_target = sum(it["tv"] for it in items) / len(items)
    lo, hi = int((tv_target - 0.20) * V + 0.999), int(min(1.0, tv_target + 0.20) * V)
    say(f"[2단계] 학습 쪽 talking {len(items)}개 · 세션 {len(sess)} · TV 목표 {tv_target:.2f} (각 split {lo}~{hi}/{V})")

    def pick(p):
        tv = [it for it in p if it["tv"]]; sp = [it for it in p if not it["tv"]]
        n_tv = min(max(int(round(tv_target * V)), V - len(sp)), len(tv))
        if not (lo <= n_tv <= hi) or V - n_tv > len(sp):
            return None
        out = []
        for part, n in ((tv, n_tv), (sp, V - n_tv)):
            out += RS.rr_by(part, lambda x: (x["chime"], sub_key(x["row"])), n, SEED)
        return out

    for n in names:
        say(f"   세션 {n[1]}: {len(sess[n])}개 · TV {sum(it['tv'] for it in sess[n])} · 말소리 {sum(not it['tv'] for it in sess[n])}"
            f" · 하위라벨 {sorted({sub_key(it['row']) for it in sess[n]})}")
    best, why = None, Counter()
    for mask in range(1, 2 ** len(names) - 1):
        side = {s: [it for i, n in enumerate(names) if ((mask >> i) & 1) == (s == "val") for it in sess[n]] for s in ("val", "test")}
        if any(len(side[s]) < V for s in side):
            why["크기"] += 1; continue
        got = {s: pick(side[s]) for s in side}
        if any(g is None for g in got.values()):
            why["TV 범위"] += 1; continue
        if not {sub_key(it["row"]) for it in got["test"]} <= {sub_key(it["row"]) for it in got["val"]}:
            why["하위라벨"] += 1; continue
        f = {s: (sum(it["tv"] for it in got[s]) / V, sum(it["chime"] for it in got[s]) / V, sum(it["social"] for it in got[s]) / V) for s in got}
        score = (abs(f["val"][0] - tv_target) + abs(f["test"][0] - tv_target) + abs(f["val"][1] - f["test"][1])
                 + abs(f["val"][2] - f["test"][2]))
        if best is None or score < best[0]:
            best = (score, got, f)
    if best is None:
        raise SystemExit(f"[STOP] 2단계: 조건을 만족하는 세션 배정이 없습니다 — 탈락 사유 {dict(why)}")
    _, vt, f = best
    for s in ("val", "test"):
        say(f"[2단계] {s}: 세션 {len({it['key'] for it in vt[s]})} · TV {f[s][0]:.2f} · CHiME {f[s][1]:.2f} · SINS 대화 {f[s][2]:.2f}"
            f" · 구성 {dict(Counter(('CHiME' if it['chime'] else 'SINS') + ('TV' if it['tv'] else '말소리') for it in vt[s]))}")
    kv, kt = {it["key"] for it in vt["val"]}, {it["key"] for it in vt["test"]}
    chk2 = dict(size=all(len(vt[s]) == V for s in vt), sess_disjoint=not (kv & kt), vs_m53_eval=not ((kv | kt) & M_eval),
                no_child=not any(is_child(it["row"]["aihub_category"]) for s in vt for it in vt[s] if it["chime"]),
                no_dup=all(len({it["from_"] for it in vt[s]}) == V for s in vt))
    say("[2단계 검사] " + " · ".join(f"{k}={v}" for k, v in chk2.items()))
    ok = all(chk1.values()) and all(chk2.values())
    json.dump({"step1": [dict(split=r["assigned_split"], name=r["local_filename"], prev=r["aihub_src_file"], new=it["src"], role=it["role"])
                         for r, it in plan1],
               "step2": {s: [it["from_"] for it in vt[s]] for s in vt}},
              open(os.path.join(R.CACHE_DIR, f"nochild_{TODAY}_{'dryrun' if a.dry_run else 'apply'}.json"), "w"), ensure_ascii=False, indent=0)
    say(f"→ {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
    if a.dry_run:
        open(os.path.join(R.CACHE_DIR, f"nochild_{TODAY}_dryrun.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
        print("[DRY RUN] 파일·provenance 는 바꾸지 않았습니다.")
        return
    if not ok:
        raise SystemExit("[STOP] 기대값 불일치 — 실행하지 않음")

    # ── 실행 ──
    backup = R.PROV_PATH + f".bak_before_nochild_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)
    new_rows = []
    keep = os.path.join(R.PROJECT_ROOT, f"_replaced_{TODAY}")
    for r, it in plan1:                                                     # 1단계 — Mark5.3 파일 교체
        dst = m52_file(r["assigned_split"], r["local_filename"])
        old = m52_file(r["assigned_split"], r["local_filename"], root=keep)
        os.makedirs(os.path.dirname(old), exist_ok=True)
        shutil.move(dst, old)
        seg, start = R.load_segment(it)
        if seg is None:
            raise SystemExit(f"[ERROR] 게이트 캐시와 다름: {it['src']} {start}")
        sf.write(dst, seg, R.TARGET_SR, subtype="FLOAT")
        row = R.prov_row(r["local_filename"], r["target_class"], r["assigned_split"], it, start, dst)
        row["mark52_note"] = f"{NOTE_M53}(prev={r['aihub_src_file']})"
        new_rows.append(row)
        drive = m52_file(r["assigned_split"], r["local_filename"], root=STAGE_DRIVE, dirs=M52_OUT)   # 3단계 — 드라이브용 data_* 판
        os.makedirs(os.path.dirname(drive), exist_ok=True)
        process_audio_file(dst, drive)
        src_copy = m52_file(r["assigned_split"], r["local_filename"], root=STAGE_DRIVE)
        os.makedirs(os.path.dirname(src_copy), exist_ok=True)
        shutil.copyfile(dst, src_copy)
    m = prov.index.isin(tgt.index)
    prov.loc[m, "removed_20260715"] = TAG_M53
    say(f"[실행 1] Mark5.3 {len(plan1)}개 교체(옛 파일 → {keep}) · 드라이브용 판 {STAGE_DRIVE}")
    by_from = {f"mark5.2:{r['assigned_split']}:{r['local_filename']}": nr for (r, _), nr in zip(plan1, new_rows)}
    root45 = os.path.join(R.EXPERT_ROOT, "Mark4.5", "data")                 # 2단계 — 교사 val/test
    rep45 = os.path.join(root45, f"_replaced_{TODAY}")
    for s in ("val", "test"):
        os.makedirs(os.path.join(rep45, s), exist_ok=True)
        for fn in os.listdir(os.path.join(root45, s)):
            if fn.startswith(f"talking_{s}_"):
                shutil.move(os.path.join(root45, s, fn), os.path.join(rep45, s, fn))
        for i, it in enumerate(vt[s], 1):
            name = f"talking_{s}_{i:03d}.wav"
            dst = os.path.join(root45, s, name)
            shutil.copyfile(it["src_path"], dst)
            base = by_from.get(it["from_"]) or {k: v for k, v in it["row"].items() if k != "_rk"}
            row = dict(base)
            row.update(local_filename=name, assigned_split=s, mark_version="mark4.5", target_class="talking", removed_20260715="active",
                       source_type="original", sha256=sha(dst), size_bytes=os.path.getsize(dst), mark52_note=f"{NOTE_M45}(from={it['from_']})")
            new_rows.append(row)
    m45 = ((prov["mark_version"] == "mark4.5") & (prov["target_class"] == "talking") & prov["assigned_split"].isin(["val", "test"])
           & (prov["removed_20260715"].astype(str) == "active") & (prov["source_type"] == "original"))
    prov.loc[m45, "removed_20260715"] = TAG_M45
    say(f"[실행 2] Mark4.5 talking val/test {sum(len(vt[s]) for s in vt)}개 새로(옛 행 {int(m45.sum())} → {TAG_M45})")
    prov = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
    prov.to_excel(R.PROV_PATH, index=False)
    say(f"[provenance] 새 행 {len(new_rows)} · 백업 {os.path.basename(backup)} ({time.time()-t0:.0f}s)")
    open(os.path.join(R.CACHE_DIR, f"nochild_{TODAY}.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    print(f"[완료] ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
