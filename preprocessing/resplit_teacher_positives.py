#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resplit_teacher_positives.py — 교사 train 이 Mark5.3 학습 쪽과 같은 녹음을 공유하지 않게 교사 양성(타겟)을 다시 나눈다 (2026-10-23).

왜: 10-23 점검에서 교사 train 과 Mark5.3 정답 라벨·라벨 없는 클립이 같은 녹음을 82~227개씩 공유했다(09-07 방침 "전문가 train 은
    Mark5.2 train 과 겹쳐도 된다"의 결과). 교사가 이미 들은 녹음에 가짜 정답을 붙이면 그 가짜 정답은 실제보다 좋아 보인다.
    그리고 교사 val/test 가 Mark5.3 평가 쪽(val·test·eval) 녹음을 29~77개씩 써서, 교사를 고를 때 Mark5.3 평가 녹음을 보고 있었다.
무엇을 (talking 제외 7개 교사 — talking 은 출처를 따로 정한다):
    train (크기 = 지금 원본 수, --train_size 로 바꿀 수 있음)
      = Mark5.3 이 어디에도 쓰지 않은 녹음만. 우선순위: ① 지금 train 의 그런 클립(그대로) ② 지금 val/test 의 그런 AI Hub 클립(옮김)
        ③ 교사도 안 쓴 AI Hub 녹음(기존 게이트 load_segment 로 새로 자름) ④ 외부 자료(gate_external_candidates.py 통과분).
        ①~③ 은 지금 train 의 AI Hub 하위라벨 비율을 따라 채우고, ④ 는 묶음(업로더·녹음 위치)을 돌아가며 시드 42 로 고른다.
    val / test (각 215)
      = 지금 val/test 클립 중 Mark5.3 평가 쪽이 아닌 것(train 으로 옮긴 것 제외) + 모자라면 Mark5.3 학습 쪽 클립(Mark5.2 파일 그대로 복사).
        Mark5.3 학습 쪽 녹음은 교사 val/test 에만 들어간다(교사가 배우지 않고, 교사를 고르고 재는 데만 쓴다 — audit ⑨ R4).
        채울 때는 하위라벨을 val·test 에 같은 순서로 번갈아 넣는다(audit ⑧: test 하위라벨 ⊆ val 하위라벨).
    파일·기록
      옛 타겟 원본은 data/_replaced_20261023/{split}/ 로 옮긴다(지우지 않는다). 새 파일 이름은 <클래스>_<split>_NNN.wav.
      train 증강본은 건드리지 않는다 — 뒤의 augment_density --reset 이 지금 폴더의 증강본 이름으로 provenance 행을 찾아 함께 지운다.
      provenance: 옛 타겟 원본 행 removed_20260715="replaced_resplit_20261023", 새 행 mark52_note="resplit_20261023(from=…)".
      외부 클립 행: source=fsd50k(fsd50k_fname) · clotho/bbc/bedavin/snu_b36_50e(aihub_src_file=원본 ID), 이용조건·주소는 mark52_note.
기대값(실행 전 검사): 교사마다 train ∩ Mark5.3 전체(학습·평가) 녹음 = 0, val/test ∩ Mark5.3 평가 쪽 = 0, train ∩ val/test = 0,
    val ∩ test = 0, 크기 일치, test 하위라벨 ⊆ val 하위라벨.
뒤 단계: reallocate_hard_negatives.py(다른 타겟 자리를 새 풀에서 다시) → run_augment_all.sh → run_expert_index_all.sh → verify_after_rebuild.sh
사용: python preprocessing/resplit_teacher_positives.py --dry_run [--train_size water_shower=500] | (실행)
"""
import os, re, sys, csv, json, math, time, shutil, argparse, random
from collections import Counter, defaultdict
from datetime import date

import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402
from cross_reject_experts import rec_key, sub_key, data_path, sha, VER_OF, SEED   # noqa: E402
from mark5_slicer import scan_aihub                               # noqa: E402

TODAY = "20261023"
TAG_OLD, NOTE = f"replaced_resplit_{TODAY}", f"resplit_{TODAY}"
CLASSES7 = ["heavy_impact", "dragging", "construction", "machine_noise", "water_toilet", "water_shower", "dog_bark"]
VT_SIZE = 215
EXT_ROOT = "/mnt/c/Users/user/journal_external_raw"
GATE_REPORT = os.path.join(EXT_ROOT, "gated", "gate_report.csv")
M52_DIRS = {"train": "data_source_labeled", "unlabeled": "data_source_unlabeled"}


def m52_path(row):
    if row["assigned_split"] == "train":
        return os.path.join(R.PROJECT_ROOT, M52_DIRS["train"], row["target_class"], row["local_filename"])
    return os.path.join(R.PROJECT_ROOT, M52_DIRS["unlabeled"], row["local_filename"])


EXT_SUFFIX = {"snu_b36_50e": ".m4a", "bbc": ".wav", "bedavin": ".wav"}     # 원본 파일 이름(숫자로 읽히지 않게)


def ext_src_name(source, ext_id, freesound_id):
    """provenance aihub_src_file 에 적을 외부 원본 ID. Clotho 는 Freesound ID, 나머지는 원본 파일 이름."""
    if source == "voices":
        return "rm4-tele"                                                # 마이크·시각이 달라도 같은 TV 녹음 하나
    return str(freesound_id) if source == "clotho" else f"{ext_id}{EXT_SUFFIX[source]}"


def ext_key(source, ext_id, freesound_id):
    """외부 클립의 녹음 키 — audit_dataset.rec_key 의 외부 출처 규칙과 같다."""
    if source in ("fsd50k", "clotho"):
        return ("F", str(int(float(freesound_id))))
    return ("X", f"{source}:{ext_src_name(source, ext_id, freesound_id)}")


def rr_by(items, key, want, seed):
    """key 로 묶어 묶음을 정렬해 돌아가며 want 개. 묶음 안은 시드 셔플."""
    g = defaultdict(list)
    for it in items:
        g[key(it)].append(it)
    rng = random.Random(seed)
    ks = sorted(g)
    for k in ks:
        rng.shuffle(g[k])
    out = []
    while len(out) < want and any(g[k] for k in ks):
        for k in ks:
            if g[k] and len(out) < want:
                out.append(g[k].pop(0))
    return out


def fill_by_category(pools, target_mix, want):
    """pools: 우선순위 순서의 [후보 목록]. target_mix: {하위라벨: 비율}. 하위라벨 몫을 우선순위대로 채우고 남으면 아무 하위라벨로."""
    quota = {k: int(round(want * v)) for k, v in target_mix.items()}
    picked, seen = [], set()
    for cat in sorted(quota):
        for pool in pools:
            for it in pool:
                if len([p for p in picked if p["cat"] == cat]) >= quota[cat]:
                    break
                if it["cat"] == cat and it["key"] not in seen:
                    picked.append(it); seen.add(it["key"])
    for pool in pools:
        for it in pool:
            if len(picked) >= want:
                break
            if it["key"] not in seen:
                picked.append(it); seen.add(it["key"])
    return picked[:want]


# ── talking (2026-10-23 수진님 결정: train = FSD50K 말소리 + VOiCES TV, val/test = Mark5.3 학습 쪽 talking 세션을 세션 단위로 나눔) ──
# 10-23 수진님 결정: VOiCES TV 구간은 게이트 통과가 238 이라 train = 250 + 238 = 488(증강으로 1,000). --talk_quota 로 바꿀 수 있다
TALK_TRAIN_QUOTA = {"fsd50k": 250, "voices": 238}
TALK_TV_RANGE = (0.40, 0.60)                            # val·test 각각 TV 비율


def talk_is_tv(row):
    """SINS watching_tv, CHiME 라벨에 v(TV·비디오)가 든 것. (SINS social_activity 는 글자 v 가 있어도 말소리다 — 10-23 오계산 원인)"""
    if row["source"] == "sins":
        return row["aihub_category"] == "watching_tv"
    return "v" in str(row["aihub_category"])


def plan_talking(m52, gate, M_all, M_eval, say, quota=None):
    """train: 게이트 통과 FSD50K 말소리·VOiCES TV 를 quota 만큼(묶음을 돌아가며). val/test: Mark5.3 학습 쪽 talking 클립을
    세션(rec_key)이 두 split 에 걸치지 않게 나눈다 — 각 VT_SIZE · TV 비율 TALK_TV_RANGE · test 하위라벨 ⊆ val 하위라벨.
    세션 배정은 전수(2^세션 수)로 보고, 두 split 의 TV 비율이 0.5 에 가장 가까운 배정을 고른다(같으면 먼저 나온 것)."""
    quota = quota or TALK_TRAIN_QUOTA
    g = gate[(gate["cls"] == "talking") & (gate["status"] == "통과")]
    train_items = []
    for src, want in quota.items():
        pool = []
        for r in g[g["source"] == src].to_dict("records"):
            k = ext_key(r["source"], r["ext_id"], r["freesound_id"])
            if k in M_all:
                continue
            pool.append(dict(key=k, cat=f"ext_{src}", kind="external", src_path=r["out_path"], ext=r, from_=f"{src}:{r['ext_id']}"))
        pool.sort(key=lambda x: x["from_"])
        got = rr_by(pool, lambda x: x["ext"]["group"], want, SEED)
        say(f"   talking train {src}: {len(got)}/{want} (게이트 통과 후보 {len(pool)})")
        train_items += got
    rows = m52[m52["assigned_split"].isin(["train", "unlabeled"]) & m52["target_class"].isin(["talking", "unlabeled:talking"])]
    items = [dict(key=r["_rk"], cat=r.get("aihub_category") or f"src:{r['source']}", kind="m52", src_path=m52_path(r), row=r,
                  tv=talk_is_tv(r), from_=f"mark5.2:{r['assigned_split']}:{r['local_filename']}")
             for r in rows.to_dict("records") if r["_rk"] not in M_eval]
    sess = defaultdict(list)
    for it in sorted(items, key=lambda x: x["from_"]):
        sess[it["key"]].append(it)
    names = sorted(sess)
    lo, hi = math.ceil(TALK_TV_RANGE[0] * VT_SIZE), math.floor(TALK_TV_RANGE[1] * VT_SIZE)     # 215 → 86 ~ 129

    def pick(pool):
        """pool 에서 VT_SIZE 개: TV 몫을 0.5 에 가깝게(범위 안) 정하고, 하위라벨을 돌아가며 고른다. 안 되면 None."""
        tv = [it for it in pool if it["tv"]]
        sp = [it for it in pool if not it["tv"]]
        n_tv = min(max(VT_SIZE // 2, VT_SIZE - len(sp)), len(tv))
        if not (lo <= n_tv <= hi) or VT_SIZE - n_tv > len(sp):
            return None
        out = []
        for part, n in ((tv, n_tv), (sp, VT_SIZE - n_tv)):
            out += rr_by(part, lambda x: sub_key(x["row"]), n, SEED)
        return out

    best = None
    for mask in range(1, 2 ** len(names) - 1):
        side = {s: [it for i, n in enumerate(names) if ((mask >> i) & 1) == (s == "val") for it in sess[n]] for s in ("val", "test")}
        if any(len(side[s]) < VT_SIZE for s in side):
            continue
        got = {s: pick(side[s]) for s in side}
        if any(v is None for v in got.values()):
            continue
        if not {sub_key(it["row"]) for it in got["test"]} <= {sub_key(it["row"]) for it in got["val"]}:
            continue
        score = sum(abs(sum(it["tv"] for it in got[s]) / VT_SIZE - 0.5) for s in got)
        if best is None or score < best[0]:
            best = (score, mask, got)
    if best is None:
        say("   talking val/test: 조건(세션 분리 · 각 215 · TV 40~60% · 하위라벨 포함)을 만족하는 세션 배정이 없습니다")
        return train_items, [], []
    _, mask, got = best
    for s in ("val", "test"):
        ss = sorted({it["key"][1] for it in got[s]})
        say(f"   talking {s}: 세션 {len(ss)}개 {ss} · TV {sum(it['tv'] for it in got[s])}/{VT_SIZE}")
    return train_items, got["val"], got["test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--train_size", nargs="*", default=[], help="클래스=크기 (기본은 지금 원본 수)")
    ap.add_argument("--talk_quota", nargs="*", default=[], help="talking train 출처=개수 (기본 fsd50k=250 voices=238)")
    ap.add_argument("--classes", nargs="*", default=CLASSES7 + ["talking"])
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
    m52 = orig[orig["mark_version"] == "mark5.2"]
    keys_of = lambda df: set(k for k in df["_rk"] if k is not None)
    M_eval = keys_of(m52[m52["assigned_split"].isin(["val", "test"])]) | keys_of(orig[orig["mark_version"] == "mark5.2_eval"])
    M_learn = keys_of(m52[m52["assigned_split"].isin(["train", "unlabeled"])])
    M_all = M_eval | M_learn
    gate = pd.read_csv(GATE_REPORT, encoding="utf-8-sig", dtype=str).fillna("")
    sizes = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.train_size}
    cache_path = os.path.join(R.CACHE_DIR, "gate_cache.json")
    gcache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    aihub_scan = scan_aihub(R.AIHUB_ROOT)

    VOICES_KEY = ("X", "voices:rm4-tele")

    def finish(cls, ver, n_train, train_items, vt, new_aihub, ext_pool, ext_picked):
        """검사·보고. 반환: 통과 여부."""
        tk = [it["key"] for it in train_items]
        vk = {s: [it["key"] for it in vt[s]] for s in ("val", "test")}
        chk = dict(
            size=(len(train_items) == n_train and all(len(vt[s]) == VT_SIZE for s in vt)),
            train_vs_m53=len(set(tk) & M_all) == 0,
            vt_vs_m53eval=len((set(vk["val"]) | set(vk["test"])) & M_eval) == 0,
            train_vs_vt=len(set(tk) & (set(vk["val"]) | set(vk["test"]))) == 0,
            val_vs_test=len(set(vk["val"]) & set(vk["test"])) == 0,
            # val/test 는 같은 '클립 파일'이 두 번 들어갔는지 본다(from_). 녹음 키로 보면 talking 처럼 긴 세션에서 자른 클립들이 중복으로 잡힌다.
            no_dup=len(set(tk)) == len(tk) - max(0, tk.count(VOICES_KEY) - 1)
                   and all(len({it["from_"] for it in vt[s]}) == len(vt[s]) for s in vt),
            sub_ok={sub_key(it.get("row") or {"source": it["kind"], "aihub_category": it["cat"]}) for it in vt["test"]}
                   <= {sub_key(it.get("row") or {"source": it["kind"], "aihub_category": it["cat"]}) for it in vt["val"]},
        )
        ok = all(chk.values())
        comp = Counter(it["kind"] for it in train_items)
        say(f"[{ver} {cls}] train {len(train_items)}/{n_train} = 지금 train 유지 {sum(1 for it in train_items if it['kind']=='teacher' and it['from_'].split(':')[1]=='train')}"
            f" · val/test→train {sum(1 for it in train_items if it['kind']=='teacher' and it['from_'].split(':')[1]!='train')}"
            f" · 새 AI Hub {comp['aihub_new']} (후보 {len(new_aihub)}) · 외부 {comp['external']} (후보 {len(ext_pool)})")
        for s in ("val", "test"):
            c2 = Counter(it["kind"] for it in vt[s])
            say(f"   {s:4s} {len(vt[s])}/{VT_SIZE} = 지금 {s} 유지 {c2['teacher']} · Mark5.3 학습 쪽에서 {c2['m52']}")
        say(f"   검사: " + " · ".join(f"{k}={v}" for k, v in chk.items()) + f" → {'OK' if ok else 'FAIL'}")
        if comp["external"]:
            say(f"   외부 구성: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(it['ext']['source'] for it in ext_picked).items())))
        return ok

    plans, ok_all = {}, True
    for cls in a.classes:
        ver = VER_OF[cls]
        if cls == "talking":
            tq = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.talk_quota} or TALK_TRAIN_QUOTA
            train_items, vt_val, vt_test = plan_talking(m52, gate, M_all, M_eval, say, tq)
            vt = {"val": vt_val, "test": vt_test}
            ok_all &= finish(cls, ver, sizes.get(cls, sum(tq.values())), train_items, vt, [], [], train_items)
            plans[cls] = dict(ver=ver, train=train_items, val=vt["val"], test=vt["test"], cur=None)
            continue
        e = orig[(orig["mark_version"] == ver) & (orig["target_class"] == cls)]
        cur = {s: e[e["assigned_split"] == s].to_dict("records") for s in ("train", "val", "test")}
        n_train = sizes.get(cls, len(cur["train"]))
        held = {r["_rk"] for s in cur for r in cur[s]}
        # 후보 항목: dict(key, cat, kind, src_path, row)
        def item_from_teacher(r, s):
            return dict(key=r["_rk"], cat=r.get("aihub_category") if r["source"] == "aihub" else f"src:{r['source']}", kind="teacher",
                        src_path=data_path(ver, s, r["local_filename"]), row=r, from_=f"{ver}:{s}:{r['local_filename']}")
        clean_tr = [item_from_teacher(r, "train") for r in cur["train"] if r["_rk"] not in M_all]
        clean_vt_aihub = [item_from_teacher(r, s) for s in ("val", "test") for r in cur[s] if r["_rk"] not in M_all and r["source"] == "aihub"]
        # ③ 교사도 안 쓴 AI Hub 녹음 (같은 카테고리, Mark5.3 미사용) — 기존 게이트로 자른다
        cats = set(e[e["source"] == "aihub"]["aihub_category"].dropna())
        new_aihub = []
        n_fsd_keep = sum(1 for it in clean_tr if it["row"]["source"] == "fsd50k")
        have_aihub = sum(1 for it in clean_tr if it["row"]["source"] == "aihub") + len(clean_vt_aihub)
        for c in (aihub_scan if have_aihub < n_train - n_fsd_keep else []):     # 이미 충분하면 새로 자르지 않는다
            if c["category"] not in cats:
                continue
            for w in c["wavs"]:
                k = ("A", w)
                if k in M_all or k in held:
                    continue
                lp = os.path.join(c["label_dir"], os.path.splitext(w)[0] + ".json") if c["label_dir"] else None
                it = dict(src=w, path=os.path.join(c["dir"], w), label=lp, source="aihub", category=c["category"], sess="",
                          group="", loader="aihub", volume=c["volume"])
                if w not in gcache:
                    seg, info = R.load_segment(it)
                    gcache[w] = {"pass": seg is not None, "start": float(info) if seg is not None else None,
                                 "why": "" if seg is not None else str(info)}
                if gcache[w]["pass"]:
                    new_aihub.append(dict(key=k, cat=c["category"], kind="aihub_new", item=it, src_path=it["path"], from_=f"aihub:{w}"))
        new_aihub = sorted(new_aihub, key=lambda x: x["key"])
        # 지금 train 의 AI Hub 하위라벨 비율
        mix = Counter(r["aihub_category"] for r in cur["train"] if r["source"] == "aihub")
        tot = sum(mix.values())
        mix = {k: v / tot for k, v in mix.items()}
        # dog_bark: FSD50K 클립은 하위라벨 비율 밖이다 — 지금 train 의 FSD50K 는 ①로 그대로 둔다
        fsd_keep = [it for it in clean_tr if it["row"]["source"] == "fsd50k"]
        aihub_tr = [it for it in clean_tr if it["row"]["source"] == "aihub"]
        n_aihub_want = n_train - len(fsd_keep)
        rng = random.Random(SEED)
        for lst in (aihub_tr, clean_vt_aihub):
            lst.sort(key=lambda x: x["from_"])
        rng.shuffle(new_aihub)
        picked_aihub = fill_by_category([aihub_tr, clean_vt_aihub, new_aihub], mix, min(n_aihub_want, len(aihub_tr) + len(clean_vt_aihub) + len(new_aihub)))
        train_items = fsd_keep + picked_aihub
        # ④ 외부
        n_ext = n_train - len(train_items)
        ext_pool = []
        g = gate[(gate["cls"] == cls) & (gate["status"] == "통과")]
        for r in g.to_dict("records"):
            k = ext_key(r["source"], r["ext_id"], r["freesound_id"])
            if k in M_all or k in held:
                continue
            ext_pool.append(dict(key=k, cat=f"ext_{r['source']}", kind="external", src_path=r["out_path"], ext=r, from_=f"{r['source']}:{r['ext_id']}"))
        ext_picked = rr_by(ext_pool, lambda x: (x["ext"]["source"], x["ext"]["group"]), max(n_ext, 0), SEED) if n_ext > 0 else []
        train_items += ext_picked
        train_keys = {it["key"] for it in train_items}
        # val / test
        moved = {it["from_"] for it in train_items if it["kind"] == "teacher"}
        vt = {}
        used_keys = set(train_keys)
        for s in ("val", "test"):
            keep = [item_from_teacher(r, s) for r in cur[s] if r["_rk"] not in M_eval and f"{ver}:{s}:{r['local_filename']}" not in moved]
            vt[s] = keep
            used_keys |= {it["key"] for it in keep}
        learn_rows = m52[m52["assigned_split"].isin(["train", "unlabeled"]) & m52["target_class"].isin([cls, f"unlabeled:{cls}"])].to_dict("records")
        learn_pool = [dict(key=r["_rk"], cat=r.get("aihub_category") if r["source"] == "aihub" else f"src:{r['source']}", kind="m52",
                           src_path=m52_path(r), row=r, from_=f"mark5.2:{r['assigned_split']}:{r['local_filename']}")
                      for r in learn_rows if r["_rk"] not in used_keys]
        by_cat = defaultdict(list)
        for it in sorted(learn_pool, key=lambda x: x["from_"]):
            by_cat[it["cat"]].append(it)
        for c in by_cat:
            random.Random(SEED).shuffle(by_cat[c])
        # 모자란 split 에 하위라벨을 같은 순서로 번갈아 — val 먼저, 같은 하위라벨을 test 에도
        need = {s: VT_SIZE - len(vt[s]) for s in ("val", "test")}
        cats_order = sorted(by_cat, key=lambda c: -len(by_cat[c]))
        while (need["val"] > 0 or need["test"] > 0) and any(by_cat[c] for c in cats_order):
            for c in cats_order:
                for s in ("val", "test"):
                    if need[s] > 0 and by_cat[c]:
                        it = by_cat[c].pop(0)
                        if it["key"] in used_keys:
                            continue
                        vt[s].append(it); used_keys.add(it["key"]); need[s] -= 1
        ok_all &= finish(cls, ver, n_train, train_items, vt, new_aihub, ext_pool, ext_picked)
        plans[cls] = dict(ver=ver, train=train_items, val=vt["val"], test=vt["test"], cur=cur)
    json.dump(gcache, open(cache_path, "w"))
    say(f"[기대값] {'OK' if ok_all else 'FAIL'} ({time.time()-t0:.0f}s)")
    out_json = os.path.join(R.CACHE_DIR, f"resplit_{TODAY}_{'dryrun' if a.dry_run else 'apply'}.json")
    json.dump({cls: {s: [dict(kind=it["kind"], from_=it["from_"], key=list(it["key"])) for it in p[s]] for s in ("train", "val", "test")}
               for cls, p in plans.items()}, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    if a.dry_run:
        open(os.path.join(R.CACHE_DIR, f"resplit_{TODAY}_dryrun.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
        print("[DRY RUN] 파일·provenance 는 바꾸지 않았습니다.")
        return
    if not ok_all:
        raise SystemExit("[STOP] 기대값 불일치 — 실행하지 않음")

    # ── 실행 ──────────────────────────────────────────────────────────────
    backup = R.PROV_PATH + f".bak_before_resplit_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)
    new_rows = []
    for cls, p in plans.items():
        ver = p["ver"]; root = os.path.join(R.EXPERT_ROOT, "Mark" + ver[4:], "data")
        stage = os.path.join(root, f"_stage_resplit_{TODAY}")
        # 1) 새 파일을 스테이징에 먼저 만든다(옛 파일을 읽어야 하므로)
        for s in ("train", "val", "test"):
            os.makedirs(os.path.join(stage, s), exist_ok=True)
            for i, it in enumerate(p[s], 1):
                name = f"{cls}_{s}_{i:03d}.wav"; dst = os.path.join(stage, s, name)
                if it["kind"] == "aihub_new":
                    seg, start = R.load_segment(it["item"])
                    sf.write(dst, seg, R.TARGET_SR, subtype="FLOAT")
                    row = R.prov_row(name, cls, s, it["item"], start, dst, version=ver)
                else:
                    shutil.copyfile(it["src_path"], dst)
                    if it["kind"] == "external":
                        x = it["ext"]
                        row = {"local_filename": name, "original_labels": x.get("labels") or x["title"], "target_class": cls,
                               "assigned_split": s, "mark_version": ver, "source_type": "original", "removed_20260715": "active",
                               "source": x["source"], "fsd50k_fname": int(x["ext_id"]) if x["source"] == "fsd50k" else None,
                               "aihub_src_file": None if x["source"] == "fsd50k" else ext_src_name(x["source"], x["ext_id"], x["freesound_id"]),
                               "aihub_category": f"ext_{x['source']}", "aihub_volume": x["group"],
                               "seg_start_sec": float(x["seg_start_sec"]), "seg_end_sec": float(x["seg_start_sec"]) + R.SEG_SEC,
                               "download_date": date.today().isoformat()}
                    else:
                        row = dict(it["row"]); row.pop("_rk", None)
                        row.update(local_filename=name, assigned_split=s, mark_version=ver, target_class=cls, removed_20260715="active")
                row["sha256"] = sha(dst); row["size_bytes"] = os.path.getsize(dst)
                ext_note = f";license={it['ext']['license']};url={it['ext']['url']}" if it["kind"] == "external" else ""
                row["mark52_note"] = f"{NOTE}(from={it['from_']}{ext_note})"
                new_rows.append(row)
        # 2) 옛 타겟 원본 + train 증강본을 _replaced 로, 스테이징을 제자리로
        rep = os.path.join(root, f"_replaced_{TODAY}")
        moved = 0
        for s in ("train", "val", "test"):
            os.makedirs(os.path.join(rep, s), exist_ok=True)
            for f in os.listdir(os.path.join(root, s)):
                if f.startswith(f"{cls}_{s}_") and "_aug_" not in f:      # 증강본은 augment_density --reset 이 provenance 와 함께 지운다
                    shutil.move(os.path.join(root, s, f), os.path.join(rep, s, f)); moved += 1
            for f in os.listdir(os.path.join(stage, s)):
                shutil.move(os.path.join(stage, s, f), os.path.join(root, s, f))
        shutil.rmtree(stage)
        m = ((prov["mark_version"] == ver) & (prov["target_class"] == cls) & (prov["removed_20260715"].astype(str) == "active")
             & (prov["source_type"] == "original"))
        prov.loc[m, "removed_20260715"] = TAG_OLD
        say(f"[실행] {ver} {cls}: 새 파일 {sum(len(p[s]) for s in ('train', 'val', 'test'))} · 옛 파일 {moved}개 → _replaced_{TODAY}/ · provenance 옛 행 {int(m.sum())} ({time.time()-t0:.0f}s)")
    prov = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
    prov.to_excel(R.PROV_PATH, index=False)
    say(f"[provenance] 새 행 {len(new_rows)}, 백업 {os.path.basename(backup)} ({time.time()-t0:.0f}s)")
    open(os.path.join(R.CACHE_DIR, f"resplit_{TODAY}.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    print(f"[완료] ({(time.time()-t0)/60:.1f}분) — 다음: reallocate_hard_negatives.py → run_augment_all.sh → run_expert_index_all.sh → verify_after_rebuild.sh")


if __name__ == "__main__":
    main()
