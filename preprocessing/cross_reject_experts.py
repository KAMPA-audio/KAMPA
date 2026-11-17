#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cross_reject_experts.py — 전문가 8개가 "다른 타겟 소리"를 거절하도록 others 를 다시 짠다 + 전문가 train 이 Mark5.2 평가 녹음을 쓰던 것을 정리한다 (2026-09-21 밤, D·A안).

왜: Mark5.2 2차(test Acc 0.9233)의 주범은 unlabeled 가짜 정답 정확도(75%)였다. 전문가가 "내 소리 vs 배경(others 930)" 만 배워
    다른 타겟을 거절하지 못해, 물 내리는 소리에 machine_noise·water_shower 전문가가 같이 손을 들었다(세그먼트 58% 에서 2명 이상).
    → water_toilet 51% · dragging 41% · machine_noise 56%. 그리고 전문가 train 타겟 276클립이 Mark5.2 val·test 와 같은 AI Hub 녹음이었다
    (09-15 규칙은 others 풀에만 걸려 있었고 08-02 타겟 분할은 안 봤다; audit ⑨ 신설).
무엇을 (A안, 크기 유지 500/215/215):
    ① 교환: 전문가 k(4.5 제외) 의 train 타겟 중 Mark5.2 val 녹음 클립 ↔ k 의 val 타겟 중 Mark5.2 train·unlabeled 녹음 클립 (같은 카테고리 우선), test 도 같게.
       파일 내용을 자리끼리 바꾸고(파일명 유지) provenance 는 메타를 따라 바꾼다. 증강본은 4절(--reset)에서 다시 만든다.
    ② 배경 others 부분집합(8곳 공통): train 500→250, val 215→108, test 215→108 — (출처×하위라벨) 비례, 각 ≥1, 시드 42. 남는 자리 = ③.
    ③ 다른 타겟 7종 → 남는 others 자리: train 250 = 36×5 + 35×2, val·test 107 = 15×5 + 16×2 (클래스 순서 heavy_impact … dog_bark 에서 자기 클래스를 뺀 순서로).
       클래스 j 의 풀은 전문가 j 의 (교환 뒤) train 타겟에서 36, val·test 타겟에서 16씩 — (출처×하위라벨) 라운드로빈, val·test 는 같은 순서(audit ⑧ 대칭).
    ④ provenance: 교환 행 removed_20260715="swapped_cross_20260921" + 새 행, 빠지는 배경 행 "replaced_cross_20260921" + 새 행(mark52_note=cross_target_20260921(from=클래스:파일)).
    ⑤ rebuild_cache/expert_common_cross/{train,val,test}/ 에 공통 배경 부분집합을 복사해 둔다(8곳 공통성 검사용). 로그·json 은 rebuild_cache/cross_reject_20260921*.
사용: python preprocessing/cross_reject_experts.py --dry_run | (실행)
"""
import os, sys, re, json, time, shutil, hashlib, argparse, random
from collections import Counter, defaultdict
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402

TODAY = "20260921"
TAG_SWAP, TAG_BG = f"swapped_cross_{TODAY}", f"replaced_cross_{TODAY}"
CACHE = R.CACHE_DIR
STAGE_OUT = os.path.join(CACHE, "expert_common_cross")
CLASSES = ["heavy_impact", "dragging", "construction", "machine_noise", "talking", "water_toilet", "water_shower", "dog_bark"]
VER_OF = {v: k for k, v in R.EXPERT_TARGET_CLASS.items()}          # class -> mark4.k
BG_KEEP = {"train": 250, "val": 108, "test": 108}
TGT_SLOTS = {"train": 250, "val": 107, "test": 107}
POOL = {"train": 36, "val": 16, "test": 16}
SEED = 42


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rec_key(r):
    """audit_dataset.rec_key 와 같은 규칙(녹음 단위)."""
    src = str(r.get("source", ""))
    if src in ("aihub", "esc50"):
        f = r.get("aihub_src_file"); return ("A", str(f)) if pd.notna(f) else None
    if src == "fsd50k":
        f = r.get("fsd50k_fname"); return ("F", str(int(float(f)))) if pd.notna(f) else None
    if src == "clotho":                                   # [10-23] 외부 출처(audit_dataset.rec_key 와 같은 규칙)
        f = r.get("aihub_src_file"); return ("F", str(int(float(f)))) if pd.notna(f) else None
    if src in ("bbc", "bedavin", "snu_b36_50e", "voices", "chime6"):   # [10-24] chime6: 세션 단위
        f = r.get("aihub_src_file"); return ("X", f"{src}:{f}") if pd.notna(f) else None
    if src == "sins_eval":                                               # [10-24] SINS 평가 세트 — 개발 세트와 다른 세션 번호 체계
        m = re.search(r"ex(\d+)", str(r.get("aihub_src_file", ""))); return ("S", "sins_eval:ex" + m.group(1)) if m else None
    if src == "sins":
        m = re.search(r"ex(\d+)", str(r.get("aihub_src_file", ""))); return ("S", "sins:ex" + m.group(1)) if m else None
    if src == "chime_home":
        v = r.get("aihub_volume"); return ("S", "chime:" + str(v)) if pd.notna(v) else None
    return None


def sub_key(r):
    """audit ⑧ 의 하위라벨 키와 같은 규칙."""
    c = r.get("aihub_category"); c = str(c) if pd.notna(c) else ""
    src = str(r.get("source", ""))
    if src == "chime_home":
        c = "".join(sorted(set(c) & set("bov"))) + ("s" if set(c) & set("cfm") else "")
    return (src, c)


def data_path(ver, split, name):
    return os.path.join(R.EXPERT_ROOT, "Mark" + ver[4:], "data", split, name)


def proportional(counts, total):
    """그룹별 개수 dict -> 합이 total 이고 각 ≥1 인 배분(큰 나머지 순)."""
    n = sum(counts.values()); keys = sorted(counts)
    raw = {k: counts[k] * total / n for k in keys}
    alloc = {k: max(1, int(raw[k])) for k in keys}
    diff = total - sum(alloc.values())
    order = sorted(keys, key=lambda k: -(raw[k] - int(raw[k])))
    i = 0
    while diff != 0:
        k = order[i % len(order)]
        if diff > 0 and alloc[k] < counts[k]:
            alloc[k] += 1; diff -= 1
        elif diff < 0 and alloc[k] > 1:
            alloc[k] -= 1; diff += 1
        i += 1
        if i > 10000:
            raise RuntimeError("배분 실패")
    return alloc


def round_robin(rows, want, rng):
    """(출처×하위라벨) 그룹을 정렬해 돌아가며 want 개 뽑는다. 그룹 안은 시드 셔플."""
    groups = defaultdict(list)
    for r in rows:
        groups[sub_key(r)].append(r)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    out, idx = [], 0
    while len(out) < want:
        k = keys[idx % len(keys)]
        if groups[k]:
            out.append(groups[k].pop(0))
        idx += 1
        if idx > want * 50:
            raise RuntimeError(f"풀 부족: {want} 필요, {len(out)} 확보")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rng = random.Random(SEED)
    prov = pd.read_excel(R.PROV_PATH)
    act = prov[prov["removed_20260715"].astype(str) == "active"].copy()
    act["_rk"] = act.apply(rec_key, axis=1)
    orig = act[act["source_type"] == "original"]
    m52 = orig[orig["mark_version"] == "mark5.2"]
    keys_of = lambda df: set(k for k in df["_rk"] if k is not None)
    M = {s: keys_of(m52[m52["assigned_split"] == s]) for s in ("val", "test")}
    M["learn"] = keys_of(m52[m52["assigned_split"].isin(["train", "unlabeled"])])
    M["eval_all"] = M["val"] | M["test"] | keys_of(orig[orig["mark_version"] == "mark5.2_eval"])
    log = []
    def say(msg):
        print(msg); log.append(msg)

    # ── ① 교환 계획 ─────────────────────────────────────────────────────
    say(f"[①] 교환 계획 (train 의 Mark5.2 val/test 녹음 ↔ val/test 의 Mark5.2 train·unlabeled 녹음, 카테고리 우선)")
    swaps = []                                     # dict(ver, split, name_in(train), name_out(split), row_in, row_out)
    exp_rows = {}                                  # ver -> {split: [원본 타겟 행 dict] (교환 반영 뒤)}
    for ver in R.EXPERT_VERSIONS:
        cls = R.EXPERT_TARGET_CLASS[ver]
        e = orig[(orig["mark_version"] == ver) & (orig["target_class"] == cls)]
        rows = {s: [r for r in e[e["assigned_split"] == s].to_dict("records")] for s in ("train", "val", "test")}
        for s in ("val", "test"):
            incoming = [r for r in rows["train"] if r["_rk"] in M[s]]
            cands = [r for r in rows[s] if r["_rk"] in M["learn"]]
            cands_by = defaultdict(list)
            for r in cands:
                cands_by[sub_key(r)].append(r)
            for k in cands_by:
                rng.shuffle(cands_by[k])
            used = set(); pairs = []
            for r_in in incoming:
                k = sub_key(r_in); pick = None
                if cands_by.get(k):
                    pick = cands_by[k].pop(0)
                else:
                    for kk in sorted(cands_by):
                        if cands_by[kk]:
                            pick = cands_by[kk].pop(0); break
                if pick is None:
                    raise SystemExit(f"[ERROR] {ver} {s}: 내보낼 후보 부족 (들어옴 {len(incoming)})")
                pairs.append((r_in, pick))
            n_same = sum(1 for r_in, r_out in pairs if sub_key(r_in) == sub_key(r_out))
            say(f"  {ver} {cls:14s} {s}: 들어옴 {len(incoming):2d} · 후보 {len(cands):3d} · 같은 카테고리 짝 {n_same}/{len(pairs)}")
            for r_in, r_out in pairs:
                swaps.append(dict(ver=ver, split=s, name_in=r_in["local_filename"], name_out=r_out["local_filename"], row_in=r_in, row_out=r_out))
            # 교환 반영: train 목록에서 r_in 을 r_out 메타(파일명은 train 쪽 이름)로, split 목록은 반대로
            in_names = {r_in["local_filename"] for r_in, _ in pairs}; out_names = {r_out["local_filename"] for _, r_out in pairs}
            new_train = [r for r in rows["train"] if r["local_filename"] not in in_names]
            new_split = [r for r in rows[s] if r["local_filename"] not in out_names]
            for r_in, r_out in pairs:
                a_ = dict(r_out); a_["local_filename"] = r_in["local_filename"]; a_["assigned_split"] = "train"; new_train.append(a_)
                b_ = dict(r_in); b_["local_filename"] = r_out["local_filename"]; b_["assigned_split"] = s; new_split.append(b_)
            rows["train"], rows[s] = new_train, new_split
        exp_rows[ver] = rows
        left = sum(1 for r in rows["train"] if r["_rk"] in M["eval_all"])
        say(f"  {ver} 교환 뒤 train ∩ Mark5.2 평가쪽 = {left} (기대 0) · train 타겟 실제 {len(rows['train'])}")
    say(f"[①] 교환 합계 {len(swaps)} (기대 276)")

    # ── ② 배경 others 부분집합 ──────────────────────────────────────────
    say(f"[②] 배경 others 부분집합 (출처×하위라벨 비례, 각 ≥1, 시드 {SEED})")
    bg = orig[(orig["mark_version"] == "mark4.1") & (orig["target_class"] == "others")]
    keep = {}                                      # split -> set(파일명)
    bg_rows = {}
    for s in ("train", "val", "test"):
        rows = bg[bg["assigned_split"] == s].to_dict("records"); bg_rows[s] = rows
        groups = defaultdict(list)
        for r in rows:
            groups[sub_key(r)].append(r)
        alloc = proportional({k: len(v) for k, v in groups.items()}, BG_KEEP[s])
        kept = []
        for k in sorted(groups):
            g = sorted(groups[k], key=lambda r: r["local_filename"]); rng.shuffle(g); kept += g[:alloc[k]]
        keep[s] = {r["local_filename"] for r in kept}
        say(f"  {s}: {len(rows)} → {len(kept)} 유지 (하위라벨 {len(groups)}종 전부 ≥1: {all(alloc[k] >= 1 for k in groups)}), 자리 비움 {len(rows) - len(kept)} (기대 {TGT_SLOTS[s]})")
    vt_keys = {s: {sub_key(r) for r in bg_rows[s] if r["local_filename"] in keep[s]} for s in ("val", "test")}
    say(f"  val·test 유지분 하위라벨 집합 동일: {vt_keys['val'] == vt_keys['test']}")

    # ── ③ 다른 타겟 풀 ──────────────────────────────────────────────────
    say(f"[③] 클래스별 타겟 풀 (train {POOL['train']} · val/test {POOL['val']}, 출처×하위라벨 라운드로빈, val·test 같은 순서)")
    pool = {}                                      # cls -> split -> [row]
    for cls in CLASSES:
        ver = VER_OF[cls]; pool[cls] = {}
        for s in ("train", "val", "test"):
            rows = exp_rows[ver][s]
            assert all(r["_rk"] not in M["eval_all"] for r in rows) if s == "train" else True
            pool[cls][s] = round_robin(rows, POOL[s], random.Random(SEED))
        kv = Counter(sub_key(r) for r in pool[cls]["val"]); kt = Counter(sub_key(r) for r in pool[cls]["test"])
        say(f"  {cls:14s} train {len(pool[cls]['train'])} (하위라벨 {len(set(sub_key(r) for r in pool[cls]['train']))}종) · val {len(pool[cls]['val'])} · test {len(pool[cls]['test'])} · val/test 하위라벨 다중집합 동일 {kv == kt}")
    plan = []                                      # dict(ver, split, slot, cls, src_ver, src_split, src_name, row)
    for ver in R.EXPERT_VERSIONS:
        mine = R.EXPERT_TARGET_CLASS[ver]; others7 = [c for c in CLASSES if c != mine]
        for s in ("train", "val", "test"):
            slots = sorted(r["local_filename"] for r in bg_rows[s] if r["local_filename"] not in keep[s])
            assert len(slots) == TGT_SLOTS[s], (ver, s, len(slots))
            per = [36, 36, 36, 36, 36, 35, 35] if s == "train" else [15, 15, 15, 15, 15, 16, 16]
            i = 0
            for cls, n in zip(others7, per):
                for r in pool[cls][s][:n]:
                    plan.append(dict(ver=ver, split=s, slot=slots[i], cls=cls, src_ver=VER_OF[cls], src_split=s, src_name=r["local_filename"], row=r)); i += 1
            assert i == len(slots)
        say(f"  {ver} others 자리 채움: train {sum(1 for p in plan if p['ver']==ver and p['split']=='train')} · val {sum(1 for p in plan if p['ver']==ver and p['split']=='val')} · test {sum(1 for p in plan if p['ver']==ver and p['split']=='test')} (자기 클래스 {mine} 0개: {all(p['cls'] != mine for p in plan if p['ver']==ver)})")
    say(f"[③] 합계 {len(plan)} (기대 8×464 = 3712)")

    # ── 기대값 대조 ──────────────────────────────────────────────────────
    ok = (len(swaps) == 276 and len(plan) == 3712 and all(len(keep[s]) == BG_KEEP[s] for s in keep) and vt_keys["val"] == vt_keys["test"])
    say(f"[기대값] 교환 276: {len(swaps) == 276} · 타겟 자리 3712: {len(plan) == 3712} · 배경 250/108/108: {[len(keep[s]) for s in ('train','val','test')]} · 종합 {'OK' if ok else 'FAIL'}")
    if a.dry_run:
        open(os.path.join(CACHE, f"cross_reject_{TODAY}_dryrun.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
        print(f"[DRY RUN] ({time.time()-t0:.0f}s)")
        return
    if not ok:
        raise SystemExit("[STOP] 기대값 불일치 — 실행하지 않음")

    # ── 실행 ① 파일 교환 ───────────────────────────────────────────────
    n_files = 0
    for sw in swaps:
        p_in = data_path(sw["ver"], "train", sw["name_in"]); p_out = data_path(sw["ver"], sw["split"], sw["name_out"])
        b_in = open(p_in, "rb").read(); b_out = open(p_out, "rb").read()
        with open(p_in, "wb") as f: f.write(b_out)
        with open(p_out, "wb") as f: f.write(b_in)
        n_files += 2
    say(f"[실행 ①] 파일 교환 {len(swaps)}쌍 ({n_files}파일) ({time.time()-t0:.0f}s)")
    # ── 실행 ②③ others 자리 채움 (읽기는 원본 파일에서; 자기 저장소 안 교환은 위에서 끝났으므로 src 는 src_ver 폴더) ──
    for p in plan:
        src = data_path(p["src_ver"], p["src_split"], p["src_name"]); dst = data_path(p["ver"], p["split"], p["slot"])
        shutil.copyfile(src, dst); n_files += 1
    say(f"[실행 ②③] others 자리 채움 {len(plan)}파일 ({time.time()-t0:.0f}s)")
    # 공통 배경 스테이징
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(STAGE_OUT, s), exist_ok=True)
        for nm in sorted(keep[s]):
            shutil.copyfile(data_path("mark4.1", s, nm), os.path.join(STAGE_OUT, s, nm))
    say(f"[실행 ⑤] 공통 배경 스테이징 {STAGE_OUT} ({sum(len(v) for v in keep.values())}파일)")

    # ── provenance ───────────────────────────────────────────────────────
    backup = R.PROV_PATH + f".bak_before_cross_apply_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)
    st = prov["removed_20260715"].astype(str)
    new_rows = []; n_tag = 0
    def find(ver, split, name, cls):
        m = (prov["mark_version"] == ver) & (prov["assigned_split"] == split) & (prov["local_filename"] == name) & (prov["target_class"] == cls) & (st == "active") & (prov["source_type"] == "original")
        assert m.sum() == 1, (ver, split, name, cls, int(m.sum())); return prov.index[m][0]
    for sw in swaps:
        cls = R.EXPERT_TARGET_CLASS[sw["ver"]]
        i_in = find(sw["ver"], "train", sw["name_in"], cls); i_out = find(sw["ver"], sw["split"], sw["name_out"], cls)
        r_in = prov.loc[i_in].to_dict(); r_out = prov.loc[i_out].to_dict()
        prov.loc[i_in, "removed_20260715"] = TAG_SWAP; prov.loc[i_out, "removed_20260715"] = TAG_SWAP; n_tag += 2
        a_ = dict(r_out); a_.update(local_filename=sw["name_in"], assigned_split="train", removed_20260715="active", mark52_note=f"swap_cross_{TODAY}(was={sw['split']}:{sw['name_out']})")
        b_ = dict(r_in); b_.update(local_filename=sw["name_out"], assigned_split=sw["split"], removed_20260715="active", mark52_note=f"swap_cross_{TODAY}(was=train:{sw['name_in']})")
        for d in (a_, b_):
            d.pop("_rk", None); p = data_path(sw["ver"], d["assigned_split"], d["local_filename"]); d["sha256"] = sha(p); d["size_bytes"] = os.path.getsize(p)
        new_rows += [a_, b_]
    for p in plan:
        i_old = find(p["ver"], p["split"], p["slot"], "others")
        prov.loc[i_old, "removed_20260715"] = TAG_BG; n_tag += 1
        d = dict(p["row"]); d.pop("_rk", None)
        d.update(local_filename=p["slot"], assigned_split=p["split"], mark_version=p["ver"], target_class="others", removed_20260715="active",
                 mark52_note=f"cross_target_{TODAY}(from={p['cls']}:{p['src_name']})")
        dst = data_path(p["ver"], p["split"], p["slot"]); d["sha256"] = sha(dst); d["size_bytes"] = os.path.getsize(dst)
        new_rows.append(d)
    st2 = prov["removed_20260715"].astype(str)
    prov = pd.concat([prov, pd.DataFrame(new_rows)[prov.columns]], ignore_index=True)
    prov.to_excel(R.PROV_PATH, index=False)
    say(f"[provenance] 태그 {n_tag}행 (swap {len(swaps)*2} + bg {len(plan)}), 새 행 {len(new_rows)}, 백업 {os.path.basename(backup)} ({time.time()-t0:.0f}s)")
    json.dump(dict(swaps=[{k: v for k, v in sw.items() if k not in ('row_in', 'row_out')} for sw in swaps],
                   keep={s: sorted(v) for s, v in keep.items()},
                   plan=[{k: v for k, v in p.items() if k != 'row'} for p in plan]),
              open(os.path.join(CACHE, f"cross_reject_{TODAY}.json"), "w"), ensure_ascii=False, indent=0)
    open(os.path.join(CACHE, f"cross_reject_{TODAY}.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    print(f"[완료] ({(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()
