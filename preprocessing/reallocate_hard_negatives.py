#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reallocate_hard_negatives.py — 전문가 others 의 "다른 타겟" 자리를 헷갈리는 클래스 쪽에 더 배분한다 (2026-10-23, 교사 개선판 B+).

왜: 09-21 cross_reject_experts.py 는 다른 타겟 7종을 균등하게 넣었다(train 36×5+35×2 · val/test 15×5+16×2). 그런데 3차 교사는
    특정 클래스에 몰려서 손을 든다(Mark5.3 기준 dragging 클립에 machine_noise 교사 34.1%, machine_noise 클립에 dragging 교사 30.6% 등).
    헷갈리는 소리를 음성 예제로 더 보여 주면 교사가 그 소리를 거절하는 법을 더 배운다(설계 문서 '개선판(B+)에 더하는 것').
무엇을:
    ① 배분: 교사 k 의 자리 수(train 250 · val 107 · test 107)를 7종 j 에 floor(최소 개수) + 나머지를 (손들기 비율 raise_rate[k][j] + smooth)
       (measure_teacher_confusion.py, 교사 쪽 val 에서 잼 — Mark5.3 데이터를 보지 않음)에 비례해 큰 나머지 순으로 나눈다.
       val 과 test 는 같은 개수(audit ⑧ 대칭). smooth(기본 0.03 ≈ val 215클립 중 6개)는 손들기가 거의 없는 교사(talking·dog_bark,
       모두 ≤ 0.05)에서 클립 1~4개 차이가 배분을 흔들지 않게 한다 — 그런 교사는 거의 균등, 뚜렷이 헷갈리는 교사는 그쪽으로 쏠린다.
    ② 풀: 09-21 과 같다 — 클래스 j 의 풀 = 전문가 j 의 (교환 뒤) 타겟 원본, (출처×하위라벨) 라운드로빈, 시드 42, val·test 같은 순서.
       같은 순서의 앞부분을 쓰므로 09-21 에 뽑혔던 클립이 앞자리에 그대로 남는다(바뀌는 것은 개수).
    ③ 자리: 지금 교사 k others 중 mark52_note 가 cross_target_20260921 또는 hardneg_ 로 시작하는 원본 행(train 250 · val 107 · test 107)의
       파일명을 그대로 쓴다(10-23 resplit 뒤 다시 돌릴 때는 앞선 실행이 남긴 hardneg_ 자리를 쓴다 — --run_tag 로 태그를 구분).
    ④ provenance: 옛 자리 행 removed_20260715="replaced_hardneg_20261023" + 새 행(mark52_note=hardneg_20261023(from=클래스:파일)). 실행 전 백업.
뒤 단계(09-21 과 같은 순서): bash preprocessing/run_augment_all.sh (증강본 재생성, others 증강 절반이 이 자리에서 나온다)
    → bash preprocessing/run_expert_index_all.sh → bash preprocessing/verify_after_rebuild.sh(전문가 validate + audit ①~⑨)
사용: python preprocessing/reallocate_hard_negatives.py --confusion rebuild_cache/teacher_confusion_<날짜>.json --dry_run | (실행)
"""
import os, sys, json, time, shutil, argparse, random
from collections import Counter, defaultdict
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402
from cross_reject_experts import rec_key, sub_key, round_robin, data_path, sha, CLASSES, VER_OF, SEED   # noqa: E402

TODAY = "20261023"
CROSS_NOTE = ("cross_target_20260921", "hardneg_")
SLOTS = {"train": 250, "val": 107, "test": 107}
OLD_PER = {"train": [36, 36, 36, 36, 36, 35, 35], "val": [15, 15, 15, 15, 15, 16, 16], "test": [15, 15, 15, 15, 15, 16, 16]}
CACHE = R.CACHE_DIR


def allocate(rates, total, floor):
    """rates: {클래스: 손들기 비율} 7종 → 합이 total 이고 각 ≥ floor 인 개수. 나머지는 비율에 비례(큰 나머지 순, 같으면 클래스 순서)."""
    keys = list(rates)
    rest = total - floor * len(keys)
    if rest < 0:
        raise ValueError(f"floor {floor} × {len(keys)} > {total}")
    s = sum(rates.values())
    w = {k: (rates[k] / s if s > 0 else 1.0 / len(keys)) for k in keys}
    raw = {k: rest * w[k] for k in keys}
    out = {k: floor + int(raw[k]) for k in keys}
    left = total - sum(out.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - int(raw[k])), keys.index(k)))[:left]:
        out[k] += 1
    assert sum(out.values()) == total and all(v >= floor for v in out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True, help="measure_teacher_confusion.py 산출 json")
    ap.add_argument("--floor_train", type=int, default=15, help="train 에서 클래스마다 최소 개수(7종 모두 계속 보여 준다)")
    ap.add_argument("--floor_vt", type=int, default=6, help="val·test 에서 클래스마다 최소 개수")
    ap.add_argument("--smooth", type=float, default=0.03, help="손들기 비율에 더하는 값(측정 잡음 완화)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--run_tag", default=TODAY, help="provenance 태그·산출 파일 이름에 붙는 값(같은 날 두 번째 실행은 20261023b 등)")
    a = ap.parse_args()
    t0 = time.time()
    TAG_OLD, NOTE = f"replaced_hardneg_{a.run_tag}", f"hardneg_{a.run_tag}"
    conf_path = a.confusion if os.path.isabs(a.confusion) else os.path.join(BASE_DIR, a.confusion)
    conf = json.load(open(conf_path, encoding="utf-8"))
    if conf.get("split") != "val":
        raise SystemExit("[STOP] 배분 비율은 교사 쪽 val 에서 잰 값만 씁니다(test 를 보지 않음).")
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
    eval_all = keys_of(m52[m52["assigned_split"].isin(["val", "test"])]) | keys_of(orig[orig["mark_version"] == "mark5.2_eval"])

    # ── ① 배분 ───────────────────────────────────────────────────────────
    say(f"[①] 배분 (floor train {a.floor_train} · val/test {a.floor_vt}, 나머지는 (교사 쪽 val 손들기 비율 + {a.smooth:g})에 비례 — {os.path.basename(conf_path)})")
    alloc = {}
    for ver in R.EXPERT_VERSIONS:
        k = R.EXPERT_TARGET_CLASS[ver]; others7 = [c for c in CLASSES if c != k]
        rates = {j: float(conf["raise_rate"][k][j]) for j in others7}
        w = {j: rates[j] + a.smooth for j in others7}
        alloc[ver] = {"train": allocate(w, SLOTS["train"], a.floor_train)}
        alloc[ver]["val"] = allocate(w, SLOTS["val"], a.floor_vt)
        alloc[ver]["test"] = dict(alloc[ver]["val"])
        say(f"  {ver} {k:14s} 손들기 " + " ".join(f"{j[:6]}={rates[j]:.3f}" for j in others7))
        say(f"  {'':22s} train " + " ".join(f"{j[:6]}={alloc[ver]['train'][j]:3d}({o:+d})" for j, o in zip(others7, [alloc[ver]['train'][j] - n for j, n in zip(others7, OLD_PER['train'])]))
            + f"  · val/test " + " ".join(f"{j[:6]}={alloc[ver]['val'][j]}" for j in others7))

    # ── ② 풀 ─────────────────────────────────────────────────────────────
    need = {cls: {s: max(alloc[v][s][cls] for v in R.EXPERT_VERSIONS if R.EXPERT_TARGET_CLASS[v] != cls) for s in SLOTS} for cls in CLASSES}
    say("[②] 클래스별 풀 (전문가 j 의 타겟 원본, 라운드로빈 시드 42) — 필요한 최대 개수 대비 크기")
    pool, short = {}, []
    for cls in CLASSES:
        ver = VER_OF[cls]; pool[cls] = {}
        e = orig[(orig["mark_version"] == ver) & (orig["target_class"] == cls)]
        for s in SLOTS:
            rows = e[e["assigned_split"] == s].to_dict("records")
            if s == "train":
                bad = sum(1 for r in rows if r["_rk"] in eval_all)
                if bad:
                    raise SystemExit(f"[STOP] {ver} train 타겟 {bad}개가 Mark5.2 평가쪽 녹음입니다(audit ⑨ R1) — 풀로 쓰지 않음")
            if need[cls][s] > len(rows):
                short.append((cls, s, need[cls][s], len(rows)))
            pool[cls][s] = round_robin(rows, min(need[cls][s], len(rows)), random.Random(SEED))
        say(f"  {cls:14s} train 풀 {len(e[e['assigned_split']=='train'])} · 필요 {need[cls]['train']:3d} | val 풀 {len(e[e['assigned_split']=='val'])} · test 풀 {len(e[e['assigned_split']=='test'])} · 필요 {need[cls]['val']}"
            + f" | 하위라벨 train {len(set(sub_key(r) for r in pool[cls]['train']))}종")
    if short:
        raise SystemExit(f"[STOP] 풀 부족: {short}")
    # 09-21 에 뽑힌 클립이 새 순서의 앞부분과 같은지(같은 라운드로빈·시드면 앞 36/16 개가 같아야 한다)
    old_prefix_same = True
    j21 = json.load(open(os.path.join(CACHE, "cross_reject_20260921.json"), encoding="utf-8"))
    old_src = defaultdict(set)
    for p in j21["plan"]:
        old_src[(p["cls"], p["split"])].add(p["src_name"])
    for cls in CLASSES:
        for s in SLOTS:
            n_old = 36 if s == "train" else 16
            prefix = {r["local_filename"] for r in pool[cls][s][:n_old]}
            if len(pool[cls][s]) >= n_old and prefix != old_src[(cls, s)]:
                old_prefix_same = False
    say(f"  09-21 에 뽑힌 클립 = 새 순서의 앞부분: {old_prefix_same}")

    # ── ③ 자리 ───────────────────────────────────────────────────────────
    plan = []
    for ver in R.EXPERT_VERSIONS:
        k = R.EXPERT_TARGET_CLASS[ver]; others7 = [c for c in CLASSES if c != k]
        o = orig[(orig["mark_version"] == ver) & (orig["target_class"] == "others")]
        for s in SLOTS:
            slots = sorted(o[(o["assigned_split"] == s) & o["mark52_note"].astype(str).str.startswith(CROSS_NOTE)]["local_filename"])
            if len(slots) != SLOTS[s]:
                raise SystemExit(f"[STOP] {ver} {s}: 다른 타겟 자리 {len(slots)}개(기대 {SLOTS[s]})")
            i = 0
            for cls in others7:
                for r in pool[cls][s][:alloc[ver][s][cls]]:
                    plan.append(dict(ver=ver, split=s, slot=slots[i], cls=cls, src_ver=VER_OF[cls], src_split=s, src_name=r["local_filename"], row=r)); i += 1
            assert i == len(slots)
    # audit ⑧ 과 같은 기준: 교사마다 test 자리의 (출처×하위라벨) 종류가 val 자리에도 있어야 한다(test 에만 있는 종류 = FAIL).
    # 개수까지 같지는 않을 수 있다 — 하위라벨 한 종류가 val·test 에 3~5개뿐이면(talking 의 CHiME 'os') 라운드로빈이 먼저 바닥난다.
    vt_same = all({sub_key(p["row"]) for p in plan if p["ver"] == v and p["split"] == "test"}
                  <= {sub_key(p["row"]) for p in plan if p["ver"] == v and p["split"] == "val"} for v in R.EXPERT_VERSIONS)
    vt_diff = sum((Counter((p["ver"], p["cls"], sub_key(p["row"])) for p in plan if p["split"] == "val")
                   - Counter((p["ver"], p["cls"], sub_key(p["row"])) for p in plan if p["split"] == "test")).values())
    n_mine = sum(1 for p in plan if p["cls"] == R.EXPERT_TARGET_CLASS[p["ver"]])
    ok = len(plan) == 8 * sum(SLOTS.values()) and n_mine == 0 and vt_same   # 앞부분 일치는 알리기만(10-22 무음 교체 등으로 행 순서가 바뀌면 달라질 수 있다)
    say(f"[③] 자리 채움 {len(plan)} (기대 {8 * sum(SLOTS.values())}) · 자기 클래스 0개: {n_mine == 0} · test 하위라벨 ⊆ val 하위라벨: {vt_same}"
        f" · val/test 개수 차이 {vt_diff}자리(전체 {8 * SLOTS['val']} 중)")
    say(f"[기대값] {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
    tag = "dryrun" if a.dry_run else "apply"
    json.dump(dict(confusion=os.path.basename(conf_path), floor_train=a.floor_train, floor_vt=a.floor_vt, smooth=a.smooth, alloc=alloc,
                   plan=[{k_: v for k_, v in p.items() if k_ != "row"} for p in plan]),
              open(os.path.join(CACHE, f"hardneg_{a.run_tag}_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    if a.dry_run:
        open(os.path.join(CACHE, f"hardneg_{a.run_tag}_dryrun.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
        print("[DRY RUN] 파일·provenance 는 바꾸지 않았습니다.")
        return
    if not ok:
        raise SystemExit("[STOP] 기대값 불일치 — 실행하지 않음")

    # ── 실행: 파일 ────────────────────────────────────────────────────────
    for p in plan:
        shutil.copyfile(data_path(p["src_ver"], p["src_split"], p["src_name"]), data_path(p["ver"], p["split"], p["slot"]))
    say(f"[실행] 자리 채움 {len(plan)}파일 ({time.time()-t0:.0f}s)")
    # ── 실행: provenance ─────────────────────────────────────────────────
    backup = R.PROV_PATH + f".bak_before_hardneg_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(R.PROV_PATH, backup)
    new_rows = []
    for p in plan:
        m = ((prov["mark_version"] == p["ver"]) & (prov["assigned_split"] == p["split"]) & (prov["local_filename"] == p["slot"])
             & (prov["target_class"] == "others") & (st == "active") & (prov["source_type"] == "original"))
        assert m.sum() == 1, (p["ver"], p["split"], p["slot"], int(m.sum()))
        prov.loc[prov.index[m][0], "removed_20260715"] = TAG_OLD
        d = dict(p["row"]); d.pop("_rk", None)
        dst = data_path(p["ver"], p["split"], p["slot"])
        d.update(local_filename=p["slot"], assigned_split=p["split"], mark_version=p["ver"], target_class="others", removed_20260715="active",
                 mark52_note=f"{NOTE}(from={p['cls']}:{p['src_name']})", sha256=sha(dst), size_bytes=os.path.getsize(dst))
        new_rows.append(d)
    prov = pd.concat([prov, pd.DataFrame(new_rows)[prov.columns]], ignore_index=True)
    prov.to_excel(R.PROV_PATH, index=False)
    say(f"[provenance] 옛 자리 {len(plan)}행 → {TAG_OLD}, 새 행 {len(new_rows)}, 백업 {os.path.basename(backup)} ({time.time()-t0:.0f}s)")
    open(os.path.join(CACHE, f"hardneg_{a.run_tag}.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
    print(f"[완료] ({(time.time()-t0)/60:.1f}분) — 다음: run_augment_all.sh → run_expert_index_all.sh → verify_after_rebuild.sh")


if __name__ == "__main__":
    main()
