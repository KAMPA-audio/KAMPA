#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
repair_after_audit.py — rebuild 뒤 검증(validate·audit·silence_audit)에서 나온 문제를 한 번에 고친다 (2026-09-15).

  (a) 긴 정적 표시 파일 교체 — silence_audit CSV 에서 flag 가 붙은 **새로 만든** 파일(Mark5.2 talking·others,
      unlabeled 의 새 talking·others, 전문가 공통 others, Mark4.5 talking)을 같은 (역할, split) 풀에서
      아직 안 쓴 원본으로 바꾼다. 후보는 게이트(check_waveform) + 정적 기준(active >= 0.5s, 빈 구간 < 2.0s)
      을 모두 통과해야 한다. 파일명은 그대로 두고 내용·provenance 행만 바뀐다(전문가 others 는 8곳 동일).
      ⚠ mark5.0 에서 복사한 7종과 Mark4.8 dog_bark 는 손대지 않는다(수진님 결정 사항).
  (b) 경계 B 평가 세트에 SINS dishwashing 추가 — 1차 생성에서 split 이 eval 이 아니라 빠졌던 143개.
  (c) DC 오프셋 WARN 파일 평균 제거 — Mark5.2 others_val_036, 전문가 others_train_346(8곳 + staging).
  (d) 평가 세트 provenance 행의 mark_version 을 mark5.2 -> mark5.2_eval 로 바꿔 audit ⑦(디스크 vs provenance)
      이 학습 데이터만 세게 한다.
  provenance 는 한 번만 읽고 한 번만 쓴다(백업 먼저).

사용: python preprocessing/repair_after_audit.py --dry_run     # 교체 목록만
      python preprocessing/repair_after_audit.py               # 실행
"""
import os, sys, csv, json, glob, time, shutil, hashlib, argparse
from collections import defaultdict, Counter
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                      # noqa: E402
from validate_dataset import frame_rms                # noqa: E402

CACHE = R.CACHE_DIR
STAGE = os.path.join(CACHE, "expert_common")
TODAY = R.TODAY
MIN_ACTIVE, MAX_GAP = 0.5, 2.0
N_COPIED_UNLABELED = 1556      # unlabeled_0001~1556 은 mark5.0 7종 복사분


def sparse(seg, sr=R.TARGET_SR):
    """(active_sec, max_inactive_run). silence_audit.measure 와 같은 정의."""
    fr = frame_rms(seg, sr)
    thr = max(1e-4, 0.05 * float(fr.max())) if len(fr) else 1e-4
    act = fr > thr
    runs, cur = [], 0
    for a in act:
        if a:
            runs.append(cur); cur = 0
        else:
            cur += 1
    runs.append(cur)
    return float(act.sum() * 0.010), max(runs) * 0.010


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def expert_paths(split, name):
    return [os.path.join(R.EXPERT_ROOT, "Mark" + v[4:], "data", split, name) for v in R.EXPERT_VERSIONS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--audit_csv", type=str, default=sorted(glob.glob(os.path.join(CACHE, "silence_audit_*.csv")))[-1])
    a = ap.parse_args()
    t0 = time.time()

    # ── 풀·계획·캐시 ──
    cache = json.load(open(os.path.join(CACHE, "gate_cache.json")))
    plan = json.load(open(os.path.join(CACHE, "plan.json")))
    pools_raw = R.build_chime() + R.build_sins() + R.build_esc() + R.build_aihub()
    pools = R.gate_pool(pools_raw, cache)
    json.dump(cache, open(os.path.join(CACHE, "gate_cache.json"), "w"))
    by_src = {it["src"]: it for it in pools}
    P = defaultdict(list)
    for it in pools:
        P[(it["role"], it["split"])].append(it)
    used = set(s for lst in plan.values() for s in lst)
    print(f"[풀] 게이트 통과 {len(pools)} / 이미 쓴 원본 {len(used)}")

    # 파일명 -> 원본 (새로 만든 것만)
    src_of = {}
    for key, lst in plan.items():
        ds, sp, cls = key.split("|")
        if ds == "eval" or (ds == "m52" and sp == "unlabeled"):
            continue
        for i, s in enumerate(lst, 1):
            src_of[(ds, sp, f"{cls}_{sp}_{i:03d}.wav")] = (s, cls)
    k = N_COPIED_UNLABELED
    for cls in ("talking", "others"):
        for s in plan[f"m52|unlabeled|{cls}"]:
            k += 1
            src_of[("m52", "unlabeled", f"unlabeled_{k:04d}.wav")] = (s, cls)

    # ── (a) 교체 대상 ──
    rows_audit = list(csv.DictReader(open(a.audit_csv, encoding="utf-8")))
    targets = []
    for r in rows_audit:
        if not r["flag"]:
            continue
        key = (r["dataset"], r["split"], r["file"])
        if key not in src_of:
            continue                                   # 복사분·평가세트·4.8 dog 는 제외
        targets.append((key, src_of[key]))
    print(f"[교체 대상] {len(targets)}개 (정적 표시된 새 파일)")

    replacements, short = [], []
    for (ds, sp, name), (old_src, cls) in targets:
        old = by_src.get(old_src)
        if old is None:
            short.append((name, "원본을 풀에서 못 찾음")); continue
        role, split = old["role"], old["split"]
        cand = R.draw(P[(role, split)], 200, used)      # 같은 group 균등 + 세션 라운드로빈 순서로 후보 200
        # 같은 group(출처·카테고리)을 먼저 시도해 구성이 흔들리지 않게 한다
        cand.sort(key=lambda it: 0 if it["group"] == old["group"] else 1)
        picked = None
        for it in cand:
            seg, st = R.load_segment(it)
            if seg is None:
                continue
            act, gap = sparse(seg)
            if act >= MIN_ACTIVE and gap < MAX_GAP:
                picked = (it, seg, st); break
        if picked is None and role in ("house", "far"):
            # others 는 집 녹음 <-> 바깥 소리 서로 대체 가능(비율은 어림값). val 집 녹음 풀이 바닥난 경우.
            alt = "far" if role == "house" else "house"
            for it in R.draw(P[(alt, split)], 200, used):
                seg, st = R.load_segment(it)
                if seg is None:
                    continue
                act, gap = sparse(seg)
                if act >= MIN_ACTIVE and gap < MAX_GAP:
                    picked = (it, seg, st); break
        if picked is None:
            short.append((name, f"{role}/{split} 풀에 조건 맞는 후보 없음")); continue
        it, seg, st = picked
        used.add(it["src"])
        replacements.append(dict(ds=ds, split=sp, name=name, cls=cls, old=old, new=it, seg=seg, start=st))
    print(f"[교체 후보 확보] {len(replacements)}개 / 실패 {len(short)}개  ({time.time()-t0:.0f}s)")
    for name, why in short:
        print(f"   - {name}: {why}")
    same_group = sum(1 for r in replacements if r["old"]["group"] == r["new"]["group"])
    print(f"   같은 카테고리로 교체 {same_group} / 다른 카테고리 {len(replacements)-same_group}")
    if a.dry_run:
        for r in replacements[:12]:
            print(f"   {r['ds']:6s} {r['split']:9s} {r['name']:24s} {r['old']['group']:22s} -> {r['new']['group']:22s} {r['new']['src']}")
        if len(replacements) > 12:
            print(f"   ... 외 {len(replacements)-12}개")
    # (b) dishwashing
    dish = [it for it in pools if it["role"] == "boundary_B" and it["source"] == "sins"]
    print(f"[경계 B 추가] SINS dishwashing {len(dish)}개")
    if a.dry_run:
        print("\n[DRY RUN] 파일·provenance 는 바꾸지 않았습니다.")
        return

    import pandas as pd
    prov = pd.read_excel(R.PROV_PATH)
    backup = R.PROV_PATH + f".bak_before_repair_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(R.PROV_PATH, backup)
    print(f"[provenance] {len(prov)}행 읽음, 백업 {os.path.basename(backup)}  ({time.time()-t0:.0f}s)")
    if "mark52_note" not in prov.columns:
        prov["mark52_note"] = ""
    active = prov["removed_20260715"] == "active"

    def update_rows(mask, path, it, start, note):
        idx = prov.index[mask]
        if len(idx) == 0:
            print(f"   [WARN] provenance 행 없음: {os.path.basename(path)}")
        prov.loc[idx, "original_labels"] = it["category"]
        prov.loc[idx, "source"] = it["source"]
        prov.loc[idx, "aihub_src_file"] = it["src"]
        prov.loc[idx, "aihub_category"] = it["category"]
        prov.loc[idx, "aihub_volume"] = it["volume"]
        prov.loc[idx, "seg_start_sec"] = round(float(start), 3)
        prov.loc[idx, "seg_end_sec"] = round(float(start) + R.SEG_SEC, 3)
        prov.loc[idx, "sha256"] = sha(path)
        prov.loc[idx, "size_bytes"] = os.path.getsize(path)
        prov.loc[idx, "mark52_note"] = note
        return len(idx)

    # ── (a) 실행 ──
    n_rows = 0
    for r in replacements:
        note = f"replaced_silence_{TODAY}(prev={r['old']['src']})"
        if r["ds"] == "m52":
            d = {"train": "data_source_labeled", "val": "data_source_val", "test": "data_source_test",
                 "unlabeled": "data_source_unlabeled"}[r["split"]]
            path = (os.path.join(R.PROJECT_ROOT, d, r["name"]) if r["split"] == "unlabeled"
                    else os.path.join(R.PROJECT_ROOT, d, r["cls"], r["name"]))
            sf.write(path, r["seg"], R.TARGET_SR, subtype="FLOAT")
            m = active & (prov["mark_version"] == "mark5.2") & (prov["local_filename"] == r["name"]) & \
                (prov["assigned_split"] == r["split"])
            n_rows += update_rows(m, path, r["new"], r["start"], note)
        elif r["cls"] == "others":                                  # 전문가 공통 others
            paths = [os.path.join(STAGE, r["split"], r["name"])] + expert_paths(r["split"], r["name"])
            for p in paths:
                sf.write(p, r["seg"], R.TARGET_SR, subtype="FLOAT")
            m = active & prov["mark_version"].isin(R.EXPERT_VERSIONS) & (prov["local_filename"] == r["name"]) & \
                (prov["assigned_split"] == r["split"]) & (prov["target_class"] == "others")
            n_rows += update_rows(m, paths[-1], r["new"], r["start"], note)
        else:                                                       # Mark4.5 talking
            paths = [os.path.join(STAGE, "talking", r["split"], r["name"]),
                     os.path.join(R.EXPERT_ROOT, "Mark4.5", "data", r["split"], r["name"])]
            for p in paths:
                sf.write(p, r["seg"], R.TARGET_SR, subtype="FLOAT")
            m = active & (prov["mark_version"] == "mark4.5") & (prov["local_filename"] == r["name"]) & \
                (prov["assigned_split"] == r["split"])
            n_rows += update_rows(m, paths[-1], r["new"], r["start"], note)
    print(f"[교체] 파일 {len(replacements)}개, provenance 행 {n_rows}개 갱신  ({time.time()-t0:.0f}s)")

    # ── (b) dishwashing → boundary_B ──
    new_rows = []
    if dish:
        R.write_items(dish, os.path.join(R.PROJECT_ROOT, "data_eval", "boundary_B", "dishwashing"),
                      lambda i: f"boundary_B_dishwashing_{i:03d}.wav", "eval:boundary_B", "eval_boundary_B",
                      cache, new_rows, version="mark5.2_eval")
        print(f"[경계 B] dishwashing {len(new_rows)}개 추가")

    # ── (c) DC 제거 ──
    def remove_dc(path):
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x[:, 0]; dc = float(x.mean())
        sf.write(path, (x - np.float32(dc)).astype(np.float32), sr, subtype="FLOAT")
        return dc
    dc_targets = [("mark5.2", os.path.join(R.PROJECT_ROOT, "data_source_val", "others", "others_val_036.wav"), "val")]
    for ver, path, sp in dc_targets:
        dc = remove_dc(path)
        m = active & (prov["mark_version"] == ver) & (prov["local_filename"] == os.path.basename(path)) & (prov["assigned_split"] == sp)
        prov.loc[prov.index[m], "sha256"] = sha(path); prov.loc[prov.index[m], "size_bytes"] = os.path.getsize(path)
        prov.loc[prov.index[m], "mark52_note"] = f"dc_removed_{TODAY}({dc:+.4f})"
        print(f"[DC] {os.path.basename(path)} {dc:+.4f} 제거")
    name = "others_train_346.wav"
    paths = [os.path.join(STAGE, "train", name)] + expert_paths("train", name)
    dc = None
    for p in paths:
        dc = remove_dc(p)
    m = active & prov["mark_version"].isin(R.EXPERT_VERSIONS) & (prov["local_filename"] == name) & \
        (prov["assigned_split"] == "train") & (prov["target_class"] == "others")
    prov.loc[prov.index[m], "sha256"] = sha(paths[-1]); prov.loc[prov.index[m], "size_bytes"] = os.path.getsize(paths[-1])
    prov.loc[prov.index[m], "mark52_note"] = f"dc_removed_{TODAY}({dc:+.4f})"
    print(f"[DC] 전문가 {name} {dc:+.4f} 제거 (9곳), 행 {int(m.sum())}")

    # ── (e) Mark4.8 dog_bark 재정렬 — 클리핑 >= 0.5% 로 validate ERROR 난 4개. 클리핑 기준을 만족하는
    #        창(0.1초 간격) 중 RMS 최대를 고른다. 원본은 _source_long_20260915/ 에 그대로 있다.
    clip_bad = ["dog_bark_train_608.wav", "dog_bark_train_635.wav", "dog_bark_train_676.wav", "dog_bark_train_682.wav"]
    m48 = os.path.join(R.EXPERT_ROOT, "Mark4.8", "data")
    for f in clip_bad:
        src = os.path.join(m48, f"_source_long_{TODAY}", "train", f)
        x, sr = sf.read(src, dtype="float32", always_2d=True); x = np.ascontiguousarray(x[:, 0])
        n = int(R.SEG_SEC * sr); hop = int(0.1 * sr)
        best, best_s, best_clip = -1.0, None, 1.0
        for st in range(0, len(x) - n + 1, hop):
            w = x[st:st + n]; clip = float(np.mean(np.abs(w) >= 0.999))
            if clip >= 0.005:
                continue
            if R.check_waveform(w, sr, min_duration=R.SEG_SEC) is not None:
                continue
            rr = float(np.sqrt(np.mean(w ** 2)))
            if rr > best:
                best, best_s, best_clip = rr, st, clip
        if best_s is None:
            print(f"   [WARN] {f}: 클리핑 기준을 만족하는 창이 없음 — 그대로 둠"); continue
        dst = os.path.join(m48, "train", f)
        sf.write(dst, x[best_s:best_s + n].astype(np.float32), sr, subtype="FLOAT")
        m = active & (prov["mark_version"] == "mark4.8") & (prov["local_filename"] == f) & (prov["assigned_split"] == "train")
        idx = prov.index[m]
        prov.loc[idx, "seg_start_sec"] = round(best_s / sr, 3); prov.loc[idx, "seg_end_sec"] = round(best_s / sr + R.SEG_SEC, 3)
        prov.loc[idx, "sha256"] = sha(dst); prov.loc[idx, "size_bytes"] = os.path.getsize(dst)
        prov.loc[idx, "mark52_note"] = f"realign3s_{TODAY}(clip<0.5%)"
        print(f"[4.8 재정렬] {f}: 창 {best_s/sr:.1f}s 클리핑 {best_clip*100:.2f}%  행 {len(idx)}")

    # ── (d) 평가 세트 버전 표시 ──
    m = (prov["mark_version"] == "mark5.2") & prov["assigned_split"].astype(str).str.startswith("eval_")
    prov.loc[prov.index[m], "mark_version"] = "mark5.2_eval"
    print(f"[평가 세트] mark_version -> mark5.2_eval {int(m.sum())}행")

    merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else prov
    merged.to_excel(R.PROV_PATH, index=False)
    print(f"[provenance] 저장 {len(merged)}행  ({(time.time()-t0)/60:.1f}분)")
    json.dump([dict(ds=r["ds"], split=r["split"], name=r["name"], old=r["old"]["src"], new=r["new"]["src"],
                    old_group=r["old"]["group"], new_group=r["new"]["group"]) for r in replacements],
              open(os.path.join(CACHE, f"repair_{TODAY}.json"), "w"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
