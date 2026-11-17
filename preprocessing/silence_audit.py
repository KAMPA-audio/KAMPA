#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
silence_audit.py — 3초 클립의 "긴 정적" 전수 검사 (2026-09-15, rebuild 뒤 보조 검증).

왜 따로 있는가
  validate_dataset 은 silence_ratio·active_sec·max_silence_sec 를 재지만 판정에는 쓰지 않는다(2026-07-31,
  상대 임계 기반이라 짧고 강한 소리에서 오탐). LONG_SILENCE 도 연속 무음 3.0초 이상에서만 WARN 이라
  3초 클립에서는 사실상 안 걸린다. 수집 게이트(check_waveform)도 길이·rms·peak·클리핑만 본다.
  그래서 "짧은 소리 하나 + 긴 정적" 클립이 통과할 수 있다. 이 스크립트가 그것을 잡는다.

지표(프레임 RMS 25ms/10ms, validate_dataset.frame_rms 와 같은 정의)
  active_sec        : 프레임 RMS > max(1e-4, 0.05 x 그 파일 최대 프레임 RMS) 인 시간(초)
  max_inactive_run  : 그 기준 아래로 연속되는 가장 긴 구간(초)
  peak / rms        : 참고
표시 기준(기본): active_sec < 0.5  또는  max_inactive_run >= 2.0
  -> 기준의 타당성은 8월에 검증 통과한 mark5.0 복사분(7종)의 분포로 대조한다.

사용: python preprocessing/silence_audit.py            (결과: rebuild_cache/silence_audit_<시각>.csv + 요약)
"""
import os, sys, csv, time, argparse
from collections import defaultdict
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CODES_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
sys.path.insert(0, BASE_DIR)
from validate_dataset import frame_rms, max_run_seconds  # noqa: E402

COPIED = {"heavy_impact", "dragging", "construction", "machine_noise", "water_toilet", "water_shower", "dog_bark"}


def targets():
    """(dataset, split, group, path) 목록. 전문가 8개는 같은 파일이라 staging 사본 한 벌만 본다."""
    out = []
    for sp, d in (("train", "data_source_labeled"), ("val", "data_source_val"), ("test", "data_source_test")):
        root = os.path.join(PROJECT_ROOT, d)
        for cls in sorted(os.listdir(root)):
            for f in sorted(os.listdir(os.path.join(root, cls))):
                if f.endswith(".wav"):
                    out.append(("m52", sp, ("copied:" if cls in COPIED else "new:") + cls, os.path.join(root, cls, f)))
    ur = os.path.join(PROJECT_ROOT, "data_source_unlabeled")
    if os.path.isdir(ur):
        for f in sorted(os.listdir(ur)):
            if f.endswith(".wav"):
                out.append(("m52", "unlabeled", "unlabeled", os.path.join(ur, f)))
    er = os.path.join(PROJECT_ROOT, "data_eval")
    if os.path.isdir(er):
        for role in sorted(os.listdir(er)):
            for cat in sorted(os.listdir(os.path.join(er, role))):
                for f in sorted(os.listdir(os.path.join(er, role, cat))):
                    if f.endswith(".wav"):
                        out.append(("eval", role, role, os.path.join(er, role, cat, f)))
    st = os.path.join(BASE_DIR, "rebuild_cache", "expert_common")
    for sp in ("train", "val", "test"):
        d = os.path.join(st, sp)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                out.append(("expert", sp, "others", os.path.join(d, f)))
        d = os.path.join(st, "talking", sp)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                out.append(("expert", sp, "talking(4.5)", os.path.join(d, f)))
    m48 = os.path.join(CODES_ROOT, "Mark4_series", "Mark4.8", "data")
    for sp in ("train", "val", "test"):
        d = os.path.join(m48, sp)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("dog_bark_") and "_aug_" not in f and f.endswith(".wav"):
                    out.append(("m48", sp, "dog_bark", os.path.join(d, f)))
    return out


def measure(path):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        x = f.read(dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    fr = frame_rms(x, sr)
    thr = max(1e-4, 0.05 * float(fr.max())) if len(fr) else 1e-4
    act = fr > thr
    inactive_run = 0.0
    if len(act):
        runs, cur = [], 0
        for a in act:
            if a:
                runs.append(cur); cur = 0
            else:
                cur += 1
        runs.append(cur)
        inactive_run = max(runs) * 0.010
    return dict(duration=len(x) / sr, rms=float(np.sqrt(np.mean(x ** 2))), peak=float(np.abs(x).max()),
                active_sec=float(act.sum() * 0.010), max_inactive_run=inactive_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_active", type=float, default=0.5)
    ap.add_argument("--max_inactive", type=float, default=2.0)
    a = ap.parse_args()
    items = targets()
    print(f"[대상] {len(items)}개 파일")
    out_csv = os.path.join(BASE_DIR, "rebuild_cache", f"silence_audit_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    rows, t0 = [], time.time()
    for i, (ds, sp, grp, p) in enumerate(items, 1):
        try:
            m = measure(p)
        except Exception as e:
            m = dict(duration=-1, rms=-1, peak=-1, active_sec=-1, max_inactive_run=-1, error=str(e))
        m.update(dataset=ds, split=sp, group=grp, file=os.path.basename(p), path=p,
                 flag=("LOW_ACTIVE" if 0 <= m["active_sec"] < a.min_active else "") +
                      ("+LONG_GAP" if m["max_inactive_run"] >= a.max_inactive else ""))
        rows.append(m)
        if i % 1000 == 0:
            print(f"  {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "split", "group", "file", "duration", "rms", "peak",
                                          "active_sec", "max_inactive_run", "flag", "path"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"[저장] {out_csv}")

    # 요약: (dataset, group) 별 분포 + 표시 수
    print(f"\n[요약] 기준: active_sec < {a.min_active}s 또는 max_inactive_run >= {a.max_inactive}s")
    print(f"  {'dataset':7s} {'group':22s} {'n':>5s}  {'active p10':>10s} {'중앙':>6s}  {'gap p90':>8s} {'최대':>6s}  {'LOW':>4s} {'GAP':>4s} {'표시':>5s}")
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["group"])].append(r)
    total_flag = 0
    for k in sorted(by):
        rs = by[k]
        act = np.array([r["active_sec"] for r in rs]); gap = np.array([r["max_inactive_run"] for r in rs])
        low = sum(1 for r in rs if "LOW_ACTIVE" in r["flag"]); lg = sum(1 for r in rs if "LONG_GAP" in r["flag"])
        fl = sum(1 for r in rs if r["flag"]); total_flag += fl
        print(f"  {k[0]:7s} {k[1]:22s} {len(rs):5d}  {np.percentile(act,10):10.2f} {np.median(act):6.2f}  "
              f"{np.percentile(gap,90):8.2f} {gap.max():6.2f}  {low:4d} {lg:4d} {fl:5d}")
    print(f"\n[표시 합계] {total_flag} / {len(rows)}")
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"[읽기 실패] {len(errs)}개: " + ", ".join(r["file"] for r in errs[:10]))


if __name__ == "__main__":
    main()
