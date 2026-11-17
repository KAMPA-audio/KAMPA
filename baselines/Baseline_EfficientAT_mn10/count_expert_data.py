#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""전문가 8개(Mark4.1~4.8)의 학습 데이터 개수 — 논문 표 1 의 "추가 지식" 열.

[신설 2026-10-07] 제안 방법은 표준 반지도 기준선과 달리 클래스별 이진 전문가 8개를 추가로 쓴다. Noisy Student(CVPR 2020)
본표의 "Extra Data" 열처럼, 그 전문가들이 무엇으로 학습했는지를 숫자로 밝힌다.

읽는 것  <mark4_root>/Mark4.<i>/dataset_index_mark4.<i>.csv  (path,label — 분할은 경로의 data/train|val|test, label = 타겟 또는 others)
쓰는 것  plots/expert_data_counts.csv · plots/expert_data_counts.json

--hash 를 주면 오디오 파일 md5 로 서로 다른 오디오 개수도 센다(파일 이름 others_train_001.wav 등이 전문가마다 겹쳐
이름으로는 셀 수 없다 — 2026-10-07 실측). 오디오는 <mark4_root>/Mark4.<i>/data/<split>/<파일> 에서 찾는다.
<mark4_root> 기본값: 옆 폴더(코랩 드라이브는 experiments/Mark4.1 … 이 이 폴더와 나란히 있다), 없으면 ../Mark4_series(USB).
"""
import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_RE = re.compile(r"[\\/]data[\\/](train|val|test)[\\/]")


def _default_root():
    sib = os.path.abspath(os.path.join(ROOT, ".."))
    if os.path.isdir(os.path.join(sib, "Mark4.1")):
        return sib
    return os.path.join(sib, "Mark4_series")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="전문가 8개 학습 데이터 개수")
    ap.add_argument("--mark4_root", type=str, default=None)
    ap.add_argument("--hash", action="store_true", help="오디오 md5 로 서로 다른 오디오 개수도 센다")
    args = ap.parse_args()
    root = args.mark4_root or _default_root()
    rows_out, digests, missing = [], {"train": set(), "val": set(), "test": set()}, Counter()
    total = Counter()
    for i in range(1, 9):
        name = f"Mark4.{i}"
        idx = os.path.join(root, name, f"dataset_index_mark4.{i}.csv")
        if not os.path.isfile(idx):
            raise SystemExit(f"[ERROR] 인덱스가 없습니다: {idx}")
        with open(idx, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        c = Counter()
        for r in rows:
            p = r["path"].replace("\\", "/")
            m = SPLIT_RE.search(p)
            split = m.group(1) if m else "?"
            kind = "others" if r["label"] == "others" else "target"
            c[(split, kind)] += 1
            total[(split, kind)] += 1
            if args.hash and split in digests:
                local = os.path.join(root, name, "data", split, os.path.basename(p))
                if os.path.isfile(local):
                    digests[split].add(_md5(local))
                else:
                    missing[split] += 1
        target = next((r["label"] for r in rows if r["label"] != "others"), "?")
        line = {"expert": name, "target": target, "rows": len(rows)}
        for split in ("train", "val", "test"):
            for kind in ("target", "others"):
                line[f"{split}_{kind}"] = c[(split, kind)]
        rows_out.append(line)
        print(f"{name} ({target}): " + ", ".join(f"{s} {c[(s, 'target')]}+{c[(s, 'others')]}" for s in ("train", "val", "test")))
    summary = {"rows_total": sum(total.values()),
               "train_rows": total[("train", "target")] + total[("train", "others")],
               "val_rows": total[("val", "target")] + total[("val", "others")],
               "test_rows": total[("test", "target")] + total[("test", "others")]}
    if args.hash:
        summary.update({f"{s}_unique_audio_md5": len(digests[s]) for s in digests})
        summary.update({f"{s}_audio_missing": missing[s] for s in digests})
    print("합계:", summary)
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
    with open(os.path.join(ROOT, "plots", "expert_data_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with open(os.path.join(ROOT, "plots", "expert_data_counts.json"), "w", encoding="utf-8") as f:
        json.dump({"mark4_root": root, "per_expert": rows_out, "summary": summary}, f, ensure_ascii=False, indent=2)
    print("저장: plots/expert_data_counts.csv · plots/expert_data_counts.json")


if __name__ == "__main__":
    main()
