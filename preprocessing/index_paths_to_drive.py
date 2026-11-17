#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
index_paths_to_drive.py — 전문가 8개 dataset_index_mark4.x.{csv,pkl}(루트·extraction·model·preprocessing 4벌)의 경로 앞부분을
USB(로컬) 경로 ↔ 드라이브(코랩) 경로로 바꾼다 (2026-10-23).

왜: run_expert_index_all.sh(generate_dataset_index.py)는 로컬 절대경로(/mnt/d/...)로 인덱스를 만든다. 커밋·드라이브에 올라가는
    인덱스는 코랩 경로(/content/drive/MyDrive/NewProject/codes/experiments/Mark4.k/...)여야 B0(teacher_train.py)가 파일을 그대로 연다
    (09-23 판 HEAD 도 이 형식). 코랩에서 인덱스를 다시 만들면 드라이브 data/ 에 남은 옛 wav 까지 들어갈 수 있으므로, 감사를 마친
    로컬 인덱스의 경로만 바꿔 쓴다. 행 순서·라벨은 그대로다.
사용: python preprocessing/index_paths_to_drive.py [--only 5]  # 로컬 → 드라이브
      python preprocessing/index_paths_to_drive.py --to_local # 드라이브 → 로컬(로컬에서 학습·스모크할 때)
검사: 바꾸기 전 모든 행이 출발 접두사로 시작해야 하고(아니면 멈춤), 4벌 csv 가 서로 같아야 하며, 바꾼 뒤 행 수·라벨이 같아야 한다.
"""
import argparse
import csv
import io
import os
import pickle
import sys

LOCAL = "/mnt/d/workspace/journal/codes/Mark4_series/"
DRIVE = "/content/drive/MyDrive/NewProject/codes/experiments/"
SERIES = "/mnt/d/workspace/journal/codes/Mark4_series"
SUBS = ["", "extraction", "model", "preprocessing"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to_local", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", default=list(range(1, 9)),
                    help="[10-24] 바꿀 교사 번호(기본 1~8). 예: --only 5 (Mark4.5 만 다시 만든 경우)")
    a = ap.parse_args()
    src, dst = (DRIVE, LOCAL) if a.to_local else (LOCAL, DRIVE)
    for i in a.only:
        repo, mv = f"{SERIES}/Mark4.{i}", f"mark4.{i}"
        csvs = [os.path.join(repo, s, f"dataset_index_{mv}.csv") for s in SUBS]
        texts = [open(p, encoding="utf-8").read() for p in csvs]
        if len(set(texts)) != 1:
            raise SystemExit(f"[STOP] {mv}: csv 4벌 내용이 서로 다릅니다 — run_expert_index_all.sh 를 다시 돌린 뒤 하십시오")
        rows = list(csv.DictReader(io.StringIO(texts[0])))
        bad = [r["path"] for r in rows if not r["path"].startswith(src)]
        if bad:
            raise SystemExit(f"[STOP] {mv}: '{src}' 로 시작하지 않는 행 {len(bad)}개(예: {bad[0]}) — 이미 바뀌었거나 다른 형식")
        new = [{"path": dst + r["path"][len(src):], "label": r["label"]} for r in rows]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["path", "label"], lineterminator="\r\n")   # HEAD 인덱스와 같은 CRLF
        w.writeheader()
        w.writerows(new)
        for s in SUBS:
            base = os.path.join(repo, s, f"dataset_index_{mv}")
            with open(base + ".csv", "w", encoding="utf-8", newline="") as f:
                f.write(buf.getvalue())
            old = pickle.load(open(base + ".pkl", "rb"))
            if [(d["path"], d["label"]) for d in old] != [(r["path"], r["label"]) for r in rows]:
                raise SystemExit(f"[STOP] {base}.pkl 이 csv 와 다릅니다")
            pickle.dump([dict(d, path=dst + d["path"][len(src):]) for d in old], open(base + ".pkl", "wb"))
        back = list(csv.DictReader(open(csvs[0], encoding="utf-8")))
        assert len(back) == len(rows) and [r["label"] for r in back] == [r["label"] for r in rows]
        print(f"{mv}: {len(rows)}행 · 4벌 csv·pkl → '{dst}…'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
