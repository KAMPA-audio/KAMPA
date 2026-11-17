#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_data_release.py — 감사를 마친 전문가 8개 데이터의 '판'을 오디오 해시로 기록한다 (2026-10-23).

왜: dataset_index_mark4.x.csv 는 파일 이름만 담는다. 10-23 재배치처럼 같은 이름에 다른 오디오가 들어가면 인덱스 해시가
    예전과 같게 나올 수 있다(4.1~4.3 은 HEAD 와 바이트까지 같다). 그러면 코랩 노트북이 옛 오디오로 학습한 교사를 '끝난 판'으로
    보고 건너뛰거나, 드라이브에 새 오디오가 덜 올라간 채로 학습할 수 있다. 오디오 SHA-256 목록을 저장소에 두고 노트북이 대조한다.
무엇을: 저장소마다 data_release_mark4.x.json = {index_sha256, n_files, files: {data/<split>/<이름>: sha256}, release_sha256}
       release_sha256 = files 를 키 순서로 정렬한 json(구분자 ',' ':')의 SHA-256. 오디오는 인덱스 경로를 USB 경로로 바꿔 읽는다.
사용: python preprocessing/make_data_release.py [--only 5]   (index_paths_to_drive.py 뒤, 커밋 전)
"""
import csv
import hashlib
import json
import os
import re
import sys

SERIES = "/mnt/d/workspace/journal/codes/Mark4_series"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    only = [int(x) for x in sys.argv[sys.argv.index("--only") + 1:]] if "--only" in sys.argv else list(range(1, 9))   # [10-24] 예: --only 5
    for i in only:
        repo, mv = f"{SERIES}/Mark4.{i}", f"mark4.{i}"
        idx = os.path.join(repo, f"dataset_index_{mv}.csv")
        files = {}
        for r in csv.DictReader(open(idx, newline="", encoding="utf-8")):
            m = re.search(r"/Mark4\.%d/(data/(?:train|val|test)/[^/]+)$" % i, r["path"].replace("\\", "/"))
            if not m:
                raise SystemExit(f"[STOP] {mv}: 인덱스 경로 형식이 다릅니다: {r['path']}")
            files[m.group(1)] = sha256(os.path.join(repo, m.group(1)))
        rel = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        out = {"mark_version": mv, "index_sha256": sha256(idx), "n_files": len(files), "release_sha256": rel, "files": files}
        with open(os.path.join(repo, f"data_release_{mv}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, ensure_ascii=False, indent=0, sort_keys=True)
            f.write("\n")
        print(f"{mv}: 파일 {len(files)} · index {out['index_sha256'][:12]} · release {rel[:12]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
