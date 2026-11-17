#!/usr/bin/env bash
# run_expert_index_all.sh — 전문가 8개의 dataset_index_mark4.x.csv/pkl 를 다시 만든다 (2026-09-15, 증강 뒤 단계).
#   others·4.5 타겟·4.8 dog_bark·증강본이 바뀌었으므로 커밋 대상인 인덱스도 새로 만들어야 한다.
#   generate_dataset_index.py 는 vild_config 를 import 하므로 torch 가 딸려온다(~/venvs/mark5_exp 에서 2~3초).
#   경로는 로컬 절대경로(/mnt/d/...)로 들어간다 — 코랩에서는 학습 전에 generate_dataset_index 를 다시 돌린다(기존 규칙).
set -u
PY=~/venvs/mark5_exp/bin/python
CODES=/mnt/d/workspace/journal/codes
REPO=$(cd "$(dirname "$0")/.." && pwd)   # [2026-11-17] Mark5.2 에서 이 저장소(KAMPA)로 옮김
LOG=$REPO/preprocessing/rebuild_cache/expert_index_$(date +%Y%m%d_%H%M%S).log
: > "$LOG"
for i in 1 2 3 4 5 6 7 8; do
  WD=$CODES/Mark4_series/Mark4.$i
  echo "### mark4.$i" | tee -a "$LOG"
  ( cd "$WD" && $PY preprocessing/generate_dataset_index.py --mark_version mark4.$i ) >> "$LOG" 2>&1
  rc=$?
  n=$( $PY -c "import csv,sys; print(sum(1 for _ in open('$WD/dataset_index_mark4.$i.csv', encoding='utf-8'))-1)" 2>/dev/null )
  echo "  exit $rc  인덱스 행 ${n:-?}" | tee -a "$LOG"
done
echo "로그: $LOG"
