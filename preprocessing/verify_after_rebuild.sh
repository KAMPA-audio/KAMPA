#!/usr/bin/env bash
# verify_after_rebuild.sh — rebuild_mark52_data.py 뒤에 돌리는 검증 묶음 (2026-09-15)
#   KAMPA(구 Mark5.2 데이터) : validate(3 split) + audit(4 split)
#   전문가 8 : validate + audit   (증강본은 아직 없음 — augment_density --reset 전 상태)
# 판정 줄만 모아 rebuild_cache/verify_<시각>.txt 에 남긴다. 전체 출력은 같은 폴더의 verify_full_*.log.
set -u
PY=~/venvs/mark5_exp/bin/python
CODES=/mnt/d/workspace/journal/codes
REPO=$(cd "$(dirname "$0")/.." && pwd)   # [2026-11-17] Mark5.2 에서 이 저장소(KAMPA)로 옮김
TS=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/preprocessing/rebuild_cache/verify_$TS.txt
FULL=$REPO/preprocessing/rebuild_cache/verify_full_$TS.log
: > "$OUT"; : > "$FULL"
run() {  # $1 라벨, 나머지 명령
  local label="$1"; shift
  echo "### $label" | tee -a "$OUT" >> "$FULL"
  ( cd "$WD" && "$@" ) >> "$FULL" 2>&1
  local rc=$?
  # 마지막 실행분의 판정 줄만 뽑는다
  awk -v l="### $label" 'BEGIN{p=0} $0==l{p=1;next} /^### /{p=0} p' "$FULL" \
    | grep -E "^판정:|✗ |FAIL|\[STRICT\]|Traceback|Error|^\s+(WARN|ERROR|CRITICAL)\s+[A-Z_]+\s+[0-9]+건" | grep -v MISSING_FILE | head -40 >> "$OUT"
  echo "  (exit $rc)" >> "$OUT"
}

WD=$REPO
run "KAMPA validate" $PY preprocessing/validate_dataset.py --mark_version mark5.2 --data_root . \
    --splits data_source_labeled,data_source_val,data_source_test --strict
run "KAMPA audit" $PY preprocessing/audit_dataset.py --mark_version mark5.2 --data_root . \
    --splits data_source_labeled,data_source_val,data_source_test,data_source_unlabeled

for i in 1 2 3 4 5 6 7 8; do
  WD=$CODES/Mark4_series/Mark4.$i
  # 전문가 폴더는 Mark4_series/ 아래라 기본 경로(PROJECT_ROOT/../data_provenance.xlsx)가 어긋난다 — 명시
  run "Mark4.$i validate" $PY preprocessing/validate_dataset.py --mark_version mark4.$i --strict \
      --provenance_path $CODES/data_provenance.xlsx
  run "Mark4.$i audit" $PY preprocessing/audit_dataset.py --mark_version mark4.$i \
      --provenance_path $CODES/data_provenance.xlsx
done
echo; echo "===== 요약: $OUT ====="; cat "$OUT"
