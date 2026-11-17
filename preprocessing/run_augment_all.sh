#!/usr/bin/env bash
# run_augment_all.sh — 전문가 8개의 train 증강본을 다시 만든다 (2026-09-15, rebuild 뒤 단계)
#   others(전부)·4.5 타겟이 바뀌었고 옛 _aug_ 는 _replaced_20260915/ 로 옮겨졌으므로 8개 전부 재생성.
#   모드 규칙(2026-08-01 확립): --dry_run 의 "[대상] 클래스별 active 평균" 을 읽어
#     |타겟 평균 - others 평균| > 0.05 이면 기본 모드, 이하면 --balanced.
#   목표: --target_per_class 1000 (train = 클래스당 원본 + 증강 = 1000, 4.x 표준 규격)
#   provenance 는 augment_density 가 스스로 갱신한다(버전마다 read/write 1회씩 — 8번).
# 사용: bash preprocessing/run_augment_all.sh            # 실행
#       bash preprocessing/run_augment_all.sh --plan     # 모드만 정하고 실행은 안 함
set -u
PY=~/venvs/mark5_exp/bin/python
CODES=/mnt/d/workspace/journal/codes
REPO=$(cd "$(dirname "$0")/.." && pwd)   # [2026-11-17] Mark5.2 에서 이 저장소(KAMPA)로 옮김
PLAN_ONLY=${1:-}
LOG=$REPO/preprocessing/rebuild_cache/augment_$(date +%Y%m%d_%H%M%S).log
: > "$LOG"
for i in 1 2 3 4 5 6 7 8; do
  WD=$CODES/Mark4_series/Mark4.$i
  V=mark4.$i
  DRY=$( cd "$WD" && $PY preprocessing/augment_density.py --mark_version $V --target_split train --target_per_class 1000 --provenance_path $CODES/data_provenance.xlsx --dry_run 2>&1 )
  MEANS=$(echo "$DRY" | grep "클래스별 active 평균" | head -1)
  # "[대상] split=train  클래스별 active 평균: dog_bark=0.3621, others=0.8231"
  GAP=$(echo "$MEANS" | $PY -c "
import re,sys
s=sys.stdin.read(); v=[float(x) for x in re.findall(r'=([0-9.]+)', s)]
print(abs(v[0]-v[1]) if len(v)==2 else -1)")
  MODE=""; [ "$(echo "$GAP <= 0.05" | bc -l)" = "1" ] && MODE="--balanced"
  echo "### $V  $MEANS  격차 $GAP  모드 ${MODE:-기본}" | tee -a "$LOG"
  if [ "$PLAN_ONLY" = "--plan" ]; then continue; fi
  ( cd "$WD" && $PY preprocessing/augment_density.py --mark_version $V --target_split train \
      --target_per_class 1000 --provenance_path $CODES/data_provenance.xlsx --reset $MODE ) >> "$LOG" 2>&1
  echo "  exit $?  $(ls $WD/data/train | grep -c '_aug_') aug 파일" | tee -a "$LOG"
done
echo "로그: $LOG"
