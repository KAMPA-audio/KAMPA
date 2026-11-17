#!/usr/bin/env bash
# Rebuild the four result tables of the paper from the stored per-run outputs in results/runs/
# (no audio, no training) and check that they are identical to results/four_results_*.
#   KAMPA      : results/runs/kampa/plots/performance_summary_*_plain.csv (40 runs)
#   baselines  : results/runs/baselines/Baseline_*/plots/{performance_summary,prediction_details}_*.csv (60 runs)
set -euo pipefail
cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/kampa/plots"
cp results/runs/kampa/plots/*.csv "$work/kampa/plots/"
python model/make_four_results.py --root "$work/kampa" --baseline_root results/runs/baselines --no_plot
for f in four_results_1_test.csv four_results_2_test.csv four_results_3_test.csv four_results_4_test.csv four_results_test.md; do
  cmp "$work/kampa/plots/$f" "results/$f"
done
echo "[OK] results/four_results_{1,2,3,4}_test.csv and four_results_test.md were reproduced byte for byte."
