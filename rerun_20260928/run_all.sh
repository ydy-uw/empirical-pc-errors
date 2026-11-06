#!/usr/bin/env bash
# Full rerun of the 27 September package on the cleaned universes (28 September 2026).
# Run from the project root:  bash rerun_20260928/run_all.sh
# Steps 0-3 (WRDS pulls, candidates, universes, checks) were run before this script; see README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=emp/.venv/Scripts/python.exe
export HLHH_ROLLING_DIR="$PWD/emp/data/processed/rolling/rolling_topcap_clean_200601_202511_20260928"
export HLHH_RERUN_OUT="$PWD/rerun_20260928/work"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
C=rerun_20260928/code
L=rerun_20260928/work/logs; mkdir -p "$L"
PKG=paper/overleaf_upload_20260928
stamp() { date "+%H:%M:%S"; }

[ -f rerun_20260928/work/outlier_cleaning/_cache/panels_P3000.pkl ] || $PY $C/outlier_cleaning/panels.py build > $L/panels_build.log 2>&1
echo "$(stamp) 00_raw";       $PY $C/outlier_cleaning/run_method.py 00_raw > $L/00_raw.log 2>&1

echo "$(stamp) treatments 01-08 and the figure_code scripts that need only the panels, in parallel"
pids=()
for m in 01_fixed_clip 02_daily_winsorization 03_cross_sectional_mad 04_time_series_mad \
         05_log_returns 06_rank_transform 07_drop_names 08_reversal_filter; do
  ( $PY $C/outlier_cleaning/run_method.py $m > $L/$m.log 2>&1
    case $m in
      01_fixed_clip) $PY $C/outlier_cleaning/run_method.py $m --heavy-setting c050 > $L/${m}_c050.log 2>&1 ;;
      03_cross_sectional_mad) $PY $C/outlier_cleaning/run_method.py $m --heavy-setting c20 > $L/${m}_c20.log 2>&1 ;;
    esac ) &
  pids+=($!)
done
$PY $C/figure_code/outline_20260926/breadth.py > $L/breadth.log 2>&1 &   pids+=($!)
$PY $C/figure_code/outline_20260927/separation.py > $L/separation.log 2>&1 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || { echo "a parallel step failed; see $L"; exit 1; }

echo "$(stamp) scripts that need 00_raw and 01_fixed_clip c050"
pids=()
$PY $C/figure_code/outline_20260926/localized_check.py > $L/localized_check.log 2>&1 & pids+=($!)
OPENBLAS_NUM_THREADS=2 $PY $C/figure_code/outline_20260927/assumption3.py > $L/assumption3.log 2>&1 & pids+=($!)
$PY $C/outlier_cleaning/spectra.py 00_raw 01_fixed_clip 02_daily_winsorization 03_cross_sectional_mad \
    04_time_series_mad 05_log_returns 06_rank_transform 07_drop_names 08_reversal_filter > $L/spectra.log 2>&1 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || { echo "a step failed; see $L"; exit 1; }

echo "$(stamp) residual heatmaps, summaries, context, comparison"
$PY $C/figure_code/outline_20260926/residual_heatmaps.py $PKG > $L/residual_heatmaps.log 2>&1
$PY $C/outlier_cleaning/summarize.py > $L/summarize.log 2>&1
mkdir -p rerun_20260928/work/outlier_cleaning/overleaf_reports/tables/comparison
$PY $C/outlier_cleaning/context_stats.py > $L/context_stats.log 2>&1
$PY $C/outlier_cleaning/compare.py > $L/compare.log 2>&1

echo "$(stamp) exhibits into $PKG"
$PY $C/figure_code/outline_20260926/make_exhibits.py $PKG > $L/make_exhibits.log 2>&1
$PY $C/figure_code/outline_20260927/make_exhibits_0927.py $PKG > $L/make_exhibits_0927.log 2>&1
echo "$(stamp) done"
