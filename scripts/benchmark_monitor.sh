#!/bin/bash
# Continuously benchmark checkpoints/v2_reflectance/best.pt while training
# is running. Re-runs whenever the checkpoint mtime changes (i.e. a new
# best epoch was saved). Runs on CPU so it doesn't compete with the
# training process for the MPS GPU.
#
# Usage:
#     scripts/benchmark_monitor.sh [interval_seconds] [checkpoint_path]
#
# Output: timestamped subdirs in benchmarks/results/ + a single tail-able
# log at /tmp/benchmark_monitor.log

set -e
cd "$(dirname "$0")/.."

INTERVAL="${1:-300}"   # default: poll every 5 minutes
CKPT="${2:-checkpoints/v2_reflectance/best.pt}"
USGS="${SPECTRALNP_USGS:-/Users/eric/repo/atmgen/USGS_ASCIIdata/ASCIIdata_splib07a}"
LOG=/tmp/benchmark_monitor.log

last_mtime=""
echo "[monitor] starting; interval=${INTERVAL}s checkpoint=$CKPT" | tee -a "$LOG"

while true; do
    if [[ -f "$CKPT" ]]; then
        # macOS stat: -f %m
        cur_mtime=$(stat -f %m "$CKPT" 2>/dev/null || echo "0")
        if [[ "$cur_mtime" != "$last_mtime" ]]; then
            ts=$(date +"%Y-%m-%d %H:%M:%S")
            echo "[monitor] $ts new checkpoint detected (mtime=$cur_mtime), running benchmarks..." | tee -a "$LOG"

            KMP_DUPLICATE_LIB_OK=TRUE conda run -n atmgen-pyarts python3 \
                scripts/run_benchmarks.py \
                --model "$CKPT" \
                --usgs-data "$USGS" \
                --output benchmarks/results \
                --n-test-spectra 30 \
                --n-samples 8 \
                --device cpu \
                --no-open >> "$LOG" 2>&1

            echo "[monitor] $(date +"%H:%M:%S") done" | tee -a "$LOG"
            last_mtime="$cur_mtime"
        fi
    else
        echo "[monitor] $(date +"%H:%M:%S") waiting for $CKPT to exist" | tee -a "$LOG"
    fi
    sleep "$INTERVAL"
done
