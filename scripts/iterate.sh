#!/bin/bash
# Fast iteration loop: train ~5 min, run convergence test, report.
# Usage: scripts/iterate.sh <experiment_name> [extra args to pretrain]
#
# Example:
#   scripts/iterate.sh baseline
#   scripts/iterate.sh nokl --w-kl 0
#   scripts/iterate.sh bigz --z-dim 256

set -e
cd "$(dirname "$0")/.."

NAME="${1:-baseline}"
shift || true
OUT_DIR="checkpoints/iter_${NAME}"
LOG="/tmp/iter_${NAME}.log"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[iter] training run: ${NAME}"
START=$(date +%s)

KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 conda run -n atmgen-pyarts python3 -u \
    -m spectralnp.training.pretrain \
    --usgs-data /Users/eric/repo/atmgen/USGS_ASCIIdata/ASCIIdata_splib07a \
    --abs-lookup /Users/eric/.cache/atmgen/lut/abs_lookup_fa72fc35f64b.xml \
    --output-dir "$OUT_DIR" \
    --epochs 5 \
    --samples-per-epoch 2000 \
    --batch-size 64 \
    --lr 5e-4 \
    --lr-constant \
    --d-model 128 \
    --n-layers 4 \
    --z-dim 64 \
    --spectral-hidden 192 \
    --spectral-n-layers 3 \
    --wl-min-nm 380 \
    --wl-max-nm 2500 \
    --dense-n-points 425 \
    --kl-warmup-epochs 1 \
    --n-scenes-per-epoch 15 \
    --num-workers 0 \
    --device auto \
    --w-spectral 1.0 \
    --w-reflectance 0.0 \
    --w-atmos 0.0 \
    --w-material 0.0 \
    --w-kl 0.0001 \
    --w-calibration 0.0 \
    "$@" \
    > "$LOG" 2>&1

ELAPSED=$(($(date +%s) - START))
echo "[iter] training done in ${ELAPSED}s"
echo
echo "===== epoch trend ====="
grep "Epoch [0-9]*/5" "$LOG" | tail -10
echo
echo "===== convergence test ====="
KMP_DUPLICATE_LIB_OK=TRUE conda run -n atmgen-pyarts python3 scripts/convergence_test.py \
    --model "$OUT_DIR/best.pt" \
    --usgs-data /Users/eric/repo/atmgen/USGS_ASCIIdata/ASCIIdata_splib07a \
    --output "convergence_${NAME}.png" \
    --no-open 2>&1 | tail -15
echo
echo "[iter] saved convergence_${NAME}.png"

# Commit + push the convergence plot for this iteration
git add -f "convergence_${NAME}.png" "$LOG" 2>/dev/null || true
git add scripts/iterate.sh src/spectralnp/training/pretrain.py src/spectralnp/training/losses.py src/spectralnp/model/ 2>/dev/null || true
if ! git diff --cached --quiet; then
    git commit -m "iter ${NAME}: $(grep 'Epoch 5/5' "$LOG" | head -1 | sed 's/^.*Epoch/Epoch/' | head -c 100)" 2>&1 | tail -3
    git push origin "$(git branch --show-current)" 2>&1 | tail -2
fi
