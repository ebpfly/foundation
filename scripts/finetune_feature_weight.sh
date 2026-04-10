#!/bin/bash
# Fine-tune the iter_full200 model with feature-weighted CRPS loss.
# Emphasises sharp spectral features (gas absorption, mineral bands).
#
# Usage:
#   scripts/finetune_feature_weight.sh [CHECKPOINT] [EXTRA_EPOCHS]
#
# Defaults:
#   CHECKPOINT = checkpoints/iter_full200/best.pt
#   EXTRA_EPOCHS = 50

set -euo pipefail

CKPT="${1:-checkpoints/iter_full200/best.pt}"
EXTRA="${2:-50}"
USGS="/Users/eric/repo/atmgen/USGS_ASCIIdata/ASCIIdata_splib07a"
PCAVAE="checkpoints/pca_vae_swir/final.pt"
OUTDIR="checkpoints/iter_feat_finetune"

echo "=== Fine-tuning with feature-weighted loss ==="
echo "  Checkpoint: $CKPT"
echo "  Extra epochs: $EXTRA"
echo "  Output: $OUTDIR"

KMP_DUPLICATE_LIB_OK=TRUE conda run -n atmgen-pyarts \
    python -u -m spectralnp.training.pretrain \
    --usgs-data "$USGS" \
    --pca-vae "$PCAVAE" \
    --output-dir "$OUTDIR" \
    --resume "$CKPT" \
    --epochs $((200 + EXTRA)) \
    --samples-per-epoch 5000 \
    --batch-size 64 \
    --lr 1e-4 \
    --lr-constant \
    --d-model 192 --n-layers 6 --n-frequencies 256 \
    --spectral-hidden 384 --spectral-n-layers 4 \
    --z-atm-dim 64 --z-surf-dim 192 \
    --wl-min-nm 380 --wl-max-nm 2500 \
    --dense-n-points 425 \
    --kl-warmup-epochs 10 \
    --num-workers 4 --device auto \
    --w-spectral 1.0 --w-reflectance 1.0 \
    --w-atmos 0.1 --w-material 0.2 \
    --w-kl 0.01 \
    --no-r-in-decoder \
    --feature-weight 2.0

echo "=== Fine-tuning complete. Running convergence test... ==="
KMP_DUPLICATE_LIB_OK=TRUE conda run -n atmgen-pyarts \
    python scripts/convergence_test.py \
    --model "$OUTDIR/best.pt" \
    --usgs-data "$USGS" \
    --held-out \
    --output convergence_feat_finetune.png

echo "=== Done ==="
open convergence_feat_finetune.png
