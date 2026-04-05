# Spectral VAE: Next Steps

## Posterior Collapse Fix

The current trained model suffers from posterior collapse — the decoder ignores the latent code `z` and outputs near-identical spectra. The generated spectra have ~37% of the variance of real data.

**Root cause**: beta=0.1 was too high, causing the encoder to collapse `mu → 0, log_var → 0` (matching the prior exactly), so the decoder learned to ignore `z`.

**Already fixed in code** (`spectral_vae.py`):
- Added **free bits** (0.5 nats/dim) to `vae_loss()` — prevents any latent dimension from collapsing
- Clamped `log_var` to [-10, 10] and initialized bias to -2.0

**To retrain with the fix**:
```bash
python -m spectralnp.training.train_vae \
    --usgs-data data/ASCIIdata_splib07a.zip \
    --epochs 200 \
    --batch-size 64 \
    --lr 3e-4 \
    --z-dim 32 \
    --beta 0.005 \
    --beta-warmup 50 \
    --augment \
    --output-dir checkpoints/vae2
```

Key changes from the first run:
- `--beta 0.005` (was 0.1) — much lower KL weight so reconstruction dominates
- `--beta-warmup 50` (was 30) — slower warmup gives encoder time to learn useful representations
- Free bits ensures each latent dimension carries at least 0.5 nats of information

## Further Improvements

- **Spectral loss weighting**: weight absorption feature regions (1.4, 1.9, 2.2 um) more heavily in reconstruction loss to learn sharp features
- **Larger z_dim**: try 64 or 128 for more expressive latent space
- **Category-conditioned generation**: condition decoder on material category for targeted generation
- **Spectral attention**: replace conv encoder with transformer for better long-range spectral dependencies
