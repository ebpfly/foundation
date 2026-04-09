# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install (editable, in venv)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_model.py::test_forward_shapes -v

# Lint
ruff check src/ tests/

# Train demo model (no external data needed)
python train_demo.py

# Pretrain on USGS data
python -m spectralnp.training.pretrain --usgs-data /path/to/usgs_splib07 --epochs 100
```

## Architecture

SpectralNP is a sensor-agnostic spectral foundation model. It accepts spectral measurements from any sensor (arbitrary band count, wavelength positions, FWHMs) and produces predictions with uncertainty that widens when fewer bands are observed.

**Data flow:** `{(λ, FWHM, L)} → BandEncoder → SpectralAggregator → Decoders → predictions + uncertainty`

### Model (`src/spectralnp/model/`)

The model has three components wired together in `spectralnp.py` via `SpectralNPConfig` dataclass:

- **BandEncoder** (`band_encoder.py`): Learnable sinusoidal wavelength encoding (frequencies in log-space) + FWHM MLP + wavelength-conditioned radiance MLP. The radiance encoder takes spectral position as input because the physical meaning of a radiance value depends on wavelength.

- **SpectralAggregator** (`spectral_aggregator.py`): Transformer with rotary positional encoding on wavelength (not token position), plus a Neural Process dual path:
  - *Deterministic path*: learnable latent queries cross-attend into band features → pooled to vector `r`
  - *Stochastic path*: mean-pooled band features → diagonal Gaussian `q(z|context)` whose width naturally shrinks with more bands
  
- **Decoders** (`decoders.py`): Three heads consuming `(r, z)`:
  - `SpectralDecoder`: continuous operator (DeepONet-style) — predicts radiance at arbitrary query wavelengths
  - `MaterialDecoder`: classification logits
  - `AtmosphericDecoder`: uses NIG head from `evidential.py` for decomposed aleatoric/epistemic uncertainty

### Uncertainty (`model/evidential.py`)

Two-level uncertainty: (1) NP latent `z` captures epistemic uncertainty from limited spectral coverage, (2) Normal-Inverse-Gamma (NIG) evidential outputs decompose aleatoric vs. epistemic per prediction. The evidence regularizer penalizes confidence when predictions are wrong.

### Data Pipeline (`src/spectralnp/data/`)

Training data is generated on-the-fly in `dataset.py`:
1. Surface reflectance sampled from USGS library (`usgs_speclib.py`) or synthetic spectra (`synthetic_speclib.py`)
2. Atmospheric state + geometry randomly sampled
3. At-sensor radiance computed via simplified two-stream RTM (`rtm_simulator.py`), with full PyARTS (ARTS) as optional backend
4. Random virtual sensor generated (`random_sensor.py`) — random band count (3-200), wavelength positions, FWHMs, and sampling strategy (uniform/clustered/regular)
5. Spectrum convolved with sensor SRFs + signal-dependent noise added

`sensor_definitions.py` has real sensor specs (Landsat-8, Sentinel-2, AVIRIS-NG, PRISMA, EnMAP) used for validation.

Variable-length batches are handled by `collate_spectral_batch()` which pads to max bands and returns a `pad_mask`.

### Training (`src/spectralnp/training/`)

`losses.py` combines four weighted terms: heteroscedastic spectral reconstruction (Gaussian NLL), atmospheric NIG NLL + evidence regularizer, NP KL divergence (context posterior vs. full-information posterior), and material cross-entropy.

`pretrain.py` is the CLI entry point with KL annealing (linear warmup), AdamW + cosine LR, gradient clipping at 1.0, and optional W&B logging.

### Inference (`src/spectralnp/inference/`)

`SpectralNPPredictor` wraps the model for single-observation or mixed-sensor batch prediction. `predict_with_uncertainty()` draws multiple z samples and reports mean ± std across samples. Mixed-sensor batches (different instruments per sample) work via padding.

## Experiment Log

`EXPERIMENTS.md` documents every training run, what was tried, why, and the result. **Update this file** after every iteration — it's the single source of truth for what's been tested and what worked.

## Iteration Workflow

Use the fast-iteration loop to test architectural and loss changes:

```bash
# Quick 5-epoch experiment (~5 min):
scripts/iterate.sh <experiment_name> [extra pretrain args]

# Example: test a new loss weight
scripts/iterate.sh my_experiment --w-kl 0.001

# The script automatically:
# 1. Trains for 5 epochs with a small model
# 2. Runs the convergence test on the result
# 3. Saves convergence_<name>.png
# 4. Commits + pushes the result
```

The **convergence test** (`scripts/convergence_test.py`) is the definitive pass/fail signal. A working model must show:
1. **RMSE decreases** with more input bands (factor > 2×)
2. **Sharpness decreases** with more input bands (ratio > 1)
3. **Coverage@2σ ≈ 95%** (well-calibrated uncertainty)

Use `--temperature 0` to apply post-hoc temperature scaling and see what calibration WOULD look like with correct variance scale.

For longer training runs:
```bash
# 50-100 epoch run (~35-70 min with simplified RTM):
python -m spectralnp.training.pretrain \
    --usgs-data /path/to/USGS_ASCIIdata/ASCIIdata_splib07a \
    --output-dir checkpoints/<name> \
    --epochs 50 --samples-per-epoch 5000 --lr 5e-4 --lr-constant \
    --no-r-in-decoder \
    [other flags]
```

After training, always run the convergence test and update EXPERIMENTS.md.

### Training data validation

```bash
python scripts/validate_training_data.py \
    --usgs-data /path/to/USGS_ASCIIdata/ASCIIdata_splib07a
```

Runs 41 automated tests on the training inputs and targets (shapes, ranges, physics sanity, sensor convolution, bandwidth, target consistency). Should pass 100%.

### Training watchdog

```bash
python scripts/training_watchdog.py /tmp/training.log --pid <PID>
```

Parses training log and detects NaN, KL collapse, divergence, or stall. Exit code 0 = ok, nonzero = problem.
