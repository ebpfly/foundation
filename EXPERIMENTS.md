# SpectralNP Training Experiments

## Architecture

SpectralNP: sensor-agnostic spectral foundation model using a Neural Process
with band encoder → transformer aggregator → stochastic latent z → decoder.

The model takes (wavelength, fwhm, radiance) tokens from any sensor and
predicts dense at-sensor radiance at arbitrary query wavelengths with
uncertainty.

## Convergence test

`scripts/convergence_test.py` — the definitive pass/fail test. Takes one
spectrum, simulates through RTM, observes through sensors with increasing
band counts (3, 5, 7, 13, 30, 50, 100, 200, 400), and measures:

1. **RMSE vs truth** — should decrease with more bands
2. **Sharpness** (predicted σ) — should decrease with more bands
3. **Coverage@2σ** — should be near 95% (well-calibrated Gaussian)
4. **RMSE factor** (RMSE@3 / RMSE@400) — higher = model uses band info better

Band positions are **nested** (3 bands ⊂ 7 ⊂ ... ⊂ 400) so adding bands
only adds information.

---

## Experiment log

### v1–v3: multi-head, ARTS RTM (failed)

**Config**: d_model=256, n_layers=6, z_dim=128, 4 heads (radiance +
reflectance + material + atmospheric), real ARTS abs_lookup RT.

**Problem**: KL divergence collapsed/exploded in every run despite
multiple fixes (free bits, log_sigma clamping, log1p squashing, KL
warmup). Root cause: too many competing loss terms pulling the latent
in different directions.

**v3 result**: spectral loss plateaued at log_var_max ceiling (5.0);
reflectance loss diverged; KL oscillated wildly. No convergence.

### v4: VNIR/SWIR, class-balanced, calibration regulariser

**Changes**: dropped LWIR (fake data), aggregated material to 7 classes,
added calibration regulariser, class-balanced CE.

**Result**: R²=0.59 reflectance, 0.49 radiance, top-1=0.38 material.
BUT convergence test showed model produced **same prediction regardless
of band count** — posterior collapse. The deterministic representation
`r` dominated the decoder; the latent `z` was ignored.

### Baseline fast-iteration (5 epochs, simplified RTM)

Switched to fast iteration: 5 epochs, 2000 samples/epoch, d_model=128,
simplified RTM (matching the convergence test's RTM). ~5 min/run.

**Baseline**: RMSE factor 0.99×, collapsed.

### `nor` — drop r from decoder (THE KEY FIX)

**Hypothesis**: the deterministic representation `r` is bypassing the
latent `z` in the decoder. If the decoder sees `[r, z, query_enc]`, it
learns to use `r` (reliable) and ignore `z` (noisy).

**Fix**: `SpectralDecoder(use_r=False)` drops `r` from the MLP input.
The decoder only sees `[z, query_enc]`.

**Result at 5 epochs**: diversity ↑ (2.34 vs 0.83), sharpness goes in
the correct direction for the first time.

### `nor_50` / `nor_100` — longer training

Same config but 50 / 100 epochs. RMSE and sharpness both kept improving:

| Epoch | RMSE@400 (multi) | RMSE factor | Sharp ratio |
|-------|-------------------|-------------|-------------|
| 10    | 17.2             | 1.10×       | 1.09×       |
| 50    | 12.9             | 2.12×       | 1.15×       |
| 100   | 12.0             | 3.48×       | 1.15×       |

BUT: the ~12 RMSE floor turned out to be the **systematic gap between
ARTS RTM (training) and simplified RTM (evaluation)**. Not a model
capacity limit.

### `nor_simple` — matching train/eval RTM (CRITICAL FIX)

**Changed**: removed `--abs-lookup`, trained with simplified RTM
(same as convergence test eval). Also used `num_workers=4`.

| Epoch | RMSE@400 (multi) | RMSE factor | Sharp ratio |
|-------|-------------------|-------------|-------------|
| 10    | 8.8              | 2.22×       | 1.09×       |
| 20    | 6.9              | 2.63×       | 1.20×       |
| 50    | 5.7              | 6.95×       | 6.29×       |

**RMSE dropped from 12 → 5.7** just from matching the RTMs.

**Lesson**: always train and evaluate with the same forward model.

### Calibration problem: σ is 163× too small

Despite excellent mean predictions, the model's predicted σ is
meaningless. Coverage@2σ = 0.5% (should be 95%). Post-hoc temperature
scaling T=163 fixes coverage to 94%, confirming the uncertainty SHAPE
is correct but the SCALE is 163× off.

**Root cause**: Gaussian NLL has an asymmetric incentive. Correct
predictions push σ↓ (free loss reduction). Wrong predictions push σ↑
(costly). Averaged over batches, the model learns to collapse σ to
the floor.

### `calib` — calibration regulariser (w=0.5)

Explicit loss: `|log_var - log((y-μ)²)|`.

**Result**: coverage went from 0.5% → 27% (better) but RMSE went from
5.7 → 30.4 (catastrophic). The calibration term smothered the mean
gradient by inflating log_var early in training.

### `calib_light` — lighter regulariser (w=0.05)

Too slow to evaluate; abandoned.

### `beta_nll` — β-NLL (β=0.5)

Re-weights NLL by σ^(2β) to dampen the "shrink σ" incentive.

| Epoch | T (post-hoc) | RMSE@400 | Factor |
|-------|-------------|----------|--------|
| 10    | 36          | 7.8      | 3.26×  |
| 20    | 68          | 6.0      | 4.05×  |

T went from 36 → 68 as training continued — overconfidence worsened.
β=0.5 slows the collapse but doesn't prevent it.

### `crps` — Gaussian CRPS loss (CURRENT)

CRPS is a proper scoring rule: no "free shrink" term. The optimal σ
equals the actual prediction error magnitude.

| Epoch | T (post-hoc) | RMSE@400 | Factor |
|-------|-------------|----------|--------|
| 10    | 26.6        | 6.7      | 2.79×  |
| ...   | (running)   | ...      | ...    |

Early results: T=26.6 (6× better than NLL's 163). Watching whether T
stays stable or improves with more training (unlike NLL/β-NLL where it
got worse).

---

## Current best model

**Config**: `nor_simple` epoch 50 — simplified RTM, no-r decoder,
d_model=128, n_layers=4, z_dim=128, NLL loss.

**Metrics** (with post-hoc T=163 scaling):
- RMSE@400 = 5.7, factor 6.95×
- Coverage@2σ = 94% (post-hoc)
- Sharp ratio = 6.29×

**Pending**: CRPS loss should give good calibration WITHOUT post-hoc
scaling. Running now.

### Summary: variance collapse is architectural, not loss-related

All three losses tested (NLL, β-NLL, CRPS) converge to the same
failure: variance collapses as mean accuracy improves. The issue is
that the same final MLP layer outputs both μ and log_var — features
optimize for accuracy and drag log_var to the floor.

| Loss    | Epoch | RMSE@400 | Coverage@2σ | Post-hoc T |
|---------|-------|----------|-------------|------------|
| NLL     | 50    | 5.7      | 0.5%        | 163        |
| β-NLL   | 20    | 6.0      | 0.7%        | 68         |
| CRPS    | 40    | 3.3      | 1.6%        | 174        |

HOWEVER: post-hoc temperature scaling (T≈170) gives 94-98% coverage
with ALL losses. The model's **relative uncertainty** is correct — it
just needs a single scalar correction.

**Next steps (in priority order)**:
1. Accept T-scaling for now; focus on other priorities (LWIR, multi-head)
2. Later: separate mean and variance into two independent networks
3. Or: two-phase training (freeze mean, train variance head separately)
