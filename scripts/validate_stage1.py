#!/usr/bin/env python3
"""Validate the stage-1 Bayesian-PCA encoder (no training required).

Stage 1 has no learned parameters:
  1. PCA basis fitted from training spectra via SVD (offline).
  2. Given sparse sensor observations, a closed-form linear-Gaussian
     Bayesian update gives the PCA-coefficient posterior.
  3. Reconstructing the full radiance from the posterior mean is the
     stage-1 prediction.  The posterior covariance gives uncertainty.

This script runs the same convergence diagnostic as scripts/convergence_test.py
but skips the trained VAE and evaluates just the stage-1 encoder.  If stage 1
can't reconstruct radiance from sparse bands, stage 2 (VAE) has no hope.

Usage:
    python scripts/validate_stage1.py \\
        --usgs-data /path/to/USGS_ASCIIdata/ASCIIdata_splib07a \\
        --output stage1_validation.png

    # Without external data (synthetic spectra):
    python scripts/validate_stage1.py --output stage1_validation.png
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spectralnp.data.random_sensor import (
    VirtualSensor,
    add_sensor_noise,
    apply_sensor,
)
from spectralnp.data.rtm_simulator import (
    AtmosphericState,
    ViewGeometry,
    simplified_toa_radiance,
)
from spectralnp.model.foundation import FoundationConfig, SpectralFoundation

BAND_COUNTS = [3, 5, 7, 13, 30, 50, 100, 200, 400]


# ---------------------------------------------------------------------------
# PCA training data generation
# ---------------------------------------------------------------------------

def _random_atmos(rng: np.random.Generator) -> AtmosphericState:
    return AtmosphericState(
        aod_550=rng.uniform(0.01, 1.0),
        water_vapour=rng.uniform(0.2, 5.0),
        ozone_du=rng.uniform(200, 500),
        visibility_km=rng.uniform(5, 100),
        surface_altitude_km=rng.uniform(0, 3),
        surface_temperature_k=rng.uniform(260, 330),
    )


def _random_geom(rng: np.random.Generator) -> ViewGeometry:
    return ViewGeometry(
        solar_zenith_deg=rng.uniform(10, 70),
        sensor_zenith_deg=rng.uniform(0, 30),
        relative_azimuth_deg=rng.uniform(0, 180),
    )


def simulate_radiance(
    refl_matrix: np.ndarray,
    index_pool: np.ndarray,
    dense_wl: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate N radiance spectra by sampling reflectances from the given pool.

    Each draw picks a random reflectance index from ``index_pool``, combines
    it with random atmosphere and viewing geometry, and runs the simplified
    two-stream RTM.
    """
    out = np.zeros((n_samples, dense_wl.shape[0]), dtype=np.float32)
    for i in range(n_samples):
        idx = index_pool[rng.integers(0, len(index_pool))]
        refl = refl_matrix[idx]
        atmos = _random_atmos(rng)
        geom = _random_geom(rng)
        result = simplified_toa_radiance(
            surface_reflectance=refl,
            wavelength_nm=dense_wl,
            atmos=atmos,
            geometry=geom,
        )
        out[i] = result.toa_radiance.astype(np.float32)
    return out


def build_mixed_library(
    usgs_data: str | None,
    dense_wl: np.ndarray,
    n_synthetic_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Build a combined USGS + synthetic reflectance matrix.

    Returns
    -------
    refl_matrix : (N, W) reflectance values (clipped to [0, 1], NaNs filled)
    source_labels : list of "usgs" or "synthetic" per row
    """
    from spectralnp.data.synthetic_speclib import generate_synthetic_library

    labels: list[str] = []
    parts: list[np.ndarray] = []

    if usgs_data:
        from spectralnp.data.usgs_speclib import load_from_directory, load_from_zip
        p_path = Path(usgs_data)
        if p_path.suffix == ".zip":
            usgs = load_from_zip(p_path)
        else:
            usgs = load_from_directory(p_path)
        usgs = usgs.filter_wavelength_range(380, 2400)
        usgs_refl = usgs.to_array(dense_wl)
        usgs_refl = np.nan_to_num(usgs_refl, nan=0.04).clip(0, 1).astype(np.float32)
        parts.append(usgs_refl)
        labels.extend(["usgs"] * usgs_refl.shape[0])
        print(f"  USGS:      {usgs_refl.shape[0]} spectra")

    # Use an independent RNG so synthetic generation is reproducible
    synth = generate_synthetic_library(
        n_per_class=n_synthetic_per_class,
        wavelength_nm=dense_wl,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    synth_refl = synth.to_array(dense_wl)
    synth_refl = np.nan_to_num(synth_refl, nan=0.04).clip(0, 1).astype(np.float32)
    parts.append(synth_refl)
    labels.extend(["synthetic"] * synth_refl.shape[0])
    print(f"  Synthetic: {synth_refl.shape[0]} spectra")

    refl_matrix = np.concatenate(parts, axis=0)
    print(f"  Total:     {refl_matrix.shape[0]} spectra")
    return refl_matrix, labels


# ---------------------------------------------------------------------------
# Stage-1 prediction wrapper
# ---------------------------------------------------------------------------

class Stage1Predictor:
    """Wraps the Bayesian PCA update + reconstruction as a predictor.

    No learned parameters.  The only "training" is the PCA basis fit,
    which is done once from a set of training spectra.
    """

    def __init__(self, model: SpectralFoundation) -> None:
        self.model = model.eval()

    @torch.no_grad()
    def predict(
        self,
        wavelength_nm: np.ndarray,
        fwhm_nm: np.ndarray,
        radiance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the full radiance spectrum from sparse observations.

        Uses the FULL posterior covariance Σ_post (not just the diagonal).
        The off-diagonal correlations tighten the variance at observed
        wavelengths, producing local dips around each band centre.

        Returns
        -------
        mean : (W,) reconstructed radiance on the dense wavelength grid
        std  : (W,) per-wavelength standard deviation
        """
        wl = torch.from_numpy(wavelength_nm.astype(np.float32)).unsqueeze(0)
        fw = torch.from_numpy(fwhm_nm.astype(np.float32)).unsqueeze(0)
        rad = torch.from_numpy(radiance.astype(np.float32)).unsqueeze(0)
        mask = torch.ones_like(wl, dtype=torch.bool)

        # 1. Bayesian update — get full posterior covariance.
        mu_c, _var_c, Sigma = self.model.bayesian_update(
            wl, fw, rad, mask, return_full_cov=True,
        )

        # 2. Reconstruct mean radiance: L̂ = μ_c · P + μ_rad
        components = self.model.rad_pca_components           # (K, W)
        rad_mean = self.model.rad_pca_mean                   # (W,)
        mean = torch.matmul(mu_c, components) + rad_mean     # (1, W)

        # 3. Full variance propagation:
        #    Var(L[w]) = P[:, w]ᵀ Σ_post P[:, w]
        #    Computed batch-wise as diag(Pᵀ Σ P):
        #      Y = Σ @ components      →  (B, K, W)
        #      var[b, w] = Σ_k P[k, w] · Y[b, k, w]
        Y = torch.matmul(Sigma, components)                   # (B, K, W)
        var = (components.unsqueeze(0) * Y).sum(dim=-2)       # (B, W)
        std = var.clamp(min=0.0).sqrt()

        return mean.squeeze(0).cpu().numpy(), std.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Convergence eval (mirrors scripts/convergence_test.py)
# ---------------------------------------------------------------------------

def run_convergence(
    predictor: Stage1Predictor,
    refl: np.ndarray,
    dense_wl: np.ndarray,
    atmos: AtmosphericState,
    geom: ViewGeometry,
    band_counts: list[int],
    snr: float = 200.0,
    seed: int = 42,
    fwhm_override: float | None = None,
) -> dict:
    """Run stage-1 with varying band counts on a single test spectrum."""
    rng = np.random.default_rng(seed)

    truth = simplified_toa_radiance(
        surface_reflectance=refl, wavelength_nm=dense_wl,
        atmos=atmos, geometry=geom,
    ).toa_radiance.astype(np.float32)

    # Nested band positions (smaller N ⊂ larger N).
    max_n = max(band_counts)
    wl_lo, wl_hi = float(dense_wl[0]), float(dense_wl[-1])
    all_centers = np.linspace(wl_lo, wl_hi, max_n).astype(np.float32)
    if fwhm_override is not None:
        fwhm_val = float(fwhm_override)
    else:
        fwhm_val = float(rng.uniform(5.0, min(20.0, (wl_hi - wl_lo) / max_n * 2)))
    all_fwhm = np.full(max_n, fwhm_val, dtype=np.float32)

    results = []
    for n in band_counts:
        if n >= max_n:
            idx = np.arange(max_n)
        else:
            idx = np.round(np.linspace(0, max_n - 1, n)).astype(int)
        centers = all_centers[idx]
        fwhms = all_fwhm[idx]

        sensor = VirtualSensor(
            center_wavelength_nm=centers,
            fwhm_nm=fwhms,
        )
        band_rad = apply_sensor(sensor, dense_wl, truth).astype(np.float32)
        band_rad = add_sensor_noise(
            band_rad, rng, snr_range=(snr, snr),
        ).astype(np.float32)

        mean, std = predictor.predict(
            wavelength_nm=centers,
            fwhm_nm=fwhms,
            radiance=band_rad,
        )
        results.append({
            "n_bands": n,
            "pred_mean": mean,
            "pred_std": std,
            "input_wl": centers,
            "input_rad": band_rad,
        })

    return {
        "truth": truth,
        "dense_wl": dense_wl,
        "results": results,
    }


def compute_metrics(data: dict) -> dict:
    truth = data["truth"]
    results = data["results"]
    n_band_list = [r["n_bands"] for r in results]

    rmse = np.array([
        float(np.sqrt(np.mean((r["pred_mean"] - truth) ** 2)))
        for r in results
    ])
    sharpness = np.array([float(np.mean(r["pred_std"])) for r in results])
    coverage_2sigma = np.array([
        float(np.mean(np.abs(r["pred_mean"] - truth) <= 2 * np.maximum(r["pred_std"], 1e-9)))
        for r in results
    ])

    dense_wl = data["dense_wl"]
    obs_rmse = []
    for r in results:
        pred_at_obs = np.interp(r["input_wl"], dense_wl, r["pred_mean"])
        err = float(np.sqrt(np.mean((pred_at_obs - r["input_rad"]) ** 2)))
        obs_rmse.append(err)
    obs_rmse = np.array(obs_rmse)

    all_preds = np.stack([r["pred_mean"] for r in results])
    diversity_per_wl = all_preds.std(axis=0)
    mean_diversity = float(diversity_per_wl.mean())

    rmse_min_to_max = float(np.sqrt(np.mean(
        (all_preds[-1] - all_preds[0]) ** 2
    )))

    return {
        "n_bands": n_band_list,
        "rmse_vs_truth": rmse,
        "obs_rmse": obs_rmse,
        "sharpness": sharpness,
        "coverage_2sigma": coverage_2sigma,
        "mean_diversity": mean_diversity,
        "rmse_min_to_max": rmse_min_to_max,
    }


def make_plot(data: dict, metrics: dict, output_path: Path) -> None:
    truth = data["truth"]
    dense_wl = data["dense_wl"]
    results = data["results"]
    n_bands = metrics["n_bands"]

    n_rows = (len(results) + 3) // 4
    fig = plt.figure(figsize=(20, 4 + 3 * n_rows))
    gs = fig.add_gridspec(
        n_rows + 1, 4, hspace=0.45, wspace=0.30,
        height_ratios=[1.0] + [1.0] * n_rows,
    )

    # --- summary row ---
    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_rmse.plot(n_bands, metrics["rmse_vs_truth"], "o-", color="#e74c3c", linewidth=2)
    ax_rmse.set_xscale("log")
    ax_rmse.set_xlabel("# input bands")
    ax_rmse.set_ylabel("RMSE vs truth")
    ax_rmse.set_title("Accuracy (should ↓)")
    ax_rmse.grid(alpha=0.3)

    ax_sharp = fig.add_subplot(gs[0, 1])
    ax_sharp.plot(n_bands, metrics["sharpness"], "s-", color="#8e44ad", linewidth=2)
    ax_sharp.set_xscale("log")
    ax_sharp.set_xlabel("# input bands")
    ax_sharp.set_ylabel("mean σ̂")
    ax_sharp.set_title("Uncertainty (should ↓)")
    ax_sharp.grid(alpha=0.3)

    ax_cov = fig.add_subplot(gs[0, 2])
    ax_cov.plot(n_bands, metrics["coverage_2sigma"], "^-", color="#27ae60", linewidth=2)
    ax_cov.axhline(0.95, linestyle="--", color="gray", label="ideal 95%")
    ax_cov.set_xscale("log")
    ax_cov.set_xlabel("# input bands")
    ax_cov.set_ylabel("coverage @ 2σ")
    ax_cov.set_title("Calibration (target 0.95)")
    ax_cov.set_ylim(0, 1.05)
    ax_cov.legend(fontsize=8)
    ax_cov.grid(alpha=0.3)

    ax_obs = fig.add_subplot(gs[0, 3])
    obs_color = "#27ae60" if metrics["obs_rmse"][-1] <= metrics["obs_rmse"][0] * 1.2 else "#e74c3c"
    ax_obs.plot(n_bands, metrics["obs_rmse"], "D-", color=obs_color, linewidth=2)
    ax_obs.set_xscale("log")
    ax_obs.set_xlabel("# input bands")
    ax_obs.set_ylabel("RMSE at input band positions")
    ax_obs.set_title("Obs fidelity (should stay flat or ↓)", fontweight="bold",
                     color=obs_color)
    ax_obs.grid(alpha=0.3)

    # --- per-N spectral plots ---
    for i, r in enumerate(results):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        pred = r["pred_mean"]
        std = r["pred_std"]
        cov = metrics["coverage_2sigma"][i]

        error = np.abs(pred - truth)
        ax.fill_between(
            dense_wl, pred - error, pred + error,
            color="#e74c3c", alpha=0.15, label="actual error",
        )
        ax.fill_between(
            dense_wl, pred - 2 * std, pred + 2 * std,
            color="#3498db", alpha=0.35, label="predicted ±2σ",
        )
        ax.plot(dense_wl, truth, color="black", linewidth=0.8, alpha=0.5,
                label="true")
        ax.plot(dense_wl, pred, color="#3498db", linewidth=1.2,
                label="predicted")
        ax.scatter(r["input_wl"], r["input_rad"], s=8, color="black",
                   zorder=5)

        cov_color = "#27ae60" if 0.80 < cov < 0.99 else "#e74c3c"
        ax.set_title(
            f"{r['n_bands']} bands — RMSE={metrics['rmse_vs_truth'][i]:.1f}\n"
            f"cov@2σ={cov:.0%} (target 95%)",
            fontsize=9, color=cov_color, fontweight="bold",
        )
        ax.set_xlim(dense_wl[0], dense_wl[-1])
        if col == 0:
            ax.set_ylabel("radiance")
        if row == n_rows:
            ax.set_xlabel("wavelength (nm)")
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Stage-1 (Bayesian PCA) validation — no training\n"
        "PASS: RMSE↓ with n_bands, σ̂↓, coverage@2σ ∈ [80%, 99%]",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Validate the stage-1 Bayesian-PCA encoder")
    p.add_argument("--usgs-data", type=str, default=None,
                   help="Path to USGS ASCIIdata_splib07a (or omit for synthetic)")
    p.add_argument("--output", type=str, default="stage1_validation.png")
    p.add_argument("--n-pca-samples", type=int, default=2000,
                   help="Number of radiance spectra to fit PCA on")
    p.add_argument("--n-pca", type=int, default=0,
                   help="Number of PCA components (0 = full rank, recommended)")
    p.add_argument("--n-eval-spectra", type=int, default=10,
                   help="Number of held-out spectra for averaged metrics")
    p.add_argument("--n-synthetic-per-class", type=int, default=60,
                   help="Number of synthetic spectra generated per class "
                        "(5 classes total) to mix with USGS")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="Fraction of the combined library to hold out as test")
    p.add_argument("--snr", type=float, default=200.0,
                   help="SNR used in the forward simulation (noise added to obs)")
    p.add_argument("--model-snr", type=float, default=None,
                   help="SNR the Bayesian update assumes for observations. "
                        "Defaults to --snr.")
    p.add_argument("--fwhm", type=float, default=None,
                   help="Force all sensor bands to this FWHM (nm). "
                        "Use a small value (e.g. 0.01) for point-sample bands.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dense_wl = np.arange(380.0, 2501.0, 5.0)

    # --- Build mixed library (USGS + synthetic) ---
    print("Building mixed spectral library...")
    refl_matrix, source_labels = build_mixed_library(
        usgs_data=args.usgs_data,
        dense_wl=dense_wl,
        n_synthetic_per_class=args.n_synthetic_per_class,
        rng=rng,
    )
    n_total = refl_matrix.shape[0]

    # --- Train/test split (80/20 by default) ---
    perm = rng.permutation(n_total)
    n_test = max(args.n_eval_spectra, int(round(args.test_frac * n_total)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    n_train_usgs = sum(1 for i in train_idx if source_labels[i] == "usgs")
    n_train_syn = len(train_idx) - n_train_usgs
    n_test_usgs = sum(1 for i in test_idx if source_labels[i] == "usgs")
    n_test_syn = len(test_idx) - n_test_usgs
    print(f"  Train: {len(train_idx)} ({n_train_usgs} USGS + {n_train_syn} synthetic)")
    print(f"  Test:  {len(test_idx)} ({n_test_usgs} USGS + {n_test_syn} synthetic)")

    # --- Simulate training radiance from train split for PCA fitting ---
    print(f"Simulating {args.n_pca_samples} training radiance spectra for PCA...")
    train_rad = simulate_radiance(
        refl_matrix=refl_matrix,
        index_pool=train_idx,
        dense_wl=dense_wl,
        n_samples=args.n_pca_samples,
        rng=rng,
    )
    train_refl_for_pca = refl_matrix[train_idx]

    # --- Build model + fit PCA (no training needed) ---
    model_snr = args.model_snr if args.model_snr is not None else args.snr
    config = FoundationConfig(
        n_pca_radiance=args.n_pca,
        n_pca_reflectance=args.n_pca,
        assumed_snr=model_snr,
    )
    model = SpectralFoundation(config)
    diag = model.fit_pca(train_rad, train_refl_for_pca, dense_wl)
    ev = float(diag["radiance_explained_var"])
    print(f"Radiance PCA: {ev * 100:.4f}% variance explained "
          f"({(1 - ev) * 100:.4f}% residual) by "
          f"{model.config.n_pca_radiance} components")

    predictor = Stage1Predictor(model)

    # --- Visual test: pick one held-out spectrum ---
    visual_idx = int(test_idx[0])
    test_refl = refl_matrix[visual_idx]
    print(f"Visual test spectrum: index={visual_idx} source={source_labels[visual_idx]}")

    atmos = AtmosphericState(aod_550=0.20, water_vapour=2.0, ozone_du=320.0)
    geom = ViewGeometry(solar_zenith_deg=30.0, sensor_zenith_deg=0.0)

    data = run_convergence(
        predictor, test_refl, dense_wl, atmos, geom,
        band_counts=BAND_COUNTS, snr=args.snr, seed=args.seed,
        fwhm_override=args.fwhm,
    )
    metrics = compute_metrics(data)

    # --- Print verdict ---
    print()
    print("=== Stage-1 convergence metrics ===")
    print(f"  Mean diversity (should be > 0.5):        {metrics['mean_diversity']:.4f}")
    print(f"  min→max ΔRMSE (should be > 1.0):         {metrics['rmse_min_to_max']:.4f}")
    print(f"  RMSE @ {BAND_COUNTS[0]:3d} bands:                        {metrics['rmse_vs_truth'][0]:.4f}")
    print(f"  RMSE @ {BAND_COUNTS[-1]:3d} bands:                        {metrics['rmse_vs_truth'][-1]:.4f}")
    print(f"  RMSE improvement factor:                  "
          f"{metrics['rmse_vs_truth'][0] / max(metrics['rmse_vs_truth'][-1], 1e-9):.2f}×")
    print(f"  Sharpness @ {BAND_COUNTS[0]:3d} bands:                   {metrics['sharpness'][0]:.4f}")
    print(f"  Sharpness @ {BAND_COUNTS[-1]:3d} bands:                   {metrics['sharpness'][-1]:.4f}")
    print(f"  Coverage@2σ @ {BAND_COUNTS[0]:3d} bands:                 {metrics['coverage_2sigma'][0]:.1%}")
    print(f"  Coverage@2σ @ {BAND_COUNTS[-1]:3d} bands:                 {metrics['coverage_2sigma'][-1]:.1%}")

    obs_3 = metrics["obs_rmse"][0]
    obs_400 = metrics["obs_rmse"][-1]
    obs_trend = "✓ steady/improving" if obs_400 <= obs_3 * 1.2 else "✗ DEGRADING"
    print(f"  Obs-point RMSE @ {BAND_COUNTS[0]:3d} bands:              {obs_3:.4f}")
    print(f"  Obs-point RMSE @ {BAND_COUNTS[-1]:3d} bands:              {obs_400:.4f}")
    print(f"  Obs-point trend:                          {obs_trend}")

    # --- Averaged over multiple held-out spectra (from test split) ---
    if args.n_eval_spectra > 0:
        n_use = min(args.n_eval_spectra, len(test_idx))
        # Sample n_use test indices uniformly across the test split so
        # both USGS and synthetic are represented.
        step = max(len(test_idx) // n_use, 1)
        eval_indices = test_idx[::step][:n_use]
        eval_src = [source_labels[int(i)] for i in eval_indices]
        n_eval_usgs = sum(1 for s in eval_src if s == "usgs")
        n_eval_syn = n_use - n_eval_usgs
        rmse_3, rmse_400, sharp_3, sharp_400 = [], [], [], []
        cov_3_list, cov_400_list = [], []
        for si, idx in enumerate(eval_indices):
            refl_e = refl_matrix[int(idx)]
            d = run_convergence(
                predictor, refl_e, dense_wl, atmos, geom,
                band_counts=BAND_COUNTS, snr=args.snr, seed=args.seed + si,
                fwhm_override=args.fwhm,
            )
            m = compute_metrics(d)
            rmse_3.append(m["rmse_vs_truth"][0])
            rmse_400.append(m["rmse_vs_truth"][-1])
            sharp_3.append(m["sharpness"][0])
            sharp_400.append(m["sharpness"][-1])
            cov_3_list.append(m["coverage_2sigma"][0])
            cov_400_list.append(m["coverage_2sigma"][-1])
        rmse_3 = np.array(rmse_3)
        rmse_400 = np.array(rmse_400)
        sharp_3 = np.array(sharp_3)
        sharp_400 = np.array(sharp_400)
        print()
        print(f"=== Averaged over {n_use} held-out spectra "
              f"({n_eval_usgs} USGS + {n_eval_syn} synthetic) ===")
        print(f"  RMSE@{BAND_COUNTS[0]}:      mean {rmse_3.mean():.3f}  median {np.median(rmse_3):.3f}")
        print(f"  RMSE@{BAND_COUNTS[-1]}:    mean {rmse_400.mean():.3f}  median {np.median(rmse_400):.3f}")
        print(f"  RMSE factor: mean {(rmse_3 / np.maximum(rmse_400, 1e-9)).mean():.2f}×")
        print(f"  Sharp@{BAND_COUNTS[0]}:     mean {sharp_3.mean():.3f}")
        print(f"  Sharp@{BAND_COUNTS[-1]}:   mean {sharp_400.mean():.3f}")
        print(f"  Sharp ratio: mean {(sharp_3 / np.maximum(sharp_400, 1e-9)).mean():.2f}× (>1 = correct)")
        print(f"  Cov@2σ mean @{BAND_COUNTS[0]}:  {np.mean(cov_3_list):.1%}")
        print(f"  Cov@2σ mean @{BAND_COUNTS[-1]}: {np.mean(cov_400_list):.1%}")

    out = Path(args.output)
    make_plot(data, metrics, out)
    print(f"\nSaved plot: {out}")
    if not args.no_open:
        try:
            subprocess.run(["open", str(out)], check=False)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
