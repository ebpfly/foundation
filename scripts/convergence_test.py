#!/usr/bin/env python3
"""Quantitative test of model convergence as a function of input band count.

A working sensor-agnostic foundation model should:
  1. Get *closer* to the true at-sensor radiance as the number of input
     bands grows.
  2. Get *more confident* (smaller predicted sigma) as the number of input
     bands grows.
  3. Produce *different* predictions at different band counts (not collapse
     to the same output regardless of input).

If the latent variable has collapsed (decoder ignores ``z``), the model
will produce the *same* spectrum for any band count. This test catches
that failure mode directly via the "diversity score" panel.

Usage:
    python scripts/convergence_test.py \
        --model checkpoints/v4_arts/best.pt \
        --usgs-data /path/to/USGS_ASCIIdata/ASCIIdata_splib07a \
        --output convergence_v4.png
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
    sample_virtual_sensor,
)
from spectralnp.data.rtm_simulator import (
    AtmosphericState,
    ViewGeometry,
    simplified_toa_radiance,
)
from spectralnp.data.usgs_speclib import load_from_directory, load_from_zip
from spectralnp.inference.predict import SpectralNPPredictor
from spectralnp.model.spectralnp import SpectralNP

BAND_COUNTS = [3, 5, 7, 13, 30, 50, 100, 200, 400]


def run_convergence(
    predictor: SpectralNPPredictor,
    spec,
    dense_wl: np.ndarray,
    atmos: AtmosphericState,
    geom: ViewGeometry,
    band_counts: list[int],
    n_samples: int = 32,
    snr: float = 200.0,
    seed: int = 42,
) -> dict:
    """Run the model with varying band counts on a single test spectrum."""
    rng = np.random.default_rng(seed)

    # Ground-truth dense radiance.
    refl = np.clip(np.nan_to_num(spec.resample(dense_wl), nan=0.04), 0, 1)
    truth = simplified_toa_radiance(
        surface_reflectance=refl, wavelength_nm=dense_wl,
        atmos=atmos, geometry=geom,
    ).toa_radiance.astype(np.float32)

    # Generate NESTED band positions: the largest band count defines a
    # superset, and smaller counts are strict subsets. This ensures that
    # adding more bands only ADDS information — it never moves bands.
    max_n = max(band_counts)
    wl_lo, wl_hi = float(dense_wl[0]), float(dense_wl[-1])
    all_centers = np.linspace(wl_lo, wl_hi, max_n).astype(np.float32)
    fwhm_val = rng.uniform(5.0, min(20.0, (wl_hi - wl_lo) / max_n * 2))
    all_fwhm = np.full(max_n, fwhm_val, dtype=np.float32)

    # For each band count N, take the first N positions from a
    # uniformly-spaced grid that is consistent across N values.
    # Use every (max_n // N)-th band so spacing stays uniform.
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

        pred = predictor.predict(
            wavelength_nm=centers,
            fwhm_nm=fwhms,
            radiance=band_rad,
            query_wavelength_nm=dense_wl,
            n_samples=n_samples,
        )
        results.append({
            "n_bands": n,
            "pred_mean": pred.spectral_mean.astype(np.float32),
            "pred_std": pred.spectral_std.astype(np.float32),
            "input_wl": centers,
            "input_rad": band_rad,
        })

    return {
        "truth": truth,
        "dense_wl": dense_wl,
        "results": results,
        "spec_name": getattr(spec, "name", "test_spectrum"),
        "category": getattr(spec, "category", ""),
    }


def compute_metrics(data: dict) -> dict:
    """Aggregate per-N convergence metrics + cross-N diversity."""
    truth = data["truth"]
    results = data["results"]
    n_band_list = [r["n_bands"] for r in results]

    rmse = np.array([
        float(np.sqrt(np.mean((r["pred_mean"] - truth) ** 2)))
        for r in results
    ])
    sharpness = np.array([float(np.mean(r["pred_std"])) for r in results])
    coverage_2sigma = np.array([
        float(np.mean(np.abs(r["pred_mean"] - truth) <= 2 * r["pred_std"]))
        for r in results
    ])

    # Observation-point fidelity: RMSE between the predicted radiance AT
    # the input band wavelengths and the actual input band radiance.
    # This must NOT degrade as band count increases — if it does, the
    # encoder is losing per-band information (e.g. mean-pooling dilution).
    dense_wl = data["dense_wl"]
    obs_rmse = []
    for r in results:
        # Interpolate the dense prediction to the input band positions.
        pred_at_obs = np.interp(r["input_wl"], dense_wl, r["pred_mean"])
        err = float(np.sqrt(np.mean((pred_at_obs - r["input_rad"]) ** 2)))
        obs_rmse.append(err)
    obs_rmse = np.array(obs_rmse)

    # Cross-N similarity: how much do predictions change between band counts?
    all_preds = np.stack([r["pred_mean"] for r in results])
    diversity_per_wl = all_preds.std(axis=0)
    mean_diversity = float(diversity_per_wl.mean())

    # Pairwise RMSE between adjacent band counts.
    pairwise_rmse = np.array([
        float(np.sqrt(np.mean((all_preds[i + 1] - all_preds[i]) ** 2)))
        for i in range(len(results) - 1)
    ])

    # The big test: what is the RMSE between the smallest-N and largest-N
    # predictions? If the model is collapsed, this will be ~0.
    rmse_min_to_max = float(np.sqrt(np.mean(
        (all_preds[-1] - all_preds[0]) ** 2
    )))

    return {
        "n_bands": n_band_list,
        "rmse_vs_truth": rmse,
        "obs_rmse": obs_rmse,
        "sharpness": sharpness,
        "coverage_2sigma": coverage_2sigma,
        "diversity_per_wl": diversity_per_wl,
        "mean_diversity": mean_diversity,
        "pairwise_rmse": pairwise_rmse,
        "rmse_min_to_max": rmse_min_to_max,
    }


def make_plot(
    data: dict, metrics: dict, output_path: Path,
) -> None:
    """Save the convergence visualization to PNG."""
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

    # ----- Row 0: summary metrics -----
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

    # Observation-point fidelity: can the model reproduce its own inputs?
    ax_obs = fig.add_subplot(gs[0, 3])
    obs_color = "#27ae60" if metrics["obs_rmse"][-1] <= metrics["obs_rmse"][0] * 1.2 else "#e74c3c"
    ax_obs.plot(n_bands, metrics["obs_rmse"], "D-", color=obs_color, linewidth=2)
    ax_obs.set_xscale("log")
    ax_obs.set_xlabel("# input bands")
    ax_obs.set_ylabel("RMSE at input band positions")
    ax_obs.set_title("Obs fidelity (should stay flat or ↓)", fontweight="bold",
                     color=obs_color)
    ax_obs.grid(alpha=0.3)

    # ----- Per-N spectral plots -----
    for i, r in enumerate(results):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        pred = r["pred_mean"]
        std = r["pred_std"]
        cov = metrics["coverage_2sigma"][i]

        # Show the actual error as a shaded region so the user can see
        # whether the predicted ±2σ is appropriately sized.
        error = np.abs(pred - truth)
        ax.fill_between(
            dense_wl,
            pred - error,
            pred + error,
            color="#e74c3c", alpha=0.15, label="actual error",
        )
        # Predicted ±2σ — draw even if very thin, with a visible edge.
        ax.fill_between(
            dense_wl,
            pred - 2 * std,
            pred + 2 * std,
            color="#3498db", alpha=0.35, label="predicted ±2σ",
        )
        ax.plot(dense_wl, truth, color="black", linewidth=0.8, alpha=0.5,
                label="true")
        ax.plot(dense_wl, pred, color="#3498db", linewidth=1.2,
                label="predicted")
        ax.scatter(r["input_wl"], r["input_rad"], s=8, color="black",
                   zorder=5)

        # Coverage color: green if near 95%, red if way off.
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
        f"Convergence test — {data['spec_name']} ({data['category']})\n"
        f"PASS: RMSE↓ with n_bands, sharpness↓, coverage@2σ ∈ [80%, 99%]",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--usgs-data", type=str, required=True,
                   help="USGS spectral library directory or zip")
    p.add_argument("--output", type=str, default="convergence_test.png")
    p.add_argument("--category", type=str, default="vegetation",
                   help="USGS category to pick a test spectrum from (visual plot)")
    p.add_argument("--spec-index", type=int, default=2,
                   help="Index within the category (visual plot)")
    p.add_argument("--n-eval-spectra", type=int, default=10,
                   help="Number of additional held-out spectra to average "
                        "convergence metrics across (in addition to the visual one)")
    p.add_argument("--held-out", action="store_true",
                   help="Use SYNTHETIC spectra (never seen in training) for the "
                        "multi-spectrum eval. Critical for detecting memorisation.")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--snr", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=None,
                   help="Apply post-hoc temperature scaling to predicted σ. "
                        "If 'auto', compute optimal T from z-scores. "
                        "If a number, use that T directly.")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    # ----- Load model -----
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SpectralNP.from_checkpoint(ckpt)
    model.eval()
    predictor = SpectralNPPredictor(model)
    print(f"Loaded {args.model} (epoch {ckpt.get('epoch')}, "
          f"loss {ckpt.get('loss', 0):.4f})")

    # ----- Load USGS -----
    p_path = Path(args.usgs_data)
    if p_path.suffix == ".zip":
        speclib = load_from_zip(p_path)
    else:
        speclib = load_from_directory(p_path)
    speclib = speclib.filter_wavelength_range(380, 2400)
    print(f"Loaded {len(speclib)} USGS spectra")

    subset = speclib.filter_category(args.category)
    if args.spec_index >= len(subset):
        raise ValueError(
            f"category={args.category} only has {len(subset)} spectra"
        )
    spec = subset.spectra[args.spec_index]
    print(f"Test spectrum: {spec.name} ({spec.category})")

    # ----- Test setup -----
    dense_wl = np.arange(380.0, 2501.0, 5.0)
    atmos = AtmosphericState(aod_550=0.20, water_vapour=2.0, ozone_du=320.0)
    geom = ViewGeometry(solar_zenith_deg=30.0, sensor_zenith_deg=0.0)

    # ----- Run + plot -----
    data = run_convergence(
        predictor, spec, dense_wl, atmos, geom,
        band_counts=BAND_COUNTS, n_samples=args.n_samples,
        snr=args.snr, seed=args.seed,
    )

    # Optional temperature scaling: multiply predicted σ by T before
    # computing metrics and plotting. If --temperature auto, compute T
    # from z-scores. If --temperature <float>, use that value.
    if args.temperature is not None:
        if args.temperature == 0:  # sentinel for "auto"
            all_z = []
            for r in data["results"]:
                z = (data["truth"] - r["pred_mean"]) / np.maximum(r["pred_std"], 1e-9)
                all_z.extend(z.tolist())
            T = float(np.std(all_z))
            print(f"Auto temperature: T = {T:.1f}")
        else:
            T = args.temperature
            print(f"Applied temperature: T = {T:.1f}")
        for r in data["results"]:
            r["pred_std"] = r["pred_std"] * T

    metrics = compute_metrics(data)

    # ----- Print verdict -----
    print()
    print("=== Convergence metrics ===")
    print(f"  Mean diversity (should be > 0.5):        {metrics['mean_diversity']:.4f}")
    print(f"  min→max ΔRMSE (should be > 1.0):         {metrics['rmse_min_to_max']:.4f}")
    print(f"  RMSE @ {BAND_COUNTS[0]} bands:                          {metrics['rmse_vs_truth'][0]:.4f}")
    print(f"  RMSE @ {BAND_COUNTS[-1]} bands:                        {metrics['rmse_vs_truth'][-1]:.4f}")
    print(f"  RMSE improvement factor:                  "
          f"{metrics['rmse_vs_truth'][0] / max(metrics['rmse_vs_truth'][-1], 1e-9):.2f}×")
    print(f"  Sharpness @ {BAND_COUNTS[0]} bands:                     {metrics['sharpness'][0]:.4f}")
    print(f"  Sharpness @ {BAND_COUNTS[-1]} bands:                    {metrics['sharpness'][-1]:.4f}")

    # Observation-point fidelity: does the model reproduce its own inputs?
    obs_3 = metrics["obs_rmse"][0]
    obs_400 = metrics["obs_rmse"][-1]
    obs_trend = "✓ steady/improving" if obs_400 <= obs_3 * 1.2 else "✗ DEGRADING"
    print(f"  Obs-point RMSE @ {BAND_COUNTS[0]} bands:                {obs_3:.4f}")
    print(f"  Obs-point RMSE @ {BAND_COUNTS[-1]} bands:               {obs_400:.4f}")
    print(f"  Obs-point trend:                          {obs_trend}")
    if obs_400 > obs_3 * 1.5:
        print(f"  >>> OBSERVATION FIDELITY DEGRADATION <<<")
        print(f"  Accuracy at input band positions gets WORSE with more bands.")
        print(f"  Likely cause: encoder aggregation is too lossy (mean-pooling?).")

    # Coverage @ 2σ (calibration check)
    cov_3 = metrics["coverage_2sigma"][0]
    cov_400 = metrics["coverage_2sigma"][-1]
    print(f"  Coverage@2σ @ {BAND_COUNTS[0]} bands:                   {cov_3:.1%}")
    print(f"  Coverage@2σ @ {BAND_COUNTS[-1]} bands:                  {cov_400:.1%}")
    print(f"  Target coverage@2σ:                       95%")
    ideal = 0.9545
    calib_err_3 = abs(cov_3 - ideal)
    calib_err_400 = abs(cov_400 - ideal)
    print(f"  Calibration error @ {BAND_COUNTS[0]} bands:              {calib_err_3:.1%}")
    print(f"  Calibration error @ {BAND_COUNTS[-1]} bands:             {calib_err_400:.1%}")

    if metrics["mean_diversity"] < 0.05 or metrics["rmse_min_to_max"] < 0.05:
        print()
        print("  >>> POSTERIOR COLLAPSE DETECTED <<<")
        print("  Model produces ~identical predictions regardless of band count.")
        print("  Decoder is ignoring the latent z. KL regularization too strong.")

    if cov_400 < 0.10:
        print()
        print("  >>> WILDLY OVERCONFIDENT <<<")
        print(f"  Coverage@2σ = {cov_400:.1%} at {BAND_COUNTS[-1]} bands (should be ~95%).")
        print(f"  Predicted σ is {metrics['sharpness'][-1]:.4f} but RMSE is "
              f"{metrics['rmse_vs_truth'][-1]:.2f} — uncertainty is "
              f"~{metrics['rmse_vs_truth'][-1] / max(metrics['sharpness'][-1], 1e-9):.0f}× too small.")
    elif cov_400 > 0.99:
        print()
        print("  >>> UNDERCONFIDENT <<<")
        print(f"  Coverage@2σ = {cov_400:.1%} at {BAND_COUNTS[-1]} bands (should be ~95%).")
        print("  Predicted uncertainty is too wide.")

    # Post-hoc temperature scaling diagnostic: compute the optimal
    # temperature T such that (y - μ) / (T·σ) is standard normal.
    all_z = []
    for r in data["results"]:
        z = (data["truth"] - r["pred_mean"]) / np.maximum(r["pred_std"], 1e-9)
        all_z.extend(z.tolist())
    z_arr = np.array(all_z)
    T_opt = float(z_arr.std())
    cov_posthoc = float(np.mean(np.abs(z_arr / T_opt) < 2))
    print()
    print(f"  Post-hoc temperature scaling:")
    print(f"    Optimal T (σ multiplier):        {T_opt:.1f}")
    print(f"    Coverage@2σ after T-scaling:      {cov_posthoc:.1%}")

    # Multi-spectrum eval: average convergence metrics over many random
    # spectra to confirm the visual single-spectrum result generalises.
    if args.n_eval_spectra > 0:
        if args.held_out:
            # Use SYNTHETIC spectra that were NEVER in training.
            from spectralnp.data.synthetic_speclib import generate_synthetic_library
            held_out_lib = generate_synthetic_library(
                n_per_class=max(args.n_eval_spectra // 5, 5), seed=99999
            )
            eval_spectra = held_out_lib.spectra[:args.n_eval_spectra]
            label = "held-out SYNTHETIC"
        else:
            rng2 = np.random.default_rng(args.seed + 1000)
            eval_spec_indices = rng2.choice(
                len(speclib), size=min(args.n_eval_spectra, len(speclib)),
                replace=False,
            )
            eval_spectra = [speclib.spectra[int(i)] for i in eval_spec_indices]
            label = "training-set USGS"
        rmse_3, rmse_400, sharp_3, sharp_400 = [], [], [], []
        for si, spec_e in enumerate(eval_spectra):
            data_e = run_convergence(
                predictor, spec_e, dense_wl, atmos, geom,
                band_counts=BAND_COUNTS, n_samples=args.n_samples,
                snr=args.snr, seed=args.seed + si,
            )
            m_e = compute_metrics(data_e)
            rmse_3.append(m_e["rmse_vs_truth"][0])
            rmse_400.append(m_e["rmse_vs_truth"][-1])
            sharp_3.append(m_e["sharpness"][0])
            sharp_400.append(m_e["sharpness"][-1])
        rmse_3 = np.array(rmse_3)
        rmse_400 = np.array(rmse_400)
        sharp_3 = np.array(sharp_3)
        sharp_400 = np.array(sharp_400)
        print()
        print(f"=== Averaged over {len(eval_spectra)} spectra ({label}) ===")
        print(f"  RMSE@3:      mean {rmse_3.mean():.3f}  median {np.median(rmse_3):.3f}")
        print(f"  RMSE@400:    mean {rmse_400.mean():.3f}  median {np.median(rmse_400):.3f}")
        print(f"  RMSE factor: mean {(rmse_3/np.maximum(rmse_400,1e-9)).mean():.2f}×")
        print(f"  Sharp@3:     mean {sharp_3.mean():.3f}")
        print(f"  Sharp@400:   mean {sharp_400.mean():.3f}")
        print(f"  Sharp ratio: mean {(sharp_3/np.maximum(sharp_400,1e-9)).mean():.2f}× (>1 = correct)")

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
