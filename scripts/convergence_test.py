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

    results = []
    for n in band_counts:
        # Uniform spacing across the full dense wavelength range — this
        # isolates "more bands" from "different band positions".
        sensor = sample_virtual_sensor(
            rng,
            n_bands_range=(n, n),
            wavelength_range=(float(dense_wl[0]), float(dense_wl[-1])),
            strategy="regular",
        )
        band_rad = apply_sensor(sensor, dense_wl, truth).astype(np.float32)
        band_rad = add_sensor_noise(
            band_rad, rng, snr_range=(snr, snr),
        ).astype(np.float32)

        pred = predictor.predict(
            wavelength_nm=sensor.center_wavelength_nm,
            fwhm_nm=sensor.fwhm_nm,
            radiance=band_rad,
            query_wavelength_nm=dense_wl,
            n_samples=n_samples,
        )
        results.append({
            "n_bands": n,
            "pred_mean": pred.spectral_mean.astype(np.float32),
            "pred_std": pred.spectral_std.astype(np.float32),
            "input_wl": sensor.center_wavelength_nm.astype(np.float32),
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

    # Diversity panel — the collapse detector.
    ax_div = fig.add_subplot(gs[0, 3])
    ax_div.plot(dense_wl, metrics["diversity_per_wl"], color="#34495e", linewidth=1.2)
    ax_div.set_xlabel("wavelength (nm)")
    ax_div.set_ylabel("std of pred mean across N values")
    title_color = "#27ae60" if metrics["mean_diversity"] > 0.5 else "#e74c3c"
    ax_div.set_title(
        f"Diversity (mean={metrics['mean_diversity']:.3f})\n"
        f"min→max ΔRMSE={metrics['rmse_min_to_max']:.3f}",
        color=title_color, fontweight="bold",
    )
    ax_div.grid(alpha=0.3)

    # ----- Per-N spectral plots -----
    for i, r in enumerate(results):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        ax.plot(dense_wl, truth, color="black", linewidth=0.8, alpha=0.5,
                label="true")
        ax.plot(dense_wl, r["pred_mean"], color="#3498db", linewidth=1.2,
                label="predicted")
        ax.fill_between(
            dense_wl,
            r["pred_mean"] - 2 * r["pred_std"],
            r["pred_mean"] + 2 * r["pred_std"],
            color="#3498db", alpha=0.25, label="±2σ",
        )
        ax.scatter(r["input_wl"], r["input_rad"], s=8, color="black",
                   zorder=5)
        ax.set_title(f"{r['n_bands']} bands "
                     f"(RMSE={metrics['rmse_vs_truth'][i]:.2f})",
                     fontsize=10)
        ax.set_xlim(dense_wl[0], dense_wl[-1])
        if col == 0:
            ax.set_ylabel("radiance")
        if row == n_rows:
            ax.set_xlabel("wavelength (nm)")
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Convergence test — {data['spec_name']} ({data['category']})\n"
        f"PASS criteria: RMSE↓ with n_bands, sharpness↓, mean diversity > 0.5, "
        f"min→max ΔRMSE > 1.0",
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
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--snr", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    # ----- Load model -----
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SpectralNP(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
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

    if metrics["mean_diversity"] < 0.05 or metrics["rmse_min_to_max"] < 0.05:
        print()
        print("  >>> POSTERIOR COLLAPSE DETECTED <<<")
        print("  Model produces ~identical predictions regardless of band count.")
        print("  Decoder is ignoring the latent z. KL regularization too strong.")

    # Multi-spectrum eval: average convergence metrics over many random
    # spectra to confirm the visual single-spectrum result generalises.
    if args.n_eval_spectra > 0:
        rng2 = np.random.default_rng(args.seed + 1000)
        eval_spec_indices = rng2.choice(
            len(speclib), size=min(args.n_eval_spectra, len(speclib)),
            replace=False,
        )
        rmse_3, rmse_400, sharp_3, sharp_400 = [], [], [], []
        for ei in eval_spec_indices:
            spec_e = speclib.spectra[int(ei)]
            data_e = run_convergence(
                predictor, spec_e, dense_wl, atmos, geom,
                band_counts=BAND_COUNTS, n_samples=args.n_samples,
                snr=args.snr, seed=args.seed + int(ei),
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
        print(f"=== Averaged over {len(eval_spec_indices)} held-out spectra ===")
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
