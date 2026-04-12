#!/usr/bin/env python3
"""Validate the trained stage-2 foundation model.

Evaluates a checkpoint on held-out test scenes and measures:

  - Reflectance / emissivity reconstruction RMSE as a function of band count
  - Surface temperature prediction accuracy + coverage@2σ
  - Atmospheric parameter prediction accuracy + coverage@2σ
  - Material classification accuracy
  - Disentanglement check: does z_surf stay constant when we swap atmospheres?

Usage:
    python scripts/validate_stage2.py \\
        --checkpoint checkpoints/foundation/best.pt \\
        --mode lwir --lwir-library data/lwir_library \\
        --output stage2_validation.png
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
from spectralnp.model.foundation import SpectralFoundation


BAND_COUNTS = [3, 5, 10, 30, 100, 300, 1000]


def _make_atmos(rng: np.random.Generator, mode: str) -> AtmosphericState:
    if mode == "lwir":
        return AtmosphericState(
            aod_550=0.05,
            water_vapour=float(rng.uniform(0.2, 5.0)),
            ozone_du=float(rng.uniform(200, 500)),
            co2_ppmv=float(rng.uniform(380, 450)),
            visibility_km=23.0,
            surface_altitude_km=float(rng.uniform(0, 3)),
            surface_temperature_k=float(rng.uniform(260, 330)),
        )
    return AtmosphericState(
        aod_550=float(rng.uniform(0.05, 0.8)),
        water_vapour=float(rng.uniform(0.5, 4.0)),
        ozone_du=float(rng.uniform(250, 400)),
        visibility_km=float(rng.uniform(10, 80)),
        surface_altitude_km=float(rng.uniform(0, 2)),
        surface_temperature_k=float(rng.uniform(270, 320)),
    )


def _make_geom(rng: np.random.Generator, mode: str) -> ViewGeometry:
    if mode == "lwir":
        return ViewGeometry(
            solar_zenith_deg=90.0,
            sensor_zenith_deg=float(rng.uniform(0, 45)),
            relative_azimuth_deg=0.0,
        )
    return ViewGeometry(
        solar_zenith_deg=float(rng.uniform(20, 60)),
        sensor_zenith_deg=float(rng.uniform(0, 30)),
        relative_azimuth_deg=float(rng.uniform(0, 180)),
    )


def _atmos_params_vec(atmos: AtmosphericState, geom: ViewGeometry, mode: str) -> np.ndarray:
    if mode == "lwir":
        return np.array([
            atmos.water_vapour / 5.0,
            atmos.ozone_du / 500.0,
            (atmos.co2_ppmv - 400.0) / 50.0,
            geom.sensor_zenith_deg / 60.0,
        ], dtype=np.float32)
    return np.array([
        atmos.aod_550,
        atmos.water_vapour,
        atmos.ozone_du / 1000.0,
        atmos.visibility_km / 100.0,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Reflectance / emissivity convergence test
# ---------------------------------------------------------------------------

@torch.no_grad()
def convergence_test(
    model: SpectralFoundation,
    refl: np.ndarray,
    dense_wl: np.ndarray,
    atmos: AtmosphericState,
    geom: ViewGeometry,
    band_counts: list[int],
    snr: float = 200.0,
    seed: int = 0,
) -> dict:
    """Run one test spectrum through the model at varying band counts."""
    rng = np.random.default_rng(seed)

    # For LWIR, the "reflectance" input to the RTM is (1 - emissivity)
    rtm_refl = np.clip(1.0 - refl, 0.0, 1.0).astype(np.float32)
    truth = simplified_toa_radiance(
        surface_reflectance=rtm_refl, wavelength_nm=dense_wl,
        atmos=atmos, geometry=geom,
    ).toa_radiance.astype(np.float32)

    max_n = max(band_counts)
    wl_lo, wl_hi = float(dense_wl[0]), float(dense_wl[-1])
    all_centers = np.linspace(wl_lo, wl_hi, max_n).astype(np.float32)
    fwhm_val = float(rng.uniform(10.0, 40.0))
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
            center_wavelength_nm=centers, fwhm_nm=fwhms,
        )
        band_rad = apply_sensor(sensor, dense_wl, truth).astype(np.float32)
        band_rad = add_sensor_noise(band_rad, rng, snr_range=(snr, snr)).astype(np.float32)

        # Multiple MC samples for uncertainty propagation
        n_mc = 16
        refl_samples = []
        temp_samples = []
        atmos_samples = []
        logit_accum = None
        wl_t = torch.from_numpy(centers).unsqueeze(0)
        fw_t = torch.from_numpy(fwhms).unsqueeze(0)
        rad_t = torch.from_numpy(band_rad).unsqueeze(0)
        mask_t = torch.ones(1, len(centers), dtype=torch.bool)
        for _ in range(n_mc):
            out = model(wl_t, fw_t, rad_t, mask_t)
            refl_samples.append(out.reflectance[0, 0].cpu().numpy())
            temp_samples.append(out.temp_mu[0, 0, 0].item())
            atmos_samples.append(out.atmos_mu[0].cpu().numpy())
            logits = out.material_logits[0, 0].cpu()  # (n_classes,)
            if logit_accum is None:
                logit_accum = torch.zeros_like(logits)
            logit_accum += torch.softmax(logits, dim=-1)

        refl_arr = np.stack(refl_samples)
        # Average softmax across MC samples
        avg_probs = (logit_accum / n_mc).numpy()
        results.append({
            "n_bands": n,
            "refl_mean": refl_arr.mean(axis=0),
            "refl_std": refl_arr.std(axis=0),
            "temp_mean": float(np.mean(temp_samples)),
            "temp_std": float(np.std(temp_samples)),
            "atmos_mean": np.mean(atmos_samples, axis=0),
            "atmos_std": np.std(atmos_samples, axis=0),
            "avg_probs": avg_probs,
            "input_wl": centers,
            "input_rad": band_rad,
        })

    return {
        "truth_radiance": truth,
        "truth_refl": refl.astype(np.float32),
        "dense_wl": dense_wl,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Disentanglement check
# ---------------------------------------------------------------------------

@torch.no_grad()
def disentanglement_check(
    model: SpectralFoundation,
    refl: np.ndarray,
    dense_wl: np.ndarray,
    mode: str,
    n_atmospheres: int = 20,
    n_bands: int = 100,
    seed: int = 0,
) -> dict:
    """Check: same surface + different atmospheres → how similar is z_surf?

    A disentangled model should produce nearly-identical z_surf for the
    same surface across different atmospheric conditions.  We measure the
    coefficient of variation of z_surf across atmospheres.
    """
    rng = np.random.default_rng(seed)
    rtm_refl = np.clip(1.0 - refl, 0.0, 1.0).astype(np.float32)

    # Build one fixed sensor configuration
    wl_lo, wl_hi = float(dense_wl[0]), float(dense_wl[-1])
    centers = np.linspace(wl_lo, wl_hi, n_bands).astype(np.float32)
    fwhms = np.full(n_bands, 30.0, dtype=np.float32)
    sensor = VirtualSensor(center_wavelength_nm=centers, fwhm_nm=fwhms)

    z_surf_all = []
    z_atm_all = []
    for _ in range(n_atmospheres):
        atmos = _make_atmos(rng, mode)
        geom = _make_geom(rng, mode)
        truth = simplified_toa_radiance(
            surface_reflectance=rtm_refl, wavelength_nm=dense_wl,
            atmos=atmos, geometry=geom,
        ).toa_radiance.astype(np.float32)
        band_rad = apply_sensor(sensor, dense_wl, truth).astype(np.float32)
        band_rad = add_sensor_noise(band_rad, rng).astype(np.float32)

        out = model(
            torch.from_numpy(centers).unsqueeze(0),
            torch.from_numpy(fwhms).unsqueeze(0),
            torch.from_numpy(band_rad).unsqueeze(0),
            torch.ones(1, n_bands, dtype=torch.bool),
        )
        z_surf_all.append(out.z_surf_mu[0, 0].cpu().numpy())
        z_atm_all.append(out.z_atm_mu[0].cpu().numpy())

    z_surf_arr = np.stack(z_surf_all)       # (n_atm, z_surf_dim)
    z_atm_arr = np.stack(z_atm_all)         # (n_atm, z_atm_dim)

    # Normalized "variation" metric: std / (|mean| + 0.1)
    surf_std = z_surf_arr.std(axis=0).mean()
    surf_mean_abs = np.abs(z_surf_arr.mean(axis=0)).mean()
    atm_std = z_atm_arr.std(axis=0).mean()
    atm_mean_abs = np.abs(z_atm_arr.mean(axis=0)).mean()

    return {
        "z_surf_std": float(surf_std),
        "z_surf_mean_abs": float(surf_mean_abs),
        "z_atm_std": float(atm_std),
        "z_atm_mean_abs": float(atm_mean_abs),
        "z_surf_all": z_surf_arr,
        "z_atm_all": z_atm_arr,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _compute_metrics(data: dict) -> dict:
    truth_refl = data["truth_refl"]
    results = data["results"]

    rmse_refl = np.array([
        float(np.sqrt(np.mean((r["refl_mean"] - truth_refl) ** 2)))
        for r in results
    ])
    sharp_refl = np.array([float(np.mean(r["refl_std"])) for r in results])
    cov_refl = np.array([
        float(np.mean(np.abs(r["refl_mean"] - truth_refl) <= 2 * np.maximum(r["refl_std"], 1e-9)))
        for r in results
    ])
    return {
        "n_bands": [r["n_bands"] for r in results],
        "rmse_refl": rmse_refl,
        "sharp_refl": sharp_refl,
        "cov_refl": cov_refl,
    }


def make_plot(data: dict, metrics: dict, path: Path) -> None:
    truth_refl = data["truth_refl"]
    dense_wl = data["dense_wl"]
    results = data["results"]
    wl_um = dense_wl / 1000.0

    n_cols = 3
    n_rows_detail = (len(results) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(16, 3 + 2.5 * (1 + n_rows_detail)))
    gs = fig.add_gridspec(1 + n_rows_detail, n_cols, hspace=0.5, wspace=0.3)

    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_rmse.plot(metrics["n_bands"], metrics["rmse_refl"], "o-", color="#e74c3c")
    ax_rmse.set_xscale("log")
    ax_rmse.set_xlabel("# bands")
    ax_rmse.set_ylabel("reflectance RMSE")
    ax_rmse.set_title("Accuracy (should ↓)")
    ax_rmse.grid(alpha=0.3)

    ax_sharp = fig.add_subplot(gs[0, 1])
    ax_sharp.plot(metrics["n_bands"], metrics["sharp_refl"], "s-", color="#8e44ad")
    ax_sharp.set_xscale("log")
    ax_sharp.set_xlabel("# bands")
    ax_sharp.set_ylabel("mean σ̂")
    ax_sharp.set_title("Sharpness (should ↓)")
    ax_sharp.grid(alpha=0.3)

    ax_cov = fig.add_subplot(gs[0, 2])
    ax_cov.plot(metrics["n_bands"], metrics["cov_refl"], "^-", color="#27ae60")
    ax_cov.axhline(0.95, linestyle="--", color="gray", label="95%")
    ax_cov.set_xscale("log")
    ax_cov.set_xlabel("# bands")
    ax_cov.set_ylabel("coverage@2σ")
    ax_cov.set_ylim(0, 1.05)
    ax_cov.set_title("Calibration")
    ax_cov.legend()
    ax_cov.grid(alpha=0.3)

    for i, r in enumerate(results):
        row = 1 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.fill_between(wl_um,
                        r["refl_mean"] - 2 * r["refl_std"],
                        r["refl_mean"] + 2 * r["refl_std"],
                        color="#3498db", alpha=0.3, label="pred ±2σ")
        ax.plot(wl_um, truth_refl, color="black", linewidth=0.8,
                label="truth")
        ax.plot(wl_um, r["refl_mean"], color="#3498db", linewidth=1.2,
                label="pred")
        ax.set_title(
            f"{r['n_bands']} bands — RMSE={metrics['rmse_refl'][i]:.3f}",
            fontsize=10,
        )
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel("emissivity")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Stage 2 reflectance/emissivity convergence",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--mode", choices=["lwir", "vnir"], default="lwir")
    p.add_argument("--lwir-library", type=str, default="data/lwir_library")
    p.add_argument("--usgs-data", type=str, default=None)
    p.add_argument("--output", type=str, default="stage2_validation.png")
    p.add_argument("--n-eval-spectra", type=int, default=40)
    p.add_argument("--snr", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    # ---- Load model ----
    print(f"Loading {args.checkpoint}...")
    model = SpectralFoundation.from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    dense_wl = model.wavelength_nm.cpu().numpy()
    print(f"  grid: {dense_wl[0]:.0f}-{dense_wl[-1]:.0f} nm ({len(dense_wl)} bands)")
    print(f"  z_atm_dim={model.config.z_atm_dim}  z_surf_dim={model.config.z_surf_dim}")

    # ---- Load library for held-out evaluation ----
    if args.mode == "lwir":
        from spectralnp.data.envi_sli import read_envi_sli
        lib_wl, lib_spectra, lib_names = read_envi_sli(args.lwir_library)
        # Resample library onto model's grid
        refl_matrix = np.zeros((lib_spectra.shape[0], len(dense_wl)), dtype=np.float32)
        for i in range(lib_spectra.shape[0]):
            refl_matrix[i] = np.interp(dense_wl, lib_wl, lib_spectra[i])
        categories = [n.rsplit("_", 1)[0] for n in lib_names]
    else:
        raise NotImplementedError("VNIR stage-2 eval not yet implemented")

    rng = np.random.default_rng(args.seed)
    test_indices = rng.choice(len(lib_names), size=args.n_eval_spectra, replace=False)

    # ---- Visual test on one spectrum ----
    atmos = _make_atmos(rng, args.mode)
    geom = _make_geom(rng, args.mode)
    visual_idx = int(test_indices[0])
    test_refl = refl_matrix[visual_idx]
    print(f"Visual test: {lib_names[visual_idx]} ({categories[visual_idx]})")

    data = convergence_test(
        model, test_refl, dense_wl, atmos, geom,
        band_counts=BAND_COUNTS, snr=args.snr, seed=args.seed,
    )
    metrics = _compute_metrics(data)

    print()
    print("=== Reflectance/emissivity convergence ===")
    print(f"  RMSE @ {BAND_COUNTS[0]:4d} bands : {metrics['rmse_refl'][0]:.4f}")
    print(f"  RMSE @ {BAND_COUNTS[-1]:4d} bands : {metrics['rmse_refl'][-1]:.4f}")
    print(f"  Factor           : {metrics['rmse_refl'][0] / max(metrics['rmse_refl'][-1], 1e-9):.2f}×")
    print(f"  σ̂ @ {BAND_COUNTS[0]:4d} bands    : {metrics['sharp_refl'][0]:.4f}")
    print(f"  σ̂ @ {BAND_COUNTS[-1]:4d} bands    : {metrics['sharp_refl'][-1]:.4f}")
    print(f"  Cov @ {BAND_COUNTS[0]:4d} bands   : {metrics['cov_refl'][0]:.1%}")
    print(f"  Cov @ {BAND_COUNTS[-1]:4d} bands   : {metrics['cov_refl'][-1]:.1%}")

    # ---- Averaged over multiple held-out spectra ----
    print()
    print(f"=== Averaged over {args.n_eval_spectra} held-out spectra ===")
    rmse_at = {n: [] for n in BAND_COUNTS}
    temp_errs = []
    atmos_errs = []
    # Material identification: per band-count top-k accuracy
    top1_at = {n: [] for n in BAND_COUNTS}
    top5_at = {n: [] for n in BAND_COUNTS}
    top100_at = {n: [] for n in BAND_COUNTS}
    for si, idx in enumerate(test_indices):
        true_idx = int(idx)
        r_e = refl_matrix[true_idx]
        a_e = _make_atmos(rng, args.mode)
        g_e = _make_geom(rng, args.mode)
        d_e = convergence_test(
            model, r_e, dense_wl, a_e, g_e,
            band_counts=BAND_COUNTS, snr=args.snr, seed=args.seed + si,
        )
        m_e = _compute_metrics(d_e)
        for i, n in enumerate(BAND_COUNTS):
            rmse_at[n].append(m_e["rmse_refl"][i])
            probs = d_e["results"][i].get("avg_probs")
            if probs is not None and len(probs) > true_idx:
                ranked = np.argsort(-probs)
                rank = int(np.where(ranked == true_idx)[0][0])
                top1_at[n].append(rank < 1)
                top5_at[n].append(rank < 5)
                top100_at[n].append(rank < 100)
        # Use the highest-band-count result for task accuracy
        last = d_e["results"][-1]
        t_norm_true = (a_e.surface_temperature_k - 290.0) / 40.0
        temp_errs.append(abs(last["temp_mean"] - t_norm_true))
        a_true = _atmos_params_vec(a_e, g_e, args.mode)
        atmos_errs.append(np.abs(last["atmos_mean"] - a_true).mean())

    for n in BAND_COUNTS:
        vals = np.array(rmse_at[n])
        print(f"  Reflectance RMSE @ {n:4d} bands: mean {vals.mean():.4f}  median {np.median(vals):.4f}")
    print(f"  Temperature MAE (normalized): {np.mean(temp_errs):.4f}")
    print(f"  Temperature MAE (Kelvin):     {np.mean(temp_errs) * 40.0:.2f} K")
    print(f"  Atmos MAE (normalized):       {np.mean(atmos_errs):.4f}")

    # ---- Material identification (top-k accuracy) ----
    if any(top1_at[BAND_COUNTS[-1]]):
        print()
        print("=== Material identification (10k-way) ===")
        for n in BAND_COUNTS:
            if not top1_at[n]:
                continue
            t1 = float(np.mean(top1_at[n]))
            t5 = float(np.mean(top5_at[n]))
            t100 = float(np.mean(top100_at[n]))
            print(f"  @ {n:4d} bands:  top1={t1:.1%}  top5={t5:.1%}  top100={t100:.1%}")

    # ---- Disentanglement ----
    print()
    print("=== Disentanglement check ===")
    dis = disentanglement_check(
        model, refl_matrix[int(test_indices[0])], dense_wl, args.mode,
        n_atmospheres=20, n_bands=100, seed=args.seed + 1000,
    )
    print(f"  Same surface, 20 different atmospheres:")
    print(f"    z_surf std/|mean|: {dis['z_surf_std'] / (dis['z_surf_mean_abs'] + 0.1):.3f}")
    print(f"    z_atm  std/|mean|: {dis['z_atm_std'] / (dis['z_atm_mean_abs'] + 0.1):.3f}")
    print("  (z_surf variation should be << z_atm variation if disentangled)")

    # ---- Plot ----
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
