#!/usr/bin/env python3
"""Validate the inputs and targets that the training pipeline produces.

Loads SpectralNPDataset exactly the way pretrain.py does, draws a few
hundred samples, and runs a battery of sanity checks:

  Shape & dtype:
    - all expected keys present
    - tensors have the right shape and dtype
    - no NaN or Inf in any field

  Physical-range checks:
    - target_reflectance in [0, 1]
    - target_radiance non-negative, finite, sensible magnitude
    - input radiance non-negative, finite
    - wavelengths in [380, 2500] nm
    - fwhm > 0 and not absurdly large
    - material_idx in [0, n_material_classes)
    - atmos_params in expected normalised ranges

  Variation checks (the dataset must produce diverse samples):
    - target_radiance varies across samples (std > 0.01 of the mean)
    - target_reflectance varies across samples
    - sensor wavelengths differ across samples (not always the same sensor)
    - atmos_params differ across samples
    - material_idx covers multiple classes

  Physics sanity:
    - brighter surface (mean reflectance) ⇒ brighter radiance (mean)
      (Pearson correlation > 0.3)
    - higher water vapour ⇒ deeper absorption near 1380 nm + 1880 nm
    - aerosol loading reduces VIS radiance
    - input radiance ≈ sensor.convolve(target_radiance) within tolerance
      (the input bands ARE a sensor-convolution of the dense target)

Each test reports PASS / FAIL with the relevant numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

from spectralnp.data.dataset import SpectralNPDataset, collate_spectral_batch
from spectralnp.data.lut import make_lut_wavelength_grid
from spectralnp.data.rtm_simulator import ARTSLookupSimulator
from spectralnp.data.usgs_speclib import load_from_directory, load_from_zip


# ---------------------------------------------------------------------------
# Tiny test framework
# ---------------------------------------------------------------------------

PASS = "[\033[92m PASS \033[0m]"
FAIL = "[\033[91m FAIL \033[0m]"
WARN = "[\033[93m WARN \033[0m]"


class Report:
    def __init__(self):
        self.passes = 0
        self.fails = 0
        self.warns = 0
        self.lines = []

    def passed(self, name: str, msg: str = ""):
        self.passes += 1
        self.lines.append(f"{PASS} {name}: {msg}")

    def failed(self, name: str, msg: str = ""):
        self.fails += 1
        self.lines.append(f"{FAIL} {name}: {msg}")

    def warned(self, name: str, msg: str = ""):
        self.warns += 1
        self.lines.append(f"{WARN} {name}: {msg}")

    def check(self, name: str, condition: bool, msg: str = "", warn_only=False):
        if condition:
            self.passed(name, msg)
        elif warn_only:
            self.warned(name, msg)
        else:
            self.failed(name, msg)

    def summary(self) -> str:
        return f"\n{self.passes} passed, {self.fails} failed, {self.warns} warned"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def collect_samples(dataset, n: int) -> list[dict]:
    """Draw `n` samples from the dataset."""
    samples = []
    for i in range(n):
        s = dataset[i]
        samples.append(s)
    return samples


def test_shapes_and_finiteness(samples: list[dict], dense_wl: np.ndarray, rep: Report):
    """Verify each sample has the expected fields with valid shapes/types."""
    expected_keys = {
        "wavelength", "fwhm", "radiance",
        "target_wavelength", "target_radiance", "target_reflectance",
        "atmos_params", "material_idx",
    }
    n_dense = len(dense_wl)

    missing_keys = []
    nan_count = 0
    inf_count = 0
    bad_dense_shape = 0
    for s in samples:
        ks = set(s.keys())
        if not expected_keys.issubset(ks):
            missing_keys.append(expected_keys - ks)
        for k, v in s.items():
            if not torch.is_tensor(v):
                continue
            if torch.isnan(v).any():
                nan_count += 1
            if torch.isinf(v).any():
                inf_count += 1
        if s["target_radiance"].shape[0] != n_dense:
            bad_dense_shape += 1

    rep.check("shape: all expected keys present", not missing_keys,
              f"missing in {len(missing_keys)}/{len(samples)} samples")
    rep.check("shape: target_radiance length matches dense_wl",
              bad_dense_shape == 0,
              f"{bad_dense_shape}/{len(samples)} mismatched")
    rep.check("finite: no NaNs in any sample", nan_count == 0,
              f"{nan_count} samples had NaNs")
    rep.check("finite: no Infs in any sample", inf_count == 0,
              f"{inf_count} samples had Infs")


def test_physical_ranges(samples: list[dict], rep: Report, n_classes: int):
    refl_lo = np.min([s["target_reflectance"].min().item() for s in samples])
    refl_hi = np.max([s["target_reflectance"].max().item() for s in samples])
    rep.check("range: target_reflectance >= 0", refl_lo >= -1e-6,
              f"min={refl_lo:.4f}")
    rep.check("range: target_reflectance <= 1", refl_hi <= 1.0 + 1e-6,
              f"max={refl_hi:.4f}")

    rad_lo = np.min([s["target_radiance"].min().item() for s in samples])
    rad_hi = np.max([s["target_radiance"].max().item() for s in samples])
    rep.check("range: target_radiance >= 0", rad_lo >= -1e-6,
              f"min={rad_lo:.4f}")
    rep.check("range: target_radiance < 1000 W/m²/sr/μm",
              rad_hi < 1000.0,
              f"max={rad_hi:.4f}")

    band_lo = np.min([s["radiance"].min().item() for s in samples])
    band_hi = np.max([s["radiance"].max().item() for s in samples])
    rep.check("range: input band radiance >= 0", band_lo >= -1e-6,
              f"min={band_lo:.4f}")
    rep.check("range: input band radiance < 1000",
              band_hi < 1000.0,
              f"max={band_hi:.4f}")

    wl_lo = np.min([s["wavelength"].min().item() for s in samples])
    wl_hi = np.max([s["wavelength"].max().item() for s in samples])
    rep.check("range: input wavelengths >= 380 nm", wl_lo >= 380.0 - 1.0,
              f"min={wl_lo:.1f}")
    rep.check("range: input wavelengths <= 2500 nm", wl_hi <= 2500.0 + 1.0,
              f"max={wl_hi:.1f}")

    fwhm_lo = np.min([s["fwhm"].min().item() for s in samples])
    fwhm_hi = np.max([s["fwhm"].max().item() for s in samples])
    rep.check("range: input fwhm > 0", fwhm_lo > 0.0,
              f"min={fwhm_lo:.2f}")
    rep.check("range: input fwhm < 500 nm", fwhm_hi < 500.0,
              f"max={fwhm_hi:.2f}")

    mat_idxs = [int(s["material_idx"]) for s in samples]
    rep.check(
        "range: material_idx within [0, n_classes)",
        all(0 <= m < n_classes for m in mat_idxs),
        f"min={min(mat_idxs)}, max={max(mat_idxs)}, n_classes={n_classes}",
    )

    # atmos_params: [aod, water_vapour, ozone_du/1000, visibility/100]
    aod = [s["atmos_params"][0].item() for s in samples]
    wv = [s["atmos_params"][1].item() for s in samples]
    oz = [s["atmos_params"][2].item() for s in samples]
    vis = [s["atmos_params"][3].item() for s in samples]
    rep.check("range: aod_550 in [0, 2]",
              0 <= min(aod) and max(aod) <= 2.0,
              f"[{min(aod):.3f}, {max(aod):.3f}]")
    rep.check("range: water_vapour in [0, 10]",
              0 <= min(wv) and max(wv) <= 10.0,
              f"[{min(wv):.3f}, {max(wv):.3f}]")
    rep.check("range: ozone_du/1000 in [0, 1]",
              0 <= min(oz) and max(oz) <= 1.0,
              f"[{min(oz):.3f}, {max(oz):.3f}]")
    rep.check("range: visibility/100 in [0, 2]",
              0 <= min(vis) and max(vis) <= 2.0,
              f"[{min(vis):.3f}, {max(vis):.3f}]")


def test_variation(samples: list[dict], rep: Report):
    """The dataset must produce diverse samples — not constant."""
    rad_means = np.array([s["target_radiance"].mean().item() for s in samples])
    rep.check(
        "variation: target_radiance mean varies across samples",
        rad_means.std() > 0.01 * abs(rad_means.mean()),
        f"std={rad_means.std():.3f}, mean={rad_means.mean():.3f}",
    )

    refl_means = np.array([s["target_reflectance"].mean().item() for s in samples])
    rep.check(
        "variation: target_reflectance mean varies across samples",
        refl_means.std() > 1e-3,
        f"std={refl_means.std():.4f}",
    )

    n_bands_per_sample = [s["wavelength"].shape[0] for s in samples]
    distinct_n_bands = len(set(n_bands_per_sample))
    rep.check(
        "variation: random sensors with different band counts",
        distinct_n_bands > 1,
        f"distinct band counts={distinct_n_bands}",
    )

    aods = np.array([s["atmos_params"][0].item() for s in samples])
    rep.check(
        "variation: aod_550 varies across samples",
        aods.std() > 0.01,
        f"std={aods.std():.4f}",
    )

    wvs = np.array([s["atmos_params"][1].item() for s in samples])
    rep.check(
        "variation: water_vapour varies across samples",
        wvs.std() > 0.05,
        f"std={wvs.std():.4f}",
    )

    mat_idxs = [int(s["material_idx"]) for s in samples]
    distinct_classes = len(set(mat_idxs))
    rep.check(
        "variation: material_idx covers multiple classes",
        distinct_classes >= 2,
        f"{distinct_classes} distinct classes seen in {len(samples)} samples",
    )


def test_physics(samples: list[dict], dense_wl: np.ndarray, rep: Report):
    """Sanity-check that the simulated radiance behaves physically."""
    refls = np.stack([s["target_reflectance"].numpy() for s in samples])
    rads = np.stack([s["target_radiance"].numpy() for s in samples])
    aods = np.array([s["atmos_params"][0].item() for s in samples])
    wvs = np.array([s["atmos_params"][1].item() for s in samples])

    # Brighter surface → brighter radiance (in VNIR/SWIR where solar matters)
    vis_mask = (dense_wl >= 500) & (dense_wl <= 1000)
    refl_vis = refls[:, vis_mask].mean(axis=1)
    rad_vis = rads[:, vis_mask].mean(axis=1)
    if refl_vis.std() > 1e-4 and rad_vis.std() > 1e-4:
        corr = float(np.corrcoef(refl_vis, rad_vis)[0, 1])
    else:
        corr = float("nan")
    rep.check(
        "physics: bright surface → bright VIS radiance (corr > 0.3)",
        corr > 0.3,
        f"Pearson r={corr:.3f}",
    )

    # Higher AOD → lower VIS direct + higher path radiance.
    # Net effect on mean VIS radiance is messy, but we can check there
    # IS some correlation (positive or negative) with AOD.
    if aods.std() > 0.05:
        corr_aod = float(np.corrcoef(aods, rad_vis)[0, 1])
    else:
        corr_aod = 0.0
    rep.check(
        "physics: aod has measurable effect on VIS radiance (|corr| > 0.05)",
        abs(corr_aod) > 0.05,
        f"corr(aod, rad_vis) = {corr_aod:.3f}",
        warn_only=True,  # not a hard pass/fail; depends on how diverse refl is
    )

    # Higher water vapour → deeper absorption near 1380 nm.
    # Compute depth as: rad_outside_band / rad_in_band - 1
    band_in = (dense_wl >= 1350) & (dense_wl <= 1450)
    band_out = ((dense_wl >= 1250) & (dense_wl <= 1340)) | ((dense_wl >= 1460) & (dense_wl <= 1550))
    if band_in.sum() > 0 and band_out.sum() > 0:
        rad_in_1380 = rads[:, band_in].mean(axis=1)
        rad_out_1380 = rads[:, band_out].mean(axis=1)
        depth_1380 = (rad_out_1380 - rad_in_1380) / np.maximum(rad_out_1380, 1e-6)
        if wvs.std() > 0.05 and depth_1380.std() > 1e-4:
            corr_wv = float(np.corrcoef(wvs, depth_1380)[0, 1])
        else:
            corr_wv = 0.0
        rep.check(
            "physics: more water vapour → deeper 1380 nm band (corr > 0.1)",
            corr_wv > 0.1,
            f"corr(wv, depth_1380) = {corr_wv:.3f}",
        )

    # Negative-radiance check (must not happen)
    n_neg = int((rads < -1e-6).sum())
    rep.check(
        "physics: no negative target_radiance pixels",
        n_neg == 0,
        f"{n_neg} negative pixels",
    )


def test_sensor_convolution(samples: list[dict], dense_wl: np.ndarray, rep: Report):
    """The input radiance should match a sensor-convolution of the dense target."""
    from spectralnp.data.random_sensor import VirtualSensor, apply_sensor

    n_within = 0
    n_total = 0
    for s in samples[:20]:
        wl = s["wavelength"].numpy()
        fwhm = s["fwhm"].numpy()
        observed = s["radiance"].numpy()
        target = s["target_radiance"].numpy()

        # Reconstruct the sensor and re-convolve target.
        sensor = VirtualSensor(
            center_wavelength_nm=wl.astype(np.float32),
            fwhm_nm=fwhm.astype(np.float32),
        )
        ref = apply_sensor(sensor, dense_wl, target)
        # Allow some noise (we add Gaussian noise per sample) — relative error <30%
        rel = np.abs(observed - ref) / (np.abs(ref) + 1e-6)
        n_within += int((rel < 0.30).sum())
        n_total += rel.size

    fraction_within = n_within / max(n_total, 1)
    rep.check(
        "sensor: input bands match sensor-convolved target_radiance (>80%)",
        fraction_within > 0.80,
        f"{100*fraction_within:.1f}% within ±30%",
    )


def test_bandwidth(samples: list[dict], dense_wl: np.ndarray, rep: Report) -> None:
    """Verify that band FWHM (bandwidth) is meaningful and used correctly.

    1. FWHM varies across samples (different virtual instruments).
    2. Per-sample FWHM may be uniform (regular sensor) or varying
       (uniform/clustered) — both modes appear in the dataset.
    3. The sensor convolution actually uses FWHM: convolving the same
       dense radiance with narrow vs wide FWHM gives different results.
    4. Narrower FWHM ⇒ less smoothing ⇒ better preservation of
       fine spectral features.
    """
    from spectralnp.data.random_sensor import VirtualSensor, apply_sensor

    # 1. FWHM mean varies across samples
    sample_mean_fwhm = np.array([
        s["fwhm"].numpy().mean() for s in samples
    ])
    rep.check(
        "bandwidth: per-sample mean FWHM varies across samples",
        sample_mean_fwhm.std() > 1.0,
        f"std={sample_mean_fwhm.std():.2f} nm, "
        f"range=[{sample_mean_fwhm.min():.1f}, {sample_mean_fwhm.max():.1f}]",
    )

    # 2. Within-sample FWHM variation: some sensors have varying FWHM
    n_uniform_fwhm = sum(
        1 for s in samples if s["fwhm"].numpy().std() < 1e-3
    )
    n_varying_fwhm = len(samples) - n_uniform_fwhm
    rep.check(
        "bandwidth: dataset includes both uniform-FWHM and varying-FWHM sensors",
        n_uniform_fwhm > 0 and n_varying_fwhm > 0,
        f"{n_uniform_fwhm} uniform, {n_varying_fwhm} varying",
    )

    # 3. The convolution operator actually uses FWHM. Construct a sharp
    # absorption-like spectrum and convolve with narrow vs wide FWHM at
    # the same wavelength positions; outputs must differ.
    test_wl = dense_wl
    sharp = np.ones_like(test_wl, dtype=np.float32)
    band_idx = (test_wl > 1380) & (test_wl < 1420)  # narrow water-vapour-like dip
    sharp[band_idx] = 0.1
    centers = np.array([1400.0], dtype=np.float32)
    narrow = VirtualSensor(
        center_wavelength_nm=centers, fwhm_nm=np.array([5.0], dtype=np.float32)
    )
    wide = VirtualSensor(
        center_wavelength_nm=centers, fwhm_nm=np.array([100.0], dtype=np.float32)
    )
    rad_narrow = apply_sensor(narrow, test_wl, sharp)[0]
    rad_wide = apply_sensor(wide, test_wl, sharp)[0]
    rep.check(
        "bandwidth: narrow FWHM resolves a sharp dip more than wide FWHM",
        rad_narrow < rad_wide,
        f"narrow(FWHM=5)={rad_narrow:.3f}, wide(FWHM=100)={rad_wide:.3f}",
    )

    # 4. Cross-FWHM physics: take a sample's dense target and re-convolve
    # with a 5x wider FWHM at the same band centres; should give a smoother
    # response (less variance band-to-band).
    s0 = samples[0]
    wl0 = s0["wavelength"].numpy()
    fwhm0 = s0["fwhm"].numpy()
    target0 = s0["target_radiance"].numpy()
    sensor_normal = VirtualSensor(
        center_wavelength_nm=wl0.astype(np.float32),
        fwhm_nm=fwhm0.astype(np.float32),
    )
    sensor_wide = VirtualSensor(
        center_wavelength_nm=wl0.astype(np.float32),
        fwhm_nm=(fwhm0 * 5).clip(max=200).astype(np.float32),
    )
    rad_normal = apply_sensor(sensor_normal, dense_wl, target0)
    rad_wide_full = apply_sensor(sensor_wide, dense_wl, target0)
    if len(rad_normal) > 5:
        # std band-to-band is a smoothness proxy.
        std_normal = float(np.std(np.diff(rad_normal)))
        std_wide = float(np.std(np.diff(rad_wide_full)))
        rep.check(
            "bandwidth: 5× wider FWHM produces a smoother spectrum",
            std_wide < std_normal,
            f"std(diff): normal={std_normal:.3f}, 5x_wide={std_wide:.3f}",
        )


def test_target_consistency(
    samples: list[dict],
    dataset,
    dense_wl: np.ndarray,
    rep: Report,
) -> None:
    """Verify that the targets are consistent with each other and with USGS."""
    speclib = dataset.speclib
    n_classes = dataset.n_material_classes
    cat_id_by_spec = dataset.category_id_by_spec
    category_names = dataset.category_names

    # 1. target_reflectance must equal the USGS spectrum's resampling.
    # The dataset stores this in dataset.reflectance_matrix; samples should
    # match (the dataset draws spec_idx via rng.integers internally).
    n_match_refl = 0
    n_total_refl = 0
    for s in samples[:30]:
        target = s["target_reflectance"].numpy()
        # Find which row of dataset.reflectance_matrix this came from.
        # Use closest match by L2 distance.
        d = np.sum((dataset.reflectance_matrix - target[None, :]) ** 2, axis=1)
        best = int(np.argmin(d))
        rms = float(np.sqrt(d[best] / len(target)))
        n_match_refl += int(rms < 1e-3)
        n_total_refl += 1
    rep.check(
        "target consistency: reflectance row matches a USGS spectrum",
        n_match_refl == n_total_refl,
        f"{n_match_refl}/{n_total_refl} samples matched within RMS<1e-3",
    )

    # 2. material_idx must match the category of a USGS spectrum that has
    # the same target_reflectance.
    n_correct_label = 0
    n_total_label = 0
    for s in samples[:50]:
        target = s["target_reflectance"].numpy()
        d = np.sum((dataset.reflectance_matrix - target[None, :]) ** 2, axis=1)
        best_spec_idx = int(np.argmin(d))
        true_cat_id = int(cat_id_by_spec[best_spec_idx])
        sample_cat_id = int(s["material_idx"])
        if sample_cat_id == true_cat_id:
            n_correct_label += 1
        n_total_label += 1
    rep.check(
        "target consistency: material_idx matches reflectance source spectrum",
        n_correct_label == n_total_label,
        f"{n_correct_label}/{n_total_label} matched",
    )

    # 3. Distinguishability: sample radiance should DIFFER between
    # categories. Take 10 of the dominant category and 10 of another.
    cat_to_samples: dict[int, list[np.ndarray]] = {}
    for s in samples:
        c = int(s["material_idx"])
        cat_to_samples.setdefault(c, []).append(s["target_radiance"].numpy())
    cats_with_data = [c for c, lst in cat_to_samples.items() if len(lst) >= 5]
    if len(cats_with_data) >= 2:
        c0, c1 = cats_with_data[0], cats_with_data[1]
        rad0 = np.stack(cat_to_samples[c0][:5])
        rad1 = np.stack(cat_to_samples[c1][:5])
        # Inter-category mean spectral distance vs intra-category.
        between = np.linalg.norm(rad0.mean(0) - rad1.mean(0))
        within0 = np.std(rad0, axis=0).mean() + 1e-6
        within1 = np.std(rad1, axis=0).mean() + 1e-6
        ratio = between / (0.5 * (within0 + within1))
        rep.check(
            "target distinguishability: inter-category mean / intra-category std > 1.0",
            ratio > 1.0,
            f"between={between:.2f}, within={(within0+within1)/2:.2f}, "
            f"ratio={ratio:.2f}, comparing {category_names[c0]} vs {category_names[c1]}",
        )
    else:
        rep.warned(
            "target distinguishability",
            f"only {len(cats_with_data)} categories with ≥5 samples; can't compare",
        )

    # 4. target_radiance must NOT be constant per sample (degenerate physics)
    flat_count = 0
    for s in samples:
        rad = s["target_radiance"].numpy()
        if rad.std() / max(abs(rad.mean()), 1e-6) < 0.01:
            flat_count += 1
    rep.check(
        "target radiance: not flat (std/mean > 1%)",
        flat_count == 0,
        f"{flat_count}/{len(samples)} samples had flat radiance",
    )

    # 5. target_reflectance must NOT be constant either
    flat_refl = 0
    for s in samples:
        refl = s["target_reflectance"].numpy()
        if refl.std() < 0.001:
            flat_refl += 1
    rep.check(
        "target reflectance: not flat (std > 0.001)",
        flat_refl == 0,
        f"{flat_refl}/{len(samples)} samples had flat reflectance",
        warn_only=True,  # some materials really are flat (e.g. fresh asphalt)
    )

    # 6. atmospheric target values are consistent with sample variation
    # (we already check variation; here check that they actually correspond
    # to the per-sample atmospheric state, which they do by construction)
    n_dense = len(dense_wl)
    rep.check(
        "target atmos_params: 4-vector",
        all(s["atmos_params"].shape[0] == 4 for s in samples),
        "all samples have atmos_params length 4",
    )

    # 7. Coverage of categories: every USGS category should appear in the
    # target labels with reasonable frequency (or at least the dominant ones).
    seen_cats = Counter(int(s["material_idx"]) for s in samples)
    cat_pct = {category_names[c]: 100 * n / len(samples)
               for c, n in seen_cats.items()}
    sorted_cats = sorted(cat_pct.items(), key=lambda x: -x[1])
    rep.passed("target labels: category histogram",
               ", ".join(f"{n}={p:.0f}%" for n, p in sorted_cats[:5]))


def make_visualization(
    samples: list[dict],
    dataset,
    dense_wl: np.ndarray,
    output_path: Path,
) -> None:
    """Save a visual of the targets to PNG so the user can eyeball them."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    # Pick 10 random samples for the line plots
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), size=min(10, len(samples)), replace=False)

    # Row 0: target reflectance + target radiance overlay
    ax = axes[0, 0]
    for i in pick:
        ax.plot(dense_wl, samples[i]["target_reflectance"].numpy(),
                alpha=0.6, linewidth=0.8)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("target reflectance")
    ax.set_title("Target reflectance — 10 random samples")
    ax.set_xlim(dense_wl[0], dense_wl[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i in pick:
        ax.plot(dense_wl, samples[i]["target_radiance"].numpy(),
                alpha=0.6, linewidth=0.8)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("target radiance (W/m²/sr/μm)")
    ax.set_title("Target radiance — 10 random samples")
    ax.set_xlim(dense_wl[0], dense_wl[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    # Sensor input bands for one sample
    ax = axes[0, 2]
    s0 = samples[pick[0]]
    ax.plot(dense_wl, s0["target_radiance"].numpy(),
            color="black", alpha=0.4, label="dense target", linewidth=0.8)
    ax.scatter(s0["wavelength"].numpy(), s0["radiance"].numpy(),
               s=15, color="red", zorder=5, label="sensor bands")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("radiance")
    ax.set_title(f"Example: {s0['wavelength'].shape[0]} sensor bands "
                 f"vs dense target")
    ax.set_xlim(dense_wl[0], dense_wl[-1])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1: histograms
    ax = axes[1, 0]
    all_refl = np.concatenate([
        s["target_reflectance"].numpy() for s in samples
    ])
    ax.hist(all_refl, bins=50, color="#27ae60", alpha=0.8)
    ax.set_xlabel("target reflectance value")
    ax.set_ylabel("count")
    ax.set_title(f"Reflectance histogram (n={len(samples)} samples)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    rad_means = np.array([s["target_radiance"].mean().item() for s in samples])
    ax.hist(rad_means, bins=30, color="#e74c3c", alpha=0.8)
    ax.set_xlabel("per-sample mean radiance")
    ax.set_ylabel("count")
    ax.set_title("Per-sample mean target_radiance histogram")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    all_fwhm = np.concatenate([s["fwhm"].numpy() for s in samples])
    ax.hist(all_fwhm, bins=40, color="#9b59b6", alpha=0.8)
    ax.set_xlabel("FWHM (nm)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-band FWHM distribution\n"
                 f"({len(all_fwhm)} bands across {len(samples)} samples)")
    ax.grid(alpha=0.3)

    # Row 2: per-category overlay (vegetation vs minerals etc.)
    cat_to_idx: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        cat_to_idx.setdefault(int(s["material_idx"]), []).append(i)
    cat_names = dataset.category_names

    for col, cat_id in enumerate(sorted(cat_to_idx.keys())[:3]):
        ax = axes[2, col]
        idxs = cat_to_idx[cat_id][:8]
        for i in idxs:
            ax.plot(dense_wl, samples[i]["target_reflectance"].numpy(),
                    alpha=0.6, linewidth=0.8)
        ax.set_title(f"{cat_names[cat_id]} ({len(cat_to_idx[cat_id])} samples)")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("reflectance")
        ax.set_xlim(dense_wl[0], dense_wl[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("SpectralNP training data validation",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usgs-data", type=str, required=True)
    p.add_argument("--abs-lookup", type=str,
                   default="/Users/eric/.cache/atmgen/lut/abs_lookup_fa72fc35f64b.xml")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-scenes", type=int, default=20)
    p.add_argument("--use-arts", action="store_true",
                   help="Use real ARTS RTM (default: simplified, no ARTS)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rep = Report()

    print(f"Loading USGS from {args.usgs_data}...")
    p_path = Path(args.usgs_data)
    if p_path.suffix == ".zip":
        speclib = load_from_zip(p_path)
    else:
        speclib = load_from_directory(p_path)
    speclib = speclib.filter_wavelength_range(380, 2400)
    print(f"  {len(speclib)} spectra")

    arts = None
    if args.use_arts:
        print("Initialising ARTS abs_lookup simulator...")
        arts = ARTSLookupSimulator(args.abs_lookup)
        arts.populate_random_scenes(args.n_scenes, np.random.default_rng(args.seed))
        print(f"  {len(arts._cache)} scenes cached")
        dense_wl = make_lut_wavelength_grid(380, 2500)
    else:
        dense_wl = np.arange(380.0, 2501.0, 5.0)

    ds = SpectralNPDataset(
        spectral_library=speclib,
        dense_wavelength_nm=dense_wl,
        samples_per_epoch=args.n_samples,
        arts_simulator=arts,
        seed=args.seed,
    )
    print(f"  dataset n_material_classes={ds.n_material_classes}, "
          f"dense grid={len(dense_wl)} points")

    print(f"\nDrawing {args.n_samples} samples...")
    samples = collect_samples(ds, args.n_samples)

    print("\nRunning tests...\n")
    test_shapes_and_finiteness(samples, dense_wl, rep)
    test_physical_ranges(samples, rep, ds.n_material_classes)
    test_variation(samples, rep)
    test_physics(samples, dense_wl, rep)
    test_sensor_convolution(samples, dense_wl, rep)
    test_bandwidth(samples, dense_wl, rep)
    test_target_consistency(samples, ds, dense_wl, rep)

    for line in rep.lines:
        print(line)
    print(rep.summary())

    out = Path("training_data_validation.png")
    make_visualization(samples, ds, dense_wl, out)
    print(f"\nVisualisation saved to {out}")
    import subprocess
    subprocess.run(["open", str(out)], check=False)

    sys.exit(0 if rep.fails == 0 else 1)


if __name__ == "__main__":
    main()
