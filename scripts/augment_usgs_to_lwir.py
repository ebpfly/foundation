#!/usr/bin/env python3
"""Build a 10k LWIR spectral library from augmented USGS spectra.

Takes the ~1748 real USGS spectra, remaps their wavelengths to the
7-16 μm LWIR range, and augments to 10k unique spectra using
physically-agnostic transforms that preserve feature diversity:

  - identity (original, relabeled)
  - flip (reversed in wavelength)
  - stretch/squeeze (warp the wavelength axis by a random factor)
  - shift + scale (random baseline offset and amplitude scaling)
  - mix (linear blend with another random spectrum)
  - smooth warp (non-linear wavelength distortion via random spline)

The result has real spectral complexity (absorption features, slopes,
fine structure) that parametric generators can't match — just with
wavelength labels that happen to say 7–16 μm instead of 0.38–2.5 μm.

Usage:
    python scripts/augment_usgs_to_lwir.py \
        --usgs-data /path/to/ASCIIdata_splib07a \
        --output data/lwir_library \
        --n-total 10000 --n-bands 4000
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectralnp.data.envi_sli import write_envi_sli
from spectralnp.data.usgs_speclib import load_from_directory, load_from_zip


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

def _resample(spectrum: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a 1-D spectrum to n_out points (linear interp)."""
    x_in = np.linspace(0, 1, len(spectrum))
    x_out = np.linspace(0, 1, n_out)
    return np.interp(x_out, x_in, spectrum).astype(np.float32)


def aug_identity(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return spec.copy()


def aug_flip(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reverse the spectrum in wavelength."""
    return spec[::-1].copy()


def aug_stretch(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stretch or squeeze the wavelength axis by a random factor.

    Selects a random sub-interval of the spectrum and resamples it
    to the full length, effectively "zooming in" on a portion.
    """
    n = len(spec)
    # Random crop: keep 50-100% of the spectrum, then resample to full
    frac = rng.uniform(0.5, 1.0)
    crop_len = max(int(n * frac), 2)
    start = rng.integers(0, n - crop_len + 1)
    cropped = spec[start : start + crop_len]
    return _resample(cropped, n)


def aug_squeeze(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Squeeze features: embed the spectrum in a wider baseline, leaving
    flat regions on either side."""
    n = len(spec)
    frac = rng.uniform(0.5, 0.85)
    target_len = max(int(n * frac), 2)
    squeezed = _resample(spec, target_len)
    # Embed in a flat baseline (value = edge value)
    out = np.full(n, float(spec[0]), dtype=np.float32)
    start = rng.integers(0, n - target_len + 1)
    out[start : start + target_len] = squeezed
    # Blend edges smoothly
    blend = min(20, target_len // 4)
    if blend > 1:
        t = np.linspace(0, 1, blend).astype(np.float32)
        out[start : start + blend] = (1 - t) * out[start] + t * squeezed[:blend]
        out[start + target_len - blend : start + target_len] = (
            t * out[start + target_len - 1]
            + (1 - t) * squeezed[-blend:]
        )
    return out


def aug_shift_scale(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random baseline shift + amplitude scaling."""
    scale = rng.uniform(0.6, 1.4)
    offset = rng.uniform(-0.1, 0.1)
    # Also add a slight tilt
    tilt = rng.uniform(-0.05, 0.05)
    x = np.linspace(-0.5, 0.5, len(spec))
    return np.clip(spec * scale + offset + tilt * x, 0.0, 1.0).astype(np.float32)


def aug_mix(
    spec: np.ndarray,
    rng: np.random.Generator,
    all_spectra: np.ndarray,
) -> np.ndarray:
    """Linear blend with a random other spectrum."""
    other_idx = rng.integers(0, len(all_spectra))
    other = all_spectra[other_idx]
    alpha = rng.uniform(0.3, 0.7)
    return (alpha * spec + (1 - alpha) * other).astype(np.float32)


def aug_smooth_warp(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Non-linear wavelength distortion using random control points.

    Generates a smooth monotonic warping function that shifts features
    non-uniformly — simulating how different materials have similar
    features at slightly different positions.
    """
    n = len(spec)
    # Random control points for the warping function
    n_ctrl = rng.integers(3, 7)
    ctrl_x = np.sort(rng.uniform(0, 1, size=n_ctrl))
    ctrl_x = np.concatenate([[0.0], ctrl_x, [1.0]])
    # Perturbed y-values (must be monotonically increasing)
    ctrl_y = ctrl_x + rng.uniform(-0.08, 0.08, size=len(ctrl_x))
    ctrl_y = np.sort(ctrl_y)  # enforce monotonicity
    ctrl_y = (ctrl_y - ctrl_y[0]) / (ctrl_y[-1] - ctrl_y[0])  # renorm to [0, 1]

    # Interpolate warping function
    x_in = np.linspace(0, 1, n)
    x_warped = np.interp(x_in, ctrl_x, ctrl_y)
    # Resample spectrum at warped positions
    return np.interp(x_warped, x_in, spec).astype(np.float32)


# All transforms (excluding mix, which needs the full array)
TRANSFORMS = [
    ("identity", aug_identity),
    ("flip", aug_flip),
    ("stretch", aug_stretch),
    ("squeeze", aug_squeeze),
    ("shift_scale", aug_shift_scale),
    ("smooth_warp", aug_smooth_warp),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build LWIR library from augmented USGS spectra",
    )
    p.add_argument("--usgs-data", type=str, required=True,
                   help="Path to USGS ASCIIdata_splib07a (or .zip)")
    p.add_argument("--output", type=str, default="data/lwir_library")
    p.add_argument("--n-total", type=int, default=10_000)
    p.add_argument("--n-bands", type=int, default=4_000)
    p.add_argument("--wl-lo", type=float, default=7.0, help="LWIR lower [μm]")
    p.add_argument("--wl-hi", type=float, default=16.0, help="LWIR upper [μm]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- Load USGS ----
    p_path = Path(args.usgs_data)
    if p_path.suffix == ".zip":
        speclib = load_from_zip(p_path)
    else:
        speclib = load_from_directory(p_path)
    speclib = speclib.filter_wavelength_range(380, 2500)
    print(f"Loaded {len(speclib)} USGS spectra")

    # Resample all spectra to n_bands points on the USGS native grid,
    # then relabel the wavelength axis to LWIR.
    usgs_wl = np.linspace(380.0, 2500.0, args.n_bands)
    base_matrix = speclib.to_array(usgs_wl)  # (N_usgs, n_bands)
    base_matrix = np.nan_to_num(base_matrix, nan=0.04).clip(0.0, 1.0).astype(np.float32)
    base_names = [s.name for s in speclib.spectra]
    base_categories = [s.category for s in speclib.spectra]
    n_base = base_matrix.shape[0]
    print(f"Base spectra: {n_base}, resampled to {args.n_bands} bands")

    # ---- LWIR wavelength grid (just relabeled) ----
    lwir_wl = np.linspace(
        args.wl_lo * 1000.0, args.wl_hi * 1000.0, args.n_bands,
    ).astype(np.float32)

    # ---- Augment to n_total ----
    # Strategy: cycle through transforms, each applied to each base spectrum,
    # until we hit n_total.
    out_spectra = []
    out_names = []
    out_categories = []

    # First pass: identity (all originals)
    for i in range(n_base):
        out_spectra.append(base_matrix[i].copy())
        out_names.append(f"{base_categories[i]}_{len(out_names):06d}")
        out_categories.append(base_categories[i])

    n_remaining = args.n_total - len(out_spectra)
    print(f"Need {n_remaining} more augmented spectra...")

    # Shuffle and cycle through transforms
    transform_names = []
    while len(out_spectra) < args.n_total:
        # Pick a random base spectrum
        base_idx = rng.integers(0, n_base)
        base_spec = base_matrix[base_idx]
        base_cat = base_categories[base_idx]

        # Pick a random transform (or mix)
        if rng.random() < 0.15:
            # Mix with another spectrum
            aug = aug_mix(base_spec, rng, base_matrix)
            tname = "mix"
        else:
            tname, tfn = TRANSFORMS[rng.integers(1, len(TRANSFORMS))]  # skip identity
            aug = tfn(base_spec, rng)

        aug = np.clip(aug, 0.0, 1.0).astype(np.float32)
        out_spectra.append(aug)
        out_names.append(f"{base_cat}_{len(out_spectra) - 1:06d}")
        out_categories.append(base_cat)
        transform_names.append(tname)

    out_matrix = np.stack(out_spectra[:args.n_total])
    out_names = out_names[:args.n_total]
    out_categories = out_categories[:args.n_total]

    print(f"\nFinal library: {out_matrix.shape[0]} spectra × {out_matrix.shape[1]} bands")
    print(f"  range: [{out_matrix.min():.3f}, {out_matrix.max():.3f}]")
    print(f"  mean:  {out_matrix.mean():.3f}")

    # Transform distribution
    tcounts = Counter(transform_names)
    print(f"  originals: {n_base}")
    for t, c in sorted(tcounts.items()):
        print(f"  {t:12s}: {c}")

    # Category distribution
    cat_counts = Counter(out_categories)
    print(f"\n  Categories ({len(cat_counts)}):")
    for c in sorted(cat_counts, key=cat_counts.get, reverse=True)[:10]:
        print(f"    {c:15s}: {cat_counts[c]}")

    # Uniqueness
    means = out_matrix.mean(axis=1).round(6)
    n_unique = len(np.unique(means))
    print(f"\n  Unique-mean spectra: {n_unique} / {out_matrix.shape[0]}")

    # ---- Write ENVI sli ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_envi_sli(
        path=output_path,
        wavelength_nm=lwir_wl,
        spectra=out_matrix,
        spectra_names=out_names,
        description=f"USGS-augmented LWIR library ({args.n_total} spectra)",
    )
    sli_path = output_path.with_suffix(".sli")
    hdr_path = output_path.with_suffix(".hdr")
    print(f"\nWrote {sli_path} ({sli_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {hdr_path}")

    # ---- Plot ----
    if not args.no_plot:
        plot_path = output_path.with_suffix(".png")
        _plot_samples(lwir_wl, out_matrix, out_categories, plot_path)
        print(f"Wrote {plot_path}")
        if not args.no_open:
            try:
                subprocess.run(["open", str(plot_path)], check=False)
            except FileNotFoundError:
                pass


def _plot_samples(wl_nm, matrix, categories, path):
    cats = sorted(set(categories))
    n_show = min(8, len(cats))
    cats = cats[:n_show]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(n_show, 1, figsize=(12, 1.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    wl_um = wl_nm / 1000.0
    for ax, cat in zip(axes, cats):
        idx = [i for i, c in enumerate(categories) if c == cat]
        chosen = rng.choice(idx, size=min(8, len(idx)), replace=False)
        for i in chosen:
            ax.plot(wl_um, matrix[i], alpha=0.65, linewidth=0.8)
        ax.set_ylabel(cat[:10], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("wavelength (μm)")
    fig.suptitle(
        f"USGS-augmented LWIR library — {matrix.shape[0]:,} spectra, "
        f"{matrix.shape[1]:,} bands",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
