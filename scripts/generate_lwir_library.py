#!/usr/bin/env python3
"""Generate a synthetic LWIR emissivity library and save as ENVI sli.

Usage:
    python scripts/generate_lwir_library.py \
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

from spectralnp.data.envi_sli import read_envi_sli, write_envi_sli
from spectralnp.data.synthetic_lwir import generate_lwir_library


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic LWIR emissivity library")
    p.add_argument("--output", type=str, default="data/lwir_library",
                   help="Base path (.sli and .hdr will be written)")
    p.add_argument("--n-total", type=int, default=10_000)
    p.add_argument("--n-bands", type=int, default=4_000)
    p.add_argument("--wl-lo", type=float, default=7.0, help="Lower bound [μm]")
    p.add_argument("--wl-hi", type=float, default=16.0, help="Upper bound [μm]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify", action="store_true",
                   help="Read back the written file and check round-trip")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    print(
        f"Generating {args.n_total:,} spectra × {args.n_bands:,} bands "
        f"({args.wl_lo}–{args.wl_hi} μm) ..."
    )
    lib = generate_lwir_library(
        n_total=args.n_total,
        wavelength_lo_um=args.wl_lo,
        wavelength_hi_um=args.wl_hi,
        n_bands=args.n_bands,
        seed=args.seed,
    )

    counts = Counter(lib.categories)
    print(f"\nClass distribution ({len(counts)} classes):")
    for cat in sorted(counts, key=counts.get, reverse=True):
        print(f"  {cat:12s}: {counts[cat]:5d}")

    print(f"\nEmissivity statistics:")
    print(f"  shape             : {lib.emissivity.shape}")
    print(f"  range             : [{lib.emissivity.min():.4f}, {lib.emissivity.max():.4f}]")
    print(f"  mean              : {lib.emissivity.mean():.4f}")
    print(f"  std               : {lib.emissivity.std():.4f}")

    # Uniqueness check — no two spectra should be exactly identical
    n_unique = len(np.unique(lib.emissivity.mean(axis=1).round(decimals=6)))
    print(f"  unique-mean rows  : {n_unique} / {lib.emissivity.shape[0]}")

    # ---- Write ENVI sli ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_envi_sli(
        path=output_path,
        wavelength_nm=lib.wavelength_nm,
        spectra=lib.emissivity,
        spectra_names=lib.names,
        description=f"Synthetic LWIR emissivity library ({args.n_total} spectra)",
    )
    sli_path = output_path.with_suffix(".sli")
    hdr_path = output_path.with_suffix(".hdr")
    print(f"\nWrote {sli_path} ({sli_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {hdr_path}")

    # ---- Round-trip verify ----
    if args.verify:
        wl_r, spectra_r, names_r = read_envi_sli(output_path)
        assert wl_r.shape == lib.wavelength_nm.shape, \
            f"wavelength shape mismatch: {wl_r.shape} vs {lib.wavelength_nm.shape}"
        assert spectra_r.shape == lib.emissivity.shape, \
            f"spectra shape mismatch: {spectra_r.shape} vs {lib.emissivity.shape}"
        assert np.allclose(wl_r, lib.wavelength_nm, atol=1e-3), \
            "wavelengths differ after round-trip"
        assert np.allclose(spectra_r, lib.emissivity, atol=1e-5), \
            "spectra differ after round-trip"
        assert names_r == lib.names, "names differ after round-trip"
        print("Round-trip verify: OK")

    # ---- Diagnostic plot ----
    if not args.no_plot:
        plot_path = output_path.with_suffix(".png")
        _plot_samples(lib, plot_path)
        print(f"Wrote {plot_path}")
        if not args.no_open:
            try:
                subprocess.run(["open", str(plot_path)], check=False)
            except FileNotFoundError:
                pass


def _plot_samples(lib, path: Path) -> None:
    """Plot a grid of sample spectra from each class."""
    categories = sorted(set(lib.categories))
    n_classes = len(categories)
    n_per_class = 6

    n_rows = (n_classes + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.0 * n_rows), sharex=True)
    axes = axes.flatten()

    wl_um = lib.wavelength_nm / 1000.0
    rng = np.random.default_rng(0)
    for ax, cat in zip(axes, categories):
        idx = [i for i, c in enumerate(lib.categories) if c == cat]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for i in chosen:
            ax.plot(wl_um, lib.emissivity[i], alpha=0.75, linewidth=1.0)
        ax.set_title(cat, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("ε", fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[len(categories):]:
        ax.axis("off")

    for ax in axes[-2:]:
        ax.set_xlabel("wavelength (μm)", fontsize=9)

    fig.suptitle(
        f"Synthetic LWIR emissivity library — {lib.emissivity.shape[0]:,} spectra, "
        f"{lib.emissivity.shape[1]:,} bands, {len(categories)} classes",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
