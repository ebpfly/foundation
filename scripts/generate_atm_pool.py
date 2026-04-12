#!/usr/bin/env python3
"""Pre-generate a pool of atmospheric RT components (τ, Lup, Ldown).

Uses atmgen's MultiClimateSampler for physically diverse atmospheres
and RTCalculator (with LUT when available) for accurate LWIR RT.

The pool is saved as a single .npz file.  At training time, the dataset
picks a random atmosphere from the pool, optionally perturbs it, and
combines it with a random surface emissivity + temperature to produce
TOA radiance analytically:

    L_TOA(λ) = τ(λ) · [ε(λ) · B(λ, T_s) + (1-ε(λ)) · L_down(λ)] + L_up(λ)

Usage:
    python scripts/generate_atm_pool.py --n-atm 500 --output data/atm_pool.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Generate atmospheric RT pool")
    p.add_argument("--n-atm", type=int, default=500,
                   help="Number of atmospheres to generate")
    p.add_argument("--n-points", type=int, default=500,
                   help="Number of spectral points (interpolated to training grid later)")
    p.add_argument("--wl-lo", type=float, default=7.0, help="Lower wavelength [μm]")
    p.add_argument("--wl-hi", type=float, default=16.0, help="Upper wavelength [μm]")
    p.add_argument("--output", type=str, default="data/atm_pool.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from atmgen.sampling import MultiClimateSampler
    from atmgen.radiative_transfer import RTCalculator

    rng = np.random.default_rng(args.seed)

    print(f"Creating multi-climate sampler...")
    sampler = MultiClimateSampler(seed=args.seed)

    print(f"Creating RT calculator ({args.wl_lo}-{args.wl_hi} μm, {args.n_points} pts)...")
    calc = RTCalculator.for_wavelength_range(
        args.wl_lo, args.wl_hi, n_points=args.n_points,
    )
    wl_um = calc.wavelength_um.copy()
    # atmgen returns descending wavelength; sort ascending for consistency
    if wl_um[0] > wl_um[-1]:
        wl_um = wl_um[::-1]
        flip = True
    else:
        flip = False

    n_wl = len(wl_um)
    wl_nm = (wl_um * 1000.0).astype(np.float32)

    # Conversion factor: W/(m²·sr·Hz) → W/(m²·sr·μm)
    c_ms = 2.99792458e8
    wl_m = wl_um * 1e-6
    hz_to_um = c_ms / (wl_m ** 2) * 1e-6  # per wavelength point

    # Allocate output arrays
    tau_pool = np.zeros((args.n_atm, n_wl), dtype=np.float32)
    lup_pool = np.zeros((args.n_atm, n_wl), dtype=np.float32)
    ldn_pool = np.zeros((args.n_atm, n_wl), dtype=np.float32)

    print(f"Generating {args.n_atm} atmospheres...")
    t0 = time.time()
    for i in range(args.n_atm):
        atm = sampler.sample_one()

        # Transmission
        trans_r = calc.compute_transmission(atm)
        tau = trans_r.transmission
        if flip:
            tau = tau[::-1]

        # Upwelling path radiance (atmosphere only, no surface)
        up_r = calc.compute_upwelling(
            atm, observer_altitude=800e3, atmosphere_only=True,
        )
        lup = up_r.radiance
        if flip:
            lup = lup[::-1]

        # Downwelling
        dn_r = calc.compute_downwelling(atm)
        ldn = dn_r.radiance
        if flip:
            ldn = ldn[::-1]

        # Convert radiance units to W/(m²·sr·μm)
        tau_pool[i] = tau.astype(np.float32)
        lup_pool[i] = (lup * hz_to_um).astype(np.float32)
        ldn_pool[i] = (ldn * hz_to_um).astype(np.float32)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (args.n_atm - i - 1) / rate
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:4d}/{args.n_atm}]  "
                  f"τ=[{tau.min():.3f},{tau.max():.3f}]  "
                  f"Lup=[{lup_pool[i].min():.2f},{lup_pool[i].max():.2f}]  "
                  f"{rate:.2f} atm/s  ETA {eta/60:.0f}m")

    total = time.time() - t0
    print(f"\nDone in {total/60:.1f} min ({total/args.n_atm:.1f} s/atm)")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        wavelength_nm=wl_nm,
        tau=tau_pool,
        lup=lup_pool,
        ldn=ldn_pool,
    )
    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  shape: ({args.n_atm}, {n_wl})")
    print(f"  wavelength: {wl_nm[0]:.0f}-{wl_nm[-1]:.0f} nm")


if __name__ == "__main__":
    main()
