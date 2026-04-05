#!/usr/bin/env python3
"""Generate an atmospheric RT lookup table from PyARTS.

Caches layer-by-layer gas absorption optical depths for 0.3–16 μm
across a grid of atmospheric states.  All six trace gases
(H₂O, CO₂, O₃, N₂O, CH₄, CO) are included.

Path integration (transmittance, emission, any geometry) is done at
runtime — the LUT stores only the expensive spectroscopy output.

Usage::

    # Default grid (~2900 atmospheric states)
    python scripts/build_lut.py -o lut/arts_lut.h5 --workers 8

    # Restrict to VNIR/SWIR only
    python scripts/build_lut.py -o lut/arts_vnir.h5 --wl-max 2500

    # Coarse grid for quick testing
    python scripts/build_lut.py -o lut/arts_test.h5 --quick
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate atmospheric RT LUT from PyARTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output HDF5 file path."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--wl-min", type=float, default=300.0, help="Min wavelength [nm]."
    )
    parser.add_argument(
        "--wl-max", type=float, default=16000.0, help="Max wavelength [nm]."
    )
    parser.add_argument(
        "--n-layers", type=int, default=50, help="Atmospheric layers."
    )
    parser.add_argument(
        "--arts-data",
        type=str,
        default=None,
        help="Path to ARTS spectroscopic data.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a coarse grid for quick testing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    from spectralnp.data.lut import ARTSLUTGenerator, LUTConfig, make_lut_wavelength_grid

    wl = make_lut_wavelength_grid(args.wl_min, args.wl_max)

    if args.quick:
        cfg = LUTConfig(
            wavelength_nm=wl,
            water_vapour=np.array([0.5, 2.0, 5.0]),
            ozone_du=np.array([250.0, 350.0]),
            co2_ppmv=np.array([420.0]),
            ch4_ppbv=np.array([1900.0]),
            n2o_ppbv=np.array([332.0]),
            co_ppbv=np.array([120.0]),
            surface_altitude_km=np.array([0.0, 2.0]),
            n_layers=args.n_layers,
        )
    else:
        cfg = LUTConfig(wavelength_nm=wl, n_layers=args.n_layers)

    log.info("Wavelength grid: %.0f–%.0f nm  (%d points)", wl[0], wl[-1], len(wl))
    log.info("Atmospheric states: %s  (%d total)", cfg.shape, cfg.n_grid_points)
    log.info("Layers: %d", cfg.n_layers)
    log.info("Workers: %d", args.workers)

    gen = ARTSLUTGenerator(cfg, arts_data_path=args.arts_data)

    t0 = time.perf_counter()
    gen.generate(args.output, n_workers=args.workers)
    elapsed = time.perf_counter() - t0

    log.info(
        "Done in %.1f min (%.1f s per atmospheric state).",
        elapsed / 60,
        elapsed / cfg.n_grid_points,
    )


if __name__ == "__main__":
    main()
