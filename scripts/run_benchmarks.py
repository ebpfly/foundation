#!/usr/bin/env python3
"""Run the SpectralNP benchmark suite against a model checkpoint.

Outputs a timestamped directory under ``benchmarks/results/`` containing
JSON metrics, a markdown summary, and matplotlib plots — all auto-opened
when the script finishes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import subprocess
from pathlib import Path

import torch

from spectralnp.benchmarks import data as bench_data
from spectralnp.benchmarks import report as bench_report
from spectralnp.benchmarks.material import run_material_benchmark
from spectralnp.benchmarks.radiance import run_radiance_benchmark
from spectralnp.benchmarks.reflectance import run_reflectance_benchmark
from spectralnp.data.usgs_speclib import load_from_directory, load_from_zip
from spectralnp.inference.predict import SpectralNPPredictor
from spectralnp.model.spectralnp import SpectralNP


def main():
    p = argparse.ArgumentParser(description="Run SpectralNP benchmarks.")
    p.add_argument("--model", type=str, required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--usgs-data", type=str, required=True,
                   help="USGS spectral library (directory or .zip)")
    p.add_argument("--output", type=str, default="benchmarks/results",
                   help="Output base directory")
    p.add_argument("--n-samples", type=int, default=16,
                   help="Number of latent samples per prediction")
    p.add_argument("--n-test-spectra", type=int, default=None,
                   help="Limit test set size (default: full 150)")
    p.add_argument("--snr", type=float, default=200.0,
                   help="Signal-to-noise ratio for sensor noise injection")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-open", action="store_true",
                   help="Do not auto-open the summary at the end")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    # ---- Load model ----
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = SpectralNP(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    log.info(f"Loaded model from {args.model}")

    predictor = SpectralNPPredictor(model)
    predictor.device = device

    # ---- Load data ----
    usgs_path = Path(args.usgs_data)
    if usgs_path.suffix == ".zip":
        full_speclib = load_from_zip(usgs_path)
    else:
        full_speclib = load_from_directory(usgs_path)
    full_speclib = full_speclib.filter_wavelength_range(380, 2400)
    log.info(f"Loaded {len(full_speclib)} spectra from USGS")

    test_speclib, test_indices = bench_data.held_out_speclib(full_speclib)
    if args.n_test_spectra is not None and args.n_test_spectra < len(test_speclib):
        test_speclib.spectra = test_speclib.spectra[: args.n_test_spectra]
        test_indices = test_indices[: args.n_test_spectra]
    log.info(f"Held-out test set: {len(test_speclib)} spectra (indices seed={bench_data.TEST_SPLIT_SEED})")

    # ---- Run benchmarks ----
    log.info("Running radiance benchmark...")
    rad_results = run_radiance_benchmark(predictor, test_speclib, n_samples=args.n_samples, snr=args.snr)

    log.info("Running reflectance benchmark...")
    refl_results = run_reflectance_benchmark(predictor, test_speclib, n_samples=args.n_samples, snr=args.snr)

    log.info("Running material benchmark...")
    mat_results = run_material_benchmark(
        predictor, full_speclib, test_indices,
        n_samples=args.n_samples, snr=args.snr,
    )

    # ---- Assemble + write ----
    results = {
        "model_path": args.model,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "n_test_spectra": len(test_speclib),
        "n_samples": args.n_samples,
        "snr": args.snr,
        "checkpoint_loss": float(ckpt.get("loss", 0)) if ckpt.get("loss") is not None else None,
        "checkpoint_epoch": int(ckpt.get("epoch", 0)),
        "radiance": rad_results,
        "reflectance": refl_results,
        "material": mat_results,
    }

    model_name = Path(args.model).stem
    out_dir = bench_report.make_results_dir(args.output, model_name)
    paths = bench_report.write_all(out_dir, results)

    log.info(f"Results written to {out_dir}")
    for label, p_ in paths.items():
        log.info(f"  {label}: {p_.name}")

    if not args.no_open:
        try:
            subprocess.run(["open", str(paths["summary"])], check=False)
            for k in ("radiance_scaling", "reflectance_scaling", "confusion"):
                if k in paths:
                    subprocess.run(["open", str(paths[k])], check=False)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
