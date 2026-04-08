"""Pytest wrapper for the benchmark suite — regression thresholds.

Skips when no checkpoint is available. Asserts on minimum thresholds we
can ratchet up over time as the model improves.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from spectralnp.benchmarks import data as bench_data
from spectralnp.benchmarks.material import run_material_benchmark
from spectralnp.benchmarks.radiance import run_radiance_benchmark
from spectralnp.benchmarks.reflectance import run_reflectance_benchmark
from spectralnp.data.usgs_speclib import load_from_directory, load_from_zip
from spectralnp.inference.predict import SpectralNPPredictor
from spectralnp.model.spectralnp import SpectralNP

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = REPO_ROOT / "demo_model.pt"
DEFAULT_USGS = Path(
    os.environ.get(
        "SPECTRALNP_USGS",
        "/Users/eric/repo/atmgen/USGS_ASCIIdata/ASCIIdata_splib07a",
    )
)
# Run a small subset of test spectra in the pytest version so it stays fast.
PYTEST_TEST_SPECTRA = 30
PYTEST_N_SAMPLES = 8


@pytest.fixture(scope="module")
def predictor():
    if not DEFAULT_MODEL.exists():
        pytest.skip(f"No model checkpoint at {DEFAULT_MODEL}")
    if not DEFAULT_USGS.exists():
        pytest.skip(f"No USGS data at {DEFAULT_USGS}")
    ckpt = torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=False)
    model = SpectralNP(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return SpectralNPPredictor(model)


@pytest.fixture(scope="module")
def test_speclib():
    if DEFAULT_USGS.suffix == ".zip":
        speclib = load_from_zip(DEFAULT_USGS)
    else:
        speclib = load_from_directory(DEFAULT_USGS)
    speclib = speclib.filter_wavelength_range(380, 2400)
    test_lib, indices = bench_data.held_out_speclib(speclib)
    # Limit for speed.
    test_lib.spectra = test_lib.spectra[:PYTEST_TEST_SPECTRA]
    indices = indices[:PYTEST_TEST_SPECTRA]
    return speclib, test_lib, indices


# ---------------------------------------------------------------------------
# Use case 1 — radiance
# ---------------------------------------------------------------------------


def test_radiance_basic(predictor, test_speclib):
    _, test_lib, _ = test_speclib
    # AVIRIS-NG only — high band count, our headline sensor.
    aviris = [s for s in bench_data.real_sensors() if s.name == "AVIRIS-NG"]
    moderate = [bench_data.ATMOSPHERE_SCENARIOS[1]]
    results = run_radiance_benchmark(
        predictor, test_lib, sensors=aviris, atmospheres=moderate,
        n_samples=PYTEST_N_SAMPLES,
    )
    metrics = results["by_sensor"]["AVIRIS-NG"]
    assert metrics["sam_deg"] < 30.0, f"SAM too high: {metrics['sam_deg']:.2f}°"
    assert metrics["coverage_2sigma"] > 0.30, f"2σ coverage too low: {metrics['coverage_2sigma']:.3f}"
    assert metrics["r2"] > -1.0, f"R² is degenerate: {metrics['r2']:.3f}"


# ---------------------------------------------------------------------------
# Use case 2 — reflectance
# ---------------------------------------------------------------------------


def test_reflectance_basic(predictor, test_speclib):
    _, test_lib, _ = test_speclib
    aviris = [s for s in bench_data.real_sensors() if s.name == "AVIRIS-NG"]
    moderate = [bench_data.ATMOSPHERE_SCENARIOS[1]]
    results = run_reflectance_benchmark(
        predictor, test_lib, sensors=aviris, atmospheres=moderate,
        n_samples=PYTEST_N_SAMPLES,
    )
    metrics = results["by_sensor"]["AVIRIS-NG"]
    # Reflectance is in [0,1], so RMSE has a meaningful absolute scale.
    assert metrics["rmse"] < 0.5, f"Reflectance RMSE too high: {metrics['rmse']:.3f}"
    assert metrics["sam_deg"] < 45.0, f"Reflectance SAM too high: {metrics['sam_deg']:.2f}°"


# ---------------------------------------------------------------------------
# Use case 3 — material classification
# ---------------------------------------------------------------------------


def test_material_above_chance(predictor, test_speclib):
    full_lib, _, indices = test_speclib
    aviris = [s for s in bench_data.real_sensors() if s.name == "AVIRIS-NG"]
    moderate = [bench_data.ATMOSPHERE_SCENARIOS[1]]
    results = run_material_benchmark(
        predictor, full_lib, indices, sensors=aviris, atmospheres=moderate,
        n_samples=PYTEST_N_SAMPLES,
    )
    n_classes = results["n_categories"]
    chance = 1.0 / n_classes
    assert results["top1_category"] > chance, (
        f"Top-1 ({results['top1_category']:.3f}) below chance ({chance:.3f})"
    )
    assert results["top3_category"] >= results["top1_category"]
