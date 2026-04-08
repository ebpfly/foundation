"""Use case 3 — material classification with uncertainty.

The model outputs probabilities over training-spectrum indices (1748 in
practice). To get human-meaningful metrics, aggregate to the USGS category
level (minerals, vegetation, soils, ...).
"""

from __future__ import annotations

import numpy as np

from spectralnp.benchmarks import data as bench_data
from spectralnp.benchmarks import metrics as M
from spectralnp.benchmarks.radiance import _ground_truth_radiance
from spectralnp.data.random_sensor import add_sensor_noise
from spectralnp.data.usgs_speclib import SpectralLibrary
from spectralnp.inference.predict import SpectralNPPredictor


def _build_category_table(train_speclib: SpectralLibrary) -> tuple[list[str], list[str], np.ndarray]:
    """Build the index → category mapping in training order.

    Returns
    -------
    category_by_idx : list[str]
        Length = n_train_spectra. Category for each index.
    category_names  : list[str]
        Sorted unique category names.
    cat_id_by_idx   : np.ndarray
        Length n_train_spectra of category indices into ``category_names``.
    """
    category_by_idx = [s.category for s in train_speclib.spectra]
    category_names = sorted(set(category_by_idx))
    name_to_id = {n: i for i, n in enumerate(category_names)}
    cat_id_by_idx = np.array([name_to_id[c] for c in category_by_idx], dtype=np.int64)
    return category_by_idx, category_names, cat_id_by_idx


def _aggregate_to_category(probs: np.ndarray, cat_id_by_idx: np.ndarray, n_categories: int) -> np.ndarray:
    """Sum per-spectrum probabilities into per-category probabilities."""
    cat_probs = np.zeros(n_categories, dtype=np.float64)
    np.add.at(cat_probs, cat_id_by_idx, probs)
    s = cat_probs.sum()
    if s > 0:
        cat_probs /= s
    return cat_probs


def run_material_benchmark(
    predictor: SpectralNPPredictor,
    train_speclib: SpectralLibrary,
    test_indices: list[int],
    sensors: list[bench_data.SensorScenario] | None = None,
    atmospheres: list[bench_data.AtmosphereScenario] | None = None,
    n_samples: int = 16,
    snr: float = 200.0,
) -> dict:
    """Run use case 3 across all sensor + atmosphere combinations.

    Parameters
    ----------
    train_speclib : SpectralLibrary
        Full training library (used to look up category-by-index).
    test_indices : list[int]
        Indices into the training library that form the test set.
    """
    if sensors is None:
        sensors = bench_data.all_sensors()
    if atmospheres is None:
        atmospheres = bench_data.ATMOSPHERE_SCENARIOS

    wl_dense = bench_data.DENSE_WL
    rng = np.random.default_rng(636363)

    _, category_names, cat_id_by_idx = _build_category_table(train_speclib)
    n_categories = len(category_names)

    y_true_cat: list[int] = []
    cat_probs_list: list[np.ndarray] = []
    entropies: list[float] = []

    for idx in test_indices:
        spec = train_speclib.spectra[idx]
        true_cat_id = int(cat_id_by_idx[idx])

        for s_scn in sensors:
            wl_band, fw_band = bench_data.get_sensor_bands(s_scn.sensor)
            for a_scn in atmospheres:
                y_rad = _ground_truth_radiance(spec, a_scn.atmos, a_scn.geom, wl_dense)
                band_rad = bench_data.convolve_sensor(s_scn.sensor, wl_dense, y_rad)
                band_rad = add_sensor_noise(band_rad, rng, snr_range=(snr, snr)).astype(np.float32)

                pred = predictor.predict(
                    wavelength_nm=wl_band,
                    fwhm_nm=fw_band,
                    radiance=band_rad,
                    n_samples=n_samples,
                )
                if pred.material_probs is None:
                    continue
                cat_probs = _aggregate_to_category(
                    pred.material_probs, cat_id_by_idx, n_categories
                )
                cat_probs_list.append(cat_probs)
                y_true_cat.append(true_cat_id)
                entropies.append(float(pred.material_entropy) if pred.material_entropy is not None else 0.0)

    if not cat_probs_list:
        return {"error": "no predictions"}

    probs_arr = np.stack(cat_probs_list)        # (N, C)
    y_true = np.array(y_true_cat, dtype=np.int64)
    y_pred = probs_arr.argmax(axis=-1)

    per_class = M.per_class_prf1(y_true, y_pred, category_names)
    cm = M.confusion_matrix(y_true, y_pred, n_categories)

    result = {
        "category_names": category_names,
        "n_categories": n_categories,
        "n_test_cases": int(len(y_true)),
        "top1_category": M.topk_accuracy(y_true, probs_arr, k=1),
        "top3_category": M.topk_accuracy(y_true, probs_arr, k=3),
        "macro_f1": M.macro_f1(per_class),
        "ece_category": M.ece(y_true, probs_arr, n_bins=15),
        "brier_category": M.brier_multiclass(y_true, probs_arr),
        "entropy_error_correlation": M.entropy_error_correlation(
            (y_pred != y_true).astype(np.float64),
            np.array(entropies),
        ),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    return result
