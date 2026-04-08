"""Use case 1 — at-sensor radiance prediction with uncertainty.

For each test spectrum, simulate dense at-sensor radiance, observe through
each sensor in the benchmark sensor list, run the predictor, and compute
all continuous metrics against the dense ground truth.
"""

from __future__ import annotations

import numpy as np

from spectralnp.benchmarks import data as bench_data
from spectralnp.benchmarks import metrics as M
from spectralnp.data.random_sensor import add_sensor_noise
from spectralnp.data.rtm_simulator import simplified_toa_radiance
from spectralnp.data.usgs_speclib import SpectralLibrary
from spectralnp.inference.predict import SpectralNPPredictor


def _ground_truth_radiance(
    spec, atmos, geom, wl_dense
) -> np.ndarray:
    """Ground-truth dense at-sensor radiance for one (spec, atmos) pair."""
    refl = np.clip(np.nan_to_num(spec.resample(wl_dense), nan=0.04), 0, 1)
    rtm = simplified_toa_radiance(
        surface_reflectance=refl, wavelength_nm=wl_dense, atmos=atmos, geometry=geom
    )
    return rtm.toa_radiance.astype(np.float32)


def _metrics_block(y: np.ndarray, yh: np.ndarray, s: np.ndarray) -> dict:
    """Compute the standard continuous-prediction metrics block."""
    return {
        "rmse": M.rmse(y, yh),
        "mae": M.mae(y, yh),
        "mape": M.mape(y, yh),
        "sam_deg": M.sam_deg(y, yh),
        "r2": M.r2_score(y, yh),
        "coverage_1sigma": M.coverage(y, yh, s, k=1.0),
        "coverage_2sigma": M.coverage(y, yh, s, k=2.0),
        "coverage_3sigma": M.coverage(y, yh, s, k=3.0),
        "picp_95": M.picp(y, yh, s, level=0.95),
        "sharpness": M.sharpness(s),
        "crps": M.gaussian_crps(y, yh, s),
        "nll": M.gaussian_nll(y, yh, s),
    }


def run_radiance_benchmark(
    predictor: SpectralNPPredictor,
    test_speclib: SpectralLibrary,
    sensors: list[bench_data.SensorScenario] | None = None,
    atmospheres: list[bench_data.AtmosphereScenario] | None = None,
    n_samples: int = 16,
    snr: float = 200.0,
) -> dict:
    """Run use case 1 across all (sensor, atmosphere) pairs.

    Returns a metrics dict structured by sensor, plus a scaling sub-block.
    """
    if sensors is None:
        sensors = bench_data.all_sensors()
    if atmospheres is None:
        atmospheres = bench_data.ATMOSPHERE_SCENARIOS

    wl_dense = bench_data.DENSE_WL
    rng = np.random.default_rng(424242)

    by_sensor: dict[str, dict] = {}

    for s_scn in sensors:
        wl_band, fw_band = bench_data.get_sensor_bands(s_scn.sensor)
        all_y, all_yh, all_sigma = [], [], []

        for a_scn in atmospheres:
            for spec in test_speclib.spectra:
                # Ground truth dense radiance.
                y_dense = _ground_truth_radiance(spec, a_scn.atmos, a_scn.geom, wl_dense)
                # Sensor input.
                band_rad = bench_data.convolve_sensor(s_scn.sensor, wl_dense, y_dense)
                band_rad = add_sensor_noise(band_rad, rng, snr_range=(snr, snr)).astype(np.float32)

                pred = predictor.predict(
                    wavelength_nm=wl_band,
                    fwhm_nm=fw_band,
                    radiance=band_rad,
                    query_wavelength_nm=wl_dense,
                    n_samples=n_samples,
                )
                if pred.spectral_mean is None:
                    continue
                all_y.append(y_dense)
                all_yh.append(pred.spectral_mean.astype(np.float32))
                all_sigma.append(pred.spectral_std.astype(np.float32))

        if not all_y:
            continue
        y_arr = np.stack(all_y)
        yh_arr = np.stack(all_yh)
        s_arr = np.stack(all_sigma)
        block = _metrics_block(y_arr, yh_arr, s_arr)
        block["n_bands"] = int(s_scn.n_bands)
        block["n_test_cases"] = int(y_arr.shape[0])
        by_sensor[s_scn.name] = block

    # ---- Scaling curve: error vs. n_bands at fixed atmosphere ----
    scaling_sensors = bench_data.scaling_sensors()
    scaling_atm = atmospheres[len(atmospheres) // 2]  # moderate
    scaling: dict[str, list] = {
        "n_bands": [],
        "rmse": [],
        "sharpness": [],
        "crps": [],
    }
    for s_scn in scaling_sensors:
        wl_band, fw_band = bench_data.get_sensor_bands(s_scn.sensor)
        ys, yhs, ss = [], [], []
        for spec in test_speclib.spectra:
            y_dense = _ground_truth_radiance(spec, scaling_atm.atmos, scaling_atm.geom, wl_dense)
            band_rad = bench_data.convolve_sensor(s_scn.sensor, wl_dense, y_dense)
            band_rad = add_sensor_noise(band_rad, rng, snr_range=(snr, snr)).astype(np.float32)
            pred = predictor.predict(
                wavelength_nm=wl_band, fwhm_nm=fw_band, radiance=band_rad,
                query_wavelength_nm=wl_dense, n_samples=n_samples,
            )
            if pred.spectral_mean is None:
                continue
            ys.append(y_dense)
            yhs.append(pred.spectral_mean.astype(np.float32))
            ss.append(pred.spectral_std.astype(np.float32))
        if not ys:
            continue
        y_arr = np.stack(ys)
        yh_arr = np.stack(yhs)
        s_arr = np.stack(ss)
        scaling["n_bands"].append(int(s_scn.n_bands))
        scaling["rmse"].append(M.rmse(y_arr, yh_arr))
        scaling["sharpness"].append(M.sharpness(s_arr))
        scaling["crps"].append(M.gaussian_crps(y_arr, yh_arr, s_arr))

    return {"by_sensor": by_sensor, "scaling": scaling}
