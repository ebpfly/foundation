"""Use case 2 — surface reflectance prediction with uncertainty.

For each test spectrum, observe through each sensor (with realistic
atmosphere + noise), then predict reflectance via the new reflectance head
and compare against the original USGS reflectance.
"""

from __future__ import annotations

import numpy as np

from spectralnp.benchmarks import data as bench_data
from spectralnp.benchmarks import metrics as M
from spectralnp.benchmarks.radiance import _ground_truth_radiance, _metrics_block
from spectralnp.data.random_sensor import add_sensor_noise
from spectralnp.data.usgs_speclib import SpectralLibrary
from spectralnp.inference.predict import SpectralNPPredictor


def _ground_truth_reflectance(spec, wl_dense: np.ndarray) -> np.ndarray:
    """Resample the USGS spectrum to the dense grid (no RTM needed)."""
    refl = np.clip(np.nan_to_num(spec.resample(wl_dense)), 0, 1).astype(np.float32)
    return refl


def run_reflectance_benchmark(
    predictor: SpectralNPPredictor,
    test_speclib: SpectralLibrary,
    sensors: list[bench_data.SensorScenario] | None = None,
    atmospheres: list[bench_data.AtmosphereScenario] | None = None,
    n_samples: int = 16,
    snr: float = 200.0,
) -> dict:
    """Run use case 2 across all (sensor, atmosphere) pairs."""
    if sensors is None:
        sensors = bench_data.all_sensors()
    if atmospheres is None:
        atmospheres = bench_data.ATMOSPHERE_SCENARIOS

    wl_dense = bench_data.DENSE_WL
    rng = np.random.default_rng(525252)

    by_sensor: dict[str, dict] = {}

    for s_scn in sensors:
        wl_band, fw_band = bench_data.get_sensor_bands(s_scn.sensor)
        all_y, all_yh, all_sigma = [], [], []

        for a_scn in atmospheres:
            for spec in test_speclib.spectra:
                # Truth = USGS reflectance, no RTM.
                y_refl = _ground_truth_reflectance(spec, wl_dense)
                # Sensor input = simulated radiance.
                y_rad = _ground_truth_radiance(spec, a_scn.atmos, a_scn.geom, wl_dense)
                band_rad = bench_data.convolve_sensor(s_scn.sensor, wl_dense, y_rad)
                band_rad = add_sensor_noise(band_rad, rng, snr_range=(snr, snr)).astype(np.float32)

                pred = predictor.predict(
                    wavelength_nm=wl_band,
                    fwhm_nm=fw_band,
                    radiance=band_rad,
                    query_wavelength_nm=wl_dense,
                    n_samples=n_samples,
                )
                if pred.reflectance_mean is None:
                    continue
                all_y.append(y_refl)
                all_yh.append(pred.reflectance_mean.astype(np.float32))
                all_sigma.append(pred.reflectance_std.astype(np.float32))

        if not all_y:
            continue
        y_arr = np.stack(all_y)
        yh_arr = np.stack(all_yh)
        s_arr = np.stack(all_sigma)
        block = _metrics_block(y_arr, yh_arr, s_arr)
        block["n_bands"] = int(s_scn.n_bands)
        block["n_test_cases"] = int(y_arr.shape[0])
        by_sensor[s_scn.name] = block

    # ---- Scaling curve: error vs n_bands at moderate atmosphere ----
    scaling_sensors = bench_data.scaling_sensors()
    scaling_atm = atmospheres[len(atmospheres) // 2]
    scaling: dict[str, list] = {
        "n_bands": [], "rmse": [], "sharpness": [], "crps": [],
    }
    for s_scn in scaling_sensors:
        wl_band, fw_band = bench_data.get_sensor_bands(s_scn.sensor)
        ys, yhs, ss = [], [], []
        for spec in test_speclib.spectra:
            y_refl = _ground_truth_reflectance(spec, wl_dense)
            y_rad = _ground_truth_radiance(spec, scaling_atm.atmos, scaling_atm.geom, wl_dense)
            band_rad = bench_data.convolve_sensor(s_scn.sensor, wl_dense, y_rad)
            band_rad = add_sensor_noise(band_rad, rng, snr_range=(snr, snr)).astype(np.float32)
            pred = predictor.predict(
                wavelength_nm=wl_band, fwhm_nm=fw_band, radiance=band_rad,
                query_wavelength_nm=wl_dense, n_samples=n_samples,
            )
            if pred.reflectance_mean is None:
                continue
            ys.append(y_refl)
            yhs.append(pred.reflectance_mean.astype(np.float32))
            ss.append(pred.reflectance_std.astype(np.float32))
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
