"""Tests for the data generation pipeline."""

import numpy as np
import pytest

from spectralnp.data.random_sensor import (
    add_sensor_noise,
    apply_sensor,
    sample_virtual_sensor,
)
from spectralnp.data.rtm_simulator import (
    AtmosphericState,
    simplified_toa_radiance,
)
from spectralnp.data.sensor_definitions import LANDSAT8_OLI


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestSimplifiedRTM:
    def test_output_shape(self):
        wl = np.arange(380, 2501, 5.0)
        refl = np.full_like(wl, 0.3)
        result = simplified_toa_radiance(refl, wl)
        assert result.toa_radiance.shape == wl.shape
        assert result.wavelength_nm.shape == wl.shape

    def test_radiance_nonnegative(self):
        wl = np.arange(380, 2501, 5.0)
        refl = np.full_like(wl, 0.5)
        result = simplified_toa_radiance(refl, wl)
        assert np.all(result.toa_radiance >= 0)

    def test_higher_reflectance_higher_radiance(self):
        wl = np.arange(400, 2501, 10.0)
        low = simplified_toa_radiance(np.full_like(wl, 0.1), wl)
        high = simplified_toa_radiance(np.full_like(wl, 0.9), wl)
        # On average, higher surface reflectance should give more radiance.
        assert high.toa_radiance.mean() > low.toa_radiance.mean()

    def test_atmospheric_absorption_features(self):
        wl = np.arange(380, 2501, 1.0)
        refl = np.full_like(wl, 0.5)
        dry = simplified_toa_radiance(refl, wl, AtmosphericState(water_vapour=0.2))
        wet = simplified_toa_radiance(refl, wl, AtmosphericState(water_vapour=5.0))
        # At water absorption bands (~1380 nm), wet should have less radiance.
        idx_1380 = np.argmin(np.abs(wl - 1380))
        assert wet.toa_radiance[idx_1380] < dry.toa_radiance[idx_1380]


class TestRandomSensor:
    def test_band_count_range(self, rng):
        for _ in range(20):
            sensor = sample_virtual_sensor(rng, n_bands_range=(5, 50))
            assert 5 <= sensor.n_bands <= 50

    def test_wavelength_range(self, rng):
        sensor = sample_virtual_sensor(rng, wavelength_range=(400, 2400))
        assert sensor.center_wavelength_nm.min() >= 400
        assert sensor.center_wavelength_nm.max() <= 2400

    def test_sorted_wavelengths(self, rng):
        for _ in range(10):
            sensor = sample_virtual_sensor(rng)
            assert np.all(np.diff(sensor.center_wavelength_nm) >= 0)

    def test_strategies(self, rng):
        for strategy in ["uniform", "clustered", "regular"]:
            sensor = sample_virtual_sensor(rng, strategy=strategy)
            assert sensor.n_bands >= 3


class TestSensorConvolution:
    def test_landsat_bands(self):
        wl = np.arange(380, 2501, 1.0)
        spectrum = np.sin(wl / 500) * 0.3 + 0.5
        bands = LANDSAT8_OLI.convolve(wl, spectrum)
        assert bands.shape == (7,)
        assert np.all(np.isfinite(bands))

    def test_virtual_sensor_apply(self, rng):
        wl = np.arange(380, 2501, 5.0)
        spectrum = np.random.rand(len(wl))
        sensor = sample_virtual_sensor(rng, n_bands_range=(10, 10))
        bands = apply_sensor(sensor, wl, spectrum)
        assert bands.shape == (10,)

    def test_noise_preserves_shape(self, rng):
        rad = np.random.rand(20) * 100
        noisy = add_sensor_noise(rad, rng)
        assert noisy.shape == rad.shape
        # Noisy should differ from clean.
        assert not np.allclose(rad, noisy)
