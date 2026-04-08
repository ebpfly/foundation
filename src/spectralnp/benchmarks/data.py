"""Test data and scenario generation for benchmarks.

Defines a deterministic held-out USGS split, the list of sensors to evaluate,
and the atmospheric conditions to average over.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from spectralnp.data.random_sensor import VirtualSensor, sample_virtual_sensor
from spectralnp.data.rtm_simulator import AtmosphericState, ViewGeometry
from spectralnp.data.sensor_definitions import (
    AVIRIS_NG,
    EnMAP,
    LANDSAT8_OLI,
    SENTINEL2_MSI,
)
from spectralnp.data.usgs_speclib import SpectralLibrary

# Dense reconstruction grid (matches training).
DENSE_WL = np.arange(380.0, 2501.0, 5.0)

# Deterministic held-out test split: same indices every run, regardless of
# training. Tests we want to ratchet up over time.
TEST_SPLIT_SEED = 12345
TEST_SPLIT_SIZE = 150


def held_out_indices(n_total: int, size: int = TEST_SPLIT_SIZE) -> list[int]:
    """Return a deterministic list of held-out indices into a speclib."""
    rng = random.Random(TEST_SPLIT_SEED)
    return sorted(rng.sample(range(n_total), min(size, n_total)))


def held_out_speclib(speclib: SpectralLibrary) -> tuple[SpectralLibrary, list[int]]:
    """Return (test_speclib, indices_into_full)."""
    indices = held_out_indices(len(speclib))
    test_spectra = [speclib.spectra[i] for i in indices]
    return SpectralLibrary(spectra=test_spectra), indices


# ---------------------------------------------------------------------------
# Atmospheres
# ---------------------------------------------------------------------------


@dataclass
class AtmosphereScenario:
    name: str
    atmos: AtmosphericState
    geom: ViewGeometry


ATMOSPHERE_SCENARIOS: list[AtmosphereScenario] = [
    AtmosphereScenario(
        name="clean",
        atmos=AtmosphericState(aod_550=0.05, water_vapour=0.5, ozone_du=300.0),
        geom=ViewGeometry(solar_zenith_deg=30.0, sensor_zenith_deg=0.0),
    ),
    AtmosphereScenario(
        name="moderate",
        atmos=AtmosphericState(aod_550=0.20, water_vapour=2.0, ozone_du=320.0),
        geom=ViewGeometry(solar_zenith_deg=30.0, sensor_zenith_deg=0.0),
    ),
    AtmosphereScenario(
        name="hazy",
        atmos=AtmosphericState(aod_550=0.60, water_vapour=4.0, ozone_du=300.0),
        geom=ViewGeometry(solar_zenith_deg=45.0, sensor_zenith_deg=0.0),
    ),
]


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


@dataclass
class SensorScenario:
    name: str
    sensor: object  # SensorDefinition or VirtualSensor
    n_bands: int


def real_sensors() -> list[SensorScenario]:
    """Real Earth-observation sensors covering low → high band counts."""
    return [
        SensorScenario("Landsat-8", LANDSAT8_OLI, LANDSAT8_OLI.n_bands),
        SensorScenario("Sentinel-2", SENTINEL2_MSI, SENTINEL2_MSI.n_bands),
        SensorScenario("EnMAP", EnMAP, EnMAP.n_bands),
        SensorScenario("AVIRIS-NG", AVIRIS_NG, AVIRIS_NG.n_bands),
    ]


VIRTUAL_SENSOR_SEED = 67890


def virtual_sensors(n_bands_list: list[int] | None = None) -> list[SensorScenario]:
    """Random virtual sensors with fixed band counts (deterministic seed)."""
    if n_bands_list is None:
        n_bands_list = [5, 30, 100]
    rng = np.random.default_rng(VIRTUAL_SENSOR_SEED)
    out = []
    for n in n_bands_list:
        sensor = sample_virtual_sensor(rng, n_bands_range=(n, n))
        out.append(SensorScenario(f"random-{n}", sensor, n))
    return out


def all_sensors() -> list[SensorScenario]:
    """Default benchmark sensor list: real + virtual."""
    return real_sensors() + virtual_sensors()


# Scaling-curve band counts (use case 1 & 2 sub-benchmark).
SCALING_BAND_COUNTS: list[int] = [3, 5, 10, 20, 50, 100, 200]


def scaling_sensors(n_bands_list: list[int] | None = None) -> list[SensorScenario]:
    """One random virtual sensor per band count for the scaling curve."""
    if n_bands_list is None:
        n_bands_list = SCALING_BAND_COUNTS
    rng = np.random.default_rng(VIRTUAL_SENSOR_SEED + 1)
    return [
        SensorScenario(f"scale-{n}", sample_virtual_sensor(rng, n_bands_range=(n, n)), n)
        for n in n_bands_list
    ]


def get_sensor_bands(sensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (center_wavelength_nm, fwhm_nm) regardless of sensor type."""
    return (
        np.asarray(sensor.center_wavelength_nm, dtype=np.float32),
        np.asarray(sensor.fwhm_nm, dtype=np.float32),
    )


def convolve_sensor(sensor, wl_dense: np.ndarray, dense_radiance: np.ndarray) -> np.ndarray:
    """Convolve a dense spectrum to a sensor's bands.

    SensorDefinition has a ``convolve()`` method; VirtualSensor uses
    ``apply_sensor()``.
    """
    if isinstance(sensor, VirtualSensor):
        from spectralnp.data.random_sensor import apply_sensor

        return apply_sensor(sensor, wl_dense, dense_radiance).astype(np.float32)
    return sensor.convolve(wl_dense, dense_radiance).astype(np.float32)
