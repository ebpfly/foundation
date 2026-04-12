"""LWIR (7–16 μm) training dataset.

Generates training samples on-the-fly:
  1. Sample a surface emissivity from the synthetic LWIR library
  2. Sample atmospheric state (T_s, water vapour, ozone, CO₂)
  3. Sample viewing geometry (sensor zenith — no solar path in LWIR)
  4. Run the simplified two-stream RTM at LWIR to get TOA radiance
  5. Generate a random virtual LWIR sensor
  6. Convolve radiance with sensor SRFs + add noise

Unlike the VNIR-SWIR pipeline, the dominant signal at LWIR is thermal
emission from the surface (ε · B(λ, T_s)) plus the atmospheric radiance
path.  Surface temperature is a first-class learning target here, not
an afterthought.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spectralnp.data.envi_sli import read_envi_sli
from spectralnp.data.random_sensor import (
    add_sensor_noise,
    apply_sensor,
    sample_virtual_sensor,
)
from spectralnp.data.rtm_simulator import (
    AtmosphericState,
    ViewGeometry,
    simplified_toa_radiance,
)


# Default training grid: 7–16 μm at 2.25 nm spacing = 4001 bands
LWIR_WL_LO_NM = 7000.0
LWIR_WL_HI_NM = 16000.0
DEFAULT_N_WAVELENGTHS = 4001


class LWIRDataset(Dataset):
    """On-the-fly LWIR training dataset.

    Parameters
    ----------
    library_path : path to ENVI .sli library (base or with .sli/.hdr suffix)
    dense_wavelength_nm : optional custom wavelength grid; defaults to
        4001 bands uniformly from 7000–16000 nm
    samples_per_epoch : number of samples per epoch
    n_bands_range : (min, max) bands for the random virtual sensor
    seed : RNG seed
    """

    def __init__(
        self,
        library_path: str | Path,
        dense_wavelength_nm: np.ndarray | None = None,
        samples_per_epoch: int = 10_000,
        n_bands_range: tuple[int, int] = (3, 200),
        fwhm_range_nm: tuple[float, float] = (10.0, 200.0),
        seed: int = 42,
    ) -> None:
        # Load library
        lib_wl, lib_emiss, lib_names = read_envi_sli(library_path)

        # Each spectrum is its own material class (10k-way identification).
        # The model must learn to distinguish all individual library members.
        self.n_material_classes = lib_emiss.shape[0]

        # Also keep category-level info for diagnostics / grouping.
        categories = [n.rsplit("_", 1)[0] for n in lib_names]
        self.category_names = sorted(set(categories))
        name_to_id = {c: i for i, c in enumerate(self.category_names)}
        self.category_id_by_spec = np.array(
            [name_to_id[c] for c in categories], dtype=np.int64,
        )

        # Dense wavelength grid (for simulation and targets)
        if dense_wavelength_nm is None:
            dense_wavelength_nm = np.linspace(
                LWIR_WL_LO_NM, LWIR_WL_HI_NM, DEFAULT_N_WAVELENGTHS,
            )
        self.dense_wl = np.asarray(dense_wavelength_nm, dtype=np.float32)
        self.wl_range_nm = (float(self.dense_wl[0]), float(self.dense_wl[-1]))

        # Resample library emissivity onto the dense grid
        self.emissivity_matrix = np.zeros(
            (lib_emiss.shape[0], self.dense_wl.shape[0]), dtype=np.float32,
        )
        for i in range(lib_emiss.shape[0]):
            self.emissivity_matrix[i] = np.interp(
                self.dense_wl, lib_wl, lib_emiss[i],
            ).astype(np.float32)
        self.emissivity_matrix = np.clip(self.emissivity_matrix, 0.0, 1.0)
        self.n_spectra = self.emissivity_matrix.shape[0]

        self.samples_per_epoch = samples_per_epoch
        self.n_bands_range = n_bands_range
        self.fwhm_range_nm = fwhm_range_nm
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_atmos(self) -> AtmosphericState:
        """LWIR-relevant atmospheric state."""
        return AtmosphericState(
            aod_550=0.05,  # ~irrelevant at LWIR
            water_vapour=float(self.rng.uniform(0.2, 5.0)),
            ozone_du=float(self.rng.uniform(200.0, 500.0)),
            co2_ppmv=float(self.rng.uniform(380.0, 450.0)),
            ch4_ppbv=1900.0,
            n2o_ppbv=330.0,
            co_ppbv=150.0,
            visibility_km=23.0,
            surface_altitude_km=float(self.rng.uniform(0.0, 3.0)),
            surface_temperature_k=float(self.rng.uniform(260.0, 330.0)),
        )

    def _sample_geometry(self) -> ViewGeometry:
        """Only sensor zenith matters in LWIR (no solar path)."""
        return ViewGeometry(
            solar_zenith_deg=90.0,  # night — no solar contribution
            sensor_zenith_deg=float(self.rng.uniform(0.0, 45.0)),
            relative_azimuth_deg=0.0,
            sensor_altitude_km=float(self.rng.choice([20.0, 100.0, 400.0, 700.0, 800.0])),
        )

    def __getitem__(self, idx: int) -> dict:
        # 1. Surface emissivity from library
        spec_idx = int(self.rng.integers(0, self.n_spectra))
        emissivity = self.emissivity_matrix[spec_idx]
        # Reflectance = 1 - emissivity for opaque surfaces (Kirchhoff)
        reflectance = np.clip(1.0 - emissivity, 0.0, 1.0).astype(np.float32)

        # 2. Atmosphere + geometry
        atmos = self._sample_atmos()
        geom = self._sample_geometry()

        # 3. Run RTM — simplified two-stream with thermal emission
        result = simplified_toa_radiance(
            surface_reflectance=reflectance,
            wavelength_nm=self.dense_wl,
            atmos=atmos,
            geometry=geom,
        )
        dense_radiance = result.toa_radiance.astype(np.float32)

        # 4. Random virtual LWIR sensor
        sensor = sample_virtual_sensor(
            self.rng,
            n_bands_range=self.n_bands_range,
            wavelength_range=self.wl_range_nm,
            fwhm_range=self.fwhm_range_nm,
        )

        # 5. Convolve + add noise
        band_radiance = apply_sensor(sensor, self.dense_wl, dense_radiance)
        band_radiance = add_sensor_noise(band_radiance, self.rng).astype(np.float32)

        # 6. LWIR atmos parameter target vector (normalised):
        #    [water_vapour/5, ozone/500, (CO2-400)/50, sensor_zenith/60]
        atmos_params = np.array(
            [
                atmos.water_vapour / 5.0,
                atmos.ozone_du / 500.0,
                (atmos.co2_ppmv - 400.0) / 50.0,
                geom.sensor_zenith_deg / 60.0,
            ],
            dtype=np.float32,
        )

        # Note: target_reflectance is used by the model's reflectance head
        # and its loss.  We train the head to predict *emissivity* directly
        # because that's the physically meaningful surface property in LWIR.
        # The refl-PCA basis will be fit on emissivity, not reflectance.
        return {
            "wavelength": torch.from_numpy(sensor.center_wavelength_nm),
            "fwhm": torch.from_numpy(sensor.fwhm_nm),
            "radiance": torch.from_numpy(band_radiance),
            "target_wavelength": torch.from_numpy(self.dense_wl),
            "target_radiance": torch.from_numpy(dense_radiance),
            "target_reflectance": torch.from_numpy(emissivity),  # = emissivity in LWIR mode
            "atmos_params": torch.from_numpy(atmos_params),
            "surface_temperature_k": torch.tensor(
                atmos.surface_temperature_k or 300.0, dtype=torch.float32,
            ),
            "material_idx": spec_idx,  # spectrum ID (0–N), not class ID
        }


def collate_lwir_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-length LWIR samples (shared logic with VNIR collate)."""
    from spectralnp.data.dataset import collate_spectral_batch
    return collate_spectral_batch(samples)
