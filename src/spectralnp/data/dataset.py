"""PyTorch dataset for training SpectralNP.

Generates training samples on-the-fly:
1. Sample a surface reflectance from the USGS spectral library
2. Sample random atmospheric state and viewing geometry
3. Simulate at-sensor radiance (via LUT, full ARTS, or simplified model)
4. Generate a random virtual sensor (augmentation)
5. Convolve radiance with virtual sensor SRFs + add noise
6. Return (input bands, target dense spectrum, atmospheric params)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

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
from spectralnp.data.usgs_speclib import SpectralLibrary


@dataclass
class SpectralSample:
    """A single training sample."""

    # Input: what the sensor observes.
    wavelength: torch.Tensor     # (N_bands,) center wavelengths nm
    fwhm: torch.Tensor           # (N_bands,) band widths nm
    radiance: torch.Tensor       # (N_bands,) at-sensor radiance

    # Target: dense spectrum for reconstruction loss.
    target_wavelength: torch.Tensor   # (N_dense,) nm
    target_radiance: torch.Tensor     # (N_dense,) at-sensor radiance
    target_reflectance: torch.Tensor  # (N_dense,) surface reflectance

    # Target: atmospheric parameters.
    atmos_params: torch.Tensor   # (4,) [AOD, water_vapour, ozone_du, visibility]

    # Metadata.
    material_idx: int = 0


class SpectralNPDataset(Dataset):
    """On-the-fly training dataset for SpectralNP.

    Parameters
    ----------
    spectral_library : SpectralLibrary
        USGS spectra providing surface reflectances.
    dense_wavelength_nm : array-like
        The dense wavelength grid for simulation and reconstruction targets.
    samples_per_epoch : int
        Number of samples per "epoch" (since data is generated on-the-fly).
    n_bands_range : (min, max) bands for random sensor.
    lut_path : str or Path, optional
        Path to a pre-computed ARTS LUT (HDF5).  When provided, samples use
        the LUT for radiative transfer instead of the simplified model.
    use_full_rtm : bool
        If True, use PyARTS for simulation (slow but accurate).
        If False, use the simplified two-stream model.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        spectral_library: SpectralLibrary,
        dense_wavelength_nm: np.ndarray | None = None,
        samples_per_epoch: int = 100_000,
        n_bands_range: tuple[int, int] = (3, 200),
        lut_path: str | Path | None = None,
        use_full_rtm: bool = False,
        seed: int = 42,
    ) -> None:
        self.speclib = spectral_library
        self.n_spectra = len(spectral_library)
        if self.n_spectra == 0:
            raise ValueError("Spectral library is empty")

        # Load LUT if provided.
        self.lut = None
        if lut_path is not None:
            from spectralnp.data.lut import SpectralLUT

            self.lut = SpectralLUT(lut_path)

        if dense_wavelength_nm is None:
            dense_wavelength_nm = np.arange(380.0, 2501.0, 5.0)
        self.dense_wl = np.asarray(dense_wavelength_nm, dtype=np.float32)
        self.samples_per_epoch = samples_per_epoch
        self.n_bands_range = n_bands_range
        self.use_full_rtm = use_full_rtm
        self.rng = np.random.default_rng(seed)

        # Pre-resample all spectra to the dense grid.
        self.reflectance_matrix = spectral_library.to_array(self.dense_wl)  # (N, W)
        # Replace NaN with 0 (out-of-range wavelengths).
        self.reflectance_matrix = np.nan_to_num(self.reflectance_matrix, nan=0.0)
        # Clip to physical range.
        self.reflectance_matrix = np.clip(self.reflectance_matrix, 0.0, 1.0).astype(np.float32)

        # If LUT is used, also pre-resample spectra onto LUT wavelength grid.
        if self.lut is not None:
            self._lut_reflectance = spectral_library.to_array(self.lut.wavelength_nm)
            self._lut_reflectance = np.nan_to_num(self._lut_reflectance, nan=0.0)
            self._lut_reflectance = np.clip(self._lut_reflectance, 0.0, 1.0).astype(np.float32)

        self._arts_sim = None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_atmospheric_state(self) -> AtmosphericState:
        """Sample a random atmospheric state from physically plausible ranges."""
        return AtmosphericState(
            aod_550=self.rng.uniform(0.01, 1.0),
            water_vapour=self.rng.uniform(0.2, 5.0),
            ozone_du=self.rng.uniform(200, 500),
            co2_ppmv=self.rng.uniform(400, 450),
            ch4_ppbv=self.rng.uniform(1800, 2000),
            n2o_ppbv=self.rng.uniform(320, 345),
            co_ppbv=self.rng.uniform(60, 250),
            visibility_km=self.rng.uniform(5, 100),
            surface_altitude_km=self.rng.uniform(0, 3),
            surface_temperature_k=self.rng.uniform(260, 330),
        )

    def _sample_geometry(self) -> ViewGeometry:
        """Sample random viewing geometry."""
        return ViewGeometry(
            solar_zenith_deg=self.rng.uniform(10, 70),
            sensor_zenith_deg=self.rng.uniform(0, 30),
            relative_azimuth_deg=self.rng.uniform(0, 180),
            sensor_altitude_km=self.rng.choice([20, 100, 400, 700, 800]),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 1. Sample a surface reflectance.
        spec_idx = self.rng.integers(0, self.n_spectra)
        reflectance = self.reflectance_matrix[spec_idx]  # (W,)

        # 2. Sample atmospheric state and geometry.
        atmos = self._sample_atmospheric_state()
        geometry = self._sample_geometry()

        # 3. Simulate at-sensor radiance.
        if self.lut is not None:
            # LUT-based RT: compute on LUT grid, resample to dense grid.
            lut_refl = self._lut_reflectance[spec_idx].astype(np.float64)
            lut_radiance = self.lut.toa_radiance(
                lut_refl,
                water_vapour=atmos.water_vapour,
                ozone_du=atmos.ozone_du,
                co2_ppmv=atmos.co2_ppmv,
                ch4_ppbv=atmos.ch4_ppbv,
                n2o_ppbv=atmos.n2o_ppbv,
                co_ppbv=atmos.co_ppbv,
                surface_altitude_km=atmos.surface_altitude_km,
                solar_zenith_deg=geometry.solar_zenith_deg,
                sensor_zenith_deg=geometry.sensor_zenith_deg,
                relative_azimuth_deg=geometry.relative_azimuth_deg,
                sensor_altitude_km=geometry.sensor_altitude_km,
                surface_temperature_k=atmos.surface_temperature_k or 300.0,
                aod_550=atmos.aod_550,
            )
            dense_radiance = np.interp(
                self.dense_wl, self.lut.wavelength_nm, lut_radiance
            ).astype(np.float32)
        else:
            result = simplified_toa_radiance(
                surface_reflectance=reflectance,
                wavelength_nm=self.dense_wl,
                atmos=atmos,
                geometry=geometry,
            )
            dense_radiance = result.toa_radiance.astype(np.float32)

        # 4. Generate random virtual sensor.
        sensor = sample_virtual_sensor(
            self.rng,
            n_bands_range=self.n_bands_range,
        )

        # 5. Convolve and add noise.
        band_radiance = apply_sensor(sensor, self.dense_wl, dense_radiance)
        band_radiance = add_sensor_noise(band_radiance, self.rng).astype(np.float32)

        # 6. Atmospheric parameter target vector.
        atmos_params = np.array([
            atmos.aod_550,
            atmos.water_vapour,
            atmos.ozone_du / 1000.0,  # normalise to ~[0, 0.5]
            atmos.visibility_km / 100.0,  # normalise to ~[0, 1]
        ], dtype=np.float32)

        return {
            "wavelength": torch.from_numpy(sensor.center_wavelength_nm),
            "fwhm": torch.from_numpy(sensor.fwhm_nm),
            "radiance": torch.from_numpy(band_radiance),
            "target_wavelength": torch.from_numpy(self.dense_wl),
            "target_radiance": torch.from_numpy(dense_radiance),
            "target_reflectance": torch.from_numpy(reflectance),
            "atmos_params": torch.from_numpy(atmos_params),
            "material_idx": spec_idx,
        }


def collate_spectral_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-length spectral samples into a padded batch.

    Pads input bands to the maximum number in the batch and creates
    a pad_mask.  Target dense spectra are already fixed-length.
    """
    max_bands = max(s["wavelength"].shape[0] for s in samples)
    batch_size = len(samples)

    # Dense targets are fixed-length.
    n_dense = samples[0]["target_wavelength"].shape[0]

    wavelength = torch.zeros(batch_size, max_bands)
    fwhm = torch.zeros(batch_size, max_bands)
    radiance = torch.zeros(batch_size, max_bands)
    pad_mask = torch.zeros(batch_size, max_bands, dtype=torch.bool)

    target_wavelength = torch.zeros(batch_size, n_dense)
    target_radiance = torch.zeros(batch_size, n_dense)
    target_reflectance = torch.zeros(batch_size, n_dense)
    atmos_params = torch.zeros(batch_size, 4)
    material_idx = torch.zeros(batch_size, dtype=torch.long)

    for i, s in enumerate(samples):
        n = s["wavelength"].shape[0]
        wavelength[i, :n] = s["wavelength"]
        fwhm[i, :n] = s["fwhm"]
        radiance[i, :n] = s["radiance"]
        pad_mask[i, :n] = True
        target_wavelength[i] = s["target_wavelength"]
        target_radiance[i] = s["target_radiance"]
        target_reflectance[i] = s["target_reflectance"]
        atmos_params[i] = s["atmos_params"]
        material_idx[i] = s["material_idx"]

    return {
        "wavelength": wavelength,
        "fwhm": fwhm,
        "radiance": radiance,
        "pad_mask": pad_mask,
        "target_wavelength": target_wavelength,
        "target_radiance": target_radiance,
        "target_reflectance": target_reflectance,
        "atmos_params": atmos_params,
        "material_idx": material_idx,
    }
