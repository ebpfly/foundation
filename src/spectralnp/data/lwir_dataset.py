"""LWIR (7–16 μm) training dataset.

Two atmospheric models:

  **Pool mode** (recommended): loads a pre-generated pool of
  (τ, Lup, Ldown) computed by ARTS via scripts/generate_atm_pool.py.
  At training time, picks a random atmosphere from the pool, applies a
  small random perturbation, and computes radiance analytically:

      L_TOA(λ) = τ(λ) · [ε(λ) · B(λ, T_s) + (1-ε(λ)) · L_down(λ)] + L_up(λ)

  This is both fast and atmospherically diverse.

  **Fallback mode**: uses the simplified two-stream RTM from
  spectralnp.data.rtm_simulator when no atmosphere pool is available.
"""

from __future__ import annotations

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


# Default training grid: 7–16 μm at 2.25 nm spacing = 4001 bands
LWIR_WL_LO_NM = 7000.0
LWIR_WL_HI_NM = 16000.0
DEFAULT_N_WAVELENGTHS = 4001

# Physical constants
_C = 2.99792458e8   # speed of light [m/s]
_H = 6.62607015e-34
_K = 1.380649e-23


def _planck_um(wavelength_nm: np.ndarray, T: float) -> np.ndarray:
    """Planck spectral radiance B(λ, T) in W/(m²·sr·μm)."""
    lam_m = wavelength_nm * 1e-9
    x = np.minimum(_H * _C / (lam_m * _K * T), 500.0)
    # B in W/(m²·sr·m)
    B_m = (2.0 * _H * _C**2 / lam_m**5) / (np.exp(x) - 1.0)
    # Convert to W/(m²·sr·μm)
    return B_m * 1e-6


class LWIRDataset(Dataset):
    """On-the-fly LWIR training dataset.

    Parameters
    ----------
    library_path : path to ENVI .sli library
    atm_pool_path : path to atmosphere pool .npz (from generate_atm_pool.py).
        If None, falls back to simplified RTM.
    dense_wavelength_nm : custom wavelength grid
    samples_per_epoch : number of samples per epoch
    n_bands_range : (min, max) bands for random virtual sensor
    fwhm_range_nm : (min, max) FWHM in nm
    atm_perturb_std : std of log-normal perturbation applied to pool
        atmospheres (0 = no perturbation, 0.1 = ~10% jitter)
    seed : RNG seed
    """

    def __init__(
        self,
        library_path: str | Path,
        atm_pool_path: str | Path | None = None,
        dense_wavelength_nm: np.ndarray | None = None,
        samples_per_epoch: int = 10_000,
        n_bands_range: tuple[int, int] = (3, 200),
        fwhm_range_nm: tuple[float, float] = (10.0, 200.0),
        atm_perturb_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        # Load emissivity library
        lib_wl, lib_emiss, lib_names = read_envi_sli(library_path)

        # Each spectrum is its own class (10k-way identification)
        self.n_material_classes = lib_emiss.shape[0]

        categories = [n.rsplit("_", 1)[0] for n in lib_names]
        self.category_names = sorted(set(categories))
        name_to_id = {c: i for i, c in enumerate(self.category_names)}
        self.category_id_by_spec = np.array(
            [name_to_id[c] for c in categories], dtype=np.int64,
        )

        # Dense wavelength grid
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

        # Load atmosphere pool (if provided)
        self.atm_pool = None
        self.atm_perturb_std = atm_perturb_std
        if atm_pool_path is not None:
            pool_path = Path(atm_pool_path)
            if pool_path.exists():
                pool = np.load(pool_path)
                pool_wl = pool["wavelength_nm"]
                # Resample pool to our dense grid
                n_pool = pool["tau"].shape[0]
                self.atm_pool = {
                    "tau": np.zeros((n_pool, len(self.dense_wl)), dtype=np.float32),
                    "lup": np.zeros((n_pool, len(self.dense_wl)), dtype=np.float32),
                    "ldn": np.zeros((n_pool, len(self.dense_wl)), dtype=np.float32),
                }
                for i in range(n_pool):
                    self.atm_pool["tau"][i] = np.interp(
                        self.dense_wl, pool_wl, pool["tau"][i],
                    )
                    self.atm_pool["lup"][i] = np.interp(
                        self.dense_wl, pool_wl, pool["lup"][i],
                    )
                    self.atm_pool["ldn"][i] = np.interp(
                        self.dense_wl, pool_wl, pool["ldn"][i],
                    )
                # Clamp transmission to [0, 1]
                self.atm_pool["tau"] = np.clip(self.atm_pool["tau"], 0.0, 1.0)
                self.atm_pool["lup"] = np.clip(self.atm_pool["lup"], 0.0, None)
                self.atm_pool["ldn"] = np.clip(self.atm_pool["ldn"], 0.0, None)

        self.samples_per_epoch = samples_per_epoch
        self.n_bands_range = n_bands_range
        self.fwhm_range_nm = fwhm_range_nm
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _radiance_from_pool(
        self, emissivity: np.ndarray, T_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute TOA radiance using a random atmosphere from the pool.

        Returns (dense_radiance, atmos_params_normalised).
        """
        pool = self.atm_pool
        n_pool = pool["tau"].shape[0]
        idx = int(self.rng.integers(0, n_pool))

        tau = pool["tau"][idx].copy()
        lup = pool["lup"][idx].copy()
        ldn = pool["ldn"][idx].copy()

        # Random perturbation so we never see the exact same atm twice
        if self.atm_perturb_std > 0:
            scale_tau = np.exp(
                self.rng.normal(0, self.atm_perturb_std, size=tau.shape)
            ).astype(np.float32)
            scale_rad = np.exp(
                self.rng.normal(0, self.atm_perturb_std, size=tau.shape)
            ).astype(np.float32)
            tau = np.clip(tau * scale_tau, 0.0, 1.0)
            lup = lup * scale_rad
            ldn = ldn * scale_rad

        # TOA radiance: L = τ·(ε·B(T_s) + (1-ε)·Ldn) + Lup
        B_Ts = _planck_um(self.dense_wl, T_s)
        surface_leaving = emissivity * B_Ts + (1.0 - emissivity) * ldn
        radiance = tau * surface_leaving + lup

        # Normalised atmos params — we don't have exact scalar values
        # from the pool, so use summary statistics as targets.
        # tau_mean serves as a proxy for total atmospheric opacity.
        atmos_params = np.array([
            tau.mean(),           # mean transmittance [0-1]
            lup.mean() / 10.0,   # normalised upwelling
            ldn.mean() / 10.0,   # normalised downwelling
            0.0,                  # placeholder (sensor zenith not stored in pool)
        ], dtype=np.float32)

        return radiance.astype(np.float32), atmos_params

    def _radiance_from_rtm(
        self, emissivity: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Fallback: compute radiance using simplified two-stream RTM."""
        from spectralnp.data.rtm_simulator import (
            AtmosphericState,
            ViewGeometry,
            simplified_toa_radiance,
        )
        reflectance = np.clip(1.0 - emissivity, 0.0, 1.0).astype(np.float32)
        atmos = AtmosphericState(
            aod_550=0.05,
            water_vapour=float(self.rng.uniform(0.2, 5.0)),
            ozone_du=float(self.rng.uniform(200, 500)),
            co2_ppmv=float(self.rng.uniform(380, 450)),
            visibility_km=23.0,
            surface_altitude_km=float(self.rng.uniform(0, 3)),
            surface_temperature_k=float(self.rng.uniform(260, 330)),
        )
        geom = ViewGeometry(
            solar_zenith_deg=90.0,
            sensor_zenith_deg=float(self.rng.uniform(0, 45)),
            relative_azimuth_deg=0.0,
            sensor_altitude_km=float(self.rng.choice([20, 100, 400, 700, 800])),
        )
        result = simplified_toa_radiance(
            surface_reflectance=reflectance,
            wavelength_nm=self.dense_wl,
            atmos=atmos,
            geometry=geom,
        )
        atmos_params = np.array([
            atmos.water_vapour / 5.0,
            atmos.ozone_du / 500.0,
            (atmos.co2_ppmv - 400.0) / 50.0,
            geom.sensor_zenith_deg / 60.0,
        ], dtype=np.float32)
        return (
            result.toa_radiance.astype(np.float32),
            atmos.surface_temperature_k,
            atmos_params,
        )

    def __getitem__(self, idx: int) -> dict:
        # 1. Random surface emissivity
        spec_idx = int(self.rng.integers(0, self.n_spectra))
        emissivity = self.emissivity_matrix[spec_idx]

        # 2. Random surface temperature
        T_s = float(self.rng.uniform(260.0, 330.0))

        # 3. Compute TOA radiance
        if self.atm_pool is not None:
            dense_radiance, atmos_params = self._radiance_from_pool(emissivity, T_s)
        else:
            dense_radiance, T_s, atmos_params = self._radiance_from_rtm(emissivity)

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

        return {
            "wavelength": torch.from_numpy(sensor.center_wavelength_nm),
            "fwhm": torch.from_numpy(sensor.fwhm_nm),
            "radiance": torch.from_numpy(band_radiance),
            "target_wavelength": torch.from_numpy(self.dense_wl),
            "target_radiance": torch.from_numpy(dense_radiance),
            "target_reflectance": torch.from_numpy(emissivity),
            "atmos_params": torch.from_numpy(atmos_params),
            "surface_temperature_k": torch.tensor(T_s, dtype=torch.float32),
            "material_idx": spec_idx,
        }


def collate_lwir_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-length LWIR samples."""
    from spectralnp.data.dataset import collate_spectral_batch
    return collate_spectral_batch(samples)
