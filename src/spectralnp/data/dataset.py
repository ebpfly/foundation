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
        n_bands_range: tuple[int, int] = (3, 425),
        lut_path: str | Path | None = None,
        arts_simulator=None,
        pca_vae_path: str | Path | None = None,
        use_full_rtm: bool = False,
        seed: int = 42,
    ) -> None:
        self.speclib = spectral_library
        self.n_spectra = len(spectral_library)
        if self.n_spectra == 0:
            raise ValueError("Spectral library is empty")

        # Build category-id-by-spectrum-index lookup. The model classifies
        # by USGS *category* (minerals, vegetation, ...) rather than by
        # individual spectrum (which would just be memorisation).
        category_by_idx = [s.category for s in spectral_library.spectra]
        self.category_names = sorted(set(category_by_idx))
        name_to_id = {n: i for i, n in enumerate(self.category_names)}
        self.category_id_by_spec = np.array(
            [name_to_id[c] for c in category_by_idx], dtype=np.int64
        )
        self.n_material_classes = len(self.category_names)

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

        # USGS reflectance is mostly VNIR/SWIR (380-2500 nm). For wavelengths
        # outside the measured range we fill with a graybody value (~0.04)
        # which corresponds to emissivity ~0.96 — typical for natural
        # surfaces in MWIR/LWIR. Filling with 0 (the old default) implied
        # ε=1 (perfect blackbody), which is physically wrong.
        GRAYBODY_FILL = 0.04

        # Pre-resample all spectra to the dense grid.
        self.reflectance_matrix = spectral_library.to_array(self.dense_wl)  # (N, W)
        self.reflectance_matrix = np.nan_to_num(self.reflectance_matrix, nan=GRAYBODY_FILL)
        # Clip to physical range.
        self.reflectance_matrix = np.clip(self.reflectance_matrix, 0.0, 1.0).astype(np.float32)

        # If LUT is used, also pre-resample spectra onto LUT wavelength grid.
        if self.lut is not None:
            self._lut_reflectance = spectral_library.to_array(self.lut.wavelength_nm)
            self._lut_reflectance = np.nan_to_num(self._lut_reflectance, nan=GRAYBODY_FILL)
            self._lut_reflectance = np.clip(self._lut_reflectance, 0.0, 1.0).astype(np.float32)

        # ARTS abs_lookup-based simulator (with per-atmosphere caching).
        self._arts_sim = arts_simulator
        if self._arts_sim is not None:
            # Trigger workspace init so wavelength_nm is populated.
            self._arts_sim._init_ws()
            self._arts_reflectance = spectral_library.to_array(self._arts_sim.wavelength_nm)
            self._arts_reflectance = np.nan_to_num(self._arts_reflectance, nan=GRAYBODY_FILL)
            self._arts_reflectance = np.clip(self._arts_reflectance, 0.0, 1.0).astype(np.float32)

        # Optional PCA-VAE generator for novel spectra (anti-memorisation).
        self._pca_vae = None
        self._pca_vae_bank = None  # pre-generated numpy spectra (fork-safe)
        self._pca_vae_wl_lo = 350.0
        self._pca_vae_wl_hi = 2500.0
        if pca_vae_path is not None:
            import torch
            from spectralnp.model.pca_vae import PCAVAE

            ckpt = torch.load(pca_vae_path, map_location="cpu", weights_only=False)
            vae = PCAVAE(ckpt["config"])
            for k in ["pca_mean", "pca_components", "pca_singular_values",
                       "z_mean", "z_cholesky"]:
                if k in ckpt["model_state_dict"]:
                    setattr(vae, k,
                            torch.zeros_like(ckpt["model_state_dict"][k]))
            vae.load_state_dict(ckpt["model_state_dict"])
            vae.eval()
            # wavelength grid is stored directly in the checkpoint.
            vae_wl = ckpt.get("wavelength_nm")
            if vae_wl is not None:
                vae_wl = vae_wl if isinstance(vae_wl, np.ndarray) else np.array(vae_wl)
                self._pca_vae_wl_lo = float(vae_wl[0])
                self._pca_vae_wl_hi = float(vae_wl[-1])
            # Pre-generate a bank of spectra as numpy (fork-safe, no PyTorch
            # model in workers). Regenerate each epoch via refresh_vae_bank().
            with torch.no_grad():
                self._pca_vae_bank = vae.generate(n_samples=5000).numpy()
            del vae  # don't keep the PyTorch model — prevents fork deadlock

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_atmospheric_state(self) -> AtmosphericState:
        """Sample a random atmospheric state from physically plausible ranges.

        Modes (mutually exclusive):
        - **scene-based**: if the simulator has pre-cached scenes, return the
          AtmosphericState from a randomly chosen scene. The dataset's sample
          handling reads :attr:`_arts_sim._scenes` directly via
          :meth:`_arts_sim.random_scene_index`.
        - **per-atm**: if the simulator has pre-cached τ tables (no scenes),
          draw (wv, oz, alt) from the cached tuples + continuous geometry.
        - **continuous**: fully random, uses simplified RTM.
        """
        if self._arts_sim is not None and self._arts_sim._available_states:
            wv, oz, alt = self._arts_sim.random_atmospheric_values(self.rng)
            return AtmosphericState(
                aod_550=self.rng.uniform(0.01, 1.0),
                water_vapour=wv,
                ozone_du=oz,
                co2_ppmv=self._arts_sim.RANDOM_GAS_DEFAULTS["co2_ppmv"],
                ch4_ppbv=self._arts_sim.RANDOM_GAS_DEFAULTS["ch4_ppbv"],
                n2o_ppbv=self._arts_sim.RANDOM_GAS_DEFAULTS["n2o_ppbv"],
                co_ppbv=self._arts_sim.RANDOM_GAS_DEFAULTS["co_ppbv"],
                visibility_km=self.rng.uniform(5, 100),
                surface_altitude_km=alt,
                surface_temperature_k=self.rng.uniform(260, 330),
            )
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

    def _augment_reflectance(self, reflectance: np.ndarray) -> np.ndarray:
        """Augment a surface reflectance spectrum to increase diversity.

        Without augmentation, the model memorizes the 1748 USGS spectra
        (factor 6.63× on training vs 1.27× on held-out). Augmentation
        forces the model to learn spectral physics, not a lookup table.

        If a PCA-VAE generator is loaded (``self._pca_vae``), 50% of
        samples are replaced entirely with novel generated spectra.
        """
        # 50% chance: use a novel spectrum from pre-generated PCA-VAE bank
        if self._pca_vae_bank is not None and self.rng.random() < 0.5:
            idx = self.rng.integers(len(self._pca_vae_bank))
            gen = self._pca_vae_bank[idx]
            # Resample from the VAE's native grid to the dense grid.
            vae_wl = np.linspace(
                self._pca_vae_wl_lo, self._pca_vae_wl_hi,
                gen.shape[0],
            )
            refl = np.interp(self.dense_wl, vae_wl, gen).astype(np.float32)
            return np.clip(refl, 0.0, 1.0).astype(np.float32)

        refl = reflectance.copy()

        # 1. Spectral mixing: blend with another random spectrum (50% chance)
        if self.rng.random() < 0.5:
            other_idx = self.rng.integers(0, self.n_spectra)
            other = self.reflectance_matrix[other_idx]
            alpha = self.rng.uniform(0.2, 0.8)
            refl = alpha * refl + (1 - alpha) * other

        # 2. Random scale + offset (always applied)
        scale = self.rng.uniform(0.7, 1.3)
        offset = self.rng.uniform(-0.03, 0.03)
        refl = refl * scale + offset

        # 3. Per-wavelength Gaussian noise (always applied, small)
        noise_std = self.rng.uniform(0.005, 0.02)
        refl = refl + self.rng.normal(0, noise_std, size=refl.shape).astype(np.float32)

        return np.clip(refl, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 1. Sample a surface reflectance + augment.
        spec_idx = self.rng.integers(0, self.n_spectra)
        reflectance = self._augment_reflectance(self.reflectance_matrix[spec_idx])

        # 2. Simulate at-sensor radiance.
        if self._arts_sim is not None and self._arts_sim._scenes:
            # ---- Scene-based fast path ----
            # Atmosphere + geometry + aerosol all come from a precomputed scene.
            # Per-sample work is just T_s + reflectance + the surface coupling.
            scene_idx = self._arts_sim.random_scene_index(self.rng)
            scene = self._arts_sim._scenes[scene_idx]
            T_s = float(self.rng.uniform(260.0, 330.0))
            arts_refl = self._arts_reflectance[spec_idx].astype(np.float64)
            result = self._arts_sim.simulate_with_scene(
                scene_idx=scene_idx,
                surface_reflectance=arts_refl,
                surface_temperature_k=T_s,
            )
            scene_atmos = scene["atmos"]
            geometry = scene["geometry"]
            atmos = AtmosphericState(
                aod_550=scene_atmos.aod_550,
                water_vapour=scene_atmos.water_vapour,
                ozone_du=scene_atmos.ozone_du,
                co2_ppmv=scene_atmos.co2_ppmv,
                ch4_ppbv=scene_atmos.ch4_ppbv,
                n2o_ppbv=scene_atmos.n2o_ppbv,
                co_ppbv=scene_atmos.co_ppbv,
                visibility_km=23.0,  # not used in scene-based RT
                surface_altitude_km=scene_atmos.surface_altitude_km,
                surface_temperature_k=T_s,
            )
            dense_radiance = np.interp(
                self.dense_wl, self._arts_sim.wavelength_nm, result.toa_radiance
            ).astype(np.float32)
        else:
            # ---- Slower paths (full per-sample RT) ----
            atmos = self._sample_atmospheric_state()
            geometry = self._sample_geometry()
            if self._arts_sim is not None:
                arts_refl = self._arts_reflectance[spec_idx].astype(np.float64)
                result = self._arts_sim.simulate(
                    surface_reflectance=arts_refl,
                    atmos=atmos,
                    geometry=geometry,
                )
                dense_radiance = np.interp(
                    self.dense_wl, self._arts_sim.wavelength_nm, result.toa_radiance
                ).astype(np.float32)
            elif self.lut is not None:
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

        # Return numpy arrays (not torch tensors) so that forked DataLoader
        # workers never touch the PyTorch runtime.  Conversion to tensors
        # happens in collate_spectral_batch(), which runs in the main process.
        return {
            "wavelength": sensor.center_wavelength_nm,
            "fwhm": sensor.fwhm_nm,
            "radiance": band_radiance,
            "target_wavelength": self.dense_wl,
            "target_radiance": dense_radiance,
            "target_reflectance": reflectance,
            "atmos_params": atmos_params,
            "material_idx": int(self.category_id_by_spec[spec_idx]),
        }


def collate_spectral_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-length spectral samples into a padded batch.

    Pads input bands to the maximum number in the batch and creates
    a pad_mask.  Target dense spectra are already fixed-length.

    Accepts numpy arrays or torch tensors from __getitem__ — all
    tensor conversion happens here (in the main process) to avoid
    touching PyTorch in forked DataLoader workers on macOS/MPS.
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
        # Support both numpy arrays and torch tensors.
        _to_t = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        wavelength[i, :n] = _to_t(s["wavelength"])
        fwhm[i, :n] = _to_t(s["fwhm"])
        radiance[i, :n] = _to_t(s["radiance"])
        pad_mask[i, :n] = True
        target_wavelength[i] = _to_t(s["target_wavelength"])
        target_radiance[i] = _to_t(s["target_radiance"])
        target_reflectance[i] = _to_t(s["target_reflectance"])
        atmos_params[i] = _to_t(s["atmos_params"])
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
