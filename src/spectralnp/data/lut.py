"""Radiative transfer lookup tables built from PyARTS simulations.

Caches the **expensive** part of atmospheric RT — line-by-line gas absorption
computed by ARTS — as layer optical depths.  The **cheap** part — path
integration (Beer's law, emission summation) — runs at inference time for
arbitrary sensor altitude, viewing geometry, and surface properties.

LUT contents per atmospheric state
-----------------------------------
  τ_layers   (n_layers, n_λ)  gas + Rayleigh optical depth per layer
  T_layers   (n_layers,)      temperature at each layer centre  [K]
  z_layers   (n_layers,)      altitude of each layer centre     [m]
  p_layers   (n_layers,)      pressure at each layer centre     [Pa]

Atmospheric state axes (7 dimensions, no geometry)
--------------------------------------------------
  water_vapour, ozone_du, co2_ppmv, ch4_ppbv, n2o_ppbv, co_ppbv,
  surface_altitude_km

All six absorbing gases (H₂O, CO₂, O₃, N₂O, CH₄, CO) plus O₂/N₂ CIA
are included for the 0.3–16 μm range.

Runtime path integration
------------------------
Given layer optical depths, the module computes transmittance, upwelling /
downwelling thermal emission, solar irradiance at the ground, and
surface-coupled TOA radiance — all vectorised over wavelength.  Rayleigh
scattering and aerosol are added analytically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_C = 2.99792458e8  # speed of light  [m s⁻¹]
_H = 6.62607015e-34  # Planck constant  [J s]
_K = 1.380649e-23  # Boltzmann constant  [J K⁻¹]
_SOLAR_OMEGA = 6.794e-5  # solar disk solid angle  [sr]

# Gas species for ARTS spanning 0.3–16 μm.
ARTS_ABS_SPECIES = [
    "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
    "CO2, CO2-CKDMT252",
    "O3",
    "N2O",
    "CH4",
    "CO",
    "O2-CIAfunCKDMT100",
    "N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def planck(wavelength_nm: np.ndarray, temperature_k: float) -> np.ndarray:
    """Planck spectral radiance *B*(λ, T) in W m⁻² sr⁻¹ μm⁻¹."""
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9  # → m
    x = np.minimum(_H * _C / (lam * _K * temperature_k), 500.0)
    return (2.0 * _H * _C**2 / lam**5) / (np.exp(x) - 1.0) * 1e-6


def planck_array(wavelength_nm: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Planck radiance for an array of temperatures.

    Parameters
    ----------
    wavelength_nm : (N_λ,)
    T : (N,) temperatures in K

    Returns
    -------
    (N, N_λ) array of B(λ, T) in W m⁻² sr⁻¹ μm⁻¹.
    """
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9  # (N_λ,)
    T = np.asarray(T, dtype=np.float64)
    # (N, N_λ)
    x = np.minimum(
        _H * _C / (lam[np.newaxis, :] * _K * T[:, np.newaxis]), 500.0
    )
    return (
        (2.0 * _H * _C**2 / lam[np.newaxis, :] ** 5)
        / (np.exp(x) - 1.0)
        * 1e-6
    )


def make_lut_wavelength_grid(
    wl_min: float = 300.0, wl_max: float = 16000.0
) -> np.ndarray:
    """Variable-resolution wavelength grid (nm) for 0.3–16 μm.

    UV-VIS    300– 700 nm  @  1 nm   ( 400 pts)
    NIR-SWIR  700–2500 nm  @  2 nm   ( 900 pts)
    MWIR     2500–5000 nm  @  5 nm   ( 500 pts)
    TIR      5000–16000 nm @ 10 nm   (1101 pts)
    Total  ≈ 2901 points
    """
    breaks = [
        (300.0, 700.0, 1.0),
        (700.0, 2500.0, 2.0),
        (2500.0, 5000.0, 5.0),
        (5000.0, 16001.0, 10.0),
    ]
    parts: list[np.ndarray] = []
    for lo, hi, step in breaks:
        a, b = max(lo, wl_min), min(hi, wl_max + 1)
        if a < b:
            parts.append(np.arange(a, b, step))
    return np.concatenate(parts).astype(np.float64)


# ---------------------------------------------------------------------------
# LUT configuration
# ---------------------------------------------------------------------------


@dataclass
class LUTConfig:
    """Grid definition for the atmospheric RT lookup table."""

    wavelength_nm: np.ndarray = field(default_factory=make_lut_wavelength_grid)

    # Atmospheric state axes.
    water_vapour: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.5, 1.0, 2.0, 3.5, 5.5])
    )
    ozone_du: np.ndarray = field(
        default_factory=lambda: np.array([200.0, 300.0, 450.0])
    )
    co2_ppmv: np.ndarray = field(
        default_factory=lambda: np.array([400.0, 420.0, 450.0])
    )
    ch4_ppbv: np.ndarray = field(
        default_factory=lambda: np.array([1800.0, 1900.0, 2000.0])
    )
    n2o_ppbv: np.ndarray = field(
        default_factory=lambda: np.array([320.0, 332.0, 345.0])
    )
    co_ppbv: np.ndarray = field(
        default_factory=lambda: np.array([80.0, 120.0, 200.0])
    )
    surface_altitude_km: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 2.0, 3.0])
    )

    # Number of atmospheric layers in ARTS.
    n_layers: int = 50
    # Top of atmosphere altitude [m].
    toa_m: float = 100e3

    _axis_names: tuple[str, ...] = (
        "water_vapour",
        "ozone_du",
        "co2_ppmv",
        "ch4_ppbv",
        "n2o_ppbv",
        "co_ppbv",
        "surface_altitude_km",
    )

    @property
    def axis_names(self) -> tuple[str, ...]:
        return self._axis_names

    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        return tuple(getattr(self, n) for n in self._axis_names)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the atmospheric-state grid (no geometry, no layers)."""
        return tuple(len(a) for a in self.axes)

    @property
    def n_grid_points(self) -> int:
        from math import prod

        return prod(self.shape)


# ---------------------------------------------------------------------------
# ARTS LUT generator
# ---------------------------------------------------------------------------


def _lut_worker(args):
    """Compute layer optical depths for one atmospheric state (multiprocessing)."""
    config, atmos_idx, atmos_vals, arts_data_path = args
    gen = ARTSLUTGenerator(config, arts_data_path)
    ws = gen._init_arts()
    tau, T_layer, z_layer, p_layer = gen._compute_layer_optics(ws, atmos_vals)
    return atmos_idx, tau, T_layer, z_layer, p_layer


class ARTSLUTGenerator:
    """Build a layer-optical-depth LUT using PyARTS (ARTS ≥ 2.6).

    For each atmospheric state, sets up the ARTS workspace and extracts
    the per-layer gas absorption + Rayleigh optical depth.  This is the
    computationally expensive step (line-by-line spectroscopy).  Path
    integration is done at runtime.

    Parameters
    ----------
    config : LUTConfig
        Grid definition.
    arts_data_path : str, optional
        Path to ARTS spectroscopic catalogues.
    """

    def __init__(
        self,
        config: LUTConfig | None = None,
        arts_data_path: str | None = None,
    ):
        self.config = config or LUTConfig()
        self._arts_data_path = arts_data_path

    def _init_arts(self):
        """Create and return a fresh ARTS workspace for 0.3–16 μm."""
        import pyarts

        if self._arts_data_path:
            pyarts.cat.download.retrieve(arts_data_path=self._arts_data_path)
        else:
            pyarts.cat.download.retrieve()

        ws = pyarts.Workspace()
        ws.atmosphere_dim = 1
        ws.stokes_dim = 1

        wl_m = self.config.wavelength_nm * 1e-9
        freq_hz = _C / wl_m
        ws.f_grid = np.sort(freq_hz)[::-1].copy()

        ws.abs_species = list(ARTS_ABS_SPECIES)
        ws.propmat_clearsky_agendaAuto()

        return ws

    def _setup_atmosphere(self, ws, atmos_vals: dict[str, float]):
        """Configure the 1-D atmosphere and scale gas profiles."""
        import pyarts

        ws.atm = pyarts.arts.AtmField.fromMeanState(
            ws.f_grid, toa=self.config.toa_m, nlay=self.config.n_layers
        )

        _REF = {
            "water_vapour": ("H2O", 1.5),
            "ozone_du": ("O3", 300.0),
            "co2_ppmv": ("CO2", 400.0),
            "ch4_ppbv": ("CH4", 1800.0),
            "n2o_ppbv": ("N2O", 332.0),
            "co_ppbv": ("CO", 120.0),
        }
        for key, (species, ref_val) in _REF.items():
            val = atmos_vals.get(key, ref_val)
            if val != ref_val:
                ws.atm[species] = ws.atm[species] * (val / ref_val)

        alt_km = atmos_vals.get("surface_altitude_km", 0.0)
        ws.z_surface = np.array([[alt_km * 1e3]])

    def _compute_layer_optics(
        self, ws, atmos_vals: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract per-layer optical depths from ARTS.

        Returns
        -------
        tau_layers : (n_layers, n_λ)  optical depth per layer
        T_layers   : (n_layers,)      temperature at layer centre [K]
        z_layers   : (n_layers,)      altitude at layer centre [m]
        p_layers   : (n_layers,)      pressure at layer centre [Pa]
        """
        import pyarts

        self._setup_atmosphere(ws, atmos_vals)

        n_wl = len(self.config.wavelength_nm)

        # Extract the atmospheric grid from the AtmField.
        atm = ws.atm
        z_grid = np.array(atm.grid("z"))  # altitude grid [m]
        T_grid = np.array(atm["t"])  # temperature [K]
        p_grid = np.array(atm["p"])  # pressure [Pa]

        # Layer centres: midpoints between grid levels.
        z_layers = 0.5 * (z_grid[:-1] + z_grid[1:])
        T_layers = 0.5 * (T_grid[:-1] + T_grid[1:])
        p_layers = np.sqrt(p_grid[:-1] * p_grid[1:])  # geometric mean
        dz = np.abs(z_grid[1:] - z_grid[:-1])  # layer thickness [m]

        # Compute absorption coefficients at each layer.
        # propmat_clearsky gives the propagation matrix (absorption + scattering)
        # at specified (f, T, p, VMR).
        tau_layers = np.zeros((len(z_layers), n_wl), dtype=np.float64)

        for i in range(len(z_layers)):
            # Set atmospheric point for this layer.
            ws.atm_point = pyarts.arts.AtmPoint(atm, z_layers[i])

            # Compute propagation matrix at all frequencies.
            ws.propmat_clearskyInit()
            ws.propmat_clearskyAddFromLookup()  # or Auto depending on setup

            # Extract absorption coefficient [1/m] from propagation matrix.
            # propmat_clearsky is (n_freq, stokes, stokes) → take [i,0,0].
            pm = np.array(ws.propmat_clearsky)
            kappa = pm[:, 0, 0]  # absorption coefficient [1/m]

            # Optical depth = κ · Δz  (vertical, nadir).
            tau_layers[i, :] = kappa * dz[i]

        return (
            tau_layers.astype(np.float32),
            T_layers.astype(np.float32),
            z_layers.astype(np.float32),
            p_layers.astype(np.float32),
        )

    def generate(self, output_path: str | Path, n_workers: int = 1) -> Path:
        """Build the LUT and write to HDF5.

        Parameters
        ----------
        output_path : where to save the ``.h5`` file.
        n_workers : parallel processes (1 = sequential).
        """
        cfg = self.config
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        n_wl = len(cfg.wavelength_nm)
        n_layers = cfg.n_layers

        # Allocate.  Shape: (*atmos_shape, n_layers, n_wl) for tau,
        # (*atmos_shape, n_layers) for T, z, p.
        tau_shape = (*cfg.shape, n_layers, n_wl)
        profile_shape = (*cfg.shape, n_layers)

        tau_all = np.zeros(tau_shape, np.float32)
        T_all = np.zeros(profile_shape, np.float32)
        z_all = np.zeros(profile_shape, np.float32)
        p_all = np.zeros(profile_shape, np.float32)

        # Build job list — one per atmospheric state.
        atmos_axis_names = cfg._axis_names
        atmos_grids = [getattr(cfg, n) for n in atmos_axis_names]
        jobs: list[tuple] = []
        for idx in np.ndindex(*[len(g) for g in atmos_grids]):
            vals = {
                name: float(grid[i])
                for name, grid, i in zip(atmos_axis_names, atmos_grids, idx)
            }
            jobs.append((cfg, idx, vals, self._arts_data_path))

        total = len(jobs)
        done = 0

        if n_workers <= 1:
            ws = self._init_arts()
            for _, atmos_idx, vals, _ in jobs:
                tau, T_lay, z_lay, p_lay = self._compute_layer_optics(ws, vals)
                tau_all[atmos_idx] = tau
                T_all[atmos_idx] = T_lay
                z_all[atmos_idx] = z_lay
                p_all[atmos_idx] = p_lay
                done += 1
                if done % 10 == 0 or done == total:
                    logger.info(
                        "LUT %d / %d (%.0f%%)", done, total, 100 * done / total
                    )
        else:
            from multiprocessing import Pool

            with Pool(n_workers) as pool:
                for result in pool.imap_unordered(_lut_worker, jobs):
                    atmos_idx, tau, T_lay, z_lay, p_lay = result
                    tau_all[atmos_idx] = tau
                    T_all[atmos_idx] = T_lay
                    z_all[atmos_idx] = z_lay
                    p_all[atmos_idx] = p_lay
                    done += 1
                    logger.info(
                        "LUT %d / %d (%.0f%%)", done, total, 100 * done / total
                    )

        self._write_hdf5(out, tau_all, T_all, z_all, p_all)
        logger.info("Wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
        return out

    def _write_hdf5(self, path, tau_all, T_all, z_all, p_all):
        cfg = self.config
        comp = dict(compression="gzip", compression_opts=4)
        with h5py.File(path, "w") as f:
            f.attrs["source"] = "spectralnp.data.lut.ARTSLUTGenerator"
            f.attrs["axis_order"] = list(cfg.axis_names)
            f.attrs["n_layers"] = cfg.n_layers
            f.attrs["toa_m"] = cfg.toa_m

            f.create_dataset("wavelength_nm", data=cfg.wavelength_nm)

            g = f.create_group("axes")
            for name in cfg.axis_names:
                g.create_dataset(name, data=getattr(cfg, name))

            f.create_dataset("tau_layers", data=tau_all, **comp)
            f.create_dataset("T_layers", data=T_all, **comp)
            f.create_dataset("z_layers", data=z_all, **comp)
            f.create_dataset("p_layers", data=p_all, **comp)


# ---------------------------------------------------------------------------
# LUT interpolator + runtime path integration
# ---------------------------------------------------------------------------


class SpectralLUT:
    """Load a layer-optical-depth LUT and compute RT at runtime.

    Interpolates layer optical depths across the atmospheric state grid,
    then performs fast vectorised path integration for any sensor altitude,
    viewing geometry, and surface properties.

    Parameters
    ----------
    lut_path : path to the HDF5 file produced by :class:`ARTSLUTGenerator`.
    """

    def __init__(self, lut_path: str | Path):
        with h5py.File(lut_path, "r") as f:
            self.wavelength_nm: np.ndarray = f["wavelength_nm"][:].astype(
                np.float64
            )
            self._axis_order: list[str] = list(f.attrs["axis_order"])
            self._axes = tuple(
                f[f"axes/{name}"][:].astype(np.float64)
                for name in self._axis_order
            )
            # (n_atmos..., n_layers, n_wl)
            self._tau = f["tau_layers"][:].astype(np.float64)
            # (n_atmos..., n_layers)
            self._T = f["T_layers"][:].astype(np.float64)
            self._z = f["z_layers"][:].astype(np.float64)
            self._p = f["p_layers"][:].astype(np.float64)

        self._build_interpolators()

    def _build_interpolators(self):
        from scipy.interpolate import RegularGridInterpolator as RGI

        # tau has shape (*atmos_shape, n_layers, n_wl) — treat last two dims
        # as "values" that get interpolated together.
        self._interp_tau = RGI(
            self._axes, self._tau,
            method="linear", bounds_error=False, fill_value=None,
        )
        self._interp_T = RGI(
            self._axes, self._T,
            method="linear", bounds_error=False, fill_value=None,
        )
        self._interp_z = RGI(
            self._axes, self._z,
            method="linear", bounds_error=False, fill_value=None,
        )

    @property
    def axis_order(self) -> list[str]:
        return list(self._axis_order)

    def interpolate(self, **kwargs: float):
        """Interpolate layer properties at a single atmospheric state.

        Returns
        -------
        tau_layers : (n_layers, n_λ)  optical depth per layer (nadir)
        T_layers   : (n_layers,)      temperature [K]
        z_layers   : (n_layers,)      altitude [m]
        """
        pt = np.array([[kwargs[name] for name in self._axis_order]])
        tau = self._interp_tau(pt).reshape(
            self._tau.shape[-2], self._tau.shape[-1]
        )
        T_lay = self._interp_T(pt).ravel()
        z_lay = self._interp_z(pt).ravel()
        return tau, T_lay, z_lay

    # ----------------------------------------------------------------
    # Runtime path integration
    # ----------------------------------------------------------------

    def toa_radiance(
        self,
        surface_reflectance: np.ndarray,
        *,
        water_vapour: float,
        ozone_du: float,
        co2_ppmv: float = 420.0,
        ch4_ppbv: float = 1900.0,
        n2o_ppbv: float = 332.0,
        co_ppbv: float = 120.0,
        surface_altitude_km: float = 0.0,
        solar_zenith_deg: float = 30.0,
        sensor_zenith_deg: float = 0.0,
        relative_azimuth_deg: float = 0.0,
        sensor_altitude_km: float = 800.0,
        surface_temperature_k: float = 300.0,
        aod_550: float = 0.0,
    ) -> np.ndarray:
        """Compute at-sensor radiance on the LUT wavelength grid.

        Performs full path integration through the atmosphere from the
        surface to the sensor altitude, including thermal emission at
        every layer.

        Returns (N_λ,) array in W m⁻² sr⁻¹ μm⁻¹.
        """
        tau_layers, T_layers, z_layers = self.interpolate(
            water_vapour=water_vapour,
            ozone_du=ozone_du,
            co2_ppmv=co2_ppmv,
            ch4_ppbv=ch4_ppbv,
            n2o_ppbv=n2o_ppbv,
            co_ppbv=co_ppbv,
            surface_altitude_km=surface_altitude_km,
        )

        return path_integrate(
            tau_layers=tau_layers,
            T_layers=T_layers,
            z_layers=z_layers,
            wavelength_nm=self.wavelength_nm,
            surface_reflectance=surface_reflectance,
            solar_zenith_deg=solar_zenith_deg,
            sensor_zenith_deg=sensor_zenith_deg,
            relative_azimuth_deg=relative_azimuth_deg,
            sensor_altitude_m=sensor_altitude_km * 1e3,
            surface_altitude_m=surface_altitude_km * 1e3,
            surface_temperature_k=surface_temperature_k,
            aod_550=aod_550,
        )

    def resample(
        self, spectrum: np.ndarray, target_wavelength_nm: np.ndarray
    ) -> np.ndarray:
        """Linearly interpolate *spectrum* from the LUT grid to *target*."""
        return np.interp(target_wavelength_nm, self.wavelength_nm, spectrum)


# ---------------------------------------------------------------------------
# Path integration engine
# ---------------------------------------------------------------------------


def path_integrate(
    tau_layers: np.ndarray,
    T_layers: np.ndarray,
    z_layers: np.ndarray,
    wavelength_nm: np.ndarray,
    surface_reflectance: np.ndarray,
    solar_zenith_deg: float = 30.0,
    sensor_zenith_deg: float = 0.0,
    relative_azimuth_deg: float = 0.0,
    sensor_altitude_m: float = 800e3,
    surface_altitude_m: float = 0.0,
    surface_temperature_k: float = 300.0,
    aod_550: float = 0.0,
) -> np.ndarray:
    """Vectorised path integration from layer optical depths.

    Parameters
    ----------
    tau_layers : (n_layers, n_λ)  nadir optical depth per layer
    T_layers   : (n_layers,)      layer temperature [K]
    z_layers   : (n_layers,)      layer altitude [m]
    wavelength_nm : (n_λ,)
    surface_reflectance : (n_λ,)
    solar_zenith_deg, sensor_zenith_deg, relative_azimuth_deg : geometry
    sensor_altitude_m : sensor altitude [m]
    surface_altitude_m : surface altitude [m]
    surface_temperature_k : surface skin temperature [K]
    aod_550 : aerosol optical depth at 550 nm (analytical)

    Returns
    -------
    (n_λ,) at-sensor radiance in W m⁻² sr⁻¹ μm⁻¹.
    """
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    rho = np.asarray(surface_reflectance, dtype=np.float64)
    n_wl = len(wl)

    cos_sza = np.cos(np.radians(solar_zenith_deg))
    cos_vza = max(np.cos(np.radians(sensor_zenith_deg)), 0.05)

    # Airmass factors for slant paths.
    mu_sun = max(cos_sza, 0.05)
    mu_sen = cos_vza

    # --- Select layers between surface and sensor ---
    mask = (z_layers >= surface_altitude_m) & (z_layers <= sensor_altitude_m)
    tau_col = tau_layers[mask]  # (n_active, n_λ)
    T_col = T_layers[mask]  # (n_active,)
    z_col = z_layers[mask]

    # Also need full column for solar path (sun → surface).
    mask_full = z_layers >= surface_altitude_m
    tau_full = tau_layers[mask_full]

    if len(tau_col) == 0:
        # Sensor below the lowest layer — return surface emission only.
        B_s = planck(wl, surface_temperature_k)
        return (1.0 - rho) * B_s

    # --- Aerosol optical depth per layer (analytical, distributed) ---
    wl_um = wl * 1e-3
    if aod_550 > 0:
        tau_aer_total = aod_550 * (0.55 / wl_um) ** 1.3  # (n_λ,)
        # Distribute aerosol in the lowest 5 km (exponential scale height 2 km).
        aer_weights = np.exp(-(z_col - surface_altitude_m) / 2000.0)
        aer_weights /= aer_weights.sum() + 1e-30
        # (n_active, n_λ)
        tau_aer_layers = aer_weights[:, np.newaxis] * tau_aer_total[np.newaxis, :]

        # Same for full column (solar path).
        z_full = z_layers[mask_full]
        aer_w_full = np.exp(-(z_full - surface_altitude_m) / 2000.0)
        aer_w_full /= aer_w_full.sum() + 1e-30
        tau_aer_full = aer_w_full[:, np.newaxis] * tau_aer_total[np.newaxis, :]
    else:
        tau_aer_layers = np.zeros_like(tau_col)
        tau_aer_full = np.zeros_like(tau_full)

    # Total optical depth per layer (gas + aerosol).
    tau_col_total = tau_col + tau_aer_layers
    tau_full_total = tau_full + tau_aer_full

    # --- Upward transmittance (surface → sensor), sensor path ---
    tau_slant_up = tau_col_total / mu_sen  # (n_active, n_λ)
    tau_cum_up = np.cumsum(tau_slant_up, axis=0)  # cumulative from bottom
    T_up_total = np.exp(-tau_cum_up[-1])  # (n_λ,) total transmittance

    # --- Upwelling atmospheric thermal emission (surface → sensor) ---
    # L_up = Σ_i  B(T_i, λ) · (1 - exp(-dτ_i)) · exp(-τ_below_i)
    B_layers = planck_array(wl, T_col)  # (n_active, n_λ)
    dtau_up = tau_slant_up  # slant optical depth of each layer
    layer_emissivity = 1.0 - np.exp(-dtau_up)  # (n_active, n_λ)
    # Transmittance from layer i to sensor = exp(-(τ_total - τ_cum_to_i)).
    tau_above_layer = tau_cum_up[-1][np.newaxis, :] - tau_cum_up  # τ from layer to sensor
    T_above_layer = np.exp(-tau_above_layer)  # (n_active, n_λ)
    L_up_atm = np.sum(B_layers * layer_emissivity * T_above_layer, axis=0)

    # --- Downwelling atmospheric thermal emission (TOA → surface) ---
    # Same sum but top-down, using full column.
    tau_slant_down_full = tau_full_total / 1.66  # diffusivity approximation
    B_full = planck_array(wl, T_layers[mask_full])
    n_full = len(tau_slant_down_full)
    # Cumulate from top.
    tau_cum_down = np.cumsum(tau_slant_down_full[::-1], axis=0)[::-1]
    # τ from layer i down to surface = τ_cum_down[i] - τ_cum_down[-1] ... no.
    # τ below layer i to surface:
    tau_below_down = np.zeros_like(tau_cum_down)
    tau_below_down[:-1] = np.cumsum(tau_slant_down_full[1:], axis=0)[::-1]
    # Wait — let me do this more carefully.
    # Going top to bottom: cumulative τ from TOA down to each layer.
    tau_from_top = np.zeros((n_full, n_wl), dtype=np.float64)
    tau_from_top[0] = 0.0
    for i in range(1, n_full):
        tau_from_top[i] = tau_from_top[i - 1] + tau_slant_down_full[n_full - i]
    # Reverse: tau_from_top[0] = from TOA, tau_from_top[-1] = full column from TOA to surface.
    # Actually, let's index from top:
    idx_top = np.arange(n_full - 1, -1, -1)
    tau_down_slant_topdown = tau_slant_down_full[idx_top]  # layers from top to bottom
    B_topdown = B_full[idx_top]  # corresponding B
    tau_cum_topdown = np.cumsum(tau_down_slant_topdown, axis=0)
    tau_cum_topdown_shifted = np.zeros_like(tau_cum_topdown)
    tau_cum_topdown_shifted[1:] = tau_cum_topdown[:-1]
    layer_emissivity_down = 1.0 - np.exp(-tau_down_slant_topdown)
    T_below_topdown = np.exp(
        -(tau_cum_topdown[-1][np.newaxis, :] - tau_cum_topdown)
    )
    L_down_atm = np.sum(
        B_topdown * layer_emissivity_down * T_below_topdown, axis=0
    )

    # --- Solar irradiance at ground ---
    tau_slant_sun = tau_full_total / mu_sun  # (n_full, n_λ)
    T_sun_total = np.exp(-np.sum(tau_slant_sun, axis=0))  # (n_λ,)
    E_sun_toa = planck(wl, 5778.0) * _SOLAR_OMEGA  # (n_λ,)
    E_direct_ground = E_sun_toa * cos_sza * T_sun_total  # (n_λ,)

    # --- Rayleigh scattering (analytical) ---
    tau_ray = 0.00864 * (wl_um ** (-3.916 - 0.074 * wl_um + 0.05 / wl_um))
    # Rayleigh already included in ARTS tau_layers for gas species.
    # Add only the path radiance contribution (single-scatter).
    omega_ray = 1.0
    P_ray = 0.75 * (1.0 + cos_vza**2)
    L_path_ray = (
        E_sun_toa * cos_sza / (4.0 * np.pi)
        * omega_ray * tau_ray * P_ray
        / (mu_sun + mu_sen)
        * (1.0 - np.exp(-(tau_ray) * (1.0 / mu_sun + 1.0 / mu_sen)))
    )

    # --- Aerosol path radiance (if any) ---
    L_path_aer = np.zeros(n_wl)
    if aod_550 > 0:
        sin_s = np.sin(np.radians(solar_zenith_deg))
        sin_v = np.sin(np.radians(sensor_zenith_deg))
        cos_scat = -(cos_sza * cos_vza + sin_s * sin_v * np.cos(np.radians(relative_azimuth_deg)))
        g = 0.65
        P_HG = (1 - g**2) / np.maximum((1 + g**2 - 2 * g * cos_scat) ** 1.5, 1e-10)
        omega_aer = 0.92
        tau_aer_col = aod_550 * (0.55 / wl_um) ** 1.3
        L_path_aer = (
            E_sun_toa * cos_sza * omega_aer * tau_aer_col * P_HG
            / (4.0 * np.pi * (mu_sun + mu_sen))
        )

    # --- Spherical albedo (Rayleigh + aerosol) ---
    S_ray = 0.05 * tau_ray
    S_aer = 0.0
    if aod_550 > 0:
        S_aer = 0.92 * (1 - 0.65) * aod_550 * (0.55 / wl_um) ** 1.3 * 0.5
    S_atm = np.clip(S_ray + S_aer, 0, 0.999)

    # --- Surface terms ---
    eps = 1.0 - rho
    B_surface = planck(wl, surface_temperature_k)

    # Diffuse solar at ground (approximate: fraction of Rayleigh-scattered light).
    E_diffuse_ground = E_sun_toa * cos_sza * (1.0 - np.exp(-tau_ray / mu_sun)) * 0.5

    E_ground_total = E_direct_ground + E_diffuse_ground + np.pi * L_down_atm

    # --- Combine ---
    # Upwelling from surface (before multiple reflections):
    L_surface_up = (rho * E_ground_total / np.pi + eps * B_surface) * T_up_total

    # Multiple reflection correction.
    L_surface = L_surface_up / np.maximum(1.0 - rho * S_atm, 1e-6)

    # Total at sensor.
    L_total = L_up_atm + L_surface + L_path_ray + L_path_aer

    return np.maximum(L_total, 0.0)
