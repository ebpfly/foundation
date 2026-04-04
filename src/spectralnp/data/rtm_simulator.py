"""Radiative transfer simulation using PyARTS (ARTS).

Simulates at-sensor (top-of-atmosphere) radiance given:
  - Surface spectral reflectance
  - Atmospheric state (temperature/pressure profile, gas concentrations)
  - Viewing geometry (solar zenith, sensor zenith, relative azimuth)
  - Sensor altitude

Uses the ARTS (Atmospheric Radiative Transfer Simulator) via its Python
interface `pyarts` to perform line-by-line or correlated-k calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AtmosphericState:
    """Parameterisation of the atmospheric state for simulation.

    All profiles are 1-D (altitude layers).  If scalar values are given,
    a standard atmosphere is perturbed by those values.
    """

    # Aerosol optical depth at 550 nm.
    aod_550: float = 0.1
    # Columnar water vapour (g/cm^2).
    water_vapour: float = 1.5
    # Total ozone column (Dobson units).
    ozone_du: float = 300.0
    # Visibility (km) — alternative aerosol parameterisation.
    visibility_km: float = 23.0
    # Surface altitude (km above sea level).
    surface_altitude_km: float = 0.0
    # Surface temperature (K).  If None, derived from atmospheric profile.
    surface_temperature_k: float | None = None


@dataclass
class ViewGeometry:
    """Solar-sensor geometry."""

    solar_zenith_deg: float = 30.0
    sensor_zenith_deg: float = 0.0   # 0 = nadir
    relative_azimuth_deg: float = 0.0
    sensor_altitude_km: float = 800.0  # satellite altitude


@dataclass
class SimulationResult:
    """Output of a single radiative transfer simulation."""

    wavelength_nm: np.ndarray          # (N_wl,)
    toa_radiance: np.ndarray           # (N_wl,) in W/m^2/sr/um
    surface_reflectance: np.ndarray    # (N_wl,) input reflectance used
    atmospheric_state: AtmosphericState
    geometry: ViewGeometry
    # Diagnostic: atmospheric transmittance and path radiance.
    transmittance: np.ndarray | None = None
    path_radiance: np.ndarray | None = None


class ARTSSimulator:
    """Interface to PyARTS for at-sensor radiance simulation.

    This wraps the ARTS workspace to provide a simplified API for
    generating training data.  The simulator handles:
    1. Setting up a 1-D atmosphere with specified gas profiles
    2. Configuring the surface as a Lambertian reflector
    3. Running the radiative transfer for a nadir-looking sensor
    4. Returning at-sensor spectral radiance

    Parameters
    ----------
    wavelength_nm : array-like
        Wavelength grid for simulation (nm).  Typically 350-2500 nm
        at 1 nm or 5 nm resolution for training data.
    arts_data_path : str, optional
        Path to ARTS XML data files.  If None, pyarts downloads
        them automatically.
    """

    def __init__(
        self,
        wavelength_nm: np.ndarray | None = None,
        arts_data_path: str | None = None,
    ) -> None:
        if wavelength_nm is None:
            wavelength_nm = np.arange(350.0, 2501.0, 5.0)
        self.wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
        self.wavelength_m = self.wavelength_nm * 1e-9
        self.frequency_hz = 2.998e8 / self.wavelength_m  # c / lambda
        self._arts_data_path = arts_data_path
        self._ws = None

    def _init_workspace(self) -> None:
        """Lazy-initialise the ARTS workspace."""
        if self._ws is not None:
            return

        import pyarts

        if self._arts_data_path:
            pyarts.cat.download.retrieve(arts_data_path=self._arts_data_path)
        else:
            pyarts.cat.download.retrieve()

        self._ws = pyarts.Workspace()
        ws = self._ws

        # Basic settings: 1-D atmosphere, scalar (no polarisation).
        ws.atmosphere_dim = 1
        ws.stokes_dim = 1

        # Frequency grid (ARTS uses Hz, descending frequency = ascending wavelength).
        ws.f_grid = np.sort(self.frequency_hz)[::-1].copy()

        # Absorption species relevant for solar shortwave.
        ws.abs_species = [
            "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
            "CO2, CO2-CKDMT252",
            "O3",
            "O2-CIAfunCKDMT100",
            "N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        ]

    def simulate(
        self,
        surface_reflectance: np.ndarray,
        atmos: AtmosphericState | None = None,
        geometry: ViewGeometry | None = None,
    ) -> SimulationResult:
        """Run a single radiative transfer simulation.

        Parameters
        ----------
        surface_reflectance : (N_wl,) array
            Surface spectral reflectance on self.wavelength_nm grid.
        atmos : AtmosphericState, optional
        geometry : ViewGeometry, optional

        Returns
        -------
        SimulationResult with at-sensor radiance.
        """
        if atmos is None:
            atmos = AtmosphericState()
        if geometry is None:
            geometry = ViewGeometry()

        self._init_workspace()
        ws = self._ws

        # --- Atmosphere setup ---
        # Use a built-in standard atmosphere and scale gas columns.
        ws.atm = pyarts.arts.AtmField.fromMeanState(
            ws.f_grid,
            toa=100e3,  # top of atmosphere at 100 km
            nlay=50,
        )

        # Scale water vapour and ozone to match requested columns.
        # (Simplified: in production you'd modify the profile more carefully.)
        if atmos.water_vapour != 1.5:
            scale = atmos.water_vapour / 1.5
            ws.atm["H2O"] = ws.atm["H2O"] * scale

        # --- Surface ---
        # Lambertian surface with given spectral reflectance.
        ws.surface_type = "Lambertian"
        ws.surface_reflectivity = surface_reflectance.reshape(-1, 1, 1)

        # Surface altitude.
        ws.z_surface = np.array([[atmos.surface_altitude_km * 1e3]])

        # --- Geometry ---
        ws.sensor_pos = np.array([[geometry.sensor_altitude_km * 1e3, 0, 0]])
        ws.sensor_los = np.array([[geometry.sensor_zenith_deg, 0]])

        # Solar source.
        ws.sunFromGrid(
            suns_do=1,
            sun_spectrum_raw="sun/solar_spectrum_QUIET",
            sun_za=geometry.solar_zenith_deg,
            sun_aa=geometry.relative_azimuth_deg,
        )

        # --- Run RT calculation ---
        ws.propmat_clearsky_agendaAuto()
        ws.iy_main_agendaFromPath()
        ws.ppath_agendaFromGeometric()

        ws.yCalc()

        # Extract radiance.  ARTS outputs Stokes-I in W/(m^2 Hz sr).
        # Convert to W/(m^2 sr um) using: L_lambda = L_nu * c / lambda^2
        y_wm2_hz_sr = np.array(ws.y).flatten()
        # Map from frequency-ordered to wavelength-ordered.
        freq_order = np.argsort(self.frequency_hz)[::-1]
        y_sorted = y_wm2_hz_sr  # already sorted by f_grid ordering
        # Convert spectral units: L_λ = L_ν · |dν/dλ| = L_ν · c / λ²
        c = 2.998e8
        lambda_m = self.wavelength_m
        toa_radiance_wm2_sr_um = y_sorted * c / (lambda_m**2) * 1e-6  # per um

        return SimulationResult(
            wavelength_nm=self.wavelength_nm.copy(),
            toa_radiance=toa_radiance_wm2_sr_um,
            surface_reflectance=surface_reflectance.copy(),
            atmospheric_state=atmos,
            geometry=geometry,
        )


def simplified_toa_radiance(
    surface_reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    atmos: AtmosphericState | None = None,
    geometry: ViewGeometry | None = None,
) -> SimulationResult:
    """Fast approximate TOA radiance using a simplified two-stream model.

    This is useful for rapid dataset generation and testing when full
    ARTS simulation is too slow.  The model accounts for:
    - Solar irradiance (6S-style solar spectrum approximation)
    - Rayleigh scattering
    - Aerosol extinction (wavelength-dependent)
    - Water vapour and ozone absorption (parameterised)
    - Lambertian surface with given reflectance

    NOT a substitute for full RTM training data, but useful for
    prototyping the ML pipeline.
    """
    if atmos is None:
        atmos = AtmosphericState()
    if geometry is None:
        geometry = ViewGeometry()

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    wl_um = wl / 1000.0
    rho = np.asarray(surface_reflectance, dtype=np.float64)

    # Solar zenith cosine.
    cos_sza = np.cos(np.radians(geometry.solar_zenith_deg))

    # --- Solar irradiance approximation (W/m^2/um) ---
    # Planck function at ~5778 K scaled to ~1360 W/m^2 total.
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    T_sun = 5778.0
    lam_m = wl_um * 1e-6
    B = (2 * h * c**2 / lam_m**5) / (np.exp(h * c / (lam_m * k * T_sun)) - 1)
    # Scale: solar disk solid angle ~6.8e-5 sr.
    E_sun = B * 6.8e-5 * 1e-6  # W/m^2/um at TOA

    # --- Rayleigh optical depth ---
    tau_ray = 0.00864 * (wl_um ** (-3.916 - 0.074 * wl_um + 0.05 / wl_um))

    # --- Aerosol optical depth (Angstrom law) ---
    alpha_angstrom = 1.3
    tau_aer = atmos.aod_550 * (0.55 / wl_um) ** alpha_angstrom

    # --- Gas absorption (simplified) ---
    # Water vapour: dominant bands around 940, 1130, 1380, 1870 nm.
    tau_h2o = np.zeros_like(wl)
    for center, width, strength in [
        (940, 40, 0.15), (1130, 50, 0.10), (1380, 60, 0.40), (1870, 80, 0.30),
    ]:
        tau_h2o += strength * atmos.water_vapour * np.exp(-0.5 * ((wl - center) / width) ** 2)

    # Ozone: Huggins/Chappuis bands.
    tau_o3 = np.zeros_like(wl)
    # Chappuis band (500-700 nm).
    tau_o3 += 0.0002 * (atmos.ozone_du / 300.0) * np.exp(-0.5 * ((wl - 600) / 80) ** 2)
    # Huggins band (300-350 nm).
    tau_o3 += 0.05 * (atmos.ozone_du / 300.0) * np.exp(-0.5 * ((wl - 320) / 15) ** 2)

    # Total optical depth.
    tau_total = tau_ray + tau_aer + tau_h2o + tau_o3

    # --- Two-stream approximation ---
    # Direct transmittance (sun path + sensor path).
    cos_vza = np.cos(np.radians(geometry.sensor_zenith_deg))
    airmass = 1.0 / cos_sza + 1.0 / max(cos_vza, 0.1)
    T_direct = np.exp(-tau_total * airmass)

    # Diffuse transmittance (rough approximation).
    T_diffuse = np.exp(-tau_total * 1.66)  # diffusivity factor

    # Path radiance (Rayleigh + aerosol scattering).
    # Simplified: fraction of scattered light reaching sensor.
    omega_ray = 1.0  # Rayleigh single-scatter albedo
    omega_aer = 0.9
    # Single-scatter approximation for path radiance.
    P_ray = 0.75 * (1 + np.cos(np.radians(geometry.sensor_zenith_deg)) ** 2)  # phase function
    L_path = (
        E_sun * cos_sza / (4 * np.pi)
        * (omega_ray * tau_ray * P_ray + omega_aer * tau_aer * 0.7)
        * (1 - np.exp(-tau_total / cos_sza))
        / (tau_total + 1e-10)
    )

    # Surface-reflected radiance.
    L_surface = (rho * E_sun * cos_sza * T_direct) / np.pi

    # Spherical albedo approximation for multiple surface-atmosphere interactions.
    s_atm = 0.05 * (tau_ray + tau_aer)  # atmospheric backscatter
    L_total = L_path + L_surface / (1 - s_atm * rho + 1e-10)

    # Ensure non-negative.
    L_total = np.maximum(L_total, 0.0)

    return SimulationResult(
        wavelength_nm=wl.copy(),
        toa_radiance=L_total,
        surface_reflectance=rho.copy(),
        atmospheric_state=atmos,
        geometry=geometry,
        transmittance=T_direct,
        path_radiance=L_path,
    )
