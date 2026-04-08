"""Radiative transfer simulation using PyARTS (ARTS).

Simulates at-sensor (top-of-atmosphere) radiance given:
  - Surface spectral reflectance
  - Atmospheric state (temperature/pressure profile, gas concentrations)
  - Viewing geometry (solar zenith, sensor zenith, relative azimuth)
  - Sensor altitude

Provides three backends:
  1. ``ARTSSimulator``          – full PyARTS line-by-line (accurate, slow)
  2. ``LUTSimulator``           – pre-computed LUT + analytical aerosol (fast)
  3. ``simplified_toa_radiance`` – parametric two-stream (fast, approximate)

All six major absorbing gases are included for the 0.3–16 μm range:
H₂O, CO₂, O₃, N₂O, CH₄, CO  (plus O₂ / N₂ CIA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (local copies to avoid circular imports)
# ---------------------------------------------------------------------------
_C = 2.99792458e8
_H = 6.62607015e-34
_K = 1.380649e-23
_SOLAR_OMEGA = 6.794e-5


def _planck(wavelength_nm: np.ndarray, T: float) -> np.ndarray:
    """Planck spectral radiance B(λ,T) in W m⁻² sr⁻¹ μm⁻¹."""
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    x = np.minimum(_H * _C / (lam * _K * T), 500.0)
    return (2.0 * _H * _C**2 / lam**5) / (np.exp(x) - 1.0) * 1e-6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
    # CO₂ volume mixing ratio (ppmv).
    co2_ppmv: float = 420.0
    # CH₄ volume mixing ratio (ppbv).
    ch4_ppbv: float = 1900.0
    # N₂O volume mixing ratio (ppbv).
    n2o_ppbv: float = 332.0
    # CO volume mixing ratio (ppbv).
    co_ppbv: float = 120.0
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


# ---------------------------------------------------------------------------
# Full ARTS simulator (0.3–16 μm)
# ---------------------------------------------------------------------------


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
        Wavelength grid for simulation (nm).  Default covers 300–16000 nm.
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
        self.frequency_hz = _C / self.wavelength_m
        self._arts_data_path = arts_data_path
        self._ws = None

    def _init_workspace(self) -> None:
        """Lazy-initialise the ARTS workspace."""
        if self._ws is not None:
            return

        import pyarts
        from pathlib import Path
        from spectralnp.data.lut import ARTS_ABS_SPECIES

        pyarts.cat.download.retrieve()

        self._ws = pyarts.Workspace()
        ws = self._ws

        # Basic settings: 1-D atmosphere, scalar (no polarisation).
        ws.atmosphere_dim = 1
        ws.stokes_dim = 1

        # Frequency grid (ARTS uses Hz, descending frequency = ascending wavelength).
        ws.f_grid = np.sort(self.frequency_hz)[::-1].copy()

        # All six trace gases + O₂/N₂ CIA for 0.3–16 μm.
        ws.abs_species = list(ARTS_ABS_SPECIES)

        # Load spectral line catalogue (ARTS 2.6 API).
        cat_base = Path.home() / ".cache" / "arts"
        cat_dirs = sorted(cat_base.glob("arts-cat-data-*"))
        if cat_dirs:
            lines_dir = str(cat_dirs[-1] / "lines") + "/"
        else:
            raise RuntimeError(
                "ARTS catalogue data not found in ~/.cache/arts/. "
                "Run pyarts.cat.download.retrieve() first."
            )
        ws.abs_linesReadSpeciesSplitCatalog(basename=lines_dir, robust=1)
        ws.abs_linesTurnOffLineMixing()
        ws.abs_linesCutoff(option="ByLine", value=750e9)
        ws.abs_lines_per_speciesCreateFromLines()
        ws.propmat_clearsky_agendaAuto()
        ws.jacobian_quantities = pyarts.arts.ArrayOfRetrievalQuantity()

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

        import pyarts
        from spectralnp.data.lut import ARTS_ABS_SPECIES

        # --- Atmosphere setup (ARTS 2.6 API) ---
        n_levels = 51
        z_surface = atmos.surface_altitude_km * 1e3
        z_grid = np.linspace(z_surface, 100e3, n_levels)
        T_profile = np.where(
            z_grid < 11e3, 288.15 - 6.5e-3 * z_grid,
            np.where(z_grid < 20e3, 216.65,
                     np.where(z_grid < 32e3, 216.65 + 1e-3 * (z_grid - 20e3),
                              228.65)))
        T_profile = np.clip(T_profile, 180.0, 320.0)
        p_profile = 101325.0 * np.exp(-z_grid / 8500.0)

        p_sorted = np.sort(p_profile)[::-1]
        sort_idx = np.argsort(p_profile)[::-1]

        ws.p_grid = p_sorted
        ws.lat_grid = np.array([])
        ws.lon_grid = np.array([])
        ws.t_field = pyarts.arts.Tensor3(
            T_profile[sort_idx].reshape(-1, 1, 1)
        )
        ws.z_field = pyarts.arts.Tensor3(
            z_grid[sort_idx].reshape(-1, 1, 1)
        )
        ws.z_surface = pyarts.arts.Matrix(np.array([[z_surface]]))

        n_sp = len(ARTS_ABS_SPECIES)
        vmr = np.zeros((n_sp, n_levels, 1, 1))
        z_sorted = z_grid[sort_idx]
        vmr[0] = 0.01 * np.exp(-z_sorted.reshape(-1, 1, 1) / 2000.0)
        vmr[1] = 400e-6
        vmr[2] = 3e-6
        vmr[3] = 332e-9
        vmr[4] = 1800e-9
        vmr[5] = 120e-9
        vmr[6] = 0.21
        vmr[7] = 0.78

        _REF = {
            "water_vapour": (0, 1.5),
            "ozone_du": (2, 300.0),
            "co2_ppmv": (1, 400.0),
            "ch4_ppbv": (4, 1800.0),
            "n2o_ppbv": (3, 332.0),
            "co_ppbv": (5, 120.0),
        }
        for attr_name, (sp_idx, ref_val) in _REF.items():
            val = getattr(atmos, attr_name)
            if val != ref_val:
                vmr[sp_idx] *= val / ref_val

        ws.vmr_field = pyarts.arts.Tensor4(vmr)

        # --- Surface ---
        ws.surface_type = "Lambertian"
        ws.surface_reflectivity = surface_reflectance.reshape(-1, 1, 1)

        # Surface skin temperature for thermal emission.
        if atmos.surface_temperature_k is not None:
            ws.surface_skin_t = float(atmos.surface_temperature_k)

        # --- Geometry ---
        ws.sensor_pos = np.array([[geometry.sensor_altitude_km * 1e3, 0, 0]])
        ws.sensor_los = np.array([[geometry.sensor_zenith_deg, 0]])

        ws.sunFromGrid(
            suns_do=1,
            sun_spectrum_raw="sun/solar_spectrum_QUIET",
            sun_za=geometry.solar_zenith_deg,
            sun_aa=geometry.relative_azimuth_deg,
        )

        # --- Run RT calculation ---
        ws.lbl_checkedCalc()
        ws.propmat_clearsky_agenda_checkedCalc()
        ws.atmfields_checkedCalc()
        ws.atmgeom_checkedCalc()
        ws.iy_main_agendaFromPath()
        ws.ppath_agendaFromGeometric()

        ws.yCalc()

        # Extract radiance.  ARTS outputs Stokes-I in W/(m² Hz sr).
        # Convert to W/(m² sr μm) using: L_λ = L_ν · c / λ²
        y_wm2_hz_sr = np.array(ws.y.value).flatten()
        lambda_m = self.wavelength_m
        toa_radiance_wm2_sr_um = y_wm2_hz_sr * _C / (lambda_m**2) * 1e-6

        return SimulationResult(
            wavelength_nm=self.wavelength_nm.copy(),
            toa_radiance=toa_radiance_wm2_sr_um,
            surface_reflectance=surface_reflectance.copy(),
            atmospheric_state=atmos,
            geometry=geometry,
        )


# ---------------------------------------------------------------------------
# ARTS abs_lookup-backed simulator (caches tau per atmosphere)
# ---------------------------------------------------------------------------


class ARTSLookupSimulator:
    """RTM using ARTS ``abs_lookup`` for gas absorption + cached layer τ.

    Each ``simulate()`` call:
      1. Quantises the atmospheric state into a coarse cache key.
      2. If the (atmos_key) is not in cache, calls ARTS once per layer to
         compute layer-by-layer optical depths via ``propmat_clearskyAddFromLookup``.
      3. Runs the existing ``path_integrate()`` on the cached τ tensor with
         the *actual* (un-quantised) geometry, surface, and aerosol params.

    The ARTS workspace is created lazily and is **not fork-safe**, so callers
    must use ``num_workers=0`` in any DataLoader.

    Parameters
    ----------
    abs_lookup_path : str
        Path to an ARTS ``abs_lookup`` XML file (e.g. produced by ``atmgen``).
    n_layers : int
        Number of atmospheric layers.  Default 50.
    toa_m : float
        Top-of-atmosphere altitude in metres.  Default 100 km.
    cache_quantize : dict, optional
        Per-parameter rounding for the cache key.  Default rounds
        water_vapour to 1.0, ozone_du to 100, and surface_altitude_km to 1.0.
    """

    DEFAULT_QUANTIZE = {
        "water_vapour": 1.0,
        "ozone_du": 100.0,
        "co2_ppmv": 200.0,    # effectively fixed
        "ch4_ppbv": 500.0,    # effectively fixed
        "n2o_ppbv": 100.0,    # effectively fixed
        "co_ppbv": 500.0,     # effectively fixed
        "surface_altitude_km": 1.0,
    }

    def __init__(
        self,
        abs_lookup_path: str,
        n_layers: int = 50,
        toa_m: float = 100e3,
        cache_quantize: dict | None = None,
    ) -> None:
        self.abs_lookup_path = abs_lookup_path
        self.n_layers = n_layers
        self.toa_m = toa_m
        self.cache_quantize = cache_quantize or dict(self.DEFAULT_QUANTIZE)
        self._ws = None
        self.wavelength_nm: np.ndarray | None = None
        self._cache: dict[tuple, tuple[np.ndarray, ...]] = {}
        self._n_misses = 0
        self._n_hits = 0

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _init_ws(self) -> None:
        """Create the ARTS workspace and load the abs_lookup."""
        if self._ws is not None:
            return
        import pyarts

        pyarts.cat.download.retrieve()

        ws = pyarts.Workspace()
        ws.atmosphere_dim = 1
        ws.stokes_dim = 1
        ws.abs_species = [
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            "CO2", "O3", "N2O", "CH4", "CO", "O2", "N2",
        ]
        ws.ReadXML(ws.abs_lookup, str(self.abs_lookup_path))
        ws.f_gridFromGasAbsLookup()
        ws.abs_lookupAdapt()

        @pyarts.workspace.arts_agenda
        def propmat_clearsky_agenda(ws):
            ws.Ignore(ws.rtp_mag)
            ws.Ignore(ws.rtp_los)
            ws.Ignore(ws.rtp_nlte)
            ws.propmat_clearskyInit()
            ws.propmat_clearskyAddFromLookup()

        ws.propmat_clearsky_agenda = propmat_clearsky_agenda
        ws.jacobian_quantities = pyarts.arts.ArrayOfRetrievalQuantity()
        ws.propmat_clearsky_agenda_checkedCalc()

        self._ws = ws
        f_grid = np.array(ws.f_grid.value)
        self.wavelength_nm = (_C / f_grid * 1e9).astype(np.float64)
        # ARTS f_grid is descending, so wl is ascending — make it strictly ascending.
        order = np.argsort(self.wavelength_nm)
        self.wavelength_nm = self.wavelength_nm[order]
        self._wl_order = order

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def _atmos_key(self, atmos: AtmosphericState) -> tuple:
        q = self.cache_quantize
        def r(val, step):
            return round(val / step) * step
        return (
            r(atmos.water_vapour, q["water_vapour"]),
            r(atmos.ozone_du, q["ozone_du"]),
            r(atmos.co2_ppmv, q["co2_ppmv"]),
            r(atmos.ch4_ppbv, q["ch4_ppbv"]),
            r(atmos.n2o_ppbv, q["n2o_ppbv"]),
            r(atmos.co_ppbv, q["co_ppbv"]),
            r(atmos.surface_altitude_km, q["surface_altitude_km"]),
        )

    # ------------------------------------------------------------------
    # Per-layer τ extraction (called on cache miss)
    # ------------------------------------------------------------------

    def _compute_tau_layers(
        self, atmos_key: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run ARTS to compute layer optical depths for an atmospheric state."""
        import pyarts

        self._init_ws()
        ws = self._ws

        (water_vapour, ozone_du, co2_ppmv, ch4_ppbv,
         n2o_ppbv, co_ppbv, surface_altitude_km) = atmos_key

        # ---- Build a US-Standard atmosphere on n_layers+1 levels ----
        n_levels = self.n_layers + 1
        z_surface = surface_altitude_km * 1e3
        z = np.linspace(z_surface, self.toa_m, n_levels)
        T = np.where(
            z < 11e3, 288.15 - 6.5e-3 * z,
            np.where(z < 20e3, 216.65,
                     np.where(z < 32e3, 216.65 + 1e-3 * (z - 20e3), 228.65)))
        T = np.clip(T, 180.0, 320.0)
        p = 101325.0 * np.exp(-z / 8500.0)
        p_sorted = np.sort(p)[::-1]
        sort_idx = np.argsort(p)[::-1]

        ws.p_grid = p_sorted
        ws.lat_grid = np.array([])
        ws.lon_grid = np.array([])
        ws.t_field = pyarts.arts.Tensor3(T[sort_idx].reshape(-1, 1, 1))
        ws.z_field = pyarts.arts.Tensor3(z[sort_idx].reshape(-1, 1, 1))
        ws.z_surface = pyarts.arts.Matrix(np.array([[z_surface]]))

        n_sp = 8
        vmr = np.zeros((n_sp, n_levels, 1, 1))
        z_s = z[sort_idx]
        # Reference VMR profiles, scaled by atmos parameters.
        vmr[0] = 0.01 * np.exp(-z_s.reshape(-1, 1, 1) / 2000.0) * (water_vapour / 1.5)
        vmr[1] = co2_ppmv * 1e-6
        vmr[2] = 3e-6 * (ozone_du / 300.0)
        vmr[3] = n2o_ppbv * 1e-9
        vmr[4] = ch4_ppbv * 1e-9
        vmr[5] = co_ppbv * 1e-9
        vmr[6] = 0.21
        vmr[7] = 0.78
        ws.vmr_field = pyarts.arts.Tensor4(vmr)

        # ---- Per-layer propmat extraction ----
        n_freq = len(np.array(ws.f_grid.value))
        n_layers = self.n_layers
        z_arr = z[sort_idx]
        T_arr = T[sort_idx]
        z_layers = 0.5 * (z_arr[:-1] + z_arr[1:])
        T_layers = 0.5 * (T_arr[:-1] + T_arr[1:])
        p_layers = np.sqrt(p_sorted[:-1] * p_sorted[1:])
        dz = np.abs(z_arr[1:] - z_arr[:-1])

        tau_layers = np.zeros((n_layers, n_freq), dtype=np.float64)
        for i in range(n_layers):
            ws.rtp_pressure = float(p_layers[i])
            ws.rtp_temperature = float(T_layers[i])
            ws.rtp_vmr = vmr[:, i, 0, 0].copy()
            ws.rtp_mag = np.array([0.0, 0.0, 0.0])
            ws.rtp_los = np.array([0.0, 0.0])
            ws.rtp_nlte = pyarts.arts.EnergyLevelMap()
            ws.propmat_clearskyInit()
            ws.propmat_clearskyAddFromLookup()
            pm = np.array(ws.propmat_clearsky.value.data)
            kappa = pm[0, 0, :, 0]
            tau_layers[i, :] = kappa * dz[i]

        # Reorder along the wavelength axis to match self.wavelength_nm (ascending).
        tau_layers = tau_layers[:, self._wl_order]

        return (
            tau_layers.astype(np.float32),
            T_layers.astype(np.float32),
            z_layers.astype(np.float32),
            p_layers.astype(np.float32),
        )

    def get_tau(self, atmos: AtmosphericState):
        """Cached τ lookup for an atmospheric state."""
        key = self._atmos_key(atmos)
        if key in self._cache:
            self._n_hits += 1
            return self._cache[key]
        self._n_misses += 1
        result = self._compute_tau_layers(key)
        self._cache[key] = result
        return result

    @property
    def cache_stats(self) -> dict:
        return {
            "size": len(self._cache),
            "hits": self._n_hits,
            "misses": self._n_misses,
        }

    # ------------------------------------------------------------------
    # Public API: forward simulate
    # ------------------------------------------------------------------

    def simulate(
        self,
        surface_reflectance: np.ndarray,
        atmos: AtmosphericState | None = None,
        geometry: ViewGeometry | None = None,
    ) -> SimulationResult:
        """Compute TOA radiance using cached τ + path integration."""
        from spectralnp.data.lut import path_integrate

        if atmos is None:
            atmos = AtmosphericState()
        if geometry is None:
            geometry = ViewGeometry()

        tau_layers, T_layers, z_layers, _p_layers = self.get_tau(atmos)

        # Resample reflectance to ARTS wavelength grid if needed.
        refl = np.asarray(surface_reflectance, dtype=np.float64)
        if len(refl) != len(self.wavelength_nm):
            src_wl = np.linspace(
                self.wavelength_nm[0], self.wavelength_nm[-1], len(refl)
            )
            refl = np.interp(self.wavelength_nm, src_wl, refl)

        toa = path_integrate(
            tau_layers=tau_layers,
            T_layers=T_layers,
            z_layers=z_layers,
            wavelength_nm=self.wavelength_nm,
            surface_reflectance=refl,
            solar_zenith_deg=geometry.solar_zenith_deg,
            sensor_zenith_deg=geometry.sensor_zenith_deg,
            relative_azimuth_deg=geometry.relative_azimuth_deg,
            sensor_altitude_m=geometry.sensor_altitude_km * 1e3,
            surface_altitude_m=atmos.surface_altitude_km * 1e3,
            surface_temperature_k=atmos.surface_temperature_k or 300.0,
            aod_550=atmos.aod_550,
        )

        return SimulationResult(
            wavelength_nm=self.wavelength_nm.copy(),
            toa_radiance=toa,
            surface_reflectance=refl,
            atmospheric_state=atmos,
            geometry=geometry,
        )


# ---------------------------------------------------------------------------
# LUT-backed simulator
# ---------------------------------------------------------------------------


class LUTSimulator:
    """Fast at-sensor radiance from a pre-computed ARTS layer-optical-depth LUT.

    Drop-in replacement for ``ARTSSimulator``.  Interpolates cached layer
    optical depths then runs vectorised path integration at runtime, so
    it supports arbitrary sensor altitude and viewing geometry.

    Parameters
    ----------
    lut_path : path to the HDF5 LUT file.
    """

    def __init__(self, lut_path: str) -> None:
        from spectralnp.data.lut import SpectralLUT

        self.lut = SpectralLUT(lut_path)
        self.wavelength_nm = self.lut.wavelength_nm

    def simulate(
        self,
        surface_reflectance: np.ndarray,
        atmos: AtmosphericState | None = None,
        geometry: ViewGeometry | None = None,
    ) -> SimulationResult:
        """Simulate at-sensor radiance via LUT + path integration.

        *surface_reflectance* should be on the LUT wavelength grid.
        If it is on a different grid, it is linearly interpolated.
        """
        if atmos is None:
            atmos = AtmosphericState()
        if geometry is None:
            geometry = ViewGeometry()

        # Resample reflectance to LUT grid if lengths differ.
        refl = np.asarray(surface_reflectance, dtype=np.float64)
        if len(refl) != len(self.wavelength_nm):
            src_wl = np.linspace(
                self.wavelength_nm[0], self.wavelength_nm[-1], len(refl)
            )
            refl = np.interp(self.wavelength_nm, src_wl, refl)

        T_s = atmos.surface_temperature_k if atmos.surface_temperature_k else 300.0

        toa = self.lut.toa_radiance(
            refl,
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
            surface_temperature_k=T_s,
            aod_550=atmos.aod_550,
        )

        return SimulationResult(
            wavelength_nm=self.wavelength_nm.copy(),
            toa_radiance=toa,
            surface_reflectance=refl.copy(),
            atmospheric_state=atmos,
            geometry=geometry,
        )


# ---------------------------------------------------------------------------
# Simplified two-stream model (extended to 0.3–16 μm)
# ---------------------------------------------------------------------------


def simplified_toa_radiance(
    surface_reflectance: np.ndarray,
    wavelength_nm: np.ndarray,
    atmos: AtmosphericState | None = None,
    geometry: ViewGeometry | None = None,
) -> SimulationResult:
    """Fast approximate TOA radiance using a simplified two-stream model.

    This is useful for rapid dataset generation and testing when full
    ARTS simulation or a LUT is not available.  The model accounts for:
    - Solar irradiance (Planck @ 5778 K)
    - Rayleigh scattering
    - Aerosol extinction (wavelength-dependent)
    - Six-gas absorption (H₂O, CO₂, O₃, CH₄, N₂O, CO — parameterised)
    - Surface thermal emission (Kirchhoff emissivity)
    - Atmospheric thermal emission (simplified)

    NOT a substitute for full RTM or LUT training data, but useful for
    prototyping the ML pipeline.
    """
    if atmos is None:
        atmos = AtmosphericState()
    if geometry is None:
        geometry = ViewGeometry()

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    wl_um = wl / 1000.0
    rho = np.asarray(surface_reflectance, dtype=np.float64)

    cos_sza = np.cos(np.radians(geometry.solar_zenith_deg))
    cos_vza = np.cos(np.radians(geometry.sensor_zenith_deg))
    cos_vza = max(cos_vza, 0.1)

    # --- Solar irradiance approximation (W/m^2/um) ---
    E_sun = _planck(wl, 5778.0) * _SOLAR_OMEGA

    # --- Rayleigh optical depth ---
    tau_ray = 0.00864 * (wl_um ** (-3.916 - 0.074 * wl_um + 0.05 / wl_um))

    # --- Aerosol optical depth (Angstrom law) ---
    alpha_angstrom = 1.3
    tau_aer = atmos.aod_550 * (0.55 / wl_um) ** alpha_angstrom

    # --- Gas absorption (parameterised) ---
    tau_gas = np.zeros_like(wl)

    # H₂O: dominant bands at 720, 820, 940, 1130, 1380, 1870, 2700, 6300 nm
    # plus far-IR rotation band.
    for center, width, strength in [
        (720, 20, 0.02), (820, 20, 0.03),
        (940, 40, 0.15), (1130, 50, 0.10),
        (1380, 60, 0.40), (1870, 80, 0.30),
        (2700, 150, 0.80), (6300, 400, 1.50),
    ]:
        tau_gas += (
            strength * atmos.water_vapour
            * np.exp(-0.5 * ((wl - center) / width) ** 2)
        )
    # H₂O far-IR continuum (> 12 μm, increases strongly).
    tau_gas += atmos.water_vapour * 0.3 * np.exp((wl - 16000) / 3000)
    tau_gas = np.maximum(tau_gas, 0.0)

    # O₃: Huggins (320 nm), Chappuis (600 nm), 9.6 μm band.
    oz_scale = atmos.ozone_du / 300.0
    tau_gas += 0.05 * oz_scale * np.exp(-0.5 * ((wl - 320) / 15) ** 2)
    tau_gas += 0.0002 * oz_scale * np.exp(-0.5 * ((wl - 600) / 80) ** 2)
    tau_gas += 0.15 * oz_scale * np.exp(-0.5 * ((wl - 9600) / 300) ** 2)

    # CO₂: 2.0, 2.7, 4.3, 15 μm bands.
    co2_scale = atmos.co2_ppmv / 400.0
    for center, width, strength in [
        (2000, 40, 0.05), (2700, 100, 0.20),
        (4300, 120, 1.50), (15000, 800, 2.00),
    ]:
        tau_gas += (
            strength * co2_scale
            * np.exp(-0.5 * ((wl - center) / width) ** 2)
        )

    # CH₄: 3.3 μm and 7.7 μm bands.
    ch4_scale = atmos.ch4_ppbv / 1800.0
    for center, width, strength in [
        (3300, 100, 0.25), (7700, 300, 0.40),
    ]:
        tau_gas += (
            strength * ch4_scale
            * np.exp(-0.5 * ((wl - center) / width) ** 2)
        )

    # N₂O: 4.5 μm and 7.8 μm bands.
    n2o_scale = atmos.n2o_ppbv / 332.0
    for center, width, strength in [
        (4500, 80, 0.20), (7800, 250, 0.30),
    ]:
        tau_gas += (
            strength * n2o_scale
            * np.exp(-0.5 * ((wl - center) / width) ** 2)
        )

    # CO: 4.7 μm band.
    co_scale = atmos.co_ppbv / 120.0
    tau_gas += 0.10 * co_scale * np.exp(-0.5 * ((wl - 4700) / 60) ** 2)

    # Total optical depth.
    tau_total = tau_ray + tau_aer + tau_gas

    # --- Two-stream transmittances ---
    T_sun = np.exp(-tau_total / cos_sza)
    T_sensor = np.exp(-tau_total / cos_vza)
    T_two_way = T_sun * T_sensor

    # Diffuse transmittance (rough approximation, used in path radiance).
    _T_diffuse = np.exp(-tau_total * 1.66)  # noqa: F841

    # --- Path radiance (Rayleigh + aerosol single scatter) ---
    omega_ray = 1.0
    omega_aer = 0.9
    P_ray = 0.75 * (1 + np.cos(np.radians(geometry.sensor_zenith_deg)) ** 2)
    L_path = (
        E_sun * cos_sza / (4 * np.pi)
        * (omega_ray * tau_ray * P_ray + omega_aer * tau_aer * 0.7)
        * (1 - np.exp(-tau_total / cos_sza))
        / (tau_total + 1e-10)
    )

    # --- Surface-reflected solar radiance ---
    L_surface_solar = (rho * E_sun * cos_sza * T_two_way) / np.pi

    # Spherical albedo approximation.
    s_atm = 0.05 * (tau_ray + tau_aer)

    # --- Surface thermal emission (significant for λ > 3 μm) ---
    T_s = atmos.surface_temperature_k if atmos.surface_temperature_k else 288.15
    eps = 1.0 - rho
    B_surface = _planck(wl, T_s)
    L_surface_thermal = eps * B_surface * T_sensor

    # --- Atmospheric thermal emission (simplified) ---
    # Approximate: the atmosphere emits (1 − T_sensor) × B(T_atm_eff)
    T_atm_eff = T_s - 30.0  # rough effective atmospheric temperature
    B_atm = _planck(wl, T_atm_eff)
    L_atm_thermal = (1.0 - T_sensor) * B_atm

    # --- Combine ---
    L_total = (
        L_path
        + L_surface_solar / (1 - s_atm * rho + 1e-10)
        + L_surface_thermal
        + L_atm_thermal
    )
    L_total = np.maximum(L_total, 0.0)

    return SimulationResult(
        wavelength_nm=wl.copy(),
        toa_radiance=L_total,
        surface_reflectance=rho.copy(),
        atmospheric_state=atmos,
        geometry=geometry,
        transmittance=T_two_way,
        path_radiance=L_path,
    )
