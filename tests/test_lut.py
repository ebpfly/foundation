"""Tests for the atmospheric RT lookup table module (layer optical depth LUT)."""

from __future__ import annotations

import numpy as np
import pytest

from spectralnp.data.lut import (
    LUTConfig,
    SpectralLUT,
    make_lut_wavelength_grid,
    path_integrate,
    planck,
    planck_array,
)


# ---------------------------------------------------------------------------
# Planck function
# ---------------------------------------------------------------------------


class TestPlanck:
    def test_wien_peak(self):
        """Wien's law: peak of B(λ,T) near λ_max = 2898 μm·K / T."""
        T = 5778.0
        wl = np.arange(200, 2000, 1.0)
        B = planck(wl, T)
        peak_wl = wl[np.argmax(B)]
        expected = 2898e3 / T  # nm
        assert abs(peak_wl - expected) < 20

    def test_stefan_boltzmann(self):
        """Integral of πB over all λ should approximate σT⁴."""
        T = 300.0
        wl = np.arange(100, 100000, 10.0)
        B = planck(wl, T)
        integral = np.trapezoid(np.pi * B, wl * 1e-3)  # W/m²
        sigma_T4 = 5.6704e-8 * T**4
        assert abs(integral / sigma_T4 - 1) < 0.05

    def test_positive(self):
        assert np.all(planck(np.array([500.0, 10000.0]), 300.0) > 0)

    def test_monotonic_in_temperature(self):
        wl = np.array([10000.0])
        assert planck(wl, 310.0) > planck(wl, 300.0)

    def test_planck_array(self):
        wl = np.arange(8000, 12000, 100.0)
        temps = np.array([280.0, 300.0, 320.0])
        B = planck_array(wl, temps)
        assert B.shape == (3, len(wl))
        # Hotter → more radiance at all wavelengths.
        assert np.all(B[2] > B[1])
        assert np.all(B[1] > B[0])
        # Matches scalar version.
        np.testing.assert_allclose(B[1], planck(wl, 300.0), rtol=1e-10)


# ---------------------------------------------------------------------------
# Wavelength grid
# ---------------------------------------------------------------------------


class TestWavelengthGrid:
    def test_default_range(self):
        wl = make_lut_wavelength_grid()
        assert wl[0] == 300.0
        assert wl[-1] == 16000.0

    def test_custom_range(self):
        wl = make_lut_wavelength_grid(wl_min=400.0, wl_max=2500.0)
        assert wl[0] == 400.0
        assert wl[-1] <= 2500.0

    def test_sorted_and_unique(self):
        wl = make_lut_wavelength_grid()
        assert np.all(np.diff(wl) > 0)
        assert len(wl) == len(np.unique(wl))

    def test_approximate_size(self):
        wl = make_lut_wavelength_grid()
        assert 2800 < len(wl) < 3100


# ---------------------------------------------------------------------------
# Path integration
# ---------------------------------------------------------------------------


class TestPathIntegrate:
    """Test the runtime path integration engine with synthetic layers."""

    @pytest.fixture
    def simple_atm(self):
        """A simple 10-layer atmosphere with known properties."""
        n_layers = 10
        wl = np.arange(8000.0, 12001.0, 200.0)  # TIR window
        z_layers = np.linspace(500, 95000, n_layers)  # 0.5 to 95 km
        T_layers = np.linspace(288, 220, n_layers)  # surface warm, top cool
        # Uniform small optical depth per layer.
        tau_layers = 0.01 * np.ones((n_layers, len(wl)))
        return tau_layers, T_layers, z_layers, wl

    def test_transparent_atmosphere(self):
        """τ=0 everywhere → sensor sees surface directly."""
        wl = np.arange(8000.0, 12001.0, 200.0)
        n_layers = 5
        tau = np.zeros((n_layers, len(wl)))
        T_lay = 250.0 * np.ones(n_layers)
        z_lay = np.linspace(1000, 50000, n_layers)
        rho = np.zeros(len(wl))  # blackbody surface
        T_s = 300.0

        L = path_integrate(
            tau, T_lay, z_lay, wl, rho,
            sensor_altitude_m=100e3,
            surface_temperature_k=T_s,
            solar_zenith_deg=90.0,  # nighttime
            aod_550=0.0,
        )

        # Should just be B(T_s) (blackbody surface, no atmosphere, no sun).
        B_s = planck(wl, T_s)
        # Won't be exact because of Rayleigh path radiance, but close.
        np.testing.assert_allclose(L, B_s, rtol=0.05)

    def test_opaque_atmosphere(self):
        """Very thick atmosphere → sensor sees atmospheric emission."""
        wl = np.arange(8000.0, 12001.0, 200.0)
        n_layers = 5
        tau = 100.0 * np.ones((n_layers, len(wl)))  # very opaque
        T_lay = 260.0 * np.ones(n_layers)
        z_lay = np.linspace(1000, 50000, n_layers)
        rho = np.zeros(len(wl))
        T_s = 300.0

        L = path_integrate(
            tau, T_lay, z_lay, wl, rho,
            sensor_altitude_m=100e3,
            surface_temperature_k=T_s,
            solar_zenith_deg=90.0,  # nighttime
            aod_550=0.0,
        )

        # Should be approximately B(T_top_layer) since the atmosphere is
        # opaque — we see the top layer emission.
        assert np.mean(L) > 0
        # The radiance should be between the top layer and the warmest layer.
        assert np.mean(L) < np.max(planck(wl, T_lay[0]))

    def test_sensor_altitude_matters(self, simple_atm):
        """Lower sensor altitude → less atmospheric path → different signal."""
        tau, T_lay, z_lay, wl = simple_atm
        rho = 0.05 * np.ones(len(wl))
        T_s = 300.0

        L_satellite = path_integrate(
            tau, T_lay, z_lay, wl, rho,
            sensor_altitude_m=100e3,
            surface_temperature_k=T_s,
            solar_zenith_deg=90.0,
            aod_550=0.0,
        )
        L_airborne = path_integrate(
            tau, T_lay, z_lay, wl, rho,
            sensor_altitude_m=10e3,
            surface_temperature_k=T_s,
            solar_zenith_deg=90.0,
            aod_550=0.0,
        )

        # Should be different (less atmosphere for airborne).
        assert not np.allclose(L_satellite, L_airborne, rtol=1e-3)

    def test_hotter_surface_more_radiance_tir(self, simple_atm):
        """In TIR, hotter surface → more radiance."""
        tau, T_lay, z_lay, wl = simple_atm
        rho = 0.05 * np.ones(len(wl))

        L_cool = path_integrate(
            tau, T_lay, z_lay, wl, rho,
            sensor_altitude_m=100e3,
            surface_temperature_k=280.0,
            solar_zenith_deg=90.0,
            aod_550=0.0,
        )
        L_hot = path_integrate(
            tau, T_lay, z_lay, wl, rho,
            sensor_altitude_m=100e3,
            surface_temperature_k=320.0,
            solar_zenith_deg=90.0,
            aod_550=0.0,
        )
        assert np.all(L_hot > L_cool)

    def test_nonnegative(self, simple_atm):
        tau, T_lay, z_lay, wl = simple_atm
        rho = 0.3 * np.ones(len(wl))
        L = path_integrate(
            tau, T_lay, z_lay, wl, rho,
            sensor_altitude_m=100e3,
            surface_temperature_k=300.0,
            solar_zenith_deg=30.0,
            aod_550=0.2,
        )
        assert np.all(L >= 0)


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------


class TestHDF5RoundTrip:
    def test_write_read(self, tmp_path):
        """Verify layer data survives HDF5 serialisation."""
        import h5py

        wl = np.arange(8000.0, 10001.0, 200.0)
        n_wl = len(wl)
        n_layers = 5

        cfg = LUTConfig(
            wavelength_nm=wl,
            water_vapour=np.array([1.0, 3.0]),
            ozone_du=np.array([300.0]),
            co2_ppmv=np.array([420.0]),
            ch4_ppbv=np.array([1900.0]),
            n2o_ppbv=np.array([332.0]),
            co_ppbv=np.array([120.0]),
            surface_altitude_km=np.array([0.0]),
            n_layers=n_layers,
        )

        shape = cfg.shape  # (2, 1, 1, 1, 1, 1, 1)
        rng = np.random.default_rng(0)
        tau = rng.uniform(0, 0.1, (*shape, n_layers, n_wl)).astype(np.float32)
        T_lay = rng.uniform(220, 290, (*shape, n_layers)).astype(np.float32)
        z_lay = np.broadcast_to(
            np.linspace(500, 50000, n_layers), (*shape, n_layers)
        ).astype(np.float32)
        p_lay = np.broadcast_to(
            np.linspace(1e5, 100, n_layers), (*shape, n_layers)
        ).astype(np.float32)

        path = tmp_path / "test_lut.h5"
        comp = dict(compression="gzip", compression_opts=4)
        with h5py.File(path, "w") as f:
            f.attrs["source"] = "test"
            f.attrs["axis_order"] = list(cfg.axis_names)
            f.attrs["n_layers"] = n_layers
            f.attrs["toa_m"] = 100e3
            f.create_dataset("wavelength_nm", data=wl)
            g = f.create_group("axes")
            for name in cfg.axis_names:
                g.create_dataset(name, data=getattr(cfg, name))
            f.create_dataset("tau_layers", data=tau, **comp)
            f.create_dataset("T_layers", data=T_lay, **comp)
            f.create_dataset("z_layers", data=z_lay, **comp)
            f.create_dataset("p_layers", data=p_lay, **comp)

        lut = SpectralLUT(path)
        np.testing.assert_array_equal(lut.wavelength_nm, wl)
        assert lut.axis_order == list(cfg.axis_names)

        # Interpolate at a grid node.
        tau_out, T_out, z_out = lut.interpolate(
            water_vapour=1.0, ozone_du=300.0, co2_ppmv=420.0,
            ch4_ppbv=1900.0, n2o_ppbv=332.0, co_ppbv=120.0,
            surface_altitude_km=0.0,
        )
        assert tau_out.shape == (n_layers, n_wl)
        np.testing.assert_allclose(
            tau_out, tau[0, 0, 0, 0, 0, 0, 0].astype(np.float64), atol=1e-5
        )

    def test_toa_radiance_runs(self, tmp_path):
        """Verify toa_radiance produces output from a synthetic LUT."""
        import h5py

        wl = np.arange(8000.0, 10001.0, 200.0)
        n_wl = len(wl)
        n_layers = 5

        cfg = LUTConfig(
            wavelength_nm=wl,
            water_vapour=np.array([1.0, 3.0]),
            ozone_du=np.array([300.0]),
            co2_ppmv=np.array([420.0]),
            ch4_ppbv=np.array([1900.0]),
            n2o_ppbv=np.array([332.0]),
            co_ppbv=np.array([120.0]),
            surface_altitude_km=np.array([0.0]),
            n_layers=n_layers,
        )

        shape = cfg.shape
        tau = 0.02 * np.ones((*shape, n_layers, n_wl), np.float32)
        T_lay = np.broadcast_to(
            np.linspace(288, 220, n_layers), (*shape, n_layers)
        ).astype(np.float32)
        z_lay = np.broadcast_to(
            np.linspace(500, 50000, n_layers), (*shape, n_layers)
        ).astype(np.float32)
        p_lay = np.broadcast_to(
            np.linspace(1e5, 100, n_layers), (*shape, n_layers)
        ).astype(np.float32)

        path = tmp_path / "test_lut2.h5"
        comp = dict(compression="gzip", compression_opts=4)
        with h5py.File(path, "w") as f:
            f.attrs["source"] = "test"
            f.attrs["axis_order"] = list(cfg.axis_names)
            f.attrs["n_layers"] = n_layers
            f.attrs["toa_m"] = 100e3
            f.create_dataset("wavelength_nm", data=wl)
            g = f.create_group("axes")
            for name in cfg.axis_names:
                g.create_dataset(name, data=getattr(cfg, name))
            f.create_dataset("tau_layers", data=tau, **comp)
            f.create_dataset("T_layers", data=T_lay, **comp)
            f.create_dataset("z_layers", data=z_lay, **comp)
            f.create_dataset("p_layers", data=p_lay, **comp)

        lut = SpectralLUT(path)
        rho = 0.05 * np.ones(n_wl)

        L = lut.toa_radiance(
            rho,
            water_vapour=2.0, ozone_du=300.0,
            surface_temperature_k=300.0,
            sensor_altitude_km=800.0,
            solar_zenith_deg=30.0,
        )
        assert L.shape == (n_wl,)
        assert np.all(L >= 0)
        assert np.any(L > 0)
