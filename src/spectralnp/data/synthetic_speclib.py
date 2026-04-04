"""Synthetic spectral library for training and demos.

Generates physically-motivated reflectance spectra for common material
classes without requiring the 5GB USGS download.  Each material is
parameterised by known spectral features (absorption bands, slopes, etc.)
with random variation to produce diverse training samples.
"""

from __future__ import annotations

import numpy as np

from spectralnp.data.usgs_speclib import Spectrum, SpectralLibrary


def _gaussian_absorption(wl: np.ndarray, center: float, width: float, depth: float) -> np.ndarray:
    """Absorption feature (dip) at given center wavelength."""
    return depth * np.exp(-0.5 * ((wl - center) / width) ** 2)


def _generate_mineral(wl: np.ndarray, rng: np.random.Generator, idx: int) -> Spectrum:
    """Generate a mineral-like spectrum with Fe/Al-OH/carbonate absorptions."""
    # Base reflectance: moderate, slightly red-sloping.
    base = rng.uniform(0.15, 0.6)
    slope = rng.uniform(-0.0001, 0.0002)
    refl = base + slope * (wl - 1000)

    # Iron absorptions near 500nm and 900nm.
    if rng.random() > 0.3:
        refl -= _gaussian_absorption(wl, rng.uniform(480, 520), rng.uniform(50, 120), rng.uniform(0.05, 0.2))
        refl -= _gaussian_absorption(wl, rng.uniform(850, 950), rng.uniform(80, 200), rng.uniform(0.05, 0.25))

    # Al-OH / clay absorption near 2200nm.
    if rng.random() > 0.4:
        refl -= _gaussian_absorption(wl, rng.uniform(2150, 2250), rng.uniform(20, 60), rng.uniform(0.03, 0.15))

    # Carbonate absorption near 2340nm.
    if rng.random() > 0.6:
        refl -= _gaussian_absorption(wl, rng.uniform(2300, 2360), rng.uniform(20, 50), rng.uniform(0.02, 0.1))

    # Water/OH near 1400nm and 1900nm.
    if rng.random() > 0.5:
        refl -= _gaussian_absorption(wl, rng.uniform(1380, 1420), rng.uniform(30, 60), rng.uniform(0.02, 0.1))
        refl -= _gaussian_absorption(wl, rng.uniform(1880, 1920), rng.uniform(30, 60), rng.uniform(0.02, 0.12))

    refl = np.clip(refl, 0.01, 0.95)
    return Spectrum(
        name=f"mineral_{idx:04d}",
        category="minerals",
        wavelength_um=wl / 1000.0,
        reflectance=refl.astype(np.float64),
    )


def _generate_vegetation(wl: np.ndarray, rng: np.random.Generator, idx: int) -> Spectrum:
    """Generate a vegetation spectrum with chlorophyll, red edge, water bands."""
    # Low visible reflectance with green peak.
    refl = np.full_like(wl, 0.03, dtype=np.float64)

    # Green peak around 550nm.
    green_peak = rng.uniform(0.05, 0.15)
    refl += green_peak * np.exp(-0.5 * ((wl - 550) / 40) ** 2)

    # Red edge: sharp rise from 680-750nm.
    nir_plateau = rng.uniform(0.3, 0.6)
    red_edge_center = rng.uniform(710, 730)
    red_edge_width = rng.uniform(20, 40)
    refl += nir_plateau / (1 + np.exp(-(wl - red_edge_center) / red_edge_width))

    # Chlorophyll absorption dips.
    refl -= _gaussian_absorption(wl, 430, 20, rng.uniform(0.01, 0.03))
    refl -= _gaussian_absorption(wl, 680, 15, rng.uniform(0.02, 0.05))

    # NIR plateau with slight decline.
    mask_nir = wl > 750
    refl[mask_nir] *= (1 - 0.0001 * (wl[mask_nir] - 750))

    # Water absorption bands.
    water_strength = rng.uniform(0.5, 0.9)  # dry to wet vegetation
    refl -= _gaussian_absorption(wl, 970, 30, 0.03 * water_strength)
    refl -= _gaussian_absorption(wl, 1200, 50, 0.08 * water_strength)
    refl -= _gaussian_absorption(wl, 1450, 50, 0.15 * water_strength)
    refl -= _gaussian_absorption(wl, 1940, 60, 0.2 * water_strength)

    # SWIR decline.
    mask_swir = wl > 1400
    refl[mask_swir] *= np.exp(-0.0003 * (wl[mask_swir] - 1400))

    refl = np.clip(refl, 0.01, 0.95)
    return Spectrum(
        name=f"vegetation_{idx:04d}",
        category="vegetation",
        wavelength_um=wl / 1000.0,
        reflectance=refl.astype(np.float64),
    )


def _generate_soil(wl: np.ndarray, rng: np.random.Generator, idx: int) -> Spectrum:
    """Generate a soil spectrum: smooth, increasing towards NIR."""
    # Base: monotonically increasing with some curvature.
    brightness = rng.uniform(0.05, 0.4)
    slope = rng.uniform(0.00005, 0.0003)
    curve = rng.uniform(-5e-8, 5e-8)
    refl = brightness + slope * (wl - 400) + curve * (wl - 400) ** 2

    # Iron absorption.
    if rng.random() > 0.3:
        refl -= _gaussian_absorption(wl, rng.uniform(480, 530), rng.uniform(60, 150), rng.uniform(0.02, 0.1))
        refl -= _gaussian_absorption(wl, rng.uniform(850, 950), rng.uniform(80, 200), rng.uniform(0.01, 0.08))

    # Clay/water absorption in SWIR.
    if rng.random() > 0.4:
        refl -= _gaussian_absorption(wl, rng.uniform(1400, 1450), rng.uniform(30, 60), rng.uniform(0.01, 0.05))
        refl -= _gaussian_absorption(wl, rng.uniform(1900, 1950), rng.uniform(40, 80), rng.uniform(0.01, 0.06))
        refl -= _gaussian_absorption(wl, rng.uniform(2200, 2250), rng.uniform(20, 50), rng.uniform(0.01, 0.04))

    refl = np.clip(refl, 0.01, 0.95)
    return Spectrum(
        name=f"soil_{idx:04d}",
        category="soils",
        wavelength_um=wl / 1000.0,
        reflectance=refl.astype(np.float64),
    )


def _generate_water(wl: np.ndarray, rng: np.random.Generator, idx: int) -> Spectrum:
    """Generate a water spectrum: low overall, rapidly decreasing in NIR/SWIR."""
    # Very low reflectance, peaks in blue-green.
    turbidity = rng.uniform(0.0, 0.3)
    chlorophyll = rng.uniform(0.0, 0.1)

    refl = 0.02 * np.exp(-0.002 * (wl - 400))
    # Blue-green scatter from turbidity.
    refl += turbidity * 0.1 * np.exp(-0.5 * ((wl - 550) / 100) ** 2)
    # Chlorophyll peak at 550nm with absorption at 670nm.
    refl += chlorophyll * np.exp(-0.5 * ((wl - 550) / 50) ** 2)
    refl -= chlorophyll * 0.5 * np.exp(-0.5 * ((wl - 670) / 20) ** 2)

    # Near-zero above 900nm.
    refl[wl > 900] *= np.exp(-0.01 * (wl[wl > 900] - 900))

    refl = np.clip(refl, 0.001, 0.5)
    return Spectrum(
        name=f"water_{idx:04d}",
        category="water",
        wavelength_um=wl / 1000.0,
        reflectance=refl.astype(np.float64),
    )


def _generate_manmade(wl: np.ndarray, rng: np.random.Generator, idx: int) -> Spectrum:
    """Generate a man-made material spectrum (concrete, asphalt, paint, roof)."""
    material_type = rng.choice(["concrete", "asphalt", "paint", "metal"])

    if material_type == "concrete":
        refl = np.full_like(wl, rng.uniform(0.2, 0.4), dtype=np.float64)
        refl += 0.05 * np.sin(wl / 300)
        refl -= _gaussian_absorption(wl, 2340, 40, rng.uniform(0.02, 0.06))
    elif material_type == "asphalt":
        refl = np.full_like(wl, rng.uniform(0.05, 0.15), dtype=np.float64)
        refl += 0.0001 * (wl - 400)  # slight increase with wavelength
    elif material_type == "paint":
        # Coloured paint: pick a colour peak.
        colour_center = rng.choice([450, 520, 580, 620, 700])
        refl = rng.uniform(0.1, 0.3) + rng.uniform(0.2, 0.5) * np.exp(
            -0.5 * ((wl - colour_center) / rng.uniform(30, 80)) ** 2
        )
    else:  # metal
        refl = np.full_like(wl, rng.uniform(0.3, 0.8), dtype=np.float64)
        refl += rng.uniform(-0.0002, 0.0002) * (wl - 1000)

    # Add slight noise texture.
    refl += rng.normal(0, 0.005, size=len(wl))
    refl = np.clip(refl, 0.01, 0.95)
    return Spectrum(
        name=f"manmade_{material_type}_{idx:04d}",
        category="manmade",
        wavelength_um=wl / 1000.0,
        reflectance=refl.astype(np.float64),
    )


GENERATORS = {
    "minerals": _generate_mineral,
    "vegetation": _generate_vegetation,
    "soils": _generate_soil,
    "water": _generate_water,
    "manmade": _generate_manmade,
}


def generate_synthetic_library(
    n_per_class: int = 100,
    wavelength_nm: np.ndarray | None = None,
    seed: int = 42,
) -> SpectralLibrary:
    """Generate a diverse synthetic spectral library.

    Parameters
    ----------
    n_per_class : number of spectra per material category.
    wavelength_nm : wavelength grid (default: 380-2500nm at 1nm).
    seed : random seed.

    Returns
    -------
    SpectralLibrary with n_per_class * 5 spectra.
    """
    if wavelength_nm is None:
        wavelength_nm = np.arange(380.0, 2501.0, 1.0)

    rng = np.random.default_rng(seed)
    spectra = []
    idx = 0
    for category, gen_fn in GENERATORS.items():
        for _ in range(n_per_class):
            spectra.append(gen_fn(wavelength_nm, rng, idx))
            idx += 1
    return SpectralLibrary(spectra)
