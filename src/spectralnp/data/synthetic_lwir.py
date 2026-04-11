"""Synthetic LWIR (7–16 μm) emissivity library generator.

Generates physically-motivated emissivity spectra for common Earth
surface materials in the LWIR thermal window.  Each material class has
a base model derived from the actual molecular/vibrational features of
that material family:

  - Reststrahlen bands from Si-O vibrations (silicates)
  - Carbonate bending/stretching modes (CO₃²⁻: 11.4, 7, 14 μm)
  - Sulfate modes (SO₄²⁻: 8.6–9.2 μm)
  - Clay Si-O-Al features (9–11 μm)
  - Christiansen feature (emissivity peak before reststrahlen)

Within each class, features are randomized in position, width, depth,
and subset of optional features, so individual spectra are distinct
while preserving class identity.  Target: thousands of unique spectra
per class with inter-spectrum similarity comparable to real libraries
(e.g. ECOSTRESS, ASTER).

Physical reference (approximate feature positions in μm):
    quartz:      8.2, 9.1 (doublet, very strong)
    feldspar:    9.0–10.5 (broad)
    olivine:     10.0–11.0 (broad single)
    pyroxene:    9.5–11.5 (broad)
    carbonate:   7.0, 11.4 (doublet), 14.0
    sulfate:     8.6, 8.8 (gypsum), 9.2
    phosphate:   9.3, 9.6
    clay:        9.0, 9.5, 10.9, 11.0
    vegetation:  flat ~0.98, subtle 10, 11 μm
    water:       flat ~0.99
    ice:         subtle 11, 12 μm
    metal:       low (0.05–0.40) flat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Feature primitives
# ---------------------------------------------------------------------------

def _gaussian(wl: np.ndarray, center: float, fwhm: float, depth: float) -> np.ndarray:
    """Gaussian band shape (positive value = absorption dip)."""
    sigma = fwhm / 2.354820045  # FWHM → σ
    return depth * np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def _lorentzian(wl: np.ndarray, center: float, fwhm: float, depth: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return depth / (1.0 + ((wl - center) / gamma) ** 2)


def _voigt(
    wl: np.ndarray, center: float, fwhm: float, depth: float, eta: float = 0.5,
) -> np.ndarray:
    """Pseudo-Voigt: linear blend of Gaussian and Lorentzian."""
    return (
        (1.0 - eta) * _gaussian(wl, center, fwhm, depth)
        + eta * _lorentzian(wl, center, fwhm, depth)
    )


def _smooth_perturbation(
    wl: np.ndarray,
    rng: np.random.Generator,
    amplitude: float = 0.02,
    n_components: int = 5,
) -> np.ndarray:
    """Smooth sinusoidal baseline perturbation for uniqueness."""
    wl_norm = (wl - wl.min()) / (wl.max() - wl.min())
    out = np.zeros_like(wl)
    for _ in range(n_components):
        freq = rng.uniform(0.3, 3.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(-amplitude, amplitude) / n_components
        out += amp * np.sin(2 * np.pi * freq * wl_norm + phase)
    return out


def _linear_slope(
    wl: np.ndarray, rng: np.random.Generator, max_amp: float = 0.02,
) -> np.ndarray:
    """Slight linear tilt in emissivity across the spectrum."""
    wl_norm = (wl - wl.min()) / (wl.max() - wl.min())
    return (wl_norm - 0.5) * rng.uniform(-max_amp, max_amp)


# ---------------------------------------------------------------------------
# Class generators — each returns one emissivity spectrum in [0, 1]
# ---------------------------------------------------------------------------

def _gen_quartz(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Quartz: characteristic Si-O reststrahlen doublet at 8.2 & 9.1 μm."""
    base = 0.955 + rng.uniform(-0.020, 0.020)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.015)
    e += _linear_slope(wl, rng, max_amp=0.02)

    # Christiansen peak (sharp emissivity rise ~7.4–7.8 μm)
    chris_pos = rng.uniform(7400, 7800)
    e += _gaussian(wl, chris_pos, rng.uniform(200, 350), rng.uniform(0.020, 0.040))

    # Primary reststrahlen (8.2 μm, always present)
    r1 = rng.uniform(8050, 8300)
    e -= _voigt(wl, r1,
                fwhm=rng.uniform(300, 550),
                depth=rng.uniform(0.12, 0.28),
                eta=rng.uniform(0.3, 0.7))

    # Secondary reststrahlen (9.1 μm, stronger, always present)
    r2 = rng.uniform(8900, 9250)
    e -= _voigt(wl, r2,
                fwhm=rng.uniform(400, 700),
                depth=rng.uniform(0.18, 0.38),
                eta=rng.uniform(0.3, 0.7))

    # 12.5 μm Si-O bending (sometimes)
    if rng.random() < 0.65:
        e -= _gaussian(wl, rng.uniform(12200, 12800), rng.uniform(300, 550),
                        rng.uniform(0.02, 0.08))

    # 14 μm tail
    if rng.random() < 0.3:
        e -= _gaussian(wl, rng.uniform(13800, 14400), rng.uniform(400, 800),
                        rng.uniform(0.015, 0.05))

    return np.clip(e, 0.0, 1.0)


def _gen_feldspar(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Feldspar: broad reststrahlen 9–10.5 μm with secondary structure."""
    base = 0.935 + rng.uniform(-0.025, 0.025)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.015)
    e += _linear_slope(wl, rng, max_amp=0.025)

    e += _gaussian(wl, rng.uniform(7600, 8000), rng.uniform(250, 400),
                    rng.uniform(0.015, 0.030))

    # Main reststrahlen (broad)
    e -= _voigt(wl, rng.uniform(9400, 10300),
                fwhm=rng.uniform(700, 1400),
                depth=rng.uniform(0.15, 0.30),
                eta=rng.uniform(0.3, 0.7))

    # Triplet substructure (1–3 extra features)
    for _ in range(rng.integers(1, 4)):
        e -= _gaussian(
            wl, rng.uniform(8500, 11500),
            rng.uniform(150, 350),
            rng.uniform(0.03, 0.10),
        )

    return np.clip(e, 0.0, 1.0)


def _gen_olivine(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Olivine: single broad Si-O reststrahlen 10–11 μm."""
    base = 0.920 + rng.uniform(-0.025, 0.025)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.015)
    e += _linear_slope(wl, rng, max_amp=0.03)

    e -= _voigt(wl, rng.uniform(9800, 11000),
                fwhm=rng.uniform(1200, 2500),
                depth=rng.uniform(0.10, 0.22),
                eta=rng.uniform(0.3, 0.7))

    if rng.random() < 0.4:
        e -= _gaussian(wl, rng.uniform(11500, 12500), rng.uniform(300, 600),
                        rng.uniform(0.02, 0.07))
    return np.clip(e, 0.0, 1.0)


def _gen_pyroxene(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pyroxene: broad reststrahlen 9.5–11.5 μm."""
    base = 0.925 + rng.uniform(-0.025, 0.025)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.015)

    e -= _voigt(wl, rng.uniform(9500, 11500),
                fwhm=rng.uniform(1000, 2200),
                depth=rng.uniform(0.10, 0.22),
                eta=rng.uniform(0.3, 0.7))
    if rng.random() < 0.5:
        e -= _gaussian(wl, rng.uniform(8800, 9500), rng.uniform(200, 400),
                        rng.uniform(0.02, 0.08))
    return np.clip(e, 0.0, 1.0)


def _gen_carbonate(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Carbonate (calcite, dolomite, magnesite): CO₃²⁻ modes at 7, 11.4, 14 μm."""
    base = 0.950 + rng.uniform(-0.015, 0.020)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.012)
    e += _linear_slope(wl, rng, max_amp=0.02)

    # Main 11.4 μm doublet (CO3 ν3 asymmetric stretch)
    center1 = rng.uniform(11100, 11450)
    e -= _gaussian(wl, center1,
                    fwhm=rng.uniform(200, 400),
                    depth=rng.uniform(0.08, 0.20))
    # Second peak slightly shifted
    e -= _gaussian(wl, center1 + rng.uniform(250, 500),
                    fwhm=rng.uniform(200, 400),
                    depth=rng.uniform(0.05, 0.14))

    # 7 μm feature (CO3 ν1 + translational)
    if wl.min() < 7200 and rng.random() < 0.75:
        e -= _gaussian(wl, rng.uniform(6900, 7200),
                        fwhm=rng.uniform(150, 300),
                        depth=rng.uniform(0.05, 0.13))

    # 14 μm (CO3 out-of-plane bend ν2)
    if rng.random() < 0.55:
        e -= _gaussian(wl, rng.uniform(13800, 14300),
                        fwhm=rng.uniform(200, 400),
                        depth=rng.uniform(0.03, 0.10))

    return np.clip(e, 0.0, 1.0)


def _gen_sulfate(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sulfate (gypsum, anhydrite, barite): SO₄²⁻ modes 8.6–9.2 μm."""
    base = 0.935 + rng.uniform(-0.020, 0.020)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.015)

    # SO4 ν3 doublet/triplet near 8.6–9.2 μm
    for _ in range(rng.integers(2, 5)):
        e -= _gaussian(wl, rng.uniform(8400, 9300),
                        fwhm=rng.uniform(120, 280),
                        depth=rng.uniform(0.06, 0.18))

    # Gypsum has bound water — weak broad feature ~12–13 μm
    if rng.random() < 0.5:
        e -= _gaussian(wl, rng.uniform(12000, 13000),
                        fwhm=rng.uniform(400, 800),
                        depth=rng.uniform(0.02, 0.08))

    # Occasionally a 15 μm feature
    if rng.random() < 0.3:
        e -= _gaussian(wl, rng.uniform(14800, 15500),
                        fwhm=rng.uniform(300, 600),
                        depth=rng.uniform(0.02, 0.06))

    return np.clip(e, 0.0, 1.0)


def _gen_phosphate(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phosphate (apatite): PO₄³⁻ modes at 9.3, 9.6 μm."""
    base = 0.935 + rng.uniform(-0.020, 0.020)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.013)

    # PO4 doublet
    c1 = rng.uniform(9200, 9400)
    e -= _gaussian(wl, c1, rng.uniform(150, 280), rng.uniform(0.07, 0.16))
    e -= _gaussian(wl, c1 + rng.uniform(200, 400),
                    rng.uniform(150, 280), rng.uniform(0.05, 0.13))

    # OH libration feature if hydroxyapatite
    if rng.random() < 0.4:
        e -= _gaussian(wl, rng.uniform(11000, 11800),
                        rng.uniform(300, 600), rng.uniform(0.03, 0.09))

    return np.clip(e, 0.0, 1.0)


def _gen_clay(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clay minerals (kaolinite, illite, smectite): multiple narrow features 9–11 μm."""
    base = 0.940 + rng.uniform(-0.025, 0.020)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.015)
    e += _linear_slope(wl, rng, max_amp=0.025)

    # Multiple Si-O-Al / Si-O features
    for _ in range(rng.integers(3, 7)):
        e -= _gaussian(
            wl,
            rng.uniform(8800, 11500),
            fwhm=rng.uniform(80, 250),
            depth=rng.uniform(0.03, 0.12),
        )

    # Optional broader reststrahlen backdrop
    if rng.random() < 0.6:
        e -= _voigt(wl, rng.uniform(9500, 10500),
                    fwhm=rng.uniform(800, 1500),
                    depth=rng.uniform(0.04, 0.13),
                    eta=rng.uniform(0.3, 0.7))

    return np.clip(e, 0.0, 1.0)


def _gen_vegetation(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vegetation: very high emissivity, nearly flat, subtle cellulose features."""
    base = rng.uniform(0.960, 0.990)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.006)
    e += _linear_slope(wl, rng, max_amp=0.008)

    # Cellulose / lignin subtle features
    if rng.random() < 0.6:
        e -= _gaussian(wl, rng.uniform(9500, 10500),
                        rng.uniform(400, 800), rng.uniform(0.004, 0.015))
    if rng.random() < 0.5:
        e -= _gaussian(wl, rng.uniform(11000, 12500),
                        rng.uniform(400, 900), rng.uniform(0.003, 0.012))
    if rng.random() < 0.3:
        e -= _gaussian(wl, rng.uniform(13500, 14500),
                        rng.uniform(500, 1200), rng.uniform(0.003, 0.010))

    return np.clip(e, 0.0, 1.0)


def _gen_water(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Liquid water: very flat, emissivity ~0.99."""
    base = rng.uniform(0.985, 0.995)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.003)
    e += _linear_slope(wl, rng, max_amp=0.005)
    # Broad very subtle feature at ~12 μm (liquid water OH bend tail)
    if rng.random() < 0.4:
        e -= _gaussian(wl, rng.uniform(11500, 13500),
                        rng.uniform(1000, 2500), rng.uniform(0.002, 0.006))
    return np.clip(e, 0.0, 1.0)


def _gen_ice(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ice / snow: high emissivity with subtle 11 μm feature."""
    base = rng.uniform(0.975, 0.990)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.005)

    e -= _gaussian(wl, rng.uniform(10800, 11400),
                    rng.uniform(500, 900), rng.uniform(0.02, 0.06))
    if rng.random() < 0.5:
        e -= _gaussian(wl, rng.uniform(12200, 13200),
                        rng.uniform(600, 1200), rng.uniform(0.01, 0.035))
    return np.clip(e, 0.0, 1.0)


def _gen_soil(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soil: mixture of quartz, clay, carbonate (composition-dependent)."""
    base = rng.uniform(0.900, 0.960)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.018)
    e += _linear_slope(wl, rng, max_amp=0.03)

    # Weakened quartz reststrahlen (most soils have some quartz)
    if rng.random() < 0.85:
        e -= _voigt(wl, rng.uniform(8800, 9300),
                    fwhm=rng.uniform(500, 900),
                    depth=rng.uniform(0.04, 0.18),
                    eta=rng.uniform(0.3, 0.7))
    # Secondary quartz
    if rng.random() < 0.6:
        e -= _gaussian(wl, rng.uniform(8100, 8400),
                        rng.uniform(200, 500), rng.uniform(0.02, 0.10))

    # Clay features
    if rng.random() < 0.65:
        for _ in range(rng.integers(1, 5)):
            e -= _gaussian(wl, rng.uniform(9500, 11500),
                            rng.uniform(150, 400),
                            rng.uniform(0.015, 0.08))

    # Occasional carbonate (calcareous soils)
    if rng.random() < 0.25:
        e -= _gaussian(wl, rng.uniform(11200, 11500),
                        rng.uniform(200, 400), rng.uniform(0.03, 0.11))

    # Organic matter (very subtle, reduces contrast)
    if rng.random() < 0.4:
        e += _smooth_perturbation(wl, rng, amplitude=0.005)

    return np.clip(e, 0.0, 1.0)


def _gen_concrete(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Concrete: quartz-dominated with carbonate (cement) signatures."""
    base = rng.uniform(0.920, 0.960)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.015)

    # Quartz (aggregate)
    e -= _voigt(wl, rng.uniform(8900, 9300),
                fwhm=rng.uniform(500, 900),
                depth=rng.uniform(0.06, 0.18),
                eta=rng.uniform(0.3, 0.7))
    if rng.random() < 0.6:
        e -= _gaussian(wl, rng.uniform(8100, 8400),
                        rng.uniform(200, 450), rng.uniform(0.02, 0.09))
    # Carbonate (cement calcite)
    if rng.random() < 0.6:
        e -= _gaussian(wl, rng.uniform(11200, 11500),
                        rng.uniform(200, 400), rng.uniform(0.03, 0.10))
    return np.clip(e, 0.0, 1.0)


def _gen_asphalt(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Asphalt / bitumen: nearly flat, slight hydrocarbon dips."""
    base = rng.uniform(0.930, 0.960)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.018)
    e += _linear_slope(wl, rng, max_amp=0.02)

    # Aggregate quartz can show through
    if rng.random() < 0.35:
        e -= _voigt(wl, rng.uniform(8900, 9300),
                    fwhm=rng.uniform(400, 800),
                    depth=rng.uniform(0.02, 0.08),
                    eta=rng.uniform(0.3, 0.7))
    # Weak C-H bending overtones
    if rng.random() < 0.5:
        e -= _gaussian(wl, rng.uniform(13500, 14500),
                        rng.uniform(400, 800), rng.uniform(0.01, 0.03))
    return np.clip(e, 0.0, 1.0)


def _gen_paint(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint: smooth with 1–3 polymer/pigment features."""
    base = rng.uniform(0.900, 0.975)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.025)
    e += _linear_slope(wl, rng, max_amp=0.04)

    # Polymer / pigment features
    for _ in range(rng.integers(0, 4)):
        e -= _gaussian(
            wl,
            rng.uniform(wl.min() + 500, wl.max() - 500),
            fwhm=rng.uniform(200, 800),
            depth=rng.uniform(0.02, 0.10),
        )
    return np.clip(e, 0.0, 1.0)


def _gen_metal(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Metal: low, nearly flat emissivity with slight slope."""
    base = rng.uniform(0.05, 0.40)
    e = np.full_like(wl, base, dtype=np.float64)
    e += _smooth_perturbation(wl, rng, amplitude=0.025)

    # Slight slope (metals often emit more at longer wavelengths)
    slope = rng.uniform(-0.03, 0.10)
    wl_norm = (wl - wl.min()) / (wl.max() - wl.min())
    e += slope * wl_norm

    # Oxide layers can add features
    if rng.random() < 0.35:
        e += _gaussian(
            wl,
            rng.uniform(wl.min(), wl.max()),
            fwhm=rng.uniform(800, 2000),
            depth=rng.uniform(-0.15, -0.04),  # positive bump
        )
    return np.clip(e, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "quartz":     _gen_quartz,
    "feldspar":   _gen_feldspar,
    "olivine":    _gen_olivine,
    "pyroxene":   _gen_pyroxene,
    "carbonate":  _gen_carbonate,
    "sulfate":    _gen_sulfate,
    "phosphate":  _gen_phosphate,
    "clay":       _gen_clay,
    "vegetation": _gen_vegetation,
    "water":      _gen_water,
    "ice":        _gen_ice,
    "soil":       _gen_soil,
    "concrete":   _gen_concrete,
    "asphalt":    _gen_asphalt,
    "paint":      _gen_paint,
    "metal":      _gen_metal,
}

CATEGORY_NAMES: list[str] = list(GENERATORS.keys())


# ---------------------------------------------------------------------------
# Library generation
# ---------------------------------------------------------------------------

@dataclass
class LWIRLibrary:
    wavelength_nm: np.ndarray     # (n_bands,) float32
    emissivity: np.ndarray        # (n_spectra, n_bands) float32
    names: list[str]
    categories: list[str]

    def __len__(self) -> int:
        return self.emissivity.shape[0]

    @property
    def n_bands(self) -> int:
        return self.emissivity.shape[1]


def generate_lwir_library(
    n_total: int = 10_000,
    wavelength_lo_um: float = 7.0,
    wavelength_hi_um: float = 16.0,
    n_bands: int = 4_000,
    seed: int = 42,
    weights: dict[str, float] | None = None,
) -> LWIRLibrary:
    """Generate a synthetic LWIR emissivity library.

    Parameters
    ----------
    n_total : total number of spectra to generate
    wavelength_lo_um : lower bound in μm
    wavelength_hi_um : upper bound in μm
    n_bands : number of wavelength samples (linearly spaced)
    seed : RNG seed
    weights : optional dict of class name → relative weight.
        If None, classes are approximately equally represented.

    Returns
    -------
    LWIRLibrary with fields wavelength_nm, emissivity, names, categories.
    """
    rng = np.random.default_rng(seed)

    wl = np.linspace(
        wavelength_lo_um * 1000.0,
        wavelength_hi_um * 1000.0,
        n_bands,
    )

    class_names = list(GENERATORS.keys())
    n_classes = len(class_names)

    if weights is None:
        weights = {c: 1.0 for c in class_names}
    w_arr = np.array([weights.get(c, 1.0) for c in class_names], dtype=np.float64)
    w_arr = w_arr / w_arr.sum()
    # Largest-remainder allocation for integer counts
    raw = w_arr * n_total
    counts = np.floor(raw).astype(int)
    remainder_rank = np.argsort(-(raw - counts))
    deficit = n_total - counts.sum()
    for i in remainder_rank[:deficit]:
        counts[i] += 1

    emissivity = np.zeros((n_total, n_bands), dtype=np.float32)
    names: list[str] = []
    categories: list[str] = []

    idx = 0
    for cat, count in zip(class_names, counts):
        gen = GENERATORS[cat]
        for i in range(int(count)):
            spec_seed = int(rng.integers(0, 2**31 - 1))
            spec_rng = np.random.default_rng(spec_seed)
            spec = gen(wl, spec_rng)
            emissivity[idx] = spec.astype(np.float32)
            names.append(f"{cat}_{i:06d}")
            categories.append(cat)
            idx += 1

    return LWIRLibrary(
        wavelength_nm=wl.astype(np.float32),
        emissivity=emissivity,
        names=names,
        categories=categories,
    )
