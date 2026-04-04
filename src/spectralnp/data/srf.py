"""Spectral response function: pseudo-Voigt profile.

A pseudo-Voigt is a linear combination of Gaussian and Lorentzian
with the same FWHM.  The mixing parameter eta (0=pure Gaussian,
1=pure Lorentzian) controls the wing shape.  Real sensor SRFs
typically have heavier tails than a pure Gaussian, making pVoigt
a better approximation.
"""

from __future__ import annotations

import numpy as np


def pseudo_voigt(
    wavelength: np.ndarray,
    center: np.ndarray,
    fwhm: np.ndarray,
    eta: float = 0.5,
) -> np.ndarray:
    """Compute pseudo-Voigt spectral response functions.

    Parameters
    ----------
    wavelength : (W,) or (1, W) wavelength grid
    center : (N,) or (N, 1) band centers
    fwhm : (N,) or (N, 1) full-width at half-maximum
    eta : Lorentzian mixing fraction in [0, 1].
          0 = pure Gaussian, 1 = pure Lorentzian.

    Returns
    -------
    (N, W) array of response weights, each normalised to sum to 1.
    """
    wl = np.atleast_2d(wavelength)          # (1, W)
    c = np.atleast_2d(center)               # (N, 1) or (N,)->needs reshape
    f = np.atleast_2d(fwhm)
    if c.shape[0] == 1 and c.shape[1] > 1:
        c = c.T
        f = f.T

    sigma = f / 2.355                       # Gaussian sigma
    gamma = f / 2.0                         # Lorentzian half-width

    dx = wl - c
    gauss = np.exp(-0.5 * (dx / sigma) ** 2)
    lorentz = 1.0 / (1.0 + (dx / gamma) ** 2)

    profile = (1.0 - eta) * gauss + eta * lorentz
    profile /= profile.sum(axis=-1, keepdims=True) + 1e-30
    return profile
