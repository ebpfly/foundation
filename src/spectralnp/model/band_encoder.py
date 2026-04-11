"""Component A: Spectral Band Encoder.

Encodes each spectral band measurement (center wavelength, FWHM, radiance)
into a rich feature vector. The wavelength encoding uses learnable sinusoidal
frequencies so the model can allocate representational capacity to spectrally
dense regions. The radiance encoder is wavelength-conditioned because the
physical meaning of a radiance value depends on where in the spectrum it falls.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SpectralPositionalEncoding(nn.Module):
    """Learnable sinusoidal encoding of wavelength and FWHM.

    Parameters
    ----------
    n_frequencies : int
        Number of sin/cos frequency pairs for wavelength encoding.
    d_model : int
        Output dimensionality.
    wl_range : tuple[float, float]
        Expected wavelength range in nm, used to initialise frequencies
        so they span roughly one period across the range.
    """

    def __init__(
        self,
        n_frequencies: int = 64,
        d_model: int = 256,
        wl_range: tuple[float, float] = (350.0, 2500.0),
    ) -> None:
        super().__init__()
        self.n_frequencies = n_frequencies
        # Initialise log-spaced frequencies spanning the wavelength range.
        # The lowest frequency has ~1 cycle across the full range;
        # the highest has ~n_frequencies cycles.
        wl_span = wl_range[1] - wl_range[0]
        init_freqs = torch.linspace(
            math.log(2 * math.pi / wl_span),
            math.log(2 * math.pi * n_frequencies / wl_span),
            n_frequencies,
        )
        self.log_freqs = nn.Parameter(init_freqs)  # learnable in log-space

        # Small MLP to encode FWHM (bandwidth tells the model spectral resolution).
        self.fwhm_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # Project concatenated [sin/cos wavelength; fwhm features] to d_model.
        self.proj = nn.Linear(2 * n_frequencies + 32, d_model)

    def forward(self, wavelength: Tensor, fwhm: Tensor) -> Tensor:
        """Encode wavelength and FWHM into a positional feature vector.

        Parameters
        ----------
        wavelength : Tensor[...,]
            Center wavelength in nm.
        fwhm : Tensor[...,]
            Full-width at half-maximum in nm.

        Returns
        -------
        Tensor[..., d_model]
        """
        freqs = self.log_freqs.exp()  # (n_frequencies,)
        # wavelength: (...,) -> (..., n_frequencies)
        phase = wavelength.unsqueeze(-1) * freqs
        wl_enc = torch.cat([phase.sin(), phase.cos()], dim=-1)  # (..., 2*n_freq)

        fwhm_enc = self.fwhm_mlp(fwhm.unsqueeze(-1))  # (..., 32)

        return self.proj(torch.cat([wl_enc, fwhm_enc], dim=-1))


class BandEncoder(nn.Module):
    """Encode a set of spectral band measurements into per-band feature vectors.

    Each band is represented by (center_wavelength, fwhm, radiance).
    The radiance MLP is conditioned on the spectral position so the model
    interprets radiance values in their spectral context.

    Parameters
    ----------
    d_model : int
        Feature dimensionality throughout the model.
    n_frequencies : int
        Number of learnable frequency pairs for wavelength encoding.
    """

    def __init__(self, d_model: int = 256, n_frequencies: int = 64) -> None:
        super().__init__()
        self.spectral_pos = SpectralPositionalEncoding(n_frequencies, d_model)

        # Wavelength-conditioned radiance encoder.
        # Input: radiance (1) + spectral position embedding (d_model) -> d_model.
        # The radiance value is the primary signal — spectral position tells
        # the model what the radiance means, but radiance must dominate.
        self.radiance_mlp = nn.Sequential(
            nn.Linear(1 + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Combine: only radiance features go through (no position skip path).
        # Position is already baked into rad_feat via the conditioned MLP.
        # Adding a raw position skip path lets the model bypass radiance.
        self.combine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
    ) -> Tensor:
        """Encode spectral band measurements.

        Parameters
        ----------
        wavelength : Tensor[B, N]
            Center wavelengths in nm.
        fwhm : Tensor[B, N]
            Band widths in nm.
        radiance : Tensor[B, N]
            At-sensor radiance values (W/m^2/sr/um).

        Returns
        -------
        Tensor[B, N, d_model]
            Per-band feature vectors.
        """
        pos = self.spectral_pos(wavelength, fwhm)  # (B, N, d_model)

        # Radiance encoder is conditioned on spectral position.
        # Position tells the model what wavelength region this is;
        # radiance is the actual measured value.
        rad_input = torch.cat([radiance.unsqueeze(-1), pos], dim=-1)
        rad_feat = self.radiance_mlp(rad_input)  # (B, N, d_model)

        # No position skip path — all information must flow through
        # the radiance-conditioned features.
        return self.combine(rad_feat)
