"""SpectralNP: full model assembly.

Wires together BandEncoder -> SpectralAggregator -> Decoders.
Provides forward pass with reparameterised sampling and multi-sample
uncertainty inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from spectralnp.model.band_encoder import BandEncoder
from spectralnp.model.spectral_aggregator import SpectralAggregator
from spectralnp.model.decoders import AtmosphericDecoder, MaterialDecoder, SpectralDecoder


@dataclass
class SpectralNPConfig:
    """Model hyper-parameters."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    n_frequencies: int = 64
    n_latents: int = 64
    z_dim: int = 128
    dropout: float = 0.0
    # Decoder settings
    spectral_hidden: int = 512
    spectral_n_layers: int = 4
    n_material_classes: int = 100
    n_atmos_params: int = 4


@dataclass
class SpectralNPOutput:
    """Container for all model outputs from a single forward pass."""

    # Spectral reconstruction
    spectral_mu: Tensor | None = None       # (B, Q)
    spectral_log_var: Tensor | None = None  # (B, Q)

    # Material classification
    material_logits: Tensor | None = None   # (B, n_classes)

    # Atmospheric parameters (NIG)
    atmos_gamma: Tensor | None = None       # (B, n_params) — predicted mean
    atmos_nu: Tensor | None = None          # (B, n_params)
    atmos_alpha: Tensor | None = None       # (B, n_params)
    atmos_beta: Tensor | None = None        # (B, n_params)

    # Latent distribution (for KL loss)
    z_mu: Tensor | None = None              # (B, z_dim)
    z_log_sigma: Tensor | None = None       # (B, z_dim)

    # Deterministic representation (for downstream probing)
    r: Tensor | None = None                 # (B, d_model)


class SpectralNP(nn.Module):
    """Sensor-agnostic spectral foundation model with calibrated uncertainty.

    Accepts spectral measurements from any sensor (arbitrary number of bands,
    arbitrary wavelength positions and widths) and produces:
    - Continuous spectral reconstruction at any query wavelength
    - Material / surface property classification
    - Atmospheric parameter estimation
    All with principled uncertainty that widens when fewer bands are observed.
    """

    def __init__(self, cfg: SpectralNPConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SpectralNPConfig()
        self.cfg = cfg

        self.encoder = BandEncoder(cfg.d_model, cfg.n_frequencies)
        self.aggregator = SpectralAggregator(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            n_latents=cfg.n_latents,
            z_dim=cfg.z_dim,
            dropout=cfg.dropout,
        )
        self.spectral_decoder = SpectralDecoder(
            d_model=cfg.d_model,
            z_dim=cfg.z_dim,
            n_frequencies=cfg.n_frequencies,
            hidden=cfg.spectral_hidden,
            n_layers=cfg.spectral_n_layers,
        )
        self.material_decoder = MaterialDecoder(
            d_model=cfg.d_model,
            z_dim=cfg.z_dim,
            n_classes=cfg.n_material_classes,
        )
        self.atmos_decoder = AtmosphericDecoder(
            d_model=cfg.d_model,
            z_dim=cfg.z_dim,
            n_params=cfg.n_atmos_params,
        )

    def _reparameterise(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
        """Sample z via the reparameterisation trick."""
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * log_sigma.exp()
        return mu  # deterministic at eval (unless using multi-sample)

    def encode(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode spectral bands to (r, z_mu, z_log_sigma).

        This is exposed separately so users can cache the representation
        and run multiple decoders or sample z multiple times.
        """
        h = self.encoder(wavelength, fwhm, radiance)
        r, z_mu, z_log_sigma = self.aggregator(h, wavelength, pad_mask)
        return r, z_mu, z_log_sigma

    def forward(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
        query_wavelength: Tensor | None = None,
        query_fwhm: Tensor | None = None,
    ) -> SpectralNPOutput:
        """Full forward pass.

        Parameters
        ----------
        wavelength : Tensor[B, N]
            Center wavelengths of input bands (nm).
        fwhm : Tensor[B, N]
            Band widths (nm).
        radiance : Tensor[B, N]
            At-sensor radiance (W/m^2/sr/um).
        pad_mask : Tensor[B, N] bool, optional
            True = valid band, False = padding.
        query_wavelength : Tensor[B, Q], optional
            Wavelengths for spectral reconstruction. If None, spectral
            decoder is skipped.
        query_fwhm : Tensor[B, Q], optional
            FWHM for query wavelengths.

        Returns
        -------
        SpectralNPOutput
        """
        r, z_mu, z_log_sigma = self.encode(wavelength, fwhm, radiance, pad_mask)
        z = self._reparameterise(z_mu, z_log_sigma)

        out = SpectralNPOutput(r=r, z_mu=z_mu, z_log_sigma=z_log_sigma)

        # Spectral reconstruction (only if query wavelengths provided).
        if query_wavelength is not None:
            out.spectral_mu, out.spectral_log_var = self.spectral_decoder(
                r, z, query_wavelength, query_fwhm
            )

        # Material classification.
        out.material_logits = self.material_decoder(r, z)

        # Atmospheric parameters.
        out.atmos_gamma, out.atmos_nu, out.atmos_alpha, out.atmos_beta = (
            self.atmos_decoder(r, z)
        )

        return out

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
        query_wavelength: Tensor | None = None,
        query_fwhm: Tensor | None = None,
        n_samples: int = 16,
    ) -> dict[str, Tensor]:
        """Multi-sample inference for calibrated uncertainty.

        Draws n_samples from q(z|context) and aggregates predictions.
        Epistemic uncertainty comes from the spread across samples.

        Returns dict with keys like 'spectral_mean', 'spectral_std',
        'material_probs', 'material_entropy', 'atmos_mean', 'atmos_std', etc.
        """
        self.eval()
        r, z_mu, z_log_sigma = self.encode(wavelength, fwhm, radiance, pad_mask)

        spectral_samples = []
        material_logit_samples = []
        atmos_gamma_samples = []
        atmos_epistemic_samples = []

        for _ in range(n_samples):
            eps = torch.randn_like(z_mu)
            z = z_mu + eps * z_log_sigma.exp()

            if query_wavelength is not None:
                s_mu, s_logvar = self.spectral_decoder(r, z, query_wavelength, query_fwhm)
                spectral_samples.append(s_mu)

            material_logit_samples.append(self.material_decoder(r, z))

            ag, an, aa, ab = self.atmos_decoder(r, z)
            atmos_gamma_samples.append(ag)
            # Per-sample evidential epistemic uncertainty.
            atmos_epistemic_samples.append(ab / (an * (aa - 1.0)))

        results: dict[str, Tensor] = {}

        if spectral_samples:
            stacked = torch.stack(spectral_samples)  # (S, B, Q)
            results["spectral_mean"] = stacked.mean(0)
            results["spectral_std"] = stacked.std(0)

        mat_stack = torch.stack(material_logit_samples)  # (S, B, C)
        mat_probs = mat_stack.softmax(dim=-1).mean(0)  # (B, C)
        results["material_probs"] = mat_probs
        results["material_entropy"] = -(mat_probs * (mat_probs + 1e-10).log()).sum(-1)

        atmos_stack = torch.stack(atmos_gamma_samples)  # (S, B, P)
        results["atmos_mean"] = atmos_stack.mean(0)
        results["atmos_std"] = atmos_stack.std(0)  # epistemic from z
        # Mean of per-sample evidential epistemic (additional uncertainty).
        results["atmos_evidential_epistemic"] = torch.stack(atmos_epistemic_samples).mean(0)

        return results
