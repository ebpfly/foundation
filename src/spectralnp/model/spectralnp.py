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
from spectralnp.model.spectral_aggregator import CrossPixelAggregator, SpectralAggregator
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
    # Hierarchical latent: z_atm (shared across pixels) + z_surf (per-pixel).
    # z_atm_dim + z_surf_dim = z_dim.  When context pixels are provided,
    # z_atm is inferred from all pixels jointly; otherwise it uses a wide prior.
    z_atm_dim: int = 32
    z_surf_dim: int = 96
    # Decoder settings
    spectral_hidden: int = 512
    spectral_n_layers: int = 4
    spectral_decoder_use_r: bool = True  # if False, drop r from decoder input (forces use of z)
    n_material_classes: int = 100
    n_atmos_params: int = 4


@dataclass
class SpectralNPOutput:
    """Container for all model outputs from a single forward pass."""

    # Spectral reconstruction (at-sensor radiance)
    spectral_mu: Tensor | None = None       # (B, Q)
    spectral_log_var: Tensor | None = None  # (B, Q)

    # Surface reflectance reconstruction (atmosphere-corrected)
    reflectance_mu: Tensor | None = None        # (B, Q)
    reflectance_log_var: Tensor | None = None   # (B, Q)

    # Material classification
    material_logits: Tensor | None = None   # (B, n_classes)

    # Atmospheric parameters (NIG)
    atmos_gamma: Tensor | None = None       # (B, n_params) — predicted mean
    atmos_nu: Tensor | None = None          # (B, n_params)
    atmos_alpha: Tensor | None = None       # (B, n_params)
    atmos_beta: Tensor | None = None        # (B, n_params)

    # Latent distributions (for KL loss)
    z_mu: Tensor | None = None              # (B, z_dim) — combined z for compat
    z_log_sigma: Tensor | None = None       # (B, z_dim)
    z_atm_mu: Tensor | None = None          # (B, z_atm_dim)
    z_atm_log_sigma: Tensor | None = None   # (B, z_atm_dim)
    z_surf_mu: Tensor | None = None         # (B, z_surf_dim)
    z_surf_log_sigma: Tensor | None = None  # (B, z_surf_dim)

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

    Supports optional multi-pixel context: if context pixels from the same
    scene are provided, a shared atmospheric latent z_atm is inferred jointly,
    improving atmospheric separation.  With a single pixel (default), z_atm
    has a wide posterior and the model degrades gracefully.
    """

    def __init__(self, cfg: SpectralNPConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SpectralNPConfig()
        self.cfg = cfg

        z_total = cfg.z_atm_dim + cfg.z_surf_dim

        self.encoder = BandEncoder(cfg.d_model, cfg.n_frequencies)
        self.aggregator = SpectralAggregator(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            n_latents=cfg.n_latents,
            z_dim=cfg.z_surf_dim,       # per-pixel surface latent
            dropout=cfg.dropout,
        )
        self.cross_pixel = CrossPixelAggregator(
            d_model=cfg.d_model,
            z_atm_dim=cfg.z_atm_dim,
            n_heads=cfg.n_heads,
        )

        # Decoders consume z = cat(z_atm, z_surf).
        # Radiance head — at-sensor TOA radiance.
        self.spectral_decoder = SpectralDecoder(
            d_model=cfg.d_model,
            z_dim=z_total,
            n_frequencies=cfg.n_frequencies,
            hidden=cfg.spectral_hidden,
            n_layers=cfg.spectral_n_layers,
            use_r=cfg.spectral_decoder_use_r,
        )
        # Reflectance head — atmosphere-corrected surface reflectance.
        # Same architecture as the radiance decoder; learns its own weights.
        self.reflectance_decoder = SpectralDecoder(
            d_model=cfg.d_model,
            z_dim=z_total,
            n_frequencies=cfg.n_frequencies,
            hidden=cfg.spectral_hidden,
            n_layers=cfg.spectral_n_layers,
            use_r=cfg.spectral_decoder_use_r,
        )
        self.material_decoder = MaterialDecoder(
            d_model=cfg.d_model,
            z_dim=z_total,
            n_classes=cfg.n_material_classes,
        )
        # Atmospheric decoder uses z_atm only — the atmosphere is shared,
        # it should not depend on per-pixel surface properties.
        self.atmos_decoder = AtmosphericDecoder(
            d_model=cfg.d_model,
            z_dim=cfg.z_atm_dim,
            n_params=cfg.n_atmos_params,
        )

    def _reparameterise(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
        """Sample z via the reparameterisation trick."""
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * log_sigma.exp()
        return mu

    def _encode_single_pixel(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode bands for a single pixel, return (r, z_surf_mu, z_surf_log_sigma)."""
        h = self.encoder(wavelength, fwhm, radiance)
        r, z_surf_mu, z_surf_log_sigma = self.aggregator(h, wavelength, pad_mask)
        return r, z_surf_mu, z_surf_log_sigma

    def encode(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
        context_wavelength: Tensor | None = None,
        context_fwhm: Tensor | None = None,
        context_radiance: Tensor | None = None,
        context_pad_mask: Tensor | None = None,
        context_pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode spectral bands with optional multi-pixel context.

        Parameters
        ----------
        wavelength, fwhm, radiance : Tensor[B, N]
            Target pixel bands.
        pad_mask : Tensor[B, N], optional
            Band padding mask for target pixel.
        context_wavelength : Tensor[B, K, N_ctx], optional
            K context pixel bands.  If None, z_atm comes from the target
            pixel alone (single-pixel mode).
        context_fwhm : Tensor[B, K, N_ctx], optional
        context_radiance : Tensor[B, K, N_ctx], optional
        context_pad_mask : Tensor[B, K, N_ctx], optional
            Band padding within each context pixel.
        context_pixel_mask : Tensor[B, K], optional
            Which context pixels are valid (for variable K).

        Returns
        -------
        r, z_atm_mu, z_atm_log_sigma, z_surf_mu, z_surf_log_sigma, z_mu, z_log_sigma
        """
        # Encode target pixel.
        r, z_surf_mu, z_surf_log_sigma = self._encode_single_pixel(
            wavelength, fwhm, radiance, pad_mask
        )

        # Encode context pixels for atmospheric latent.
        if context_wavelength is not None:
            B, K, N_ctx = context_wavelength.shape
            # Flatten context pixels: (B*K, N_ctx).
            flat_wl = context_wavelength.reshape(B * K, N_ctx)
            flat_fw = context_fwhm.reshape(B * K, N_ctx)
            flat_rad = context_radiance.reshape(B * K, N_ctx)
            flat_mask = context_pad_mask.reshape(B * K, N_ctx) if context_pad_mask is not None else None

            # Get per-context-pixel representations.
            ctx_r, _, _ = self._encode_single_pixel(flat_wl, flat_fw, flat_rad, flat_mask)
            ctx_r = ctx_r.reshape(B, K, -1)  # (B, K, d_model)

            # Also include the target pixel as context.
            pixel_reps = torch.cat([r.unsqueeze(1), ctx_r], dim=1)  # (B, K+1, d_model)
            if context_pixel_mask is not None:
                target_valid = torch.ones(B, 1, dtype=torch.bool, device=r.device)
                pixel_mask = torch.cat([target_valid, context_pixel_mask], dim=1)
            else:
                pixel_mask = None

            z_atm_mu, z_atm_log_sigma = self.cross_pixel(pixel_reps, pixel_mask)
        else:
            # Single pixel: z_atm from target pixel alone.
            z_atm_mu, z_atm_log_sigma = self.cross_pixel(r.unsqueeze(1))

        # Combined z for backward compatibility.
        z_mu = torch.cat([z_atm_mu, z_surf_mu], dim=-1)
        z_log_sigma = torch.cat([z_atm_log_sigma, z_surf_log_sigma], dim=-1)

        return r, z_atm_mu, z_atm_log_sigma, z_surf_mu, z_surf_log_sigma, z_mu, z_log_sigma

    def forward(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
        query_wavelength: Tensor | None = None,
        query_fwhm: Tensor | None = None,
        context_wavelength: Tensor | None = None,
        context_fwhm: Tensor | None = None,
        context_radiance: Tensor | None = None,
        context_pad_mask: Tensor | None = None,
        context_pixel_mask: Tensor | None = None,
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
            Wavelengths for spectral reconstruction.
        query_fwhm : Tensor[B, Q], optional
        context_wavelength : Tensor[B, K, N_ctx], optional
            Context pixel bands (same atmosphere).
        context_fwhm, context_radiance, context_pad_mask, context_pixel_mask:
            See encode() for details.

        Returns
        -------
        SpectralNPOutput
        """
        r, z_atm_mu, z_atm_ls, z_surf_mu, z_surf_ls, z_mu, z_log_sigma = self.encode(
            wavelength, fwhm, radiance, pad_mask,
            context_wavelength, context_fwhm, context_radiance,
            context_pad_mask, context_pixel_mask,
        )

        z_atm = self._reparameterise(z_atm_mu, z_atm_ls)
        z_surf = self._reparameterise(z_surf_mu, z_surf_ls)
        z = torch.cat([z_atm, z_surf], dim=-1)

        out = SpectralNPOutput(
            r=r,
            z_mu=z_mu, z_log_sigma=z_log_sigma,
            z_atm_mu=z_atm_mu, z_atm_log_sigma=z_atm_ls,
            z_surf_mu=z_surf_mu, z_surf_log_sigma=z_surf_ls,
        )

        # Spectral reconstruction (surface + atmosphere).
        if query_wavelength is not None:
            out.spectral_mu, out.spectral_log_var = self.spectral_decoder(
                r, z, query_wavelength, query_fwhm
            )
            out.reflectance_mu, out.reflectance_log_var = self.reflectance_decoder(
                r, z, query_wavelength, query_fwhm
            )

        # Material classification (surface + atmosphere).
        out.material_logits = self.material_decoder(r, z)

        # Atmospheric parameters (z_atm only).
        out.atmos_gamma, out.atmos_nu, out.atmos_alpha, out.atmos_beta = (
            self.atmos_decoder(r, z_atm)
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
        context_wavelength: Tensor | None = None,
        context_fwhm: Tensor | None = None,
        context_radiance: Tensor | None = None,
        context_pad_mask: Tensor | None = None,
        context_pixel_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Multi-sample inference for calibrated uncertainty.

        Draws n_samples from q(z_atm, z_surf | context) and aggregates.
        """
        self.eval()
        r, z_atm_mu, z_atm_ls, z_surf_mu, z_surf_ls, _, _ = self.encode(
            wavelength, fwhm, radiance, pad_mask,
            context_wavelength, context_fwhm, context_radiance,
            context_pad_mask, context_pixel_mask,
        )

        spectral_samples = []       # s_mu per z-sample
        spectral_logvar_samples = [] # s_logvar per z-sample (aleatoric)
        reflectance_samples = []
        material_logit_samples = []
        atmos_gamma_samples = []
        atmos_epistemic_samples = []

        for _ in range(n_samples):
            z_atm = z_atm_mu + torch.randn_like(z_atm_mu) * z_atm_ls.exp()
            z_surf = z_surf_mu + torch.randn_like(z_surf_mu) * z_surf_ls.exp()
            z = torch.cat([z_atm, z_surf], dim=-1)

            if query_wavelength is not None:
                s_mu, s_logvar = self.spectral_decoder(r, z, query_wavelength, query_fwhm)
                spectral_samples.append(s_mu)
                spectral_logvar_samples.append(s_logvar)
                rho_mu, rho_logvar = self.reflectance_decoder(r, z, query_wavelength, query_fwhm)
                reflectance_samples.append(rho_mu)

            material_logit_samples.append(self.material_decoder(r, z))

            ag, an, aa, ab = self.atmos_decoder(r, z_atm)
            atmos_gamma_samples.append(ag)
            atmos_epistemic_samples.append(ab / (an * (aa - 1.0)))

        results: dict[str, Tensor] = {}

        if spectral_samples:
            stacked = torch.stack(spectral_samples)
            results["spectral_mean"] = stacked.mean(0)
            # Total uncertainty = epistemic (z-sample variance) + aleatoric
            # (mean of per-sample predicted variance from the decoder).
            epistemic_var = stacked.var(0)
            aleatoric_var = torch.stack(spectral_logvar_samples).exp().mean(0)
            results["spectral_std"] = (epistemic_var + aleatoric_var).sqrt()
        if reflectance_samples:
            rho_stack = torch.stack(reflectance_samples)
            results["reflectance_mean"] = rho_stack.mean(0)
            results["reflectance_std"] = rho_stack.std(0)

        mat_stack = torch.stack(material_logit_samples)
        mat_probs = mat_stack.softmax(dim=-1).mean(0)
        results["material_probs"] = mat_probs
        results["material_entropy"] = -(mat_probs * (mat_probs + 1e-10).log()).sum(-1)

        atmos_stack = torch.stack(atmos_gamma_samples)
        results["atmos_mean"] = atmos_stack.mean(0)
        results["atmos_std"] = atmos_stack.std(0)
        results["atmos_evidential_epistemic"] = torch.stack(atmos_epistemic_samples).mean(0)

        return results
