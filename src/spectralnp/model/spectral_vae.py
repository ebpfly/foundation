"""Spectral VAE: deep generative model for reflectance spectra.

A 1D convolutional variational autoencoder that learns to generate
realistic reflectance spectra from the USGS spectral library.

The encoder compresses a spectrum into a low-dimensional latent space.
The decoder reconstructs spectra from latent codes.  Sampling from the
prior p(z) = N(0, I) produces novel, physically plausible spectra.

Architecture:
    Encoder: 1D conv stack → flatten → (mu, log_var)
    Decoder: linear → unflatten → 1D transposed conv stack → spectrum
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SpectralVAEConfig:
    """Hyperparameters for the spectral VAE."""

    n_wavelengths: int = 2151       # ASD: 350-2500 nm at 1 nm
    z_dim: int = 32                 # latent dimension
    base_channels: int = 64         # first conv layer channels
    n_layers: int = 4               # number of conv blocks in encoder/decoder
    kernel_size: int = 7            # conv kernel size (odd)
    beta: float = 1.0               # KL weight (beta-VAE)
    dropout: float = 0.1


class ResBlock1d(nn.Module):
    """Residual block with 1D convolutions."""

    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class SpectralEncoder(nn.Module):
    """Encode a spectrum to latent (mu, log_var).

    Input:  (B, 1, W)  — single-channel 1D signal
    Output: (B, z_dim), (B, z_dim)
    """

    def __init__(self, cfg: SpectralVAEConfig) -> None:
        super().__init__()
        ch = cfg.base_channels
        ks = cfg.kernel_size
        pad = ks // 2

        layers: list[nn.Module] = [
            nn.Conv1d(1, ch, ks, stride=2, padding=pad),
            nn.BatchNorm1d(ch),
            nn.GELU(),
        ]

        for i in range(cfg.n_layers - 1):
            out_ch = ch * 2
            layers.extend([
                nn.Conv1d(ch, out_ch, ks, stride=2, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                ResBlock1d(out_ch, ks, cfg.dropout),
            ])
            ch = out_ch

        self.conv = nn.Sequential(*layers)

        # Compute flattened size by doing a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.n_wavelengths)
            flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.fc_mu = nn.Linear(flat_size, cfg.z_dim)
        self.fc_log_var = nn.Linear(flat_size, cfg.z_dim)
        # Init log_var bias near 0 so KL starts small.
        nn.init.zeros_(self.fc_log_var.weight)
        nn.init.constant_(self.fc_log_var.bias, -2.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.conv(x).flatten(1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10, 10)
        return mu, log_var


class SpectralDecoder(nn.Module):
    """Decode latent z to a reconstructed spectrum.

    Input:  (B, z_dim)
    Output: (B, 1, W)
    """

    def __init__(self, cfg: SpectralVAEConfig) -> None:
        super().__init__()
        ks = cfg.kernel_size
        pad = ks // 2

        # Figure out the spatial shape after the encoder's conv stack.
        ch = cfg.base_channels
        for _ in range(cfg.n_layers - 1):
            ch *= 2
        self.top_channels = ch

        # Compute the spatial size after encoder convolutions.
        size = cfg.n_wavelengths
        for _ in range(cfg.n_layers):
            size = (size + 2 * pad - ks) // 2 + 1
        self.top_size = size

        self.fc = nn.Linear(cfg.z_dim, ch * size)

        layers: list[nn.Module] = []
        for i in range(cfg.n_layers - 1):
            out_ch = ch // 2
            layers.extend([
                ResBlock1d(ch, ks, cfg.dropout),
                nn.ConvTranspose1d(ch, out_ch, ks, stride=2, padding=pad, output_padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            ch = out_ch

        layers.append(
            nn.ConvTranspose1d(ch, 1, ks, stride=2, padding=pad, output_padding=1),
        )
        self.deconv = nn.Sequential(*layers)
        self.target_len = cfg.n_wavelengths

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, self.top_channels, self.top_size)
        out = self.deconv(h)
        # Trim or pad to exact target length (transposed convs may differ by 1-2).
        if out.shape[-1] > self.target_len:
            out = out[..., :self.target_len]
        elif out.shape[-1] < self.target_len:
            out = nn.functional.pad(out, (0, self.target_len - out.shape[-1]))
        return out


class SpectralVAE(nn.Module):
    """Variational autoencoder for reflectance spectra.

    Generates new spectra by sampling z ~ N(0, I) and decoding.
    """

    def __init__(self, cfg: SpectralVAEConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SpectralVAEConfig()
        self.cfg = cfg
        self.encoder = SpectralEncoder(cfg)
        self.decoder = SpectralDecoder(cfg)

    def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, W) or (B, 1, W) reflectance spectrum

        Returns
        -------
        recon : (B, W) reconstructed spectrum
        mu : (B, z_dim)
        log_var : (B, z_dim)
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decoder(z)

        if squeeze:
            recon = recon.squeeze(1)
        return recon, mu, log_var

    @torch.no_grad()
    def generate(self, n_samples: int = 1, device: str | torch.device = "cpu") -> Tensor:
        """Generate spectra by sampling from the prior."""
        self.eval()
        z = torch.randn(n_samples, self.cfg.z_dim, device=device)
        spectra = self.decoder(z).squeeze(1)
        return spectra

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """Encode and decode (deterministic at eval)."""
        self.eval()
        recon, _, _ = self.forward(x)
        return recon

    @torch.no_grad()
    def interpolate(self, x1: Tensor, x2: Tensor, steps: int = 10) -> Tensor:
        """Linear interpolation in latent space between two spectra."""
        self.eval()
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0).unsqueeze(0)
        elif x1.dim() == 2:
            x1 = x1.unsqueeze(1)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0).unsqueeze(0)
        elif x2.dim() == 2:
            x2 = x2.unsqueeze(1)

        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)

        alphas = torch.linspace(0, 1, steps, device=x1.device)
        zs = torch.stack([mu1 * (1 - a) + mu2 * a for a in alphas]).squeeze(1)
        return self.decoder(zs).squeeze(1)


def vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    log_var: Tensor,
    beta: float = 1.0,
    free_bits: float = 0.5,
) -> dict[str, Tensor]:
    """VAE ELBO loss = reconstruction + beta * KL divergence.

    Uses MSE for reconstruction since reflectance is continuous.
    free_bits: minimum KL per latent dimension (prevents posterior collapse).
    """
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
    # KL divergence per dimension: -0.5 * (1 + log_var - mu^2 - exp(log_var))
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # (B, z_dim)
    # Free bits: clamp each dimension's KL to at least `free_bits` nats.
    # This prevents the encoder from collapsing any dimension to the prior.
    kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl_loss = kl_per_dim.sum(dim=-1).mean()
    total = recon_loss + beta * kl_loss
    return {"total": total, "recon": recon_loss, "kl": kl_loss}
