"""PCA-Latent VAE: lightweight generative model for reflectance spectra.

Instead of learning to compress raw spectra with convolutions, this model
first projects spectra into PCA space and then trains a small MLP-VAE in
that low-dimensional representation.  PCA handles the linear structure;
the VAE captures nonlinear variation and enables smooth sampling.

    spectrum → PCA → z_pca → Encoder → (mu, log_var) → z → Decoder → z_pca_hat → PCA⁻¹ → spectrum

The PCA basis is fit once on the training data and stored alongside the
model checkpoint so generation only requires the checkpoint file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PCAVAEConfig:
    """Hyperparameters for the PCA-latent VAE."""

    n_pca: int = 64             # number of PCA components kept
    z_dim: int = 16             # VAE latent dimension
    hidden_dims: tuple[int, ...] = (256, 128)  # MLP hidden layers
    beta: float = 1.0           # KL weight
    dropout: float = 0.1


class PCAVAE(nn.Module):
    """VAE that operates in PCA space.

    Call ``fit_pca`` before training to compute the PCA basis from data.
    """

    def __init__(self, cfg: PCAVAEConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = PCAVAEConfig()
        self.cfg = cfg

        # PCA buffers (set by fit_pca).
        self.register_buffer("pca_mean", torch.zeros(1))
        self.register_buffer("pca_components", torch.zeros(1))
        self.register_buffer("pca_singular_values", torch.zeros(1))
        self._pca_fitted = False

        # Latent prior buffers (set by fit_latent_prior).
        self.register_buffer("z_mean", torch.zeros(1))
        self.register_buffer("z_cholesky", torch.zeros(1))

        # Encoder: PCA space → (mu, log_var).
        enc_layers: list[nn.Module] = []
        in_dim = cfg.n_pca
        for h in cfg.hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, cfg.z_dim)
        self.fc_log_var = nn.Linear(in_dim, cfg.z_dim)
        nn.init.zeros_(self.fc_log_var.weight)
        nn.init.constant_(self.fc_log_var.bias, -2.0)

        # Decoder: z → PCA space.
        dec_layers: list[nn.Module] = []
        in_dim = cfg.z_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.n_pca))
        self.decoder = nn.Sequential(*dec_layers)

    def fit_pca(self, spectra: np.ndarray) -> dict[str, np.ndarray]:
        """Compute PCA basis from (N, W) spectra array.

        Returns dict with PCA diagnostics (explained variance etc.).
        """
        mean = spectra.mean(axis=0)
        centered = spectra - mean

        # SVD for PCA.
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        n_pca = self.cfg.n_pca
        components = Vt[:n_pca]          # (n_pca, W)
        singular_values = S[:n_pca]      # (n_pca,)

        # Store as buffers.
        self.pca_mean = torch.from_numpy(mean.astype(np.float32))
        self.pca_components = torch.from_numpy(components.astype(np.float32))
        self.pca_singular_values = torch.from_numpy(singular_values.astype(np.float32))
        self._pca_fitted = True

        # Diagnostics.
        total_var = (S ** 2).sum()
        explained = (S[:n_pca] ** 2).cumsum() / total_var
        return {
            "explained_variance_ratio": explained,
            "singular_values": S,
            "n_components": n_pca,
            "total_variance": total_var,
        }

    def to_pca(self, spectra: Tensor) -> Tensor:
        """Project (B, W) spectra to (B, n_pca) PCA coefficients."""
        centered = spectra - self.pca_mean
        return centered @ self.pca_components.T

    def from_pca(self, coeffs: Tensor) -> Tensor:
        """Reconstruct (B, W) spectra from (B, n_pca) PCA coefficients."""
        return coeffs @ self.pca_components + self.pca_mean

    def encode(self, pca_coeffs: Tensor) -> tuple[Tensor, Tensor]:
        """Encode PCA coefficients to (mu, log_var)."""
        h = self.encoder_body(pca_coeffs)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h).clamp(-10, 10)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent z to PCA coefficients."""
        return self.decoder(z)

    def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, spectra: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass.

        Parameters
        ----------
        spectra : (B, W) reflectance spectra

        Returns
        -------
        recon : (B, W) reconstructed spectra
        mu, log_var : (B, z_dim) latent distribution parameters
        pca_recon : (B, n_pca) reconstructed PCA coefficients (for PCA-space loss)
        """
        pca_coeffs = self.to_pca(spectra)
        mu, log_var = self.encode(pca_coeffs)
        z = self.reparameterise(mu, log_var)
        pca_recon = self.decode(z)
        recon = self.from_pca(pca_recon)
        return recon, mu, log_var, pca_recon

    @torch.no_grad()
    def fit_latent_prior(self, spectra: np.ndarray) -> None:
        """Fit a Gaussian to the encoded training data for realistic sampling.

        Call this after training, passing the training spectra.  The fitted
        mean and Cholesky factor are stored as buffers so they persist in
        checkpoints.
        """
        self.eval()
        t = torch.from_numpy(spectra.astype(np.float32))
        pca = self.to_pca(t)
        mu, _ = self.encode(pca)
        mu_np = mu.cpu().numpy()

        z_mean = mu_np.mean(axis=0)
        z_cov = np.cov(mu_np, rowvar=False)
        # Cholesky for sampling: z = mean + L @ eps, eps ~ N(0,I)
        L = np.linalg.cholesky(z_cov + 1e-6 * np.eye(len(z_cov)))

        self.register_buffer("z_mean", torch.from_numpy(z_mean.astype(np.float32)))
        self.register_buffer("z_cholesky", torch.from_numpy(L.astype(np.float32)))
        self._latent_prior_fitted = True

    @torch.no_grad()
    def generate(self, n_samples: int = 1, device: str | torch.device = "cpu") -> Tensor:
        """Generate spectra by sampling from the fitted latent distribution.

        If ``fit_latent_prior`` was called, samples from the aggregate
        posterior N(z_mean, z_cov).  Otherwise falls back to N(0, I).
        """
        self.eval()
        if hasattr(self, "z_mean") and self.z_mean.dim() > 0 and self.z_mean.shape[0] > 1:
            eps = torch.randn(n_samples, self.cfg.z_dim, device=device)
            z = self.z_mean.to(device) + eps @ self.z_cholesky.to(device).T
        else:
            z = torch.randn(n_samples, self.cfg.z_dim, device=device)
        pca_coeffs = self.decode(z)
        return self.from_pca(pca_coeffs)

    @torch.no_grad()
    def reconstruct(self, spectra: Tensor) -> Tensor:
        """Encode and decode (deterministic at eval)."""
        self.eval()
        recon, _, _, _ = self.forward(spectra)
        return recon

    @torch.no_grad()
    def interpolate(self, x1: Tensor, x2: Tensor, steps: int = 10) -> Tensor:
        """Linear interpolation in latent space between two spectra."""
        self.eval()
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)

        pca1 = self.to_pca(x1)
        pca2 = self.to_pca(x2)
        mu1, _ = self.encode(pca1)
        mu2, _ = self.encode(pca2)

        alphas = torch.linspace(0, 1, steps, device=x1.device)
        zs = torch.stack([mu1 * (1 - a) + mu2 * a for a in alphas]).squeeze(1)
        pca_coeffs = self.decode(zs)
        return self.from_pca(pca_coeffs)


def pca_vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    log_var: Tensor,
    pca_recon: Tensor,
    pca_target: Tensor,
    beta: float = 1.0,
    free_bits: float = 0.5,
) -> dict[str, Tensor]:
    """Combined loss: spectral reconstruction + PCA reconstruction + KL.

    The PCA-space loss ensures the VAE learns the PCA structure well;
    the spectral-space loss ensures end-to-end fidelity.
    """
    # Spectral-space reconstruction.
    recon_loss = nn.functional.mse_loss(recon, target)

    # PCA-space reconstruction (weighted higher since it's the direct target).
    pca_loss = nn.functional.mse_loss(pca_recon, pca_target)

    # KL with free bits.
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl_loss = kl_per_dim.sum(dim=-1).mean()

    total = recon_loss + pca_loss + beta * kl_loss
    return {
        "total": total,
        "recon": recon_loss,
        "pca_recon": pca_loss,
        "kl": kl_loss,
    }
