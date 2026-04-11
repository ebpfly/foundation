"""Spectral foundation model with Bayesian PCA observation encoding.

Architecture:
    1. Bayesian update: sparse sensor observations → PCA coefficient posterior
       (exact linear-Gaussian, no learned parameters)
    2. VAE encoder: PCA posterior (μ, σ²) → latent z
    3. VAE decoder: z → three heads (reflectance, atmosphere, temperature)

The PCA basis decorrelates the spectrum so that representing the posterior
as independent means + variances per component is justified.  The Bayesian
update is closed-form because the observation model (SRF convolution) is
linear in PCA space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FoundationConfig:
    """Configuration for the spectral foundation model."""

    # PCA dimensions.  Set to 0 (or any value larger than the wavelength
    # grid) to use full-rank PCA — recommended, since the point of PCA
    # here is decorrelation, not dimensionality reduction.
    n_pca_radiance: int = 0
    n_pca_reflectance: int = 0

    # Latent
    z_dim: int = 128

    # Encoder MLP (PCA posterior → z)
    encoder_hidden: tuple[int, ...] = (512, 256)

    # Decoder MLPs (z → task outputs)
    decoder_hidden: tuple[int, ...] = (256, 512)

    # Task heads
    n_atmos_params: int = 4  # AOD, water_vapour, ozone, visibility

    # Observation noise model for Bayesian update
    assumed_snr: float = 200.0
    read_noise: float = 0.1

    # Regularisation
    dropout: float = 0.1

    # Training
    beta: float = 1.0  # KL weight


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class FoundationOutput:
    """All outputs from a forward pass."""

    # Reflectance reconstruction
    reflectance: Tensor          # (B, n_wl)

    # Atmospheric parameters (heteroscedastic)
    atmos_mu: Tensor             # (B, n_atmos)
    atmos_log_var: Tensor        # (B, n_atmos)

    # Surface temperature (heteroscedastic)
    temp_mu: Tensor              # (B, 1)
    temp_log_var: Tensor         # (B, 1)

    # Latent distribution
    z: Tensor                    # (B, z_dim) — sampled
    z_mu: Tensor                 # (B, z_dim)
    z_log_var: Tensor            # (B, z_dim)

    # Bayesian update diagnostics
    pca_mu: Tensor               # (B, n_pca_rad)
    pca_var: Tensor              # (B, n_pca_rad)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP with ReLU activations and optional dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpectralFoundation(nn.Module):
    """Bayesian-PCA → VAE spectral foundation model.

    Call :meth:`fit_pca` once before training to set the PCA bases from
    a collection of radiance and reflectance spectra on the dense grid.
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.config = config

        # ---- PCA buffers (populated by fit_pca) ----
        self.register_buffer("wavelength_nm", torch.zeros(1))
        self.register_buffer("rad_pca_mean", torch.zeros(1))
        self.register_buffer("rad_pca_components", torch.zeros(1, 1))      # (n_pca, W)
        self.register_buffer("rad_pca_variances", torch.zeros(1))          # (n_pca,)
        self.register_buffer("refl_pca_mean", torch.zeros(1))
        self.register_buffer("refl_pca_components", torch.zeros(1, 1))     # (n_pca, W)

        # ---- Encoder: (μ_c, σ²_c) → (μ_z, log_σ²_z) ----
        # Placeholder sizes are used if n_pca is 0 (auto full-rank).
        # The actual layers are built/rebuilt in fit_pca once PCA is known.
        enc_in = 2 * max(config.n_pca_radiance, 1)
        self.encoder = MLP(enc_in, config.encoder_hidden, 2 * config.z_dim, config.dropout)

        # ---- Reflectance head: z → PCA coefficients ----
        self.reflectance_head = MLP(
            config.z_dim, config.decoder_hidden,
            max(config.n_pca_reflectance, 1), config.dropout,
        )

        # ---- Atmosphere head: z → (μ, log_var) per parameter ----
        self.atmosphere_head = MLP(
            config.z_dim, (256, 128), 2 * config.n_atmos_params, config.dropout,
        )

        # ---- Temperature head: z → (μ, log_var) ----
        self.temperature_head = MLP(
            config.z_dim, (256, 128), 2, config.dropout,
        )

    # ------------------------------------------------------------------
    # PCA fitting (offline, before training)
    # ------------------------------------------------------------------

    def fit_pca(
        self,
        radiance_spectra: np.ndarray,
        reflectance_spectra: np.ndarray,
        wavelength_nm: np.ndarray,
    ) -> dict[str, float]:
        """Fit PCA bases from training data.

        Parameters
        ----------
        radiance_spectra : (N, W) at-sensor radiance on the dense grid.
        reflectance_spectra : (N, W) surface reflectance on the dense grid.
        wavelength_nm : (W,) the dense wavelength grid in nm.

        Returns
        -------
        dict with explained-variance diagnostics.
        """
        device = self.rad_pca_mean.device

        self.wavelength_nm = torch.from_numpy(
            wavelength_nm.astype(np.float32)
        ).to(device)

        # --- Radiance PCA ---
        rad_mean = radiance_spectra.mean(axis=0)
        centered = radiance_spectra - rad_mean
        _U, S_rad, Vt_rad = np.linalg.svd(centered, full_matrices=False)
        max_rad = len(S_rad)
        n = self.config.n_pca_radiance
        if n <= 0 or n > max_rad:
            n = max_rad
        variances = S_rad[:n] ** 2 / max(len(radiance_spectra) - 1, 1)

        self.rad_pca_mean = torch.from_numpy(rad_mean.astype(np.float32)).to(device)
        self.rad_pca_components = torch.from_numpy(Vt_rad[:n].astype(np.float32)).to(device)
        self.rad_pca_variances = torch.from_numpy(variances.astype(np.float32)).to(device)

        # --- Reflectance PCA ---
        refl_mean = reflectance_spectra.mean(axis=0)
        centered = reflectance_spectra - refl_mean
        _U, S_refl, Vt_refl = np.linalg.svd(centered, full_matrices=False)
        max_refl = len(S_refl)
        m = self.config.n_pca_reflectance
        if m <= 0 or m > max_refl:
            m = max_refl

        self.refl_pca_mean = torch.from_numpy(refl_mean.astype(np.float32)).to(device)
        self.refl_pca_components = torch.from_numpy(Vt_refl[:m].astype(np.float32)).to(device)

        # --- Rebuild encoder + reflectance head to match actual PCA sizes ---
        self.config.n_pca_radiance = n
        self.config.n_pca_reflectance = m
        self.encoder = MLP(
            2 * n, self.config.encoder_hidden, 2 * self.config.z_dim,
            self.config.dropout,
        ).to(device)
        self.reflectance_head = MLP(
            self.config.z_dim, self.config.decoder_hidden, m,
            self.config.dropout,
        ).to(device)

        explained_rad = float(S_rad[:n].sum() ** 2 / (S_rad ** 2).sum())
        explained_refl = float(S_refl[:m].sum() ** 2 / (S_refl ** 2).sum())
        return {
            "radiance_explained_var": (S_rad[:n] ** 2).sum() / (S_rad ** 2).sum(),
            "reflectance_explained_var": (S_refl[:m] ** 2).sum() / (S_refl ** 2).sum(),
        }

    # ------------------------------------------------------------------
    # Bayesian update (no learned parameters)
    # ------------------------------------------------------------------

    def bayesian_update(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor,
        return_full_cov: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Exact linear-Gaussian update from sparse bands to PCA posterior.

        Observation model (per band *i*):
            y_i = SRF_i · (P^T c + μ) + noise_i

        This is linear in the PCA coefficients *c*, so the posterior is
        Gaussian with closed-form mean and covariance.

        Parameters
        ----------
        wavelength : (B, N)  band centre wavelengths [nm]
        fwhm       : (B, N)  band FWHMs [nm]
        radiance   : (B, N)  observed at-sensor radiance
        pad_mask   : (B, N)  True = valid band
        return_full_cov : if True, also return the full (B, K, K) covariance.
            Needed for correct per-wavelength variance propagation; the
            diagonal alone discards the off-diagonal correlations that
            tighten the variance at observed wavelengths.

        Returns
        -------
        mu_c  : (B, K)  posterior mean of PCA coefficients
        var_c : (B, K)  posterior marginal variance (diagonal of Σ_post)
        Sigma : (B, K, K)  full covariance (only if return_full_cov=True)
        """
        B, N = wavelength.shape
        K = self.config.n_pca_radiance
        device = wavelength.device

        wl_grid = self.wavelength_nm                          # (W,)
        W = wl_grid.shape[0]

        # --- Build pseudo-Voigt SRFs: (B, N, W) ---
        # Matches spectralnp.data.srf.pseudo_voigt with eta=0.5.
        # Must match the forward simulation's SRF model to avoid a
        # systematic posterior bias at high band counts.
        wl = wl_grid.view(1, 1, W)                           # broadcast grid
        ctr = wavelength.unsqueeze(-1)                        # (B, N, 1)
        width = fwhm.unsqueeze(-1).clamp(min=0.01)           # (B, N, 1)

        sigma = width / 2.355                                 # Gaussian σ
        gamma = width / 2.0                                   # Lorentzian γ
        dx = wl - ctr
        gauss = torch.exp(-0.5 * (dx / sigma) ** 2)
        lorentz = 1.0 / (1.0 + (dx / gamma) ** 2)
        eta = 0.5
        srf = (1.0 - eta) * gauss + eta * lorentz
        srf = srf / (srf.sum(dim=-1, keepdim=True) + 1e-30)  # normalise

        # Zero out padded bands
        srf = srf * pad_mask.unsqueeze(-1).float()

        # --- Observation matrix in PCA space: A = H P^T  (B, N, K) ---
        A = torch.matmul(srf, self.rad_pca_components.T)     # (B, N, K)

        # --- Residual: b = y − H μ  (B, N) ---
        offset = torch.matmul(srf, self.rad_pca_mean)        # (B, N)
        b = (radiance - offset) * pad_mask.float()

        # --- Per-band noise variance (signal-dependent) ---
        noise_var = (radiance.abs() / self.config.assumed_snr) ** 2 \
                    + self.config.read_noise ** 2
        noise_var = noise_var + (~pad_mask).float() * 1e10    # kill padding
        noise_prec = 1.0 / noise_var                          # (B, N)

        # --- Weighted observation matrices ---
        w = noise_prec.sqrt().unsqueeze(-1)                   # (B, N, 1)
        Aw = A * w                                            # (B, N, K)
        bw = b * noise_prec.sqrt()                            # (B, N)

        # --- Posterior precision: Λ = Σ_prior⁻¹ + Aᵀ diag(1/σ²) A ---
        prior_prec = 1.0 / (self.rad_pca_variances + 1e-10)  # (K,)
        AtWA = torch.matmul(Aw.transpose(-1, -2), Aw)        # (B, K, K)
        eye_K = torch.eye(K, device=device)
        Lambda = torch.diag(prior_prec).unsqueeze(0) + AtWA
        Lambda = Lambda + 1e-6 * eye_K.unsqueeze(0)          # jitter

        # --- Solve via Cholesky ---
        L = torch.linalg.cholesky(Lambda)                     # (B, K, K)
        Atb = torch.matmul(Aw.transpose(-1, -2), bw.unsqueeze(-1))  # (B, K, 1)
        mu_c = torch.cholesky_solve(Atb, L).squeeze(-1)      # (B, K)

        # Marginal variances = diag(Λ⁻¹)
        Sigma = torch.cholesky_solve(
            eye_K.unsqueeze(0).expand(B, -1, -1), L,
        )
        var_c = torch.diagonal(Sigma, dim1=-2, dim2=-1)      # (B, K)

        if return_full_cov:
            return mu_c, var_c, Sigma
        return mu_c, var_c

    # ------------------------------------------------------------------
    # Encoder / decoder
    # ------------------------------------------------------------------

    def encode(self, mu_c: Tensor, var_c: Tensor) -> tuple[Tensor, Tensor]:
        """Map PCA posterior to latent distribution q(z | observations).

        Inputs are normalised by prior std so the encoder sees
        roughly unit-scale features regardless of PCA component ordering.
        """
        prior_std = (self.rad_pca_variances + 1e-10).sqrt()
        mu_norm = mu_c / prior_std
        var_norm = var_c / (self.rad_pca_variances + 1e-10)

        x = torch.cat([mu_norm, var_norm], dim=-1)
        h = self.encoder(x)
        mu_z, log_var_z = h.chunk(2, dim=-1)
        return mu_z, log_var_z

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample z ~ N(mu, diag(exp(log_var))) with the reparameterization trick."""
        if not torch.is_grad_enabled():
            return mu
        std = (0.5 * log_var).exp()
        return mu + std * torch.randn_like(std)

    def decode_reflectance(self, z: Tensor) -> Tensor:
        """z → surface reflectance on the dense wavelength grid."""
        coeffs = self.reflectance_head(z)                     # (B, n_pca_refl)
        refl = torch.matmul(coeffs, self.refl_pca_components) + self.refl_pca_mean
        return refl.clamp(0.0, 1.0)

    def decode_atmosphere(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """z → atmospheric parameters with heteroscedastic uncertainty."""
        out = self.atmosphere_head(z)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var.clamp(-7.0, 4.0)

    def decode_temperature(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """z → surface temperature with heteroscedastic uncertainty."""
        out = self.temperature_head(z)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var.clamp(-7.0, 4.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
    ) -> FoundationOutput:
        B, N = wavelength.shape
        if pad_mask is None:
            pad_mask = torch.ones(B, N, dtype=torch.bool, device=wavelength.device)

        # 1. Bayesian update → PCA posterior
        mu_c, var_c = self.bayesian_update(wavelength, fwhm, radiance, pad_mask)

        # 2. Encode → latent distribution
        mu_z, log_var_z = self.encode(mu_c, var_c)
        z = self.reparameterize(mu_z, log_var_z)

        # 3. Decode → task outputs
        reflectance = self.decode_reflectance(z)
        atmos_mu, atmos_log_var = self.decode_atmosphere(z)
        temp_mu, temp_log_var = self.decode_temperature(z)

        return FoundationOutput(
            reflectance=reflectance,
            atmos_mu=atmos_mu,
            atmos_log_var=atmos_log_var,
            temp_mu=temp_mu,
            temp_log_var=temp_log_var,
            z=z,
            z_mu=mu_z,
            z_log_var=log_var_z,
            pca_mu=mu_c,
            pca_var=var_c,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss(
        output: FoundationOutput,
        target_reflectance: Tensor,
        target_atmos: Tensor,
        target_temperature: Tensor,
        beta: float = 1.0,
        w_refl: float = 1.0,
        w_atmos: float = 1.0,
        w_temp: float = 1.0,
    ) -> dict[str, Tensor]:
        """Compute combined training loss.

        Parameters
        ----------
        target_reflectance : (B, W) surface reflectance on dense grid
        target_atmos       : (B, n_atmos) normalised atmospheric parameters
        target_temperature : (B, 1) normalised surface temperature
        beta               : KL weight
        w_refl, w_atmos, w_temp : task loss weights
        """
        # --- Reflectance reconstruction (MSE) ---
        refl_loss = torch.nn.functional.mse_loss(
            output.reflectance, target_reflectance,
        )

        # --- Atmosphere (Gaussian NLL, heteroscedastic) ---
        atmos_nll = 0.5 * (
            output.atmos_log_var
            + (target_atmos - output.atmos_mu) ** 2 / output.atmos_log_var.exp()
        )
        atmos_loss = atmos_nll.mean()

        # --- Temperature (Gaussian NLL, heteroscedastic) ---
        temp_nll = 0.5 * (
            output.temp_log_var
            + (target_temperature - output.temp_mu) ** 2 / output.temp_log_var.exp()
        )
        temp_loss = temp_nll.mean()

        # --- KL divergence: q(z|x) || N(0, I) ---
        kl = -0.5 * (
            1.0 + output.z_log_var - output.z_mu.pow(2) - output.z_log_var.exp()
        )
        kl_loss = kl.sum(dim=-1).mean()

        total = w_refl * refl_loss + w_atmos * atmos_loss + w_temp * temp_loss + beta * kl_loss

        return {
            "total": total,
            "reflectance": refl_loss.detach(),
            "atmosphere": atmos_loss.detach(),
            "temperature": temp_loss.detach(),
            "kl": kl_loss.detach(),
        }
