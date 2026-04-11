"""Spectral foundation model with Bayesian PCA + disentangled latents.

Two-stage architecture:

  Stage 1 — Bayesian update (no learned parameters):
      sparse (λ, FWHM, L) → PCA coefficient posterior (μ_c, σ²_c)

  Stage 2 — disentangled VAE:
      PCA posterior → per-pixel features
                    → z_atm  (scene-shared, fused across pixels by Gaussian product)
                    → z_surf (per-pixel, conditioned on z_atm)
                    → task heads

Task heads:
    • reflectance  — per-pixel, linear from z_surf via reflectance-PCA basis
    • material     — per-pixel classification, from z_surf
    • temperature  — per-pixel, from z_surf  (surface temperature)
    • atmosphere   — scene-shared, from z_atm  (AOD, H2O, O3, visibility)
    • physics      — per-pixel radiance reconstruction from (z_atm, z_surf)
                     as a consistency loss that forces disentanglement

The architecture is always multi-pixel: inputs have shape (B, K, N) with
K pixels per scene.  K=1 is the normal single-pixel case and the Gaussian
product over one pixel is trivially that pixel's belief.  When the
dataloader starts producing multi-pixel scenes (K>1), the model
automatically tightens z_atm (precisions add), which flows through to
tighter z_surf via the SurfMLP([h_i, z_atm]) dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FoundationConfig:
    """Configuration for the spectral foundation model."""

    # ---- PCA dimensions ----
    # Set to 0 to use full rank (recommended — the point of PCA here is
    # decorrelation, not dimensionality reduction).
    n_pca_radiance: int = 0
    n_pca_reflectance: int = 0

    # ---- Latents ----
    z_atm_dim: int = 32       # scene-shared atmospheric latent (small)
    z_surf_dim: int = 96      # per-pixel surface latent
    feature_dim: int = 256    # per-pixel shared feature dimension

    # ---- Network sizes ----
    trunk_hidden: tuple[int, ...] = (512, 256)
    head_hidden: tuple[int, ...] = (256, 128)
    decoder_hidden: tuple[int, ...] = (256, 512)

    # ---- Task outputs ----
    n_atmos_params: int = 4          # AOD, H2O, O3, visibility
    n_material_classes: int = 10     # set at training time to match dataset

    # ---- Observation noise model (Bayesian update) ----
    assumed_snr: float = 200.0
    read_noise: float = 0.1

    # ---- Regularisation ----
    dropout: float = 0.1

    # ---- Training loss weights ----
    w_refl: float = 1.0
    w_atmos: float = 1.0
    w_temp: float = 1.0
    w_material: float = 0.0   # disabled by default — enable once classes are set
    w_physics: float = 0.5
    beta_atm: float = 1.0     # KL weight for z_atm
    beta_surf: float = 1.0    # KL weight for z_surf


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class FoundationOutput:
    """All outputs from a forward pass.

    All per-pixel tensors have a K dimension (pixels per scene).
    """

    # --- Per-pixel reconstructions ---
    reflectance: Tensor           # (B, K, n_wl)  surface reflectance
    radiance_recon: Tensor        # (B, K, n_wl)  physics-consistency radiance
    rad_coeffs_normed: Tensor     # (B, K, n_pca_rad)  predicted coeffs in prior-std units

    # --- Per-pixel task outputs ---
    temp_mu: Tensor               # (B, K, 1)     surface temperature
    temp_log_var: Tensor          # (B, K, 1)
    material_logits: Tensor       # (B, K, n_classes)

    # --- Scene-shared task outputs (from z_atm) ---
    atmos_mu: Tensor              # (B, n_atmos)
    atmos_log_var: Tensor         # (B, n_atmos)

    # --- Latents ---
    z_atm: Tensor                 # (B, z_atm_dim)
    z_atm_mu: Tensor              # (B, z_atm_dim)
    z_atm_log_var: Tensor         # (B, z_atm_dim)

    z_surf: Tensor                # (B, K, z_surf_dim)
    z_surf_mu: Tensor             # (B, K, z_surf_dim)
    z_surf_log_var: Tensor        # (B, K, z_surf_dim)

    # --- Stage-1 diagnostics (per pixel) ---
    pca_mu: Tensor                # (B, K, n_pca_rad)
    pca_var: Tensor               # (B, K, n_pca_rad)

    # --- Masks carried through for losses ---
    pixel_mask: Tensor            # (B, K) bool


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP with LayerNorm + GELU + dropout."""

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


def _gaussian_kl_standard(mu: Tensor, log_var: Tensor) -> Tensor:
    """KL( N(μ, σ²) || N(0, I) ), summed over the last dim."""
    return -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpectralFoundation(nn.Module):
    """Disentangled spectral foundation model.

    Call :meth:`fit_pca` once before training to set PCA bases and size
    all downstream layers.
    """

    def __init__(self, config: FoundationConfig) -> None:
        super().__init__()
        self.config = config

        # ---- PCA buffers (populated by fit_pca) ----
        self.register_buffer("wavelength_nm", torch.zeros(1))
        self.register_buffer("rad_pca_mean", torch.zeros(1))
        self.register_buffer("rad_pca_components", torch.zeros(1, 1))      # (K, W)
        self.register_buffer("rad_pca_variances", torch.zeros(1))          # (K,)
        self.register_buffer("refl_pca_mean", torch.zeros(1))
        self.register_buffer("refl_pca_components", torch.zeros(1, 1))     # (M, W)

        # ---- Stage 2 networks (placeholder sizes; rebuilt in fit_pca) ----
        self._build_networks(
            n_pca_rad=max(config.n_pca_radiance, 1),
            n_pca_refl=max(config.n_pca_reflectance, 1),
        )

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_networks(self, n_pca_rad: int, n_pca_refl: int) -> None:
        c = self.config

        # Per-pixel shared trunk: PCA posterior → feature_dim
        self.shared_trunk = MLP(
            in_dim=2 * n_pca_rad,
            hidden_dims=c.trunk_hidden,
            out_dim=c.feature_dim,
            dropout=c.dropout,
        )

        # Atmospheric pseudo-observation head: per-pixel (μ_atm, log_var_atm)
        self.atm_pseudo_head = MLP(
            in_dim=c.feature_dim,
            hidden_dims=c.head_hidden,
            out_dim=2 * c.z_atm_dim,
            dropout=c.dropout,
        )

        # Surface trunk: [h_i, z_atm] → (μ_surf, log_var_surf)
        self.surf_trunk = MLP(
            in_dim=c.feature_dim + c.z_atm_dim,
            hidden_dims=c.head_hidden,
            out_dim=2 * c.z_surf_dim,
            dropout=c.dropout,
        )

        # ---- Task heads ----
        # Reflectance: z_surf → reflectance PCA coefficients → spectrum
        self.reflectance_head = MLP(
            in_dim=c.z_surf_dim,
            hidden_dims=c.decoder_hidden,
            out_dim=n_pca_refl,
            dropout=c.dropout,
        )

        # Surface temperature: per-pixel, from z_surf
        self.temperature_head = MLP(
            in_dim=c.z_surf_dim,
            hidden_dims=c.head_hidden,
            out_dim=2,
            dropout=c.dropout,
        )

        # Material classification: per-pixel, from z_surf
        self.material_head = MLP(
            in_dim=c.z_surf_dim,
            hidden_dims=c.head_hidden,
            out_dim=c.n_material_classes,
            dropout=c.dropout,
        )

        # Atmospheric parameters: scene-shared, from z_atm
        self.atmosphere_head = MLP(
            in_dim=c.z_atm_dim,
            hidden_dims=c.head_hidden,
            out_dim=2 * c.n_atmos_params,
            dropout=c.dropout,
        )

        # Physics consistency: (z_atm, z_surf) → radiance PCA coefficients
        # This is a reconstruction head that forces (z_atm, z_surf) to
        # jointly explain the observation, enforcing disentanglement.
        self.physics_head = MLP(
            in_dim=c.z_atm_dim + c.z_surf_dim,
            hidden_dims=c.decoder_hidden,
            out_dim=n_pca_rad,
            dropout=c.dropout,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def resize_from_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Resize PCA buffers and rebuild stage-2 networks to match a checkpoint.

        The stage-2 layer dimensions depend on the fitted PCA rank, which
        is only known after ``fit_pca``.  When loading a saved model we
        don't want to re-fit PCA — instead, inspect the state_dict and
        resize the placeholder buffers + rebuild the MLPs with the right
        input/output sizes.  After this call, ``load_state_dict`` will
        succeed.
        """
        device = self.rad_pca_mean.device

        # Required PCA buffers
        required_keys = [
            "wavelength_nm",
            "rad_pca_mean",
            "rad_pca_components",
            "rad_pca_variances",
            "refl_pca_mean",
            "refl_pca_components",
        ]
        for key in required_keys:
            if key not in state_dict:
                raise KeyError(
                    f"state_dict missing required PCA buffer '{key}'"
                )

        # Resize buffers by setting empty tensors with the target shape.
        self.wavelength_nm = torch.empty_like(state_dict["wavelength_nm"]).to(device)
        self.rad_pca_mean = torch.empty_like(state_dict["rad_pca_mean"]).to(device)
        self.rad_pca_components = torch.empty_like(
            state_dict["rad_pca_components"]
        ).to(device)
        self.rad_pca_variances = torch.empty_like(
            state_dict["rad_pca_variances"]
        ).to(device)
        self.refl_pca_mean = torch.empty_like(state_dict["refl_pca_mean"]).to(device)
        self.refl_pca_components = torch.empty_like(
            state_dict["refl_pca_components"]
        ).to(device)

        # Derive the actual PCA dimensions from the stored components.
        n_pca_rad = state_dict["rad_pca_components"].shape[0]
        n_pca_refl = state_dict["refl_pca_components"].shape[0]
        self.config.n_pca_radiance = n_pca_rad
        self.config.n_pca_reflectance = n_pca_refl

        # Rebuild stage-2 networks with the correct sizes.
        self._build_networks(n_pca_rad=n_pca_rad, n_pca_refl=n_pca_refl)
        self.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        path,
        map_location: str | torch.device = "cpu",
    ) -> "SpectralFoundation":
        """Load a model from a checkpoint saved by the training script.

        Handles the auto-sized stage-2 layers: reads the PCA dimensions
        from the stored state_dict, rebuilds the network, then loads.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        config: FoundationConfig = ckpt["config"]
        model = cls(config)
        model.resize_from_state_dict(ckpt["model_state_dict"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    # PCA fitting
    # ------------------------------------------------------------------

    def fit_pca(
        self,
        radiance_spectra: np.ndarray,
        reflectance_spectra: np.ndarray,
        wavelength_nm: np.ndarray,
    ) -> dict[str, float]:
        """Fit PCA bases and rebuild stage-2 networks to match."""
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
        # Floor small eigenvalues at 1e-6 of the largest.  This keeps
        # the Bayesian update well-conditioned and prevents the physics
        # loss (which divides by prior_std) from exploding when
        # sample-covariance tails are noisy.
        var_floor = float(variances.max()) * 1e-6
        variances = np.maximum(variances, var_floor)

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

        # --- Rebuild stage 2 networks with the actual PCA sizes ---
        self.config.n_pca_radiance = n
        self.config.n_pca_reflectance = m
        self._build_networks(n_pca_rad=n, n_pca_refl=m)
        self.to(device)

        return {
            "radiance_explained_var": float((S_rad[:n] ** 2).sum() / (S_rad ** 2).sum()),
            "reflectance_explained_var": float((S_refl[:m] ** 2).sum() / (S_refl ** 2).sum()),
        }

    # ------------------------------------------------------------------
    # Stage 1 — Bayesian update (flat, per-pixel)
    # ------------------------------------------------------------------

    def bayesian_update(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor,
        return_full_cov: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Closed-form linear-Gaussian update from sparse bands to PCA posterior.

        Operates on a flat (B, N) batch — multi-pixel inputs are flattened
        to (B·K, N) by the caller before invoking this method.

        Parameters
        ----------
        wavelength : (B, N)  band centre wavelengths [nm]
        fwhm       : (B, N)  band FWHMs [nm]
        radiance   : (B, N)  observed at-sensor radiance
        pad_mask   : (B, N)  True = valid band
        return_full_cov : also return the full (B, K, K) posterior covariance

        Returns
        -------
        mu_c  : (B, K)  posterior mean of PCA coefficients
        var_c : (B, K)  posterior marginal variance
        Sigma : (B, K, K) full covariance (only if return_full_cov)
        """
        B, N = wavelength.shape
        K = self.config.n_pca_radiance
        device = wavelength.device

        wl_grid = self.wavelength_nm                          # (W,)
        W = wl_grid.shape[0]

        # --- Build pseudo-Voigt SRFs (matches data.srf.pseudo_voigt, eta=0.5) ---
        wl = wl_grid.view(1, 1, W)
        ctr = wavelength.unsqueeze(-1)
        width = fwhm.unsqueeze(-1).clamp(min=0.01)

        sigma = width / 2.355
        gamma = width / 2.0
        dx = wl - ctr
        gauss = torch.exp(-0.5 * (dx / sigma) ** 2)
        lorentz = 1.0 / (1.0 + (dx / gamma) ** 2)
        eta = 0.5
        srf = (1.0 - eta) * gauss + eta * lorentz
        srf = srf / (srf.sum(dim=-1, keepdim=True) + 1e-30)
        srf = srf * pad_mask.unsqueeze(-1).float()

        # --- Linear operator in PCA space: A = H P^T ---
        A = torch.matmul(srf, self.rad_pca_components.T)

        # --- Residual: b = y − H μ_rad ---
        offset = torch.matmul(srf, self.rad_pca_mean)
        b = (radiance - offset) * pad_mask.float()

        # --- Per-band noise variance (signal-dependent) ---
        noise_var = (radiance.abs() / self.config.assumed_snr) ** 2 \
                    + self.config.read_noise ** 2
        noise_var = noise_var + (~pad_mask).float() * 1e10
        noise_prec = 1.0 / noise_var

        sqrt_prec = noise_prec.sqrt().unsqueeze(-1)
        Aw = A * sqrt_prec
        bw = b * noise_prec.sqrt()

        prior_prec = 1.0 / (self.rad_pca_variances + 1e-10)
        AtWA = torch.matmul(Aw.transpose(-1, -2), Aw)
        eye_K = torch.eye(K, device=device)
        Lambda = torch.diag(prior_prec).unsqueeze(0) + AtWA
        Lambda = Lambda + 1e-6 * eye_K.unsqueeze(0)

        # Use linalg.solve instead of cholesky_solve for MPS compatibility.
        # Lambda is SPD by construction so either works mathematically.
        Atb = torch.matmul(Aw.transpose(-1, -2), bw.unsqueeze(-1))
        mu_c = torch.linalg.solve(Lambda, Atb).squeeze(-1)

        Sigma = torch.linalg.solve(
            Lambda, eye_K.unsqueeze(0).expand(B, -1, -1),
        )
        var_c = torch.diagonal(Sigma, dim1=-2, dim2=-1)

        if return_full_cov:
            return mu_c, var_c, Sigma
        return mu_c, var_c

    # ------------------------------------------------------------------
    # Stage 2 — disentangled encoding
    # ------------------------------------------------------------------

    def _encode_pixels(
        self,
        pca_mu: Tensor,     # (B, K, n_pca_rad)
        pca_var: Tensor,    # (B, K, n_pca_rad)
    ) -> Tensor:
        """Map per-pixel PCA posterior to per-pixel feature vector h_i."""
        prior_std = (self.rad_pca_variances + 1e-10).sqrt()
        mu_norm = pca_mu / prior_std
        var_norm = pca_var / (self.rad_pca_variances + 1e-10)
        x = torch.cat([mu_norm, var_norm], dim=-1)            # (B, K, 2*n_pca)
        return self.shared_trunk(x)                           # (B, K, feature_dim)

    def _fuse_atmosphere(
        self,
        h: Tensor,                  # (B, K, feature_dim)
        pixel_mask: Tensor,         # (B, K) bool
    ) -> tuple[Tensor, Tensor]:
        """Gaussian-product fusion of per-pixel atmospheric pseudo-observations.

        Each pixel i produces a Gaussian belief (μ_i, σ²_i) about z_atm.
        The combined posterior is N(μ, σ²) with

            precision_i  = 1 / σ²_i     (masked by pixel_mask)
            σ²           = 1 / Σ_i precision_i
            μ            = σ² · Σ_i μ_i precision_i

        With K=1 this degrades to the single pixel's belief.  With K>>1
        the precision sums and the posterior tightens by √K (in the
        homoscedastic limit).
        """
        out = self.atm_pseudo_head(h)                         # (B, K, 2*z_atm)
        mu_i, log_var_i = out.chunk(2, dim=-1)
        # Clamp log_var for stability
        log_var_i = log_var_i.clamp(-10.0, 10.0)

        prec_i = (-log_var_i).exp()                           # (B, K, z_atm)
        mask_f = pixel_mask.unsqueeze(-1).float()             # (B, K, 1)
        prec_i = prec_i * mask_f

        prec_sum = prec_i.sum(dim=1) + 1e-10                  # (B, z_atm)
        mu_atm = (mu_i * prec_i).sum(dim=1) / prec_sum        # (B, z_atm)
        log_var_atm = -torch.log(prec_sum)                    # (B, z_atm)
        return mu_atm, log_var_atm

    def _encode_surface(
        self,
        h: Tensor,          # (B, K, feature_dim)
        z_atm: Tensor,      # (B, z_atm_dim)
    ) -> tuple[Tensor, Tensor]:
        """Per-pixel surface latent, conditioned on the shared atmosphere."""
        B, K, _ = h.shape
        z_atm_bcast = z_atm.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([h, z_atm_bcast], dim=-1)               # (B, K, feat+z_atm)
        out = self.surf_trunk(x)                              # (B, K, 2*z_surf)
        mu_surf, log_var_surf = out.chunk(2, dim=-1)
        return mu_surf, log_var_surf.clamp(-10.0, 10.0)

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = (0.5 * log_var).exp()
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        wavelength: Tensor,
        fwhm: Tensor,
        radiance: Tensor,
        pad_mask: Tensor | None = None,
        pixel_mask: Tensor | None = None,
    ) -> FoundationOutput:
        """Run the full pipeline.

        Accepts either (B, N) single-pixel or (B, K, N) multi-pixel input.
        Single-pixel input is automatically wrapped to K=1.
        """
        # --- Normalize to multi-pixel shape ---
        if wavelength.dim() == 2:
            wavelength = wavelength.unsqueeze(1)
            fwhm = fwhm.unsqueeze(1)
            radiance = radiance.unsqueeze(1)
            if pad_mask is not None:
                pad_mask = pad_mask.unsqueeze(1)

        B, K, N = wavelength.shape
        device = wavelength.device
        if pad_mask is None:
            pad_mask = torch.ones(B, K, N, dtype=torch.bool, device=device)
        if pixel_mask is None:
            pixel_mask = torch.ones(B, K, dtype=torch.bool, device=device)

        # --- Stage 1: Bayesian update (flatten over pixels) ---
        flat_wl = wavelength.reshape(B * K, N)
        flat_fw = fwhm.reshape(B * K, N)
        flat_rad = radiance.reshape(B * K, N)
        flat_mask = pad_mask.reshape(B * K, N)

        mu_c_flat, var_c_flat = self.bayesian_update(
            flat_wl, flat_fw, flat_rad, flat_mask,
        )
        n_pca_rad = self.config.n_pca_radiance
        pca_mu = mu_c_flat.view(B, K, n_pca_rad)
        pca_var = var_c_flat.view(B, K, n_pca_rad)

        # --- Stage 2: per-pixel features ---
        h = self._encode_pixels(pca_mu, pca_var)              # (B, K, feat)

        # --- Atmospheric fusion across pixels ---
        mu_atm, log_var_atm = self._fuse_atmosphere(h, pixel_mask)
        z_atm = self.reparameterize(mu_atm, log_var_atm)      # (B, z_atm)

        # --- Per-pixel surface latent, conditioned on z_atm ---
        mu_surf, log_var_surf = self._encode_surface(h, z_atm)
        z_surf = self.reparameterize(mu_surf, log_var_surf)   # (B, K, z_surf)

        # --- Task heads ---
        # Reflectance: per-pixel linear PCA reconstruction
        refl_coeffs = self.reflectance_head(z_surf)           # (B, K, n_pca_refl)
        reflectance = torch.matmul(
            refl_coeffs, self.refl_pca_components,
        ) + self.refl_pca_mean                                # (B, K, W)
        reflectance = reflectance.clamp(0.0, 1.0)

        # Per-pixel temperature
        temp_out = self.temperature_head(z_surf)              # (B, K, 2)
        temp_mu, temp_log_var = temp_out.chunk(2, dim=-1)
        temp_log_var = temp_log_var.clamp(-7.0, 4.0)

        # Per-pixel material logits
        material_logits = self.material_head(z_surf)          # (B, K, n_classes)

        # Scene-shared atmospheric parameters
        atm_out = self.atmosphere_head(z_atm)                 # (B, 2*n_atmos)
        atmos_mu, atmos_log_var = atm_out.chunk(2, dim=-1)
        atmos_log_var = atmos_log_var.clamp(-7.0, 4.0)

        # Physics consistency: joint (z_atm, z_surf) → radiance PCA coeffs
        # The head outputs NORMALIZED coefficients (in units of prior std)
        # for a well-scaled loss.  Real coefficients: pred_norm · prior_std.
        z_atm_bcast = z_atm.unsqueeze(1).expand(-1, K, -1)
        phys_in = torch.cat([z_atm_bcast, z_surf], dim=-1)
        rad_coeffs_normed = self.physics_head(phys_in)        # (B, K, n_pca_rad)
        prior_std = (self.rad_pca_variances + 1e-10).sqrt()
        rad_coeffs_recon = rad_coeffs_normed * prior_std
        radiance_recon = torch.matmul(
            rad_coeffs_recon, self.rad_pca_components,
        ) + self.rad_pca_mean                                 # (B, K, W)

        return FoundationOutput(
            reflectance=reflectance,
            radiance_recon=radiance_recon,
            rad_coeffs_normed=rad_coeffs_normed,
            temp_mu=temp_mu,
            temp_log_var=temp_log_var,
            material_logits=material_logits,
            atmos_mu=atmos_mu,
            atmos_log_var=atmos_log_var,
            z_atm=z_atm,
            z_atm_mu=mu_atm,
            z_atm_log_var=log_var_atm,
            z_surf=z_surf,
            z_surf_mu=mu_surf,
            z_surf_log_var=log_var_surf,
            pca_mu=pca_mu,
            pca_var=pca_var,
            pixel_mask=pixel_mask,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        output: FoundationOutput,
        target_reflectance: Tensor,       # (B, K, W) or (B, W)
        target_radiance: Tensor,          # (B, K, W) or (B, W)
        target_atmos: Tensor,             # (B, n_atmos)
        target_temperature: Tensor,       # (B, K, 1) or (B, 1)
        target_material: Tensor | None = None,  # (B, K) or (B,) long
    ) -> dict[str, Tensor]:
        """Multi-task loss with physics consistency.

        All per-pixel tensors are internally reshaped to (B, K, ...) to
        match the model's output.
        """
        c = self.config
        pixel_mask = output.pixel_mask                        # (B, K)
        mask_f = pixel_mask.float()
        n_valid = mask_f.sum().clamp(min=1.0)

        # Ensure per-pixel targets have K dim
        def _maybe_unsqueeze(t: Tensor) -> Tensor:
            return t.unsqueeze(1) if t.dim() == output.reflectance.dim() - 1 else t

        target_reflectance = _maybe_unsqueeze(target_reflectance)
        target_radiance = _maybe_unsqueeze(target_radiance)
        target_temperature = _maybe_unsqueeze(target_temperature)

        # ---- Reflectance (MSE per wavelength, masked by pixel) ----
        refl_err = (output.reflectance - target_reflectance) ** 2    # (B, K, W)
        refl_loss = (refl_err.mean(dim=-1) * mask_f).sum() / n_valid

        # ---- Physics consistency (in normalized PCA-coefficient space) ----
        # Project the target radiance into normalized PCA space so the loss
        # is in units of "squared prior std" and scale-matched to the other
        # per-pixel losses.
        prior_std = (self.rad_pca_variances + 1e-10).sqrt()
        centered = target_radiance - self.rad_pca_mean                # (B, K, W)
        target_coeffs_normed = torch.matmul(
            centered, self.rad_pca_components.T,
        ) / prior_std                                                 # (B, K, n_pca)
        phys_err = (output.rad_coeffs_normed - target_coeffs_normed) ** 2
        phys_loss = (phys_err.mean(dim=-1) * mask_f).sum() / n_valid

        # ---- Atmosphere: scene-shared Gaussian NLL ----
        atm_err_sq = (target_atmos - output.atmos_mu) ** 2
        atm_nll = 0.5 * (output.atmos_log_var + atm_err_sq / output.atmos_log_var.exp())
        atm_loss = atm_nll.mean()

        # ---- Temperature: per-pixel Gaussian NLL ----
        temp_err_sq = (target_temperature - output.temp_mu) ** 2
        temp_nll = 0.5 * (output.temp_log_var + temp_err_sq / output.temp_log_var.exp())
        temp_nll = temp_nll.squeeze(-1)                              # (B, K)
        temp_loss = (temp_nll * mask_f).sum() / n_valid

        # ---- Material (optional): per-pixel cross-entropy ----
        if target_material is not None and c.w_material > 0:
            target_material = _maybe_unsqueeze(
                target_material.unsqueeze(-1)
            ).squeeze(-1) if target_material.dim() == 1 else target_material
            B, K, C = output.material_logits.shape
            ce = F.cross_entropy(
                output.material_logits.reshape(B * K, C),
                target_material.reshape(B * K).long(),
                reduction="none",
            ).view(B, K)
            mat_loss = (ce * mask_f).sum() / n_valid
        else:
            mat_loss = torch.tensor(0.0, device=output.reflectance.device)

        # ---- KL divergences ----
        kl_atm = _gaussian_kl_standard(output.z_atm_mu, output.z_atm_log_var).mean()
        kl_surf_per = _gaussian_kl_standard(output.z_surf_mu, output.z_surf_log_var)
        kl_surf = (kl_surf_per * mask_f).sum() / n_valid

        total = (
            c.w_refl * refl_loss
            + c.w_physics * phys_loss
            + c.w_atmos * atm_loss
            + c.w_temp * temp_loss
            + c.w_material * mat_loss
            + c.beta_atm * kl_atm
            + c.beta_surf * kl_surf
        )

        return {
            "total": total,
            "reflectance": refl_loss.detach(),
            "physics": phys_loss.detach(),
            "atmosphere": atm_loss.detach(),
            "temperature": temp_loss.detach(),
            "material": mat_loss.detach(),
            "kl_atm": kl_atm.detach(),
            "kl_surf": kl_surf.detach(),
        }
