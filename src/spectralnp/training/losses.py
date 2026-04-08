"""Loss functions for SpectralNP training.

The combined loss balances four objectives:
1. Spectral reconstruction (MSE at held-out wavelengths)
2. Atmospheric parameter estimation (evidential NIG loss)
3. NP KL divergence (consistency between context and target posteriors)
4. Material classification (cross-entropy, when labels available)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from spectralnp.model.evidential import evidential_regulariser, nig_nll


def spectral_reconstruction_loss(
    pred_mu: Tensor,
    pred_log_var: Tensor,
    target: Tensor,
) -> Tensor:
    """Heteroscedastic Gaussian NLL for spectral reconstruction.

    The model predicts both mean and log-variance at each wavelength,
    so it can express per-wavelength confidence.

    Parameters
    ----------
    pred_mu : (B, Q) predicted radiance
    pred_log_var : (B, Q) predicted log-variance
    target : (B, Q) true radiance

    Returns
    -------
    Scalar loss.
    """
    # Gaussian NLL: 0.5 * (log_var + (y - mu)^2 / exp(log_var))
    precision = torch.exp(-pred_log_var)
    nll = 0.5 * (pred_log_var + precision * (target - pred_mu) ** 2)
    return nll.mean()


def np_kl_divergence(
    z_mu: Tensor,
    z_log_sigma: Tensor,
    prior_mu: Tensor | None = None,
    prior_log_sigma: Tensor | None = None,
) -> Tensor:
    """KL divergence between the context posterior and a prior.

    For self-supervised pretraining, the prior is the posterior conditioned
    on ALL bands (the "target" posterior).  During inference, we use a
    standard normal prior.

    KL(q(z|context) || p(z)) where both are diagonal Gaussians.
    """
    if prior_mu is None:
        # Standard normal prior.
        prior_mu = torch.zeros_like(z_mu)
        prior_log_sigma = torch.zeros_like(z_log_sigma)

    sigma_q = z_log_sigma.exp()
    sigma_p = prior_log_sigma.exp()

    kl = (
        prior_log_sigma - z_log_sigma
        + (sigma_q**2 + (z_mu - prior_mu) ** 2) / (2 * sigma_p**2)
        - 0.5
    )
    return kl.sum(dim=-1).mean()


def atmospheric_loss(
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
    target: Tensor,
    reg_weight: float = 0.01,
) -> Tensor:
    """Combined NIG NLL + evidence regulariser for atmospheric parameters."""
    nll = nig_nll(target, gamma, nu, alpha, beta)
    reg = evidential_regulariser(target, gamma, nu, alpha, beta)
    return nll + reg_weight * reg


def material_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Cross-entropy for material classification."""
    return F.cross_entropy(logits, target)


class SpectralNPLoss(torch.nn.Module):
    """Combined training loss for SpectralNP.

    Parameters
    ----------
    w_spectral : weight for spectral reconstruction
    w_atmos : weight for atmospheric estimation
    w_kl : weight for NP KL divergence
    w_material : weight for material classification (0 to disable)
    w_evidence_reg : weight for evidential regulariser within atmos loss
    """

    def __init__(
        self,
        w_spectral: float = 1.0,
        w_reflectance: float = 1.0,
        w_atmos: float = 0.1,
        w_kl: float = 0.01,
        w_material: float = 0.1,
        w_evidence_reg: float = 0.01,
    ) -> None:
        super().__init__()
        self.w_spectral = w_spectral
        self.w_reflectance = w_reflectance
        self.w_atmos = w_atmos
        self.w_kl = w_kl
        self.w_material = w_material
        self.w_evidence_reg = w_evidence_reg

    def forward(
        self,
        output,  # SpectralNPOutput
        target_radiance: Tensor,        # (B, Q)
        target_atmos: Tensor,           # (B, n_params)
        target_reflectance: Tensor | None = None,  # (B, Q)
        target_material: Tensor | None = None,  # (B,) long
        prior_mu: Tensor | None = None,
        prior_log_sigma: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all loss components and total.

        Returns dict with individual losses and 'total'.
        """
        losses: dict[str, Tensor] = {}

        # Spectral reconstruction (radiance).
        if output.spectral_mu is not None:
            losses["spectral"] = spectral_reconstruction_loss(
                output.spectral_mu, output.spectral_log_var, target_radiance
            )

        # Surface reflectance reconstruction.
        if output.reflectance_mu is not None and target_reflectance is not None:
            losses["reflectance"] = spectral_reconstruction_loss(
                output.reflectance_mu, output.reflectance_log_var, target_reflectance
            )

        # Atmospheric parameters.
        if output.atmos_gamma is not None:
            losses["atmos"] = atmospheric_loss(
                output.atmos_gamma,
                output.atmos_nu,
                output.atmos_alpha,
                output.atmos_beta,
                target_atmos,
                reg_weight=self.w_evidence_reg,
            )

        # KL divergence.
        if output.z_mu is not None:
            losses["kl"] = np_kl_divergence(
                output.z_mu, output.z_log_sigma,
                prior_mu, prior_log_sigma,
            )

        # Material classification.
        if self.w_material > 0 and target_material is not None and output.material_logits is not None:
            losses["material"] = material_loss(output.material_logits, target_material)

        # Weighted total.
        total = torch.tensor(0.0, device=target_radiance.device)
        if "spectral" in losses:
            total = total + self.w_spectral * losses["spectral"]
        if "reflectance" in losses:
            total = total + self.w_reflectance * losses["reflectance"]
        if "atmos" in losses:
            total = total + self.w_atmos * losses["atmos"]
        if "kl" in losses:
            total = total + self.w_kl * losses["kl"]
        if "material" in losses:
            total = total + self.w_material * losses["material"]
        losses["total"] = total

        return losses
