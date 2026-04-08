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
    log_var_min: float = -10.0,
    log_var_max: float = 10.0,
) -> Tensor:
    """Heteroscedastic Gaussian NLL for spectral reconstruction.

    The model predicts both mean and log-variance at each wavelength.

    Stability: ``log_var`` is clamped so that the NLL formula stays
    bounded. With ``log_var_min=-10`` we cap precision at e^10 ≈ 22000,
    which prevents the (y-mu)^2 / exp(log_var) term from exploding when
    the model becomes overconfident on wrong predictions. The clamp is
    differentiable (straight-through).
    """
    pred_log_var = pred_log_var.clamp(log_var_min, log_var_max)
    precision = torch.exp(-pred_log_var)
    nll = 0.5 * (pred_log_var + precision * (target - pred_mu) ** 2)
    return nll.mean()


def np_kl_divergence(
    z_mu: Tensor,
    z_log_sigma: Tensor,
    prior_mu: Tensor | None = None,
    prior_log_sigma: Tensor | None = None,
    log_sigma_min: float = -2.0,
    log_sigma_max: float = 2.0,
    free_bits: float = 0.05,
) -> Tensor:
    """KL divergence between the context posterior and a prior.

    For self-supervised pretraining, the prior is the posterior conditioned
    on ALL bands (the "target" posterior). During inference, we use a
    standard normal prior.

    Stability:
      - log_sigma is clamped to ``[log_sigma_min, log_sigma_max]`` to
        prevent the KL from blowing up when the prior is overconfident
        (small sigma_p) or the context posterior collapses
      - "free bits": per-dimension KL is floored at ``free_bits``, which
        prevents the latent from collapsing into the prior (deactivates
        the dimension)

    KL(q(z|context) || p(z)) where both are diagonal Gaussians.
    """
    if prior_mu is None:
        # Standard normal prior.
        prior_mu = torch.zeros_like(z_mu)
        prior_log_sigma = torch.zeros_like(z_log_sigma)

    # Clamp for stability — prevents the formula from exploding when
    # sigma_p is very small (overconfident prior) or sigma_q is huge.
    z_log_sigma = z_log_sigma.clamp(log_sigma_min, log_sigma_max)
    prior_log_sigma = prior_log_sigma.clamp(log_sigma_min, log_sigma_max)

    sigma_q_sq = (2 * z_log_sigma).exp()
    sigma_p_sq = (2 * prior_log_sigma).exp()

    kl_per_dim = (
        prior_log_sigma - z_log_sigma
        + (sigma_q_sq + (z_mu - prior_mu) ** 2) / (2 * sigma_p_sq)
        - 0.5
    )
    # Free bits: floor each dimension's KL at `free_bits`, sum over dims.
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_total = kl_per_dim.sum(dim=-1).mean()
    # Soft ceiling: log(1 + KL) keeps gradient flowing for any KL value
    # but prevents the absolute scale from dominating the total loss when
    # KL is large (e.g. early training, before posteriors align).
    return torch.log1p(kl_total)


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


def material_loss(
    logits: Tensor,
    target: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Cross-entropy for material classification, optionally class-weighted.

    USGS categories are highly imbalanced (~880 minerals vs ~11 coatings),
    so unweighted CE collapses to predicting "minerals" all the time.
    Pass per-class weights inversely proportional to category frequency.
    """
    return F.cross_entropy(logits, target, weight=class_weights)


def calibration_loss(pred_mu: Tensor, pred_log_var: Tensor, target: Tensor) -> Tensor:
    """Calibration regulariser: penalise mismatch between predicted variance
    and the empirical squared error.

    The standard heteroscedastic NLL has a known failure mode where the
    optimum drives ``log_var`` toward -∞ for any near-perfect prediction.
    This regulariser pulls ``log_var`` toward ``log((y-mu)^2)``, i.e. the
    empirical squared error, which is what a well-calibrated Gaussian
    should report.

    Returns the mean absolute log-ratio.
    """
    sq_err = (target - pred_mu) ** 2 + 1e-8
    log_sq_err = torch.log(sq_err)
    return (pred_log_var - log_sq_err).abs().mean()


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
        w_calibration: float = 0.0,
        material_class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.w_spectral = w_spectral
        self.w_reflectance = w_reflectance
        self.w_atmos = w_atmos
        self.w_kl = w_kl
        self.w_material = w_material
        self.w_evidence_reg = w_evidence_reg
        self.w_calibration = w_calibration
        # Register so it moves with .to(device).
        if material_class_weights is not None:
            self.register_buffer("material_class_weights", material_class_weights)
        else:
            self.material_class_weights = None

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
            if self.w_calibration > 0:
                losses["spectral_calib"] = calibration_loss(
                    output.spectral_mu, output.spectral_log_var, target_radiance
                )

        # Surface reflectance reconstruction.
        if output.reflectance_mu is not None and target_reflectance is not None:
            losses["reflectance"] = spectral_reconstruction_loss(
                output.reflectance_mu, output.reflectance_log_var, target_reflectance
            )
            if self.w_calibration > 0:
                losses["reflectance_calib"] = calibration_loss(
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
            losses["material"] = material_loss(
                output.material_logits, target_material,
                class_weights=self.material_class_weights,
            )

        # Weighted total.
        total = torch.tensor(0.0, device=target_radiance.device)
        if "spectral" in losses:
            total = total + self.w_spectral * losses["spectral"]
        if "reflectance" in losses:
            total = total + self.w_reflectance * losses["reflectance"]
        if "spectral_calib" in losses:
            total = total + self.w_calibration * losses["spectral_calib"]
        if "reflectance_calib" in losses:
            total = total + self.w_calibration * losses["reflectance_calib"]
        if "atmos" in losses:
            total = total + self.w_atmos * losses["atmos"]
        if "kl" in losses:
            total = total + self.w_kl * losses["kl"]
        if "material" in losses:
            total = total + self.w_material * losses["material"]
        losses["total"] = total

        return losses
