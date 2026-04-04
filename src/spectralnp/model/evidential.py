"""Evidential deep learning for calibrated aleatoric + epistemic uncertainty.

Implements the Normal-Inverse-Gamma (NIG) parameterisation from
"Deep Evidential Regression" (Amini et al., NeurIPS 2020).

For a scalar regression target y, the NIG distribution is parameterised by
(gamma, nu, alpha, beta) where:
    gamma  = predicted mean
    nu     = precision of the mean (pseudo-observation count)
    alpha  = shape of the inverse-gamma on variance  (alpha > 1)
    beta   = scale of the inverse-gamma on variance  (beta > 0)

Derived uncertainties:
    aleatoric  = beta / (alpha - 1)          (expected data noise)
    epistemic  = beta / (nu * (alpha - 1))   (model uncertainty)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NIGHead(nn.Module):
    """Output head that produces NIG parameters for each regression target.

    Parameters
    ----------
    d_in : int
        Input feature dimension.
    n_targets : int
        Number of independent regression targets.
    """

    def __init__(self, d_in: int, n_targets: int) -> None:
        super().__init__()
        self.n_targets = n_targets
        # Output 4 parameters per target: gamma, nu, alpha, beta.
        self.linear = nn.Linear(d_in, 4 * n_targets)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Produce NIG parameters.

        Returns gamma, nu, alpha, beta  each (B, n_targets).
        """
        out = self.linear(x)  # (B, 4 * n_targets)
        out = out.reshape(x.shape[0], 4, self.n_targets)
        gamma = out[:, 0]
        # nu > 0, alpha > 1, beta > 0 — use softplus to enforce.
        nu = torch.nn.functional.softplus(out[:, 1]) + 1e-6
        alpha = torch.nn.functional.softplus(out[:, 2]) + 1.0 + 1e-6
        beta = torch.nn.functional.softplus(out[:, 3]) + 1e-6
        return gamma, nu, alpha, beta


def nig_nll(
    y: Tensor,
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> Tensor:
    """Negative log-likelihood of observations under the NIG predictive (Student-t).

    Parameters
    ----------
    y : (B, T)  ground-truth targets
    gamma, nu, alpha, beta : (B, T)  NIG parameters

    Returns
    -------
    Scalar loss (mean over batch and targets).
    """
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll.mean()


def evidential_regulariser(
    y: Tensor,
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
) -> Tensor:
    """Evidence regulariser: penalise evidence (nu) when the prediction is wrong.

    This prevents the model from being confidently wrong — it can only
    accumulate evidence (reduce epistemic uncertainty) when the prediction
    is actually close to the target.
    """
    error = torch.abs(y - gamma)
    reg = error * (2.0 * nu + alpha)
    return reg.mean()


def nig_uncertainty(
    nu: Tensor, alpha: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute aleatoric and epistemic uncertainty from NIG parameters.

    Returns
    -------
    aleatoric : Tensor  — expected data noise variance
    epistemic : Tensor  — model uncertainty variance
    """
    aleatoric = beta / (alpha - 1.0)
    epistemic = beta / (nu * (alpha - 1.0))
    return aleatoric, epistemic
