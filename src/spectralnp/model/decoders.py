"""Component C: Task-specific decoders.

All decoders consume the same representation (r, z) from the spectral aggregator.

1. SpectralDecoder  — continuous spectral reconstruction (DeepONet-style)
2. MaterialDecoder  — material / surface property classification
3. AtmosphericDecoder — atmospheric parameter estimation with evidential uncertainty
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spectralnp.model.band_encoder import SpectralPositionalEncoding
from spectralnp.model.evidential import NIGHead


class SpectralDecoder(nn.Module):
    """Predict radiance (or reflectance) at arbitrary query wavelengths.

    This is a continuous operator: given the latent representation (r, z)
    from the encoder and a set of query wavelengths, it outputs predicted
    spectral values at those wavelengths.  This enables spectral super-
    resolution, gap-filling, and reconstruction to any target sensor.

    Parameters
    ----------
    d_model : int
        Dimension of the deterministic representation r.
    z_dim : int
        Dimension of the stochastic latent z.
    n_frequencies : int
        Frequencies for encoding query wavelengths.
    hidden : int
        Hidden width of the decoder MLP.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        d_model: int = 256,
        z_dim: int = 128,
        n_frequencies: int = 64,
        hidden: int = 512,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.query_enc = SpectralPositionalEncoding(n_frequencies, d_model)

        layers: list[nn.Module] = []
        in_dim = d_model + z_dim + d_model  # r + z + query_encoding
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else hidden
            layers.extend([nn.Linear(in_dim if i == 0 else hidden, out_dim), nn.GELU()])
        # Final output: predicted radiance (mean) + log_var for aleatoric noise.
        layers.append(nn.Linear(hidden, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        r: Tensor,
        z: Tensor,
        query_wavelength: Tensor,
        query_fwhm: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict spectral radiance at query wavelengths.

        Parameters
        ----------
        r : Tensor[B, d_model]
            Deterministic representation.
        z : Tensor[B, z_dim]
            Sampled stochastic latent.
        query_wavelength : Tensor[B, Q]
            Wavelengths to predict at (nm).
        query_fwhm : Tensor[B, Q], optional
            FWHM for query bands.  Defaults to a narrow 1 nm if not given.

        Returns
        -------
        mu : Tensor[B, Q]
            Predicted mean radiance.
        log_var : Tensor[B, Q]
            Log variance (aleatoric uncertainty per wavelength).
        """
        B, Q = query_wavelength.shape
        if query_fwhm is None:
            query_fwhm = torch.ones_like(query_wavelength)

        q_enc = self.query_enc(query_wavelength, query_fwhm)  # (B, Q, d_model)

        # Broadcast r and z across query wavelengths.
        r_exp = r.unsqueeze(1).expand(-1, Q, -1)   # (B, Q, d_model)
        z_exp = z.unsqueeze(1).expand(-1, Q, -1)   # (B, Q, z_dim)

        inp = torch.cat([r_exp, z_exp, q_enc], dim=-1)  # (B, Q, d_model+z_dim+d_model)
        out = self.mlp(inp)  # (B, Q, 2)
        mu = out[..., 0]
        log_var = out[..., 1]
        return mu, log_var


class MaterialDecoder(nn.Module):
    """Classify surface material from spectral representation.

    Parameters
    ----------
    d_model : int
    z_dim : int
    n_classes : int
        Number of material classes.
    hidden : int
    """

    def __init__(
        self,
        d_model: int = 256,
        z_dim: int = 128,
        n_classes: int = 100,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, r: Tensor, z: Tensor) -> Tensor:
        """Return logits (B, n_classes)."""
        return self.mlp(torch.cat([r, z], dim=-1))


class AtmosphericDecoder(nn.Module):
    """Estimate atmospheric state parameters with evidential uncertainty.

    Default targets (can be configured):
        0: aerosol optical depth (AOD at 550 nm)
        1: columnar water vapour (g/cm^2)
        2: ozone column (DU)
        3: visibility (km)

    Parameters
    ----------
    d_model : int
    z_dim : int
    n_params : int
        Number of atmospheric parameters to estimate.
    hidden : int
    """

    def __init__(
        self,
        d_model: int = 256,
        z_dim: int = 128,
        n_params: int = 4,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_model + z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.nig_head = NIGHead(hidden, n_params)

    def forward(
        self, r: Tensor, z: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return NIG parameters (gamma, nu, alpha, beta) each (B, n_params)."""
        feat = self.backbone(torch.cat([r, z], dim=-1))
        return self.nig_head(feat)
