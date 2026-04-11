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
        use_r: bool = True,
    ) -> None:
        super().__init__()
        self.query_enc = SpectralPositionalEncoding(n_frequencies, d_model)
        self.use_r = use_r

        layers: list[nn.Module] = []
        # Optionally drop r from the decoder input. Forces the decoder
        # to depend on the latent z, mitigating posterior collapse where
        # the deterministic representation r dominates and z is ignored.
        in_dim = (d_model if use_r else 0) + z_dim + d_model
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

        # Broadcast z (and optionally r) across query wavelengths.
        z_exp = z.unsqueeze(1).expand(-1, Q, -1)   # (B, Q, z_dim)
        if self.use_r:
            r_exp = r.unsqueeze(1).expand(-1, Q, -1)   # (B, Q, d_model)
            inp = torch.cat([r_exp, z_exp, q_enc], dim=-1)
        else:
            inp = torch.cat([z_exp, q_enc], dim=-1)
        out = self.mlp(inp)  # (B, Q, 2)
        mu = out[..., 0]
        log_var = out[..., 1]
        return mu, log_var


class GridDecoder(nn.Module):
    """Decode latent z to a fixed wavelength grid via 1D conv stack.

    Unlike SpectralDecoder which decodes each query wavelength independently,
    this produces the full spectrum in one shot.  Adjacent wavelengths share
    information through convolutional kernels, enabling better reconstruction
    of sharp spectral features (absorption bands, red edges).

    Output is (mu, log_var) at each grid point for heteroscedastic uncertainty.

    Parameters
    ----------
    z_dim : int
        Dimension of the stochastic latent z.
    n_grid : int
        Number of output wavelength grid points.
    hidden_channels : int
        Channel width in the conv stack.
    n_blocks : int
        Number of residual conv blocks.
    """

    def __init__(
        self,
        z_dim: int = 256,
        n_grid: int = 425,
        hidden_channels: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_grid = n_grid

        # Project z to a spatial feature map: (B, z_dim) → (B, C, L_init)
        self.l_init = max(n_grid // 8, 16)
        self.proj = nn.Linear(z_dim, hidden_channels * self.l_init)
        self.z_drop = nn.Dropout(dropout)

        # Progressive upsampling: interleave conv blocks with 2× upsamples
        # so each block operates at increasing resolution.
        # 53 → 106 → 212 → 425 (3 upsamples for 4 blocks)
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        sizes = [self.l_init]
        for i in range(n_blocks):
            self.stages.append(_ResConvBlock(hidden_channels, dropout))
            if i < n_blocks - 1:
                next_size = min(sizes[-1] * 2, n_grid)
                self.upsamples.append(
                    nn.Upsample(size=next_size, mode="linear", align_corners=False)
                )
                sizes.append(next_size)
        # Final upsample to exact grid size.
        self.final_upsample = nn.Upsample(size=n_grid, mode="linear", align_corners=False)

        # Separate heads for mu and log_var so they don't compete for capacity.
        self.mu_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )
        self.logvar_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels // 4, 1, kernel_size=1),
        )

    def forward(
        self,
        r: Tensor,
        z: Tensor,
        query_wavelength: Tensor | None = None,
        query_fwhm: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Decode z to the fixed grid.

        Parameters
        ----------
        r : Tensor[B, d_model]
            Ignored (kept for API compatibility).
        z : Tensor[B, z_dim]
        query_wavelength, query_fwhm : ignored (grid is fixed).

        Returns
        -------
        mu : Tensor[B, n_grid]
        log_var : Tensor[B, n_grid]
        """
        B = z.shape[0]
        h = self.z_drop(self.proj(z)).reshape(B, -1, self.l_init)  # (B, C, L_init)
        for i, stage in enumerate(self.stages):
            h = stage(h)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)
        h = self.final_upsample(h)  # (B, C, n_grid)
        mu = self.mu_head(h).squeeze(1)          # (B, n_grid)
        log_var = self.logvar_head(h).squeeze(1)  # (B, n_grid)
        # Clamp log_var to prevent numerical explosion.
        log_var = log_var.clamp(-7.0, 4.0)
        return mu, log_var


class _ResConvBlock(nn.Module):
    """Pre-norm residual 1D conv block: norm → conv → GELU → conv + skip."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


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
        use_r: bool = True,
    ) -> None:
        super().__init__()
        self.use_r = use_r
        in_dim = (d_model if use_r else 0) + z_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, r: Tensor, z: Tensor) -> Tensor:
        """Return logits (B, n_classes)."""
        if self.use_r:
            return self.mlp(torch.cat([r, z], dim=-1))
        return self.mlp(z)


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
        use_r: bool = True,
    ) -> None:
        super().__init__()
        self.use_r = use_r
        in_dim = (d_model if use_r else 0) + z_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.nig_head = NIGHead(hidden, n_params)

    def forward(
        self, r: Tensor, z: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return NIG parameters (gamma, nu, alpha, beta) each (B, n_params)."""
        inp = torch.cat([r, z], dim=-1) if self.use_r else z
        feat = self.backbone(inp)
        return self.nig_head(feat)
