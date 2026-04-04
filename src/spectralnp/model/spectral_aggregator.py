"""Component B: Spectral Aggregator.

Transformer encoder with rotary positional encoding on wavelength,
plus a Neural Process dual-path architecture:
  - Deterministic path: cross-attention into a latent bottleneck
  - Stochastic path: mean-pooled posterior over a latent variable z

The stochastic path is what gives the model uncertainty that naturally
widens when fewer bands are observed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Rotary positional encoding on wavelength
# ---------------------------------------------------------------------------

def _rotary_embedding(wavelength: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Compute rotary sin/cos embeddings from wavelength values.

    We use the wavelength (in nm) as the "position" for RoPE so that
    attention naturally favours spectrally nearby bands.

    Parameters
    ----------
    wavelength : Tensor[B, N]
    dim : int  — must be even

    Returns
    -------
    cos, sin : Tensor[B, N, dim//2]
    """
    half = dim // 2
    # Frequency schedule: geometric from 1/10000 to 1 (in 1/nm units).
    freq_seq = torch.arange(half, device=wavelength.device, dtype=wavelength.dtype)
    inv_freq = 1.0 / (10000.0 ** (freq_seq / half))  # (half,)
    # Outer product: (B, N, 1) * (half,) -> (B, N, half)
    phase = wavelength.unsqueeze(-1) * inv_freq
    return phase.cos(), phase.sin()


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to the last dimension of x.

    x : (..., dim)  where dim is even.
    cos, sin : (..., dim//2)
    """
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# Transformer layer with RoPE on wavelength
# ---------------------------------------------------------------------------

class SpectralSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self, x: Tensor, wavelength: Tensor, pad_mask: Tensor | None = None
    ) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, n_heads, head_dim)

        # Apply RoPE to q, k using wavelength as position.
        cos, sin = _rotary_embedding(wavelength, self.head_dim)  # (B, N, head_dim//2)
        cos = cos.unsqueeze(2)  # (B, N, 1, head_dim//2)
        sin = sin.unsqueeze(2)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # (B, n_heads, N, head_dim) for scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if pad_mask is not None:
            # pad_mask: (B, N) True=valid -> convert to attn_mask: (B, 1, 1, N)
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # broadcast over heads & queries
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = SpectralSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, wavelength: Tensor, pad_mask: Tensor | None = None
    ) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x), wavelength, pad_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Latent bottleneck cross-attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to keys/values."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, queries: Tensor, context: Tensor, mask: Tensor | None = None) -> Tensor:
        B, Nq, D = queries.shape
        Nk = context.shape[1]
        q = self.q_proj(queries).reshape(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, Nk, 2, self.n_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, Nq, D))


# ---------------------------------------------------------------------------
# Neural Process stochastic path
# ---------------------------------------------------------------------------

class StochasticEncoder(nn.Module):
    """Map a variable-length set of band features to a diagonal-Gaussian latent z.

    Uses masked mean-pooling so padding tokens don't contribute.
    """

    def __init__(self, d_model: int, z_dim: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.to_mu = nn.Linear(d_model, z_dim)
        self.to_log_sigma = nn.Linear(d_model, z_dim)

    def forward(self, h: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return (mu, log_sigma) of q(z | context).

        h : (B, N, d_model)
        mask : (B, N) bool, True = valid
        """
        s = self.pre(h)  # (B, N, d_model)
        if mask is not None:
            s = s * mask.unsqueeze(-1).float()
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, 1)
            pooled = s.sum(dim=1) / counts  # (B, d_model)
        else:
            pooled = s.mean(dim=1)
        mu = self.to_mu(pooled)
        log_sigma = self.to_log_sigma(pooled)
        # Clamp for numerical stability.
        log_sigma = log_sigma.clamp(-10.0, 2.0)
        return mu, log_sigma


# ---------------------------------------------------------------------------
# Full spectral aggregator
# ---------------------------------------------------------------------------

class SpectralAggregator(nn.Module):
    """Transformer encoder + NP dual-path aggregator.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads in transformer & cross-attention.
    n_layers : int
        Number of transformer self-attention blocks.
    n_latents : int
        Number of learnable latent bottleneck vectors.
    z_dim : int
        Dimensionality of the stochastic latent variable.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_latents: int = 64,
        z_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Self-attention transformer over input band tokens.
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Deterministic path: learnable latent queries cross-attend into band features.
        self.latent_queries = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.det_cross_attn = CrossAttention(d_model, n_heads)
        self.det_norm = nn.LayerNorm(d_model)
        # Collapse latent queries into a single deterministic vector.
        self.det_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Stochastic path.
        self.stochastic = StochasticEncoder(d_model, z_dim)

    def forward(
        self,
        h: Tensor,
        wavelength: Tensor,
        pad_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Process band features through transformer and NP paths.

        Parameters
        ----------
        h : Tensor[B, N, d_model]
            Per-band features from BandEncoder.
        wavelength : Tensor[B, N]
            Center wavelengths (needed for RoPE in self-attention).
        pad_mask : Tensor[B, N] bool, optional
            True for valid bands, False for padding.

        Returns
        -------
        r : Tensor[B, d_model]
            Deterministic representation.
        z_mu : Tensor[B, z_dim]
            Mean of the latent posterior.
        z_log_sigma : Tensor[B, z_dim]
            Log-std of the latent posterior.
        """
        # Self-attention over bands.
        for layer in self.layers:
            h = layer(h, wavelength, pad_mask)

        # --- Deterministic path ---
        B = h.shape[0]
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        det = self.det_cross_attn(queries, h, mask=pad_mask)  # (B, K, d)
        det = self.det_norm(det)
        # Mean-pool over K latent vectors.
        r = self.det_pool(det.mean(dim=1))  # (B, d_model)

        # --- Stochastic path ---
        z_mu, z_log_sigma = self.stochastic(h, pad_mask)

        return r, z_mu, z_log_sigma
