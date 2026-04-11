"""Component B: Spectral Aggregator.

Transformer encoder with rotary positional encoding on wavelength,
plus a Neural Process dual-path architecture:
  - Deterministic path: cross-attention into a latent bottleneck
  - Stochastic path: mean-pooled posterior over a latent variable z

The stochastic path is what gives the model uncertainty that naturally
widens when fewer bands are observed.
"""

from __future__ import annotations


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

    Uses **learned-query cross-attention** to aggregate band features
    into a fixed-size representation, then projects to (mu, log_sigma).

    Previous implementation used mean-pooling, which destructively
    averages per-band information as band count grows (accuracy at
    observation points degraded from 13→30 bands because each band's
    contribution was diluted by 1/N). Cross-attention lets the model
    learn WHICH information to extract regardless of N.
    """

    def __init__(self, d_model: int, z_dim: int, n_heads: int = 4, n_queries: int = 32) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.pre = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Learned queries that attend to band features.
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        # Flatten K attended queries and project to z — preserves the
        # per-query structure instead of destroying it with mean-pool.
        # K*d_model → z_dim captures much more spectral detail.
        self.to_mu = nn.Linear(n_queries * d_model, z_dim)
        self.to_log_sigma = nn.Linear(n_queries * d_model, z_dim)

    def forward(self, h: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return (mu, log_sigma) of q(z | context).

        h : (B, N, d_model)
        mask : (B, N) bool, True = valid
        """
        s = self.pre(h)  # (B, N, d_model)
        B = s.shape[0]
        # Cross-attend: learned queries → band features.
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
        attended = self.cross_attn(queries, s, mask=mask)       # (B, K, d)
        attended = self.norm(attended)
        # Flatten K×d into a single vector — preserves per-query structure.
        flat = attended.reshape(B, -1)  # (B, K*d)
        mu = self.to_mu(flat)
        log_sigma = self.to_log_sigma(flat)
        # Clamp for numerical stability only. Allow tight posteriors
        # (low log_sigma) when the encoder is confident — this is what
        # gives us band-count-dependent uncertainty.
        # -6 → sigma ≈ 0.0025 (very tight but not zero)
        log_sigma = log_sigma.clamp(-6.0, 2.0)
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

        # Stochastic path: per-pixel surface latent.
        self.stochastic = StochasticEncoder(d_model, z_dim, n_heads=n_heads)

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
        h = self.encode_bands(h, wavelength, pad_mask)

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

    def encode_bands(
        self,
        h: Tensor,
        wavelength: Tensor,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        """Run self-attention over bands only (shared by single- and multi-pixel paths).

        Returns the transformed band features h.
        """
        for layer in self.layers:
            h = layer(h, wavelength, pad_mask)
        return h


class CrossPixelAggregator(nn.Module):
    """Aggregate per-pixel representations into a shared atmospheric latent.

    Given deterministic representations from K context pixels that share
    the same atmosphere, produce a posterior q(z_atm | pixel_1, ..., pixel_K).

    With K=1 the posterior is wide (uninformative) and the model degrades
    gracefully to single-pixel behaviour.

    Parameters
    ----------
    d_model : int
        Per-pixel representation dimension.
    z_atm_dim : int
        Atmospheric latent dimensionality.
    n_heads : int
        Attention heads for cross-pixel attention.
    n_queries : int
        Number of learnable atmospheric query vectors.
    """

    def __init__(
        self,
        d_model: int = 256,
        z_atm_dim: int = 32,
        n_heads: int = 4,
        n_queries: int = 8,
    ) -> None:
        super().__init__()
        self.atm_queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.to_mu = nn.Linear(d_model, z_atm_dim)
        self.to_log_sigma = nn.Linear(d_model, z_atm_dim)

    def forward(
        self,
        pixel_reps: Tensor,
        pixel_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Produce atmospheric latent posterior from context pixels.

        Parameters
        ----------
        pixel_reps : Tensor[B, K, d_model]
            Deterministic representations of K context pixels.
        pixel_mask : Tensor[B, K] bool, optional
            True = valid pixel, False = padding.

        Returns
        -------
        z_atm_mu : Tensor[B, z_atm_dim]
        z_atm_log_sigma : Tensor[B, z_atm_dim]
        """
        B = pixel_reps.shape[0]
        queries = self.atm_queries.unsqueeze(0).expand(B, -1, -1)
        attended = self.cross_attn(queries, pixel_reps, mask=pixel_mask)
        pooled = self.norm(attended).mean(dim=1)  # (B, d_model)
        mu = self.to_mu(pooled)
        log_sigma = self.to_log_sigma(pooled).clamp(-6.0, 2.0)
        return mu, log_sigma
