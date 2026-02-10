"""Causal self-attention with RoPE, QK-norm, fused QKV (BitLinear)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from onebit_llm.linear import BitLinearLayer
from onebit_llm.norm import RMSNorm


def rope_cos_sin(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos/sin for positions [0, seq_len). Shape (seq_len, head_dim/2)."""
    d2 = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(d2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q, k. q,k: (B, H, L, D). cos, sin: (1, 1, L, D/2)."""
    d = q.size(-1)
    d2 = d // 2
    q1, q2 = q[..., :d2], q[..., d2:]
    k1, k2 = k[..., :d2], k[..., d2:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        quant_mode: str,
        use_qk_norm: bool,
        ste_scale_factor: float,
        latent_clamp_max: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        # Fused QKV: 3 * hidden_size
        self.qkv = BitLinearLayer(
            hidden_size, 3 * hidden_size, quant_mode, ste_scale_factor, latent_clamp_max
        )
        self.o_proj = BitLinearLayer(
            hidden_size, hidden_size, quant_mode, ste_scale_factor, latent_clamp_max
        )
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, layer_norm_eps)
            self.k_norm = RMSNorm(head_dim, layer_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, L, H_d = x.shape
        qkv = self.qkv(x)  # (B, L, 3*H)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, H, L, 3, D)
        q, k, v = qkv.unbind(-2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if cos_sin is not None:
            cos, sin = cos_sin
            L_q = q.size(2)
            d2 = q.size(-1) // 2  # RoPE uses half of head_dim
            cos = cos[:L_q].view(1, 1, L_q, d2)
            sin = sin[:L_q].view(1, 1, L_q, d2)
            q, k = apply_rope(q, k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)
        attn = attn.masked_fill(
            torch.tril(torch.ones(L, L, device=attn.device, dtype=torch.bool)).logical_not(),
            float("-inf"),
        )
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, H, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, H_d)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """Standard 2-layer FFN with ReLU² or SiLU."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_mode: str,
        use_relu2: bool,
        ste_scale_factor: float,
        latent_clamp_max: float,
    ):
        super().__init__()
        self.w_up = BitLinearLayer(
            hidden_size, intermediate_size, quant_mode, ste_scale_factor, latent_clamp_max
        )
        self.w_down = BitLinearLayer(
            intermediate_size, hidden_size, quant_mode, ste_scale_factor, latent_clamp_max
        )
        self.use_relu2 = use_relu2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w_up(x)
        h = (h.relu() ** 2) if self.use_relu2 else h.silu()
        return self.w_down(h)
