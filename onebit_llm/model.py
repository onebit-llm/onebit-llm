"""Decoder-only transformer (OneBitLlm) with gradient checkpointing."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from onebit_llm.config import OneBitLlmConfig
from onebit_llm.norm import RMSNorm
from onebit_llm.attention import CausalSelfAttention, rope_cos_sin
from onebit_llm.attention import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, config: OneBitLlmConfig, layer_idx: int):
        super().__init__()
        self.config = config
        quant = config.decoder_layer_quant_mode(layer_idx)
        self.ln1 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.attn = CausalSelfAttention(
            config.hidden_size,
            config.num_heads,
            config.head_dim,
            quant,
            config.use_qk_norm,
            config.ste_scale_factor,
            config.latent_clamp_max,
            config.layer_norm_eps,
        )
        self.ln2 = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.ffn = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            quant,
            config.use_relu2,
            config.ste_scale_factor,
            config.latent_clamp_max,
        )
        self.residual_scale = 1.0 / math.sqrt(2.0) if config.use_residual_scaling else 1.0

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        arenas_coef: float | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = self.attn(x, cos_sin)
        x = residual + x * self.residual_scale
        if arenas_coef is not None:
            x = x + residual * arenas_coef

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x * self.residual_scale
        if arenas_coef is not None:
            x = x + residual * arenas_coef
        return x


class OneBitLlm(nn.Module):
    """Decoder-only 1-bit/ternary LLM. Weight tying: wte and lm_head share weights."""

    def __init__(self, config: OneBitLlmConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            DecoderBlock(config, i) for i in range(config.num_layers)
        ])
        self.ln_f = RMSNorm(config.hidden_size, config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        arenas_coef: float | None = None,
        use_checkpointing: bool = False,
        segment_size: int = 0,
    ) -> torch.Tensor:
        """
        input_ids: (B, L) long.
        use_checkpointing: if True, use torch.utils.checkpoint for blocks (saves memory).
        segment_size: if > 0, checkpoint every this many blocks; else checkpoint each block.
        """
        x = self.wte(input_ids)
        L = x.size(1)
        # Build cos/sin with same dtype/device as x for checkpoint
        if self.config.use_rope:
            cos, sin = rope_cos_sin(L, self.config.head_dim, x.device)
            cos_sin = (cos.to(x.dtype), sin.to(x.dtype))
        else:
            cos_sin = None

        if use_checkpointing and segment_size <= 0:
            segment_size = 1
        if use_checkpointing and segment_size > 0:
            num_segments = (self.config.num_layers + segment_size - 1) // segment_size
            for seg in range(num_segments):
                start = seg * segment_size
                end = min(start + segment_size, self.config.num_layers)
                # Checkpoint this segment (recompute in backward)
                x = checkpoint(
                    self._forward_blocks,
                    x,
                    cos_sin,
                    arenas_coef,
                    start,
                    end,
                    use_reentrant=True,
                )
        else:
            x = self._forward_blocks(x, cos_sin, arenas_coef, 0, self.config.num_layers)

        x = self.ln_f(x)
        logits = torch.nn.functional.linear(x, self.wte.weight)  # weight tying
        return logits

    def _forward_blocks(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
        arenas_coef: float | None,
        start: int,
        end: int,
    ) -> torch.Tensor:
        for i in range(start, end):
            x = self.blocks[i](x, cos_sin=cos_sin, arenas_coef=arenas_coef)
        return x
