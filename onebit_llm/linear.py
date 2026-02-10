"""BitLinear: binary/ternary linear with STE (matches Rust linear.rs)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from onebit_llm.quantize import (
    current_anneal_frac,
    ste_sign_scaled,
    ste_tanh_scaled,
    ternary_quantize_dynamic,
    binary_quantize,
)


class BitLinear(nn.Module):
    """Ternary or binary linear: latent weights, quantized forward, STE backward."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_ternary: bool = True,
        ste_scale_factor: float = 2.0,
        latent_clamp_max: float = 1.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_ternary = use_ternary
        self.ste_scale_factor = ste_scale_factor
        self.latent_clamp_max = latent_clamp_max
        # Small init so weights start in ternary zero band
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        w = self.weight.clamp(-self.latent_clamp_max, self.latent_clamp_max)
        anneal = current_anneal_frac()

        if self.use_ternary:
            # Dynamic threshold ternary
            w_q, gamma = ternary_quantize_dynamic(w)
            if anneal < 1.0:
                alpha = 1.0 + 7.0 * anneal
                w_ste = ste_tanh_scaled(w, alpha, self.ste_scale_factor)
                x_ste = ste_tanh_scaled(x, alpha, self.ste_scale_factor)
            else:
                w_ste = ste_sign_scaled(w, self.ste_scale_factor)
                x_ste = ste_sign_scaled(x, self.ste_scale_factor)
            # Use w_ste for gradient flow, but scale as with quantized gamma
            scale = gamma / (self.in_dim ** 0.5)
            out = torch.nn.functional.linear(x_ste, w_ste)
            return out * scale
        else:
            w_q, gamma = binary_quantize(w)
            if anneal < 1.0:
                alpha = 1.0 + 7.0 * anneal
                w_ste = ste_tanh_scaled(w, alpha, self.ste_scale_factor)
                x_ste = ste_tanh_scaled(x, alpha, self.ste_scale_factor)
            else:
                w_ste = ste_sign_scaled(w, self.ste_scale_factor)
                x_ste = ste_sign_scaled(x, self.ste_scale_factor)
            scale = gamma / (self.in_dim ** 0.5)
            return torch.nn.functional.linear(x_ste, w_ste) * scale


class BitLinearLayer(nn.Module):
    """Wrapper that chooses BitLinear or nn.Linear by quant mode."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        quant_mode: str,
        ste_scale_factor: float = 2.0,
        latent_clamp_max: float = 1.5,
    ):
        super().__init__()
        if quant_mode == "f16":
            self.layer = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.layer = BitLinear(
                in_dim,
                out_dim,
                use_ternary=(quant_mode == "ternary"),
                ste_scale_factor=ste_scale_factor,
                latent_clamp_max=latent_clamp_max,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
