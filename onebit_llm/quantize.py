"""STE and ternary/binary quantization (matches Rust quantize.rs)."""

from __future__ import annotations

import torch
import torch.nn as nn

# Global anneal fraction 0..1: 0 = soft (tanh), 1 = hard (sign). Set by trainer.
_anneal_frac: float = 1.0


def set_quant_anneal_frac(frac: float) -> None:
    global _anneal_frac
    _anneal_frac = max(0.0, min(1.0, frac))


def current_anneal_frac() -> float:
    return _anneal_frac


def ste_sign_scaled(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Hard STE: forward ≈ sign(x), backward = scale * identity."""
    sign_x = x.sign()
    return sign_x + (x - x.detach()) * scale


def ste_tanh_scaled(x: torch.Tensor, alpha: float, scale: float) -> torch.Tensor:
    """Soft STE: forward ≈ tanh(α*x), backward = scale * identity."""
    y = torch.tanh(x * alpha)
    return y + (x - x.detach()) * scale


def ternary_quantize_dynamic(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Ternary with dynamic threshold δ = 0.7 * mean(|W|). Returns (W_q, scale)."""
    w_abs = w.abs()
    delta = 0.7 * w_abs.mean()
    w_q = torch.where(w_abs > delta, w.sign(), torch.zeros_like(w))
    gamma = w_abs.mean().clamp(min=1e-8)
    return w_q, gamma.item()


def binary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Binary ±1. Returns (W_bin, gamma)."""
    w_q = w.sign()
    gamma = w.abs().mean().clamp(min=1e-8)
    return w_q, gamma.item()
