"""RMSNorm (and LayerNorm) for 1-bit transformer."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps) ** 0.5
        return x / rms * self.weight
