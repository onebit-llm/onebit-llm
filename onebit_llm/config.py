"""Model configuration (JSON-compatible with Rust config)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class QuantMode:
    F16 = "f16"
    TERNARY = "ternary"
    BINARY = "binary"


class LayerBitMap:
    def __init__(
        self,
        embedding: str = QuantMode.F16,
        layer_modes: Optional[list[str]] = None,
        lm_head: Optional[str] = None,
    ):
        self.embedding = embedding
        self.layer_modes = layer_modes or []
        self.lm_head = lm_head

    def layer_mode(self, i: int) -> str:
        return self.layer_modes[i] if i < len(self.layer_modes) else QuantMode.TERNARY

    @classmethod
    def sandwich_default(cls, num_layers: int) -> "LayerBitMap":
        return cls(
            embedding=QuantMode.F16,
            layer_modes=[QuantMode.TERNARY] * num_layers,
            lm_head=None,
        )


class OneBitLlmConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 16,
        intermediate_size: int = 2816,
        max_seq_len: int = 1024,
        layer_norm_eps: float = 1e-5,
        use_ternary: bool = True,
        use_relu2: bool = True,
        use_swiglu: bool = False,
        use_subln: bool = True,
        use_rope: bool = True,
        use_qk_norm: bool = True,
        use_residual_scaling: bool = True,
        use_dynamic_threshold: bool = True,
        ste_scale_factor: float = 2.0,
        latent_clamp_max: float = 1.5,
        latent_clip_max_training: float = 1.2,
        anneal_fraction: float = 0.3,
        layer_bit_map: Optional[dict[str, Any]] = None,
        **_: Any,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.layer_norm_eps = layer_norm_eps
        self.use_ternary = use_ternary
        self.use_relu2 = use_relu2
        self.use_swiglu = use_swiglu
        self.use_subln = use_subln
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_residual_scaling = use_residual_scaling
        self.use_dynamic_threshold = use_dynamic_threshold
        self.ste_scale_factor = ste_scale_factor
        self.latent_clamp_max = latent_clamp_max
        self.latent_clip_max_training = latent_clip_max_training
        self.anneal_fraction = anneal_fraction
        self.head_dim = hidden_size // num_heads
        self._layer_bit_map = layer_bit_map
        self._bit_map: Optional[LayerBitMap] = None

    @property
    def layer_bit_map(self) -> LayerBitMap:
        if self._bit_map is None:
            if self._layer_bit_map:
                lb = self._layer_bit_map
                self._bit_map = LayerBitMap(
                    embedding=lb.get("embedding", QuantMode.F16),
                    layer_modes=lb.get("layer_modes"),
                    lm_head=lb.get("lm_head"),
                )
            else:
                self._bit_map = LayerBitMap.sandwich_default(self.num_layers)
        return self._bit_map

    def decoder_layer_quant_mode(self, layer_idx: int) -> str:
        return self.layer_bit_map.layer_mode(layer_idx)

    @classmethod
    def load(cls, path: str | Path) -> "OneBitLlmConfig":
        with open(path) as f:
            d = json.load(f)
        # Preserve layer_bit_map as dict for LayerBitMap parsing
        layer_bit_map = d.pop("layer_bit_map", None)
        cfg = cls(**d)
        cfg._layer_bit_map = layer_bit_map
        return cfg

    def save(self, path: str | Path) -> None:
        d = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
            "layer_norm_eps": self.layer_norm_eps,
            "use_ternary": self.use_ternary,
            "use_relu2": self.use_relu2,
            "use_swiglu": self.use_swiglu,
            "use_subln": self.use_subln,
            "use_rope": self.use_rope,
            "use_qk_norm": self.use_qk_norm,
            "use_residual_scaling": self.use_residual_scaling,
            "use_dynamic_threshold": self.use_dynamic_threshold,
            "ste_scale_factor": self.ste_scale_factor,
            "latent_clamp_max": self.latent_clamp_max,
            "latent_clip_max_training": self.latent_clip_max_training,
            "anneal_fraction": self.anneal_fraction,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
