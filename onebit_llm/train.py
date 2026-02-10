"""Training loop: scheduler, optimizer, step with optional gradient checkpointing."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from onebit_llm.config import OneBitLlmConfig
from onebit_llm.model import OneBitLlm
from onebit_llm.quantize import set_quant_anneal_frac


def cross_entropy_with_label_smoothing(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
    vocab_size: int,
) -> torch.Tensor:
    """logits (N, V), labels (N,) long. Returns scalar."""
    if smoothing <= 0:
        return F.cross_entropy(logits, labels)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(log_probs, labels, reduction="mean")
    neg_sum_mean = -log_probs.sum(dim=-1).mean()
    return (1.0 - smoothing) * nll + (smoothing / vocab_size) * neg_sum_mean


class LRScheduler:
    """Warmup then constant (max_steps=0) or cosine decay."""

    def __init__(
        self,
        lr: float,
        lr_min: float,
        warmup_steps: int,
        max_steps: int = 0,
        decay: str = "none",
    ):
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.decay = decay
        self.step_count = 0

    def current_lr(self) -> float:
        s = self.step_count
        if self.warmup_steps > 0 and s < self.warmup_steps:
            return self.lr * (s + 1) / self.warmup_steps
        if self.max_steps == 0 or self.decay == "none":
            return self.lr
        s = min(s, self.max_steps)
        if s <= self.warmup_steps:
            return self.lr
        decay_steps = max(1, self.max_steps - self.warmup_steps)
        progress = (s - self.warmup_steps) / decay_steps
        if self.decay == "cosine":
            return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * progress))
        return self.lr

    def advance(self) -> None:
        self.step_count += 1

    def set_step(self, step: int) -> None:
        self.step_count = step


class Trainer:
    def __init__(
        self,
        model: OneBitLlm,
        config: OneBitLlmConfig,
        lr: float = 5e-3,
        lr_min: float = 1e-6,
        lr_warmup_steps: int = 2000,
        lr_decay: str = "none",
        weight_decay: float = 0.0,
        grad_clip_max_norm: float = 1.0,
        label_smoothing: float = 0.1,
        quant_warmup_steps: int = 2000,
        quant_anneal_steps: int = 8000,
        output_dir: str | Path = "checkpoints",
        gradient_checkpointing_segments: int = 0,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.gradient_checkpointing_segments = gradient_checkpointing_segments
        self.label_smoothing = label_smoothing
        self.quant_warmup_steps = quant_warmup_steps
        self.quant_anneal_steps = quant_anneal_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = LRScheduler(lr, lr_min, lr_warmup_steps, max_steps=0, decay=lr_decay)
        self.grad_clip_max_norm = grad_clip_max_norm
        self.global_step = 0

    def step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        arenas_coef: float | None = None,
    ) -> dict:
        """One training step. input_ids/labels (B, L). Returns dict with loss, lr."""
        self.model.train()
        self.optimizer.zero_grad()
        use_ckpt = self.gradient_checkpointing_segments > 0
        segment_size = self.gradient_checkpointing_segments if use_ckpt else 0
        logits = self.model(
            input_ids,
            arenas_coef=arenas_coef,
            use_checkpointing=use_ckpt,
            segment_size=segment_size,
        )
        B, L, V = logits.shape
        logits_flat = logits.view(B * L, V)
        labels_flat = labels.view(B * L).long().clamp(0, V - 1)
        loss = cross_entropy_with_label_smoothing(
            logits_flat,
            labels_flat,
            self.label_smoothing,
            self.config.vocab_size,
        )
        loss.backward()
        if self.grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
        self.optimizer.step()
        lr = self.scheduler.current_lr()
        self.optimizer.param_groups[0]["lr"] = lr
        self.scheduler.advance()
        self.global_step += 1
        # Quant annealing
        if self.global_step < self.quant_warmup_steps:
            set_quant_anneal_frac(0.0)
        else:
            progress = (self.global_step - self.quant_warmup_steps) / max(1, self.quant_anneal_steps)
            set_quant_anneal_frac(min(1.0, progress))
        return {"loss": loss.item(), "lr": lr}

    def save_checkpoint(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "model.safetensors"
        from safetensors.torch import save_file
        save_file(self.model.state_dict(), path)
        (self.output_dir / "global_step.txt").write_text(str(self.global_step))
        self.config.save(self.output_dir / "config.json")
        return path

    def load_weights(self, path: str | Path) -> None:
        from safetensors.torch import load_file
        state = load_file(path)
        self.model.load_state_dict(state, strict=False)

    @staticmethod
    def read_global_step(output_dir: str | Path) -> int | None:
        p = Path(output_dir) / "global_step.txt"
        if not p.exists():
            return None
        return int(p.read_text().strip())
