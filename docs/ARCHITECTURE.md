# Architecture

## Model

- **Decoder-only transformer** (GPT-style): embedding → N × (attention + FFN) → final norm → tied LM head.
- **Weight tying**: token embedding and output projection share the same weight matrix.
- **RoPE**: rotary position embeddings in attention.
- **QK-norm**: RMSNorm on Q and K before attention (optional).
- **Residual scaling**: sublayer outputs scaled by \(1/\sqrt{2}\) before adding to the residual (optional).

## Quantization

- **Ternary** \(\{-1, 0, +1\}\): dynamic threshold \(\delta = 0.7 \times \operatorname{mean}(|W|)\); weights inside the band map to 0.
- **Binary** \(\pm 1\): sign quantization.
- **STE (straight-through estimator)**: forward uses quantized values; backward passes gradient through as if identity (scaled). Soft phase uses \(\tanh(\alpha x)\) with \(\alpha\) increasing during annealing; hard phase uses \(\operatorname{sign}(x)\).
- **Sandwich layout**: embedding and LM head stay in full precision (f16); hidden layers use ternary or binary per config.

## Training

- **Gradient checkpointing**: `torch.utils.checkpoint` recomputes block segments in the backward pass to reduce activation memory.
- **LR schedule**: linear warmup then constant (streaming) or cosine decay.
- **Quant annealing**: global fraction moves from 0 (soft) to 1 (hard) over `quant_anneal_steps` after `quant_warmup_steps`.

## Stack-stream

- Shards are listed from the Hugging Face dataset repo (`data/*.parquet`).
- Language-weighted sampling; each shard is downloaded to `local_root`, converted to `content.jsonl`, then trained for `steps_per_shard` steps.
- Disk usage is capped; oldest shards are evicted when over budget (current and recent shards are protected).
