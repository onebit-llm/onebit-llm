# OneBit-LLM

1-bit decoder-only LLM in Rust with **binary/ternary** layers: weights are quantized to ±1 (binary) or {-1,0,+1} (ternary) in the forward pass; gradients use the straight-through estimator (STE); training remains in full precision.

---

## ⚠️ NOT PRODUCTION READY

**This project is experimental and not suitable for production use.**

- **Research / proof-of-concept only.** Trained models produce structurally plausible but often repetitive or incoherent text. Output quality is not comparable to full-precision or production LLMs.
- **No guarantees.** Code, configs, and checkpoints may change without notice. Use at your own risk.
- **No benchmarks or reproducibility claims.** Loss curves and sample outputs are for development only. We do not claim parity with any baseline.

If you need a production-ready model, use an established framework and full-precision or officially supported quantized models instead.

---

## Current status

- **Architecture:** Decoder-only transformer (GPT-style) with ternary linear layers (BitNet-style AbsMean), ReLU² in FFN, RMSNorm (subln), RoPE, optional Arenas residual, **QK-norm** (RMSNorm on Q/K before attention), **residual scaling** (1/√2), and **weight scaling (γ)** after quantized matmuls.
- **Training:** Full pipeline — text/JSONL loading, BPE tokenizer, AdamW, gradient clipping, cosine LR decay, label smoothing, checkpointing. Supports **streaming** for large datasets and **max-epochs** for fixed-length runs.
- **Inference:** Load `model.safetensors` or the latest `checkpoint-*.safetensors`, generate from a prompt with configurable temperature.

---

## Repository structure

```
onebit-llm/
├── Cargo.toml
├── Cargo.lock
├── config.json              # Model config (hidden_size 512, QK-norm, residual scaling, etc.)
├── config.bitnet.json       # Alternative BitNet-style config
├── README.md
├── docs/
│   └── REFERENCE.md
├── src/
│   ├── lib.rs
│   ├── config.rs
│   ├── binary.rs             # Binary/Ternary linear + weight scaling γ
│   ├── model.rs              # Transformer + QK-norm, residual scaling
│   ├── data.rs               # TextDataset, StreamingBatchIter
│   └── bin/
│       ├── train.rs          # Train CLI (streaming, max-epochs, label smoothing)
│       ├── export.rs
│       └── run.rs            # Inference (loads latest checkpoint if no model.safetensors)
├── data/                     # Gitignored; add tokenizer and datasets here
├── checkpoints/              # Gitignored; training output
└── exported/                 # Gitignored; export output
```

---

## Build

```bash
cargo build --release
```

CUDA is enabled in `Cargo.toml` by default (candle with `cuda`). For CPU-only, remove the `cuda` feature from the candle dependencies.

---

## Usage

### Train

You need a **config** (JSON), a **data** path (file or directory of `.txt` / `.jsonl`), and a **tokenizer** (`tokenizer.json`, e.g. GPT-2 BPE).

**Non-streaming (small data, e.g. Wikitext-2):**

```bash
cargo run --bin train -- \
  --config config.json \
  --data-dir ./data/wikitext2 \
  --tokenizer ./data/superior-reasoning/tokenizer.json \
  --output-dir ./checkpoints \
  --batch-size 8 \
  --max-steps 0 \
  --max-epochs 10 \
  --save-every 1000 \
  --log-every 100
```

**Streaming (large data):**

```bash
cargo run --bin train -- \
  --config config.json \
  --data-dir ./data/superior-reasoning \
  --tokenizer ./data/superior-reasoning/tokenizer.json \
  --output-dir ./checkpoints \
  --streaming \
  --batch-size 4 \
  --max-steps 10000 \
  --save-every 1000
```

**Training options:** `--lr`, `--lr-min` (e.g. 1e-6 for cosine), `--lr-warmup-steps` (e.g. 200 for small data), `--label-smoothing` (default 0.1), `--grad-clip-max-norm` (default 1.0).

### Run (inference)

Load from `--model-dir`; if `model.safetensors` is missing, the latest `checkpoint-*.safetensors` is loaded.

```bash
cargo run --bin run -- \
  --model-dir ./checkpoints \
  --tokenizer ./data/superior-reasoning/tokenizer.json \
  --prompt "The" \
  --max-tokens 64 \
  --temperature 0.4
```

Lower temperature (e.g. 0.4) tends to give more stable output for 1-bit models.

### Export

Bundle a checkpoint for distribution:

```bash
cargo run --bin export -- \
  --checkpoint-dir ./checkpoints \
  --output-dir ./exported \
  --tokenizer ./data/superior-reasoning/tokenizer.json
```

---

## Config

`OneBitLlmConfig` (JSON) includes:

| Field | Description |
|-------|-------------|
| `vocab_size`, `hidden_size`, `num_heads`, `num_layers`, `intermediate_size`, `max_seq_len`, `layer_norm_eps` | Core dimensions |
| `use_ternary` | Ternary {-1,0,+1} (AbsMean) instead of binary ±1 |
| `use_relu2` | ReLU² in FFN (BitNet-style) |
| `use_subln` | RMSNorm instead of LayerNorm |
| `use_rope` | Rotary position embeddings |
| `use_qk_norm` | RMSNorm on Q and K before attention (stability, lower loss) |
| `use_residual_scaling` | Scale sublayer output by 1/√2 before residual add |
| `arenas_initial`, `arenas_anneal_steps` | Full-precision residual path (anneals to 0) |

Defaults are in `src/config.rs`. `vocab_size` must match the tokenizer.

---

## Data

- **Wikitext-2:** Small; use non-streaming, `--max-epochs 10`. Place `train.txt` (and optionally `test.txt`) under e.g. `data/wikitext2/`.
- **Superior-Reasoning (JSONL):** Large; use `--streaming`. Each line can have `"text"` or `"input"`/`"output"`; the loader concatenates input+output. Put tokenizer in the same or another path and pass via `--tokenizer`.

---

## Project layout

- `src/config.rs` — Config struct and JSON load/save.
- `src/binary.rs` — Binary/ternary linear layers with STE and weight scaling (γ).
- `src/model.rs` — Decoder blocks, attention (with QK-norm), FFN, residual scaling.
- `src/data.rs` — Text dataset and streaming batch iterator.
- `src/bin/train.rs` — Training CLI.
- `src/bin/run.rs` — Inference CLI.
- `src/bin/export.rs` — Export CLI.
- `docs/REFERENCE.md` — Paper/framework references.

---

## License

MIT
