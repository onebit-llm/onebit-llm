<p align="center">
  <strong>OneBit-LLM</strong><br>
  <em>A comprehensive Rust framework for training, optimising, and deploying 1-bit &amp; 1.58-bit Large Language Models.</em>
</p>

<p align="center">
  <a href="https://github.com/onebit-llm/onebit-llm/actions"><img src="https://img.shields.io/github/actions/workflow/status/onebit-llm/onebit-llm/ci.yml?label=build" alt="Build"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="License"></a>
  <a href="https://crates.io/crates/ternary-core"><img src="https://img.shields.io/crates/v/ternary-core?label=crates.io" alt="Crates.io"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-2021%20edition-orange" alt="Rust"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions"></a>
</p>

---

## Why OneBit?

Standard LLMs store every weight as a 16-bit or 32-bit floating-point number.
**OneBit-LLM** compresses weights to just **1 bit** (Binary: +/-1) or **1.58 bits**
(Ternary: {-1, 0, +1}) during training -- not as a post-hoc quantisation step,
but as a *first-class training objective* via the Straight-Through Estimator.

### Memory comparison

```
Model (45M params)       Memory (weights only)
---------------------    ---------------------
FP32 (baseline)          ~180 MB
FP16                      ~90 MB
INT8                      ~45 MB
Ternary (1.58-bit)       ~11 MB    <-- OneBit-LLM
Binary  (1-bit)           ~5.6 MB
```

> **Core principle:** *Compute is cheap, Memory is expensive.*
> On edge devices, DRAM bandwidth is the bottleneck -- not FLOPs.
> A 16x memory reduction means 16x more model fits in cache.

---

## Architecture

```
                            +------------------------------------------+
                            |           onebit-llm workspace           |
                            +------------------------------------------+
                                              |
                +-------------+---------------+---------------+-------------+
                v             v               v               v             v
        +--------------+ +------------+ +--------------+ +----------+ +----------+
        | ternary-core | |  ternary-  | | ternary-     | | ternary- | | ternary- |
        |              | |  common    | | train        | | search   | | infer    |
        |  BitLinear   | |  Config    | |  AdamW +     | | Expander | | Sampler  |
        |  RMSNorm     | |  Tokenizer | |  Clipping    | | Graph    | | Top-P    |
        |  SwiGLU      | |  Data I/O  | |  Annealing   | | Coord.   | | .1bit    |
        |  RoPE        | |  Safetens. | |  Scheduler   | | Eval.    | | Export   |
        +--------------+ +------------+ +--------------+ +----------+ +----------+
               |               |               |              |            |
               +---------------+---------------+--------------+------------+
                                         candle-core
                                    (CPU / CUDA / Metal / WASM)
```

### Crate overview

| Crate | Role | Binaries |
|-------|------|----------|
| **ternary-core** | Mathematical engine: `BitLinear`, `TernaryLinear`, RMSNorm, SwiGLU, RoPE, QK-Norm, STE, soft-to-hard annealing | -- |
| **ternary-common** | Shared config, tokenizer wrapper, data pipeline (`TextDataset`, `StreamingBatchIter`) | -- |
| **ternary-train** | Trainer with AdamW, latent weight clipping, gradient accumulation, cosine LR scheduler, annealing schedule | `onebit-train` |
| **ternary-search** | Graph-based quantisation search: Expander decomposition, Dijkstra, cached evaluator | `onebit-search`, `onebit-eval-config` |
| **ternary-infer** | Inference runtime, sampler (top-k/top-p/temperature/repetition penalty), `.1bit` binary export | `onebit-chat`, `onebit-export`, `onebit-test-generate` |

### Training loop (Annealing -> Quantise -> Backprop)

```
 +-----------------------------------------------------------------------+
 |                        Training Step t                                |
 |                                                                       |
 |  1. Read annealing fraction  a(t) in [0,1]                           |
 |     +-------------------------------------+                           |
 |     |  a < 1  ->  W_q = tanh(a * W_latent) |  (soft, differentiable) |
 |     |  a = 1  ->  W_q = sign(W_latent)     |  (hard, STE gradient)   |
 |     +-------------------------------------+                           |
 |                                                                       |
 |  2. Forward:  logits = Transformer(x, W_q)                           |
 |  3. Loss:     L = CrossEntropy(logits, labels)                       |
 |  4. Backward: dL/dW_latent  via STE  (scale factor S)                |
 |  5. Update:   W_latent <- AdamW(W_latent, dL/dW_latent)              |
 |  6. Clamp:    W_latent <- clamp(W_latent, -1.5, +1.5)               |
 |                                                                       |
 +-----------------------------------------------------------------------+
```

---

## Repository Structure

```
onebit-llm/
|-- Cargo.toml                  # Workspace root
|-- README.md
|-- LICENSE
|-- CONTRIBUTING.md
|-- CODE_OF_CONDUCT.md
|-- config.json                 # Default model config (512-dim, 6-layer)
|
|-- crates/
|   |-- core/                   # (ternary-core)  BitLinear, RMSNorm, SwiGLU, Quantisation
|   |-- common/                 # (ternary-common) Config, Tokenizer, Data loading
|   |-- train/                  # (ternary-train)  Trainer, AdamW+Clipping, LR Scheduler
|   |-- search/                 # (ternary-search) Expander Graph, Coordinator, Evaluator
|   +-- inference/              # (ternary-infer)  Sampler, Decoding, .1bit Export
|
|-- scripts/                    # Python helpers (random search)
|-- docs/                       # MASTER_PLAN.md
|-- data/                       # (gitignored) datasets and tokenisers
+-- checkpoints/                # (gitignored) training output
```

---

## Quick Start

### Build

```bash
# CPU-only (default -- compiles everywhere including WASM)
cargo build --release

# With CUDA acceleration
cargo build --release --features cuda

# With Metal acceleration (Apple Silicon)
cargo build --release --features metal
```

### Train

```bash
cargo run --release -p ternary-train --bin onebit-train --features cuda -- \
  --config config.json \
  --data-dir ./data/wikitext-103-raw \
  --tokenizer ./data/tokenizer.json \
  --output-dir ./checkpoints \
  --batch-size 4 \
  --accumulation-steps 4 \
  --lr 5e-3 \
  --lr-decay cosine \
  --lr-warmup-steps 200 \
  --max-steps 10000 \
  --save-every 1000 \
  --log-every 100
```

For large datasets that don't fit in RAM, add `--streaming`:

```bash
cargo run --release -p ternary-train --bin onebit-train --features cuda -- \
  --config config.json \
  --data-dir ./data/openwebtext \
  --tokenizer ./data/tokenizer.json \
  --output-dir ./checkpoints \
  --streaming \
  --batch-size 8 \
  --max-steps 100000
```

### Search (optimal bit-width map)

```bash
cargo run --release -p ternary-search --bin onebit-search --features cuda -- \
  --model-config config.json \
  --checkpoint checkpoints/model.safetensors \
  --val-data data/val.txt \
  --tokenizer data/tokenizer.json \
  --max-size-mb 100 \
  --output best_config.json
```

### Chat (interactive inference)

```bash
cargo run --release -p ternary-infer --bin onebit-chat --features cuda -- \
  --model-dir ./checkpoints \
  --temperature 0.7 \
  --top-p 0.9 \
  --repetition-penalty 1.2
```

### Export to `.1bit` format

```bash
cargo run --release -p ternary-infer --bin onebit-export -- \
  --model-dir ./checkpoints \
  --output model.1bit \
  --pack-mode ternary \
  --emit-c-header
```

---

## Configuration

`config.json` controls every aspect of the model architecture:

| Field | Default | Description |
|-------|---------|-------------|
| `vocab_size` | 50257 | Vocabulary size (must match tokeniser) |
| `hidden_size` | 512 | Model dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 6 | Decoder layers |
| `intermediate_size` | 2048 | FFN intermediate dim |
| `max_seq_len` | 512 | Maximum sequence length |
| `use_ternary` | false | Ternary {-1,0,+1} vs Binary +/-1 |
| `use_relu2` | false | ReLU squared activation (BitNet-style) |
| `use_swiglu` | false | SwiGLU activation (LLaMA/Mistral-style, 3 projections) |
| `use_subln` | false | RMSNorm instead of LayerNorm |
| `use_rope` | false | Rotary position embeddings |
| `use_qk_norm` | true | RMSNorm on Q/K (stabilises attention) |
| `use_residual_scaling` | true | Scale sub-layer by 1/sqrt(2) |
| `use_dynamic_threshold` | true | delta = 0.7 x mean(\|W\|) for ternary |
| `ste_scale_factor` | 2.0 | STE gradient multiplier |
| `latent_clamp_max` | 1.5 | Latent weight clamp bound |
| `anneal_fraction` | 0.3 | Fraction of training in soft annealing regime |
| `arenas_initial` | null | Arenas FP residual coefficient (null = disabled) |
| `arenas_anneal_steps` | 10000 | Steps to anneal Arenas -> 0 |

---

## Key Concepts

### Straight-Through Estimator (STE)

1-bit weights are non-differentiable. The STE lets gradients flow through the
quantisation step by using the identity function on the backward pass:

```
Forward:  W_q = sign(W_latent)            # or ternary quantise
Backward: dL/dW_latent = dL/dW_q * S     # S = ste_scale_factor (default 2.0)
```

### Soft -> Hard Annealing

During the first 30% of training (configurable via `anneal_fraction`), we use
`tanh(alpha * x)` with alpha growing from 1 to 8 as a smooth approximation to
`sign(x)`. This prevents gradient death at initialisation when all latent
weights are near zero.

### Latent Weight Clipping

After every optimiser step, latent weights are clamped to `[-latent_clamp_max, +latent_clamp_max]`
(default 1.5). This keeps them within reach of the quantisation thresholds so
they can still flip between {-1, 0, +1} as the model learns.

### KV-Cache (O(1) per-token decoding)

During generation, the inference runtime uses a per-layer key/value cache. The prompt is run once (prefill); then each new token is decoded with sequence length 1, reusing the cache. This gives O(1) work per token instead of reprocessing the full context every step.

### `.1bit` Binary Export

Trained models can be exported to a compact `.1bit` format:
- **Ternary packing:** 4 values per byte (2-bit encoding: 00=0, 01=+1, 11=-1)
- **Binary packing:** 8 values per byte (1-bit encoding: 0=-1, 1=+1)
- Auto-generated C header for embedded deployment

---

## Datasets

| Dataset | Size | Recommendation |
|---------|------|----------------|
| WikiText-2 | 12 MB | Quick experiments only |
| WikiText-103 | 500 MB | Good for ~45M params |
| OpenWebText | 38 GB | Production-quality training |

```bash
# WikiText-103
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip -d data/
```

For WikiText-2, a small Python script is provided to download and format data plus the GPT-2 tokenizer: see `scripts/download_wikitext.py`. It writes `data/wikitext-2/train.txt`, `valid.txt`, `test.txt` and `data/tokenizer.json`.

---

## Experimental results (WikiText-2)

The following describes a real training and inference run on WikiText-2. **This model is not ready for production or general use.** Results are documented for reproducibility and to set accurate expectations.

### Hardware

- **GPU:** NVIDIA RTX 4080 (16 GB VRAM)
- Training and inference were run on this single GPU.

### Settings

- **Config:** `config_wikitext.json` — binary 1-bit (no ternary), 512 hidden size, 8 layers, 8 heads, 1536 intermediate size, **max_seq_len 256** (reduced from 512 to avoid OOM). RMSNorm, RoPE, QK-norm, residual scaling, and dynamic threshold enabled. `ste_scale_factor` 1.0, `latent_clamp_max` 1.5, `anneal_fraction` 0.2, `arenas_initial` 0.1, `arenas_anneal_steps` 5000.
- **Data:** WikiText-2 via `scripts/download_wikitext.py` — train ~998 paragraphs (~413K tokens), validation 176 paragraphs; GPT-2 tokenizer (`data/tokenizer.json`).
- **Training:** AdamW, lr 1e-3, cosine decay to 1e-5, warmup 500 steps, batch_size 2, accumulation_steps 4 (effective batch 8), grad_clip_max_norm 1.0, label_smoothing 0.05. **50,000 steps** with streaming over `data/wikitext-2/`, saving checkpoints every 10K steps and the final model to `checkpoints/wikitext/`.

### Training results

| Step   | Train loss | Val loss | Notes                    |
|--------|------------|----------|--------------------------|
| 5,000  | ~11        | —        | Early checkpoint         |
| 20,000 | ~9.5       | ~26.6    | Mid training             |
| 50,000 | **8.07**   | **25.15**| Final run (completed)    |

Validation perplexity at 50K steps was very high (~8.4e10), indicating the model has not learned to predict text reliably.

### Inference results

Generation was tested with `onebit-test-generate` (temperature 0.7, top-k 40, top-p 0.9, repetition penalty 1.4) on prompts such as *"The history of"*, *"The rain fell"*, *"A small boat"*, *"The scientist"*, *"Stars twinkled"*.

- **Observed behaviour:** The model sometimes produces plausible short beginnings (e.g. *"The history of"*, *"The rain fell"*) and WikiText-style fragments (names, places, numbers), but output quickly degrades into word salad, list-like tokens, and severe repetition (e.g. repeated words or subwords). It has not learned coherent sentence structure or long-range fluency.
- **Conclusion:** The 1-bit model has captured local token and phrase statistics to some degree but is **not suitable for use** as a language model in its current form. Further work (more/better data, longer training, architecture or inference tuning) would be needed to approach usable quality.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) | Detailed architectural blueprint with Rust code specifications |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute (fork, PR, join the org) |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards |

---

## Roadmap

- [x] Cargo Workspace with 5 focused crates
- [x] Binary & Ternary linear layers with STE
- [x] Soft -> Hard annealing schedule (global atomic)
- [x] RMSNorm, RoPE, QK-Norm, residual scaling
- [x] SwiGLU activation (3-projection FFN)
- [x] Arenas residual annealing
- [x] `Trainer` struct with AdamW, latent clipping, gradient accumulation
- [x] Cosine / linear / constant LR scheduler with warmup
- [x] Cross-entropy with label smoothing
- [x] Streaming data pipeline (`TextDataset` + `StreamingBatchIter`)
- [x] Expander-based quantisation search (graph + Dijkstra)
- [x] Cached config evaluator with LRU
- [x] Inference sampler: top-k, top-p, temperature, repetition penalty
- [x] `.1bit` binary export format with C-compatible header
- [x] End-to-end train -> checkpoint -> inference pipeline verified on GPU
- [x] KV-Cache for O(1) per-token decoding
- [ ] `memmap2` zero-copy dataset loading (`MmapDataset`)
- [ ] `tokio`-based async search coordinator
- [ ] WASM compilation target
- [ ] `no_std` core support for embedded
- [ ] Hugging Face model hub integration
- [ ] Speculative decoding with tiny draft model

---

## Contributing

We welcome contributions: code, documentation, issues, and ideas.
See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to contribute.
By participating, you agree to our **[Code of Conduct](CODE_OF_CONDUCT.md)**.

---

## References

- Ma et al., *"The Era of 1-bit LLMs"* (BitNet b1.58), 2024
- Wang et al., *"BitNet: Scaling 1-bit Transformers"*, 2023
- Microsoft Research, *"bitnet.cpp"* -- Inference framework for 1-bit LLMs
- Chang et al., *"Deterministic Distributed Expander Decomposition"*, STOC 2024

---

## License

Dual-licensed under **MIT** and **Apache-2.0**. See [LICENSE](LICENSE) for details.
