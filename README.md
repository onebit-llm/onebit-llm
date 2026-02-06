<p align="center">
  <strong>Ternary-Core: 1.58-bit LLM Framework in Rust</strong><br>
  <em>Mixed-precision training, search, and inference for models small enough for embedded devices.</em>
</p>

<p align="center">
  <a href="https://github.com/onebit-llm/ternary-core/actions"><img src="https://img.shields.io/github/actions/workflow/status/onebit-llm/onebit-llm/ci.yml?label=build" alt="Build"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="License"></a>
  <a href="https://crates.io/crates/ternary-core"><img src="https://img.shields.io/crates/v/ternary-core?label=crates.io" alt="Crates.io"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-2021%20edition-orange" alt="Rust"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions"></a>
</p>

---

## Why Ternary-Core?

Standard LLMs use 16-bit or 32-bit weights. **Ternary-Core** uses **mixed-precision** (the **Sandwich Rule**): keep **embedding** and **LM head** in high precision (F16 / 8-bit) and compress **hidden layers** to **1.58-bit Ternary** (or 1-bit Binary). That gives **~20× compression vs FP16** while avoiding the **Information Collapse** that full quantization caused.

### How we achieve ~20× compression vs FP16

| Precision   | Bits/param | Relative size |
|------------|------------|----------------|
| FP32       | 32         | 1× (baseline)  |
| FP16       | 16         | 0.5×           |
| 8-bit      | 8          | 0.25×          |
| **Ternary**| **~1.58**  | **~0.1×**      |
| Binary     | 1          | ~0.06×         |

By keeping only the middle layers in Ternary and the input/output in F16, we get a model that is **small enough for embedded devices (e.g. ESP32)** but **smart enough to generate coherent text**.

---

## Architecture

### Sandwich quantization (mixed precision)

```
  [Embedding]          F16 / 8-bit   (high precision — avoid information collapse)
       ↓
  [Decoder Layer 0]     Ternary / Binary
  [Decoder Layer 1]     Ternary / Binary
  ...                   ...
  [Decoder Layer N]     Ternary / Binary
       ↓
  [LM Head]             F16 / 8-bit   (high precision — coherent logits)
```

The **search** crate produces a **bit-map JSON** that pins embedding and LM head to high precision and assigns Ternary/Binary per decoder layer; **training** and **inference** consume this map.

### Workspace layout

```
                            +------------------------------------------+
                            |           ternary-core workspace        |
                            +------------------------------------------+
                                              |
                +-------------+---------------+---------------+-------------+
                v             v               v               v             v
        +--------------+ +------------+ +--------------+ +----------+ +----------+
        | ternary-core | |  ternary-  | | ternary-     | | ternary- | | ternary- |
        |              | |  common   | | train        | | search   | | infer    |
        |  BitLinear   | |  Config   | |  AdamW +     | | Bit-map  | | KV-cache |
        |  QuantMode   | |  LayerBit | |  Latent clip | | Min-ppl  | | Top-P    |
        |  RMSNorm    | |  Tokenizer| |  Annealing   | | Pinned   | | Export   |
        |  RoPE       | |  Mmap     | |  Scheduler   | | Eval.    | | .1bit    |
        +--------------+ +------------+ +--------------+ +----------+ +----------+
               |               |               |              |            |
               +---------------+---------------+--------------+------------+
                                         candle-core
                                    (CPU / CUDA / Metal / WASM)
```

### Crate overview

| Crate | Role | Binaries |
|-------|------|----------|
| **ternary-core** | Math engine: `BitLinear` (F16/8bit/Ternary/Binary), RMSNorm, SwiGLU, RoPE, STE, annealing | -- |
| **ternary-common** | Config, `QuantMode`, `LayerBitMap`, tokenizer, mmap datasets | -- |
| **ternary-train** | AdamW, latent clipping (default 1.2), cosine LR, annealing | `onebit-train`, `onebit-tokenize` |
| **ternary-search** | Bit-width search with **min-perplexity** constraint; output **bit-map JSON**; embedding/head **pinned** F16 | `onebit-search`, `onebit-eval-config` |
| **ternary-infer** | KV-cache, repetition penalty, Top-P sampling, `.1bit` export | `onebit-chat`, `onebit-export`, `onebit-test-generate` |

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

### 1. Download data

```bash
python scripts/download_wikitext.py   # or wget WikiText-103 / OpenWebText
```

### 2. Tokenize (optional; for mmap or pre-tokenized)

```bash
cargo run -p ternary-train --bin onebit-tokenize -- \
  --data-dir data/wikitext-103-raw --tokenizer tokenizer.json \
  --out-dir data/tokenized
```

### 3. Train (mixed-precision: Sandwich Rule by default)

```bash
cargo run --release -p ternary-train --bin onebit-train --features cuda -- \
  --config config.json \
  --data-dir ./data/tokenized \
  --tokenizer ./data/tokenizer.json \
  --output-dir ./checkpoints \
  --batch-size 4 --accumulation-steps 4 \
  --lr 5e-3 --lr-decay cosine --lr-warmup-steps 200 \
  --max-steps 10000 --save-every 1000 --log-every 100
```

### 4. Search (bit-map with min-perplexity; embedding/head pinned)

```bash
cargo run --release -p ternary-search --bin onebit-search --features cuda -- \
  --model-config config.json --checkpoint checkpoints/model.safetensors \
  --val-data data/val.txt --tokenizer data/tokenizer.json \
  --min-perplexity-max 50 \
  --output search_result.json
```

`search_result.json` contains `layer_bit_map` (embedding/lm_head F16, per-layer Ternary/Binary) for inference.

### 5. Chat (inference)

```bash
cargo run --release -p ternary-infer --bin onebit-chat --features cuda -- \
  --model-dir ./checkpoints \
  --temperature 0.7 --top-p 0.9 --repetition-penalty 1.2
```

### Build

```bash
# CPU-only (default)
cargo build --release

# With CUDA acceleration
cargo build --release --features cuda

# With Metal (Apple Silicon)
cargo build --release --features metal
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

### Search (optimal bit-width map; min-perplexity constraint)

```bash
cargo run --release -p ternary-search --bin onebit-search --features cuda -- \
  --model-config config.json \
  --checkpoint checkpoints/model.safetensors \
  --val-data data/val.txt \
  --tokenizer data/tokenizer.json \
  --min-perplexity-max 50 \
  --max-size-mb 100 \
  --output search_result.json
```

The output JSON includes `layer_bit_map` (embedding/lm_head pinned F16).

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

`config.json` controls the model. For **mixed precision**, use `layer_bit_map` (or leave unset for Sandwich default: embedding/lm_head F16, layers Ternary).

| Field | Default | Description |
|-------|---------|-------------|
| `vocab_size` | 50257 | Vocabulary size (must match tokeniser) |
| `hidden_size` | 512 | Model dimension |
| `num_heads` | 8 | Attention heads |
| `num_layers` | 6 | Decoder layers |
| `intermediate_size` | 2048 | FFN intermediate dim |
| `max_seq_len` | 512 | Maximum sequence length |
| `use_ternary` | false | Global: Ternary vs Binary (ignored if `layer_bit_map` set) |
| `layer_bit_map` | null | Per-layer `QuantMode` (embedding, layer_modes, lm_head); Sandwich default when null |
| `use_swiglu` | false | SwiGLU (3 projections) |
| `use_rope` | false | Rotary position embeddings |
| `use_qk_norm` | true | RMSNorm on Q/K |
| `use_residual_scaling` | true | Scale sub-layer by 1/√2 |
| `use_dynamic_threshold` | true | δ = 0.7×mean(\|W\|) for ternary |
| `ste_scale_factor` | 2.0 | STE gradient multiplier |
| `latent_clamp_max` | 1.5 | Forward clamp for latent weights |
| `latent_clip_max_training` | 1.2 | Clamp after each optimizer step |
| `anneal_fraction` | 0.3 | Fraction of training in soft annealing |
| `arenas_initial` | null | Arenas FP residual (null = disabled) |
| `arenas_anneal_steps` | 10000 | Steps to anneal Arenas → 0 |

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

After every optimiser step, latent weights are clamped to `[-latent_clip_max_training, +latent_clip_max_training]`
(default **1.2**) to prevent gradient explosion. Forward pass uses `latent_clamp_max` (default 1.5).

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

### Zero-copy with MmapDataset

For large datasets, avoid loading all tokens into RAM by using a pre-tokenized binary file and **MmapDataset** (backed by `memmap2`). Only the pages touched for each batch are paged in.

1. **Create a `.tokens` file** (one-off): load text, tokenize, and write the binary format:
   ```bash
   cargo run -p ternary-train --bin onebit-tokenize -- \
     --data-dir data/wikitext-2 --tokenizer data/tokenizer.json \
     --seq-len 256 --output data/wikitext-2/train.tokens
   ```
   Or from code: `TextDataset::load()` then `dataset.write_tokenized("path/to/train.tokens")`.

2. **Train with mmap**: point `--data-dir` at the `.tokens` file. The trainer detects the extension and uses `MmapDataset` automatically:
   ```bash
   cargo run -p ternary-train --bin onebit-train --features cuda -- \
     --config config_wikitext.json --data-dir data/wikitext-2/train.tokens \
     --tokenizer data/tokenizer.json --output-dir checkpoints/wikitext ...
   ```
   Validation can use a `.tokens` file too: `--val-data-dir data/wikitext-2/valid.tokens`.

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

## Benchmarks (Size vs Perplexity)

| Config | Size (approx) | Perplexity (target) | Notes |
|--------|----------------|---------------------|-------|
| Pure Ternary (all layers) | Smallest | High (collapse risk) | Not recommended |
| **Sandwich (F16 embed/head + Ternary body)** | **~20× vs FP16** | **Controlled** | **Default; use min-perplexity in search** |
| Mixed (search with min-perplexity-max) | Variable | ≤ threshold | Search output bit-map |

Use `--min-perplexity-max` in search to enforce a perplexity ceiling; the search outputs a bit-map JSON that meets the constraint while minimizing size.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/SANDWICH_RULE.md](docs/SANDWICH_RULE.md) | Mixed-precision and why embedding/head stay high precision |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Workspace layout and crate roles |
| [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) | Detailed architectural blueprint |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards |

---

## Roadmap

- [x] Cargo Workspace with 5 crates (core, common, train, search, infer)
- [x] **Mixed-precision BitLinear** (QuantMode: F16, EightBit, Ternary, Binary)
- [x] **Sandwich Rule** (embedding/lm_head high precision; LayerBitMap)
- [x] Soft → hard annealing, STE, latent clipping (default 1.2)
- [x] RMSNorm, RoPE, QK-Norm, SwiGLU, Arenas
- [x] Trainer: AdamW, gradient accumulation, cosine LR
- [x] **Search: min-perplexity constraint, bit-map JSON output, pin embed/head**
- [x] Inference: KV-cache, Top-P, repetition penalty, `.1bit` export
- [x] `memmap2` zero-copy datasets; `tokio` async search
- [ ] WASM target; `no_std` core for embedded (e.g. ESP32)
- [ ] Hugging Face integration; speculative decoding

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
