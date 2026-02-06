# Ternary-Core Architecture

## Cargo Workspace Layout

```
ternary-core/
├── Cargo.toml              # Workspace definition
├── README.md
├── docs/                    # Architecture & math
├── scripts/                 # Data prep (e.g. Python)
├── crates/
│   ├── core/               # ternary-core — mathematical engine
│   ├── common/             # ternary-common — config, tokenizer, datasets
│   ├── train/              # ternary-train — training loop, AdamW, annealing
│   ├── search/             # ternary-search — bit-width search, min-perplexity
│   └── inference/          # ternary-infer — KV-cache, sampling, export
└── examples/
    └── train_wikitext.rs   # Example: train on WikiText
```

## Crate Responsibilities

| Crate | Role | Key types |
|-------|------|-----------|
| **ternary-core** | Math engine: BitLinear (mixed precision), RMSNorm, quantization, STE | `QuantMode`, `BitLinear`, `OneBitLlm` |
| **ternary-common** | Shared config, tokenizer, mmap datasets | `OneBitLlmConfig`, `LayerBitMap`, `QuantMode` |
| **ternary-train** | Training: AdamW, cosine LR, annealing, latent clipping | `Trainer`, `AnnealSchedule` |
| **ternary-search** | Search: coordinate evaluator, min-perplexity constraint, bit-map output | Bit-map JSON, layer pinning |
| **ternary-infer** | Runtime: KV-cache, repetition penalty, Top-P | Inference engine |

## Data Flow

1. **Config** (JSON) + optional **LayerBitMap** (JSON) → define model and per-layer precision.
2. **Training** uses annealing (soft→hard) and latent clipping; respects LayerBitMap (embedding/lm_head high precision).
3. **Search** produces a LayerBitMap that minimizes size subject to **min-perplexity** and **pinned** embedding/head.
4. **Inference** loads model + LayerBitMap and runs with KV-cache and sampling.

## Design Principles

- **Separation of concerns**: Core has no training loop; train has no search logic.
- **no_std (where possible)**: Core aims for alloc-only / no_std for future embedded use; currently uses std for candle.
- **Pure Rust stack**: candle (tensors), tokio/rayon (async/parallel), serde/safetensors (serialization).
