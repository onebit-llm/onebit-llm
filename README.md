# OneBit-LLM

1-bit decoder-only LLM in Rust with **binary/XNOR-style** layers: weights and activations are 1-bit in the forward pass; gradients use the straight-through estimator (STE) so training remains in full precision.

---

## ⚠️ Project status: very early stage

**We are at the very beginning of this project.** This repository is experimental research/exploration, not a usable product.

- **No consistent or coherent text yet.** Trained models (including on small corpora like Tiny Shakespeare) do **not** produce meaningful, coherent, or reliable outputs. Generation is often repetitive, nonsensical, or token soup (e.g. repeated punctuation and high-frequency tokens). This is expected at this stage.
- **Architecture and training are under active development.** Binary/ternary layers, RoPE, ReLU², RMSNorm, and Arenas-style residual paths are implemented for experimentation, but hyperparameters, scaling, and stability are not tuned. We have not yet demonstrated that this 1-bit setup can match or approach full-precision baselines.
- **No benchmarks or reproducibility claims.** We do not report perplexity, benchmarks, or “results” because we have not established a stable training setup or evaluation protocol. Any numbers you see (e.g. loss curves) are for internal debugging only.
- **Use at your own risk.** Code and configs may change without notice. If you try training or inference, treat it as a proof-of-concept and expect rough edges and failures.

We are sharing this early to iterate in the open and to invite feedback from anyone interested in 1-bit LLMs in Rust. If you expect a working, production-ready model, this is not it—yet.

---

## Repository structure

```
onebit-llm/
├── Cargo.toml
├── Cargo.lock
├── config.json              # Default model config (created if missing)
├── config.bitnet.json       # BitNet-style config (ternary, RoPE, ReLU², subln, Arenas)
├── README.md
├── docs/
│   └── REFERENCE.md         # Framework / paper reference
├── src/
│   ├── lib.rs
│   ├── config.rs
│   ├── binary.rs
│   ├── model.rs
│   ├── data.rs
│   └── bin/
│       ├── train.rs
│       ├── export.rs
│       └── run.rs
├── data/                    # Ignored; add tokenizer and datasets here
├── checkpoints/             # Ignored; training output (safetensors, config)
└── exported/                # Ignored; export output
```

## Features

- **Architecture**: Decoder-only transformer (GPT-style) with binary linear layers in attention and FFN.
- **Training**: Full pipeline — data loading, tokenizer (BPE), AdamW optimizer, checkpointing, single-GPU training.
- **Export**: Save model weights (safetensors) and config (JSON) for distribution.
- **Import**: Load a checkpoint and run inference or resume training.

## Build

```bash
cargo build --release
```

CUDA is optional; the project builds on CPU by default. To enable CUDA, add to `Cargo.toml`:

```toml
candle = { ..., features = ["cuda"] }
candle-nn = { ..., features = ["cuda"] }
```

## Usage

### 1. Train from scratch

You need:

- A **config** (JSON). If missing, a default is created at `config.json`.
- A **data** path: file or directory of `.txt` / `.jsonl` files (one text per line; JSONL can have a `"text"` field).
- A **tokenizer** (e.g. GPT-2 BPE) as `tokenizer.json`.

```bash
cargo run --bin train -- \
  --config config.json \
  --data-dir /path/to/texts \
  --tokenizer /path/to/tokenizer.json \
  --output-dir ./checkpoints \
  --batch-size 8 \
  --max-steps 10000 \
  --save-every 1000
```

Checkpoints are written to `--output-dir` as `checkpoint-N.safetensors` and `config.json`; the final run saves `model.safetensors` and `config.json`.

**If loss plateaus (e.g. never drops below ~5):** Academic work on 1-bit/ternary LLMs (BitNet b1.58, “BitNet b1.58 reloaded”, low-bit quantization surveys) finds that 1-bit training is *more robust to higher learning rates* than FP16; BitNet recipes often use a *higher* peak LR. The default `--lr` is now `1e-3`. If loss still plateaus, try `--lr 3e-3` or `--lr 1e-2` (with warmup and decay). Gradient clipping (default 1.0) keeps training stable.

### 2. Export

Bundle a checkpoint for distribution: copy weights and config (and optionally the tokenizer) into an output directory.

```bash
cargo run --bin export -- \
  --checkpoint-dir ./checkpoints \
  --output-dir ./exported \
  --tokenizer /path/to/tokenizer.json
```

This writes `./exported/model.safetensors`, `./exported/config.json`, and (if `--tokenizer` is given) `./exported/tokenizer.json`.

### 3. Run (import + inference)

Load a model from an exported directory and run a dummy forward pass or generate from a prompt.

```bash
# Dummy forward (no tokenizer needed)
cargo run --bin run -- --model-dir ./exported

# Generate from prompt (requires tokenizer.json in model dir or --tokenizer)
cargo run --bin run -- \
  --model-dir ./exported \
  --prompt "Hello, world" \
  --max-tokens 64
```

## Small dataset (Tiny Shakespeare) — quick run

**Note:** Even after a full run, the model will not produce coherent text; use this only to verify that training and inference pipelines run end-to-end.

For a short, reproducible run (no large download):

1. **Data** — Tiny Shakespeare (~1.1 MB) is already in `data/wikitext/train.txt` (Karpathy’s char-rnn source).
2. **Tokenizer** — Use the same GPT-2 `tokenizer.json` (e.g. from `data/superior-reasoning/tokenizer.json` or download from Hugging Face).
3. **Train** (finishes in minutes; use `--max-steps` so it stops):
   ```bash
   cargo run --release --bin train -- \
     --config config.bitnet.json \
     --data-dir ./data/wikitext \
     --tokenizer ./data/superior-reasoning/tokenizer.json \
     --output-dir ./checkpoints \
     --batch-size 8 \
     --max-steps 5000 \
     --save-every 1000
   ```
4. **GPU / batch size** — Check usage with `nvidia-smi`. Use `--batch-size 8` by default; increase (e.g. 16) if GPU has headroom, or reduce if OOM.

## Using Alibaba Superior-Reasoning-SFT for testing

The dataset [Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b](https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b) is supported out of the box. Each JSONL row has `input` and `output`; the loader concatenates them as `input + "\n" + output` for next-token training.

1. Download (e.g. stage1, ~4.6 GB):
   ```bash
   huggingface-cli download Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b \
     Superior-Reasoning-SFT-gpt-oss-120b-stage1-train-data.jsonl \
     --local-dir ./data/superior-reasoning
   ```
2. Use a compatible tokenizer (e.g. GPT-2) and set `vocab_size` in config to match.
3. Train:
   ```bash
   cargo run --bin train -- \
     --data-dir ./data/superior-reasoning \
     --tokenizer /path/to/tokenizer.json \
     --output-dir ./checkpoints \
     --batch-size 4 \
     --max-steps 1000
   ```

For a quick test with a small subset, take the first N lines of the JSONL file into a separate file and point `--data-dir` to the folder containing that file.

## Config

`OneBitLlmConfig` (JSON) includes:

- **Core:** `vocab_size`, `hidden_size`, `num_heads`, `num_layers`, `intermediate_size`, `max_seq_len`, `layer_norm_eps`
- **BitNet-style (for lower loss):** `use_ternary` (ternary weights {-1,0,+1} with AbsMean), `use_relu2` (ReLU² in FFN), `use_subln` (RMSNorm instead of LayerNorm), `use_rope` (rotary position embeddings), `arenas_initial` and `arenas_anneal_steps` (full-precision residual path that anneals to 0).

**For lower loss**, use a config with BitNet-style options enabled (e.g. `config.bitnet.json` or set `use_ternary`, `use_relu2`, `use_subln`, `use_rope` to `true` and `arenas_initial` to a value like `0.1`). Defaults in code are conservative; the example `config.bitnet.json` enables all of these. Training uses full-precision activations; 8-bit activation quantization (W1.58A8) in the doc is for inference only.

Default values are in `src/config.rs`. `vocab_size` must match the tokenizer.

## Project layout

- `docs/REFERENCE.md` — Framework / paper reference (1.58-bit paradigm, BitNet, Sherry, Arenas).
- `src/config.rs` — Model config and JSON save/load.
- `src/binary.rs` — Binary and ternary linear layers with STE.
- `src/model.rs` — Decoder-only transformer (embedding, causal attention, decoder blocks, lm_head).
- `src/data.rs` — Text dataset, tokenization, batching.
- `src/bin/train.rs` — Training CLI.
- `src/bin/export.rs` — Export CLI.
- `src/bin/run.rs` — Import and inference CLI.

## Organization

This project is developed under the [onebit-llm](https://github.com/onebit-llm) GitHub organization. The repo is in an early, experimental state; see the [project status](#-project-status-very-early-stage) section above.

## License

MIT
