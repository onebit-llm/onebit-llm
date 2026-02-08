# OneBit-LLM: 1.58-bit Ternary LLM Framework in Rust

**OneBit-LLM** is a high-performance Rust framework for training and deploying extremely compressed Large Language Models. Based on the **BitNet 1.58b** architecture, it uses ternary quantization ({-1, 0, +1}) to achieve up to **10-20x compression** compared to standard FP16 models, making it possible to run LLMs on edge devices and embedded systems.

<p align="center">
  <img src="https://img.shields.io/badge/rust-2021%20edition-orange" alt="Rust">
  <img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue" alt="License">
  <img src="https://img.shields.io/badge/compression-~20x-brightgreen" alt="Compression">
</p>

---

## 🚀 Key Features

- **Ternary Quantization (1.58-bit):** Compresses hidden layers to {-1, 0, 1} while maintaining high performance using the Straight-Through Estimator (STE).
- **The Sandwich Rule (Mixed-Precision):** Automatically keeps critical layers (Embedding and LM Head) in high precision (FP16/8-bit) to prevent information collapse.
- **Advanced Architecture:** Includes RoPE (Rotary Positional Embeddings), Sub-Layer Norm, QK-Norm, and Residual Scaling for stable 1-bit training.
- **Zero-Copy Data Loading:** Uses `mmap` for instant loading of multi-gigabyte tokenized datasets without RAM overhead.
- **Optimized Inference:** O(1) per-token decoding via KV-Caching and highly optimized ternary kernels.
- **Soft-to-Hard Annealing:** Smoothly transitions from continuous weights to discrete ternary values during training to stabilize gradients.

---

## 🏗️ Project Structure

The project is organized into a clean Rust workspace:

- **`crates/core`**: The math engine. Implements `BitLinear`, STE primitives, and custom normalization.
- **`crates/train`**: Full-featured trainer with AdamW, learning rate schedulers (Cosine/Linear), and gradient accumulation.
- **`crates/infer`**: Runtime for low-latency generation, featuring Top-P sampling and KV-cache management.
- **`crates/common`**: Shared utilities for configuration, serialization, and high-performance data loading.
- **`crates/search`**: (Experimental) Automated bit-width search to find the optimal quantization map for a given perplexity target.

---

## 🛠️ Quick Start

### 1. Installation

Ensure you have the Rust toolchain installed. For GPU acceleration, the appropriate CUDA/Metal headers are required.

```bash
# Clone the repository
git clone https://github.com/anil/onebit-llm
cd onebit-llm

# Build with CUDA support
cargo build --release --features cuda
```

### 2. Prepare Data

Download and tokenize the WikiText-2 dataset (or use your own raw text files):

```bash
# Download dataset and tokenizer
python3 scripts/download_wikitext.py

# Tokenize for high-performance training
cargo run -p ternary-train --bin onebit-tokenize -- \
  --data-dir data/wikitext-2 \
  --tokenizer data/tokenizer.json \
  --output data/wikitext-2/train.tokens
```

### 3. Training

Start training a ternary model with the **Sandwich Rule** enabled:

```bash
cargo run --release --features cuda -p ternary-train --bin onebit-train -- \
  --config config.json \
  --data-dir data/wikitext-2/train.tokens \
  --tokenizer data/tokenizer.json \
  --output-dir ./checkpoints \
  --batch-size 4 \
  --accumulation-steps 8 \
  --lr 1e-3
```

---

## 📈 Training Progress & Stability

OneBit-LLM has been verified to effectively overfit small datasets to **Loss < 0.1** in ternary mode, confirming that the Straight-Through Estimator and annealing schedules are correctly routing gradients. 

| Metric | Status |
|--------|--------|
| **Quantization Stability** | Verified (Annealing + STE) |
| **Memory Efficiency** | ~1.2GB VRAM for 25M params at SeqLen 512 |
| **Inference Correctness** | Verified via KV-cache consistency tests |

---

## 📄 License

Dual-licensed under MIT and Apache-2.0. See `LICENSE` for more details.

---

## 📚 References

- *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (BitNet b1.58)
- *BitNet: Scaling 1-bit Transformers for Large Language Models*
