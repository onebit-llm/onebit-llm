# OneBit-LLM — Master Plan

> **Version:** 0.2.0  
> **Status:** Implemented -- workspace migration complete, end-to-end pipeline verified  
> **Audience:** Contributors, reviewers, and the future you at 3 AM debugging gradient death

---

## Table of Contents

1. [Vision & Philosophy](#1-vision--philosophy)
2. [Tech Stack Rationale](#2-tech-stack-rationale)
3. [Workspace Architecture](#3-workspace-architecture)
4. [Component Specifications](#4-component-specifications)
   - 4A. [crates/core — The Mathematical Engine](#4a-cratescore--the-mathematical-engine)
   - 4B. [crates/train — The Stability Engine](#4b-cratestrain--the-stability-engine)
   - 4C. [crates/search — The Optimisation Engine](#4c-cratessearch--the-optimisation-engine)
   - 4D. [crates/inference — The Deployment Engine](#4d-cratesinference--the-deployment-engine)
   - 4E. [crates/common — Shared Primitives](#4e-cratescommon--shared-primitives)
5. [Migration Plan](#5-migration-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Future Work](#7-future-work)

---

## 1. Vision & Philosophy

### From Script to Framework

The codebase has been migrated from a monolithic single crate (~2,500 lines) to a
**Cargo Workspace** with five focused crates. This migration was the single most
important architectural decision — each crate can now be embedded independently,
compiled to WASM, and tested in isolation.

### Core Principles

1. **"Compute is cheap, Memory is expensive."**
   On edge devices, DRAM bandwidth dominates latency. A ternary matmul on
   packed 2-bit weights running through L1 cache beats an FP16 matmul that
   stalls on L2 cache misses. Every design decision must be evaluated through
   this lens.

2. **Quantisation is a first-class training objective.**
   We do not train an FP32 model and then quantise it. The quantisation
   function is *inside* the forward pass. The optimiser sees quantised
   activations. The loss surface reflects the true inference model.

3. **Hardware agnostic.**
   `ternary-core` must compile on:
   - `x86_64-unknown-linux-gnu` (data centre, CUDA optional)
   - `aarch64-apple-darwin` (Apple Silicon, Metal optional)
   - `wasm32-unknown-unknown` (browser, edge)
   - `thumbv7em-none-eabihf` (embedded, `no_std`, future)

   This means: no `std::fs`, no `println!`, no `tokio` in `ternary-core`.
   Everything goes through `candle-core` tensors and pure arithmetic.

4. **Deterministic.**
   Given identical inputs, annealing fraction, and RNG seed, two forward
   passes must produce bit-identical output. This is critical for
   reproducible research and debugging gradient issues.

---

## 2. Tech Stack Rationale

| Dependency | Why |
|------------|-----|
| **candle-core** | Pure Rust tensor library. Compiles to CPU, CUDA, Metal, WASM without linking PyTorch/ONNX. Critical for edge deployment. |
| **candle-nn** | Provides `VarBuilder`, `VarMap`, `AdamW`, `RmsNorm`, `Embedding`, `rotary_emb` — saves us from reimplementing training infra. |
| **memmap2** | Zero-copy file mapping. Loading a 38 GB OpenWebText dataset must not allocate 38 GB of heap. `mmap` maps the file into virtual address space; the OS pages in/out on demand. |
| **rayon** | CPU-parallel data loading, batch preparation, and gradient accumulation. Thread pool is created once and reused. |
| **tokio** | Async runtime for the search coordinator. Allows the evaluator to yield while waiting for GPU kernels. Future: HTTP API for serving. |
| **serde + serde_json** | Config serialisation. Every config must round-trip through JSON without data loss. |
| **safetensors** | Industry-standard checkpoint format. Compatible with Hugging Face ecosystem. Supports memory-mapped loading. |
| **clap** (derive) | Type-safe CLI argument parsing. Derive macros eliminate boilerplate. |
| **tracing** | Structured logging with span-based context. Essential for debugging: "which layer's gradient exploded at step 3,417?" |
| **petgraph** | Graph data structure for the quantisation search space. Dijkstra's algorithm runs on it. |

### What we do NOT use

| Avoided | Reason |
|---------|--------|
| `tch-rs` (libtorch) | Links 2 GB C++ library. Cannot compile to WASM. Defeats the purpose. |
| `ndarray` | No autograd. No GPU support. Candle does everything ndarray does, plus backprop. |
| `polars` | Overkill for our data pipeline. We tokenise text, not query DataFrames. |

---

## 3. Workspace Architecture

### Dependency Graph

```
                 ternary-common
                   │       │
            ┌──────┘       └──────┐
            ▼                     ▼
      ternary-core          (standalone)
       │    │    │
       │    │    └──────────────┐
       ▼    ▼                   ▼
  ternary-train    ternary-search    ternary-infer
```

**Rules:**
- `ternary-core` depends only on `candle-*` and `serde`. No IO, no CLI.
- `ternary-common` depends on `candle-core`, `tokenizers`, `serde`, `memmap2`. No model logic.
- `ternary-train`, `ternary-search`, `ternary-infer` depend on both `core` and `common`.
- No circular dependencies. No cross-binary dependencies.

### Feature Flags

```toml
[features]
default = []             # Pure CPU — WASM compatible
cuda    = ["candle-core/cuda", "candle-nn/cuda"]
metal   = ["candle-core/metal", "candle-nn/metal"]
```

Features propagate: enabling `cuda` on `ternary-train` automatically enables
it on `ternary-core` via the dependency chain.

---

## 4. Component Specifications

### 4A. `crates/core` — The Mathematical Engine

This is the heart of the framework. Every mathematical operation lives here.

#### 4A.1 BitLinear Layer (the "Soft-to-Hard" flow)

The forward pass of a BitLinear layer has two regimes controlled by the global
annealing fraction `α(t) ∈ [0, 1]`:

**Soft regime (α < 1.0):**

```
W_q = tanh(α_eff · W_latent)      where α_eff = 1 + 7·α(t)
x_q = tanh(α_eff · x)
out = (x_q @ W_q^T) · γ           where γ = mean(|W_latent|) / √d_in
```

**Hard regime (α = 1.0):**

```
W_q = sign(W_latent)              via STE: ∂W_q/∂W_latent ≈ S·I
x_q = sign(x)                     (binary) or identity (ternary)
out = (x_q @ W_q^T) · γ
```

**Reference Rust implementation:**

```rust
/// Unified BitLinear forward pass with soft→hard annealing.
///
/// During training:  anneal ∈ [0, 1) → uses tanh(α·x) as smooth sign proxy
/// At convergence:   anneal = 1.0    → hard sign with STE
/// At inference:     uses pre-cached quantised weights (no re-quantise)
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // Fast path: inference with cached weights
    if let Some((w_q, scale)) = self.cache.lock().as_ref() {
        let out = matmul_reshape(x, &w_q.t()?)?;
        return out.affine(*scale, 0.0);
    }

    // Training path
    let w = self.weight.weight();
    let w_clamped = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
    let d_in = w_clamped.dim(1)?;
    let anneal = current_anneal_frac();

    let (w_q, x_q) = if anneal < 1.0 {
        // Soft regime: tanh approximation
        let alpha = 1.0 + 7.0 * anneal;  // α grows 1→8 over schedule
        (
            ste_tanh_scaled(&w_clamped, alpha, self.ste_scale)?,
            ste_tanh_scaled(x, alpha, self.ste_scale)?,
        )
    } else {
        // Hard regime: true sign with STE
        (
            ste_sign_scaled(&w_clamped, self.ste_scale)?,
            ste_sign_scaled(x, self.ste_scale)?,
        )
    };

    let out = matmul_reshape(&x_q, &w_q.t()?)?;
    let gamma = w_clamped.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
    let scale = gamma / (d_in as f64).sqrt();
    out.affine(scale, 0.0)
}
```

**Key insight — the STE residual trick:**

```rust
/// STE for sign: forward ≈ sign(x), backward = scale · ∂x
fn ste_sign_scaled(x: &Tensor, scale: f64) -> Result<Tensor> {
    let sign_x = x.sign()?;            // Non-differentiable
    let detach_x = x.detach();         // Cut the gradient here
    let residual = (x - &detach_x)?;   // = 0 in forward, carries grad in backward
    &sign_x + &residual.affine(scale, 0.0)?
    //  ^^^^^ this is the quantised value
    //         ^^^^^^^^^^^^^^^^^^^^^^^^^ this is zero in forward but
    //                                   ∂/∂x = scale in backward
}
```

This is the same trick originally from the legacy `src/binary.rs`.
In the workspace version, `RefCell<Option<...>>` caches have been replaced
with `parking_lot::Mutex<Option<...>>` to make all layers `Send + Sync`.

#### 4A.2 Ternary Quantisation (AbsMean)

For ternary {-1, 0, +1}, there are two strategies:

**Strategy 1: AbsMean (default)**
```
β = mean(|W|)
W_scaled = W / β
W_q = clamp(round(W_scaled), -1, +1)
```

**Strategy 2: Dynamic Threshold (recommended)**
```
β = mean(|W|)
δ = 0.7 × β
W_q[i] = { sign(W[i])  if |W[i]| > δ
          { 0           otherwise
```

Strategy 2 is preferred because it creates a clear "dead zone" around zero,
making the ternary distribution more balanced (fewer weights get stuck at ±1).

#### 4A.3 Activation Functions

**Current: ReLU²**
```rust
fn relu_squared(x: &Tensor) -> Result<Tensor> {
    x.relu()?.sqr()
}
```

**New: SwiGLU (to be added)**

SwiGLU splits the FFN intermediate dimension in half and uses one half as a
gate. This is the activation used in LLaMA, Mistral, and most modern LLMs:

```rust
/// SwiGLU Feed-Forward Network
///
/// Given input x ∈ R^{batch × seq × d_model}:
///   gate = x @ W_gate    ∈ R^{... × d_ff}
///   up   = x @ W_up      ∈ R^{... × d_ff}
///   out  = (SiLU(gate) ⊙ up) @ W_down
///
/// Note: d_ff is the *half* intermediate size. The effective parameter
/// count is 3 × d_model × d_ff (three projections instead of two).
pub struct SwiGLUFeedForward {
    w_gate: BitLinearLayer,  // d_model → d_ff
    w_up:   BitLinearLayer,  // d_model → d_ff
    w_down: BitLinearLayer,  // d_ff → d_model
}

impl SwiGLUFeedForward {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        let activated = (gate * up)?;
        self.w_down.forward(&activated)
    }
}
```

**Config change:** When `use_swiglu = true`, `intermediate_size` represents
`d_ff` (the half-size). The total parameter count becomes
`3 × hidden_size × intermediate_size` instead of `2 × hidden_size × intermediate_size`.

#### 4A.4 RMSNorm

Already implemented. The key property for 1-bit models: RMSNorm is
scale-invariant, which prevents the normalisation from fighting the
quantisation. LayerNorm subtracts the mean, which can shift weights
across the quantisation threshold unpredictably.

```rust
// RMSNorm(x) = x / RMS(x) × γ
// where RMS(x) = √(mean(x²) + ε)
//
// No mean subtraction → stable with ternary weights.
```

#### 4A.5 RoPE (Rotary Position Embeddings)

Already implemented using `candle_nn::rotary_emb::rope_i`. No changes needed
for the workspace migration.

---

### 4B. `crates/train` — The Stability Engine

#### 4B.1 Optimiser: AdamW with Latent Weight Clipping

The critical insight: standard AdamW with weight decay can *prevent* ternary
weights from flipping. If weight decay shrinks a latent weight toward zero,
and the quantisation threshold is at `δ = 0.7 × mean(|W|)`, the weight
gets trapped at zero forever.

**Solution:** Zero weight decay on latent weights (set `--weight-decay 0.0`),
and instead use **hard clamping** after every optimiser step:

```rust
/// Post-optimiser latent weight management.
///
/// After AdamW updates W_latent, clamp every parameter to [-C, +C]
/// where C = latent_clamp_max (default 1.5).
///
/// Why 1.5 and not 1.0?
/// - The ternary threshold δ ≈ 0.7 × mean(|W|) ≈ 0.3–0.5 for typical init.
/// - Clamping at 1.0 would prevent the distribution from spreading.
/// - Clamping at 1.5 allows weights to "overshoot" past 1.0 during
///   training but not diverge to infinity.
/// - Empirically, 1.5 gives the best balance of stability and flexibility.
fn clamp_latent_weights(vars: &[Var], clamp_max: f64) -> anyhow::Result<()> {
    for var in vars {
        let t = var.as_tensor();
        if t.dtype() == DType::F32 {
            let clamped = t.clamp(-clamp_max as f32, clamp_max as f32)?;
            var.set(&clamped)?;
        }
    }
    Ok(())
}
```

**Note:** The legacy `src/bin/train.rs` hard-coded `clamp(-1.2, 1.2)`.
This was fixed in the workspace migration: the trainer now uses
`config.latent_clamp_max` (default 1.5).

#### 4B.2 Scheduler: Cosine Annealing with Warmup

The scheduler has two independent schedules running simultaneously:

**Schedule 1: Learning Rate**
```
step < warmup_steps:
    lr(step) = lr_max × (step + 1) / warmup_steps

step ≥ warmup_steps:
    progress = (step - warmup) / (max_steps - warmup)
    lr(step) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × progress))
```

**Schedule 2: Quantisation Annealing (Soft→Hard)**
```
anneal_steps = 0.3 × max_steps       # First 30% of training
α(step) = clamp(step / anneal_steps, 0, 1)
```

These two schedules are **intentionally decoupled.** The LR schedule controls
the magnitude of weight updates. The annealing schedule controls the
sharpness of the quantisation function. They serve different purposes:

- Early training: Low α (soft quantisation) + high LR → large updates,
  smooth loss landscape, weights explore freely.
- Mid training: Rising α + decaying LR → quantisation sharpens,
  weights commit to their ternary values.
- Late training: α=1 (hard sign) + very low LR → fine-tuning within
  the quantised regime.

```rust
/// Quantisation annealing schedule.
///
/// Controls the global `ANNEAL_FRAC` atomic that all BitLinear layers read.
/// This is a global schedule because all layers must anneal in lockstep —
/// having different layers at different annealing stages creates gradient
/// scale mismatches that destabilise training.
pub struct AnnealSchedule {
    total_steps: usize,
    anneal_fraction: f32,  // What fraction of training to spend annealing (default 0.3)
}

impl AnnealSchedule {
    pub fn step(&self, global_step: usize) {
        let anneal_steps = (self.total_steps as f32 * self.anneal_fraction) as usize;
        let frac = if anneal_steps == 0 {
            1.0
        } else {
            (global_step as f32 / anneal_steps as f32).min(1.0)
        };
        set_quant_anneal_frac(frac);
    }
}
```

#### 4B.3 Trainer Loop Architecture

The trainer must cleanly separate three concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        Trainer                               │
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ Data Source  │───▶│ Compute Graph│───▶│ Optimiser Step │  │
│  │             │    │              │    │                │  │
│  │ TextDataset │    │  model.fwd() │    │  backward()    │  │
│  │ or Streaming│    │  loss()      │    │  clip_grads()  │  │
│  │             │    │              │    │  optimizer.step │  │
│  │             │    │              │    │  clamp_latent() │  │
│  └─────────────┘    └──────────────┘    └────────────────┘  │
│                                                               │
│  Hooks:                                                       │
│    on_step_end   → log loss, advance LR, advance annealing   │
│    on_eval       → compute val perplexity, log to CSV         │
│    on_checkpoint  → save safetensors + config.json             │
│    on_epoch_end  → log epoch summary                          │
└─────────────────────────────────────────────────────────────┘
```

**Gradient accumulation** (already implemented) is handled by running N
micro-batches through the compute graph, scaling each loss by `1/N`,
and calling `backward()` once on the sum. This is correct because:

```
∂(L₁/N + L₂/N + ... + Lₙ/N)/∂W = (1/N) × Σᵢ ∂Lᵢ/∂W
```

**Key change from current code:** The trainer should be a `struct Trainer`
with builder-pattern configuration, not a 600-line `main()` function.
This allows:
- Unit testing the training loop with mock data
- Embedding the trainer in a Jupyter-like notebook (via Python bindings later)
- Running multiple training jobs with different configs in the same process

```rust
pub struct Trainer {
    model: OneBitLlm,
    optimizer: AdamW,
    lr_scheduler: LrScheduler,
    anneal_schedule: AnnealSchedule,
    config: TrainerConfig,
    varmap: VarMap,
}

pub struct TrainerConfig {
    pub batch_size: usize,
    pub accumulation_steps: usize,
    pub max_steps: usize,
    pub max_epochs: usize,
    pub grad_clip_max_norm: f64,
    pub label_smoothing: f64,
    pub save_every: usize,
    pub eval_every: usize,
    pub output_dir: PathBuf,
}

impl Trainer {
    pub fn step(&mut self, batches: &[(Vec<u32>, Vec<u32>)]) -> Result<StepMetrics> {
        // 1. Forward + loss accumulation
        // 2. Backward
        // 3. Gradient clipping
        // 4. Optimizer step
        // 5. Latent weight clamping
        // 6. Advance schedules
        // Returns: StepMetrics { loss, grad_norm, lr, anneal_frac }
    }
}
```

#### 4B.4 Arenas Residual Annealing

Arenas adds a full-precision residual shortcut that bypasses the quantised
layers. This stabilises early training when quantised weights are essentially
random. The coefficient anneals to zero:

```
arenas_coef(t) = arenas_initial × max(0, 1 - t / arenas_anneal_steps)
```

In the forward pass of each decoder block:

```rust
// After attention sublayer:
x = residual + scale * attn_out;
if let Some(c) = arenas_coef {
    x = x + c * block_input;   // Full-precision shortcut
}
```

When `arenas_coef` reaches zero, the shortcut disappears and the model
relies entirely on quantised pathways. This is implemented in `crates/core/src/model.rs` (`DecoderBlock::forward`).

---

### 4C. `crates/search` — The Optimisation Engine

#### 4C.1 Problem Statement

Given a trained model with N layers, find the optimal assignment of each
layer to Binary (1-bit) or Ternary (1.58-bit) that minimises perplexity
while respecting a size budget.

This is a combinatorial optimisation problem with 2^N possible configurations.
For N=6 this is trivial (64 configs), but for N=32 or N=80 it's intractable.

#### 4C.2 Architecture: Coordinator-Evaluator Pattern

```
┌────────────────────────────────────────────────────┐
│                  SearchCoordinator                  │
│                                                      │
│  1. Build QuantGraph (nodes=configs, edges=transitions)
│  2. Expander decompose → partitions                  │
│  3. For each partition (SEQUENTIAL — one GPU):       │
│     ├── Run Dijkstra from start_node                 │
│     ├── For top-K reachable nodes:                   │
│     │   └── Evaluator.evaluate(config)               │
│     └── Track best (config, score) in partition      │
│  4. Merge partition results → global best            │
│  5. Final evaluation of global best                  │
│                                                      │
│  ⚠ Sequential search prevents GPU OOM!              │
│  ⚠ Each evaluate() loads full model into VRAM.      │
└────────────────────────────────────────────────────┘
```

#### 4C.3 Metrics

The search should track three metrics per configuration:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Perplexity** | exp(avg_loss) | Lower = better language modelling |
| **Size** | Σ(params_i × bits_i) / 8 MB | Model footprint |
| **Sensitivity** | (loss_ternary - loss_binary) per layer | How much a layer "cares" about precision |

Sensitivity is computed by perturbing one layer at a time: set layer i to
Binary while keeping everything else Ternary, measure the loss increase.
Layers with high sensitivity should stay Ternary; layers with low sensitivity
can be safely compressed to Binary.

#### 4C.4 Future: Async Coordinator with Tokio

The current coordinator is synchronous. For multi-GPU search, we need
an async coordinator that can dispatch evaluation tasks to different
GPU workers:

```rust
/// Future architecture (not yet implemented)
pub struct AsyncSearchCoordinator {
    workers: Vec<tokio::sync::mpsc::Sender<EvalRequest>>,
    results: tokio::sync::mpsc::Receiver<EvalResult>,
}
```

---

### 4D. `crates/inference` — The Deployment Engine

#### 4D.1 KV-Cache

The current inference runtime recomputes the full attention over all past
tokens for every new token. This is O(n^2) per generation. A future
KV-Cache implementation will bring this to O(n) per token:

```rust
/// Pre-allocated KV-Cache for a single attention layer.
///
/// Shape: (batch, num_heads, max_seq_len, head_dim)
///
/// The cache is a ring buffer: when the sequence exceeds max_seq_len,
/// we slide the window forward (or truncate the oldest entries).
pub struct KvCache {
    k: Tensor,   // (batch, heads, max_seq, head_dim)
    v: Tensor,   // (batch, heads, max_seq, head_dim)
    len: usize,  // Current number of cached positions
    max_len: usize,
}

impl KvCache {
    /// Create empty cache on the given device.
    pub fn new(
        batch: usize,
        heads: usize,
        max_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let k = Tensor::zeros((batch, heads, max_len, head_dim), DType::F32, device)?;
        let v = Tensor::zeros((batch, heads, max_len, head_dim), DType::F32, device)?;
        Ok(Self { k, v, len: 0, max_len })
    }

    /// Append new K, V for the current position(s).
    /// Returns the full K, V tensors up to current length.
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_len = new_k.dim(2)?;  // number of new positions

        // Write into the pre-allocated buffer at position [self.len .. self.len + new_len]
        // (Implementation uses narrow + copy or slice_assign)
        self.len += new_len;

        // Return view of cache up to current length
        let k_out = self.k.narrow(2, 0, self.len)?;
        let v_out = self.v.narrow(2, 0, self.len)?;
        Ok((k_out, v_out))
    }

    /// Reset the cache (new conversation).
    pub fn reset(&mut self) {
        self.len = 0;
    }
}
```

**Integration with attention:**

```rust
/// Modified attention forward for incremental decoding.
///
/// When kv_cache is Some, x contains only the NEW tokens.
/// The cache provides all previous K, V values.
pub fn forward_incremental(
    &self,
    x: &Tensor,                      // (batch, new_seq, hidden)
    kv_cache: &mut Option<KvCache>,
) -> Result<Tensor> {
    let (b, t, _) = x.dims3()?;

    // Project Q, K, V for the NEW tokens only
    let qkv = self.c_attn.forward(x)?;
    // ... reshape, split into q, k, v ...

    // Append to cache and get full history
    let (k_full, v_full) = if let Some(cache) = kv_cache {
        cache.append(&k, &v)?
    } else {
        (k.clone(), v.clone())
    };

    // Attention: Q (new tokens) attends to K_full, V_full (all tokens)
    let scores = q.matmul(&k_full.t()?)? * self.scale;
    // ... causal mask (offset by cache length) ...
    let att = softmax(&scores, D::Minus1)?;
    let y = att.matmul(&v_full)?;
    // ...
}
```

#### 4D.2 Decoding Strategies

The sampler is implemented in `crates/inference/src/sampler.rs` with
temperature, top-k, top-p, and repetition penalty. A future `Sampler` trait
could enable additional strategies:

```rust
pub trait Sampler: Send + Sync {
    fn sample(&self, logits: &Tensor, context: &[u32]) -> Result<u32>;
}

pub struct NucleusSampler {
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f64,
}

pub struct GreedySampler;

pub struct BeamSearchSampler {
    pub beam_width: usize,
    pub length_penalty: f64,
}
```

#### 4D.3 `.1bit` Binary Export Format

For deployment on microcontrollers and C-only environments, we need a
custom binary format that can be loaded with a single `fread()`:

```
┌─────────────────────────────────────────────────────────────┐
│                    .1bit File Format                         │
├─────────────────────────────────────────────────────────────┤
│ Magic:        "1BIT" (4 bytes, ASCII)                       │
│ Version:      u32 (little-endian)                           │
│ Config size:  u32 (bytes of JSON config that follows)       │
│ Config:       UTF-8 JSON (OneBitLlmConfig)                  │
│ Padding:      0-3 bytes to align to 4-byte boundary         │
│ Num tensors:  u32                                           │
│ For each tensor:                                            │
│   Name len:   u32                                           │
│   Name:       UTF-8 string                                  │
│   Dtype:      u8 (0=F32, 1=I8, 2=Packed2Bit)              │
│   Num dims:   u32                                           │
│   Dims:       [u32; num_dims]                               │
│   Data len:   u64 (bytes)                                   │
│   Data:       raw bytes (little-endian)                     │
│   Padding:    0-7 bytes to align to 8-byte boundary         │
└─────────────────────────────────────────────────────────────┘
```

**Packed 2-bit encoding for ternary weights:**

Each weight is 2 bits: `00` = 0, `01` = +1, `11` = -1 (sign bit + value bit).
Four weights per byte. A 1024×1024 weight matrix becomes 256 KB instead of 4 MB.

```rust
/// Pack ternary weights into 2-bit representation.
/// Input: F32 tensor with values in {-1.0, 0.0, 1.0}
/// Output: Vec<u8> where each byte holds 4 weights
pub fn pack_ternary_2bit(tensor: &Tensor) -> Result<Vec<u8>> {
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    let num_bytes = (flat.len() + 3) / 4;  // ceil division
    let mut packed = vec![0u8; num_bytes];

    for (i, &val) in flat.iter().enumerate() {
        let bits: u8 = match val as i8 {
            1  => 0b01,   // +1
            -1 => 0b11,   // -1
            _  => 0b00,   // 0
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        packed[byte_idx] |= bits << bit_offset;
    }

    Ok(packed)
}
```

**C header for loading:**

```c
// onebit.h — Load .1bit models in C (header-only)
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    char     magic[4];      // "1BIT"
    uint32_t version;
    uint32_t config_size;
    // followed by config JSON, then tensor data
} OnebitHeader;

static inline int8_t unpack_ternary(const uint8_t* packed, size_t index) {
    uint8_t byte = packed[index / 4];
    uint8_t bits = (byte >> ((index % 4) * 2)) & 0x03;
    if (bits == 0x01) return  1;
    if (bits == 0x03) return -1;
    return 0;
}
```

---

### 4E. `crates/common` — Shared Primitives

#### 4E.1 Config

Implemented in `crates/common/src/config.rs`. New fields added in v0.2:

```rust
/// New fields for v0.2.0
pub struct OneBitLlmConfig {
    // ... existing fields ...

    /// Use SwiGLU activation instead of ReLU²/SiLU.
    /// When true, intermediate_size is the half-size (gate and up share it).
    #[serde(default)]
    pub use_swiglu: bool,

    /// Quantisation annealing: fraction of training steps spent in soft regime.
    /// Default 0.3 (30% of training is soft, 70% is hard).
    #[serde(default = "default_anneal_fraction")]
    pub anneal_fraction: f32,
}
```

#### 4E.2 Data Pipeline: MmapDataset

For datasets larger than RAM, `memmap2` provides zero-copy access:

```rust
use memmap2::Mmap;

/// Memory-mapped dataset: pre-tokenised binary file.
///
/// File format: flat array of u32 token IDs, little-endian.
/// Created by a one-time preprocessing step:
///   tokenise(text) → Vec<u32> → write_to_file()
///
/// At training time, the OS pages in only the data we actually read.
/// A 38 GB dataset uses ~0 MB of heap.
pub struct MmapDataset {
    mmap: Mmap,
    len: usize,   // number of u32 tokens
    seq_len: usize,
}

impl MmapDataset {
    pub fn open(path: &Path, seq_len: usize) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let len = mmap.len() / 4;  // u32 = 4 bytes
        Ok(Self { mmap, len, seq_len })
    }

    /// Get a contiguous slice of tokens starting at byte offset.
    pub fn get_tokens(&self, start: usize, count: usize) -> &[u32] {
        let byte_start = start * 4;
        let byte_end = byte_start + count * 4;
        let bytes = &self.mmap[byte_start..byte_end];
        // SAFETY: mmap is aligned, u32 is 4-byte aligned, we checked bounds
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, count) }
    }
}
```

---

## 5. Migration Plan

The migration from monolith to workspace is done in phases:

### Phase 1: Workspace Shell -- DONE
- [x] Create `Cargo.toml` workspace root
- [x] Create all five crate directories with stub `lib.rs`
- [x] Create stub binaries
- [x] Verify `cargo check` passes for the workspace

### Phase 2: Migrate `common` + `core` -- DONE
- [x] `crates/common/src/config.rs` — `OneBitLlmConfig` with `use_swiglu`, `anneal_fraction`
- [x] `crates/common/src/data.rs` — `TextDataset`, `StreamingBatchIter`, `batch_to_tensors`
- [x] `crates/core/src/quantize.rs` — STE primitives, global annealing atomic, ternary/binary quantise
- [x] `crates/core/src/linear.rs` — `BinaryLinear`, `TernaryLinear`, `BitLinearLayer` with `parking_lot::Mutex`
- [x] `crates/core/src/norm.rs` — `NormLayer` (RMSNorm / LayerNorm)
- [x] `crates/core/src/activation.rs` — `FeedForward`, `SwiGLUFeedForward`, `FfnLayer`
- [x] `crates/core/src/attention.rs` — `CausalSelfAttention` with RoPE and QK-Norm
- [x] `crates/core/src/model.rs` — `OneBitLlm`, `CompressionStats`, weight tying
- [x] Unit tests for quantisation (round-trip, distributions, annealing)

### Phase 3: Migrate `train` -- DONE
- [x] `crates/train/src/scheduler.rs` — `LrScheduler` (warmup + cosine/linear/constant) + `AnnealSchedule`
- [x] `crates/train/src/trainer.rs` — `Trainer` struct with AdamW, gradient accumulation, label smoothing, latent clamping
- [x] `onebit-train` CLI binary with full argument parsing
- [x] Latent clamp uses `config.latent_clamp_max` (not hard-coded)

### Phase 4: Migrate `search` -- DONE
- [x] `crates/search/src/types.rs` — `QuantLevel`, `QuantConfig`, `SearchResult`, `SearchConfig`
- [x] `crates/search/src/graph.rs` — `GraphBuilder` with BFS, cost estimation
- [x] `crates/search/src/expander.rs` — `ExpanderDecomposer` with configurable partitions
- [x] `crates/search/src/evaluator.rs` — `ConfigEvaluator` with LRU cache
- [x] `crates/search/src/coordinator.rs` — `SearchCoordinator` with Dijkstra
- [x] `onebit-search` and `onebit-eval-config` CLI binaries

### Phase 5: Migrate `inference` -- DONE
- [x] `crates/inference/src/sampler.rs` — top-k, top-p, temperature, repetition penalty
- [x] `crates/inference/src/runtime.rs` — `InferenceRuntime` (load, generate, chat loop)
- [x] `crates/inference/src/export.rs` — `.1bit` binary format with ternary/binary packing + C header
- [x] `onebit-chat`, `onebit-export`, `onebit-test-generate` CLI binaries

### Phase 6: End-to-End Verification -- DONE
- [x] Trained binary model (64-dim, 2-layer) for 5K steps on GPU — loss 62.3 -> 3.45
- [x] Trained ternary model (128-dim, 4-layer) for 10K steps on GPU — loss 128.8 -> 3.66
- [x] Inference generates recognisable fragments from training data
- [x] Full test suite: 18 tests passing, zero warnings

### Phase 7: Future Work
- [ ] KV-Cache for O(1) per-token decoding
- [ ] `MmapDataset` in `ternary-common`
- [ ] `tokio`-based async search coordinator
- [ ] WASM build target with example web page
- [ ] Hugging Face model hub integration
- [ ] Speculative decoding with tiny draft model

---

## 6. Testing Strategy

### Unit Tests

Every mathematical function must have tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    #[test]
    fn ternary_quantise_round_trip() {
        let device = Device::Cpu;
        let w = Tensor::new(&[-0.8f32, -0.1, 0.0, 0.1, 0.9], &device).unwrap();
        let q = ternary_quantize_forward(&w, true).unwrap();
        let vals: Vec<f32> = q.to_vec1().unwrap();
        assert_eq!(vals, vec![-1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn ste_gradient_flows() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.5f32, -0.3, 0.0], &device).unwrap();
        // After STE sign, forward = [1, -1, 0-ish]
        // But backward should have non-zero gradients
        let y = ste_sign_scaled(&x, 2.0).unwrap();
        let loss = y.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        let grad_x = grads.get(&x).unwrap();
        // Gradient should be 2.0 * scale = 2.0 * 2.0 for each element
        let g: Vec<f32> = grad_x.to_vec1().unwrap();
        for &g_i in &g {
            assert!(g_i.abs() > 0.0, "STE gradient must be non-zero");
        }
    }

    #[test]
    fn config_json_round_trip() {
        let config = OneBitLlmConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let loaded: OneBitLlmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.vocab_size, loaded.vocab_size);
        assert_eq!(config.hidden_size, loaded.hidden_size);
    }
}
```

### Integration Tests

```rust
// tests/train_smoke.rs (in workspace root)
#[test]
fn train_one_step_does_not_panic() {
    // Create tiny config (vocab=100, hidden=32, layers=2)
    // Create random "dataset" of 100 tokens
    // Run one training step
    // Assert loss is finite and non-NaN
}
```

### Property-Based Tests

For numerical stability, use `proptest` or `quickcheck`:

```rust
proptest! {
    #[test]
    fn ternary_weights_always_in_range(w in prop::collection::vec(-2.0f32..2.0, 1..1000)) {
        let t = Tensor::from_vec(w, &Device::Cpu)?;
        let q = ternary_quantize_forward(&t, true)?;
        let vals: Vec<f32> = q.to_vec1()?;
        for v in vals {
            prop_assert!(v == -1.0 || v == 0.0 || v == 1.0);
        }
    }
}
```

---

## 7. Future Work

### 7.1 `no_std` Core

`ternary-core` should eventually compile with `#![no_std]` for embedded
deployment. This requires:
- Replacing `Vec` with `heapless::Vec` or fixed-size arrays
- No `std::sync::atomic` (use `core::sync::atomic`)
- No `println!` / `eprintln!` (use `defmt` or silent mode)
- candle-core `no_std` support (requires upstream work)

### 7.2 WASM Target

```bash
cargo build --target wasm32-unknown-unknown -p ternary-infer --no-default-features
```

This should produce a .wasm file that can be loaded in a browser.
Combined with the `.1bit` export format, this enables "download model and
run inference entirely in the browser" — no server needed.

### 7.3 Mixture of Quantisation (MoQ)

Instead of assigning a single bit-width per layer, allow different bit-widths
for attention vs FFN within the same layer:

```rust
pub struct LayerQuantConfig {
    pub attn_qkv: QuantLevel,     // Q, K, V projection
    pub attn_proj: QuantLevel,    // Output projection
    pub ffn_gate: QuantLevel,     // SwiGLU gate
    pub ffn_up: QuantLevel,       // SwiGLU up
    pub ffn_down: QuantLevel,     // SwiGLU down
}
```

This expands the search space from 2^N to 2^(5N) but could find much better
Pareto-optimal configurations.

### 7.4 Speculative Decoding

Use a tiny 1-bit "draft" model to propose K tokens, then verify with a
larger model in parallel. Since 1-bit models are extremely fast for
inference, this could provide 3-5× speedup on the verification model.

---

*This document is the single source of truth for the OneBit-LLM architecture.
Update it when designs change. If the code and this document disagree,
the code is wrong.*
