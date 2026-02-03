# Architecture: Candle, Quantization, and Model

Technical reference for Candle usage, binary/ternary quantization, STE, and model layout. English only.

---

## 1. Candle usage

| Layer | Usage |
|-------|--------|
| **Tensor ops** | All compute via `candle::Tensor`: `matmul`, `reshape`, `permute`, `affine`, `clamp`, `sign`, `abs`, `relu`, `sqr`, `softmax`, RoPE cos/sin, causal mask, etc. |
| **Autograd** | Training: `loss.backward()`; STE implemented as `detach()` + residual so gradients flow to latent weights. |
| **Model definition** | Full Candle: `candle_nn::Module`, `Linear`, `Embedding`, `RmsNorm`, `LayerNorm`, `VarBuilder`, custom `BinaryLinear` / `TernaryLinear`. |
| **Inference & training** | Both use Candle: `model.forward()`, training loop with AdamW and GradStore. |

**Features used:** `candle-core` (tensor, Device, DType, Var, GradStore) and `candle-nn` (Linear, Embedding, RmsNorm, LayerNorm, AdamW, loss, ops, rotary_emb, sampling). **Not used:** `candle_transformers`; transformer blocks are custom in `model.rs`. **Custom kernels:** none; STE and quantize are plain Rust + Tensor API.

**Backend:** `Device::cuda_if_available(0)` (runtime GPU/CPU). CUDA enabled in `Cargo.toml`. No Metal.

---

## 2. Quantization implementation

- **Weights:** Stored as F32 latent in `VarMap`; no separate quantized tensor on disk.
- **Forward:** Latent → clamp → quantize (with or without STE) → F32 tensor with values {-1,0,+1} or ±1 → matmul → scale (γ).
- **Dtype:** Everything stays F32; values are discrete. No i8/i2 dtype or conversion.

**Ternary modes (config `use_dynamic_threshold`):**

- **Dynamic (true):** β = mean(\|W\|), δ = 0.7×β; \|w\| ≤ δ → 0, else sign(w).
- **AbsMean (false):** β = mean(\|W\|), w_scaled = w/β, round, clamp(-1, 1). No fixed global δ.

Both use a per-layer, per-forward statistic (β).

---

## 3. STE (Straight-Through Estimator)

- **Formula:** `out = quantized.detach() + (x - x.detach()).affine(ste_scale, 0.0)`. Forward looks quantized; backward passes gradient through the second term (identity × scale) to the latent.
- **Implementation:** In `binary.rs`: `ste_sign_scaled`, `ternary_delta_quantize` (with `apply_ste`), `ternary_absmean_ste`. No custom backward hooks; Candle autograd only.
- **Reference:** Standard STE (e.g. Bengio et al., 2013); in quantization, “straight-through” for round/sign.

---

## 4. Model state and checkpoint

- **VarMap / VarBuilder:** Single `VarMap`; `VarBuilder::from_varmap(..., DType::F32, &device)`; model built with `OneBitLlm::new(vb, &config)`; all parameters registered via `vb.pp("...")`.
- **Checkpoint:** `varmap.save(path)` / `varmap.load(path)` (safetensors via Candle). Only latent F32 weights are saved.
- **Inference cache:** Optional in-memory cache of (quantized weight, scale) per bit-linear layer, filled by `cache_quantized_weights()` so repeated forwards skip re-quantize.

---

## 5. Model architecture

- **Style:** Decoder-only, GPT-like; **not** using `candle_transformers::models`.
- **Components:** `OneBitLlm`: `wte` (embedding), `blocks: Vec<DecoderBlock>`, `ln_f`. **Weight tying:** output logits use `wte` weight (no separate `lm_head`). Each `DecoderBlock`: pre-norm attention + residual, pre-norm FFN + residual; optional Arenas (`block_input * c`), residual scaling (1/√2).
- **Attention:** `CausalSelfAttention`: QKV and output projection are `BitLinearLayer` (binary or ternary); RoPE; optional QK-norm (RmsNorm on Q, K); causal mask; softmax.
- **FFN:** Two `BitLinearLayer`s with ReLU² or SiLU.
- **Quantized layers:** Only the bit linears: attention `c_attn`, `c_proj`; FFN `c_fc`, `c_proj`. Embedding and all norms are F32.

---

## 6. Arenas (Annealing Residual Synapse)

- **Formula:** After attention and after FFN: `x = (residual + sublayer_out) + c × block_input`. The extra term is **block input** scaled by `c`, not the sublayer output.
- **Schedule:** `c` starts at `arenas_initial` and anneals linearly to 0 over `arenas_anneal_steps`. At inference, `c = 0` (no extra cost).
- **Purpose:** Full-precision residual path to mitigate “weight trapping” / gradient homogenization (see REFERENCE.md).

---

## 7. Code layout

```
onebit-llm/
├── src/
│   ├── lib.rs       # config, binary, model, data re-export; CompressionStats
│   ├── config.rs    # OneBitLlmConfig (JSON serde)
│   ├── binary.rs    # BinaryLinear, TernaryLinear, STE, ternary_*; optional cache
│   ├── model.rs     # BitLinearLayer, CausalSelfAttention, FeedForward, NormLayer,
│   │                 # DecoderBlock, OneBitLlm; cache API; CompressionStats
│   ├── data.rs      # TextDataset, StreamingBatchIter, batch_to_tensors
│   └── bin/
│       ├── train.rs # Training CLI (accumulation, validation, LrScheduler, compression log)
│       ├── export.rs
│       └── run.rs   # Inference (cache, benchmark, eval-perplexity)
└── docs/
    ├── REFERENCE.md      # Paper/framework references
    ├── ARCHITECTURE.md   # This file
    └── IMPLEMENTATION.md # Done / to-do
```

---

## 8. Training step (pseudo)

```text
# With gradient accumulation (streaming): collect N batches
total_loss = average(loss_1, ..., loss_N)
grads = total_loss.backward()
clip_grad_norm(grads, vars, max_norm)
optimizer.step(grads)
lr_scheduler.advance()
# Validation every eval_every steps: forward on val_data_dir, log perplexity
```

Quantization is not a separate update step; latent weights are updated by the optimizer, and the next forward re-quantizes from the updated latent (or uses cache if set for inference).
