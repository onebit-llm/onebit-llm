# Implementation Status

What has been implemented and what is planned. All in English.

---

## Done

### Training (`train` binary)

| Feature | Description |
|--------|-------------|
| **Gradient accumulation** | `--accumulation-steps N` (default 1). In streaming mode: collect N batches, average loss, single backward + step. In non-streaming: same (peekable iterator, take N batches per step). Effective batch = batch_size × N. |
| **LR scheduler** | `LrScheduler` struct: warmup (linear 0→lr) then cosine/linear decay via `--lr-decay` (default cosine). `current_lr()`, `advance()`. |
| **Validation perplexity** | `--val-data-dir <path>`: validation dataset. `--eval-every N` (default 500): run validation every N steps. `--eval-batches N` (default 50): max batches per eval. Logs `[eval] step X val_loss=... perplexity=...`. When validation is enabled, **metrics.csv** is written to output dir with columns `step,val_loss,perplexity`. |
| **Compression metrics** | At training start: log `total_params`, `quantized_params`, `effective_bits_per_param`, `compression_ratio_vs_f32`. `OneBitLlmConfig::compression_stats()` and `CompressionStats` in `model.rs`. |

### Inference cache (static quantize at load time)

| Feature | Description |
|--------|-------------|
| **Cached quantized weights** | `BinaryLinear` / `TernaryLinear`: optional `RefCell<Option<(Tensor, f64)>>` cache (quantized weight + scale). `cache_quantized()` fills it once; `forward()` uses cache when set (no re-quantize per forward). |
| **Model API** | `OneBitLlm::cache_quantized_weights()`, `clear_quantized_cache()`. |

### Run binary

| Feature | Description |
|--------|-------------|
| **`--use-cached-quantized`** | After loading weights, call `model.cache_quantized_weights()` so subsequent forwards use cached quantized weights (faster inference). |
| **`--benchmark N`** | Run N forward passes (after 3 warmup), report elapsed time and throughput (forwards/sec, ms/forward). |
| **`--eval-perplexity <path>`** | Evaluate cross-entropy loss and perplexity on a text file (same format as training data). Requires tokenizer. |

### Model (weight tying)

| Feature | Description |
|--------|-------------|
| **Embedding weight tying** | `wte` (token embedding) and output projection share the same weights; `lm_head` is removed. Forward uses `hidden.matmul(wte.embeddings().t())`. Checkpoint size ~280 MB → ~180 MB; params 70M → 45M. |

### Export

| Feature | Description |
|--------|-------------|
| **export** | Copy checkpoint + config to output dir. Tip: use `run --model-dir <output_dir> --use-cached-quantized`. |
| **export_quantized** | `export_quantized` binary: load F32 checkpoint, quantize c_attn/c_proj/c_fc to ternary {-1,0,+1} (F32), keep embeddings and norms; save to new safetensors. |

---

## To Do (roadmap)

### Phase 1: Training (optional refinements)

- [ ] Optional: use Candle built-in LR scheduler if available (current manual schedule is sufficient).
- [ ] Ablation: label smoothing 0 vs 0.1 (document or script).
- [ ] Larger dataset runs (e.g. OpenWebText subset) and document results.

### Phase 2: Inference optimization

- [x] **Static quantization export**: `export_quantized` binary saves pre-quantized ternary layers (F32 tensors -1/0/+1). Runtime cache (`--use-cached-quantized`) still available for F32 checkpoints.
- [ ] **i8 / packed matmul**: custom CUDA or Candle extension for int8/packed ternary matmul (potential 2–3× speedup).
- [ ] **LUT-based inference**: as in REFERENCE.md (Sherry/bitnet.cpp style); eliminate matmul in favor of lookups. Large engineering effort.

### Phase 3: Model scaling

- [ ] Scale to 355M / 774M (GPT-2 medium/large).
- [ ] Multi-GPU training (Candle distributed, if available).
- [ ] Mixed precision: F16 activations + ternary weights (if Candle supports).

### Phase 4: Publication and release

- [ ] Documentation: architecture deep-dive, usage, tuning guide.
- [ ] Benchmarks: perplexity on WikiText-103 / PTB; optional downstream (HellaSwag, etc.).
- [ ] Crates.io release (`onebit-llm`).
- [ ] Hugging Face integration: safetensors + config format already compatible; add model card / example.

---

## File changes (summary)

| File | Changes |
|------|--------|
| `src/binary.rs` | `RefCell` cache, `cache_quantized()`, `clear_cache()`, cache path in `forward()`. |
| `src/model.rs` | Weight tying (no lm_head; logits via wte.embeddings().t()). Cache API; `CompressionStats`; `compression_stats()` excludes lm_head. |
| `src/lib.rs` | Re-export `CompressionStats`, `ternary_quantize_forward`. |
| `src/binary.rs` | `ternary_quantize_forward` (pub) for export. |
| `src/bin/train.rs` | `LrScheduler`, gradient accumulation (streaming + non-streaming), metrics.csv when val set present, validation perplexity. |
| `src/bin/run.rs` | `--use-cached-quantized`, `--benchmark`, `--eval-perplexity`. |
| `src/bin/export.rs` | Print inference tip. |
| `src/bin/export_quantized.rs` | Load F32 checkpoint, quantize bit layers, save safetensors. |
