# Implementation Status

What has been implemented and what is planned. All in English.

---

## Done

### Training (`train` binary)

| Feature | Description |
|--------|-------------|
| **Gradient accumulation** | `--accumulation-steps N` (default 1). In streaming mode: collect N batches, average loss, single backward + step. Effective batch = batch_size × N. |
| **LR scheduler** | `LrScheduler` struct: warmup (linear 0→lr) then cosine/linear decay via `--lr-decay` (default cosine). `current_lr()`, `advance()`. |
| **Validation perplexity** | `--val-data-dir <path>`: validation dataset. `--eval-every N` (default 500): run validation every N steps. `--eval-batches N` (default 50): max batches per eval. Logs `[eval] step X val_loss=... perplexity=...`. |
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

### Export

| Feature | Description |
|--------|-------------|
| **Tip** | After export, print: use `run --model-dir <output_dir> --use-cached-quantized` for faster inference. |

---

## To Do (roadmap)

### Phase 1: Training (optional refinements)

- [ ] Optional: use Candle built-in LR scheduler if available (current manual schedule is sufficient).
- [ ] Ablation: label smoothing 0 vs 0.1 (document or script).
- [ ] Larger dataset runs (e.g. OpenWebText subset) and document results.

### Phase 2: Inference optimization

- [ ] **Static quantization export**: optional export path that saves pre-quantized weights only (e.g. F32 tensors with values -1/0/+1) so inference can load without latent weights. Current design uses runtime cache instead.
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
| `src/model.rs` | Cache API on bit layers and blocks; `CompressionStats` and `OneBitLlmConfig::compression_stats()`. |
| `src/lib.rs` | Re-export `CompressionStats`. |
| `src/bin/train.rs` | `LrScheduler`, gradient accumulation (streaming), compression log, `--val-data-dir`, `--eval-every`, `--eval-batches`, validation perplexity; non-streaming: `lr_scheduler` and validation. |
| `src/bin/run.rs` | `--use-cached-quantized`, `--benchmark`, `--eval-perplexity`. |
| `src/bin/export.rs` | Print inference tip. |
