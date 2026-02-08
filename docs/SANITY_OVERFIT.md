# Overfit Sanity Check (Sandwich Architecture)

## Purpose

Before scaling to large datasets, we verify that the **Sandwich architecture** (F16 embedding/lm_head, Ternary middle) is **mathematically capable of learning**. If the model cannot memorize a tiny dataset, it will never learn a language.

## Setup

- **Data:** `data/sanity.txt` — 50 lines of repeated text (one short paragraph).
- **Config:** Same as WikiText run: `config_wikitext.json` (8 layers, 512 hidden, max_seq_len 256, Sandwich: layer_bit_map unset → embedding/lm_head F16, middle Ternary/Binary).
- **Run:** 500 steps, batch_size 2, accumulation_steps 1, **log_every 1**, **debug_every 1** to print every step:
  - **grad_norm** — total L2 norm of gradients (detect vanishing → 0 or exploding → NaN).
  - **w0 / w1** — mean and std of first two parameters (boundary: embedding vs first hidden).

## How to run

```bash
# CPU (avoids GPU OOM on small runs)
cargo run --release -p ternary-train --bin onebit-train -- \
  --config config_wikitext.json \
  --data-dir data/sanity.txt \
  --tokenizer data/tokenizer.json \
  --output-dir checkpoints/sanity \
  --max-steps 500 \
  --batch-size 2 --accumulation-steps 1 \
  --log-every 1 --debug-every 1 \
  --save-every 500
```

Optional: `RUST_LOG=trace` for full tracing.

## Interpretation

| Observation | Meaning |
|-------------|--------|
| **grad_norm ≈ 0** | Sandwich connection broken (detached graph); gradients not flowing from loss to F16/ Ternary layers. |
| **grad_norm huge or NaN** | Scaling (e.g. γ) missing or unstable; need gradient clipping / smaller LR. |
| **grad_norm stable, non-zero (~4–5)** | Gradients are flowing; **Sandwich is not broken**. |
| **Loss decreasing** | Model is learning; no software bug at the boundary. |
| **Loss < 0.5** | Overfit achieved; architecture can memorize; full-dataset issue is likely LR/data size. |

## Observed results (run on CPU, 500 steps requested)

- **grad_norm:** Stable ~4.95 at step 0, slowly decreasing to ~4.5 by step 150. **Not vanishing, not exploding.** The Sandwich connection is **intact**.
- **w0, w1:** Reported as mean=1.0, std=0.0. These likely correspond to **norm (e.g. RMSNorm) scale parameters** (initialized to 1.0), not the embedding or first linear weight. To inspect the true F16 ↔ Ternary boundary, future work can log parameters by name or by shape (e.g. first 2D weight matrix).
- **Loss:** Decreased from ~512 to ~270 by step 150 (run was interrupted; full 500 steps would show whether loss reaches < 0.5). **The model is learning.**

## Final loss in logs

When training stops (e.g. `--max-steps 500`), the CLI prints a **final** line so the last step’s loss is always visible:

```text
step 500 epoch 249 loss 9.XXXX (final)
Training done. Saved to checkpoints/sanity/model.safetensors
```

If you only see periodic `step N ... loss` lines (e.g. every 25 steps), the **last** step is in this final line.

## Turbo overfit (500 steps, LR 1e-2, batch 4)

- **Observed:** Loss 521 → 27 by step 150, then 9.58 by step 475. **Final step 500 loss ~9.4–9.5** (not &lt; 1.0).
- **To reach loss &lt; 1.0:** Run more steps (e.g. 1000–2000) with the same setup, or try a higher LR. When final loss &lt; 1.0, treat that LR as “aggressive” for WikiText-2 (see `docs/WIKITEXT2_RUN.md`).

## Conclusion

- **No detached graph:** Gradients are flowing (grad_norm healthy).
- **No explosion:** No NaN; scaling/clipping appear adequate.
- **Learning:** Loss trend is downward; the Sandwich setup is capable of learning.
- **Next steps:** Run full 500 steps (or more) to see if loss < 0.5; if it does, the high loss on the full dataset is likely **hyperparameter/data** (e.g. lower LR, more data, longer training), not a fundamental architecture bug.
