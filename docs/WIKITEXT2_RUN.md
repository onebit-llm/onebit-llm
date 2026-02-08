# WikiText-2 Training Run

## When to run

Run the **Turbo Overfit** sanity check first (500 steps on `data/sanity.txt`, LR 1e-2, batch 4). If loss drops **below 1.0** by step 500, use the same aggressive LR for WikiText-2.

## Strategy

- **Config:** Sandwich (F16 embed/head, Ternary middle) — use `config_wikitext_sandwich.json`.
- **LR:** From overfit (e.g. **1e-2**); warmup 200 steps, cosine decay, min 1e-5. Reduce to 5e-3 if training is unstable.
- **Sequence length:** **max_seq_len 256** (no gradient checkpointing in codebase; 256 is enough for sentence structure).
- **Batch:** On ~16 GB VRAM use **batch_size 8**, **accumulation_steps 4**. If OOM, try batch 4.

## Quick run

```bash
./scripts/run_wikitext2.sh
```

Defaults: `config_wikitext_sandwich.json`, `data/wikitext-2`, LR 1e-2, batch 8, accum 4, out `checkpoints/wikitext2`.

Override args (config, data_dir, tokenizer, output_dir, lr, batch, accum):

```bash
./scripts/run_wikitext2.sh config_wikitext_sandwich.json data/wikitext-2 data/tokenizer.json checkpoints/wikitext2 1e-2 8 4
```

With less VRAM:

```bash
./scripts/run_wikitext2.sh config_wikitext_sandwich.json data/wikitext-2 data/tokenizer.json checkpoints/wikitext2 1e-2 4 4
```

## Data

- Tokenized WikiText-2 in `data/wikitext-2` (from `onebit-tokenize`).
- Tokenizer: `data/tokenizer.json`.

## Gradient checkpointing

Not implemented. To fit in VRAM we keep **max_seq_len 256** and adjust batch size (8 or 4).
