#!/usr/bin/env bash
# WikiText-2 training with Sandwich config and aggressive LR (from Turbo Overfit).
# Config: F16 Embed/Head, Ternary middle. max_seq_len=256 (no gradient checkpointing in codebase).
# Use batch_size 8 on 16 GB VRAM; reduce to 4 if OOM.

set -e
cd "$(dirname "$0")/.."

# Sandwich = F16 embed/head, Ternary middle (config_wikitext_sandwich.json)
CONFIG="${1:-config_wikitext_sandwich.json}"
DATA_DIR="${2:-data/wikitext-2}"
TOKENIZER="${3:-data/tokenizer.json}"
OUT_DIR="${4:-checkpoints/wikitext2}"
# Aggressive LR from successful overfit (1e-2); reduce to 5e-3 if training is unstable.
LR="${5:-1e-2}"
BATCH="${6:-8}"
ACCUM="${7:-4}"

echo "Config: $CONFIG | Data: $DATA_DIR | LR: $LR | batch=$BATCH accum=$ACCUM | out: $OUT_DIR"

cargo run --release -p ternary-train --bin onebit-train --features cuda -- \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUT_DIR" \
  --lr "$LR" \
  --lr-warmup-steps 200 \
  --lr-min 1e-5 \
  --lr-decay cosine \
  --batch-size "$BATCH" \
  --accumulation-steps "$ACCUM" \
  --max-steps 20000 \
  --save-every 2000 \
  --log-every 100 \
  --grad-clip-max-norm 1.0 \
  --label-smoothing 0.1 \
  "$@"
