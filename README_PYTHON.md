# OneBit LLM — Python rewrite

PyTorch reimplementation with **gradient checkpointing** so training runs on smaller GPUs.

## Setup

```bash
pip install -e .
# or: pip install torch transformers huggingface-hub pyarrow pandas tqdm python-dotenv safetensors
```

## Stack-stream training

Same workflow as the Rust CLI: list shards from Hugging Face, download, convert parquet→jsonl, train, evict by disk budget.

```bash
# .env: HF_ACCESS_TOKEN=...
python scripts/stack_stream.py \
  --local-root /media/anil/DATA/stack_shards \
  --output-dir ./checkpoints_stack_stream_0_3B \
  --dataset-id bigcode/the-stack \
  --disk-budget-gb 900 \
  --model-config config_0_3B_stack.json \
  --tokenizer data/tokenizer.json \
  --batch-size 1 \
  --accumulation-steps 1 \
  --gradient-checkpointing-segments 8 \
  --save-every 500
```

Gradient checkpointing uses `torch.utils.checkpoint` (recompute segments in backward), so it works correctly and reduces VRAM.

## Layout

- `onebit_llm/config.py` — JSON config (compatible with Rust)
- `onebit_llm/quantize.py` — STE, ternary/binary quant
- `onebit_llm/linear.py` — BitLinear, BitLinearLayer
- `onebit_llm/norm.py` — RMSNorm
- `onebit_llm/attention.py` — Causal self-attention, RoPE, QK-norm
- `onebit_llm/model.py` — DecoderBlock, OneBitLlm (with checkpointing)
- `onebit_llm/train.py` — Trainer, LR scheduler, quant annealing
- `onebit_llm/data.py` — Streaming batches from jsonl
- `onebit_llm/stack_stream.py` — HF list/download, parquet→jsonl, evict, train loop
- `scripts/stack_stream.py` — CLI entrypoint
