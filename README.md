# OneBit LLM

PyTorch implementation of a 1-bit / ternary decoder-only language model (BitNet-style). Training uses **gradient checkpointing** so it runs on smaller GPUs.

## Features

- **Ternary and binary linear layers** with straight-through estimators (STE)
- **Sandwich layout**: embedding and LM head in full precision; hidden layers quantized
- **Stack-stream training**: stream shards from Hugging Face (e.g. The Stack), download, convert parquet→jsonl, train, evict by disk budget—no separate Python data pipeline
- **Gradient checkpointing** via `torch.utils.checkpoint` to reduce VRAM

## Setup

**With CUDA 13.0** (install PyTorch from the CUDA index first, then the project):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

**CPU only:**

```bash
pip install -e .
```

Optional: install remaining dependencies from `requirements-cuda.txt` after installing PyTorch with the cu130 index.

## Stack-stream training

List dataset shards from Hugging Face, download on demand, convert parquet to jsonl, train, and evict by disk budget. Set `HF_ACCESS_TOKEN` or `HF_TOKEN` in the environment (or in `.env`) for private datasets.

```bash
python scripts/stack_stream.py \
  --local-root /path/to/stack_shards \
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

- `--gradient-checkpointing-segments 8`: recompute 8 blocks per segment in the backward pass (saves VRAM).
- `--save-every 500`: write a checkpoint every 500 steps so a crash loses at most 500 steps.

## Project layout

| Path | Description |
|------|-------------|
| `onebit_llm/` | Python package |
| `onebit_llm/config.py` | JSON model config (vocab, layers, quant options) |
| `onebit_llm/quantize.py` | STE, ternary/binary quantization, annealing |
| `onebit_llm/linear.py` | BitLinear, BitLinearLayer (f16 / ternary / binary) |
| `onebit_llm/norm.py` | RMSNorm |
| `onebit_llm/attention.py` | Causal self-attention, RoPE, QK-norm |
| `onebit_llm/model.py` | DecoderBlock, OneBitLlm (with checkpointing) |
| `onebit_llm/train.py` | Trainer, LR scheduler, quant annealing |
| `onebit_llm/data.py` | Streaming batches from jsonl |
| `onebit_llm/stack_stream.py` | HF list/download, parquet→jsonl, evict, train loop |
| `scripts/stack_stream.py` | CLI for stack-stream training |
| `config_0_3B_stack.json` | Example 0.3B config (16 layers, 1024 hidden, seq 1024) |

See [docs/](docs/) for architecture and setup details.

## License

See [LICENSE](LICENSE).
