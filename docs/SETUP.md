# Setup

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (with CUDA if you want GPU training)
- See `pyproject.toml` for full dependencies: `transformers`, `huggingface-hub`, `pyarrow`, `pandas`, `tqdm`, `python-dotenv`, `safetensors`.

## Install

**GPU (CUDA 13.0):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

**CPU:**

```bash
pip install -e .
```

## Config and tokenizer

- **Model config**: JSON file (e.g. `config_0_3B_stack.json`) with `vocab_size`, `hidden_size`, `num_heads`, `num_layers`, `intermediate_size`, `max_seq_len`, and quant options. See existing configs in the repo root.
- **Tokenizer**: Hugging Face tokenizer (directory containing `tokenizer.json` or a model id). Pass the path to the directory or to `tokenizer.json`; the loader uses the parent directory.

## Hugging Face

For datasets like `bigcode/the-stack`, set `HF_ACCESS_TOKEN` or `HF_TOKEN` in the environment (or in a `.env` file in the project root) if the repo is gated or private.
