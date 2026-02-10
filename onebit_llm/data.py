"""Streaming batch iterator: read jsonl, tokenize, yield (input_ids, labels) batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch


def iter_jsonl_texts(path: Path) -> Iterator[str]:
    """Yield non-empty 'text' lines from content.jsonl."""
    jsonl = path / "content.jsonl"
    if not jsonl.exists():
        return
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("text") or obj.get("content") or obj.get("code")
                if t and isinstance(t, str) and t.strip():
                    yield t.strip()
            except Exception:
                continue


def chunked_token_ids(
    tokenizer,
    texts: Iterator[str],
    seq_len: int,
    eos_id: int | None = None,
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield (input_ids, labels) chunks of length seq_len. labels[i] = input_ids[i+1]."""
    buffer: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            ids = list(ids) + [eos_id]
        buffer.extend(ids)
        while len(buffer) >= seq_len + 1:
            inp = buffer[:seq_len]
            lab = buffer[1 : seq_len + 1]
            yield inp, lab
            buffer = buffer[seq_len:]
    # Remainder dropped


def streaming_batches(
    data_dir: Path,
    tokenizer_path: str | Path,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, labels) tensors of shape (batch_size, seq_len)."""
    from transformers import AutoTokenizer
    path = Path(tokenizer_path)
    # If path is a file (e.g. tokenizer.json), use parent dir
    tokenizer_dir = path.parent if path.suffix else path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    texts = iter_jsonl_texts(Path(data_dir))
    chunk_iter = chunked_token_ids(tokenizer, texts, seq_len, eos_id)
    batch_inp: list[list[int]] = []
    batch_lab: list[list[int]] = []
    for inp, lab in chunk_iter:
        batch_inp.append(inp)
        batch_lab.append(lab)
        if len(batch_inp) == batch_size:
            inp = torch.tensor(batch_inp, dtype=torch.long, device=device)
            lab = torch.tensor(batch_lab, dtype=torch.long, device=device)
            yield inp, lab
            batch_inp = []
            batch_lab = []
    if batch_inp:
        while len(batch_inp) < batch_size:
            batch_inp.append(batch_inp[0])
            batch_lab.append(batch_lab[0])
        inp = torch.tensor(batch_inp[:batch_size], dtype=torch.long, device=device)
        lab = torch.tensor(batch_lab[:batch_size], dtype=torch.long, device=device)
        yield inp, lab
