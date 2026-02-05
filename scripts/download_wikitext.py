#!/usr/bin/env python3
"""Download and format WikiText-2 for onebit-llm training.

Creates:
    data/wikitext-2/train.txt   ~10 MB, ~2M tokens
    data/wikitext-2/valid.txt   ~1 MB
    data/wikitext-2/test.txt    ~1 MB
    data/tokenizer.json          GPT-2 tokenizer

Each output file is one paragraph per line (blank lines and section
headers stripped). This format works directly with both TextDataset
(in-memory) and StreamingBatchIter (streaming).

Usage:
    python3 scripts/download_wikitext.py
    python3 scripts/download_wikitext.py --dataset wikitext-103  # larger
"""

import argparse
import json
import os
import sys
import urllib.request

# ── HuggingFace datasets rows API ──────────────────────────────────────────
# We fetch text via the REST API to avoid needing pyarrow/pandas.

HF_BASE = "https://datasets-server.huggingface.co/rows"

CONFIGS = {
    "wikitext-2": "wikitext-2-raw-v1",
    "wikitext-103": "wikitext-103-raw-v1",
}

SPLITS = ["train", "validation", "test"]
SPLIT_RENAME = {"validation": "valid"}  # we rename for consistency

TOKENIZER_URL = "https://huggingface.co/gpt2/resolve/main/tokenizer.json"
MAX_ROWS_PER_REQUEST = 100  # HF API limit


def fetch_json(url: str) -> dict:
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "onebit-llm/0.2"})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode("utf-8"))


def fetch_split(dataset: str, config: str, split: str) -> list[str]:
    """Fetch all rows for a split via HuggingFace rows API."""
    rows = []
    offset = 0
    while True:
        url = (
            f"{HF_BASE}?dataset={dataset}&config={config}"
            f"&split={split}&offset={offset}&length={MAX_ROWS_PER_REQUEST}"
        )
        try:
            data = fetch_json(url)
        except Exception as e:
            print(f"\n  API error at offset {offset}: {e}")
            break

        batch = data.get("rows", [])
        if not batch:
            break

        for row in batch:
            text = row.get("row", {}).get("text", "")
            if text:
                rows.append(text)

        num_total = data.get("num_rows_total", 0)
        offset += len(batch)
        pct = min(100, offset * 100 // max(num_total, 1))
        print(f"\r  Fetching {split}: {offset:,} / {num_total:,} rows ({pct}%)", end="", flush=True)

        if offset >= num_total:
            break

    print()
    return rows


def clean_wikitext(lines: list[str]) -> list[str]:
    """Clean raw WikiText lines into one-paragraph-per-line format.

    Removes:
      - Section headers (lines starting with ' = ')
      - Blank lines
      - Very short fragments (<20 chars)

    Joins consecutive non-blank lines into paragraphs.
    """
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        # Skip wiki section headers
        if stripped.startswith("=") and stripped.endswith("="):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        # Skip very short fragments
        if len(stripped) < 20:
            if current:
                current.append(stripped)
            continue

        current.append(stripped)

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def download_file(url: str, path: str, desc: str):
    """Download a file with progress."""
    print(f"Downloading {desc}...")
    req = urllib.request.Request(url, headers={"User-Agent": "onebit-llm/0.2"})
    resp = urllib.request.urlopen(req)
    total = resp.headers.get("Content-Length")
    data = bytearray()
    chunk_size = 1 << 20
    while True:
        chunk = resp.read(chunk_size)
        if not chunk:
            break
        data.extend(chunk)
        if total:
            pct = len(data) * 100 / int(total)
            print(f"\r  {len(data) / 1e6:.1f} / {int(total) / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
        else:
            print(f"\r  {len(data) / 1e6:.1f} MB", end="", flush=True)
    print()
    with open(path, "wb") as f:
        f.write(data)


def main():
    parser = argparse.ArgumentParser(description="Download WikiText for onebit-llm")
    parser.add_argument(
        "--dataset",
        choices=list(CONFIGS.keys()),
        default="wikitext-2",
        help="Which dataset to download (default: wikitext-2)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Root output directory (default: data/)",
    )
    args = parser.parse_args()

    config_name = CONFIGS[args.dataset]
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    hf_dataset = "Salesforce/wikitext"

    # --- Fetch each split via the rows API ---
    for split in SPLITS:
        out_name = SPLIT_RENAME.get(split, split)
        out_path = os.path.join(out_dir, f"{out_name}.txt")

        if os.path.exists(out_path):
            print(f"  {out_name}.txt already exists, skipping.")
            continue

        raw_lines = fetch_split(hf_dataset, config_name, split)
        paragraphs = clean_wikitext(raw_lines)

        with open(out_path, "w", encoding="utf-8") as f:
            for para in paragraphs:
                f.write(para + "\n")

        word_count = sum(len(p.split()) for p in paragraphs)
        token_est = int(word_count * 1.3)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  -> {out_name}: {len(paragraphs):,} paragraphs, ~{word_count:,} words "
              f"(~{token_est:,} tokens), {size_mb:.1f} MB -> {out_path}")

    # --- Download GPT-2 tokenizer ---
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"Tokenizer already exists: {tokenizer_path}")
    else:
        download_file(TOKENIZER_URL, tokenizer_path, "GPT-2 tokenizer")
        size_mb = os.path.getsize(tokenizer_path) / 1e6
        print(f"  Saved tokenizer: {tokenizer_path} ({size_mb:.1f} MB)")

    # --- Print summary ---
    train_path = os.path.join(out_dir, "train.txt")
    valid_path = os.path.join(out_dir, "valid.txt")

    print()
    print("=" * 70)
    print("Ready! Suggested training command:")
    print("=" * 70)
    print()
    print(f"cargo run --release -p ternary-train --bin onebit-train --features cuda -- \\")
    print(f"  --config config_wikitext.json \\")
    print(f"  --data-dir {train_path} \\")
    print(f"  --tokenizer {tokenizer_path} \\")
    print(f"  --output-dir checkpoints/wikitext \\")
    print(f"  --val-data-dir {valid_path} \\")
    print(f"  --batch-size 4 \\")
    print(f"  --accumulation-steps 8 \\")
    print(f"  --max-steps 20000 \\")
    print(f"  --lr 1e-3 \\")
    print(f"  --lr-warmup-steps 500 \\")
    print(f"  --lr-decay cosine \\")
    print(f"  --save-every 2000 \\")
    print(f"  --eval-every 1000 \\")
    print(f"  --log-every 100")
    print()


if __name__ == "__main__":
    main()
