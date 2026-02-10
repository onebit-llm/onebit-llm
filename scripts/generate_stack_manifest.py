#!/usr/bin/env python3
"""
generate_stack_manifest.py
---------------------------

Generate a lightweight **manifest JSONL** file for The Stack dataset, instead
of trying to download all ~6 TB up front. This manifest is then consumed by
`stack_stream_train.py` to drive the shard download–train–evict loop.

Notes:
- This script only uses Hugging Face metadata; it does **not** download data.
- You must already be logged in and authorized via `hf` / `huggingface_hub`.

Output format (JSON Lines, one shard per line):

    {"id": "python_train_0000", "language": "python", "url": "bigcode/the-stack:data/python/train-00000-of-01024.parquet"}

The `bytes` field is optional; if omitted, the training script falls back to 0.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from huggingface_hub import HfApi


@dataclass
class StackFile:
    path: str  # e.g. "data/python/train-00000-of-01024.parquet"

    @property
    def language(self) -> str:
        parts = self.path.split("/")
        # Expected layout: ["data", "<language>", ...]
        if len(parts) >= 2:
            return parts[1]
        return "unknown"

    @property
    def id(self) -> str:
        # Example: "data/python/train-00000-of-01024.parquet" ->
        # language="python", basename="train-00000-of-01024.parquet"
        lang = self.language
        basename = self.path.split("/")[-1]
        base_no_ext = basename.split(".")[0]
        return f"{lang}_{base_no_ext}"

    @property
    def url(self) -> str:
        # URL format compatible with stack_stream_train:
        #   "bigcode/the-stack:data/python/train-00000-of-01024.parquet"
        return f"bigcode/the-stack:{self.path}"


def list_stack_files(dataset_id: str) -> List[StackFile]:
    api = HfApi()
    files = api.list_repo_files(dataset_id, repo_type="dataset")
    out: List[StackFile] = []
    for path in files:
        # The Stack typically stores shards under "data/<language>/...".
        if not path.startswith("data/"):
            continue
        # You can further restrict to parquet / jsonl.zst here if desired, e.g.:
        # if not path.endswith(".parquet"): continue
        out.append(StackFile(path=path))
    if not out:
        raise RuntimeError(f"{dataset_id} için data/* altında dosya bulamadım.")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a manifest JSONL for The Stack.")
    p.add_argument(
        "--dataset-id",
        type=str,
        default="bigcode/the-stack",
        help="HF dataset ID (default: bigcode/the-stack)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/the_stack_manifest.jsonl",
        help="Output manifest path (JSONL).",
    )
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[generate_stack_manifest] listing files for {args.dataset_id} ...")
    stack_files = list_stack_files(args.dataset_id)
    print(f"[generate_stack_manifest] found {len(stack_files)} files under data/*")

    with out_path.open("w", encoding="utf-8") as f:
        for sf in stack_files:
            record = {
                "id": sf.id,
                "language": sf.language,
                "url": sf.url,
                # `bytes` is optional; the training script falls back to 0 if missing.
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[generate_stack_manifest] wrote manifest to {out_path}")


if __name__ == "__main__":
    main()

