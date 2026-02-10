#!/usr/bin/env python3
"""
Shard-based streaming trainer for The Stack-style massive datasets.

Goal
-----
- Train a model on ~TB-scale data while:
  * Only keeping <= disk_budget_gb on local disk at any time.
  * Interleaving languages / domains across time (avoid "English phase" then "French phase").
  * Driving the existing `onebit-train` Rust binary (no changes needed there).

High-level design
------------------
- Input: a JSONL manifest of shards, e.g. `data/the_stack_manifest.jsonl`:

    {"id": "py_0001", "language": "python", "url": "s3://...", "bytes": 120000000000}
    {"id": "js_0001", "language": "javascript", "url": "s3://...", "bytes": 90000000000}
    ...

- This script:
  1. Loads all shards + basic metadata.
  2. Builds a global, interleaved shard schedule (language-weighted sampling).
  3. For each shard in schedule:
     - Ensures disk usage under budget (deletes old shards that are not pinned).
     - Downloads the shard into `--local-root / <language> / <id>`.
     - Runs `onebit-train` for a fixed number of steps on that shard (streaming mode).
     - Optionally keeps a small "replay buffer" of recent shards on disk.

Notes / assumptions
--------------------
- This is an orchestration script; you still need:
  * A downloader that can fetch each shard given its `url`. We expose this as a shell
    command template (`--download-cmd`) and format it with `{url}` and `{out_dir}`.
  * A tokenizer compatible with The Stack (pointed by `--tokenizer`).
- Mixing between languages happens at *shard* granularity here. If you later extend
  `onebit-train` to accept multiple data dirs at once, you can upgrade this script
  to launch multi-dir runs per iteration to get per-batch mixing.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    # When available, prefer the Hugging Face Python API for downloads.
    # This avoids CLI argument pitfalls and shell-escaping issues.
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None


@dataclass
class Shard:
    id: str
    language: str
    url: str
    bytes: int


def load_manifest(path: Path) -> List[Shard]:
    shards: List[Shard] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            shards.append(
                Shard(
                    id=str(obj["id"]),
                    language=str(obj.get("language", "unknown")),
                    url=str(obj["url"]),
                    bytes=int(obj.get("bytes", 0)),
                )
            )
    if not shards:
        raise RuntimeError(f"Manifest {path} is empty.")
    return shards


def group_by_language(shards: List[Shard]) -> Dict[str, List[Shard]]:
    out: Dict[str, List[Shard]] = {}
    for s in shards:
        out.setdefault(s.language, []).append(s)
    # Shuffle shards within each language to avoid fixed ordering
    for lst in out.values():
        random.shuffle(lst)
    return out


def build_language_weights(lang_to_shards: Dict[str, List[Shard]]) -> Dict[str, float]:
    """Simple heuristic: weight by total bytes per language."""
    totals: Dict[str, int] = {}
    for lang, shards in lang_to_shards.items():
        totals[lang] = sum(s.bytes for s in shards) or len(shards)
    total_bytes = sum(totals.values())
    if total_bytes == 0:
        # Fallback: uniform over languages
        return {lang: 1.0 / len(lang_to_shards) for lang in lang_to_shards}
    return {lang: v / total_bytes for lang, v in totals.items()}


def sample_language(lang_weights: Dict[str, float]) -> str:
    langs = list(lang_weights.keys())
    weights = [lang_weights[l] for l in langs]
    r = random.random()
    acc = 0.0
    for l, w in zip(langs, weights):
        acc += w
        if r <= acc:
            return l
    return langs[-1]


def current_disk_usage_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                st = os.stat(os.path.join(root, fn))
            except FileNotFoundError:
                continue
            total += st.st_size
    return total


def run_cmd(cmd: List[str]) -> None:
    print(f"[stack_stream_train] running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def format_download_cmd(template: str, url: str, out_dir: Path) -> List[str]:
    """
    Expand a user-provided download command template.

    Example:
      --download-cmd "aws s3 cp {url} {out_dir} --recursive"
      --download-cmd "bash -lc 'huggingface-cli download ... -d {out_dir}'"
    """
    cmd_str = template.format(url=url, out_dir=str(out_dir))
    return ["bash", "-lc", cmd_str]


def ensure_shard_downloaded(
    shard: Shard,
    shard_dir: Path,
    download_cmd_template: str,
) -> None:
    if shard_dir.exists():
        return
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Özel durum: The Stack (bigcode/the-stack) URL formatını Python API ile çöz.
    # URL şablonu: "bigcode/the-stack:data/<language>/train-....parquet"
    if shard.url.startswith("bigcode/the-stack:") and hf_hub_download is not None:
        repo_id, subpath = shard.url.split(":", 1)
        print(
            f"[stack_stream_train] downloading via huggingface_hub: repo_id={repo_id} path={subpath}",
            file=sys.stderr,
        )
        hf_hub_download(
            repo_id=repo_id,
            filename=subpath,
            repo_type="dataset",
            local_dir=str(shard_dir),
            local_dir_use_symlinks=False,
        )
        return

    # Genel durum: Kullanıcının verdiği shell komut şablonunu kullan.
    cmd = format_download_cmd(download_cmd_template, shard.url, shard_dir)
    run_cmd(cmd)


def evict_if_needed(
    local_root: Path,
    disk_budget_bytes: int,
    protected_shards: List[Path],
) -> None:
    """Keep disk usage under budget by deleting old shard dirs not in protected_shards."""
    usage = current_disk_usage_bytes(local_root)
    if usage <= disk_budget_bytes:
        return

    # Collect candidate shard dirs (sorted by mtime, oldest first)
    candidates: List[Tuple[float, Path]] = []
    protected_set = {p.resolve() for p in protected_shards}
    for child in local_root.iterdir():
        if not child.is_dir():
            continue
        for shard_dir in child.iterdir():
            if not shard_dir.is_dir():
                continue
            if shard_dir.resolve() in protected_set:
                continue
            try:
                mtime = shard_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            candidates.append((mtime, shard_dir))

    candidates.sort(key=lambda x: x[0])  # oldest first

    for _mtime, shard_path in candidates:
        if usage <= disk_budget_bytes:
            break
        print(f"[stack_stream_train] evicting shard {shard_path}", file=sys.stderr)
        try:
            shutil.rmtree(shard_path)
        except FileNotFoundError:
            pass
        usage = current_disk_usage_bytes(local_root)


def run_training_on_shard(
    shard_dir: Path,
    args: argparse.Namespace,
) -> None:
    """
    Launch onebit-train for a fixed number of steps on a single shard directory.

    We rely on the existing `--streaming` mode, which will iterate files inside
    `--data-dir` without trying to pre-load everything into memory at once.
    """
    cmd = [
        "cargo",
        "run",
        "--features",
        "cuda",
        "-p",
        "ternary-train",
        "--bin",
        "onebit-train",
        "--release",
        "--",
        "--config",
        args.model_config,
        "--data-dir",
        str(shard_dir),
        "--tokenizer",
        args.tokenizer,
        "--output-dir",
        str(args.output_root),
        "--streaming",
        "--batch-size",
        str(args.batch_size),
        "--accumulation-steps",
        str(args.accumulation_steps),
        "--max-steps",
        str(args.steps_per_shard),
        "--save-every",
        str(args.save_every),
        "--lr",
        str(args.lr),
        "--lr-warmup-steps",
        str(args.lr_warmup_steps),
        "--lr-min",
        str(args.lr_min),
        "--lr-decay",
        args.lr_decay,
        "--label-smoothing",
        str(args.label_smoothing),
        "--quant-warmup-steps",
        str(args.quant_warmup_steps),
        "--quant-anneal-steps",
        str(args.quant_anneal_steps),
    ]
    run_cmd(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Shard-based streaming trainer for The Stack.")
    p.add_argument("--manifest", type=str, required=True, help="JSONL manifest of shards.")
    p.add_argument(
        "--local-root",
        type=str,
        default="data/stack_shards",
        help="Root directory for locally cached shards.",
    )
    p.add_argument(
        "--disk-budget-gb",
        type=float,
        default=900.0,
        help="Max local disk usage for shards (in GB).",
    )
    p.add_argument(
        "--replay-capacity",
        type=int,
        default=4,
        help="Number of most-recent shard dirs to keep as a simple replay buffer.",
    )
    p.add_argument(
        "--download-cmd",
        type=str,
        required=True,
        help="Shell command template to download a shard. Use {url} and {out_dir} placeholders.",
    )

    # Training hyper-parameters / onebit-train args
    p.add_argument("--model-config", type=str, default="config.json")
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--output-root", type=str, default="checkpoints_stack_stream")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--steps-per-shard", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lr-warmup-steps", type=int, default=2000)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-decay", type=str, default="cosine")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--quant-warmup-steps", type=int, default=2000)
    p.add_argument("--quant-anneal-steps", type=int, default=8000)

    # Total training budget
    p.add_argument(
        "--max-global-steps",
        type=int,
        default=0,
        help="Optional cap on total training steps across all shards (0 = unlimited).",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="Optional cap on number of shards to process (0 = all).",
    )

    args = p.parse_args()

    manifest_path = Path(args.manifest)
    shards = load_manifest(manifest_path)
    lang_to_shards = group_by_language(shards)
    lang_weights = build_language_weights(lang_to_shards)

    local_root = Path(args.local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    disk_budget_bytes = int(args.disk_budget_gb * (1024**3))
    global_steps = 0
    processed_shards = 0
    replay_buffer: List[Path] = []

    print(
        f"[stack_stream_train] starting: {len(shards)} shards, "
        f"languages={list(lang_to_shards.keys())}, disk_budget={args.disk_budget_gb} GB",
        file=sys.stderr,
    )

    # We use a simple loop: at each iteration, pick a language by weight, then pop
    # the next shard for that language. This naturally interleaves languages over time.
    while True:
        if args.max_shards and processed_shards >= args.max_shards:
            print("[stack_stream_train] reached max_shards, stopping.", file=sys.stderr)
            break
        if args.max_global_steps and global_steps >= args.max_global_steps:
            print("[stack_stream_train] reached max_global_steps, stopping.", file=sys.stderr)
            break

        # Filter out languages with no remaining shards
        available_langs = {l: w for l, w in lang_weights.items() if lang_to_shards.get(l)}
        if not available_langs:
            print("[stack_stream_train] no more shards available, stopping.", file=sys.stderr)
            break

        lang = sample_language(available_langs)
        shard_list = lang_to_shards[lang]
        shard = shard_list.pop()

        shard_dir = local_root / shard.language / shard.id
        print(
            f"[stack_stream_train] next shard: id={shard.id} lang={shard.language} "
            f"bytes={shard.bytes}",
            file=sys.stderr,
        )

        # Evict old shards if needed
        evict_if_needed(local_root, disk_budget_bytes, replay_buffer)

        # Download shard if missing
        ensure_shard_downloaded(shard, shard_dir, args.download_cmd)

        # Update replay buffer
        replay_buffer.append(shard_dir)
        if len(replay_buffer) > args.replay_capacity:
            replay_buffer = replay_buffer[-args.replay_capacity :]

        # Train on this shard for a fixed number of steps
        run_training_on_shard(shard_dir, args)
        global_steps += args.steps_per_shard
        processed_shards += 1

        print(
            f"[stack_stream_train] finished shard {shard.id} "
            f"(global_steps={global_steps}, processed_shards={processed_shards})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

