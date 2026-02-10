#!/usr/bin/env python3
"""CLI: onebit stack-stream (Python rewrite)."""

import argparse
from pathlib import Path

from onebit_llm.stack_stream import run_stack_stream


def main() -> None:
    p = argparse.ArgumentParser(description="OneBit stack-stream: HF download, train, evict")
    p.add_argument("--local-root", type=Path, default=Path("data/stack_shards"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints_stack_stream"))
    p.add_argument("--dataset-id", type=str, default="bigcode/the-stack")
    p.add_argument("--disk-budget-gb", type=float, default=900)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--steps-per-shard", type=int, default=1000)
    p.add_argument("--lr", type=str, default="5e-3")
    p.add_argument("--lr-min", type=str, default="1e-6")
    p.add_argument("--lr-warmup-steps", type=int, default=2000)
    p.add_argument("--lr-decay", type=str, default="none")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--gradient-checkpointing-segments", type=int, default=8)
    args = p.parse_args()
    lr = float(args.lr)
    lr_min = float(args.lr_min)
    run_stack_stream(
        dataset_id=args.dataset_id,
        local_root=args.local_root,
        output_dir=args.output_dir,
        model_config_path=args.model_config,
        tokenizer_path=args.tokenizer,
        disk_budget_gb=args.disk_budget_gb,
        steps_per_shard=args.steps_per_shard,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=lr,
        lr_min=lr_min,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        log_every=args.log_every,
        save_every=args.save_every,
        gradient_checkpointing_segments=args.gradient_checkpointing_segments,
    )


if __name__ == "__main__":
    main()
