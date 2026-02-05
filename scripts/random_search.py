#!/usr/bin/env python3
"""
Random quantization search baseline for onebit-llm.

This script samples random layer-wise quantization configs and evaluates each
by calling the Rust `eval_config` binary. The results are written to:

- A JSON summary with the best configuration found.
- A CSV file with one row per random sample.

Example:

  python scripts/random_search.py \
    --num-samples 400 \
    --model-config config.json \
    --checkpoint checkpoints/model.safetensors \
    --val-data data/wikitext2/wiki.valid.raw \
    --tokenizer data/superior-reasoning/tokenizer.json \
    --output-json random_result.json \
    --output-csv random_results.csv
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from typing import List, Dict, Any


QUANT_LEVELS = ["Binary", "Ternary", "FourBit", "EightBit", "Float16"]


def load_num_layers(model_config_path: str) -> int:
    with open(model_config_path, "r") as f:
        cfg = json.load(f)
    # onebit-llm config has `num_layers`
    return int(cfg["num_layers"])


def run_eval_config(
    eval_bin: str,
    args: argparse.Namespace,
    quant_config_path: str,
) -> Dict[str, Any]:
    cmd = [
        eval_bin,
        "--model-config",
        args.model_config,
        "--checkpoint",
        args.checkpoint,
        "--val-data",
        args.val_data,
        "--tokenizer",
        args.tokenizer,
        "--quant-config",
        quant_config_path,
        "--max-eval-batches",
        str(args.max_eval_batches),
        "--batch-size",
        str(args.batch_size),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"[random_search] eval_config failed (code {proc.returncode}):\n")
        sys.stderr.write(proc.stderr + "\n")
        raise RuntimeError("eval_config failed")
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[random_search] failed to parse eval_config output: {e}\n")
        sys.stderr.write(proc.stdout + "\n")
        raise


def sample_quant_config(num_layers: int) -> Dict[str, Any]:
    layer_quant = {
        str(i): random.choice(QUANT_LEVELS)
        for i in range(num_layers)
    }
    return {
        "layer_quant": layer_quant,
        "num_layers": num_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Random search baseline for onebit-llm quantization")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of random configs to evaluate")
    parser.add_argument("--model-config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.safetensors")
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="random_result.json")
    parser.add_argument("--output-csv", type=str, default="random_results.csv")
    parser.add_argument("--eval-bin", type=str, default="./target/release/eval_config")
    parser.add_argument("--max-eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    if not os.path.exists(args.eval_bin):
        sys.stderr.write(f"[random_search] eval_bin not found at {args.eval_bin}. "
                         "Build it with `cargo build --release --bin eval_config`.\n")
        sys.exit(1)

    num_layers = load_num_layers(args.model_config)

    # Prepare CSV
    csv_fp = open(args.output_csv, "w", encoding="utf-8")
    csv_fp.write("sample_id,config_key,loss,perplexity,quant_json\n")
    csv_fp.flush()

    best: Dict[str, Any] = {
        "loss": float("inf"),
        "perplexity": float("inf"),
        "config": None,
        "config_key": None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(args.num_samples):
            quant_cfg = sample_quant_config(num_layers)
            q_path = os.path.join(tmpdir, f"quant_{i}.json")
            with open(q_path, "w", encoding="utf-8") as f:
                json.dump(quant_cfg, f)

            try:
                res = run_eval_config(args.eval_bin, args, q_path)
            except Exception:
                # Skip failed evals but continue search
                continue

            loss = float(res["loss"])
            ppl = float(res["perplexity"])
            cfg_key = str(res.get("config_key", ""))

            csv_fp.write(
                f"{i},{cfg_key},{loss},{ppl},\"{json.dumps(quant_cfg).replace('\"', '\"\"')}\"\n"
            )
            csv_fp.flush()

            if loss < best["loss"]:
                best["loss"] = loss
                best["perplexity"] = ppl
                best["config"] = quant_cfg
                best["config_key"] = cfg_key

            sys.stderr.write(
                f"[random_search] {i+1}/{args.num_samples}: loss={loss:.4f}, ppl={ppl:.2e}\n"
            )

    csv_fp.close()

    summary = {
        "best_loss": best["loss"],
        "best_perplexity": best["perplexity"],
        "best_config": best["config"],
        "best_config_key": best["config_key"],
        "num_samples": args.num_samples,
        "model_config": args.model_config,
        "checkpoint": args.checkpoint,
        "val_data": args.val_data,
        "tokenizer": args.tokenizer,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sys.stderr.write(f"[random_search] Done. Summary written to {args.output_json}, "
                     f"CSV to {args.output_csv}.\n")


if __name__ == "__main__":
    main()

