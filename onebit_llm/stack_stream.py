"""Stack streaming: list HF shards, download, parquet→jsonl, evict, train."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from huggingface_hub import HfApi, list_repo_files
from tqdm import tqdm

from onebit_llm.config import OneBitLlmConfig
from onebit_llm.model import OneBitLlm
from onebit_llm.train import Trainer
from onebit_llm.data import streaming_batches, iter_jsonl_texts


def disk_usage_bytes(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def list_stack_shards(dataset_id: str, token: str | None = None) -> list[dict]:
    """List data/*.parquet files in the dataset repo."""
    api = HfApi(token=token)
    files = list_repo_files(dataset_id, repo_type="dataset", token=token)
    shards = []
    for path in files:
        if not path.startswith("data/") or not path.endswith(".parquet"):
            continue
        parts = path.split("/")
        language = parts[1] if len(parts) > 1 else "unknown"
        name = Path(path).stem
        shards.append({"path": path, "language": language, "id": f"{language}_{name}"})
    return shards


def group_by_language(shards: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in shards:
        out.setdefault(s["language"], []).append(s)
    for lst in out.values():
        random.shuffle(lst)
    return out


def language_weights(lang_to_shards: dict[str, list]) -> dict[str, float]:
    n = sum(len(v) for v in lang_to_shards.values())
    if n == 0:
        return {}
    return {lang: len(v) / n for lang, v in lang_to_shards.items()}


def sample_language(weights: dict[str, float]) -> str | None:
    r = random.random()
    acc = 0.0
    for lang, w in weights.items():
        acc += w
        if r <= acc:
            return lang
    return next(iter(weights.keys())) if weights else None


def evict_until_under_budget(
    local_root: Path,
    budget_bytes: int,
    protected: set[Path],
) -> None:
    candidates: list[tuple[float, Path]] = []
    for lang_dir in local_root.iterdir():
        if not lang_dir.is_dir():
            continue
        for shard_dir in lang_dir.iterdir():
            if not shard_dir.is_dir():
                continue
            try:
                canon = shard_dir.resolve()
                if canon in protected:
                    continue
                mtime = shard_dir.stat().st_mtime
                candidates.append((mtime, shard_dir))
            except OSError:
                continue
    candidates.sort(key=lambda x: x[0])
    usage = disk_usage_bytes(local_root)
    for _, shard_path in candidates:
        if usage <= budget_bytes:
            break
        size = disk_usage_bytes(shard_path)
        import shutil
        shutil.rmtree(shard_path, ignore_errors=True)
        usage -= size
        print(f"[stack-stream] evicting {shard_path}", flush=True)


def download_shard(
    dataset_id: str,
    repo_path: str,
    token: str | None,
    dest_dir: Path,
) -> Path:
    from huggingface_hub import hf_hub_download
    dest_dir.mkdir(parents=True, exist_ok=True)
    fp = hf_hub_download(
        repo_id=dataset_id,
        filename=repo_path,
        repo_type="dataset",
        token=token,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    return Path(fp)


def parquet_to_jsonl(parquet_path: Path) -> Path:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    for col in ["content", "code", "text"]:
        if col in df.columns:
            break
    else:
        raise ValueError(f"No content/code/text column in {parquet_path}")
    jsonl_path = parquet_path.parent / "content.jsonl"
    with open(jsonl_path, "w") as f:
        for _, row in df.iterrows():
            text = row[col]
            if isinstance(text, str) and text.strip():
                f.write(json.dumps({"text": text}) + "\n")
    parquet_path.unlink(missing_ok=True)
    return jsonl_path


def run_streaming_steps(
    trainer: Trainer,
    data_dir: Path,
    tokenizer_path: Path,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    log_every: int = 100,
    save_every: int = 500,
) -> None:
    steps_done = 0
    for inp, lab in streaming_batches(data_dir, tokenizer_path, seq_len, batch_size, device):
        if steps_done >= max_steps:
            break
        m = trainer.step(inp, lab)
        steps_done += 1
        if log_every > 0 and steps_done % log_every == 0:
            print(f"  [stack-stream] step {trainer.global_step} loss {m['loss']:.4f} lr {m['lr']:.2e}", flush=True)
        if save_every > 0 and steps_done % save_every == 0:
            p = trainer.save_checkpoint()
            print(f"  [stack-stream] checkpoint saved to {p}", flush=True)
    trainer.save_checkpoint()


def run_stack_stream(
    dataset_id: str = "bigcode/the-stack",
    local_root: str | Path = "data/stack_shards",
    output_dir: str | Path = "checkpoints_stack_stream",
    model_config_path: str | Path = "config.json",
    tokenizer_path: str | Path = "data/tokenizer.json",
    disk_budget_gb: float = 900,
    steps_per_shard: int = 1000,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    lr: float = 5e-3,
    lr_min: float = 1e-6,
    lr_warmup_steps: int = 2000,
    lr_decay: str = "none",
    log_every: int = 100,
    save_every: int = 500,
    gradient_checkpointing_segments: int = 8,
    **kwargs: object,
) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    token = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("[stack-stream] warning: no HF_ACCESS_TOKEN or HF_TOKEN in env; private datasets may fail.", flush=True)

    shards = list_stack_shards(dataset_id, token)
    print(f"[stack-stream] listed {len(shards)} shards", flush=True)
    lang_to_shards = group_by_language(shards)
    weights = language_weights(lang_to_shards)
    budget_bytes = int(disk_budget_gb * 1024**3)
    config = OneBitLlmConfig.load(model_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneBitLlm(config).to(device)
    trainer = Trainer(
        model,
        config,
        lr=lr,
        lr_min=lr_min,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay=lr_decay,
        output_dir=output_dir,
        gradient_checkpointing_segments=gradient_checkpointing_segments,
        **kwargs,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "model.safetensors"
    if ckpt.exists():
        trainer.load_weights(ckpt)
        step = Trainer.read_global_step(output_dir)
        if step is not None:
            trainer.global_step = step
            trainer.scheduler.set_step(step)
            print(f"[stack-stream] resumed global_step={step}", flush=True)
    Path(local_root).mkdir(parents=True, exist_ok=True)
    replay: list[Path] = []
    replay_capacity = 10
    seq_len = config.max_seq_len

    while lang_to_shards:
        available = {k: v for k, v in lang_to_shards.items() if v}
        if not available:
            print("[stack-stream] no more shards", flush=True)
            break
        lang = sample_language(weights)
        if lang is None or lang not in lang_to_shards or not lang_to_shards[lang]:
            break
        shard = lang_to_shards[lang].pop()
        shard_dir = Path(local_root) / shard["language"] / shard["id"]
        if shard_dir.exists():
            print(f"[stack-stream] reusing {shard_dir}", flush=True)
        else:
            evict_until_under_budget(Path(local_root), budget_bytes, {Path(p).resolve() for p in replay})
            print(f"[stack-stream] downloading {shard['path']} -> {shard_dir}", flush=True)
            downloaded = download_shard(dataset_id, shard["path"], token, shard_dir)
            if downloaded.suffix == ".parquet":
                parquet_to_jsonl(downloaded)
        replay.append(shard_dir.resolve())
        if len(replay) > replay_capacity:
            replay = replay[-replay_capacity:]
        print(f"[stack-stream] training on {shard_dir} (steps={steps_per_shard})", flush=True)
        run_streaming_steps(
            trainer,
            shard_dir,
            Path(tokenizer_path),
            steps_per_shard,
            batch_size,
            seq_len,
            device,
            log_every=log_every,
            save_every=save_every,
        )
