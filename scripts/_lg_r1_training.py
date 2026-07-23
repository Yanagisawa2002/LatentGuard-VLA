"""Shared remote-only data, checkpoint, and training utilities for LG-R1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    write_json,
)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the repository-required bounded training controls."""

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-every-steps", type=int)


def resolved_training_config(
    config_path: Path, args: argparse.Namespace, *, run_name: str
) -> tuple[dict[str, Any], Path]:
    """Resolve overrides and create an ignored run output directory."""

    config = read_yaml(resolve_repo_path(config_path))
    for argument, key in (
        ("max_steps", "max_steps"),
        ("seed", "seed"),
        ("eval_every_steps", "eval_every_steps"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    if int(config["max_steps"]) < 1:
        raise ValueError("max_steps must be positive")
    if int(config["eval_every_steps"]) < 1:
        raise ValueError("eval_every_steps must be positive")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("limit_samples must be positive")
    run_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else output_root() / "runs" / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "resolved_config.json", config)
    return config, run_dir


def load_current_samples(runtime: Path) -> list[dict[str, Any]]:
    """Load model-input-only current samples from the accepted dataset."""

    manifest = read_json(runtime / "dataset_manifest.json")
    if manifest.get("status") != "pass":
        raise ValueError("accepted dataset manifest is required")
    if manifest.get("privileged_model_inputs"):
        raise ValueError("privileged fields are present in model inputs")
    rows: list[dict[str, Any]] = []
    path = runtime / "progress_dataset" / "current_samples.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("privileged_fields"):
                raise ValueError("invalid or privileged current sample")
            rows.append(row)
    if not rows:
        raise ValueError("current-state dataset is empty")
    return rows


def bounded_samples(
    rows: list[dict[str, Any]], *, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    """Apply an outcome-independent deterministic cap per split."""

    if limit is None:
        return rows
    caps = {
        "train": limit,
        "validation": max(limit // 3, 1),
        "test": max(limit // 3, 1),
    }
    selected: list[dict[str, Any]] = []
    for split, cap in caps.items():
        candidates = [row for row in rows if row["split"] == split]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                (f"{seed}|{row['episode_id']}|{row['frame_index']}").encode()
            ).hexdigest()
        )
        selected.extend(candidates[:cap])
    return selected


class EpisodeDatasetCache:
    """Lazily load one local LeRobotDataset per complete episode."""

    def __init__(self, runtime: Path) -> None:
        self._runtime = runtime
        self._datasets: dict[str, Any] = {}

    def get(self, episode_id: str) -> Any:
        """Return the exact local episode dataset."""

        if episode_id not in self._datasets:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._datasets[episode_id] = LeRobotDataset(
                "lg_r1_eval_recording",
                root=self._runtime / "datasets" / episode_id,
            )
        return self._datasets[episode_id]

    def frame(self, episode_id: str, index: int) -> dict[str, Any]:
        """Read one bounded frame."""

        dataset = self.get(episode_id)
        bounded = min(max(index, 0), len(dataset) - 1)
        value = dataset[bounded]
        if not isinstance(value, dict):
            raise ValueError("LeRobotDataset returned a non-mapping frame")
        return value


def set_training_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch deterministically."""

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_digest(module: Any) -> str:
    """Hash deterministic sentinel tensors to detect backbone mutation."""

    import torch

    parameters = list(module.named_parameters())
    if not parameters:
        raise ValueError("module has no parameters")
    indices = sorted({0, len(parameters) // 2, len(parameters) - 1})
    digest = hashlib.sha256()
    for index in indices:
        name, parameter = parameters[index]
        digest.update(name.encode())
        digest.update(
            np.ascontiguousarray(
                parameter.detach().to("cpu", dtype=torch.float32).numpy()
            ).tobytes()
        )
    return digest.hexdigest()


def save_torch_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save a resumable torch checkpoint outside Git."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def peak_cuda_memory() -> dict[str, Any]:
    """Return measured CUDA allocation counters."""

    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def write_run_manifest(
    run_dir: Path,
    *,
    run_type: str,
    config: dict[str, Any],
    status: str,
    optimizer_steps: int,
    checkpoint_path: Path | None,
    extra: dict[str, Any],
) -> None:
    """Write the required sanitized remote run manifest."""

    payload = {
        "schema_version": "latentguard.lg_r1.training_run.v1",
        "run_type": run_type,
        "status": status,
        "runtime_identity": runtime_identity(),
        "resolved_config": config,
        "optimizer_steps": optimizer_steps,
        "checkpoint_locator": (
            str(checkpoint_path.name) if checkpoint_path is not None else None
        ),
        "cuda_memory": peak_cuda_memory(),
        **extra,
    }
    write_json(run_dir / "run_manifest.json", payload)
