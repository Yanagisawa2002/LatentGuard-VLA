"""Evaluate the frozen LG-R1 SARM checkpoint on all active unseen tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1_training import load_current_samples, peak_cuda_memory
from _lg_r1b_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    sha256_path,
    write_json,
)
from lg_r1_train_or_eval_sarm import (
    _build_clip_cache,
    _load_arrays,
    _models,
    _sample_identity,
    _split_metrics,
    _write_predictions,
)


def _balanced_rows(
    rows: list[dict[str, Any]],
    maximum_per_task: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["suite"]), int(row["task_id"])), []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda item: (str(item["episode_id"]), int(item["frame_index"])),
        )
        if len(items) <= maximum_per_task:
            chosen = items
        else:
            indices = np.linspace(
                0,
                len(items) - 1,
                num=maximum_per_task,
                dtype=np.int64,
            )
            chosen = [items[int(index)] for index in indices]
        for row in chosen:
            copied = dict(row)
            copied["split"] = "test"
            selected.append(copied)
    return selected


def _task_metrics(
    stage_model: Any,
    subtask_model: Any,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    *,
    device: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    keys = [(str(row["suite"]), int(row["task_id"])) for row in rows]
    for key in sorted(set(keys)):
        mask = np.asarray([candidate == key for candidate in keys], dtype=bool)
        subset = {name: values[mask] for name, values in arrays.items()}
        subset["split"] = np.asarray(["test"] * int(mask.sum()))
        metrics[f"{key[0]}/task{key[1]}"] = _split_metrics(
            stage_model,
            subtask_model,
            subset,
            split="test",
            device=device,
        )
    return metrics


def _triggered(metrics: dict[str, Any], trigger: dict[str, Any]) -> bool:
    any_rule = trigger["any"]
    return bool(
        float(metrics["progress"]["mae"]) > float(any_rule["progress_mae_above"])
        or float(metrics["progress"]["spearman"])
        < float(any_rule["progress_spearman_below"])
        or float(metrics["pairwise"]["accuracy"])
        < float(any_rule["pairwise_progress_accuracy_below"])
    )


def main() -> None:
    """Load only a frozen checkpoint; optimizer construction is prohibited."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1b/eval_zero_shot.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    eval_config = read_yaml(resolve_repo_path(args.config))
    base = read_yaml(resolve_repo_path(Path(str(eval_config["base_sarm_config"]))))
    config = {**base, **eval_config}
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    checkpoint = args.checkpoint
    if checkpoint is None:
        source_root = os.environ.get("LG_R1_SOURCE_ROOT")
        if not source_root:
            raise ValueError("--checkpoint or LG_R1_SOURCE_ROOT is required")
        checkpoint = Path(source_root) / str(config["frozen_lg_r1_checkpoint_locator"])
    checkpoint = resolve_repo_path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen LG-R1 SARM checkpoint missing: {checkpoint}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "checkpoint_exists": True,
                    "optimizer_steps": 0,
                },
                sort_keys=True,
            )
        )
        return
    stage_report = read_json(runtime / "stage_annotation_report.json")
    if stage_report.get("status") != "pass":
        raise ValueError("stage QA must pass before zero-shot evaluation")
    maximum = int(config["maximum_samples_per_task"])
    if args.limit_samples is not None:
        maximum = min(maximum, int(args.limit_samples))
    rows = _balanced_rows(load_current_samples(runtime), maximum)
    cache_root = runtime / "sarm-zero-shot"
    cache_path = cache_root / "frozen_clip_features.npz"
    feature_manifest_path = cache_root / "feature_manifest.json"
    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        feature_manifest = _build_clip_cache(
            rows,
            runtime=runtime,
            cache_path=cache_path,
            config=config,
        )
        feature_manifest["sample_identity"] = _sample_identity(rows)
        write_json(feature_manifest_path, feature_manifest)
    else:
        feature_manifest = read_json(feature_manifest_path)
        if feature_manifest["sample_identity"] != _sample_identity(rows):
            raise ValueError("zero-shot feature cache sample identity drift")
    arrays = _load_arrays(cache_path)
    if len(arrays["video"]) != len(rows):
        raise ValueError("zero-shot feature cache sample mismatch")
    import torch

    device = torch.device("cuda")
    stage_model, subtask_model = _models(config, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    stage_model.load_state_dict(state["stage_model"])
    subtask_model.load_state_dict(state["subtask_model"])
    held_out = _split_metrics(
        stage_model,
        subtask_model,
        arrays,
        split="test",
        device=device,
    )
    per_task = _task_metrics(
        stage_model,
        subtask_model,
        arrays,
        rows,
        device=device,
    )
    lg_r1 = read_json(resolve_repo_path(Path(str(config["frozen_lg_r1_results"]))))
    prediction_path = runtime / "sarm_zero_shot_predictions.jsonl"
    _write_predictions(
        prediction_path,
        stage_model,
        subtask_model,
        arrays,
        device=device,
    )
    result = {
        "schema_version": "latentguard.lg_r1b.sarm_zero_shot_results.v1",
        "status": "pass",
        "evaluation": "held-out-task zero-shot",
        "frozen_before_adaptation": True,
        "checkpoint_sha256": sha256_path(checkpoint),
        "checkpoint_selection": config["checkpoint_selection"],
        "optimizer_steps": 0,
        "trained_parameters": [],
        "clip_frozen": True,
        "vlajepa_frozen": True,
        "policy_frozen": True,
        "held_out_task_samples": len(rows),
        "held_out_task_count": len(per_task),
        "held_out_task": held_out,
        "per_task": per_task,
        "lg_r1_in_distribution_test": lg_r1["test"],
        "adaptation_trigger": config["adaptation_trigger"],
        "adaptation_triggered": _triggered(held_out, config["adaptation_trigger"]),
        "prediction_locator": "sarm_zero_shot_predictions.jsonl",
        "feature_manifest": feature_manifest,
        "cuda_memory": peak_cuda_memory(),
        "claim_boundary": (
            "Progress/stage generalization evidence only; not a safety detector."
        ),
    }
    result_path = (
        resolve_repo_path(args.result)
        if args.result is not None
        else runtime / "sarm_zero_shot_results.json"
    )
    write_json(result_path, result)
    freeze_record = {
        "schema_version": "latentguard.lg_r1b.zero_shot_freeze.v1",
        "status": "frozen_before_adaptation",
        "result_sha256": sha256_path(result_path),
        "checkpoint_sha256": sha256_path(checkpoint),
        "optimizer_steps": 0,
    }
    write_json(runtime / "sarm_zero_shot_freeze.json", freeze_record)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
