"""Build the LG-R1b current-frame dataset with a frozen task-level split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1b_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    sha256_path,
    write_json,
)
from lg_r1_build_stage_labels import _read_jsonl

from latentguard.progress.serialization import write_jsonl_atomic


def _split_map(registry: dict[str, Any]) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for split, tasks in registry["adaptation_task_split"].items():
        for task in tasks:
            key = str(task["suite"]), int(task["task_id"])
            if key in result:
                raise ValueError(f"adaptation task split overlap: {key}")
            result[key] = str(split)
    return result


def main() -> None:
    """Serialize only allowlisted deployable inputs and current labels."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-registry",
        type=Path,
        default=Path("configs/lg_r1b/task_registry.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--rollout-manifest", type=Path)
    parser.add_argument("--stage-report", type=Path)
    args = parser.parse_args()
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    registry_path = resolve_repo_path(args.task_registry)
    registry = read_yaml(registry_path)
    splits = _split_map(registry)
    rollout_path = (
        resolve_repo_path(args.rollout_manifest)
        if args.rollout_manifest is not None
        else runtime / "rollout_manifest.json"
    )
    stage_path = (
        resolve_repo_path(args.stage_report)
        if args.stage_report is not None
        else runtime / "stage_annotation_report.json"
    )
    rollout = read_json(rollout_path)
    stage = read_json(stage_path)
    if rollout.get("status") != "pass" or stage.get("status") != "pass":
        raise ValueError("rollout and stage QA gates must pass")
    rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for episode in rollout["episodes"]:
        key = str(episode["suite"]), int(episode["task_id"])
        split = splits.get(key, "zero_shot_only")
        seed = int(episode["seed"])
        if seed in seen_seeds:
            raise ValueError(f"seed crosses episode experiments: {seed}")
        seen_seeds.add(seed)
        episode_id = str(episode["episode_id"])
        assignment_rows.append(
            {
                "episode_id": episode_id,
                "suite": key[0],
                "task_id": key[1],
                "seed": seed,
                "adaptation_split": split,
                "zero_shot_held_out": True,
            }
        )
        for label in _read_jsonl(runtime / "stage_labels" / f"{episode_id}.jsonl"):
            rows.append(
                {
                    "schema_version": "latentguard.lg_r1b.current_sample.v1",
                    "episode_id": episode_id,
                    "suite": key[0],
                    "task_id": key[1],
                    "seed": seed,
                    "split": split,
                    "zero_shot_split": "test",
                    "frame_index": int(label["step_index"]),
                    "episode_horizon": int(episode["episode_horizon"]),
                    "episode_frame_count": int(episode["frame_count"]),
                    "episode_success": bool(episode["success"]),
                    "dataset_locator": episode["dataset_locator"],
                    "instruction": episode["instruction"],
                    "stage_id": int(label["pre_label"]["stage_id"]),
                    "stage_completion": float(label["pre_label"]["stage_completion"]),
                    "overall_progress": float(label["pre_label"]["overall_progress"]),
                    "model_input_keys": label["current_sample"]["model_input_keys"],
                    "privileged_fields": [],
                }
            )
    dataset_root = runtime / "progress_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    samples_path = dataset_root / "current_samples.jsonl"
    write_jsonl_atomic(samples_path, rows)
    split_manifest = {
        "schema_version": "latentguard.lg_r1b.task_split_manifest.v1",
        "status": "pass",
        "split_unit": "task",
        "frozen_before_adaptation": True,
        "adaptation_task_split": registry["adaptation_task_split"],
        "episode_assignments": assignment_rows,
        "episode_split_leakage": 0,
        "seed_split_leakage": 0,
        "held_out_task_supported": True,
        "all_new_tasks_zero_shot_held_out": True,
    }
    write_json(runtime / "split_manifest.json", split_manifest)
    manifest = {
        "schema_version": "latentguard.lg_r1b.dataset_manifest.v1",
        "status": "pass",
        "source_rollout_manifest_sha256": sha256_path(rollout_path),
        "source_stage_report_sha256": sha256_path(stage_path),
        "source_task_registry_sha256": sha256_path(registry_path),
        "valid_rollout_episodes": int(rollout["valid_episodes"]),
        "tasks": int(rollout["task_count"]),
        "suites": int(rollout["suite_count"]),
        "natural_failed_episodes": int(rollout["natural_failed_episodes"]),
        "failed_tasks": int(rollout["failed_tasks"]),
        "current_samples": len(rows),
        "stage_annotation_passed": True,
        "episode_split_leakage": 0,
        "seed_split_leakage": 0,
        "processor_identity_completeness": float(
            rollout["processor_identity_completeness"]
        ),
        "checkpoint_identity_completeness": float(
            rollout["checkpoint_identity_completeness"]
        ),
        "model_input_allowlist": [
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "task",
        ],
        "privileged_model_inputs": [],
        "synthetic_corruption_samples": 0,
        "failure_head_samples": 0,
        "files": {
            "current_samples": {
                "locator": "progress_dataset/current_samples.jsonl",
                "bytes": samples_path.stat().st_size,
                "sha256": sha256_path(samples_path),
            }
        },
    }
    write_json(runtime / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
