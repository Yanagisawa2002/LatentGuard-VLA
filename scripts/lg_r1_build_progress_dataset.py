"""Build episode-grouped current, pairwise, and action-window datasets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _lg_r1_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    sha256_path,
    write_json,
)

from latentguard.progress.dataset import (
    EpisodeAssignment,
    assert_no_split_leakage,
    assign_episode_split,
    build_progress_windows,
)
from latentguard.progress.models import StageLabel
from latentguard.progress.serialization import write_jsonl_atomic


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object rows in {path}")
            values.append(payload)
    if not values:
        raise ValueError(f"empty label file: {path}")
    return values


def _split_seeds(
    seeds: list[int],
    fractions: dict[str, Any],
    *,
    split_seed: int,
) -> dict[str, set[int]]:
    ordered = sorted(set(seeds))
    if len(ordered) != len(seeds):
        raise ValueError("the episode schedule duplicates a seed/task record")
    unique = sorted(set(seeds))
    random.Random(split_seed).shuffle(unique)
    train_count = int(len(unique) * float(fractions["train"]))
    validation_count = int(len(unique) * float(fractions["validation"]))
    if train_count < 1 or validation_count < 1:
        raise ValueError("too few seeds for train/validation/test")
    return {
        "train": set(unique[:train_count]),
        "validation": set(unique[train_count : train_count + validation_count]),
        "test": set(unique[train_count + validation_count :]),
    }


def _label(value: dict[str, Any]) -> StageLabel:
    return StageLabel(
        stage_id=int(value["stage_id"]),
        stage_name=str(value["stage_name"]),
        stage_completion=float(value["stage_completion"]),
        overall_progress=float(value["overall_progress"]),
        terminal_success=bool(value["terminal_success"]),
        terminal_failure=bool(value["terminal_failure"]),
        evidence=dict(value["evidence"]),
    )


def main() -> None:
    """Build all dataset views only after rollout and stage gates pass."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/dataset.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rollout-manifest", type=Path)
    parser.add_argument("--stage-report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    runtime = (
        resolve_repo_path(args.output_root)
        if args.output_root is not None
        else output_root()
    )
    rollout_path = (
        resolve_repo_path(args.rollout_manifest)
        if args.rollout_manifest is not None
        else runtime / "rollout_manifest.json"
    )
    report_path = (
        resolve_repo_path(args.stage_report)
        if args.stage_report is not None
        else runtime / "stage_annotation_report.json"
    )
    rollout = read_json(rollout_path)
    stage_report = read_json(report_path)
    if rollout.get("status") != "pass":
        raise ValueError("rollout data gate is incomplete")
    if stage_report.get("status") != "pass":
        raise ValueError("stage QA must pass before dataset construction")
    seed_values = sorted({int(episode["seed"]) for episode in rollout["episodes"]})
    seed_splits = _split_seeds(
        seed_values,
        config["split_fractions"],
        split_seed=int(config["split_seed"]),
    )
    assignments: list[EpisodeAssignment] = []
    for episode in rollout["episodes"]:
        assignments.append(
            assign_episode_split(
                episode_id=str(episode["episode_id"]),
                suite=str(episode["suite"]),
                task_id=int(episode["task_id"]),
                seed=int(episode["seed"]),
                train_seeds=seed_splits["train"],
                validation_seeds=seed_splits["validation"],
                test_seeds=seed_splits["test"],
            )
        )
    assert_no_split_leakage(assignments)
    assignment_by_id = {assignment.episode_id: assignment for assignment in assignments}
    current_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    failure_windows = 0
    successful_windows = 0
    episode_by_id = {
        str(episode["episode_id"]): episode for episode in rollout["episodes"]
    }
    for episode_id, assignment in sorted(assignment_by_id.items()):
        episode = episode_by_id[episode_id]
        rows = _read_jsonl(runtime / "stage_labels" / f"{episode_id}.jsonl")
        labels = [_label(row["pre_label"]) for row in rows]
        for row in rows:
            current_rows.append(
                {
                    "schema_version": ("latentguard.lg_r1.current_sample.v1"),
                    "episode_id": episode_id,
                    "suite": assignment.suite,
                    "task_id": assignment.task_id,
                    "seed": assignment.seed,
                    "split": assignment.split,
                    "frame_index": int(row["step_index"]),
                    "episode_horizon": int(episode["episode_horizon"]),
                    "episode_frame_count": int(episode["frame_count"]),
                    "episode_success": bool(episode["success"]),
                    "dataset_locator": episode["dataset_locator"],
                    "instruction": episode["instruction"],
                    "stage_id": int(row["pre_label"]["stage_id"]),
                    "stage_completion": float(row["pre_label"]["stage_completion"]),
                    "overall_progress": float(row["pre_label"]["overall_progress"]),
                    "model_input_keys": row["current_sample"]["model_input_keys"],
                    "privileged_fields": [],
                }
            )
        windows = build_progress_windows(
            episode_id=episode_id,
            split=assignment.split,
            labels=labels,
            window_steps={
                str(key): int(value) for key, value in config["window_steps"].items()
            },
            stride=int(config["frame_stride"]),
            progress_epsilon=0.02,
        )
        for window in windows:
            base = {
                **asdict(window),
                "suite": assignment.suite,
                "task_id": assignment.task_id,
                "seed": assignment.seed,
                "instruction": episode["instruction"],
                "start_frame_locator": (
                    f"{episode['dataset_locator']}#frame={window.start_index}"
                ),
                "end_frame_locator": (
                    f"{episode['dataset_locator']}#frame={window.end_index}"
                ),
                "pair_target": (
                    1
                    if window.progress_delta > 0.02
                    else -1
                    if window.progress_delta < -0.02
                    else 0
                ),
                "privileged_fields": [],
            }
            pairwise_rows.append(
                {
                    "schema_version": ("latentguard.lg_r1.pairwise_sample.v1"),
                    **base,
                }
            )
            actions = [
                row["executed_action_window"]
                for row in rows[window.start_index : window.end_index]
            ]
            action_rows.append(
                {
                    "schema_version": ("latentguard.lg_r1.action_window.v1"),
                    **base,
                    "executed_actions": [item["executed_action"] for item in actions],
                    "action_masks": [item["action_mask"] for item in actions],
                    "action_is_pad": [item["action_is_pad"] for item in actions],
                }
            )
            if bool(episode["success"]):
                successful_windows += 1
            else:
                failure_windows += 1
    dataset_root = runtime / "progress_dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    files = {
        "current_samples": dataset_root / "current_samples.jsonl",
        "pairwise_samples": dataset_root / "pairwise_samples.jsonl",
        "action_windows": dataset_root / "action_windows.jsonl",
    }
    write_jsonl_atomic(files["current_samples"], current_rows)
    write_jsonl_atomic(files["pairwise_samples"], pairwise_rows)
    write_jsonl_atomic(files["action_windows"], action_rows)
    split_manifest = {
        "schema_version": "latentguard.lg_r1.split_manifest.v1",
        "status": "pass",
        "split_seed": int(config["split_seed"]),
        "seed_assignments": {
            name: sorted(values) for name, values in seed_splits.items()
        },
        "episode_assignments": [asdict(assignment) for assignment in assignments],
        "episode_split_leakage": 0,
        "seed_split_leakage": 0,
        "held_out_seed_supported": True,
        "held_out_task_supported": False,
        "held_out_suite_supported": False,
        "unsupported_reason": (
            "Every frozen seed is shared by every task, so a disjoint "
            "held-out task or suite would violate seed exclusivity."
        ),
    }
    split_path = (
        resolve_repo_path(args.split_manifest)
        if args.split_manifest is not None
        else runtime / "split_manifest.json"
    )
    write_json(split_path, split_manifest)
    manifest = {
        "schema_version": "latentguard.lg_r1.dataset_manifest.v1",
        "status": "pass",
        "source_rollout_manifest_sha256": sha256_path(rollout_path),
        "source_stage_report_sha256": sha256_path(report_path),
        "source_config_sha256": sha256_path(config_path),
        "valid_rollout_episodes": int(rollout["valid_episodes"]),
        "tasks": int(rollout["task_count"]),
        "suites": int(rollout["suite_count"]),
        "natural_failed_episodes": int(rollout["natural_failed_episodes"]),
        "current_samples": len(current_rows),
        "pairwise_windows": len(pairwise_rows),
        "executed_action_windows": len(action_rows),
        "failure_windows": failure_windows,
        "successful_windows": successful_windows,
        "stage_annotation_passed": True,
        "episode_split_leakage": 0,
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
        "files": {
            name: {
                "locator": f"progress_dataset/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            for name, path in files.items()
        },
    }
    manifest_path = (
        resolve_repo_path(args.manifest)
        if args.manifest is not None
        else runtime / "dataset_manifest.json"
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
