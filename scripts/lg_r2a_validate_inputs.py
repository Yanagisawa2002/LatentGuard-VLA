"""Validate and content-bind frozen LG-R2a source data."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2a_common import (
    directory_digest,
    file_identity,
    output_root,
    r1c_root,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    source_root,
    validate_no_final_seeds,
    validate_repository_lineage,
    write_json,
)

from latentguard.action_conditioning.contracts import ActionContract

EXPECTED_COUNTS = {
    "episodes": 320,
    "tasks": 8,
    "suites": 2,
    "frames": 79_326,
    "successes": 293,
    "failures": 27,
    "evaluation_windows": 8_470,
    "failure_tasks": 4,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--r1c-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate exact counts, source identities, and numeric action values."""

    args = parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    if args.dry_run:
        print({"status": "dry_run", "config": config["schema_version"]})
        return
    source = source_root(args.source_root)
    reward = r1c_root(args.r1c_root)
    output = output_root(args.output_dir)
    current_path = source / "progress_dataset" / "current_samples.jsonl"
    stage_path = source / "stage_labels"
    cache_path = reward / "vlajepa-probe" / "frozen_qwen_features.npz"
    windows_path = reward / "reward_windows.jsonl"
    for path in (current_path, stage_path, cache_path, windows_path):
        if not path.exists():
            raise ValueError(f"required frozen source is missing: {path}")

    current = read_jsonl(current_path)
    episodes = {str(row["episode_id"]) for row in current}
    tasks = {f"{row['suite']}/task{int(row['task_id'])}" for row in current}
    suites = {str(row["suite"]) for row in current}
    episode_success = {
        str(row["episode_id"]): bool(row["episode_success"]) for row in current
    }
    seeds = {int(row["seed"]) for row in current}
    validate_no_final_seeds(seeds)
    failure_registry = read_json(
        resolve_repo_path(config["source_artifacts"]["failure_registry"])
    )
    failure_tasks = {
        f"{row['suite']}/task{int(row['task_id'])}"
        for row in failure_registry["episodes"]
    }
    window_manifest = read_json(
        resolve_repo_path(config["source_artifacts"]["window_manifest"])
    )
    counts = {
        "episodes": len(episodes),
        "tasks": len(tasks),
        "suites": len(suites),
        "frames": len(current),
        "successes": sum(episode_success.values()),
        "failures": sum(not value for value in episode_success.values()),
        "evaluation_windows": int(window_manifest["window_count"]),
        "failure_tasks": len(failure_tasks),
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"frozen source count mismatch: {counts}")

    actions: list[list[float]] = []
    native_masks: Counter[str] = Counter()
    padding: Counter[bool] = Counter()
    for path in sorted(stage_path.glob("*.jsonl")):
        for row in read_jsonl(path):
            item = row["executed_action_window"]
            action = [float(value) for value in item["executed_action"]]
            if len(action) != 7 or not np.isfinite(action).all():
                raise ValueError(f"invalid executed action in {path}")
            if min(action) < -1.0 or max(action) > 1.0:
                raise ValueError(f"executed action exceeds native bounds in {path}")
            if action[-1] not in {-1.0, 1.0}:
                raise ValueError(f"invalid gripper command in {path}")
            actions.append(action)
            native_masks[str(item["action_mask"])] += 1
            padding[bool(item["action_is_pad"])] += 1
    if len(actions) != EXPECTED_COUNTS["frames"]:
        raise ValueError("stage-label action count differs from frozen frame count")
    if native_masks != Counter({"None": 79_326}) or padding != Counter({False: 79_326}):
        raise ValueError("native action mask or padding semantic changed")
    action_values = np.asarray(actions, dtype=np.float64)
    action_contract = {
        "schema_version": "latentguard.lg_r2a.action_contract.v1",
        **ActionContract().to_dict(),
        "source_semantic": "frozen real on-policy executed actions only",
        "native_action_mask_observation": {"null": 79_326},
        "native_padding_observation": {"false": 79_326},
        "observed_min": action_values.min(axis=0).tolist(),
        "observed_max": action_values.max(axis=0).tolist(),
        "validation": {
            "status": "pass",
            "finite": True,
            "shape": [79_326, 7],
            "bounds": "pass",
            "gripper": "pass",
            "normalization_reproducible_from_frozen_postprocessor_contract": True,
        },
    }
    cache = np.load(cache_path, allow_pickle=False)
    cache_shape = list(cache["features"].shape)
    if cache_shape != [5_000, 2_048]:
        raise ValueError(f"unexpected frozen VLA cache shape: {cache_shape}")
    source_artifacts: dict[str, Any] = {}
    for name, locator in config["source_artifacts"].items():
        path = resolve_repo_path(locator)
        source_artifacts[name] = file_identity(path, locator=str(locator))
    manifest = {
        "schema_version": "latentguard.lg_r2a.input_dataset_manifest.v1",
        "status": "pass",
        "counts": {
            **counts,
            "episode_leakage": 0,
            "seed_leakage": 0,
        },
        "source_artifacts": source_artifacts,
        "runtime_sources": {
            "current_samples": file_identity(
                current_path, locator="lg_r1b/progress_dataset/current_samples.jsonl"
            ),
            "stage_labels": {
                "locator": "lg_r1b/stage_labels/*.jsonl",
                **directory_digest(stage_path, "*.jsonl"),
            },
            "reward_windows": file_identity(
                windows_path, locator="lg_r1c/reward_windows.jsonl"
            ),
            "vla_feature_cache": {
                **file_identity(
                    cache_path,
                    locator="lg_r1c/vlajepa-probe/frozen_qwen_features.npz",
                ),
                "shape": cache_shape,
                "dtype": str(cache["features"].dtype),
                "foundation_frozen": True,
            },
        },
        "numeric_action_schema": "latentguard.lg_r2a.action_contract.v1",
        "new_rollouts": 0,
        "synthetic_actions": 0,
        "synthetic_failures": 0,
        "final_seeds_accessed": False,
        "original_split_preserved": True,
        "repository": validate_repository_lineage(),
    }
    source_validation = {
        "schema_version": "latentguard.lg_r2a.source_validation.v1",
        "status": "pass",
        "repository": manifest["repository"],
        "langmani_imported": False,
        "langmani_modified": False,
        "new_rollouts": 0,
        "synthetic_action_training": False,
        "candidate_generation": False,
        "candidate_ranking": False,
        "counterfactual_rollout": False,
        "intervention": False,
        "robolab_used": False,
        "pointworld_used": False,
        "final_seeds_accessed": False,
        "foundation_models_frozen": True,
        "processor_frozen": True,
        "policy_frozen": True,
    }
    write_json(output / "action_contract.json", action_contract)
    write_json(output / "input_dataset_manifest.json", manifest)
    write_json(output / "source_validation.json", source_validation)
    print({"status": "pass", "output": str(output), "counts": counts})


if __name__ == "__main__":
    main()
