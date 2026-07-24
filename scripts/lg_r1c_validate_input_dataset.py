"""Bind and optionally reload-validate the frozen LG-R1b input dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1c_common import (
    file_identity,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    runtime_root,
    validate_no_final_seeds,
    validate_repository_lineage,
    write_json,
)


def _tracked_identities(config: dict[str, Any]) -> dict[str, Any]:
    sources = config.get("source_artifacts")
    if not isinstance(sources, dict):
        raise ValueError("source_artifacts must be a mapping")
    identities: dict[str, Any] = {}
    for name, locator in sources.items():
        path = resolve_repo_path(Path(str(locator)))
        if not path.is_file():
            raise FileNotFoundError(path)
        identities[str(name)] = file_identity(path, locator=str(locator))
    return identities


def _validate_counts(
    config: dict[str, Any],
    rollout: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, int]:
    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("expected counts must be a mapping")
    observed = {
        "episodes": int(rollout["valid_episodes"]),
        "tasks": int(rollout["task_count"]),
        "suites": int(rollout["suite_count"]),
        "successes": int(rollout["valid_episodes"])
        - int(rollout["natural_failed_episodes"]),
        "failures": int(rollout["natural_failed_episodes"]),
        "failure_producing_tasks": int(rollout["failed_tasks"]),
        "frames": int(dataset["current_samples"]),
        "failure_windows": int(dataset["failure_windows"]),
        "matched_success_windows": int(dataset["matched_success_windows"]),
        "episode_leakage": int(dataset["episode_split_leakage"]),
        "seed_leakage": int(dataset["seed_split_leakage"]),
    }
    drift = {
        key: {"expected": int(expected[key]), "observed": value}
        for key, value in observed.items()
        if value != int(expected[key])
    }
    if drift:
        raise ValueError(f"LG-R1b input identity drift: {drift}")
    return observed


def _runtime_validation(root: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    current_path = root / "progress_dataset" / "current_samples.jsonl"
    failure_path = root / "failure_windows" / "failure_windows.jsonl"
    matched_path = root / "failure_windows" / "matched_success_windows.jsonl"
    expected_files = dataset.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("dataset manifest files must be a mapping")
    current_identity = file_identity(
        current_path,
        locator="progress_dataset/current_samples.jsonl",
    )
    expected_current = expected_files.get("current_samples")
    if not isinstance(expected_current, dict):
        raise ValueError("current_samples identity missing")
    if current_identity["sha256"] != expected_current.get("sha256"):
        raise ValueError("runtime current_samples hash mismatch")
    current_rows = read_jsonl(current_path)
    failure_rows = read_jsonl(failure_path)
    matched_rows = read_jsonl(matched_path)
    seeds = [int(row["seed"]) for row in current_rows]
    validate_no_final_seeds(seeds)
    episodes = {str(row["episode_id"]) for row in current_rows}
    successes = {
        str(row["episode_id"]) for row in current_rows if bool(row["episode_success"])
    }
    failures = episodes - successes
    splits_by_episode: dict[str, set[str]] = {}
    seeds_by_split: dict[str, set[int]] = {}
    for row in current_rows:
        episode_id = str(row["episode_id"])
        split = str(row["split"])
        splits_by_episode.setdefault(episode_id, set()).add(split)
        seeds_by_split.setdefault(split, set()).add(int(row["seed"]))
    episode_leakage = sum(len(values) > 1 for values in splits_by_episode.values())
    split_names = sorted(seeds_by_split)
    seed_leakage = sum(
        len(seeds_by_split[left] & seeds_by_split[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    )
    if episode_leakage or seed_leakage:
        raise ValueError("runtime episode or seed split leakage detected")
    return {
        "status": "pass",
        "current_samples": len(current_rows),
        "episodes": len(episodes),
        "successes": len(successes),
        "failures": len(failures),
        "failure_windows": len(failure_rows),
        "matched_success_windows": len(matched_rows),
        "episode_leakage": episode_leakage,
        "seed_leakage": seed_leakage,
        "sealed_final_seed_overlap": [],
        "files": {
            "current_samples": current_identity,
            "failure_windows": file_identity(
                failure_path,
                locator="failure_windows/failure_windows.jsonl",
            ),
            "matched_success_windows": file_identity(
                matched_path,
                locator="failure_windows/matched_success_windows.jsonl",
            ),
        },
    }


def build_manifest(
    config_path: Path,
    *,
    runtime: Path | None,
) -> dict[str, Any]:
    """Build the content-bound frozen input manifest."""

    config = read_yaml(resolve_repo_path(config_path))
    if config.get("schema_version") != "latentguard.lg_r1c.dataset.v1":
        raise ValueError("unexpected LG-R1c dataset schema")
    sources = config["source_artifacts"]
    rollout = read_json(resolve_repo_path(Path(str(sources["rollout_manifest"]))))
    dataset = read_json(resolve_repo_path(Path(str(sources["dataset_manifest"]))))
    counts = _validate_counts(config, rollout, dataset)
    episodes = rollout.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("rollout manifest episodes must be a list")
    validate_no_final_seeds(int(item["seed"]) for item in episodes)
    runtime_result: dict[str, Any] = {"status": "not_run"}
    if runtime is not None:
        runtime_result = _runtime_validation(runtime, dataset)
        for key in (
            "episodes",
            "successes",
            "failures",
            "current_samples",
            "failure_windows",
            "matched_success_windows",
            "episode_leakage",
            "seed_leakage",
        ):
            expected_key = "frames" if key == "current_samples" else key
            if int(runtime_result[key]) != int(counts[expected_key]):
                raise ValueError(f"runtime count drift for {key}")
    return {
        "schema_version": "latentguard.lg_r1c.input_dataset_manifest.v1",
        "status": "pass",
        "repository": validate_repository_lineage(),
        "source_lg_r1b_commit": config["baseline_commit"],
        "counts": counts,
        "source_artifacts": _tracked_identities(config),
        "runtime_reload_validation": runtime_result,
        "original_split_reused": True,
        "resplit_performed": False,
        "new_rollouts": 0,
        "synthetic_failures": 0,
        "sealed_final_seed_overlap": [],
    }


def main() -> None:
    """Validate the frozen input data locally or against the remote runtime."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/dataset.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1c/input_dataset_manifest.json"),
    )
    args = parser.parse_args()
    runtime = runtime_root(args.runtime_root) if args.runtime_root is not None else None
    payload = build_manifest(args.config, runtime=runtime)
    write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
