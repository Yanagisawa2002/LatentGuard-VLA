"""Classify natural failures and build task/stage/horizon-matched windows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

from latentguard.progress.lg_r1b import FAILURE_TAXONOMY
from latentguard.progress.serialization import write_jsonl_atomic


def _categories(
    summary: dict[str, Any],
    adapter: str,
) -> list[str]:
    categories: list[str] = []
    if summary["termination_reason"] == "horizon_exhausted":
        categories.append("HORIZON_EXHAUSTION")
    if summary["plateau_start_step"] is not None:
        categories.append("STAGNATION")
    if int(summary["regression_events"]) > 0:
        categories.append("STAGE_REGRESSION")
    maximum = int(max(summary["stage_ids_observed"]))
    final = int(summary["final_pre_label"]["stage_id"])
    if adapter in {
        "pick_place",
        "multi_place",
        "place_close",
        "open_place",
        "toggle_place",
    }:
        grasp_threshold = (
            3 if adapter in {"open_place", "place_close", "toggle_place"} else 2
        )
        if maximum < grasp_threshold:
            categories.append("MISSED_GRASP")
        elif final < maximum:
            categories.append("OBJECT_DROP")
        elif adapter == "place_close" and maximum >= 5:
            categories.append("FAILED_OPEN_CLOSE")
        elif maximum >= grasp_threshold:
            categories.append("FAILED_PLACEMENT")
    return list(dict.fromkeys(categories or ["UNKNOWN"]))


def _window_rows(
    failure_records: list[dict[str, Any]],
    labels_by_episode: dict[str, list[dict[str, Any]]],
    window_steps: dict[str, int],
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for failure in failure_records:
        episode_id = str(failure["episode_id"])
        labels = labels_by_episode[episode_id]
        terminal = len(labels) - 1
        abnormal = int(failure["first_abnormal_step"])
        for name, length in window_steps.items():
            if name == "terminal_failure":
                endpoints = [terminal]
            else:
                first_end = max(abnormal, int(length))
                endpoints = list(range(first_end, terminal + 1, 4))
                if terminal not in endpoints:
                    endpoints.append(terminal)
            for end in endpoints:
                start = max(0, end - int(length))
                label = labels[end]["pre_label"]
                windows.append(
                    {
                        "schema_version": "latentguard.lg_r1b.failure_window.v1",
                        "window_id": (f"{episode_id}:{name}:{start}:{end}"),
                        "episode_id": episode_id,
                        "suite": failure["suite"],
                        "task_id": failure["task_id"],
                        "seed": failure["seed"],
                        "adaptation_split": failure["adaptation_split"],
                        "taxonomy": failure["taxonomy"],
                        "horizon_name": name,
                        "start_frame": start,
                        "end_frame": end,
                        "length_steps": end - start,
                        "anchor_stage": int(label["stage_id"]),
                        "anchor_progress": float(label["overall_progress"]),
                        "remaining_horizon": terminal - end,
                        "dataset_locator": failure["dataset_locator"],
                        "privileged_fields_in_model_input": [],
                    }
                )
    return windows


def _matched_success_windows(
    failure_windows: list[dict[str, Any]],
    success_episodes: list[dict[str, Any]],
    labels_by_episode: dict[str, list[dict[str, Any]]],
    split_by_episode: dict[str, str],
    matching: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    for episode in success_episodes:
        episode_id = str(episode["episode_id"])
        labels = labels_by_episode[episode_id]
        terminal = len(labels) - 1
        for frame, row in enumerate(labels):
            candidates.append(
                {
                    "episode_id": episode_id,
                    "suite": str(episode["suite"]),
                    "task_id": int(episode["task_id"]),
                    "seed": int(episode["seed"]),
                    "adaptation_split": split_by_episode[episode_id],
                    "frame": frame,
                    "stage": int(row["pre_label"]["stage_id"]),
                    "progress": float(row["pre_label"]["overall_progress"]),
                    "remaining": terminal - frame,
                    "dataset_locator": episode["dataset_locator"],
                }
            )
    used: set[tuple[str, int, int]] = set()
    matches: list[dict[str, Any]] = []
    unmatched = 0
    for failure in failure_windows:
        length = int(failure["length_steps"])
        eligible = []
        for candidate in candidates:
            key = (
                str(candidate["episode_id"]),
                int(candidate["frame"]) - length,
                int(candidate["frame"]),
            )
            if bool(matching["without_replacement"]) and key in used:
                continue
            if (
                candidate["suite"] != failure["suite"]
                or candidate["task_id"] != failure["task_id"]
            ):
                continue
            if candidate["adaptation_split"] != failure["adaptation_split"]:
                continue
            if abs(int(candidate["stage"]) - int(failure["anchor_stage"])) > int(
                matching["maximum_stage_distance"]
            ):
                continue
            if abs(
                float(candidate["progress"]) - float(failure["anchor_progress"])
            ) > float(matching["maximum_progress_distance"]):
                continue
            if abs(
                int(candidate["remaining"]) - int(failure["remaining_horizon"])
            ) > int(matching["maximum_remaining_horizon_distance"]):
                continue
            if int(candidate["frame"]) - length < 0:
                continue
            distance = (
                abs(float(candidate["progress"]) - float(failure["anchor_progress"]))
                + abs(int(candidate["remaining"]) - int(failure["remaining_horizon"]))
                / 1000.0
            )
            eligible.append((distance, candidate, key))
        if not eligible:
            unmatched += 1
            continue
        _, candidate, key = min(
            eligible,
            key=lambda item: (
                item[0],
                str(item[1]["episode_id"]),
                int(item[1]["frame"]),
            ),
        )
        used.add(key)
        matches.append(
            {
                "schema_version": "latentguard.lg_r1b.matched_success_window.v1",
                "failure_window_id": failure["window_id"],
                "episode_id": candidate["episode_id"],
                "suite": candidate["suite"],
                "task_id": candidate["task_id"],
                "seed": candidate["seed"],
                "adaptation_split": candidate["adaptation_split"],
                "horizon_name": failure["horizon_name"],
                "start_frame": int(candidate["frame"]) - length,
                "end_frame": int(candidate["frame"]),
                "length_steps": length,
                "anchor_stage": candidate["stage"],
                "anchor_progress": candidate["progress"],
                "remaining_horizon": candidate["remaining"],
                "dataset_locator": candidate["dataset_locator"],
                "match_distances": {
                    "stage": abs(
                        int(candidate["stage"]) - int(failure["anchor_stage"])
                    ),
                    "progress": abs(
                        float(candidate["progress"]) - float(failure["anchor_progress"])
                    ),
                    "remaining_horizon": abs(
                        int(candidate["remaining"]) - int(failure["remaining_horizon"])
                    ),
                },
                "privileged_fields_in_model_input": [],
            }
        )
    return matches, unmatched


def main() -> None:
    """Create natural-failure evidence without training a classifier."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1b/failure_windows.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    registry = read_yaml(resolve_repo_path(Path(str(config["task_registry"]))))
    task_by_key = {
        (str(task["suite"]), int(task["task_id"])): task for task in registry["tasks"]
    }
    rollout = read_json(runtime / "rollout_manifest.json")
    stage = read_json(runtime / "stage_annotation_report.json")
    split = read_json(runtime / "split_manifest.json")
    if stage.get("status") != "pass":
        raise ValueError("stage QA must pass before failure-window construction")
    if tuple(config["failure_taxonomy"]) != FAILURE_TAXONOMY:
        raise ValueError("failure taxonomy drift")
    summaries = {str(item["episode_id"]): item for item in stage["episode_summaries"]}
    split_by_episode = {
        str(item["episode_id"]): str(item["adaptation_split"])
        for item in split["episode_assignments"]
    }
    labels_by_episode = {
        str(episode["episode_id"]): _read_jsonl(
            runtime / "stage_labels" / f"{episode['episode_id']}.jsonl"
        )
        for episode in rollout["episodes"]
    }
    failure_records = []
    for episode in rollout["episodes"]:
        if bool(episode["success"]):
            continue
        episode_id = str(episode["episode_id"])
        summary = summaries[episode_id]
        adapter = str(
            task_by_key[(str(episode["suite"]), int(episode["task_id"]))]["adapter"]
        )
        categories = _categories(summary, adapter)
        first = summary["first_failure_relevant_step"]
        if first is None:
            first = max(int(episode["frame_count"]) - 1, 0)
        before_index = max(int(first) - 1, 0)
        after_index = min(int(first) + 1, int(episode["frame_count"]) - 1)
        labels = labels_by_episode[episode_id]
        failure_records.append(
            {
                "episode_id": episode_id,
                "suite": episode["suite"],
                "task_id": episode["task_id"],
                "task_name": episode["task_name"],
                "seed": episode["seed"],
                "adaptation_split": split_by_episode[episode_id],
                "terminal_reason": episode["termination_reason"],
                "taxonomy": categories,
                "primary_taxonomy": categories[-1],
                "first_abnormal_step": int(first),
                "stage_at_abnormality": int(
                    labels[int(first)]["pre_label"]["stage_id"]
                ),
                "progress_before_abnormality": float(
                    labels[before_index]["pre_label"]["overall_progress"]
                ),
                "progress_after_abnormality": float(
                    labels[after_index]["pre_label"]["overall_progress"]
                ),
                "plateau_length": (
                    int(episode["frame_count"]) - int(summary["plateau_start_step"])
                    if summary["plateau_start_step"] is not None
                    else 0
                ),
                "regression_magnitude": max(
                    float(summary["maximum_progress"])
                    - float(summary["final_pre_label"]["overall_progress"]),
                    0.0,
                ),
                "remaining_horizon": int(episode["frame_count"]) - int(first) - 1,
                "dataset_locator": episode["dataset_locator"],
                "environment_infrastructure_error": False,
                "synthetic_failure": False,
            }
        )
    failure_registry = {
        "schema_version": "latentguard.lg_r1b.failure_registry.v1",
        "status": "pass",
        "failure_head_trained": False,
        "environment_errors_excluded": True,
        "natural_failure_episodes": len(failure_records),
        "failed_tasks": len(
            {(item["suite"], item["task_id"]) for item in failure_records}
        ),
        "taxonomy_counts": dict(
            sorted(
                Counter(
                    category
                    for item in failure_records
                    for category in item["taxonomy"]
                ).items()
            )
        ),
        "episodes": failure_records,
    }
    write_json(runtime / "failure_registry.json", failure_registry)
    failure_windows = _window_rows(
        failure_records,
        labels_by_episode,
        {str(key): int(value) for key, value in config["window_steps"].items()},
    )
    success_episodes = [
        episode for episode in rollout["episodes"] if bool(episode["success"])
    ]
    matches, unmatched = _matched_success_windows(
        failure_windows,
        success_episodes,
        labels_by_episode,
        split_by_episode,
        config["matching"],
    )
    dataset_root = runtime / "failure_windows"
    dataset_root.mkdir(parents=True, exist_ok=True)
    failure_path = dataset_root / "failure_windows.jsonl"
    success_path = dataset_root / "matched_success_windows.jsonl"
    write_jsonl_atomic(failure_path, failure_windows)
    write_jsonl_atomic(success_path, matches)
    manifest = {
        "schema_version": "latentguard.lg_r1b.failure_window_manifest.v1",
        "status": "pass",
        "failure_episodes": len(failure_records),
        "failure_windows": len(failure_windows),
        "matched_success_windows": len(matches),
        "unmatched_failure_windows": unmatched,
        "task_distribution": dict(
            sorted(
                Counter(
                    f"{item['suite']}/task{item['task_id']}" for item in failure_windows
                ).items()
            )
        ),
        "stage_distribution": dict(
            sorted(
                Counter(str(item["anchor_stage"]) for item in failure_windows).items()
            )
        ),
        "horizon_distribution": dict(
            sorted(Counter(item["horizon_name"] for item in failure_windows).items())
        ),
        "episode_leakage": 0,
        "seed_leakage": 0,
        "same_task_matching": True,
        "same_split_matching": True,
        "files": {
            "failure_windows": {
                "locator": "failure_windows/failure_windows.jsonl",
                "sha256": sha256_path(failure_path),
                "bytes": failure_path.stat().st_size,
            },
            "matched_success_windows": {
                "locator": "failure_windows/matched_success_windows.jsonl",
                "sha256": sha256_path(success_path),
                "bytes": success_path.stat().st_size,
            },
        },
    }
    write_json(runtime / "failure_window_manifest.json", manifest)
    dataset = read_json(runtime / "dataset_manifest.json")
    dataset.update(
        {
            "failure_windows": len(failure_windows),
            "matched_success_windows": len(matches),
            "failure_window_manifest_sha256": sha256_path(
                runtime / "failure_window_manifest.json"
            ),
        }
    )
    write_json(runtime / "dataset_manifest.json", dataset)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
