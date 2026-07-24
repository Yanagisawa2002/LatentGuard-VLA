"""Build deterministic common and matched LG-R1c reward windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _lg_r1c_common import (
    file_identity,
    output_root,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    runtime_root,
    validate_no_final_seeds,
    write_json,
    write_jsonl,
)


def _uniform_indices(start: int, end: int, count: int) -> list[int]:
    if start < 0 or end < start or count < 1:
        raise ValueError("invalid inclusive uniform-index request")
    if count == 1:
        return [end]
    extent = end - start
    return [
        start + (extent * index + (count - 1) // 2) // (count - 1)
        for index in range(count)
    ]


def _sample(
    row: dict[str, Any],
    *,
    start: int,
    end: int,
    frame_count: int,
) -> dict[str, Any]:
    indices = _uniform_indices(start, end, frame_count)
    progress_by_frame = row["progress_by_frame"]
    stage_by_frame = row["stage_by_frame"]
    return {
        "frame_indices": indices,
        "frame_progress_targets": [
            float(progress_by_frame[index]) for index in indices
        ],
        "frame_stage_targets": [int(stage_by_frame[index]) for index in indices],
    }


def _episode_rows(current: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current:
        grouped[str(row["episode_id"])].append(row)
    episodes: dict[str, dict[str, Any]] = {}
    for episode_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: int(item["frame_index"]))
        expected = list(range(len(ordered)))
        observed = [int(item["frame_index"]) for item in ordered]
        if observed != expected:
            raise ValueError(f"non-contiguous current samples for {episode_id}")
        first = ordered[0]
        episodes[episode_id] = {
            "episode_id": episode_id,
            "suite": str(first["suite"]),
            "task_id": int(first["task_id"]),
            "seed": int(first["seed"]),
            "split": str(first["split"]),
            "instruction": str(first["instruction"]),
            "episode_horizon": int(first["episode_horizon"]),
            "episode_frame_count": len(ordered),
            "episode_success": bool(first["episode_success"]),
            "progress_by_frame": [float(item["overall_progress"]) for item in ordered],
            "stage_by_frame": [int(item["stage_id"]) for item in ordered],
        }
    validate_no_final_seeds(int(item["seed"]) for item in episodes.values())
    return episodes


def _base_window(
    episode: dict[str, Any],
    *,
    window_id: str,
    purpose: str,
    context: str,
    start: int,
    end: int,
    sampled_frames: int,
) -> dict[str, Any]:
    sample = _sample(
        episode,
        start=start,
        end=end,
        frame_count=sampled_frames,
    )
    return {
        "schema_version": "latentguard.lg_r1c.reward_window.v1",
        "window_id": window_id,
        "purpose": purpose,
        "context": context,
        "episode_id": episode["episode_id"],
        "suite": episode["suite"],
        "task_id": episode["task_id"],
        "task_key": f"{episode['suite']}/task{episode['task_id']}",
        "seed": episode["seed"],
        "split": episode["split"],
        "instruction": episode["instruction"],
        "start_frame": start,
        "end_frame": end,
        "episode_frame_count": episode["episode_frame_count"],
        "episode_horizon": episode["episode_horizon"],
        "terminal_success": episode["episode_success"],
        "progress_target": float(episode["progress_by_frame"][end]),
        "stage_target": int(episode["stage_by_frame"][end]),
        "remaining_horizon": int(episode["episode_horizon"]) - end - 1,
        "time_to_episode_end": int(episode["episode_frame_count"]) - end - 1,
        "camera_key": "observation.images.image",
        "common_frame_indices": sample["frame_indices"],
        "frame_progress_targets": sample["frame_progress_targets"],
        "frame_stage_targets": sample["frame_stage_targets"],
        "future_observation_used_for_score": True,
        "candidate_action_conditioned": False,
    }


def _anchor_windows(
    episodes: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    fractions = config["anchor_fractions"]
    contexts = config["contexts"]
    rows: list[dict[str, Any]] = []
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        maximum = int(episode["episode_frame_count"]) - 1
        endpoints = sorted(
            {
                min(max(round(float(fraction) * maximum), 0), maximum)
                for fraction in fractions
            }
        )
        if len(endpoints) != len(fractions):
            raise ValueError(f"anchor endpoint collision for {episode_id}")
        for endpoint_ordinal, end in enumerate(endpoints):
            for context_name in ("short", "medium", "long"):
                context = contexts[context_name]
                span = int(context["span_steps"])
                sampled = int(context["common_sampled_frames"])
                start = max(0, end - span + 1)
                window = _base_window(
                    episode,
                    window_id=(
                        f"anchor:{episode_id}:{endpoint_ordinal}:{context_name}"
                    ),
                    purpose="anchor_grid",
                    context=context_name,
                    start=start,
                    end=end,
                    sampled_frames=sampled,
                )
                if context_name == "long":
                    native = int(context["topreward_native_diagnostic_frames"])
                    window["topreward_native_frame_indices"] = _uniform_indices(
                        start,
                        end,
                        native,
                    )
                rows.append(window)
    return rows


def _matched_windows(
    episodes: dict[str, dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    success_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    failure_by_id = {str(row["window_id"]): row for row in failure_rows}
    if len(failure_by_id) != len(failure_rows):
        raise ValueError("duplicate failure window IDs")
    protocol = config["matched_failure_protocol"]
    sampled = int(protocol["common_sampled_frames"])
    strict = protocol["strict_match"]
    windows: list[dict[str, Any]] = []
    for match_index, success in enumerate(success_rows):
        failure_id = str(success["failure_window_id"])
        failure = failure_by_id.get(failure_id)
        if failure is None:
            raise ValueError(f"matched failure window missing: {failure_id}")
        failure_episode = episodes[str(failure["episode_id"])]
        success_episode = episodes[str(success["episode_id"])]
        context = str(failure["horizon_name"]).removeprefix("pre_failure_")
        if context not in {"short", "medium", "long", "terminal_failure"}:
            raise ValueError(f"unexpected failure context: {context}")
        failure_window = _base_window(
            failure_episode,
            window_id=f"matched:failure:{match_index}:{failure_id}",
            purpose="failure_matched",
            context=context,
            start=int(failure["start_frame"]),
            end=int(failure["end_frame"]),
            sampled_frames=sampled,
        )
        success_window = _base_window(
            success_episode,
            window_id=(
                f"matched:success:{match_index}:{success_episode['episode_id']}"
            ),
            purpose="failure_matched",
            context=context,
            start=int(success["start_frame"]),
            end=int(success["end_frame"]),
            sampled_frames=sampled,
        )
        distances = success["match_distances"]
        strict_match = bool(
            float(distances["progress"]) <= float(strict["maximum_progress_distance"])
            and int(distances["stage"]) <= int(strict["maximum_stage_distance"])
            and int(distances["remaining_horizon"])
            <= int(strict["maximum_remaining_horizon_distance"])
        )
        taxonomy = [str(value) for value in failure["taxonomy"]]
        shared = {
            "match_id": f"match:{match_index}:{failure_id}",
            "strict_match": strict_match,
            "match_distances": distances,
            "failure_taxonomy": taxonomy,
            "failure_episode_id": failure_episode["episode_id"],
        }
        failure_window.update({**shared, "failure_label": True})
        success_window.update({**shared, "failure_label": False})
        windows.extend((failure_window, success_window))
    return windows


def build_windows(
    config_path: Path,
    *,
    source_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Build the immutable JSONL window stream and compact manifest."""

    config = read_yaml(resolve_repo_path(config_path))
    if config.get("schema_version") != "latentguard.lg_r1c.windows.v1":
        raise ValueError("unexpected LG-R1c window schema")
    current = read_jsonl(source_root / "progress_dataset" / "current_samples.jsonl")
    failures = read_jsonl(source_root / "failure_windows" / "failure_windows.jsonl")
    successes = read_jsonl(
        source_root / "failure_windows" / "matched_success_windows.jsonl"
    )
    episodes = _episode_rows(current)
    anchors = _anchor_windows(episodes, config)
    matched = _matched_windows(episodes, failures, successes, config)
    rows = anchors + matched
    ids = [str(row["window_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("reward window identities are not unique")
    destination.mkdir(parents=True, exist_ok=True)
    windows_path = destination / "reward_windows.jsonl"
    write_jsonl(windows_path, rows)
    calibration_rows = [row for row in rows if row["split"] == "validation"]
    calibration_path = destination / "calibration_validation_windows.jsonl"
    write_jsonl(calibration_path, calibration_rows)
    endpoints = hashlib.sha256()
    for row in rows:
        endpoints.update(
            (
                f"{row['window_id']}|{row['episode_id']}|{row['end_frame']}|"
                f"{row['common_frame_indices']}\n"
            ).encode()
        )
    contexts = Counter(str(row["context"]) for row in rows)
    tasks = Counter(str(row["task_key"]) for row in rows)
    strict_count = sum(
        bool(row.get("strict_match"))
        for row in matched
        if bool(row.get("failure_label"))
    )
    payload = {
        "schema_version": "latentguard.lg_r1c.window_manifest.v1",
        "status": "pass",
        "frozen_before_model_inference": True,
        "source_episodes": len(episodes),
        "source_frames": len(current),
        "window_count": len(rows),
        "anchor_windows": len(anchors),
        "matched_windows": len(matched),
        "matched_pairs": len(successes),
        "strict_matched_pairs": strict_count,
        "context_counts": dict(sorted(contexts.items())),
        "task_counts": dict(sorted(tasks.items())),
        "common_sampled_frames": 8,
        "topreward_native_diagnostic_windows": sum(
            "topreward_native_frame_indices" in row for row in rows
        ),
        "same_endpoint_across_models": True,
        "same_instruction_across_models": True,
        "same_camera_across_models": True,
        "model_result_dependent_sampling": False,
        "window_endpoint_digest": endpoints.hexdigest(),
        "windows_file": file_identity(
            windows_path,
            locator="reward_windows.jsonl",
        ),
        "calibration_validation_windows": {
            "selection_split": "validation",
            "test_labels_included": False,
            "rows": len(calibration_rows),
            "file": file_identity(
                calibration_path,
                locator="calibration_validation_windows.jsonl",
            ),
        },
        "configuration": config,
    }
    write_json(destination / "window_manifest.json", payload)
    return payload


def main() -> None:
    """Build the frozen reward window benchmark."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/windows.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    source = runtime_root(args.runtime_root)
    destination = output_root(args.output_root)
    payload = build_windows(args.config, source_root=source, destination=destination)
    if args.manifest_output is not None:
        write_json(resolve_repo_path(args.manifest_output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
