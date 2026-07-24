"""Materialize frozen real-action LG-R2a probe targets outside Git."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2a_common import (
    file_identity,
    output_root,
    r1c_root,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    source_root,
    write_json,
    write_jsonl,
)

from latentguard.action_conditioning.contracts import ActionContract


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--r1c-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _episode_proprio(source: Path, episode_id: str) -> np.ndarray:
    """Load current-time proprioception from one frozen LeRobot parquet."""

    import pyarrow.parquet as pq

    path = source / "datasets" / episode_id / "data" / "chunk-000" / "file-000.parquet"
    if not path.is_file():
        raise ValueError(f"missing frozen episode parquet: {path}")
    table = pq.read_table(path, columns=["frame_index", "observation.state"])
    frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    order = np.argsort(frames, kind="mergesort")
    if not np.array_equal(frames[order], np.arange(frames.size)):
        raise ValueError(f"non-contiguous frame indices in {path}")
    if states.shape != (frames.size, 8) or not np.isfinite(states).all():
        raise ValueError(f"invalid current proprioception in {path}")
    return states[order]


def main() -> None:
    """Join frozen representation, action, proprioception, and future labels."""

    args = parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    horizons = [int(value) for value in config["horizons"]]
    if horizons != [7, 21, 49]:
        raise ValueError("LG-R2a horizons must remain frozen at 7, 21, and 49")
    if args.dry_run:
        print({"status": "dry_run", "horizons": horizons})
        return
    source = source_root(args.source_root)
    reward = r1c_root(args.r1c_root)
    output = output_root(args.output_dir)
    split_manifest = read_json(output / "cv_split_manifest.json")
    fold_by_episode = {
        str(row["episode_id"]): int(row["fold"])
        for row in split_manifest["assignments"]
    }
    current_rows = read_jsonl(source / "progress_dataset" / "current_samples.jsonl")
    current_by_id = {
        f"{row['episode_id']}:{int(row['frame_index'])}": row for row in current_rows
    }
    failures = {
        str(row["episode_id"]): row
        for row in read_json(resolve_repo_path(config["failure_registry"]))["episodes"]
    }
    stage_by_episode: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((source / "stage_labels").glob("*.jsonl")):
        rows = read_jsonl(path)
        episode_id = str(rows[0]["episode_id"])
        if any(str(row["episode_id"]) != episode_id for row in rows):
            raise ValueError(f"mixed episode IDs in {path}")
        if [int(row["step_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous stage-label steps in {path}")
        stage_by_episode[episode_id] = rows
    cache_path = reward / "vlajepa-probe" / "frozen_qwen_features.npz"
    cache = np.load(cache_path, allow_pickle=False)
    cache_features = np.asarray(cache["features"], dtype=np.float32)
    cache_sample_ids = [str(value) for value in cache["sample_id"].tolist()]
    sample_ids = cache_sample_ids
    if args.limit_samples is not None:
        sample_ids = sample_ids[: args.limit_samples]
    feature_index = {
        str(sample_id): index for index, sample_id in enumerate(cache_sample_ids)
    }
    proprio_cache: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    progress: list[list[float]] = []
    progress_valid: list[list[bool]] = []
    stagnation: list[list[float]] = []
    regression: list[list[float]] = []
    events: list[list[float]] = []
    terminal: list[float] = []
    metadata: list[dict[str, Any]] = []
    excluded_tail = 0
    for sample_id in sample_ids:
        episode_id, frame_text = sample_id.rsplit(":", 1)
        frame = int(frame_text)
        rows = stage_by_episode[episode_id]
        if frame + 7 > len(rows):
            excluded_tail += 1
            continue
        current = current_by_id[sample_id]
        if episode_id not in proprio_cache:
            proprio_cache[episode_id] = _episode_proprio(source, episode_id)
        chunk = np.asarray(
            [
                row["executed_action_window"]["executed_action"]
                for row in rows[frame : frame + 7]
            ],
            dtype=np.float32,
        )
        action_mask = np.ones(7, dtype=np.bool_)
        current_progress = float(rows[frame]["pre_label"]["overall_progress"])
        delta_values: list[float] = []
        valid_values: list[bool] = []
        stagnant_values: list[float] = []
        regressed_values: list[float] = []
        for horizon in horizons:
            valid = frame + horizon <= len(rows)
            valid_values.append(valid)
            if valid:
                future_progress = float(
                    rows[frame + horizon - 1]["post_label"]["overall_progress"]
                )
                delta = future_progress - current_progress
                regressed = delta < -float(config["regression_threshold"]) or any(
                    bool(row["stage_regression"])
                    for row in rows[frame : frame + horizon]
                )
            else:
                delta = 0.0
                regressed = False
            delta_values.append(delta)
            stagnant_values.append(
                float(valid and delta < float(config["stagnation_threshold"]))
            )
            regressed_values.append(float(valid and regressed))
        failure = failures.get(episode_id)
        taxonomy = str(failure["primary_taxonomy"]) if failure is not None else None
        abnormal_step = (
            int(failure["first_abnormal_step"]) if failure is not None else None
        )
        within_long = (
            abnormal_step is not None and frame <= abnormal_step < frame + horizons[-1]
        )
        terminal_distance = len(rows) - frame
        terminal_failure = bool(
            failure is not None and terminal_distance <= horizons[-1]
        )
        if abnormal_step is None:
            time_bucket = None
        else:
            distance = abnormal_step - frame
            time_bucket = (
                "past"
                if distance < 0
                else "imminent"
                if distance < 7
                else "medium"
                if distance < 21
                else "distant"
            )
        features.append(cache_features[feature_index[sample_id]])
        actions.append(chunk)
        masks.append(action_mask)
        proprio.append(proprio_cache[episode_id][frame])
        progress.append(delta_values)
        progress_valid.append(valid_values)
        stagnation.append(stagnant_values)
        regression.append(regressed_values)
        events.append(
            [
                float(within_long and taxonomy == "OBJECT_DROP"),
                float(within_long and taxonomy == "FAILED_PLACEMENT"),
            ]
        )
        terminal.append(float(terminal_failure))
        metadata.append(
            {
                "sample_id": sample_id,
                "episode_id": episode_id,
                "frame_index": frame,
                "seed": int(current["seed"]),
                "suite": str(current["suite"]),
                "task_id": int(current["task_id"]),
                "task": f"{current['suite']}/task{int(current['task_id'])}",
                "original_split": str(current["split"]),
                "fold": fold_by_episode[episode_id],
                "episode_success": bool(current["episode_success"]),
                "current_progress": current_progress,
                "progress_bin": min(9, int(current_progress * 10)),
                "current_stage": int(rows[frame]["pre_label"]["stage_id"]),
                "failure_taxonomy": taxonomy,
                "time_to_failure_bucket": time_bucket,
                "effective_target_steps": [
                    min(horizon, len(rows) - frame) for horizon in horizons
                ],
                "effective_target_steps_reporting_only": True,
                "action_source": "frozen_real_on_policy_executed",
                "synthetic_action": False,
            }
        )
    action_array = np.stack(actions)
    mask_array = np.stack(masks)
    action_validation = ActionContract().validate(
        action_array, mask_array, require_full_horizon=True
    )
    prepared_path = output / "prepared_data.npz"
    np.savez_compressed(
        prepared_path,
        features=np.stack(features),
        actions=action_array,
        action_mask=mask_array,
        proprio=np.stack(proprio),
        progress=np.asarray(progress, dtype=np.float32),
        progress_valid=np.asarray(progress_valid, dtype=np.bool_),
        stagnation=np.asarray(stagnation, dtype=np.float32),
        regression=np.asarray(regression, dtype=np.float32),
        events=np.asarray(events, dtype=np.float32),
        terminal=np.asarray(terminal, dtype=np.float32)[:, None],
    )
    write_jsonl(output / "prepared_metadata.jsonl", metadata)
    target_manifest = {
        "schema_version": "latentguard.lg_r2a.target_manifest.v1",
        "status": "pass",
        "samples": len(metadata),
        "excluded_incomplete_action_tail_samples": excluded_tail,
        "horizons": {"short": 7, "medium": 21, "long": 49},
        "progress_delta": "post_progress(t+k-1)-pre_progress(t)",
        "stagnation_threshold": float(config["stagnation_threshold"]),
        "regression_threshold": float(config["regression_threshold"]),
        "event_targets": ["OBJECT_DROP", "FAILED_PLACEMENT"],
        "event_timing_source": "LG-R1b frozen first_abnormal_step",
        "terminal_failure": (
            "natural unsuccessful episode whose terminal boundary is within "
            "the frozen long horizon"
        ),
        "time_to_failure": {
            "role": "diagnostic_only",
            "buckets": ["imminent_lt7", "medium_lt21", "distant_ge21", "past"],
        },
        "label_only_fields": [
            "future progress",
            "future stage regression",
            "failure taxonomy",
            "terminal result",
            "effective target steps",
        ],
        "deployable_model_inputs": [
            "current_vla_representation",
            "numeric_action_chunk",
            "action_mask",
            "effective_action_horizon",
            "current_proprioception_for_model_d",
        ],
        "prepared_data": file_identity(
            prepared_path, locator="external/prepared_data.npz"
        ),
        "action_validation": action_validation,
        "supports": {
            "OBJECT_DROP": int(np.asarray(events)[:, 0].sum()),
            "FAILED_PLACEMENT": int(np.asarray(events)[:, 1].sum()),
            "terminal_failure": int(np.asarray(terminal).sum()),
        },
        "new_rollouts": 0,
        "synthetic_actions": 0,
    }
    write_json(output / "target_manifest.json", target_manifest)
    print(
        {
            "status": "pass",
            "samples": len(metadata),
            "excluded_tail": excluded_tail,
            "supports": target_manifest["supports"],
        }
    )


if __name__ == "__main__":
    main()
