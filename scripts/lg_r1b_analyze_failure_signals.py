"""Analyze frozen SARM progress signals without fitting a failure head."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1b_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    write_json,
)
from lg_r1_build_stage_labels import _read_jsonl


def _signal_step(
    rows: list[dict[str, Any]],
    *,
    stagnation_delta: float,
    stagnation_steps: int,
    regression_delta: float,
) -> tuple[int | None, str | None, int, int]:
    values = [float(row["predicted_progress"]) for row in rows]
    deltas = np.diff(values)
    regression_positions = np.flatnonzero(deltas < regression_delta)
    first_regression = (
        int(regression_positions[0] + 1) if len(regression_positions) else None
    )
    first_stagnation = None
    for end in range(stagnation_steps, len(values)):
        start = end - stagnation_steps
        if (
            max(values[start : end + 1]) - min(values[start : end + 1])
            <= stagnation_delta
        ):
            first_stagnation = start
            break
    candidates = [
        (step, name)
        for step, name in (
            (first_regression, "regression"),
            (first_stagnation, "stagnation"),
        )
        if step is not None
    ]
    if not candidates:
        return None, None, int(len(regression_positions)), 0
    step, name = min(candidates)
    plateau = len(values) - first_stagnation if first_stagnation is not None else 0
    return int(step), name, int(len(regression_positions)), plateau


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    """Compute pre-registered progress-signal precision, recall, and lead time."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1b/analysis.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    predictions = _read_jsonl(runtime / "sarm_zero_shot_predictions.jsonl")
    failure = read_json(runtime / "failure_registry.json")
    failure_by_episode = {str(item["episode_id"]): item for item in failure["episodes"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["episode_id"])].append(row)
    thresholds = config["frozen_thresholds"]
    episode_results = []
    true_positive = false_positive = true_negative = false_negative = 0
    lead_to_terminal = []
    lead_to_abnormal = []
    for episode_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
        step, signal, regressions, plateau = _signal_step(
            ordered,
            stagnation_delta=float(thresholds["stagnation_delta_absolute_max"]),
            stagnation_steps=int(thresholds["stagnation_min_steps"]),
            regression_delta=float(thresholds["regression_delta_below"]),
        )
        failed = episode_id in failure_by_episode
        fired = step is not None
        if failed and fired:
            true_positive += 1
        elif failed:
            false_negative += 1
        elif fired:
            false_positive += 1
        else:
            true_negative += 1
        target = [float(row["target_progress"]) for row in ordered]
        predicted = [float(row["predicted_progress"]) for row in ordered]
        terminal_frame = int(ordered[-1]["frame_index"])
        abnormal_frame = (
            int(failure_by_episode[episode_id]["first_abnormal_step"])
            if failed
            else None
        )
        if failed and step is not None:
            signal_frame = int(ordered[step]["frame_index"])
            lead_to_terminal.append(terminal_frame - signal_frame)
            if abnormal_frame is not None:
                lead_to_abnormal.append(abnormal_frame - signal_frame)
        episode_results.append(
            {
                "episode_id": episode_id,
                "failed": failed,
                "signal_fired": fired,
                "signal_type": signal,
                "signal_sample_index": step,
                "regression_count": regressions,
                "plateau_sample_length": plateau,
                "pre_terminal_progress_delta": (
                    predicted[-1] - predicted[max(0, len(predicted) - 6)]
                ),
                "decline_from_maximum_progress": max(predicted) - predicted[-1],
                "target_decline_from_maximum": max(target) - target[-1],
            }
        )
    result = {
        "schema_version": "latentguard.lg_r1b.progress_failure_analysis.v1",
        "status": "pass",
        "scope": "progress-based failure signal",
        "threshold_source": "pre-registered frozen validation rules",
        "thresholds": thresholds,
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "failure_episode_detection_recall": _rate(
            true_positive, true_positive + false_negative
        ),
        "failure_episode_detection_precision": _rate(
            true_positive, true_positive + false_positive
        ),
        "false_alarm_rate": _rate(false_positive, false_positive + true_negative),
        "early_warning_lead_steps_to_terminal_mean": (
            float(np.mean(lead_to_terminal)) if lead_to_terminal else None
        ),
        "early_warning_lead_steps_to_first_abnormal_mean": (
            float(np.mean(lead_to_abnormal)) if lead_to_abnormal else None
        ),
        "episode_results": episode_results,
        "failure_head_trained": False,
        "intervention_threshold_selected": False,
        "safety_detector_claim": False,
        "limitations": [
            (
                "Feature-cache sampling means reported lead lengths are sampled "
                "frame steps."
            ),
            "Descriptive association does not establish causal or safety value.",
        ],
    }
    write_json(runtime / "progress_failure_analysis.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
