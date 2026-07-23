"""Evaluate online and oracle episode-time progress baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _lg_r1_common import (
    output_root,
    read_yaml,
    resolve_repo_path,
    write_json,
)

from latentguard.progress.metrics import (
    evaluate_progress,
    pairwise_accuracy,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL rows in {path}")
    return rows


def _kendall_episode(targets: list[float], predictions: list[float]) -> float:
    concordant = 0
    discordant = 0
    target_ties = 0
    prediction_ties = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            target_delta = targets[right] - targets[left]
            predicted_delta = predictions[right] - predictions[left]
            if target_delta == 0.0 and predicted_delta == 0.0:
                continue
            if target_delta == 0.0:
                target_ties += 1
            elif predicted_delta == 0.0:
                prediction_ties += 1
            elif target_delta * predicted_delta > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = (
        (concordant + discordant + target_ties)
        * (concordant + discordant + prediction_ties)
    ) ** 0.5
    return (concordant - discordant) / denominator if denominator else 0.0


def _progress_summary(
    rows: list[dict[str, Any]], prediction_key: str
) -> dict[str, Any]:
    targets = [float(row["overall_progress"]) for row in rows]
    predictions = [float(row[prediction_key]) for row in rows]
    metrics = evaluate_progress(targets, predictions)
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episodes[str(row["episode_id"])].append(row)
    kendall_values = []
    for episode_rows in episodes.values():
        ordered = sorted(episode_rows, key=lambda item: int(item["frame_index"]))
        kendall_values.append(
            _kendall_episode(
                [float(item["overall_progress"]) for item in ordered],
                [float(item[prediction_key]) for item in ordered],
            )
        )
    return {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "spearman": metrics.spearman,
        "episode_macro_kendall_tau_b": (sum(kendall_values) / len(kendall_values)),
        "samples": len(rows),
        "episodes": len(episodes),
    }


def main() -> None:
    """Evaluate time without fitting or accessing test outcomes."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/eval.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    runtime = (
        resolve_repo_path(args.output_root)
        if args.output_root is not None
        else output_root()
    )
    current = _read_jsonl(runtime / "progress_dataset" / "current_samples.jsonl")
    pairs = _read_jsonl(runtime / "progress_dataset" / "pairwise_samples.jsonl")
    for row in current:
        step = int(row["frame_index"])
        row["online_time_progress"] = min(
            step / max(int(row["episode_horizon"]) - 1, 1), 1.0
        )
        row["oracle_episode_time_progress"] = min(
            step / max(int(row["episode_frame_count"]) - 1, 1), 1.0
        )
    results: dict[str, Any] = {}
    for split in ("validation", "test"):
        split_rows = [row for row in current if row["split"] == split]
        split_pairs = [row for row in pairs if row["split"] == split]
        predicted_deltas = [
            (int(row["end_index"]) - int(row["start_index"]))
            / max(
                next(
                    int(item["episode_horizon"])
                    for item in split_rows
                    if item["episode_id"] == row["episode_id"]
                )
                - 1,
                1,
            )
            for row in split_pairs
        ]
        target_deltas = [float(row["progress_delta"]) for row in split_pairs]
        results[split] = {
            "online_available": _progress_summary(split_rows, "online_time_progress"),
            "offline_oracle": _progress_summary(
                split_rows, "oracle_episode_time_progress"
            ),
            "online_pairwise_accuracy": pairwise_accuracy(
                target_deltas, predicted_deltas, epsilon=0.02
            ),
            "online_regression_recall": 0.0,
            "online_stagnation_recall": 0.0,
        }
    payload = {
        "schema_version": ("latentguard.lg_r1.time_baseline_results.v1"),
        "status": "pass",
        "selection_split": config["selection_split"],
        "thresholds_fit_on_test": False,
        "online_baseline": "step_index / episode_horizon",
        "offline_oracle": "step_index / observed_episode_length",
        "offline_oracle_is_deployable": False,
        "results": results,
    }
    output = (
        resolve_repo_path(args.output)
        if args.output is not None
        else runtime / "time_baseline_results.json"
    )
    write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
