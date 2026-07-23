"""Evaluate LG-R1 progress baselines and descriptive failure trends."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r0_runtime import load_stack
from _lg_r1_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    write_json,
)
from _lg_r1_training import EpisodeDatasetCache

from latentguard.adapters.vla_jepa.world_model_adapter import (
    inspect_training_world_model,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL rows in {path}")
    return rows


def _mean_curve(
    episodes: dict[str, list[dict[str, Any]]],
    *,
    success: bool,
    bins: int = 10,
) -> list[dict[str, Any]]:
    buckets: list[list[float]] = [[] for _ in range(bins)]
    targets: list[list[float]] = [[] for _ in range(bins)]
    for rows in episodes.values():
        if bool(rows[0]["episode_success"]) is not success:
            continue
        ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
        maximum = max(int(row["frame_index"]) for row in ordered)
        for row in ordered:
            fraction = int(row["frame_index"]) / max(maximum, 1)
            index = min(int(fraction * bins), bins - 1)
            buckets[index].append(float(row["predicted_progress"]))
            targets[index].append(float(row["target_progress"]))
    return [
        {
            "normalized_time_bin": index,
            "predicted_progress_mean": (float(np.mean(bucket)) if bucket else None),
            "target_progress_mean": (
                float(np.mean(targets[index])) if targets[index] else None
            ),
            "samples": len(bucket),
        }
        for index, bucket in enumerate(buckets)
    ]


def _failure_analysis(
    predictions: list[dict[str, Any]], sarm: dict[str, Any]
) -> dict[str, Any]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        episodes[str(row["episode_id"])].append(row)
    failure_deltas: list[float] = []
    failure_plateaus: list[int] = []
    regression_leads: list[int] = []
    for rows in episodes.values():
        if bool(rows[0]["episode_success"]):
            continue
        ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
        deltas = [
            float(right["predicted_progress"]) - float(left["predicted_progress"])
            for left, right in zip(ordered, ordered[1:], strict=False)
        ]
        failure_deltas.extend(deltas[-5:])
        plateau = 0
        for delta in reversed(deltas):
            if abs(delta) <= 0.02:
                plateau += 1
            else:
                break
        failure_plateaus.append(plateau)
        regression_positions = [
            index for index, delta in enumerate(deltas) if delta < -0.02
        ]
        if regression_positions:
            regression_leads.append(len(deltas) - regression_positions[0])
    return {
        "schema_version": ("latentguard.lg_r1.progress_failure_analysis.v1"),
        "status": "pass",
        "scope": "descriptive progress analysis, not safety detection",
        "threshold_source": "validation",
        "thresholds": {
            "stagnation_absolute_delta": 0.02,
            "negative_progress_delta": -0.02,
        },
        "success_mean_progress_curve": _mean_curve(episodes, success=True),
        "failure_mean_progress_curve": _mean_curve(episodes, success=False),
        "failure_pre_terminal_delta_mean": (
            float(np.mean(failure_deltas)) if failure_deltas else None
        ),
        "timeout_terminal_plateau_sample_count_mean": (
            float(np.mean(failure_plateaus)) if failure_plateaus else None
        ),
        "regression_before_timeout_sample_count_mean": (
            float(np.mean(regression_leads)) if regression_leads else None
        ),
        "validation_temporal_metrics": sarm["validation"]["temporal"],
        "test_temporal_metrics": sarm["test"]["temporal"],
        "terminal_failure_label_used_for_timeout": False,
        "failure_classifier_trained": False,
        "intervention_threshold_selected": False,
        "limitations": [
            (
                "Feature-cache sampling makes lead and plateau lengths sample "
                "counts, not control steps."
            ),
            (
                "Associations are descriptive and do not establish failure "
                "prediction or safety value."
            ),
        ],
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.std() == 0.0 or right_array.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _vlajepa_descriptive(
    runtime: Path,
    predictions: list[dict[str, Any]],
    *,
    max_samples: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    candidates = sorted(
        [row for row in predictions if row["split"] == "test"],
        key=lambda row: (
            bool(row["episode_success"]),
            str(row["episode_id"]),
            int(row["frame_index"]),
        ),
    )
    if len(candidates) > max_samples:
        indices = np.linspace(0, len(candidates) - 1, max_samples, dtype=int)
        candidates = [candidates[int(index)] for index in indices]
    stack = load_stack()
    stack.policy.requires_grad_(False)
    stack.policy.eval()
    datasets = EpisodeDatasetCache(runtime)
    instruction_by_episode = {
        str(item["episode_id"]): str(item["instruction"])
        for item in _read_jsonl(runtime / "progress_dataset" / "current_samples.jsonl")
    }
    records = []
    failures = []
    for row in candidates:
        try:
            dataset = datasets.get(str(row["episode_id"]))
            start = min(
                int(row["frame_index"]),
                max(len(dataset) - int(stack.config.num_video_frames), 0),
            )
            frames = [
                dataset[start + offset]
                for offset in range(int(stack.config.num_video_frames))
            ]
            batch = {
                f"{OBS_IMAGES}.image": torch.stack(
                    [frame[f"{OBS_IMAGES}.image"] for frame in frames]
                )
                .unsqueeze(0)
                .to(stack.config.device),
                f"{OBS_IMAGES}.image2": torch.stack(
                    [frame[f"{OBS_IMAGES}.image2"] for frame in frames]
                )
                .unsqueeze(0)
                .to(stack.config.device),
                OBS_STATE: frames[0][OBS_STATE].unsqueeze(0).to(stack.config.device),
                "task": [instruction_by_episode[str(row["episode_id"])]],
            }
            output = inspect_training_world_model(stack.policy, batch)
            predicted = output.predicted_future_latents
            target = output.target_future_latents
            if predicted is None or target is None:
                raise RuntimeError("world-model tensors unavailable")
            cosine = 1.0 - functional.cosine_similarity(
                predicted.float().flatten(1),
                target.float().flatten(1),
                dim=1,
            )
            metadata = output.predictor_metadata
            records.append(
                {
                    "episode_id": row["episode_id"],
                    "frame_index": int(row["frame_index"]),
                    "episode_success": bool(row["episode_success"]),
                    "sarm_progress": float(row["predicted_progress"]),
                    "observed_progress": float(row["target_progress"]),
                    "predictor_variance": float(
                        predicted.float().var(unbiased=False).item()
                    ),
                    "predictor_target_l1": float(metadata["l1_distance"]),
                    "predictor_target_cosine_distance": float(cosine.item()),
                    "action_token_norm_mean": float(metadata["action_token_norm_mean"]),
                    "current_latent_norm_mean": float(
                        metadata["current_latent_norm_mean"]
                    ),
                    "predicted_latent_norm_mean": float(
                        metadata["predicted_latent_norm_mean"]
                    ),
                    "current_predicted_l1": float(metadata["current_predicted_l1"]),
                    "predicted_temporal_change_mean": float(
                        metadata["predicted_temporal_change_mean"]
                    ),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "episode_id": row["episode_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    fields = [
        "predictor_variance",
        "action_token_norm_mean",
        "current_latent_norm_mean",
        "predicted_latent_norm_mean",
        "current_predicted_l1",
        "predicted_temporal_change_mean",
    ]
    correlations = {
        field: {
            "with_sarm_progress": _pearson(
                [float(record[field]) for record in records],
                [float(record["sarm_progress"]) for record in records],
            ),
            "with_observed_progress": _pearson(
                [float(record[field]) for record in records],
                [float(record["observed_progress"]) for record in records],
            ),
        }
        for field in fields
    }
    return {
        "schema_version": ("latentguard.lg_r1.vlajepa_descriptive_comparison.v1"),
        "status": ("pass" if records and not failures else "partial"),
        "path_semantic": ("official training-style offline diagnostic"),
        "samples": records,
        "sample_count": len(records),
        "failures": failures,
        "correlations": correlations,
        "external_numeric_candidates_supported": False,
        "native_progress_score_available": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "failure_head_trained": False,
        "claims": (
            "descriptive association only; no predictive, causal, "
            "candidate-ranking, or safety claim"
        ),
    }


def main() -> None:
    """Run formal post-selection comparison without test tuning."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/eval.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-vlajepa", action="store_true")
    parser.add_argument("--max-vlajepa-samples", type=int, default=32)
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    time_baseline = read_json(runtime / "time_baseline_results.json")
    probe = read_json(runtime / "representation_probe_results.json")
    sarm = read_json(runtime / "sarm_results.json")
    predictions = _read_jsonl(runtime / "sarm_predictions.jsonl")
    validation_improvement = float(sarm["validation"]["progress"]["mae"]) < float(
        time_baseline["results"]["validation"]["online_available"]["mae"]
    )
    test_retention = float(sarm["test"]["progress"]["mae"]) < float(
        time_baseline["results"]["test"]["online_available"]["mae"]
    )
    comparison = {
        "schema_version": ("latentguard.lg_r1.progress_evaluation.v1"),
        "status": "pass",
        "selection_split": config["selection_split"],
        "test_used_for_selection": False,
        "time_baseline": time_baseline,
        "representation_probe": probe,
        "sarm": sarm,
        "promotion_checks": {
            "sarm_validation_mae_better_than_online_time": (validation_improvement),
            "sarm_test_mae_retains_improvement": test_retention,
        },
        "sarm_progress_baseline_promoted": (validation_improvement and test_retention),
        "task_success_claim": False,
        "failure_detection_claim": False,
    }
    failure = _failure_analysis(predictions, sarm)
    write_json(runtime / "progress_failure_analysis.json", failure)
    if not args.skip_vlajepa:
        vlajepa = _vlajepa_descriptive(
            runtime,
            predictions,
            max_samples=args.max_vlajepa_samples,
        )
        write_json(runtime / "vlajepa_descriptive_comparison.json", vlajepa)
        comparison["vlajepa_descriptive_status"] = vlajepa["status"]
    output = (
        resolve_repo_path(args.output)
        if args.output is not None
        else runtime / "progress_evaluation.json"
    )
    write_json(output, comparison)
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
