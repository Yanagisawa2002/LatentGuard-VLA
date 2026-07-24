"""Evaluate frozen zero-shot and validation-calibrated LG-R1c rewards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1c_common import (
    output_root,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    sha256_path,
    write_json,
)
from _lg_r1c_reward_runtime import load_windows

from latentguard.rewards.metrics import (
    binary_metrics,
    pairwise_progress_accuracy,
    progress_metrics,
    recall_operating_point,
    task_macro_binary_metrics,
)
from latentguard.rewards.schemas import evaluate_reward_gate


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["window_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate prediction identity in {path}")
    return result


def _mean_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else None


def _progress_summary(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        targets = [float(row["progress_target"]) for row in subset]
        values = [scores[str(row["window_id"])] for row in subset]
        result = progress_metrics(targets, values)
        result["pairwise_accuracy"] = pairwise_progress_accuracy(
            [str(row["episode_id"]) for row in subset],
            [int(row["end_frame"]) for row in subset],
            targets,
            values,
        )
        return result

    by_split = {
        split: summarize([row for row in rows if row["split"] == split])
        for split in ("train", "validation", "test")
    }
    per_task = {
        task: summarize([row for row in rows if row["task_key"] == task])
        for task in sorted({str(row["task_key"]) for row in rows})
    }
    task_records = list(per_task.values())
    return {
        "all": summarize(rows),
        "by_split": by_split,
        "task_macro": {
            "eligible_tasks": len(task_records),
            "mae": _mean_metric(task_records, "mae"),
            "rmse": _mean_metric(task_records, "rmse"),
            "spearman": _mean_metric(task_records, "spearman"),
            "kendall_tau_b": _mean_metric(task_records, "kendall_tau_b"),
            "pairwise_accuracy": _mean_metric(
                task_records,
                "pairwise_accuracy",
            ),
        },
        "per_task": per_task,
    }


def _success_summary(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [bool(row["terminal_success"]) for row in subset]
        values = [scores[str(row["window_id"])] for row in subset]
        return binary_metrics(labels, values, probability_scores=values).to_dict()

    labels = [bool(row["terminal_success"]) for row in rows]
    values = [scores[str(row["window_id"])] for row in rows]
    tasks = [str(row["task_key"]) for row in rows]
    return {
        "all": summarize(rows),
        "by_split": {
            split: summarize([row for row in rows if row["split"] == split])
            for split in ("train", "validation", "test")
        },
        "task_macro": task_macro_binary_metrics(labels, values, tasks),
    }


def _binary_failure_scope(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    labels = [bool(row["failure_label"]) for row in rows]
    values = [scores[str(row["window_id"])] for row in rows]
    result = binary_metrics(labels, values).to_dict()
    if any(labels):
        result["operating_points"] = {
            str(recall): recall_operating_point(
                labels,
                values,
                requested_recall=recall,
            ).to_dict()
            for recall in (0.5, 0.6, 0.8)
        }
    else:
        result["operating_points"] = {}
    return result


def _matched_failure_summary(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    all_result = _binary_failure_scope(rows, scores)
    strict_rows = [row for row in rows if bool(row["strict_match"])]
    leave_rows = [
        row
        for row in rows
        if not (row["suite"] == "libero_10" and int(row["task_id"]) == 6)
    ]
    labels = [bool(row["failure_label"]) for row in rows]
    values = [scores[str(row["window_id"])] for row in rows]
    tasks = [str(row["task_key"]) for row in rows]
    taxonomy: dict[str, Any] = {}
    for name in ("FAILED_PLACEMENT", "OBJECT_DROP"):
        subset = [row for row in rows if name in row["failure_taxonomy"]]
        taxonomy[name] = _binary_failure_scope(subset, scores) if subset else None
    return {
        "all_matched": all_result,
        "strictly_matched": _binary_failure_scope(strict_rows, scores),
        "by_split": {
            split: _binary_failure_scope(
                [row for row in rows if row["split"] == split],
                scores,
            )
            for split in ("train", "validation", "test")
        },
        "task_macro": task_macro_binary_metrics(labels, values, tasks),
        "leave_task6_out": _binary_failure_scope(leave_rows, scores),
        "taxonomy": taxonomy,
    }


def _episode_failure_summary(
    anchor_rows: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    terminal_by_episode: dict[str, dict[str, Any]] = {}
    for row in anchor_rows:
        episode_id = str(row["episode_id"])
        prior = terminal_by_episode.get(episode_id)
        if prior is None or int(row["end_frame"]) > int(prior["end_frame"]):
            terminal_by_episode[episode_id] = row
    rows = list(terminal_by_episode.values())
    labels = [not bool(row["terminal_success"]) for row in rows]
    values = [scores[str(row["window_id"])] for row in rows]
    tasks = [str(row["task_key"]) for row in rows]
    leave = [
        row
        for row in rows
        if not (row["suite"] == "libero_10" and int(row["task_id"]) == 6)
    ]
    return {
        "all": _binary_failure_scope(
            [{**row, "failure_label": not row["terminal_success"]} for row in rows],
            scores,
        ),
        "by_split": {
            split: _binary_failure_scope(
                [
                    {**row, "failure_label": not row["terminal_success"]}
                    for row in rows
                    if row["split"] == split
                ],
                scores,
            )
            for split in ("train", "validation", "test")
        },
        "task_macro": task_macro_binary_metrics(labels, values, tasks),
        "leave_task6_out": _binary_failure_scope(
            [{**row, "failure_label": not row["terminal_success"]} for row in leave],
            scores,
        ),
    }


def _early_warning(
    rows: list[dict[str, Any]],
    scores: dict[str, float],
) -> dict[str, Any]:
    validation = [row for row in rows if row["split"] == "validation"]
    labels = [bool(row["failure_label"]) for row in validation]
    values = [scores[str(row["window_id"])] for row in validation]
    operating = recall_operating_point(
        labels,
        values,
        requested_recall=0.6,
    )
    threshold = operating.threshold
    failure_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row["failure_label"]):
            failure_rows[str(row["episode_id"])].append(row)
    leads = []
    detected = 0
    for episode_rows in failure_rows.values():
        triggered = [
            row for row in episode_rows if scores[str(row["window_id"])] >= threshold
        ]
        if triggered:
            first = min(triggered, key=lambda row: int(row["end_frame"]))
            leads.append(int(first["time_to_episode_end"]))
            detected += 1
    success_rows = [row for row in rows if not bool(row["failure_label"])]
    false_alarm_rate = float(
        np.mean([scores[str(row["window_id"])] >= threshold for row in success_rows])
    )
    return {
        "threshold_fit_split": "validation",
        "threshold_fit_operating_point": operating.to_dict(),
        "failure_episodes": len(failure_rows),
        "detected_failure_episodes": detected,
        "episode_detection_recall": detected / max(len(failure_rows), 1),
        "mean_early_warning_lead_frames": (float(np.mean(leads)) if leads else None),
        "false_alarm_rate_on_matched_success_windows": false_alarm_rate,
    }


def _robometer_temporal(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_deltas = []
    predicted_deltas = []
    for row in rows:
        target = np.asarray(row["frame_progress_targets"], dtype=np.float64)
        predicted = np.asarray(
            predictions[str(row["window_id"])]["per_frame_progress"],
            dtype=np.float64,
        )
        if target.shape != predicted.shape:
            raise ValueError("ROBOMETER temporal prediction shape mismatch")
        target_deltas.extend(np.diff(target).tolist())
        predicted_deltas.extend(np.diff(predicted).tolist())
    target_array = np.asarray(target_deltas)
    predicted_array = np.asarray(predicted_deltas)
    stagnation = np.abs(target_array) <= 0.02
    regression = target_array < -0.02
    return {
        "transitions": len(target_array),
        "stagnation_detection_recall": (
            float((np.abs(predicted_array[stagnation]) <= 0.02).mean())
            if stagnation.any()
            else None
        ),
        "regression_detection_recall": (
            float((predicted_array[regression] < -0.02).mean())
            if regression.any()
            else None
        ),
        "monotonicity_violation_rate": float((predicted_array < -0.02).mean()),
    }


def _model_summary(
    *,
    name: str,
    progress_scores: dict[str, float],
    success_scores: dict[str, float],
    failure_scores: dict[str, float],
    anchor_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    fitted: bool,
) -> dict[str, Any]:
    return {
        "model": name,
        "validation_fitted": fitted,
        "progress": _progress_summary(anchor_rows, progress_scores),
        "success": _success_summary(anchor_rows, success_scores),
        "episode_failure": _episode_failure_summary(
            anchor_rows,
            failure_scores,
        ),
        "matched_failure": _matched_failure_summary(
            matched_rows,
            failure_scores,
        ),
        "early_warning": _early_warning(matched_rows, failure_scores),
    }


def _build_score_sets(
    destination: Path,
    *,
    include_calibrated: bool,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, Any]]]:
    time = _prediction_map(destination / "time_predictions.jsonl")
    sarm = _prediction_map(destination / "sarm_predictions.jsonl")
    robometer = _prediction_map(destination / "robometer_predictions.jsonl")
    topreward = _prediction_map(destination / "topreward_predictions.jsonl")
    identities = set(time)
    if not all(set(values) == identities for values in (sarm, robometer, topreward)):
        raise ValueError("reward models did not score identical frozen windows")
    scores: dict[str, dict[str, dict[str, float]]] = {
        "online_time": {
            "progress": {key: float(value["score"]) for key, value in time.items()},
            "success": {key: float(value["score"]) for key, value in time.items()},
            "failure": {
                key: 1.0 - float(value["score"]) for key, value in time.items()
            },
        },
        "frozen_sarm": {
            "progress": {
                key: float(value["predicted_progress"]) for key, value in sarm.items()
            },
            "success": {
                key: float(value["predicted_progress"]) for key, value in sarm.items()
            },
            "failure": {
                key: 1.0 - float(value["predicted_progress"])
                for key, value in sarm.items()
            },
        },
        "robometer_zero_shot": {
            "progress": {
                key: float(value["last_frame_progress"])
                for key, value in robometer.items()
            },
            "success": {
                key: float(value["last_frame_success_probability"])
                for key, value in robometer.items()
            },
            "failure": {
                key: 1.0 - float(value["last_frame_success_probability"])
                for key, value in robometer.items()
            },
        },
        "topreward_zero_shot": {
            "progress": {
                key: float(value["normalized_window_reward"])
                for key, value in topreward.items()
            },
            "success": {
                key: float(value["normalized_window_reward"])
                for key, value in topreward.items()
            },
            "failure": {
                key: 1.0 - float(value["normalized_window_reward"])
                for key, value in topreward.items()
            },
        },
    }
    raw = {
        "time": time,
        "sarm": sarm,
        "robometer": robometer,
        "topreward": topreward,
    }
    if include_calibrated:
        calibrated = _prediction_map(destination / "calibrated_predictions.jsonl")
        if set(calibrated) != identities:
            raise ValueError("calibrated predictions changed frozen endpoints")
        mappings = {
            "robometer_calibrated": (
                "robometer_calibrated_progress",
                "robometer_calibrated_success",
                None,
            ),
            "topreward_calibrated": (
                "topreward_calibrated_progress",
                "topreward_calibrated_success",
                None,
            ),
            "task_agnostic_ensemble": (
                "ensemble_progress",
                "ensemble_success",
                "ensemble_failure",
            ),
        }
        for name, (progress_key, success_key, failure_key) in mappings.items():
            success_values = {
                key: float(value[success_key]) for key, value in calibrated.items()
            }
            scores[name] = {
                "progress": {
                    key: float(value[progress_key]) for key, value in calibrated.items()
                },
                "success": success_values,
                "failure": (
                    {
                        key: float(value[failure_key])
                        for key, value in calibrated.items()
                    }
                    if failure_key is not None
                    else {key: 1.0 - value for key, value in success_values.items()}
                ),
            }
        raw["calibrated"] = calibrated
    return scores, raw


def _candidate_gate(
    summary: dict[str, Any],
    *,
    fitted: bool,
) -> dict[str, Any]:
    scope = "test" if fitted else "all"
    progress = (
        summary["progress"]["by_split"]["test"]
        if fitted
        else summary["progress"]["task_macro"]
    )
    success_auroc = (
        summary["success"]["by_split"]["test"]["auroc"]
        if fitted
        else summary["success"]["task_macro"]["auroc"]
    )
    failure = summary["episode_failure"][scope]
    leave = summary["episode_failure"]["leave_task6_out"]
    operating = failure.get("operating_points", {}).get("0.6")
    candidate = {
        "progress_spearman": progress.get("spearman"),
        "pairwise_accuracy": progress.get("pairwise_accuracy"),
        "task_macro_success_auroc": success_auroc,
        "failure_auprc": failure.get("auprc"),
        "failure_precision_at_recall": (
            operating.get("precision") if operating else None
        ),
        "failure_recall": operating.get("recall") if operating else None,
        "failure_fpr_at_recall": (
            operating.get("false_positive_rate") if operating else None
        ),
        "leave_task6_failure_auprc": leave.get("auprc"),
    }
    prevalence = float(failure.get("prevalence", 0.0))
    gate = evaluate_reward_gate(candidate, prevalence=prevalence)
    gate["evaluation_scope"] = scope
    gate["candidate_metrics"] = candidate
    return gate


def main() -> None:
    """Write zero-shot or final calibrated reward evaluation artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/evaluation.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--phase",
        choices=("zero-shot", "final"),
        default="final",
    )
    args = parser.parse_args()
    destination = output_root(args.output_root)
    config = read_yaml(resolve_repo_path(args.config))
    windows = load_windows(destination)
    anchor_rows = [
        row
        for row in windows
        if row["purpose"] == "anchor_grid"
        and row["context"] == config["primary_progress_context"]
    ]
    matched_rows = [row for row in windows if row["purpose"] == "failure_matched"]
    include_calibrated = args.phase == "final"
    scores, raw = _build_score_sets(
        destination,
        include_calibrated=include_calibrated,
    )
    fitted_models = {
        "robometer_calibrated",
        "topreward_calibrated",
        "task_agnostic_ensemble",
    }
    summaries = {
        name: _model_summary(
            name=name,
            progress_scores=values["progress"],
            success_scores=values["success"],
            failure_scores=values["failure"],
            anchor_rows=anchor_rows,
            matched_rows=matched_rows,
            fitted=name in fitted_models,
        )
        for name, values in scores.items()
    }
    summaries["robometer_zero_shot"]["temporal"] = _robometer_temporal(
        anchor_rows,
        raw["robometer"],
    )
    zero_shot_outputs = {
        "online_time": "time_baseline_results.json",
        "frozen_sarm": "sarm_results.json",
        "robometer_zero_shot": "robometer_zero_shot_results.json",
        "topreward_zero_shot": "topreward_zero_shot_results.json",
    }
    for model, filename in zero_shot_outputs.items():
        prior_path = destination / filename
        prior = read_json(prior_path) if prior_path.is_file() else {}
        write_json(
            prior_path,
            {
                **prior,
                "schema_version": f"latentguard.lg_r1c.{model}_results.v1",
                "status": "pass",
                "evaluation": summaries[model],
            },
        )
    if args.phase == "zero-shot":
        prediction_files = (
            "time_predictions.jsonl",
            "sarm_predictions.jsonl",
            "robometer_predictions.jsonl",
            "topreward_predictions.jsonl",
        )
        result_files = tuple(zero_shot_outputs.values())
        freeze = {
            "schema_version": "latentguard.lg_r1c.zero_shot_freeze.v1",
            "status": "frozen_before_calibration",
            "prediction_sha256": {
                name: sha256_path(destination / name) for name in prediction_files
            },
            "result_sha256": {
                name: sha256_path(destination / name) for name in result_files
            },
            "foundation_model_training": False,
            "optimizer_steps": 0,
        }
        write_json(destination / "zero_shot_freeze.json", freeze)
        print(json.dumps(freeze, sort_keys=True))
        return
    calibrated_path = destination / "calibrated_reward_results.json"
    calibrated = read_json(calibrated_path)
    calibrated["evaluation"] = {
        name: summaries[name]
        for name in (
            "robometer_calibrated",
            "topreward_calibrated",
            "task_agnostic_ensemble",
        )
    }
    write_json(calibrated_path, calibrated)
    gates = {
        name: _candidate_gate(
            summary,
            fitted=name in fitted_models,
        )
        for name, summary in summaries.items()
    }
    task_agnostic = [
        "robometer_zero_shot",
        "topreward_zero_shot",
        "robometer_calibrated",
        "topreward_calibrated",
        "task_agnostic_ensemble",
    ]
    authorized = [
        name
        for name in task_agnostic
        if gates[name]["LG_R2_REWARD_BASELINE_AUTHORIZED"]
    ]
    nearest = max(
        task_agnostic,
        key=lambda name: sum(bool(value) for value in gates[name]["checks"].values()),
    )
    sarm_progress = float(
        summaries["frozen_sarm"]["progress"]["task_macro"]["spearman"]
    )
    sarm_failure = float(summaries["frozen_sarm"]["episode_failure"]["all"]["auprc"])
    nearest_progress = float(summaries[nearest]["progress"]["task_macro"]["spearman"])
    nearest_failure = float(summaries[nearest]["episode_failure"]["all"]["auprc"])
    result_class = (
        "Result A"
        if authorized
        else "Result B"
        if nearest_progress > sarm_progress or nearest_failure > sarm_failure
        else "Result C"
    )
    gate_payload = {
        "schema_version": "latentguard.lg_r1c.lg_r2_gate.v1",
        "status": "pass" if authorized else "not_promoted",
        "result_class": result_class,
        "authorized_models": authorized,
        "nearest_task_agnostic_baseline": nearest,
        "LG_R2_REWARD_BASELINE_AUTHORIZED": bool(authorized),
        "candidate_gates": gates,
        "thresholds_frozen_before_test": True,
    }
    write_json(destination / "lg_r2_gate.json", gate_payload)
    task_macro = {
        "schema_version": "latentguard.lg_r1c.task_macro_results.v1",
        "status": "pass",
        "models": {
            name: {
                "progress": summary["progress"]["task_macro"],
                "success": summary["success"]["task_macro"],
                "failure": summary["matched_failure"]["task_macro"],
            }
            for name, summary in summaries.items()
        },
    }
    leave = {
        "schema_version": "latentguard.lg_r1c.leave_task6_out_results.v1",
        "status": "pass",
        "diagnostic_only": True,
        "models": {
            name: {
                "episode_failure": summary["episode_failure"]["leave_task6_out"],
                "matched_failure": summary["matched_failure"]["leave_task6_out"],
            }
            for name, summary in summaries.items()
        },
    }
    failure = {
        "schema_version": "latentguard.lg_r1c.failure_signal_results.v1",
        "status": "pass",
        "natural_episode_prevalence": 27 / 320,
        "models": {
            name: {
                "episode": summary["episode_failure"],
                "window": summary["matched_failure"],
                "early_warning": summary["early_warning"],
            }
            for name, summary in summaries.items()
        },
    }
    evaluation = {
        "schema_version": "latentguard.lg_r1c.evaluation_summary.v1",
        "status": "pass",
        "result_class": result_class,
        "models": summaries,
        "gate": gate_payload,
        "window_count": len(windows),
        "anchor_primary_windows": len(anchor_rows),
        "matched_windows": len(matched_rows),
        "test_used_for_selection": False,
        "foundation_models_trained": False,
        "failure_head_trained": False,
        "intervention_executed": False,
        "new_rollouts": 0,
        "task_success_claim": False,
    }
    for name, payload in (
        ("task_macro_results.json", task_macro),
        ("leave_task6_out_results.json", leave),
        ("failure_signal_results.json", failure),
        ("evaluation_summary.json", evaluation),
    ):
        write_json(destination / name, payload)
    print(json.dumps(gate_payload, sort_keys=True))


if __name__ == "__main__":
    main()
