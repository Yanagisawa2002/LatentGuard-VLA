"""Aggregate grouped-CV LG-R2a metrics, uncertainty, and sensitivity."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2a_common import (
    output_root,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    write_json,
)

from latentguard.action_conditioning.gates import evaluate_lg_r2b_gate
from latentguard.rewards.metrics import (
    binary_metrics,
    recall_operating_point,
    task_macro_binary_metrics,
)

HORIZON_NAMES = ("short", "medium", "long")
CLASSIFICATION_TARGETS = (
    "stagnation_short",
    "regression_short",
    "OBJECT_DROP",
    "FAILED_PLACEMENT",
    "terminal_failure",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def _spearman(target: np.ndarray, score: np.ndarray) -> float:
    left = _rank(target)
    right = _rank(score)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _kendall(target: np.ndarray, score: np.ndarray) -> float:
    try:
        from scipy.stats import kendalltau

        value = kendalltau(target, score, variant="b").statistic
        return 0.0 if value is None or not math.isfinite(value) else float(value)
    except ImportError:
        if target.size > 2_000:
            positions = np.linspace(0, target.size - 1, 2_000, dtype=np.int64)
            target = target[positions]
            score = score[positions]
        concordant = 0
        discordant = 0
        x_ties = 0
        y_ties = 0
        for first in range(target.size):
            dx = target[first + 1 :] - target[first]
            dy = score[first + 1 :] - score[first]
            concordant += int(np.sum(dx * dy > 0))
            discordant += int(np.sum(dx * dy < 0))
            x_ties += int(np.sum((dx == 0) & (dy != 0)))
            y_ties += int(np.sum((dy == 0) & (dx != 0)))
        denominator = math.sqrt(
            (concordant + discordant + x_ties) * (concordant + discordant + y_ties)
        )
        return (concordant - discordant) / denominator if denominator else 0.0


def _progress_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    difference = score - target
    epsilon = 0.02
    target_sign = np.where(target > epsilon, 1, np.where(target < -epsilon, -1, 0))
    score_sign = np.where(score > epsilon, 1, np.where(score < -epsilon, -1, 0))
    return {
        "samples": int(target.size),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "spearman": _spearman(target, score),
        "kendall_tau_b": _kendall(target, score),
        "sign_accuracy": float(np.mean(target_sign == score_sign)),
    }


def _classification_arrays(
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "stagnation_short":
        return data["stagnation"][:, 0], predictions["stagnation"][:, 0]
    if name == "regression_short":
        return data["regression"][:, 0], predictions["regression"][:, 0]
    if name == "OBJECT_DROP":
        return data["events"][:, 0], predictions["events"][:, 0]
    if name == "FAILED_PLACEMENT":
        return data["events"][:, 1], predictions["events"][:, 1]
    if name == "terminal_failure":
        return data["terminal"][:, 0], predictions["terminal"][:, 0]
    raise ValueError(f"unknown classification target: {name}")


def _fold_thresholds(
    run_manifest: dict[str, Any],
    folds: np.ndarray,
    name: str,
) -> np.ndarray:
    if name == "stagnation_short":
        key, column = "stagnation", 0
    elif name == "regression_short":
        key, column = "regression", 0
    elif name == "OBJECT_DROP":
        key, column = "events", 0
    elif name == "FAILED_PLACEMENT":
        key, column = "events", 1
    else:
        key, column = "terminal", 0
    by_fold = {
        int(row["fold"]): float(row["selected_thresholds"][key][column])
        for row in run_manifest["folds"]
    }
    return np.asarray([by_fold[int(fold)] for fold in folds], dtype=np.float64)


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (scores >= lower) & (scores <= upper if upper >= 1.0 else scores < upper)
        if mask.any():
            total += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(scores[mask].mean())
            )
    return total


def _binary_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    tasks: np.ndarray,
) -> dict[str, Any]:
    base = binary_metrics(
        labels.astype(int).tolist(),
        scores.tolist(),
        probability_scores=scores.tolist(),
    ).to_dict()
    predicted = scores >= thresholds
    true_positive = int(np.sum(predicted & (labels == 1)))
    false_positive = int(np.sum(predicted & (labels == 0)))
    false_negative = int(np.sum(~predicted & (labels == 1)))
    true_negative = int(np.sum(~predicted & (labels == 0)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    operating: dict[str, Any] | None = None
    if labels.sum() > 0:
        operating = recall_operating_point(
            labels.astype(int).tolist(),
            scores.tolist(),
            requested_recall=0.5,
        ).to_dict()
    return {
        **base,
        "precision_validation_selected_threshold": precision,
        "recall_validation_selected_threshold": recall,
        "f1_validation_selected_threshold": f1,
        "false_positive_rate_validation_selected_threshold": (
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else 0.0
        ),
        "descriptive_test_curve_at_recall_0_5": operating,
        "expected_calibration_error_10_bins": _ece(labels, scores),
        "task_macro": task_macro_binary_metrics(
            labels.astype(int).tolist(), scores.tolist(), tasks.tolist()
        ),
    }


def _model_metrics(
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    run_manifest: dict[str, Any],
    metadata: list[dict[str, Any]],
    mask: np.ndarray,
) -> dict[str, Any]:
    folds = np.asarray([int(row["fold"]) for row in metadata], dtype=np.int64)
    tasks = np.asarray([str(row["task"]) for row in metadata])
    progress_results: dict[str, Any] = {}
    for column, horizon in enumerate(HORIZON_NAMES):
        valid = mask & data["progress_valid"][:, column]
        progress_results[horizon] = _progress_metrics(
            data["progress"][valid, column], predictions["progress"][valid, column]
        )
        per_task = {}
        for task in sorted(set(tasks[valid].tolist())):
            task_mask = valid & (tasks == task)
            per_task[task] = _progress_metrics(
                data["progress"][task_mask, column],
                predictions["progress"][task_mask, column],
            )
        progress_results[horizon]["task_macro"] = {
            "eligible_tasks": len(per_task),
            "mae": float(np.mean([row["mae"] for row in per_task.values()])),
            "spearman": float(np.mean([row["spearman"] for row in per_task.values()])),
            "per_task": per_task,
        }
    classification = {}
    for name in CLASSIFICATION_TARGETS:
        labels, scores = _classification_arrays(data, predictions, name)
        thresholds = _fold_thresholds(run_manifest, folds, name)
        classification[name] = _binary_summary(
            labels[mask], scores[mask], thresholds[mask], tasks[mask]
        )
    return {"progress_delta": progress_results, "classification": classification}


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {
        key: payload[key]
        for key in ("progress", "stagnation", "regression", "events", "terminal")
    }


def _bootstrap_delta(
    *,
    metadata: list[dict[str, Any]],
    repeats: int,
    seed: int,
    metric: Any,
) -> dict[str, Any]:
    episode_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        episode_indices[str(row["episode_id"])].append(index)
    episodes = sorted(episode_indices)
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = generator.choice(episodes, size=len(episodes), replace=True)
        indices = np.concatenate(
            [
                np.asarray(episode_indices[str(episode)], dtype=np.int64)
                for episode in sampled
            ]
        )
        values.append(float(metric(indices)))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "repeats": repeats,
        "unit": "episode",
        "seed": seed,
        "mean": float(np.mean(values)),
        "ci_95": [float(lower), float(upper)],
    }


def _subset_model(model: dict[str, Any], subset_name: str) -> dict[str, Any]:
    return {
        "progress_short": model["progress_delta"]["short"],
        "classification": model["classification"],
        "subset": subset_name,
    }


def main() -> None:
    """Produce all compact model, generalization, sensitivity, and gate artifacts."""

    args = parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    if args.dry_run:
        print({"status": "dry_run", "bootstrap_repeats": config["bootstrap_repeats"]})
        return
    output = output_root(args.output_dir)
    prepared = np.load(output / "prepared_data.npz", allow_pickle=False)
    data = {key: prepared[key] for key in prepared.files}
    metadata = read_jsonl(output / "prepared_metadata.jsonl")
    sample_count = len(metadata)
    all_mask = np.ones(sample_count, dtype=np.bool_)
    tasks = np.asarray([str(row["task"]) for row in metadata])
    folds = np.asarray([int(row["fold"]) for row in metadata], dtype=np.int64)
    variants = {
        "state_only": "state_only",
        "action_only": "action_only",
        "state_action": "state_action",
        "state_action_proprio": "state_action_proprio",
    }
    predictions = {
        key: _load_predictions(output / "runs" / folder / "oof_predictions.npz")
        for key, folder in variants.items()
    }
    manifests = {
        key: read_json(output / "runs" / folder / "run_manifest.json")
        for key, folder in variants.items()
    }
    metrics = {
        key: _model_metrics(data, predictions[key], manifests[key], metadata, all_mask)
        for key in variants
    }
    without_task6 = tasks != "libero_10/task6"
    leave_task6 = {
        key: _model_metrics(
            data, predictions[key], manifests[key], metadata, without_task6
        )
        for key in variants
    }
    failure_tasks = {
        str(row["task"])
        for row in read_json(output / "cv_split_manifest.json")["assignments"]
        if not bool(row["success"])
    }
    failure_task_mask = np.isin(tasks, sorted(failure_tasks))
    failure_task_macro = {
        key: _model_metrics(
            data, predictions[key], manifests[key], metadata, failure_task_mask
        )
        for key in variants
    }
    original_split_metrics = {}
    original_splits = np.asarray([str(row["original_split"]) for row in metadata])
    for split in sorted(set(original_splits.tolist())):
        split_mask = original_splits == split
        original_split_metrics[split] = {
            key: _model_metrics(
                data, predictions[key], manifests[key], metadata, split_mask
            )
            for key in variants
        }

    state = metrics["state_only"]
    combined = metrics["state_action"]
    short_mae_state = state["progress_delta"]["short"]["mae"]
    short_mae_combined = combined["progress_delta"]["short"]["mae"]
    short_mae_reduction = (short_mae_state - short_mae_combined) / short_mae_state
    short_spearman_delta = (
        combined["progress_delta"]["short"]["spearman"]
        - state["progress_delta"]["short"]["spearman"]
    )
    class_deltas = {
        name: (
            combined["classification"][name]["auprc"]
            - state["classification"][name]["auprc"]
        )
        for name in CLASSIFICATION_TARGETS
    }
    best_class_name = max(
        class_deltas,
        key=lambda name: (
            -math.inf if class_deltas[name] is None else class_deltas[name]
        ),
    )
    best_class_delta = float(class_deltas[best_class_name] or 0.0)
    repeats = int(config["bootstrap_repeats"])
    bootstrap_seed = int(config["bootstrap_seed"])
    progress_ci = _bootstrap_delta(
        metadata=metadata,
        repeats=repeats,
        seed=bootstrap_seed,
        metric=lambda indices: (
            np.mean(
                np.abs(
                    data["progress"][indices, 0]
                    - predictions["state_action"]["progress"][indices, 0]
                )
            )
            - np.mean(
                np.abs(
                    data["progress"][indices, 0]
                    - predictions["state_only"]["progress"][indices, 0]
                )
            )
        ),
    )
    combined_class_labels, combined_class_scores = _classification_arrays(
        data, predictions["state_action"], best_class_name
    )
    state_class_labels, state_class_scores = _classification_arrays(
        data, predictions["state_only"], best_class_name
    )
    class_ci = _bootstrap_delta(
        metadata=metadata,
        repeats=repeats,
        seed=bootstrap_seed + 1,
        metric=lambda indices: (
            (
                binary_metrics(
                    combined_class_labels[indices].astype(int).tolist(),
                    combined_class_scores[indices].tolist(),
                ).auprc
                or 0.0
            )
            - (
                binary_metrics(
                    state_class_labels[indices].astype(int).tolist(),
                    state_class_scores[indices].tolist(),
                ).auprc
                or 0.0
            )
        ),
    )
    fold_progress_deltas = []
    for fold in range(5):
        fold_mask = folds == fold
        target = data["progress"][fold_mask, 0]
        state_mae = float(
            np.mean(
                np.abs(target - predictions["state_only"]["progress"][fold_mask, 0])
            )
        )
        combined_mae = float(
            np.mean(
                np.abs(target - predictions["state_action"]["progress"][fold_mask, 0])
            )
        )
        fold_progress_deltas.append(
            {
                "fold": fold,
                "state_only_mae": state_mae,
                "state_action_mae": combined_mae,
                "direction_positive": combined_mae < state_mae,
            }
        )
    positive_folds = sum(row["direction_positive"] for row in fold_progress_deltas)

    sensitivity_payload = np.load(
        output / "sensitivity_predictions.npz", allow_pickle=False
    )
    sensitivity_full = {
        key: sensitivity_payload[f"full__{key}"]
        for key in ("progress", "stagnation", "regression", "events", "terminal")
    }
    sensitivity_permuted = {
        key: sensitivity_payload[f"permuted_real__{key}"]
        for key in ("progress", "stagnation", "regression", "events", "terminal")
    }
    combined_run = manifests["state_action"]
    permuted_metrics = _model_metrics(
        data, sensitivity_permuted, combined_run, metadata, all_mask
    )
    full_sensitivity_metrics = _model_metrics(
        data, sensitivity_full, combined_run, metadata, all_mask
    )
    permuted_short_mae = permuted_metrics["progress_delta"]["short"]["mae"]
    permutation_mae_worsening = (
        permuted_short_mae - full_sensitivity_metrics["progress_delta"]["short"]["mae"]
    ) / full_sensitivity_metrics["progress_delta"]["short"]["mae"]
    permutation_class_drops = {
        name: (
            full_sensitivity_metrics["classification"][name]["auprc"]
            - permuted_metrics["classification"][name]["auprc"]
        )
        for name in CLASSIFICATION_TARGETS
    }
    largest_permutation_drop = max(
        float(value or 0.0) for value in permutation_class_drops.values()
    )
    sensitivity_run = read_json(output / "action_sensitivity_run.json")
    swap_predictions = {
        key: sensitivity_payload[f"swap_surrogate_real__{key}"]
        for key in ("progress", "stagnation", "regression", "events", "terminal")
    }
    swap_changes = {
        key: float(np.mean(np.abs(swap_predictions[key] - sensitivity_full[key])))
        for key in swap_predictions
    }
    ablation_results = {}
    for name in ("first_action_only", "endpoint_summary", "action_masked"):
        ablation_predictions = {
            key: sensitivity_payload[f"{name}__{key}"]
            for key in ("progress", "stagnation", "regression", "events", "terminal")
        }
        ablation_results[name] = _subset_model(
            _model_metrics(
                data,
                ablation_predictions,
                combined_run,
                metadata,
                all_mask,
            ),
            name,
        )

    best_task_macro_delta = float(
        combined["classification"][best_class_name]["task_macro"]["auprc"]
        - state["classification"][best_class_name]["task_macro"]["auprc"]
    )
    leave_task6_best = leave_task6["state_action"]["classification"][best_class_name]
    summary_for_gate = {
        "required_checks": {
            "episode_leakage": read_json(output / "input_dataset_manifest.json")[
                "counts"
            ]["episode_leakage"],
            "seed_leakage": read_json(output / "input_dataset_manifest.json")["counts"][
                "seed_leakage"
            ],
            "foundation_models_frozen": True,
            "action_contract_status": read_json(output / "action_contract.json")[
                "validation"
            ]["status"],
            "cv_status": read_json(output / "cv_split_manifest.json")["status"],
        },
        "incremental_value": {
            "short_progress_mae_relative_reduction": short_mae_reduction,
            "short_progress_spearman_delta": short_spearman_delta,
            "best_action_sensitive_classification_target": best_class_name,
            "best_action_sensitive_classification_auprc_delta": best_class_delta,
        },
        "permutation_sensitivity": {
            "short_progress_mae_relative_worsening": permutation_mae_worsening,
            "largest_classification_auprc_drop": largest_permutation_drop,
        },
        "robustness": {
            "positive_folds": positive_folds,
            "task_macro_delta": best_task_macro_delta,
            "leave_task6_best_auprc": leave_task6_best["auprc"],
            "leave_task6_corresponding_prevalence": leave_task6_best["prevalence"],
        },
        "deployability": {
            "pre_execution_only": True,
            "model_e_used_for_gate": False,
        },
    }
    gate = evaluate_lg_r2b_gate(summary_for_gate)
    if gate["LG_R2B_AUTHORIZED"]:
        result_class = "Result A"
    elif (
        short_mae_reduction > 0
        or best_class_delta > 0
        or permutation_mae_worsening > 0
        or largest_permutation_drop > 0
    ):
        result_class = "Result B"
    else:
        result_class = "Result C"
    evaluation_summary = {
        "schema_version": "latentguard.lg_r2a.evaluation_summary.v1",
        "status": "pass",
        "result_class": result_class,
        "research_question": (
            "Does explicit numeric executed action add stable, task-balanced, "
            "pre-execution predictive information beyond current VLA representation?"
        ),
        "samples": sample_count,
        "incremental_value": summary_for_gate["incremental_value"],
        "permutation_sensitivity": {
            **summary_for_gate["permutation_sensitivity"],
            "classification_auprc_drops": permutation_class_drops,
        },
        "robustness": {
            **summary_for_gate["robustness"],
            "folds": fold_progress_deltas,
        },
        "bootstrap": {
            "short_progress_mae_difference_state_action_minus_state_only": progress_ci,
            f"{best_class_name}_auprc_difference": class_ci,
        },
        "model_metrics": metrics,
        "original_split_compatibility": original_split_metrics,
        "failure_task_macro": failure_task_macro,
        "gate": {**gate, "inputs": summary_for_gate},
        "claims": {
            "causal_action_consequence_proved": False,
            "same_state_candidate_comparison_proved": False,
            "world_model_planning": False,
            "safety_detection": False,
            "online_intervention_authorized": False,
        },
        "new_rollouts": 0,
        "candidate_generation": False,
        "candidate_ranking": False,
        "intervention": False,
    }
    write_json(output / "evaluation_summary.json", evaluation_summary)
    for name, model_key in (
        ("state_only_results.json", "state_only"),
        ("action_only_results.json", "action_only"),
        ("state_action_results.json", "state_action"),
    ):
        write_json(
            output / name,
            {
                "schema_version": f"latentguard.lg_r2a.{model_key}.results.v1",
                "status": "pass",
                "model": model_key,
                "metrics": metrics[model_key],
                "folds": manifests[model_key]["folds"],
                "runtime": {
                    "peak_gpu_memory_bytes": manifests[model_key][
                        "peak_gpu_memory_bytes"
                    ],
                    "elapsed_seconds": manifests[model_key]["elapsed_seconds"],
                },
            },
        )
    write_json(
        output / "internal_token_results.json",
        {
            "schema_version": "latentguard.lg_r2a.internal_token.results.v1",
            "status": "not_run_optional",
            "reason": (
                "Frozen LG-R1c cache contains current pooled Qwen representation "
                "only; recomputing action-token or JEPA temporal caches was not "
                "necessary for the primary numeric-action identifiability question."
            ),
            "self_generated_action_only_if_run": True,
            "general_external_candidate_deployable": False,
            "used_for_gate": False,
        },
    )
    write_json(
        output / "action_permutation_results.json",
        {
            "schema_version": "latentguard.lg_r2a.action_permutation.results.v1",
            "status": "pass",
            "protocol": sensitivity_run["real_action_permutation"],
            "full_metrics": _subset_model(full_sensitivity_metrics, "full"),
            "permuted_metrics": _subset_model(permuted_metrics, "permuted_real"),
            "short_progress_mae_relative_worsening": permutation_mae_worsening,
            "classification_auprc_drops": permutation_class_drops,
            "largest_classification_auprc_drop": largest_permutation_drop,
        },
    )
    write_json(
        output / "action_swap_surrogate_results.json",
        {
            "schema_version": "latentguard.lg_r2a.action_swap_surrogate.results.v1",
            "status": "pass",
            "protocol": sensitivity_run["same_state_swap_surrogate"],
            "mean_absolute_prediction_changes": swap_changes,
            "true_same_state_counterfactual": False,
            "interpretation": "offline action-sensitivity surrogate only",
        },
    )
    write_json(
        output / "task_macro_results.json",
        {
            "schema_version": "latentguard.lg_r2a.task_macro.results.v1",
            "status": "pass",
            "all_tasks_equal_weight": True,
            "models": {
                key: {
                    "progress": {
                        horizon: metrics[key]["progress_delta"][horizon]["task_macro"]
                        for horizon in HORIZON_NAMES
                    },
                    "classification": {
                        name: metrics[key]["classification"][name]["task_macro"]
                        for name in CLASSIFICATION_TARGETS
                    },
                }
                for key in variants
            },
        },
    )
    write_json(
        output / "leave_task6_out_results.json",
        {
            "schema_version": "latentguard.lg_r2a.leave_task6_out.results.v1",
            "status": "pass",
            "excluded_task": "libero_10/task6",
            "samples": int(without_task6.sum()),
            "models": leave_task6,
        },
    )
    write_json(
        output / "lg_r2b_gate.json",
        {
            **gate,
            "result_class": result_class,
            "inputs": summary_for_gate,
        },
    )
    write_json(
        output / "action_ablation_results.json",
        {
            "schema_version": "latentguard.lg_r2a.action_ablation.results.v1",
            "status": "pass",
            "results": ablation_results,
            "attribution": {
                "method": "absolute input gradient of short progress output",
                "by_action_dimension": sensitivity_run[
                    "short_progress_input_gradient_absolute_mean_by_action_dimension"
                ],
            },
        },
    )
    print(
        {
            "status": "pass",
            "result_class": result_class,
            "LG_R2B_AUTHORIZED": gate["LG_R2B_AUTHORIZED"],
        }
    )


if __name__ == "__main__":
    main()
