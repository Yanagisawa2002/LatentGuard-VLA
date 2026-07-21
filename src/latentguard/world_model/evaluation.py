"""Deterministic WM-v0 prediction, calibration, and candidate-ranking metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray


def _vectors(
    labels: NDArray[np.float64], scores: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != predictions.shape or values.size == 0:
        raise ValueError("labels and scores must be non-empty aligned vectors")
    if not bool(np.isfinite(values).all() and np.isfinite(predictions).all()):
        raise ValueError("labels and scores must be finite")
    if not bool(np.logical_or(values == 0.0, values == 1.0).all()):
        raise ValueError("labels must be binary")
    return values, predictions


def binary_auroc(
    labels: NDArray[np.float64], scores: NDArray[np.float64]
) -> float | None:
    """Return tie-aware empirical AUROC, or None for one-class targets."""

    values, predictions = _vectors(labels, scores)
    positives = predictions[values == 1.0]
    negatives = predictions[values == 0.0]
    if positives.size == 0 or negatives.size == 0:
        return None
    comparison = positives[:, None] - negatives[None, :]
    return float(
        (np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)) / comparison.size
    )


def binary_auprc(
    labels: NDArray[np.float64], scores: NDArray[np.float64]
) -> float | None:
    """Return average precision under descending score order."""

    values, predictions = _vectors(labels, scores)
    positive_count = int(values.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-predictions, kind="stable")
    ordered = values[order]
    precisions = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float(np.sum(precisions * ordered) / positive_count)


def calibration_error(
    labels: NDArray[np.float64], probabilities: NDArray[np.float64], *, bins: int
) -> float:
    """Return fixed-width expected calibration error."""

    values, predictions = _vectors(labels, probabilities)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be positive")
    if not bool(np.logical_and(predictions >= 0.0, predictions <= 1.0).all()):
        raise ValueError("probabilities must lie in [0,1]")
    assignments = np.minimum((predictions * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if bool(selected.any()):
            error += float(selected.mean()) * abs(
                float(predictions[selected].mean()) - float(values[selected].mean())
            )
    return error


def rank_correlation(
    expected: NDArray[np.float64], observed: NDArray[np.float64]
) -> float | None:
    """Return Pearson correlation of stable ordinal ranks."""

    left = np.asarray(expected, dtype=np.float64).reshape(-1)
    right = np.asarray(observed, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        return None
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(
        np.float64
    )
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(
        np.float64
    )
    if float(left_rank.std()) == 0.0 or float(right_rank.std()) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    """One outcome prediction in a shared candidate group."""

    group_id: str
    candidate_id: str
    policy_source: str
    success_probability: float
    terminal_success: bool


def candidate_ranking_metrics(
    predictions: Iterable[CandidatePrediction],
) -> dict[str, object]:
    """Evaluate pairwise order, top-1 success, and oracle regret per source."""

    groups: dict[str, list[CandidatePrediction]] = defaultdict(list)
    for prediction in predictions:
        if not 0.0 <= prediction.success_probability <= 1.0:
            raise ValueError("candidate probability must lie in [0,1]")
        groups[prediction.group_id].append(prediction)
    if not groups:
        raise ValueError("candidate prediction groups must not be empty")

    def summarize(items: list[list[CandidatePrediction]]) -> dict[str, object]:
        correct_pairs = 0
        compared_pairs = 0
        selected_success = 0
        oracle_success = 0
        for group in items:
            selected = max(
                group, key=lambda item: (item.success_probability, item.candidate_id)
            )
            selected_success += int(selected.terminal_success)
            oracle_success += int(any(item.terminal_success for item in group))
            for left_index, left in enumerate(group):
                for right in group[left_index + 1 :]:
                    if left.terminal_success == right.terminal_success:
                        continue
                    compared_pairs += 1
                    successful = left if left.terminal_success else right
                    failed = right if left.terminal_success else left
                    correct_pairs += int(
                        successful.success_probability > failed.success_probability
                    )
        group_count = len(items)
        return {
            "group_count": group_count,
            "oracle_regret": (oracle_success - selected_success) / group_count,
            "pairwise_accuracy": (
                None if compared_pairs == 0 else correct_pairs / compared_pairs
            ),
            "pairwise_support": compared_pairs,
            "top1_success_rate": selected_success / group_count,
        }

    result: dict[str, object] = {"all": summarize(list(groups.values()))}
    for source in sorted(
        {item.policy_source for group in groups.values() for item in group}
    ):
        source_groups = [
            selected
            for group in groups.values()
            if (selected := [item for item in group if item.policy_source == source])
        ]
        result[source] = summarize(source_groups)
    return result


def compare_candidate_selectors(
    predictions_by_model: dict[str, tuple[CandidatePrediction, ...]],
) -> dict[str, object]:
    """Compare three selectors only after proving identical candidate inventory."""

    required = {"direct_verifier", "outcome_only", "wm_v0"}
    if set(predictions_by_model) != required:
        raise ValueError("comparison requires direct, outcome-only, and WM-v0 models")
    inventories: dict[str, tuple[tuple[str, str, str, bool], ...]] = {}
    for name, values in predictions_by_model.items():
        inventories[name] = tuple(
            sorted(
                (
                    item.group_id,
                    item.candidate_id,
                    item.policy_source,
                    item.terminal_success,
                )
                for item in values
            )
        )
    reference = inventories["wm_v0"]
    if not reference or any(
        inventory != reference for inventory in inventories.values()
    ):
        raise ValueError("selector candidate inventories or outcomes differ")
    summaries = {
        name: candidate_ranking_metrics(values)
        for name, values in predictions_by_model.items()
    }
    top1 = {
        name: cast(float, cast(dict[str, object], result["all"])["top1_success_rate"])
        for name, result in summaries.items()
    }
    versus_outcome = top1["wm_v0"] - top1["outcome_only"]
    versus_direct = top1["wm_v0"] - top1["direct_verifier"]
    return {
        "does_future_latent_improve_selection": (
            "yes_on_point_estimate"
            if versus_outcome > 0.0
            else "no_on_point_estimate"
            if versus_outcome < 0.0
            else "tied_on_point_estimate"
        ),
        "model_ranking_metrics": summaries,
        "wm_v0_minus_direct_top1": versus_direct,
        "wm_v0_minus_outcome_only_top1": versus_outcome,
    }


def prediction_metrics(
    *,
    predicted_latents: NDArray[np.float64],
    target_latents: NDArray[np.float64],
    predicted_progress: NDArray[np.float64],
    target_progress: NDArray[np.float64],
    progress_mask: NDArray[np.bool_],
    event_probabilities: NDArray[np.float64],
    event_labels: NDArray[np.float64],
    event_mask: NDArray[np.bool_],
    event_names: tuple[str, ...],
    success_probabilities: NDArray[np.float64],
    terminal_success: NDArray[np.float64],
    terminal_success_mask: NDArray[np.bool_],
    calibration_bins: int = 10,
) -> dict[str, object]:
    """Return the complete deterministic offline metric inventory."""

    predicted = np.asarray(predicted_latents, dtype=np.float64)
    expected = np.asarray(target_latents, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 4:
        raise ValueError("latent arrays must be aligned [N,K,V,D]")
    squared = np.mean((predicted - expected) ** 2, axis=(-1, -2))
    numerator = np.sum(predicted * expected, axis=-1)
    denominator = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(expected, axis=-1)
    cosine = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    cosine = np.mean(cosine, axis=-1)
    progress_values = np.asarray(progress_mask, dtype=np.bool_)
    progress_error = np.abs(
        np.asarray(predicted_progress, dtype=np.float64)
        - np.asarray(target_progress, dtype=np.float64)
    )
    progress_expected = np.asarray(target_progress, dtype=np.float64)[progress_values]
    progress_observed = np.asarray(predicted_progress, dtype=np.float64)[
        progress_values
    ]
    event_result: dict[str, object] = {}
    for index, name in enumerate(event_names):
        selected = np.asarray(event_mask[:, :, index], dtype=np.bool_)
        labels = np.asarray(event_labels[:, :, index], dtype=np.float64)[selected]
        scores = np.asarray(event_probabilities[:, :, index], dtype=np.float64)[
            selected
        ]
        if labels.size == 0:
            event_result[name] = {"support": 0}
            continue
        predictions = scores >= 0.5
        true = labels == 1.0
        tp = int(np.logical_and(predictions, true).sum())
        fp = int(np.logical_and(predictions, ~true).sum())
        fn = int(np.logical_and(~predictions, true).sum())
        precision = None if tp + fp == 0 else tp / (tp + fp)
        recall = None if tp + fn == 0 else tp / (tp + fn)
        event_result[name] = {
            "auprc": binary_auprc(labels, scores),
            "auroc": binary_auroc(labels, scores),
            "f1": (
                None
                if precision is None or recall is None or precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            ),
            "precision": precision,
            "recall": recall,
            "support": int(labels.size),
            "positive_support": int(true.sum()),
        }
    raw_success = np.asarray(terminal_success, dtype=np.float64).reshape(-1)
    raw_success_scores = np.asarray(success_probabilities, dtype=np.float64).reshape(-1)
    success_mask = np.asarray(terminal_success_mask, dtype=np.bool_).reshape(-1)
    if (
        raw_success.shape != raw_success_scores.shape
        or raw_success.shape != success_mask.shape
    ):
        raise ValueError("terminal outcome arrays and mask must align")
    terminal_metrics: dict[str, object]
    if bool(success_mask.any()):
        success_labels, success_scores = _vectors(
            raw_success[success_mask], raw_success_scores[success_mask]
        )
        terminal_metrics = {
            "auprc": binary_auprc(success_labels, success_scores),
            "auroc": binary_auroc(success_labels, success_scores),
            "brier": float(np.mean((success_scores - success_labels) ** 2)),
            "ece": calibration_error(
                success_labels, success_scores, bins=calibration_bins
            ),
            "support": int(success_labels.size),
        }
    else:
        terminal_metrics = {
            "auprc": None,
            "auroc": None,
            "brier": None,
            "ece": None,
            "support": 0,
        }
    return {
        "events": event_result,
        "future_latent": {
            "cosine_by_horizon": [float(value) for value in cosine.mean(axis=0)],
            "cosine_mean": float(cosine.mean()),
            "mse_by_horizon": [float(value) for value in squared.mean(axis=0)],
            "mse_mean": float(squared.mean()),
        },
        "progress": {
            "mae": (
                None
                if not bool(progress_values.any())
                else float(progress_error[progress_values].mean())
            ),
            "rank_correlation": rank_correlation(progress_expected, progress_observed),
            "support": int(progress_values.sum()),
        },
        "terminal_outcome": terminal_metrics,
    }
