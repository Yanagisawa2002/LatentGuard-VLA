"""Dependency-light stage and progress metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class StageMetrics:
    """Stage classification summary."""

    accuracy: float
    macro_f1: float
    per_stage_f1: dict[int, float]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class ProgressMetrics:
    """Continuous progress regression summary."""

    mae: float
    rmse: float
    spearman: float


def evaluate_stage_predictions(
    targets: Sequence[int], predictions: Sequence[int], *, stage_count: int
) -> StageMetrics:
    """Compute accuracy, macro F1, and a fixed-size confusion matrix."""

    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    target = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if target.shape != predicted.shape or target.ndim != 1 or target.size == 0:
        raise ValueError(
            "stage targets and predictions must be equal non-empty vectors"
        )
    if (
        (target < 0).any()
        or (target >= stage_count).any()
        or (predicted < 0).any()
        or (predicted >= stage_count).any()
    ):
        raise ValueError("stage indices are outside the declared stage_count")
    matrix = np.zeros((stage_count, stage_count), dtype=np.int64)
    np.add.at(matrix, (target, predicted), 1)
    f1: dict[int, float] = {}
    for stage in range(stage_count):
        true_positive = int(matrix[stage, stage])
        false_positive = int(matrix[:, stage].sum() - true_positive)
        false_negative = int(matrix[stage, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        f1[stage] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    return StageMetrics(
        accuracy=float((target == predicted).mean()),
        macro_f1=float(np.mean(list(f1.values()))),
        per_stage_f1=f1,
        confusion_matrix=matrix.tolist(),
    )


def evaluate_progress(
    targets: Sequence[float], predictions: Sequence[float]
) -> ProgressMetrics:
    """Compute finite MAE, RMSE, and tie-aware Spearman correlation."""

    target = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if target.shape != predicted.shape or target.ndim != 1 or target.size == 0:
        raise ValueError(
            "progress targets and predictions must be equal non-empty vectors"
        )
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise ValueError("progress metrics require finite inputs")
    errors = predicted - target
    return ProgressMetrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(math.sqrt(float(np.mean(errors**2)))),
        spearman=_pearson(_rankdata(target), _rankdata(predicted)),
    )


def pairwise_accuracy(
    target_delta: Sequence[float],
    predicted_delta: Sequence[float],
    *,
    epsilon: float,
) -> float:
    """Evaluate positive/tied/negative pairwise ordering."""

    target = np.asarray(target_delta, dtype=np.float64)
    predicted = np.asarray(predicted_delta, dtype=np.float64)
    if target.shape != predicted.shape or target.ndim != 1 or target.size == 0:
        raise ValueError("pairwise deltas must be equal non-empty vectors")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    target_sign = np.where(target > epsilon, 1, np.where(target < -epsilon, -1, 0))
    predicted_sign = np.where(
        predicted > epsilon, 1, np.where(predicted < -epsilon, -1, 0)
    )
    return float((target_sign == predicted_sign).mean())


def _rankdata(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def _pearson(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(float(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)
