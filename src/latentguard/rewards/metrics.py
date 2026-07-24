"""Dependency-light metrics for frozen LG-R1c reward predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from latentguard.progress.metrics import evaluate_progress


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary ranking and probability metrics."""

    samples: int
    positives: int
    prevalence: float
    auroc: float | None
    auprc: float | None
    brier: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible metric mapping."""

        return asdict(self)


@dataclass(frozen=True)
class RecallOperatingPoint:
    """Highest-score threshold that reaches a requested recall."""

    requested_recall: float
    threshold: float
    recall: float
    precision: float
    false_positive_rate: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-compatible operating point."""

        return asdict(self)


def binary_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    probability_scores: Sequence[float] | None = None,
) -> BinaryMetrics:
    """Compute tie-aware AUROC, average precision, prevalence, and Brier."""

    target, value = _binary_arrays(labels, scores)
    probabilities = (
        value
        if probability_scores is None
        else np.asarray(probability_scores, dtype=np.float64)
    )
    if probabilities.shape != target.shape or not np.isfinite(probabilities).all():
        raise ValueError("probability scores must be a finite vector matching labels")
    positives = int(target.sum())
    negatives = int(target.size - positives)
    auroc = _auroc(target, value) if positives and negatives else None
    auprc = _average_precision(target, value) if positives else None
    return BinaryMetrics(
        samples=int(target.size),
        positives=positives,
        prevalence=float(target.mean()),
        auroc=auroc,
        auprc=auprc,
        brier=float(np.mean((probabilities - target) ** 2)),
    )


def recall_operating_point(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    *,
    requested_recall: float,
) -> RecallOperatingPoint:
    """Select the highest threshold whose empirical recall reaches the target."""

    target, value = _binary_arrays(labels, scores)
    if not 0.0 < requested_recall <= 1.0:
        raise ValueError("requested recall must lie in (0, 1]")
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0:
        raise ValueError("recall operating point requires positive examples")
    order = np.argsort(-value, kind="mergesort")
    sorted_target = target[order]
    sorted_value = value[order]
    true_positive = 0
    false_positive = 0
    for index, label in enumerate(sorted_target):
        true_positive += int(label)
        false_positive += int(not label)
        at_boundary = index == len(sorted_target) - 1 or (
            sorted_value[index + 1] < sorted_value[index]
        )
        recall = true_positive / positives
        if at_boundary and recall >= requested_recall:
            precision = true_positive / (true_positive + false_positive)
            false_positive_rate = false_positive / negatives if negatives else 0.0
            return RecallOperatingPoint(
                requested_recall=requested_recall,
                threshold=float(sorted_value[index]),
                recall=float(recall),
                precision=float(precision),
                false_positive_rate=float(false_positive_rate),
            )
    raise RuntimeError("recall operating point was not found")


def progress_metrics(
    targets: Sequence[float],
    scores: Sequence[float],
) -> dict[str, float | int]:
    """Compute regression, rank, and Kendall metrics for progress."""

    summary = evaluate_progress(targets, scores)
    return {
        "samples": len(targets),
        "mae": summary.mae,
        "rmse": summary.rmse,
        "spearman": summary.spearman,
        "kendall_tau_b": kendall_tau_b(targets, scores),
    }


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Kendall tau-b with explicit tie accounting."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError(
            "Kendall inputs must be equal vectors with at least two values"
        )
    concordant = 0
    discordant = 0
    x_ties = 0
    y_ties = 0
    for first in range(x.size):
        for second in range(first + 1, x.size):
            delta_x = x[second] - x[first]
            delta_y = y[second] - y[first]
            if delta_x == 0.0 and delta_y == 0.0:
                continue
            if delta_x == 0.0:
                x_ties += 1
            elif delta_y == 0.0:
                y_ties += 1
            elif delta_x * delta_y > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + x_ties) * (concordant + discordant + y_ties)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def task_macro_binary_metrics(
    labels: Sequence[int | bool],
    scores: Sequence[float],
    tasks: Sequence[str],
) -> dict[str, Any]:
    """Give every task with both classes equal weight."""

    target, value = _binary_arrays(labels, scores)
    task_values = np.asarray(tasks)
    if task_values.shape != target.shape:
        raise ValueError("tasks must match labels")
    per_task: dict[str, dict[str, Any]] = {}
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    for task in sorted(set(str(item) for item in task_values.tolist())):
        mask = task_values == task
        summary = binary_metrics(target[mask].tolist(), value[mask].tolist())
        per_task[task] = summary.to_dict()
        if summary.auroc is not None:
            auroc_values.append(summary.auroc)
        if summary.auprc is not None:
            auprc_values.append(summary.auprc)
    return {
        "eligible_auroc_tasks": len(auroc_values),
        "eligible_auprc_tasks": len(auprc_values),
        "auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "auprc": float(np.mean(auprc_values)) if auprc_values else None,
        "per_task": per_task,
    }


def pairwise_progress_accuracy(
    episode_ids: Sequence[str],
    frame_indices: Sequence[int],
    targets: Sequence[float],
    scores: Sequence[float],
    *,
    epsilon: float = 0.02,
) -> float:
    """Compare consecutive within-episode progress deltas."""

    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    episodes = np.asarray(episode_ids)
    frames = np.asarray(frame_indices, dtype=np.int64)
    target = np.asarray(targets, dtype=np.float64)
    value = np.asarray(scores, dtype=np.float64)
    if not (episodes.shape == frames.shape == target.shape == value.shape):
        raise ValueError("pairwise progress inputs must have equal shapes")
    target_signs: list[int] = []
    score_signs: list[int] = []
    for episode in sorted(set(str(item) for item in episodes.tolist())):
        positions = np.flatnonzero(episodes == episode)
        positions = positions[np.argsort(frames[positions], kind="mergesort")]
        for left, right in zip(positions, positions[1:], strict=False):
            target_signs.append(_sign(target[right] - target[left], epsilon))
            score_signs.append(_sign(value[right] - value[left], epsilon))
    if not target_signs:
        raise ValueError("pairwise progress requires consecutive samples")
    return float(np.mean(np.asarray(target_signs) == np.asarray(score_signs)))


def _binary_arrays(
    labels: Sequence[int | bool],
    scores: Sequence[float],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    target = np.asarray(labels)
    value = np.asarray(scores, dtype=np.float64)
    if target.shape != value.shape or target.ndim != 1 or target.size == 0:
        raise ValueError("binary labels and scores must be equal non-empty vectors")
    if not np.isfinite(value).all():
        raise ValueError("binary scores must be finite")
    if not np.isin(target, [0, 1, False, True]).all():
        raise ValueError("binary labels must contain only 0 and 1")
    return target.astype(np.bool_), value


def _average_precision(
    target: npt.NDArray[np.bool_],
    value: npt.NDArray[np.float64],
) -> float:
    order = np.argsort(-value, kind="mergesort")
    ranked = target[order]
    positives = int(target.sum())
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, target.size + 1)
    return float(np.sum(precision * ranked) / positives)


def _auroc(
    target: npt.NDArray[np.bool_],
    value: npt.NDArray[np.float64],
) -> float:
    ranks = _rankdata(value)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    positive_rank_sum = float(ranks[target].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _rankdata(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _sign(value: float, epsilon: float) -> int:
    return 1 if value > epsilon else -1 if value < -epsilon else 0
