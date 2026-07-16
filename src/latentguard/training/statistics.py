"""Trajectory-level paired bootstrap comparisons for M3B predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from latentguard.training.metrics import (
    GroupCandidate,
    MetricStatus,
    MetricValue,
    average_precision_score,
)


def _fail(context: str, reason: str) -> NoReturn:
    raise ValueError(f"{context}: {reason}")


def _targets(value: ArrayLike) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "biu":
        _fail("targets", "expected a rank-one boolean/integer array")
    result = np.asarray(array, dtype=np.int64)
    if not bool(np.all((result == 0) | (result == 1))):
        _fail("targets", "values must be zero or one")
    return result


def _probabilities(value: ArrayLike, count: int, context: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != count or array.dtype.kind not in "fiu":
        _fail(context, "expected a matching rank-one numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))) or not bool(
        np.all((result >= 0.0) & (result <= 1.0))
    ):
        _fail(context, "values must be finite and lie in [0, 1]")
    return result


def _trajectories(value: Sequence[str], count: int) -> NDArray[np.str_]:
    if len(value) != count:
        _fail("source_trajectory_ids", "length must match targets")
    if any(not item or item != item.strip() for item in value):
        _fail("source_trajectory_ids", "values must be canonical text")
    return np.asarray(value, dtype=np.str_)


@dataclass(frozen=True, slots=True)
class BootstrapIntervalReport:
    """One deterministic trajectory-bootstrap difference interval."""

    metric: str
    difference_semantic: str
    sampling_unit: str
    requested_resamples: int
    valid_resamples: int
    skipped_resamples: int
    confidence_level: float
    bootstrap_seed: int
    observed_difference: MetricValue
    confidence_lower: float | None
    confidence_upper: float | None
    interval_method: str = "trajectory_percentile_linear_v1"

    def __post_init__(self) -> None:
        """Reject non-finite, incomplete, or inconsistent intervals."""

        if type(self.requested_resamples) is not int or self.requested_resamples <= 0:
            _fail("requested_resamples", "expected a positive integer")
        if (
            type(self.valid_resamples) is not int
            or type(self.skipped_resamples) is not int
            or self.valid_resamples < 0
            or self.skipped_resamples < 0
            or self.valid_resamples + self.skipped_resamples != self.requested_resamples
        ):
            _fail("resamples", "valid and skipped inventories must be exact")
        if not math.isfinite(self.confidence_level) or not (
            0.0 < self.confidence_level < 1.0
        ):
            _fail("confidence_level", "expected a finite value in (0, 1)")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            _fail("bootstrap_seed", "expected a non-negative integer")
        if self.valid_resamples:
            if (
                self.confidence_lower is None
                or self.confidence_upper is None
                or not math.isfinite(self.confidence_lower)
                or not math.isfinite(self.confidence_upper)
                or self.confidence_lower > self.confidence_upper
            ):
                _fail("confidence", "valid resamples require finite ordered bounds")
        elif self.confidence_lower is not None or self.confidence_upper is not None:
            _fail("confidence", "zero valid resamples cannot carry bounds")
        if self.sampling_unit != "original_source_trajectory":
            _fail("sampling_unit", "M3B comparisons must resample trajectories")
        if self.difference_semantic != "first_model_minus_second_model":
            _fail("difference_semantic", "unsupported comparison direction")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native interval report."""

        return {
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "difference_semantic": self.difference_semantic,
            "interval_method": self.interval_method,
            "metric": self.metric,
            "observed_difference": self.observed_difference.to_dict(),
            "requested_resamples": self.requested_resamples,
            "sampling_unit": self.sampling_unit,
            "skipped_resamples": self.skipped_resamples,
            "valid_resamples": self.valid_resamples,
        }


def _brier(targets: NDArray[np.int64], probabilities: NDArray[np.float64]) -> float:
    return float(np.mean((probabilities - targets) ** 2))


def _failure_auprc(
    targets: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> float | None:
    value = average_precision_score(targets, probabilities)
    return value.value


def _make_report(
    *,
    metric: str,
    observed: float | None,
    replicates: Sequence[float],
    requested_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
    undefined_reason: str,
) -> BootstrapIntervalReport:
    if observed is None:
        observed_value = MetricValue.undefined(
            MetricStatus.SINGLE_CLASS, undefined_reason
        )
    else:
        observed_value = MetricValue.defined(observed)
    if replicates:
        alpha = (1.0 - confidence_level) / 2.0
        array = np.asarray(replicates, dtype=np.float64)
        lower = float(np.quantile(array, alpha, method="linear"))
        upper = float(np.quantile(array, 1.0 - alpha, method="linear"))
    else:
        lower = None
        upper = None
    return BootstrapIntervalReport(
        metric=metric,
        difference_semantic="first_model_minus_second_model",
        sampling_unit="original_source_trajectory",
        requested_resamples=requested_resamples,
        valid_resamples=len(replicates),
        skipped_resamples=requested_resamples - len(replicates),
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        observed_difference=observed_value,
        confidence_lower=lower,
        confidence_upper=upper,
    )


def paired_trajectory_bootstrap(
    targets: ArrayLike,
    first_failure_probabilities: ArrayLike,
    second_failure_probabilities: ArrayLike,
    source_trajectory_ids: Sequence[str],
    *,
    metric: str,
    resamples: int = 2000,
    seed: int = 1729,
    confidence_level: float = 0.95,
) -> BootstrapIntervalReport:
    """Compare two models by resampling complete source trajectories."""

    labels = _targets(targets)
    first = _probabilities(
        first_failure_probabilities, labels.size, "first_failure_probabilities"
    )
    second = _probabilities(
        second_failure_probabilities, labels.size, "second_failure_probabilities"
    )
    trajectories = _trajectories(source_trajectory_ids, labels.size)
    if labels.size == 0:
        _fail("targets", "bootstrap comparison requires samples")
    if type(resamples) is not int or resamples <= 0:
        _fail("resamples", "expected a positive integer")
    if type(seed) is not int or seed < 0:
        _fail("seed", "expected a non-negative integer")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        _fail("confidence_level", "expected a finite value in (0, 1)")
    metric_function: Callable[[NDArray[np.int64], NDArray[np.float64]], float | None]
    if metric == "failure_auprc":
        metric_function = _failure_auprc
    elif metric == "brier_score":
        metric_function = _brier
    else:
        _fail("metric", "expected failure_auprc or brier_score")
    observed_first = metric_function(labels, first)
    observed_second = metric_function(labels, second)
    observed = (
        None
        if observed_first is None or observed_second is None
        else observed_first - observed_second
    )
    trajectory_inventory = tuple(sorted(set(trajectories.tolist())))
    by_trajectory = {
        trajectory: np.flatnonzero(trajectories == trajectory)
        for trajectory in trajectory_inventory
    }
    generator = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(resamples):
        sampled = generator.choice(
            trajectory_inventory,
            size=len(trajectory_inventory),
            replace=True,
        )
        indices = np.concatenate([by_trajectory[str(item)] for item in sampled])
        first_value = metric_function(labels[indices], first[indices])
        second_value = metric_function(labels[indices], second[indices])
        if first_value is not None and second_value is not None:
            differences.append(first_value - second_value)
    return _make_report(
        metric=metric,
        observed=observed,
        replicates=differences,
        requested_resamples=resamples,
        confidence_level=confidence_level,
        bootstrap_seed=seed,
        undefined_reason="failure AUPRC requires both classes",
    )


def _paired_group_outcomes(
    first_candidates: Sequence[GroupCandidate],
    second_candidates: Sequence[GroupCandidate],
) -> tuple[dict[str, list[float]], float | None]:
    first_by_id = {item.sample_id: item for item in first_candidates}
    second_by_id = {item.sample_id: item for item in second_candidates}
    if len(first_by_id) != len(first_candidates) or len(second_by_id) != len(
        second_candidates
    ):
        _fail("candidates", "sample identifiers must be unique")
    if set(first_by_id) != set(second_by_id):
        _fail("candidates", "paired models must cover identical samples")
    for sample_id, first in first_by_id.items():
        second = second_by_id[sample_id]
        if (
            first.group_id != second.group_id
            or first.source_trajectory_id != second.source_trajectory_id
            or first.candidate_type != second.candidate_type
            or first.failure_target != second.failure_target
        ):
            _fail("candidates", "paired sample metadata or labels differ")
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for sample_id, candidate in first_by_id.items():
        grouped[candidate.group_id].append(sample_id)
    all_trajectories = sorted(
        {candidate.source_trajectory_id for candidate in first_candidates}
    )
    differences: defaultdict[str, list[float]] = defaultdict(list)
    for trajectory in all_trajectories:
        differences[trajectory] = []
    for group_id in sorted(grouped):
        sample_ids = grouped[group_id]
        corrupted = [
            sample_id
            for sample_id in sample_ids
            if first_by_id[sample_id].candidate_type == "corrupted"
        ]
        if not corrupted or not any(
            first_by_id[sample_id].failure_target == 0 for sample_id in corrupted
        ):
            continue
        first_top = min(
            corrupted,
            key=lambda sample_id: (
                first_by_id[sample_id].failure_probability,
                sample_id,
            ),
        )
        second_top = min(
            corrupted,
            key=lambda sample_id: (
                second_by_id[sample_id].failure_probability,
                sample_id,
            ),
        )
        first_success = float(first_by_id[first_top].failure_target == 0)
        second_success = float(second_by_id[second_top].failure_target == 0)
        trajectory = first_by_id[corrupted[0]].source_trajectory_id
        differences[trajectory].append(first_success - second_success)
    flattened = [item for values in differences.values() for item in values]
    observed = float(np.mean(flattened)) if flattened else None
    return dict(differences), observed


def group_bootstrap_top1_success_difference(
    first_candidates: Sequence[GroupCandidate],
    second_candidates: Sequence[GroupCandidate],
    *,
    resamples: int = 2000,
    seed: int = 1729,
    confidence_level: float = 0.95,
) -> BootstrapIntervalReport:
    """Compare corrupted-only top-1 success by resampling trajectories."""

    if type(resamples) is not int or resamples <= 0:
        _fail("resamples", "expected a positive integer")
    if type(seed) is not int or seed < 0:
        _fail("seed", "expected a non-negative integer")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        _fail("confidence_level", "expected a finite value in (0, 1)")
    by_trajectory, observed = _paired_group_outcomes(
        first_candidates, second_candidates
    )
    trajectories = tuple(sorted(by_trajectory))
    differences: list[float] = []
    if trajectories:
        generator = np.random.default_rng(seed)
        for _ in range(resamples):
            sampled = generator.choice(
                trajectories, size=len(trajectories), replace=True
            )
            values = [
                difference
                for item in sampled
                for difference in by_trajectory[str(item)]
            ]
            if values:
                differences.append(float(np.mean(values)))
    return _make_report(
        metric="corrupted_only_top1_success",
        observed=observed,
        replicates=differences,
        requested_resamples=resamples,
        confidence_level=confidence_level,
        bootstrap_seed=seed,
        undefined_reason="no eligible corrupted-only candidate groups",
    )


__all__ = [
    "BootstrapIntervalReport",
    "group_bootstrap_top1_success_difference",
    "paired_trajectory_bootstrap",
]
