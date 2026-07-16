"""Deterministic paired trajectory bootstrap for M3C selector comparisons."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import numpy as np

from latentguard.selection.metrics import (
    CandidateOutcomeV1,
    evaluate_selector_metrics,
)
from latentguard.selection.models import SelectorDecisionV1

BOOTSTRAP_METRICS = (
    "selected_success_rate",
    "task_failure_rate",
    "oracle_success_regret",
)
BOOTSTRAP_SAMPLING_UNIT = "original_m3c_source_trajectory_v1"
BOOTSTRAP_INTERVAL_METHOD = "trajectory_percentile_linear_v1"


class SelectionBootstrapError(ValueError):
    """Raised when selectors cannot be compared by paired trajectory resampling."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionBootstrapError(f"{context}: {reason}")


@dataclass(frozen=True, slots=True)
class BootstrapMetricIntervalV1:
    """One selector-minus-selector trajectory-bootstrap interval."""

    metric: str
    observed_difference: float | None
    confidence_lower: float | None
    confidence_upper: float | None
    valid_resamples: int
    skipped_resamples: int
    undefined_reason: str | None = None
    difference_semantic: str = "first_selector_minus_second_selector"
    interval_method: str = BOOTSTRAP_INTERVAL_METHOD

    def __post_init__(self) -> None:
        """Reject incomplete bounds, invalid counts, and unsupported metrics."""

        if self.metric not in BOOTSTRAP_METRICS:
            _fail("BootstrapMetricIntervalV1.metric", "unsupported metric")
        if (
            type(self.valid_resamples) is not int
            or type(self.skipped_resamples) is not int
            or self.valid_resamples < 0
            or self.skipped_resamples < 0
        ):
            _fail("BootstrapMetricIntervalV1.resamples", "invalid counts")
        if self.observed_difference is None:
            if not self.undefined_reason:
                _fail(
                    "BootstrapMetricIntervalV1.undefined_reason",
                    "undefined estimate requires an explicit reason",
                )
        elif (
            not math.isfinite(self.observed_difference)
            or self.undefined_reason is not None
        ):
            _fail("BootstrapMetricIntervalV1.observed_difference", "invalid estimate")
        if self.valid_resamples:
            if (
                self.confidence_lower is None
                or self.confidence_upper is None
                or not math.isfinite(self.confidence_lower)
                or not math.isfinite(self.confidence_upper)
                or self.confidence_lower > self.confidence_upper
            ):
                _fail("BootstrapMetricIntervalV1.confidence", "invalid interval")
        elif self.confidence_lower is not None or self.confidence_upper is not None:
            _fail(
                "BootstrapMetricIntervalV1.confidence",
                "no valid replicate has no bounds",
            )
        if self.difference_semantic != "first_selector_minus_second_selector":
            _fail("BootstrapMetricIntervalV1.difference_semantic", "semantic changed")
        if self.interval_method != BOOTSTRAP_INTERVAL_METHOD:
            _fail("BootstrapMetricIntervalV1.interval_method", "method changed")

    def to_dict(self) -> dict[str, object]:
        """Return the compact interval with explicit valid/skip counts."""

        return {
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "difference_semantic": self.difference_semantic,
            "interval_method": self.interval_method,
            "metric": self.metric,
            "observed_difference": self.observed_difference,
            "skipped_resamples": self.skipped_resamples,
            "undefined_reason": self.undefined_reason,
            "valid_resamples": self.valid_resamples,
        }


@dataclass(frozen=True, slots=True)
class PairedTrajectoryBootstrapReportV1:
    """Three required paired intervals using original trajectories as units."""

    first_selector_id: str
    second_selector_id: str
    trajectory_count: int
    requested_resamples: int
    bootstrap_seed: int
    confidence_level: float
    intervals: tuple[BootstrapMetricIntervalV1, ...]
    sampling_unit: str = BOOTSTRAP_SAMPLING_UNIT
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Require the fixed M3C metric inventory and at least 2,000 replicates."""

        for name in ("first_selector_id", "second_selector_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                _fail(f"PairedTrajectoryBootstrapReportV1.{name}", "invalid text")
        if self.first_selector_id == self.second_selector_id:
            _fail("PairedTrajectoryBootstrapReportV1", "selectors must differ")
        if type(self.trajectory_count) is not int or self.trajectory_count <= 0:
            _fail("PairedTrajectoryBootstrapReportV1.trajectory_count", "invalid count")
        if type(self.requested_resamples) is not int or self.requested_resamples < 2000:
            _fail(
                "PairedTrajectoryBootstrapReportV1.requested_resamples",
                "M3C requires at least 2,000 resamples",
            )
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**64:
            _fail("PairedTrajectoryBootstrapReportV1.bootstrap_seed", "expected uint64")
        if (
            type(self.confidence_level) not in (int, float)
            or not math.isfinite(float(self.confidence_level))
            or not 0.0 < float(self.confidence_level) < 1.0
        ):
            _fail("PairedTrajectoryBootstrapReportV1.confidence_level", "invalid level")
        intervals = tuple(self.intervals)
        if tuple(item.metric for item in intervals) != BOOTSTRAP_METRICS:
            _fail(
                "PairedTrajectoryBootstrapReportV1.intervals",
                "metric inventory changed",
            )
        if any(
            item.valid_resamples + item.skipped_resamples != self.requested_resamples
            for item in intervals
        ):
            _fail(
                "PairedTrajectoryBootstrapReportV1.intervals", "replicate counts differ"
            )
        if self.sampling_unit != BOOTSTRAP_SAMPLING_UNIT:
            _fail("PairedTrajectoryBootstrapReportV1.sampling_unit", "unit changed")
        if self.schema_version != "1.0":
            _fail(
                "PairedTrajectoryBootstrapReportV1.schema_version",
                "unsupported version",
            )
        object.__setattr__(self, "confidence_level", float(self.confidence_level))
        object.__setattr__(self, "intervals", intervals)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-native bootstrap summary."""

        return {
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "first_selector_id": self.first_selector_id,
            "intervals": [item.to_dict() for item in self.intervals],
            "requested_resamples": self.requested_resamples,
            "sampling_unit": self.sampling_unit,
            "schema_version": self.schema_version,
            "second_selector_id": self.second_selector_id,
            "trajectory_count": self.trajectory_count,
        }


@dataclass(frozen=True, slots=True)
class _MetricCounts:
    selected_success_numerator: int = 0
    selected_outcome_denominator: int = 0
    task_failure_numerator: int = 0
    oracle_regret_numerator: int = 0
    executed_solvable_denominator: int = 0

    def __add__(self, other: _MetricCounts) -> _MetricCounts:
        return _MetricCounts(
            selected_success_numerator=(
                self.selected_success_numerator + other.selected_success_numerator
            ),
            selected_outcome_denominator=(
                self.selected_outcome_denominator + other.selected_outcome_denominator
            ),
            task_failure_numerator=(
                self.task_failure_numerator + other.task_failure_numerator
            ),
            oracle_regret_numerator=(
                self.oracle_regret_numerator + other.oracle_regret_numerator
            ),
            executed_solvable_denominator=(
                self.executed_solvable_denominator + other.executed_solvable_denominator
            ),
        )

    def value(self, metric: str) -> float | None:
        if metric == "selected_success_rate":
            numerator = self.selected_success_numerator
            denominator = self.selected_outcome_denominator
        elif metric == "task_failure_rate":
            numerator = self.task_failure_numerator
            denominator = self.selected_outcome_denominator
        elif metric == "oracle_success_regret":
            numerator = self.oracle_regret_numerator
            denominator = self.executed_solvable_denominator
        else:
            _fail("metric", "unsupported metric")
        return None if denominator == 0 else numerator / denominator


def _selector_counts_by_trajectory(
    decisions: tuple[SelectorDecisionV1, ...],
    outcomes_by_group: dict[str, dict[str, CandidateOutcomeV1]],
) -> dict[str, _MetricCounts]:
    result: defaultdict[str, _MetricCounts] = defaultdict(_MetricCounts)
    for decision in decisions:
        members = outcomes_by_group[decision.group_id]
        trajectory_ids = {item.source_trajectory_id for item in members.values()}
        if len(trajectory_ids) != 1:
            _fail("outcomes", "one group spans multiple source trajectories")
        trajectory_id = trajectory_ids.pop()
        if decision.abstained:
            continue
        if decision.selected_proposal_id is None:
            _fail("decisions", "non-abstained decision has no selected proposal")
        selected = members[decision.selected_proposal_id]
        if selected.status != "conclusive":
            _fail("outcomes", "bootstrap requires conclusive selected outcomes")
        solvable = any(item.success is True for item in members.values())
        increment = _MetricCounts(
            selected_success_numerator=int(bool(selected.success)),
            selected_outcome_denominator=1,
            task_failure_numerator=int(selected.success is False),
            oracle_regret_numerator=int(solvable and selected.success is False),
            executed_solvable_denominator=int(solvable),
        )
        result[trajectory_id] = result[trajectory_id] + increment
    return dict(result)


def _sum_counts(values: Sequence[_MetricCounts]) -> _MetricCounts:
    result = _MetricCounts()
    for value in values:
        result = result + value
    return result


def paired_trajectory_selector_bootstrap(
    first_decisions: Sequence[SelectorDecisionV1],
    second_decisions: Sequence[SelectorDecisionV1],
    complete_outcomes: Sequence[CandidateOutcomeV1],
    *,
    resamples: int = 2000,
    seed: int = 1729,
    confidence_level: float = 0.95,
) -> PairedTrajectoryBootstrapReportV1:
    """Compare two selectors by resampling whole original M3C trajectories."""

    first = tuple(first_decisions)
    second = tuple(second_decisions)
    outcomes = tuple(complete_outcomes)
    if not first or not second:
        _fail("decisions", "both selector inventories must be non-empty")
    if any(not isinstance(item, SelectorDecisionV1) for item in (*first, *second)):
        _fail("decisions", "expected SelectorDecisionV1 inventories")
    if any(not isinstance(item, CandidateOutcomeV1) for item in outcomes):
        _fail("complete_outcomes", "expected CandidateOutcomeV1 inventory")
    first_report = evaluate_selector_metrics(cast(Sequence[Any], first), outcomes)
    second_report = evaluate_selector_metrics(cast(Sequence[Any], second), outcomes)
    if first_report.selector_id == second_report.selector_id:
        _fail("decisions", "selectors must have distinct identities")
    first_groups = {item.group_id for item in first}
    second_groups = {item.group_id for item in second}
    if first_groups != second_groups:
        _fail("decisions", "paired selectors must cover identical groups")
    if type(resamples) is not int or resamples < 2000:
        _fail("resamples", "M3C requires at least 2,000 resamples")
    if type(seed) is not int or not 0 <= seed < 2**64:
        _fail("seed", "expected uint64")
    if (
        type(confidence_level) not in (int, float)
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        _fail("confidence_level", "expected a value in (0, 1)")
    outcomes_by_group: defaultdict[str, dict[str, CandidateOutcomeV1]] = defaultdict(
        dict
    )
    for outcome in outcomes:
        outcomes_by_group[outcome.group_id][outcome.proposal_id] = outcome
    first_counts = _selector_counts_by_trajectory(first, dict(outcomes_by_group))
    second_counts = _selector_counts_by_trajectory(second, dict(outcomes_by_group))
    trajectory_inventory = tuple(
        sorted(
            {
                item.source_trajectory_id
                for item in outcomes
                if item.group_id in first_groups
            }
        )
    )
    if not trajectory_inventory:
        _fail("complete_outcomes", "no source trajectories")
    zero = _MetricCounts()
    first_ordered = tuple(first_counts.get(item, zero) for item in trajectory_inventory)
    second_ordered = tuple(
        second_counts.get(item, zero) for item in trajectory_inventory
    )
    observed_first = _sum_counts(first_ordered)
    observed_second = _sum_counts(second_ordered)
    replicates: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    generator = np.random.default_rng(seed)
    for _ in range(resamples):
        sampled_indices = generator.integers(
            0,
            len(trajectory_inventory),
            size=len(trajectory_inventory),
            endpoint=False,
        )
        sampled_first = _sum_counts(
            [first_ordered[int(item)] for item in sampled_indices]
        )
        sampled_second = _sum_counts(
            [second_ordered[int(item)] for item in sampled_indices]
        )
        for metric in BOOTSTRAP_METRICS:
            first_value = sampled_first.value(metric)
            second_value = sampled_second.value(metric)
            if first_value is not None and second_value is not None:
                replicates[metric].append(first_value - second_value)

    alpha = (1.0 - float(confidence_level)) / 2.0
    intervals: list[BootstrapMetricIntervalV1] = []
    for metric in BOOTSTRAP_METRICS:
        first_value = observed_first.value(metric)
        second_value = observed_second.value(metric)
        observed = (
            None
            if first_value is None or second_value is None
            else first_value - second_value
        )
        values = np.asarray(replicates[metric], dtype=np.float64)
        intervals.append(
            BootstrapMetricIntervalV1(
                metric=metric,
                observed_difference=observed,
                confidence_lower=(
                    None
                    if values.size == 0
                    else float(np.quantile(values, alpha, method="linear"))
                ),
                confidence_upper=(
                    None
                    if values.size == 0
                    else float(np.quantile(values, 1.0 - alpha, method="linear"))
                ),
                valid_resamples=int(values.size),
                skipped_resamples=resamples - int(values.size),
                undefined_reason=(
                    "one or both selectors have zero eligible outcomes"
                    if observed is None
                    else None
                ),
            )
        )
    return PairedTrajectoryBootstrapReportV1(
        first_selector_id=first_report.selector_id,
        second_selector_id=second_report.selector_id,
        trajectory_count=len(trajectory_inventory),
        requested_resamples=resamples,
        bootstrap_seed=seed,
        confidence_level=float(confidence_level),
        intervals=tuple(intervals),
    )


__all__ = [
    "BOOTSTRAP_INTERVAL_METHOD",
    "BOOTSTRAP_METRICS",
    "BOOTSTRAP_SAMPLING_UNIT",
    "BootstrapMetricIntervalV1",
    "PairedTrajectoryBootstrapReportV1",
    "SelectionBootstrapError",
    "paired_trajectory_selector_bootstrap",
]
