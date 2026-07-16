"""Deterministic NumPy metrics for direct action-verifier evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MetricStatus(StrEnum):
    """Machine-readable status for a metric that may be undefined."""

    DEFINED = "defined"
    EMPTY = "empty"
    SINGLE_CLASS = "single_class"
    ZERO_DENOMINATOR = "zero_denominator"
    NO_ELIGIBLE_GROUPS = "no_eligible_groups"


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One finite metric value or an explicit undefined status."""

    value: float | None
    status: MetricStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        """Reject NaN, infinity, and ambiguous undefined values."""

        if self.status is MetricStatus.DEFINED:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("defined metric values must be finite")
            if self.reason is not None:
                raise ValueError("defined metric values cannot carry a reason")
        elif self.value is not None or not self.reason:
            raise ValueError("undefined metric values require a reason and no value")

    @classmethod
    def defined(cls, value: float) -> MetricValue:
        """Construct a finite defined value."""

        return cls(value=float(value), status=MetricStatus.DEFINED)

    @classmethod
    def undefined(cls, status: MetricStatus, reason: str) -> MetricValue:
        """Construct an explicitly undefined value."""

        if status is MetricStatus.DEFINED:
            raise ValueError("undefined metrics cannot use the defined status")
        return cls(value=None, status=status, reason=reason)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native representation."""

        return {
            "reason": self.reason,
            "status": self.status.value,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Binary confusion counts with task failure as the positive class."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def __post_init__(self) -> None:
        """Require non-negative integer counts."""

        if any(
            type(value) is not int or value < 0
            for value in (
                self.true_negative,
                self.false_positive,
                self.false_negative,
                self.true_positive,
            )
        ):
            raise ValueError("confusion counts must be non-negative integers")

    def to_dict(self) -> dict[str, int]:
        """Return stable JSON-native confusion counts."""

        return {
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "true_positive": self.true_positive,
        }


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One fixed-width reliability bin."""

    lower_bound: float
    upper_bound: float
    upper_inclusive: bool
    count: int
    mean_failure_probability: float | None
    empirical_failure_rate: float | None

    def __post_init__(self) -> None:
        """Validate finite bounds and defined values for non-empty bins."""

        if not (
            math.isfinite(self.lower_bound)
            and math.isfinite(self.upper_bound)
            and 0.0 <= self.lower_bound < self.upper_bound <= 1.0
        ):
            raise ValueError("reliability-bin bounds must lie in [0, 1]")
        if type(self.count) is not int or self.count < 0:
            raise ValueError("reliability-bin count must be non-negative")
        values = (self.mean_failure_probability, self.empirical_failure_rate)
        if self.count == 0:
            if any(value is not None for value in values):
                raise ValueError("empty reliability bins cannot carry values")
        elif any(
            value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError("non-empty reliability bins require finite rates")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native representation."""

        return {
            "count": self.count,
            "empirical_failure_rate": self.empirical_failure_rate,
            "lower_bound": self.lower_bound,
            "mean_failure_probability": self.mean_failure_probability,
            "upper_bound": self.upper_bound,
            "upper_inclusive": self.upper_inclusive,
        }


@dataclass(frozen=True, slots=True)
class BinaryMetricReport:
    """Complete candidate-level binary classification metrics."""

    sample_count: int
    failure_count: int
    success_count: int
    threshold: float
    failure_prevalence: MetricValue
    roc_auc: MetricValue
    failure_auprc: MetricValue
    success_auprc: MetricValue
    negative_log_likelihood: MetricValue
    brier_score: MetricValue
    balanced_accuracy: MetricValue
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    specificity: MetricValue
    matthews_correlation_coefficient: MetricValue
    expected_calibration_error: MetricValue
    confusion_matrix: ConfusionMatrix
    reliability_bins: tuple[ReliabilityBin, ...]
    probability_clip_epsilon: float

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native report."""

        return {
            "balanced_accuracy": self.balanced_accuracy.to_dict(),
            "brier_score": self.brier_score.to_dict(),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "expected_calibration_error": self.expected_calibration_error.to_dict(),
            "f1": self.f1.to_dict(),
            "failure_auprc": self.failure_auprc.to_dict(),
            "failure_count": self.failure_count,
            "failure_prevalence": self.failure_prevalence.to_dict(),
            "matthews_correlation_coefficient": (
                self.matthews_correlation_coefficient.to_dict()
            ),
            "negative_log_likelihood": self.negative_log_likelihood.to_dict(),
            "precision": self.precision.to_dict(),
            "probability_clip_epsilon": self.probability_clip_epsilon,
            "recall": self.recall.to_dict(),
            "reliability_bins": [item.to_dict() for item in self.reliability_bins],
            "roc_auc": self.roc_auc.to_dict(),
            "sample_count": self.sample_count,
            "specificity": self.specificity.to_dict(),
            "success_auprc": self.success_auprc.to_dict(),
            "success_count": self.success_count,
            "threshold": self.threshold,
        }


def _binary_targets(value: ArrayLike) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("failure targets must be rank one")
    if array.dtype.kind not in "biu":
        raise ValueError("failure targets must use a boolean or integer dtype")
    converted = np.asarray(array, dtype=np.int64)
    if not bool(np.all((converted == 0) | (converted == 1))):
        raise ValueError("failure targets must contain only zero and one")
    return converted


def _probabilities(value: ArrayLike, *, expected_count: int) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != expected_count:
        raise ValueError("failure probabilities must match the target vector")
    if array.dtype.kind not in "fiu":
        raise ValueError("failure probabilities must use a numeric dtype")
    converted = np.asarray(array, dtype=np.float64)
    if not bool(np.all(np.isfinite(converted))):
        raise ValueError("failure probabilities must be finite")
    if not bool(np.all((converted >= 0.0) & (converted <= 1.0))):
        raise ValueError("failure probabilities must lie in [0, 1]")
    return converted


def _ranking_undefined(targets: NDArray[np.int64]) -> MetricValue | None:
    if targets.size == 0:
        return MetricValue.undefined(MetricStatus.EMPTY, "no samples")
    if int(np.min(targets)) == int(np.max(targets)):
        return MetricValue.undefined(
            MetricStatus.SINGLE_CLASS,
            "ranking metrics require both failure and success samples",
        )
    return None


def roc_auc_score(targets: ArrayLike, probabilities: ArrayLike) -> MetricValue:
    """Compute tie-aware ROC AUC without external metric dependencies."""

    labels = _binary_targets(targets)
    scores = _probabilities(probabilities, expected_count=labels.size)
    undefined = _ranking_undefined(labels)
    if undefined is not None:
        return undefined
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_count = int(np.sum(labels))
    negative_count = labels.size - positive_count
    rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )
    return MetricValue.defined(auc)


def average_precision_score(
    targets: ArrayLike,
    probabilities: ArrayLike,
) -> MetricValue:
    """Compute tie-invariant average precision for a binary positive class."""

    labels = _binary_targets(targets)
    scores = _probabilities(probabilities, expected_count=labels.size)
    undefined = _ranking_undefined(labels)
    if undefined is not None:
        return undefined
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_count = int(np.sum(sorted_labels))
    cumulative_positive = 0
    cumulative_count = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        cumulative_positive += int(np.sum(sorted_labels[start:end]))
        cumulative_count = end
        recall = cumulative_positive / positive_count
        precision = cumulative_positive / cumulative_count
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return MetricValue.defined(average_precision)


def _safe_ratio(
    numerator: int,
    denominator: int,
    *,
    reason: str,
) -> MetricValue:
    if denominator == 0:
        return MetricValue.undefined(MetricStatus.ZERO_DENOMINATOR, reason)
    return MetricValue.defined(numerator / denominator)


def _reliability(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    bin_count: int,
) -> tuple[MetricValue, tuple[ReliabilityBin, ...]]:
    if type(bin_count) is not int or bin_count <= 0:
        raise ValueError("reliability bin count must be a positive integer")
    bins: list[ReliabilityBin] = []
    weighted_error = 0.0
    if labels.size:
        assignments = np.minimum(
            np.floor(probabilities * bin_count).astype(np.int64), bin_count - 1
        )
    else:
        assignments = np.empty(0, dtype=np.int64)
    for index in range(bin_count):
        selected = assignments == index
        count = int(np.sum(selected))
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if count:
            mean_probability = float(np.mean(probabilities[selected]))
            empirical_rate = float(np.mean(labels[selected]))
            weighted_error += count * abs(mean_probability - empirical_rate)
        else:
            mean_probability = None
            empirical_rate = None
        bins.append(
            ReliabilityBin(
                lower_bound=lower,
                upper_bound=upper,
                upper_inclusive=index == bin_count - 1,
                count=count,
                mean_failure_probability=mean_probability,
                empirical_failure_rate=empirical_rate,
            )
        )
    if labels.size == 0:
        ece = MetricValue.undefined(MetricStatus.EMPTY, "no samples")
    else:
        ece = MetricValue.defined(weighted_error / labels.size)
    return ece, tuple(bins)


def evaluate_binary_metrics(
    targets: ArrayLike,
    failure_probabilities: ArrayLike,
    *,
    threshold: float = 0.5,
    reliability_bin_count: int = 10,
) -> BinaryMetricReport:
    """Evaluate failure-positive candidate metrics with explicit edge cases."""

    labels = _binary_targets(targets)
    probabilities = _probabilities(failure_probabilities, expected_count=labels.size)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("classification threshold must lie in [0, 1]")
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))
    true_negative = int(np.sum(~predictions & negatives))
    confusion = ConfusionMatrix(
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )
    count = labels.size
    failure_count = int(np.sum(labels))
    success_count = count - failure_count
    if count == 0:
        empty = MetricValue.undefined(MetricStatus.EMPTY, "no samples")
        prevalence = nll = brier = empty
    else:
        prevalence = MetricValue.defined(failure_count / count)
        epsilon = float(np.finfo(np.float64).eps)
        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        nll = MetricValue.defined(
            float(
                -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
            )
        )
        brier = MetricValue.defined(float(np.mean((probabilities - labels) ** 2)))
    recall = _safe_ratio(
        true_positive,
        true_positive + false_negative,
        reason="recall requires at least one failure sample",
    )
    specificity = _safe_ratio(
        true_negative,
        true_negative + false_positive,
        reason="specificity requires at least one success sample",
    )
    precision = _safe_ratio(
        true_positive,
        true_positive + false_positive,
        reason="precision requires at least one predicted failure",
    )
    if recall.value is None or specificity.value is None:
        balanced_accuracy = MetricValue.undefined(
            MetricStatus.SINGLE_CLASS,
            "balanced accuracy requires both failure and success samples",
        )
    else:
        balanced_accuracy = MetricValue.defined(
            (recall.value + specificity.value) / 2.0
        )
    if precision.value is None or recall.value is None:
        f1 = MetricValue.undefined(
            MetricStatus.ZERO_DENOMINATOR,
            "F1 requires defined precision and recall",
        )
    elif precision.value + recall.value == 0.0:
        f1 = MetricValue.defined(0.0)
    else:
        f1 = MetricValue.defined(
            2.0 * precision.value * recall.value / (precision.value + recall.value)
        )
    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if mcc_denominator == 0.0:
        mcc = MetricValue.undefined(
            MetricStatus.ZERO_DENOMINATOR,
            "Matthews correlation coefficient denominator is zero",
        )
    else:
        mcc = MetricValue.defined(
            (true_positive * true_negative - false_positive * false_negative)
            / mcc_denominator
        )
    ece, reliability_bins = _reliability(labels, probabilities, reliability_bin_count)
    return BinaryMetricReport(
        sample_count=count,
        failure_count=failure_count,
        success_count=success_count,
        threshold=float(threshold),
        failure_prevalence=prevalence,
        roc_auc=roc_auc_score(labels, probabilities),
        failure_auprc=average_precision_score(labels, probabilities),
        success_auprc=average_precision_score(1 - labels, 1.0 - probabilities),
        negative_log_likelihood=nll,
        brier_score=brier,
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        matthews_correlation_coefficient=mcc,
        expected_calibration_error=ece,
        confusion_matrix=confusion,
        reliability_bins=reliability_bins,
        probability_clip_epsilon=float(np.finfo(np.float64).eps),
    )


@dataclass(frozen=True, slots=True)
class GroupCandidate:
    """Reporting-only candidate record used by group-ranking metrics."""

    sample_id: str
    group_id: str
    source_trajectory_id: str
    candidate_type: str
    failure_target: int
    failure_probability: float

    def __post_init__(self) -> None:
        """Validate one compact group-metric record."""

        for value in (self.sample_id, self.group_id, self.source_trajectory_id):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError("group candidate identifiers must be canonical text")
        if self.candidate_type not in {"source", "corrupted"}:
            raise ValueError("candidate_type must be source or corrupted")
        if type(self.failure_target) is not int or self.failure_target not in {0, 1}:
            raise ValueError("failure_target must be zero or one")
        if not math.isfinite(self.failure_probability) or not (
            0.0 <= self.failure_probability <= 1.0
        ):
            raise ValueError("failure_probability must be finite and normalized")


@dataclass(frozen=True, slots=True)
class GroupRankingReport:
    """Candidate ranking metrics within state-indexed candidate groups."""

    group_count: int
    corrupted_top1_eligible_group_count: int
    corrupted_top1_success: MetricValue
    random_choice_success_expectation: MetricValue
    both_class_group_count: int
    pairwise_comparison_count: int
    pairwise_success_over_failure_concordance: MetricValue
    corrupted_top1_failure_rate: MetricValue
    mean_reciprocal_rank_first_success: MetricValue
    source_margin_group_count: int
    source_margin_pair_count: int
    mean_source_success_margin_over_failed_corruptions: MetricValue

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native group report."""

        return {
            "both_class_group_count": self.both_class_group_count,
            "corrupted_top1_eligible_group_count": (
                self.corrupted_top1_eligible_group_count
            ),
            "corrupted_top1_failure_rate": (self.corrupted_top1_failure_rate.to_dict()),
            "corrupted_top1_success": self.corrupted_top1_success.to_dict(),
            "group_count": self.group_count,
            "mean_reciprocal_rank_first_success": (
                self.mean_reciprocal_rank_first_success.to_dict()
            ),
            "mean_source_success_margin_over_failed_corruptions": (
                self.mean_source_success_margin_over_failed_corruptions.to_dict()
            ),
            "pairwise_comparison_count": self.pairwise_comparison_count,
            "pairwise_success_over_failure_concordance": (
                self.pairwise_success_over_failure_concordance.to_dict()
            ),
            "random_choice_success_expectation": (
                self.random_choice_success_expectation.to_dict()
            ),
            "source_margin_group_count": self.source_margin_group_count,
            "source_margin_pair_count": self.source_margin_pair_count,
        }


def _mean_or_undefined(
    values: Sequence[float],
    *,
    reason: str,
) -> MetricValue:
    if not values:
        return MetricValue.undefined(MetricStatus.NO_ELIGIBLE_GROUPS, reason)
    return MetricValue.defined(float(np.mean(np.asarray(values, dtype=np.float64))))


def evaluate_group_ranking(
    candidates: Sequence[GroupCandidate],
) -> GroupRankingReport:
    """Evaluate deterministic corrupted-only and source-margin group rankings."""

    if len({candidate.sample_id for candidate in candidates}) != len(candidates):
        raise ValueError("group-ranking sample identifiers must be unique")
    grouped: defaultdict[str, list[GroupCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.group_id].append(candidate)
    top1_successes: list[float] = []
    random_expectations: list[float] = []
    pairwise_group_scores: list[float] = []
    both_class_top1_failures: list[float] = []
    reciprocal_ranks: list[float] = []
    source_margins: list[float] = []
    source_margin_groups = 0
    pairwise_count = 0
    for group_id in sorted(grouped):
        members = grouped[group_id]
        if len({item.source_trajectory_id for item in members}) != 1:
            raise ValueError("one candidate group cannot span source trajectories")
        source = [item for item in members if item.candidate_type == "source"]
        corrupted = [item for item in members if item.candidate_type == "corrupted"]
        if len(source) != 1 or not corrupted:
            raise ValueError("each group requires one source and corrupted candidates")
        successful = [item for item in corrupted if item.failure_target == 0]
        failed = [item for item in corrupted if item.failure_target == 1]
        ranked = sorted(
            corrupted,
            key=lambda item: (item.failure_probability, item.sample_id),
        )
        if successful:
            top1_successes.append(float(ranked[0].failure_target == 0))
            random_expectations.append(len(successful) / len(corrupted))
        if successful and failed:
            concordant = 0.0
            group_pairs = 0
            for success in successful:
                for failure in failed:
                    group_pairs += 1
                    if success.failure_probability < failure.failure_probability:
                        concordant += 1.0
                    elif success.failure_probability == failure.failure_probability:
                        concordant += 0.5
            pairwise_count += group_pairs
            pairwise_group_scores.append(concordant / group_pairs)
            both_class_top1_failures.append(float(ranked[0].failure_target == 1))
            first_success_rank = next(
                index
                for index, item in enumerate(ranked, start=1)
                if item.failure_target == 0
            )
            reciprocal_ranks.append(1.0 / first_success_rank)
        if failed:
            source_margin_groups += 1
            source_probability = source[0].failure_probability
            source_margins.extend(
                failure.failure_probability - source_probability for failure in failed
            )
    return GroupRankingReport(
        group_count=len(grouped),
        corrupted_top1_eligible_group_count=len(top1_successes),
        corrupted_top1_success=_mean_or_undefined(
            top1_successes,
            reason="no group contains a successful corrupted candidate",
        ),
        random_choice_success_expectation=_mean_or_undefined(
            random_expectations,
            reason="no group contains a successful corrupted candidate",
        ),
        both_class_group_count=len(pairwise_group_scores),
        pairwise_comparison_count=pairwise_count,
        pairwise_success_over_failure_concordance=_mean_or_undefined(
            pairwise_group_scores,
            reason="no group contains both successful and failed corruptions",
        ),
        corrupted_top1_failure_rate=_mean_or_undefined(
            both_class_top1_failures,
            reason="no group contains both successful and failed corruptions",
        ),
        mean_reciprocal_rank_first_success=_mean_or_undefined(
            reciprocal_ranks,
            reason="no group contains both successful and failed corruptions",
        ),
        source_margin_group_count=source_margin_groups,
        source_margin_pair_count=len(source_margins),
        mean_source_success_margin_over_failed_corruptions=_mean_or_undefined(
            source_margins,
            reason="no failed corrupted candidate is available for source margin",
        ),
    )


@dataclass(frozen=True, slots=True)
class CoverageRiskPoint:
    """Observed selective-execution risk at one requested coverage."""

    requested_coverage: float
    threshold: float
    retained_count: int
    rejected_count: int
    observed_coverage: float
    retained_failure_rate: MetricValue
    retained_success_rate: MetricValue
    rejected_failure_recall: MetricValue

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native point."""

        return {
            "observed_coverage": self.observed_coverage,
            "rejected_count": self.rejected_count,
            "rejected_failure_recall": self.rejected_failure_recall.to_dict(),
            "requested_coverage": self.requested_coverage,
            "retained_count": self.retained_count,
            "retained_failure_rate": self.retained_failure_rate.to_dict(),
            "retained_success_rate": self.retained_success_rate.to_dict(),
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class CoverageRiskReport:
    """Selective execution table with deterministic threshold semantics."""

    sample_count: int
    failure_count: int
    threshold_semantic: str
    area_semantic: str
    area_under_risk_coverage_curve: MetricValue
    points: tuple[CoverageRiskPoint, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native report."""

        return {
            "area_semantic": self.area_semantic,
            "area_under_risk_coverage_curve": (
                self.area_under_risk_coverage_curve.to_dict()
            ),
            "failure_count": self.failure_count,
            "points": [point.to_dict() for point in self.points],
            "sample_count": self.sample_count,
            "threshold_semantic": self.threshold_semantic,
        }


def evaluate_coverage_risk(
    targets: ArrayLike,
    failure_probabilities: ArrayLike,
    *,
    requested_coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> CoverageRiskReport:
    """Evaluate deterministic score-only coverage thresholds and retained risk.

    The threshold is the ``ceil(coverage * n)``-th lowest score. All candidates
    tied at that score are retained, so observed coverage can exceed the request.
    The risk-coverage area is a right-step integral over complete equal-score
    blocks (``tie_block_right_step_mean_v1``), so input ordering cannot affect it.
    """

    labels = _binary_targets(targets)
    probabilities = _probabilities(failure_probabilities, expected_count=labels.size)
    normalized_coverages: list[float] = []
    for value in requested_coverages:
        number = float(value)
        if not math.isfinite(number) or not 0.0 < number <= 1.0:
            raise ValueError("requested coverages must lie in (0, 1]")
        if number in normalized_coverages:
            raise ValueError("requested coverages must be unique")
        normalized_coverages.append(number)
    points: list[CoverageRiskPoint] = []
    failure_count = int(np.sum(labels))
    if labels.size:
        sorted_probabilities = np.sort(probabilities, kind="stable")
        cumulative_count = 0
        cumulative_failures = 0
        weighted_risk = 0.0
        for score in sorted(set(probabilities.tolist())):
            tied = probabilities == score
            tied_count = int(np.sum(tied))
            cumulative_count += tied_count
            cumulative_failures += int(np.sum(labels[tied]))
            weighted_risk += (
                (tied_count / labels.size) * cumulative_failures / cumulative_count
            )
        area = MetricValue.defined(weighted_risk)
    else:
        sorted_probabilities = np.empty(0, dtype=np.float64)
        area = MetricValue.undefined(MetricStatus.EMPTY, "no samples")
    for requested in normalized_coverages:
        if labels.size == 0:
            threshold = 1.0
            retained = np.zeros(0, dtype=np.bool_)
        else:
            rank = max(1, math.ceil(requested * labels.size))
            threshold = float(sorted_probabilities[rank - 1])
            retained = probabilities <= threshold
        retained_count = int(np.sum(retained))
        rejected = ~retained
        rejected_count = int(np.sum(rejected))
        if retained_count:
            retained_failures = int(np.sum(labels[retained]))
            retained_failure_rate = MetricValue.defined(
                retained_failures / retained_count
            )
            retained_success_rate = MetricValue.defined(
                1.0 - retained_failures / retained_count
            )
        else:
            retained_failure_rate = MetricValue.undefined(
                MetricStatus.EMPTY, "no retained candidates"
            )
            retained_success_rate = MetricValue.undefined(
                MetricStatus.EMPTY, "no retained candidates"
            )
        if failure_count:
            rejected_failure_recall = MetricValue.defined(
                int(np.sum(labels[rejected])) / failure_count
            )
        else:
            rejected_failure_recall = MetricValue.undefined(
                MetricStatus.SINGLE_CLASS,
                "failure recall requires at least one failure sample",
            )
        points.append(
            CoverageRiskPoint(
                requested_coverage=requested,
                threshold=threshold,
                retained_count=retained_count,
                rejected_count=rejected_count,
                observed_coverage=(
                    retained_count / labels.size if labels.size else 0.0
                ),
                retained_failure_rate=retained_failure_rate,
                retained_success_rate=retained_success_rate,
                rejected_failure_recall=rejected_failure_recall,
            )
        )
    return CoverageRiskReport(
        sample_count=labels.size,
        failure_count=failure_count,
        threshold_semantic="ceil_rank_retain_all_boundary_ties_v1",
        area_semantic="tie_block_right_step_mean_v1",
        area_under_risk_coverage_curve=area,
        points=tuple(points),
    )


@dataclass(frozen=True, slots=True)
class SliceMetricReport:
    """Binary metrics for one reporting-only metadata slice."""

    dimension: str
    value: str
    metrics: BinaryMetricReport

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native slice report."""

        return {
            "dimension": self.dimension,
            "metrics": self.metrics.to_dict(),
            "value": self.value,
        }


def evaluate_slices(
    targets: ArrayLike,
    failure_probabilities: ArrayLike,
    metadata: Mapping[str, Sequence[str | None]],
    *,
    threshold: float = 0.5,
    reliability_bin_count: int = 10,
) -> tuple[SliceMetricReport, ...]:
    """Evaluate reporting-only slices without fitting slice-specific thresholds."""

    labels = _binary_targets(targets)
    probabilities = _probabilities(failure_probabilities, expected_count=labels.size)
    reports: list[SliceMetricReport] = []
    for dimension in sorted(metadata):
        if not dimension or dimension != dimension.strip():
            raise ValueError("slice dimensions must be canonical text")
        raw_values = metadata[dimension]
        if len(raw_values) != labels.size:
            raise ValueError(f"slice dimension {dimension!r} has the wrong length")
        values = np.asarray(
            ["<not_applicable>" if value is None else value for value in raw_values],
            dtype=np.str_,
        )
        if any(not value or value != value.strip() for value in values.tolist()):
            raise ValueError("slice values must be canonical text")
        for value in sorted(set(values.tolist())):
            selected = values == value
            reports.append(
                SliceMetricReport(
                    dimension=dimension,
                    value=value,
                    metrics=evaluate_binary_metrics(
                        labels[selected],
                        probabilities[selected],
                        threshold=threshold,
                        reliability_bin_count=reliability_bin_count,
                    ),
                )
            )
    return tuple(reports)


__all__ = [
    "BinaryMetricReport",
    "ConfusionMatrix",
    "CoverageRiskPoint",
    "CoverageRiskReport",
    "GroupCandidate",
    "GroupRankingReport",
    "MetricStatus",
    "MetricValue",
    "ReliabilityBin",
    "SliceMetricReport",
    "average_precision_score",
    "evaluate_binary_metrics",
    "evaluate_coverage_risk",
    "evaluate_group_ranking",
    "evaluate_slices",
    "roc_auc_score",
]
