"""Outcome-safe selector and complete-pool ranking metrics for M3C."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn, Protocol


class SelectionMetricError(ValueError):
    """Raised when M3C outcome or decision inventories cannot be compared."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionMetricError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class RateV1:
    """One explicit numerator/denominator rate without NaN sentinels."""

    numerator: int
    denominator: int
    value: float | None
    undefined_reason: str | None = None

    def __post_init__(self) -> None:
        """Require counts and the represented value to agree exactly."""

        if (
            type(self.numerator) is not int
            or type(self.denominator) is not int
            or self.numerator < 0
            or self.denominator < 0
            or self.numerator > self.denominator
        ):
            _fail("RateV1", "invalid numerator or denominator")
        expected = None if self.denominator == 0 else self.numerator / self.denominator
        if expected is None:
            if self.value is not None or not self.undefined_reason:
                _fail("RateV1", "zero denominator requires an explicit reason")
        elif (
            self.value is None
            or not math.isfinite(self.value)
            or self.value != expected
            or self.undefined_reason is not None
        ):
            _fail("RateV1", "value does not match its counts")

    @classmethod
    def from_count(cls, numerator: int, denominator: int, reason: str) -> RateV1:
        """Build a defined or explicitly undefined rate from exact counts."""

        return cls(
            numerator=numerator,
            denominator=denominator,
            value=None if denominator == 0 else numerator / denominator,
            undefined_reason=reason if denominator == 0 else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-native count-preserving representation."""

        return {
            "denominator": self.denominator,
            "numerator": self.numerator,
            "undefined_reason": self.undefined_reason,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class CandidateOutcomeV1:
    """One replay result kept distinct from the Stage-A selection decision."""

    proposal_id: str
    group_id: str
    source_trajectory_id: str
    distribution: str
    evidence_id: str
    status: str
    success: bool | None
    unsafe: bool | None
    simulator_replay_verified: bool
    label_strength: str
    state_component_count: int
    execution_error: bool = False

    def __post_init__(self) -> None:
        """Reject fabricated task values and incomplete strong evidence."""

        for name in (
            "proposal_id",
            "group_id",
            "source_trajectory_id",
            "evidence_id",
        ):
            _text(getattr(self, name), f"CandidateOutcomeV1.{name}")
        if self.distribution not in {"id_like", "shifted"}:
            _fail("CandidateOutcomeV1.distribution", "unsupported distribution")
        if type(self.state_component_count) is not int:
            _fail("CandidateOutcomeV1.state_component_count", "expected integer")
        if self.status == "conclusive":
            if (
                type(self.success) is not bool
                or type(self.unsafe) is not bool
                or self.execution_error
                or not self.simulator_replay_verified
                or self.label_strength != "strong"
                or self.state_component_count != 70
            ):
                _fail(
                    "CandidateOutcomeV1",
                    "conclusive M3C outcomes require strong 70-component replay",
                )
        elif self.status == "execution_error":
            if (
                self.success is not None
                or self.unsafe is not None
                or not self.execution_error
                or self.simulator_replay_verified
            ):
                _fail(
                    "CandidateOutcomeV1",
                    "execution errors cannot carry fabricated task outcomes",
                )
        else:
            _fail("CandidateOutcomeV1.status", "unsupported terminal status")


class _Decision(Protocol):
    selector_id: str
    group_id: str
    selected_proposal_id: str | None
    ranking: tuple[str, ...]
    predicted_failure_probabilities: object
    abstained: bool


def _probability_mapping(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = tuple(value)
    else:
        _fail("predicted_failure_probabilities", "expected mapping or pairs")
    result: dict[str, float] = {}
    for item in pairs:
        if not isinstance(item, Sequence) or len(item) != 2:
            _fail("predicted_failure_probabilities", "expected ID/value pairs")
        proposal_id = _text(item[0], "predicted_failure_probabilities.proposal_id")
        raw = item[1]
        if type(raw) not in (int, float):
            _fail("predicted_failure_probabilities", "expected numeric probability")
        probability = float(raw)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            _fail("predicted_failure_probabilities", "probability outside [0, 1]")
        if proposal_id in result:
            _fail("predicted_failure_probabilities", "duplicate proposal")
        result[proposal_id] = probability
    return result


@dataclass(frozen=True, slots=True)
class RankingMetricReportV1:
    """Complete-pool ranking metrics with an eligible count for each estimand."""

    group_count: int
    top1_success: RateV1
    top2_success: RateV1
    pairwise_concordant_credit: float
    pairwise_comparison_count: int
    pairwise_concordance: float | None
    reciprocal_rank_sum: float
    reciprocal_rank_eligible_group_count: int
    mean_reciprocal_rank_first_success: float | None
    ndcg_sum: float
    ndcg_eligible_group_count: int
    mean_binary_ndcg: float | None
    oracle_solvable_group_rate: RateV1
    all_success_group_count: int
    all_failure_group_count: int
    mixed_outcome_group_count: int

    def to_dict(self) -> dict[str, object]:
        """Return the compact ranking report."""

        return {
            "all_failure_group_count": self.all_failure_group_count,
            "all_success_group_count": self.all_success_group_count,
            "group_count": self.group_count,
            "mean_binary_ndcg": self.mean_binary_ndcg,
            "mean_reciprocal_rank_first_success": (
                self.mean_reciprocal_rank_first_success
            ),
            "mixed_outcome_group_count": self.mixed_outcome_group_count,
            "ndcg_eligible_group_count": self.ndcg_eligible_group_count,
            "ndcg_sum": self.ndcg_sum,
            "oracle_solvable_group_rate": self.oracle_solvable_group_rate.to_dict(),
            "pairwise_comparison_count": self.pairwise_comparison_count,
            "pairwise_concordance": self.pairwise_concordance,
            "pairwise_concordant_credit": self.pairwise_concordant_credit,
            "reciprocal_rank_eligible_group_count": (
                self.reciprocal_rank_eligible_group_count
            ),
            "reciprocal_rank_sum": self.reciprocal_rank_sum,
            "top1_success": self.top1_success.to_dict(),
            "top2_success": self.top2_success.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DistributionSelectionReportV1:
    """Outcome rates for selected ID-like or shifted candidates only."""

    distribution: str
    executed_count: int
    conclusive_count: int
    execution_error_count: int
    success_rate: RateV1
    task_failure_rate: RateV1
    unsafe_rate: RateV1

    def __post_init__(self) -> None:
        """Require exact execution and outcome inventories."""

        if self.distribution not in {"id_like", "shifted"}:
            _fail("DistributionSelectionReportV1.distribution", "unsupported value")
        if self.executed_count != self.conclusive_count + self.execution_error_count:
            _fail("DistributionSelectionReportV1", "execution inventory differs")

    def to_dict(self) -> dict[str, object]:
        """Return a compact denominator-preserving slice report."""

        return {
            "conclusive_count": self.conclusive_count,
            "distribution": self.distribution,
            "executed_count": self.executed_count,
            "execution_error_count": self.execution_error_count,
            "success_rate": self.success_rate.to_dict(),
            "task_failure_rate": self.task_failure_rate.to_dict(),
            "unsafe_rate": self.unsafe_rate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SelectorMetricReportV1:
    """Outcome metrics that preserve execution, abstention, and error counts."""

    selector_id: str
    group_count: int
    executed_count: int
    abstained_count: int
    conclusive_count: int
    execution_error_count: int
    coverage: RateV1
    selected_success_rate: RateV1
    task_failure_rate: RateV1
    unsafe_rate: RateV1
    solvable_group_count: int
    executed_solvable_group_count: int
    solvable_success_rate: RateV1
    mixed_group_count: int
    executed_mixed_group_count: int
    mixed_failure_rate: RateV1
    oracle_success_regret: float | None
    mean_selected_oracle_rank: float | None
    mean_selected_predicted_failure_probability: float | None
    selected_id_like_count: int
    selected_shifted_count: int
    distribution_reports: tuple[DistributionSelectionReportV1, ...]
    ranking: RankingMetricReportV1

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-native selector summary with all denominators."""

        return {
            "abstained_count": self.abstained_count,
            "conclusive_count": self.conclusive_count,
            "coverage": self.coverage.to_dict(),
            "executed_count": self.executed_count,
            "executed_mixed_group_count": self.executed_mixed_group_count,
            "executed_solvable_group_count": self.executed_solvable_group_count,
            "execution_error_count": self.execution_error_count,
            "distribution_reports": [
                item.to_dict() for item in self.distribution_reports
            ],
            "group_count": self.group_count,
            "mean_selected_oracle_rank": self.mean_selected_oracle_rank,
            "mean_selected_predicted_failure_probability": (
                self.mean_selected_predicted_failure_probability
            ),
            "mixed_failure_rate": self.mixed_failure_rate.to_dict(),
            "mixed_group_count": self.mixed_group_count,
            "oracle_success_regret": self.oracle_success_regret,
            "ranking": self.ranking.to_dict(),
            "selected_id_like_count": self.selected_id_like_count,
            "selected_shifted_count": self.selected_shifted_count,
            "selected_success_rate": self.selected_success_rate.to_dict(),
            "selector_id": self.selector_id,
            "solvable_group_count": self.solvable_group_count,
            "solvable_success_rate": self.solvable_success_rate.to_dict(),
            "task_failure_rate": self.task_failure_rate.to_dict(),
            "unsafe_rate": self.unsafe_rate.to_dict(),
        }


def _validate_inventory(
    decisions: Sequence[_Decision], outcomes: Sequence[CandidateOutcomeV1]
) -> tuple[
    tuple[_Decision, ...],
    dict[str, dict[str, CandidateOutcomeV1]],
    str,
]:
    choices = tuple(decisions)
    values = tuple(outcomes)
    if not choices:
        _fail("decisions", "at least one group is required")
    selector_ids = {_text(item.selector_id, "decision.selector_id") for item in choices}
    if len(selector_ids) != 1:
        _fail("decisions", "one report covers exactly one selector")
    if len({item.group_id for item in choices}) != len(choices):
        _fail("decisions", "duplicate group decision")
    by_group: defaultdict[str, dict[str, CandidateOutcomeV1]] = defaultdict(dict)
    seen_evidence: set[str] = set()
    for outcome in values:
        if outcome.proposal_id in by_group[outcome.group_id]:
            _fail("outcomes", "duplicate proposal outcome")
        if outcome.evidence_id in seen_evidence:
            _fail("outcomes", "duplicate evidence identity")
        by_group[outcome.group_id][outcome.proposal_id] = outcome
        seen_evidence.add(outcome.evidence_id)
    if set(by_group) != {item.group_id for item in choices}:
        _fail("outcomes", "group inventory differs from decisions")
    for decision in choices:
        members = by_group[decision.group_id]
        if len(members) != 8 or set(decision.ranking) != set(members):
            _fail("outcomes", "every ranking must cover its exact eight outcomes")
        trajectories = {item.source_trajectory_id for item in members.values()}
        if len(trajectories) != 1:
            _fail("outcomes", "one group cannot span source trajectories")
        probabilities = _probability_mapping(decision.predicted_failure_probabilities)
        if probabilities and set(probabilities) != set(members):
            _fail("decisions", "prediction inventory differs from the pool")
        if decision.abstained != (decision.selected_proposal_id is None):
            _fail("decisions", "abstention and selected proposal disagree")
        if (
            not decision.abstained
            and decision.selected_proposal_id != decision.ranking[0]
        ):
            _fail("decisions", "selected proposal must remain ranking top-1")
    return choices, dict(by_group), selector_ids.pop()


def evaluate_ranking_metrics(
    decisions: Sequence[_Decision], outcomes: Sequence[CandidateOutcomeV1]
) -> RankingMetricReportV1:
    """Evaluate complete-pool rankings without using abstention as an outcome."""

    choices, by_group, _ = _validate_inventory(decisions, outcomes)
    top1 = top2 = solvable = all_success = all_failure = mixed = 0
    concordant = 0.0
    comparisons = 0
    reciprocal_sum = 0.0
    reciprocal_eligible = 0
    ndcg_sum = 0.0
    ndcg_eligible = 0
    for decision in choices:
        members = by_group[decision.group_id]
        if any(item.status != "conclusive" for item in members.values()):
            _fail("outcomes", "ranking requires complete conclusive pool outcomes")
        ranked = [members[item] for item in decision.ranking]
        successes = [item for item in members.values() if item.success]
        failures = [item for item in members.values() if item.success is False]
        success_count = len(successes)
        if success_count:
            solvable += 1
            top1 += int(bool(ranked[0].success))
            top2 += int(any(bool(item.success) for item in ranked[:2]))
            first = next(index for index, item in enumerate(ranked, 1) if item.success)
            reciprocal_sum += 1.0 / first
            reciprocal_eligible += 1
            dcg = sum(
                (1.0 / math.log2(index + 1.0))
                for index, item in enumerate(ranked, 1)
                if item.success
            )
            ideal = sum(
                1.0 / math.log2(index + 1.0) for index in range(1, success_count + 1)
            )
            ndcg_sum += dcg / ideal
            ndcg_eligible += 1
        if success_count == 8:
            all_success += 1
        elif success_count == 0:
            all_failure += 1
        else:
            mixed += 1
        rank_index = {
            proposal_id: index for index, proposal_id in enumerate(decision.ranking)
        }
        for success in successes:
            for failure in failures:
                comparisons += 1
                if rank_index[success.proposal_id] < rank_index[failure.proposal_id]:
                    concordant += 1.0
    return RankingMetricReportV1(
        group_count=len(choices),
        top1_success=RateV1.from_count(top1, len(choices), "no groups"),
        top2_success=RateV1.from_count(top2, len(choices), "no groups"),
        pairwise_concordant_credit=concordant,
        pairwise_comparison_count=comparisons,
        pairwise_concordance=None if not comparisons else concordant / comparisons,
        reciprocal_rank_sum=reciprocal_sum,
        reciprocal_rank_eligible_group_count=reciprocal_eligible,
        mean_reciprocal_rank_first_success=(
            None if not reciprocal_eligible else reciprocal_sum / reciprocal_eligible
        ),
        ndcg_sum=ndcg_sum,
        ndcg_eligible_group_count=ndcg_eligible,
        mean_binary_ndcg=None if not ndcg_eligible else ndcg_sum / ndcg_eligible,
        oracle_solvable_group_rate=RateV1.from_count(
            solvable, len(choices), "no groups"
        ),
        all_success_group_count=all_success,
        all_failure_group_count=all_failure,
        mixed_outcome_group_count=mixed,
    )


def evaluate_selector_metrics(
    decisions: Sequence[_Decision], outcomes: Sequence[CandidateOutcomeV1]
) -> SelectorMetricReportV1:
    """Join immutable Stage-A decisions to complete Stage-C outcomes."""

    choices, by_group, selector_id = _validate_inventory(decisions, outcomes)
    ranking = evaluate_ranking_metrics(choices, outcomes)
    executed = abstained = conclusive = errors = 0
    successes = failures = unsafe = 0
    solvable_groups = mixed_groups = 0
    executed_solvable = solvable_successes = 0
    executed_mixed = mixed_failures = 0
    oracle_ranks: list[float] = []
    probabilities: list[float] = []
    id_like = shifted = 0
    distribution_counts = {
        name: {"executed": 0, "conclusive": 0, "errors": 0, "success": 0, "unsafe": 0}
        for name in ("id_like", "shifted")
    }
    for decision in choices:
        members = by_group[decision.group_id]
        successful = [item for item in members.values() if item.success is True]
        failed = [item for item in members.values() if item.success is False]
        solvable = bool(successful)
        mixed = bool(successful and failed)
        solvable_groups += int(solvable)
        mixed_groups += int(mixed)
        if decision.abstained:
            abstained += 1
            continue
        executed += 1
        assert decision.selected_proposal_id is not None
        selected = members[decision.selected_proposal_id]
        selected_counts = distribution_counts[selected.distribution]
        selected_counts["executed"] += 1
        if selected.distribution == "id_like":
            id_like += 1
        else:
            shifted += 1
        predicted = _probability_mapping(decision.predicted_failure_probabilities)
        if predicted:
            probabilities.append(predicted[selected.proposal_id])
        if selected.status == "execution_error":
            errors += 1
            selected_counts["errors"] += 1
            continue
        conclusive += 1
        selected_counts["conclusive"] += 1
        selected_counts["success"] += int(bool(selected.success))
        selected_counts["unsafe"] += int(bool(selected.unsafe))
        successes += int(bool(selected.success))
        failures += int(selected.success is False)
        unsafe += int(bool(selected.unsafe))
        if solvable:
            executed_solvable += 1
            solvable_successes += int(bool(selected.success))
        if mixed:
            executed_mixed += 1
            mixed_failures += int(selected.success is False)
        oracle_order = sorted(
            members.values(),
            key=lambda item: (not bool(item.success), item.proposal_id),
        )
        oracle_ranks.append(float(oracle_order.index(selected) + 1))
    solvable_rate = RateV1.from_count(
        solvable_successes,
        executed_solvable,
        "no executed solvable groups",
    )
    distribution_reports = tuple(
        DistributionSelectionReportV1(
            distribution=name,
            executed_count=values["executed"],
            conclusive_count=values["conclusive"],
            execution_error_count=values["errors"],
            success_rate=RateV1.from_count(
                values["success"],
                values["conclusive"],
                f"no conclusive selected {name} candidates",
            ),
            task_failure_rate=RateV1.from_count(
                values["conclusive"] - values["success"],
                values["conclusive"],
                f"no conclusive selected {name} candidates",
            ),
            unsafe_rate=RateV1.from_count(
                values["unsafe"],
                values["conclusive"],
                f"no conclusive selected {name} candidates",
            ),
        )
        for name, values in sorted(distribution_counts.items())
    )
    return SelectorMetricReportV1(
        selector_id=selector_id,
        group_count=len(choices),
        executed_count=executed,
        abstained_count=abstained,
        conclusive_count=conclusive,
        execution_error_count=errors,
        coverage=RateV1.from_count(executed, len(choices), "no groups"),
        selected_success_rate=RateV1.from_count(
            successes, conclusive, "no conclusive executed groups"
        ),
        task_failure_rate=RateV1.from_count(
            failures, conclusive, "no conclusive executed groups"
        ),
        unsafe_rate=RateV1.from_count(
            unsafe, conclusive, "no conclusive executed groups"
        ),
        solvable_group_count=solvable_groups,
        executed_solvable_group_count=executed_solvable,
        solvable_success_rate=solvable_rate,
        mixed_group_count=mixed_groups,
        executed_mixed_group_count=executed_mixed,
        mixed_failure_rate=RateV1.from_count(
            mixed_failures, executed_mixed, "no executed mixed-outcome groups"
        ),
        oracle_success_regret=(
            None if solvable_rate.value is None else 1.0 - solvable_rate.value
        ),
        mean_selected_oracle_rank=(
            None if not oracle_ranks else sum(oracle_ranks) / len(oracle_ranks)
        ),
        mean_selected_predicted_failure_probability=(
            None if not probabilities else sum(probabilities) / len(probabilities)
        ),
        selected_id_like_count=id_like,
        selected_shifted_count=shifted,
        distribution_reports=distribution_reports,
        ranking=ranking,
    )


@dataclass(frozen=True, slots=True)
class SelectorComparisonV1:
    """Difference and relative-failure comparison against deterministic random."""

    selector_id: str
    random_selector_id: str
    absolute_success_improvement: float | None
    absolute_failure_difference: float | None
    relative_failure_reduction: float | None
    oracle_regret_difference: float | None


def compare_selector_to_random(
    selector: SelectorMetricReportV1,
    random: SelectorMetricReportV1,
) -> SelectorComparisonV1:
    """Compare like-for-like full-coverage selector reports."""

    success = (
        None
        if selector.selected_success_rate.value is None
        or random.selected_success_rate.value is None
        else selector.selected_success_rate.value - random.selected_success_rate.value
    )
    failure = (
        None
        if selector.task_failure_rate.value is None
        or random.task_failure_rate.value is None
        else selector.task_failure_rate.value - random.task_failure_rate.value
    )
    random_failure_rate = random.task_failure_rate.value
    relative = (
        None
        if failure is None or random_failure_rate is None or random_failure_rate == 0.0
        else -failure / random_failure_rate
    )
    regret = (
        None
        if selector.oracle_success_regret is None
        or random.oracle_success_regret is None
        else selector.oracle_success_regret - random.oracle_success_regret
    )
    return SelectorComparisonV1(
        selector_id=selector.selector_id,
        random_selector_id=random.selector_id,
        absolute_success_improvement=success,
        absolute_failure_difference=failure,
        relative_failure_reduction=relative,
        oracle_regret_difference=regret,
    )


__all__ = [
    "CandidateOutcomeV1",
    "DistributionSelectionReportV1",
    "RankingMetricReportV1",
    "RateV1",
    "SelectionMetricError",
    "SelectorComparisonV1",
    "SelectorMetricReportV1",
    "compare_selector_to_random",
    "evaluate_ranking_metrics",
    "evaluate_selector_metrics",
]
