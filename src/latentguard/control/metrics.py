"""Episode-level metrics, interventions, and trajectory bootstrap for M4C."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from latentguard.control.config import BootstrapConfigurationV1
from latentguard.control.models import (
    ClosedLoopDecisionRecordV1,
    ClosedLoopEpisodeRecordV1,
    EpisodeState,
    content_digest,
)


@dataclass(frozen=True, slots=True)
class EpisodeMetricRowV1:
    """Compact per-episode values used by aggregate M4C evaluation."""

    source_trajectory_id: str
    selector_id: str
    visual_domain: str
    outcome: EpisodeState
    executed_control_steps: int
    decision_count: int
    intervention_count: int
    decisions_before_first_intervention: int | None
    selected_probability_sum: float
    selected_probability_count: int
    risk_margin_sum: float
    risk_margin_count: int
    tail_fallback_count: int
    recovery_count: int
    execution_seconds: float

    def __post_init__(self) -> None:
        """Validate finite non-negative report values."""

        for name in ("source_trajectory_id", "selector_id", "visual_domain"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"EpisodeMetricRowV1.{name}: invalid text")
        if not isinstance(self.outcome, EpisodeState) or not self.outcome.terminal:
            raise ValueError("EpisodeMetricRowV1.outcome must be terminal")
        for name in (
            "executed_control_steps",
            "decision_count",
            "intervention_count",
            "selected_probability_count",
            "risk_margin_count",
            "tail_fallback_count",
            "recovery_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"EpisodeMetricRowV1.{name}: invalid count")
        if self.decisions_before_first_intervention is not None:
            if (
                type(self.decisions_before_first_intervention) is not int
                or self.decisions_before_first_intervention < 0
                or self.decisions_before_first_intervention >= self.decision_count
            ):
                raise ValueError(
                    "EpisodeMetricRowV1.decisions_before_first_intervention: "
                    "invalid count"
                )
        if (self.intervention_count == 0) != (
            self.decisions_before_first_intervention is None
        ):
            raise ValueError(
                "EpisodeMetricRowV1 first-intervention count disagrees with "
                "intervention count"
            )
        for name in (
            "selected_probability_sum",
            "risk_margin_sum",
            "execution_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"EpisodeMetricRowV1.{name}: invalid value")
        if self.execution_seconds < 0.0:
            raise ValueError("EpisodeMetricRowV1.execution_seconds: negative")

    @property
    def success(self) -> float:
        """Return the binary episode-success indicator."""

        return float(self.outcome is EpisodeState.SUCCESS)

    @property
    def unsuccessful(self) -> float:
        """Return unsuccessful episode indicator, including horizon exhaustion."""

        return float(self.outcome is not EpisodeState.SUCCESS)


def episode_metric_row(
    episode: ClosedLoopEpisodeRecordV1,
    decisions: Sequence[ClosedLoopDecisionRecordV1],
    *,
    primary_candidate_ordinal: int,
    execution_seconds: float,
) -> EpisodeMetricRowV1:
    """Project one strict episode and its immutable selections into metrics."""

    values = tuple(decisions)
    if len(values) != len(episode.boundaries):
        raise ValueError("episode decisions do not match boundary count")
    interventions = 0
    first_intervention: int | None = None
    probability_sum = 0.0
    probability_count = 0
    margin_sum = 0.0
    margin_count = 0
    for decision in values:
        if (
            decision.content_digest
            != episode.boundaries[decision.decision_ordinal].decision_record_digest
        ):
            raise ValueError("decision digest differs from episode ledger")
        primary_id = decision.ordered_candidate_ids[primary_candidate_ordinal]
        intervened = decision.selected_candidate_id != primary_id
        interventions += int(intervened)
        if intervened and first_intervention is None:
            first_intervention = decision.decision_ordinal
        if decision.selected_predicted_failure_probability is not None:
            probability_sum += decision.selected_predicted_failure_probability
            probability_count += 1
            scores = dict(decision.selector_scores)
            margin_sum += scores[primary_id] - scores[decision.selected_candidate_id]
            margin_count += 1
    return EpisodeMetricRowV1(
        source_trajectory_id=episode.source_trajectory_id,
        selector_id=episode.selector_id,
        visual_domain=episode.visual_domain,
        outcome=episode.state,
        executed_control_steps=episode.executed_control_steps,
        decision_count=len(episode.boundaries),
        intervention_count=interventions,
        decisions_before_first_intervention=first_intervention,
        selected_probability_sum=probability_sum,
        selected_probability_count=probability_count,
        risk_margin_sum=margin_sum,
        risk_margin_count=margin_count,
        tail_fallback_count=episode.tail_fallback_count,
        recovery_count=episode.recovery_count,
        execution_seconds=execution_seconds,
    )


def _distribution(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError("metric distribution requires finite values")
    return {
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
    }


def aggregate_episode_metrics(rows: Sequence[EpisodeMetricRowV1]) -> dict[str, object]:
    """Aggregate distinct outcome rates and required operational distributions."""

    values = tuple(rows)
    if not values:
        raise ValueError("aggregate metrics require episodes")
    total = len(values)
    outcomes = {
        state.value: sum(item.outcome is state for item in values)
        for state in (
            EpisodeState.SUCCESS,
            EpisodeState.TASK_FAILURE,
            EpisodeState.UNSAFE,
            EpisodeState.HORIZON_EXHAUSTED,
            EpisodeState.EXECUTION_ERROR,
        )
    }
    probability_count = sum(item.selected_probability_count for item in values)
    margin_count = sum(item.risk_margin_count for item in values)
    interventions = sum(item.intervention_count for item in values)
    decisions = sum(item.decision_count for item in values)
    payload: dict[str, object] = {
        "average_episode_execution_seconds": float(
            np.mean([item.execution_seconds for item in values])
        ),
        "decision_count": _distribution(
            [float(item.decision_count) for item in values]
        ),
        "episode_count": total,
        "executed_control_steps": _distribution(
            [float(item.executed_control_steps) for item in values]
        ),
        "execution_error_rate": outcomes[EpisodeState.EXECUTION_ERROR.value] / total,
        "horizon_exhaustion_rate": outcomes[EpisodeState.HORIZON_EXHAUSTED.value]
        / total,
        "mean_risk_margin_selected_vs_primary": (
            None
            if margin_count == 0
            else sum(item.risk_margin_sum for item in values) / margin_count
        ),
        "mean_selected_predicted_failure_probability": (
            None
            if probability_count == 0
            else sum(item.selected_probability_sum for item in values)
            / probability_count
        ),
        "non_primary_candidate_intervention_rate": (
            0.0 if decisions == 0 else interventions / decisions
        ),
        "outcome_counts": outcomes,
        "recovery_count": sum(item.recovery_count for item in values),
        "schema_version": "1.0",
        "success_rate": outcomes[EpisodeState.SUCCESS.value] / total,
        "tail_fallback_count": sum(item.tail_fallback_count for item in values),
        "task_failure_rate": outcomes[EpisodeState.TASK_FAILURE.value] / total,
        "unsafe_rate": outcomes[EpisodeState.UNSAFE.value] / total,
    }
    return {
        **payload,
        "content_digest": content_digest(payload, context="M4CAggregateMetricsV1"),
    }


def paired_trajectory_bootstrap(
    primary: Sequence[EpisodeMetricRowV1],
    comparator: Sequence[EpisodeMetricRowV1],
    *,
    configuration: BootstrapConfigurationV1,
) -> dict[str, object]:
    """Bootstrap paired source-trajectory episode differences, never decisions."""

    left = {item.source_trajectory_id: item for item in primary}
    right = {item.source_trajectory_id: item for item in comparator}
    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap requires identical source trajectories")
    ids = tuple(sorted(left))
    metrics = {
        "episode_success_rate_difference": np.asarray(
            [left[key].success - right[key].success for key in ids], dtype=np.float64
        ),
        "unsuccessful_episode_rate_difference": np.asarray(
            [left[key].unsuccessful - right[key].unsuccessful for key in ids],
            dtype=np.float64,
        ),
        "mean_control_step_difference": np.asarray(
            [
                left[key].executed_control_steps - right[key].executed_control_steps
                for key in ids
            ],
            dtype=np.float64,
        ),
        "mean_decision_count_difference": np.asarray(
            [left[key].decision_count - right[key].decision_count for key in ids],
            dtype=np.float64,
        ),
    }
    generator = np.random.default_rng(configuration.seed)
    sampled = generator.integers(
        0, len(ids), size=(configuration.resamples, len(ids)), endpoint=False
    )
    intervals: dict[str, object] = {}
    for name, values in metrics.items():
        estimates = np.mean(values[sampled], axis=1)
        intervals[name] = {
            "confidence_lower": float(np.quantile(estimates, 0.025, method="linear")),
            "confidence_upper": float(np.quantile(estimates, 0.975, method="linear")),
            "estimate": float(np.mean(values)),
        }
    payload: dict[str, object] = {
        "bootstrap_configuration_digest": configuration.content_digest,
        "intervals": intervals,
        "paired_source_trajectory_count": len(ids),
        "resampling_unit": configuration.sampling_unit,
        "schema_version": "1.0",
    }
    return {
        **payload,
        "content_digest": content_digest(payload, context="M4CPairedBootstrapV1"),
    }


def intervention_diagnostics(
    rows: Sequence[EpisodeMetricRowV1],
) -> dict[str, object]:
    """Report intervention-conditioned outcomes without making causal claims."""

    values = tuple(rows)
    if not values:
        raise ValueError("intervention diagnostics require episodes")
    intervened = tuple(item for item in values if item.intervention_count > 0)
    untouched = tuple(item for item in values if item.intervention_count == 0)

    def success_rate(items: Sequence[EpisodeMetricRowV1]) -> float | None:
        return None if not items else sum(item.success for item in items) / len(items)

    payload: dict[str, object] = {
        "causal_interpretation_permitted": False,
        "episode_count": len(values),
        "episode_intervention_rate": len(intervened) / len(values),
        "mean_repeated_interventions_per_episode": float(
            np.mean([item.intervention_count for item in values])
        ),
        "decisions_before_first_intervention": (
            None
            if not intervened
            else _distribution(
                [
                    float(item.decisions_before_first_intervention)
                    for item in intervened
                    if item.decisions_before_first_intervention is not None
                ]
            )
        ),
        "schema_version": "1.0",
        "success_conditional_on_intervention": success_rate(intervened),
        "success_conditional_on_no_intervention": success_rate(untouched),
    }
    return {
        **payload,
        "content_digest": content_digest(
            payload, context="M4CInterventionDiagnosticsV1"
        ),
    }


def content_identical_decision_agreement(
    decisions: Sequence[ClosedLoopDecisionRecordV1],
) -> dict[str, object]:
    """Compare rankings only where selectors reached the same physical state."""

    grouped: dict[tuple[str, int, str], list[ClosedLoopDecisionRecordV1]] = defaultdict(
        list
    )
    for item in decisions:
        grouped[
            (
                item.source_trajectory_id,
                item.nominal_plan_index,
                item.pre_decision_state_digest,
            )
        ].append(item)
    eligible = tuple(group for group in grouped.values() if len(group) > 1)
    pair_count = 0
    same_top = 0
    same_ranking = 0
    pairwise: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for group in eligible:
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                left_participant = f"{left.selector_id}/{left.visual_domain}"
                right_participant = f"{right.selector_id}/{right.visual_domain}"
                if left_participant == right_participant:
                    continue
                pair_key = (
                    (left_participant, right_participant)
                    if left_participant < right_participant
                    else (right_participant, left_participant)
                )
                pair_count += 1
                top_match = int(
                    left.selected_candidate_id == right.selected_candidate_id
                )
                ranking_match = int(
                    left.deterministic_ranking == right.deterministic_ranking
                )
                same_top += top_match
                same_ranking += ranking_match
                pairwise[pair_key][0] += 1
                pairwise[pair_key][1] += top_match
                pairwise[pair_key][2] += ranking_match
    pairwise_payload = {}
    for (pair_left, pair_right), (
        count,
        pair_top,
        pair_ranking,
    ) in sorted(pairwise.items()):
        pairwise_payload[f"{pair_left}_vs_{pair_right}"] = {
            "content_identical_boundary_count": count,
            "full_ranking_agreement_rate": pair_ranking / count,
            "top_selection_agreement_rate": pair_top / count,
            "top_selection_disagreement_rate": 1.0 - (pair_top / count),
        }
    payload: dict[str, object] = {
        "content_identical_boundary_group_count": len(eligible),
        "cross_selector_pair_count": pair_count,
        "full_ranking_agreement_rate": None
        if pair_count == 0
        else same_ranking / pair_count,
        "join_keys": [
            "source_trajectory_id",
            "nominal_plan_index",
            "pre_decision_state_digest",
        ],
        "ordinal_only_join_prohibited": True,
        "pairwise_selector_agreement": pairwise_payload,
        "schema_version": "1.0",
        "top_selection_agreement_rate": None
        if pair_count == 0
        else same_top / pair_count,
    }
    return {
        **payload,
        "content_digest": content_digest(payload, context="M4CSelectionConsistencyV1"),
    }


__all__ = [
    "EpisodeMetricRowV1",
    "aggregate_episode_metrics",
    "content_identical_decision_agreement",
    "episode_metric_row",
    "intervention_diagnostics",
    "paired_trajectory_bootstrap",
]
