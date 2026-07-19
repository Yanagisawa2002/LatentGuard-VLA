"""M4D episode, fault-detection, gate-selection, and paired metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from latentguard.control.fallback import (
    FallbackShieldDecisionRecordV1,
    InterventionType,
)
from latentguard.control.models import (
    ClosedLoopEpisodeRecordV1,
    EpisodeState,
    content_digest,
)


@dataclass(frozen=True, slots=True)
class FallbackEpisodeMetricRowV1:
    """One terminal episode projected with finalized M4D decisions."""

    source_trajectory_id: str
    policy_id: str
    scenario: str
    visual_domain: str
    outcome: EpisodeState
    executed_control_steps: int
    decision_count: int
    intervention_count: int
    fallback_count: int
    alternative_count: int
    injected_fault_count: int
    injected_fault_override_count: int
    injected_fault_fallback_count: int
    clean_boundary_count: int
    clean_override_count: int
    fault_exposure_before_first_override: int
    nominal_risks: tuple[float, ...]
    injected_labels: tuple[int, ...]
    execution_seconds: float

    @property
    def success(self) -> float:
        """Return binary success."""

        return float(self.outcome is EpisodeState.SUCCESS)

    @property
    def unsuccessful(self) -> float:
        """Return binary non-success, including horizon exhaustion."""

        return 1.0 - self.success

    @property
    def intervention_rate(self) -> float:
        """Return boundary intervention rate."""

        return (
            0.0
            if self.decision_count == 0
            else self.intervention_count / self.decision_count
        )

    @property
    def fallback_rate(self) -> float:
        """Return fixed-fallback invocation rate."""

        return (
            0.0
            if self.decision_count == 0
            else self.fallback_count / self.decision_count
        )


def fallback_episode_metric_row(
    episode: ClosedLoopEpisodeRecordV1,
    decisions: Sequence[FallbackShieldDecisionRecordV1],
    *,
    execution_seconds: float,
) -> FallbackEpisodeMetricRowV1:
    """Validate base-ledger linkage and project one M4D episode."""

    values = tuple(decisions)
    if not episode.state.terminal or len(values) != len(episode.boundaries):
        raise ValueError("M4D metric episode is incomplete")
    ordered = tuple(sorted(values, key=lambda item: item.decision_ordinal))
    if tuple(item.decision_ordinal for item in ordered) != tuple(range(len(ordered))):
        raise ValueError("M4D decision ordinals are discontinuous")
    if any(
        item.source_trajectory_id != episode.source_trajectory_id for item in ordered
    ):
        raise ValueError("M4D decision source differs from episode")
    for item, ledger in zip(ordered, episode.boundaries, strict=True):
        if (
            item.decision_record_digest != ledger.decision_record_digest
            or item.candidate_pool_digest != ledger.candidate_pool_digest
        ):
            raise ValueError("M4D decision differs from base ledger")
    policies = {item.policy_id for item in ordered}
    scenarios = {item.scenario for item in ordered}
    if len(policies) != 1 or len(scenarios) != 1:
        raise ValueError("M4D episode decision roles changed")
    interventions = [
        item.intervention_type is not InterventionType.ACCEPT_NOMINAL
        for item in ordered
    ]
    injected = [item.injected_fault_reporting_only for item in ordered]
    first_override = next((i for i, value in enumerate(interventions) if value), None)
    exposure = (
        sum(injected[:first_override]) if first_override is not None else sum(injected)
    )
    risks = tuple(
        float(item.nominal_risk) for item in ordered if item.nominal_risk is not None
    )
    labels = tuple(
        int(item.injected_fault_reporting_only)
        for item in ordered
        if item.nominal_risk is not None
    )
    return FallbackEpisodeMetricRowV1(
        source_trajectory_id=episode.source_trajectory_id,
        policy_id=next(iter(policies)),
        scenario=next(iter(scenarios)),
        visual_domain=episode.visual_domain,
        outcome=episode.state,
        executed_control_steps=episode.executed_control_steps,
        decision_count=len(ordered),
        intervention_count=sum(interventions),
        fallback_count=sum(
            item.intervention_type is InterventionType.OVERRIDE_TO_FIXED_FALLBACK
            for item in ordered
        ),
        alternative_count=sum(
            item.intervention_type is InterventionType.OVERRIDE_TO_OTHER_ALTERNATIVE
            for item in ordered
        ),
        injected_fault_count=sum(injected),
        injected_fault_override_count=sum(
            fault and override
            for fault, override in zip(injected, interventions, strict=True)
        ),
        injected_fault_fallback_count=sum(
            fault
            and item.intervention_type is InterventionType.OVERRIDE_TO_FIXED_FALLBACK
            for fault, item in zip(injected, ordered, strict=True)
        ),
        clean_boundary_count=sum(not value for value in injected),
        clean_override_count=sum(
            not fault and override
            for fault, override in zip(injected, interventions, strict=True)
        ),
        fault_exposure_before_first_override=exposure,
        nominal_risks=risks,
        injected_labels=labels,
        execution_seconds=execution_seconds,
    )


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Return tie-aware rank AUROC without optional dependencies."""

    y = np.asarray(tuple(labels), dtype=np.int64)
    x = np.asarray(tuple(scores), dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(tuple(labels), dtype=np.int64)
    x = np.asarray(tuple(scores), dtype=np.float64)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return None
    order = np.argsort(-x, kind="stable")
    ranked = y[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, ranked.size + 1)
    return float(np.sum(precision * ranked) / positives)


def aggregate_fallback_metrics(
    rows: Sequence[FallbackEpisodeMetricRowV1],
) -> dict[str, object]:
    """Aggregate outcome, intervention, fault, and timing evidence."""

    values = tuple(rows)
    if not values:
        raise ValueError("M4D aggregate requires episodes")
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
    decisions = sum(item.decision_count for item in values)
    interventions = sum(item.intervention_count for item in values)
    fallbacks = sum(item.fallback_count for item in values)
    alternatives = sum(item.alternative_count for item in values)
    injected = sum(item.injected_fault_count for item in values)
    injected_overrides = sum(item.injected_fault_override_count for item in values)
    injected_fallbacks = sum(item.injected_fault_fallback_count for item in values)
    clean = sum(item.clean_boundary_count for item in values)
    clean_overrides = sum(item.clean_override_count for item in values)
    labels = tuple(label for item in values for label in item.injected_labels)
    risks = tuple(risk for item in values for risk in item.nominal_risks)
    faulted_episodes = [item for item in values if item.injected_fault_count > 0]
    payload: dict[str, object] = {
        "alternative_invocation_rate": _safe_rate(alternatives, decisions),
        "average_episode_execution_seconds": float(
            np.mean([item.execution_seconds for item in values])
        ),
        "clean_boundary_count": clean,
        "decision_count": decisions,
        "episode_count": len(values),
        "execution_error_rate": outcomes[EpisodeState.EXECUTION_ERROR.value]
        / len(values),
        "false_override_rate": _safe_rate(clean_overrides, clean),
        "fallback_invocation_precision": _safe_rate(injected_fallbacks, fallbacks),
        "fallback_invocation_rate": _safe_rate(fallbacks, decisions),
        "fault_exposure_before_first_override_mean": float(
            np.mean([item.fault_exposure_before_first_override for item in values])
        ),
        "faulted_episodes_recovered_to_success": sum(
            item.outcome is EpisodeState.SUCCESS for item in faulted_episodes
        ),
        "faulted_episodes_unsuccessful": sum(
            item.outcome is not EpisodeState.SUCCESS for item in faulted_episodes
        ),
        "fault_recovery_rate": _safe_rate(
            sum(item.outcome is EpisodeState.SUCCESS for item in faulted_episodes),
            len(faulted_episodes),
        ),
        "horizon_exhaustion_rate": outcomes[EpisodeState.HORIZON_EXHAUSTED.value]
        / len(values),
        "injected_fault_boundary_count": injected,
        "injected_fault_override_recall": _safe_rate(injected_overrides, injected),
        "intervention_precision": _safe_rate(injected_overrides, interventions),
        "intervention_rate": _safe_rate(interventions, decisions),
        "mean_control_steps": float(
            np.mean([item.executed_control_steps for item in values])
        ),
        "mean_decision_count": float(np.mean([item.decision_count for item in values])),
        "nominal_risk_fault_auprc": _average_precision(labels, risks),
        "nominal_risk_fault_auroc": _binary_auc(labels, risks),
        "outcome_counts": outcomes,
        "success_rate": outcomes[EpisodeState.SUCCESS.value] / len(values),
        "task_failure_rate": outcomes[EpisodeState.TASK_FAILURE.value] / len(values),
        "unsafe_rate": outcomes[EpisodeState.UNSAFE.value] / len(values),
    }
    return {
        **payload,
        "content_digest": content_digest(payload, context="M4DAggregateMetricsV1"),
    }


def select_gate_profiles(
    aggregate_metrics: Mapping[str, Mapping[str, object]],
    *,
    gate_profiles_digest: str,
    development_manifest_digest: str,
) -> dict[str, object]:
    """Select one profile per scorer using only frozen development aggregates."""

    scorer_policies = (
        "gated_action_only",
        "gated_direct_visual",
        "gated_distilled_visual",
        "gated_privileged_structured",
    )
    profiles = ("conservative", "balanced", "responsive")
    selected: dict[str, object] = {}
    fallback = aggregate_metrics[
        "always_fixed_primary_fallback/none/clean/not_applicable"
    ]
    fallback_success = float(cast(float, fallback["success_rate"]))
    for policy in scorer_policies:
        candidates: list[dict[str, object]] = []
        for conservatism, profile in zip((2, 1, 0), profiles, strict=True):
            clean_domain = "canonical" if "visual" in policy else "not_applicable"
            clean = aggregate_metrics[f"{policy}/{profile}/clean/{clean_domain}"]
            fault = aggregate_metrics[
                f"{policy}/{profile}/fault_injected/{clean_domain}"
            ]
            clean_counts = cast(Mapping[str, int], clean["outcome_counts"])
            clean_pass = bool(
                float(cast(float, clean["success_rate"])) >= fallback_success - 0.02
                and clean_counts[EpisodeState.UNSAFE.value] == 0
                and clean_counts[EpisodeState.EXECUTION_ERROR.value] == 0
            )
            candidates.append(
                {
                    "clean_constraint_passed": clean_pass,
                    "clean_success_rate": clean["success_rate"],
                    "conservatism_ordinal": conservatism,
                    "fault_success_rate": fault["success_rate"],
                    "fault_unsuccessful_rate": 1.0
                    - float(cast(float, fault["success_rate"])),
                    "fallback_invocation_rate": fault["fallback_invocation_rate"],
                    "intervention_rate": fault["intervention_rate"],
                    "profile_id": profile,
                }
            )
        passing = [item for item in candidates if item["clean_constraint_passed"]]
        if passing:
            winner = min(
                passing,
                key=lambda item: (
                    -float(cast(float, item["fault_success_rate"])),
                    float(cast(float, item["fault_unsuccessful_rate"])),
                    float(cast(float, item["fallback_invocation_rate"])),
                    float(cast(float, item["intervention_rate"])),
                    -int(cast(int, item["conservatism_ordinal"])),
                ),
            )
        else:
            winner = min(
                candidates,
                key=lambda item: (
                    -float(cast(float, item["clean_success_rate"])),
                    -float(cast(float, item["fault_success_rate"])),
                    float(cast(float, item["intervention_rate"])),
                    -int(cast(int, item["conservatism_ordinal"])),
                ),
            )
        selected[policy] = {
            "candidates": candidates,
            "clean_constraint_passed": bool(winner["clean_constraint_passed"]),
            "selected_profile_id": winner["profile_id"],
        }
    payload: dict[str, object] = {
        "development_manifest_digest": development_manifest_digest,
        "gate_profiles_digest": gate_profiles_digest,
        "outcomes_opened": "m4d_development_only",
        "schema_version": "1.0",
        "selected_profiles": selected,
        "selection_semantic": "clean_constraint_then_fault_success_cost_tiebreak_v1",
    }
    return {
        **payload,
        "content_digest": content_digest(payload, context="M4DGateSelectionV1"),
    }


def paired_fallback_bootstrap(
    primary: Sequence[FallbackEpisodeMetricRowV1],
    comparator: Sequence[FallbackEpisodeMetricRowV1],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Bootstrap paired source identities for five predeclared differences."""

    left = {item.source_trajectory_id: item for item in primary}
    right = {item.source_trajectory_id: item for item in comparator}
    if set(left) != set(right) or not left:
        raise ValueError("M4D paired bootstrap source inventories differ")
    ids = tuple(sorted(left))
    metrics = {
        "success_rate_difference": np.asarray(
            [left[key].success - right[key].success for key in ids], dtype=np.float64
        ),
        "unsuccessful_rate_difference": np.asarray(
            [left[key].unsuccessful - right[key].unsuccessful for key in ids],
            dtype=np.float64,
        ),
        "fallback_invocation_rate_difference": np.asarray(
            [left[key].fallback_rate - right[key].fallback_rate for key in ids],
            dtype=np.float64,
        ),
        "total_intervention_rate_difference": np.asarray(
            [left[key].intervention_rate - right[key].intervention_rate for key in ids],
            dtype=np.float64,
        ),
        "mean_control_step_difference": np.asarray(
            [
                left[key].executed_control_steps - right[key].executed_control_steps
                for key in ids
            ],
            dtype=np.float64,
        ),
    }
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(ids), size=(resamples, len(ids)))
    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, object] = {}
    for name, values in metrics.items():
        samples = np.mean(values[indices], axis=1)
        intervals[name] = {
            "confidence_level": confidence_level,
            "estimate": float(np.mean(values)),
            "lower": float(np.quantile(samples, alpha, method="linear")),
            "upper": float(np.quantile(samples, 1.0 - alpha, method="linear")),
        }
    payload: dict[str, object] = {
        "intervals": intervals,
        "resamples": resamples,
        "sampling_unit": "source_trajectory_v1",
        "seed": seed,
        "source_trajectory_count": len(ids),
    }
    return {
        **payload,
        "content_digest": content_digest(payload, context="M4DPairedBootstrapV1"),
    }


def group_fallback_rows(
    rows: Sequence[FallbackEpisodeMetricRowV1],
) -> Mapping[tuple[str, str, str, str], tuple[FallbackEpisodeMetricRowV1, ...]]:
    """Group rows by policy, profile, scenario, and domain."""

    grouped: defaultdict[
        tuple[str, str, str, str], list[FallbackEpisodeMetricRowV1]
    ] = defaultdict(list)
    for row in rows:
        # Policy IDs expanded for development carry a profile suffix.
        parts = row.policy_id.rsplit("__", maxsplit=1)
        if len(parts) == 2 and parts[1] in {"conservative", "balanced", "responsive"}:
            policy, profile = parts
        else:
            policy, profile = row.policy_id, "none"
        grouped[(policy, profile, row.scenario, row.visual_domain)].append(row)
    return {key: tuple(value) for key, value in grouped.items()}


__all__ = [
    "FallbackEpisodeMetricRowV1",
    "aggregate_fallback_metrics",
    "fallback_episode_metric_row",
    "group_fallback_rows",
    "paired_fallback_bootstrap",
    "select_gate_profiles",
]
