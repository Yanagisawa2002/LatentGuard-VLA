"""Fail-closed distribution and promotion gates for LG-R2b0 evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CounterfactualGateEvidence:
    """Scalar evidence required by the pre-registered LG-R2b1 gate."""

    restore_mismatches: int
    branch_contamination: int
    candidate_metadata_completeness: float
    identity_completeness: float
    policy_generated_candidate_ratio: float
    synthetic_candidate_ratio: float
    valid_anchors: int
    valid_branches: int
    anchors_with_three_distinct_ratio: float
    exact_duplicate_rate: float
    meaningful_progress_spread_ratio: float
    local_event_disagreement_anchors: int
    mixed_terminal_outcome_anchors: int
    non_task6_outcome_divergence: bool
    ranking_tie_rate: float


def validate_anchor_registry(
    anchors: list[dict[str, Any]],
    *,
    expected_anchors: int = 60,
    minimum_tasks: int = 6,
    minimum_suites: int = 2,
    minimum_per_task: int = 8,
    maximum_task6_ratio: float = 0.35,
) -> dict[str, Any]:
    """Validate frozen anchor identities and task/suite balance."""
    if len(anchors) != expected_anchors:
        raise ValueError(
            f"anchor registry contains {len(anchors)} anchors, "
            f"expected {expected_anchors}"
        )
    ids = [str(item.get("anchor_id", "")) for item in anchors]
    if any(not anchor_id for anchor_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("anchor ids must be complete and unique")
    episodes = [str(item.get("episode_id", "")) for item in anchors]
    if any(not episode for episode in episodes) or len(set(episodes)) != len(episodes):
        raise ValueError("pilot requires independent single-anchor source episodes")
    tasks = Counter(
        (str(item.get("suite", "")), int(item.get("task_id", -1))) for item in anchors
    )
    suites = {suite for suite, _ in tasks}
    if len(tasks) < minimum_tasks or len(suites) < minimum_suites:
        raise ValueError("anchor task/suite coverage is below the registered minimum")
    if min(tasks.values(), default=0) < minimum_per_task:
        raise ValueError("one or more tasks has too few anchors")
    task6 = sum(count for (_, task_id), count in tasks.items() if task_id == 6)
    ratio = task6 / len(anchors)
    if ratio > maximum_task6_ratio:
        raise ValueError("task 6 anchor ratio exceeds the registered maximum")
    final_seed_hits = [
        int(item["seed"])
        for item in anchors
        if 900_000 <= int(item.get("seed", -1)) <= 900_099
    ]
    if final_seed_hits:
        raise ValueError("sealed final seed accessed")
    return {
        "anchors": len(anchors),
        "tasks": len(tasks),
        "suites": len(suites),
        "minimum_task_anchors": min(tasks.values()),
        "task6_anchor_ratio": ratio,
        "final_seed_hits": final_seed_hits,
    }


def evaluate_counterfactual_gate(
    evidence: CounterfactualGateEvidence,
) -> dict[str, Any]:
    """Return Result A authorization only when every required gate passes."""
    infrastructure = {
        "restore_mismatch": evidence.restore_mismatches == 0,
        "branch_contamination": evidence.branch_contamination == 0,
        "candidate_metadata": evidence.candidate_metadata_completeness == 1.0,
        "policy_checkpoint_identity": evidence.identity_completeness == 1.0,
    }
    authenticity = {
        "policy_generated_only": evidence.policy_generated_candidate_ratio == 1.0,
        "synthetic_candidates_zero": evidence.synthetic_candidate_ratio == 0.0,
        "valid_anchors": evidence.valid_anchors >= 50,
        "valid_branches": evidence.valid_branches >= 150,
    }
    diversity = {
        "three_distinct_candidates": (
            evidence.anchors_with_three_distinct_ratio >= 0.70
        ),
        "exact_duplicate_rate": evidence.exact_duplicate_rate <= 0.20,
    }
    outcome_alternatives = {
        "short_horizon_progress": (evidence.meaningful_progress_spread_ratio >= 0.30),
        "local_events": evidence.local_event_disagreement_anchors >= 15,
        "mixed_terminal": evidence.mixed_terminal_outcome_anchors >= 10,
    }
    outcome = {
        "at_least_one_divergence_target": any(outcome_alternatives.values()),
        "non_task6_divergence": evidence.non_task6_outcome_divergence,
    }
    ranking = {
        "distinguishable_targets": evidence.ranking_tie_rate < 1.0,
        "tie_rate": evidence.ranking_tie_rate <= 0.70,
    }
    authorized = all(
        all(group.values())
        for group in (infrastructure, authenticity, diversity, outcome, ranking)
    )
    return {
        "schema_version": "latentguard.lg_r2b0.lg_r2b1_gate.v1",
        "status": "pass" if authorized else "fail",
        "result": "A" if authorized else "B_OR_C",
        "LG_R2B1_AUTHORIZED": authorized,
        "infrastructure": infrastructure,
        "candidate_authenticity": authenticity,
        "candidate_diversity": diversity,
        "outcome_diversity": outcome,
        "outcome_alternatives": outcome_alternatives,
        "ranking": ranking,
    }


__all__ = [
    "CounterfactualGateEvidence",
    "evaluate_counterfactual_gate",
    "validate_anchor_registry",
]
