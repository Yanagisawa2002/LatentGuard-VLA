"""Machine-derived summaries and concise audit text for paired replay runs."""

from __future__ import annotations

from dataclasses import dataclass

from latentguard.evaluation.models import EvaluationStatus
from latentguard.evaluation.serialization import EvaluationDataset

BASELINE_VALID_METRIC = "replay_baseline_valid"
REPLAY_CASE_ID_METRIC = "replay_case_id"
BASELINE_STEPS_METRIC = "replay_baseline_steps"
CORRUPTED_STEPS_METRIC = "replay_corrupted_steps"


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Exact attempt-level replay counts for CLI reporting."""

    source_episode_count: int
    source_candidate_count: int
    proposal_count: int
    valid_baseline_count: int
    invalid_baseline_count: int
    conclusive_success_count: int
    conclusive_task_failure_count: int
    indeterminate_count: int
    invalid_count: int
    execution_error_count: int
    skipped_count: int
    projected_outcome_count: int
    resumed_without_rerun_count: int
    adapter_trust_tier: str


def build_replay_summary(
    dataset: EvaluationDataset,
    *,
    source_episode_count: int,
    source_candidate_count: int,
    adapter_trust_tier: str,
    resumed_without_rerun_count: int = 0,
) -> ReplaySummary:
    """Derive paired-replay counts from validated M2A evidence and ledger data."""
    for field, value in (
        ("source_episode_count", source_episode_count),
        ("source_candidate_count", source_candidate_count),
        ("resumed_without_rerun_count", resumed_without_rerun_count),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"ReplaySummary.{field}: expected a non-negative integer")
    if (
        not isinstance(adapter_trust_tier, str)
        or not adapter_trust_tier
        or adapter_trust_tier != adapter_trust_tier.strip()
    ):
        raise ValueError(
            "ReplaySummary.adapter_trust_tier: expected a non-empty canonical string"
        )

    valid_baselines = 0
    invalid_baselines = 0
    conclusive_successes = 0
    conclusive_failures = 0
    for evidence in dataset.evidence:
        baseline_valid = evidence.metrics.get(BASELINE_VALID_METRIC)
        if baseline_valid is True:
            valid_baselines += 1
        elif baseline_valid is False:
            invalid_baselines += 1
        if evidence.status is EvaluationStatus.CONCLUSIVE:
            if evidence.success is True:
                conclusive_successes += 1
            elif evidence.success is False:
                conclusive_failures += 1

    summary = dataset.summary
    return ReplaySummary(
        source_episode_count=source_episode_count,
        source_candidate_count=source_candidate_count,
        proposal_count=len(dataset.selected_proposal_ids),
        valid_baseline_count=valid_baselines,
        invalid_baseline_count=invalid_baselines,
        conclusive_success_count=conclusive_successes,
        conclusive_task_failure_count=conclusive_failures,
        indeterminate_count=summary.indeterminate,
        invalid_count=summary.invalid,
        execution_error_count=summary.execution_error,
        skipped_count=summary.skipped,
        projected_outcome_count=summary.projected_outcome_labels,
        resumed_without_rerun_count=resumed_without_rerun_count,
        adapter_trust_tier=adapter_trust_tier,
    )


def format_replay_summary(summary: ReplaySummary, *, resumed: bool) -> str:
    """Format one stable replay CLI result line with an explicit fixture warning."""
    warning = (
        " fixture_warning=non-physical-weak-non-simulator-infrastructure-only"
        if summary.adapter_trust_tier == "fixture"
        else ""
    )
    return (
        "replay-data result: "
        f"resumed={str(resumed).lower()} "
        f"source_episodes={summary.source_episode_count} "
        f"source_candidates={summary.source_candidate_count} "
        f"proposals={summary.proposal_count} "
        f"valid_baselines={summary.valid_baseline_count} "
        f"invalid_baselines={summary.invalid_baseline_count} "
        f"conclusive_success={summary.conclusive_success_count} "
        f"conclusive_task_failure={summary.conclusive_task_failure_count} "
        f"indeterminate={summary.indeterminate_count} "
        f"invalid={summary.invalid_count} "
        f"execution_error={summary.execution_error_count} "
        f"skipped={summary.skipped_count} "
        f"projected_outcomes={summary.projected_outcome_count} "
        f"resumed_without_rerun={summary.resumed_without_rerun_count} "
        f"adapter_trust_tier={summary.adapter_trust_tier}{warning}"
    )


def replay_audit_lines(dataset: EvaluationDataset) -> tuple[str, ...]:
    """Return deterministic attempt audit lines without runtime paths."""
    evidence_by_id = {item.evidence_id: item for item in dataset.evidence}
    lines: list[str] = []
    for entry in dataset.ledger:
        evidence = (
            None if entry.evidence_id is None else evidence_by_id.get(entry.evidence_id)
        )
        case_id = (
            "<pending>"
            if evidence is None
            else str(evidence.metrics.get(REPLAY_CASE_ID_METRIC, "<unavailable>"))
        )
        lines.append(
            "replay-data audit: "
            f"proposal={entry.proposal_id} attempt={entry.attempt_ordinal} "
            f"seed={entry.evaluation_seed} state={entry.state.value} "
            f"case={case_id} evidence={entry.evidence_id or '<pending>'}"
        )
    return tuple(lines)


__all__ = [
    "BASELINE_STEPS_METRIC",
    "BASELINE_VALID_METRIC",
    "CORRUPTED_STEPS_METRIC",
    "REPLAY_CASE_ID_METRIC",
    "ReplaySummary",
    "build_replay_summary",
    "format_replay_summary",
    "replay_audit_lines",
]
