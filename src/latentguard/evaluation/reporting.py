"""Deterministic summaries and concise audit text for evaluation runs."""

from __future__ import annotations

from collections.abc import Sequence

from latentguard.evaluation.models import EvaluationEvidence
from latentguard.evaluation.serialization import (
    EvaluationDataset,
    EvaluationSummary,
    LedgerEntry,
    LedgerState,
)
from latentguard.evaluation.validation import evidence_to_outcome_label


def build_evaluation_summary(
    evidence: Sequence[EvaluationEvidence],
    ledger: Sequence[LedgerEntry],
) -> EvaluationSummary:
    """Recompute attempt-level counts without depending on completion order."""
    counts = {state: 0 for state in LedgerState}
    for entry in ledger:
        counts[entry.state] += 1
    evidence_by_id = {item.evidence_id: item for item in evidence}
    projected = 0
    for entry in ledger:
        if entry.state is LedgerState.COMPLETED and entry.evidence_id is not None:
            item = evidence_by_id.get(entry.evidence_id)
            if item is not None:
                evidence_to_outcome_label(item)
                projected += 1
    return EvaluationSummary(
        conclusive=counts[LedgerState.COMPLETED],
        indeterminate=counts[LedgerState.INDETERMINATE],
        invalid=counts[LedgerState.INVALID],
        skipped=counts[LedgerState.SKIPPED],
        execution_error=counts[LedgerState.EXECUTION_ERROR],
        projected_outcome_labels=projected,
        retried_attempts=sum(entry.attempt_ordinal > 0 for entry in ledger),
        total_attempts=len(ledger),
    )


def format_evaluation_summary(
    summary: EvaluationSummary,
    *,
    evaluator_id: str,
    resumed: bool = False,
    evaluated_attempts: int = 0,
    recovered_attempts: int = 0,
) -> str:
    """Format one stable single-line CLI summary with a fixture warning."""
    prefix = "evaluate-data result"
    warning = ""
    if evaluator_id == "deterministic_fixture":
        warning = " fixture_evidence=synthetic-weak-non-simulator"
    return (
        f"{prefix}: evaluator={evaluator_id} resumed={str(resumed).lower()} "
        f"evaluated_attempts={evaluated_attempts} "
        f"recovered_attempts={recovered_attempts} "
        f"conclusive={summary.conclusive} "
        f"indeterminate={summary.indeterminate} invalid={summary.invalid} "
        f"skipped={summary.skipped} execution_error={summary.execution_error} "
        f"projected_outcome_labels={summary.projected_outcome_labels} "
        f"retried_attempts={summary.retried_attempts}{warning}"
    )


def evaluation_audit_lines(dataset: EvaluationDataset) -> tuple[str, ...]:
    """Return concise, deterministic ledger lines without private paths."""
    return tuple(
        "evaluate-data audit: "
        f"proposal={entry.proposal_id} attempt={entry.attempt_ordinal} "
        f"seed={entry.evaluation_seed} state={entry.state.value} "
        f"evidence={entry.evidence_id or '<pending>'}"
        for entry in dataset.ledger
    )


__all__ = [
    "build_evaluation_summary",
    "evaluation_audit_lines",
    "format_evaluation_summary",
]
