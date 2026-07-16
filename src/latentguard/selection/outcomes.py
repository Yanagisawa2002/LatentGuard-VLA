"""Project joined strong replay evidence into M3C outcome records."""

from __future__ import annotations

from typing import NoReturn

from latentguard.evaluation.models import EvaluationStatus
from latentguard.selection.evaluation import CompletePoolReplayJoin
from latentguard.selection.metrics import CandidateOutcomeV1
from latentguard.selection.models import CandidatePoolV1


class SelectionOutcomeError(ValueError):
    """Raised when replay evidence cannot form exact complete-pool outcomes."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionOutcomeError(f"{context}: {reason}")


def candidate_outcomes_from_replay_join(
    candidate_pool: CandidatePoolV1,
    replay: CompletePoolReplayJoin,
) -> tuple[CandidateOutcomeV1, ...]:
    """Join reporting-only pool metadata after physical replay has completed."""

    if replay.candidate_pool_digest != candidate_pool.content_digest:
        _fail("replay join", "candidate pool digest differs")
    if replay.proposal_ids != candidate_pool.proposal_ids:
        _fail("replay join", "proposal order differs")
    locations = {
        candidate.proposal_id: (group, candidate)
        for group in candidate_pool.groups
        for candidate in group.candidates
    }
    outcomes: list[CandidateOutcomeV1] = []
    for proposal_id in candidate_pool.proposal_ids:
        evidence = replay.evidence_by_proposal[proposal_id]
        group, candidate = locations[proposal_id]
        if (
            evidence.status is not EvaluationStatus.CONCLUSIVE
            or type(evidence.success) is not bool
            or type(evidence.unsafe) is not bool
            or evidence.label_strength is None
        ):
            _fail(proposal_id, "joined evidence is not conclusive")
        component_count = evidence.metrics.get(
            "replay_baseline_restoration_compared_component_count"
        )
        if type(component_count) is not int:
            _fail(proposal_id, "state component count is absent")
        outcomes.append(
            CandidateOutcomeV1(
                proposal_id=proposal_id,
                group_id=group.group_id,
                source_trajectory_id=group.source_trajectory_id,
                distribution=candidate.distribution.value,
                evidence_id=evidence.evidence_id,
                status="conclusive",
                success=evidence.success,
                unsafe=evidence.unsafe,
                simulator_replay_verified=evidence.simulator_replay_verified,
                label_strength=evidence.label_strength.value,
                state_component_count=component_count,
            )
        )
    return tuple(outcomes)


__all__ = [
    "SelectionOutcomeError",
    "candidate_outcomes_from_replay_join",
]
