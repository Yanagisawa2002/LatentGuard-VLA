"""Tests for the explicit M2A evidence-to-outcome projection boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from latentguard.evaluation.models import (
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.evaluation.validation import (
    OutcomeProjectionError,
    evidence_to_outcome_label,
)
from latentguard.models import FailureEvent, LabelSource, LabelStrength


def _conclusive_evidence() -> EvaluationEvidence:
    digest = compute_configuration_digest({"schema_version": "1.0", "limit": 1.0})
    proposal_id = "cap-sha256-" + "2" * 64
    evidence_id = compute_evidence_identifier(
        proposal_id=proposal_id,
        evaluator_id="projection_test",
        evaluator_version="1.0.0",
        evaluator_configuration_digest=digest,
        evaluation_seed=91,
        attempt_ordinal=0,
    )
    return EvaluationEvidence(
        evidence_id=evidence_id,
        proposal_id=proposal_id,
        source_dataset_id="sha256:source",
        source_episode_id="episode-2",
        source_candidate_id="candidate-2",
        split_group_id="episode-2",
        evaluator_id="projection_test",
        evaluator_version="1.0.0",
        evaluator_configuration_digest=digest,
        evaluation_seed=91,
        attempt_ordinal=0,
        status=EvaluationStatus.CONCLUSIVE,
        success=False,
        progress_before=0.75,
        progress_after=0.25,
        progress_delta=-0.5,
        unsafe=True,
        failure_events=(
            FailureEvent(
                failure_type="synthetic_collision",
                timestamp_s=0.5,
                probability=0.8,
                description="fixture annotation",
            ),
        ),
        termination_reason="synthetic fixture completed",
        replayed_control_steps=0,
        metrics={"fixture_score": 0.25},
        artifact_references=("artifacts/summary.json",),
        label_source=LabelSource.DETERMINISTIC_EVALUATOR,
        label_strength=LabelStrength.WEAK,
        simulator_replay_verified=False,
        notes="projection fixture",
    )


def _snapshot(evidence: EvaluationEvidence) -> tuple[object, ...]:
    return (
        evidence.evidence_id,
        evidence.status,
        evidence.success,
        evidence.progress_before,
        evidence.progress_after,
        evidence.progress_delta,
        evidence.unsafe,
        evidence.failure_events,
        dict(evidence.metrics),
        evidence.artifact_references,
        evidence.label_source,
        evidence.label_strength,
        evidence.simulator_replay_verified,
    )


def test_complete_conclusive_evidence_projects_using_progress_after() -> None:
    evidence = _conclusive_evidence()

    outcome = evidence_to_outcome_label(evidence)

    assert outcome.success is False
    assert outcome.progress == evidence.progress_after == 0.25
    assert outcome.unsafe is True
    assert outcome.label_source is LabelSource.DETERMINISTIC_EVALUATOR
    assert outcome.label_strength is LabelStrength.WEAK
    assert outcome.simulator_replay_verified is False
    assert outcome.failure_events == evidence.failure_events
    assert outcome.success_probability is None
    assert outcome.unsafe_probability is None


def test_projection_does_not_mutate_evidence() -> None:
    evidence = _conclusive_evidence()
    before = _snapshot(evidence)

    evidence_to_outcome_label(evidence)

    assert _snapshot(evidence) == before


@pytest.mark.parametrize("field", ["success", "progress_after", "unsafe"])
def test_incomplete_conclusive_evidence_is_rejected(field: str) -> None:
    evidence = replace(_conclusive_evidence(), **{field: None})

    with pytest.raises(OutcomeProjectionError, match="malformed|incomplete"):
        evidence_to_outcome_label(evidence)


@pytest.mark.parametrize(
    "status",
    [
        EvaluationStatus.INDETERMINATE,
        EvaluationStatus.INVALID,
        EvaluationStatus.SKIPPED,
        EvaluationStatus.EXECUTION_ERROR,
    ],
)
def test_nonconclusive_evidence_cannot_be_projected(
    status: EvaluationStatus,
) -> None:
    evidence = replace(
        _conclusive_evidence(),
        status=status,
        success=None,
        progress_before=None,
        progress_after=None,
        progress_delta=None,
        unsafe=None,
        failure_events=(),
        termination_reason=(
            None if status is EvaluationStatus.INDETERMINATE else "reason"
        ),
        label_source=None,
        label_strength=None,
    )

    with pytest.raises(OutcomeProjectionError, match="only conclusive"):
        evidence_to_outcome_label(evidence)


def test_verified_strong_simulator_evidence_can_project() -> None:
    evidence = replace(
        _conclusive_evidence(),
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
    )

    outcome = evidence_to_outcome_label(evidence)

    assert outcome.label_source is LabelSource.SIMULATOR
    assert outcome.label_strength is LabelStrength.STRONG
    assert outcome.simulator_replay_verified is True


@pytest.mark.parametrize(
    ("source", "strength", "verified"),
    [
        (LabelSource.SIMULATOR, LabelStrength.STRONG, False),
        (LabelSource.HEURISTIC, LabelStrength.STRONG, False),
        (LabelSource.HUMAN, LabelStrength.WEAK, True),
    ],
)
def test_invalid_verification_metadata_cannot_cross_projection_boundary(
    source: LabelSource, strength: LabelStrength, verified: bool
) -> None:
    evidence = replace(
        _conclusive_evidence(),
        label_source=source,
        label_strength=strength,
        simulator_replay_verified=verified,
    )

    with pytest.raises(OutcomeProjectionError, match="malformed"):
        evidence_to_outcome_label(evidence)


def test_tampered_identity_cannot_cross_projection_boundary() -> None:
    evidence = replace(_conclusive_evidence(), evidence_id="evd-sha256-" + "f" * 64)

    with pytest.raises(OutcomeProjectionError, match="identifier mismatch"):
        evidence_to_outcome_label(evidence)
