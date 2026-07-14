"""Tests for deterministic, mutation-safe M2A evaluation evidence."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from latentguard.evaluation.models import (
    CONFIGURATION_DIGEST_PREFIX,
    EVIDENCE_ID_PREFIX,
    EvaluationEvidence,
    EvaluationIdentityError,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.evaluation.validation import (
    EvaluationEvidenceValidationError,
    validate_evaluation_evidence,
)
from latentguard.models import FailureEvent, LabelSource, LabelStrength


def _evidence(**updates: object) -> EvaluationEvidence:
    digest = compute_configuration_digest({"schema_version": "1.0", "threshold": 0.5})
    values: dict[str, object] = {
        "proposal_id": "cap-sha256-" + "1" * 64,
        "source_dataset_id": "sha256:source-dataset",
        "source_episode_id": "episode-0001",
        "source_candidate_id": "candidate-0001-00",
        "split_group_id": "episode-0001",
        "evaluator_id": "unit_test_evaluator",
        "evaluator_version": "1.2.3",
        "evaluator_configuration_digest": digest,
        "evaluation_seed": 17,
        "attempt_ordinal": 0,
        "status": EvaluationStatus.CONCLUSIVE,
        "success": True,
        "progress_before": 0.2,
        "progress_after": 0.7,
        "progress_delta": 0.5,
        "unsafe": False,
        "failure_events": (),
        "termination_reason": "fixture completed",
        "replayed_control_steps": 0,
        "metrics": {"score": 0.75, "accepted": True},
        "artifact_references": ("artifacts/metrics.json",),
        "label_source": LabelSource.DETERMINISTIC_EVALUATOR,
        "label_strength": LabelStrength.WEAK,
        "simulator_replay_verified": False,
        "notes": "synthetic evidence",
    }
    supplied_identifier = updates.pop("evidence_id", None)
    values.update(updates)
    values["evidence_id"] = supplied_identifier or compute_evidence_identifier(
        proposal_id=str(values["proposal_id"]),
        evaluator_id=str(values["evaluator_id"]),
        evaluator_version=str(values["evaluator_version"]),
        evaluator_configuration_digest=str(values["evaluator_configuration_digest"]),
        evaluation_seed=values["evaluation_seed"],  # type: ignore[arg-type]
        attempt_ordinal=values["attempt_ordinal"],  # type: ignore[arg-type]
    )
    return EvaluationEvidence(**values)  # type: ignore[arg-type]


def _nonconclusive(status: EvaluationStatus) -> EvaluationEvidence:
    return _evidence(
        status=status,
        success=None,
        progress_before=None,
        progress_after=None,
        progress_delta=None,
        unsafe=None,
        failure_events=(),
        label_source=None,
        label_strength=None,
        termination_reason=(
            None if status is EvaluationStatus.INDETERMINATE else "reason"
        ),
    )


def test_evidence_detaches_collections_and_is_frozen() -> None:
    metrics = {"score": 0.75}
    artifacts = ["artifacts/result.json"]
    failures = [FailureEvent(failure_type="fixture_failure")]
    evidence = _evidence(
        metrics=metrics,
        artifact_references=artifacts,
        failure_events=failures,
    )

    metrics["score"] = 0.0
    artifacts.append("artifacts/late.json")
    failures.clear()

    assert evidence.metrics == {"score": 0.75}
    assert evidence.artifact_references == ("artifacts/result.json",)
    assert len(evidence.failure_events) == 1
    with pytest.raises(TypeError):
        evidence.metrics["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        evidence.success = False  # type: ignore[misc]
    validate_evaluation_evidence(evidence)


def test_configuration_digest_is_canonical_and_path_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = {
        "schema_version": "1.0",
        "nested": {"enabled": True, "values": [1, 2.5, None]},
    }
    second = {
        "nested": {"values": [1, 2.5, None], "enabled": True},
        "schema_version": "1.0",
    }
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()

    monkeypatch.chdir(first_directory)
    first_digest = compute_configuration_digest(first)
    monkeypatch.chdir(second_directory)
    second_digest = compute_configuration_digest(second)

    assert first_digest == second_digest
    assert re.fullmatch(
        re.escape(CONFIGURATION_DIGEST_PREFIX) + r"[0-9a-f]{64}", first_digest
    )


@pytest.mark.parametrize(
    "configuration",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": object()},
        {"": 1},
        {" spaced ": 1},
        {"value": "\ud800"},
    ],
)
def test_configuration_digest_rejects_noncanonical_values(
    configuration: dict[str, object],
) -> None:
    with pytest.raises(EvaluationIdentityError):
        compute_configuration_digest(configuration)


def test_evidence_identifier_is_stable_and_every_identity_input_matters() -> None:
    evidence = _evidence()
    identity = {
        "proposal_id": evidence.proposal_id,
        "evaluator_id": evidence.evaluator_id,
        "evaluator_version": evidence.evaluator_version,
        "evaluator_configuration_digest": evidence.evaluator_configuration_digest,
        "evaluation_seed": evidence.evaluation_seed,
        "attempt_ordinal": evidence.attempt_ordinal,
    }

    assert evidence.evidence_id == compute_evidence_identifier(**identity)
    assert re.fullmatch(
        re.escape(EVIDENCE_ID_PREFIX) + r"[0-9a-f]{64}", evidence.evidence_id
    )
    replacements: tuple[tuple[str, object], ...] = (
        ("proposal_id", "another-proposal"),
        ("evaluator_id", "another-evaluator"),
        ("evaluator_version", "2.0.0"),
        (
            "evaluator_configuration_digest",
            compute_configuration_digest({"different": True}),
        ),
        ("evaluation_seed", 18),
        ("attempt_ordinal", 1),
    )
    for field, value in replacements:
        changed = dict(identity)
        changed[field] = value
        assert compute_evidence_identifier(**changed) != evidence.evidence_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("proposal_id", "proposal\x00private"),
        ("evaluator_id", "fixture\nspoofed"),
        ("evaluator_version", "1.0.0\x1b[2J"),
    ],
)
def test_evidence_identity_rejects_embedded_control_characters(
    field: str, value: str
) -> None:
    evidence = _evidence()
    identity: dict[str, object] = {
        "proposal_id": evidence.proposal_id,
        "evaluator_id": evidence.evaluator_id,
        "evaluator_version": evidence.evaluator_version,
        "evaluator_configuration_digest": evidence.evaluator_configuration_digest,
        "evaluation_seed": evidence.evaluation_seed,
        "attempt_ordinal": evidence.attempt_ordinal,
    }
    identity[field] = value

    with pytest.raises(EvaluationIdentityError, match=field):
        compute_evidence_identifier(**identity)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, 2**64, True, 1.5])
def test_evidence_identifier_rejects_invalid_seed(value: object) -> None:
    evidence = _evidence()
    with pytest.raises(EvaluationIdentityError, match="evaluation_seed"):
        compute_evidence_identifier(
            proposal_id=evidence.proposal_id,
            evaluator_id=evidence.evaluator_id,
            evaluator_version=evidence.evaluator_version,
            evaluator_configuration_digest=evidence.evaluator_configuration_digest,
            evaluation_seed=value,  # type: ignore[arg-type]
            attempt_ordinal=0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_dataset_id", ""),
        ("source_episode_id", "   "),
        ("evaluator_version", "version-one"),
        ("evaluation_seed", True),
        ("attempt_ordinal", -1),
        ("replayed_control_steps", True),
        ("schema_version", "99.0"),
        ("status", "conclusive"),
    ],
)
def test_validation_rejects_invalid_identity_and_schema_fields(
    field: str, value: object
) -> None:
    evidence = _evidence()
    malformed = replace(evidence, **{field: value})

    with pytest.raises(EvaluationEvidenceValidationError, match=field):
        validate_evaluation_evidence(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("progress_before", -0.01),
        ("progress_after", 1.01),
        ("progress_after", float("nan")),
        ("progress_delta", -1.01),
        ("progress_delta", float("inf")),
    ],
)
def test_validation_rejects_invalid_progress(field: str, value: object) -> None:
    with pytest.raises(EvaluationEvidenceValidationError, match=field):
        validate_evaluation_evidence(replace(_evidence(), **{field: value}))


def test_negative_consistent_progress_delta_is_valid() -> None:
    evidence = _evidence(
        progress_before=0.9,
        progress_after=0.2,
        progress_delta=-0.7,
    )

    validate_evaluation_evidence(evidence)


def test_inconsistent_progress_delta_is_rejected() -> None:
    with pytest.raises(EvaluationEvidenceValidationError, match="progress_delta"):
        validate_evaluation_evidence(replace(_evidence(), progress_delta=0.25))


@pytest.mark.parametrize(
    "metrics",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": [1, 2]},
        {"": 1},
        {" spaced ": 1},
    ],
)
def test_metrics_reject_nonfinite_nested_or_invalid_values(
    metrics: dict[str, object],
) -> None:
    evidence = replace(_evidence(), metrics=metrics)  # type: ignore[arg-type]
    with pytest.raises(EvaluationEvidenceValidationError, match="metrics"):
        validate_evaluation_evidence(evidence)


@pytest.mark.parametrize(
    "reference",
    [
        "../outside.json",
        "artifacts/../../outside.json",
        "/absolute/result.json",
        "C:/private/result.json",
        "C:\\private\\result.json",
        "//server/share/result.json",
        "https://user:password@example.test/result.json",
        "artifacts/password=do-not-store.log",
        "artifacts/api_key:do-not-store.log",
        "artifacts/credentials.json",
        "artifacts/secret.txt",
        "artifacts/token.txt",
        "artifacts/private_key.pem",
        "artifacts/passwords.txt",
        "artifacts/tokens.json",
        "artifacts/secrets.json",
        "artifacts/api_keys.json",
        "artifacts/access_keys.json",
        "artifacts/client_secrets.json",
        "artifacts/private_keys.pem",
        "https://[malformed/result.json",
        "file:private.json",
        "artifacts//result.json",
        "artifacts/./result.json",
        "artifacts/result.json\x00suffix",
    ],
)
def test_artifact_references_reject_escape_absolute_and_url_values(
    reference: str,
) -> None:
    evidence = replace(_evidence(), artifact_references=(reference,))
    with pytest.raises(EvaluationEvidenceValidationError, match="artifact_references"):
        validate_evaluation_evidence(evidence)


def test_duplicate_artifact_references_are_rejected() -> None:
    evidence = replace(
        _evidence(),
        artifact_references=("artifacts/result.json", "artifacts/result.json"),
    )
    with pytest.raises(EvaluationEvidenceValidationError, match="duplicate"):
        validate_evaluation_evidence(evidence)


@pytest.mark.parametrize(
    ("source", "strength", "verified"),
    [
        (LabelSource.HEURISTIC, LabelStrength.STRONG, False),
        (LabelSource.SIMULATOR, LabelStrength.STRONG, False),
        (LabelSource.HUMAN, LabelStrength.WEAK, True),
        (LabelSource.SIMULATOR, None, False),
    ],
)
def test_invalid_label_source_strength_and_verification_combinations(
    source: LabelSource, strength: LabelStrength | None, verified: bool
) -> None:
    evidence = replace(
        _evidence(),
        label_source=source,
        label_strength=strength,
        simulator_replay_verified=verified,
    )
    with pytest.raises(EvaluationEvidenceValidationError):
        validate_evaluation_evidence(evidence)


@pytest.mark.parametrize(
    "status",
    [
        EvaluationStatus.INDETERMINATE,
        EvaluationStatus.INVALID,
        EvaluationStatus.SKIPPED,
        EvaluationStatus.EXECUTION_ERROR,
    ],
)
def test_every_nonconclusive_status_can_be_represented_without_defaults(
    status: EvaluationStatus,
) -> None:
    validate_evaluation_evidence(_nonconclusive(status))


def test_projection_complete_indeterminate_evidence_is_rejected() -> None:
    evidence = replace(_evidence(), status=EvaluationStatus.INDETERMINATE)

    with pytest.raises(EvaluationEvidenceValidationError, match="indeterminate"):
        validate_evaluation_evidence(evidence)


@pytest.mark.parametrize(
    "status",
    [
        EvaluationStatus.INVALID,
        EvaluationStatus.SKIPPED,
        EvaluationStatus.EXECUTION_ERROR,
    ],
)
def test_non_task_statuses_reject_fabricated_outcomes(status: EvaluationStatus) -> None:
    evidence = replace(_nonconclusive(status), success=False)
    with pytest.raises(EvaluationEvidenceValidationError, match="task outcomes"):
        validate_evaluation_evidence(evidence)


def test_execution_error_rejects_task_named_diagnostic_metric() -> None:
    evidence = replace(
        _nonconclusive(EvaluationStatus.EXECUTION_ERROR), metrics={"success": False}
    )
    with pytest.raises(EvaluationEvidenceValidationError, match="task-outcome metric"):
        validate_evaluation_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [("success", None), ("progress_after", None), ("unsafe", None)],
)
def test_conclusive_evidence_requires_projection_fields(
    field: str, value: object
) -> None:
    with pytest.raises(EvaluationEvidenceValidationError, match=field):
        validate_evaluation_evidence(replace(_evidence(), **{field: value}))


def test_failure_events_are_validated() -> None:
    invalid = FailureEvent(failure_type="", probability=1.5)
    evidence = replace(_evidence(), failure_events=(invalid,))
    with pytest.raises(EvaluationEvidenceValidationError, match="failure_events"):
        validate_evaluation_evidence(evidence)


def test_tampered_evidence_identifier_is_rejected() -> None:
    evidence = replace(_evidence(), evidence_id=EVIDENCE_ID_PREFIX + "0" * 64)
    with pytest.raises(EvaluationEvidenceValidationError, match="identifier mismatch"):
        validate_evaluation_evidence(evidence)
