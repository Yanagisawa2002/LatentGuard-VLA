"""Strict validation and outcome projection for evaluation evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import NoReturn, cast
from urllib.parse import urlsplit

from latentguard.evaluation.models import (
    CONFIGURATION_DIGEST_PREFIX,
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_ID_PREFIX,
    EvaluationEvidence,
    EvaluationIdentityError,
    EvaluationStatus,
    compute_evidence_identifier,
)
from latentguard.models import (
    CURRENT_SCHEMA_VERSION,
    FailureEvent,
    LabelSource,
    LabelStrength,
    OutcomeLabel,
)
from latentguard.validation import (
    DataValidationError,
    validate_failure_event,
    validate_outcome_label,
)

_CONFIGURATION_DIGEST_PATTERN = re.compile(r"^cfg-sha256-[0-9a-f]{64}$")
_EVIDENCE_ID_PATTERN = re.compile(r"^evd-sha256-[0-9a-f]{64}$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_CREDENTIAL_REFERENCE_PATTERN = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9])"
    r"(?:passwords?|passwds?|tokens?|secrets?|credentials?|api[_-]?keys?|"
    r"access[_-]?keys?|auth[_-]?tokens?|client[_-]?secrets?|"
    r"private[_-]?keys?|ssh[_-]?passwords?)"
    r"(?:$|[^A-Za-z0-9])"
)
_PROGRESS_TOLERANCE = 1e-12
_TASK_METRIC_NAMES = frozenset(
    {
        "failure",
        "failure_event",
        "failure_events",
        "progress",
        "progress_after",
        "progress_before",
        "progress_delta",
        "success",
        "unsafe",
    }
)


class EvaluationEvidenceValidationError(ValueError):
    """Raised when evidence is incomplete, inconsistent, or unsafe."""

    def __init__(self, field: str, evidence_id: str, reason: str) -> None:
        """Build an error that identifies the malformed evidence field."""
        self.field = field
        self.evidence_id = evidence_id
        self.reason = reason
        super().__init__(
            f"EvaluationEvidence[id={evidence_id or '<missing>'}].{field}: {reason}"
        )


class OutcomeProjectionError(ValueError):
    """Raised when evidence cannot validly become an M0 outcome label."""


def _fail(evidence: EvaluationEvidence, field: str, reason: str) -> NoReturn:
    raise EvaluationEvidenceValidationError(field, evidence.evidence_id, reason)


def _required_text(evidence: EvaluationEvidence, field: str) -> str:
    value: object = getattr(evidence, field)
    if not isinstance(value, str) or not value:
        _fail(evidence, field, "must be a non-empty string")
    if value != value.strip():
        _fail(evidence, field, "must not contain surrounding whitespace")
    if "\x00" in value:
        _fail(evidence, field, "must not contain NUL characters")
    return value


def _optional_text(evidence: EvaluationEvidence, field: str) -> None:
    value: object = getattr(evidence, field)
    if value is None:
        return
    _required_text(evidence, field)


def _normalized_progress(
    evidence: EvaluationEvidence, field: str, value: object
) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        _fail(evidence, field, "must be a finite number or null")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail(evidence, field, "must be finite")
    if not 0.0 <= number <= 1.0:
        _fail(evidence, field, f"must be within [0, 1], got {number}")
    return number


def _progress_change(evidence: EvaluationEvidence, value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        _fail(evidence, "progress_delta", "must be a finite number or null")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail(evidence, "progress_delta", "must be finite")
    if not -1.0 <= number <= 1.0:
        _fail(evidence, "progress_delta", f"must be within [-1, 1], got {number}")
    return number


def _validate_metrics(evidence: EvaluationEvidence) -> None:
    metrics: object = evidence.metrics
    if not isinstance(metrics, Mapping):
        _fail(evidence, "metrics", "must be a mapping of JSON scalar values")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key or key != key.strip():
            _fail(
                evidence,
                "metrics",
                "keys must be non-empty strings without surrounding whitespace",
            )
        if value is not None and type(value) not in (str, int, float, bool):
            _fail(
                evidence,
                "metrics",
                f"metric {key!r} has unsupported type {type(value).__name__}",
            )
        if type(value) is float and not math.isfinite(value):
            _fail(evidence, "metrics", f"metric {key!r} must be finite")


def _validate_artifact_reference(
    evidence: EvaluationEvidence, reference: object, index: int
) -> str:
    context = f"artifact_references[{index}]"
    if not isinstance(reference, str) or not reference:
        _fail(evidence, context, "must be a non-empty relative path")
    if reference != reference.strip() or "\x00" in reference or "\\" in reference:
        _fail(evidence, context, "must be a canonical portable relative path")
    try:
        parsed = urlsplit(reference)
    except ValueError as exc:
        _fail(evidence, context, f"malformed URL-like reference: {exc}")
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        _fail(evidence, context, "URLs and URI-like references are not allowed")
    if _CREDENTIAL_REFERENCE_PATTERN.search(reference) is not None:
        _fail(evidence, context, "credential-like path components are not allowed")
    pure = PurePosixPath(reference)
    parts = reference.split("/")
    if (
        pure.is_absolute()
        or any(part in ("", ".", "..") for part in parts)
        or ":" in reference
    ):
        _fail(evidence, context, "path must remain within the evaluation run directory")
    return reference


def _validate_failure_events(evidence: EvaluationEvidence) -> None:
    failure_events: object = evidence.failure_events
    if not isinstance(failure_events, tuple):
        _fail(evidence, "failure_events", "must be an immutable tuple")
    for index, failure in enumerate(failure_events):
        if not isinstance(failure, FailureEvent):
            _fail(
                evidence,
                "failure_events",
                f"item {index} is not a FailureEvent",
            )
        try:
            validate_failure_event(failure, f"{evidence.evidence_id}:failure:{index}")
        except DataValidationError as exc:
            _fail(evidence, "failure_events", str(exc))


def _validate_label_metadata(evidence: EvaluationEvidence) -> None:
    source: object = evidence.label_source
    strength: object = evidence.label_strength
    if (source is None) != (strength is None):
        _fail(
            evidence,
            "label_source",
            "label source and label strength must either both be present "
            "or both be null",
        )
    if source is not None and not isinstance(source, LabelSource):
        _fail(evidence, "label_source", f"unsupported label source {source!r}")
    if strength is not None and not isinstance(strength, LabelStrength):
        _fail(evidence, "label_strength", f"unsupported label strength {strength!r}")
    if type(evidence.simulator_replay_verified) is not bool:
        _fail(evidence, "simulator_replay_verified", "must be a boolean")
    if evidence.simulator_replay_verified and source is not LabelSource.SIMULATOR:
        _fail(
            evidence,
            "simulator_replay_verified",
            "verification may only be true for simulator-sourced evidence",
        )
    if source is LabelSource.HEURISTIC and strength is LabelStrength.STRONG:
        _fail(evidence, "label_strength", "heuristic evidence must be weak")
    if (
        source is LabelSource.SIMULATOR
        and strength is LabelStrength.STRONG
        and not evidence.simulator_replay_verified
    ):
        _fail(
            evidence,
            "simulator_replay_verified",
            "strong simulator evidence requires verified exact replay",
        )


def _validate_status_fields(evidence: EvaluationEvidence) -> None:
    if evidence.status is EvaluationStatus.CONCLUSIVE:
        if type(evidence.success) is not bool:
            _fail(evidence, "success", "conclusive evidence requires a boolean value")
        if evidence.progress_after is None:
            _fail(
                evidence,
                "progress_after",
                "conclusive evidence requires canonical post-evaluation progress",
            )
        if type(evidence.unsafe) is not bool:
            _fail(evidence, "unsafe", "conclusive evidence requires a boolean value")
        if evidence.label_source is None or evidence.label_strength is None:
            _fail(
                evidence,
                "label_source",
                "conclusive evidence requires label source and strength",
            )
        return

    if evidence.status is EvaluationStatus.INDETERMINATE:
        projection_fields = (
            evidence.success,
            evidence.progress_after,
            evidence.unsafe,
            evidence.label_source,
            evidence.label_strength,
        )
        if all(value is not None for value in projection_fields):
            _fail(
                evidence,
                "status",
                "indeterminate evidence must remain insufficient for outcome "
                "projection rather than carrying every required task field",
            )
        return

    if evidence.status in {
        EvaluationStatus.INVALID,
        EvaluationStatus.SKIPPED,
        EvaluationStatus.EXECUTION_ERROR,
    }:
        task_fields = {
            "success": evidence.success,
            "progress_before": evidence.progress_before,
            "progress_after": evidence.progress_after,
            "progress_delta": evidence.progress_delta,
            "unsafe": evidence.unsafe,
        }
        populated = [name for name, value in task_fields.items() if value is not None]
        if populated:
            _fail(
                evidence,
                populated[0],
                f"{evidence.status.value} evidence cannot contain task outcomes",
            )
        if evidence.failure_events:
            _fail(
                evidence,
                "failure_events",
                f"{evidence.status.value} evidence cannot contain task failure labels",
            )
        if evidence.label_source is not None or evidence.label_strength is not None:
            _fail(
                evidence,
                "label_source",
                f"{evidence.status.value} evidence cannot claim task label metadata",
            )
        if evidence.simulator_replay_verified:
            _fail(
                evidence,
                "simulator_replay_verified",
                f"{evidence.status.value} evidence cannot claim verified replay",
            )
        if evidence.termination_reason is None:
            _fail(
                evidence,
                "termination_reason",
                f"{evidence.status.value} evidence requires an explicit reason",
            )
        if evidence.status is EvaluationStatus.EXECUTION_ERROR:
            reserved = sorted(_TASK_METRIC_NAMES.intersection(evidence.metrics))
            if reserved:
                _fail(
                    evidence,
                    "metrics",
                    "execution-error diagnostics cannot use task-outcome metric "
                    f"names: {', '.join(reserved)}",
                )


def validate_evaluation_evidence(evidence: EvaluationEvidence) -> None:
    """Validate identity, provenance, status semantics, and safe diagnostics."""
    for field in (
        "evidence_id",
        "proposal_id",
        "source_dataset_id",
        "source_episode_id",
        "source_candidate_id",
        "split_group_id",
        "evaluator_id",
        "evaluator_version",
        "evaluator_configuration_digest",
    ):
        _required_text(evidence, field)
    _optional_text(evidence, "termination_reason")
    _optional_text(evidence, "notes")

    if _EVIDENCE_ID_PATTERN.fullmatch(evidence.evidence_id) is None:
        _fail(
            evidence,
            "evidence_id",
            f"expected {EVIDENCE_ID_PREFIX}<64 lowercase hex characters>",
        )
    if (
        _CONFIGURATION_DIGEST_PATTERN.fullmatch(evidence.evaluator_configuration_digest)
        is None
    ):
        _fail(
            evidence,
            "evaluator_configuration_digest",
            f"expected {CONFIGURATION_DIGEST_PREFIX}<64 lowercase hex characters>",
        )
    if _SEMANTIC_VERSION_PATTERN.fullmatch(evidence.evaluator_version) is None:
        _fail(evidence, "evaluator_version", "must be a semantic version such as 1.0.0")
    if evidence.schema_version != EVALUATION_EVIDENCE_SCHEMA_VERSION:
        _fail(
            evidence,
            "schema_version",
            f"unsupported version {evidence.schema_version!r}; supported: "
            f"{EVALUATION_EVIDENCE_SCHEMA_VERSION}",
        )
    status: object = evidence.status
    if not isinstance(status, EvaluationStatus):
        _fail(evidence, "status", f"unsupported evaluation status {evidence.status!r}")
    if type(evidence.evaluation_seed) is not int or not (
        0 <= evidence.evaluation_seed < 2**64
    ):
        _fail(evidence, "evaluation_seed", "must be an integer in [0, 2**64)")
    if type(evidence.attempt_ordinal) is not int or evidence.attempt_ordinal < 0:
        _fail(evidence, "attempt_ordinal", "must be a non-negative integer")
    if type(evidence.replayed_control_steps) is not int or (
        evidence.replayed_control_steps < 0
    ):
        _fail(evidence, "replayed_control_steps", "must be a non-negative integer")
    if evidence.success is not None and type(evidence.success) is not bool:
        _fail(evidence, "success", "must be a boolean or null")
    if evidence.unsafe is not None and type(evidence.unsafe) is not bool:
        _fail(evidence, "unsafe", "must be a boolean or null")

    before = _normalized_progress(evidence, "progress_before", evidence.progress_before)
    after = _normalized_progress(evidence, "progress_after", evidence.progress_after)
    delta = _progress_change(evidence, evidence.progress_delta)
    if before is not None and after is not None and delta is not None:
        expected_delta = after - before
        if not math.isclose(
            delta, expected_delta, rel_tol=0.0, abs_tol=_PROGRESS_TOLERANCE
        ):
            _fail(
                evidence,
                "progress_delta",
                f"expected progress_after - progress_before ({expected_delta}), "
                f"got {delta}",
            )

    _validate_metrics(evidence)
    artifact_references: object = evidence.artifact_references
    if not isinstance(artifact_references, tuple):
        _fail(evidence, "artifact_references", "must be an immutable tuple")
    references = [
        _validate_artifact_reference(evidence, reference, index)
        for index, reference in enumerate(artifact_references)
    ]
    if len(set(references)) != len(references):
        _fail(evidence, "artifact_references", "duplicate references are not allowed")
    _validate_failure_events(evidence)
    _validate_label_metadata(evidence)
    _validate_status_fields(evidence)

    try:
        expected_identifier = compute_evidence_identifier(
            proposal_id=evidence.proposal_id,
            evaluator_id=evidence.evaluator_id,
            evaluator_version=evidence.evaluator_version,
            evaluator_configuration_digest=evidence.evaluator_configuration_digest,
            evaluation_seed=evidence.evaluation_seed,
            attempt_ordinal=evidence.attempt_ordinal,
        )
    except EvaluationIdentityError as exc:
        _fail(evidence, "evidence_id", f"invalid identity inputs: {exc}")
    if evidence.evidence_id != expected_identifier:
        _fail(
            evidence,
            "evidence_id",
            "deterministic identifier mismatch; "
            f"expected {expected_identifier!r}, got {evidence.evidence_id!r}",
        )


def evidence_to_outcome_label(evidence: EvaluationEvidence) -> OutcomeLabel:
    """Project complete conclusive evidence using ``progress_after`` as progress."""
    try:
        validate_evaluation_evidence(evidence)
    except EvaluationEvidenceValidationError as exc:
        raise OutcomeProjectionError(
            f"cannot project malformed evidence: {exc}"
        ) from exc
    if evidence.status is not EvaluationStatus.CONCLUSIVE:
        raise OutcomeProjectionError(
            f"evidence {evidence.evidence_id!r} has status "
            f"{evidence.status.value!r}; only conclusive evidence can be projected"
        )

    success = evidence.success
    progress = evidence.progress_after
    unsafe = evidence.unsafe
    label_source = evidence.label_source
    label_strength = evidence.label_strength
    if (
        type(success) is not bool
        or progress is None
        or type(unsafe) is not bool
        or label_source is None
        or label_strength is None
    ):
        raise OutcomeProjectionError(
            f"evidence {evidence.evidence_id!r} is incomplete for projection"
        )

    outcome = OutcomeLabel(
        success=success,
        progress=float(progress),
        unsafe=unsafe,
        label_source=label_source,
        label_strength=label_strength,
        simulator_replay_verified=evidence.simulator_replay_verified,
        success_probability=None,
        unsafe_probability=None,
        failure_events=evidence.failure_events,
        schema_version=CURRENT_SCHEMA_VERSION,
    )
    try:
        validate_outcome_label(outcome, evidence.proposal_id)
    except DataValidationError as exc:
        raise OutcomeProjectionError(
            f"evidence {evidence.evidence_id!r} produced an invalid outcome: {exc}"
        ) from exc
    return outcome


__all__ = [
    "EvaluationEvidenceValidationError",
    "OutcomeProjectionError",
    "evidence_to_outcome_label",
    "validate_evaluation_evidence",
]
