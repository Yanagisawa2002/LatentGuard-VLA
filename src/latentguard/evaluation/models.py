"""Mutation-safe data models and deterministic identities for evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

from latentguard.models import FailureEvent, JsonScalar, LabelSource, LabelStrength

EVALUATION_EVIDENCE_SCHEMA_VERSION = "1.0"
"""Logical schema version supported for M2A evaluation evidence."""

SUPPORTED_EVALUATION_EVIDENCE_SCHEMA_VERSIONS = frozenset(
    {EVALUATION_EVIDENCE_SCHEMA_VERSION}
)
"""Evaluation-evidence schema versions supported by this release."""

EVIDENCE_ID_PREFIX = "evd-sha256-"
"""Prefix for canonical SHA-256 evaluation-evidence identifiers."""

CONFIGURATION_DIGEST_PREFIX = "cfg-sha256-"
"""Prefix for canonical evaluator-configuration digests."""

_CONFIGURATION_DIGEST_PATTERN = re.compile(r"^cfg-sha256-[0-9a-f]{64}$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_MAX_SEED = 2**64

CanonicalJsonValue: TypeAlias = (
    JsonScalar | Sequence["CanonicalJsonValue"] | Mapping[str, "CanonicalJsonValue"]
)


class EvaluationIdentityError(ValueError):
    """Raised when deterministic identity inputs are malformed."""


class EvaluationStatus(StrEnum):
    """Outcome-evidence state for one evaluator attempt."""

    CONCLUSIVE = "conclusive"
    INDETERMINATE = "indeterminate"
    INVALID = "invalid"
    SKIPPED = "skipped"
    EXECUTION_ERROR = "execution_error"


def _canonical_json_value(value: object, context: str) -> object:
    """Return a JSON-native copy without coercing unsupported values."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EvaluationIdentityError(f"{context}: float values must be finite")
        return value
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or key != key.strip():
                raise EvaluationIdentityError(
                    f"{context}: object keys must be non-empty strings without "
                    "surrounding whitespace"
                )
            canonical[key] = _canonical_json_value(item, f"{context}.{key}")
        return canonical
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_json_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise EvaluationIdentityError(
        f"{context}: unsupported JSON value {type(value).__name__}"
    )


def _canonical_json_bytes(value: object, context: str) -> bytes:
    canonical = _canonical_json_value(value, context)
    try:
        serialized = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return serialized.encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise EvaluationIdentityError(f"{context}: not canonical JSON: {exc}") from exc


def _require_identity_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationIdentityError(
            f"EvaluationIdentity.{field}: must be a non-empty string without "
            "surrounding whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise EvaluationIdentityError(
            f"EvaluationIdentity.{field}: control characters are unsupported"
        )
    return value


def validate_evaluator_identity(
    *, evaluator_id: object, evaluator_version: object
) -> None:
    """Validate stable evaluator identity before planning or hashing attempts."""
    _require_identity_text(evaluator_id, "evaluator_id")
    version = _require_identity_text(evaluator_version, "evaluator_version")
    if _SEMANTIC_VERSION_PATTERN.fullmatch(version) is None:
        raise EvaluationIdentityError(
            "EvaluationIdentity.evaluator_version: must be a semantic version "
            "such as 1.0.0"
        )


def compute_configuration_digest(configuration: Mapping[str, object]) -> str:
    """Compute a path-independent digest of a resolved evaluator configuration."""
    if not isinstance(configuration, Mapping):
        raise EvaluationIdentityError(
            "EvaluatorConfiguration: expected a mapping of canonical JSON values"
        )
    digest = hashlib.sha256(
        _canonical_json_bytes(configuration, "EvaluatorConfiguration")
    ).hexdigest()
    return f"{CONFIGURATION_DIGEST_PREFIX}{digest}"


def compute_evidence_identifier(
    *,
    proposal_id: str,
    evaluator_id: str,
    evaluator_version: str,
    evaluator_configuration_digest: str,
    evaluation_seed: int,
    attempt_ordinal: int,
) -> str:
    """Compute the canonical SHA-256 identity for one evaluator attempt."""
    _require_identity_text(proposal_id, "proposal_id")
    validate_evaluator_identity(
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
    )
    digest = _require_identity_text(
        evaluator_configuration_digest, "evaluator_configuration_digest"
    )
    if _CONFIGURATION_DIGEST_PATTERN.fullmatch(digest) is None:
        raise EvaluationIdentityError(
            "EvaluationIdentity.evaluator_configuration_digest: expected "
            f"{CONFIGURATION_DIGEST_PREFIX}<64 lowercase hex characters>"
        )
    if type(evaluation_seed) is not int or not 0 <= evaluation_seed < _MAX_SEED:
        raise EvaluationIdentityError(
            "EvaluationIdentity.evaluation_seed: must be an integer in "
            f"[0, {_MAX_SEED})"
        )
    if type(attempt_ordinal) is not int or attempt_ordinal < 0:
        raise EvaluationIdentityError(
            "EvaluationIdentity.attempt_ordinal: must be a non-negative integer"
        )
    payload = {
        "attempt_ordinal": attempt_ordinal,
        "evaluation_seed": evaluation_seed,
        "evaluator_configuration_digest": digest,
        "evaluator_id": evaluator_id,
        "evaluator_version": evaluator_version,
        "proposal_id": proposal_id,
    }
    hexadecimal = hashlib.sha256(
        _canonical_json_bytes(payload, "EvaluationIdentity")
    ).hexdigest()
    return f"{EVIDENCE_ID_PREFIX}{hexadecimal}"


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationEvidence:
    """Immutable task evidence and diagnostics for one proposal attempt."""

    evidence_id: str
    proposal_id: str
    source_dataset_id: str
    source_episode_id: str
    source_candidate_id: str
    split_group_id: str
    evaluator_id: str
    evaluator_version: str
    evaluator_configuration_digest: str
    evaluation_seed: int
    attempt_ordinal: int
    status: EvaluationStatus
    success: bool | None
    progress_before: float | None
    progress_after: float | None
    progress_delta: float | None
    unsafe: bool | None
    failure_events: tuple[FailureEvent, ...]
    termination_reason: str | None
    replayed_control_steps: int
    metrics: Mapping[str, JsonScalar]
    artifact_references: tuple[str, ...]
    label_source: LabelSource | None
    label_strength: LabelStrength | None
    simulator_replay_verified: bool
    notes: str | None
    schema_version: str = EVALUATION_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach mutable collections without inferring or repairing evidence."""
        failure_events: object = self.failure_events
        if isinstance(failure_events, Iterable) and not isinstance(
            failure_events, (str, bytes)
        ):
            object.__setattr__(self, "failure_events", tuple(failure_events))
        if isinstance(self.metrics, Mapping):
            object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        artifact_references: object = self.artifact_references
        if isinstance(artifact_references, Iterable) and not isinstance(
            artifact_references, (str, bytes)
        ):
            object.__setattr__(self, "artifact_references", tuple(artifact_references))


__all__ = [
    "CONFIGURATION_DIGEST_PREFIX",
    "EVALUATION_EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_ID_PREFIX",
    "SUPPORTED_EVALUATION_EVIDENCE_SCHEMA_VERSIONS",
    "EvaluationEvidence",
    "EvaluationIdentityError",
    "EvaluationStatus",
    "compute_configuration_digest",
    "compute_evidence_identifier",
    "validate_evaluator_identity",
]
