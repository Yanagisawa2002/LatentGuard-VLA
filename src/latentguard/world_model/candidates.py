"""Strict candidate-generation contracts for WM-v0 D1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

from .collector import ActionCandidate
from .data_schema import CandidateSource, WorldModelSchemaError


class CandidateOrigin(StrEnum):
    """Auditable origin of an action candidate."""

    POLICY_GENERATED = "policy_generated"
    SYNTHETIC_CORRUPTION = "synthetic_corruption"
    REPLAY_REFERENCE = "replay_reference"


class CandidateRejectionCode(StrEnum):
    """Stable reason codes emitted before simulator execution."""

    DUPLICATE_ACTION = "DUPLICATE_ACTION"
    DUPLICATE_IDENTITY = "DUPLICATE_IDENTITY"
    INVALID_ACTION_SHAPE = "INVALID_ACTION_SHAPE"
    NONFINITE_ACTION = "NONFINITE_ACTION"
    POLICY_INFERENCE_FAILED = "POLICY_INFERENCE_FAILED"
    INCOMPLETE_METADATA = "INCOMPLETE_METADATA"
    RESTORE_MISMATCH = "RESTORE_MISMATCH"
    OBSERVATION_MISALIGNMENT = "OBSERVATION_MISALIGNMENT"


class CandidateValidationError(WorldModelSchemaError):
    """Raised when one generated candidate violates the D1 contract."""

    def __init__(self, code: CandidateRejectionCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CandidateContext:
    """Outcome-free context supplied to one policy inference call."""

    episode_id: str
    source_episode_id: str
    anchor_id: str
    task_id: str
    scene_group: str
    episode_phase: str
    time_index: int
    seed: int
    expected_action_dimension: int
    minimum_action_horizon: int
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "source_episode_id",
            "anchor_id",
            "task_id",
            "scene_group",
            "episode_phase",
        ):
            _text(getattr(self, name), name)
        if type(self.time_index) is not int or self.time_index < 0:
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "time_index must be a non-negative integer",
            )
        if type(self.seed) is not int or self.seed < 0:
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "seed must be a non-negative integer",
            )
        if (
            type(self.expected_action_dimension) is not int
            or self.expected_action_dimension < 1
            or type(self.minimum_action_horizon) is not int
            or self.minimum_action_horizon < 1
        ):
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "action dimension and minimum horizon must be positive",
            )
        _canonical_configuration(self.extra, "context.extra")


@dataclass(frozen=True, slots=True, eq=False)
class PolicyCandidate:
    """One policy or non-policy candidate with reproduction provenance."""

    candidate_id: str
    action_chunk: NDArray[np.float32]
    action_mask: NDArray[np.bool_] | None
    candidate_origin: CandidateOrigin
    policy_family: str
    policy_name: str
    checkpoint_id: str
    checkpoint_hash: str | None
    inference_seed: int | None
    sampling_config: Mapping[str, object]
    action_horizon: int
    generation_latency_ms: float
    corruption_type: str | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate_id")
        for name in ("policy_family", "policy_name", "checkpoint_id"):
            _text(getattr(self, name), name)
        if not isinstance(self.candidate_origin, CandidateOrigin):
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "candidate_origin must be a CandidateOrigin",
            )
        if not isinstance(self.action_chunk, np.ndarray) or self.action_chunk.ndim != 2:
            raise CandidateValidationError(
                CandidateRejectionCode.INVALID_ACTION_SHAPE,
                "action_chunk must be a 2-D ndarray",
            )
        if self.action_chunk.shape[0] < 1 or self.action_chunk.shape[1] < 1:
            raise CandidateValidationError(
                CandidateRejectionCode.INVALID_ACTION_SHAPE,
                "action_chunk must have a non-empty horizon and action dimension",
            )
        if not np.issubdtype(self.action_chunk.dtype, np.number) or not bool(
            np.isfinite(self.action_chunk).all()
        ):
            raise CandidateValidationError(
                CandidateRejectionCode.NONFINITE_ACTION,
                "action_chunk must contain finite numeric values",
            )
        if type(self.action_horizon) is not int or self.action_horizon < 1:
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "action_horizon must be positive",
            )
        if self.action_horizon != self.action_chunk.shape[0]:
            raise CandidateValidationError(
                CandidateRejectionCode.INVALID_ACTION_SHAPE,
                "action_horizon differs from action_chunk rows",
            )
        if self.action_mask is not None:
            if (
                not isinstance(self.action_mask, np.ndarray)
                or self.action_mask.dtype != np.dtype("bool")
                or self.action_mask.shape != (self.action_horizon,)
                or not bool(self.action_mask.any())
            ):
                raise CandidateValidationError(
                    CandidateRejectionCode.INVALID_ACTION_SHAPE,
                    "action_mask must be a non-empty boolean horizon mask",
                )
            false_rows = np.flatnonzero(~self.action_mask)
            if false_rows.size and bool(self.action_mask[int(false_rows[0]) :].any()):
                raise CandidateValidationError(
                    CandidateRejectionCode.INVALID_ACTION_SHAPE,
                    "action_mask must select a contiguous prefix",
                )
        if self.inference_seed is not None and (
            type(self.inference_seed) is not int or self.inference_seed < 0
        ):
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "inference_seed must be a non-negative integer or None",
            )
        if (
            type(self.generation_latency_ms) not in (int, float)
            or not math.isfinite(float(self.generation_latency_ms))
            or float(self.generation_latency_ms) < 0.0
        ):
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "generation_latency_ms must be finite and non-negative",
            )
        _canonical_configuration(self.sampling_config, "sampling_config")
        if self.candidate_origin is CandidateOrigin.POLICY_GENERATED:
            if not _is_sha256(self.checkpoint_hash):
                raise CandidateValidationError(
                    CandidateRejectionCode.INCOMPLETE_METADATA,
                    "policy-generated candidates require a SHA-256 checkpoint hash",
                )
            if self.corruption_type is not None:
                raise CandidateValidationError(
                    CandidateRejectionCode.INCOMPLETE_METADATA,
                    "policy-generated candidates cannot declare corruption_type",
                )
        elif self.candidate_origin is CandidateOrigin.SYNTHETIC_CORRUPTION:
            if self.corruption_type is None:
                raise CandidateValidationError(
                    CandidateRejectionCode.INCOMPLETE_METADATA,
                    "synthetic candidates require corruption_type",
                )
            _text(self.corruption_type, "corruption_type")
        elif self.corruption_type is not None:
            raise CandidateValidationError(
                CandidateRejectionCode.INCOMPLETE_METADATA,
                "replay references cannot declare corruption_type",
            )

    @property
    def normalized_action_mask(self) -> NDArray[np.bool_]:
        """Return the explicit immutable action mask used for execution."""

        if self.action_mask is None:
            return np.ones(self.action_horizon, dtype=np.bool_)
        return np.array(self.action_mask, copy=True, dtype=np.bool_)

    @property
    def action_content_digest(self) -> str:
        """Return a dtype- and mask-bound action-content identity."""

        actions = np.asarray(self.action_chunk, dtype=np.dtype("<f4"), order="C")
        mask = np.asarray(self.normalized_action_mask, dtype=np.bool_, order="C")
        digest = hashlib.sha256()
        digest.update(b"PolicyCandidateActionV1\0")
        digest.update(str(actions.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(actions.tobytes(order="C"))
        digest.update(mask.tobytes(order="C"))
        return f"sha256:{digest.hexdigest()}"

    def as_metadata_mapping(self) -> dict[str, object]:
        """Return path-independent candidate metadata without action bytes."""

        return {
            "action_content_digest": self.action_content_digest,
            "action_horizon": self.action_horizon,
            "candidate_id": self.candidate_id,
            "candidate_origin": self.candidate_origin.value,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_id": self.checkpoint_id,
            "corruption_type": self.corruption_type,
            "generation_latency_ms": float(self.generation_latency_ms),
            "inference_seed": self.inference_seed,
            "policy_family": self.policy_family,
            "policy_name": self.policy_name,
            "sampling_config": dict(self.sampling_config),
            "schema_version": "wm-v0-policy-candidate-v1",
        }

    def to_action_candidate(self) -> ActionCandidate:
        """Project strict D1 provenance into the generic collector contract."""

        return ActionCandidate(
            candidate_id=self.candidate_id,
            actions=np.array(self.action_chunk, copy=True, dtype=np.float32),
            action_mask=self.normalized_action_mask,
            policy_source=CandidateSource(self.candidate_origin.value),
            corruption_type=self.corruption_type,
        )


class CandidateProvider(Protocol):
    """Generate actions from one current outcome-free anchor observation."""

    def generate(
        self,
        observation: dict[str, NDArray[np.uint8]],
        proprioception: NDArray[np.float32],
        instruction: str | None,
        context: CandidateContext,
    ) -> list[PolicyCandidate]:
        """Return candidates without using future observations or outcomes."""
        ...


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    """One explicit candidate exclusion before simulator execution."""

    code: CandidateRejectionCode
    candidate_id: str | None
    detail: str
    action_content_digest: str | None = None

    def as_mapping(self) -> dict[str, object]:
        """Return a JSON-native rejection record."""

        return {
            "action_content_digest": self.action_content_digest,
            "candidate_id": self.candidate_id,
            "detail": self.detail,
            "reason_code": self.code.value,
        }


def validate_and_deduplicate_candidates(
    candidates: Sequence[PolicyCandidate],
    *,
    context: CandidateContext,
) -> tuple[tuple[PolicyCandidate, ...], tuple[CandidateRejection, ...]]:
    """Validate candidate shapes and deterministically reject duplicates."""

    accepted: list[PolicyCandidate] = []
    rejected: list[CandidateRejection] = []
    identities: set[str] = set()
    contents: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in identities:
            rejected.append(
                CandidateRejection(
                    CandidateRejectionCode.DUPLICATE_IDENTITY,
                    candidate.candidate_id,
                    "candidate_id already occurred at this anchor",
                    candidate.action_content_digest,
                )
            )
            continue
        identities.add(candidate.candidate_id)
        if candidate.action_chunk.shape[1] != context.expected_action_dimension:
            rejected.append(
                CandidateRejection(
                    CandidateRejectionCode.INVALID_ACTION_SHAPE,
                    candidate.candidate_id,
                    "action dimension differs from the adapter contract",
                    candidate.action_content_digest,
                )
            )
            continue
        if int(candidate.normalized_action_mask.sum()) < context.minimum_action_horizon:
            rejected.append(
                CandidateRejection(
                    CandidateRejectionCode.INVALID_ACTION_SHAPE,
                    candidate.candidate_id,
                    "valid action prefix is shorter than the collection horizon",
                    candidate.action_content_digest,
                )
            )
            continue
        digest = candidate.action_content_digest
        if digest in contents:
            rejected.append(
                CandidateRejection(
                    CandidateRejectionCode.DUPLICATE_ACTION,
                    candidate.candidate_id,
                    "identical action content already occurred at this anchor",
                    digest,
                )
            )
            continue
        contents.add(digest)
        accepted.append(candidate)
    return tuple(accepted), tuple(rejected)


def checkpoint_sha256(path: Path) -> str:
    """Hash one regular checkpoint file without loading executable content."""

    checkpoint = Path(path).absolute()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise CandidateValidationError(
            CandidateRejectionCode.INCOMPLETE_METADATA,
            "checkpoint must be one regular non-link file",
        )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def verify_checkpoint_hash(path: Path, expected: str) -> str:
    """Verify one declared checkpoint hash and return the observed identity."""

    if not _is_sha256(expected):
        raise CandidateValidationError(
            CandidateRejectionCode.INCOMPLETE_METADATA,
            "expected checkpoint hash is not a SHA-256 identity",
        )
    observed = checkpoint_sha256(path)
    if observed != expected:
        raise CandidateValidationError(
            CandidateRejectionCode.INCOMPLETE_METADATA,
            "checkpoint content differs from the declared hash",
        )
    return observed


def candidate_set_digest(candidates: Sequence[PolicyCandidate]) -> str:
    """Return a deterministic identity for a complete ordered candidate set."""

    payload: list[dict[str, object]] = []
    for candidate in candidates:
        identity = candidate.as_metadata_mapping()
        identity.pop("generation_latency_ms")
        payload.append(identity)
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(payload, context="PolicyCandidateSetV1")
        ).hexdigest()
    )


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CandidateValidationError(
            CandidateRejectionCode.INCOMPLETE_METADATA,
            f"{name} must be non-empty stripped text",
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _canonical_configuration(value: Mapping[str, object], name: str) -> None:
    if not isinstance(value, Mapping):
        raise CandidateValidationError(
            CandidateRejectionCode.INCOMPLETE_METADATA,
            f"{name} must be a mapping",
        )
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CandidateValidationError(
            CandidateRejectionCode.INCOMPLETE_METADATA,
            f"{name} must be finite JSON: {exc}",
        ) from exc
    if not isinstance(decoded, dict):
        raise CandidateValidationError(
            CandidateRejectionCode.INCOMPLETE_METADATA,
            f"{name} must encode one JSON object",
        )


__all__ = [
    "CandidateContext",
    "CandidateOrigin",
    "CandidateProvider",
    "CandidateRejection",
    "CandidateRejectionCode",
    "CandidateValidationError",
    "PolicyCandidate",
    "candidate_set_digest",
    "checkpoint_sha256",
    "validate_and_deduplicate_candidates",
    "verify_checkpoint_hash",
]
