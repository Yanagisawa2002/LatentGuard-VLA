"""Mutation-safe models for real policy candidates and executed branches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

_ACTION_DIMENSION = 7
_FINAL_SEED_START = 900_000
_FINAL_SEED_END = 900_099


class CandidateDisposition(StrEnum):
    """Content-based relation of one candidate to earlier anchor candidates."""

    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    MEANINGFULLY_DISTINCT = "MEANINGFULLY_DISTINCT"
    INVALID = "INVALID"


class TerminalOutcome(StrEnum):
    """Conclusive or explicitly unresolved result of one branch."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNRESOLVED_HORIZON = "UNRESOLVED_HORIZON"
    ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"


class Recoverability(StrEnum):
    """Outcome of the frozen common continuation after a candidate chunk."""

    RECOVERED = "continuation_policy_recovered"
    FAILED = "continuation_policy_failed"
    NOT_DETERMINED = "not_determined"


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return MappingProxyType(decoded)


def _freeze_action(
    value: npt.ArrayLike,
    field_name: str,
    *,
    check_native_bounds: bool,
) -> npt.NDArray[np.float32]:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != _ACTION_DIMENSION:
        raise ValueError(f"{field_name} must have shape [horizon, 7]")
    if array.shape[0] < 1:
        raise ValueError(f"{field_name} must contain at least one action")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} contains non-finite values")
    frozen = np.ascontiguousarray(array, dtype=np.float32)
    if check_native_bounds and (
        bool(np.any(frozen < -1.0)) or bool(np.any(frozen > 1.0))
    ):
        raise ValueError(f"{field_name} exceeds the frozen LIBERO [-1, 1] bounds")
    if check_native_bounds and not bool(np.all(np.isin(frozen[:, -1], (-1.0, 1.0)))):
        raise ValueError(f"{field_name} has invalid binary gripper commands")
    immutable = np.frombuffer(frozen.tobytes(), dtype=np.float32).reshape(frozen.shape)
    return immutable


def _freeze_mask(
    value: npt.ArrayLike | None,
    horizon: int,
) -> npt.NDArray[np.bool_] | None:
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.shape != (horizon,):
        raise ValueError("action_mask must have shape [horizon]")
    frozen = np.ascontiguousarray(mask, dtype=np.bool_)
    return np.frombuffer(frozen.tobytes(), dtype=np.bool_)


def action_content_sha256(
    normalized_action_chunk: npt.ArrayLike,
    native_action_chunk: npt.ArrayLike,
    action_mask: npt.ArrayLike | None,
) -> str:
    """Hash both float32 action representations and the exact mask semantic."""
    normalized = np.ascontiguousarray(
        np.asarray(normalized_action_chunk, dtype=np.float32)
    )
    native = np.ascontiguousarray(np.asarray(native_action_chunk, dtype=np.float32))
    if normalized.shape != native.shape:
        raise ValueError("normalized and native chunks must align")
    mask = (
        None
        if action_mask is None
        else np.ascontiguousarray(np.asarray(action_mask, dtype=np.bool_))
    )
    if mask is not None and mask.shape != (native.shape[0],):
        raise ValueError("action mask does not align with chunk horizon")
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "mask": (
                None
                if mask is None
                else {"dtype": mask.dtype.str, "shape": list(mask.shape)}
            ),
            "native": {"dtype": native.dtype.str, "shape": list(native.shape)},
            "normalized": {
                "dtype": normalized.dtype.str,
                "shape": list(normalized.shape),
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(normalized.tobytes())
    digest.update(native.tobytes())
    if mask is not None:
        digest.update(mask.tobytes())
    return digest.hexdigest()


def _validate_digest(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class RealPolicyCandidate:
    """One native policy sample generated again from the current anchor."""

    candidate_id: str
    anchor_id: str
    policy_id: str
    checkpoint_revision: str
    processor_revision: str
    inference_seed: int | None
    sampling_config: Mapping[str, Any]
    normalized_action_chunk: npt.NDArray[np.float32]
    native_action_chunk: npt.NDArray[np.float32]
    action_mask: npt.NDArray[np.bool_] | None
    content_sha256: str
    generation_latency_ms: float
    source_kind: str = "native_policy_sampling"
    disposition: CandidateDisposition = CandidateDisposition.MEANINGFULLY_DISTINCT

    def __post_init__(self) -> None:
        """Validate identity, provenance, shapes, bounds, and no-training source."""
        for field_name in (
            "candidate_id",
            "anchor_id",
            "policy_id",
            "checkpoint_revision",
            "processor_revision",
        ):
            _required_text(str(getattr(self, field_name)), field_name)
        if self.inference_seed is not None:
            if isinstance(self.inference_seed, bool) or not isinstance(
                self.inference_seed, int
            ):
                raise ValueError("inference_seed must be an integer or null")
            if _FINAL_SEED_START <= self.inference_seed <= _FINAL_SEED_END:
                raise ValueError("sealed final seed accessed")
        if self.source_kind != "native_policy_sampling":
            raise ValueError("synthetic or transferred candidates are prohibited")
        if not isinstance(self.disposition, CandidateDisposition):
            raise ValueError("disposition must be a CandidateDisposition")
        normalized = _freeze_action(
            self.normalized_action_chunk,
            "normalized_action_chunk",
            check_native_bounds=False,
        )
        native = _freeze_action(
            self.native_action_chunk,
            "native_action_chunk",
            check_native_bounds=True,
        )
        if normalized.shape != native.shape:
            raise ValueError("normalized and native action chunks must align")
        mask = _freeze_mask(self.action_mask, int(native.shape[0]))
        sampling = _freeze_mapping(self.sampling_config, "sampling_config")
        _validate_digest(self.content_sha256, "content_sha256")
        if self.content_sha256 != action_content_sha256(normalized, native, mask):
            raise ValueError("candidate content_sha256 does not match action content")
        if (
            isinstance(self.generation_latency_ms, bool)
            or not isinstance(self.generation_latency_ms, (int, float))
            or not math.isfinite(float(self.generation_latency_ms))
            or float(self.generation_latency_ms) < 0
        ):
            raise ValueError("generation_latency_ms must be finite and non-negative")
        object.__setattr__(self, "normalized_action_chunk", normalized)
        object.__setattr__(self, "native_action_chunk", native)
        object.__setattr__(self, "action_mask", mask)
        object.__setattr__(self, "sampling_config", sampling)


@dataclass(frozen=True, slots=True, eq=False)
class CounterfactualBranch:
    """One real candidate executed from a content-bound restored anchor."""

    anchor_id: str
    candidate_id: str
    policy_metadata: Mapping[str, Any]
    sampling_metadata: Mapping[str, Any]
    action_chunk: npt.NDArray[np.float32]
    action_mask: npt.NDArray[np.bool_] | None
    short_horizon_trajectory: Mapping[str, Any]
    continuation_trajectory: Mapping[str, Any] | None
    progress_deltas: Mapping[str, float]
    event_labels: Mapping[str, bool]
    terminal_outcome: TerminalOutcome
    recoverability: Recoverability
    simulator_state_hash_before: str
    simulator_state_hash_after: str

    def __post_init__(self) -> None:
        """Validate branch completeness without inventing missing outcomes."""
        _required_text(self.anchor_id, "anchor_id")
        _required_text(self.candidate_id, "candidate_id")
        action = _freeze_action(
            self.action_chunk,
            "action_chunk",
            check_native_bounds=True,
        )
        mask = _freeze_mask(self.action_mask, int(action.shape[0]))
        policy = _freeze_mapping(self.policy_metadata, "policy_metadata")
        sampling = _freeze_mapping(self.sampling_metadata, "sampling_metadata")
        short = _freeze_mapping(
            self.short_horizon_trajectory,
            "short_horizon_trajectory",
        )
        continuation = (
            None
            if self.continuation_trajectory is None
            else _freeze_mapping(
                self.continuation_trajectory,
                "continuation_trajectory",
            )
        )
        progress = {
            str(key): float(value) for key, value in self.progress_deltas.items()
        }
        if not progress or not all(math.isfinite(value) for value in progress.values()):
            raise ValueError("progress_deltas must contain finite values")
        events = dict(self.event_labels)
        if not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in events.items()
        ):
            raise ValueError("event_labels must map strings to booleans")
        if not isinstance(self.terminal_outcome, TerminalOutcome):
            raise ValueError("terminal_outcome must be explicit")
        if not isinstance(self.recoverability, Recoverability):
            raise ValueError("recoverability must be explicit")
        _validate_digest(
            self.simulator_state_hash_before,
            "simulator_state_hash_before",
        )
        _validate_digest(
            self.simulator_state_hash_after,
            "simulator_state_hash_after",
        )
        object.__setattr__(self, "action_chunk", action)
        object.__setattr__(self, "action_mask", mask)
        object.__setattr__(self, "policy_metadata", policy)
        object.__setattr__(self, "sampling_metadata", sampling)
        object.__setattr__(self, "short_horizon_trajectory", short)
        object.__setattr__(self, "continuation_trajectory", continuation)
        object.__setattr__(self, "progress_deltas", MappingProxyType(progress))
        object.__setattr__(self, "event_labels", MappingProxyType(events))


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    """Frozen identity and selection evidence for one independent anchor."""

    anchor_id: str
    episode_id: str
    task: str
    suite: str
    task_id: int
    seed: int
    step_index: int
    stage: str
    stage_id: int
    progress: float
    simulator_state_hash: str
    observation_hash: str
    proprioception_hash: str
    policy_observation_history_hash: str
    source_rollout_identity: Mapping[str, Any]
    snapshot_locator: str

    def __post_init__(self) -> None:
        """Validate an outcome-independent anchor record."""
        for field_name in (
            "anchor_id",
            "episode_id",
            "task",
            "suite",
            "stage",
            "snapshot_locator",
        ):
            _required_text(str(getattr(self, field_name)), field_name)
        for field_name in (
            "simulator_state_hash",
            "observation_hash",
            "proprioception_hash",
        ):
            _validate_digest(str(getattr(self, field_name)), field_name)
        _validate_digest(
            self.policy_observation_history_hash,
            "policy_observation_history_hash",
        )
        if _FINAL_SEED_START <= self.seed <= _FINAL_SEED_END:
            raise ValueError("sealed final seed accessed")
        if self.step_index < 0 or self.task_id < 0 or self.stage_id < 0:
            raise ValueError("anchor integer indices must be non-negative")
        if not math.isfinite(self.progress):
            raise ValueError("anchor progress must be finite")
        object.__setattr__(
            self,
            "source_rollout_identity",
            _freeze_mapping(
                self.source_rollout_identity,
                "source_rollout_identity",
            ),
        )


def action_array_sha256(array: npt.ArrayLike) -> str:
    """Hash action dtype, shape, and exact C-order bytes."""
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "AnchorRecord",
    "CandidateDisposition",
    "CounterfactualBranch",
    "RealPolicyCandidate",
    "Recoverability",
    "TerminalOutcome",
    "action_array_sha256",
    "action_content_sha256",
]
