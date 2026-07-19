"""Strict, JSON-only cross-project records for the LangMani boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self, cast

SCHEMA_VERSION = "latentguard-langmani-contract-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)


class LangManiContractError(ValueError):
    """Raised when an integration record violates its versioned contract."""


class ObservationRole(StrEnum):
    """A field's only authorized use at the integration boundary."""

    POLICY_INPUT = "required_policy_input"
    VERIFIER_INPUT = "available_verifier_input"
    PRIVILEGED_VERIFIER = "privileged_verifier_only_input"
    RESTORATION_ONLY = "restoration_only_state"
    REPORTING_ONLY = "reporting_only_metadata"
    PROHIBITED_LEARNED = "prohibited_learned_input"
    UNRESOLVED = "unstable_or_unresolved"


class ProposalStage(StrEnum):
    """The non-interchangeable stage represented by an action proposal."""

    RAW = "raw_policy_proposal"
    PROJECTED = "projected_executable_proposal"
    EXECUTED = "actually_executed_action_sequence"


class OutcomeStatus(StrEnum):
    """Lossless outcome classes before any binary training projection."""

    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    HORIZON_EXHAUSTED = "horizon_exhausted"
    PROJECTION_FAILURE = "projection_failure"
    INVALID_ACTION = "invalid_action"
    EXECUTION_ERROR = "execution_error"
    INDETERMINATE = "indeterminate"
    SKIPPED = "skipped"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LangManiContractError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    text = _text(value, name)
    if _DIGEST.fullmatch(text) is None:
        raise LangManiContractError(f"{name} must be a sha256 digest")
    return text


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LangManiContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LangManiContractError(f"{name} must be finite")
    return result


def _freeze_json(value: object, name: str = "value") -> JsonValue:
    if value is None or type(value) in (bool, int, str):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LangManiContractError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LangManiContractError(f"{name} contains a non-string key")
            frozen[key] = _freeze_json(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, f"{name}[]") for item in value)
    raise LangManiContractError(
        f"{name} contains unsupported type {type(value).__name__}"
    )


def _thaw(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _thaw(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Return path-independent canonical JSON for a contract value."""

    return json.dumps(
        _thaw(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_digest(value: object) -> str:
    """Return the canonical SHA-256 identity of a contract value."""

    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _exact_payload(cls: type[Any], value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise LangManiContractError(f"{cls.__name__} must be a mapping")
    payload = dict(value)
    expected = {field.name for field in fields(cls)}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise LangManiContractError(
            f"{cls.__name__} fields differ; missing={missing}, extra={extra}"
        )
    return payload


class _Record:
    schema_version: str

    def to_dict(self) -> dict[str, object]:
        """Return a safe JSON-compatible representation."""

        return cast(dict[str, object], _thaw(self))

    @property
    def digest(self) -> str:
        """Return the record's canonical semantic digest."""

        return content_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class LangManiTaskContextV1(_Record):
    """Task identity with explicit model, identity, and reporting allowlists."""

    task_family: str
    instruction_text: str
    normalized_task_id: str
    object_id: str
    goal_id: str
    conditioning_id: str
    conditioning_vector: tuple[float, ...]
    simulator_task_configuration: Mapping[str, JsonValue]
    identity_only_fields: tuple[str, ...]
    model_input_fields: tuple[str, ...]
    reporting_only_fields: tuple[str, ...]
    reporting_labels: Mapping[str, JsonValue]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LangManiContractError("unsupported task-context schema")
        for name in (
            "task_family",
            "instruction_text",
            "normalized_task_id",
            "object_id",
            "goal_id",
            "conditioning_id",
        ):
            _text(getattr(self, name), name)
        vector = tuple(
            _finite(value, "conditioning_vector") for value in self.conditioning_vector
        )
        object.__setattr__(self, "conditioning_vector", vector)
        object.__setattr__(
            self,
            "simulator_task_configuration",
            cast(
                Mapping[str, JsonValue], _freeze_json(self.simulator_task_configuration)
            ),
        )
        object.__setattr__(
            self,
            "reporting_labels",
            cast(Mapping[str, JsonValue], _freeze_json(self.reporting_labels)),
        )
        groups = (
            self.identity_only_fields,
            self.model_input_fields,
            self.reporting_only_fields,
        )
        for group in groups:
            if len(group) != len(set(group)) or any(not item for item in group):
                raise LangManiContractError(
                    "task-context allowlists must contain unique names"
                )
        if any(
            set(left) & set(right)
            for left in groups
            for right in groups
            if left is not right
        ):
            raise LangManiContractError("task-context allowlists must be disjoint")
        if "instruction_text" in self.model_input_fields:
            raise LangManiContractError(
                "instruction text is not an authorized first-integration input"
            )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load a task context while rejecting unknown fields."""

        payload = _exact_payload(cls, value)
        payload["conditioning_vector"] = tuple(
            cast(Sequence[float], payload["conditioning_vector"])
        )
        for key in (
            "identity_only_fields",
            "model_input_fields",
            "reporting_only_fields",
        ):
            payload[key] = tuple(cast(Sequence[str], payload[key]))
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LangManiObservationFieldV1(_Record):
    """One audited observation field and its authorized role."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    role: ObservationRole
    units: str
    coordinate_frame: str
    normalization: str
    update_cadence: str
    source_location: str
    determinism: str
    reconstructable_after_restore: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LangManiContractError("unsupported observation-field schema")
        for name in (
            "name",
            "dtype",
            "units",
            "coordinate_frame",
            "normalization",
            "update_cadence",
            "source_location",
            "determinism",
        ):
            _text(getattr(self, name), name)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in self.shape
        ):
            raise LangManiContractError(
                "observation shape must contain positive integers"
            )
        object.__setattr__(self, "role", ObservationRole(self.role))
        if not isinstance(self.reconstructable_after_restore, bool):
            raise LangManiContractError("reconstructable_after_restore must be boolean")

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load an observation field while rejecting unknown fields."""

        payload = _exact_payload(cls, value)
        payload["shape"] = tuple(cast(Sequence[int], payload["shape"]))
        payload["role"] = ObservationRole(cast(str, payload["role"]))
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LangManiObservationEnvelopeV1(_Record):
    """Separated policy, verifier, privileged, restoration, and report payloads."""

    observation_id: str
    task_context_digest: str
    fields: tuple[LangManiObservationFieldV1, ...]
    policy_inputs: Mapping[str, JsonValue]
    verifier_inputs: Mapping[str, JsonValue]
    privileged_verifier_inputs: Mapping[str, JsonValue]
    restoration_state_digest: str
    reporting_metadata: Mapping[str, JsonValue]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LangManiContractError("unsupported observation-envelope schema")
        _text(self.observation_id, "observation_id")
        _digest(self.task_context_digest, "task_context_digest")
        _digest(self.restoration_state_digest, "restoration_state_digest")
        names = [field.name for field in self.fields]
        if not self.fields or len(names) != len(set(names)):
            raise LangManiContractError(
                "observation fields must be non-empty and unique"
            )
        payloads = {
            ObservationRole.POLICY_INPUT: self.policy_inputs,
            ObservationRole.VERIFIER_INPUT: self.verifier_inputs,
            ObservationRole.PRIVILEGED_VERIFIER: self.privileged_verifier_inputs,
            ObservationRole.REPORTING_ONLY: self.reporting_metadata,
        }
        roles = {field.name: field.role for field in self.fields}
        seen: set[str] = set()
        for role, payload in payloads.items():
            frozen = cast(Mapping[str, JsonValue], _freeze_json(payload, role.value))
            attribute = {
                ObservationRole.POLICY_INPUT: "policy_inputs",
                ObservationRole.VERIFIER_INPUT: "verifier_inputs",
                ObservationRole.PRIVILEGED_VERIFIER: "privileged_verifier_inputs",
                ObservationRole.REPORTING_ONLY: "reporting_metadata",
            }[role]
            object.__setattr__(self, attribute, frozen)
            for name in frozen:
                if name in seen or roles.get(name) is not role:
                    raise LangManiContractError(
                        f"observation payload role mismatch for {name!r}"
                    )
                seen.add(name)
        prohibited = {
            ObservationRole.PROHIBITED_LEARNED,
            ObservationRole.RESTORATION_ONLY,
        }
        learned_names = set(self.policy_inputs) | set(self.verifier_inputs)
        if any(roles[name] in prohibited for name in learned_names):
            raise LangManiContractError(
                "restoration/prohibited fields cannot be learned inputs"
            )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load an envelope while rejecting unknown and leaked fields."""

        payload = _exact_payload(cls, value)
        payload["fields"] = tuple(
            LangManiObservationFieldV1.from_dict(item)
            for item in cast(Sequence[object], payload["fields"])
        )
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LangManiPolicyBindingV1(_Record):
    """Content identities and temporal contract of one frozen policy."""

    policy_family: str
    variant: str
    checkpoint_fingerprint: str
    model_component_fingerprint: str
    preprocessor_fingerprint: str
    postprocessor_fingerprint: str
    producer_git_commit: str
    conditioning_id: str
    image_shape_chw: tuple[int, int, int]
    state_dimension: int
    action_dimension: int
    policy_chunk_size: int
    execution_horizon: int
    observation_steps: int
    control_frequency_hz: int
    normalization: Mapping[str, JsonValue]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LangManiContractError("unsupported policy-binding schema")
        for name in (
            "policy_family",
            "variant",
            "producer_git_commit",
            "conditioning_id",
        ):
            _text(getattr(self, name), name)
        for name in (
            "checkpoint_fingerprint",
            "model_component_fingerprint",
            "preprocessor_fingerprint",
            "postprocessor_fingerprint",
        ):
            _digest(getattr(self, name), name)
        if not re.fullmatch(r"[0-9a-f]{40}", self.producer_git_commit):
            raise LangManiContractError("producer_git_commit must be a full Git SHA")
        if self.image_shape_chw != (3, 256, 256):
            raise LangManiContractError("LangMani ACT image shape must be (3,256,256)")
        for name in (
            "state_dimension",
            "action_dimension",
            "policy_chunk_size",
            "execution_horizon",
            "observation_steps",
            "control_frequency_hz",
        ):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) < 1:
                raise LangManiContractError(f"{name} must be positive")
        if self.execution_horizon > self.policy_chunk_size:
            raise LangManiContractError(
                "execution horizon cannot exceed policy chunk size"
            )
        object.__setattr__(
            self,
            "normalization",
            cast(Mapping[str, JsonValue], _freeze_json(self.normalization)),
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load a policy binding while rejecting unknown fields."""

        payload = _exact_payload(cls, value)
        payload["image_shape_chw"] = tuple(
            cast(Sequence[int], payload["image_shape_chw"])
        )
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LangManiActionProposalV1(_Record):
    """One raw, projected, or executed action sequence with a unique identity."""

    proposal_id: str
    stage: ProposalStage
    actions: tuple[tuple[float, ...], ...]
    mask: tuple[bool, ...]
    action_dtype: str
    horizon: int
    action_dimension: int
    control_mode: str
    control_period_s: float
    action_lower_bounds: tuple[float, ...]
    action_upper_bounds: tuple[float, ...]
    normalization: str
    projection_status: tuple[str, ...]
    policy_binding_digest: str
    task_context_digest: str
    proposal_generation_seed: int
    source_proposal_digest: str | None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LangManiContractError("unsupported action-proposal schema")
        _text(self.proposal_id, "proposal_id")
        object.__setattr__(self, "stage", ProposalStage(self.stage))
        for name in ("action_dtype", "control_mode", "normalization"):
            _text(getattr(self, name), name)
        if self.horizon < 1 or self.action_dimension < 1:
            raise LangManiContractError("action horizon and dimension must be positive")
        actions = tuple(
            tuple(_finite(value, "actions") for value in action)
            for action in self.actions
        )
        object.__setattr__(self, "actions", actions)
        if len(actions) != self.horizon or any(
            len(row) != self.action_dimension for row in actions
        ):
            raise LangManiContractError(
                "actions must have shape [horizon, action_dimension]"
            )
        if len(self.mask) != self.horizon or any(
            not isinstance(item, bool) for item in self.mask
        ):
            raise LangManiContractError("mask must contain one boolean per action")
        if len(self.projection_status) != self.horizon:
            raise LangManiContractError(
                "projection_status must contain one value per action"
            )
        low = tuple(
            _finite(value, "action_lower_bounds") for value in self.action_lower_bounds
        )
        high = tuple(
            _finite(value, "action_upper_bounds") for value in self.action_upper_bounds
        )
        object.__setattr__(self, "action_lower_bounds", low)
        object.__setattr__(self, "action_upper_bounds", high)
        if len(low) != self.action_dimension or len(high) != self.action_dimension:
            raise LangManiContractError("bounds must match the action dimension")
        if any(lower > upper for lower, upper in zip(low, high, strict=True)):
            raise LangManiContractError(
                "action lower bounds cannot exceed upper bounds"
            )
        _finite(self.control_period_s, "control_period_s")
        if self.control_period_s <= 0:
            raise LangManiContractError("control_period_s must be positive")
        _digest(self.policy_binding_digest, "policy_binding_digest")
        _digest(self.task_context_digest, "task_context_digest")
        if (
            isinstance(self.proposal_generation_seed, bool)
            or self.proposal_generation_seed < 0
        ):
            raise LangManiContractError("proposal_generation_seed must be non-negative")
        if self.stage is ProposalStage.RAW and self.source_proposal_digest is not None:
            raise LangManiContractError(
                "raw proposals cannot reference another proposal"
            )
        if self.stage is not ProposalStage.RAW:
            _digest(self.source_proposal_digest, "source_proposal_digest")

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load an action proposal while rejecting unknown fields."""

        payload = _exact_payload(cls, value)
        payload["stage"] = ProposalStage(cast(str, payload["stage"]))
        payload["actions"] = tuple(
            tuple(cast(Sequence[float], row))
            for row in cast(Sequence[object], payload["actions"])
        )
        for key in (
            "mask",
            "action_lower_bounds",
            "action_upper_bounds",
            "projection_status",
        ):
            payload[key] = tuple(cast(Sequence[Any], payload[key]))
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LangManiReplaySnapshotV1(_Record):
    """References required to restore simulator and policy execution state."""

    snapshot_id: str
    task_context_digest: str
    simulator_state_digest: str
    simulator_state_schema: str
    component_count: int
    comparison_semantic: str
    maximum_absolute_tolerance: float
    restoration_mode: str
    policy_state_digest: str | None
    observation_history_digest: str | None
    wrapper_state_digest: str | None
    rng_state_digest: str | None
    renderer_state_digest: str | None
    state_payload: Mapping[str, JsonValue]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LangManiContractError("unsupported replay-snapshot schema")
        for name in (
            "snapshot_id",
            "simulator_state_schema",
            "comparison_semantic",
            "restoration_mode",
        ):
            _text(getattr(self, name), name)
        for name in ("task_context_digest", "simulator_state_digest"):
            _digest(getattr(self, name), name)
        for name in (
            "policy_state_digest",
            "observation_history_digest",
            "wrapper_state_digest",
            "rng_state_digest",
            "renderer_state_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name)
        if isinstance(self.component_count, bool) or self.component_count < 1:
            raise LangManiContractError("component_count must be positive")
        tolerance = _finite(
            self.maximum_absolute_tolerance, "maximum_absolute_tolerance"
        )
        if tolerance < 0:
            raise LangManiContractError(
                "maximum_absolute_tolerance must be non-negative"
            )
        object.__setattr__(self, "maximum_absolute_tolerance", tolerance)
        object.__setattr__(
            self,
            "state_payload",
            cast(Mapping[str, JsonValue], _freeze_json(self.state_payload)),
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load a replay snapshot while rejecting unknown fields."""

        return cls(**_exact_payload(cls, value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LangManiOutcomeEvidenceV1(_Record):
    """Conclusive or non-conclusive evidence without boolean collapse."""

    evidence_id: str
    proposal_digest: str
    replay_snapshot_digest: str
    status: OutcomeStatus
    official_success: bool | None
    official_failure: bool | None
    horizon_exhausted: bool
    unsafe_proxy: bool | None
    execution_error: str | None
    executed_action_count: int
    continuation_semantic: str
    simulator_verified: bool
    reporting_metadata: Mapping[str, JsonValue]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise LangManiContractError("unsupported outcome-evidence schema")
        _text(self.evidence_id, "evidence_id")
        _digest(self.proposal_digest, "proposal_digest")
        _digest(self.replay_snapshot_digest, "replay_snapshot_digest")
        object.__setattr__(self, "status", OutcomeStatus(self.status))
        for name in ("official_success", "official_failure", "unsafe_proxy"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise LangManiContractError(f"{name} must be boolean or null")
        if self.official_success is True and self.official_failure is True:
            raise LangManiContractError(
                "official success and failure cannot both be true"
            )
        if self.executed_action_count < 0:
            raise LangManiContractError("executed_action_count must be non-negative")
        _text(self.continuation_semantic, "continuation_semantic")
        if self.status is OutcomeStatus.EXECUTION_ERROR and not self.execution_error:
            raise LangManiContractError(
                "execution_error status requires an error description"
            )
        if (
            self.status is not OutcomeStatus.EXECUTION_ERROR
            and self.execution_error is not None
        ):
            raise LangManiContractError(
                "non-error evidence cannot carry an execution error"
            )
        object.__setattr__(
            self,
            "reporting_metadata",
            cast(Mapping[str, JsonValue], _freeze_json(self.reporting_metadata)),
        )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load outcome evidence while rejecting unknown fields."""

        payload = _exact_payload(cls, value)
        payload["status"] = OutcomeStatus(cast(str, payload["status"]))
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class LangManiIntegrationManifestV1(_Record):
    """Path-independent binding of the audited cross-project contract."""

    latentguard_commit: str
    langmani_commit: str
    integration_schema_version: str
    task_contract_digest: str
    observation_contract_digest: str
    action_contract_digest: str
    projection_contract_digest: str
    replay_contract_digest: str
    success_contract_digest: str
    compatibility_matrix_digest: str
    readiness_gate_digest: str
    overall_readiness: str
    unresolved_blocker_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or self.integration_schema_version != SCHEMA_VERSION
        ):
            raise LangManiContractError("unsupported integration-manifest schema")
        for name in ("latentguard_commit", "langmani_commit"):
            value = _text(getattr(self, name), name)
            if not re.fullmatch(r"[0-9a-f]{40}", value):
                raise LangManiContractError(f"{name} must be a full Git SHA")
        for name in (
            "task_contract_digest",
            "observation_contract_digest",
            "action_contract_digest",
            "projection_contract_digest",
            "replay_contract_digest",
            "success_contract_digest",
            "compatibility_matrix_digest",
            "readiness_gate_digest",
        ):
            _digest(getattr(self, name), name)
        if self.overall_readiness not in {
            "ready_for_m6b",
            "conditionally_ready",
            "blocked",
        }:
            raise LangManiContractError("unknown overall readiness")
        if len(self.unresolved_blocker_ids) != len(set(self.unresolved_blocker_ids)):
            raise LangManiContractError("blocker IDs must be unique")
        if self.overall_readiness == "ready_for_m6b" and self.unresolved_blocker_ids:
            raise LangManiContractError(
                "ready manifests cannot contain unresolved blockers"
            )

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Load an integration manifest while rejecting unknown fields."""

        payload = _exact_payload(cls, value)
        payload["unresolved_blocker_ids"] = tuple(
            cast(Sequence[str], payload["unresolved_blocker_ids"])
        )
        return cls(**payload)  # type: ignore[arg-type]


__all__ = [
    "LangManiActionProposalV1",
    "LangManiContractError",
    "LangManiIntegrationManifestV1",
    "LangManiObservationEnvelopeV1",
    "LangManiObservationFieldV1",
    "LangManiOutcomeEvidenceV1",
    "LangManiPolicyBindingV1",
    "LangManiReplaySnapshotV1",
    "LangManiTaskContextV1",
    "ObservationRole",
    "OutcomeStatus",
    "ProposalStage",
    "SCHEMA_VERSION",
    "canonical_json",
    "content_digest",
]
