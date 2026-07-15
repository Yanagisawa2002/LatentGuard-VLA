"""Mutation-safe, simulator-independent models for exact paired replay."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from latentguard.evaluation.models import EvaluationStatus
from latentguard.models import (
    ActionChunk,
    FailureEvent,
    JsonScalar,
    LabelSource,
    LabelStrength,
)

REPLAY_SCHEMA_VERSION = "1.0"
"""Schema version shared by the M2B-Core replay models."""

REPLAY_TRUST_CONTRACT_VERSION = "1.1"
"""Trust-contract version implemented by M2B-Core."""


class StateComparisonSemantic(StrEnum):
    """Semantic used by an adapter to compare a restored state."""

    EXACT_DIGEST = "exact_digest"
    NUMERIC_TOLERANCE = "numeric_tolerance"


class StateMatchKind(StrEnum):
    """Result of comparing restored state with a content-bound reference."""

    EXACT = "exact"
    WITHIN_TOLERANCE = "within_tolerance"
    MISMATCH = "mismatch"


class TerminalTaskStatus(StrEnum):
    """Completeness of task evidence returned by a replay session."""

    COMPLETE = "complete"
    INDETERMINATE = "indeterminate"


class ReplayTrustTier(StrEnum):
    """Explicit trust tier assigned to a replay adapter."""

    FIXTURE = "fixture"
    DETERMINISTIC_NON_SIMULATOR = "deterministic_non_simulator"
    EXACT_SIMULATOR = "exact_simulator"


class ReplayExecutionRole(StrEnum):
    """Role of one independently created replay environment session."""

    BASELINE = "baseline"
    CORRUPTED = "corrupted"


def _freeze_json(value: object) -> object:
    """Recursively detach JSON-shaped data without coercing unsupported values."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json(value)
    if isinstance(frozen, Mapping):
        return frozen
    raise TypeError("expected a mapping")


def _detach_action(action: object) -> object:
    if not isinstance(action, ActionChunk):
        return action
    return ActionChunk(
        actions=action.actions,
        coordinate_frame=action.coordinate_frame,
        control_period_s=action.control_period_s,
        schema_version=action.schema_version,
    )


@dataclass(frozen=True, slots=True)
class ReplayStateReference:
    """Opaque, content-bound reference to one restorable initial state."""

    adapter_id: str
    adapter_version: str
    source_reference_id: str
    expected_state_digest: str
    comparison_semantic: StateComparisonSemantic
    state_key: str | None = None
    state_index: int | None = None
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach metadata and validate the complete reference."""
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))
        from latentguard.replay.validation import validate_replay_state_reference

        validate_replay_state_reference(self)


@dataclass(frozen=True, slots=True)
class ReplayTaskReference:
    """Stable task identity and uninterpreted canonical task metadata."""

    task_id: str
    task_contract_version: str
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach metadata and validate the complete reference."""
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))
        from latentguard.replay.validation import validate_replay_task_reference

        validate_replay_task_reference(self)


@dataclass(frozen=True, slots=True, eq=False)
class ReplayCase:
    """One deterministic source/corruption pair and its replay references."""

    case_id: str
    proposal_id: str
    source_dataset_id: str
    source_dataset_digest: str
    corruption_dataset_digest: str
    source_episode_id: str
    source_candidate_id: str
    split_group_id: str
    original_action: ActionChunk
    transformed_action: ActionChunk
    state_reference: ReplayStateReference
    task_reference: ReplayTaskReference
    adapter_id: str
    adapter_version: str
    progress_semantic: str
    unsafe_semantic: str
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach both actions and validate the complete replay case."""
        object.__setattr__(
            self, "original_action", _detach_action(self.original_action)
        )
        object.__setattr__(
            self, "transformed_action", _detach_action(self.transformed_action)
        )
        from latentguard.replay.validation import validate_replay_case

        validate_replay_case(self)


@dataclass(frozen=True, slots=True)
class StateRestorationEvidence:
    """Scalar-only evidence that an adapter restored a referenced state."""

    expected_state_digest: str
    observed_state_digest: str
    comparison_semantic: StateComparisonSemantic
    comparison_tolerance: float
    compared_component_count: int
    maximum_absolute_error: float | None
    match_kind: StateMatchKind
    restoration_verified: bool
    diagnostics: Mapping[str, JsonScalar] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = REPLAY_SCHEMA_VERSION
    complete_state_comparison: bool = False

    def __post_init__(self) -> None:
        """Detach diagnostics and validate restoration semantics."""
        if isinstance(self.diagnostics, Mapping):
            object.__setattr__(
                self, "diagnostics", MappingProxyType(dict(self.diagnostics))
            )
        from latentguard.replay.validation import validate_state_restoration_evidence

        validate_state_restoration_evidence(self)


@dataclass(frozen=True, slots=True)
class ActionExecutionEvidence:
    """Step-count and termination evidence for one complete action chunk."""

    requested_step_count: int
    executed_step_count: int
    complete: bool
    termination_reason: str | None = None
    diagnostics: Mapping[str, JsonScalar] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach diagnostics and validate execution counters."""
        if isinstance(self.diagnostics, Mapping):
            object.__setattr__(
                self, "diagnostics", MappingProxyType(dict(self.diagnostics))
            )
        from latentguard.replay.validation import validate_action_execution_evidence

        validate_action_execution_evidence(self)


@dataclass(frozen=True, slots=True)
class TerminalTaskEvidence:
    """Complete or explicitly indeterminate task evidence from one session."""

    status: TerminalTaskStatus
    success: bool | None
    progress: float | None
    unsafe: bool | None
    failure_events: tuple[FailureEvent, ...] = ()
    termination_reason: str | None = None
    diagnostics: Mapping[str, JsonScalar] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach collections and validate task-evidence completeness."""
        events: object = self.failure_events
        if isinstance(events, Iterable) and not isinstance(events, (str, bytes)):
            object.__setattr__(self, "failure_events", tuple(events))
        if isinstance(self.diagnostics, Mapping):
            object.__setattr__(
                self, "diagnostics", MappingProxyType(dict(self.diagnostics))
            )
        from latentguard.replay.validation import validate_terminal_task_evidence

        validate_terminal_task_evidence(self)


@dataclass(frozen=True, slots=True)
class ReplayTrustDescriptor:
    """Maximum label authority and verification claims allowed for an adapter."""

    trust_tier: ReplayTrustTier
    label_source: LabelSource
    maximum_label_strength: LabelStrength
    simulator_verification_allowed: bool
    exact_state_verification_required: bool
    trust_contract_version: str = REPLAY_TRUST_CONTRACT_VERSION
    schema_version: str = REPLAY_SCHEMA_VERSION
    state_verification_semantic: StateComparisonSemantic = (
        StateComparisonSemantic.EXACT_DIGEST
    )
    state_verification_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Validate the trust boundary without granting physical authority."""
        from latentguard.replay.validation import validate_replay_trust_descriptor

        validate_replay_trust_descriptor(self)


@dataclass(frozen=True, slots=True)
class PairedReplayResult:
    """Structured result of independent baseline and corrupted replay sessions."""

    replay_case_id: str
    proposal_id: str
    status: EvaluationStatus
    termination_reason: str
    baseline_restoration: StateRestorationEvidence | None = None
    baseline_initial_task: TerminalTaskEvidence | None = None
    baseline_execution: ActionExecutionEvidence | None = None
    baseline_terminal_task: TerminalTaskEvidence | None = None
    corrupted_restoration: StateRestorationEvidence | None = None
    corrupted_initial_task: TerminalTaskEvidence | None = None
    corrupted_execution: ActionExecutionEvidence | None = None
    corrupted_terminal_task: TerminalTaskEvidence | None = None
    replayed_control_steps: int = 0
    diagnostics: Mapping[str, JsonScalar] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach diagnostics and validate status/phase consistency."""
        if isinstance(self.diagnostics, Mapping):
            object.__setattr__(
                self, "diagnostics", MappingProxyType(dict(self.diagnostics))
            )
        from latentguard.replay.validation import validate_paired_replay_result

        validate_paired_replay_result(self)


@dataclass(frozen=True, slots=True)
class ReplayBundle:
    """Versioned ordered replay cases bound to source and adapter content."""

    source_dataset_id: str
    source_dataset_digest: str
    corruption_dataset_digest: str
    adapter_id: str
    adapter_version: str
    adapter_configuration_digest: str
    replay_cases: tuple[ReplayCase, ...]
    bundle_digest: str
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: str = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach cases/metadata and validate the complete bundle."""
        object.__setattr__(self, "replay_cases", tuple(self.replay_cases))
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))
        from latentguard.replay.validation import validate_replay_bundle

        validate_replay_bundle(self)


__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "REPLAY_TRUST_CONTRACT_VERSION",
    "ActionExecutionEvidence",
    "PairedReplayResult",
    "ReplayBundle",
    "ReplayCase",
    "ReplayExecutionRole",
    "ReplayStateReference",
    "ReplayTaskReference",
    "ReplayTrustDescriptor",
    "ReplayTrustTier",
    "StateComparisonSemantic",
    "StateMatchKind",
    "StateRestorationEvidence",
    "TerminalTaskEvidence",
    "TerminalTaskStatus",
]
