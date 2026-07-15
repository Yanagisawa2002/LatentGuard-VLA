"""Deterministic non-physical replay fixture used for CPU infrastructure tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.models import FailureEvent, LabelSource, LabelStrength
from latentguard.replay.base import ReplayEnvironmentSession, ReplayExecutionError
from latentguard.replay.identity import (
    compute_replay_bundle_digest,
    compute_replay_case_identifier,
)
from latentguard.replay.models import (
    REPLAY_SCHEMA_VERSION,
    ReplayBundle,
    ReplayCase,
    ReplayExecutionRole,
    ReplayStateReference,
    ReplayTaskReference,
    ReplayTrustDescriptor,
    ReplayTrustTier,
    StateComparisonSemantic,
    StateMatchKind,
    StateRestorationEvidence,
    TerminalTaskEvidence,
    TerminalTaskStatus,
)
from latentguard.replay.source import ReplaySourceBinding

FIXTURE_ADAPTER_ID = "deterministic_replay_fixture"
FIXTURE_ADAPTER_VERSION = "1.0.0"
FIXTURE_CONFIGURATION_SCHEMA_VERSION = "1.0"

_CONFIGURATION_FIELDS = frozenset(
    {
        "schema_version",
        "state_dimension",
        "action_scale",
        "progress_scale",
        "success_tolerance",
        "unsafe_bound",
        "baseline_failure_ordinals",
        "restoration_mismatch_ordinals",
        "indeterminate_ordinals",
        "step_exception_ordinals",
        "close_exception_ordinals",
    }
)


class FixtureReplayConfigurationError(ValueError):
    """Raised when deterministic fixture configuration is ambiguous or invalid."""


class FixtureReplayExecutionError(RuntimeError):
    """Raised for an explicitly configured synthetic infrastructure failure."""


@dataclass(frozen=True, slots=True)
class _FixtureConfigurationValues:
    state_dimension: int
    action_scale: float
    progress_scale: float
    success_tolerance: float
    unsafe_bound: float
    baseline_failure_ordinals: tuple[int, ...]
    restoration_mismatch_ordinals: tuple[int, ...]
    indeterminate_ordinals: tuple[int, ...]
    step_exception_ordinals: tuple[int, ...]
    close_exception_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FixtureReplayConfiguration:
    """Strict resolved settings for the deterministic replay fixture."""

    state_dimension: int
    action_scale: float
    progress_scale: float
    success_tolerance: float
    unsafe_bound: float
    baseline_failure_ordinals: tuple[int, ...]
    restoration_mismatch_ordinals: tuple[int, ...]
    indeterminate_ordinals: tuple[int, ...]
    step_exception_ordinals: tuple[int, ...]
    close_exception_ordinals: tuple[int, ...]
    schema_version: str = FIXTURE_CONFIGURATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate and detach direct SDK construction just like JSON resolution."""

        values = _validate_fixture_configuration_values(self.as_mapping())
        for field in (
            "state_dimension",
            "action_scale",
            "progress_scale",
            "success_tolerance",
            "unsafe_bound",
            "baseline_failure_ordinals",
            "restoration_mismatch_ordinals",
            "indeterminate_ordinals",
            "step_exception_ordinals",
            "close_exception_ordinals",
        ):
            object.__setattr__(self, field, getattr(values, field))

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical JSON-compatible resolved configuration values."""
        return MappingProxyType(
            {
                "action_scale": self.action_scale,
                "baseline_failure_ordinals": self.baseline_failure_ordinals,
                "close_exception_ordinals": self.close_exception_ordinals,
                "indeterminate_ordinals": self.indeterminate_ordinals,
                "progress_scale": self.progress_scale,
                "restoration_mismatch_ordinals": self.restoration_mismatch_ordinals,
                "schema_version": self.schema_version,
                "state_dimension": self.state_dimension,
                "step_exception_ordinals": self.step_exception_ordinals,
                "success_tolerance": self.success_tolerance,
                "unsafe_bound": self.unsafe_bound,
            }
        )


def _finite_number(
    configuration: Mapping[str, object],
    field: str,
    *,
    allow_zero: bool,
) -> float:
    value = configuration[field]
    if type(value) not in (int, float):
        raise FixtureReplayConfigurationError(
            f"FixtureReplayConfiguration.{field}: expected a finite number"
        )
    try:
        number = float(cast(int | float, value))
    except OverflowError as exc:
        raise FixtureReplayConfigurationError(
            f"FixtureReplayConfiguration.{field}: number is outside the finite range"
        ) from exc
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        relation = "non-negative" if allow_zero else "positive"
        raise FixtureReplayConfigurationError(
            f"FixtureReplayConfiguration.{field}: expected a finite {relation} number"
        )
    return number


def _ordinal_selector(
    configuration: Mapping[str, object], field: str
) -> tuple[int, ...]:
    raw = configuration[field]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise FixtureReplayConfigurationError(
            f"FixtureReplayConfiguration.{field}: expected an array of ordinals"
        )
    values = tuple(raw)
    if any(type(value) is not int or value < 0 for value in values):
        raise FixtureReplayConfigurationError(
            f"FixtureReplayConfiguration.{field}: ordinals must be "
            "non-negative integers"
        )
    typed = cast(tuple[int, ...], values)
    if typed != tuple(sorted(typed)):
        raise FixtureReplayConfigurationError(
            f"FixtureReplayConfiguration.{field}: ordinals must be sorted"
        )
    if len(set(typed)) != len(typed):
        raise FixtureReplayConfigurationError(
            f"FixtureReplayConfiguration.{field}: duplicate ordinals are unsupported"
        )
    return typed


def _validate_fixture_configuration_values(
    configuration: Mapping[str, object],
) -> _FixtureConfigurationValues:
    """Validate exact fixture fields and return detached canonical values."""
    if not isinstance(configuration, Mapping):
        raise FixtureReplayConfigurationError(
            "FixtureReplayConfiguration: expected a JSON object"
        )
    if not all(isinstance(key, str) for key in configuration):
        raise FixtureReplayConfigurationError(
            "FixtureReplayConfiguration: object keys must be strings"
        )
    actual = set(configuration)
    missing = sorted(_CONFIGURATION_FIELDS - actual)
    unexpected = sorted(actual - _CONFIGURATION_FIELDS)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise FixtureReplayConfigurationError(
            "FixtureReplayConfiguration: invalid fields (" + "; ".join(details) + ")"
        )
    if configuration["schema_version"] != FIXTURE_CONFIGURATION_SCHEMA_VERSION:
        raise FixtureReplayConfigurationError(
            "FixtureReplayConfiguration.schema_version: unsupported version "
            f"{configuration['schema_version']!r}"
        )
    dimension = configuration["state_dimension"]
    if type(dimension) is not int or dimension <= 0:
        raise FixtureReplayConfigurationError(
            "FixtureReplayConfiguration.state_dimension: expected a positive integer"
        )
    selectors = {
        field: _ordinal_selector(configuration, field)
        for field in (
            "baseline_failure_ordinals",
            "restoration_mismatch_ordinals",
            "indeterminate_ordinals",
            "step_exception_ordinals",
            "close_exception_ordinals",
        )
    }
    owners: dict[int, str] = {}
    for field, ordinals in selectors.items():
        for ordinal in ordinals:
            prior = owners.get(ordinal)
            if prior is not None:
                raise FixtureReplayConfigurationError(
                    "FixtureReplayConfiguration: selector ordinal "
                    f"{ordinal} appears in both {prior} and {field}"
                )
            owners[ordinal] = field
    return _FixtureConfigurationValues(
        state_dimension=dimension,
        action_scale=_finite_number(configuration, "action_scale", allow_zero=False),
        progress_scale=_finite_number(
            configuration, "progress_scale", allow_zero=False
        ),
        success_tolerance=_finite_number(
            configuration, "success_tolerance", allow_zero=True
        ),
        unsafe_bound=_finite_number(configuration, "unsafe_bound", allow_zero=False),
        baseline_failure_ordinals=selectors["baseline_failure_ordinals"],
        restoration_mismatch_ordinals=selectors["restoration_mismatch_ordinals"],
        indeterminate_ordinals=selectors["indeterminate_ordinals"],
        step_exception_ordinals=selectors["step_exception_ordinals"],
        close_exception_ordinals=selectors["close_exception_ordinals"],
    )


def resolve_fixture_configuration(
    configuration: Mapping[str, object],
) -> FixtureReplayConfiguration:
    """Validate exact fixture fields without changing their configured meaning."""

    values = _validate_fixture_configuration_values(configuration)
    return FixtureReplayConfiguration(
        state_dimension=values.state_dimension,
        action_scale=values.action_scale,
        progress_scale=values.progress_scale,
        success_tolerance=values.success_tolerance,
        unsafe_bound=values.unsafe_bound,
        baseline_failure_ordinals=values.baseline_failure_ordinals,
        restoration_mismatch_ordinals=values.restoration_mismatch_ordinals,
        indeterminate_ordinals=values.indeterminate_ordinals,
        step_exception_ordinals=values.step_exception_ordinals,
        close_exception_ordinals=values.close_exception_ordinals,
    )


def _fixture_state_digest(state: NDArray[Any]) -> str:
    payload = {
        "content_sha256": hashlib.sha256(state.tobytes(order="C")).hexdigest(),
        "dtype": state.dtype.str,
        "shape": list(state.shape),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _fixture_target(
    replay_case: ReplayCase, configuration: FixtureReplayConfiguration
) -> NDArray[np.float64]:
    target = np.zeros(configuration.state_dimension, dtype=np.float64)
    for row in replay_case.original_action.actions:
        target += np.asarray(row, dtype=np.float64) * configuration.action_scale
    return target


class DeterministicReplayFixtureSession:
    """One isolated numeric fixture session with no physical interpretation."""

    def __init__(
        self,
        replay_case: ReplayCase,
        configuration: FixtureReplayConfiguration,
        execution_role: ReplayExecutionRole,
    ) -> None:
        """Create detached state for exactly one baseline or corrupted execution."""
        self._case = replay_case
        self._configuration = configuration
        self._role = execution_role
        self._state = np.zeros(configuration.state_dimension, dtype=np.float64)
        self._target = _fixture_target(replay_case, configuration)
        self._restored = False
        self._closed = False
        self._executed_rows = 0

    @property
    def state_snapshot(self) -> NDArray[np.float64]:
        """Return a detached state copy for infrastructure-only independence tests."""
        return np.array(self._state, copy=True)

    def restore_state(
        self, reference: ReplayStateReference
    ) -> StateRestorationEvidence:
        """Restore deterministic zero state and return scalar-only comparison proof."""
        self._require_open()
        if reference != self._case.state_reference:
            raise FixtureReplayExecutionError(
                "fixture session received a state reference from another replay case"
            )
        self._state = np.zeros(self._configuration.state_dimension, dtype=np.float64)
        mismatch = (
            self._role is ReplayExecutionRole.BASELINE
            and self._case_generation_ordinal()
            in self._configuration.restoration_mismatch_ordinals
        )
        maximum_error = 0.0
        if mismatch:
            self._state[0] = 1.0
            maximum_error = 1.0
        observed_digest = _fixture_state_digest(self._state)
        self._restored = not mismatch
        return StateRestorationEvidence(
            expected_state_digest=reference.expected_state_digest,
            observed_state_digest=observed_digest,
            comparison_semantic=reference.comparison_semantic,
            comparison_tolerance=0.0,
            compared_component_count=self._configuration.state_dimension,
            maximum_absolute_error=maximum_error,
            match_kind=(StateMatchKind.MISMATCH if mismatch else StateMatchKind.EXACT),
            restoration_verified=not mismatch,
            complete_state_comparison=True,
            diagnostics={"fixture_non_physical": True},
        )

    def evaluate_task(self, task: ReplayTaskReference) -> TerminalTaskEvidence:
        """Evaluate synthetic distance-to-target progress without inferring values."""
        self._require_ready()
        if task != self._case.task_reference:
            raise FixtureReplayExecutionError(
                "fixture session received a task reference from another replay case"
            )
        expected_steps = self._case.original_action.actions.shape[0]
        if self._executed_rows not in {0, expected_steps}:
            raise FixtureReplayExecutionError(
                "fixture task evaluation is only valid before or after a full action"
            )
        distance = float(np.linalg.norm(self._state - self._target))
        progress = max(
            0.0,
            min(1.0, 1.0 - distance / self._configuration.progress_scale),
        )
        unsafe = bool(np.max(np.abs(self._state)) > self._configuration.unsafe_bound)
        ordinal = self._case_generation_ordinal()
        terminal = self._executed_rows == expected_steps
        if (
            terminal
            and self._role is ReplayExecutionRole.CORRUPTED
            and ordinal in self._configuration.indeterminate_ordinals
        ):
            return TerminalTaskEvidence(
                status=TerminalTaskStatus.INDETERMINATE,
                success=None,
                progress=progress,
                unsafe=None,
                failure_events=(),
                termination_reason="fixture_terminal_indeterminate",
                diagnostics={"fixture_non_physical": True},
            )
        forced_baseline_failure = bool(
            terminal
            and self._role is ReplayExecutionRole.BASELINE
            and ordinal in self._configuration.baseline_failure_ordinals
        )
        success = bool(
            not forced_baseline_failure
            and distance <= self._configuration.success_tolerance
        )
        failure_events = (
            ()
            if success
            else (
                FailureEvent(
                    failure_type="synthetic_fixture_target_miss",
                    description=(
                        "Non-physical deterministic fixture target was not reached."
                    ),
                ),
            )
        )
        return TerminalTaskEvidence(
            status=TerminalTaskStatus.COMPLETE,
            success=success,
            progress=progress,
            unsafe=unsafe,
            failure_events=failure_events,
            termination_reason="fixture_task_complete",
            diagnostics={
                "fixture_non_physical": True,
                "synthetic_distance": distance,
            },
        )

    def step_action(self, action: NDArray[Any]) -> None:
        """Apply one exact validated action row to the synthetic numeric state."""
        self._require_ready()
        if (
            self._role is ReplayExecutionRole.CORRUPTED
            and self._case_generation_ordinal()
            in self._configuration.step_exception_ordinals
        ):
            raise FixtureReplayExecutionError(
                "controlled deterministic fixture step exception"
            )
        if not isinstance(action, np.ndarray) or action.dtype.hasobject:
            raise FixtureReplayExecutionError(
                "fixture action row must be a non-object NumPy array"
            )
        if action.shape != (self._configuration.state_dimension,):
            raise FixtureReplayExecutionError(
                "fixture action row does not match configured state dimension"
            )
        if not np.issubdtype(action.dtype, np.number) or not np.all(
            np.isfinite(action)
        ):
            raise FixtureReplayExecutionError(
                "fixture action row must contain finite numeric values"
            )
        expected_action = (
            self._case.original_action
            if self._role is ReplayExecutionRole.BASELINE
            else self._case.transformed_action
        )
        if self._executed_rows >= expected_action.actions.shape[0]:
            raise FixtureReplayExecutionError(
                "fixture received more steps than the action horizon"
            )
        expected_row = expected_action.actions[self._executed_rows]
        if not np.array_equal(action, expected_row):
            raise FixtureReplayExecutionError(
                "fixture received an action row outside the replay case contract"
            )
        self._state += (
            np.asarray(action, dtype=np.float64) * self._configuration.action_scale
        )
        self._executed_rows += 1

    def close(self) -> None:
        """Close this independent session, optionally raising a controlled error."""
        if self._closed:
            raise FixtureReplayExecutionError(
                "fixture session was closed more than once"
            )
        self._closed = True
        if (
            self._role is ReplayExecutionRole.CORRUPTED
            and self._case_generation_ordinal()
            in self._configuration.close_exception_ordinals
        ):
            raise FixtureReplayExecutionError(
                "controlled deterministic fixture close exception"
            )

    def _case_generation_ordinal(self) -> int:
        value = self._case.state_reference.metadata.get("generation_ordinal")
        if type(value) is not int or value < 0:
            raise FixtureReplayExecutionError(
                "fixture replay case is missing a valid generation ordinal"
            )
        return value

    def _require_open(self) -> None:
        if self._closed:
            raise FixtureReplayExecutionError("fixture session is already closed")

    def _require_ready(self) -> None:
        self._require_open()
        if not self._restored:
            raise FixtureReplayExecutionError(
                "fixture state must be successfully restored before use"
            )


class DeterministicReplayFixtureAdapter:
    """Explicit weak non-simulator adapter for paired replay infrastructure tests."""

    def __init__(
        self,
        source_binding: ReplaySourceBinding,
        configuration: FixtureReplayConfiguration,
    ) -> None:
        """Resolve all cases and bind their identities before any session exists."""
        if not isinstance(configuration, FixtureReplayConfiguration):
            raise FixtureReplayConfigurationError(
                "DeterministicReplayFixtureAdapter.configuration: expected "
                "FixtureReplayConfiguration"
            )
        configuration = resolve_fixture_configuration(configuration.as_mapping())
        proposal_count = source_binding.proposal_count
        for field in (
            "baseline_failure_ordinals",
            "restoration_mismatch_ordinals",
            "indeterminate_ordinals",
            "step_exception_ordinals",
            "close_exception_ordinals",
        ):
            invalid = tuple(
                ordinal
                for ordinal in getattr(configuration, field)
                if ordinal >= proposal_count
            )
            if invalid:
                raise FixtureReplayConfigurationError(
                    f"FixtureReplayConfiguration.{field}: proposal ordinals "
                    f"{invalid!r} are outside the bound dataset range "
                    f"[0, {proposal_count})"
                )
        self._source_binding = source_binding
        self._configuration = configuration
        self._resolved_configuration = configuration.as_mapping()
        self._configuration_digest = compute_configuration_digest(
            self._resolved_configuration
        )
        cases = tuple(
            self._case_from_proposal(proposal)
            for proposal in source_binding.corruption_dataset.proposals
        )
        self._case_by_proposal = MappingProxyType(
            {case.proposal_id: case for case in cases}
        )
        bundle_metadata = MappingProxyType(
            {
                "fixture_non_physical": True,
                "research_claims_allowed": False,
            }
        )
        bundle_digest = compute_replay_bundle_digest(
            source_dataset_id=source_binding.source_dataset_id,
            source_dataset_digest=source_binding.source_dataset_digest,
            corruption_dataset_digest=source_binding.corruption_dataset_digest,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_configuration_digest=self.configuration_digest,
            replay_cases=cases,
            metadata=bundle_metadata,
        )
        self._bundle = ReplayBundle(
            source_dataset_id=source_binding.source_dataset_id,
            source_dataset_digest=source_binding.source_dataset_digest,
            corruption_dataset_digest=source_binding.corruption_dataset_digest,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_configuration_digest=self.configuration_digest,
            replay_cases=cases,
            bundle_digest=bundle_digest,
            metadata=bundle_metadata,
        )
        self._trust = ReplayTrustDescriptor(
            trust_tier=ReplayTrustTier.FIXTURE,
            label_source=LabelSource.DETERMINISTIC_EVALUATOR,
            maximum_label_strength=LabelStrength.WEAK,
            simulator_verification_allowed=False,
            exact_state_verification_required=True,
            state_verification_semantic=StateComparisonSemantic.EXACT_DIGEST,
            state_verification_tolerance=0.0,
        )

    @property
    def adapter_id(self) -> str:
        """Return the stable built-in fixture registry identifier."""
        return FIXTURE_ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        """Return the fixture adapter semantic version."""
        return FIXTURE_ADAPTER_VERSION

    @property
    def configuration_digest(self) -> str:
        """Return the digest of strict resolved fixture settings."""
        return self._configuration_digest

    @property
    def replay_bundle(self) -> ReplayBundle:
        """Return the in-memory content-bound bundle without duplicated arrays."""
        return self._bundle

    def trust_descriptor(self) -> ReplayTrustDescriptor:
        """Return the fixed weak non-simulator trust ceiling."""
        return self._trust

    def resolved_configuration(self) -> Mapping[str, object]:
        """Return canonical fixture settings without paths or runtime metadata."""
        return self._resolved_configuration

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        """Resolve and revalidate one bound proposal without creating a session."""
        pair = self._source_binding.resolve(proposal)
        try:
            replay_case = self._case_by_proposal[pair.proposal_id]
        except KeyError as exc:
            raise FixtureReplayExecutionError(
                f"fixture has no replay case for proposal {pair.proposal_id!r}"
            ) from exc
        return replay_case

    def create_session(
        self,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
    ) -> ReplayEnvironmentSession:
        """Create a fresh state-owning session for one execution role."""
        if not isinstance(execution_role, ReplayExecutionRole):
            raise ReplayExecutionError("unsupported replay execution role")
        bound = self._case_by_proposal.get(replay_case.proposal_id)
        if bound is not replay_case and (
            bound is None or bound.case_id != replay_case.case_id
        ):
            raise ReplayExecutionError(
                "replay case is not bound to this fixture adapter"
            )
        return DeterministicReplayFixtureSession(
            replay_case, self._configuration, execution_role
        )

    def _case_from_proposal(self, proposal: CorruptedActionProposal) -> ReplayCase:
        pair = self._source_binding.resolve(proposal)
        action_dimension = pair.original_action.actions.shape[1]
        if action_dimension != self._configuration.state_dimension:
            raise FixtureReplayConfigurationError(
                "FixtureReplayConfiguration.state_dimension: must match the bound "
                f"action dimension {action_dimension}"
            )
        initial_state = np.zeros(self._configuration.state_dimension, dtype=np.float64)
        state_reference = ReplayStateReference(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            source_reference_id=pair.source_episode_id,
            expected_state_digest=_fixture_state_digest(initial_state),
            comparison_semantic=StateComparisonSemantic.EXACT_DIGEST,
            state_index=0,
            metadata={
                "fixture_non_physical": True,
                "generation_ordinal": pair.generation_ordinal,
            },
        )
        task_reference = ReplayTaskReference(
            task_id=pair.source_task_id,
            task_contract_version="1.0.0",
            metadata={"fixture_non_physical": True},
        )
        progress_semantic = "fixture_normalized_distance_to_original_target_v1"
        unsafe_semantic = "fixture_state_linf_bound_v1"
        case_id = compute_replay_case_identifier(
            proposal_id=pair.proposal_id,
            source_dataset_id=pair.source_dataset_id,
            source_dataset_digest=pair.source_dataset_digest,
            corruption_dataset_digest=pair.corruption_dataset_digest,
            source_episode_id=pair.source_episode_id,
            source_candidate_id=pair.source_candidate_id,
            split_group_id=pair.split_group_id,
            original_action=pair.original_action,
            transformed_action=pair.transformed_action,
            state_reference=state_reference,
            task_reference=task_reference,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            progress_semantic=progress_semantic,
            unsafe_semantic=unsafe_semantic,
        )
        return ReplayCase(
            case_id=case_id,
            proposal_id=pair.proposal_id,
            source_dataset_id=pair.source_dataset_id,
            source_dataset_digest=pair.source_dataset_digest,
            corruption_dataset_digest=pair.corruption_dataset_digest,
            source_episode_id=pair.source_episode_id,
            source_candidate_id=pair.source_candidate_id,
            split_group_id=pair.split_group_id,
            original_action=pair.original_action,
            transformed_action=pair.transformed_action,
            state_reference=state_reference,
            task_reference=task_reference,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            progress_semantic=progress_semantic,
            unsafe_semantic=unsafe_semantic,
            schema_version=REPLAY_SCHEMA_VERSION,
        )


def create_deterministic_replay_fixture_adapter(
    configuration: Mapping[str, object], source_binding: ReplaySourceBinding
) -> DeterministicReplayFixtureAdapter:
    """Create the sole built-in adapter from strict synthetic settings and data."""
    return DeterministicReplayFixtureAdapter(
        source_binding, resolve_fixture_configuration(configuration)
    )


__all__ = [
    "FIXTURE_ADAPTER_ID",
    "FIXTURE_ADAPTER_VERSION",
    "FIXTURE_CONFIGURATION_SCHEMA_VERSION",
    "FixtureReplayConfiguration",
    "FixtureReplayConfigurationError",
    "FixtureReplayExecutionError",
    "DeterministicReplayFixtureAdapter",
    "DeterministicReplayFixtureSession",
    "create_deterministic_replay_fixture_adapter",
    "resolve_fixture_configuration",
]
