"""CPU-only paired replay execution and deterministic fixture tests."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import CorruptionDataset
from latentguard.evaluation.models import EvaluationStatus
from latentguard.models import ActionChunk, LabelSource, LabelStrength
from latentguard.replay.base import ReplayEnvironmentSession
from latentguard.replay.executor import (
    PairedReplayExecutionError,
    _execute_replay_case,
    execute_paired_replay,
)
from latentguard.replay.fixture import (
    DeterministicReplayFixtureAdapter,
    create_deterministic_replay_fixture_adapter,
)
from latentguard.replay.models import (
    PairedReplayResult,
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
)
from latentguard.replay.source import ReplaySourceBinding
from latentguard.synthetic import generate_synthetic_episodes

_CORRUPTION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "corruptions" / "m1-smoke.json"
)


def _configuration(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "state_dimension": 7,
        "action_scale": 1.0,
        "progress_scale": 1.0,
        "success_tolerance": 1e-6,
        "unsafe_bound": 10.0,
        "baseline_failure_ordinals": [],
        "restoration_mismatch_ordinals": [],
        "indeterminate_ordinals": [],
        "step_exception_ordinals": [],
        "close_exception_ordinals": [],
    }
    value.update(updates)
    return value


def _stack(
    configuration: Mapping[str, object] | None = None,
) -> tuple[CorruptionDataset, ReplaySourceBinding, DeterministicReplayFixtureAdapter]:
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=8,
        action_dim=7,
        robot_state_dim=10,
        camera_count=0,
    )
    plan = load_corruption_plan(_CORRUPTION_CONFIG)
    generated = generate_corruption_proposals(
        episodes,
        plan.action_layout,
        plan.corruptions,
        base_seed=314159,
        proposal_limit=6,
    )
    dataset = CorruptionDataset(
        source_dataset_id="sha256:" + "a" * 64,
        action_layout=plan.action_layout,
        proposals=generated.proposals,
    )
    binding = ReplaySourceBinding.from_datasets(episodes, dataset)
    adapter = create_deterministic_replay_fixture_adapter(
        configuration or _configuration(), binding
    )
    return dataset, binding, adapter


def _case(adapter: DeterministicReplayFixtureAdapter, ordinal: int) -> ReplayCase:
    return adapter.replay_bundle.replay_cases[ordinal]


def _execute(
    adapter: DeterministicReplayFixtureAdapter | _AdapterWrapper,
    replay_case: ReplayCase,
) -> PairedReplayResult:
    bundle = (
        adapter.replay_bundle
        if isinstance(adapter, DeterministicReplayFixtureAdapter)
        else adapter.delegate.replay_bundle
    )
    return _execute_replay_case(adapter, replay_case, bundle)


class _SessionWrapper:
    def __init__(
        self,
        delegate: ReplayEnvironmentSession,
        owner: _AdapterWrapper,
        role: ReplayExecutionRole,
    ) -> None:
        self.delegate = delegate
        self.owner = owner
        self.role = role

    def restore_state(
        self, reference: ReplayStateReference
    ) -> StateRestorationEvidence:
        self.owner.restore_references.append((self.role, reference))
        if (
            self.owner.malformed_restoration
            and self.role is ReplayExecutionRole.BASELINE
        ):
            return object()  # type: ignore[return-value]
        evidence = self.delegate.restore_state(reference)
        if self.owner.incomplete_restoration and (
            self.role is ReplayExecutionRole.BASELINE
        ):
            return replace(evidence, complete_state_comparison=False)
        if self.owner.corrupted_restoration_mismatch and (
            self.role is ReplayExecutionRole.CORRUPTED
        ):
            return StateRestorationEvidence(
                expected_state_digest=reference.expected_state_digest,
                observed_state_digest="sha256:" + "f" * 64,
                comparison_semantic=reference.comparison_semantic,
                comparison_tolerance=0.0,
                compared_component_count=evidence.compared_component_count,
                maximum_absolute_error=1.0,
                match_kind=StateMatchKind.MISMATCH,
                restoration_verified=False,
                complete_state_comparison=False,
            )
        return evidence

    def evaluate_task(self, task: ReplayTaskReference) -> TerminalTaskEvidence:
        return self.delegate.evaluate_task(task)

    def step_action(self, action: NDArray[Any]) -> None:
        self.owner.rows.append(action)
        if self.owner.baseline_step_error and self.role is ReplayExecutionRole.BASELINE:
            raise RuntimeError("controlled baseline infrastructure failure")
        if self.owner.keyboard_interrupt and self.role is ReplayExecutionRole.BASELINE:
            raise KeyboardInterrupt
        self.delegate.step_action(action)

    def close(self) -> None:
        self.owner.close_counts[self.role] += 1
        if self.owner.close_during_interrupt and self.owner.keyboard_interrupt:
            raise RuntimeError("secondary close failure")
        self.delegate.close()


class _AdapterWrapper:
    def __init__(self, delegate: DeterministicReplayFixtureAdapter) -> None:
        self.delegate = delegate
        self.sessions: list[ReplayEnvironmentSession] = []
        self.restore_references: list[
            tuple[ReplayExecutionRole, ReplayStateReference]
        ] = []
        self.rows: list[NDArray[Any]] = []
        self.close_counts = {
            ReplayExecutionRole.BASELINE: 0,
            ReplayExecutionRole.CORRUPTED: 0,
        }
        self.same_session = False
        self.incomplete_restoration = False
        self.corrupted_restoration_mismatch = False
        self.malformed_restoration = False
        self.keyboard_interrupt = False
        self.close_during_interrupt = False
        self.baseline_step_error = False
        self.configuration_digest_override: str | None = None
        self.configuration_drift_on_create = False
        self.resolved_case_override: ReplayCase | None = None
        self.ignore_proposal_content = False
        self.trust_descriptor_override: ReplayTrustDescriptor | None = None

    @property
    def adapter_id(self) -> str:
        return self.delegate.adapter_id

    @property
    def adapter_version(self) -> str:
        return self.delegate.adapter_version

    @property
    def configuration_digest(self) -> str:
        if self.configuration_digest_override is not None:
            return self.configuration_digest_override
        if self.configuration_drift_on_create and self.sessions:
            return "cfg-sha256-" + "0" * 64
        return self.delegate.configuration_digest

    def trust_descriptor(self) -> ReplayTrustDescriptor:
        return self.trust_descriptor_override or self.delegate.trust_descriptor()

    def resolved_configuration(self) -> Mapping[str, object]:
        return self.delegate.resolved_configuration()

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        if self.resolved_case_override is not None:
            return self.resolved_case_override
        if self.ignore_proposal_content:
            return next(
                replay_case
                for replay_case in self.delegate.replay_bundle.replay_cases
                if replay_case.proposal_id == proposal.proposal_id
            )
        return self.delegate.resolve_case(proposal)

    def create_session(
        self, replay_case: ReplayCase, *, execution_role: ReplayExecutionRole
    ) -> ReplayEnvironmentSession:
        if self.same_session and self.sessions:
            return self.sessions[0]
        delegate = self.delegate.create_session(
            replay_case, execution_role=execution_role
        )
        wrapped = _SessionWrapper(delegate, self, execution_role)
        self.sessions.append(wrapped)
        return wrapped


def test_fixture_conclusive_failure_and_success_remain_conclusive() -> None:
    _, _, adapter = _stack()

    failure = _execute(adapter, _case(adapter, 0))
    success = _execute(adapter, _case(adapter, 5))

    assert failure.status is EvaluationStatus.CONCLUSIVE
    assert failure.corrupted_terminal_task is not None
    assert failure.corrupted_terminal_task.success is False
    assert failure.corrupted_terminal_task.failure_events
    assert success.status is EvaluationStatus.CONCLUSIVE
    assert success.corrupted_terminal_task is not None
    assert success.corrupted_terminal_task.success is True
    assert failure.replayed_control_steps == success.replayed_control_steps == 16


def test_baseline_failure_is_invalid_and_never_runs_corruption() -> None:
    _, _, adapter = _stack(_configuration(baseline_failure_ordinals=[0]))

    result = _execute(adapter, _case(adapter, 0))

    assert result.status is EvaluationStatus.INVALID
    assert result.termination_reason == "baseline_task_not_successful"
    assert result.baseline_terminal_task is not None
    assert result.baseline_terminal_task.success is False
    assert result.corrupted_restoration is None
    assert result.diagnostics["replay_baseline_valid"] is False


def test_restoration_mismatch_is_invalid_and_session_closes_once() -> None:
    _, _, delegate = _stack(_configuration(restoration_mismatch_ordinals=[1]))
    adapter = _AdapterWrapper(delegate)

    result = _execute(adapter, _case(delegate, 1))

    assert result.status is EvaluationStatus.INVALID
    assert result.baseline_restoration is not None
    assert result.baseline_restoration.restoration_verified is False
    assert result.baseline_restoration.complete_state_comparison is True
    assert adapter.close_counts[ReplayExecutionRole.BASELINE] == 1
    assert adapter.close_counts[ReplayExecutionRole.CORRUPTED] == 0


def test_incomplete_legacy_restoration_is_rejected_before_any_action() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.incomplete_restoration = True

    result = _execute(adapter, _case(delegate, 0))

    assert result.status is EvaluationStatus.INVALID
    assert result.termination_reason == "baseline_state_restoration_mismatch"
    assert result.baseline_restoration is not None
    assert result.baseline_restoration.restoration_verified is True
    assert result.baseline_restoration.complete_state_comparison is False
    assert adapter.rows == []
    assert adapter.close_counts[ReplayExecutionRole.BASELINE] == 1
    assert adapter.close_counts[ReplayExecutionRole.CORRUPTED] == 0


def test_trust_comparison_drift_is_rejected_before_any_action() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.trust_descriptor_override = ReplayTrustDescriptor(
        trust_tier=ReplayTrustTier.EXACT_SIMULATOR,
        label_source=LabelSource.SIMULATOR,
        maximum_label_strength=LabelStrength.STRONG,
        simulator_verification_allowed=True,
        exact_state_verification_required=True,
        state_verification_semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
        state_verification_tolerance=1e-6,
    )

    result = _execute(adapter, _case(delegate, 0))

    assert result.status is EvaluationStatus.EXECUTION_ERROR
    assert "baseline_restore_state" in result.termination_reason
    assert adapter.rows == []
    assert adapter.close_counts[ReplayExecutionRole.BASELINE] == 1
    assert adapter.close_counts[ReplayExecutionRole.CORRUPTED] == 0


def test_corrupted_restoration_mismatch_is_invalid_after_valid_baseline() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.corrupted_restoration_mismatch = True

    result = _execute(adapter, _case(delegate, 0))

    assert result.status is EvaluationStatus.INVALID
    assert result.diagnostics["replay_baseline_valid"] is True
    assert result.corrupted_restoration is not None
    assert result.corrupted_restoration.restoration_verified is False


def test_corrupted_terminal_indeterminacy_preserves_missing_values() -> None:
    _, _, adapter = _stack(_configuration(indeterminate_ordinals=[2]))

    result = _execute(adapter, _case(adapter, 2))

    assert result.status is EvaluationStatus.INDETERMINATE
    assert result.corrupted_terminal_task is not None
    assert result.corrupted_terminal_task.success is None
    assert result.corrupted_terminal_task.progress is not None
    assert result.corrupted_terminal_task.unsafe is None


@pytest.mark.parametrize(
    ("field", "ordinal", "expected_reason"),
    [
        ("step_exception_ordinals", 3, "corrupted_step_action"),
        ("close_exception_ordinals", 4, "corrupted_close"),
    ],
)
def test_fixture_runtime_and_close_failures_are_execution_errors(
    field: str, ordinal: int, expected_reason: str
) -> None:
    adapter_configuration = _configuration(**{field: [ordinal]})
    _, _, adapter = _stack(adapter_configuration)

    result = _execute(adapter, _case(adapter, ordinal))

    assert result.status is EvaluationStatus.EXECUTION_ERROR
    assert expected_reason in result.termination_reason
    assert result.diagnostics["replay_baseline_valid"] is True
    if field == "close_exception_ordinals":
        assert result.corrupted_terminal_task is not None


def test_sessions_are_distinct_restore_same_reference_and_rows_are_detached() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    replay_case = _case(delegate, 0)

    result = _execute(adapter, replay_case)

    assert result.status is EvaluationStatus.CONCLUSIVE
    assert len(adapter.sessions) == 2
    assert adapter.sessions[0] is not adapter.sessions[1]
    assert adapter.restore_references == [
        (ReplayExecutionRole.BASELINE, replay_case.state_reference),
        (ReplayExecutionRole.CORRUPTED, replay_case.state_reference),
    ]
    assert len(adapter.rows) == 16
    assert all(not row.flags.writeable for row in adapter.rows)
    assert all(
        not np.shares_memory(row, replay_case.original_action.actions)
        and not np.shares_memory(row, replay_case.transformed_action.actions)
        for row in adapter.rows
    )


def test_reused_session_object_is_an_execution_error() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.same_session = True

    result = _execute(adapter, _case(delegate, 0))

    assert result.status is EvaluationStatus.EXECUTION_ERROR
    assert "distinct" not in result.termination_reason
    assert "corrupted_create_session" in result.termination_reason
    assert adapter.close_counts[ReplayExecutionRole.BASELINE] == 1
    assert adapter.close_counts[ReplayExecutionRole.CORRUPTED] == 0


def test_public_executor_rejects_adapter_configuration_mismatch() -> None:
    dataset, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.configuration_digest_override = "cfg-sha256-" + "0" * 64

    with pytest.raises(PairedReplayExecutionError, match="configuration_digest"):
        execute_paired_replay(adapter, dataset.proposals[0], delegate.replay_bundle)

    assert adapter.sessions == []


def test_public_executor_rejects_configuration_drift_after_baseline() -> None:
    dataset, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.configuration_drift_on_create = True

    with pytest.raises(PairedReplayExecutionError, match="configuration_digest"):
        execute_paired_replay(adapter, dataset.proposals[0], delegate.replay_bundle)

    assert adapter.close_counts[ReplayExecutionRole.BASELINE] == 1
    assert adapter.close_counts[ReplayExecutionRole.CORRUPTED] == 0


@pytest.mark.parametrize("shape_mismatch", [False, True])
def test_public_executor_rejects_tampered_resolved_case_before_sessions(
    shape_mismatch: bool,
) -> None:
    dataset, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    forged = copy.copy(_case(delegate, 0))
    original = forged.original_action
    object.__setattr__(
        forged,
        "original_action",
        ActionChunk(
            actions=(
                original.actions[:, :-1]
                if shape_mismatch
                else original.actions + np.float32(0.25)
            ),
            coordinate_frame=original.coordinate_frame,
            control_period_s=original.control_period_s,
            schema_version=original.schema_version,
        ),
    )
    adapter.resolved_case_override = forged

    with pytest.raises(PairedReplayExecutionError, match="failed validation"):
        execute_paired_replay(adapter, dataset.proposals[0], delegate.replay_bundle)

    assert adapter.sessions == []


def test_public_executor_rejects_proposal_content_ignored_by_adapter() -> None:
    dataset, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.ignore_proposal_content = True
    proposal = dataset.proposals[0]
    forged = replace(
        proposal,
        transformed_action=ActionChunk(
            actions=proposal.transformed_action.actions + np.float32(0.25),
            coordinate_frame=proposal.transformed_action.coordinate_frame,
            control_period_s=proposal.transformed_action.control_period_s,
            schema_version=proposal.transformed_action.schema_version,
        ),
    )

    with pytest.raises(PairedReplayExecutionError, match="transformed action"):
        execute_paired_replay(adapter, forged, delegate.replay_bundle)

    assert adapter.sessions == []


def test_malformed_restoration_evidence_is_an_execution_error() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.malformed_restoration = True

    result = _execute(adapter, _case(delegate, 0))

    assert result.status is EvaluationStatus.EXECUTION_ERROR
    assert "baseline_restore_state" in result.termination_reason
    assert adapter.close_counts[ReplayExecutionRole.BASELINE] == 1


def test_baseline_step_exception_is_execution_error_not_task_failure() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.baseline_step_error = True

    result = _execute(adapter, _case(delegate, 0))

    assert result.status is EvaluationStatus.EXECUTION_ERROR
    assert "baseline_step_action" in result.termination_reason
    assert result.corrupted_restoration is None
    assert result.baseline_execution is not None
    assert result.baseline_execution.complete is False


def test_static_action_contract_mismatch_is_invalid_before_session_creation() -> None:
    _, _, delegate = _stack()
    replay_case = _case(delegate, 0)
    malformed = ActionChunk(
        actions=replay_case.transformed_action.actions[:, :-1],
        coordinate_frame=replay_case.transformed_action.coordinate_frame,
        control_period_s=replay_case.transformed_action.control_period_s,
    )
    object.__setattr__(replay_case, "transformed_action", malformed)
    adapter = _AdapterWrapper(delegate)

    result = _execute(adapter, replay_case)

    assert result.status is EvaluationStatus.INVALID
    assert result.termination_reason == "static_action_shape_mismatch"
    assert adapter.sessions == []


def test_keyboard_interrupt_is_not_masked_by_close_failure() -> None:
    _, _, delegate = _stack()
    adapter = _AdapterWrapper(delegate)
    adapter.keyboard_interrupt = True
    adapter.close_during_interrupt = True

    with pytest.raises(KeyboardInterrupt) as caught:
        _execute(adapter, _case(delegate, 0))

    assert caught.value.__notes__
    assert "close also failed" in caught.value.__notes__[0]
    assert adapter.close_counts[ReplayExecutionRole.BASELINE] == 1


def test_source_and_transformed_actions_remain_byte_identical() -> None:
    dataset, binding, adapter = _stack()
    replay_case = _case(adapter, 0)
    original = replay_case.original_action.actions.tobytes(order="C")
    transformed = replay_case.transformed_action.actions.tobytes(order="C")
    proposal = dataset.proposals[0]
    proposal_snapshot = proposal.transformed_action.actions.tobytes(order="C")

    _execute(adapter, replay_case)

    assert replay_case.original_action.actions.tobytes(order="C") == original
    assert replay_case.transformed_action.actions.tobytes(order="C") == transformed
    assert proposal.transformed_action.actions.tobytes(order="C") == proposal_snapshot
    binding.assert_unchanged()
