"""CPU-only fake-runtime tests for the ManiSkill PickCube replay adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from latentguard.corruptions.models import (
    CorruptedActionProposal,
    compute_proposal_identifier,
)
from latentguard.evaluation.models import (
    EvaluationStatus,
    compute_configuration_digest,
)
from latentguard.integrations.maniskill_pickcube.adapter import (
    MANISKILL_PICKCUBE_ADAPTER_ID,
    MANISKILL_PICKCUBE_ADAPTER_VERSION,
    ManiSkillPickCubeAdapter,
    ManiSkillPickCubeAdapterConfigurationError,
    ManiSkillPickCubeSemanticIdentity,
    TrustedManiSkillRuntimeAttestation,
    resolve_maniskill_pickcube_configuration,
)
from latentguard.integrations.maniskill_pickcube.archive import (
    ManiSkillReferenceArchive,
    ManiSkillReferenceEpisode,
    save_reference_archive,
)
from latentguard.integrations.maniskill_pickcube.session import (
    ArchivePickCubeReferenceStateStore,
    LoadedReferenceState,
    ManiSkillPickCubeEnvironmentSettings,
    ManiSkillPickCubeReplaySession,
    ManiSkillPickCubeSessionError,
    ManiSkillPickCubeSessionFactory,
    PickCubeReplayActionContract,
    StateTreeComparison,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_TASK_CONTRACT_VERSION,
    PICKCUBE_TASK_ID,
    PICKCUBE_UNSAFE_SEMANTIC,
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
    build_pickcube_task_evidence,
)
from latentguard.models import ActionChunk, LabelSource, LabelStrength
from latentguard.replay.base import ReplayInvalidContextError
from latentguard.replay.evaluator import create_exact_state_paired_replay_evaluator
from latentguard.replay.identity import (
    compute_replay_bundle_digest,
    compute_replay_case_identifier,
)
from latentguard.replay.models import (
    ReplayBundle,
    ReplayCase,
    ReplayExecutionRole,
    ReplayStateReference,
    ReplayTaskReference,
    StateComparisonSemantic,
    StateMatchKind,
    TerminalTaskStatus,
)

_SOURCE_DIGEST = "sha256:" + "1" * 64
_CORRUPTION_DIGEST = "sha256:" + "2" * 64
_STATE_DIGEST = "sha256:" + "3" * 64


def _task_keys() -> PickCubeTaskKeyContract:
    return PickCubeTaskKeyContract(
        success="success",
        object_placed="is_obj_placed",
        robot_static="is_robot_static",
        grasped="is_grasped",
    )


def _action_contract() -> PickCubeReplayActionContract:
    return PickCubeReplayActionContract(
        total_dimension=2,
        environment_numpy_dtype=np.dtype(np.float32).str,
        lower_bounds=np.array([-1.0, -1.0], dtype=np.float32),
        upper_bounds=np.array([1.0, 1.0], dtype=np.float32),
        environment_shape=(1, 2),
        coordinate_frame="panda_pd_joint_pos_v1",
        control_period_s=0.02,
    )


def _settings() -> ManiSkillPickCubeEnvironmentSettings:
    return ManiSkillPickCubeEnvironmentSettings(
        obs_mode="none",
        state_tolerance=1e-6,
    )


def _proposal_and_case(
    *, transformed: np.ndarray | None = None
) -> tuple[CorruptedActionProposal, ReplayCase]:
    original_array = np.array([[0.2, 0.1], [0.3, -0.1]], dtype=np.float32)
    transformed_array = (
        np.zeros_like(original_array) if transformed is None else transformed
    )
    original = ActionChunk(
        actions=original_array,
        coordinate_frame="panda_pd_joint_pos_v1",
        control_period_s=0.02,
    )
    changed = ActionChunk(
        actions=transformed_array,
        coordinate_frame="panda_pd_joint_pos_v1",
        control_period_s=0.02,
    )
    proposal_id = compute_proposal_identifier(
        source_episode_id="episode-0",
        source_candidate_id="candidate-0",
        corruption_name="action_hold",
        resolved_parameters={"hold_steps": 2},
        seed=7,
        generation_ordinal=0,
    )
    proposal = CorruptedActionProposal(
        proposal_id=proposal_id,
        source_episode_id="episode-0",
        source_candidate_id="candidate-0",
        source_policy_id="official-maniskill-solver",
        source_task_id=PICKCUBE_TASK_ID,
        split_group_id="split-group-0",
        transformed_action=changed,
        corruption_type="action_hold",
        resolved_parameters={"hold_steps": 2},
        seed=7,
        generation_ordinal=0,
    )
    state_reference = ReplayStateReference(
        adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
        adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
        source_reference_id="episode-0",
        expected_state_digest=_STATE_DIGEST,
        comparison_semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
        state_key="initial",
        metadata={"state_semantic": "maniskill_state_tree_v1"},
    )
    task_reference = ReplayTaskReference(
        task_id=PICKCUBE_TASK_ID,
        task_contract_version=PICKCUBE_TASK_CONTRACT_VERSION,
    )
    case_id = compute_replay_case_identifier(
        proposal_id=proposal_id,
        source_dataset_id="source-dataset",
        source_dataset_digest=_SOURCE_DIGEST,
        corruption_dataset_digest=_CORRUPTION_DIGEST,
        source_episode_id="episode-0",
        source_candidate_id="candidate-0",
        split_group_id="split-group-0",
        original_action=original,
        transformed_action=changed,
        state_reference=state_reference,
        task_reference=task_reference,
        adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
        adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
        progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
        unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
    )
    replay_case = ReplayCase(
        case_id=case_id,
        proposal_id=proposal_id,
        source_dataset_id="source-dataset",
        source_dataset_digest=_SOURCE_DIGEST,
        corruption_dataset_digest=_CORRUPTION_DIGEST,
        source_episode_id="episode-0",
        source_candidate_id="candidate-0",
        split_group_id="split-group-0",
        original_action=original,
        transformed_action=changed,
        state_reference=state_reference,
        task_reference=task_reference,
        adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
        adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
        progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
        unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
    )
    return proposal, replay_case


@dataclass
class _Environment:
    role: ReplayExecutionRole
    state: object | None = None
    steps: int = 0
    closed: bool = False


class _FakeRuntime:
    def __init__(
        self,
        *,
        mutate_action: bool = False,
        baseline_succeeds: bool = True,
        corrupted_succeeds: bool = False,
        corrupted_step_exception: bool = False,
    ) -> None:
        self.environments: list[_Environment] = []
        self.mutate_action = mutate_action
        self.baseline_succeeds = baseline_succeeds
        self.corrupted_succeeds = corrupted_succeeds
        self.corrupted_step_exception = corrupted_step_exception

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        execution_role: ReplayExecutionRole,
    ) -> object:
        assert settings.sim_backend == "gpu"
        assert action_contract.environment_shape == (1, 2)
        environment = _Environment(execution_role)
        self.environments.append(environment)
        return environment

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        assert isinstance(environment, _Environment)
        return state_tree

    def set_state_dict(self, environment: object, state_tree: object) -> None:
        assert isinstance(environment, _Environment)
        environment.state = state_tree

    def get_state_dict(self, environment: object) -> object:
        assert isinstance(environment, _Environment)
        return environment.state

    def capture_task_snapshot(
        self,
        environment: object,
        key_contract: PickCubeTaskKeyContract,
    ) -> RawPickCubeTaskSnapshot:
        assert isinstance(environment, _Environment)
        terminal = environment.steps == 2
        success = bool(
            terminal
            and (
                (
                    environment.role is ReplayExecutionRole.BASELINE
                    and self.baseline_succeeds
                )
                or (
                    environment.role is ReplayExecutionRole.CORRUPTED
                    and self.corrupted_succeeds
                )
            )
        )
        return RawPickCubeTaskSnapshot(
            evaluator_values={
                key_contract.success: np.array([success], dtype=np.bool_),
                key_contract.object_placed: np.array([success], dtype=np.bool_),
                key_contract.robot_static: np.array([terminal], dtype=np.bool_),
                key_contract.grasped: np.array([False], dtype=np.bool_),
            },
            cube_center_z=np.array([0.05], dtype=np.float32),
            cube_to_goal_distance=np.array([0.0 if success else 0.4], dtype=np.float32),
        )

    def step_action(
        self,
        environment: object,
        action: np.ndarray,
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        assert isinstance(environment, _Environment)
        assert action.shape == action_contract.row_shape
        if (
            self.corrupted_step_exception
            and environment.role is ReplayExecutionRole.CORRUPTED
        ):
            raise RuntimeError("controlled fake simulator exception")
        environment.steps += 1
        if self.mutate_action:
            action[0] = -0.75

    def close_environment(self, environment: object) -> None:
        assert isinstance(environment, _Environment)
        environment.closed = True


class _StateLoader:
    def __init__(self) -> None:
        self.references: list[ReplayStateReference] = []

    def load_reference_state(
        self, reference: ReplayStateReference
    ) -> LoadedReferenceState:
        self.references.append(reference)
        assert reference.state_key is not None
        return LoadedReferenceState(
            source_reference_id=reference.source_reference_id,
            state_key=reference.state_key,
            state_digest=_STATE_DIGEST,
            state_tree={"actors": np.array([1.0, 2.0], dtype=np.float32)},
            compared_component_count=1,
        )


class _Comparator:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.mismatch = mismatch

    def compare_state_trees(
        self,
        expected: object,
        observed: object,
        *,
        tolerance: float,
    ) -> StateTreeComparison:
        assert expected is observed
        if self.mismatch:
            return StateTreeComparison(
                expected_digest=_STATE_DIGEST,
                observed_digest="sha256:" + "4" * 64,
                structure_matches=True,
                compared_component_count=1,
                maximum_absolute_error=tolerance + 0.1,
            )
        return StateTreeComparison(
            expected_digest=_STATE_DIGEST,
            observed_digest=_STATE_DIGEST,
            structure_matches=True,
            compared_component_count=1,
            maximum_absolute_error=0.0,
        )


class _CaseProvider:
    def __init__(self, replay_case: ReplayCase) -> None:
        self.replay_case = replay_case

    @property
    def provider_id(self) -> str:
        return "fake-content-bound-provider"

    @property
    def provider_version(self) -> str:
        return "1.0.0"

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        assert proposal.proposal_id == self.replay_case.proposal_id
        return self.replay_case


class _ReferenceValidator:
    def __init__(self) -> None:
        self.references: list[ReplayStateReference] = []

    def validate_reference(self, reference: ReplayStateReference) -> None:
        assert reference.expected_state_digest == _STATE_DIGEST
        self.references.append(reference)


def _adapter_fixture() -> tuple[
    CorruptedActionProposal,
    ReplayCase,
    ReplayBundle,
    ManiSkillPickCubeAdapter,
    _FakeRuntime,
    _StateLoader,
]:
    return _adapter_fixture_with_runtime(_FakeRuntime())


def _adapter_fixture_with_runtime(
    runtime: _FakeRuntime,
) -> tuple[
    CorruptedActionProposal,
    ReplayCase,
    ReplayBundle,
    ManiSkillPickCubeAdapter,
    _FakeRuntime,
    _StateLoader,
]:
    proposal, replay_case = _proposal_and_case()
    identity = ManiSkillPickCubeSemanticIdentity(
        compatibility_identity="compat-sha256-" + "5" * 64,
        solver_identity="solver-sha256-" + "6" * 64,
        task_implementation_identity="task-sha256-" + "7" * 64,
        controller_configuration_identity="controller-sha256-" + "8" * 64,
        action_layout_digest="sha256:" + "9" * 64,
    )
    settings = _settings()
    action_contract = _action_contract()
    keys = _task_keys()
    resolved = resolve_maniskill_pickcube_configuration(
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_keys=keys,
    )
    configuration_digest = compute_configuration_digest(resolved)
    metadata = MappingProxyType({"real_simulator_reference": True})
    bundle_digest = compute_replay_bundle_digest(
        source_dataset_id="source-dataset",
        source_dataset_digest=_SOURCE_DIGEST,
        corruption_dataset_digest=_CORRUPTION_DIGEST,
        adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
        adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
        adapter_configuration_digest=configuration_digest,
        replay_cases=(replay_case,),
        metadata=metadata,
    )
    bundle = ReplayBundle(
        source_dataset_id="source-dataset",
        source_dataset_digest=_SOURCE_DIGEST,
        corruption_dataset_digest=_CORRUPTION_DIGEST,
        adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
        adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
        adapter_configuration_digest=configuration_digest,
        replay_cases=(replay_case,),
        bundle_digest=bundle_digest,
        metadata=metadata,
    )
    state_loader = _StateLoader()
    session_factory = ManiSkillPickCubeSessionFactory(
        settings=settings,
        action_contract=action_contract,
        task_key_contract=keys,
        state_loader=state_loader,
        state_comparator=_Comparator(),
        runtime=runtime,
    )
    adapter = ManiSkillPickCubeAdapter(
        case_provider=_CaseProvider(replay_case),
        replay_bundle=bundle,
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_key_contract=keys,
        reference_validator=_ReferenceValidator(),
        session_factory=session_factory,
        trust_attestation=TrustedManiSkillRuntimeAttestation(
            compatibility_identity=identity.compatibility_identity,
            semantic_configuration_digest=configuration_digest,
            compatibility_probe_passed=True,
            dependency_versions_verified=True,
            action_contract_verified=True,
            state_api_verified=True,
            state_round_trip_verified=True,
            task_contract_verified=True,
            solver_source_verified=True,
            task_source_verified=True,
            real_simulator_runtime=True,
        ),
    )
    return proposal, replay_case, bundle, adapter, runtime, state_loader


def test_task_evidence_binary_progress_and_narrow_unsafe_events() -> None:
    evidence = build_pickcube_task_evidence(
        RawPickCubeTaskSnapshot(
            evaluator_values={
                "success": np.array([False], dtype=np.bool_),
                "is_obj_placed": np.array([False], dtype=np.bool_),
                "is_robot_static": np.array([False], dtype=np.bool_),
                "is_grasped": np.array([True], dtype=np.bool_),
            },
            cube_center_z=np.array([-0.01], dtype=np.float32),
            cube_to_goal_distance=np.array([0.3], dtype=np.float32),
        ),
        _task_keys(),
    )

    assert evidence.status is TerminalTaskStatus.COMPLETE
    assert evidence.success is False
    assert evidence.progress == 0.0
    assert evidence.unsafe is True
    assert [event.failure_type for event in evidence.failure_events] == [
        "task_not_completed",
        "cube_not_at_goal",
        "robot_not_static",
        "cube_below_world_zero",
    ]
    assert not any("grasp" in event.failure_type for event in evidence.failure_events)


def test_complete_success_has_binary_one_progress_and_no_failure_events() -> None:
    evidence = build_pickcube_task_evidence(
        RawPickCubeTaskSnapshot(
            evaluator_values={
                "success": np.array([True], dtype=np.bool_),
                "is_obj_placed": np.array([True], dtype=np.bool_),
                "is_robot_static": np.array([True], dtype=np.bool_),
                "is_grasped": np.array([False], dtype=np.bool_),
            },
            cube_center_z=0.05,
            cube_to_goal_distance=0.0,
        ),
        _task_keys(),
    )
    assert evidence.status is TerminalTaskStatus.COMPLETE
    assert evidence.success is True
    assert evidence.progress == 1.0
    assert evidence.unsafe is False
    assert evidence.failure_events == ()


@pytest.mark.parametrize(
    "values",
    [
        {
            "success": np.array([False], dtype=np.bool_),
            "is_obj_placed": np.array([False], dtype=np.bool_),
            "is_robot_static": np.array([False], dtype=np.bool_),
        },
        {
            "success": np.array([0], dtype=np.int64),
            "is_obj_placed": np.array([False], dtype=np.bool_),
            "is_robot_static": np.array([False], dtype=np.bool_),
            "is_grasped": np.array([False], dtype=np.bool_),
        },
        {
            "success": np.array([False, True], dtype=np.bool_),
            "is_obj_placed": np.array([False], dtype=np.bool_),
            "is_robot_static": np.array([False], dtype=np.bool_),
            "is_grasped": np.array([False], dtype=np.bool_),
        },
    ],
)
def test_task_evidence_missing_non_boolean_or_non_scalar_is_indeterminate(
    values: dict[str, object],
) -> None:
    evidence = build_pickcube_task_evidence(
        RawPickCubeTaskSnapshot(
            evaluator_values=values,
            cube_center_z=0.1,
            cube_to_goal_distance=0.2,
        ),
        _task_keys(),
    )
    assert evidence.status is TerminalTaskStatus.INDETERMINATE
    assert evidence.success is None
    assert evidence.failure_events == ()


def test_session_restores_same_reference_in_two_fresh_environments() -> None:
    _, replay_case, _, _, runtime, state_loader = _adapter_fixture()
    factory = ManiSkillPickCubeSessionFactory(
        settings=_settings(),
        action_contract=_action_contract(),
        task_key_contract=_task_keys(),
        state_loader=state_loader,
        state_comparator=_Comparator(),
        runtime=runtime,
    )
    baseline = factory.create_session(
        replay_case, execution_role=ReplayExecutionRole.BASELINE
    )
    corrupted = factory.create_session(
        replay_case, execution_role=ReplayExecutionRole.CORRUPTED
    )
    assert baseline is not corrupted
    assert runtime.environments[0] is not runtime.environments[1]
    first = baseline.restore_state(replay_case.state_reference)
    second = corrupted.restore_state(replay_case.state_reference)
    assert first.match_kind is StateMatchKind.EXACT
    assert second.match_kind is StateMatchKind.EXACT
    assert state_loader.references == [
        replay_case.state_reference,
        replay_case.state_reference,
    ]
    baseline.close()
    corrupted.close()


def test_restoration_mismatch_is_unverified_not_repaired() -> None:
    _, replay_case = _proposal_and_case()
    session = ManiSkillPickCubeReplaySession(
        replay_case,
        execution_role=ReplayExecutionRole.BASELINE,
        settings=_settings(),
        action_contract=_action_contract(),
        task_key_contract=_task_keys(),
        state_loader=_StateLoader(),
        state_comparator=_Comparator(mismatch=True),
        runtime=_FakeRuntime(),
    )
    evidence = session.restore_state(replay_case.state_reference)
    assert evidence.match_kind is StateMatchKind.MISMATCH
    assert evidence.restoration_verified is False
    with pytest.raises(ManiSkillPickCubeSessionError, match="verified"):
        session.evaluate_task(replay_case.task_reference)
    session.close()


def test_runtime_archive_store_loads_content_bound_initial_state(
    tmp_path: Path,
) -> None:
    action_contract = _action_contract()
    episode = ManiSkillReferenceEpisode(
        compatibility_identity="sha256:" + "5" * 64,
        source_solver_identity={
            "module_name": "mani_skill.examples.pick_cube",
            "source_sha256": "sha256:" + "6" * 64,
        },
        environment_configuration=_settings().as_mapping(),
        seed=17,
        source_actions=np.array([[0.1, 0.0]], dtype=np.float32),
        initial_state={"actors": np.array([1.0, 2.0], dtype=np.float32)},
        terminal_state={"actors": np.array([2.0, 3.0], dtype=np.float32)},
        terminal_task_evidence={
            "success": True,
            "is_obj_placed": True,
            "is_robot_static": True,
            "is_grasped": False,
            "cube_center_z": 0.05,
            "cube_to_goal_distance": 0.0,
        },
        initial_robot_state=np.array([0.1, 0.0], dtype=np.float32),
        robot_state_joint_names=("panda_joint1",),
        robot_state_semantic="panda_active_joint_qpos_qvel_named_v1",
        action_contract=action_contract.as_mapping(),
        action_coordinate_frame=action_contract.coordinate_frame,
        control_period_s=action_contract.control_period_s,
        source_generation_success=True,
        independent_baseline_success=True,
    )
    archive_directory = tmp_path / "runtime-archive"
    save_reference_archive(
        ManiSkillReferenceArchive(episodes=(episode,)), archive_directory
    )
    reference = ReplayStateReference(
        adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
        adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
        source_reference_id=episode.episode_id,
        expected_state_digest=episode.initial_state_digest,
        comparison_semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
        state_key="initial_state",
    )
    store = ArchivePickCubeReferenceStateStore(archive_directory)

    store.validate_reference(reference)
    loaded = store.load_reference_state(reference)
    assert loaded.source_reference_id == episode.episode_id
    assert loaded.state_digest == episode.initial_state_digest
    assert loaded.compared_component_count == 2


def test_runtime_action_mutation_is_detected() -> None:
    _, replay_case = _proposal_and_case()
    session = ManiSkillPickCubeReplaySession(
        replay_case,
        execution_role=ReplayExecutionRole.BASELINE,
        settings=_settings(),
        action_contract=_action_contract(),
        task_key_contract=_task_keys(),
        state_loader=_StateLoader(),
        state_comparator=_Comparator(),
        runtime=_FakeRuntime(mutate_action=True),
    )
    session.restore_state(replay_case.state_reference)
    with pytest.raises(ManiSkillPickCubeSessionError, match="mutated"):
        session.step_action(replay_case.original_action.actions[0])
    session.close()


def test_adapter_complete_task_failure_is_conclusive_strong_simulator_evidence() -> (
    None
):
    proposal, _, bundle, adapter, runtime, _ = _adapter_fixture()
    evaluator = create_exact_state_paired_replay_evaluator(adapter, bundle)
    evidence = evaluator.evaluate(
        proposal,
        source_dataset_id="source-dataset",
        evaluation_seed=11,
        attempt_ordinal=0,
    )

    assert evidence.status is EvaluationStatus.CONCLUSIVE
    assert evidence.success is False
    assert evidence.label_source is LabelSource.SIMULATOR
    assert evidence.label_strength is LabelStrength.STRONG
    assert evidence.simulator_replay_verified is True
    assert evidence.metrics["replay_baseline_requested_steps"] == 2
    assert evidence.metrics["replay_baseline_steps"] == 2
    assert evidence.metrics["replay_corrupted_requested_steps"] == 2
    assert evidence.metrics["replay_corrupted_steps"] == 2
    assert evidence.metrics["replay_baseline_restoration_maximum_absolute_error"] == 0.0
    assert (
        evidence.metrics["replay_corrupted_restoration_maximum_absolute_error"] == 0.0
    )
    assert evidence.metrics["initial_cube_to_goal_distance"] == pytest.approx(0.4)
    assert evidence.metrics["final_cube_to_goal_distance"] == pytest.approx(0.4)
    assert evidence.metrics["final_cube_center_z"] == pytest.approx(0.05)
    assert evidence.metrics["object_placed"] is False
    assert evidence.metrics["robot_static"] is True
    assert evidence.metrics["grasped"] is False
    assert evidence.metrics[
        "source_corrupted_action_mean_absolute_difference"
    ] == pytest.approx(0.175)
    assert evidence.metrics[
        "source_corrupted_action_maximum_absolute_difference"
    ] == pytest.approx(0.3)
    assert len(runtime.environments) == 2
    assert runtime.environments[0] is not runtime.environments[1]
    assert all(environment.closed for environment in runtime.environments)


def test_adapter_complete_success_is_conclusive_strong_simulator_evidence() -> None:
    proposal, _, bundle, adapter, _, _ = _adapter_fixture_with_runtime(
        _FakeRuntime(corrupted_succeeds=True)
    )
    evidence = create_exact_state_paired_replay_evaluator(adapter, bundle).evaluate(
        proposal,
        source_dataset_id="source-dataset",
        evaluation_seed=12,
        attempt_ordinal=0,
    )
    assert evidence.status is EvaluationStatus.CONCLUSIVE
    assert evidence.success is True
    assert evidence.progress_after == 1.0
    assert evidence.simulator_replay_verified is True


def test_baseline_failure_is_invalid_context_and_skips_corrupted_session() -> None:
    runtime = _FakeRuntime(baseline_succeeds=False)
    proposal, _, bundle, adapter, _, _ = _adapter_fixture_with_runtime(runtime)
    evidence = create_exact_state_paired_replay_evaluator(adapter, bundle).evaluate(
        proposal,
        source_dataset_id="source-dataset",
        evaluation_seed=13,
        attempt_ordinal=0,
    )
    assert evidence.status is EvaluationStatus.INVALID
    assert evidence.success is None
    assert evidence.simulator_replay_verified is False
    assert len(runtime.environments) == 1


def test_simulator_exception_is_execution_error_not_task_failure() -> None:
    runtime = _FakeRuntime(corrupted_step_exception=True)
    proposal, _, bundle, adapter, _, _ = _adapter_fixture_with_runtime(runtime)
    evidence = create_exact_state_paired_replay_evaluator(adapter, bundle).evaluate(
        proposal,
        source_dataset_id="source-dataset",
        evaluation_seed=14,
        attempt_ordinal=0,
    )
    assert evidence.status is EvaluationStatus.EXECUTION_ERROR
    assert evidence.success is None
    assert evidence.failure_events == ()
    assert evidence.label_source is None
    assert evidence.simulator_replay_verified is False


def test_trusted_attestation_rejects_a_missing_real_runtime_gate() -> None:
    with pytest.raises(
        ManiSkillPickCubeAdapterConfigurationError, match="every real-runtime gate"
    ):
        TrustedManiSkillRuntimeAttestation(
            compatibility_identity="compat-sha256-" + "5" * 64,
            semantic_configuration_digest="cfg-sha256-" + "6" * 64,
            compatibility_probe_passed=True,
            dependency_versions_verified=True,
            action_contract_verified=True,
            state_api_verified=True,
            state_round_trip_verified=True,
            task_contract_verified=True,
            solver_source_verified=True,
            task_source_verified=True,
            real_simulator_runtime=False,
        )


def test_out_of_bounds_corruption_is_invalid_before_session_creation() -> None:
    transformed = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    _, replay_case = _proposal_and_case(transformed=transformed)
    with pytest.raises(
        ReplayInvalidContextError, match="corrupted_action_out_of_bounds"
    ):
        _action_contract().validate_action_chunk(
            replay_case.transformed_action, role="corrupted"
        )


def test_action_contract_preserves_content_bound_floating_dtype() -> None:
    contract = _action_contract()
    action = ActionChunk(
        actions=np.array([[0.2, 0.1], [0.3, -0.1]], dtype=np.float64),
        coordinate_frame=contract.coordinate_frame,
        control_period_s=contract.control_period_s,
    )

    contract.validate_action_chunk(action, role="baseline")
    assert action.actions.dtype == np.dtype(np.float64)


def test_action_contract_rejects_integer_rows() -> None:
    with pytest.raises(ReplayInvalidContextError, match="dtype_not_floating"):
        _action_contract().validate_row(np.array([0, 0], dtype=np.int64))


def test_importing_adapter_does_not_import_optional_simulator_modules() -> None:
    forbidden = {"mani_skill", "sapien", "mplib", "torch", "gymnasium"}
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import json,sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "import latentguard.integrations.maniskill_pickcube.adapter;"
        "print(json.dumps(sorted(set(sys.modules).intersection("
        f"{sorted(forbidden)!r}))))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []
