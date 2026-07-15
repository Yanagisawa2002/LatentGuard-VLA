from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from latentguard.integrations.maniskill_pickcube.anchors import PickCubeStateAnchor
from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_baseline import (
    ANCHOR_BASELINE_PURPOSE,
    AnchorBaselineFailureError,
    AnchorBaselineIntegrityError,
    AnchorBaselineInvalidContextError,
    AnchorBaselineRuntimeError,
    PickCubeAnchorBaselineValidator,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    PickCubeActionControlContractV1,
    build_source_continuation_identity,
    build_state_indexed_anchor_sources,
)
from latentguard.integrations.maniskill_pickcube.state_tree import (
    clone_state_tree,
    compute_state_tree_digest,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    build_pickcube_verifier_state_v1,
)
from latentguard.models import LabelSource, LabelStrength

_COMPATIBILITY = f"sha256:{'a' * 64}"


def _task(index: int, action_count: int) -> PickCubeTaskSnapshotV1:
    terminal = index == action_count
    return PickCubeTaskSnapshotV1(
        success=terminal,
        is_obj_placed=terminal,
        is_robot_static=terminal,
        is_grasped=index >= 1,
        cube_center_z=0.1,
        cube_to_goal_distance=float(action_count - index) * 0.1,
        tcp_to_cube_distance=float(max(1 - index, 0)) * 0.1,
    )


def _state(
    index: int,
    action_count: int,
    *,
    source_initial_grasped: bool = False,
) -> PickCubeIndexedStateV1:
    terminal = index == action_count
    task_snapshot = _task(index, action_count)
    if index == 0 and source_initial_grasped:
        task_snapshot = replace(task_snapshot, is_grasped=True)
    return PickCubeIndexedStateV1(
        state_index=index,
        source_action_index=index,
        tree={
            "actors": np.array([index, index + 0.25], dtype=np.float32),
            "robot": {"qpos": np.array([index * 0.1], dtype=np.float64)},
        },
        task_snapshot=task_snapshot,
        restored_task_snapshot=_task(index, action_count),
        verifier_state=build_pickcube_verifier_state_v1(
            joint_names=("joint_a", "joint_b"),
            qpos=np.array([index, -index], dtype=np.float32),
            qvel=np.array([0.1, -0.1], dtype=np.float32),
            tcp_position=np.array([0.0, 0.0, 0.1 + index], dtype=np.float32),
            tcp_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            cube_position=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            cube_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            goal_position=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            is_grasped=index >= 1,
            is_obj_placed=terminal,
            is_robot_static=terminal,
        ),
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_trajectory_id="mspc-trajectory-a",
    )


def _source_episode(
    *, source_initial_grasped: bool = False
) -> PickCubeStateIndexedEpisodeV1:
    action_count = 4
    return PickCubeStateIndexedEpisodeV1(
        episode_id="mspc-sequence-a",
        source_trajectory_id="mspc-trajectory-a",
        source_policy_identity="maniskill/3.0.1/official-pickcube-solver-v1",
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_actions=np.array(
            [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
            dtype=np.float32,
        ),
        states=tuple(
            _state(
                index,
                action_count,
                source_initial_grasped=source_initial_grasped,
            )
            for index in range(action_count + 1)
        ),
    )


def _anchor() -> PickCubeStateAnchor:
    return PickCubeStateAnchor(
        anchor_id=f"mspc-anchor-{'b' * 64}",
        state_index=0,
        selection_reason="early_trajectory",
        source_trajectory_id="mspc-trajectory-a",
        source_seed=7,
        split_group_id=f"mspc-split-{'c' * 64}",
        remaining_horizon=4,
        candidate_horizon=2,
    )


def _build_action_contract() -> PickCubeActionControlContractV1:
    return PickCubeActionControlContractV1(
        coordinate_frame="joint_position",
        control_period_s=0.05,
        action_dtype="<f4",
        action_dimension=2,
    )


def _runtime_action_contract() -> PickCubeReplayActionContract:
    return PickCubeReplayActionContract(
        total_dimension=2,
        environment_numpy_dtype="<f4",
        lower_bounds=np.full(2, -1.0, dtype=np.float32),
        upper_bounds=np.full(2, 1.0, dtype=np.float32),
        environment_shape=(2,),
        coordinate_frame="joint_position",
        control_period_s=0.05,
    )


def _settings() -> ManiSkillPickCubeEnvironmentSettings:
    return ManiSkillPickCubeEnvironmentSettings(
        obs_mode="state_dict", state_tolerance=1e-6
    )


def _keys() -> PickCubeTaskKeyContract:
    return PickCubeTaskKeyContract(
        success="success",
        object_placed="is_obj_placed",
        robot_static="is_robot_static",
        grasped="is_grasped",
    )


@dataclass
class _FakeEnvironment:
    total_actions: int
    restore_error: float = 0.0
    terminal_success: bool = True
    mutate_action: bool = False
    step_error: bool = False
    state: object | None = None
    step_count: int = 0
    reset_seed: int | None = None
    closed: bool = False
    task_capture_steps: list[int] = field(default_factory=list)
    agent: object | None = None
    cube: object | None = None
    goal_site: object | None = None

    @property
    def unwrapped(self) -> _FakeEnvironment:
        return self

    def reset(self, *, seed: int) -> None:
        self.reset_seed = seed
        self.step_count = 0

    def set_state_dict(self, state: object) -> None:
        self.state = clone_state_tree(state)

    def get_state_dict(self) -> object:
        assert self.state is not None
        observed = clone_state_tree(self.state)
        if self.restore_error:
            assert isinstance(observed, dict)
            actors = observed["actors"]
            assert isinstance(actors, np.ndarray)
            actors[0] += self.restore_error
        return observed

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeFactory:
    restore_error: float = 0.0
    verifier_error: float = 0.0
    terminal_success: bool = True
    mutate_action: bool = False
    step_error: bool = False
    environments: list[_FakeEnvironment] = field(default_factory=list)
    purposes: list[str] = field(default_factory=list)

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        purpose: str,
    ) -> object:
        assert settings.state_tolerance == 1e-6
        assert action_contract.total_dimension == 2
        environment = _FakeEnvironment(
            total_actions=4,
            restore_error=self.restore_error,
            terminal_success=self.terminal_success,
            mutate_action=self.mutate_action,
            step_error=self.step_error,
        )
        environment.agent = SimpleNamespace(
            tcp=_pose([0.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.0])
        )
        environment.cube = _pose([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0])
        environment.goal_site = _pose([0.4, 0.5, 0.6], [1.0, 0.0, 0.0, 0.0])
        self.environments.append(environment)
        self.purposes.append(purpose)
        return environment

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        return clone_state_tree(state_tree)

    def capture_task_snapshot(
        self, environment: object, key_contract: PickCubeTaskKeyContract
    ) -> RawPickCubeTaskSnapshot:
        assert isinstance(environment, _FakeEnvironment)
        environment.task_capture_steps.append(environment.step_count)
        terminal = environment.step_count == environment.total_actions
        success = terminal and environment.terminal_success
        return RawPickCubeTaskSnapshot(
            evaluator_values={
                key_contract.success: np.bool_(success),
                key_contract.object_placed: np.bool_(success),
                key_contract.robot_static: np.bool_(success),
                key_contract.grasped: np.bool_(environment.step_count >= 1),
            },
            cube_center_z=np.float32(0.1),
            cube_to_goal_distance=np.float32(
                max(environment.total_actions - environment.step_count, 0) * 0.1
            ),
        )

    def step_action(
        self,
        environment: object,
        action: NDArray[Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        assert isinstance(environment, _FakeEnvironment)
        assert action_contract.total_dimension == action.shape[0]
        if environment.step_error:
            raise RuntimeError("fake step failed")
        environment.step_count += 1
        if environment.mutate_action:
            action[0] += np.float32(0.01)

    def extract_named_robot_state(
        self, environment: object
    ) -> tuple[tuple[str, ...], NDArray[Any]]:
        del environment
        return (
            ("joint_a", "joint_b"),
            np.array([self.verifier_error, 0.0, 0.1, -0.1], dtype=np.float32),
        )


def _pose(position: list[float], quaternion: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        pose=SimpleNamespace(
            p=np.array([position], dtype=np.float32),
            q=np.array([quaternion], dtype=np.float32),
        )
    )


def _validator(factory: _FakeFactory) -> PickCubeAnchorBaselineValidator:
    return PickCubeAnchorBaselineValidator(
        environment_factory=factory,
        settings=_settings(),
        runtime_action_contract=_runtime_action_contract(),
        task_key_contract=_keys(),
    )


def _validate(
    factory: _FakeFactory,
    *,
    source: PickCubeStateIndexedEpisodeV1 | None = None,
):
    source = source or _source_episode()
    anchor = _anchor()
    build_contract = _build_action_contract()
    continuation = build_source_continuation_identity(
        source_episode=source,
        anchor_state_index=anchor.state_index,
        candidate_horizon=anchor.candidate_horizon,
        action_control_contract=build_contract,
    )
    return _validator(factory).validate_anchor_baseline(
        archive_content_digest=f"sha256:{'d' * 64}",
        source_episode=source,
        source_state=source.states[anchor.state_index],
        anchor=anchor,
        source_remaining_actions=source.source_actions[anchor.state_index :],
        continuation=continuation,
        action_control_contract=build_contract,
    )


def test_successful_anchor_baseline_uses_fresh_environment_and_three_boundaries() -> (
    None
):
    factory = _FakeFactory()
    source = _source_episode()
    actions_before = source.source_actions.tobytes(order="C")
    state_before = compute_state_tree_digest(source.states[0].tree)
    evidence = _validate(factory)

    assert evidence.label_source is LabelSource.SIMULATOR
    assert evidence.label_strength is LabelStrength.STRONG
    assert evidence.simulator_replay_verified is True
    assert evidence.state_restoration_verified is True
    assert evidence.complete_state_comparison is True
    assert evidence.compared_component_count == source.states[0].numeric_component_count
    assert evidence.maximum_absolute_error == 0.0
    assert evidence.verifier_state_restoration_verified is True
    assert evidence.verifier_state_compared_component_count == (
        source.states[0].verifier_state.values.size
    )
    assert evidence.verifier_state_maximum_absolute_error == 0.0
    assert evidence.executed_action_count == 4
    assert evidence.progress_before == 0.0
    assert evidence.progress_after_candidate == 0.0
    assert evidence.progress_delta == 0.0
    assert evidence.terminal_progress == 1.0
    assert evidence.official_terminal_success is True
    assert evidence.continuation_identity.startswith("sha256:")
    assert factory.purposes == [ANCHOR_BASELINE_PURPOSE]
    assert factory.environments[0].reset_seed == 7
    assert factory.environments[0].task_capture_steps == [0, 0, 2, 4]
    assert factory.environments[0].closed is True
    assert source.source_actions.tobytes(order="C") == actions_before
    assert compute_state_tree_digest(source.states[0].tree) == state_before

    _validate(factory)
    assert len(factory.environments) == 2
    assert factory.environments[0] is not factory.environments[1]
    assert all(environment.closed for environment in factory.environments)


def test_baseline_binds_restored_snapshot_not_source_contact_annotation() -> None:
    factory = _FakeFactory()
    source = _source_episode(source_initial_grasped=True)

    evidence = _validate(factory, source=source)

    assert source.states[0].task_snapshot.is_grasped is True
    assert source.states[0].restored_task_snapshot.is_grasped is False
    assert evidence.simulator_replay_verified is True


def test_restoration_mismatch_fails_closed_and_closes_environment() -> None:
    factory = _FakeFactory(restore_error=1e-3)
    with pytest.raises(AnchorBaselineInvalidContextError, match="restoration"):
        _validate(factory)

    assert len(factory.environments) == 1
    assert factory.environments[0].step_count == 0
    assert factory.environments[0].closed is True


def test_verifier_state_mismatch_fails_closed_before_actions() -> None:
    factory = _FakeFactory(verifier_error=1e-3)
    with pytest.raises(
        AnchorBaselineInvalidContextError, match="verifier-state values"
    ):
        _validate(factory)

    assert len(factory.environments) == 1
    assert factory.environments[0].step_count == 0
    assert factory.environments[0].closed is True


def test_complete_baseline_task_failure_is_not_strong_evidence() -> None:
    factory = _FakeFactory(terminal_success=False)
    with pytest.raises(AnchorBaselineFailureError, match="official task success"):
        _validate(factory)

    assert factory.environments[0].step_count == 4
    assert factory.environments[0].task_capture_steps == [0, 0, 2, 4]
    assert factory.environments[0].closed is True


def test_builder_excludes_typed_baseline_failure_without_source_episode() -> None:
    factory = _FakeFactory(terminal_success=False)
    result = build_state_indexed_anchor_sources(
        PickCubeStateIndexedArchiveV1(episodes=(_source_episode(),)),
        action_control_contract=_build_action_contract(),
        baseline_validator=_validator(factory),
        candidate_horizon=2,
        maximum_anchors_per_trajectory=1,
    )

    assert result.source_episodes == ()
    assert result.manifest.records == ()
    assert result.manifest.baseline_evidence == ()
    assert len(result.manifest.exclusions) == 1
    assert result.manifest.exclusions[0].reason_code == "baseline_task_failure"
    assert factory.environments[0].closed is True


def test_forwarded_action_mutation_is_rejected_and_environment_is_closed() -> None:
    factory = _FakeFactory(mutate_action=True)
    source = _source_episode()
    actions_before = source.source_actions.tobytes(order="C")
    with pytest.raises(AnchorBaselineIntegrityError, match="mutated"):
        _validate(factory)

    assert factory.environments[0].step_count == 1
    assert factory.environments[0].closed is True
    assert source.source_actions.tobytes(order="C") == actions_before


def test_runtime_step_error_is_distinct_and_environment_is_closed() -> None:
    factory = _FakeFactory(step_error=True)
    with pytest.raises(AnchorBaselineRuntimeError, match="RuntimeError"):
        _validate(factory)

    assert factory.environments[0].closed is True


@pytest.mark.parametrize(
    "purpose",
    [
        "official_state_indexed_source_generation",
        "indexed_state_fresh_validation",
        ANCHOR_BASELINE_PURPOSE,
    ],
)
def test_lazy_production_factory_allows_fixed_m3a_purposes(purpose: str) -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.roles: list[object] = []

        def create_environment(
            self,
            settings: ManiSkillPickCubeEnvironmentSettings,
            action_contract: PickCubeReplayActionContract,
            *,
            execution_role: object,
        ) -> object:
            self.roles.append(execution_role)
            return object()

    factory = LazyManiSkillSourceEnvironmentFactory()
    runtime = _Runtime()
    factory._runtime = runtime  # type: ignore[assignment]

    created = factory.create_environment(
        _settings(),
        _runtime_action_contract(),
        purpose=purpose,
    )

    assert type(created) is object
    assert len(runtime.roles) == 1
