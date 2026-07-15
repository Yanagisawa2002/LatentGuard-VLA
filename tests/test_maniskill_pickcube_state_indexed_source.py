from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    OFFICIAL_SOLVER_EXPORT,
    OFFICIAL_SOLVER_MODULE,
    PickCubeSourceGenerationError,
    SolverSourceIdentity,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_source import (
    collect_state_indexed_reference_archive,
    verify_all_indexed_states_fresh,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
)


@dataclass
class _Pose:
    p: np.ndarray[Any, Any]
    q: np.ndarray[Any, Any]


@dataclass
class _Entity:
    pose: _Pose


class _Joint:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _Robot:
    def __init__(self, environment: _Environment) -> None:
        self.environment = environment

    def get_active_joints(self) -> list[_Joint]:
        return [_Joint("joint-a"), _Joint("joint-b")]

    def get_qpos(self) -> np.ndarray[Any, Any]:
        count = float(self.environment.count)
        return np.array([[count, count + 0.25]], dtype=np.float32)

    def get_qvel(self) -> np.ndarray[Any, Any]:
        count = float(self.environment.count)
        return np.array([[count * 0.1, count * 0.2]], dtype=np.float32)


class _Agent:
    def __init__(self, environment: _Environment) -> None:
        self.robot = _Robot(environment)
        self.tcp = _Entity(
            _Pose(
                p=np.zeros((1, 3), dtype=np.float32),
                q=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        )


class _Environment:
    def __init__(self, target_steps: int) -> None:
        self.target_steps = target_steps
        self.count = 0
        self.grasp_contact = False
        self.closed = False
        self.agent = _Agent(self)
        self.cube = _Entity(
            _Pose(
                p=np.zeros((1, 3), dtype=np.float32),
                q=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        )
        self.goal_site = _Entity(
            _Pose(
                p=np.array([[1.0, 0.0, 0.1]], dtype=np.float32),
                q=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        )
        self._sync()

    @property
    def unwrapped(self) -> _Environment:
        return self

    def _sync(self) -> None:
        fraction = self.count / self.target_steps
        self.agent.tcp.pose.p = np.array([[fraction, 0.0, 0.2]], dtype=np.float32)
        self.cube.pose.p = np.array([[fraction, 0.0, 0.1]], dtype=np.float32)

    def reset(self, *, seed: int) -> tuple[None, dict[str, int]]:
        assert 0 <= seed < 2**32
        self.count = 0
        self.grasp_contact = False
        self._sync()
        return None, {"elapsed_steps": 0}

    def step(self, action: object) -> tuple[None, float, bool, bool, dict[str, int]]:
        array = np.asarray(action)
        assert array.shape in ((8,), (1, 8))
        self.count += 1
        self.grasp_contact = 5 <= self.count < self.target_steps
        self._sync()
        return None, 0.0, False, False, {"elapsed_steps": self.count}

    def get_state_dict(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "count": np.array([self.count], dtype=np.int64),
            "continuous": np.array(
                [self.count / self.target_steps, self.count * 0.125],
                dtype=np.float32,
            ),
        }

    def set_state_dict(self, state: object) -> None:
        assert isinstance(state, dict)
        self.count = int(np.asarray(state["count"]).reshape(()))
        # ManiSkill's public grasp result is contact-impulse-derived and is not
        # reconstructed by set_state_dict until physics advances again.
        self.grasp_contact = False
        self._sync()

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, target_steps: int) -> None:
        self.target_steps = target_steps
        self.environments: list[_Environment] = []
        self.restored_grasp_override: bool | None = None
        self.mutate_state_during_task_capture = False

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        purpose: str,
    ) -> _Environment:
        del settings, action_contract, purpose
        environment = _Environment(self.target_steps)
        self.environments.append(environment)
        return environment

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        del environment
        from latentguard.integrations.maniskill_pickcube.state_tree import (
            clone_state_tree,
        )

        return clone_state_tree(state_tree)

    def capture_task_snapshot(
        self,
        environment: object,
        key_contract: PickCubeTaskKeyContract,
    ) -> RawPickCubeTaskSnapshot:
        assert isinstance(environment, _Environment)
        success = environment.count >= environment.target_steps
        grasped = environment.grasp_contact
        if (
            self.restored_grasp_override is not None
            and environment.count > 0
            and not environment.grasp_contact
        ):
            grasped = self.restored_grasp_override
        cube = environment.cube.pose.p.reshape(3)
        goal = environment.goal_site.pose.p.reshape(3)
        snapshot = RawPickCubeTaskSnapshot(
            evaluator_values={
                key_contract.success: np.array([success], dtype=np.bool_),
                key_contract.object_placed: np.array([success], dtype=np.bool_),
                key_contract.robot_static: np.array([success], dtype=np.bool_),
                key_contract.grasped: np.array([grasped], dtype=np.bool_),
            },
            cube_center_z=np.array([cube[2]], dtype=np.float32),
            cube_to_goal_distance=np.array(
                [np.linalg.norm(cube - goal)], dtype=np.float32
            ),
        )
        if self.mutate_state_during_task_capture:
            environment.count += 1
            environment._sync()
        return snapshot

    def step_action(
        self,
        environment: object,
        action: np.ndarray[Any, Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        assert isinstance(environment, _Environment)
        action_contract.validate_row(action)
        environment.step(action)

    def extract_named_robot_state(
        self, environment: object
    ) -> tuple[tuple[str, ...], np.ndarray[Any, Any]]:
        assert isinstance(environment, _Environment)
        qpos = environment.agent.robot.get_qpos().reshape(-1)
        qvel = environment.agent.robot.get_qvel().reshape(-1)
        return ("joint-a", "joint-b"), np.concatenate((qpos, qvel))


def _settings() -> ManiSkillPickCubeEnvironmentSettings:
    return ManiSkillPickCubeEnvironmentSettings(
        obs_mode="none",
        state_tolerance=1e-6,
    )


def _action_contract() -> PickCubeReplayActionContract:
    return PickCubeReplayActionContract(
        total_dimension=8,
        environment_numpy_dtype=np.dtype(np.float32).str,
        lower_bounds=np.full(8, -1.0, dtype=np.float32),
        upper_bounds=np.full(8, 1.0, dtype=np.float32),
        environment_shape=(1, 8),
        coordinate_frame="unspecified",
        control_period_s=0.05,
    )


def _solver(environment: object, *, seed: int, debug: bool, vis: bool) -> int:
    del debug, vis
    runtime: Any = environment
    runtime.reset(seed=seed)
    for _ in range(18):
        runtime.step(np.zeros(8, dtype=np.float32))
    return 0


def test_collection_records_t_plus_one_and_freshly_verifies_every_state() -> None:
    factory = _Factory(target_steps=18)
    key_contract = PickCubeTaskKeyContract(
        success="success",
        object_placed="is_obj_placed",
        robot_static="is_robot_static",
        grasped="is_grasped",
    )
    result = collect_state_indexed_reference_archive(
        requested_success_count=1,
        starting_seed=3,
        maximum_attempts=1,
        compatibility_identity="sha256:" + "1" * 64,
        environment_factory=factory,
        settings=_settings(),
        action_contract=_action_contract(),
        key_contract=key_contract,
        solver=_solver,
        solver_identity=SolverSourceIdentity(
            module_name=OFFICIAL_SOLVER_MODULE,
            export_name=OFFICIAL_SOLVER_EXPORT,
            source_sha256="2" * 64,
        ),
    )
    assert result.archive is not None
    assert result.reference_archive is not None
    episode = result.archive.episodes[0]
    assert episode.source_actions.shape == (18, 8)
    assert len(episode.states) == 19
    assert episode.states[0].state_index == 0
    assert episode.states[-1].state_index == 18
    assert episode.states[-1].task_snapshot.success is True
    assert (
        episode.source_trajectory_id
        == result.reference_archive.episodes[0].source_trajectory_id
    )
    grasped_state = episode.states[5]
    assert grasped_state.task_snapshot.is_grasped is True
    assert grasped_state.restored_task_snapshot.is_grasped is False
    grasped_component = grasped_state.verifier_state.component_names.index(
        "task/is_grasped"
    )
    assert grasped_state.verifier_state.values[grasped_component] == 0.0

    audit = verify_all_indexed_states_fresh(
        episode,
        environment_factory=factory,
        settings=_settings(),
        action_contract=_action_contract(),
        key_contract=key_contract,
    )
    assert len(audit) == 19
    assert {record.compared_component_count for record in audit} == {3}
    assert max(record.maximum_absolute_error for record in audit) == 0.0
    assert max(record.verifier_maximum_absolute_error for record in audit) == 0.0
    assert audit[5].source_to_restored_task_mismatch_fields == ("is_grasped",)
    assert audit[4].source_to_restored_task_mismatch_fields == ()
    assert all(environment.closed for environment in factory.environments)


def test_independent_verification_rejects_restored_task_projection_drift() -> None:
    factory = _Factory(target_steps=18)
    key_contract = PickCubeTaskKeyContract(
        success="success",
        object_placed="is_obj_placed",
        robot_static="is_robot_static",
        grasped="is_grasped",
    )
    result = collect_state_indexed_reference_archive(
        requested_success_count=1,
        starting_seed=3,
        maximum_attempts=1,
        compatibility_identity="sha256:" + "1" * 64,
        environment_factory=factory,
        settings=_settings(),
        action_contract=_action_contract(),
        key_contract=key_contract,
        solver=_solver,
        solver_identity=SolverSourceIdentity(
            module_name=OFFICIAL_SOLVER_MODULE,
            export_name=OFFICIAL_SOLVER_EXPORT,
            source_sha256="2" * 64,
        ),
    )
    assert result.archive is not None
    factory.restored_grasp_override = True

    with pytest.raises(
        PickCubeSourceGenerationError,
        match="restored task snapshot fields: is_grasped",
    ):
        verify_all_indexed_states_fresh(
            result.archive.episodes[0],
            environment_factory=factory,
            settings=_settings(),
            action_contract=_action_contract(),
            key_contract=key_contract,
        )

    assert all(environment.closed for environment in factory.environments)


def test_public_projection_extraction_must_not_mutate_restored_state() -> None:
    factory = _Factory(target_steps=18)
    key_contract = PickCubeTaskKeyContract(
        success="success",
        object_placed="is_obj_placed",
        robot_static="is_robot_static",
        grasped="is_grasped",
    )
    result = collect_state_indexed_reference_archive(
        requested_success_count=1,
        starting_seed=3,
        maximum_attempts=1,
        compatibility_identity="sha256:" + "1" * 64,
        environment_factory=factory,
        settings=_settings(),
        action_contract=_action_contract(),
        key_contract=key_contract,
        solver=_solver,
        solver_identity=SolverSourceIdentity(
            module_name=OFFICIAL_SOLVER_MODULE,
            export_name=OFFICIAL_SOLVER_EXPORT,
            source_sha256="2" * 64,
        ),
    )
    assert result.archive is not None
    factory.mutate_state_during_task_capture = True

    with pytest.raises(
        PickCubeSourceGenerationError,
        match="public projection extraction mutated the restored state",
    ):
        verify_all_indexed_states_fresh(
            result.archive.episodes[0],
            environment_factory=factory,
            settings=_settings(),
            action_contract=_action_contract(),
            key_contract=key_contract,
        )

    assert all(environment.closed for environment in factory.environments)
