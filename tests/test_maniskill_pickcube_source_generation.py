from __future__ import annotations

import hashlib
import warnings
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
    PickCubeSourceCollectionIncompleteError,
    PickCubeSourceGenerationError,
    PickCubeTrajectoryActionLimitError,
    RecordingEnvironmentProxy,
    SolverSourceIdentity,
    collect_reference_archive,
    ordered_source_seeds,
    record_official_source_trajectory,
    run_official_solver,
    validate_independent_source_baseline,
)
from latentguard.integrations.maniskill_pickcube.state_tree import (
    compute_state_tree_digest,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
)


def _contract() -> PickCubeReplayActionContract:
    return PickCubeReplayActionContract(
        total_dimension=2,
        environment_numpy_dtype=np.dtype(np.float32).str,
        lower_bounds=np.array([-1.0, -1.0], dtype=np.float32),
        upper_bounds=np.array([1.0, 1.0], dtype=np.float32),
        environment_shape=(2,),
        coordinate_frame="joint_position_target",
        control_period_s=0.05,
    )


def _settings() -> ManiSkillPickCubeEnvironmentSettings:
    return ManiSkillPickCubeEnvironmentSettings(obs_mode="none", state_tolerance=1e-6)


def _identity() -> SolverSourceIdentity:
    return SolverSourceIdentity(
        module_name=OFFICIAL_SOLVER_MODULE,
        export_name=OFFICIAL_SOLVER_EXPORT,
        source_sha256="a" * 64,
    )


def _keys() -> PickCubeTaskKeyContract:
    return PickCubeTaskKeyContract(
        success="success",
        object_placed="is_obj_placed",
        robot_static="is_robot_static",
        grasped="is_grasped",
    )


def test_official_solver_filters_only_the_pinned_pose_deprecation() -> None:
    exact_message = (
        "component.pose can be ambiguous thus deprecated. It is equivalent to "
        "component.entity_pose, which should be used instead"
    )

    def solver(
        environment: object, *, seed: int, debug: bool, vis: bool
    ) -> tuple[object, int, bool, bool]:
        warnings.warn_explicit(
            exact_message,
            DeprecationWarning,
            filename="mani_skill/utils/geometry/trimesh_utils.py",
            lineno=111,
            module="mani_skill.utils.geometry.trimesh_utils",
        )
        return environment, seed, debug, vis

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run_official_solver(solver, "environment", seed=7) == (
            "environment",
            7,
            False,
            False,
        )

    def changed_warning(
        environment: object, *, seed: int, debug: bool, vis: bool
    ) -> None:
        del environment, seed, debug, vis
        warnings.warn_explicit(
            exact_message + " changed",
            DeprecationWarning,
            filename="mani_skill/utils/geometry/trimesh_utils.py",
            lineno=111,
            module="mani_skill.utils.geometry.trimesh_utils",
        )

    with warnings.catch_warnings(), pytest.raises(DeprecationWarning):
        warnings.simplefilter("error")
        run_official_solver(changed_warning, "environment", seed=7)


class _Joint:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _Robot:
    def __init__(self, environment: _Environment) -> None:
        self._environment = environment

    def get_active_joints(self) -> list[_Joint]:
        return [_Joint("joint_a"), _Joint("joint_b")]

    def get_qpos(self) -> np.ndarray[Any, Any]:
        return self._environment.state["robot"][None, :]

    def get_qvel(self) -> np.ndarray[Any, Any]:
        return np.zeros((1, 2), dtype=np.float32)


class _Environment:
    def __init__(
        self,
        *,
        terminal_success: bool = True,
        mutate_action: bool = False,
        elapsed_offset: int = 0,
        restore_offset: float = 0.0,
        restore_structure_mismatch: bool = False,
    ) -> None:
        self.state = {"robot": np.zeros(2, dtype=np.float32)}
        self.elapsed = 0
        self.terminal_success = terminal_success
        self.mutate_action = mutate_action
        self.elapsed_offset = elapsed_offset
        self.restore_offset = restore_offset
        self.restore_structure_mismatch = restore_structure_mismatch
        self.state_was_restored = False
        self.last_restored_state: dict[str, np.ndarray[Any, Any]] | None = None
        self.agent = type("Agent", (), {})()
        self.agent.robot = _Robot(self)
        self.closed = False

    @property
    def unwrapped(self) -> _Environment:
        return self

    def reset(self, *, seed: int) -> tuple[None, dict[str, object]]:
        self.state = {"robot": np.array([seed % 3, -(seed % 3)], dtype=np.float32)}
        self.elapsed = 0
        self.state_was_restored = False
        self.last_restored_state = None
        return None, {}

    def get_state_dict(self) -> dict[str, np.ndarray[Any, Any]]:
        state = {"robot": np.array(self.state["robot"], copy=True)}
        if self.state_was_restored and self.restore_structure_mismatch:
            state["unexpected"] = np.zeros(1, dtype=np.float32)
        return state

    def set_state_dict(self, state: dict[str, np.ndarray[Any, Any]]) -> None:
        restored = np.array(state["robot"], copy=True)
        restored[0] += np.float32(self.restore_offset)
        self.state = {"robot": restored}
        self.state_was_restored = True
        self.last_restored_state = self.get_state_dict()

    def step(
        self, action: np.ndarray[Any, Any]
    ) -> tuple[None, float, bool, bool, dict[str, np.ndarray[Any, Any]]]:
        self.elapsed += 1
        self.state["robot"] = self.state["robot"] + action
        if self.mutate_action:
            action[...] = 0
        return (
            None,
            0.0,
            False,
            False,
            {
                "elapsed_steps": np.array(
                    [self.elapsed + self.elapsed_offset], dtype=np.int64
                )
            },
        )

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(
        self,
        *,
        terminal_success: bool = True,
        mutate_action: bool = False,
        elapsed_offset: int = 0,
        missing_close: bool = False,
        restore_offset: float = 0.0,
        restore_structure_mismatch: bool = False,
    ) -> None:
        self.terminal_success = terminal_success
        self.mutate_action = mutate_action
        self.elapsed_offset = elapsed_offset
        self.missing_close = missing_close
        self.restore_offset = restore_offset
        self.restore_structure_mismatch = restore_structure_mismatch
        self.created: list[_Environment] = []

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        purpose: str,
    ) -> _Environment:
        assert settings == _settings()
        assert action_contract.total_dimension == 2
        assert purpose in {
            "official_source_generation",
            "independent_source_baseline",
        }
        environment = _Environment(
            terminal_success=self.terminal_success,
            mutate_action=self.mutate_action,
            elapsed_offset=self.elapsed_offset,
            restore_offset=self.restore_offset,
            restore_structure_mismatch=self.restore_structure_mismatch,
        )
        if self.missing_close:
            environment.close = None  # type: ignore[assignment, method-assign]
        self.created.append(environment)
        return environment

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        return state_tree

    def capture_task_snapshot(
        self,
        environment: object,
        key_contract: PickCubeTaskKeyContract,
    ) -> RawPickCubeTaskSnapshot:
        assert isinstance(environment, _Environment)
        del key_contract
        complete = environment.elapsed >= 2 and environment.terminal_success
        return RawPickCubeTaskSnapshot(
            evaluator_values={
                "success": np.array([complete]),
                "is_obj_placed": np.array([complete]),
                "is_robot_static": np.array([True]),
                "is_grasped": np.array([False]),
            },
            cube_center_z=0.02,
            cube_to_goal_distance=0.0 if complete else 0.2,
        )

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
        return (
            ("joint_a", "joint_b"),
            np.concatenate(
                (
                    environment.state["robot"],
                    np.zeros(2, dtype=np.float32),
                )
            ),
        )


def _solver(environment: object, *, seed: int, debug: bool, vis: bool) -> object:
    assert isinstance(environment, RecordingEnvironmentProxy)
    assert not debug and not vis
    environment.reset(seed=seed)
    result: object = None
    for action in (
        np.array([0.1, 0.0], dtype=np.float32),
        np.array([0.0, 0.1], dtype=np.float32),
    ):
        result = environment.step(action)
    return result


def _record(factory: _Factory):
    return record_official_source_trajectory(
        seed=7,
        environment_factory=factory,
        settings=_settings(),
        action_contract=_contract(),
        key_contract=_keys(),
        solver=_solver,
        solver_identity=_identity(),
    )


def test_ordered_source_seeds_are_bounded_and_deterministic() -> None:
    assert ordered_source_seeds(5, 4) == (5, 6, 7, 8)
    assert ordered_source_seeds(5, 4) == ordered_source_seeds(5, 4)
    with pytest.raises(PickCubeSourceGenerationError, match="exceeds uint32"):
        ordered_source_seeds(2**32 - 1, 2)


def test_recording_captures_actions_state_and_named_initial_robot_state() -> None:
    factory = _Factory()
    trajectory = _record(factory)

    assert trajectory.seed == 7
    assert trajectory.source_actions.shape == (2, 2)
    assert trajectory.joint_names == ("joint_a", "joint_b")
    assert np.array_equal(
        trajectory.robot_state, np.array([1, -1, 0, 0], dtype=np.float32)
    )
    assert (
        trajectory.source_action_digest
        == hashlib.sha256(trajectory.source_actions.tobytes(order="C")).hexdigest()
    )
    assert factory.created[0].closed


def test_recording_accepts_subtolerance_raw_state_drift() -> None:
    factory = _Factory(restore_offset=float(np.finfo(np.float32).eps))

    trajectory = _record(factory)

    restored = factory.created[0].last_restored_state
    assert restored is not None
    assert trajectory.initial_state_digest == compute_state_tree_digest(
        trajectory.initial_state
    )
    assert compute_state_tree_digest(restored) != trajectory.initial_state_digest


@pytest.mark.parametrize(
    "factory",
    (
        _Factory(restore_offset=2e-6),
        _Factory(restore_structure_mismatch=True),
    ),
    ids=("over-tolerance", "structure-mismatch"),
)
def test_recording_rejects_invalid_state_round_trip(factory: _Factory) -> None:
    with pytest.raises(
        PickCubeSourceGenerationError,
        match="complete structure within tolerance",
    ):
        _record(factory)


def test_recording_preserves_solver_dtype_distinct_from_environment_space() -> None:
    def float64_solver(
        environment: object, *, seed: int, debug: bool, vis: bool
    ) -> object:
        assert isinstance(environment, RecordingEnvironmentProxy)
        environment.reset(seed=seed)
        result: object = None
        for action in (
            np.array([0.1, 0.0], dtype=np.float64),
            np.array([0.0, 0.1], dtype=np.float64),
        ):
            result = environment.step(action)
        return result

    trajectory = record_official_source_trajectory(
        seed=7,
        environment_factory=_Factory(),
        settings=_settings(),
        action_contract=_contract(),
        key_contract=_keys(),
        solver=float64_solver,
        solver_identity=_identity(),
    )

    assert trajectory.source_actions.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        trajectory.source_actions,
        np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64),
    )


def test_recording_rejects_environment_action_mutation() -> None:
    with pytest.raises(PickCubeSourceGenerationError, match="mutated"):
        _record(_Factory(mutate_action=True))


def test_recording_rejects_incomplete_interception_count() -> None:
    with pytest.raises(PickCubeSourceGenerationError, match="incomplete"):
        _record(_Factory(elapsed_offset=1))


def test_recording_rejects_solver_reset_seed_drift() -> None:
    factory = _Factory()

    def wrong_seed_solver(
        environment: object, *, seed: int, debug: bool, vis: bool
    ) -> object:
        assert isinstance(environment, RecordingEnvironmentProxy)
        environment.reset(seed=seed + 1)
        result: object = None
        for action in (
            np.array([0.1, 0.0], dtype=np.float32),
            np.array([0.0, 0.1], dtype=np.float32),
        ):
            result = environment.step(action)
        return result

    with pytest.raises(PickCubeSourceGenerationError, match="reset seed differs"):
        record_official_source_trajectory(
            seed=7,
            environment_factory=factory,
            settings=_settings(),
            action_contract=_contract(),
            key_contract=_keys(),
            solver=wrong_seed_solver,
            solver_identity=_identity(),
        )

    assert factory.created[0].closed


def test_recording_rejects_solver_failure() -> None:
    def failed_solver(environment: object, *, seed: int, debug: bool, vis: bool) -> int:
        assert isinstance(environment, RecordingEnvironmentProxy)
        environment.reset(seed=seed)
        return -1

    with pytest.raises(PickCubeSourceGenerationError, match="motion-planning"):
        record_official_source_trajectory(
            seed=7,
            environment_factory=_Factory(),
            settings=_settings(),
            action_contract=_contract(),
            key_contract=_keys(),
            solver=failed_solver,
            solver_identity=_identity(),
        )


def test_recording_action_limit_aborts_before_extra_step() -> None:
    factory = _Factory()

    def excessive_solver(
        environment: object, *, seed: int, debug: bool, vis: bool
    ) -> object:
        assert isinstance(environment, RecordingEnvironmentProxy)
        environment.reset(seed=seed)
        result: object = None
        for _ in range(3):
            result = environment.step(np.zeros(2, dtype=np.float32))
        return result

    with pytest.raises(PickCubeTrajectoryActionLimitError, match="before step"):
        record_official_source_trajectory(
            seed=7,
            environment_factory=factory,
            settings=_settings(),
            action_contract=_contract(),
            key_contract=_keys(),
            solver=excessive_solver,
            solver_identity=_identity(),
            trajectory_action_limit=2,
        )

    assert factory.created[0].elapsed == 2
    assert factory.created[0].closed


def test_independent_baseline_uses_fresh_environment_and_preserves_source() -> None:
    generation_factory = _Factory()
    trajectory = _record(generation_factory)
    action_snapshot = trajectory.source_actions.tobytes(order="C")
    baseline_factory = _Factory()

    terminal = validate_independent_source_baseline(
        trajectory,
        environment_factory=baseline_factory,
        settings=_settings(),
        action_contract=_contract(),
        key_contract=_keys(),
    )

    assert terminal.evaluator_values["success"].item() is True
    assert baseline_factory.created[0] is not generation_factory.created[0]
    assert baseline_factory.created[0].closed
    assert trajectory.source_actions.tobytes(order="C") == action_snapshot


def test_independent_baseline_accepts_subtolerance_raw_state_drift() -> None:
    trajectory = _record(_Factory())
    state_digest = compute_state_tree_digest(trajectory.initial_state)
    baseline_factory = _Factory(restore_offset=float(np.finfo(np.float32).eps))

    terminal = validate_independent_source_baseline(
        trajectory,
        environment_factory=baseline_factory,
        settings=_settings(),
        action_contract=_contract(),
        key_contract=_keys(),
    )

    restored = baseline_factory.created[0].last_restored_state
    assert restored is not None
    assert terminal.evaluator_values["success"].item() is True
    assert compute_state_tree_digest(restored) != state_digest
    assert compute_state_tree_digest(trajectory.initial_state) == state_digest


@pytest.mark.parametrize(
    "baseline_factory",
    (
        _Factory(restore_offset=2e-6),
        _Factory(restore_structure_mismatch=True),
    ),
    ids=("over-tolerance", "structure-mismatch"),
)
def test_independent_baseline_rejects_invalid_state_round_trip(
    baseline_factory: _Factory,
) -> None:
    trajectory = _record(_Factory())

    with pytest.raises(
        PickCubeSourceGenerationError,
        match="complete structure within tolerance",
    ):
        validate_independent_source_baseline(
            trajectory,
            environment_factory=baseline_factory,
            settings=_settings(),
            action_contract=_contract(),
            key_contract=_keys(),
        )


def test_failed_independent_baseline_is_rejected() -> None:
    trajectory = _record(_Factory())
    with pytest.raises(PickCubeSourceGenerationError, match="task success"):
        validate_independent_source_baseline(
            trajectory,
            environment_factory=_Factory(terminal_success=False),
            settings=_settings(),
            action_contract=_contract(),
            key_contract=_keys(),
        )


def test_independent_baseline_requires_callable_close() -> None:
    trajectory = _record(_Factory())
    baseline_factory = _Factory(missing_close=True)

    with pytest.raises(PickCubeSourceGenerationError, match="required callable close"):
        validate_independent_source_baseline(
            trajectory,
            environment_factory=baseline_factory,
            settings=_settings(),
            action_contract=_contract(),
            key_contract=_keys(),
        )

    assert baseline_factory.created[0].closed is False


def test_collection_uses_ordered_seeds_and_accepts_successes_only() -> None:
    def selective_solver(
        environment: object, *, seed: int, debug: bool, vis: bool
    ) -> object:
        if seed == 7:
            assert isinstance(environment, RecordingEnvironmentProxy)
            environment.reset(seed=seed)
            return -1
        return _solver(environment, seed=seed, debug=debug, vis=vis)

    result = collect_reference_archive(
        requested_success_count=2,
        starting_seed=7,
        maximum_attempts=3,
        compatibility_identity="sha256:" + "b" * 64,
        environment_factory=_Factory(),
        settings=_settings(),
        action_contract=_contract(),
        key_contract=_keys(),
        solver=selective_solver,
        solver_identity=_identity(),
    )

    assert tuple(attempt.seed for attempt in result.attempts) == (7, 8, 9)
    assert tuple(attempt.accepted for attempt in result.attempts) == (
        False,
        True,
        True,
    )
    assert result.accepted_count == 2
    assert result.archive is not None
    assert len(result.archive.episodes) == 2


def test_collection_records_action_limit_as_bounded_failed_attempt() -> None:
    def first_attempt_exceeds_limit(
        environment: object, *, seed: int, debug: bool, vis: bool
    ) -> object:
        if seed != 7:
            return _solver(environment, seed=seed, debug=debug, vis=vis)
        assert isinstance(environment, RecordingEnvironmentProxy)
        environment.reset(seed=seed)
        result: object = None
        for _ in range(3):
            result = environment.step(np.zeros(2, dtype=np.float32))
        return result

    factory = _Factory()
    result = collect_reference_archive(
        requested_success_count=1,
        starting_seed=7,
        maximum_attempts=2,
        compatibility_identity="sha256:" + "b" * 64,
        environment_factory=factory,
        settings=_settings(),
        action_contract=_contract(),
        key_contract=_keys(),
        solver=first_attempt_exceeds_limit,
        solver_identity=_identity(),
        trajectory_action_limit=2,
    )

    assert tuple(attempt.seed for attempt in result.attempts) == (7, 8)
    assert result.attempts[0].accepted is False
    assert result.attempts[0].failure_category == "PickCubeTrajectoryActionLimitError"
    assert result.attempts[1].accepted is True
    assert factory.created[0].elapsed == 2
    assert factory.created[0].closed


def test_collection_failure_does_not_publish_partial_archive() -> None:
    with pytest.raises(PickCubeSourceCollectionIncompleteError) as captured:
        collect_reference_archive(
            requested_success_count=2,
            starting_seed=1,
            maximum_attempts=2,
            compatibility_identity="sha256:" + "b" * 64,
            environment_factory=_Factory(terminal_success=False),
            settings=_settings(),
            action_contract=_contract(),
            key_contract=_keys(),
            solver=_solver,
            solver_identity=_identity(),
        )

    assert captured.value.result.archive is None
    assert captured.value.result.accepted_count == 0
