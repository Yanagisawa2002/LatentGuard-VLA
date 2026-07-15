"""Official-solver recording and bounded PickCube source generation."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.models import ReplayExecutionRole

from .archive import ManiSkillReferenceArchive, ManiSkillReferenceEpisode
from .session import (
    LazyManiSkillPickCubeRuntime,
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from .state_tree import (
    clone_state_tree,
    compare_state_trees,
    compute_state_tree_digest,
)
from .task_evidence import (
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
    build_pickcube_task_evidence,
)

OFFICIAL_SOLVER_MODULE = "mani_skill.examples.motionplanning.panda.solutions.pick_cube"
OFFICIAL_SOLVER_EXPORT = "solve"
OFFICIAL_SOURCE_POLICY_ID = "maniskill/official-panda-motionplanning-pickcube-v1"
ROBOT_STATE_SEMANTIC = "panda_active_joint_qpos_qvel_named_v1"


class PickCubeSourceGenerationError(RuntimeError):
    """Raised when official-source recording or baseline validation fails."""


class PickCubeTrajectoryActionLimitError(PickCubeSourceGenerationError):
    """Raised before an official solver can execute beyond its action limit."""


class PickCubeSourceCollectionIncompleteError(PickCubeSourceGenerationError):
    """Raised when bounded attempts cannot produce all requested references."""

    def __init__(self, result: SourceCollectionResult) -> None:
        self.result = result
        super().__init__(
            "bounded official source collection accepted "
            f"{result.accepted_count}/{result.requested_success_count} trajectories"
        )


def _immutable_array(value: object, *, context: str) -> NDArray[Any]:
    array = np.array(value, copy=True, order="C", subok=False)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise PickCubeSourceGenerationError(f"{context} must be numeric")
    if not np.all(np.isfinite(array)):
        raise PickCubeSourceGenerationError(f"{context} must be finite")
    immutable = array.tobytes(order="C")
    return np.frombuffer(immutable, dtype=array.dtype).reshape(array.shape)


def _runtime_array(value: object, *, context: str) -> NDArray[Any]:
    candidate = value
    for method_name in ("detach", "cpu"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    return _immutable_array(candidate, context=context)


def _base_environment(environment: object) -> object:
    return getattr(environment, "unwrapped", environment)


def _required_method(environment: object, name: str) -> Callable[..., Any]:
    method = getattr(_base_environment(environment), name, None)
    if not callable(method):
        raise PickCubeSourceGenerationError(
            f"PickCube environment is missing required callable {name}"
        )
    return cast(Callable[..., Any], method)


def _scalar_int(value: object, *, context: str) -> int:
    array = _runtime_array(value, context=context)
    if array.size != 1:
        raise PickCubeSourceGenerationError(f"{context} must contain one scalar")
    scalar = array.reshape(()).item()
    if type(scalar) not in (int, np.int32, np.int64):
        if not isinstance(scalar, int):
            raise PickCubeSourceGenerationError(f"{context} must be an integer")
    result = int(scalar)
    if result < 0:
        raise PickCubeSourceGenerationError(f"{context} must be non-negative")
    return result


def _trajectory_action_limit(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise PickCubeSourceGenerationError(
            "trajectory action limit must be a positive integer or null"
        )
    return value


class RecordingEnvironmentProxy:
    """Narrow proxy that records every official solver reset and step call."""

    def __init__(
        self,
        environment: object,
        action_contract: PickCubeReplayActionContract,
        *,
        trajectory_action_limit: int | None = None,
    ) -> None:
        """Wrap one environment without altering solver-visible attributes."""
        self._environment = environment
        self._action_contract = action_contract
        self._trajectory_action_limit = _trajectory_action_limit(
            trajectory_action_limit
        )
        self._action_limit_exceeded = False
        self._initial_state: object | None = None
        self._actions: list[NDArray[Any]] = []
        self._reset_seed: int | None = None
        self._reset_count = 0
        self._last_step_result: object | None = None
        self._closed = False

    @property
    def unwrapped(self) -> object:
        """Expose the same underlying task object expected by the official solver."""
        return _base_environment(self._environment)

    @property
    def reset_seed(self) -> int:
        """Return the sole explicitly recorded reset seed."""
        if self._reset_seed is None:
            raise PickCubeSourceGenerationError(
                "official solver did not reset through the recording wrapper"
            )
        return self._reset_seed

    @property
    def initial_state(self) -> object:
        """Return a detached copy of the reset-boundary state tree."""
        if self._initial_state is None:
            raise PickCubeSourceGenerationError(
                "official solver did not expose a reset-boundary state"
            )
        return clone_state_tree(self._initial_state)

    @property
    def actions(self) -> tuple[NDArray[Any], ...]:
        """Return immutable copies of all intercepted action rows."""
        return tuple(
            _immutable_array(action, context="recorded action")
            for action in self._actions
        )

    @property
    def last_step_result(self) -> object:
        """Return the final unchanged environment step result."""
        if self._last_step_result is None:
            raise PickCubeSourceGenerationError(
                "official solver did not execute an intercepted action"
            )
        return self._last_step_result

    def reset(self, *args: object, **kwargs: object) -> object:
        """Forward exactly one seeded reset and capture the resulting state."""
        if self._closed:
            raise PickCubeSourceGenerationError("recording wrapper is closed")
        self._reset_count += 1
        if self._reset_count != 1:
            raise PickCubeSourceGenerationError(
                "official solver performed more than one reset"
            )
        if args:
            raise PickCubeSourceGenerationError(
                "official solver reset seed must be passed by keyword"
            )
        seed = kwargs.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise PickCubeSourceGenerationError(
                "official solver must reset with an explicit seed in [0, 2**32)"
            )
        reset = getattr(self._environment, "reset", None)
        if not callable(reset):
            raise PickCubeSourceGenerationError(
                "PickCube environment is missing required callable reset"
            )
        result = reset(**kwargs)
        state = _required_method(self._environment, "get_state_dict")()
        self._initial_state = clone_state_tree(state)
        self._reset_seed = seed
        return result

    def step(self, action: object) -> object:
        """Forward one unchanged detached action and verify post-call immutability."""
        if self._closed:
            raise PickCubeSourceGenerationError("recording wrapper is closed")
        if (
            self._trajectory_action_limit is not None
            and len(self._actions) >= self._trajectory_action_limit
        ):
            self._action_limit_exceeded = True
            raise PickCubeTrajectoryActionLimitError(
                "official solver trajectory action limit exceeded before step"
            )
        source = _immutable_array(action, context="official solver action")
        if source.shape != self._action_contract.row_shape:
            raise PickCubeSourceGenerationError(
                "official solver action shape differs from the probed contract"
            )
        self._action_contract.validate_row(source)
        source_snapshot = source.tobytes(order="C")
        forwarded = np.array(source, copy=True, order="C")
        forwarded_snapshot = forwarded.tobytes(order="C")
        step = getattr(self._environment, "step", None)
        if not callable(step):
            raise PickCubeSourceGenerationError(
                "PickCube environment is missing required callable step"
            )
        result = step(forwarded)
        if source.tobytes(order="C") != source_snapshot:
            raise PickCubeSourceGenerationError(
                "official solver action changed during interception"
            )
        if forwarded.tobytes(order="C") != forwarded_snapshot:
            raise PickCubeSourceGenerationError(
                "environment mutated the forwarded official solver action"
            )
        self._actions.append(source)
        self._last_step_result = result
        return result

    def verify_interception_complete(self) -> None:
        """Bind recorded steps to the official elapsed-step value."""
        if self._action_limit_exceeded:
            raise PickCubeTrajectoryActionLimitError(
                "official solver trajectory action limit was exceeded"
            )
        if self._reset_count != 1 or not self._actions:
            raise PickCubeSourceGenerationError(
                "official solver recording is incomplete"
            )
        result = self.last_step_result
        if not isinstance(result, Sequence) or len(result) != 5:
            raise PickCubeSourceGenerationError(
                "official solver final step result must be a Gymnasium 5-tuple"
            )
        info = result[4]
        if not isinstance(info, Mapping) or "elapsed_steps" not in info:
            raise PickCubeSourceGenerationError(
                "official solver result lacks elapsed_steps interception evidence"
            )
        elapsed = _scalar_int(info["elapsed_steps"], context="elapsed_steps")
        if elapsed != len(self._actions):
            raise PickCubeSourceGenerationError(
                "official solver action interception count is incomplete"
            )

    def close(self) -> None:
        """Close the underlying environment at most once."""
        if self._closed:
            return
        close = getattr(self._environment, "close", None)
        if not callable(close):
            raise PickCubeSourceGenerationError(
                "PickCube environment is missing required callable close"
            )
        try:
            close()
        finally:
            self._closed = True

    def __getattr__(self, name: str) -> object:
        """Forward solver-required public attributes to the wrapped environment."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._environment, name)


@dataclass(frozen=True, slots=True)
class SolverSourceIdentity:
    """Path-independent identity of the installed official source solver."""

    module_name: str
    export_name: str
    source_sha256: str

    def __post_init__(self) -> None:
        """Validate the fixed official module/export and SHA-256 value."""
        if self.module_name != OFFICIAL_SOLVER_MODULE:
            raise PickCubeSourceGenerationError("unexpected PickCube solver module")
        if self.export_name != OFFICIAL_SOLVER_EXPORT:
            raise PickCubeSourceGenerationError("unexpected PickCube solver export")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.source_sha256
            )
        ):
            raise PickCubeSourceGenerationError(
                "solver source digest must be 64 lowercase hexadecimal characters"
            )

    def as_mapping(self) -> Mapping[str, str]:
        """Return canonical path-independent solver identity."""
        return MappingProxyType(
            {
                "export_name": self.export_name,
                "module_name": self.module_name,
                "source_sha256": self.source_sha256,
            }
        )


def load_official_solver() -> tuple[Callable[..., object], SolverSourceIdentity]:
    """Lazily import and content-bind the packaged official PickCube solver."""
    try:
        module = importlib.import_module(OFFICIAL_SOLVER_MODULE)
    except Exception as exc:
        raise PickCubeSourceGenerationError(
            "official ManiSkill PickCube solver module is unavailable"
        ) from exc
    solver = getattr(module, OFFICIAL_SOLVER_EXPORT, None)
    if not callable(solver):
        raise PickCubeSourceGenerationError(
            "official ManiSkill PickCube solver export is not callable"
        )
    source_path = inspect.getsourcefile(solver)
    if source_path is None:
        raise PickCubeSourceGenerationError(
            "official ManiSkill PickCube solver source file is unavailable"
        )
    try:
        source = open(source_path, "rb").read()
    except OSError as exc:
        raise PickCubeSourceGenerationError(
            "official ManiSkill PickCube solver source could not be read"
        ) from exc
    identity = SolverSourceIdentity(
        module_name=OFFICIAL_SOLVER_MODULE,
        export_name=OFFICIAL_SOLVER_EXPORT,
        source_sha256=hashlib.sha256(source).hexdigest(),
    )
    return solver, identity


@runtime_checkable
class SourceEnvironmentFactory(Protocol):
    """Create fixed-scope environments for source generation and validation."""

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        purpose: str,
    ) -> object:
        """Create one fresh environment for the named bounded purpose."""
        ...

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        """Convert safe state leaves to the environment runtime representation."""
        ...

    def capture_task_snapshot(
        self,
        environment: object,
        key_contract: PickCubeTaskKeyContract,
    ) -> RawPickCubeTaskSnapshot:
        """Capture complete official task values and narrow geometry metrics."""
        ...

    def step_action(
        self,
        environment: object,
        action: NDArray[Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        """Execute one validated row using the probed runtime shape/device."""
        ...

    def extract_named_robot_state(
        self, environment: object
    ) -> tuple[tuple[str, ...], NDArray[Any]]:
        """Return active-joint names and qpos-then-qvel vector."""
        ...


class LazyManiSkillSourceEnvironmentFactory:
    """Production source factory backed by the lazy replay runtime bridge."""

    def __init__(self) -> None:
        """Create the bridge without importing any optional simulator module."""
        self._runtime = LazyManiSkillPickCubeRuntime()

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        purpose: str,
    ) -> object:
        """Create one fixed real environment; purpose is audit-only."""
        if purpose not in {
            "official_source_generation",
            "independent_source_baseline",
        }:
            raise PickCubeSourceGenerationError(
                f"unsupported source environment purpose {purpose!r}"
            )
        return self._runtime.create_environment(
            settings,
            action_contract,
            execution_role=ReplayExecutionRole.BASELINE,
        )

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        """Convert safe NumPy state leaves to the environment device."""
        return self._runtime.prepare_state_tree(environment, state_tree)

    def capture_task_snapshot(
        self,
        environment: object,
        key_contract: PickCubeTaskKeyContract,
    ) -> RawPickCubeTaskSnapshot:
        """Capture official terminal values through the replay runtime bridge."""
        return self._runtime.capture_task_snapshot(environment, key_contract)

    def step_action(
        self,
        environment: object,
        action: NDArray[Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        """Execute one source row without clipping or semantic conversion."""
        self._runtime.step_action(environment, action, action_contract)

    def extract_named_robot_state(
        self, environment: object
    ) -> tuple[tuple[str, ...], NDArray[Any]]:
        """Extract named active-joint qpos then qvel at the restored state."""
        base = _base_environment(environment)
        agent = getattr(base, "agent", None)
        robot = getattr(agent, "robot", None)
        get_joints = getattr(robot, "get_active_joints", None)
        get_qpos = getattr(robot, "get_qpos", None)
        get_qvel = getattr(robot, "get_qvel", None)
        if not callable(get_joints) or not callable(get_qpos) or not callable(get_qvel):
            raise PickCubeSourceGenerationError(
                "Panda runtime does not expose public active-joint state APIs"
            )
        names: list[str] = []
        for joint in get_joints():
            get_name = getattr(joint, "get_name", None)
            name = get_name() if callable(get_name) else getattr(joint, "name", None)
            if not isinstance(name, str) or not name or name != name.strip():
                raise PickCubeSourceGenerationError(
                    "Panda active joint returned an invalid public name"
                )
            names.append(name)
        qpos = _runtime_array(get_qpos(), context="Panda qpos")
        qvel = _runtime_array(get_qvel(), context="Panda qvel")
        if qpos.size != len(names) or qvel.size != len(names):
            raise PickCubeSourceGenerationError(
                "Panda named joints do not match qpos/qvel dimensions"
            )
        vector = np.concatenate((qpos.reshape(-1), qvel.reshape(-1)))
        return tuple(names), _immutable_array(vector, context="Panda robot state")


@dataclass(frozen=True, slots=True)
class RecordedSourceTrajectory:
    """Detached official solver result before independent baseline acceptance."""

    seed: int
    source_actions: NDArray[Any]
    initial_state: object
    terminal_state: object
    terminal_task: RawPickCubeTaskSnapshot
    joint_names: tuple[str, ...]
    robot_state: NDArray[Any]
    solver_identity: SolverSourceIdentity
    initial_state_digest: str
    terminal_state_digest: str
    source_action_digest: str

    def __post_init__(self) -> None:
        """Detach arrays and validate all content digests."""
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise PickCubeSourceGenerationError("trajectory seed is invalid")
        actions = _immutable_array(
            self.source_actions, context="recorded source actions"
        )
        robot_state = _immutable_array(self.robot_state, context="robot state")
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise PickCubeSourceGenerationError(
                "recorded source actions must be a non-empty rank-2 array"
            )
        if robot_state.ndim != 1:
            raise PickCubeSourceGenerationError("robot state must be rank one")
        joint_names = tuple(self.joint_names)
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise PickCubeSourceGenerationError(
                "robot state requires unique recorded joint names"
            )
        if robot_state.shape != (2 * len(joint_names),):
            raise PickCubeSourceGenerationError(
                "robot state must contain named qpos followed by named qvel"
            )
        object.__setattr__(self, "source_actions", actions)
        object.__setattr__(self, "robot_state", robot_state)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "initial_state", clone_state_tree(self.initial_state))
        object.__setattr__(
            self, "terminal_state", clone_state_tree(self.terminal_state)
        )
        if compute_state_tree_digest(self.initial_state) != self.initial_state_digest:
            raise PickCubeSourceGenerationError("initial state digest mismatch")
        if compute_state_tree_digest(self.terminal_state) != self.terminal_state_digest:
            raise PickCubeSourceGenerationError("terminal state digest mismatch")
        expected_action_digest = hashlib.sha256(actions.tobytes(order="C")).hexdigest()
        if self.source_action_digest != expected_action_digest:
            raise PickCubeSourceGenerationError("source action digest mismatch")


@dataclass(frozen=True, slots=True)
class SourceAttemptRecord:
    """Compact deterministic result of one bounded source seed attempt."""

    seed: int
    accepted: bool
    failure_category: str | None

    def __post_init__(self) -> None:
        """Validate a credential-free, stack-free attempt summary."""
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise PickCubeSourceGenerationError("attempt seed is invalid")
        if type(self.accepted) is not bool:
            raise PickCubeSourceGenerationError("attempt acceptance must be boolean")
        if self.accepted != (self.failure_category is None):
            raise PickCubeSourceGenerationError(
                "accepted attempts cannot carry a failure category"
            )
        if self.failure_category is not None and (
            not self.failure_category
            or self.failure_category != self.failure_category.strip()
            or any(
                not (character.isalnum() or character in "_.-")
                for character in self.failure_category
            )
        ):
            raise PickCubeSourceGenerationError(
                "attempt failure category must be a safe type name"
            )


@dataclass(frozen=True, slots=True)
class SourceCollectionResult:
    """In-memory all-or-nothing collection result and bounded attempt audit."""

    requested_success_count: int
    attempts: tuple[SourceAttemptRecord, ...]
    archive: ManiSkillReferenceArchive | None

    def __post_init__(self) -> None:
        """Validate request, deterministic attempt order, and archive cardinality."""
        if (
            type(self.requested_success_count) is not int
            or self.requested_success_count <= 0
        ):
            raise PickCubeSourceGenerationError(
                "requested successful trajectory count must be positive"
            )
        attempts = tuple(self.attempts)
        object.__setattr__(self, "attempts", attempts)
        seeds = tuple(attempt.seed for attempt in attempts)
        if seeds and seeds != tuple(range(seeds[0], seeds[0] + len(seeds))):
            raise PickCubeSourceGenerationError(
                "source attempt seeds must be consecutive and ordered"
            )
        accepted = sum(attempt.accepted for attempt in attempts)
        if self.archive is None:
            if accepted >= self.requested_success_count:
                raise PickCubeSourceGenerationError(
                    "complete source collection is missing its archive"
                )
        elif (
            accepted != self.requested_success_count
            or len(self.archive.episodes) != self.requested_success_count
        ):
            raise PickCubeSourceGenerationError(
                "source archive count does not match accepted attempts"
            )

    @property
    def accepted_count(self) -> int:
        """Return independently baseline-validated accepted trajectory count."""
        return sum(attempt.accepted for attempt in self.attempts)


def _archive_task_evidence(
    snapshot: RawPickCubeTaskSnapshot,
    key_contract: PickCubeTaskKeyContract,
) -> Mapping[str, object]:
    evidence = build_pickcube_task_evidence(snapshot, key_contract)
    if evidence.status.value != "complete" or evidence.success is None:
        raise PickCubeSourceGenerationError(
            "accepted source requires complete terminal task evidence"
        )
    diagnostics = evidence.diagnostics
    required = {
        "is_obj_placed": diagnostics.get("pickcube_object_placed"),
        "is_robot_static": diagnostics.get("pickcube_robot_static"),
        "is_grasped": diagnostics.get("pickcube_grasped"),
        "cube_center_z": diagnostics.get("pickcube_cube_center_z"),
        "cube_to_goal_distance": diagnostics.get("pickcube_cube_to_goal_distance"),
    }
    if any(value is None for value in required.values()):
        raise PickCubeSourceGenerationError(
            "accepted source terminal diagnostics are incomplete"
        )
    return MappingProxyType({"success": evidence.success, **required})


def build_reference_episode(
    trajectory: RecordedSourceTrajectory,
    *,
    independently_validated_terminal: RawPickCubeTaskSnapshot,
    compatibility_identity: str,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
) -> ManiSkillReferenceEpisode:
    """Create one archive episode only after independent baseline validation."""
    if (
        not isinstance(compatibility_identity, str)
        or not compatibility_identity
        or compatibility_identity != compatibility_identity.strip()
    ):
        raise PickCubeSourceGenerationError(
            "collection requires a path-independent compatibility identity"
        )
    terminal_evidence = _archive_task_evidence(
        independently_validated_terminal, key_contract
    )
    if terminal_evidence["success"] is not True:
        raise PickCubeSourceGenerationError(
            "independent baseline must succeed before archive construction"
        )
    return ManiSkillReferenceEpisode(
        compatibility_identity=compatibility_identity,
        source_solver_identity={
            "module_name": trajectory.solver_identity.module_name,
            "source_sha256": (f"sha256:{trajectory.solver_identity.source_sha256}"),
        },
        environment_configuration=settings.as_mapping(),
        seed=trajectory.seed,
        source_actions=trajectory.source_actions,
        initial_state=trajectory.initial_state,
        terminal_state=trajectory.terminal_state,
        terminal_task_evidence=terminal_evidence,
        initial_robot_state=trajectory.robot_state,
        robot_state_joint_names=trajectory.joint_names,
        robot_state_semantic=ROBOT_STATE_SEMANTIC,
        action_contract=action_contract.as_mapping(),
        action_coordinate_frame=action_contract.coordinate_frame,
        control_period_s=action_contract.control_period_s,
        source_generation_success=True,
        independent_baseline_success=True,
    )


def collect_reference_archive(
    *,
    requested_success_count: int,
    starting_seed: int,
    maximum_attempts: int,
    compatibility_identity: str,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
    solver: Callable[..., object],
    solver_identity: SolverSourceIdentity,
    trajectory_action_limit: int | None = None,
) -> SourceCollectionResult:
    """Collect accepted sources with deterministic seeds and a strict attempt cap."""
    if type(requested_success_count) is not int or requested_success_count <= 0:
        raise PickCubeSourceGenerationError(
            "requested successful trajectory count must be positive"
        )
    if requested_success_count > maximum_attempts:
        raise PickCubeSourceGenerationError(
            "requested successes cannot exceed maximum attempts"
        )
    action_limit = _trajectory_action_limit(trajectory_action_limit)
    attempts: list[SourceAttemptRecord] = []
    accepted: list[ManiSkillReferenceEpisode] = []
    for seed in ordered_source_seeds(starting_seed, maximum_attempts):
        try:
            recorded = record_official_source_trajectory(
                seed=seed,
                environment_factory=environment_factory,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
                solver=solver,
                solver_identity=solver_identity,
                trajectory_action_limit=action_limit,
            )
            baseline_terminal = validate_independent_source_baseline(
                recorded,
                environment_factory=environment_factory,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
            )
            reference = build_reference_episode(
                recorded,
                independently_validated_terminal=baseline_terminal,
                compatibility_identity=compatibility_identity,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
            )
        except Exception as exc:
            attempts.append(
                SourceAttemptRecord(
                    seed=seed,
                    accepted=False,
                    failure_category=type(exc).__name__,
                )
            )
            continue
        attempts.append(
            SourceAttemptRecord(seed=seed, accepted=True, failure_category=None)
        )
        accepted.append(reference)
        if len(accepted) == requested_success_count:
            return SourceCollectionResult(
                requested_success_count=requested_success_count,
                attempts=tuple(attempts),
                archive=ManiSkillReferenceArchive(episodes=tuple(accepted)),
            )
    incomplete = SourceCollectionResult(
        requested_success_count=requested_success_count,
        attempts=tuple(attempts),
        archive=None,
    )
    raise PickCubeSourceCollectionIncompleteError(incomplete)


def ordered_source_seeds(starting_seed: int, maximum_attempts: int) -> tuple[int, ...]:
    """Return deterministic bounded uint32 seeds without wraparound."""
    if type(starting_seed) is not int or not 0 <= starting_seed < 2**32:
        raise PickCubeSourceGenerationError(
            "starting seed must be an integer in [0, 2**32)"
        )
    if type(maximum_attempts) is not int or maximum_attempts <= 0:
        raise PickCubeSourceGenerationError("maximum attempts must be positive")
    if starting_seed + maximum_attempts > 2**32:
        raise PickCubeSourceGenerationError("source seed range exceeds uint32")
    return tuple(range(starting_seed, starting_seed + maximum_attempts))


def record_official_source_trajectory(
    *,
    seed: int,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
    solver: Callable[..., object],
    solver_identity: SolverSourceIdentity,
    trajectory_action_limit: int | None = None,
) -> RecordedSourceTrajectory:
    """Run one official solver attempt and return only complete successful output."""
    action_limit = _trajectory_action_limit(trajectory_action_limit)
    environment = environment_factory.create_environment(
        settings, action_contract, purpose="official_source_generation"
    )
    recorder = RecordingEnvironmentProxy(
        environment,
        action_contract,
        trajectory_action_limit=action_limit,
    )
    primary: BaseException | None = None
    try:
        result = solver(recorder, seed=seed, debug=False, vis=False)
        if result == -1:
            raise PickCubeSourceGenerationError(
                "official solver reported motion-planning failure"
            )
        recorder.verify_interception_complete()
        terminal_task = environment_factory.capture_task_snapshot(
            environment, key_contract
        )
        task_evidence = build_pickcube_task_evidence(terminal_task, key_contract)
        if (
            task_evidence.status.value != "complete"
            or task_evidence.success is not True
        ):
            raise PickCubeSourceGenerationError(
                "official solver terminal task evidence is not successful"
            )
        initial_state = recorder.initial_state
        terminal_state = clone_state_tree(
            _required_method(environment, "get_state_dict")()
        )
        prepared_initial = environment_factory.prepare_state_tree(
            environment, initial_state
        )
        _required_method(environment, "set_state_dict")(prepared_initial)
        restored_initial = clone_state_tree(
            _required_method(environment, "get_state_dict")()
        )
        initial_comparison = compare_state_trees(
            initial_state,
            restored_initial,
            atol=settings.state_tolerance,
        )
        if (
            not initial_comparison.structure_matches
            or not initial_comparison.within_tolerance
        ):
            raise PickCubeSourceGenerationError(
                "recorded source initial state did not round-trip with complete "
                "structure within tolerance"
            )
        joint_names, robot_state = environment_factory.extract_named_robot_state(
            environment
        )
        actions = np.stack(recorder.actions, axis=0)
        return RecordedSourceTrajectory(
            seed=seed,
            source_actions=actions,
            initial_state=initial_state,
            terminal_state=terminal_state,
            terminal_task=terminal_task,
            joint_names=joint_names,
            robot_state=robot_state,
            solver_identity=solver_identity,
            initial_state_digest=compute_state_tree_digest(initial_state),
            terminal_state_digest=compute_state_tree_digest(terminal_state),
            source_action_digest=hashlib.sha256(actions.tobytes(order="C")).hexdigest(),
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            recorder.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.add_note(
                "PickCube source environment close also failed: "
                f"{type(close_error).__name__}"
            )


def validate_independent_source_baseline(
    trajectory: RecordedSourceTrajectory,
    *,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
) -> RawPickCubeTaskSnapshot:
    """Restore and replay one source in a fresh session before M0 acceptance."""
    action_snapshot = trajectory.source_actions.tobytes(order="C")
    state_digest = compute_state_tree_digest(trajectory.initial_state)
    environment = environment_factory.create_environment(
        settings, action_contract, purpose="independent_source_baseline"
    )
    primary: BaseException | None = None
    try:
        reset = getattr(environment, "reset", None)
        if not callable(reset):
            raise PickCubeSourceGenerationError(
                "baseline environment is missing required callable reset"
            )
        reset(seed=trajectory.seed)
        prepared = environment_factory.prepare_state_tree(
            environment, trajectory.initial_state
        )
        _required_method(environment, "set_state_dict")(prepared)
        observed = clone_state_tree(_required_method(environment, "get_state_dict")())
        comparison = compare_state_trees(
            trajectory.initial_state,
            observed,
            atol=settings.state_tolerance,
        )
        if not comparison.structure_matches or not comparison.within_tolerance:
            raise PickCubeSourceGenerationError(
                "independent baseline state round trip did not preserve complete "
                "structure within tolerance"
            )
        for row in trajectory.source_actions:
            action_contract.validate_row(row)
            forwarded = np.array(row, copy=True, order="C")
            snapshot = forwarded.tobytes(order="C")
            environment_factory.step_action(environment, forwarded, action_contract)
            if forwarded.tobytes(order="C") != snapshot:
                raise PickCubeSourceGenerationError(
                    "baseline environment mutated a source action"
                )
        terminal_task = environment_factory.capture_task_snapshot(
            environment, key_contract
        )
        evidence = build_pickcube_task_evidence(terminal_task, key_contract)
        if evidence.status.value != "complete" or evidence.success is not True:
            raise PickCubeSourceGenerationError(
                "independent baseline replay did not establish task success"
            )
        if trajectory.source_actions.tobytes(order="C") != action_snapshot:
            raise PickCubeSourceGenerationError(
                "independent baseline changed archived source actions"
            )
        if compute_state_tree_digest(trajectory.initial_state) != state_digest:
            raise PickCubeSourceGenerationError(
                "independent baseline changed archived initial state"
            )
        return terminal_task
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close = getattr(environment, "close", None)
        if not callable(close):
            missing_close = PickCubeSourceGenerationError(
                "baseline environment is missing required callable close"
            )
            if primary is None:
                raise missing_close
            primary.add_note(str(missing_close))
        else:
            try:
                close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "PickCube baseline environment close also failed: "
                    f"{type(close_error).__name__}"
                )


__all__ = [
    "OFFICIAL_SOLVER_EXPORT",
    "OFFICIAL_SOLVER_MODULE",
    "OFFICIAL_SOURCE_POLICY_ID",
    "ROBOT_STATE_SEMANTIC",
    "LazyManiSkillSourceEnvironmentFactory",
    "PickCubeSourceCollectionIncompleteError",
    "PickCubeSourceGenerationError",
    "PickCubeTrajectoryActionLimitError",
    "RecordedSourceTrajectory",
    "RecordingEnvironmentProxy",
    "SourceAttemptRecord",
    "SourceCollectionResult",
    "SolverSourceIdentity",
    "SourceEnvironmentFactory",
    "build_reference_episode",
    "collect_reference_archive",
    "load_official_solver",
    "ordered_source_seeds",
    "record_official_source_trajectory",
    "validate_independent_source_baseline",
]
