"""PickCube exact-replay session with a lazy ManiSkill runtime boundary."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from latentguard.models import ActionChunk
from latentguard.replay.base import (
    ReplayEnvironmentSession,
    ReplayExecutionError,
    ReplayInvalidContextError,
)
from latentguard.replay.models import (
    ReplayCase,
    ReplayExecutionRole,
    ReplayStateReference,
    ReplayTaskReference,
    StateComparisonSemantic,
    StateMatchKind,
    StateRestorationEvidence,
    TerminalTaskEvidence,
)

from .configuration import (
    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
)
from .task_evidence import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_TASK_CONTRACT_VERSION,
    PICKCUBE_TASK_ID,
    PICKCUBE_UNSAFE_SEMANTIC,
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
    build_pickcube_task_evidence,
)

PICKCUBE_SOURCE_RESET_SEED_METADATA_KEY = "source_reset_seed"


class ManiSkillPickCubeSessionError(ReplayExecutionError):
    """Raised when a PickCube replay session cannot execute its contract."""


class ManiSkillIntegrationUnavailableError(ManiSkillPickCubeSessionError):
    """Raised when optional ManiSkill runtime dependencies are unavailable."""


@dataclass(frozen=True, slots=True)
class ManiSkillPickCubeEnvironmentSettings:
    """Fixed M2C environment settings, excluding all runtime paths."""

    obs_mode: str
    state_tolerance: float
    state_verification_semantic: str = PICKCUBE_STATE_VERIFICATION_SEMANTIC
    environment_id: str = "PickCube-v1"
    robot_uid: str = "panda"
    num_envs: int = 1
    control_mode: str = "pd_joint_pos"
    sim_backend: str = "gpu"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Reject every environment setting outside the fixed M2C scope."""
        expected = {
            "environment_id": "PickCube-v1",
            "robot_uid": "panda",
            "num_envs": 1,
            "control_mode": "pd_joint_pos",
            "sim_backend": "gpu",
            "schema_version": "1.0",
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ReplayInvalidContextError(
                    f"PickCube environment contract mismatch: {field}"
                )
        if (
            not isinstance(self.obs_mode, str)
            or not self.obs_mode
            or self.obs_mode != self.obs_mode.strip()
        ):
            raise ReplayInvalidContextError(
                "PickCube environment contract requires an explicit observation mode"
            )
        if type(self.state_tolerance) is not float or not math.isfinite(
            self.state_tolerance
        ):
            raise ReplayInvalidContextError(
                "PickCube state tolerance must be an explicit finite float"
            )
        if self.state_tolerance != PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE:
            raise ReplayInvalidContextError(
                "PickCube state tolerance must equal the fixed maximum absolute "
                f"tolerance {PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE!r}"
            )
        if self.state_verification_semantic != PICKCUBE_STATE_VERIFICATION_SEMANTIC:
            raise ReplayInvalidContextError(
                "PickCube state verification semantic mismatch"
            )

    def as_mapping(self) -> Mapping[str, object]:
        """Return deterministic environment semantics without runtime metadata."""
        return MappingProxyType(
            {
                "control_mode": self.control_mode,
                "environment_id": self.environment_id,
                "num_envs": self.num_envs,
                "obs_mode": self.obs_mode,
                "robot_uid": self.robot_uid,
                "schema_version": self.schema_version,
                "sim_backend": self.sim_backend,
                "state_tolerance": self.state_tolerance,
                "state_verification_semantic": self.state_verification_semantic,
            }
        )

    @classmethod
    def from_compatibility_binding(
        cls, binding: object
    ) -> ManiSkillPickCubeEnvironmentSettings:
        """Resolve fixed settings only from a trusted checked-in probe binding."""
        require_ready = getattr(binding, "require_trusted_replay_ready", None)
        if not callable(require_ready):
            raise ReplayInvalidContextError(
                "compatibility binding cannot authorize trusted replay"
            )
        require_ready()
        try:
            trusted_binding = cast(Any, binding)
            report = trusted_binding.report
            expected = trusted_binding.expected_contract
            return cls(
                obs_mode=str(report.observation_mode),
                state_tolerance=float(expected.state_round_trip_tolerance),
                environment_id=str(report.environment_id),
                robot_uid=str(report.robot_uid),
                num_envs=int(report.num_envs),
                control_mode=str(report.control_mode),
                sim_backend=str(report.sim_backend_request),
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ReplayInvalidContextError(
                "compatibility binding lacks fixed PickCube environment settings"
            ) from exc


@dataclass(frozen=True, slots=True)
class PickCubeReplayActionContract:
    """Probe-derived environment contract used before physical replay.

    The environment action-space dtype is not necessarily the dtype emitted by
    the official motion-planning solver. Recorded action rows keep their own
    content-bound floating dtype and are never coerced to this environment
    dtype.
    """

    total_dimension: int
    environment_numpy_dtype: str
    lower_bounds: NDArray[Any]
    upper_bounds: NDArray[Any]
    environment_shape: tuple[int, ...]
    coordinate_frame: str
    control_period_s: float
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Detach finite bounds and reject guessed or partial numeric contracts."""
        if self.schema_version != "1.0":
            raise ReplayInvalidContextError(
                "PickCube action contract uses an unsupported schema version"
            )
        if type(self.total_dimension) is not int or self.total_dimension <= 0:
            raise ReplayInvalidContextError(
                "PickCube action dimension must be a positive integer"
            )
        try:
            dtype = np.dtype(self.environment_numpy_dtype)
        except TypeError as exc:
            raise ReplayInvalidContextError(
                "PickCube action dtype is not a valid NumPy dtype"
            ) from exc
        if dtype.hasobject or not np.issubdtype(dtype, np.floating):
            raise ReplayInvalidContextError(
                "PickCube action dtype must be a non-object floating dtype"
            )
        if self.environment_numpy_dtype != dtype.str:
            raise ReplayInvalidContextError(
                "PickCube action dtype must use canonical NumPy dtype syntax"
            )
        lower = np.array(self.lower_bounds, copy=True, order="C", subok=False)
        upper = np.array(self.upper_bounds, copy=True, order="C", subok=False)
        if lower.shape != (self.total_dimension,) or upper.shape != (
            self.total_dimension,
        ):
            raise ReplayInvalidContextError(
                "PickCube action bounds must match the verified total dimension"
            )
        if lower.dtype.str != dtype.str or upper.dtype.str != dtype.str:
            raise ReplayInvalidContextError(
                "PickCube action bounds must use the verified action dtype"
            )
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ReplayInvalidContextError("PickCube action bounds must be finite")
        if np.any(lower > upper):
            raise ReplayInvalidContextError(
                "PickCube action lower bounds must not exceed upper bounds"
            )
        if self.environment_shape not in (
            (self.total_dimension,),
            (1, self.total_dimension),
        ):
            raise ReplayInvalidContextError(
                "PickCube action-space shape must be unbatched or one-env batched"
            )
        if (
            not isinstance(self.coordinate_frame, str)
            or not self.coordinate_frame
            or self.coordinate_frame != self.coordinate_frame.strip()
        ):
            raise ReplayInvalidContextError(
                "PickCube action coordinate frame must be explicit"
            )
        if (
            type(self.control_period_s) is not float
            or not math.isfinite(self.control_period_s)
            or self.control_period_s <= 0.0
        ):
            raise ReplayInvalidContextError(
                "PickCube action control period must be a positive finite float"
            )
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    @property
    def row_shape(self) -> tuple[int]:
        """Return the non-vectorized action-row shape stored by M0/M1."""
        return (self.total_dimension,)

    @classmethod
    def from_compatibility_report(
        cls,
        report: object,
        *,
        coordinate_frame: str,
    ) -> PickCubeReplayActionContract:
        """Build the numeric replay contract from one validated probe report."""
        try:
            compatibility_report = cast(Any, report)
            action_space = compatibility_report.action_space
            dimension = action_space.action_dimension
            dtype = np.dtype(action_space.dtype)
            lower = np.asarray(action_space.lower_bounds, dtype=dtype)
            upper = np.asarray(action_space.upper_bounds, dtype=dtype)
            environment_shape = tuple(action_space.batched_shape)
            control_period = float(compatibility_report.control_period_s)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ReplayInvalidContextError(
                "compatibility report lacks a complete action contract"
            ) from exc
        return cls(
            total_dimension=dimension,
            environment_numpy_dtype=dtype.str,
            lower_bounds=lower,
            upper_bounds=upper,
            environment_shape=environment_shape,
            coordinate_frame=coordinate_frame,
            control_period_s=control_period,
        )

    def as_mapping(self) -> Mapping[str, object]:
        """Return a path-independent JSON-shaped action contract."""
        return MappingProxyType(
            {
                "control_period_s": self.control_period_s,
                "coordinate_frame": self.coordinate_frame,
                "environment_shape": self.environment_shape,
                "lower_bounds": tuple(float(item) for item in self.lower_bounds),
                "environment_numpy_dtype": self.environment_numpy_dtype,
                "schema_version": self.schema_version,
                "total_dimension": self.total_dimension,
                "upper_bounds": tuple(float(item) for item in self.upper_bounds),
            }
        )

    def validate_action_chunk(self, action: ActionChunk, *, role: str) -> None:
        """Reject shape, semantics, non-floating data, or bounds mismatches."""
        if action.actions.shape[1:] != self.row_shape:
            raise ReplayInvalidContextError(f"{role}_action_shape_mismatch")
        if action.actions.dtype.hasobject or not np.issubdtype(
            action.actions.dtype, np.floating
        ):
            raise ReplayInvalidContextError(f"{role}_action_dtype_not_floating")
        if action.coordinate_frame != self.coordinate_frame:
            raise ReplayInvalidContextError(f"{role}_action_coordinate_frame_mismatch")
        if action.control_period_s != self.control_period_s:
            raise ReplayInvalidContextError(f"{role}_action_control_period_mismatch")
        if not np.all(np.isfinite(action.actions)):
            raise ReplayInvalidContextError(f"{role}_action_non_finite")
        if np.any(action.actions < self.lower_bounds) or np.any(
            action.actions > self.upper_bounds
        ):
            raise ReplayInvalidContextError(f"{role}_action_out_of_bounds")

    def validate_row(self, action: NDArray[Any]) -> None:
        """Revalidate one detached row without clipping or coercion."""
        if not isinstance(action, np.ndarray) or action.shape != self.row_shape:
            raise ReplayInvalidContextError("runtime_action_row_shape_mismatch")
        if action.dtype.hasobject or not np.issubdtype(action.dtype, np.floating):
            raise ReplayInvalidContextError("runtime_action_row_dtype_not_floating")
        if not np.all(np.isfinite(action)):
            raise ReplayInvalidContextError("runtime_action_row_non_finite")
        if np.any(action < self.lower_bounds) or np.any(action > self.upper_bounds):
            raise ReplayInvalidContextError("runtime_action_row_out_of_bounds")


@dataclass(frozen=True, slots=True)
class LoadedReferenceState:
    """One archive-loaded initial state detached from its runtime path."""

    source_reference_id: str
    source_reset_seed: int
    state_key: str
    state_digest: str
    state_tree: object
    compared_component_count: int

    def __post_init__(self) -> None:
        """Validate identity fields while leaving the normalized tree opaque."""
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in (
                self.source_reference_id,
                self.state_key,
                self.state_digest,
            )
        ):
            raise ReplayInvalidContextError(
                "loaded PickCube state has malformed identity fields"
            )
        if (
            type(self.compared_component_count) is not int
            or self.compared_component_count <= 0
        ):
            raise ReplayInvalidContextError(
                "loaded PickCube state must contain numeric components"
            )
        if (
            type(self.source_reset_seed) is not int
            or not 0 <= self.source_reset_seed < 2**32
        ):
            raise ReplayInvalidContextError(
                "loaded PickCube state has an invalid source reset seed"
            )


def require_pickcube_source_reset_seed(reference: ReplayStateReference) -> int:
    """Return the exact archive-bound seed used to initialize the Gym wrapper."""
    seed = reference.metadata.get(PICKCUBE_SOURCE_RESET_SEED_METADATA_KEY)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ReplayInvalidContextError(
            "PickCube replay reference requires a content-bound source reset seed"
        )
    return seed


@dataclass(frozen=True, slots=True)
class StateTreeComparison:
    """Complete scalar comparison between archived and restored state trees."""

    expected_digest: str
    observed_digest: str
    structure_matches: bool
    compared_component_count: int
    maximum_absolute_error: float

    def __post_init__(self) -> None:
        """Require finite scalar comparison evidence."""
        if type(self.structure_matches) is not bool:
            raise ManiSkillPickCubeSessionError(
                "state comparison structure flag must be boolean"
            )
        if (
            type(self.compared_component_count) is not int
            or self.compared_component_count <= 0
        ):
            raise ManiSkillPickCubeSessionError(
                "state comparison must cover at least one component"
            )
        if (
            type(self.maximum_absolute_error) is not float
            or not math.isfinite(self.maximum_absolute_error)
            or self.maximum_absolute_error < 0.0
        ):
            raise ManiSkillPickCubeSessionError(
                "state comparison maximum error must be finite and non-negative"
            )


class CanonicalPickCubeStateTreeComparator:
    """Adapt the safe state-tree implementation to scalar replay evidence."""

    def compare_state_trees(
        self,
        expected: object,
        observed: object,
        *,
        tolerance: float,
    ) -> StateTreeComparison:
        """Compare every state component without coercion or missing-leaf repair."""
        from .state_tree import compare_state_trees, flatten_state_tree

        comparison = compare_state_trees(expected, observed, atol=tolerance)
        expected_components = sum(
            int(leaf.value.size) for leaf in flatten_state_tree(expected)
        )
        if expected_components <= 0:
            raise ManiSkillPickCubeSessionError(
                "PickCube archived state has no numeric components"
            )
        maximum_error = comparison.maximum_absolute_error
        if maximum_error is None:
            maximum_error = tolerance + 1.0
        compared_components = comparison.compared_component_count
        if not comparison.structure_matches:
            compared_components = expected_components
        return StateTreeComparison(
            expected_digest=comparison.expected_digest,
            observed_digest=comparison.observed_digest,
            structure_matches=comparison.structure_matches,
            compared_component_count=compared_components,
            maximum_absolute_error=float(maximum_error),
        )


@runtime_checkable
class PickCubeReferenceStateLoader(Protocol):
    """Load one content-bound initial state without exposing archive paths."""

    def load_reference_state(
        self, reference: ReplayStateReference
    ) -> LoadedReferenceState:
        """Load and revalidate the state named by a replay reference."""
        ...


@dataclass(frozen=True, slots=True)
class ArchivePickCubeReferenceStateStore:
    """Load and validate initial states from one runtime archive directory."""

    runtime_archive_directory: Path

    def __post_init__(self) -> None:
        """Store an absolute runtime path without placing it in semantic config."""
        directory = Path(self.runtime_archive_directory).absolute()
        object.__setattr__(self, "runtime_archive_directory", directory)

    def load_reference_state(
        self, reference: ReplayStateReference
    ) -> LoadedReferenceState:
        """Load a digest-verified initial state and its full component inventory."""
        from .archive import ReferenceArchiveError, load_archive_state
        from .state_tree import flatten_state_tree

        source_reset_seed = require_pickcube_source_reset_seed(reference)
        if reference.state_key != "initial_state":
            raise ReplayInvalidContextError(
                "PickCube replay supports only the archived initial state"
            )
        try:
            loaded = load_archive_state(
                self.runtime_archive_directory,
                reference.source_reference_id,
                "initial_state",
            )
        except ReferenceArchiveError as exc:
            raise ReplayInvalidContextError(
                "PickCube runtime archive state failed validation"
            ) from exc
        component_count = sum(
            int(leaf.value.size) for leaf in flatten_state_tree(loaded.tree)
        )
        result = LoadedReferenceState(
            source_reference_id=loaded.episode_id,
            source_reset_seed=loaded.source_reset_seed,
            state_key=loaded.state_key,
            state_digest=loaded.state_digest,
            state_tree=loaded.tree,
            compared_component_count=component_count,
        )
        if result.source_reset_seed != source_reset_seed:
            raise ReplayInvalidContextError(
                "PickCube source reset seed differs from the validated runtime archive"
            )
        return result

    def validate_reference(self, reference: ReplayStateReference) -> None:
        """Reject a reference whose archive identity or digest changed."""
        loaded = self.load_reference_state(reference)
        if loaded.state_digest != reference.expected_state_digest:
            raise ReplayInvalidContextError(
                "PickCube replay reference digest does not match runtime archive"
            )


@runtime_checkable
class PickCubeStateTreeComparator(Protocol):
    """Compare complete normalized state trees using the archive semantic."""

    def compare_state_trees(
        self,
        expected: object,
        observed: object,
        *,
        tolerance: float,
    ) -> StateTreeComparison:
        """Return complete scalar restoration evidence."""
        ...


@runtime_checkable
class PickCubeRuntime(Protocol):
    """Lazy boundary around ManiSkill/SAPIEN/torch environment objects."""

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        execution_role: ReplayExecutionRole,
    ) -> object:
        """Create one fresh fixed-scope PickCube environment."""
        ...

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        """Move numeric leaves to the environment's expected device and type."""
        ...

    def reset_environment(self, environment: object, *, seed: int) -> None:
        """Initialize the public Gym wrapper with the archive-bound source seed."""
        ...

    def set_state_dict(self, environment: object, state_tree: object) -> None:
        """Restore one complete environment state tree."""
        ...

    def get_state_dict(self, environment: object) -> object:
        """Read one complete environment state tree."""
        ...

    def capture_task_snapshot(
        self,
        environment: object,
        key_contract: PickCubeTaskKeyContract,
    ) -> RawPickCubeTaskSnapshot:
        """Capture official task scalars and narrow geometry metrics."""
        ...

    def step_action(
        self,
        environment: object,
        action: NDArray[Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        """Execute exactly one detached action without repair or clipping."""
        ...

    def close_environment(self, environment: object) -> None:
        """Close one environment and surface close failures."""
        ...


class LazyManiSkillPickCubeRuntime:
    """Production runtime bridge whose optional imports occur only on use."""

    @staticmethod
    def _base_environment(environment: object) -> object:
        return getattr(environment, "unwrapped", environment)

    @classmethod
    def _method(cls, environment: object, name: str) -> Any:
        base = cls._base_environment(environment)
        method = getattr(base, name, None)
        if not callable(method):
            raise ManiSkillPickCubeSessionError(
                f"PickCube runtime is missing required callable {name}"
            )
        return method

    @staticmethod
    def _numeric_array(value: object, *, field: str) -> NDArray[Any]:
        candidate = value
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        to_numpy = getattr(candidate, "numpy", None)
        if callable(to_numpy):
            candidate = to_numpy()
        array = np.asarray(candidate)
        if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
            raise ManiSkillPickCubeSessionError(
                f"PickCube runtime {field} must be numeric"
            )
        if not np.all(np.isfinite(array)):
            raise ManiSkillPickCubeSessionError(
                f"PickCube runtime {field} must be finite"
            )
        return array

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        execution_role: ReplayExecutionRole,
    ) -> object:
        """Create and contract-check one real GPU PickCube environment lazily."""
        del execution_role
        try:
            importlib.import_module("mani_skill.envs")
            gymnasium = importlib.import_module("gymnasium")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ManiSkillIntegrationUnavailableError(
                "ManiSkill PickCube integration dependencies are unavailable; "
                "install the verified remote integration environment"
            ) from exc
        make = getattr(gymnasium, "make", None)
        if not callable(make):
            raise ManiSkillIntegrationUnavailableError(
                "gymnasium.make is unavailable in the integration environment"
            )
        environment = make(
            settings.environment_id,
            robot_uids=settings.robot_uid,
            num_envs=settings.num_envs,
            obs_mode=settings.obs_mode,
            control_mode=settings.control_mode,
            sim_backend=settings.sim_backend,
        )
        try:
            action_space = environment.action_space
            shape = tuple(int(item) for item in action_space.shape)
            dtype = np.dtype(action_space.dtype).str
            lower = np.asarray(action_space.low)
            upper = np.asarray(action_space.high)
            if shape != action_contract.environment_shape:
                raise ReplayInvalidContextError(
                    "runtime_environment_action_shape_mismatch"
                )
            if dtype != action_contract.environment_numpy_dtype:
                raise ReplayInvalidContextError(
                    "runtime_environment_action_dtype_mismatch"
                )
            expected_lower = action_contract.lower_bounds.reshape(shape)
            expected_upper = action_contract.upper_bounds.reshape(shape)
            if not np.array_equal(lower, expected_lower) or not np.array_equal(
                upper, expected_upper
            ):
                raise ReplayInvalidContextError(
                    "runtime_environment_action_bounds_mismatch"
                )
        except BaseException:
            try:
                environment.close()
            finally:
                raise
        return environment

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        """Convert normalized numeric leaves to torch tensors on the env device."""
        from .state_tree import clone_state_tree

        try:
            torch = importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ManiSkillIntegrationUnavailableError(
                "PyTorch is unavailable in the ManiSkill integration environment"
            ) from exc
        base = self._base_environment(environment)
        device = getattr(base, "device", None)
        if device is None:
            raise ManiSkillPickCubeSessionError(
                "PickCube environment does not expose its runtime device"
            )

        def convert(value: object) -> object:
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(convert(item) for item in value)
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, np.ndarray):
                return torch.as_tensor(np.array(value, copy=True), device=device)
            return value

        return convert(clone_state_tree(state_tree))

    def reset_environment(self, environment: object, *, seed: int) -> None:
        """Reset through the public wrapper before restoring archived state."""
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ReplayInvalidContextError(
                "PickCube runtime source reset seed is invalid"
            )
        reset = getattr(environment, "reset", None)
        if not callable(reset):
            raise ManiSkillPickCubeSessionError(
                "PickCube environment is missing required callable reset"
            )
        reset(seed=seed)

    def set_state_dict(self, environment: object, state_tree: object) -> None:
        """Call the verified public state restoration API."""
        self._method(environment, "set_state_dict")(state_tree)

    def get_state_dict(self, environment: object) -> object:
        """Call the verified public state readback API."""
        return self._method(environment, "get_state_dict")()

    def capture_task_snapshot(
        self,
        environment: object,
        key_contract: PickCubeTaskKeyContract,
    ) -> RawPickCubeTaskSnapshot:
        """Capture official evaluation, grasp, cube, and goal values."""
        evaluated = self._method(environment, "evaluate")()
        if not isinstance(evaluated, Mapping):
            raise ManiSkillPickCubeSessionError(
                "PickCube evaluate() did not return a mapping"
            )
        values = dict(evaluated)
        base = self._base_environment(environment)
        agent = getattr(base, "agent", None)
        cube = getattr(base, "cube", None)
        if key_contract.grasped not in values:
            is_grasping = getattr(agent, "is_grasping", None)
            if not callable(is_grasping) or cube is None:
                raise ManiSkillPickCubeSessionError(
                    "PickCube runtime cannot obtain the verified grasp status"
                )
            values[key_contract.grasped] = is_grasping(cube)
        cube_pose = getattr(cube, "pose", None)
        cube_position = self._numeric_array(
            getattr(cube_pose, "p", None), field="cube position"
        )
        goal_site = getattr(base, "goal_site", None)
        goal_pose = getattr(goal_site, "pose", None)
        goal_position = self._numeric_array(
            getattr(goal_pose, "p", None), field="goal position"
        )
        if cube_position.shape[-1:] != (3,) or goal_position.shape[-1:] != (3,):
            raise ManiSkillPickCubeSessionError(
                "PickCube cube and goal positions must end in xyz coordinates"
            )
        if cube_position.size != 3 or goal_position.size != 3:
            raise ManiSkillPickCubeSessionError(
                "PickCube num_envs=1 positions must contain exactly one xyz value"
            )
        cube_xyz = cube_position.reshape(3)
        goal_xyz = goal_position.reshape(3)
        return RawPickCubeTaskSnapshot(
            evaluator_values=values,
            cube_center_z=float(cube_xyz[2]),
            cube_to_goal_distance=float(np.linalg.norm(cube_xyz - goal_xyz)),
        )

    def step_action(
        self,
        environment: object,
        action: NDArray[Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        """Move one unchanged row to the GPU and call env.step exactly once."""
        try:
            torch = importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ManiSkillIntegrationUnavailableError(
                "PyTorch is unavailable in the ManiSkill integration environment"
            ) from exc
        base = self._base_environment(environment)
        device = getattr(base, "device", None)
        if device is None:
            raise ManiSkillPickCubeSessionError(
                "PickCube environment does not expose its runtime device"
            )
        shaped = np.array(action, copy=True).reshape(action_contract.environment_shape)
        runtime_action = torch.as_tensor(shaped, device=device)
        step = getattr(environment, "step", None)
        if not callable(step):
            raise ManiSkillPickCubeSessionError(
                "PickCube environment is missing required callable step"
            )
        step(runtime_action)

    def close_environment(self, environment: object) -> None:
        """Close one real environment and surface any close exception."""
        close = getattr(environment, "close", None)
        if not callable(close):
            raise ManiSkillPickCubeSessionError(
                "PickCube environment is missing required callable close"
            )
        close()


class ManiSkillPickCubeReplaySession:
    """One independently owned exact-state baseline or corrupted session."""

    def __init__(
        self,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        task_key_contract: PickCubeTaskKeyContract,
        state_loader: PickCubeReferenceStateLoader,
        state_comparator: PickCubeStateTreeComparator,
        runtime: PickCubeRuntime,
    ) -> None:
        """Create a fresh environment and bind all replay contracts."""
        if not isinstance(execution_role, ReplayExecutionRole):
            raise ReplayInvalidContextError("unsupported PickCube replay role")
        if replay_case.task_reference.task_id != PICKCUBE_TASK_ID:
            raise ReplayInvalidContextError("PickCube replay task ID mismatch")
        if (
            replay_case.task_reference.task_contract_version
            != PICKCUBE_TASK_CONTRACT_VERSION
        ):
            raise ReplayInvalidContextError("PickCube task contract version mismatch")
        if replay_case.progress_semantic != PICKCUBE_PROGRESS_SEMANTIC:
            raise ReplayInvalidContextError("PickCube progress semantic mismatch")
        if replay_case.unsafe_semantic != PICKCUBE_UNSAFE_SEMANTIC:
            raise ReplayInvalidContextError("PickCube unsafe semantic mismatch")
        state_reference = replay_case.state_reference
        if (
            state_reference.comparison_semantic
            is not StateComparisonSemantic.NUMERIC_TOLERANCE
            or state_reference.metadata.get("state_verification_semantic")
            != PICKCUBE_STATE_VERIFICATION_SEMANTIC
            or state_reference.metadata.get(
                "state_verification_maximum_absolute_tolerance"
            )
            != PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
        ):
            raise ReplayInvalidContextError(
                "PickCube replay state verification contract mismatch"
            )
        action = (
            replay_case.original_action
            if execution_role is ReplayExecutionRole.BASELINE
            else replay_case.transformed_action
        )
        action_contract.validate_action_chunk(action, role=execution_role.value)
        source_reset_seed = require_pickcube_source_reset_seed(state_reference)
        self._case = replay_case
        self._role = execution_role
        self._settings = settings
        self._action_contract = action_contract
        self._task_key_contract = task_key_contract
        self._state_loader = state_loader
        self._state_comparator = state_comparator
        self._runtime = runtime
        self._source_reset_seed = source_reset_seed
        self._expected_action = action
        self._restored = False
        self._closed = False
        self._executed_rows = 0
        self._environment = runtime.create_environment(
            settings, action_contract, execution_role=execution_role
        )

    @property
    def execution_role(self) -> ReplayExecutionRole:
        """Return the immutable role owned by this session."""
        return self._role

    def restore_state(
        self, reference: ReplayStateReference
    ) -> StateRestorationEvidence:
        """Set and read back the complete archived initial state exactly once."""
        self._require_open()
        if self._restored or self._executed_rows:
            raise ManiSkillPickCubeSessionError(
                "PickCube state restoration is allowed only once before stepping"
            )
        if reference != self._case.state_reference:
            raise ReplayInvalidContextError(
                "PickCube session received a different state reference"
            )
        loaded = self._state_loader.load_reference_state(reference)
        if (
            loaded.source_reference_id != reference.source_reference_id
            or loaded.state_key != reference.state_key
        ):
            raise ReplayInvalidContextError(
                "PickCube archive state identity does not match the replay reference"
            )
        if loaded.source_reset_seed != self._source_reset_seed:
            raise ReplayInvalidContextError(
                "PickCube source reset seed differs from the loaded archive state"
            )
        tolerance = (
            0.0
            if reference.comparison_semantic is StateComparisonSemantic.EXACT_DIGEST
            else self._settings.state_tolerance
        )
        if loaded.state_digest != reference.expected_state_digest:
            return StateRestorationEvidence(
                expected_state_digest=reference.expected_state_digest,
                observed_state_digest=loaded.state_digest,
                comparison_semantic=reference.comparison_semantic,
                comparison_tolerance=tolerance,
                compared_component_count=loaded.compared_component_count,
                maximum_absolute_error=tolerance + 1.0,
                match_kind=StateMatchKind.MISMATCH,
                restoration_verified=False,
                complete_state_comparison=False,
                diagnostics={
                    "pickcube_archive_reference_mismatch": True,
                    "pickcube_state_structure_matches": False,
                    "pickcube_state_verification_maximum_absolute_tolerance": (
                        PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
                    ),
                    "pickcube_state_verification_semantic": (
                        PICKCUBE_STATE_VERIFICATION_SEMANTIC
                    ),
                },
            )
        self._runtime.reset_environment(self._environment, seed=self._source_reset_seed)
        runtime_state = self._runtime.prepare_state_tree(
            self._environment, loaded.state_tree
        )
        self._runtime.set_state_dict(self._environment, runtime_state)
        observed_tree = self._runtime.get_state_dict(self._environment)
        comparison = self._state_comparator.compare_state_trees(
            loaded.state_tree, observed_tree, tolerance=tolerance
        )
        if comparison.expected_digest != loaded.state_digest:
            raise ManiSkillPickCubeSessionError(
                "state comparator changed the archived expected digest"
            )
        if comparison.compared_component_count != loaded.compared_component_count:
            raise ManiSkillPickCubeSessionError(
                "state comparator changed the archived component inventory"
            )
        exact = bool(
            comparison.structure_matches
            and comparison.expected_digest == comparison.observed_digest
            and comparison.maximum_absolute_error == 0.0
        )
        within_tolerance = bool(
            comparison.structure_matches
            and comparison.maximum_absolute_error <= tolerance
        )
        if exact:
            match_kind = StateMatchKind.EXACT
        elif (
            reference.comparison_semantic is StateComparisonSemantic.NUMERIC_TOLERANCE
            and within_tolerance
        ):
            match_kind = StateMatchKind.WITHIN_TOLERANCE
        else:
            match_kind = StateMatchKind.MISMATCH
        self._restored = match_kind is not StateMatchKind.MISMATCH
        return StateRestorationEvidence(
            expected_state_digest=reference.expected_state_digest,
            observed_state_digest=comparison.observed_digest,
            comparison_semantic=reference.comparison_semantic,
            comparison_tolerance=tolerance,
            compared_component_count=comparison.compared_component_count,
            maximum_absolute_error=comparison.maximum_absolute_error,
            match_kind=match_kind,
            restoration_verified=self._restored,
            complete_state_comparison=comparison.structure_matches,
            diagnostics={
                "pickcube_state_exact_digest_match": exact,
                "pickcube_state_structure_matches": comparison.structure_matches,
                "pickcube_state_verification_maximum_absolute_tolerance": (
                    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
                ),
                "pickcube_state_verification_semantic": (
                    PICKCUBE_STATE_VERIFICATION_SEMANTIC
                ),
                "pickcube_source_reset_seed": self._source_reset_seed,
            },
        )

    def evaluate_task(self, task: ReplayTaskReference) -> TerminalTaskEvidence:
        """Evaluate official PickCube scalars at initial or terminal state."""
        self._require_ready()
        if task != self._case.task_reference:
            raise ReplayInvalidContextError(
                "PickCube session received a different task reference"
            )
        requested = int(self._expected_action.actions.shape[0])
        if self._executed_rows not in (0, requested):
            raise ManiSkillPickCubeSessionError(
                "PickCube task evaluation requires an initial or terminal state"
            )
        snapshot = self._runtime.capture_task_snapshot(
            self._environment, self._task_key_contract
        )
        return build_pickcube_task_evidence(snapshot, self._task_key_contract)

    def step_action(self, action: NDArray[Any]) -> None:
        """Execute one exact detached row, detecting mutation by the runtime."""
        self._require_ready()
        if self._executed_rows >= self._expected_action.actions.shape[0]:
            raise ManiSkillPickCubeSessionError(
                "PickCube session received too many action rows"
            )
        self._action_contract.validate_row(action)
        expected = self._expected_action.actions[self._executed_rows]
        if not np.array_equal(action, expected):
            raise ReplayInvalidContextError(
                "PickCube action row does not match the content-bound replay case"
            )
        detached = np.array(action, copy=True, order="C", subok=False)
        snapshot = detached.tobytes(order="C")
        self._runtime.step_action(self._environment, detached, self._action_contract)
        if detached.tobytes(order="C") != snapshot:
            raise ManiSkillPickCubeSessionError(
                "PickCube runtime mutated a replay action row"
            )
        self._executed_rows += 1

    def close(self) -> None:
        """Close the independently owned environment exactly once."""
        if self._closed:
            raise ManiSkillPickCubeSessionError(
                "PickCube replay session was closed more than once"
            )
        self._closed = True
        self._runtime.close_environment(self._environment)

    def _require_open(self) -> None:
        if self._closed:
            raise ManiSkillPickCubeSessionError(
                "PickCube replay session is already closed"
            )

    def _require_ready(self) -> None:
        self._require_open()
        if not self._restored:
            raise ManiSkillPickCubeSessionError(
                "PickCube state must be verified before task evaluation or stepping"
            )


@dataclass(frozen=True, slots=True)
class ManiSkillPickCubeSessionFactory:
    """Create a new state-owning PickCube session for every replay role."""

    settings: ManiSkillPickCubeEnvironmentSettings
    action_contract: PickCubeReplayActionContract
    task_key_contract: PickCubeTaskKeyContract
    state_loader: PickCubeReferenceStateLoader
    state_comparator: PickCubeStateTreeComparator
    runtime: PickCubeRuntime

    def create_session(
        self,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
    ) -> ReplayEnvironmentSession:
        """Create one fresh session and one fresh physical environment."""
        return ManiSkillPickCubeReplaySession(
            replay_case,
            execution_role=execution_role,
            settings=self.settings,
            action_contract=self.action_contract,
            task_key_contract=self.task_key_contract,
            state_loader=self.state_loader,
            state_comparator=self.state_comparator,
            runtime=self.runtime,
        )


__all__ = [
    "ArchivePickCubeReferenceStateStore",
    "CanonicalPickCubeStateTreeComparator",
    "LazyManiSkillPickCubeRuntime",
    "LoadedReferenceState",
    "ManiSkillIntegrationUnavailableError",
    "ManiSkillPickCubeEnvironmentSettings",
    "ManiSkillPickCubeReplaySession",
    "ManiSkillPickCubeSessionError",
    "ManiSkillPickCubeSessionFactory",
    "PickCubeReferenceStateLoader",
    "PickCubeReplayActionContract",
    "PickCubeRuntime",
    "PICKCUBE_SOURCE_RESET_SEED_METADATA_KEY",
    "PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE",
    "PICKCUBE_STATE_VERIFICATION_SEMANTIC",
    "PickCubeStateTreeComparator",
    "StateTreeComparison",
    "require_pickcube_source_reset_seed",
]
