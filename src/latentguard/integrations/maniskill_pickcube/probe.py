"""Bounded real-runtime compatibility probe for ManiSkill PickCube-v1."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import inspect
import math
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType
from typing import Any, Protocol, cast

import numpy as np

from latentguard.evaluation.security import sanitize_operational_text
from latentguard.integrations.maniskill_pickcube.availability import (
    ManiSkillDistributionVersions,
    ManiSkillRuntimeModules,
    ModuleImporter,
    VersionResolver,
    require_maniskill_pickcube_runtime,
)
from latentguard.integrations.maniskill_pickcube.compatibility import (
    REQUIRED_PICKCUBE_TASK_MODULE,
    REQUIRED_SOURCE_SOLVER_MODULE,
    ActionSpaceContract,
    CompatibilityBinding,
    CompatibilityReport,
    ControllerComponentContract,
    ControllerContract,
    ManiSkillCompatibilityError,
    ModuleSourceIdentity,
    OperationalRuntimeMetadata,
    ProbeCheckResult,
    RuntimeApiContract,
    StateRoundTripResult,
    validate_compatibility_report,
    write_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    REQUIRED_CONTROL_MODE,
    REQUIRED_ENVIRONMENT_ID,
    REQUIRED_NUM_ENVS,
    REQUIRED_OBSERVATION_MODE,
    REQUIRED_ROBOT_UID,
    REQUIRED_SIM_BACKEND_REQUEST,
    ExpectedManiSkillPickCubeContract,
)
from latentguard.integrations.maniskill_pickcube.state_tree import (
    compare_state_trees,
    compute_state_tree_digest,
    compute_state_tree_structure_digest,
    normalize_state_tree,
)
from latentguard.replay.identity import canonical_json_bytes

SOURCE_SOLVER_MODULE = REQUIRED_SOURCE_SOLVER_MODULE
SOURCE_SOLVER_CALLABLE = "solve"
PICKCUBE_TASK_MODULE = REQUIRED_PICKCUBE_TASK_MODULE

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


class ManiSkillProbeRuntime(Protocol):
    """Injectable runtime boundary used by CPU-only compatibility tests."""

    def collect_report(
        self,
        *,
        reset_seed: int,
        state_tolerance: float,
    ) -> CompatibilityReport:
        """Inspect one bounded environment and return a sanitized report."""
        ...


@dataclass(frozen=True, slots=True)
class _OpenEnvironmentObservation:
    environment_id: str
    robot_uid: str
    num_envs: int
    observation_mode: str
    control_mode: str
    resolved_sim_backend: str
    gpu_simulation: bool
    action_space: ActionSpaceContract
    control_frequency_hz: float
    simulation_frequency_hz: float
    runtime_apis: RuntimeApiContract
    task_evaluator_keys: tuple[str, ...]
    task_implementation: ModuleSourceIdentity
    controller: ControllerContract
    state_tree_structure_digest: str
    state_round_trip: StateRoundTripResult
    bounded_action_step: ProbeCheckResult


@dataclass(frozen=True, slots=True)
class InstalledManiSkillProbeRuntime:
    """Default runtime inspector; construction itself imports no simulator package."""

    version_resolver: VersionResolver = importlib.metadata.version
    module_importer: ModuleImporter = importlib.import_module

    def collect_report(
        self,
        *,
        reset_seed: int,
        state_tolerance: float,
    ) -> CompatibilityReport:
        """Create one GPU environment, inspect it, step once, and close it."""
        if type(reset_seed) is not int or reset_seed < 0:
            raise ManiSkillCompatibilityError(
                "ManiSkill probe reset_seed: expected a non-negative integer"
            )
        if (
            type(state_tolerance) not in (int, float)
            or not math.isfinite(float(state_tolerance))
            or float(state_tolerance) < 0.0
        ):
            raise ManiSkillCompatibilityError(
                "ManiSkill probe state_tolerance: expected a non-negative finite number"
            )
        versions, modules = require_maniskill_pickcube_runtime(
            version_resolver=self.version_resolver,
            module_importer=self.module_importer,
        )
        operational = _operational_metadata(versions, modules)
        source_solver = _load_solver_identity(self.module_importer)
        environment = _create_environment(modules)
        observation: _OpenEnvironmentObservation | None = None
        primary_error: BaseException | None = None
        try:
            observation = _inspect_environment(
                environment,
                modules=modules,
                module_importer=self.module_importer,
                reset_seed=reset_seed,
                state_tolerance=float(state_tolerance),
            )
        except BaseException as exc:
            primary_error = exc

        close_result = _close_environment(environment)
        if primary_error is not None:
            if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                raise primary_error
            raise ManiSkillCompatibilityError(
                "ManiSkill compatibility inspection failed: "
                + sanitize_operational_text(primary_error)
            ) from primary_error
        if observation is None:
            raise ManiSkillCompatibilityError(
                "ManiSkill compatibility inspection produced no observation"
            )
        return CompatibilityReport(
            mani_skill_version=versions.mani_skill,
            sapien_version=versions.sapien,
            mplib_version=versions.mplib,
            environment_id=observation.environment_id,
            robot_uid=observation.robot_uid,
            num_envs=observation.num_envs,
            reset_seed=reset_seed,
            observation_mode=observation.observation_mode,
            control_mode=observation.control_mode,
            sim_backend_request=REQUIRED_SIM_BACKEND_REQUEST,
            resolved_sim_backend=observation.resolved_sim_backend,
            gpu_simulation=observation.gpu_simulation,
            action_space=observation.action_space,
            control_frequency_hz=observation.control_frequency_hz,
            simulation_frequency_hz=observation.simulation_frequency_hz,
            runtime_apis=observation.runtime_apis,
            task_evaluator_keys=observation.task_evaluator_keys,
            source_solver=source_solver,
            task_implementation=observation.task_implementation,
            controller=observation.controller,
            state_tree_structure_digest=observation.state_tree_structure_digest,
            state_round_trip=observation.state_round_trip,
            bounded_action_step=observation.bounded_action_step,
            environment_close=close_result,
            operational=operational,
        )


def probe_maniskill_pickcube(
    expected: ExpectedManiSkillPickCubeContract,
    *,
    runtime: ManiSkillProbeRuntime | None = None,
    report_path: Path | None = None,
    reset_seed: int = 0,
    require_trusted: bool = False,
) -> CompatibilityBinding:
    """Collect, optionally persist, and bind one bounded compatibility report."""
    selected_runtime = runtime or InstalledManiSkillProbeRuntime()
    report = selected_runtime.collect_report(
        reset_seed=reset_seed,
        state_tolerance=expected.state_round_trip_tolerance,
    )
    if report.reset_seed != reset_seed:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility report reset seed does not match the "
            "requested probe seed"
        )
    if report_path is not None:
        write_compatibility_report(report, report_path)
    return validate_compatibility_report(
        report,
        expected,
        require_trusted=require_trusted,
    )


def _operational_metadata(
    versions: ManiSkillDistributionVersions,
    modules: ManiSkillRuntimeModules,
) -> OperationalRuntimeMetadata:
    torch = cast(Any, modules.torch)
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: PyTorch CUDA API is unavailable"
        )
    if not bool(cuda.is_available()):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: one visible CUDA GPU is required"
        )
    device_count = int(cuda.device_count())
    if device_count != 1:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: exactly one visible CUDA GPU is required, "
            f"found {device_count}"
        )
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: PyTorch did not report a CUDA runtime"
        )
    capability_value = cuda.get_device_capability(0)
    if (
        not isinstance(capability_value, Sequence)
        or isinstance(capability_value, (str, bytes))
        or len(capability_value) != 2
    ):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: invalid GPU capability response"
        )
    major, minor = capability_value
    if type(major) is not int or type(minor) is not int:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: GPU capability must contain integers"
        )
    return OperationalRuntimeMetadata(
        python_version=platform.python_version(),
        torch_version=versions.torch,
        cuda_runtime_version=cuda_version,
        gpu_model=sanitize_operational_text(cuda.get_device_name(0)),
        gpu_capability=f"{major}.{minor}",
    )


def _load_solver_identity(module_importer: ModuleImporter) -> ModuleSourceIdentity:
    try:
        module = module_importer(SOURCE_SOLVER_MODULE)
    except Exception as exc:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: official PickCube solver module is "
            "unavailable"
        ) from exc
    solver = getattr(module, SOURCE_SOLVER_CALLABLE, None)
    if not callable(solver):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: official PickCube solve callable is "
            "unavailable"
        )
    if getattr(solver, "__module__", None) != SOURCE_SOLVER_MODULE:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: official PickCube solver module "
            "identity changed"
        )
    return _module_source_identity(module)


def _create_environment(modules: ManiSkillRuntimeModules) -> Any:
    gymnasium = cast(Any, modules.gymnasium)
    make = getattr(gymnasium, "make", None)
    if not callable(make):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: gymnasium.make is unavailable"
        )
    try:
        return make(
            REQUIRED_ENVIRONMENT_ID,
            robot_uids=REQUIRED_ROBOT_UID,
            num_envs=REQUIRED_NUM_ENVS,
            obs_mode=REQUIRED_OBSERVATION_MODE,
            control_mode=REQUIRED_CONTROL_MODE,
            sim_backend=REQUIRED_SIM_BACKEND_REQUEST,
            render_backend="none",
        )
    except Exception as exc:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: fixed PickCube GPU environment "
            "creation failed"
        ) from exc


def _inspect_environment(
    environment: Any,
    *,
    modules: ManiSkillRuntimeModules,
    module_importer: ModuleImporter,
    reset_seed: int,
    state_tolerance: float,
) -> _OpenEnvironmentObservation:
    base = getattr(environment, "unwrapped", environment)
    reset = getattr(environment, "reset", None)
    if not callable(reset):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: environment reset is not callable"
        )
    reset(seed=reset_seed)

    runtime_apis = RuntimeApiContract(
        get_state_dict_callable=callable(getattr(base, "get_state_dict", None)),
        set_state_dict_callable=callable(getattr(base, "set_state_dict", None)),
        evaluate_callable=callable(getattr(base, "evaluate", None)),
    )
    if not runtime_apis.complete:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: required state or evaluator API is missing"
        )

    action_space = _action_space_contract(environment, base)
    controller = _controller_contract(base, action_space.action_dimension)
    evaluation = base.evaluate()
    if not isinstance(evaluation, Mapping) or not all(
        isinstance(key, str) for key in evaluation
    ):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: evaluate() must return a string-keyed "
            "mapping"
        )
    task_keys = tuple(sorted(cast(Mapping[str, object], evaluation)))
    task_module_name = type(base).__module__
    if task_module_name != PICKCUBE_TASK_MODULE:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: PickCube task module identity changed"
        )
    task_module = module_importer(task_module_name)

    get_state = cast(Callable[[], object], base.get_state_dict)
    set_state = cast(Callable[[object], object], base.set_state_dict)
    initial_runtime_state = get_state()
    expected_tree = normalize_state_tree(initial_runtime_state)
    expected_digest = compute_state_tree_digest(expected_tree)
    structure_digest = compute_state_tree_structure_digest(expected_tree)
    set_state(initial_runtime_state)
    observed_tree = normalize_state_tree(get_state())
    comparison = compare_state_trees(
        expected_tree,
        observed_tree,
        atol=state_tolerance,
    )
    round_trip = StateRoundTripResult(
        passed=(
            comparison.structure_matches
            and comparison.within_tolerance
            and comparison.exact_digest_match
        ),
        expected_state_digest=expected_digest,
        observed_state_digest=comparison.observed_digest,
        compared_leaf_count=comparison.expected_leaf_count,
        maximum_absolute_error=comparison.maximum_absolute_error,
        tolerance=comparison.comparison_tolerance,
    )
    action_step = _bounded_action_step(
        environment,
        base=base,
        modules=modules,
        action_space=action_space,
    )
    return _OpenEnvironmentObservation(
        environment_id=_canonical_environment_id(environment),
        robot_uid=_canonical_robot_uid(base),
        num_envs=_positive_integer_attribute(base, "num_envs"),
        observation_mode=_text_attribute(base, "obs_mode"),
        control_mode=_text_attribute(base, "control_mode"),
        resolved_sim_backend=_resolved_sim_backend(base),
        gpu_simulation=_strict_bool_attribute(base, "gpu_sim_enabled"),
        action_space=action_space,
        control_frequency_hz=_positive_number_attribute(base, "control_freq"),
        simulation_frequency_hz=_positive_number_attribute(base, "sim_freq"),
        runtime_apis=runtime_apis,
        task_evaluator_keys=task_keys,
        task_implementation=_module_source_identity(task_module),
        controller=controller,
        state_tree_structure_digest=structure_digest,
        state_round_trip=round_trip,
        bounded_action_step=action_step,
    )


def _action_space_contract(environment: Any, base: Any) -> ActionSpaceContract:
    batched = getattr(environment, "action_space", None)
    single = getattr(base, "single_action_space", None)
    if batched is None or single is None:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: batched and single action spaces are "
            "required"
        )
    batched_shape = _space_shape(batched, "batched action space")
    single_shape = _space_shape(single, "single action space")
    dtype_value = getattr(single, "dtype", None)
    try:
        dtype = np.dtype(dtype_value).name
    except TypeError as exc:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: action-space dtype is unavailable"
        ) from exc
    lower = _space_bound(single, "low", single_shape)
    upper = _space_bound(single, "high", single_shape)
    return ActionSpaceContract(
        batched_shape=batched_shape,
        single_shape=single_shape,
        dtype=dtype,
        lower_bounds=tuple(float(value) for value in lower.reshape(-1)),
        upper_bounds=tuple(float(value) for value in upper.reshape(-1)),
    )


def _controller_contract(base: Any, action_dimension: int) -> ControllerContract:
    agent = getattr(base, "agent", None)
    controller = getattr(agent, "controller", None)
    components_value = getattr(controller, "controllers", None)
    if not isinstance(components_value, Mapping) or not components_value:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: public combined controller mapping is "
            "unavailable"
        )
    components_mapping = cast(Mapping[object, object], components_value)
    components: list[ControllerComponentContract] = []
    configuration_records: list[object] = []
    offset = 0
    for raw_name, component in components_mapping.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ManiSkillCompatibilityError(
                "ManiSkill compatibility probe: controller component names must be text"
            )
        dimension = _component_action_dimension(component)
        indices = tuple(range(offset, offset + dimension))
        component_type = _qualified_type_name(component)
        config = getattr(component, "config", None)
        if config is None:
            raise ManiSkillCompatibilityError(
                "ManiSkill compatibility probe: public controller component config "
                "is missing"
            )
        config_payload = _semantic_public_value(config, depth=0)
        identity = _semantic_digest(
            {
                "configuration": config_payload,
                "controller_type": component_type,
                "indices": list(indices),
                "name": raw_name,
            },
            "ManiSkillControllerComponent",
        )
        components.append(
            ControllerComponentContract(
                name=raw_name,
                indices=indices,
                controller_type=component_type,
                configuration_identity=identity,
            )
        )
        configuration_records.append(
            {
                "configuration": config_payload,
                "controller_type": component_type,
                "indices": list(indices),
                "name": raw_name,
            }
        )
        offset += dimension
    if offset != action_dimension:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: controller component dimensions do not "
            "match the environment action dimension"
        )
    if not {"arm", "gripper"}.issubset(component.name for component in components):
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: expected public arm and gripper "
            "controller components"
        )
    combined_payload = {
        "combined_controller_type": _qualified_type_name(controller),
        "components": configuration_records,
    }
    return ControllerContract(
        components=tuple(components),
        configuration_identity=_semantic_digest(
            combined_payload, "ManiSkillCombinedController"
        ),
    )


def _component_action_dimension(component: object) -> int:
    space = getattr(component, "single_action_space", None)
    if space is not None:
        shape = _space_shape(space, "controller single action space")
    else:
        space = getattr(component, "action_space", None)
        if space is None:
            raise ManiSkillCompatibilityError(
                "ManiSkill compatibility probe: controller component action space "
                "is missing"
            )
        shape = _space_shape(space, "controller action space")
        if len(shape) == 2 and shape[0] == REQUIRED_NUM_ENVS:
            shape = (shape[1],)
    if len(shape) != 1:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: controller component must expose one "
            "action vector"
        )
    return shape[0]


def _bounded_action_step(
    environment: Any,
    *,
    base: Any,
    modules: ManiSkillRuntimeModules,
    action_space: ActionSpaceContract,
) -> ProbeCheckResult:
    lower = np.asarray(action_space.lower_bounds, dtype=np.float64)
    upper = np.asarray(action_space.upper_bounds, dtype=np.float64)
    midpoint = lower + (upper - lower) * 0.5
    action = midpoint.astype(np.dtype(action_space.dtype), copy=False).reshape(
        action_space.batched_shape
    )
    torch = cast(Any, modules.torch)
    try:
        runtime_action = torch.as_tensor(action, device=base.device)
        environment.step(runtime_action)
    except Exception as exc:
        return ProbeCheckResult(
            passed=False,
            detail=sanitize_operational_text(exc),
        )
    return ProbeCheckResult(
        passed=True,
        detail="bounded midpoint action executed once",
    )


def _close_environment(environment: Any) -> ProbeCheckResult:
    close = getattr(environment, "close", None)
    if not callable(close):
        return ProbeCheckResult(
            passed=False,
            detail="environment close method is not callable",
        )
    try:
        close()
    except Exception as exc:
        return ProbeCheckResult(
            passed=False,
            detail=sanitize_operational_text(exc),
        )
    return ProbeCheckResult(passed=True, detail="environment closed cleanly")


def _module_source_identity(module: ModuleType) -> ModuleSourceIdentity:
    module_name = getattr(module, "__name__", None)
    source_path = inspect.getsourcefile(module)
    if not isinstance(module_name, str) or not module_name or source_path is None:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: imported module has no source identity"
        )
    path = Path(source_path)
    try:
        if not path.is_file():
            raise OSError("source is not a regular file")
        content = path.read_bytes()
    except OSError as exc:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: imported module source could not be hashed"
        ) from exc
    return ModuleSourceIdentity(
        module_name=module_name,
        source_sha256=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def _semantic_digest(value: object, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _semantic_public_value(value: object, *, depth: int) -> object:
    if depth > 32:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: controller config nesting is too deep"
        )
    if value is None or type(value) in (bool, int, float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ManiSkillCompatibilityError(
                "ManiSkill compatibility probe: controller config contains a "
                "non-finite value"
            )
        return value
    if isinstance(value, str):
        return "<runtime-path-omitted>" if _looks_like_runtime_path(value) else value
    if isinstance(value, Enum):
        return _semantic_public_value(value.value, depth=depth + 1)
    if isinstance(value, np.generic):
        return _semantic_public_value(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        if contiguous.dtype.hasobject:
            raise ManiSkillCompatibilityError(
                "ManiSkill compatibility probe: object arrays are invalid "
                "controller config"
            )
        return {
            "content_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _semantic_public_value(
                getattr(value, field.name), depth=depth + 1
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ManiSkillCompatibilityError(
                    "ManiSkill compatibility probe: controller config keys must be text"
                )
            result[key] = _semantic_public_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_semantic_public_value(item, depth=depth + 1) for item in value]
    if isinstance(value, type):
        return {"python_type": f"{value.__module__}.{value.__qualname__}"}
    return {"python_type": _qualified_type_name(value)}


def _looks_like_runtime_path(value: str) -> bool:
    if value.startswith(("/", "\\\\", "//")) or _WINDOWS_DRIVE_PATTERN.match(value):
        return True
    try:
        return (
            PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
        )
    except (OSError, ValueError):
        return True


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _space_shape(space: object, context: str) -> tuple[int, ...]:
    shape = getattr(space, "shape", None)
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: {context} shape is unavailable"
        )
    result: list[int] = []
    for size in shape:
        if type(size) is not int or size <= 0:
            raise ManiSkillCompatibilityError(
                f"ManiSkill compatibility probe: {context} has invalid shape"
            )
        result.append(size)
    if not result:
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: {context} must not be scalar"
        )
    return tuple(result)


def _space_bound(
    space: object, name: str, shape: tuple[int, ...]
) -> np.ndarray[Any, Any]:
    value = getattr(space, name, None)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: action-space {name} bound is unavailable"
        ) from exc
    if array.shape != shape:
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: action-space {name} shape changed"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: action-space {name} must be finite"
        )
    return cast(np.ndarray[Any, Any], array)


def _canonical_environment_id(environment: Any) -> str:
    spec = getattr(environment, "spec", None)
    value = getattr(spec, "id", None)
    if not isinstance(value, str) or not value:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: environment spec ID is unavailable"
        )
    return value


def _canonical_robot_uid(base: Any) -> str:
    value = getattr(base, "robot_uids", None)
    if not isinstance(value, str) or not value:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: robot UID is unavailable"
        )
    return value


def _resolved_sim_backend(base: Any) -> str:
    backend = getattr(base, "backend", None)
    value = getattr(backend, "sim_backend", None)
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value:
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility probe: resolved simulation backend is unavailable"
        )
    return value


def _text_attribute(value: object, name: str) -> str:
    result = getattr(value, name, None)
    if not isinstance(result, str) or not result:
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: {name} is unavailable"
        )
    return result


def _positive_integer_attribute(value: object, name: str) -> int:
    result = getattr(value, name, None)
    if type(result) is not int or result <= 0:
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: {name} must be a positive integer"
        )
    return result


def _positive_number_attribute(value: object, name: str) -> float:
    result = getattr(value, name, None)
    if type(result) not in (int, float):
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: {name} must be a positive finite number"
        )
    numeric = cast(int | float, result)
    if not math.isfinite(float(numeric)) or float(numeric) <= 0.0:
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: {name} must be a positive finite number"
        )
    return float(numeric)


def _strict_bool_attribute(value: object, name: str) -> bool:
    result = getattr(value, name, None)
    if type(result) is not bool:
        raise ManiSkillCompatibilityError(
            f"ManiSkill compatibility probe: {name} must be a boolean"
        )
    return result


__all__ = [
    "InstalledManiSkillProbeRuntime",
    "ManiSkillProbeRuntime",
    "PICKCUBE_TASK_MODULE",
    "SOURCE_SOLVER_CALLABLE",
    "SOURCE_SOLVER_MODULE",
    "probe_maniskill_pickcube",
]
