"""Versioned compatibility observations and trust binding for PickCube."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from latentguard.evaluation.security import is_sanitized_operational_text
from latentguard.integrations.maniskill_pickcube.configuration import (
    PROGRESS_SEMANTIC,
    REQUIRED_CONTROL_MODE,
    REQUIRED_ENVIRONMENT_ID,
    REQUIRED_MANISKILL_VERSION,
    REQUIRED_NUM_ENVS,
    REQUIRED_OBSERVATION_MODE,
    REQUIRED_ROBOT_UID,
    REQUIRED_SIM_BACKEND_REQUEST,
    STATE_TREE_SEMANTIC,
    TASK_CONTRACT_VERSION,
    UNSAFE_SEMANTIC,
    ExpectedManiSkillPickCubeContract,
)
from latentguard.replay.identity import canonical_json_bytes, canonical_json_value

COMPATIBILITY_REPORT_SCHEMA_VERSION = "1.0"
REQUIRED_SOURCE_SOLVER_MODULE = (
    "mani_skill.examples.motionplanning.panda.solutions.pick_cube"
)
REQUIRED_PICKCUBE_TASK_MODULE = "mani_skill.envs.tasks.tabletop.pick_cube"
REQUIRED_TASK_EVALUATOR_KEYS = frozenset(
    {"success", "is_obj_placed", "is_robot_static", "is_grasped"}
)


class ManiSkillCompatibilityError(ValueError):
    """Raised when an observed runtime cannot satisfy the M2C contract."""


def _digest(value: object, context: str) -> str:
    import hashlib

    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value.removeprefix("sha256:")
    return len(suffix) == 64 and all(
        character in "0123456789abcdef" for character in suffix
    )


def _require_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ManiSkillCompatibilityError(
            f"{context}: expected canonical non-empty text"
        )
    canonical_json_value(value, context=context)
    return value


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ManiSkillCompatibilityError(
            f"{context}: expected sha256:<64 lowercase hex>"
        )
    return cast(str, value)


def _require_positive_number(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise ManiSkillCompatibilityError(
            f"{context}: expected a positive finite number"
        )
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)) or float(numeric) <= 0.0:
        raise ManiSkillCompatibilityError(
            f"{context}: expected a positive finite number"
        )
    return float(numeric)


@dataclass(frozen=True, slots=True)
class ModuleSourceIdentity:
    """Path-independent identity of one imported Python source module."""

    module_name: str
    source_sha256: str

    def __post_init__(self) -> None:
        """Reject empty names, runtime paths, and malformed source digests."""
        _require_text(self.module_name, "ModuleSourceIdentity.module_name")
        _require_sha256(self.source_sha256, "ModuleSourceIdentity.source_sha256")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "module_name": self.module_name,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class ControllerComponentContract:
    """One publicly exposed controller component and its explicit action indices."""

    name: str
    indices: tuple[int, ...]
    controller_type: str
    configuration_identity: str

    def __post_init__(self) -> None:
        """Detach indices and validate the component identity."""
        object.__setattr__(self, "indices", tuple(self.indices))
        _require_text(self.name, "ControllerComponentContract.name")
        _require_text(
            self.controller_type, "ControllerComponentContract.controller_type"
        )
        _require_sha256(
            self.configuration_identity,
            "ControllerComponentContract.configuration_identity",
        )
        if not self.indices:
            raise ManiSkillCompatibilityError(
                "ControllerComponentContract.indices: must not be empty"
            )
        if any(type(index) is not int or index < 0 for index in self.indices):
            raise ManiSkillCompatibilityError(
                "ControllerComponentContract.indices: expected non-negative integers"
            )
        if len(set(self.indices)) != len(self.indices):
            raise ManiSkillCompatibilityError(
                "ControllerComponentContract.indices: duplicates are unsupported"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "configuration_identity": self.configuration_identity,
            "controller_type": self.controller_type,
            "indices": list(self.indices),
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class ControllerContract:
    """Ordered public controller structure bound by a semantic digest."""

    components: tuple[ControllerComponentContract, ...]
    configuration_identity: str

    def __post_init__(self) -> None:
        """Detach components and require unique names and index ownership."""
        object.__setattr__(self, "components", tuple(self.components))
        _require_sha256(
            self.configuration_identity, "ControllerContract.configuration_identity"
        )
        if not self.components:
            raise ManiSkillCompatibilityError(
                "ControllerContract.components: must not be empty"
            )
        names = tuple(component.name for component in self.components)
        if len(set(names)) != len(names):
            raise ManiSkillCompatibilityError(
                "ControllerContract.components: duplicate names are unsupported"
            )
        indices = tuple(
            index for component in self.components for index in component.indices
        )
        if len(set(indices)) != len(indices):
            raise ManiSkillCompatibilityError(
                "ControllerContract.components: action indices overlap"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "components": [component.to_dict() for component in self.components],
            "configuration_identity": self.configuration_identity,
        }


@dataclass(frozen=True, slots=True)
class ActionSpaceContract:
    """Observed batched and single-environment finite action-space contract."""

    batched_shape: tuple[int, ...]
    single_shape: tuple[int, ...]
    dtype: str
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    space_digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Detach arrays, validate finite bounds, and compute a stable digest."""
        object.__setattr__(self, "batched_shape", tuple(self.batched_shape))
        object.__setattr__(self, "single_shape", tuple(self.single_shape))
        object.__setattr__(self, "lower_bounds", tuple(self.lower_bounds))
        object.__setattr__(self, "upper_bounds", tuple(self.upper_bounds))
        _require_text(self.dtype, "ActionSpaceContract.dtype")
        for context, shape in (
            ("ActionSpaceContract.batched_shape", self.batched_shape),
            ("ActionSpaceContract.single_shape", self.single_shape),
        ):
            if not shape or any(type(size) is not int or size <= 0 for size in shape):
                raise ManiSkillCompatibilityError(
                    f"{context}: expected positive integer dimensions"
                )
        if len(self.single_shape) != 1:
            raise ManiSkillCompatibilityError(
                "ActionSpaceContract.single_shape: M2C requires one action vector"
            )
        action_dimension = self.single_shape[0]
        if (
            len(self.lower_bounds) != action_dimension
            or len(self.upper_bounds) != action_dimension
        ):
            raise ManiSkillCompatibilityError(
                "ActionSpaceContract.bounds: expected one finite bound per "
                "action dimension"
            )
        for index, (lower, upper) in enumerate(
            zip(self.lower_bounds, self.upper_bounds, strict=True)
        ):
            if (
                type(lower) not in (int, float)
                or type(upper) not in (int, float)
                or not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or float(lower) > float(upper)
            ):
                raise ManiSkillCompatibilityError(
                    "ActionSpaceContract.bounds: invalid finite interval at index "
                    f"{index}"
                )
        object.__setattr__(
            self,
            "space_digest",
            _digest(self._identity_payload(), "ManiSkillActionSpaceContract"),
        )

    @property
    def action_dimension(self) -> int:
        """Return the verified unbatched action-vector width."""
        return self.single_shape[0]

    def _identity_payload(self) -> dict[str, object]:
        return {
            "batched_shape": list(self.batched_shape),
            "dtype": self.dtype,
            "lower_bounds": list(self.lower_bounds),
            "single_shape": list(self.single_shape),
            "upper_bounds": list(self.upper_bounds),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation including its digest."""
        return {**self._identity_payload(), "space_digest": self.space_digest}


@dataclass(frozen=True, slots=True)
class RuntimeApiContract:
    """Callable status of the required public task state and evaluation APIs."""

    get_state_dict_callable: bool
    set_state_dict_callable: bool
    evaluate_callable: bool

    def __post_init__(self) -> None:
        """Require explicit booleans without integer coercion."""
        if any(
            type(value) is not bool
            for value in (
                self.get_state_dict_callable,
                self.set_state_dict_callable,
                self.evaluate_callable,
            )
        ):
            raise ManiSkillCompatibilityError(
                "RuntimeApiContract: callable flags must be booleans"
            )

    @property
    def complete(self) -> bool:
        """Return whether every required public API is callable."""
        return (
            self.get_state_dict_callable
            and self.set_state_dict_callable
            and self.evaluate_callable
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "evaluate_callable": self.evaluate_callable,
            "get_state_dict_callable": self.get_state_dict_callable,
            "set_state_dict_callable": self.set_state_dict_callable,
        }


@dataclass(frozen=True, slots=True)
class StateRoundTripResult:
    """Complete reset-boundary state restoration comparison result."""

    passed: bool
    expected_state_digest: str
    observed_state_digest: str
    compared_leaf_count: int
    maximum_absolute_error: float | None
    tolerance: float

    def __post_init__(self) -> None:
        """Validate finite error, digest, and comparison-count fields."""
        if type(self.passed) is not bool:
            raise ManiSkillCompatibilityError(
                "StateRoundTripResult.passed: expected a boolean"
            )
        _require_sha256(
            self.expected_state_digest,
            "StateRoundTripResult.expected_state_digest",
        )
        _require_sha256(
            self.observed_state_digest,
            "StateRoundTripResult.observed_state_digest",
        )
        if type(self.compared_leaf_count) is not int or self.compared_leaf_count <= 0:
            raise ManiSkillCompatibilityError(
                "StateRoundTripResult.compared_leaf_count: expected a positive integer"
            )
        if self.maximum_absolute_error is not None and (
            type(self.maximum_absolute_error) not in (int, float)
            or not math.isfinite(float(self.maximum_absolute_error))
            or float(self.maximum_absolute_error) < 0.0
        ):
            raise ManiSkillCompatibilityError(
                "StateRoundTripResult.maximum_absolute_error: expected a "
                "non-negative finite number or null"
            )
        if (
            type(self.tolerance) not in (int, float)
            or not math.isfinite(float(self.tolerance))
            or float(self.tolerance) < 0.0
        ):
            raise ManiSkillCompatibilityError(
                "StateRoundTripResult.tolerance: expected a non-negative finite number"
            )
        if self.passed and self.maximum_absolute_error is None:
            raise ManiSkillCompatibilityError(
                "StateRoundTripResult.maximum_absolute_error: passed comparisons "
                "require a finite error"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "compared_leaf_count": self.compared_leaf_count,
            "expected_state_digest": self.expected_state_digest,
            "maximum_absolute_error": self.maximum_absolute_error,
            "observed_state_digest": self.observed_state_digest,
            "passed": self.passed,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class ProbeCheckResult:
    """Sanitized result of one bounded runtime behavior check."""

    passed: bool
    detail: str

    def __post_init__(self) -> None:
        """Require an explicit boolean and disclosure-safe concise detail."""
        if type(self.passed) is not bool:
            raise ManiSkillCompatibilityError(
                "ProbeCheckResult.passed: expected a boolean"
            )
        if not is_sanitized_operational_text(self.detail):
            raise ManiSkillCompatibilityError(
                "ProbeCheckResult.detail: expected concise sanitized operational text"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {"detail": self.detail, "passed": self.passed}


@dataclass(frozen=True, slots=True)
class OperationalRuntimeMetadata:
    """Sanitized runtime facts excluded from deterministic compatibility identity."""

    python_version: str
    torch_version: str
    cuda_runtime_version: str
    gpu_model: str
    gpu_capability: str

    def __post_init__(self) -> None:
        """Reject paths, credentials, hostnames, and uncontrolled runtime text."""
        for name in (
            "python_version",
            "torch_version",
            "cuda_runtime_version",
            "gpu_model",
            "gpu_capability",
        ):
            value = getattr(self, name)
            if not is_sanitized_operational_text(value):
                raise ManiSkillCompatibilityError(
                    f"OperationalRuntimeMetadata.{name}: expected sanitized text"
                )

    def to_dict(self) -> dict[str, object]:
        """Return sanitized operational metadata without a hostname or path."""
        return {
            "cuda_runtime_version": self.cuda_runtime_version,
            "gpu_capability": self.gpu_capability,
            "gpu_model": self.gpu_model,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Sanitized real-runtime observation used to bind trusted PickCube replay."""

    mani_skill_version: str
    sapien_version: str
    mplib_version: str
    environment_id: str
    robot_uid: str
    num_envs: int
    reset_seed: int
    observation_mode: str
    control_mode: str
    sim_backend_request: str
    resolved_sim_backend: str
    gpu_simulation: bool
    action_space: ActionSpaceContract
    control_frequency_hz: float
    simulation_frequency_hz: float
    runtime_apis: RuntimeApiContract
    task_evaluator_keys: tuple[str, ...]
    source_solver: ModuleSourceIdentity
    task_implementation: ModuleSourceIdentity
    controller: ControllerContract
    state_tree_structure_digest: str
    state_round_trip: StateRoundTripResult
    bounded_action_step: ProbeCheckResult
    environment_close: ProbeCheckResult
    operational: OperationalRuntimeMetadata
    schema_version: str = COMPATIBILITY_REPORT_SCHEMA_VERSION
    action_contract_digest: str = field(init=False)
    compatibility_identity: str = field(init=False)

    def __post_init__(self) -> None:
        """Detach collections, validate structure, and compute both identities."""
        object.__setattr__(self, "task_evaluator_keys", tuple(self.task_evaluator_keys))
        if self.schema_version != COMPATIBILITY_REPORT_SCHEMA_VERSION:
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.schema_version: unsupported version "
                f"{self.schema_version!r}"
            )
        for name in (
            "mani_skill_version",
            "sapien_version",
            "mplib_version",
            "environment_id",
            "robot_uid",
            "observation_mode",
            "control_mode",
            "sim_backend_request",
            "resolved_sim_backend",
        ):
            _require_text(getattr(self, name), f"CompatibilityReport.{name}")
        if type(self.num_envs) is not int or self.num_envs <= 0:
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.num_envs: expected a positive integer"
            )
        if type(self.reset_seed) is not int or self.reset_seed < 0:
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.reset_seed: expected a non-negative integer"
            )
        if type(self.gpu_simulation) is not bool:
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.gpu_simulation: expected a boolean"
            )
        _require_positive_number(
            self.control_frequency_hz, "CompatibilityReport.control_frequency_hz"
        )
        _require_positive_number(
            self.simulation_frequency_hz,
            "CompatibilityReport.simulation_frequency_hz",
        )
        if not self.task_evaluator_keys or any(
            not isinstance(key, str) or not key.strip()
            for key in self.task_evaluator_keys
        ):
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.task_evaluator_keys: expected non-empty "
                "canonical keys"
            )
        if tuple(sorted(set(self.task_evaluator_keys))) != self.task_evaluator_keys:
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.task_evaluator_keys: keys must be sorted "
                "and unique"
            )
        _require_sha256(
            self.state_tree_structure_digest,
            "CompatibilityReport.state_tree_structure_digest",
        )
        action_indices = tuple(
            index
            for component in self.controller.components
            for index in component.indices
        )
        if action_indices != tuple(range(self.action_space.action_dimension)):
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.controller: components must cover action indices "
                "exactly once in public controller order"
            )
        action_payload = {
            "action_space": self.action_space.to_dict(),
            "control_period_s": 1.0 / float(self.control_frequency_hz),
            "controller": self.controller.to_dict(),
        }
        object.__setattr__(
            self,
            "action_contract_digest",
            _digest(action_payload, "ManiSkillActionContract"),
        )
        object.__setattr__(
            self,
            "compatibility_identity",
            _digest(self._semantic_payload(), "ManiSkillPickCubeCompatibility"),
        )

    @property
    def control_period_s(self) -> float:
        """Return the observed controller period in seconds."""
        return 1.0 / float(self.control_frequency_hz)

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "action_contract_digest": self.action_contract_digest,
            "bounded_action_step_passed": self.bounded_action_step.passed,
            "control_frequency_hz": self.control_frequency_hz,
            "control_mode": self.control_mode,
            "controller_configuration_identity": self.controller.configuration_identity,
            "environment_close_passed": self.environment_close.passed,
            "environment_id": self.environment_id,
            "gpu_simulation": self.gpu_simulation,
            "mani_skill_version": self.mani_skill_version,
            "mplib_version": self.mplib_version,
            "num_envs": self.num_envs,
            "reset_seed": self.reset_seed,
            "observation_mode": self.observation_mode,
            "resolved_sim_backend": self.resolved_sim_backend,
            "robot_uid": self.robot_uid,
            "runtime_apis": self.runtime_apis.to_dict(),
            "sapien_version": self.sapien_version,
            "schema_version": self.schema_version,
            "sim_backend_request": self.sim_backend_request,
            "simulation_frequency_hz": self.simulation_frequency_hz,
            "source_solver": self.source_solver.to_dict(),
            "state_round_trip": self.state_round_trip.to_dict(),
            "state_tree_structure_digest": self.state_tree_structure_digest,
            "task_evaluator_keys": list(self.task_evaluator_keys),
            "task_implementation": self.task_implementation.to_dict(),
        }

    def semantic_configuration(self) -> Mapping[str, object]:
        """Return adapter semantics with no runtime paths or operational metadata."""
        value = {
            "action_contract_digest": self.action_contract_digest,
            "compatibility_identity": self.compatibility_identity,
            "control_mode": self.control_mode,
            "environment_id": self.environment_id,
            "mani_skill_version": self.mani_skill_version,
            "num_envs": self.num_envs,
            "observation_mode": self.observation_mode,
            "progress_semantic": PROGRESS_SEMANTIC,
            "robot_uid": self.robot_uid,
            "sim_backend_requirement": REQUIRED_SIM_BACKEND_REQUEST,
            "solver_identity": self.source_solver.source_sha256,
            "state_semantic": STATE_TREE_SEMANTIC,
            "task_contract_version": TASK_CONTRACT_VERSION,
            "unsafe_semantic": UNSAFE_SEMANTIC,
        }
        return MappingProxyType(value)

    def to_dict(self) -> dict[str, object]:
        """Return the complete sanitized report including computed identities."""
        return {
            "action_contract_digest": self.action_contract_digest,
            "action_space": self.action_space.to_dict(),
            "bounded_action_step": self.bounded_action_step.to_dict(),
            "compatibility_identity": self.compatibility_identity,
            "control_frequency_hz": self.control_frequency_hz,
            "control_mode": self.control_mode,
            "controller": self.controller.to_dict(),
            "environment_close": self.environment_close.to_dict(),
            "environment_id": self.environment_id,
            "gpu_simulation": self.gpu_simulation,
            "mani_skill_version": self.mani_skill_version,
            "mplib_version": self.mplib_version,
            "num_envs": self.num_envs,
            "reset_seed": self.reset_seed,
            "observation_mode": self.observation_mode,
            "operational": self.operational.to_dict(),
            "resolved_sim_backend": self.resolved_sim_backend,
            "robot_uid": self.robot_uid,
            "runtime_apis": self.runtime_apis.to_dict(),
            "sapien_version": self.sapien_version,
            "schema_version": self.schema_version,
            "sim_backend_request": self.sim_backend_request,
            "simulation_frequency_hz": self.simulation_frequency_hz,
            "source_solver": self.source_solver.to_dict(),
            "state_round_trip": self.state_round_trip.to_dict(),
            "state_tree_structure_digest": self.state_tree_structure_digest,
            "task_evaluator_keys": list(self.task_evaluator_keys),
            "task_implementation": self.task_implementation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityBinding:
    """Result of comparing one observation with the checked-in expectation."""

    report: CompatibilityReport
    expected_contract: ExpectedManiSkillPickCubeContract
    unresolved_fields: tuple[str, ...]

    @property
    def trusted_replay_ready(self) -> bool:
        """Return whether every trusted replay prerequisite is locally bound."""
        return (
            not self.unresolved_fields and self.expected_contract.trusted_replay_ready
        )

    def require_trusted_replay_ready(self) -> None:
        """Fail closed unless the exact observed report is fully checked in."""
        if self.trusted_replay_ready:
            return
        unresolved = ", ".join(self.unresolved_fields) or "contract_status"
        raise ManiSkillCompatibilityError(
            "ManiSkill compatibility is probe-only and cannot authorize trusted "
            f"replay; unresolved checked-in fields: {unresolved}"
        )

    def semantic_configuration(self) -> Mapping[str, object]:
        """Return semantic adapter configuration only after strict binding."""
        self.require_trusted_replay_ready()
        return self.report.semantic_configuration()


def validate_compatibility_report(
    report: CompatibilityReport,
    expected: ExpectedManiSkillPickCubeContract,
    *,
    require_trusted: bool = False,
) -> CompatibilityBinding:
    """Validate fixed scope and bind all populated checked-in expectations."""
    python_parts = report.operational.python_version.split(".")
    if python_parts[:2] != ["3", "11"]:
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.operational.python_version: Python 3.11 is required"
        )
    fixed = (
        ("mani_skill_version", report.mani_skill_version, REQUIRED_MANISKILL_VERSION),
        ("environment_id", report.environment_id, REQUIRED_ENVIRONMENT_ID),
        ("robot_uid", report.robot_uid, REQUIRED_ROBOT_UID),
        ("num_envs", report.num_envs, REQUIRED_NUM_ENVS),
        ("observation_mode", report.observation_mode, REQUIRED_OBSERVATION_MODE),
        ("control_mode", report.control_mode, REQUIRED_CONTROL_MODE),
        (
            "sim_backend_request",
            report.sim_backend_request,
            REQUIRED_SIM_BACKEND_REQUEST,
        ),
    )
    for name, observed, required in fixed:
        if observed != required:
            raise ManiSkillCompatibilityError(
                f"CompatibilityReport.{name}: expected {required!r}, got {observed!r}"
            )
    if not report.gpu_simulation or "cpu" in report.resolved_sim_backend.lower():
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.resolved_sim_backend: GPU simulation is required"
        )
    if report.action_space.batched_shape not in (
        (report.action_space.action_dimension,),
        (REQUIRED_NUM_ENVS, report.action_space.action_dimension),
    ):
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.action_space: expected the observed single-env "
            "shape or its explicit one-row batch"
        )
    if not report.runtime_apis.complete:
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.runtime_apis: get_state_dict, set_state_dict, and "
            "evaluate must all be callable"
        )
    missing_task_keys = sorted(
        REQUIRED_TASK_EVALUATOR_KEYS - set(report.task_evaluator_keys)
    )
    if missing_task_keys:
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.task_evaluator_keys: missing required keys "
            + ", ".join(missing_task_keys)
        )
    if report.source_solver.module_name != REQUIRED_SOURCE_SOLVER_MODULE:
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.source_solver: official solver module identity changed"
        )
    if report.task_implementation.module_name != REQUIRED_PICKCUBE_TASK_MODULE:
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.task_implementation: PickCube task module "
            "identity changed"
        )
    if not report.state_round_trip.passed:
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.state_round_trip: exact reset-boundary round "
            "trip failed"
        )
    if (
        report.state_round_trip.expected_state_digest
        != report.state_round_trip.observed_state_digest
    ):
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.state_round_trip: expected and observed state "
            "digests must match exactly"
        )
    if report.state_round_trip.maximum_absolute_error is None:
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.state_round_trip: comparison did not produce a "
            "maximum absolute error"
        )
    if (
        report.state_round_trip.maximum_absolute_error
        > expected.state_round_trip_tolerance
        or report.state_round_trip.tolerance != expected.state_round_trip_tolerance
    ):
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.state_round_trip: observed tolerance or error does "
            "not satisfy the checked-in tolerance"
        )
    for name, check in (
        ("bounded_action_step", report.bounded_action_step),
        ("environment_close", report.environment_close),
    ):
        if not check.passed:
            raise ManiSkillCompatibilityError(
                f"CompatibilityReport.{name}: required bounded probe check failed"
            )

    observed_values: dict[str, object] = {
        "sapien_version": report.sapien_version,
        "mplib_version": report.mplib_version,
        "observation_mode": report.observation_mode,
        "resolved_sim_backend": report.resolved_sim_backend,
        "action_dimension": report.action_space.action_dimension,
        "action_dtype": report.action_space.dtype,
        "action_contract_digest": report.action_contract_digest,
        "control_frequency_hz": report.control_frequency_hz,
        "simulation_frequency_hz": report.simulation_frequency_hz,
        "solver_sha256": report.source_solver.source_sha256,
        "task_sha256": report.task_implementation.source_sha256,
        "controller_configuration_identity": report.controller.configuration_identity,
        "state_tree_structure_digest": report.state_tree_structure_digest,
        "compatibility_identity": report.compatibility_identity,
    }
    for field_name, observed_value in observed_values.items():
        configured = getattr(expected, field_name)
        if configured is not None and observed_value != configured:
            raise ManiSkillCompatibilityError(
                f"CompatibilityReport.{field_name}: observed {observed_value!r} "
                "does not match "
                f"checked-in value {configured!r}"
            )

    binding = CompatibilityBinding(
        report=report,
        expected_contract=expected,
        unresolved_fields=expected.unresolved_fields,
    )
    if require_trusted:
        binding.require_trusted_replay_ready()
    return binding


def write_compatibility_report(report: CompatibilityReport, path: Path) -> Path:
    """Write one new sanitized compatibility report without overwriting evidence."""
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.output: refusing to overwrite an existing path"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.output: refusing to overwrite an existing path"
            ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return destination


def load_compatibility_report(path: Path) -> CompatibilityReport:
    """Load, structurally validate, and recompute every report identity."""
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
            raise ManiSkillCompatibilityError(
                "CompatibilityReport.input: missing or unsafe regular file"
            )
        raw = cast(
            object,
            json.loads(
                source.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_fields,
            ),
        )
    except ManiSkillCompatibilityError:
        raise
    except Exception as exc:
        raise ManiSkillCompatibilityError(
            f"CompatibilityReport.input: could not read safely: {exc}"
        ) from exc
    item = _mapping(raw, "CompatibilityReport")
    _exact_fields(
        item,
        {
            "action_contract_digest",
            "action_space",
            "bounded_action_step",
            "compatibility_identity",
            "control_frequency_hz",
            "control_mode",
            "controller",
            "environment_close",
            "environment_id",
            "gpu_simulation",
            "mani_skill_version",
            "mplib_version",
            "num_envs",
            "reset_seed",
            "observation_mode",
            "operational",
            "resolved_sim_backend",
            "robot_uid",
            "runtime_apis",
            "sapien_version",
            "schema_version",
            "sim_backend_request",
            "simulation_frequency_hz",
            "source_solver",
            "state_round_trip",
            "state_tree_structure_digest",
            "task_evaluator_keys",
            "task_implementation",
        },
        "CompatibilityReport",
    )
    report = _report_from_mapping(item)
    for name, observed in (
        ("action_contract_digest", report.action_contract_digest),
        ("compatibility_identity", report.compatibility_identity),
    ):
        persisted = _string(item, name, "CompatibilityReport")
        if persisted != observed:
            raise ManiSkillCompatibilityError(
                f"CompatibilityReport.{name}: persisted identity does not match content"
            )
    return report


def _report_from_mapping(item: Mapping[str, object]) -> CompatibilityReport:
    action = _mapping(item["action_space"], "CompatibilityReport.action_space")
    _exact_fields(
        action,
        {
            "batched_shape",
            "single_shape",
            "dtype",
            "lower_bounds",
            "upper_bounds",
            "space_digest",
        },
        "CompatibilityReport.action_space",
    )
    action_space = ActionSpaceContract(
        batched_shape=_integer_tuple(
            action["batched_shape"], "CompatibilityReport.action_space.batched_shape"
        ),
        single_shape=_integer_tuple(
            action["single_shape"], "CompatibilityReport.action_space.single_shape"
        ),
        dtype=_string(action, "dtype", "CompatibilityReport.action_space"),
        lower_bounds=_number_tuple(
            action["lower_bounds"], "CompatibilityReport.action_space.lower_bounds"
        ),
        upper_bounds=_number_tuple(
            action["upper_bounds"], "CompatibilityReport.action_space.upper_bounds"
        ),
    )
    if (
        _string(action, "space_digest", "CompatibilityReport.action_space")
        != action_space.space_digest
    ):
        raise ManiSkillCompatibilityError(
            "CompatibilityReport.action_space.space_digest: persisted identity "
            "does not match content"
        )
    controller_item = _mapping(item["controller"], "CompatibilityReport.controller")
    _exact_fields(
        controller_item,
        {"components", "configuration_identity"},
        "CompatibilityReport.controller",
    )
    component_values = _sequence(
        controller_item["components"], "CompatibilityReport.controller.components"
    )
    components: list[ControllerComponentContract] = []
    for index, value in enumerate(component_values):
        context = f"CompatibilityReport.controller.components[{index}]"
        component = _mapping(value, context)
        _exact_fields(
            component,
            {"name", "indices", "controller_type", "configuration_identity"},
            context,
        )
        components.append(
            ControllerComponentContract(
                name=_string(component, "name", context),
                indices=_integer_tuple(component["indices"], f"{context}.indices"),
                controller_type=_string(component, "controller_type", context),
                configuration_identity=_string(
                    component, "configuration_identity", context
                ),
            )
        )
    runtime_api_item = _mapping(
        item["runtime_apis"], "CompatibilityReport.runtime_apis"
    )
    _exact_fields(
        runtime_api_item,
        {"get_state_dict_callable", "set_state_dict_callable", "evaluate_callable"},
        "CompatibilityReport.runtime_apis",
    )
    state_item = _mapping(
        item["state_round_trip"], "CompatibilityReport.state_round_trip"
    )
    _exact_fields(
        state_item,
        {
            "passed",
            "expected_state_digest",
            "observed_state_digest",
            "compared_leaf_count",
            "maximum_absolute_error",
            "tolerance",
        },
        "CompatibilityReport.state_round_trip",
    )
    operational_item = _mapping(item["operational"], "CompatibilityReport.operational")
    _exact_fields(
        operational_item,
        {
            "python_version",
            "torch_version",
            "cuda_runtime_version",
            "gpu_model",
            "gpu_capability",
        },
        "CompatibilityReport.operational",
    )
    return CompatibilityReport(
        schema_version=_string(item, "schema_version", "CompatibilityReport"),
        mani_skill_version=_string(item, "mani_skill_version", "CompatibilityReport"),
        sapien_version=_string(item, "sapien_version", "CompatibilityReport"),
        mplib_version=_string(item, "mplib_version", "CompatibilityReport"),
        environment_id=_string(item, "environment_id", "CompatibilityReport"),
        robot_uid=_string(item, "robot_uid", "CompatibilityReport"),
        num_envs=_integer(item, "num_envs", "CompatibilityReport"),
        reset_seed=_integer(item, "reset_seed", "CompatibilityReport"),
        observation_mode=_string(item, "observation_mode", "CompatibilityReport"),
        control_mode=_string(item, "control_mode", "CompatibilityReport"),
        sim_backend_request=_string(item, "sim_backend_request", "CompatibilityReport"),
        resolved_sim_backend=_string(
            item, "resolved_sim_backend", "CompatibilityReport"
        ),
        gpu_simulation=_boolean(item, "gpu_simulation", "CompatibilityReport"),
        action_space=action_space,
        control_frequency_hz=_number(
            item, "control_frequency_hz", "CompatibilityReport"
        ),
        simulation_frequency_hz=_number(
            item, "simulation_frequency_hz", "CompatibilityReport"
        ),
        runtime_apis=RuntimeApiContract(
            get_state_dict_callable=_boolean(
                runtime_api_item,
                "get_state_dict_callable",
                "CompatibilityReport.runtime_apis",
            ),
            set_state_dict_callable=_boolean(
                runtime_api_item,
                "set_state_dict_callable",
                "CompatibilityReport.runtime_apis",
            ),
            evaluate_callable=_boolean(
                runtime_api_item,
                "evaluate_callable",
                "CompatibilityReport.runtime_apis",
            ),
        ),
        task_evaluator_keys=tuple(
            _text_sequence(
                item["task_evaluator_keys"],
                "CompatibilityReport.task_evaluator_keys",
            )
        ),
        source_solver=_module_identity(
            item["source_solver"], "CompatibilityReport.source_solver"
        ),
        task_implementation=_module_identity(
            item["task_implementation"], "CompatibilityReport.task_implementation"
        ),
        controller=ControllerContract(
            components=tuple(components),
            configuration_identity=_string(
                controller_item,
                "configuration_identity",
                "CompatibilityReport.controller",
            ),
        ),
        state_tree_structure_digest=_string(
            item, "state_tree_structure_digest", "CompatibilityReport"
        ),
        state_round_trip=StateRoundTripResult(
            passed=_boolean(
                state_item, "passed", "CompatibilityReport.state_round_trip"
            ),
            expected_state_digest=_string(
                state_item,
                "expected_state_digest",
                "CompatibilityReport.state_round_trip",
            ),
            observed_state_digest=_string(
                state_item,
                "observed_state_digest",
                "CompatibilityReport.state_round_trip",
            ),
            compared_leaf_count=_integer(
                state_item,
                "compared_leaf_count",
                "CompatibilityReport.state_round_trip",
            ),
            maximum_absolute_error=_optional_number_value(
                state_item["maximum_absolute_error"],
                "CompatibilityReport.state_round_trip.maximum_absolute_error",
            ),
            tolerance=_number(
                state_item, "tolerance", "CompatibilityReport.state_round_trip"
            ),
        ),
        bounded_action_step=_check_result(
            item["bounded_action_step"], "CompatibilityReport.bounded_action_step"
        ),
        environment_close=_check_result(
            item["environment_close"], "CompatibilityReport.environment_close"
        ),
        operational=OperationalRuntimeMetadata(
            python_version=_string(
                operational_item, "python_version", "CompatibilityReport.operational"
            ),
            torch_version=_string(
                operational_item, "torch_version", "CompatibilityReport.operational"
            ),
            cuda_runtime_version=_string(
                operational_item,
                "cuda_runtime_version",
                "CompatibilityReport.operational",
            ),
            gpu_model=_string(
                operational_item, "gpu_model", "CompatibilityReport.operational"
            ),
            gpu_capability=_string(
                operational_item,
                "gpu_capability",
                "CompatibilityReport.operational",
            ),
        ),
    )


def _module_identity(value: object, context: str) -> ModuleSourceIdentity:
    item = _mapping(value, context)
    _exact_fields(item, {"module_name", "source_sha256"}, context)
    return ModuleSourceIdentity(
        module_name=_string(item, "module_name", context),
        source_sha256=_string(item, "source_sha256", context),
    )


def _check_result(value: object, context: str) -> ProbeCheckResult:
    item = _mapping(value, context)
    _exact_fields(item, {"passed", "detail"}, context)
    return ProbeCheckResult(
        passed=_boolean(item, "passed", context),
        detail=_string(item, "detail", context),
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManiSkillCompatibilityError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ManiSkillCompatibilityError(f"{context}: expected an array")
    return cast(list[object], value)


def _exact_fields(item: Mapping[str, object], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(item))
    unexpected = sorted(set(item) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ManiSkillCompatibilityError(
            f"{context}: invalid fields ({'; '.join(details)})"
        )


def _string(item: Mapping[str, object], name: str, context: str) -> str:
    return _require_text(item[name], f"{context}.{name}")


def _integer(item: Mapping[str, object], name: str, context: str) -> int:
    value = item[name]
    if type(value) is not int:
        raise ManiSkillCompatibilityError(f"{context}.{name}: expected an integer")
    return value


def _number(item: Mapping[str, object], name: str, context: str) -> float:
    value = item[name]
    if type(value) not in (int, float):
        raise ManiSkillCompatibilityError(f"{context}.{name}: expected a finite number")
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)):
        raise ManiSkillCompatibilityError(f"{context}.{name}: expected a finite number")
    return float(numeric)


def _boolean(item: Mapping[str, object], name: str, context: str) -> bool:
    value = item[name]
    if type(value) is not bool:
        raise ManiSkillCompatibilityError(f"{context}.{name}: expected a boolean")
    return value


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    values = _sequence(value, context)
    result: list[int] = []
    for index, item in enumerate(values):
        if type(item) is not int:
            raise ManiSkillCompatibilityError(
                f"{context}[{index}]: expected an integer"
            )
        result.append(item)
    return tuple(result)


def _number_tuple(value: object, context: str) -> tuple[float, ...]:
    values = _sequence(value, context)
    result: list[float] = []
    for index, item in enumerate(values):
        if type(item) not in (int, float):
            raise ManiSkillCompatibilityError(
                f"{context}[{index}]: expected a finite number"
            )
        numeric = cast(int | float, item)
        if not math.isfinite(float(numeric)):
            raise ManiSkillCompatibilityError(
                f"{context}[{index}]: expected a finite number"
            )
        result.append(float(numeric))
    return tuple(result)


def _optional_number_value(value: object, context: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ManiSkillCompatibilityError(
            f"{context}: expected a finite number or null"
        )
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)):
        raise ManiSkillCompatibilityError(
            f"{context}: expected a finite number or null"
        )
    return float(numeric)


def _text_sequence(value: object, context: str) -> tuple[str, ...]:
    values = _sequence(value, context)
    return tuple(
        _require_text(item, f"{context}[{index}]") for index, item in enumerate(values)
    )


def _reject_json_constant(value: str) -> NoReturn:
    raise ManiSkillCompatibilityError(
        f"CompatibilityReport.input: {value!r} is unsupported"
    )


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManiSkillCompatibilityError(
                f"CompatibilityReport.input: duplicate field {key!r}"
            )
        result[key] = value
    return result


__all__ = [
    "ActionSpaceContract",
    "COMPATIBILITY_REPORT_SCHEMA_VERSION",
    "CompatibilityBinding",
    "CompatibilityReport",
    "ControllerComponentContract",
    "ControllerContract",
    "ManiSkillCompatibilityError",
    "ModuleSourceIdentity",
    "OperationalRuntimeMetadata",
    "ProbeCheckResult",
    "REQUIRED_PICKCUBE_TASK_MODULE",
    "REQUIRED_SOURCE_SOLVER_MODULE",
    "REQUIRED_TASK_EVALUATOR_KEYS",
    "RuntimeApiContract",
    "StateRoundTripResult",
    "load_compatibility_report",
    "validate_compatibility_report",
    "write_compatibility_report",
]
