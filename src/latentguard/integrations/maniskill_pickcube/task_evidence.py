"""Strict PickCube task-evidence conversion without simulator imports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from latentguard.models import FailureEvent, JsonScalar
from latentguard.replay.models import (
    TerminalTaskEvidence,
    TerminalTaskStatus,
)

PICKCUBE_TASK_ID = "maniskill/PickCube-v1"
PICKCUBE_TASK_CONTRACT_VERSION = "1.0.0"
PICKCUBE_PROGRESS_SEMANTIC = "pickcube_binary_completion_v0"
PICKCUBE_UNSAFE_SEMANTIC = "pickcube_cube_center_below_world_zero_v0"


class PickCubeTaskEvidenceError(ValueError):
    """Raised when a task-evidence contract itself is malformed."""


@dataclass(frozen=True, slots=True)
class PickCubeTaskKeyContract:
    """Verified evaluator keys used to interpret one installed PickCube task."""

    success: str
    object_placed: str
    robot_static: str
    grasped: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Require four explicit, distinct keys rather than guessing aliases."""
        if self.schema_version != "1.0":
            raise PickCubeTaskEvidenceError(
                "PickCubeTaskKeyContract.schema_version: unsupported version"
            )
        values = (self.success, self.object_placed, self.robot_static, self.grasped)
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in values
        ):
            raise PickCubeTaskEvidenceError(
                "PickCubeTaskKeyContract: keys must be explicit non-empty strings"
            )
        if len(set(values)) != len(values):
            raise PickCubeTaskEvidenceError(
                "PickCubeTaskKeyContract: evaluator keys must be distinct"
            )

    def as_mapping(self) -> Mapping[str, str]:
        """Return the content-bound canonical key mapping."""
        return MappingProxyType(
            {
                "grasped": self.grasped,
                "object_placed": self.object_placed,
                "robot_static": self.robot_static,
                "schema_version": self.schema_version,
                "success": self.success,
            }
        )

    @classmethod
    def from_compatibility_report(cls, report: object) -> PickCubeTaskKeyContract:
        """Bind canonical meanings only after the probe observed all four keys."""
        try:
            observed = set(cast(Any, report).task_evaluator_keys)
        except (AttributeError, TypeError) as exc:
            raise PickCubeTaskEvidenceError(
                "compatibility report lacks task evaluator keys"
            ) from exc
        required = {
            "success",
            "is_obj_placed",
            "is_robot_static",
            "is_grasped",
        }
        if not required.issubset(observed):
            missing = ", ".join(sorted(required - observed))
            raise PickCubeTaskEvidenceError(
                "compatibility report is missing PickCube task keys: " + missing
            )
        return cls(
            success="success",
            object_placed="is_obj_placed",
            robot_static="is_robot_static",
            grasped="is_grasped",
        )


@dataclass(frozen=True, slots=True)
class RawPickCubeTaskSnapshot:
    """Simulator-owned scalar task values captured at one replay state."""

    evaluator_values: Mapping[str, object]
    cube_center_z: object
    cube_to_goal_distance: object
    diagnostics: Mapping[str, JsonScalar] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Detach only scalar diagnostics; simulator values remain uninterpreted."""
        if not isinstance(self.evaluator_values, Mapping):
            raise PickCubeTaskEvidenceError(
                "RawPickCubeTaskSnapshot.evaluator_values: expected a mapping"
            )
        object.__setattr__(
            self, "evaluator_values", MappingProxyType(dict(self.evaluator_values))
        )
        if not isinstance(self.diagnostics, Mapping):
            raise PickCubeTaskEvidenceError(
                "RawPickCubeTaskSnapshot.diagnostics: expected a mapping"
            )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


def _numpy_scalar(value: object) -> np.ndarray[Any, Any]:
    """Detach a scalar-like value without importing torch or another runtime."""
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
    if array.shape not in ((), (1,)):
        raise PickCubeTaskEvidenceError(
            f"expected a scalar or one-element tensor, got shape {array.shape!r}"
        )
    if array.dtype.hasobject:
        raise PickCubeTaskEvidenceError("object-valued task evidence is unsupported")
    return array.reshape(())


def _strict_bool(value: object) -> bool:
    array = _numpy_scalar(value)
    if not np.issubdtype(array.dtype, np.bool_):
        raise PickCubeTaskEvidenceError(
            f"expected boolean task evidence, got dtype {array.dtype.str!r}"
        )
    return bool(array.item())


def _finite_float(value: object) -> float:
    array = _numpy_scalar(value)
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.number
    ):
        raise PickCubeTaskEvidenceError(
            f"expected numeric task evidence, got dtype {array.dtype.str!r}"
        )
    number = float(array.item())
    if not math.isfinite(number):
        raise PickCubeTaskEvidenceError("numeric task evidence must be finite")
    return number


def _indeterminate_evidence(
    *,
    reasons: list[str],
    diagnostics: Mapping[str, JsonScalar],
) -> TerminalTaskEvidence:
    joined = ";".join(reasons)
    safe_reason = joined[:512] if joined else "unknown_task_contract_error"
    merged: dict[str, JsonScalar] = dict(diagnostics)
    merged["pickcube_task_contract_complete"] = False
    merged["pickcube_task_contract_error_count"] = len(reasons)
    return TerminalTaskEvidence(
        status=TerminalTaskStatus.INDETERMINATE,
        success=None,
        progress=None,
        unsafe=None,
        failure_events=(),
        termination_reason=f"pickcube_task_evidence_indeterminate:{safe_reason}",
        diagnostics=merged,
    )


def build_pickcube_task_evidence(
    snapshot: RawPickCubeTaskSnapshot,
    key_contract: PickCubeTaskKeyContract,
) -> TerminalTaskEvidence:
    """Convert verified official scalars into binary PickCube task evidence.

    Missing keys, unexpected shapes, non-boolean task flags, and non-finite
    geometry yield indeterminate evidence. Runtime exceptions are deliberately
    not caught here so the generic executor records them as execution errors.
    """
    values = snapshot.evaluator_values
    resolved: dict[str, bool] = {}
    reasons: list[str] = []
    for semantic, key in (
        ("success", key_contract.success),
        ("object_placed", key_contract.object_placed),
        ("robot_static", key_contract.robot_static),
        ("grasped", key_contract.grasped),
    ):
        if key not in values:
            reasons.append(f"missing_{semantic}")
            continue
        try:
            resolved[semantic] = _strict_bool(values[key])
        except (PickCubeTaskEvidenceError, TypeError, ValueError, OverflowError):
            reasons.append(f"invalid_{semantic}")

    cube_center_z: float | None = None
    cube_to_goal_distance: float | None = None
    try:
        cube_center_z = _finite_float(snapshot.cube_center_z)
    except (PickCubeTaskEvidenceError, TypeError, ValueError, OverflowError):
        reasons.append("invalid_cube_center_z")
    try:
        cube_to_goal_distance = _finite_float(snapshot.cube_to_goal_distance)
    except (PickCubeTaskEvidenceError, TypeError, ValueError, OverflowError):
        reasons.append("invalid_cube_to_goal_distance")

    diagnostics: dict[str, JsonScalar] = dict(snapshot.diagnostics)
    diagnostics["pickcube_task_contract_complete"] = not reasons
    for name, value in resolved.items():
        diagnostics[f"pickcube_{name}"] = value
    if cube_center_z is not None:
        diagnostics["pickcube_cube_center_z"] = cube_center_z
    if cube_to_goal_distance is not None:
        diagnostics["pickcube_cube_to_goal_distance"] = cube_to_goal_distance

    if reasons:
        return _indeterminate_evidence(reasons=reasons, diagnostics=diagnostics)

    success = resolved["success"]
    object_placed = resolved["object_placed"]
    robot_static = resolved["robot_static"]
    grasped = resolved["grasped"]
    assert cube_center_z is not None
    assert cube_to_goal_distance is not None
    if success and (not object_placed or not robot_static):
        return _indeterminate_evidence(
            reasons=["inconsistent_success_components"], diagnostics=diagnostics
        )

    unsafe = cube_center_z < 0.0
    events: list[FailureEvent] = []
    if not success:
        events.append(
            FailureEvent(
                failure_type="task_not_completed",
                description="Official PickCube terminal success was false.",
            )
        )
        if not object_placed:
            events.append(
                FailureEvent(
                    failure_type="cube_not_at_goal",
                    description="Official PickCube object-placed status was false.",
                )
            )
        if not robot_static:
            events.append(
                FailureEvent(
                    failure_type="robot_not_static",
                    description="Official PickCube robot-static status was false.",
                )
            )
    if unsafe:
        events.append(
            FailureEvent(
                failure_type="cube_below_world_zero",
                description="Cube center was below world z=0.",
            )
        )

    diagnostics.update(
        {
            "pickcube_grasped": grasped,
            "pickcube_object_placed": object_placed,
            "pickcube_robot_static": robot_static,
            "pickcube_unsafe_proxy": unsafe,
        }
    )
    return TerminalTaskEvidence(
        status=TerminalTaskStatus.COMPLETE,
        success=success,
        progress=1.0 if success else 0.0,
        unsafe=unsafe,
        failure_events=tuple(events),
        termination_reason="pickcube_official_task_evaluation_complete",
        diagnostics=diagnostics,
    )


__all__ = [
    "PICKCUBE_PROGRESS_SEMANTIC",
    "PICKCUBE_TASK_CONTRACT_VERSION",
    "PICKCUBE_TASK_ID",
    "PICKCUBE_UNSAFE_SEMANTIC",
    "PickCubeTaskEvidenceError",
    "PickCubeTaskKeyContract",
    "RawPickCubeTaskSnapshot",
    "build_pickcube_task_evidence",
]
