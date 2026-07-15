"""Public-runtime extraction for state-indexed PickCube sequence boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .source_generation import (
    PickCubeSourceGenerationError,
    SourceEnvironmentFactory,
)
from .state_tree import clone_state_tree
from .task_evidence import PickCubeTaskKeyContract, RawPickCubeTaskSnapshot
from .verifier_state import (
    PickCubeVerifierStateV1,
    build_pickcube_verifier_state_v1,
)


@dataclass(frozen=True, slots=True)
class CapturedPickCubeStateBoundary:
    """Complete state, public task snapshot, and compact verifier vector at s[t]."""

    state_index: int
    state_tree: object
    task_snapshot: Mapping[str, object]
    verifier_state: PickCubeVerifierStateV1

    def __post_init__(self) -> None:
        """Detach the state tree and task scalars from the live runtime."""
        if type(self.state_index) is not int or self.state_index < 0:
            raise PickCubeSourceGenerationError("state boundary index is invalid")
        object.__setattr__(self, "state_tree", clone_state_tree(self.state_tree))
        object.__setattr__(
            self, "task_snapshot", MappingProxyType(dict(self.task_snapshot))
        )


def capture_pickcube_state_boundary(
    environment: object,
    *,
    state_index: int,
    environment_factory: SourceEnvironmentFactory,
    key_contract: PickCubeTaskKeyContract,
) -> CapturedPickCubeStateBoundary:
    """Capture one boundary only through the verified public integration APIs."""
    if type(state_index) is not int or state_index < 0:
        raise PickCubeSourceGenerationError("state boundary index is invalid")
    base = getattr(environment, "unwrapped", environment)
    get_state_dict = getattr(base, "get_state_dict", None)
    if not callable(get_state_dict):
        raise PickCubeSourceGenerationError(
            "PickCube environment lacks public get_state_dict"
        )
    state_tree = clone_state_tree(get_state_dict())
    raw_task = environment_factory.capture_task_snapshot(environment, key_contract)
    flags = _task_flags(raw_task, key_contract)
    joint_names, robot_state = environment_factory.extract_named_robot_state(
        environment
    )
    joint_count = len(joint_names)
    if robot_state.shape != (2 * joint_count,):
        raise PickCubeSourceGenerationError(
            "named Panda state does not contain qpos followed by qvel"
        )
    agent = getattr(base, "agent", None)
    tcp = getattr(agent, "tcp", None)
    cube = getattr(base, "cube", None)
    goal_site = getattr(base, "goal_site", None)
    tcp_position, tcp_quaternion = _public_pose(tcp, "TCP")
    cube_position, cube_quaternion = _public_pose(cube, "cube")
    goal_position, _ = _public_pose(goal_site, "goal")
    verifier_state = build_pickcube_verifier_state_v1(
        joint_names=joint_names,
        qpos=np.asarray(robot_state[:joint_count]),
        qvel=np.asarray(robot_state[joint_count:]),
        tcp_position=tcp_position,
        tcp_quaternion=tcp_quaternion,
        cube_position=cube_position,
        cube_quaternion=cube_quaternion,
        goal_position=goal_position,
        is_grasped=flags["is_grasped"],
        is_obj_placed=flags["is_obj_placed"],
        is_robot_static=flags["is_robot_static"],
    )
    cube_to_goal = _finite_scalar(
        raw_task.cube_to_goal_distance, "cube-to-goal distance"
    )
    cube_center_z = _finite_scalar(raw_task.cube_center_z, "cube center z")
    tcp_to_cube = float(np.linalg.norm(tcp_position - cube_position))
    task_snapshot: Mapping[str, object] = MappingProxyType(
        {
            "success": flags["success"],
            "is_obj_placed": flags["is_obj_placed"],
            "is_robot_static": flags["is_robot_static"],
            "is_grasped": flags["is_grasped"],
            "cube_center_z": cube_center_z,
            "cube_to_goal_distance": cube_to_goal,
            "tcp_to_cube_distance": tcp_to_cube,
        }
    )
    return CapturedPickCubeStateBoundary(
        state_index=state_index,
        state_tree=state_tree,
        task_snapshot=task_snapshot,
        verifier_state=verifier_state,
    )


def _task_flags(
    snapshot: RawPickCubeTaskSnapshot,
    contract: PickCubeTaskKeyContract,
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, key in (
        ("success", contract.success),
        ("is_obj_placed", contract.object_placed),
        ("is_robot_static", contract.robot_static),
        ("is_grasped", contract.grasped),
    ):
        if key not in snapshot.evaluator_values:
            raise PickCubeSourceGenerationError(
                f"PickCube boundary task snapshot is missing {name}"
            )
        result[name] = _strict_bool(snapshot.evaluator_values[key], name)
    return result


def _public_pose(entity: object, context: str) -> tuple[NDArray[Any], NDArray[Any]]:
    pose = getattr(entity, "pose", None)
    position = _finite_vector(getattr(pose, "p", None), 3, f"{context} position")
    quaternion = _finite_vector(getattr(pose, "q", None), 4, f"{context} quaternion")
    return position, quaternion


def _runtime_array(value: object, context: str) -> NDArray[Any]:
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
        raise PickCubeSourceGenerationError(f"{context} must be numeric")
    if not bool(np.all(np.isfinite(array))):
        raise PickCubeSourceGenerationError(f"{context} must be finite")
    return array


def _finite_vector(value: object, size: int, context: str) -> NDArray[Any]:
    array = _runtime_array(value, context)
    if array.shape not in ((size,), (1, size)):
        raise PickCubeSourceGenerationError(
            f"{context} must have shape {(size,)} or {(1, size)}, got {array.shape}"
        )
    return np.array(array.reshape(size), copy=True, order="C")


def _finite_scalar(value: object, context: str) -> float:
    array = _runtime_array(value, context)
    if array.size != 1:
        raise PickCubeSourceGenerationError(f"{context} must be scalar")
    result = float(array.reshape(()))
    if not math.isfinite(result):
        raise PickCubeSourceGenerationError(f"{context} must be finite")
    return result


def _strict_bool(value: object, context: str) -> bool:
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
    if array.size != 1 or not np.issubdtype(array.dtype, np.bool_):
        raise PickCubeSourceGenerationError(f"{context} must be boolean scalar")
    return bool(array.reshape(()))


__all__ = [
    "CapturedPickCubeStateBoundary",
    "capture_pickcube_state_boundary",
]
