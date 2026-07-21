"""PickCube-specific phase metadata for WM-v0 D1 anchor preparation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from latentguard.world_model.d1_collection import EpisodePhase


@dataclass(frozen=True, slots=True)
class PickCubeD1AnchorMetadata:
    """Outcome-free task projection recorded at one exact PickCube anchor."""

    episode_phase: EpisodePhase
    task_progress_t: float
    distance_to_target: float
    grasp_state: bool
    object_state: str


def pickcube_d1_anchor_metadata(
    *,
    selection_reason: str,
    state_index: int,
    task_snapshot: Mapping[str, object],
) -> PickCubeD1AnchorMetadata:
    """Classify one anchor from its current task state, never future outcomes."""

    if type(state_index) is not int or state_index < 0:
        raise ValueError("PickCube D1 state_index must be non-negative")
    grasped = _bool(task_snapshot, "is_grasped")
    placed = _bool(task_snapshot, "is_obj_placed")
    static = _bool(task_snapshot, "is_robot_static")
    tcp_distance = _float(task_snapshot, "tcp_to_cube_distance")
    goal_distance = _float(task_snapshot, "cube_to_goal_distance")
    if state_index == 0 or selection_reason == "early_trajectory":
        phase = EpisodePhase.INITIAL
    elif selection_reason == "first_grasp_transition":
        phase = EpisodePhase.GRASP_ATTEMPT
    elif selection_reason == "near_placement":
        phase = EpisodePhase.PRE_PLACE
    elif selection_reason == "late_transport":
        phase = EpisodePhase.TRANSPORT
    elif selection_reason == "approach_phase":
        phase = (
            EpisodePhase.PRE_GRASP if tcp_distance <= 0.06 else EpisodePhase.APPROACH
        )
    elif selection_reason == "evenly_spaced_fill":
        if placed:
            phase = EpisodePhase.RELEASE
        elif grasped:
            phase = (
                EpisodePhase.POST_GRASP
                if goal_distance > 0.10
                else EpisodePhase.PRE_PLACE
            )
        elif tcp_distance <= 0.06:
            phase = EpisodePhase.PRE_GRASP
        else:
            phase = EpisodePhase.APPROACH
    else:
        phase = EpisodePhase.UNKNOWN
    if placed:
        progress = 1.0 if static else 0.8
        object_state = "placed"
        distance = goal_distance
    elif grasped:
        progress = 0.5 + 0.25 * (1.0 - min(goal_distance / 0.5, 1.0))
        object_state = "grasped"
        distance = goal_distance
    else:
        progress = 0.25 * (1.0 - min(tcp_distance / 0.5, 1.0))
        object_state = "ungrasped"
        distance = tcp_distance
    return PickCubeD1AnchorMetadata(
        episode_phase=phase,
        task_progress_t=progress,
        distance_to_target=distance,
        grasp_state=grasped,
        object_state=object_state,
    )


def _bool(value: Mapping[str, object], name: str) -> bool:
    result = value.get(name)
    if type(result) is not bool:
        raise ValueError(f"PickCube D1 task snapshot {name} must be boolean")
    return result


def _float(value: Mapping[str, object], name: str) -> float:
    result = value.get(name)
    if type(result) is not float or not math.isfinite(result):
        raise ValueError(f"PickCube D1 task snapshot {name} must be finite float")
    return result


__all__ = ["PickCubeD1AnchorMetadata", "pickcube_d1_anchor_metadata"]
