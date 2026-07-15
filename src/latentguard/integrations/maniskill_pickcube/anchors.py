"""Deterministic public-evidence anchor selection for PickCube sequences."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from latentguard.replay.identity import canonical_json_bytes

DEFAULT_CANDIDATE_HORIZON = 16
DEFAULT_MAX_ANCHORS_PER_TRAJECTORY = 6


class PickCubeAnchorError(ValueError):
    """Raised when state facts cannot define safe deterministic anchors."""


@dataclass(frozen=True, slots=True)
class PickCubeAnchorStateFacts:
    """Public task facts captured at one pre-action state boundary."""

    state_index: int
    success: bool
    grasped: bool
    object_placed: bool
    robot_static: bool
    cube_to_goal_distance: float
    tcp_to_cube_distance: float
    complete: bool = True

    def __post_init__(self) -> None:
        """Validate exact scalar task facts without inferring missing values."""
        if type(self.state_index) is not int or self.state_index < 0:
            raise PickCubeAnchorError("anchor state index must be non-negative")
        for name in (
            "success",
            "grasped",
            "object_placed",
            "robot_static",
            "complete",
        ):
            if type(getattr(self, name)) is not bool:
                raise PickCubeAnchorError(f"anchor fact {name} must be boolean")
        for name in ("cube_to_goal_distance", "tcp_to_cube_distance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise PickCubeAnchorError(f"anchor fact {name} must be finite")
            if float(value) < 0.0:
                raise PickCubeAnchorError(f"anchor fact {name} must be non-negative")


@dataclass(frozen=True, slots=True)
class PickCubeStateAnchor:
    """One content-independent scheduled anchor before runtime baseline gating."""

    anchor_id: str
    state_index: int
    selection_reason: str
    source_trajectory_id: str
    source_seed: int
    split_group_id: str
    remaining_horizon: int
    candidate_horizon: int


def split_group_id_for_trajectory(source_trajectory_id: str) -> str:
    """Return the stable split group owned by one complete source trajectory."""
    _require_text(source_trajectory_id, "source trajectory ID")
    digest = hashlib.sha256(source_trajectory_id.encode("utf-8")).hexdigest()
    return f"mspc-split-{digest}"


def select_pickcube_state_anchors(
    *,
    source_trajectory_id: str,
    source_seed: int,
    action_count: int,
    state_facts: tuple[PickCubeAnchorStateFacts, ...],
    candidate_horizon: int = DEFAULT_CANDIDATE_HORIZON,
    maximum_anchor_count: int = DEFAULT_MAX_ANCHORS_PER_TRAJECTORY,
) -> tuple[PickCubeStateAnchor, ...]:
    """Select event-informed anchors and deterministically fill collisions."""
    _require_text(source_trajectory_id, "source trajectory ID")
    if type(source_seed) is not int or not 0 <= source_seed < 2**32:
        raise PickCubeAnchorError("source seed must be in [0, 2**32)")
    if type(action_count) is not int or action_count <= 0:
        raise PickCubeAnchorError("action count must be positive")
    if type(candidate_horizon) is not int or candidate_horizon <= 0:
        raise PickCubeAnchorError("candidate horizon must be positive")
    if type(maximum_anchor_count) is not int or maximum_anchor_count <= 0:
        raise PickCubeAnchorError("maximum anchor count must be positive")
    if len(state_facts) != action_count + 1:
        raise PickCubeAnchorError("anchor facts must cover every T+1 state")
    if tuple(fact.state_index for fact in state_facts) != tuple(
        range(action_count + 1)
    ):
        raise PickCubeAnchorError("anchor fact indices must be contiguous and ordered")

    eligible = tuple(
        fact.state_index
        for fact in state_facts
        if fact.complete
        and not fact.success
        and fact.state_index + candidate_horizon <= action_count
    )
    if not eligible:
        return ()
    eligible_set = set(eligible)
    target_count = min(maximum_anchor_count, len(eligible))

    grasp_transition = next(
        (
            index
            for index in range(1, len(state_facts))
            if not state_facts[index - 1].grasped and state_facts[index].grasped
        ),
        None,
    )
    placed_index = next(
        (fact.state_index for fact in state_facts if fact.object_placed), None
    )
    pre_grasp = tuple(
        index
        for index in eligible
        if grasp_transition is None or index <= grasp_transition
    )
    transport = tuple(
        index
        for index in eligible
        if grasp_transition is not None and index >= grasp_transition
    )

    proposals: list[tuple[str, int | None]] = [
        ("early_trajectory", eligible[0]),
        (
            "approach_phase",
            _minimum_by_distance(pre_grasp or eligible, state_facts, tcp=True),
        ),
        (
            "first_grasp_transition",
            _nearest_eligible(grasp_transition, eligible),
        ),
        (
            "early_transport",
            None if not transport else transport[0],
        ),
        (
            "late_transport",
            _minimum_by_distance(transport, state_facts, tcp=False),
        ),
        (
            "near_placement",
            _nearest_eligible(
                None if placed_index is None else placed_index - candidate_horizon,
                eligible,
            ),
        ),
    ]
    selected: list[tuple[int, str]] = []
    selected_indices: set[int] = set()
    for reason, index in proposals:
        if (
            index is not None
            and index in eligible_set
            and index not in selected_indices
            and len(selected) < target_count
        ):
            selected.append((index, reason))
            selected_indices.add(index)

    for index in _evenly_spaced_indices(eligible, target_count):
        if len(selected) >= target_count:
            break
        if index not in selected_indices:
            selected.append((index, "evenly_spaced_fill"))
            selected_indices.add(index)
    if len(selected) < target_count:
        for index in eligible:
            if len(selected) >= target_count:
                break
            if index not in selected_indices:
                selected.append((index, "ordered_eligible_fill"))
                selected_indices.add(index)

    split_group_id = split_group_id_for_trajectory(source_trajectory_id)
    return tuple(
        PickCubeStateAnchor(
            anchor_id=_anchor_identifier(
                source_trajectory_id=source_trajectory_id,
                state_index=index,
                action_count=action_count,
                candidate_horizon=candidate_horizon,
            ),
            state_index=index,
            selection_reason=reason,
            source_trajectory_id=source_trajectory_id,
            source_seed=source_seed,
            split_group_id=split_group_id,
            remaining_horizon=action_count - index,
            candidate_horizon=candidate_horizon,
        )
        for index, reason in selected
    )


def _minimum_by_distance(
    indices: tuple[int, ...],
    facts: tuple[PickCubeAnchorStateFacts, ...],
    *,
    tcp: bool,
) -> int | None:
    if not indices:
        return None
    return min(
        indices,
        key=lambda index: (
            facts[index].tcp_to_cube_distance
            if tcp
            else facts[index].cube_to_goal_distance,
            index,
        ),
    )


def _nearest_eligible(target: int | None, eligible: tuple[int, ...]) -> int | None:
    if target is None or not eligible:
        return None
    return min(eligible, key=lambda index: (abs(index - target), index))


def _evenly_spaced_indices(eligible: tuple[int, ...], count: int) -> tuple[int, ...]:
    if count <= 0 or not eligible:
        return ()
    if count == 1:
        return (eligible[0],)
    positions = tuple(
        round(index * (len(eligible) - 1) / (count - 1)) for index in range(count)
    )
    return tuple(eligible[position] for position in positions)


def _anchor_identifier(
    *,
    source_trajectory_id: str,
    state_index: int,
    action_count: int,
    candidate_horizon: int,
) -> str:
    payload = {
        "action_count": action_count,
        "candidate_horizon": candidate_horizon,
        "source_trajectory_id": source_trajectory_id,
        "state_index": state_index,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"mspc-anchor-{digest}"


def _require_text(value: object, context: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PickCubeAnchorError(f"{context} must be non-empty canonical text")


__all__ = [
    "DEFAULT_CANDIDATE_HORIZON",
    "DEFAULT_MAX_ANCHORS_PER_TRAJECTORY",
    "PickCubeAnchorError",
    "PickCubeAnchorStateFacts",
    "PickCubeStateAnchor",
    "select_pickcube_state_anchors",
    "split_group_id_for_trajectory",
]
