from __future__ import annotations

from latentguard.integrations.maniskill_pickcube.anchors import (
    PickCubeAnchorStateFacts,
    select_pickcube_state_anchors,
)


def _facts(action_count: int = 40) -> tuple[PickCubeAnchorStateFacts, ...]:
    return tuple(
        PickCubeAnchorStateFacts(
            state_index=index,
            success=index == action_count,
            grasped=10 <= index < 35,
            object_placed=index >= 35,
            robot_static=index == action_count,
            cube_to_goal_distance=max(0.0, (35 - index) / 35),
            tcp_to_cube_distance=abs(9 - index) / 10,
        )
        for index in range(action_count + 1)
    )


def test_anchor_schedule_is_deterministic_event_informed_and_fixed_horizon() -> None:
    first = select_pickcube_state_anchors(
        source_trajectory_id="trajectory-a",
        source_seed=7,
        action_count=40,
        state_facts=_facts(),
    )
    second = select_pickcube_state_anchors(
        source_trajectory_id="trajectory-a",
        source_seed=7,
        action_count=40,
        state_facts=_facts(),
    )
    assert first == second
    assert len(first) == 6
    assert len({anchor.state_index for anchor in first}) == 6
    assert all(anchor.candidate_horizon == 16 for anchor in first)
    assert all(anchor.remaining_horizon >= 16 for anchor in first)
    assert "first_grasp_transition" in {anchor.selection_reason for anchor in first}


def test_colliding_event_anchors_are_deduplicated_and_filled() -> None:
    facts = tuple(
        PickCubeAnchorStateFacts(
            state_index=index,
            success=False,
            grasped=index >= 1,
            object_placed=index >= 2,
            robot_static=False,
            cube_to_goal_distance=float(index),
            tcp_to_cube_distance=float(index),
        )
        for index in range(25)
    )
    anchors = select_pickcube_state_anchors(
        source_trajectory_id="trajectory-b",
        source_seed=8,
        action_count=24,
        state_facts=facts,
    )
    assert len(anchors) == 6
    assert len({anchor.state_index for anchor in anchors}) == 6
    assert any("fill" in anchor.selection_reason for anchor in anchors)


def test_anchor_schedule_excludes_terminal_and_short_remainders() -> None:
    anchors = select_pickcube_state_anchors(
        source_trajectory_id="trajectory-c",
        source_seed=9,
        action_count=15,
        state_facts=_facts(action_count=15),
    )
    assert anchors == ()
