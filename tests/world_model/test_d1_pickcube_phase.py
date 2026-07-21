"""PickCube WM-v0 D1 phase classification tests."""

from latentguard.integrations.maniskill_pickcube.world_model_d1 import (
    pickcube_d1_anchor_metadata,
)
from latentguard.world_model.d1_collection import EpisodePhase


def _snapshot(
    *, grasped: bool = False, placed: bool = False, tcp: float = 0.2, goal: float = 0.3
) -> dict[str, object]:
    return {
        "cube_to_goal_distance": goal,
        "is_grasped": grasped,
        "is_obj_placed": placed,
        "is_robot_static": True,
        "tcp_to_cube_distance": tcp,
    }


def test_phase_classifier_uses_current_state_and_explicit_selection_reason() -> None:
    """Phase labels are reproducible without consulting future task outcomes."""

    initial = pickcube_d1_anchor_metadata(
        selection_reason="early_trajectory",
        state_index=0,
        task_snapshot=_snapshot(),
    )
    pregrasp = pickcube_d1_anchor_metadata(
        selection_reason="approach_phase",
        state_index=20,
        task_snapshot=_snapshot(tcp=0.03),
    )
    transport = pickcube_d1_anchor_metadata(
        selection_reason="late_transport",
        state_index=45,
        task_snapshot=_snapshot(grasped=True),
    )

    assert initial.episode_phase is EpisodePhase.INITIAL
    assert pregrasp.episode_phase is EpisodePhase.PRE_GRASP
    assert transport.episode_phase is EpisodePhase.TRANSPORT
    assert transport.object_state == "grasped"
