"""CPU-safe P0.2 temporal, phase, loss, event, and gate tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from latentguard.policies.act.grasp_supervision import (
    PickCubeGraspSupervisionError,
    PickCubeGripperEvent,
    PickCubeManipulationPhase,
    PickCubeP02Result,
    PickCubeProgressSample,
    PickCubeRolloutFailure,
    PickCubeStagedGateCounts,
    action_chunk_at_offset,
    aligned_frame_pairs,
    analyze_progress_trace,
    classify_p02_result,
    close_transition_index_in_window,
    derive_gripper_events,
    final_access_authorized,
    phase_from_expert_label,
    phase_gripper_aware_loss,
    phase_weights_from_train_labels,
    predicted_close_index,
)
from latentguard.policies.act.types import PickCubeActGraspSupervisionConfig


def test_temporal_offsets_and_terminal_padding_never_cross_episode() -> None:
    assert aligned_frame_pairs(4, -1) == ((1, 0), (2, 1), (3, 2))
    assert aligned_frame_pairs(4, 0) == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert aligned_frame_pairs(4, 2) == ((0, 2), (1, 3))
    actions = np.arange(32, dtype=np.float32).reshape(4, 8)
    chunk, padding = action_chunk_at_offset(actions, 1, 4, 2)
    np.testing.assert_array_equal(chunk[0], actions[3])
    np.testing.assert_array_equal(chunk[1:], np.zeros((3, 8), dtype=np.float32))
    np.testing.assert_array_equal(padding, [False, True, True, True])
    with pytest.raises(PickCubeGraspSupervisionError, match="leaves the episode"):
        action_chunk_at_offset(actions, 3, 4, 2)


def test_phase_labels_are_diagnostic_only_and_explicit() -> None:
    assert (
        phase_from_expert_label("REACH_PREGRASP") is PickCubeManipulationPhase.APPROACH
    )
    assert (
        phase_from_expert_label("DESCEND_TO_GRASP")
        is PickCubeManipulationPhase.PREGRASP
    )
    assert (
        phase_from_expert_label("CLOSE_GRIPPER")
        is PickCubeManipulationPhase.GRIPPER_CLOSING
    )
    with pytest.raises(PickCubeGraspSupervisionError, match="unsupported"):
        phase_from_expert_label("UNVERIFIED_CONTACT")


def test_gripper_sign_scale_and_transition_are_not_ambiguous() -> None:
    events = derive_gripper_events(
        np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32),
        close_threshold=-0.5,
        open_threshold=0.5,
    )
    np.testing.assert_array_equal(
        events,
        [
            PickCubeGripperEvent.OPEN,
            PickCubeGripperEvent.OPEN,
            PickCubeGripperEvent.CLOSING,
            PickCubeGripperEvent.CLOSED,
        ],
    )
    with pytest.raises(PickCubeGraspSupervisionError, match="ambiguous"):
        derive_gripper_events(
            [1.0, 0.0, -1.0], close_threshold=-0.5, open_threshold=0.5
        )


def test_close_event_metric_uses_transition_windows_not_closed_frames() -> None:
    phases = (
        PickCubeManipulationPhase.APPROACH.value,
        PickCubeManipulationPhase.PREGRASP.value,
        PickCubeManipulationPhase.GRIPPER_CLOSING.value,
        PickCubeManipulationPhase.GRIPPER_CLOSING.value,
        PickCubeManipulationPhase.TRANSPORT_OR_COMPLETION.value,
    )
    assert (
        close_transition_index_in_window(phases, window_start=0, window_length=4) == 2
    )
    assert (
        close_transition_index_in_window(phases, window_start=3, window_length=2)
        is None
    )
    constant_close = np.zeros((4, 8), dtype=np.float32)
    constant_close[:, 7] = -1.0
    assert predicted_close_index(constant_close, close_threshold=-0.5) == 0


def test_phase_weights_are_train_only_mean_one_and_bounded() -> None:
    labels = ["APPROACH"] * 80 + ["PREGRASP"] * 15 + ["GRIPPER_CLOSING"] * 5
    weights = phase_weights_from_train_labels(labels, exponent=0.5, maximum=3.0)
    weighted_mean = sum(weights[label] for label in labels) / len(labels)
    assert weighted_mean == pytest.approx(1.0)
    assert max(weights.values()) <= 3.0
    assert weights["GRIPPER_CLOSING"] > weights["APPROACH"]


def test_phase_aware_loss_separates_arm_gripper_and_transition() -> None:
    config = PickCubeActGraspSupervisionConfig()
    raw = torch.zeros((1, 3, 8), requires_grad=True)
    bounded = torch.tanh(raw)
    target = torch.zeros((1, 3, 8))
    target[..., 7] = torch.tensor([1.0, -1.0, -1.0])
    loss, metrics = phase_gripper_aware_loss(
        raw_actions=raw,
        bounded_actions=bounded,
        target_actions=target,
        action_is_pad=torch.tensor([[False, False, True]]),
        phase_weights=torch.ones((1, 3)),
        gripper_events=torch.tensor(
            [[PickCubeGripperEvent.OPEN, PickCubeGripperEvent.CLOSING, 0]]
        ),
        config=config,
        mean=None,
        log_variance=None,
        kl_weight=10.0,
    )
    assert metrics["arm_action_loss"] == 0.0
    assert metrics["gripper_action_loss"] == 1.0
    assert metrics["gripper_transition_loss"] > 0.0
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad[..., :7]) == 0
    assert torch.count_nonzero(raw.grad[..., 7]) > 0


def _sample(
    step: int,
    *,
    distance: float,
    close: bool = False,
    contact: bool = False,
    grasped: bool = False,
    cube_z: float = 0.02,
) -> PickCubeProgressSample:
    joints = [0.0] * 9
    joints[0] = step * 0.02
    action = [0.0] * 8
    action[7] = -1.0 if close else 1.0
    return PickCubeProgressSample(
        step_index=step,
        query_index=step // 2,
        action_index_in_chunk=step % 2,
        joint_positions=tuple(joints),
        commanded_action=tuple(action),
        gripper_position=0.02,
        commanded_gripper=action[7],
        tcp_position=(0.0, 0.0, 0.1),
        cube_position=(0.0, 0.0, cube_z),
        tcp_to_cube_distance=distance,
        left_contact_force=0.1 if contact else 0.0,
        right_contact_force=0.1 if contact else 0.0,
        grasped=grasped,
        environment_step_latency_seconds=0.01,
    )


def test_rollout_events_and_failure_taxonomy_detect_real_grasp_progress() -> None:
    trace = (
        _sample(0, distance=0.2),
        _sample(1, distance=0.07),
        _sample(2, distance=0.04, close=True, contact=True, grasped=True),
        _sample(
            3,
            distance=0.04,
            close=True,
            contact=True,
            grasped=True,
            cube_z=0.06,
        ),
    )
    summary = analyze_progress_trace(
        trace,
        initial_joint_positions=(0.0,) * 9,
        initial_cube_height=0.02,
        success=False,
    )
    assert summary.entered_pregrasp_region == 1
    assert summary.first_valid_close == 2
    assert summary.first_contact == 2
    assert summary.first_grasp == 2
    assert summary.first_lift == 3
    assert summary.failure is PickCubeRolloutFailure.TIMEOUT_AFTER_PROGRESS


def test_staged_gates_result_and_sealed_access_fail_closed() -> None:
    accepted = PickCubeStagedGateCounts(30, 0, 30, 30, 30, 30, 23)
    assert classify_p02_result(accepted) is PickCubeP02Result.RESULT_A_ACCEPTED
    assert final_access_authorized(PickCubeP02Result.RESULT_A_ACCEPTED)
    improved = PickCubeStagedGateCounts(30, 0, 24, 21, 18, 15, 20)
    assert classify_p02_result(improved) is PickCubeP02Result.RESULT_B_GRASP_UNPROMOTED
    assert not final_access_authorized(PickCubeP02Result.RESULT_B_GRASP_UNPROMOTED)
    failed = PickCubeStagedGateCounts(30, 0, 24, 10, 4, 0, 0)
    assert classify_p02_result(failed) is PickCubeP02Result.RESULT_C_NO_GRASP
    with pytest.raises(PickCubeGraspSupervisionError, match="action integrity"):
        classify_p02_result(PickCubeStagedGateCounts(30, 1, 30, 30, 30, 30, 30))
