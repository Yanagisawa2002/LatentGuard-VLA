"""CPU-only Robot Episode Toolkit API tests."""

from __future__ import annotations

import json
from dataclasses import MISSING, fields, replace

import numpy as np
import pytest

from latentguard.models import ActionChunk
from latentguard.synthetic import generate_synthetic_episodes
from latentguard.toolkit import (
    AuditConfig,
    AuditSeverity,
    DataIssue,
    ReplayAlignmentError,
    audit_episodes,
    build_replay_plan,
    iter_replay_steps,
    report_json,
    summarize_episodes,
)


def _episode():
    return generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=4,
        action_dim=2,
        robot_state_dim=3,
        camera_count=1,
        image_height=2,
        image_width=2,
        candidate_count=4,
    )[0]


def _codes(report):
    return [issue.issue_code for issue in report.issues]


def test_data_issue_details_use_an_immutable_factory_default() -> None:
    details_field = next(item for item in fields(DataIssue) if item.name == "details")
    assert details_field.default is MISSING
    assert details_field.default_factory is not MISSING

    first = DataIssue("TEST", AuditSeverity.INFO, "first")
    second = DataIssue("TEST", AuditSeverity.INFO, "second")
    assert first.details == second.details == {}
    assert first.details is not second.details
    with pytest.raises(TypeError):
        first.details["unexpected"] = 1  # type: ignore[index]


def test_audit_clean_synthetic_has_no_error_and_is_deterministic() -> None:
    episode = _episode()
    first = audit_episodes((episode,))
    second = audit_episodes((episode,))
    assert first.issues_by_severity.get("error", 0) == 0
    assert report_json(first) == report_json(second)
    assert episode.candidates[0].action.actions.flags.writeable is False


def test_audit_detects_alignment_drift_camera_and_frozen_rgb() -> None:
    episode = _episode()
    frames = list(episode.observations.frames)
    first_camera = frames[0].cameras[0]
    frames[1] = replace(frames[1], timestamp_s=0.11, cameras=(first_camera,))
    frames[2] = replace(frames[2], timestamp_s=0.21, cameras=(first_camera,))
    frames[3] = replace(frames[3], timestamp_s=0.31, cameras=())
    candidates = tuple(
        replace(
            candidate,
            action=replace(candidate.action, actions=candidate.action.actions[:3]),
        )
        for candidate in episode.candidates
    )
    changed = replace(
        episode,
        observations=replace(episode.observations, frames=tuple(frames)),
        candidates=candidates,
    )
    codes = _codes(audit_episodes((changed,)))
    assert "REPLAY_HORIZON_MISMATCH" in codes
    assert "TIMESTAMP_PERIOD_DRIFT" in codes
    assert "CAMERA_SET_CHANGED" in codes
    assert "CAMERA_FROZEN" in codes


def test_audit_detects_action_and_robot_state_quality_issues() -> None:
    episode = _episode()
    frames = tuple(
        replace(frame, robot_state=episode.observations.frames[0].robot_state)
        for frame in episode.observations.frames
    )
    zero = np.zeros_like(episode.candidates[0].action.actions)
    jump = zero.copy()
    jump[-1] = 5.0
    candidates = list(episode.candidates)
    candidates[0] = replace(
        candidates[0], action=replace(candidates[0].action, actions=zero)
    )
    candidates[1] = replace(
        candidates[1], action=replace(candidates[1].action, actions=jump)
    )
    candidates[2] = replace(
        candidates[2], action=replace(candidates[2].action, actions=zero)
    )
    changed = replace(
        episode,
        observations=replace(episode.observations, frames=frames),
        candidates=tuple(candidates),
    )
    codes = _codes(
        audit_episodes(
            (changed,),
            AuditConfig(action_jump_threshold=1.0, frozen_frame_run_length=3),
        )
    )
    assert "ACTION_NEAR_ZERO_DOMINANT" in codes
    assert "ROBOT_STATE_FROZEN" in codes
    assert "ACTION_DISCONTINUITY" in codes
    assert "DUPLICATE_CANDIDATE_ACTION" in codes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timestamp_tolerance_s", float("nan")),
        ("frozen_frame_run_length", True),
        ("near_zero_action_threshold", -1.0),
        ("near_zero_action_fraction", 1.1),
        ("action_jump_threshold", float("inf")),
    ],
)
def test_invalid_audit_config_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        AuditConfig(**{field: value})  # type: ignore[arg-type]


def test_offline_replay_exact_pairing_repeatability_and_mutation_safety() -> None:
    episode = _episode()
    candidate = episode.candidates[1]
    plan = build_replay_plan(episode, candidate.candidate_id)
    first = tuple(iter_replay_steps(episode, candidate.candidate_id))
    second = tuple(iter_replay_steps(episode, candidate.candidate_id))
    assert plan.step_count == len(first) == 4
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        assert left.observation is episode.observations.frames[index]
        assert np.array_equal(left.action, candidate.action.actions[index])
        assert np.array_equal(left.action, right.action)
        assert left.nominal_action_timestamp_s == index * plan.control_period_s
        assert left.action.flags.writeable is False
    with pytest.raises(ValueError):
        first[0].action[0] = 99.0


def test_offline_replay_rejects_missing_candidate_and_mismatch() -> None:
    episode = _episode()
    with pytest.raises(ReplayAlignmentError, match="no candidate"):
        build_replay_plan(episode, "missing")
    candidates = tuple(
        replace(
            candidate,
            action=ActionChunk(
                candidate.action.actions[:-1],
                candidate.action.coordinate_frame,
                candidate.action.control_period_s,
            ),
        )
        for candidate in episode.candidates
    )
    with pytest.raises(ReplayAlignmentError, match="horizon"):
        tuple(
            iter_replay_steps(
                replace(episode, candidates=candidates), candidates[0].candidate_id
            )
        )


def test_metrics_have_candidate_denominators_groups_and_stable_json() -> None:
    episodes = (_episode(),)
    metrics = summarize_episodes(episodes)
    assert metrics.episode_count == 1
    assert metrics.observation_count == 4
    assert metrics.candidate_count == 4
    assert metrics.successful_candidate_count == 1
    assert metrics.unsafe_candidate_count == 2
    assert metrics.candidate_success_rate == 0.25
    assert metrics.candidate_unsafe_rate == 0.5
    assert metrics.mean_progress == pytest.approx((1.0 + 0.65 + 0.25 + 0.45) / 4)
    assert metrics.failure_type_histogram == {
        "collision": 1,
        "control_limit": 1,
        "grasp_failure": 1,
    }
    assert metrics.by_task[episodes[0].task_id].candidate_count == 4
    assert metrics.by_source_policy[episodes[0].source_policy_id].episode_count == 1
    assert json.loads(report_json(metrics))["candidate_count"] == 4
    assert "NaN" not in report_json(metrics)


def test_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_episodes(())
