"""Tests for descriptive schema and cross-entity validation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from latentguard.models import (
    CameraFrame,
    Episode,
    LabelSource,
    LabelStrength,
)
from latentguard.synthetic import generate_synthetic_episodes
from latentguard.validation import DataValidationError, validate_episode


def _episode() -> Episode:
    return generate_synthetic_episodes(
        seed=5,
        episode_count=1,
        episode_length=3,
        action_dim=2,
        robot_state_dim=4,
        camera_count=1,
        image_height=3,
        image_width=4,
        depth_enabled=True,
        candidate_count=4,
    )[0]


def _replace_first_frame(episode: Episode, **changes: Any) -> Episode:
    frames = episode.observations.frames
    changed = replace(frames[0], **changes)
    history = replace(episode.observations, frames=(changed, *frames[1:]))
    return replace(episode, observations=history)


def _replace_camera(episode: Episode, camera: CameraFrame) -> Episode:
    return _replace_first_frame(episode, cameras=(camera,))


def _replace_candidate(episode: Episode, index: int, **changes: Any) -> Episode:
    candidates = list(episode.candidates)
    candidates[index] = replace(candidates[index], **changes)
    return replace(episode, candidates=tuple(candidates))


def test_validation_error_exposes_required_context() -> None:
    """Errors name the entity, field, offending identifier, and concise reason."""
    episode = _episode()
    camera = episode.observations.frames[0].cameras[0]
    invalid = _replace_camera(
        episode,
        replace(camera, rgb=np.zeros((3, 4), dtype=np.uint8)),
    )

    with pytest.raises(DataValidationError) as captured:
        validate_episode(invalid)

    error = captured.value
    assert error.entity_type == "CameraFrame"
    assert error.field_name == "rgb"
    assert error.identifier == camera.camera_id
    assert "expected positive" in error.reason
    assert "CameraFrame[id=camera-00].rgb" in str(error)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_invalid_rgb_shapes_are_rejected(rgb: np.ndarray[Any, Any]) -> None:
    """RGB must remain a positive height-width-three array."""
    episode = _episode()
    camera = episode.observations.frames[0].cameras[0]
    with pytest.raises(DataValidationError, match=r"CameraFrame.*\.rgb"):
        validate_episode(_replace_camera(episode, replace(camera, rgb=rgb)))


def test_invalid_rgb_dtype_is_rejected() -> None:
    """RGB uses an unambiguous uint8 representation."""
    episode = _episode()
    camera = episode.observations.frames[0].cameras[0]
    rgb = np.zeros((3, 4, 3), dtype=np.float32)
    with pytest.raises(DataValidationError, match="uint8"):
        validate_episode(_replace_camera(episode, replace(camera, rgb=rgb)))


def test_invalid_depth_rank_is_rejected() -> None:
    """Depth must be a two-dimensional image."""
    episode = _episode()
    camera = episode.observations.frames[0].cameras[0]
    depth = np.zeros((3, 4, 1), dtype=np.float32)
    with pytest.raises(DataValidationError, match=r"CameraFrame.*\.depth"):
        validate_episode(_replace_camera(episode, replace(camera, depth=depth)))


def test_rgb_depth_resolution_mismatch_is_rejected() -> None:
    """Depth resolution must exactly match its camera's RGB resolution."""
    episode = _episode()
    camera = episode.observations.frames[0].cameras[0]
    depth = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(DataValidationError, match="does not match RGB"):
        validate_episode(_replace_camera(episode, replace(camera, depth=depth)))


def test_negative_observation_timestamp_is_rejected() -> None:
    """Observation timestamps cannot precede episode time zero."""
    with pytest.raises(DataValidationError, match="must be non-negative"):
        validate_episode(_replace_first_frame(_episode(), timestamp_s=-0.1))


def test_non_monotonic_timestamps_are_rejected() -> None:
    """Repeated and decreasing observation timestamps fail validation."""
    episode = _episode()
    frames = episode.observations.frames
    second = replace(frames[1], timestamp_s=frames[0].timestamp_s)
    history = replace(episode.observations, frames=(frames[0], second, frames[2]))
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_episode(replace(episode, observations=history))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_robot_state_is_rejected(bad_value: float) -> None:
    """Robot-state NaN and infinity values never pass validation."""
    episode = _episode()
    state = np.array(episode.observations.frames[0].robot_state, copy=True)
    state[0] = bad_value
    with pytest.raises(DataValidationError, match="NaN or infinity"):
        validate_episode(_replace_first_frame(episode, robot_state=state))


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros(3, dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((3, 0), dtype=np.float32),
    ],
)
def test_invalid_action_shapes_are_rejected(actions: np.ndarray[Any, Any]) -> None:
    """Actions require rank two, a positive horizon, and a positive dimension."""
    episode = _episode()
    candidate = episode.candidates[0]
    action = replace(candidate.action, actions=actions)
    with pytest.raises(DataValidationError, match=r"ActionChunk.*\.actions"):
        validate_episode(_replace_candidate(episode, 0, action=action))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_actions_are_rejected(bad_value: float) -> None:
    """Action arrays reject every non-finite value."""
    episode = _episode()
    candidate = episode.candidates[0]
    actions = np.array(candidate.action.actions, copy=True)
    actions[0, 0] = bad_value
    action = replace(candidate.action, actions=actions)
    with pytest.raises(DataValidationError, match="NaN or infinity"):
        validate_episode(_replace_candidate(episode, 0, action=action))


@pytest.mark.parametrize("period", [0.0, -0.1, np.nan, np.inf])
def test_invalid_control_periods_are_rejected(period: float) -> None:
    """Control periods must be finite and strictly positive."""
    episode = _episode()
    candidate = episode.candidates[0]
    action = replace(candidate.action, control_period_s=period)
    with pytest.raises(DataValidationError, match="control_period_s"):
        validate_episode(_replace_candidate(episode, 0, action=action))


def test_mismatched_candidate_horizons_are_rejected() -> None:
    """Candidates for one episode must use a common action horizon."""
    episode = _episode()
    candidate = episode.candidates[1]
    action = replace(
        candidate.action,
        actions=np.zeros((2, 2), dtype=np.float32),
    )
    with pytest.raises(DataValidationError, match="mismatched action horizons"):
        validate_episode(_replace_candidate(episode, 1, action=action))


@pytest.mark.parametrize("progress", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_progress_is_rejected(progress: float) -> None:
    """Progress labels are finite values bounded to the unit interval."""
    episode = _episode()
    candidate = episode.candidates[0]
    outcome = replace(candidate.outcome, progress=progress)
    with pytest.raises(DataValidationError, match="progress"):
        validate_episode(_replace_candidate(episode, 0, outcome=outcome))


@pytest.mark.parametrize("probability", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_probabilities_are_rejected(probability: float) -> None:
    """Outcome probabilities reject out-of-range and non-finite values."""
    episode = _episode()
    candidate = episode.candidates[0]
    outcome = replace(candidate.outcome, success_probability=probability)
    with pytest.raises(DataValidationError, match="success_probability"):
        validate_episode(_replace_candidate(episode, 0, outcome=outcome))


def test_invalid_failure_description_is_rejected_before_serialization() -> None:
    """Failure descriptions must satisfy the same schema on save and load."""
    episode = _episode()
    candidate = episode.candidates[1]
    failure = replace(candidate.outcome.failure_events[0], description=123)
    outcome = replace(candidate.outcome, failure_events=(failure,))

    with pytest.raises(DataValidationError, match=r"FailureEvent.*description"):
        validate_episode(_replace_candidate(episode, 1, outcome=outcome))


def test_missing_provenance_is_rejected() -> None:
    """Every candidate must carry source provenance."""
    episode = _episode()
    invalid = _replace_candidate(episode, 0, provenance=None)
    with pytest.raises(
        DataValidationError, match="missing or invalid source provenance"
    ):
        validate_episode(invalid)


def test_missing_split_group_is_rejected() -> None:
    """Episode and sample split-group identifiers may not be empty."""
    with pytest.raises(DataValidationError, match="split_group_id"):
        validate_episode(replace(_episode(), split_group_id=""))


def test_duplicate_candidate_identifiers_are_rejected() -> None:
    """Candidate identifiers are unique within an episode."""
    episode = _episode()
    duplicate = replace(
        episode.candidates[1],
        candidate_id=episode.candidates[0].candidate_id,
    )
    candidates = (episode.candidates[0], duplicate, *episode.candidates[2:])
    with pytest.raises(DataValidationError, match="duplicate candidate identifier"):
        validate_episode(replace(episode, candidates=candidates))


def test_strong_heuristic_label_is_rejected() -> None:
    """Heuristically produced labels can only be weak."""
    episode = _episode()
    candidate = episode.candidates[1]
    outcome = replace(candidate.outcome, label_strength=LabelStrength.STRONG)
    with pytest.raises(DataValidationError, match="heuristic labels must be weak"):
        validate_episode(_replace_candidate(episode, 1, outcome=outcome))


def test_strong_unverified_simulator_label_is_rejected() -> None:
    """Strong simulator labels require explicit replay verification."""
    episode = _episode()
    candidate = episode.candidates[0]
    outcome = replace(
        candidate.outcome,
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=False,
    )
    with pytest.raises(DataValidationError, match="require verified simulator replay"):
        validate_episode(_replace_candidate(episode, 0, outcome=outcome))


def test_inconsistent_source_episode_identifier_is_rejected() -> None:
    """Candidate provenance must point to the containing source episode."""
    episode = _episode()
    candidate = episode.candidates[0]
    provenance = replace(candidate.provenance, source_episode_id="another-episode")
    with pytest.raises(DataValidationError, match="source_episode_id"):
        validate_episode(_replace_candidate(episode, 0, provenance=provenance))


def test_unsupported_episode_schema_version_is_rejected() -> None:
    """Top-level unsupported schema versions fail clearly."""
    with pytest.raises(DataValidationError, match="unsupported schema version"):
        validate_episode(replace(_episode(), schema_version="999.0"))


def test_unsupported_nested_schema_version_is_rejected() -> None:
    """Nested entity schema versions are validated independently."""
    episode = _episode()
    camera = episode.observations.frames[0].cameras[0]
    invalid_camera = replace(camera, schema_version="0.0")
    with pytest.raises(DataValidationError, match=r"CameraFrame.*schema_version"):
        validate_episode(_replace_camera(episode, invalid_camera))
