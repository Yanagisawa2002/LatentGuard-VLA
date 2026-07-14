"""Tests for deterministic, configurable synthetic episode generation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np
import pytest

from latentguard.models import Episode, LabelStrength
from latentguard.synthetic import generate_synthetic_episodes
from latentguard.validation import validate_episodes


def _update_fingerprint(digest: Any, value: object) -> None:
    if isinstance(value, np.ndarray):
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    elif isinstance(value, Enum):
        digest.update(value.value.encode())
    elif is_dataclass(value) and not isinstance(value, type):
        for model_field in fields(value):
            digest.update(model_field.name.encode())
            _update_fingerprint(digest, getattr(value, model_field.name))
    elif isinstance(value, Mapping):
        for key in sorted(value):
            _update_fingerprint(digest, key)
            _update_fingerprint(digest, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            _update_fingerprint(digest, item)
    else:
        digest.update(repr(value).encode())


def _fingerprint(episodes: tuple[Episode, ...]) -> str:
    digest = hashlib.sha256()
    _update_fingerprint(digest, episodes)
    return digest.hexdigest()


def _episodes(*, seed: int = 42, **overrides: Any) -> tuple[Episode, ...]:
    config: dict[str, Any] = {
        "seed": seed,
        "episode_count": 2,
        "episode_length": 4,
        "action_dim": 3,
        "robot_state_dim": 5,
        "camera_count": 1,
        "image_height": 3,
        "image_width": 4,
        "depth_enabled": False,
        "candidate_count": 4,
    }
    config.update(overrides)
    return generate_synthetic_episodes(**config)


def test_generation_is_deterministic_for_same_seed_and_config() -> None:
    """Identical seeds and settings produce byte-identical model content."""
    first = _episodes()
    second = _episodes()
    assert _fingerprint(first) == _fingerprint(second)


def test_different_seeds_change_generated_content() -> None:
    """Seed changes are detectable in identifiers, arrays, and fingerprints."""
    first = _episodes(seed=1)
    second = _episodes(seed=2)
    assert first[0].episode_id != second[0].episode_id
    assert _fingerprint(first) != _fingerprint(second)


def test_dimensions_and_lengths_are_configurable() -> None:
    """Episode, robot-state, action-horizon, and action dimensions are honored."""
    episodes = _episodes(
        episode_count=3,
        episode_length=6,
        action_dim=7,
        robot_state_dim=9,
        candidate_count=5,
    )
    assert len(episodes) == 3
    for episode in episodes:
        assert len(episode.observations.frames) == 6
        assert len(episode.candidates) == 5
        assert episode.observations.frames[0].robot_state.shape == (9,)
        assert all(
            candidate.action.actions.shape == (6, 7) for candidate in episode.candidates
        )


def test_zero_camera_operation() -> None:
    """Every observation may validly contain no camera streams."""
    episodes = _episodes(camera_count=0)
    assert all(
        not frame.cameras
        for episode in episodes
        for frame in episode.observations.frames
    )
    validate_episodes(episodes)


def test_multiple_cameras_with_optional_depth() -> None:
    """Multiple independent streams preserve RGB, calibration, and aligned depth."""
    episodes = _episodes(
        camera_count=3,
        image_height=4,
        image_width=5,
        depth_enabled=True,
    )
    for frame in episodes[0].observations.frames:
        assert len(frame.cameras) == 3
        assert {camera.camera_id for camera in frame.cameras} == {
            "camera-00",
            "camera-01",
            "camera-02",
        }
        for camera in frame.cameras:
            assert camera.rgb.shape == (4, 5, 3)
            assert camera.depth is not None and camera.depth.shape == (4, 5)
            assert camera.intrinsics is not None
            assert camera.extrinsics is not None


def test_depth_can_be_omitted() -> None:
    """Depth is absent when the feature is disabled."""
    episodes = _episodes(camera_count=2, depth_enabled=False)
    assert all(
        camera.depth is None
        for frame in episodes[0].observations.frames
        for camera in frame.cameras
    )


def test_fixture_covers_outcomes_safety_strengths_and_failure_types() -> None:
    """The default candidate set spans the required evaluation labels."""
    candidates = _episodes(episode_count=1)[0].candidates
    assert {candidate.outcome.success for candidate in candidates} == {False, True}
    assert {candidate.outcome.unsafe for candidate in candidates} == {False, True}
    assert {candidate.outcome.label_strength for candidate in candidates} == {
        LabelStrength.WEAK,
        LabelStrength.STRONG,
    }
    failure_types = {
        failure.failure_type
        for candidate in candidates
        for failure in candidate.outcome.failure_events
    }
    assert len(failure_types) >= 2
    assert all(
        candidate.provenance.source_episode_id
        == candidates[0].provenance.source_episode_id
        for candidate in candidates
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("seed", -1),
        ("episode_count", 0),
        ("episode_length", 0),
        ("action_dim", 0),
        ("robot_state_dim", 0),
        ("camera_count", -1),
        ("image_height", 0),
        ("image_width", 0),
        ("candidate_count", 0),
    ],
)
def test_invalid_generator_configuration_fails(field_name: str, value: int) -> None:
    """Malformed settings fail explicitly instead of being normalized."""
    with pytest.raises(ValueError, match=field_name):
        _episodes(**{field_name: value})


@pytest.mark.parametrize(
    ("episode_count", "candidate_count"),
    [(1, 1), (1, 2), (2, 1)],
)
def test_generator_rejects_configs_without_required_outcome_coverage(
    episode_count: int, candidate_count: int
) -> None:
    """Every accepted fixture configuration covers the required outcome set."""
    with pytest.raises(ValueError, match=r"episode_count \* candidate_count"):
        _episodes(episode_count=episode_count, candidate_count=candidate_count)
