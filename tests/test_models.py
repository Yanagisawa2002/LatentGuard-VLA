"""Tests for mutation-safe typed data models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from latentguard.models import (
    CURRENT_SCHEMA_VERSION,
    ActionChunk,
    CameraFrame,
    CandidateAction,
    Episode,
    FailureEvent,
    LabelSource,
    LabelStrength,
    ObservationFrame,
    ObservationHistory,
    OutcomeLabel,
    SampleProvenance,
)


def test_arrays_are_detached_and_read_only() -> None:
    """Model arrays cannot be changed through their source or model reference."""
    source_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    source_depth = np.ones((2, 3), dtype=np.float32)
    camera = CameraFrame(camera_id="front", rgb=source_rgb, depth=source_depth)

    source_rgb.fill(255)
    source_depth.fill(9.0)
    assert np.count_nonzero(camera.rgb) == 0
    assert np.all(camera.depth == 1.0)
    assert camera.rgb.flags.writeable is False
    assert camera.depth is not None and camera.depth.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        camera.rgb[0, 0, 0] = 1
    with pytest.raises(ValueError, match="WRITEABLE"):
        camera.rgb.setflags(write=True)


def test_nested_collections_and_provenance_are_immutable() -> None:
    """Sequences become tuples and provenance metadata becomes read-only."""
    camera = CameraFrame(camera_id="front", rgb=np.zeros((1, 1, 3), dtype=np.uint8))
    observation = ObservationFrame(
        observation_id="obs-0",
        timestamp_s=0.0,
        robot_state=np.zeros(2, dtype=np.float32),
        cameras=(camera,),
    )
    provenance = SampleProvenance(
        source_episode_id="episode-0",
        source_policy_id="policy-0",
        source_task_id="task-0",
        transformation_type="synthetic_generation",
        transformation_parameters={"variant": 1},
        seed=7,
        label_source=LabelSource.HEURISTIC,
        label_strength=LabelStrength.WEAK,
        simulator_replay_verified=False,
        split_group_id="split-0",
    )

    assert isinstance(observation.cameras, tuple)
    with pytest.raises(TypeError):
        provenance.transformation_parameters["variant"] = 2  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        observation.timestamp_s = 1.0  # type: ignore[misc]


def test_all_contract_entities_carry_explicit_schema_version() -> None:
    """Every named M0 entity exposes the supported schema version."""
    camera = CameraFrame(camera_id="front", rgb=np.zeros((1, 1, 3), dtype=np.uint8))
    observation = ObservationFrame(
        observation_id="obs-0",
        timestamp_s=0.0,
        robot_state=np.zeros(2, dtype=np.float32),
        cameras=(camera,),
    )
    history = ObservationHistory(history_id="history-0", frames=(observation,))
    action = ActionChunk(
        actions=np.zeros((1, 2), dtype=np.float32),
        coordinate_frame="robot_base",
        control_period_s=0.1,
    )
    failure = FailureEvent(failure_type="collision")
    outcome = OutcomeLabel(
        success=False,
        progress=0.5,
        unsafe=True,
        label_source=LabelSource.HEURISTIC,
        label_strength=LabelStrength.WEAK,
        failure_events=(failure,),
    )
    provenance = SampleProvenance(
        source_episode_id="episode-0",
        source_policy_id="policy-0",
        source_task_id="task-0",
        transformation_type="synthetic_generation",
        transformation_parameters={},
        seed=1,
        label_source=outcome.label_source,
        label_strength=outcome.label_strength,
        simulator_replay_verified=False,
        split_group_id="split-0",
    )
    candidate = CandidateAction(
        candidate_id="candidate-0",
        action=action,
        outcome=outcome,
        provenance=provenance,
    )
    episode = Episode(
        episode_id="episode-0",
        task_id="task-0",
        source_policy_id="policy-0",
        instruction="Do the task.",
        observations=history,
        candidates=(candidate,),
        split_group_id="split-0",
    )

    entities = (
        camera,
        observation,
        history,
        action,
        failure,
        outcome,
        provenance,
        candidate,
        episode,
    )
    assert all(entity.schema_version == CURRENT_SCHEMA_VERSION for entity in entities)


def test_label_enums_have_stable_serializable_values() -> None:
    """Label enum values are explicit strings suitable for manifests."""
    assert LabelSource.HEURISTIC.value == "heuristic"
    assert LabelSource.SIMULATOR.value == "simulator"
    assert LabelStrength.WEAK.value == "weak"
    assert LabelStrength.STRONG.value == "strong"
