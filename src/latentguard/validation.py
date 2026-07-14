"""Descriptive validation for the LatentGuard-VLA M0 data contract."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from latentguard.models import (
    SUPPORTED_SCHEMA_VERSIONS,
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


class DataValidationError(ValueError):
    """Validation error carrying entity, field, identifier, and reason."""

    def __init__(
        self,
        entity_type: str,
        field_name: str,
        identifier: str,
        reason: str,
    ) -> None:
        """Build a descriptive validation error."""
        self.entity_type = entity_type
        self.field_name = field_name
        self.identifier = identifier
        self.reason = reason
        super().__init__(f"{entity_type}[id={identifier}].{field_name}: {reason}")


def _fail(entity: object | str, field: str, identifier: str, reason: str) -> NoReturn:
    entity_name = entity if isinstance(entity, str) else type(entity).__name__
    raise DataValidationError(entity_name, field, identifier or "<missing>", reason)


def _require_nonempty_string(
    value: object, entity: object, field: str, identifier: str
) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(entity, field, identifier, "must be a non-empty string")


def _require_bool(value: object, entity: object, field: str, identifier: str) -> None:
    if type(value) is not bool:
        _fail(entity, field, identifier, "must be a boolean")


def _finite_number(value: object, entity: object, field: str, identifier: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        _fail(entity, field, identifier, "must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        _fail(entity, field, identifier, f"must be finite, got {number!r}")
    return number


def _probability(
    value: float | None, entity: object, field: str, identifier: str
) -> None:
    if value is None:
        return
    number = _finite_number(value, entity, field, identifier)
    if not 0.0 <= number <= 1.0:
        _fail(entity, field, identifier, f"must be within [0, 1], got {number}")


def _require_array(
    value: object, entity: object, field: str, identifier: str
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        _fail(entity, field, identifier, "must be a numpy.ndarray")
    if not np.issubdtype(value.dtype, np.number):
        _fail(entity, field, identifier, f"must have numeric dtype, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        _fail(entity, field, identifier, "contains NaN or infinity")
    return value


def _validate_schema(entity: object, schema_version: object, identifier: str) -> None:
    if (
        not isinstance(schema_version, str)
        or schema_version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        _fail(
            entity,
            "schema_version",
            identifier,
            f"unsupported schema version {schema_version!r}; supported: {supported}",
        )


def validate_camera_frame(camera: CameraFrame) -> None:
    """Validate one camera frame and all optional calibration arrays."""
    identifier = camera.camera_id or "<missing>"
    _validate_schema(camera, camera.schema_version, identifier)
    _require_nonempty_string(camera.camera_id, camera, "camera_id", identifier)

    rgb = _require_array(camera.rgb, camera, "rgb", identifier)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        _fail(
            camera,
            "rgb",
            identifier,
            f"expected positive [height, width, 3] shape, got {rgb.shape}",
        )
    if rgb.dtype != np.uint8:
        _fail(camera, "rgb", identifier, f"expected uint8 dtype, got {rgb.dtype}")

    if camera.depth is not None:
        depth = _require_array(camera.depth, camera, "depth", identifier)
        if depth.ndim != 2 or depth.shape[0] <= 0 or depth.shape[1] <= 0:
            _fail(
                camera,
                "depth",
                identifier,
                f"expected positive [height, width] shape, got {depth.shape}",
            )
        if depth.shape != rgb.shape[:2]:
            _fail(
                camera,
                "depth",
                identifier,
                f"resolution {depth.shape} does not match RGB {rgb.shape[:2]}",
            )

    if camera.intrinsics is not None:
        intrinsics = _require_array(camera.intrinsics, camera, "intrinsics", identifier)
        if intrinsics.shape != (3, 3):
            _fail(
                camera,
                "intrinsics",
                identifier,
                f"expected shape (3, 3), got {intrinsics.shape}",
            )

    if camera.extrinsics is not None:
        extrinsics = _require_array(camera.extrinsics, camera, "extrinsics", identifier)
        if extrinsics.shape != (4, 4):
            _fail(
                camera,
                "extrinsics",
                identifier,
                f"expected shape (4, 4), got {extrinsics.shape}",
            )


def validate_observation_frame(observation: ObservationFrame) -> None:
    """Validate a timestamped observation, robot state, and camera set."""
    identifier = observation.observation_id or "<missing>"
    _validate_schema(observation, observation.schema_version, identifier)
    _require_nonempty_string(
        observation.observation_id, observation, "observation_id", identifier
    )
    timestamp = _finite_number(
        observation.timestamp_s, observation, "timestamp_s", identifier
    )
    if timestamp < 0.0:
        _fail(observation, "timestamp_s", identifier, "must be non-negative")

    robot_state = _require_array(
        observation.robot_state, observation, "robot_state", identifier
    )
    if robot_state.ndim != 1 or robot_state.size == 0:
        _fail(
            observation,
            "robot_state",
            identifier,
            f"expected non-empty rank-1 array, got shape {robot_state.shape}",
        )

    camera_ids: set[str] = set()
    for camera in observation.cameras:
        if not isinstance(camera, CameraFrame):
            _fail(
                observation,
                "cameras",
                identifier,
                f"contains non-CameraFrame value {type(camera).__name__}",
            )
        validate_camera_frame(camera)
        if camera.camera_id in camera_ids:
            _fail(
                observation,
                "cameras",
                identifier,
                f"duplicate camera identifier {camera.camera_id!r}",
            )
        camera_ids.add(camera.camera_id)


def validate_observation_history(history: ObservationHistory) -> None:
    """Validate non-empty observations with strictly increasing timestamps."""
    identifier = history.history_id or "<missing>"
    _validate_schema(history, history.schema_version, identifier)
    _require_nonempty_string(history.history_id, history, "history_id", identifier)
    if not history.frames:
        _fail(history, "frames", identifier, "must contain at least one observation")

    previous_timestamp: float | None = None
    observation_ids: set[str] = set()
    for frame in history.frames:
        if not isinstance(frame, ObservationFrame):
            _fail(
                history,
                "frames",
                identifier,
                f"contains non-ObservationFrame value {type(frame).__name__}",
            )
        validate_observation_frame(frame)
        if frame.observation_id in observation_ids:
            _fail(
                history,
                "frames",
                identifier,
                f"duplicate observation identifier {frame.observation_id!r}",
            )
        observation_ids.add(frame.observation_id)
        if previous_timestamp is not None and frame.timestamp_s <= previous_timestamp:
            _fail(
                history,
                "frames.timestamp_s",
                identifier,
                "timestamps must be strictly increasing; "
                f"got {frame.timestamp_s} after {previous_timestamp}",
            )
        previous_timestamp = float(frame.timestamp_s)


def validate_action_chunk(action: ActionChunk, identifier: str = "<action>") -> None:
    """Validate action rank, horizon, dimension, coordinate frame, and period."""
    _validate_schema(action, action.schema_version, identifier)
    actions = _require_array(action.actions, action, "actions", identifier)
    if actions.ndim != 2:
        _fail(
            action,
            "actions",
            identifier,
            f"expected rank 2 [horizon, action_dim], got shape {actions.shape}",
        )
    if actions.shape[0] == 0:
        _fail(action, "actions", identifier, "action horizon must be positive")
    if actions.shape[1] == 0:
        _fail(action, "actions", identifier, "action dimension must be positive")
    _require_nonempty_string(
        action.coordinate_frame, action, "coordinate_frame", identifier
    )
    period = _finite_number(
        action.control_period_s, action, "control_period_s", identifier
    )
    if period <= 0.0:
        _fail(action, "control_period_s", identifier, "must be greater than zero")


def validate_failure_event(
    failure: FailureEvent, identifier: str = "<failure>"
) -> None:
    """Validate one failure type, optional timestamp, and probability."""
    _validate_schema(failure, failure.schema_version, identifier)
    _require_nonempty_string(failure.failure_type, failure, "failure_type", identifier)
    if failure.timestamp_s is not None:
        timestamp = _finite_number(
            failure.timestamp_s, failure, "timestamp_s", identifier
        )
        if timestamp < 0.0:
            _fail(failure, "timestamp_s", identifier, "must be non-negative")
    _probability(failure.probability, failure, "probability", identifier)
    if failure.description is not None and not isinstance(failure.description, str):
        _fail(
            failure,
            "description",
            identifier,
            "must be a string or null",
        )


def _validate_label_metadata(
    entity: object,
    source: object,
    strength: object,
    verified: object,
    identifier: str,
) -> None:
    if not isinstance(source, LabelSource):
        _fail(
            entity,
            "label_source",
            identifier,
            f"unsupported label source {source!r}",
        )
    if not isinstance(strength, LabelStrength):
        _fail(
            entity,
            "label_strength",
            identifier,
            f"unsupported label strength {strength!r}",
        )
    _require_bool(verified, entity, "simulator_replay_verified", identifier)
    if source is LabelSource.HEURISTIC and strength is LabelStrength.STRONG:
        _fail(
            entity,
            "label_strength",
            identifier,
            "heuristic labels must be weak",
        )
    if (
        source is LabelSource.SIMULATOR
        and strength is LabelStrength.STRONG
        and verified is not True
    ):
        _fail(
            entity,
            "simulator_replay_verified",
            identifier,
            "strong simulator labels require verified simulator replay",
        )


def validate_outcome_label(
    outcome: OutcomeLabel, identifier: str = "<outcome>"
) -> None:
    """Validate progress, probabilities, safety flags, and label provenance."""
    _validate_schema(outcome, outcome.schema_version, identifier)
    _require_bool(outcome.success, outcome, "success", identifier)
    _require_bool(outcome.unsafe, outcome, "unsafe", identifier)
    progress = _finite_number(outcome.progress, outcome, "progress", identifier)
    if not 0.0 <= progress <= 1.0:
        _fail(
            outcome,
            "progress",
            identifier,
            f"must be within [0, 1], got {progress}",
        )
    _probability(
        outcome.success_probability, outcome, "success_probability", identifier
    )
    _probability(outcome.unsafe_probability, outcome, "unsafe_probability", identifier)
    _validate_label_metadata(
        outcome,
        outcome.label_source,
        outcome.label_strength,
        outcome.simulator_replay_verified,
        identifier,
    )
    for index, failure in enumerate(outcome.failure_events):
        if not isinstance(failure, FailureEvent):
            _fail(
                outcome,
                "failure_events",
                identifier,
                f"item {index} is not a FailureEvent",
            )
        validate_failure_event(failure, f"{identifier}:failure:{index}")


def validate_sample_provenance(
    provenance: SampleProvenance, identifier: str = "<provenance>"
) -> None:
    """Validate source, transformation, split, and label provenance."""
    _validate_schema(provenance, provenance.schema_version, identifier)
    for field_name in (
        "source_episode_id",
        "source_policy_id",
        "source_task_id",
        "transformation_type",
        "split_group_id",
    ):
        _require_nonempty_string(
            getattr(provenance, field_name), provenance, field_name, identifier
        )
    if isinstance(provenance.seed, (bool, np.bool_)) or not isinstance(
        provenance.seed, (int, np.integer)
    ):
        _fail(provenance, "seed", identifier, "must be an integer")
    if not isinstance(provenance.transformation_parameters, Mapping):
        _fail(
            provenance,
            "transformation_parameters",
            identifier,
            "must be a mapping",
        )
    for key, value in provenance.transformation_parameters.items():
        if not isinstance(key, str) or not key:
            _fail(
                provenance,
                "transformation_parameters",
                identifier,
                "keys must be non-empty strings",
            )
        if value is not None and not isinstance(value, (str, int, float, bool)):
            _fail(
                provenance,
                "transformation_parameters",
                identifier,
                f"parameter {key!r} has unsupported scalar type {type(value).__name__}",
            )
        if isinstance(value, float) and not math.isfinite(value):
            _fail(
                provenance,
                "transformation_parameters",
                identifier,
                f"parameter {key!r} contains NaN or infinity",
            )
    _validate_label_metadata(
        provenance,
        provenance.label_source,
        provenance.label_strength,
        provenance.simulator_replay_verified,
        identifier,
    )


def validate_candidate_action(candidate: CandidateAction) -> None:
    """Validate a candidate action, outcome, and mutually consistent provenance."""
    identifier = candidate.candidate_id or "<missing>"
    _validate_schema(candidate, candidate.schema_version, identifier)
    _require_nonempty_string(
        candidate.candidate_id, candidate, "candidate_id", identifier
    )
    if not isinstance(candidate.action, ActionChunk):
        _fail(candidate, "action", identifier, "missing or invalid ActionChunk")
    if not isinstance(candidate.outcome, OutcomeLabel):
        _fail(candidate, "outcome", identifier, "missing or invalid OutcomeLabel")
    if not isinstance(candidate.provenance, SampleProvenance):
        _fail(
            candidate,
            "provenance",
            identifier,
            "missing or invalid source provenance",
        )
    validate_action_chunk(candidate.action, identifier)
    validate_outcome_label(candidate.outcome, identifier)
    validate_sample_provenance(candidate.provenance, identifier)
    if candidate.outcome.label_source is not candidate.provenance.label_source:
        _fail(
            candidate,
            "provenance.label_source",
            identifier,
            "does not match outcome label source",
        )
    if candidate.outcome.label_strength is not candidate.provenance.label_strength:
        _fail(
            candidate,
            "provenance.label_strength",
            identifier,
            "does not match outcome label strength",
        )
    if (
        candidate.outcome.simulator_replay_verified
        != candidate.provenance.simulator_replay_verified
    ):
        _fail(
            candidate,
            "provenance.simulator_replay_verified",
            identifier,
            "does not match outcome verification state",
        )


def validate_episode(episode: Episode) -> None:
    """Validate a complete episode and all cross-entity consistency rules."""
    identifier = episode.episode_id or "<missing>"
    _validate_schema(episode, episode.schema_version, identifier)
    for field_name in (
        "episode_id",
        "task_id",
        "source_policy_id",
        "instruction",
        "split_group_id",
    ):
        _require_nonempty_string(
            getattr(episode, field_name), episode, field_name, identifier
        )
    if not isinstance(episode.observations, ObservationHistory):
        _fail(
            episode,
            "observations",
            identifier,
            "missing or invalid ObservationHistory",
        )
    validate_observation_history(episode.observations)
    if not episode.candidates:
        _fail(episode, "candidates", identifier, "must contain at least one candidate")

    candidate_ids: set[str] = set()
    expected_horizon: int | None = None
    expected_action_dim: int | None = None
    for candidate in episode.candidates:
        if not isinstance(candidate, CandidateAction):
            _fail(
                episode,
                "candidates",
                identifier,
                f"contains non-CandidateAction value {type(candidate).__name__}",
            )
        validate_candidate_action(candidate)
        if candidate.candidate_id in candidate_ids:
            _fail(
                episode,
                "candidates.candidate_id",
                identifier,
                f"duplicate candidate identifier {candidate.candidate_id!r}",
            )
        candidate_ids.add(candidate.candidate_id)

        provenance = candidate.provenance
        if provenance.source_episode_id != episode.episode_id:
            _fail(
                candidate,
                "provenance.source_episode_id",
                candidate.candidate_id,
                "expected "
                f"{episode.episode_id!r}, got {provenance.source_episode_id!r}",
            )
        if provenance.source_policy_id != episode.source_policy_id:
            _fail(
                candidate,
                "provenance.source_policy_id",
                candidate.candidate_id,
                "expected "
                f"{episode.source_policy_id!r}, got {provenance.source_policy_id!r}",
            )
        if provenance.source_task_id != episode.task_id:
            _fail(
                candidate,
                "provenance.source_task_id",
                candidate.candidate_id,
                f"expected {episode.task_id!r}, got {provenance.source_task_id!r}",
            )
        if provenance.split_group_id != episode.split_group_id:
            _fail(
                candidate,
                "provenance.split_group_id",
                candidate.candidate_id,
                "expected "
                f"{episode.split_group_id!r}, got {provenance.split_group_id!r}",
            )

        horizon, action_dim = candidate.action.actions.shape
        if expected_horizon is None:
            expected_horizon = horizon
            expected_action_dim = action_dim
        elif horizon != expected_horizon:
            _fail(
                episode,
                "candidates.action.actions",
                identifier,
                "mismatched action horizons: "
                f"expected {expected_horizon}, got {horizon}",
            )
        elif action_dim != expected_action_dim:
            _fail(
                episode,
                "candidates.action.actions",
                identifier,
                "mismatched action dimensions: "
                f"expected {expected_action_dim}, got {action_dim}",
            )


def validate_episodes(episodes: Iterable[Episode]) -> None:
    """Validate a collection of episodes and reject duplicate episode IDs."""
    episode_ids: set[str] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Episode):
            _fail(
                "EpisodeCollection",
                "episodes",
                str(index),
                f"expected Episode, got {type(episode).__name__}",
            )
        validate_episode(episode)
        if episode.episode_id in episode_ids:
            _fail(
                "EpisodeCollection",
                "episode_id",
                episode.episode_id,
                "duplicate episode identifier",
            )
        episode_ids.add(episode.episode_id)
