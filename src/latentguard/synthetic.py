"""Deterministic, simulator-free synthetic episodes for tests and smoke runs."""

from __future__ import annotations

from typing import Final

import numpy as np

from latentguard.models import (
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
from latentguard.validation import validate_episodes

_CONTROL_PERIOD_S: Final = 0.1


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}, got {value}")


def _validate_configuration(
    *,
    seed: int,
    episode_count: int,
    episode_length: int,
    action_dim: int,
    robot_state_dim: int,
    camera_count: int,
    image_height: int,
    image_width: int,
    candidate_count: int,
) -> None:
    _positive_integer("seed", seed, allow_zero=True)
    _positive_integer("episode_count", episode_count)
    _positive_integer("episode_length", episode_length)
    _positive_integer("action_dim", action_dim)
    _positive_integer("robot_state_dim", robot_state_dim)
    _positive_integer("camera_count", camera_count, allow_zero=True)
    _positive_integer("image_height", image_height)
    _positive_integer("image_width", image_width)
    _positive_integer("candidate_count", candidate_count)
    if episode_count * candidate_count < 3:
        raise ValueError(
            "episode_count * candidate_count must be at least 3 to cover "
            "successful, unsuccessful, safe, unsafe, and multiple failure outcomes"
        )


def _camera_frame(
    rng: np.random.Generator,
    *,
    episode_index: int,
    frame_index: int,
    camera_index: int,
    image_height: int,
    image_width: int,
    depth_enabled: bool,
) -> CameraFrame:
    camera_id = f"camera-{camera_index:02d}"
    rgb = rng.integers(
        0,
        256,
        size=(image_height, image_width, 3),
        dtype=np.uint8,
    )
    depth = None
    if depth_enabled:
        depth = rng.uniform(
            0.2,
            2.0,
            size=(image_height, image_width),
        ).astype(np.float32)

    intrinsics = np.array(
        [
            [float(image_width), 0.0, (image_width - 1) / 2.0],
            [0.0, float(image_height), (image_height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[0, 3] = np.float32(camera_index * 0.05)
    extrinsics[1, 3] = np.float32(episode_index * 0.001)
    extrinsics[2, 3] = np.float32(frame_index * 0.0001)
    return CameraFrame(
        camera_id=camera_id,
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


def _outcome_for_pattern(pattern: int, episode_length: int) -> OutcomeLabel:
    failure_timestamp = _CONTROL_PERIOD_S * max(0, episode_length - 1) / 2.0
    if pattern == 0:
        return OutcomeLabel(
            success=True,
            progress=1.0,
            unsafe=False,
            label_source=LabelSource.DETERMINISTIC_EVALUATOR,
            label_strength=LabelStrength.STRONG,
            success_probability=0.95,
            unsafe_probability=0.02,
        )
    if pattern == 1:
        return OutcomeLabel(
            success=False,
            progress=0.65,
            unsafe=False,
            label_source=LabelSource.HEURISTIC,
            label_strength=LabelStrength.WEAK,
            success_probability=0.25,
            unsafe_probability=0.10,
            failure_events=(
                FailureEvent(
                    failure_type="grasp_failure",
                    timestamp_s=failure_timestamp,
                    probability=0.80,
                    description="Synthetic object grasp was not retained.",
                ),
            ),
        )
    if pattern == 2:
        return OutcomeLabel(
            success=False,
            progress=0.25,
            unsafe=True,
            label_source=LabelSource.DETERMINISTIC_EVALUATOR,
            label_strength=LabelStrength.STRONG,
            success_probability=0.05,
            unsafe_probability=0.90,
            failure_events=(
                FailureEvent(
                    failure_type="collision",
                    timestamp_s=failure_timestamp,
                    probability=0.90,
                    description="Synthetic collision guard was triggered.",
                ),
            ),
        )
    return OutcomeLabel(
        success=False,
        progress=0.45,
        unsafe=True,
        label_source=LabelSource.HEURISTIC,
        label_strength=LabelStrength.WEAK,
        success_probability=0.15,
        unsafe_probability=0.75,
        failure_events=(
            FailureEvent(
                failure_type="control_limit",
                timestamp_s=failure_timestamp,
                probability=0.75,
                description="Synthetic control limit was exceeded.",
            ),
        ),
    )


def generate_synthetic_episodes(
    *,
    seed: int,
    episode_count: int,
    episode_length: int,
    action_dim: int,
    robot_state_dim: int,
    camera_count: int,
    image_height: int = 8,
    image_width: int = 8,
    depth_enabled: bool = False,
    candidate_count: int = 4,
) -> tuple[Episode, ...]:
    """Generate deterministic, small, fully provenance-aware synthetic episodes.

    No simulator, network, GPU, downloaded model, or global random state is used.
    Configurations contain at least three total candidates so every generated
    bundle exercises success/failure, safe/unsafe, and multiple failure types.
    """
    _validate_configuration(
        seed=seed,
        episode_count=episode_count,
        episode_length=episode_length,
        action_dim=action_dim,
        robot_state_dim=robot_state_dim,
        camera_count=camera_count,
        image_height=image_height,
        image_width=image_width,
        candidate_count=candidate_count,
    )
    if type(depth_enabled) is not bool:
        raise ValueError(f"depth_enabled must be a boolean, got {depth_enabled!r}")

    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    task_id = "synthetic-manipulation"
    source_policy_id = "synthetic-policy-v1"

    for episode_index in range(episode_count):
        episode_id = f"synthetic-s{seed:08d}-e{episode_index:04d}"
        split_group_id = f"split-{episode_id}"
        frames: list[ObservationFrame] = []
        for frame_index in range(episode_length):
            cameras = tuple(
                _camera_frame(
                    rng,
                    episode_index=episode_index,
                    frame_index=frame_index,
                    camera_index=camera_index,
                    image_height=image_height,
                    image_width=image_width,
                    depth_enabled=depth_enabled,
                )
                for camera_index in range(camera_count)
            )
            frames.append(
                ObservationFrame(
                    observation_id=f"{episode_id}-obs-{frame_index:04d}",
                    timestamp_s=frame_index * _CONTROL_PERIOD_S,
                    robot_state=rng.normal(
                        loc=0.0,
                        scale=0.25,
                        size=robot_state_dim,
                    ).astype(np.float32),
                    cameras=cameras,
                )
            )

        candidates: list[CandidateAction] = []
        for candidate_index in range(candidate_count):
            candidate_id = f"{episode_id}-candidate-{candidate_index:03d}"
            pattern = (episode_index * candidate_count + candidate_index) % 4
            outcome = _outcome_for_pattern(pattern, episode_length)
            action = ActionChunk(
                actions=rng.normal(
                    loc=0.0,
                    scale=0.2 + pattern * 0.05,
                    size=(episode_length, action_dim),
                ).astype(np.float32),
                coordinate_frame="robot_base",
                control_period_s=_CONTROL_PERIOD_S,
            )
            failure_type = (
                outcome.failure_events[0].failure_type
                if outcome.failure_events
                else "none"
            )
            provenance = SampleProvenance(
                source_episode_id=episode_id,
                source_policy_id=source_policy_id,
                source_task_id=task_id,
                transformation_type="synthetic_generation",
                transformation_parameters={
                    "candidate_pattern": pattern,
                    "fixture_failure_type": failure_type,
                },
                seed=seed,
                label_source=outcome.label_source,
                label_strength=outcome.label_strength,
                simulator_replay_verified=outcome.simulator_replay_verified,
                split_group_id=split_group_id,
            )
            candidates.append(
                CandidateAction(
                    candidate_id=candidate_id,
                    action=action,
                    outcome=outcome,
                    provenance=provenance,
                )
            )

        episodes.append(
            Episode(
                episode_id=episode_id,
                task_id=task_id,
                source_policy_id=source_policy_id,
                instruction="Move the synthetic object to the target region.",
                observations=ObservationHistory(
                    history_id=f"{episode_id}-history",
                    frames=tuple(frames),
                ),
                candidates=tuple(candidates),
                split_group_id=split_group_id,
            )
        )

    result = tuple(episodes)
    validate_episodes(result)
    return result
