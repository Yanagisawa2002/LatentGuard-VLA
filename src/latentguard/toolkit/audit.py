"""Strict, read-only quality auditing for validated Episodes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from latentguard.models import Episode
from latentguard.toolkit.models import (
    AuditConfig,
    AuditReport,
    AuditSeverity,
    DataIssue,
)
from latentguard.validation import validate_episodes


def _runs(values: Sequence[object], minimum: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index < len(values) and values[index] == values[index - 1]:
            continue
        if index - start >= minimum:
            runs.append((start, index - 1))
        start = index
    return runs


def _array_key(value: np.ndarray[Any, Any]) -> tuple[str, tuple[int, ...], bytes]:
    return value.dtype.str, value.shape, value.tobytes(order="C")


def audit_episodes(
    episodes: Sequence[Episode], config: AuditConfig | None = None
) -> AuditReport:
    """Validate and deterministically audit Episodes without repairing them.

    Frozen-camera detection compares RGB only; depth is intentionally excluded.
    """
    validate_episodes(episodes)
    if config is None:
        config = AuditConfig()
    issues: list[DataIssue] = []
    for episode in episodes:
        frames = episode.observations.frames
        expected_camera_ids = tuple(sorted(c.camera_id for c in frames[0].cameras))
        for frame in frames[1:]:
            camera_ids = tuple(sorted(c.camera_id for c in frame.cameras))
            if camera_ids != expected_camera_ids:
                issues.append(
                    DataIssue(
                        "CAMERA_SET_CHANGED",
                        AuditSeverity.WARNING,
                        "camera ID set differs from the first observation",
                        episode_id=episode.episode_id,
                        observation_id=frame.observation_id,
                        details={
                            "expected_camera_ids": ",".join(expected_camera_ids),
                            "actual_camera_ids": ",".join(camera_ids),
                        },
                    )
                )

        all_camera_ids = sorted(
            {camera.camera_id for frame in frames for camera in frame.cameras}
        )
        for camera_id in all_camera_ids:
            contiguous: list[tuple[int, tuple[str, tuple[int, ...], bytes]]] = []
            for index, frame in enumerate(frames):
                camera = next(
                    (c for c in frame.cameras if c.camera_id == camera_id), None
                )
                if camera is None:
                    contiguous = []
                    continue
                key = _array_key(camera.rgb)
                if contiguous and index != contiguous[-1][0] + 1:
                    contiguous = []
                contiguous.append((index, key))
                if len(contiguous) >= config.frozen_frame_run_length and all(
                    item[1] == contiguous[-1][1]
                    for item in contiguous[-config.frozen_frame_run_length :]
                ):
                    start = index - config.frozen_frame_run_length + 1
                    if len(contiguous) == config.frozen_frame_run_length or (
                        contiguous[-config.frozen_frame_run_length - 1][1] != key
                    ):
                        issues.append(
                            DataIssue(
                                "CAMERA_FROZEN",
                                AuditSeverity.WARNING,
                                "identical RGB frames meet the configured run length",
                                episode_id=episode.episode_id,
                                observation_id=frames[start].observation_id,
                                camera_id=camera_id,
                                details={
                                    "first_step": start,
                                    "run_length_threshold": (
                                        config.frozen_frame_run_length
                                    ),
                                },
                            )
                        )

        state_keys = [_array_key(frame.robot_state) for frame in frames]
        for candidate in episode.candidates:
            actions = candidate.action.actions
            horizon = int(actions.shape[0])
            if horizon != len(frames):
                issues.append(
                    DataIssue(
                        "REPLAY_HORIZON_MISMATCH",
                        AuditSeverity.ERROR,
                        "candidate action horizon does not equal observation count",
                        episode_id=episode.episode_id,
                        candidate_id=candidate.candidate_id,
                        details={
                            "action_horizon": horizon,
                            "observation_count": len(frames),
                        },
                    )
                )

            first_timestamp = float(frames[0].timestamp_s)
            drifts = [
                abs(
                    (float(frame.timestamp_s) - first_timestamp)
                    - index * float(candidate.action.control_period_s)
                )
                for index, frame in enumerate(frames)
            ]
            exceeded = [
                index
                for index, drift in enumerate(drifts)
                if drift > config.timestamp_tolerance_s
            ]
            if exceeded:
                issues.append(
                    DataIssue(
                        "TIMESTAMP_PERIOD_DRIFT",
                        AuditSeverity.WARNING,
                        "observation cadence differs from the candidate control period",
                        episode_id=episode.episode_id,
                        candidate_id=candidate.candidate_id,
                        observation_id=frames[exceeded[0]].observation_id,
                        details={
                            "first_exceeded_step": exceeded[0],
                            "max_drift_s": max(drifts),
                            "tolerance_s": config.timestamp_tolerance_s,
                        },
                    )
                )

            near_zero = np.all(
                np.abs(actions) <= config.near_zero_action_threshold, axis=1
            )
            near_zero_count = int(np.count_nonzero(near_zero))
            fraction = near_zero_count / horizon
            if fraction > config.near_zero_action_fraction:
                issues.append(
                    DataIssue(
                        "ACTION_NEAR_ZERO_DOMINANT",
                        AuditSeverity.INFO,
                        "near-zero action rows exceed the configured fraction",
                        episode_id=episode.episode_id,
                        candidate_id=candidate.candidate_id,
                        details={
                            "near_zero_row_count": near_zero_count,
                            "total_row_count": horizon,
                            "fraction": fraction,
                            "row_threshold": config.near_zero_action_threshold,
                            "fraction_threshold": config.near_zero_action_fraction,
                        },
                    )
                )

            usable = min(horizon, len(frames))
            for start, end in _runs(
                state_keys[:usable], config.frozen_frame_run_length
            ):
                if not bool(np.all(near_zero[start : end + 1])):
                    issues.append(
                        DataIssue(
                            "ROBOT_STATE_FROZEN",
                            AuditSeverity.WARNING,
                            "robot state is frozen while corresponding actions "
                            "are not all near zero",
                            episode_id=episode.episode_id,
                            candidate_id=candidate.candidate_id,
                            observation_id=frames[start].observation_id,
                            details={"first_step": start, "last_step": end},
                        )
                    )

            if config.action_jump_threshold is not None and horizon > 1:
                jumps = np.linalg.norm(np.diff(actions, axis=0), axis=1)
                exceeded_jumps = np.flatnonzero(jumps > config.action_jump_threshold)
                if exceeded_jumps.size:
                    first = int(exceeded_jumps[0]) + 1
                    issues.append(
                        DataIssue(
                            "ACTION_DISCONTINUITY",
                            AuditSeverity.WARNING,
                            "consecutive action rows exceed the configured L2 jump",
                            episode_id=episode.episode_id,
                            candidate_id=candidate.candidate_id,
                            details={
                                "first_exceeded_step": first,
                                "max_l2_jump": float(np.max(jumps)),
                                "threshold": config.action_jump_threshold,
                            },
                        )
                    )

        for left_index, left in enumerate(episode.candidates):
            for right in episode.candidates[left_index + 1 :]:
                if (
                    left.action.actions.dtype == right.action.actions.dtype
                    and left.action.actions.shape == right.action.actions.shape
                    and np.array_equal(left.action.actions, right.action.actions)
                ):
                    issues.append(
                        DataIssue(
                            "DUPLICATE_CANDIDATE_ACTION",
                            AuditSeverity.INFO,
                            "distinct candidate IDs contain identical action arrays",
                            episode_id=episode.episode_id,
                            candidate_id=right.candidate_id,
                            details={"duplicate_of_candidate_id": left.candidate_id},
                        )
                    )

    severity_counts = Counter(issue.severity.value for issue in issues)
    code_counts = Counter(issue.issue_code for issue in issues)
    return AuditReport(
        episode_count=len(episodes),
        candidate_count=sum(len(episode.candidates) for episode in episodes),
        issue_count=len(issues),
        issues_by_severity={
            key: severity_counts[key] for key in sorted(severity_counts)
        },
        issues_by_code={key: code_counts[key] for key in sorted(code_counts)},
        issues=tuple(issues),
    )
