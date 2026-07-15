"""Deterministic offline iteration over validated Episode data."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.models import CandidateAction, Episode, ObservationFrame
from latentguard.validation import validate_episode


class ReplayAlignmentError(ValueError):
    """Offline replay cannot align a requested Episode and candidate."""


@dataclass(frozen=True, slots=True, eq=False)
class ReplayStep:
    """One exact-index offline observation/action pair."""

    step_index: int
    observation: ObservationFrame
    action: NDArray[Any]
    observation_timestamp_s: float
    nominal_action_timestamp_s: float

    def __post_init__(self) -> None:
        """Detach the action row behind an immutable buffer."""
        detached = np.array(self.action, copy=True, order="C", subok=False)
        frozen = np.frombuffer(detached.tobytes(), dtype=detached.dtype).reshape(
            detached.shape
        )
        object.__setattr__(self, "action", frozen)


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Metadata for an exact-index offline Episode traversal."""

    episode_id: str
    candidate_id: str
    step_count: int
    coordinate_frame: str
    control_period_s: float


def _candidate(episode: Episode, candidate_id: str) -> CandidateAction:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ReplayAlignmentError("candidate_id must be a non-empty string")
    for candidate in episode.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ReplayAlignmentError(
        f"episode {episode.episode_id!r} has no candidate {candidate_id!r}"
    )


def build_replay_plan(episode: Episode, candidate_id: str) -> ReplayPlan:
    """Validate and build a plan for offline data traversal only."""
    validate_episode(episode)
    candidate = _candidate(episode, candidate_id)
    horizon = int(candidate.action.actions.shape[0])
    observation_count = len(episode.observations.frames)
    if horizon != observation_count:
        raise ReplayAlignmentError(
            "offline replay requires action horizon to equal observation count: "
            f"candidate {candidate_id!r} has {horizon}, observations have "
            f"{observation_count}"
        )
    return ReplayPlan(
        episode_id=episode.episode_id,
        candidate_id=candidate.candidate_id,
        step_count=horizon,
        coordinate_frame=candidate.action.coordinate_frame,
        control_period_s=float(candidate.action.control_period_s),
    )


def iter_replay_steps(episode: Episode, candidate_id: str) -> Iterator[ReplayStep]:
    """Yield repeatable exact-index pairs without interpolation or repair."""
    plan = build_replay_plan(episode, candidate_id)
    candidate = _candidate(episode, candidate_id)
    for index in range(plan.step_count):
        observation = episode.observations.frames[index]
        yield ReplayStep(
            step_index=index,
            observation=observation,
            action=candidate.action.actions[index],
            observation_timestamp_s=float(observation.timestamp_s),
            nominal_action_timestamp_s=index * plan.control_period_s,
        )
