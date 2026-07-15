"""Candidate-denominator dataset metrics for validated Episodes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from types import MappingProxyType

from latentguard.models import CandidateAction, Episode
from latentguard.validation import validate_episodes


@dataclass(frozen=True, slots=True)
class GroupMetrics:
    """Candidate-level metrics within an Episode grouping."""

    episode_count: int
    candidate_count: int
    candidate_success_rate: float
    candidate_unsafe_rate: float
    mean_progress: float


@dataclass(frozen=True, slots=True)
class DatasetMetrics:
    """Deterministic dataset inventory and candidate-level outcome metrics."""

    episode_count: int
    observation_count: int
    task_count: int
    source_policy_count: int
    split_group_count: int
    episodes_by_task: Mapping[str, int]
    episodes_by_source_policy: Mapping[str, int]
    observation_count_min: int
    observation_count_max: int
    observation_count_mean: float
    candidate_count: int
    successful_candidate_count: int
    candidate_success_rate: float
    unsafe_candidate_count: int
    candidate_unsafe_rate: float
    mean_progress: float
    action_horizon_min: int
    action_horizon_max: int
    action_horizon_mean: float
    action_dimension_histogram: Mapping[str, int]
    coordinate_frame_histogram: Mapping[str, int]
    control_period_histogram: Mapping[str, int]
    label_source_histogram: Mapping[str, int]
    label_strength_histogram: Mapping[str, int]
    failure_type_histogram: Mapping[str, int]
    by_task: Mapping[str, GroupMetrics]
    by_source_policy: Mapping[str, GroupMetrics]

    def __post_init__(self) -> None:
        """Freeze every mapping in deterministic insertion order."""
        for name in (
            "episodes_by_task",
            "episodes_by_source_policy",
            "action_dimension_histogram",
            "coordinate_frame_histogram",
            "control_period_histogram",
            "label_source_histogram",
            "label_strength_histogram",
            "failure_type_histogram",
            "by_task",
            "by_source_policy",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _histogram(values: Sequence[str]) -> Mapping[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _mean(values: Sequence[float | int]) -> float:
    return fsum(float(value) for value in values) / len(values)


def _group(episodes: Sequence[Episode]) -> GroupMetrics:
    candidates = [candidate for episode in episodes for candidate in episode.candidates]
    count = len(candidates)
    return GroupMetrics(
        episode_count=len(episodes),
        candidate_count=count,
        candidate_success_rate=sum(c.outcome.success for c in candidates) / count,
        candidate_unsafe_rate=sum(c.outcome.unsafe for c in candidates) / count,
        mean_progress=fsum(c.outcome.progress for c in candidates) / count,
    )


def _grouped(episodes: Sequence[Episode], attribute: str) -> Mapping[str, GroupMetrics]:
    keys = sorted({str(getattr(episode, attribute)) for episode in episodes})
    return {
        key: _group(
            [episode for episode in episodes if str(getattr(episode, attribute)) == key]
        )
        for key in keys
    }


def summarize_episodes(episodes: Sequence[Episode]) -> DatasetMetrics:
    """Summarize non-empty Episodes using candidate-level outcome denominators."""
    validate_episodes(episodes)
    if not episodes:
        raise ValueError("summarize_episodes requires at least one Episode")
    candidates: list[CandidateAction] = [
        candidate for episode in episodes for candidate in episode.candidates
    ]
    observation_counts = [len(episode.observations.frames) for episode in episodes]
    horizons = [int(candidate.action.actions.shape[0]) for candidate in candidates]
    candidate_count = len(candidates)
    successful = sum(candidate.outcome.success for candidate in candidates)
    unsafe = sum(candidate.outcome.unsafe for candidate in candidates)
    return DatasetMetrics(
        episode_count=len(episodes),
        observation_count=sum(observation_counts),
        task_count=len({episode.task_id for episode in episodes}),
        source_policy_count=len({episode.source_policy_id for episode in episodes}),
        split_group_count=len({episode.split_group_id for episode in episodes}),
        episodes_by_task=_histogram([episode.task_id for episode in episodes]),
        episodes_by_source_policy=_histogram(
            [episode.source_policy_id for episode in episodes]
        ),
        observation_count_min=min(observation_counts),
        observation_count_max=max(observation_counts),
        observation_count_mean=_mean(observation_counts),
        candidate_count=candidate_count,
        successful_candidate_count=successful,
        candidate_success_rate=successful / candidate_count,
        unsafe_candidate_count=unsafe,
        candidate_unsafe_rate=unsafe / candidate_count,
        mean_progress=fsum(c.outcome.progress for c in candidates) / candidate_count,
        action_horizon_min=min(horizons),
        action_horizon_max=max(horizons),
        action_horizon_mean=_mean(horizons),
        action_dimension_histogram=_histogram(
            [str(candidate.action.actions.shape[1]) for candidate in candidates]
        ),
        coordinate_frame_histogram=_histogram(
            [candidate.action.coordinate_frame for candidate in candidates]
        ),
        control_period_histogram=_histogram(
            [
                format(float(candidate.action.control_period_s), ".17g")
                for candidate in candidates
            ]
        ),
        label_source_histogram=_histogram(
            [candidate.outcome.label_source.value for candidate in candidates]
        ),
        label_strength_histogram=_histogram(
            [candidate.outcome.label_strength.value for candidate in candidates]
        ),
        failure_type_histogram=_histogram(
            [
                event.failure_type
                for candidate in candidates
                for event in candidate.outcome.failure_events
            ]
        ),
        by_task=_grouped(episodes, "task_id"),
        by_source_policy=_grouped(episodes, "source_policy_id"),
    )
