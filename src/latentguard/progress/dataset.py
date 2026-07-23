"""Episode-grouped split and temporal-window construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from latentguard.progress.models import StageLabel


@dataclass(frozen=True)
class EpisodeAssignment:
    """One episode's immutable primary split."""

    episode_id: str
    suite: str
    task_id: int
    seed: int
    split: str
    held_out_task: bool


@dataclass(frozen=True)
class ProgressWindow:
    """One same-episode observed progress window."""

    episode_id: str
    split: str
    start_index: int
    end_index: int
    horizon_name: str
    start_stage: int
    end_stage: int
    start_progress: float
    end_progress: float
    progress_delta: float
    stagnation: bool
    regression: bool


def assign_episode_split(
    *,
    episode_id: str,
    suite: str,
    task_id: int,
    seed: int,
    train_seeds: set[int],
    validation_seeds: set[int],
    test_seeds: set[int],
    held_out_tasks: set[tuple[str, int]] | None = None,
) -> EpisodeAssignment:
    """Assign a split exclusively from seed, with held-out task as a test tag."""

    memberships = [
        name
        for name, values in (
            ("train", train_seeds),
            ("validation", validation_seeds),
            ("test", test_seeds),
        )
        if seed in values
    ]
    if len(memberships) != 1:
        raise ValueError(
            f"seed {seed} must belong to exactly one primary split, got {memberships}"
        )
    held_out = (suite, task_id) in (held_out_tasks or set())
    if held_out and memberships[0] != "test":
        held_out = False
    return EpisodeAssignment(
        episode_id=episode_id,
        suite=suite,
        task_id=task_id,
        seed=seed,
        split=memberships[0],
        held_out_task=held_out,
    )


def build_progress_windows(
    *,
    episode_id: str,
    split: str,
    labels: Sequence[StageLabel],
    window_steps: dict[str, int],
    stride: int,
    progress_epsilon: float,
) -> list[ProgressWindow]:
    """Build within-episode short, medium, and long progress windows."""

    if not labels:
        raise ValueError("cannot build windows from an empty episode")
    if stride < 1:
        raise ValueError("stride must be positive")
    if progress_epsilon < 0.0:
        raise ValueError("progress_epsilon must be non-negative")
    windows: list[ProgressWindow] = []
    for horizon_name, offset in sorted(window_steps.items()):
        if offset < 1:
            raise ValueError(f"window {horizon_name} must have a positive offset")
        for start in range(0, len(labels) - offset, stride):
            end = start + offset
            delta = labels[end].overall_progress - labels[start].overall_progress
            windows.append(
                ProgressWindow(
                    episode_id=episode_id,
                    split=split,
                    start_index=start,
                    end_index=end,
                    horizon_name=horizon_name,
                    start_stage=labels[start].stage_id,
                    end_stage=labels[end].stage_id,
                    start_progress=labels[start].overall_progress,
                    end_progress=labels[end].overall_progress,
                    progress_delta=delta,
                    stagnation=abs(delta) <= progress_epsilon,
                    regression=delta < -progress_epsilon
                    or labels[end].stage_id < labels[start].stage_id,
                )
            )
    return windows


def assert_no_split_leakage(assignments: Sequence[EpisodeAssignment]) -> None:
    """Reject episode or seed identities assigned to multiple splits."""

    episode_splits: dict[str, str] = {}
    seed_splits: dict[int, str] = {}
    for assignment in assignments:
        prior_episode = episode_splits.setdefault(
            assignment.episode_id, assignment.split
        )
        if prior_episode != assignment.split:
            raise ValueError(
                f"episode {assignment.episode_id} crosses splits: "
                f"{prior_episode}, {assignment.split}"
            )
        prior_seed = seed_splits.setdefault(assignment.seed, assignment.split)
        if prior_seed != assignment.split:
            raise ValueError(
                f"seed {assignment.seed} crosses splits: "
                f"{prior_seed}, {assignment.split}"
            )


def window_to_dict(window: ProgressWindow) -> dict[str, Any]:
    """Serialize one progress window without simulator-native objects."""

    return {
        "episode_id": window.episode_id,
        "split": window.split,
        "start_index": window.start_index,
        "end_index": window.end_index,
        "horizon_name": window.horizon_name,
        "start_stage": window.start_stage,
        "end_stage": window.end_stage,
        "start_progress": window.start_progress,
        "end_progress": window.end_progress,
        "progress_delta": window.progress_delta,
        "stagnation": window.stagnation,
        "regression": window.regression,
    }
