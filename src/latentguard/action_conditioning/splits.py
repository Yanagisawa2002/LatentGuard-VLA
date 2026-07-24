"""Deterministic episode- and seed-grouped diagnostic cross-validation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeRecord:
    """Fold-assignment metadata with no model targets beyond stratification."""

    episode_id: str
    seed: int
    task: str
    success: bool
    taxonomy: str | None
    progress_bin: int
    windows: int


def assign_grouped_folds(
    records: list[EpisodeRecord],
    *,
    folds: int = 5,
    assignment_seed: int = 22031,
) -> dict[str, int]:
    """Assign seed groups greedily while balancing task and failure support."""

    if folds < 2:
        raise ValueError("at least two folds are required")
    if not records:
        raise ValueError("episode records cannot be empty")
    if len({record.episode_id for record in records}) != len(records):
        raise ValueError("episode IDs must be unique")
    seed_groups: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        seed_groups[record.seed].append(record)
    failure_groups = sum(
        1
        for group in seed_groups.values()
        if any(not record.success for record in group)
    )
    if failure_groups < folds:
        raise ValueError("not enough independent failure seed groups for every fold")

    def digest(group: list[EpisodeRecord]) -> str:
        identity = "|".join(sorted(record.episode_id for record in group))
        return hashlib.sha256(f"{assignment_seed}|{identity}".encode()).hexdigest()

    ordered = sorted(
        seed_groups.values(),
        key=lambda group: (
            not any(not record.success for record in group),
            -sum(record.windows for record in group),
            digest(group),
        ),
    )
    fold_records: list[list[EpisodeRecord]] = [[] for _ in range(folds)]
    task_counts: list[Counter[str]] = [Counter() for _ in range(folds)]
    taxonomy_counts: list[Counter[str]] = [Counter() for _ in range(folds)]
    window_counts = [0] * folds
    failure_counts = [0] * folds

    for group in ordered:
        group_tasks = Counter(record.task for record in group)
        group_taxonomies = Counter(
            record.taxonomy for record in group if record.taxonomy is not None
        )
        group_windows = sum(record.windows for record in group)
        group_failures = sum(not record.success for record in group)

        def score(
            fold: int,
            *,
            failures_in_group: int = group_failures,
            taxonomies_in_group: Counter[str] = group_taxonomies,
            tasks_in_group: Counter[str] = group_tasks,
        ) -> tuple[float, int]:
            failure_penalty = (
                failure_counts[fold] * 1_000_000 if failures_in_group else 0
            )
            taxonomy_penalty = 20_000 * sum(
                taxonomy_counts[fold][key] * value
                for key, value in taxonomies_in_group.items()
            )
            task_penalty = 1_000 * sum(
                task_counts[fold][key] * value for key, value in tasks_in_group.items()
            )
            return (
                float(
                    failure_penalty
                    + taxonomy_penalty
                    + task_penalty
                    + window_counts[fold]
                ),
                fold,
            )

        selected = min(range(folds), key=score)
        fold_records[selected].extend(group)
        task_counts[selected].update(group_tasks)
        taxonomy_counts[selected].update(group_taxonomies)
        window_counts[selected] += group_windows
        failure_counts[selected] += group_failures

    assignments = {
        record.episode_id: fold
        for fold, group in enumerate(fold_records)
        for record in group
    }
    validate_grouped_folds(records, assignments, folds=folds)
    return assignments


def validate_grouped_folds(
    records: list[EpisodeRecord],
    assignments: dict[str, int],
    *,
    folds: int = 5,
) -> dict[str, Any]:
    """Require episode/seed isolation and natural-failure support in every fold."""

    episode_ids = {record.episode_id for record in records}
    if set(assignments) != episode_ids:
        raise ValueError("fold assignments must cover every episode exactly once")
    invalid = sorted(set(assignments.values()) - set(range(folds)))
    if invalid:
        raise ValueError(f"invalid fold indices: {invalid}")
    seed_folds: dict[int, set[int]] = defaultdict(set)
    summaries: list[dict[str, Any]] = []
    for record in records:
        seed_folds[record.seed].add(assignments[record.episode_id])
    leaking = sorted(seed for seed, values in seed_folds.items() if len(values) != 1)
    if leaking:
        raise ValueError(f"seeds cross folds: {leaking}")
    for fold in range(folds):
        members = [
            record for record in records if assignments[record.episode_id] == fold
        ]
        failures = [record for record in members if not record.success]
        if not failures:
            raise ValueError(f"fold {fold} has no natural failure episode")
        summaries.append(
            {
                "fold": fold,
                "episodes": len(members),
                "seeds": len({record.seed for record in members}),
                "windows": sum(record.windows for record in members),
                "failures": len(failures),
                "tasks": dict(
                    sorted(Counter(record.task for record in members).items())
                ),
                "taxonomies": dict(
                    sorted(
                        Counter(
                            record.taxonomy
                            for record in failures
                            if record.taxonomy is not None
                        ).items()
                    )
                ),
            }
        )
    return {
        "status": "pass",
        "episode_leakage": 0,
        "seed_leakage": 0,
        "folds": summaries,
    }


def records_as_dicts(records: list[EpisodeRecord]) -> list[dict[str, Any]]:
    """Serialize episode records deterministically."""

    return [asdict(record) for record in records]
