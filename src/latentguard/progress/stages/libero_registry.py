"""Frozen task-to-stage-adapter mapping for LIBERO integrations."""

from __future__ import annotations

from typing import Any

from latentguard.progress.stages.base import TaskStageAdapter
from latentguard.progress.stages.task_adapters import (
    DeclarativeTaskStageAdapter,
    TaskStageSpec,
)


class LiberoStageRegistry:
    """Resolve task-specific adapters without importing simulator objects."""

    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        """Build an immutable adapter map from a reviewed task registry."""

        adapters: dict[tuple[str, int], TaskStageAdapter] = {}
        for task in tasks:
            suite = str(task["suite"])
            task_id = int(task["task_id"])
            key = (suite, task_id)
            if key in adapters:
                raise ValueError(f"duplicate LIBERO task registry entry: {key}")
            manipulated = tuple(str(item) for item in task["manipulated_objects"])
            predicates = tuple(
                tuple(str(item) for item in predicate)
                for predicate in task["goal_predicates"]
            )
            auxiliary = tuple(str(item) for item in task.get("auxiliary_entities", []))
            adapters[key] = DeclarativeTaskStageAdapter(
                TaskStageSpec(
                    adapter=str(task["adapter"]),
                    manipulated_objects=manipulated,
                    goal_predicates=predicates,
                    auxiliary_entities=auxiliary,
                )
            )
        if not adapters:
            raise ValueError("LIBERO stage registry must contain at least one task")
        self._adapters = adapters

    def resolve(self, suite: str, task_id: int) -> TaskStageAdapter:
        """Return the exact adapter registered for one suite/task pair."""

        key = (suite, task_id)
        try:
            return self._adapters[key]
        except KeyError as exc:
            raise KeyError(f"no stage adapter registered for {key}") from exc

    @property
    def task_keys(self) -> tuple[tuple[str, int], ...]:
        """Return sorted registered task identities."""

        return tuple(sorted(self._adapters))
