"""Generic task-stage adapter protocol."""

from __future__ import annotations

from typing import Any, Protocol

from latentguard.progress.models import StageLabel


class TaskStageAdapter(Protocol):
    """Convert privileged simulator evidence into one progress label."""

    def label_step(
        self,
        privileged_state: dict[str, Any],
        task_metadata: dict[str, Any],
    ) -> StageLabel:
        """Label one state without using future terminal outcomes."""

        ...
