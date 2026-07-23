"""Typed progress labels shared by simulator-specific stage adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageLabel:
    """One simulator-grounded stage and progress label."""

    stage_id: int
    stage_name: str
    stage_completion: float
    overall_progress: float
    terminal_success: bool
    terminal_failure: bool
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        """Reject invalid labels instead of clipping or repairing them."""

        if self.stage_id < 0:
            raise ValueError("stage_id must be non-negative")
        if not self.stage_name:
            raise ValueError("stage_name must be non-empty")
        for name, value in (
            ("stage_completion", self.stage_completion),
            ("overall_progress", self.overall_progress),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.terminal_success and self.terminal_failure:
            raise ValueError("a step cannot be terminal success and terminal failure")
        if self.terminal_success and self.overall_progress != 1.0:
            raise ValueError("terminal success must have overall_progress == 1.0")


def compose_progress(
    stage_id: int,
    stage_count: int,
    stage_completion: float,
    *,
    terminal_success: bool,
) -> float:
    """Map a task-specific stage to a common progress interval."""

    if stage_count < 1:
        raise ValueError("stage_count must be positive")
    if stage_id < 0 or stage_id >= stage_count:
        raise ValueError(
            f"stage_id {stage_id} is outside task stage count {stage_count}"
        )
    if not math.isfinite(stage_completion):
        raise ValueError("stage_completion must be finite")
    if stage_completion < 0.0 or stage_completion > 1.0:
        raise ValueError("stage_completion must be in [0, 1]")
    if terminal_success:
        return 1.0
    return min((stage_id + stage_completion) / stage_count, 1.0)
