"""Typed boundary between LatentGuard replay logic and a RoboLab runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ReplayContract:
    """Pre-registered constants for the LG-RB0 replay feasibility gate."""

    official_state_tolerance: float
    takeover_state_tolerance: float
    repeats: int
    branch_checkpoints: tuple[int, ...]
    anchor_fractions: tuple[float, ...]
    num_envs: int = 1

    def validate(self) -> None:
        """Reject protocol changes that weaken the pre-registered gate."""
        if self.official_state_tolerance != 0.01:
            raise ValueError("official RoboLab replay tolerance must remain 0.01")
        if self.takeover_state_tolerance != 1e-6:
            raise ValueError("takeover state tolerance must remain 1e-6")
        if self.repeats != 3:
            raise ValueError("LG-RB0 requires exactly three repeats")
        if self.branch_checkpoints != (1, 5, 10):
            raise ValueError("LG-RB0 branch checkpoints must be 1, 5, and 10")
        if self.anchor_fractions != (0.25, 0.5, 0.75):
            raise ValueError("LG-RB0 anchors must be early, middle, and late")
        if self.num_envs != 1:
            raise ValueError("faithful RoboLab replay requires num_envs=1")


@dataclass(frozen=True)
class ReplaySnapshot:
    """Content-bound observable and semantic state at one replay boundary."""

    step: int
    state: Mapping[str, Any]
    observation_sha256: str
    predicates: Mapping[str, bool]
    subtask: Mapping[str, Any]
    events: Sequence[Mapping[str, Any]]
    termination_latch: bool
    success: bool


class ReplayAdapter(Protocol):
    """Minimal runtime interface implemented only on the remote RoboLab stack."""

    def reset_and_restore(self, recording: str, episode: int) -> ReplaySnapshot:
        """Reset a fresh session and restore the content-bound initial state."""

    def recorded_actions(self, recording: str, episode: int) -> Sequence[Any]:
        """Load the official recorded action sequence."""

    def step(self, action: Any) -> ReplaySnapshot:
        """Execute one legal action and capture the complete boundary state."""

    def fixed_suffix(self, branch_id: str, steps: int) -> Sequence[Any]:
        """Construct a deterministic, state-derived, pre-registered legal suffix."""

    def close(self) -> None:
        """Close the simulator session and release its resources."""
