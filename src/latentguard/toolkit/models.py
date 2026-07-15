"""Immutable models for deterministic Episode quality audits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from latentguard.models import JsonScalar


class AuditSeverity(StrEnum):
    """Severity of one non-repairing data-quality finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class DataIssue:
    """One stable, contextual data-quality finding."""

    issue_code: str
    severity: AuditSeverity
    message: str
    episode_id: str | None = None
    candidate_id: str | None = None
    observation_id: str | None = None
    camera_id: str | None = None
    details: Mapping[str, JsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach details behind an immutable mapping."""
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Deterministic aggregate of ordered quality findings."""

    episode_count: int
    candidate_count: int
    issue_count: int
    issues_by_severity: Mapping[str, int]
    issues_by_code: Mapping[str, int]
    issues: tuple[DataIssue, ...]

    def __post_init__(self) -> None:
        """Freeze mappings and issue order."""
        object.__setattr__(
            self, "issues_by_severity", MappingProxyType(dict(self.issues_by_severity))
        )
        object.__setattr__(
            self, "issues_by_code", MappingProxyType(dict(self.issues_by_code))
        )
        object.__setattr__(self, "issues", tuple(self.issues))


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Explicit thresholds for deterministic quality checks."""

    timestamp_tolerance_s: float = 1e-6
    frozen_frame_run_length: int = 3
    near_zero_action_threshold: float = 1e-6
    near_zero_action_fraction: float = 0.9
    action_jump_threshold: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid thresholds without coercion or fallback."""
        _finite_nonnegative(self.timestamp_tolerance_s, "timestamp_tolerance_s")
        if (
            isinstance(self.frozen_frame_run_length, bool)
            or not isinstance(self.frozen_frame_run_length, int)
            or self.frozen_frame_run_length < 2
        ):
            raise ValueError("frozen_frame_run_length must be an integer >= 2")
        _finite_nonnegative(
            self.near_zero_action_threshold, "near_zero_action_threshold"
        )
        fraction = _finite_nonnegative(
            self.near_zero_action_fraction, "near_zero_action_fraction"
        )
        if fraction > 1.0:
            raise ValueError("near_zero_action_fraction must be within [0, 1]")
        if self.action_jump_threshold is not None:
            _finite_nonnegative(self.action_jump_threshold, "action_jump_threshold")
