"""Simulator-independent, mutation-safe data models for LatentGuard-VLA."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

CURRENT_SCHEMA_VERSION = "1.0"
"""Schema version supported by the M0 data contract."""

SUPPORTED_SCHEMA_VERSIONS = frozenset({CURRENT_SCHEMA_VERSION})
"""Schema versions that can be validated and loaded by this release."""

JsonScalar: TypeAlias = str | int | float | bool | None


def _freeze_array(value: NDArray[Any]) -> NDArray[Any]:
    """Return a detached array backed by an immutable byte buffer."""
    detached = np.array(value, copy=True, order="C", subok=False)
    if detached.dtype.hasobject:
        detached.setflags(write=False)
        return detached
    immutable_buffer = detached.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=detached.dtype).reshape(detached.shape)


def _freeze_mapping(value: Mapping[str, JsonScalar]) -> Mapping[str, JsonScalar]:
    """Return an immutable shallow copy of scalar transformation metadata."""
    return MappingProxyType(dict(value))


class LabelSource(StrEnum):
    """Origin of an outcome label."""

    HEURISTIC = "heuristic"
    SIMULATOR = "simulator"
    HUMAN = "human"
    TRUSTED_ORACLE = "trusted_oracle"
    DETERMINISTIC_EVALUATOR = "deterministic_evaluator"


class LabelStrength(StrEnum):
    """Confidence tier assigned to an outcome label."""

    WEAK = "weak"
    STRONG = "strong"


@dataclass(frozen=True, slots=True, eq=False)
class CameraFrame:
    """One RGB camera frame with optional aligned calibration and depth."""

    camera_id: str
    rgb: NDArray[Any]
    depth: NDArray[Any] | None = None
    intrinsics: NDArray[Any] | None = None
    extrinsics: NDArray[Any] | None = None
    schema_version: str = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach and freeze array fields without changing their values."""
        if isinstance(self.rgb, np.ndarray):
            object.__setattr__(self, "rgb", _freeze_array(self.rgb))
        if isinstance(self.depth, np.ndarray):
            object.__setattr__(self, "depth", _freeze_array(self.depth))
        if isinstance(self.intrinsics, np.ndarray):
            object.__setattr__(self, "intrinsics", _freeze_array(self.intrinsics))
        if isinstance(self.extrinsics, np.ndarray):
            object.__setattr__(self, "extrinsics", _freeze_array(self.extrinsics))


@dataclass(frozen=True, slots=True, eq=False)
class ObservationFrame:
    """A timestamped robot-state observation and zero or more cameras."""

    observation_id: str
    timestamp_s: float
    robot_state: NDArray[Any]
    cameras: tuple[CameraFrame, ...] = ()
    schema_version: str = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach arrays and convert camera collections to immutable tuples."""
        if isinstance(self.robot_state, np.ndarray):
            object.__setattr__(self, "robot_state", _freeze_array(self.robot_state))
        object.__setattr__(self, "cameras", tuple(self.cameras))


@dataclass(frozen=True, slots=True, eq=False)
class ObservationHistory:
    """Chronologically ordered observations belonging to one episode."""

    history_id: str
    frames: tuple[ObservationFrame, ...]
    schema_version: str = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Convert the frame collection to an immutable tuple."""
        object.__setattr__(self, "frames", tuple(self.frames))


@dataclass(frozen=True, slots=True, eq=False)
class ActionChunk:
    """A fixed-period action sequence shaped ``[horizon, action_dim]``."""

    actions: NDArray[Any]
    coordinate_frame: str
    control_period_s: float
    schema_version: str = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach and freeze the action array."""
        if isinstance(self.actions, np.ndarray):
            object.__setattr__(self, "actions", _freeze_array(self.actions))


@dataclass(frozen=True, slots=True, eq=False)
class FailureEvent:
    """A typed failure annotation optionally localized in time."""

    failure_type: str
    timestamp_s: float | None = None
    probability: float | None = None
    description: str | None = None
    schema_version: str = CURRENT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, eq=False)
class OutcomeLabel:
    """Success, progress, safety, and failure labels for one candidate."""

    success: bool
    progress: float
    unsafe: bool
    label_source: LabelSource
    label_strength: LabelStrength
    simulator_replay_verified: bool = False
    success_probability: float | None = None
    unsafe_probability: float | None = None
    failure_events: tuple[FailureEvent, ...] = ()
    schema_version: str = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Convert failure annotations to an immutable tuple."""
        object.__setattr__(self, "failure_events", tuple(self.failure_events))


@dataclass(frozen=True, slots=True, eq=False)
class SampleProvenance:
    """Complete source, transformation, split, and label provenance."""

    source_episode_id: str
    source_policy_id: str
    source_task_id: str
    transformation_type: str
    transformation_parameters: Mapping[str, JsonScalar]
    seed: int
    label_source: LabelSource
    label_strength: LabelStrength
    simulator_replay_verified: bool
    split_group_id: str
    schema_version: str = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach transformation metadata behind an immutable mapping."""
        object.__setattr__(
            self,
            "transformation_parameters",
            _freeze_mapping(self.transformation_parameters),
        )


@dataclass(frozen=True, slots=True, eq=False)
class CandidateAction:
    """A candidate action chunk with its outcome and complete provenance."""

    candidate_id: str
    action: ActionChunk
    outcome: OutcomeLabel
    provenance: SampleProvenance
    schema_version: str = CURRENT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, eq=False)
class Episode:
    """A task episode containing observation history and candidate actions."""

    episode_id: str
    task_id: str
    source_policy_id: str
    instruction: str
    observations: ObservationHistory
    candidates: tuple[CandidateAction, ...]
    split_group_id: str
    schema_version: str = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Convert candidate collections to an immutable tuple."""
        object.__setattr__(self, "candidates", tuple(self.candidates))
