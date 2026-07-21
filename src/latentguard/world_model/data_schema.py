"""Strict, simulator-independent WM-v0 sample contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class WorldModelSchemaError(ValueError):
    """Raised when a world-model sample violates its declared contract."""


class CandidateSource(StrEnum):
    """Auditable source of one candidate action chunk."""

    POLICY_GENERATED = "policy_generated"
    SYNTHETIC_CORRUPTION = "synthetic_corruption"


class DatasetSplit(StrEnum):
    """Immutable source-group split assignment."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise WorldModelSchemaError(f"{name} must be non-empty stripped text")


def _finite_array(
    value: NDArray[Any],
    *,
    name: str,
    dimensions: int,
    nonempty: bool = True,
) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != dimensions:
        raise WorldModelSchemaError(f"{name} must be a {dimensions}-D ndarray")
    if nonempty and value.size == 0:
        raise WorldModelSchemaError(f"{name} must not be empty")
    if not np.issubdtype(value.dtype, np.number) or not bool(np.isfinite(value).all()):
        raise WorldModelSchemaError(f"{name} must contain finite numeric values")


def _rgb_views(value: NDArray[Any], name: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("uint8")
        or value.ndim != 4
        or value.shape[0] < 1
        or value.shape[1] < 1
        or value.shape[2] < 1
        or value.shape[3] != 3
    ):
        raise WorldModelSchemaError(f"{name} must be uint8 [views,height,width,3] RGB")


@dataclass(frozen=True, slots=True, eq=False)
class WorldModelSample:
    """One exact-anchor candidate and its real post-execution trajectory."""

    episode_id: str
    source_episode_id: str
    split_group_id: str
    anchor_id: str
    candidate_id: str
    task_id: str
    instruction: str | None
    camera_ids: tuple[str, ...]
    observation_t: NDArray[np.uint8]
    proprio_t: NDArray[np.float32]
    action_chunk: NDArray[np.float32]
    action_mask: NDArray[np.bool_]
    future_observations: NDArray[np.uint8]
    future_proprio: NDArray[np.float32]
    progress_t: float | None
    future_progress: NDArray[np.float32]
    progress_mask: NDArray[np.bool_]
    event_names: tuple[str, ...]
    event_labels: NDArray[np.float32]
    event_mask: NDArray[np.bool_]
    terminal_success: bool | None
    terminal_reached: bool
    policy_source: CandidateSource
    corruption_type: str | None
    split: DatasetSplit
    observation_stride: int
    restore_identity: str
    future_source: str = "simulator_execution"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate shapes, masks, provenance, and real-future semantics."""

        for name in (
            "episode_id",
            "source_episode_id",
            "split_group_id",
            "anchor_id",
            "candidate_id",
            "task_id",
            "restore_identity",
        ):
            _identifier(getattr(self, name), name)
        if self.split_group_id != self.source_episode_id:
            raise WorldModelSchemaError(
                "split_group_id must equal the original source_episode_id"
            )
        if self.instruction is not None:
            _identifier(self.instruction, "instruction")
        if self.schema_version != "1.0":
            raise WorldModelSchemaError("unsupported world-model sample schema")
        if self.future_source != "simulator_execution":
            raise WorldModelSchemaError(
                "future_source must prove real simulator execution"
            )
        if type(self.observation_stride) is not int or self.observation_stride < 1:
            raise WorldModelSchemaError("observation_stride must be positive")
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise WorldModelSchemaError("camera_ids must be non-empty and unique")
        for camera_id in self.camera_ids:
            _identifier(camera_id, "camera_id")
        _rgb_views(self.observation_t, "observation_t")
        if self.observation_t.shape[0] != len(self.camera_ids):
            raise WorldModelSchemaError("current observation camera count differs")
        if (
            not isinstance(self.future_observations, np.ndarray)
            or self.future_observations.dtype != np.dtype("uint8")
            or self.future_observations.ndim != 5
            or self.future_observations.shape[0] < 1
        ):
            raise WorldModelSchemaError(
                "future_observations must be non-empty uint8 [K,V,H,W,3]"
            )
        for future in self.future_observations:
            _rgb_views(future, "future_observations frame")
        if tuple(self.future_observations.shape[1:]) != tuple(self.observation_t.shape):
            raise WorldModelSchemaError("future RGB shape differs from current RGB")
        _finite_array(self.proprio_t, name="proprio_t", dimensions=1)
        _finite_array(self.action_chunk, name="action_chunk", dimensions=2)
        if (
            not isinstance(self.action_mask, np.ndarray)
            or self.action_mask.dtype != np.dtype("bool")
            or self.action_mask.shape != (self.action_chunk.shape[0],)
            or not bool(self.action_mask.any())
        ):
            raise WorldModelSchemaError("action_mask must select at least one row")
        false_rows = np.flatnonzero(~self.action_mask)
        if false_rows.size:
            first_false = int(false_rows[0])
            if bool(self.action_mask[first_false:].any()):
                raise WorldModelSchemaError("action_mask must be a contiguous prefix")
        _finite_array(self.future_proprio, name="future_proprio", dimensions=2)
        horizon = self.future_observations.shape[0]
        if self.future_proprio.shape != (horizon, self.proprio_t.shape[0]):
            raise WorldModelSchemaError("future proprio alignment mismatch")
        if self.progress_t is not None and (
            type(self.progress_t) is not float or not math.isfinite(self.progress_t)
        ):
            raise WorldModelSchemaError("progress_t must be a finite float or None")
        _finite_array(
            self.future_progress,
            name="future_progress",
            dimensions=1,
            nonempty=False,
        )
        if self.future_progress.shape != (horizon,):
            raise WorldModelSchemaError("future progress alignment mismatch")
        if (
            not isinstance(self.progress_mask, np.ndarray)
            or self.progress_mask.dtype != np.dtype("bool")
            or self.progress_mask.shape != (horizon,)
        ):
            raise WorldModelSchemaError("progress_mask alignment mismatch")
        if not self.event_names or len(set(self.event_names)) != len(self.event_names):
            raise WorldModelSchemaError("event_names must be non-empty and unique")
        for event_name in self.event_names:
            _identifier(event_name, "event_name")
        _finite_array(
            self.event_labels,
            name="event_labels",
            dimensions=2,
            nonempty=False,
        )
        expected_event_shape = (horizon, len(self.event_names))
        if self.event_labels.shape != expected_event_shape:
            raise WorldModelSchemaError("event label alignment mismatch")
        if (
            not isinstance(self.event_mask, np.ndarray)
            or self.event_mask.dtype != np.dtype("bool")
            or self.event_mask.shape != expected_event_shape
        ):
            raise WorldModelSchemaError("event mask alignment mismatch")
        observed_events = self.event_labels[self.event_mask]
        if observed_events.size and not bool(
            np.logical_or(observed_events == 0.0, observed_events == 1.0).all()
        ):
            raise WorldModelSchemaError("observed event targets must be binary")
        if (
            self.terminal_success is not None
            and type(self.terminal_success) is not bool
        ):
            raise WorldModelSchemaError("terminal_success must be boolean or None")
        if type(self.terminal_reached) is not bool:
            raise WorldModelSchemaError("terminal_reached must be boolean")
        if self.terminal_reached != (self.terminal_success is not None):
            raise WorldModelSchemaError(
                "terminal_success availability must match terminal_reached"
            )
        if self.policy_source is CandidateSource.POLICY_GENERATED:
            if self.corruption_type is not None:
                raise WorldModelSchemaError(
                    "policy-generated candidates cannot declare corruption_type"
                )
        else:
            if self.corruption_type is None:
                raise WorldModelSchemaError(
                    "synthetic corruptions require corruption_type"
                )
            _identifier(self.corruption_type, "corruption_type")

    @property
    def prediction_horizon(self) -> int:
        """Return the number of aligned future observation boundaries."""

        return int(self.future_observations.shape[0])

    def metadata_mapping(self) -> dict[str, object]:
        """Return JSON-native metadata without duplicating array payloads."""

        return {
            "anchor_id": self.anchor_id,
            "camera_ids": list(self.camera_ids),
            "candidate_id": self.candidate_id,
            "corruption_type": self.corruption_type,
            "episode_id": self.episode_id,
            "event_names": list(self.event_names),
            "future_source": self.future_source,
            "instruction": self.instruction,
            "observation_stride": self.observation_stride,
            "policy_source": self.policy_source.value,
            "restore_identity": self.restore_identity,
            "schema_version": self.schema_version,
            "source_episode_id": self.source_episode_id,
            "split": self.split.value,
            "split_group_id": self.split_group_id,
            "task_id": self.task_id,
            "terminal_reached": self.terminal_reached,
            "terminal_success": self.terminal_success,
        }

    @classmethod
    def from_parts(
        cls,
        metadata: Mapping[str, object],
        arrays: Mapping[str, NDArray[Any]],
    ) -> WorldModelSample:
        """Construct and strictly validate a sample from stored parts."""

        try:
            raw_camera_ids = metadata["camera_ids"]
            raw_event_names = metadata["event_names"]
            raw_stride = metadata["observation_stride"]
            if not isinstance(raw_camera_ids, list) or not isinstance(
                raw_event_names, list
            ):
                raise WorldModelSchemaError(
                    "stored camera_ids and event_names must be lists"
                )
            if type(raw_stride) is not int:
                raise WorldModelSchemaError(
                    "stored observation_stride must be an integer"
                )
            return cls(
                episode_id=str(metadata["episode_id"]),
                source_episode_id=str(metadata["source_episode_id"]),
                split_group_id=str(metadata["split_group_id"]),
                anchor_id=str(metadata["anchor_id"]),
                candidate_id=str(metadata["candidate_id"]),
                task_id=str(metadata["task_id"]),
                instruction=(
                    None
                    if metadata["instruction"] is None
                    else str(metadata["instruction"])
                ),
                camera_ids=tuple(str(item) for item in raw_camera_ids),
                observation_t=np.asarray(arrays["observation_t"], dtype=np.uint8),
                proprio_t=np.asarray(arrays["proprio_t"], dtype=np.float32),
                action_chunk=np.asarray(arrays["action_chunk"], dtype=np.float32),
                action_mask=np.asarray(arrays["action_mask"], dtype=np.bool_),
                future_observations=np.asarray(
                    arrays["future_observations"], dtype=np.uint8
                ),
                future_proprio=np.asarray(arrays["future_proprio"], dtype=np.float32),
                progress_t=(
                    None
                    if not bool(np.asarray(arrays["progress_t_present"]).item())
                    else float(np.asarray(arrays["progress_t"]).item())
                ),
                future_progress=np.asarray(arrays["future_progress"], dtype=np.float32),
                progress_mask=np.asarray(arrays["progress_mask"], dtype=np.bool_),
                event_names=tuple(str(item) for item in raw_event_names),
                event_labels=np.asarray(arrays["event_labels"], dtype=np.float32),
                event_mask=np.asarray(arrays["event_mask"], dtype=np.bool_),
                terminal_success=(
                    None
                    if metadata["terminal_success"] is None
                    else bool(metadata["terminal_success"])
                ),
                terminal_reached=bool(metadata["terminal_reached"]),
                policy_source=CandidateSource(str(metadata["policy_source"])),
                corruption_type=(
                    None
                    if metadata["corruption_type"] is None
                    else str(metadata["corruption_type"])
                ),
                split=DatasetSplit(str(metadata["split"])),
                observation_stride=raw_stride,
                restore_identity=str(metadata["restore_identity"]),
                future_source=str(metadata["future_source"]),
                schema_version=str(metadata["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, WorldModelSchemaError):
                raise
            raise WorldModelSchemaError(f"invalid stored sample: {exc}") from exc

    def array_mapping(self) -> dict[str, NDArray[Any]]:
        """Return detached C-order arrays for safe NPZ serialization."""

        progress_value = 0.0 if self.progress_t is None else self.progress_t
        return {
            "action_chunk": np.array(self.action_chunk, copy=True, order="C"),
            "action_mask": np.array(self.action_mask, copy=True, order="C"),
            "event_labels": np.array(self.event_labels, copy=True, order="C"),
            "event_mask": np.array(self.event_mask, copy=True, order="C"),
            "future_observations": np.array(
                self.future_observations, copy=True, order="C"
            ),
            "future_progress": np.array(self.future_progress, copy=True, order="C"),
            "future_proprio": np.array(self.future_proprio, copy=True, order="C"),
            "observation_t": np.array(self.observation_t, copy=True, order="C"),
            "progress_mask": np.array(self.progress_mask, copy=True, order="C"),
            "progress_t": np.asarray(progress_value, dtype=np.float32),
            "progress_t_present": np.asarray(
                self.progress_t is not None, dtype=np.bool_
            ),
            "proprio_t": np.array(self.proprio_t, copy=True, order="C"),
        }
