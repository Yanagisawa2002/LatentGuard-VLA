"""Generic exact-anchor counterfactual future collector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .data_schema import CandidateSource, DatasetSplit, WorldModelSample


@dataclass(frozen=True, slots=True)
class ExactRestoreReport:
    """Adapter-bound complete-state restoration verification."""

    anchor_id: str
    content_identity: str
    verified: bool
    compared_component_count: int
    maximum_absolute_error: float


@dataclass(frozen=True, slots=True)
class CollectionFrame:
    """One observed boundary after state-preserving visual capture."""

    observations: NDArray[np.uint8]
    proprio: NDArray[np.float32]
    progress: float | None
    events: Mapping[str, bool | None]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Public terminal facts returned after one exact action row."""

    terminated: bool
    truncated: bool
    success: bool | None


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """One immutable candidate with explicit source provenance."""

    candidate_id: str
    actions: NDArray[np.float32]
    action_mask: NDArray[np.bool_]
    policy_source: CandidateSource
    corruption_type: str | None


class WorldModelCollectionAdapter(Protocol):
    """Simulator integration boundary used by the core collector."""

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Return the fixed ordered camera identities."""
        ...

    @property
    def event_names(self) -> tuple[str, ...]:
        """Return the fixed event-channel order."""
        ...

    def restore(self, anchor_id: str) -> ExactRestoreReport:
        """Create/reset a session and verify the complete anchor state."""
        ...

    def observe(self) -> CollectionFrame:
        """Capture state-preserving RGB, proprioception, progress, and events."""
        ...

    def step(self, action: NDArray[np.float32]) -> StepResult:
        """Execute one unmodified action row."""
        ...

    def close(self) -> None:
        """Release the current integration session."""
        ...


@dataclass(slots=True)
class CounterfactualCollector:
    """Collect aligned real futures by restoring every candidate independently."""

    adapter: WorldModelCollectionAdapter
    prediction_horizon: int
    observation_stride: int

    def __post_init__(self) -> None:
        if type(self.prediction_horizon) is not int or self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        if type(self.observation_stride) is not int or self.observation_stride < 1:
            raise ValueError("observation_stride must be positive")

    def collect(
        self,
        *,
        episode_id: str,
        source_episode_id: str,
        split_group_id: str,
        anchor_id: str,
        task_id: str,
        instruction: str | None,
        split: DatasetSplit,
        candidates: tuple[ActionCandidate, ...],
    ) -> tuple[WorldModelSample, ...]:
        """Collect every candidate from the exact same verified anchor."""

        if not candidates:
            raise ValueError("candidate set must not be empty")
        results: list[WorldModelSample] = []
        anchor_identity: str | None = None
        try:
            for candidate in candidates:
                report = self.adapter.restore(anchor_id)
                if not report.verified or report.anchor_id != anchor_id:
                    raise RuntimeError("adapter did not verify the requested anchor")
                if report.compared_component_count < 1:
                    raise RuntimeError("restore verification compared no components")
                if anchor_identity is None:
                    anchor_identity = report.content_identity
                elif report.content_identity != anchor_identity:
                    raise RuntimeError("candidate restores differ in anchor content")
                current = self.adapter.observe()
                future_frames: list[CollectionFrame] = []
                final = StepResult(False, False, None)
                valid_actions = candidate.actions[candidate.action_mask]
                required_steps = self.prediction_horizon * self.observation_stride
                if valid_actions.shape[0] < required_steps:
                    raise ValueError(
                        "candidate has fewer valid rows than the collection horizon"
                    )
                for step_index, action in enumerate(valid_actions[:required_steps], 1):
                    before = action.tobytes(order="C")
                    final = self.adapter.step(action)
                    if action.tobytes(order="C") != before:
                        raise RuntimeError("adapter mutated the candidate action")
                    if step_index % self.observation_stride == 0:
                        future_frames.append(self.adapter.observe())
                    if final.terminated or final.truncated:
                        while len(future_frames) < self.prediction_horizon:
                            future_frames.append(self.adapter.observe())
                        break
                if len(future_frames) != self.prediction_horizon:
                    raise RuntimeError("future observation alignment is incomplete")
                event_labels = np.zeros(
                    (self.prediction_horizon, len(self.adapter.event_names)),
                    dtype=np.float32,
                )
                event_mask = np.zeros_like(event_labels, dtype=np.bool_)
                progress = np.zeros(self.prediction_horizon, dtype=np.float32)
                progress_mask = np.zeros(self.prediction_horizon, dtype=np.bool_)
                for future_index, frame in enumerate(future_frames):
                    if frame.progress is not None:
                        progress[future_index] = frame.progress
                        progress_mask[future_index] = True
                    for event_index, event_name in enumerate(self.adapter.event_names):
                        value = frame.events.get(event_name)
                        if value is not None:
                            event_labels[future_index, event_index] = float(value)
                            event_mask[future_index, event_index] = True
                post_report = self.adapter.restore(anchor_id)
                if (
                    not post_report.verified
                    or post_report.content_identity != anchor_identity
                ):
                    raise RuntimeError("post-candidate anchor restoration failed")
                results.append(
                    WorldModelSample(
                        episode_id=episode_id,
                        source_episode_id=source_episode_id,
                        split_group_id=split_group_id,
                        anchor_id=anchor_id,
                        candidate_id=candidate.candidate_id,
                        task_id=task_id,
                        instruction=instruction,
                        camera_ids=self.adapter.camera_ids,
                        observation_t=np.array(
                            current.observations, copy=True, dtype=np.uint8
                        ),
                        proprio_t=np.array(
                            current.proprio, copy=True, dtype=np.float32
                        ),
                        action_chunk=np.array(
                            candidate.actions, copy=True, dtype=np.float32
                        ),
                        action_mask=np.array(
                            candidate.action_mask, copy=True, dtype=np.bool_
                        ),
                        future_observations=np.stack(
                            [frame.observations for frame in future_frames]
                        ).astype(np.uint8, copy=False),
                        future_proprio=np.stack(
                            [frame.proprio for frame in future_frames]
                        ).astype(np.float32, copy=False),
                        progress_t=current.progress,
                        future_progress=progress,
                        progress_mask=progress_mask,
                        event_names=self.adapter.event_names,
                        event_labels=event_labels,
                        event_mask=event_mask,
                        terminal_success=final.success,
                        terminal_reached=final.terminated or final.truncated,
                        policy_source=candidate.policy_source,
                        corruption_type=candidate.corruption_type,
                        split=split,
                        observation_stride=self.observation_stride,
                        restore_identity=anchor_identity,
                    )
                )
        finally:
            self.adapter.close()
        return tuple(results)
