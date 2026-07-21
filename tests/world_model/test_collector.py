"""Exact-anchor counterfactual collector tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from latentguard.world_model.collector import (
    ActionCandidate,
    CollectionFrame,
    CounterfactualCollector,
    ExactRestoreReport,
    StepResult,
)
from latentguard.world_model.data_schema import CandidateSource, DatasetSplit


@dataclass(slots=True)
class _Adapter:
    camera_ids: tuple[str, ...] = ("front",)
    event_names: tuple[str, ...] = ("grasped", "collision")
    restores: int = 0
    steps: int = 0
    states: list[int] = field(default_factory=list)

    def restore(self, anchor_id: str) -> ExactRestoreReport:
        self.restores += 1
        self.steps = 0
        self.states.append(self.steps)
        return ExactRestoreReport(anchor_id, "sha256:" + "1" * 64, True, 70, 1e-7)

    def observe(self) -> CollectionFrame:
        return CollectionFrame(
            observations=np.full((1, 2, 2, 3), self.steps, dtype=np.uint8),
            proprio=np.asarray([self.steps, 0.0], dtype=np.float32),
            progress=float(self.steps) / 4.0,
            events={"grasped": self.steps >= 2, "collision": None},
        )

    def step(self, action: np.ndarray) -> StepResult:  # type: ignore[type-arg]
        self.steps += 1
        return StepResult(False, False, None)

    def close(self) -> None:
        return None


def test_exact_restore_and_future_alignment() -> None:
    """Every candidate starts and ends at the same verified content identity."""

    adapter = _Adapter()
    collector = CounterfactualCollector(
        adapter, prediction_horizon=2, observation_stride=2
    )
    action = np.zeros((4, 3), dtype=np.float32)
    candidates = tuple(
        ActionCandidate(
            candidate_id=f"candidate-{index}",
            actions=action.copy(),
            action_mask=np.ones(4, dtype=np.bool_),
            policy_source=CandidateSource.SYNTHETIC_CORRUPTION,
            corruption_type="delay",
        )
        for index in range(2)
    )
    samples = collector.collect(
        episode_id="episode",
        source_episode_id="source",
        split_group_id="source",
        anchor_id="anchor",
        task_id="task",
        instruction=None,
        split=DatasetSplit.TRAIN,
        candidates=candidates,
    )
    assert len(samples) == 2
    assert adapter.restores == 4
    assert all(
        item.future_observations[:, 0, 0, 0, 0].tolist() == [2, 4] for item in samples
    )
    assert all(not bool(item.event_mask[:, 1].any()) for item in samples)
