"""Shared strict WM-v0 test fixtures."""

from __future__ import annotations

import numpy as np

from latentguard.world_model.data_schema import (
    CandidateSource,
    DatasetSplit,
    WorldModelSample,
)


def sample(
    *, group: str = "source-1", split: DatasetSplit = DatasetSplit.TRAIN
) -> WorldModelSample:
    """Return one minimal, valid, real-future sample."""

    return WorldModelSample(
        episode_id="episode-1",
        source_episode_id=group,
        split_group_id=group,
        anchor_id="anchor-1",
        candidate_id="candidate-1",
        task_id="maniskill/PickCube-v1",
        instruction=None,
        camera_ids=("front", "side"),
        observation_t=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        proprio_t=np.arange(5, dtype=np.float32),
        action_chunk=np.zeros((4, 3), dtype=np.float32),
        action_mask=np.asarray([True, True, True, True], dtype=np.bool_),
        future_observations=np.zeros((2, 2, 4, 4, 3), dtype=np.uint8),
        future_proprio=np.zeros((2, 5), dtype=np.float32),
        progress_t=0.1,
        future_progress=np.asarray([0.2, 0.3], dtype=np.float32),
        progress_mask=np.asarray([True, True], dtype=np.bool_),
        event_names=("grasped", "collision"),
        event_labels=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        event_mask=np.asarray([[True, False], [True, False]], dtype=np.bool_),
        terminal_success=False,
        terminal_reached=True,
        policy_source=CandidateSource.SYNTHETIC_CORRUPTION,
        corruption_type="delay",
        split=split,
        observation_stride=2,
        restore_identity="sha256:" + "a" * 64,
    )
