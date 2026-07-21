"""CPU-only tests for ACT runtime data, sampling, stats, and checkpoint writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from latentguard.policies.act.data import PickCubeDemoEpisode, save_demo_episode
from latentguard.policies.act.runtime import (
    DeterministicResumeBatchSampler,
    PickCubeActDataset,
    PickCubeActInferenceRuntime,
    PickCubeActRuntimeError,
    prepare_training_batch,
    processor_statistics,
    save_checkpoint,
)
from latentguard.policies.act.types import PickCubeActExperimentConfig

_DIGEST = "sha256:" + "a" * 64


def _episode(seed: int) -> PickCubeDemoEpisode:
    return PickCubeDemoEpisode(
        scene_seed=seed,
        rgb=np.zeros((3, 224, 224, 3), dtype=np.uint8),
        proprioception=np.zeros((3, 18), dtype=np.float32),
        actions=np.arange(24, dtype=np.float64).reshape(3, 8),
        phases=("REACH_PREGRASP", "DESCEND_TO_GRASP", "CLOSE_GRIPPER"),
        compatibility_identity=_DIGEST,
        contract_digest=_DIGEST,
        camera_configuration_digest=_DIGEST,
    )


def test_runtime_dataset_emits_only_allowlisted_chunk_tensors(tmp_path: Path) -> None:
    reference = save_demo_episode(tmp_path, _episode(2))
    dataset = PickCubeActDataset(tmp_path, (reference,), chunk_size=4)
    sample = dataset[2]
    assert set(sample) == {
        "action",
        "action_is_pad",
        "observation.images.front_oblique",
        "observation.state",
    }
    assert sample["observation.images.front_oblique"].shape == (3, 224, 224)
    assert sample["observation.images.front_oblique"].dtype == torch.uint8
    assert sample["observation.state"].shape == (18,)
    assert sample["action"].shape == (4, 8)
    assert sample["action_is_pad"].tolist() == [False, True, True, True]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    prepared = prepare_training_batch(batch)
    assert prepared["observation.images.front_oblique"].dtype == torch.float32
    assert prepared["observation.images.front_oblique"].max() == 0


def test_resume_sampler_starts_at_the_same_global_batch() -> None:
    complete = iter(
        DeterministicResumeBatchSampler(
            dataset_size=10,
            batch_size=4,
            seed=7,
        )
    )
    first = [next(complete) for _ in range(5)]
    resumed = iter(
        DeterministicResumeBatchSampler(
            dataset_size=10,
            batch_size=4,
            seed=7,
            start_step=4,
        )
    )
    assert next(resumed) == first[4]


def test_processor_statistics_use_image_channel_broadcast() -> None:
    scalar = {
        "maximum": [1.0] * 18,
        "mean": [0.0] * 18,
        "minimum": [-1.0] * 18,
        "standard_deviation": [0.5] * 18,
    }
    action = {key: value[:8] for key, value in scalar.items()}
    stats = processor_statistics(
        {
            "action": action,
            "image": {
                "channel_mean": [0.1, 0.2, 0.3],
                "channel_standard_deviation": [0.4, 0.5, 0.6],
            },
            "proprioception": scalar,
        }
    )
    assert stats["observation.images.front_oblique"]["mean"].shape == (3, 1, 1)
    assert stats["observation.state"]["mean"].shape == (18,)
    assert stats["action"]["mean"].shape == (8,)


class _SavedComponent:
    def __init__(self, name: str) -> None:
        self.name = name

    def save_pretrained(self, root: Path, **kwargs: Any) -> None:
        filename = str(kwargs.get("config_filename", self.name + ".json"))
        (root / filename).write_text(self.name, encoding="utf-8")


class _FakeInferencePolicy:
    def __init__(self, chunk: torch.Tensor) -> None:
        self.chunk = chunk
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert set(batch) == {
            "observation.images.front_oblique",
            "observation.state",
        }
        return self.chunk


def _experiment() -> PickCubeActExperimentConfig:
    value = json.loads(Path("configs/pickcube_act/train.yaml").read_text())
    return PickCubeActExperimentConfig.from_mapping(value)


def test_inference_runtime_preserves_valid_chunk_and_rejects_clipping() -> None:
    policy = _FakeInferencePolicy(torch.zeros((1, 16, 8), dtype=torch.float32))
    runtime = PickCubeActInferenceRuntime(
        policy=policy,
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        experiment=_experiment(),
        action_lower=np.full(8, -1.0),
        action_upper=np.full(8, 1.0),
    )
    runtime.reset()
    chunk = runtime.predict_action_chunk(
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros(18, dtype=np.float32),
    )
    assert policy.reset_count == 1
    assert chunk.shape == (16, 8)
    assert chunk.dtype == np.dtype(np.float64)
    policy.chunk[0, 0, 0] = 1.01
    with pytest.raises(PickCubeActRuntimeError, match="clipping is prohibited"):
        runtime.predict_action_chunk(
            np.zeros((224, 224, 3), dtype=np.uint8),
            np.zeros(18, dtype=np.float32),
        )


def test_checkpoint_is_atomically_completed_with_hash_inventory(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter])
    checkpoint = save_checkpoint(
        run_root=tmp_path,
        step=3,
        examples_processed=12,
        training_identity={"training_identity_digest": _DIGEST},
        policy=_SavedComponent("policy"),
        preprocessor=_SavedComponent("preprocessor"),
        postprocessor=_SavedComponent("postprocessor"),
        optimizer=optimizer,
        metric={"step": 3, "total_loss": 1.0},
        training_control={
            "best_validation_loss": 1.0,
            "best_validation_step": 3,
            "evaluations_without_improvement": 0,
        },
    )
    assert checkpoint.name == "step-00000003"
    assert json.loads((checkpoint / "complete.json").read_text())["complete"] is True
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    assert manifest["global_step"] == 3
    assert len(manifest["artifacts"]) >= 6
    state = json.loads(
        (checkpoint / "training_state" / "training_state.json").read_text()
    )
    assert state["training_control"]["best_validation_step"] == 3
