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
    validate_checkpoint_artifacts,
)
from latentguard.policies.act.types import (
    PickCubeActExperimentConfig,
    PickCubeActGraspSupervisionConfig,
)
from latentguard.policies.actions import ActionBounds, BoundedActionTransform

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


def test_p02_dataset_adds_only_training_metadata_without_privileged_input(
    tmp_path: Path,
) -> None:
    episode = _episode(2)
    actions = np.zeros_like(episode.actions, dtype=np.float64)
    actions[:, 7] = [1.0, 1.0, -1.0]
    episode = PickCubeDemoEpisode(
        scene_seed=episode.scene_seed,
        rgb=episode.rgb,
        proprioception=episode.proprioception,
        actions=actions,
        phases=episode.phases,
        compatibility_identity=_DIGEST,
        contract_digest=_DIGEST,
        camera_configuration_digest=_DIGEST,
    )
    reference = save_demo_episode(tmp_path, episode)
    dataset = PickCubeActDataset(
        tmp_path,
        (reference,),
        chunk_size=4,
        action_transform=_bounded_transform(),
        grasp_supervision=PickCubeActGraspSupervisionConfig(),
        phase_weights={
            "APPROACH": 1.0,
            "GRIPPER_CLOSING": 1.0,
            "PREGRASP": 1.0,
        },
    )
    sample = dataset[1]
    assert set(sample) == {
        "action",
        "action_is_pad",
        "observation.images.front_oblique",
        "observation.state",
        "p02_gripper_event",
        "p02_phase_weight",
    }
    assert sample["p02_gripper_event"].tolist() == [0, 1, 0, 0]
    assert sample["action_is_pad"].tolist() == [False, False, True, True]
    assert not any("cube" in key or "phase" == key for key in sample)


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


def _bounded_experiment() -> PickCubeActExperimentConfig:
    value = json.loads(Path("configs/pickcube_act_bounded/train.yaml").read_text())
    return PickCubeActExperimentConfig.from_mapping(value)


def _bounded_transform() -> BoundedActionTransform:
    return BoundedActionTransform(
        ActionBounds(
            lower=torch.tensor([-3.0, -2.0, -3.0, -4.0, -3.0, -1.0, -3.0, -1.0]),
            upper=torch.tensor([3.0, 2.0, 3.0, 0.0, 3.0, 4.0, 3.0, 1.0]),
            names=tuple(f"action_{index}" for index in range(8)),
            units=("rad",) * 7 + ("normalized",),
        )
    )


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
    audited = runtime.predict_action_chunk_for_audit(
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros(18, dtype=np.float32),
    )
    assert audited[0, 0] == pytest.approx(1.01)


def test_bounded_runtime_applies_exactly_one_affine_transform() -> None:
    transform = _bounded_transform()
    policy = _FakeInferencePolicy(torch.zeros((1, 16, 8), dtype=torch.float32))
    runtime = PickCubeActInferenceRuntime(
        policy=policy,
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        experiment=_bounded_experiment(),
        action_lower=transform.lower.numpy(),
        action_upper=transform.upper.numpy(),
        action_transform=transform,
    )
    chunk = runtime.predict_action_chunk(
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros(18, dtype=np.float32),
    )
    assert np.allclose(chunk[0], transform.center.numpy())
    policy.chunk.fill_(0.5)
    chunk = runtime.predict_action_chunk(
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros(18, dtype=np.float32),
    )
    assert np.allclose(chunk[0], (transform.center + 0.5 * transform.scale).numpy())


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


@pytest.mark.parametrize(
    "experiment_schema",
    [
        "pickcube-native-act-bounded-experiment-v1",
        "pickcube-native-act-grasp-experiment-v1",
    ],
)
def test_bounded_checkpoint_schema_and_transform_are_required(
    tmp_path: Path, experiment_schema: str
) -> None:
    transform = _bounded_transform()
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter])
    identity = {
        "experiment": {"schema_version": experiment_schema},
        "training_identity_digest": _DIGEST,
    }
    checkpoint = save_checkpoint(
        run_root=tmp_path,
        step=1,
        examples_processed=2,
        training_identity=identity,
        policy=_SavedComponent("policy"),
        preprocessor=_SavedComponent("preprocessor"),
        postprocessor=_SavedComponent("postprocessor"),
        optimizer=optimizer,
        metric={"step": 1, "total_loss": 1.0},
        action_transform=transform,
    )
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    assert manifest["schema_version"] == "pickcube_act_bounded_v1"
    assert (checkpoint / "pretrained_model" / "action_transform.json").is_file()
    validate_checkpoint_artifacts(checkpoint, identity)
    old_identity = {"training_identity_digest": _DIGEST}
    with pytest.raises(PickCubeActRuntimeError, match="completion or resume"):
        validate_checkpoint_artifacts(checkpoint, old_identity)
