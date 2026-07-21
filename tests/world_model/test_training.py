"""Bounded CPU feature-cache, checkpoint, and resume smoke tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from latentguard.world_model.data_schema import DatasetSplit
from latentguard.world_model.features import write_feature_cache
from latentguard.world_model.losses import LossWeights
from latentguard.world_model.model import WorldModelConfig
from latentguard.world_model.training import TrainingRunConfig, train_world_model

from .conftest import sample


class _Encoder:
    identity = "sha256:" + "e" * 64
    latent_dimension = 6

    def encode(self, observations: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        values = observations.astype(np.float32).mean(axis=(2, 3, 4))
        return np.repeat(values[..., None], self.latent_dimension, axis=-1)


def test_training_checkpoint_and_resume(tmp_path: Path) -> None:
    """One-step infrastructure smoke resumes without repeating completed work."""

    values = (
        ("a" * 64, sample(group="train-1", split=DatasetSplit.TRAIN)),
        (
            "b" * 64,
            replace(
                sample(group="train-2", split=DatasetSplit.TRAIN),
                candidate_id="candidate-2",
            ),
        ),
        ("c" * 64, sample(group="val-1", split=DatasetSplit.VALIDATION)),
        (
            "d" * 64,
            replace(
                sample(group="val-2", split=DatasetSplit.VALIDATION),
                candidate_id="candidate-2",
            ),
        ),
    )
    feature_directory = tmp_path / "features"
    write_feature_cache(feature_directory, values, _Encoder())
    model = WorldModelConfig(
        latent_dimension=6,
        proprio_dimension=5,
        action_dimension=3,
        action_horizon=4,
        prediction_horizon=2,
        view_count=2,
        event_count=2,
        model_dimension=8,
        attention_heads=2,
        transformer_layers=1,
        dropout=0.0,
    )
    run = TrainingRunConfig(
        seed=0,
        model="wm_v0",
        batch_size=2,
        learning_rate=1e-3,
        maximum_steps=2,
        evaluation_interval=1,
        checkpoint_interval=1,
        output_directory=str(tmp_path / "run"),
        loss_weights=LossWeights(),
    )
    first = train_world_model(
        model_config=model,
        run_config=run,
        feature_directory=feature_directory,
        max_steps=1,
        device_name="cpu",
    )
    assert first["completed_steps"] == 1
    second = train_world_model(
        model_config=model,
        run_config=run,
        feature_directory=feature_directory,
        max_steps=2,
        resume=tmp_path / "run" / "checkpoint-1.pt",
        device_name="cpu",
    )
    assert second["completed_steps"] == 2
