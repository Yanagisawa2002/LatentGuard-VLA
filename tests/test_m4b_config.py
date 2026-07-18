"""Strict CPU-only tests for the frozen M4B configuration surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentguard.visual_training.config import (
    M4B_MODEL_IDS,
    VisualTrainingConfigurationError,
    load_backbone_config,
    load_external_evaluation_config,
    load_seed_policy,
    load_visual_model_config,
    load_visual_training_config,
)

_ROOT = Path("configs/training/m4b")


def test_all_reviewed_m4b_configurations_load() -> None:
    """Every committed M4B configuration remains strict and content-bound."""
    backbone = load_backbone_config(_ROOT / "backbone-resnet18-imagenet1k-v1.json")
    training = load_visual_training_config(_ROOT / "train-default.json")
    policy = load_seed_policy(_ROOT / "seed-policy.json")
    external = load_external_evaluation_config(_ROOT / "external-evaluation.json")
    model_paths = (
        "random-resnet18-multiview-action.json",
        "frozen-resnet18-singleview-action.json",
        "frozen-resnet18-multiview-action.json",
        "frozen-resnet18-multiview-distilled.json",
    )
    models = tuple(load_visual_model_config(_ROOT / item) for item in model_paths)

    assert tuple(item.model_id for item in models) == M4B_MODEL_IDS
    assert policy.screen_seeds == (0,)
    assert policy.promoted_seeds == (0, 1, 2)
    assert training.max_epochs == 40
    assert backbone.output_feature_dimension == 512
    assert external.bootstrap_samples == 2000
    assert external.outcomes_available_during_selection is False
    assert len({item.content_digest for item in models}) == 4


def test_model_config_rejects_unknown_fields(tmp_path: Path) -> None:
    """Architecture configuration cannot silently accept a new behavior knob."""
    source = _ROOT / "frozen-resnet18-multiview-action.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["camera_id"] = "forbidden-reporting-input"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VisualTrainingConfigurationError, match="unexpected"):
        load_visual_model_config(path)


def test_seed_policy_rejects_five_seed_drift(tmp_path: Path) -> None:
    """Seeds three and four cannot enter M4B through configuration drift."""
    payload = json.loads((_ROOT / "seed-policy.json").read_text(encoding="utf-8"))
    payload["promoted_seeds"] = [0, 1, 2, 3, 4]
    path = tmp_path / "seed-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VisualTrainingConfigurationError, match="seed sets changed"):
        load_seed_policy(path)
