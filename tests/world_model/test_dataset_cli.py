"""Regression coverage for the WM-v0 frozen-feature dataset CLI."""

import subprocess
import sys
from pathlib import Path

from latentguard.visual_training.config import load_backbone_config


def test_accepted_backbone_configuration_is_explicitly_loadable() -> None:
    """Keep the dataset CLI bound to the accepted complete M4B contract."""

    config = load_backbone_config(
        Path("configs/training/m4b/backbone-resnet18-imagenet1k-v1.json")
    )

    assert config.family == "torchvision_resnet18"
    assert config.weight_enum == "ResNet18_Weights.IMAGENET1K_V1"
    assert config.expected_input_resolution == 224
    assert config.output_feature_dimension == 512


def test_d1_collection_cli_exposes_required_resume_and_filter_controls() -> None:
    """Keep every requested bounded-collection control on the public CLI."""

    result = subprocess.run(
        [sys.executable, "scripts/collect_wm_v0.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for flag in (
        "--resume",
        "--max-samples",
        "--max-anchors",
        "--seed",
        "--policy-source",
        "--scene-group",
        "--episode-range",
        "--dry-run",
        "--validate-only",
    ):
        assert flag in result.stdout
