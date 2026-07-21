"""CPU-only tests for the native ACT experiment contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentguard.policies.act.types import (
    PICKCUBE_ACT_BOUNDED_CHECKPOINT_SCHEMA,
    PickCubeActActionParameterizationConfig,
    PickCubeActConfigurationError,
    PickCubeActExperimentConfig,
    PickCubeActModelConfig,
    build_training_identity,
    validate_bounded_final_authorization,
)

_ROOT = Path(__file__).parents[1]


def test_checked_in_native_act_config_round_trips() -> None:
    value = json.loads(
        (_ROOT / "configs" / "pickcube_act" / "train.yaml").read_text(encoding="utf-8")
    )
    experiment = PickCubeActExperimentConfig.from_mapping(value)
    assert experiment.model.image_shape_chw == (3, 224, 224)
    assert experiment.model.chunk_size == 16
    assert experiment.model.n_action_steps == 4
    assert (
        PickCubeActExperimentConfig.from_mapping(experiment.to_mapping()) == experiment
    )


def test_checked_in_bounded_act_config_is_exact_and_round_trips() -> None:
    value = json.loads(
        (_ROOT / "configs" / "pickcube_act_bounded" / "train.yaml").read_text(
            encoding="utf-8"
        )
    )
    experiment = PickCubeActExperimentConfig.from_mapping(value)
    assert experiment.bounded is True
    assert (
        experiment.action_parameterization == PickCubeActActionParameterizationConfig()
    )
    assert (
        experiment.action_parameterization.checkpoint_schema_version
        == PICKCUBE_ACT_BOUNDED_CHECKPOINT_SCHEMA
    )
    assert (
        PickCubeActExperimentConfig.from_mapping(experiment.to_mapping()) == experiment
    )


def test_bounded_training_identity_requires_transform_identity() -> None:
    experiment = PickCubeActExperimentConfig.from_mapping(
        json.loads(
            (_ROOT / "configs" / "pickcube_act_bounded" / "train.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    with pytest.raises(
        PickCubeActConfigurationError, match="requires action transform"
    ):
        build_training_identity(
            experiment=experiment,
            dataset_digest="sha256:" + "a" * 64,
            normalization_digest="sha256:" + "b" * 64,
            contract_digest="sha256:" + "c" * 64,
            source_commit="d" * 40,
        )


def test_native_act_rejects_pretrained_network_dependency() -> None:
    with pytest.raises(PickCubeActConfigurationError, match="downloads"):
        PickCubeActModelConfig(pretrained_backbone_weights="ResNet18_Weights.DEFAULT")


def test_training_identity_binds_dataset_normalization_and_commit() -> None:
    identity = build_training_identity(
        experiment=PickCubeActExperimentConfig(
            model=PickCubeActModelConfig(),
            optimization=PickCubeActExperimentConfig.from_mapping(
                json.loads(
                    (_ROOT / "configs" / "pickcube_act" / "train.yaml").read_text(
                        encoding="utf-8"
                    )
                )
            ).optimization,
        ),
        dataset_digest="sha256:" + "a" * 64,
        normalization_digest="sha256:" + "b" * 64,
        contract_digest="sha256:" + "c" * 64,
        source_commit="d" * 40,
    )
    assert str(identity["training_identity_digest"]).startswith("sha256:")


def test_bounded_final_requires_exact_promoted_checkpoint() -> None:
    report = {
        "development_gate_passed": True,
        "promoted_checkpoint_id": "step-00020000",
        "schema_version": "pickcube-act-bounded-development-report-v1",
    }
    validate_bounded_final_authorization(report, "step-00020000")
    with pytest.raises(PickCubeActConfigurationError, match="authorization"):
        validate_bounded_final_authorization(report, "step-00018000")
    rejected = dict(report)
    rejected["development_gate_passed"] = False
    with pytest.raises(PickCubeActConfigurationError, match="authorization"):
        validate_bounded_final_authorization(rejected, "step-00020000")
