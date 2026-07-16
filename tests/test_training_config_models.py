from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from latentguard.training.config import (
    LossMode,
    ModelType,
    ResolvedModelConfig,
    TrainingConfigurationError,
    load_benchmark_config,
    load_model_config,
    load_training_config,
    model_config_from_mapping,
)
from latentguard.training.manifest import (
    TrainingRunEnvironment,
    TrainingRunIdentity,
    TrainingRunManifest,
    TrainingRunStatus,
    compute_training_run_identifier,
)

_CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "training" / "m3b"
_MODEL_FILES = (
    "state-only-mlp.json",
    "action-only-mlp.json",
    "state-action-mlp.json",
    "state-action-transformer.json",
)
_EXPECTED_PARAMETER_COUNTS = {
    "state-only-mlp.json": 38_913,
    "action-only-mlp.json": 50_433,
    "state-action-mlp.json": 61_249,
    "state-action-transformer.json": 312_897,
}
_EXPECTED_MODEL_DIGESTS = {
    "state-only-mlp.json": (
        "sha256:282f4b4b8007095f59488f9ce15898008b45faa1d3cc38997c95ecdb2424cbb7"
    ),
    "action-only-mlp.json": (
        "sha256:e813a00ec0bbc6dc91b2609759c38f3b7a5b8782a031957bdae25aa903142f2c"
    ),
    "state-action-mlp.json": (
        "sha256:a2dfe19c04f557269d06bac482904d939495afa88a6f90259d3a7db791e35fb9"
    ),
    "state-action-transformer.json": (
        "sha256:1699c1a963b51cf771e7ae0fb70de061b0e87c211aabc44f482125d9be3728bb"
    ),
}


def _torch() -> Any:
    return pytest.importorskip("torch", reason="M3B training extra is not installed")


def test_checked_in_configs_are_strict_and_digest_stable() -> None:
    for name in _MODEL_FILES:
        config = load_model_config(_CONFIG_ROOT / name)
        assert config.state_dimension == 38
        assert config.action_dimension == 8
        assert config.action_horizon == 16
        assert config.content_digest == _EXPECTED_MODEL_DIGESTS[name]

    training = load_training_config(_CONFIG_ROOT / "train-default.json")
    benchmark = load_benchmark_config(_CONFIG_ROOT / "benchmark-five-seed.json")
    assert training.loss_mode is LossMode.CLASS_WEIGHTED_BCE
    assert training.limit_samples is None
    assert training.content_digest == (
        "sha256:a5d9245d33f8d89f1075e8e2b6ba467897ce5bdd20a0feb96169de75a9f996cb"
    )
    assert benchmark.model_types == tuple(ModelType)
    assert benchmark.seeds == (0, 1, 2, 3, 4)
    assert benchmark.content_digest == (
        "sha256:5174cec3390cfd738490100ae1fbf99d5b12d8a0eee8899eaaf52ef2a5580198"
    )


def test_training_cli_overrides_participate_in_configuration_digest() -> None:
    original = load_training_config(_CONFIG_ROOT / "train-default.json")
    resolved = replace(
        original,
        max_steps=50,
        max_epochs=3,
        limit_samples=64,
        num_workers=0,
    )
    assert resolved.content_digest != original.content_digest
    assert resolved.as_mapping()["limit_samples"] == 64
    with pytest.raises(TrainingConfigurationError, match="limit_samples"):
        replace(original, limit_samples=0)
    with pytest.raises(TrainingConfigurationError, match="zero workers"):
        replace(original, num_workers=2)


def test_model_config_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    payload = json.loads(
        (_CONFIG_ROOT / "state-only-mlp.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    payload["candidate_type"] = "forbidden_metadata"
    with pytest.raises(TrainingConfigurationError, match="unexpected candidate_type"):
        model_config_from_mapping(payload)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(TrainingConfigurationError, match="duplicate JSON field"):
        load_model_config(duplicate)


@pytest.mark.parametrize("config_name", _MODEL_FILES)
def test_models_have_fixed_parameter_counts_and_common_forward_contract(
    config_name: str,
) -> None:
    torch = _torch()
    from latentguard.training.models import (
        MAX_TRAINABLE_PARAMETERS,
        build_model,
        resolve_model_config,
    )

    config = load_model_config(_CONFIG_ROOT / config_name)
    model = build_model(config, seed=17)
    resolved = resolve_model_config(model)
    assert resolved.parameter_count == _EXPECTED_PARAMETER_COUNTS[config_name]
    assert resolved.parameter_count < MAX_TRAINABLE_PARAMETERS
    assert list(inspect.signature(model.forward).parameters) == [
        "state",
        "actions",
        "action_mask",
    ]

    state = torch.randn(5, 38)
    actions = torch.randn(5, 16, 8)
    mask = torch.ones(5, 16, dtype=torch.bool)
    logits = model(state, actions, mask)
    assert logits.shape == (5,)
    assert bool(torch.isfinite(logits).all())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


@pytest.mark.parametrize("config_name", _MODEL_FILES)
def test_model_initialization_is_seeded_and_masked_actions_do_not_leak(
    config_name: str,
) -> None:
    torch = _torch()
    from latentguard.training.models import build_model

    config = load_model_config(_CONFIG_ROOT / config_name)
    first = build_model(config, seed=314159)
    second = build_model(config, seed=314159)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
    )

    first.eval()
    state = torch.randn(3, 38)
    actions = torch.randn(3, 16, 8)
    mask = torch.ones(3, 16, dtype=torch.bool)
    mask[:, 12:] = False
    changed = actions.clone()
    changed[:, 12:] = 1_000_000.0
    with torch.no_grad():
        original_logits = first(state, actions, mask)
        changed_logits = first(state, changed, mask)
    assert torch.equal(original_logits, changed_logits)


def test_loss_modes_are_explicit_and_train_weight_is_deterministic() -> None:
    torch = _torch()
    from latentguard.training.losses import (
        build_failure_loss,
        compute_positive_class_weight,
        failure_bce_with_logits,
    )

    targets = torch.tensor([0.0, 0.0, 0.0, 1.0])
    logits = torch.zeros(4)
    assert compute_positive_class_weight(targets) == 3.0
    unweighted = failure_bce_with_logits(logits, targets, mode=LossMode.UNWEIGHTED_BCE)
    weighted = failure_bce_with_logits(
        logits,
        targets,
        mode=LossMode.CLASS_WEIGHTED_BCE,
        positive_class_weight=3.0,
    )
    assert bool(torch.isfinite(unweighted))
    assert bool(torch.isfinite(weighted))
    assert weighted > unweighted
    module = build_failure_loss(LossMode.CLASS_WEIGHTED_BCE, targets)
    assert module.positive_class_weight is not None
    assert module.positive_class_weight.item() == 3.0


def _run_identity() -> TrainingRunIdentity:
    return TrainingRunIdentity(
        dataset_digest="sha256:" + "1" * 64,
        split_digest="sha256:" + "2" * 64,
        preprocessing_digest="sha256:" + "3" * 64,
        model_config_digest="sha256:" + "4" * 64,
        training_config_digest="sha256:" + "5" * 64,
        seed=7,
        code_semantic_version="direct_action_verifier_v1",
    )


def test_run_identity_excludes_host_path_and_runtime_facts() -> None:
    model_configuration = load_model_config(_CONFIG_ROOT / "state-only-mlp.json")
    resolved_model = ResolvedModelConfig(
        architecture=model_configuration,
        parameter_count=_EXPECTED_PARAMETER_COUNTS["state-only-mlp.json"],
    )
    training_configuration = load_training_config(_CONFIG_ROOT / "train-default.json")
    identity = TrainingRunIdentity(
        dataset_digest="sha256:" + "1" * 64,
        split_digest="sha256:" + "2" * 64,
        preprocessing_digest="sha256:" + "3" * 64,
        model_config_digest=resolved_model.content_digest,
        training_config_digest=training_configuration.content_digest,
        seed=7,
        code_semantic_version="direct_action_verifier_v1",
    )
    first_environment = TrainingRunEnvironment(
        hostname="first-host",
        python_version="3.11.9",
        numpy_version="2.1.0",
        pytorch_version="2.6.0",
        cuda_version="12.4",
        gpu_model="RTX 5090",
        gpu_driver_version="570.00",
        cublas_workspace_config=":4096:8",
        package_versions=("latentguard-vla=0.1.0", "numpy=2.1.0", "torch=2.6.0"),
        launch_command=("latentguard", "train-action-verifier", "--output-dir", "/a"),
        deterministic_algorithms=True,
    )
    second_environment = TrainingRunEnvironment(
        hostname="second-host",
        python_version="3.11.10",
        numpy_version="2.2.0",
        pytorch_version="2.7.0",
        cuda_version="12.8",
        gpu_model="different-gpu",
        gpu_driver_version="575.00",
        cublas_workspace_config=":4096:8",
        package_versions=("latentguard-vla=0.1.0", "numpy=2.2.0", "torch=2.7.0"),
        launch_command=("latentguard", "train-action-verifier", "--output-dir", "/b"),
        deterministic_algorithms=False,
        nondeterministic_operations=("documented-op",),
    )
    first = TrainingRunManifest(
        identity=identity,
        acceptance_report_digest="sha256:" + "6" * 64,
        model_configuration=resolved_model,
        training_configuration=training_configuration,
        positive_class_weight=2.0,
        git_sha="a" * 40,
        branch="codex/m3b-direct-verifier",
        environment=first_environment,
        started_at_utc="2026-07-16T00:00:00Z",
        finished_at_utc=None,
        status=TrainingRunStatus.RUNNING,
    )
    second = TrainingRunManifest(
        identity=identity,
        acceptance_report_digest="sha256:" + "6" * 64,
        model_configuration=resolved_model,
        training_configuration=training_configuration,
        positive_class_weight=2.0,
        git_sha="b" * 40,
        branch="another-branch",
        environment=second_environment,
        started_at_utc="2026-07-17T00:00:00Z",
        finished_at_utc="2026-07-17T00:01:00Z",
        status=TrainingRunStatus.COMPLETED,
    )
    assert first.run_id == second.run_id == identity.run_id
    assert "hostname" not in identity.as_mapping()
    assert (
        "output_dir"
        not in inspect.signature(compute_training_run_identifier).parameters
    )


def test_run_identity_changes_for_every_semantic_component() -> None:
    identity = _run_identity()
    changed = compute_training_run_identifier(
        dataset_digest=identity.dataset_digest,
        split_digest=identity.split_digest,
        preprocessing_digest=identity.preprocessing_digest,
        model_config_digest=identity.model_config_digest,
        training_config_digest=identity.training_config_digest,
        seed=identity.seed + 1,
        code_semantic_version=identity.code_semantic_version,
    )
    assert changed != identity.run_id
