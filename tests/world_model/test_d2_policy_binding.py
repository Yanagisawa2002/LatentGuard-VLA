from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from latentguard.policies import (
    PolicyCompatibilityError,
    PolicyCompatibilityStatus,
    PolicyPackage,
    PolicyPackageError,
    PolicyRegistry,
    PolicyRegistryEntry,
    PolicyRegistryError,
)
from latentguard.policies.d2_gate import evaluate_d2_gate


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _package(root: Path) -> PolicyPackage:
    files = {
        "checkpoint.bin": b"checkpoint",
        "model.json": b"model",
        "preprocessor.json": b"preprocessor",
        "postprocessor.json": b"postprocessor",
        "normalization.bin": b"normalization",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    return PolicyPackage(
        policy_id="policy-v1",
        policy_family="fixture",
        source_commit="a" * 40,
        checkpoint_path="checkpoint.bin",
        checkpoint_sha256=_digest(files["checkpoint.bin"]),
        model_config_path="model.json",
        model_config_sha256=_digest(files["model.json"]),
        preprocessor_path="preprocessor.json",
        preprocessor_sha256=_digest(files["preprocessor.json"]),
        postprocessor_path="postprocessor.json",
        postprocessor_sha256=_digest(files["postprocessor.json"]),
        normalization_path="normalization.bin",
        normalization_sha256=_digest(files["normalization.bin"]),
        observation_spec={"state_shape": [9]},
        action_spec={"dimension": 8, "ordering": "joint7_gripper1"},
        environment_spec={"environment_id": "PickCube-v1"},
        action_horizon=16,
        deterministic=True,
        acceptance_evidence={"fixture": "unit-test"},
    )


def _entry(
    package: PolicyPackage,
    status: PolicyCompatibilityStatus = PolicyCompatibilityStatus.ACCEPTED_COMPATIBLE,
) -> PolicyRegistryEntry:
    blockers = (
        () if status is PolicyCompatibilityStatus.ACCEPTED_COMPATIBLE else ("blocked",)
    )
    return PolicyRegistryEntry(
        policy_id=package.policy_id,
        policy_name="fixture policy",
        policy_family=package.policy_family,
        source_repository="fixture-repository",
        compatibility_status=status,
        missing_assets=(),
        compatibility_reasons=blockers,
        known_evaluation_result={"scope": "fixture"},
        audit_record={"source": "fixture"},
        package=package,
    )


def _registry(package: PolicyPackage) -> PolicyRegistry:
    return PolicyRegistry(
        entries=(_entry(package),),
        required_observation_spec=package.observation_spec,
        required_action_spec=package.action_spec,
        required_environment_spec=package.environment_spec,
        result="RESULT_A",
    )


def test_policy_package_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (tmp_path / "checkpoint.bin").write_bytes(b"tampered")

    with pytest.raises(PolicyPackageError, match="content digest mismatch"):
        package.verify_artifacts(tmp_path)


@pytest.mark.parametrize("component", ["preprocessor", "postprocessor"])
def test_policy_package_rejects_missing_processor(
    tmp_path: Path, component: str
) -> None:
    package = _package(tmp_path)
    package = replace(
        package,
        **{
            f"{component}_path": None,
            f"{component}_sha256": None,
        },
    )

    with pytest.raises(PolicyPackageError, match=component):
        package.verify_artifacts(tmp_path)


def test_policy_package_rejects_missing_normalization(tmp_path: Path) -> None:
    package = replace(
        _package(tmp_path), normalization_path=None, normalization_sha256=None
    )

    with pytest.raises(PolicyPackageError, match="normalization"):
        package.verify_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("contract", "replacement"),
    [
        ("observation_spec", {"state_shape": [10]}),
        ("action_spec", {"dimension": 7, "ordering": "unknown"}),
        ("environment_spec", {"environment_id": "OtherTask-v0"}),
    ],
)
def test_policy_package_rejects_exact_contract_mismatch(
    tmp_path: Path, contract: str, replacement: dict[str, object]
) -> None:
    package = _package(tmp_path)
    required = {
        "observation_spec": package.observation_spec,
        "action_spec": package.action_spec,
        "environment_spec": package.environment_spec,
    }
    required[contract] = replacement

    with pytest.raises(PolicyCompatibilityError, match=contract):
        package.assert_compatible(**required)


def test_registry_returns_only_byte_verified_accepted_package(tmp_path: Path) -> None:
    package = _package(tmp_path)
    registry = _registry(package)

    observed = registry.accepted_package(package.policy_id, tmp_path)

    assert observed.package_digest == package.package_digest


def test_registry_rejects_nonaccepted_package(tmp_path: Path) -> None:
    package = _package(tmp_path)
    registry = PolicyRegistry(
        entries=(_entry(package, PolicyCompatibilityStatus.INCOMPATIBLE_ENVIRONMENT),),
        required_observation_spec=package.observation_spec,
        required_action_spec=package.action_spec,
        required_environment_spec=package.environment_spec,
        result="RESULT_B",
    )

    with pytest.raises(PolicyRegistryError, match="INCOMPATIBLE_ENVIRONMENT"):
        registry.accepted_package(package.policy_id, tmp_path)


def _passing_manifest() -> dict[str, object]:
    return {
        "schema_version": "wm-v0-d2-dataset-manifest-v1",
        "accepted_compatible_policy_count": 1,
        "policy_generated_candidate_ratio": 1.0,
        "valid_sample_count": 150,
        "independent_anchor_count": 50,
        "terminal_success_count": 25,
        "terminal_failure_count": 25,
        "terminalized_ratio": 0.70,
        "meaningfully_distinct_anchor_ratio": 0.70,
        "mixed_outcome_anchor_count": 15,
        "future_completeness": 0.99,
        "restoration_mismatch_count": 0,
        "split_leakage_count": 0,
        "duplicate_identity_count": 0,
        "policy_metadata_completeness": 1.0,
    }


def test_d2_gate_accepts_only_complete_threshold_boundary() -> None:
    result = evaluate_d2_gate(_passing_manifest())

    assert result.authorized
    assert result.failed_checks == ()


def test_synthetic_candidates_cannot_satisfy_policy_generated_gate() -> None:
    manifest = _passing_manifest()
    manifest["policy_generated_candidate_ratio"] = 0.0

    result = evaluate_d2_gate(manifest)

    assert not result.authorized
    assert (
        "policy_generated_candidate_ratio_exactly_100_percent" in result.failed_checks
    )
