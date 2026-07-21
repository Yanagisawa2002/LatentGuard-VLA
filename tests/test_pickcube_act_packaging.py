"""Native PickCube ACT PolicyPackage and registry tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from latentguard.policies.act.packaging import (
    PickCubeActPackagingError,
    build_pickcube_native_policy_registry,
)
from latentguard.policies.policy_package import PolicyPackageError


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _checkpoint(root: Path, identity: dict[str, object]) -> Path:
    checkpoint = root / "checkpoint"
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir(parents=True)
    files = (
        "model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    artifacts = []
    for index, name in enumerate(files):
        path = pretrained / name
        path.write_bytes(f"asset-{index}".encode())
        artifacts.append(
            {
                "path": f"pretrained_model/{name}",
                "sha256": _digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "artifacts": artifacts,
            "schema_version": "pickcube-native-act-checkpoint-v1",
            "step": 1000,
            "training_identity": identity,
        },
    )
    _write_json(checkpoint / "complete.json", {"complete": True})
    return checkpoint


def _inputs(tmp_path: Path) -> dict[str, object]:
    identity = {"dataset_digest": "sha256:" + "1" * 64}
    checkpoint = _checkpoint(tmp_path, identity)
    checkpoint_digest = _digest(checkpoint / "checkpoint_manifest.json")
    contract = {
        "action": {"dimension": 8},
        "contract_digest": "sha256:" + "2" * 64,
        "environment": {"environment_id": "PickCube-v1"},
        "observation": {"state_dimension": 18},
        "schema_version": "pickcube-act-contract-v1",
    }
    evaluation = {
        "action_contract_violation_count": 0,
        "checkpoint_digest": checkpoint_digest,
        "classification": "PRIMARY_ACCEPTED",
        "contract_digest": contract["contract_digest"],
        "episode_count": 100,
        "evaluation_kind": "final",
        "fixed_seed_reproduction": {"reproducible": True},
        "peak_gpu_memory_allocated_bytes": 100,
        "policy_query_latency_p95_seconds": 0.01,
        "simulator_error_count": 0,
        "source_commit": "a" * 40,
        "success_count": 80,
        "success_rate": 0.8,
        "success_rate_95_percent_wilson": [0.71, 0.86],
        "termination_taxonomy": {"success": 80, "timeout": 20},
        "workspace_violation_count": 0,
    }
    normalization = tmp_path / "normalization.json"
    _write_json(normalization, {"action": {"mean": [0.0] * 8}})
    return {
        "checkpoint": checkpoint,
        "contract": contract,
        "evaluation": evaluation,
        "identity": identity,
        "normalization": normalization,
    }


def _build(
    tmp_path: Path,
    inputs: dict[str, object],
    *,
    name: str,
    smoke: dict[str, object] | None = None,
):
    return build_pickcube_native_policy_registry(
        checkpoint=inputs["checkpoint"],  # type: ignore[arg-type]
        expected_training_identity=inputs["identity"],  # type: ignore[arg-type]
        resolved_training_config=inputs.get(
            "resolved_training_config", {"model": {"chunk_size": 16}}
        ),  # type: ignore[arg-type]
        contract=inputs["contract"],  # type: ignore[arg-type]
        final_evaluation=inputs["evaluation"],  # type: ignore[arg-type]
        normalization_stats_path=inputs["normalization"],  # type: ignore[arg-type]
        package_root=tmp_path / name,
        registry_output=tmp_path / f"{name}-registry.json",
        policy_id="pickcube-native-act-best-validation-v1",
        source_commit="a" * 40,
        builder_source={"branch": "codex/test", "commit": "b" * 40},
        smoke_audit=smoke,
    )


def test_package_requires_controlled_smoke_before_registry_acceptance(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    provisional = _build(tmp_path, inputs, name="provisional")
    assert provisional.result == "RESULT_B"
    assert provisional.accepted_count == 0
    package = provisional.entries[0].package
    assert package is not None
    runtime_digest = package.acceptance_evidence["runtime_asset_digest"]
    final = _build(
        tmp_path,
        inputs,
        name="final",
        smoke={
            "fixed_seed_reproducible": True,
            "passed": True,
            "policy_id": package.policy_id,
            "runtime_asset_digest": runtime_digest,
            "schema_version": "pickcube-native-act-package-smoke-v1",
        },
    )
    assert final.result == "RESULT_A"
    assert final.accepted_count == 1
    accepted = final.accepted_package(package.policy_id, tmp_path / "final")
    assert accepted.verify_artifacts(tmp_path / "final").verified_artifact_count > 10
    state = (
        tmp_path
        / "final"
        / "pretrained_model"
        / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    state.write_bytes(b"tampered")
    with pytest.raises(PolicyPackageError, match="content digest mismatch"):
        accepted.verify_artifacts(tmp_path / "final")


def test_package_rejects_non_final_or_weak_evidence(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    evaluation = dict(inputs["evaluation"])  # type: ignore[arg-type]
    evaluation["classification"] = "COMPATIBLE_BUT_WEAK"
    inputs["evaluation"] = evaluation
    with pytest.raises(PickCubeActPackagingError, match="not package-eligible"):
        _build(tmp_path, inputs, name="weak")


def test_bounded_package_requires_and_hashes_action_transform(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    identity = {
        "dataset_digest": "sha256:" + "1" * 64,
        "experiment": {"schema_version": "pickcube-native-act-bounded-experiment-v1"},
    }
    checkpoint = inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    transform = checkpoint / "pretrained_model" / "action_transform.json"
    _write_json(transform, {"schema_version": "bounded-action-transform-v1"})
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "pickcube_act_bounded_v1"
    manifest["training_identity"] = identity
    manifest["artifacts"].append(
        {
            "path": "pretrained_model/action_transform.json",
            "sha256": _digest(transform),
            "size_bytes": transform.stat().st_size,
        }
    )
    _write_json(manifest_path, manifest)
    evaluation = dict(inputs["evaluation"])  # type: ignore[arg-type]
    evaluation["checkpoint_digest"] = _digest(manifest_path)
    inputs.update(
        {
            "evaluation": evaluation,
            "identity": identity,
            "resolved_training_config": {
                "model": {"chunk_size": 16},
                "schema_version": "pickcube-native-act-bounded-experiment-v1",
            },
        }
    )
    registry = _build(tmp_path, inputs, name="bounded")
    package = registry.entries[0].package
    assert package is not None
    additional = package.acceptance_evidence["additional_artifacts"]
    assert isinstance(additional, dict)
    assert additional["action_transform"]["path"] == (
        "pretrained_model/action_transform.json"
    )
    package.verify_artifacts(tmp_path / "bounded")
