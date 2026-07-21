"""Build a complete content-bound PolicyPackage for native PickCube ACT."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.runtime import validate_checkpoint_artifacts
from latentguard.policies.policy_package import PolicyPackage
from latentguard.policies.policy_registry import (
    PolicyCompatibilityStatus,
    PolicyRegistry,
    PolicyRegistryEntry,
)
from latentguard.replay.identity import canonical_json_bytes, canonical_json_value

_PRETRAINED_ASSETS = {
    "checkpoint": "pretrained_model/model.safetensors",
    "model_config": "pretrained_model/config.json",
    "normalization": (
        "pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors"
    ),
    "postprocessor": "pretrained_model/policy_postprocessor.json",
    "postprocessor_state": (
        "pretrained_model/"
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    ),
    "preprocessor": "pretrained_model/policy_preprocessor.json",
}


def _pretrained_assets(training_config: Mapping[str, object]) -> dict[str, str]:
    assets = dict(_PRETRAINED_ASSETS)
    if (
        training_config.get("schema_version")
        == "pickcube-native-act-bounded-experiment-v1"
    ):
        assets["action_transform"] = "pretrained_model/action_transform.json"
    return assets


class PickCubeActPackagingError(ValueError):
    """Raised when native ACT package inputs or evidence are incomplete."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActPackagingError(f"{context}: {reason}")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    canonical = canonical_json_value(value, context=context)
    if not isinstance(canonical, dict):
        _fail(context, "expected JSON object")
    return cast(dict[str, object], canonical)


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected non-empty canonical text")
    return value


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _copy_regular(source: Path, target: Path, context: str) -> None:
    origin = Path(source).absolute()
    if not origin.is_file() or origin.is_symlink():
        _fail(context, "expected regular unlinked source file")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)


def _artifact_records(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = PurePosixPath(*path.relative_to(root).parts).as_posix()
        records[relative] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def _runtime_asset_digest(
    records: Mapping[str, Mapping[str, object]],
    assets: Mapping[str, str],
) -> str:
    inventory = {
        name: {
            "path": relative,
            "sha256": records[relative]["sha256"],
        }
        for name, relative in sorted(assets.items())
    }
    encoded = canonical_json_bytes(inventory, context="PickCubeActRuntimeAssetsV1")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _number(value: Mapping[str, object], field: str) -> float:
    raw = value.get(field)
    if type(raw) not in (int, float):
        _fail("final evaluation", f"{field} must be numeric")
    return float(cast(int | float, raw))


def build_pickcube_native_policy_registry(
    *,
    checkpoint: Path,
    expected_training_identity: Mapping[str, object],
    resolved_training_config: Mapping[str, object],
    contract: Mapping[str, object],
    final_evaluation: Mapping[str, object],
    normalization_stats_path: Path,
    package_root: Path,
    registry_output: Path,
    policy_id: str,
    source_commit: str,
    builder_source: Mapping[str, object],
    smoke_audit: Mapping[str, object] | None = None,
) -> PolicyRegistry:
    """Materialize one exact runtime package and its versioned D2 registry."""
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        _fail("source_commit", "expected full lowercase Git SHA")
    identifier = _text(policy_id, "policy_id")
    root = Path(package_root).absolute()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        _fail("package_root", "refusing to overwrite a non-empty output")
    root.mkdir(parents=True, exist_ok=True)

    checkpoint_root = Path(checkpoint).absolute()
    checkpoint_manifest = validate_checkpoint_artifacts(
        checkpoint_root, expected_training_identity
    )
    source_manifest_path = checkpoint_root / "checkpoint_manifest.json"
    checkpoint_digest = _sha256(source_manifest_path)

    evaluation = _mapping(final_evaluation, "final evaluation")
    classification = evaluation.get("classification")
    if classification not in {"PRIMARY_ACCEPTED", "SECONDARY_COMPATIBLE"}:
        _fail("final evaluation", "checkpoint is not package-eligible")
    if (
        evaluation.get("evaluation_kind") != "final"
        or evaluation.get("checkpoint_digest") != checkpoint_digest
        or evaluation.get("source_commit") != source_commit
        or type(evaluation.get("episode_count")) is not int
        or cast(int, evaluation["episode_count"]) < 100
    ):
        _fail("final evaluation", "final checkpoint identity or episode gate differs")
    reproduction = evaluation.get("fixed_seed_reproduction")
    if (
        not isinstance(reproduction, Mapping)
        or reproduction.get("reproducible") is not True
    ):
        _fail("final evaluation", "fixed-seed reproduction did not pass")
    if any(
        evaluation.get(field) != 0
        for field in (
            "action_contract_violation_count",
            "simulator_error_count",
            "workspace_violation_count",
        )
    ):
        _fail("final evaluation", "closed-loop safety or contract gate did not pass")

    native_contract = _mapping(contract, "PickCube ACT contract")
    if native_contract.get("schema_version") != "pickcube-act-contract-v1":
        _fail("PickCube ACT contract", "schema mismatch")
    observation_spec = _mapping(native_contract.get("observation"), "observation spec")
    action_spec = _mapping(native_contract.get("action"), "action spec")
    environment_spec = _mapping(native_contract.get("environment"), "environment spec")
    if evaluation.get("contract_digest") != native_contract.get("contract_digest"):
        _fail("final evaluation", "contract digest differs")

    training_config = _mapping(resolved_training_config, "resolved training config")
    pretrained_assets = _pretrained_assets(training_config)
    model_config = _mapping(training_config.get("model"), "training model config")
    action_horizon = model_config.get("chunk_size")
    if type(action_horizon) is not int or action_horizon < 1:
        _fail("training model config", "chunk_size is invalid")

    for relative in pretrained_assets.values():
        source = checkpoint_root.joinpath(*PurePosixPath(relative).parts)
        target = root.joinpath(*PurePosixPath(relative).parts)
        _copy_regular(source, target, f"checkpoint asset {relative}")

    write_atomic_json(root / "contracts" / "observation.json", observation_spec)
    write_atomic_json(root / "contracts" / "action.json", action_spec)
    write_atomic_json(root / "contracts" / "environment.json", environment_spec)
    write_atomic_json(root / "resolved_training_config.json", training_config)
    write_atomic_json(root / "evaluation_summary.json", evaluation)
    write_atomic_json(root / "source_checkpoint_manifest.json", checkpoint_manifest)
    write_atomic_json(
        root / "source_commit.json",
        {
            "builder_source": _mapping(builder_source, "builder source"),
            "checkpoint_source_commit": source_commit,
            "schema_version": "pickcube-native-act-source-binding-v1",
        },
    )
    _copy_regular(
        normalization_stats_path,
        root / "dataset_normalization_stats.json",
        "dataset normalization statistics",
    )
    if smoke_audit is not None:
        write_atomic_json(root / "package_smoke_audit.json", smoke_audit)

    records = _artifact_records(root)
    runtime_digest = _runtime_asset_digest(records, pretrained_assets)
    if smoke_audit is not None:
        smoke = _mapping(smoke_audit, "package smoke audit")
        if (
            smoke.get("schema_version") != "pickcube-native-act-package-smoke-v1"
            or smoke.get("passed") is not True
            or smoke.get("policy_id") != identifier
            or smoke.get("runtime_asset_digest") != runtime_digest
            or smoke.get("fixed_seed_reproducible") is not True
        ):
            _fail("package smoke audit", "controlled runtime smoke did not pass")

    file_hashes = {
        "artifacts": records,
        "runtime_asset_digest": runtime_digest,
        "schema_version": "pickcube-native-act-package-files-v1",
    }
    write_atomic_json(root / "file_hashes.json", file_hashes)
    records = _artifact_records(root)

    main_names = {
        "checkpoint",
        "model_config",
        "normalization",
        "postprocessor",
        "preprocessor",
    }
    additional_artifacts: dict[str, object] = {}
    for relative, record in sorted(records.items()):
        matching = [
            name for name, path in pretrained_assets.items() if path == relative
        ]
        if matching and matching[0] in main_names:
            continue
        name = (
            matching[0]
            if matching
            else relative.replace("/", "_").replace(".", "_").replace("-", "_")
        )
        additional_artifacts[name] = {
            "path": relative,
            "sha256": record["sha256"],
        }

    package = PolicyPackage(
        policy_id=identifier,
        policy_family="ACT",
        source_commit=source_commit,
        checkpoint_path=pretrained_assets["checkpoint"],
        checkpoint_sha256=cast(str, records[pretrained_assets["checkpoint"]]["sha256"]),
        model_config_path=pretrained_assets["model_config"],
        model_config_sha256=cast(
            str, records[pretrained_assets["model_config"]]["sha256"]
        ),
        preprocessor_path=pretrained_assets["preprocessor"],
        preprocessor_sha256=cast(
            str, records[pretrained_assets["preprocessor"]]["sha256"]
        ),
        postprocessor_path=pretrained_assets["postprocessor"],
        postprocessor_sha256=cast(
            str, records[pretrained_assets["postprocessor"]]["sha256"]
        ),
        normalization_path=pretrained_assets["normalization"],
        normalization_sha256=cast(
            str, records[pretrained_assets["normalization"]]["sha256"]
        ),
        observation_spec=observation_spec,
        action_spec=action_spec,
        environment_spec=environment_spec,
        action_horizon=action_horizon,
        deterministic=True,
        acceptance_evidence={
            "additional_artifacts": additional_artifacts,
            "checkpoint_classification": classification,
            "closed_loop_episode_count": evaluation["episode_count"],
            "contract_digest": native_contract["contract_digest"],
            "runtime_asset_digest": runtime_digest,
            "source_checkpoint_digest": checkpoint_digest,
            "success_rate": _number(evaluation, "success_rate"),
        },
    )
    package.verify_artifacts(root)
    write_atomic_json(root / "policy_package.json", package.to_mapping())

    accepted = smoke_audit is not None
    entry = PolicyRegistryEntry(
        policy_id=identifier,
        policy_name="PickCube-v1 native ACT best-validation checkpoint",
        policy_family="ACT",
        source_repository="Yanagisawa2002/LatentGuard-VLA",
        compatibility_status=(
            PolicyCompatibilityStatus.ACCEPTED_COMPATIBLE
            if accepted
            else PolicyCompatibilityStatus.UNVERIFIED
        ),
        missing_assets=() if accepted else ("controlled_package_smoke",),
        compatibility_reasons=()
        if accepted
        else ("controlled package load and fixed-seed execution smoke is pending",),
        known_evaluation_result={
            "classification": classification,
            "episode_count": evaluation["episode_count"],
            "failure_taxonomy": evaluation["termination_taxonomy"],
            "peak_gpu_memory_allocated_bytes": evaluation[
                "peak_gpu_memory_allocated_bytes"
            ],
            "policy_query_latency_p95_seconds": evaluation[
                "policy_query_latency_p95_seconds"
            ],
            "success_count": evaluation["success_count"],
            "success_rate": evaluation["success_rate"],
            "success_rate_95_percent_wilson": evaluation[
                "success_rate_95_percent_wilson"
            ],
        },
        audit_record={
            "builder_source": _mapping(builder_source, "builder source"),
            "checkpoint_step": checkpoint_manifest.get("step"),
            "contract_digest": native_contract["contract_digest"],
            "package_digest": package.package_digest,
            "package_role": "primary"
            if classification == "PRIMARY_ACCEPTED"
            else "secondary",
            "runtime_asset_digest": runtime_digest,
            "training_source_commit": source_commit,
        },
        package=package,
    )
    registry = PolicyRegistry(
        entries=(entry,),
        required_observation_spec=observation_spec,
        required_action_spec=action_spec,
        required_environment_spec=environment_spec,
        result="RESULT_A" if accepted else "RESULT_B",
    )
    write_atomic_json(registry_output, registry.to_mapping())
    return registry


__all__ = [
    "PickCubeActPackagingError",
    "build_pickcube_native_policy_registry",
]
