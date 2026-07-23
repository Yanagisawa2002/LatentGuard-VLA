from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import latentguard
from latentguard.adapters.vla_jepa.constants import (
    CHECKPOINT_REVISION,
    EXPECTED_ACTION_HORIZON,
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
    LIBERO_ASSET_REVISION,
)
from latentguard.adapters.vla_jepa.contracts import (
    ContractError,
    validate_action_chunk,
    validate_policy_batch,
)
from latentguard.adapters.vla_jepa.identity import (
    FileIdentity,
    IdentityValidationError,
    sha256_file,
    validate_file_identities,
    validate_snapshot_manifest,
)
from latentguard.adapters.vla_jepa.model_loader import load_local_vla_jepa
from latentguard.adapters.vla_jepa.policy_adapter import VLAJepaAdapter
from latentguard.adapters.vla_jepa.world_model_adapter import (
    ExternalCandidateUnsupportedError,
    reject_external_action_candidates,
)
from latentguard.rollout.lerobot_rollout_adapter import (
    LeRobotRolloutIdentity,
    LeRobotRolloutResult,
    build_rollout_manifest_entry,
)

ROOT = Path(__file__).resolve().parents[1]


class FakePolicy:
    def predict_action_chunk(self, batch: dict[str, object]) -> np.ndarray:
        batch_size = int(np.asarray(batch["observation.state"]).shape[0])
        return np.zeros((batch_size, EXPECTED_ACTION_HORIZON, 7), dtype=np.float32)


def test_import_resolves_to_this_repository() -> None:
    module_path = Path(latentguard.__file__).resolve()
    assert module_path.is_relative_to((ROOT / "src" / "latentguard").resolve())


def test_frozen_exact_revisions() -> None:
    assert LEROBOT_VERSION == "0.6.0"
    assert LEROBOT_COMMIT == "30da8e687a6dfc617fcd94afc367ac7071c376ce"
    assert CHECKPOINT_REVISION == "735d9f692981e286ade093b5046627eda876e5d0"


def test_stack_config_binds_exact_revisions() -> None:
    config = json.loads(
        (ROOT / "configs" / "lg_r0" / "stack.json").read_text(encoding="utf-8")
    )
    assert config["lerobot"]["version"] == LEROBOT_VERSION
    assert config["lerobot"]["commit"] == LEROBOT_COMMIT
    assert config["checkpoint"]["revision"] == CHECKPOINT_REVISION
    assert config["libero"]["assets_revision"] == LIBERO_ASSET_REVISION
    assert config["execution"]["optimizer_steps_allowed"] == 0
    assert config["execution"]["backward_calls_allowed"] == 0


def test_tracked_paths_exclude_langmani_v2() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not any("langmani_v2" in path.lower() for path in tracked)


def test_lg_r0_scope_has_no_robolab_or_ros2_runtime() -> None:
    paths = [
        path.as_posix().lower()
        for base in (
            ROOT / "src" / "latentguard" / "adapters" / "vla_jepa",
            ROOT / "configs" / "lg_r0",
        )
        for path in base.rglob("*")
        if path.is_file()
    ]
    assert not any("robolab" in path or "ros2" in path for path in paths)


def test_file_identity_passes_and_fails_closed(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"frozen")
    identity = FileIdentity(
        relative_path="asset.bin",
        bytes=asset.stat().st_size,
        sha256=sha256_file(asset),
    )
    assert validate_file_identities(tmp_path, [identity]) == (asset.resolve(),)
    asset.write_bytes(b"changed")
    with pytest.raises(IdentityValidationError, match="size mismatch"):
        validate_file_identities(tmp_path, [identity])


def test_file_identity_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"x")
    identity = FileIdentity(
        relative_path="../outside.bin",
        bytes=1,
        sha256=sha256_file(outside),
    )
    with pytest.raises(IdentityValidationError, match="escapes"):
        validate_file_identities(tmp_path, [identity])


def test_snapshot_manifest_validates_all_bound_repositories(tmp_path: Path) -> None:
    policy = tmp_path / "policy"
    qwen = tmp_path / "qwen"
    policy.mkdir()
    qwen.mkdir()
    (policy / "config.json").write_bytes(b"policy")
    (qwen / "config.json").write_bytes(b"qwen")
    manifest = {
        "schema_version": "latentguard.lg_r0.base_model_manifest.v1",
        "status": "pass",
        "assets": [
            {
                "repository_or_model_id": "policy",
                "relative_path": "config.json",
                "bytes": 6,
                "sha256": sha256_file(policy / "config.json"),
            },
            {
                "repository_or_model_id": "qwen",
                "relative_path": "config.json",
                "bytes": 4,
                "sha256": sha256_file(qwen / "config.json"),
            },
        ],
    }
    validated = validate_snapshot_manifest(
        manifest,
        {"policy": policy, "qwen": qwen},
    )
    assert len(validated) == 2


def test_snapshot_manifest_rejects_unbound_repository(tmp_path: Path) -> None:
    root = tmp_path / "policy"
    root.mkdir()
    (root / "config.json").write_bytes(b"policy")
    manifest = {
        "schema_version": "latentguard.lg_r0.base_model_manifest.v1",
        "status": "pass",
        "assets": [
            {
                "repository_or_model_id": "other",
                "relative_path": "config.json",
                "bytes": 6,
                "sha256": sha256_file(root / "config.json"),
            }
        ],
    }
    with pytest.raises(IdentityValidationError, match="unbound repository"):
        validate_snapshot_manifest(manifest, {"policy": root})


def test_model_loader_requires_an_accepted_asset_manifest(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("policy", "qwen", "vjepa")]
    for root in roots:
        root.mkdir()
    with pytest.raises(IdentityValidationError, match="not accepted"):
        load_local_vla_jepa(
            *roots,
            asset_manifest={
                "schema_version": "latentguard.lg_r0.base_model_manifest.v1",
                "status": "fail",
                "assets": [],
            },
            device="cpu",
        )


def test_policy_adapter_preserves_native_tensor_and_explicit_mask() -> None:
    policy = FakePolicy()
    adapter = VLAJepaAdapter(policy)
    batch = {"observation.state": np.zeros((2, 8), dtype=np.float32)}
    native = policy.predict_action_chunk(batch)
    output = adapter.predict_action(batch)
    assert np.array_equal(output.action_chunk, native)
    assert output.action_chunk.shape == (2, 7, 7)
    assert output.action_mask is None
    assert output.policy_metadata["output_modified"] is False
    assert output.policy_metadata["optimizer_steps"] == 0
    assert output.policy_metadata["backward_calls"] == 0


def test_action_contract_shape_finite_and_mask_semantics() -> None:
    assert validate_action_chunk(np.zeros((2, 7, 7), dtype=np.float32)) == (2, 7, 7)
    with pytest.raises(ContractError, match="non-finite"):
        validate_action_chunk(np.full((1, 7, 7), np.nan, dtype=np.float32))
    with pytest.raises(ContractError, match="shape"):
        validate_action_chunk(np.zeros((7, 7), dtype=np.float32))


def test_libero_policy_batch_contract() -> None:
    batch = {
        "observation.images.image": np.zeros((2, 3, 224, 224), dtype=np.float32),
        "observation.images.image2": np.zeros((2, 3, 224, 224), dtype=np.float32),
        "observation.state": np.zeros((2, 8), dtype=np.float32),
        "task": ["a", "b"],
    }
    validate_policy_batch(batch)
    bad = dict(batch)
    bad["observation.state"] = np.zeros((2, 7), dtype=np.float32)
    with pytest.raises(ContractError, match="8-dimensional"):
        validate_policy_batch(bad)


def test_external_candidate_unsupported_is_explicit() -> None:
    with pytest.raises(
        ExternalCandidateUnsupportedError,
        match="not externally supplied numeric action chunks",
    ):
        reject_external_action_candidates()


def test_rollout_manifest_binds_policy_identity_and_no_training() -> None:
    identity = LeRobotRolloutIdentity(
        episode_id="ep-0",
        suite="libero_spatial",
        task="task",
        task_id=0,
        seed=1000,
        instruction="do task",
        checkpoint_revision=CHECKPOINT_REVISION,
        processor_revision=CHECKPOINT_REVISION,
        policy_configuration_sha256="a" * 64,
    )
    result = LeRobotRolloutResult(
        success=False,
        termination_reason="horizon_exhausted",
        frame_count=280,
        policy_queries=40,
        action_mask_semantic="not_emitted_by_native_inference",
    )
    payload = build_rollout_manifest_entry(
        identity,
        result,
        observation_keys=["observation.images.image", "observation.state"],
        dataset_path="runtime/ep-0",
        timestamps={"started": "2026-07-23T00:00:00Z"},
    )
    assert payload["identity"]["checkpoint_revision"] == CHECKPOINT_REVISION
    assert payload["result"]["optimizer_steps"] == 0
    assert payload["result"]["backward_calls"] == 0


def test_eval_schedules_are_fixed_and_not_official_equivalent() -> None:
    single = json.loads(
        (ROOT / "configs" / "lg_r0" / "eval_single.json").read_text(encoding="utf-8")
    )
    forty = json.loads(
        (ROOT / "configs" / "lg_r0" / "eval_40ep.json").read_text(encoding="utf-8")
    )
    assert sum(len(suite["seeds"]) for suite in single["suites"]) == 1
    assert sum(len(suite["seeds"]) for suite in forty["suites"]) == 40
    assert forty["official_400_episode_equivalent"] is False
    assert forty["action_execution_horizon"] == 7


def test_libero_config_is_initialized_noninteractively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    sys.modules.pop("_lg_r0_runtime", None)
    runtime = importlib.import_module("_lg_r0_runtime")

    package_root = tmp_path / "site-packages" / "libero" / "libero"
    (package_root / "bddl_files").mkdir(parents=True)
    (package_root / "init_files").mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    output = tmp_path / "runtime-output"

    class FakeDistribution:
        def locate_file(self, relative: str) -> Path:
            assert relative == "libero/libero"
            return package_root

    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: (
            FakeDistribution()
            if name == "hf-libero"
            else pytest.fail(f"unexpected distribution: {name}")
        ),
    )
    monkeypatch.setenv("LG_R0_OUTPUT_ROOT", str(output))
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)

    config_file = runtime.prepare_libero_config(assets)
    payload = json.loads(config_file.read_text(encoding="utf-8"))

    assert config_file == output / "libero-config" / "config.yaml"
    assert Path(payload["assets"]) == assets.resolve()
    assert Path(payload["bddl_files"]) == package_root / "bddl_files"
    assert Path(payload["init_states"]) == package_root / "init_files"
    assert Path(payload["datasets"]) == output / "libero-datasets"
    assert Path(runtime.os.environ["LIBERO_CONFIG_PATH"]) == config_file.parent
