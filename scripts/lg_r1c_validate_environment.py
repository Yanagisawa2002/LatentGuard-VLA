"""Validate the isolated LG-R1c remote CUDA and LeRobot environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from _lg_r1c_common import (
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    write_json,
)
from lg_r1c_audit_reward_stack import LEROBOT_INTEGRATION_HASHES


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> None:
    """Fail closed unless the remote reward stack matches the frozen versions."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1c/environment_validation.json"),
    )
    args = parser.parse_args()
    import lerobot
    import torch

    import latentguard

    expected_versions = {
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "lerobot": "0.6.0",
        "transformers": "5.5.4",
        "qwen-vl-utils": "0.0.14",
        "huggingface-hub": "1.22.0",
        "safetensors": "0.8.0",
    }
    versions = {name: _version(name) for name in expected_versions}
    mismatches = {
        name: {"expected": expected_versions[name], "observed": observed}
        for name, observed in versions.items()
        if observed != expected_versions[name]
    }
    if mismatches:
        raise ValueError(f"LG-R1c environment version drift: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("LG-R1c remote environment requires CUDA")
    reward_root = Path(lerobot.__file__).resolve().parent / "rewards"
    source_hashes: dict[str, str] = {}
    for relative, expected in LEROBOT_INTEGRATION_HASHES.items():
        path = reward_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_path(path)
        if observed != expected:
            raise ValueError(f"LeRobot integration source hash drift: {relative}")
        source_hashes[relative] = observed
    for module_name in (
        "lerobot.rewards.robometer.modeling_robometer",
        "lerobot.rewards.robometer.processor_robometer",
        "lerobot.rewards.topreward.modeling_topreward",
        "lerobot.rewards.topreward.processor_topreward",
    ):
        importlib.import_module(module_name)
    latentguard_path = Path(latentguard.__file__).resolve()
    repository_source = resolve_repo_path(Path("src/latentguard")).resolve()
    if repository_source not in latentguard_path.parents:
        raise ValueError("latentguard import does not come from this checkout")
    payload: dict[str, Any] = {
        "schema_version": "latentguard.lg_r1c.environment_validation.v1",
        "status": "pass",
        "runtime_identity": runtime_identity(),
        "versions": versions,
        "cuda": {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "lerobot_release": "v0.6.0",
        "lerobot_integration_commit": ("30da8e687a6dfc617fcd94afc367ac7071c376ce"),
        "lerobot_integration_file_sha256": source_hashes,
        "latentguard_import": "src/latentguard/__init__.py",
        "isolated_overlay": True,
        "lg_r0_lg_r1_environment_upgraded": False,
    }
    write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
