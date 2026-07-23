"""Capture the isolated LG-R0 Linux/CUDA environment without network access."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from _lg_r0_runtime import libero_asset_path, model_paths, write_json

from latentguard.adapters.vla_jepa.constants import (
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
    LIBERO_VERSION,
)


def _distribution_versions() -> dict[str, str]:
    return dict(
        sorted(
            (
                distribution.metadata["Name"],
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        )
    )


def _lerobot_direct_url() -> dict[str, Any] | None:
    distribution = importlib.metadata.distribution("lerobot")
    direct_url = Path(distribution._path) / "direct_url.json"
    if not direct_url.is_file():
        return None
    payload = json.loads(direct_url.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _sanitize_direct_url(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Retain installation semantics without committing a machine path."""

    if payload is None:
        return None
    return {
        "editable": bool(payload.get("dir_info", {}).get("editable", False)),
        "source_kind": (
            "local_wheel"
            if str(payload.get("url", "")).startswith("file:")
            else "nonlocal_url"
        ),
        "vcs_info": payload.get("vcs_info"),
    }


def main() -> None:
    """Write both a detailed validation artifact and reproducibility manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    import mujoco
    import torch
    import transformers

    checkpoint, qwen, vjepa = model_paths()
    libero_assets = libero_asset_path()
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu_name, gpu_memory_mib, driver = [part.strip() for part in smi.split(",")]
    direct_url = _lerobot_direct_url()
    editable = bool(
        direct_url and direct_url.get("dir_info", {}).get("editable", False)
    )
    packages = _distribution_versions()
    manifest = {
        "schema_version": "latentguard.lg_r0.environment_manifest.v1",
        "python_version": platform.python_version(),
        "python_executable_name": Path(sys.executable).name,
        "os": platform.platform(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_memory_mib": int(gpu_memory_mib),
        "nvidia_driver": driver,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "lerobot_commit": LEROBOT_COMMIT,
        "vla_jepa_commit": ("2e9cd87bbdb93c23503f7eeca7317bd33027b279"),
        "libero_version": importlib.metadata.version("hf-libero"),
        "mujoco_version": mujoco.__version__,
        "all_installed_package_versions": packages,
    }
    checks = {
        "python_3_12": sys.version_info[:2] == (3, 12),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "fp16_supported": torch.cuda.is_available(),
        "lerobot_version": manifest["lerobot_version"] == LEROBOT_VERSION,
        "libero_version": manifest["libero_version"] == LIBERO_VERSION,
        "lerobot_not_editable": not editable,
        "checkpoint_snapshot_present": checkpoint.is_dir(),
        "qwen_snapshot_present": qwen.is_dir(),
        "vjepa_snapshot_present": vjepa.is_dir(),
        "libero_asset_snapshot_present": libero_assets.is_dir(),
        "no_huggingface_token_required": not any(
            name in os.environ for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        ),
    }
    validation = {
        "schema_version": "latentguard.lg_r0.environment_validation.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "manifest": manifest,
        "lerobot_direct_url": _sanitize_direct_url(direct_url),
        "editable_install": editable,
        "checkpoint_snapshot_names": {
            "policy": checkpoint.name,
            "qwen": qwen.name,
            "vjepa": vjepa.name,
            "libero_assets": libero_assets.name,
        },
        "network_access_during_validation": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    write_json(args.manifest, manifest)
    write_json(args.output, validation)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "python": manifest["python_version"],
                "torch": manifest["torch_version"],
                "output": str(args.output),
            }
        )
    )
    if validation["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
