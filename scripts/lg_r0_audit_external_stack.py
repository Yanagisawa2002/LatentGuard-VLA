"""Validate frozen LG-R0 source identities and write compact evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from _lg_r0_runtime import read_json, repo_root, write_json

from latentguard.adapters.vla_jepa.constants import (
    CHECKPOINT_REVISION,
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
    LIBERO_ASSET_REPOSITORY,
    LIBERO_ASSET_REVISION,
    QWEN_REVISION,
    VJEPA_REVISION,
    VLA_JEPA_COMMIT,
)


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_audit(source_root: Path) -> dict[str, Any]:
    """Build an exact source and configuration audit."""

    config = read_json(repo_root() / "configs" / "lg_r0" / "stack.json")
    observed_commit = _git(source_root, "rev-parse", "HEAD")
    status = _git(source_root, "status", "--short")
    checks = {
        "lerobot_version": config["lerobot"]["version"] == LEROBOT_VERSION,
        "lerobot_commit": observed_commit == LEROBOT_COMMIT,
        "source_clean": status == "",
        "vla_jepa_commit": config["lerobot"]["vla_jepa_merge_commit"]
        == VLA_JEPA_COMMIT,
        "checkpoint_revision": config["checkpoint"]["revision"] == CHECKPOINT_REVISION,
        "qwen_revision": config["qwen"]["revision"] == QWEN_REVISION,
        "vjepa_revision": config["vjepa"]["revision"] == VJEPA_REVISION,
        "libero_asset_repository": config["libero"]["assets_repository"]
        == LIBERO_ASSET_REPOSITORY,
        "libero_asset_revision": config["libero"]["assets_revision"]
        == LIBERO_ASSET_REVISION,
    }
    return {
        "schema_version": "latentguard.lg_r0.external_stack_audit.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "observed_lerobot_commit": observed_commit,
        "source_root_locator": "external_clean_lerobot_checkout",
        "source_clean": status == "",
        "config": config,
        "network_access_during_validation": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_audit(args.source_root)
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
