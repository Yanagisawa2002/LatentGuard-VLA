"""Shared dependency-light helpers for LG-R1 command-line tools."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from latentguard.progress.serialization import (  # noqa: E402
    read_json,
    write_json_atomic,
)


def repo_root() -> Path:
    """Return the local or remote LatentGuard checkout root."""

    return _REPOSITORY_ROOT


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a configured repository-relative path."""

    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping without implicit repair."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return payload


def sha256_path(path: Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    """Run one read-only Git query in the authoritative checkout."""

    result = subprocess.run(
        ["git", *args],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def runtime_identity() -> dict[str, Any]:
    """Return a sanitized execution identity for remote manifests."""

    torch_identity: dict[str, Any] = {"available": False}
    try:
        import torch

        torch_identity = {
            "available": True,
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except ImportError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_branch": git_output("branch", "--show-current"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "torch": torch_identity,
    }


def output_root() -> Path:
    """Return the ignored LG-R1 runtime root."""

    configured = os.environ.get("LG_R1_OUTPUT_ROOT")
    if configured:
        return Path(configured)
    return repo_root() / "outputs" / "lg_r1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON through the shared atomic contract."""

    write_json_atomic(path, payload)


__all__ = [
    "git_output",
    "output_root",
    "read_json",
    "read_yaml",
    "repo_root",
    "resolve_repo_path",
    "runtime_identity",
    "sha256_path",
    "write_json",
]
