"""Shared dependency-light helpers for LG-R2a commands."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
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
    write_jsonl_atomic,
)

BASELINE_COMMIT = "a2cdd476fe4c45e453f6d9535fd6c3ae6b2a83a5"
FINAL_SEEDS = frozenset(range(900000, 900100))


def repo_root() -> Path:
    """Return the authoritative local checkout."""

    return _REPOSITORY_ROOT


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path."""

    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL stream of objects."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic JSONL atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(path, rows)


def sha256_path(path: Path) -> str:
    """Hash a file without loading it fully."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(path: Path, pattern: str) -> dict[str, Any]:
    """Hash names and content identities for a deterministic file set."""

    files = sorted(item for item in path.glob(pattern) if item.is_file())
    if not files:
        raise ValueError(f"no files matching {pattern} under {path}")
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        identity = sha256_path(item)
        size = item.stat().st_size
        digest.update(f"{relative}\0{size}\0{identity}\n".encode())
        total += size
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def file_identity(path: Path, *, locator: str) -> dict[str, Any]:
    """Return path-independent file metadata."""

    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def git_output(*arguments: str) -> str:
    """Run a read-only Git query."""

    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_repository_lineage() -> dict[str, Any]:
    """Require LatentGuard origin and ancestry from the frozen baseline."""

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("HEAD does not descend from the frozen LG-R1c baseline")
    origin = git_output("remote", "get-url", "origin")
    if "LatentGuard-VLA" not in origin or "LangMani" in origin:
        raise ValueError(f"unexpected repository origin: {origin}")
    return {
        "baseline_commit": BASELINE_COMMIT,
        "head_commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "origin_repository": "Yanagisawa2002/LatentGuard-VLA",
        "baseline_is_ancestor": True,
    }


def validate_no_final_seeds(seeds: Iterable[int]) -> None:
    """Reject any use of the sealed final seed range."""

    overlap = sorted(set(seeds) & FINAL_SEEDS)
    if overlap:
        raise ValueError(f"sealed final seeds referenced: {overlap}")


def source_root(argument: Path | None = None) -> Path:
    """Return the frozen LG-R1b runtime root."""

    if argument is not None:
        return argument
    value = os.environ.get("LG_R2A_SOURCE_ROOT")
    if not value:
        raise ValueError("--source-root or LG_R2A_SOURCE_ROOT is required")
    return Path(value)


def r1c_root(argument: Path | None = None) -> Path:
    """Return the frozen LG-R1c runtime root."""

    if argument is not None:
        return argument
    value = os.environ.get("LG_R2A_R1C_ROOT")
    if not value:
        raise ValueError("--r1c-root or LG_R2A_R1C_ROOT is required")
    return Path(value)


def output_root(argument: Path | None = None) -> Path:
    """Return the external ignored LG-R2a output directory."""

    if argument is not None:
        return argument
    value = os.environ.get("LG_R2A_OUTPUT_ROOT")
    if value:
        return Path(value)
    return repo_root() / "outputs" / "lg_r2a" / "local"


def runtime_identity() -> dict[str, Any]:
    """Return a sanitized runtime identity and CUDA details."""

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
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "torch": torch_identity,
    }


__all__ = [
    "BASELINE_COMMIT",
    "FINAL_SEEDS",
    "directory_digest",
    "file_identity",
    "git_output",
    "output_root",
    "r1c_root",
    "read_json",
    "read_jsonl",
    "read_yaml",
    "repo_root",
    "resolve_repo_path",
    "runtime_identity",
    "sha256_path",
    "source_root",
    "validate_no_final_seeds",
    "validate_repository_lineage",
    "write_json",
    "write_jsonl",
]
