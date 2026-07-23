"""Shared helpers for the LG-R1b remote-only experiment tools."""

from __future__ import annotations

import os
from pathlib import Path

from _lg_r1_common import (
    git_output,
    read_json,
    read_yaml,
    repo_root,
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    write_json,
)


def output_root() -> Path:
    """Return the ignored LG-R1b runtime root."""

    configured = os.environ.get("LG_R1B_OUTPUT_ROOT")
    if configured:
        return Path(configured)
    return repo_root() / "outputs" / "lg_r1b"


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
