"""Validate and apply the exact LG-RB0.1 RoboLab compatibility patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/NVLabs/RoboLab.git"


def _git(checkout: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _patch_command(
    checkout: Path,
    patch: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *arguments, str(patch.resolve())],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_patch_state(
    *,
    checkout: Path,
    patch: Path,
    expected_base_commit: str,
    mode: str,
) -> dict[str, Any]:
    """Validate the exact base and optionally apply the patch to the index."""
    if _git(checkout, "rev-parse", "HEAD") != expected_base_commit:
        raise RuntimeError("RoboLab checkout does not match the expected base commit")
    if _git(checkout, "remote", "get-url", "origin") != UPSTREAM_REPOSITORY:
        raise RuntimeError("RoboLab checkout does not use the official origin")

    status_before = _git(checkout, "status", "--porcelain=v1")
    if mode in {"verify-unpatched", "apply"} and status_before:
        raise RuntimeError("unpatched RoboLab checkout must be clean")

    forward = _patch_command(checkout, patch, "--check")
    reverse = _patch_command(checkout, patch, "--reverse", "--check")
    if mode == "verify-unpatched":
        if forward.returncode != 0 or reverse.returncode == 0:
            raise RuntimeError("clean checkout did not identify as unpatched")
    elif mode == "apply":
        if forward.returncode != 0:
            raise RuntimeError(
                f"patch does not apply cleanly: {forward.stderr.strip()}"
            )
        applied = _patch_command(checkout, patch, "--index")
        if applied.returncode != 0:
            raise RuntimeError(f"patch application failed: {applied.stderr.strip()}")
    elif mode != "verify-patched":
        raise ValueError(f"unknown patch verification mode: {mode}")

    status_after = _git(checkout, "status", "--porcelain=v1")
    forward_after = _patch_command(checkout, patch, "--check")
    reverse_after = _patch_command(checkout, patch, "--reverse", "--check")
    patched = (
        bool(status_after)
        and forward_after.returncode != 0
        and reverse_after.returncode == 0
    )
    if mode in {"apply", "verify-patched"} and not patched:
        raise RuntimeError("RoboLab checkout did not identify as exactly patched")

    changed_files = (
        _git(checkout, "diff", "--cached", "--name-only").splitlines()
        if patched
        else []
    )
    patched_tree = _git(checkout, "write-tree") if patched else None
    return {
        "schema_version": "lg_rb01_patch_application_v1",
        "status": "pass",
        "mode": mode,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_base_commit": expected_base_commit,
        "patch_sha256": _sha256(patch),
        "identified_unpatched": not patched,
        "identified_patched": patched,
        "forward_apply_check_passed": forward_after.returncode == 0,
        "reverse_apply_check_passed": reverse_after.returncode == 0,
        "repeat_application_rejected": patched and forward_after.returncode != 0,
        "changed_files": changed_files,
        "patched_tree_digest": patched_tree,
        "checkout_clean": not bool(status_after),
    }


def main() -> None:
    """Run exact patch provenance validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--expected-base-commit", required=True)
    parser.add_argument(
        "--mode",
        choices=["verify-unpatched", "apply", "verify-patched"],
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_patch_state(
        checkout=args.checkout.resolve(),
        patch=args.patch.resolve(),
        expected_base_commit=args.expected_base_commit,
        mode=args.mode,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
