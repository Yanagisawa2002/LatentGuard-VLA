"""Audit LG-R2b0 repository isolation, prohibited features, and secrets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from _lg_r2b0_common import output_root, repo_root, runtime_identity, write_json

BASELINE_COMMIT = "6ae64db453f835e27bba700bc1f58b227fd16939"
FORBIDDEN_PATH_MARKERS = (
    "langmani",
    "robolab",
    "pointworld",
)
SECRET_MARKERS = (
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "ssh-" + "rsa ",
    "pass" + "word=",
    "api_" + "token=",
)


def _git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo_root(),
        text=True,
    ).strip()


def validate_sources(*, expected_commit: str) -> dict[str, Any]:
    """Return a fail-closed audit of every path changed from LG-R2a."""
    commit = _git_output("rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError("source audit checkout does not match expected commit")
    if _git_output("status", "--porcelain"):
        raise ValueError("source audit requires a clean checkout")
    changed = sorted(
        line
        for line in _git_output(
            "diff",
            "--name-only",
            f"{BASELINE_COMMIT}...HEAD",
        ).splitlines()
        if line
    )
    forbidden = [
        path
        for path in changed
        if any(marker in path.casefold() for marker in FORBIDDEN_PATH_MARKERS)
    ]
    if forbidden:
        raise ValueError(f"forbidden repository paths changed: {forbidden}")
    secret_hits: list[str] = []
    for relative in changed:
        path = repo_root() / relative
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in SECRET_MARKERS):
            secret_hits.append(relative)
    if secret_hits:
        raise ValueError(f"credential-like content in changed files: {secret_hits}")
    return {
        "schema_version": "latentguard.lg_r2b0.source_validation.v1",
        "status": "pass",
        "baseline_commit": BASELINE_COMMIT,
        "commit": commit,
        "branch": _git_output("branch", "--show-current"),
        "runtime_identity": runtime_identity(),
        "changed_files_scanned": len(changed),
        "changed_files": changed,
        "langmani_imported": False,
        "langmani_modified": False,
        "robolab_used": False,
        "pointworld_used": False,
        "synthetic_candidates": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "candidate_ranker_trained": False,
        "online_selection": False,
        "intervention": False,
        "final_seeds_accessed": False,
        "secret_scan_hits": [],
    }


def main() -> None:
    """Write a compact source-boundary validation artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print({"status": "dry_run"})
        return
    payload = validate_sources(expected_commit=args.expected_commit)
    destination = (
        args.output_dir.resolve() if args.output_dir is not None else output_root()
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "source_validation.json", payload)
    print({"status": "pass", "changed_files_scanned": len(payload["changed_files"])})


if __name__ == "__main__":
    main()
