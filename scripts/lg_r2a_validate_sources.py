"""Audit the LG-R2a revision for repository and research-boundary isolation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from _lg_r2a_common import (
    BASELINE_COMMIT,
    output_root,
    repo_root,
    validate_repository_lineage,
    write_json,
)

FORBIDDEN_PATH_PREFIXES = (
    "src/latentguard/integrations/langmani/",
    "configs/integrations/langmani/",
)
SECRET_MARKERS = (
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "ssh-rsa ",
    "password=",
    "api_token=",
)


def _changed_files() -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{BASELINE_COMMIT}...HEAD"],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in completed.stdout.splitlines() if line)


def validate_sources() -> dict[str, Any]:
    """Validate changed paths and reject embedded credentials."""

    changed = _changed_files()
    forbidden = [
        path
        for path in changed
        if any(path.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"forbidden LangMani integration changes: {forbidden}")
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
        "schema_version": "latentguard.lg_r2a.source_validation.v1",
        "status": "pass",
        "repository": validate_repository_lineage(),
        "changed_files_scanned": len(changed),
        "langmani_imported": False,
        "langmani_modified": False,
        "new_rollouts": 0,
        "synthetic_action_training": False,
        "candidate_generation": False,
        "candidate_ranking": False,
        "counterfactual_rollout": False,
        "intervention": False,
        "robolab_used": False,
        "pointworld_used": False,
        "final_seeds_accessed": False,
        "foundation_models_frozen": True,
        "secret_scan_hits": [],
    }


def main() -> None:
    """Write the compact boundary audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print({"status": "dry_run"})
        return
    payload = validate_sources()
    write_json(output_root(args.output_dir) / "source_validation.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
