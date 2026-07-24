"""Finalize the compact LG-RB0 remote audit without launching a simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REQUIRED_ARTIFACTS = (
    "source_validation.json",
    "robolab_stack_manifest.json",
    "environment_validation.json",
    "recording_manifest.json",
    "faithful_replay_validation.json",
    "prefix_replay_validation.json",
    "branch_determinism_validation.json",
    "branch_isolation_validation.json",
    "lg_rb1_gate.json",
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_commit(root: Path, commit: str, current: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("phase commit must be a full lowercase SHA")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        check=True,
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("phase commit is not an ancestor of audit checkout")


def _sanitize_phase(
    root: Path,
    phase: Mapping[str, Any],
    *,
    current: str,
) -> dict[str, Any]:
    name = str(phase.get("name", ""))
    run_id = str(phase.get("run_id", ""))
    commit = str(phase.get("commit", ""))
    log_path = Path(str(phase.get("log_path", ""))).resolve()
    exit_path = Path(str(phase.get("exit_path", ""))).resolve()
    if not name or not run_id:
        raise ValueError("phase name and run_id are required")
    _validate_commit(root, commit, current)
    if not log_path.is_file() or not exit_path.is_file():
        raise ValueError(f"phase evidence file is missing: {name}")
    try:
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise ValueError(f"phase exit code is invalid: {name}") from error
    if exit_code != 0:
        raise ValueError(f"successful phase audit requires exit 0: {name}")
    return {
        "name": name,
        "run_id": run_id,
        "commit": commit,
        "exit_code": exit_code,
        "log_sha256": _sha256(log_path),
        "log_size_bytes": log_path.stat().st_size,
        "simulator_used": bool(phase.get("simulator_used")),
    }


def main() -> None:
    """Validate phase evidence and write a path-free remote audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--phase-spec", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-external-commit", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    current = _git(root, "rev-parse", "HEAD")
    if current != args.expected_commit:
        raise RuntimeError("audit checkout does not match expected commit")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("audit checkout is not clean")
    external_root = args.external_root.resolve()
    if _git(external_root, "rev-parse", "HEAD") != args.expected_external_commit:
        raise RuntimeError("external checkout does not match frozen commit")
    if _git(external_root, "status", "--porcelain"):
        raise RuntimeError("external checkout is not clean")

    artifact_dir = args.artifact_dir.resolve()
    prior_path = artifact_dir / "remote_execution_audit.json"
    prior = _read_object(prior_path)
    if prior.get("status") != "pass":
        raise RuntimeError("initial remote audit did not pass")
    phase_spec = _read_object(args.phase_spec)
    raw_phases = phase_spec.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("phase spec must contain a non-empty phases list")
    phases = [
        _sanitize_phase(root, phase, current=current)
        for phase in raw_phases
        if isinstance(phase, Mapping)
    ]
    if len(phases) != len(raw_phases):
        raise ValueError("every phase spec entry must be a mapping")
    names = [phase["name"] for phase in phases]
    if len(set(names)) != len(names):
        raise ValueError("phase names must be unique")

    artifacts: dict[str, dict[str, Any]] = {}
    for name in _REQUIRED_ARTIFACTS:
        path = artifact_dir / name
        value = _read_object(path)
        artifacts[name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "status": value.get("status"),
        }
    gate = _read_object(artifact_dir / "lg_rb1_gate.json")
    faithful = _read_object(artifact_dir / "faithful_replay_validation.json")
    audit = {
        "schema_version": "lg_rb0_remote_execution_audit_v2",
        "status": "pass",
        "hostname": platform.node(),
        "branch": _git(root, "branch", "--show-current"),
        "audit_generation_commit": current,
        "tracked_checkout_clean": True,
        "external_source_clean": True,
        "external_commit": args.expected_external_commit,
        "initial_probe_audit_sha256": _sha256(prior_path),
        "network_turbo_sourced": bool(prior.get("network_turbo_sourced")),
        "phase_runs": phases,
        "artifacts": artifacts,
        "result": gate.get("result"),
        "LG_RB1_AUTHORIZED": gate.get("LG_RB1_AUTHORIZED"),
        "faithful_expected_replays": faithful.get("expected_replay_count"),
        "faithful_completed_replays": faithful.get("completed_replay_count"),
        "faithful_execution_errors": faithful.get("execution_error_count"),
        "takeover_status": "not_run",
        "training_performed": False,
        "checkpoint_generated": False,
        "policy_or_ranker_loaded": False,
        "candidate_selection_or_intervention": False,
        "final_seeds_accessed": False,
    }
    prior_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
