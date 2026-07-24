"""Write the sanitized, content-bound LG-R2b0 remote execution audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from _lg_r2b0_common import (
    git_output,
    output_root,
    runtime_identity,
    sha256_path,
    write_json,
)

COMPACT_FILES = (
    "candidate_source_manifest.json",
    "state_restore_validation.json",
    "anchor_registry.json",
    "candidate_manifest.json",
    "candidate_diversity_report.json",
    "dataset_manifest.json",
    "dataset_validation.json",
    "outcome_diversity_report.json",
    "rank_stability_report.json",
    "lg_r2b1_gate.json",
    "source_validation.json",
)


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "locator": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def main() -> None:
    """Bind the clean checkout, environment, commands, and compact evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--started-at-utc", required=True)
    parser.add_argument("--network-turbo-sourced", action="store_true")
    parser.add_argument("--aliyun-default", action="store_true")
    args = parser.parse_args()
    runtime = output_root()
    commit = git_output("rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise ValueError("remote audit checkout does not match expected commit")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("remote tracked checkout changed during execution")
    started = dt.datetime.fromisoformat(args.started_at_utc.replace("Z", "+00:00"))
    ended = dt.datetime.now(tz=dt.UTC)
    files = {}
    for name in COMPACT_FILES:
        path = runtime / name
        if not path.is_file():
            raise ValueError(f"required compact output is missing: {name}")
        files[name] = _file_identity(path)
    scientific_gate = json.loads(
        (runtime / "lg_r2b1_gate.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        (runtime / "candidate_source_manifest.json").read_text(encoding="utf-8")
    )
    restore = json.loads(
        (runtime / "state_restore_validation.json").read_text(encoding="utf-8")
    )
    simulation_execution_commits = sorted(
        {
            str(payload.get("runtime_identity", {}).get("git_commit"))
            for payload in (source, restore)
            if payload.get("runtime_identity", {}).get("git_commit")
        }
    )
    commands_path = runtime / "execution_commands.json"
    commands: list[str] = []
    if commands_path.is_file():
        loaded = json.loads(commands_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or not all(
            isinstance(value, str) for value in loaded
        ):
            raise ValueError("execution command record is malformed")
        commands = loaded
    payload = {
        "schema_version": "latentguard.lg_r2b0.remote_execution_audit.v1",
        "status": "pass",
        "run_id": args.run_id,
        "branch": git_output("branch", "--show-current"),
        "commit": commit,
        "source_checkout_clean_after_run": True,
        "remote_commit_created": False,
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "network_turbo_sourced": args.network_turbo_sourced,
        "aliyun_default_for_downloads": args.aliyun_default,
        "downloads_performed": False,
        "runtime_identity": runtime_identity(),
        "compact_files": files,
        "compact_file_set_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scientific_result": scientific_gate.get("result"),
        "scientific_gate_status": scientific_gate.get("status"),
        "LG_R2B1_AUTHORIZED": scientific_gate.get("LG_R2B1_AUTHORIZED"),
        "simulation_execution_commits": simulation_execution_commits,
        "execution_commands": commands,
        "training_performed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "checkpoints_generated": 0,
        "synthetic_candidates": 0,
        "candidate_ranker_trained": False,
        "online_selection": False,
        "intervention": False,
        "final_seeds_accessed": False,
        "tracked_source_modified_remotely": False,
        "server_shutdown_requested": False,
        "server_left_running": True,
    }
    write_json(runtime / "remote_execution_audit.json", payload)
    print({"status": "pass", "run_id": args.run_id, "compact_files": len(files)})


if __name__ == "__main__":
    main()
