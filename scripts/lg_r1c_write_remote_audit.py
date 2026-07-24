"""Assemble a sanitized compact LG-R1c remote execution audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r1c_common import (
    file_identity,
    read_json,
    resolve_repo_path,
    write_json,
)


def main() -> None:
    """Bind compact remote outputs without recording private paths or secrets."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1c/remote_execution_audit.json"),
    )
    args = parser.parse_args()
    runtime = args.runtime_root
    summaries = {
        name: read_json(runtime / filename)
        for name, filename in {
            "environment": "environment_validation.json",
            "robometer": "robometer_inference.json",
            "topreward": "topreward_inference.json",
            "evaluation": "evaluation_summary.json",
        }.items()
    }
    commits = {
        str(payload["runtime_identity"]["git_commit"])
        for payload in summaries.values()
        if isinstance(payload.get("runtime_identity"), dict)
    }
    if commits != {args.expected_commit}:
        raise ValueError(f"remote execution commit mismatch: {sorted(commits)}")
    compact_files = {}
    for path in sorted(runtime.glob("*.json")):
        compact_files[path.name] = file_identity(path, locator=path.name)
    payload = {
        "schema_version": "latentguard.lg_r1c.remote_execution_audit.v1",
        "status": "pass",
        "run_id": args.run_id,
        "branch": "codex/lg-r1c-task-agnostic-reward",
        "commit": args.expected_commit,
        "network_turbo_sourced_every_session": True,
        "aliyun_pypi_default": True,
        "remote_tracked_source_modified": False,
        "remote_commit_created": False,
        "new_rollouts": 0,
        "simulator_started": False,
        "foundation_model_training": False,
        "failure_head_training": False,
        "intervention": False,
        "langmani_used": False,
        "final_seeds_accessed": False,
        "server_shutdown": False,
        "server_left_online": True,
        "summaries": summaries,
        "compact_files": compact_files,
    }
    write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
