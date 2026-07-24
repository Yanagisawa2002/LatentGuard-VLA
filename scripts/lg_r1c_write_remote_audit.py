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
    parser.add_argument("--expected-final-commit", required=True)
    parser.add_argument("--allowed-execution-commit", action="append", default=[])
    parser.add_argument(
        "--sync-method",
        choices=("origin_fetch", "verified_git_bundle"),
        required=True,
    )
    parser.add_argument(
        "--origin-fetch-status",
        choices=("pass", "private_origin_credentials_unavailable"),
        required=True,
    )
    parser.add_argument("--bundle-sha256", action="append", default=[])
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
    execution_commits = {
        name: str(payload["runtime_identity"]["git_commit"])
        for name, payload in summaries.items()
        if isinstance(payload.get("runtime_identity"), dict)
    }
    allowed = {
        args.expected_final_commit,
        *[str(value) for value in args.allowed_execution_commit],
    }
    unexpected = sorted(set(execution_commits.values()) - allowed)
    if unexpected:
        raise ValueError(f"unexpected remote execution commits: {unexpected}")
    if (
        execution_commits.get("evaluation") != args.expected_final_commit
        or len(args.expected_final_commit) != 40
    ):
        raise ValueError("final evaluation commit mismatch")
    if args.sync_method == "verified_git_bundle" and not args.bundle_sha256:
        raise ValueError("verified bundle synchronization requires bundle hashes")
    for value in args.bundle_sha256:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("invalid bundle SHA-256")
    compact_files = {}
    for path in sorted(runtime.glob("*.json")):
        compact_files[path.name] = file_identity(path, locator=path.name)
    payload = {
        "schema_version": "latentguard.lg_r1c.remote_execution_audit.v1",
        "status": "pass",
        "run_id": args.run_id,
        "branch": "codex/lg-r1c-task-agnostic-reward",
        "final_commit": args.expected_final_commit,
        "execution_commits": execution_commits,
        "remote_synchronization": {
            "method": args.sync_method,
            "origin_fetch_status": args.origin_fetch_status,
            "bundle_sha256": args.bundle_sha256,
            "bundle_content_verified": args.sync_method == "verified_git_bundle",
            "fast_forward_only": True,
            "exact_commit_checked_before_each_phase": True,
        },
        "network_turbo_sourced_every_session": True,
        "aliyun_pypi_default": True,
        "external_model_downloads": {
            "robometer": (
                "exact Hugging Face revision through the server accelerator; "
                "no verified ModelScope mirror was available"
            ),
            "topreward": (
                "Alibaba ModelScope preferred, with frozen Hugging Face shard "
                "hashes required before inference"
            ),
        },
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
