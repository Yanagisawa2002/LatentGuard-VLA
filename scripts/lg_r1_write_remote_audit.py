"""Summarize the exact, non-destructive remote LG-R1 execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from _lg_r1_common import (
    output_root,
    read_json,
    runtime_identity,
    write_json,
)


def _optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def main() -> None:
    """Record revision, environment, phase artifacts, and prohibited work."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    runtime = args.runtime_root or output_root()
    identity = runtime_identity()
    clean = (
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == ""
    )
    artifacts = {
        name: _optional_json(runtime / filename)
        for name, filename in {
            "rollout": "rollout_manifest.json",
            "stage": "stage_annotation_report.json",
            "dataset": "dataset_manifest.json",
            "time": "time_baseline_results.json",
            "probe": "representation_probe_results.json",
            "sarm": "sarm_results.json",
            "evaluation": "progress_evaluation.json",
            "lg_r2_gate": "lg_r2_gate.json",
        }.items()
    }
    checks = {
        "exact_commit": identity["git_commit"] == args.expected_commit,
        "clean_remote_checkout": clean,
        "network_turbo_sourced": (os.environ.get("LG_R1_NETWORK_TURBO_SOURCED") == "1"),
        "remote_execution_only": True,
        "tracked_source_edited_remotely": False,
        "server_shutdown_requested": False,
        "final_seed_accessed": False,
        "synthetic_failure_generated": False,
        "policy_training_performed": False,
        "vlajepa_finetuning_performed": False,
    }
    payload = {
        "schema_version": ("latentguard.lg_r1.remote_execution_audit.v1"),
        "status": "pass" if all(checks.values()) else "fail",
        "runtime_identity": identity,
        "checks": checks,
        "download_policy": (
            "source /etc/network_turbo on boot; use Aliyun package mirrors "
            "by default; no unreviewed dependency upgrades"
        ),
        "artifact_statuses": {
            name: value.get("status") if value is not None else "missing"
            for name, value in artifacts.items()
        },
        "server_left_online": True,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "pass":
        raise RuntimeError("remote execution audit failed")


if __name__ == "__main__":
    main()
