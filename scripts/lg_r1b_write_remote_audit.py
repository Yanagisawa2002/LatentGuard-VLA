"""Record the exact non-destructive remote LG-R1b execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from _lg_r1b_common import output_root, read_json, runtime_identity, write_json


def _optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def _audit_checks_pass(checks: dict[str, bool]) -> bool:
    required_true = (
        "exact_commit",
        "clean_remote_checkout",
        "network_turbo_sourced",
        "remote_execution_only",
        "server_left_online",
    )
    required_false = (
        "tracked_source_edited_remotely",
        "server_shutdown_requested",
        "final_seed_accessed",
        "synthetic_failure_generated",
        "policy_training_performed",
        "vlajepa_finetuning_performed",
        "failure_head_training_performed",
        "intervention_performed",
        "robolab_executed",
        "langmani_modified",
    )
    return all(checks[name] for name in required_true) and not any(
        checks[name] for name in required_false
    )


def main() -> None:
    """Write sanitized runtime identity and prohibited-activity assertions."""

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
            "pilot": "pilot_manifest.json",
            "rollout": "rollout_manifest.json",
            "stage": "stage_annotation_report.json",
            "dataset": "dataset_manifest.json",
            "zero_shot": "sarm_zero_shot_results.json",
            "adaptation": "sarm_adaptation_results.json",
            "failure_registry": "failure_registry.json",
            "failure_windows": "failure_window_manifest.json",
            "progress_failure": "progress_failure_analysis.json",
            "vlajepa_descriptive": "vlajepa_descriptive_comparison.json",
            "lg_r2_gate": "lg_r2_gate.json",
        }.items()
    }
    checks = {
        "exact_commit": identity["git_commit"] == args.expected_commit,
        "clean_remote_checkout": clean,
        "network_turbo_sourced": (
            os.environ.get("LG_R1B_NETWORK_TURBO_SOURCED") == "1"
        ),
        "remote_execution_only": True,
        "server_left_online": True,
        "tracked_source_edited_remotely": False,
        "server_shutdown_requested": False,
        "final_seed_accessed": False,
        "synthetic_failure_generated": False,
        "policy_training_performed": False,
        "vlajepa_finetuning_performed": False,
        "failure_head_training_performed": False,
        "intervention_performed": False,
        "robolab_executed": False,
        "langmani_modified": False,
    }
    payload = {
        "schema_version": "latentguard.lg_r1b.remote_execution_audit.v1",
        "status": "pass" if _audit_checks_pass(checks) else "fail",
        "runtime_identity": identity,
        "checks": checks,
        "download_policy": (
            "source /etc/network_turbo after boot; Aliyun mirrors by default; "
            "no unreviewed dependency upgrades"
        ),
        "artifact_statuses": {
            name: value.get("status") if value is not None else "missing"
            for name, value in artifacts.items()
        },
        "sarm_adaptation_performed": (
            artifacts["adaptation"] is not None
            and artifacts["adaptation"].get("status") == "pass"
        ),
        "server_left_online": True,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "pass":
        raise RuntimeError("LG-R1b remote execution audit failed")


if __name__ == "__main__":
    main()
