"""Materialize an auditable LG-R2b0 Result C after an upstream hard stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r2b0_common import (
    read_json,
    runtime_identity,
    sha256_path,
    validate_execution_checkout,
    write_json,
)

NOT_RUN_ARTIFACTS = {
    "anchor_registry.json": "latentguard.lg_r2b0.anchor_registry.v1",
    "candidate_manifest.json": "latentguard.lg_r2b0.candidate_manifest.v1",
    "candidate_diversity_report.json": (
        "latentguard.lg_r2b0.candidate_diversity_report.v1"
    ),
    "dataset_manifest.json": "latentguard.lg_r2b0.dataset_manifest.v1",
    "dataset_validation.json": "latentguard.lg_r2b0.dataset_validation.v1",
    "outcome_diversity_report.json": (
        "latentguard.lg_r2b0.outcome_diversity_report.v1"
    ),
    "rank_stability_report.json": "latentguard.lg_r2b0.rank_stability_report.v1",
}


def finalize_hard_stop(
    runtime: Path,
    *,
    audit_runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    """Write explicit not-run artifacts and a Result C gate from hard-gate evidence."""
    runtime = runtime.resolve()
    source_path = runtime / "candidate_source_manifest.json"
    restore_path = runtime / "state_restore_validation.json"
    source = read_json(source_path)
    restore = read_json(restore_path)
    source_passed = source.get("status") == "pass"
    restore_passed = restore.get("status") == "pass"
    if source_passed and restore_passed:
        raise ValueError("hard-stop finalization requires a failed upstream gate")
    if not source_passed:
        failed_gate = "candidate_source"
        stop_reason = str(
            source.get("stop_reason") or "REAL_MULTI_CANDIDATE_SOURCE_NOT_AVAILABLE"
        )
    else:
        failed_gate = "state_restore"
        stop_reason = str(
            restore.get("stop_reason") or "STATE_RESTORE_DETERMINISM_GATE_FAILED"
        )
    input_sha256 = {
        "candidate_source_manifest": sha256_path(source_path),
        "state_restore_validation": sha256_path(restore_path),
    }
    execution_commits = sorted(
        {
            str(payload.get("runtime_identity", {}).get("git_commit"))
            for payload in (source, restore)
            if payload.get("runtime_identity", {}).get("git_commit")
        }
    )
    common = {
        "status": "not_run",
        "availability": "unavailable",
        "not_run_reason": f"UPSTREAM_{stop_reason}",
        "failed_upstream_gate": failed_gate,
        "input_sha256": input_sha256,
        "simulation_execution_commits": execution_commits,
        "audit_runtime_identity": audit_runtime_identity,
        "metrics_available": False,
        "metric_values": None,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "final_seeds_accessed": False,
    }
    for name, schema_version in NOT_RUN_ARTIFACTS.items():
        payload = {
            "schema_version": schema_version,
            **common,
        }
        if name == "anchor_registry.json":
            payload["registered_target_anchors"] = 60
            payload["executed_anchor_count"] = 0
        elif name == "candidate_manifest.json":
            payload["registered_candidates_per_anchor"] = 4
            payload["executed_candidate_count"] = 0
        elif name == "dataset_manifest.json":
            payload["registered_minimum_branches"] = 150
            payload["executed_branch_count"] = 0
        write_json(runtime / name, payload)
    gate = {
        "schema_version": "latentguard.lg_r2b0.lg_r2b1_gate.v1",
        "status": "blocked",
        "result": "C",
        "result_definition": (
            "real native candidates exist but complete same-state restoration is "
            "not viable under the frozen contract"
            if source_passed
            else "a real native multi-candidate source is not available"
        ),
        "LG_R2B1_AUTHORIZED": False,
        "failed_gate": failed_gate,
        "stop_reason": stop_reason,
        "candidate_source_gate_passed": source_passed,
        "state_restore_gate_passed": restore_passed,
        "downstream_collection_performed": False,
        "downstream_metrics_available": False,
        "downstream_metric_values": None,
        "input_sha256": input_sha256,
        "simulation_execution_commits": execution_commits,
        "audit_runtime_identity": audit_runtime_identity,
        "no_training": {
            "candidate_ranker": True,
            "failure_head": True,
            "world_model": True,
            "reward_model": True,
            "uncertainty_model": True,
        },
        "no_online_selection_or_intervention": True,
        "synthetic_candidates": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "final_seeds_accessed": False,
    }
    write_json(runtime / "lg_r2b1_gate.json", gate)
    return gate


def main() -> None:
    """Finalize one failed remote runtime without running downstream simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    validate_execution_checkout(args.expected_commit)
    gate = finalize_hard_stop(
        args.runtime_dir,
        audit_runtime_identity=runtime_identity(),
    )
    print(
        json.dumps(
            {
                "status": gate["status"],
                "result": gate["result"],
                "LG_R2B1_AUTHORIZED": gate["LG_R2B1_AUTHORIZED"],
                "failed_gate": gate["failed_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
