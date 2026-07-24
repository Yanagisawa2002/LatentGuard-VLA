"""Validate branch identity, action alignment, continuation, and outcome semantics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from _lg_r2b0_common import (
    read_json,
    runtime_identity,
    sha256_path,
    validate_execution_checkout,
    write_json,
)


def main() -> None:
    """Fail closed on any branch contamination or semantic drift."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    validate_execution_checkout(args.expected_commit)
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("status") != "pass":
        raise ValueError("counterfactual dataset manifest is incomplete")
    runtime = manifest_path.parent
    candidates = read_json(runtime / "candidate_manifest.json")
    anchors = read_json(runtime / "anchor_registry.json")
    candidate_by_id = {
        str(candidate["candidate_id"]): candidate
        for group in candidates["anchors"]
        for candidate in group["candidates"]
    }
    anchor_by_id = {str(anchor["anchor_id"]): anchor for anchor in anchors["anchors"]}
    errors: list[str] = []
    continuation_by_anchor: dict[str, set[tuple[int, str]]] = defaultdict(set)
    observed_ids: set[str] = set()
    unresolved = 0
    failures = 0
    environment_errors = 0
    for summary in manifest["branches"]:
        branch_id = str(summary["branch_id"])
        if branch_id in observed_ids:
            errors.append(f"duplicate branch id: {branch_id}")
            continue
        observed_ids.add(branch_id)
        branch_path = runtime / "branches" / f"{branch_id}.json"
        marker_path = runtime / "branch_completed" / f"{branch_id}.json"
        if not branch_path.is_file() or not marker_path.is_file():
            errors.append(f"incomplete branch runtime files: {branch_id}")
            continue
        marker = read_json(marker_path)
        if marker.get("branch_sha256") != sha256_path(branch_path):
            errors.append(f"branch content digest mismatch: {branch_id}")
            continue
        branch = read_json(branch_path)
        if branch.get("summary") != summary:
            errors.append(f"branch public summary drift: {branch_id}")
            continue
        candidate = candidate_by_id.get(str(summary["candidate_id"]))
        anchor = anchor_by_id.get(str(summary["anchor_id"]))
        if candidate is None or anchor is None:
            errors.append(f"branch references unknown identity: {branch_id}")
            continue
        if candidate["anchor_id"] != anchor["anchor_id"]:
            errors.append(f"candidate crosses anchors: {branch_id}")
        records = branch["records"]
        candidate_actions = np.asarray(
            candidate["native_action_chunk"],
            dtype=np.float32,
        )
        observed_actions = np.asarray(
            [item["executed_action"] for item in records[: candidate_actions.shape[0]]],
            dtype=np.float32,
        )
        if observed_actions.shape != candidate_actions.shape or not np.array_equal(
            observed_actions, candidate_actions
        ):
            errors.append(f"candidate action alignment mismatch: {branch_id}")
        if any(
            item["action_source"] != "real_policy_candidate"
            for item in records[: candidate_actions.shape[0]]
        ):
            errors.append(f"candidate source label mismatch: {branch_id}")
        if any(
            item["action_source"] != "fixed_primary_continuation"
            for item in records[candidate_actions.shape[0] :]
        ):
            errors.append(f"continuation source label mismatch: {branch_id}")
        continuation_by_anchor[str(anchor["anchor_id"])].add(
            (
                int(summary["continuation_seed"]),
                str(summary["continuation_checkpoint_revision"]),
            )
        )
        if summary["simulator_state_hash_before"] != anchor["simulator_state_hash"]:
            errors.append(f"branch start state mismatch: {branch_id}")
        if int(summary["seed"]) != int(anchor["seed"]):
            errors.append(f"episode seed mismatch: {branch_id}")
        if 900_000 <= int(summary["seed"]) <= 900_099:
            errors.append(f"sealed final seed accessed: {branch_id}")
        outcome = str(summary["terminal_outcome"])
        if outcome == "UNRESOLVED_HORIZON":
            unresolved += 1
        elif outcome == "FAILURE":
            failures += 1
        elif outcome == "ENVIRONMENT_ERROR":
            environment_errors += 1
        for horizon in ("7", "21", "49"):
            boundary = summary["boundaries"].get(horizon)
            if boundary is None:
                errors.append(f"missing {horizon}-step boundary: {branch_id}")
                continue
            expected_delta = float(boundary["progress"]) - float(
                summary["anchor_progress"]
            )
            observed_delta = float(
                summary["progress_deltas"][f"progress_delta_{horizon}"]
            )
            if abs(expected_delta - observed_delta) > 1e-12:
                errors.append(f"progress label misalignment: {branch_id}/{horizon}")
    for anchor_id, identities in continuation_by_anchor.items():
        if len(identities) != 1:
            errors.append(f"continuation configuration differs within {anchor_id}")
    if float(manifest["task6_branch_ratio"]) > 0.35:
        errors.append("task 6 branch ratio exceeds 35%")
    if int(manifest.get("optimizer_steps", -1)) != 0:
        errors.append("optimizer steps are prohibited")
    if int(manifest.get("backward_calls", -1)) != 0:
        errors.append("backward calls are prohibited")
    if manifest.get("candidate_ranking_model_trained") is not False:
        errors.append("candidate ranking training is prohibited")
    if manifest.get("online_selection") is not False:
        errors.append("online selection is prohibited")
    passed = (
        not errors
        and len(observed_ids) == int(manifest["valid_branches"])
        and len(observed_ids) == int(manifest["expected_branches"])
        and environment_errors == 0
    )
    validation = {
        "schema_version": "latentguard.lg_r2b0.dataset_validation.v1",
        "status": "pass" if passed else "fail",
        "runtime_identity": runtime_identity(),
        "dataset_manifest_sha256_before_validation": sha256_path(manifest_path),
        "validated_branches": len(observed_ids),
        "validated_anchors": len(continuation_by_anchor),
        "unresolved_outcomes": unresolved,
        "failure_outcomes": failures,
        "environment_errors": environment_errors,
        "branch_contamination": len(errors),
        "continuation_config_mismatches": sum(
            len(identities) != 1 for identities in continuation_by_anchor.values()
        ),
        "errors": errors,
        "unresolved_counted_as_failure": False,
        "final_seeds_accessed": False,
    }
    write_json(runtime / "dataset_validation.json", validation)
    manifest["validation"] = {
        "status": validation["status"],
        "validation_locator": "LG_R2B0_OUTPUT_ROOT/dataset_validation.json",
        "validated_branches": validation["validated_branches"],
        "branch_contamination": validation["branch_contamination"],
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": validation["status"],
                "branches": len(observed_ids),
                "errors": len(errors),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(9)


if __name__ == "__main__":
    main()
