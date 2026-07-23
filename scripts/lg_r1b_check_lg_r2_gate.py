"""Evaluate the strict and limited LG-R2 authorization gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r1b_common import output_root, read_json, resolve_repo_path, write_json

from latentguard.progress.lg_r1b import LGR2GateInput, evaluate_lg_r2_gate


def main() -> None:
    """Join only completed compact evidence and fail closed on missing fields."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    dataset_path = (
        resolve_repo_path(args.manifest)
        if args.manifest is not None
        else runtime / "dataset_manifest.json"
    )
    dataset = read_json(dataset_path)
    windows = read_json(runtime / "failure_window_manifest.json")
    failure = read_json(runtime / "failure_registry.json")
    zero = read_json(runtime / "sarm_zero_shot_results.json")
    test_failures = sum(
        str(item["adaptation_split"]) == "test" for item in failure["episodes"]
    )
    categories = {
        str(category)
        for item in failure["episodes"]
        for category in item["taxonomy"]
        if category not in {"HORIZON_EXHAUSTION", "UNKNOWN"}
    }
    result = evaluate_lg_r2_gate(
        LGR2GateInput(
            valid_total_rollouts=int(dataset["valid_rollout_episodes"]),
            tasks=int(dataset["tasks"]),
            suites=int(dataset["suites"]),
            natural_failed_episodes=int(dataset["natural_failed_episodes"]),
            failed_tasks=int(dataset["failed_tasks"]),
            failure_windows=int(windows["failure_windows"]),
            matched_success_windows=int(windows["matched_success_windows"]),
            stage_annotation_passed=bool(dataset["stage_annotation_passed"]),
            held_out_zero_shot_complete=zero.get("status") == "pass",
            episode_leakage=int(windows["episode_leakage"]),
            seed_leakage=int(windows["seed_leakage"]),
            processor_identity_completeness=float(
                dataset["processor_identity_completeness"]
            ),
            checkpoint_identity_completeness=float(
                dataset["checkpoint_identity_completeness"]
            ),
            infrastructure_failures_excluded=bool(
                failure["environment_errors_excluded"]
            ),
            failure_categories=len(categories),
            held_out_test_failures=test_failures,
        )
    )
    result["failure_categories_for_diversity"] = sorted(categories)
    result["source_dataset_manifest"] = str(dataset_path.as_posix())
    destination = (
        resolve_repo_path(args.output)
        if args.output is not None
        else runtime / "lg_r2_gate.json"
    )
    write_json(destination, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
