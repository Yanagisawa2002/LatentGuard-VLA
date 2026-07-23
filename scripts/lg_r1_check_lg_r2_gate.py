"""Evaluate the fail-closed LG-R2 failure-data authorization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r1_common import read_json, resolve_repo_path, write_json

from latentguard.progress.gates import LG_R2GateInput, evaluate_lg_r2_gate


def main() -> None:
    """Write the complete gate breakdown without authorizing training itself."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sarm-results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    dataset = read_json(manifest_path)
    sarm_path = (
        resolve_repo_path(args.sarm_results)
        if args.sarm_results is not None
        else manifest_path.parent / "sarm_results.json"
    )
    sarm = read_json(sarm_path)
    evidence = LG_R2GateInput(
        valid_rollout_episodes=int(dataset["valid_rollout_episodes"]),
        tasks=int(dataset["tasks"]),
        suites=int(dataset["suites"]),
        natural_failed_episodes=int(dataset["natural_failed_episodes"]),
        failure_windows=int(dataset["failure_windows"]),
        successful_windows=int(dataset["successful_windows"]),
        stage_annotation_passed=bool(dataset["stage_annotation_passed"]),
        sarm_evaluation_complete=sarm.get("status") == "pass",
        episode_split_leakage=int(dataset["episode_split_leakage"]),
        processor_identity_completeness=float(
            dataset["processor_identity_completeness"]
        ),
        checkpoint_identity_completeness=float(
            dataset["checkpoint_identity_completeness"]
        ),
    )
    result = evaluate_lg_r2_gate(evidence)
    result["authorization_scope"] = (
        "data sufficiency only; starting LG-R2 still requires explicit user "
        "authorization and a new milestone"
    )
    result["failure_head_training_started"] = False
    output = (
        resolve_repo_path(args.output)
        if args.output is not None
        else manifest_path.parent / "lg_r2_gate.json"
    )
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
