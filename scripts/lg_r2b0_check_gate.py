"""Evaluate the complete pre-registered LG-R2b1 promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r2b0_common import (
    read_json,
    runtime_identity,
    sha256_path,
    validate_execution_checkout,
    write_json,
)

from latentguard.counterfactual.gates import (
    CounterfactualGateEvidence,
    evaluate_counterfactual_gate,
)


def main() -> None:
    """Authorize LG-R2b1 only for accepted real counterfactual evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    validate_execution_checkout(args.expected_commit)
    outcome_path = args.summary.resolve()
    runtime = outcome_path.parent
    outcome = read_json(outcome_path)
    rank = read_json(runtime / "rank_stability_report.json")
    restore = read_json(runtime / "state_restore_validation.json")
    source = read_json(runtime / "candidate_source_manifest.json")
    candidates = read_json(runtime / "candidate_manifest.json")
    diversity = read_json(runtime / "candidate_diversity_report.json")
    dataset = read_json(runtime / "dataset_manifest.json")
    validation = read_json(runtime / "dataset_validation.json")
    required = (
        source,
        restore,
        candidates,
        diversity,
        dataset,
        validation,
        outcome,
        rank,
    )
    if any(item.get("status") != "pass" for item in required):
        raise ValueError("promotion gate inputs are incomplete")
    evidence = CounterfactualGateEvidence(
        restore_mismatches=(
            int(restore["restore_failures"])
            + int(restore.get("render_mismatches", 0))
            + int(restore.get("post_render_state_mismatches", 0))
            + int(restore["terminal_result_mismatches"])
            + int(restore["task_predicate_mismatches"])
        ),
        branch_contamination=int(validation["branch_contamination"]),
        candidate_metadata_completeness=float(
            dataset["candidate_metadata_completeness"]
        ),
        identity_completeness=float(dataset["policy_checkpoint_identity_completeness"]),
        policy_generated_candidate_ratio=float(
            dataset["policy_generated_candidate_ratio"]
        ),
        synthetic_candidate_ratio=float(dataset["synthetic_candidate_ratio"]),
        valid_anchors=int(outcome["valid_anchors"]),
        valid_branches=int(outcome["valid_branches"]),
        anchors_with_three_distinct_ratio=float(
            diversity["anchors_with_at_least_three_distinct_ratio"]
        ),
        exact_duplicate_rate=float(diversity["exact_duplicate_rate"]),
        meaningful_progress_spread_ratio=float(
            outcome["anchors_with_meaningful_short_horizon_progress_spread_ratio"]
        ),
        local_event_disagreement_anchors=int(
            outcome["anchors_with_local_event_disagreement"]
        ),
        mixed_terminal_outcome_anchors=int(outcome["mixed_terminal_outcome_anchors"]),
        non_task6_outcome_divergence=(
            int(outcome["non_task6_outcome_divergence_anchors"]) > 0
        ),
        ranking_tie_rate=float(rank["aggregate_tie_rate"]),
    )
    gate = evaluate_counterfactual_gate(evidence)
    gate.update(
        {
            "runtime_identity": runtime_identity(),
            "result": "A" if gate["LG_R2B1_AUTHORIZED"] else "B",
            "result_definition": (
                "real same-state counterfactual data are identifiable"
                if gate["LG_R2B1_AUTHORIZED"]
                else "real candidates exist but registered outcome/rank gates fail"
            ),
            "input_sha256": {
                "candidate_source_manifest": sha256_path(
                    runtime / "candidate_source_manifest.json"
                ),
                "state_restore_validation": sha256_path(
                    runtime / "state_restore_validation.json"
                ),
                "candidate_diversity_report": sha256_path(
                    runtime / "candidate_diversity_report.json"
                ),
                "dataset_manifest": sha256_path(runtime / "dataset_manifest.json"),
                "outcome_diversity_report": sha256_path(outcome_path),
                "rank_stability_report": sha256_path(
                    runtime / "rank_stability_report.json"
                ),
            },
            "no_training": {
                "candidate_ranker": True,
                "failure_head": True,
                "world_model": True,
                "reward_model": True,
                "uncertainty_model": True,
            },
            "no_online_selection_or_intervention": True,
            "final_seeds_accessed": False,
        }
    )
    path = runtime / "lg_r2b1_gate.json"
    write_json(path, gate)
    print(
        json.dumps(
            {
                "status": gate["status"],
                "result": gate["result"],
                "LG_R2B1_AUTHORIZED": gate["LG_R2B1_AUTHORIZED"],
                "output": str(path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
