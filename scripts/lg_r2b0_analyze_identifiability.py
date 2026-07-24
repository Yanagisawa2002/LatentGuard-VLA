"""Compute descriptive within-anchor LG-R2b0 identifiability evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2b0_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    validate_execution_checkout,
    write_json,
)


def _rank(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x = _rank(left)
    y = _rank(right)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _pairwise_rank_agreement(
    left: list[float],
    right: list[float],
    *,
    epsilon: float,
) -> dict[str, int | float | None]:
    concordant = 0
    discordant = 0
    ties = 0
    for first, second in combinations(range(len(left)), 2):
        left_delta = left[first] - left[second]
        right_delta = right[first] - right[second]
        if abs(left_delta) <= epsilon or abs(right_delta) <= epsilon:
            ties += 1
        elif left_delta * right_delta > 0:
            concordant += 1
        else:
            discordant += 1
    total = concordant + discordant + ties
    return {
        "pairs": total,
        "concordant": concordant,
        "discordant": discordant,
        "ties": ties,
        "tie_rate": ties / total if total else None,
        "agreement_without_ties": (
            concordant / (concordant + discordant) if concordant + discordant else None
        ),
    }


def _terminal_score(outcome: str) -> float:
    return {
        "FAILURE": 0.0,
        "ENVIRONMENT_ERROR": np.nan,
        "UNRESOLVED_HORIZON": 1.0,
        "SUCCESS": 2.0,
    }[outcome]


def _terminal_category(outcomes: list[str]) -> str:
    values = set(outcomes)
    if values == {"SUCCESS"}:
        return "all_success"
    if values == {"FAILURE"}:
        return "all_failure"
    if "SUCCESS" in values and "FAILURE" in values:
        return "mixed_success_failure"
    if "SUCCESS" in values and "UNRESOLVED_HORIZON" in values:
        return "success_unresolved"
    if "FAILURE" in values and "UNRESOLVED_HORIZON" in values:
        return "failure_unresolved"
    if values == {"UNRESOLVED_HORIZON"}:
        return "all_unresolved"
    return "environment_error_or_other"


def main() -> None:
    """Analyze action and outcome variance without fitting a predictor."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r2b0/analysis.yaml"),
    )
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    validate_execution_checkout(args.expected_commit)
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    runtime = output_root()
    dataset_path = runtime / "dataset_manifest.json"
    dataset = read_json(dataset_path)
    diversity = read_json(runtime / "candidate_diversity_report.json")
    if dataset.get("status") != "pass":
        raise ValueError("counterfactual dataset is incomplete")
    if dataset.get("validation", {}).get("status") != "pass":
        raise ValueError("counterfactual dataset validation has not passed")
    branches_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for branch in dataset["branches"]:
        branches_by_anchor[str(branch["anchor_id"])].append(branch)
    action_pairs_by_anchor = {
        str(item["anchor_id"]): item["pairs"] for item in diversity["anchors"]
    }
    progress_threshold = float(config["meaningful_progress_spread"])
    rank_epsilon = float(config["rank_tie_epsilon"])
    anchor_reports: list[dict[str, Any]] = []
    rank_reports: list[dict[str, Any]] = []
    action_distances: list[float] = []
    progress_distances: list[float] = []
    terminal_distances: list[float] = []
    stage_divergence: Counter[str] = Counter()
    task_divergence: Counter[str] = Counter()
    for anchor_id, branches in sorted(branches_by_anchor.items()):
        branches.sort(key=lambda item: str(item["candidate_id"]))
        candidate_ids = [str(item["candidate_id"]) for item in branches]
        progress = {
            horizon: [
                float(item["progress_deltas"][f"progress_delta_{horizon}"])
                for item in branches
            ]
            for horizon in (7, 21, 49)
        }
        ranges = {
            str(horizon): max(values) - min(values)
            for horizon, values in progress.items()
        }
        stage_disagreement = {
            str(horizon): len(
                {
                    (
                        int(item["boundaries"][str(horizon)]["stage_id"]),
                        str(item["boundaries"][str(horizon)]["stage_name"]),
                    )
                    for item in branches
                }
            )
            > 1
            for horizon in (7, 21, 49)
        }
        events = {
            event: len(
                {bool(item["event_labels"].get(event, False)) for item in branches}
            )
            > 1
            for event in sorted(
                {event for item in branches for event in item["event_labels"]}
            )
        }
        outcomes = [str(item["terminal_outcome"]) for item in branches]
        terminal_category = _terminal_category(outcomes)
        terminal_scores = [_terminal_score(value) for value in outcomes]
        object_ranges: dict[str, float] = {}
        object_names = sorted(
            {
                name
                for item in branches
                for name in item["boundaries"]["49"]["object_pose_delta"]
            }
        )
        for name in object_names:
            values = [
                float(item["boundaries"]["49"]["object_pose_delta"][name])
                for item in branches
            ]
            object_ranges[name] = max(values) - min(values)
        local_event_disagreement = any(events.values())
        meaningful_progress = ranges["49"] >= progress_threshold
        outcome_divergence = (
            meaningful_progress
            or local_event_disagreement
            or len(set(outcomes)) > 1
            or any(stage_disagreement.values())
        )
        stage = str(branches[0]["anchor_stage"])
        task = f"{branches[0]['suite']}/task{branches[0]['task_id']}"
        if outcome_divergence:
            stage_divergence[stage] += 1
            task_divergence[task] += 1
        anchor_reports.append(
            {
                "anchor_id": anchor_id,
                "suite": branches[0]["suite"],
                "task_id": branches[0]["task_id"],
                "anchor_stage": stage,
                "candidate_count": len(branches),
                "progress_delta_ranges": ranges,
                "meaningful_progress_spread": meaningful_progress,
                "object_pose_delta_ranges_49": object_ranges,
                "stage_transition_disagreement": stage_disagreement,
                "local_event_disagreement": local_event_disagreement,
                "event_disagreement": events,
                "terminal_category": terminal_category,
                "terminal_outcomes": outcomes,
                "outcome_divergence": outcome_divergence,
            }
        )
        pair_rank = {
            "7_vs_21": _pairwise_rank_agreement(
                progress[7], progress[21], epsilon=rank_epsilon
            ),
            "21_vs_49": _pairwise_rank_agreement(
                progress[21], progress[49], epsilon=rank_epsilon
            ),
            "7_vs_49": _pairwise_rank_agreement(
                progress[7], progress[49], epsilon=rank_epsilon
            ),
            "49_vs_terminal": _pairwise_rank_agreement(
                progress[49], terminal_scores, epsilon=rank_epsilon
            ),
        }
        rank_reports.append(
            {
                "anchor_id": anchor_id,
                "candidate_ids": candidate_ids,
                "rank_agreement": pair_rank,
            }
        )
        branch_by_candidate = {str(item["candidate_id"]): item for item in branches}
        for pair in action_pairs_by_anchor.get(anchor_id, []):
            left = branch_by_candidate.get(str(pair["left_candidate_id"]))
            right = branch_by_candidate.get(str(pair["right_candidate_id"]))
            if left is None or right is None:
                continue
            action_distances.append(float(pair["full_chunk_normalized_l2"]))
            progress_distances.append(
                abs(
                    float(left["progress_deltas"]["progress_delta_49"])
                    - float(right["progress_deltas"]["progress_delta_49"])
                )
            )
            left_terminal = _terminal_score(str(left["terminal_outcome"]))
            right_terminal = _terminal_score(str(right["terminal_outcome"]))
            terminal_distances.append(abs(left_terminal - right_terminal))
    anchors_total = len(anchor_reports)
    meaningful_count = sum(
        bool(item["meaningful_progress_spread"]) for item in anchor_reports
    )
    local_event_count = sum(
        bool(item["local_event_disagreement"]) for item in anchor_reports
    )
    mixed_terminal = sum(
        item["terminal_category"] == "mixed_success_failure" for item in anchor_reports
    )
    non_task6_divergent = sum(
        bool(item["outcome_divergence"]) and int(item["task_id"]) != 6
        for item in anchor_reports
    )
    terminal_distribution = Counter(
        str(branch["terminal_outcome"]) for branch in dataset["branches"]
    )
    report = {
        "schema_version": "latentguard.lg_r2b0.outcome_diversity_report.v1",
        "status": "pass",
        "runtime_identity": runtime_identity(),
        "config_sha256": sha256_path(config_path),
        "dataset_manifest_sha256": sha256_path(dataset_path),
        "analysis_kind": "descriptive_within_anchor_only",
        "predictive_model_trained": False,
        "anchors": anchor_reports,
        "valid_anchors": anchors_total,
        "valid_branches": len(dataset["branches"]),
        "anchors_with_meaningful_short_horizon_progress_spread": (meaningful_count),
        "anchors_with_meaningful_short_horizon_progress_spread_ratio": (
            meaningful_count / anchors_total if anchors_total else 0.0
        ),
        "anchors_with_local_event_disagreement": local_event_count,
        "mixed_terminal_outcome_anchors": mixed_terminal,
        "non_task6_outcome_divergence_anchors": non_task6_divergent,
        "terminal_outcome_distribution": dict(terminal_distribution),
        "stage_divergence_distribution": dict(stage_divergence),
        "task_divergence_distribution": dict(task_divergence),
        "action_outcome_relationship": {
            "pair_count": len(action_distances),
            "action_l2_vs_progress_distance_spearman": _spearman(
                action_distances,
                progress_distances,
            ),
            "action_l2_vs_terminal_distance_spearman": _spearman(
                action_distances,
                terminal_distances,
            ),
        },
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    outcome_path = runtime / "outcome_diversity_report.json"
    write_json(outcome_path, report)
    tie_values = [
        float(metric["tie_rate"])
        for item in rank_reports
        for metric in item["rank_agreement"].values()
        if metric["tie_rate"] is not None
    ]
    agreement_values = [
        float(metric["agreement_without_ties"])
        for item in rank_reports
        for metric in item["rank_agreement"].values()
        if metric["agreement_without_ties"] is not None
    ]
    rank_report = {
        "schema_version": "latentguard.lg_r2b0.rank_stability_report.v1",
        "status": "pass",
        "runtime_identity": runtime_identity(),
        "outcome_diversity_report_sha256": sha256_path(outcome_path),
        "rank_tie_epsilon": rank_epsilon,
        "anchors": rank_reports,
        "aggregate_tie_rate": (float(np.mean(tie_values)) if tie_values else 1.0),
        "aggregate_agreement_without_ties": (
            float(np.mean(agreement_values)) if agreement_values else None
        ),
        "black_box_ranker_trained": False,
    }
    rank_path = runtime / "rank_stability_report.json"
    write_json(rank_path, rank_report)
    print(
        json.dumps(
            {
                "status": "pass",
                "anchors": anchors_total,
                "progress_divergent": meaningful_count,
                "event_divergent": local_event_count,
                "mixed_terminal": mixed_terminal,
                "aggregate_tie_rate": rank_report["aggregate_tie_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
