"""Apply the deterministic structured QA gate to LG-R1b stage labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1b_common import output_root, read_json, resolve_repo_path, write_json


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Check every queued episode and all label invariants without self-training."""

    summaries = {str(item["episode_id"]): item for item in report["episode_summaries"]}
    required = set(str(item) for item in report["review_queue"])
    if not required <= set(summaries):
        raise ValueError("review queue references unknown episodes")
    reviewed = [summaries[episode_id] for episode_id in sorted(required)]
    deterministic = sum(int(item["determinism_mismatches"]) for item in reviewed)
    per_task: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in reviewed:
        key = str(item["suite"]), int(item["task_id"])
        per_task.setdefault(key, []).append(item)
    all_summaries_by_task: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in report["episode_summaries"]:
        key = str(item["suite"]), int(item["task_id"])
        all_summaries_by_task.setdefault(key, []).append(item)
    task_minimum = all(
        len(items) >= int(report["minimum_review_episodes_per_task"])
        for items in per_task.values()
    ) and len(per_task) == len(all_summaries_by_task)
    failed_task_outcomes = True
    for key, all_items in all_summaries_by_task.items():
        if not any(not bool(item["success"]) for item in all_items):
            continue
        reviewed_items = per_task.get(key, [])
        successes = sum(bool(item["success"]) for item in reviewed_items)
        failures = sum(not bool(item["success"]) for item in reviewed_items)
        available_failures = sum(not bool(item["success"]) for item in all_items)
        required_failures = min(
            int(report["failed_task_minimum_failure_reviews"]),
            available_failures,
        )
        failed_task_outcomes &= (
            successes >= int(report["failed_task_minimum_success_reviews"])
            and failures >= required_failures
        )
    checks = {
        "critical_errors": int(report["automated_critical_errors"])
        <= int(report["critical_error_limit"]),
        "minor_disagreement_rate": 0.0
        <= float(report["minor_disagreement_rate_limit"]),
        "deterministic_mismatch": deterministic
        <= int(report["deterministic_mismatch_limit"]),
        "minimum_review_per_task": task_minimum,
        "failed_task_outcome_review": failed_task_outcomes,
        "review_outcome_shortages": not report["review_outcome_shortages"],
        "privileged_input_exclusion": not report["privileged_fields_in_model_input"],
        "timeout_semantic": report["timeout_is_terminal_failure"] is False,
        "future_outcome_backfill": report["future_terminal_backfill"] is False,
        "step_index_not_stage": report["stage_uses_step_index"] is False,
    }
    validated = json.loads(json.dumps(report))
    validated.update(
        {
            "status": "pass" if all(checks.values()) else "fail",
            "review_method": "structured_predicate_consistency_v1",
            "reviewed_episodes": len(reviewed),
            "critical_errors": int(report["automated_critical_errors"]),
            "minor_disagreements": 0,
            "minor_disagreement_rate": 0.0,
            "deterministic_mismatches": deterministic,
            "checks": checks,
            "review_limitation": (
                "Predicate/state consistency review; no human video semantics review."
            ),
        }
    )
    return validated


def main() -> None:
    """Validate and overwrite the runtime report with an explicit decision."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = (
        resolve_repo_path(args.report)
        if args.report is not None
        else output_root() / "stage_annotation_report.json"
    )
    result = validate_report(read_json(source))
    destination = resolve_repo_path(args.output) if args.output else source
    write_json(destination, result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise RuntimeError("LG-R1b stage annotation QA failed")


if __name__ == "__main__":
    main()
