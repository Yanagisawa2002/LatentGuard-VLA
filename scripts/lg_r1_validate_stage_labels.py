"""Apply the fail-closed human review gate to LG-R1 stage labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1_common import read_json, resolve_repo_path, write_json


def validate_report(report: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit review decisions and evaluate frozen QA thresholds."""

    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("review decisions must be a list")
    required = set(report["review_queue"])
    by_episode: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("review decisions must be mappings")
        episode_id = str(decision["episode_id"])
        if episode_id in by_episode:
            raise ValueError(f"duplicate review decision: {episode_id}")
        by_episode[episode_id] = decision
    missing = required - set(by_episode)
    if missing:
        raise ValueError(f"missing review decisions: {sorted(missing)}")
    reviewed = [by_episode[episode_id] for episode_id in sorted(required)]
    critical = int(report["automated_critical_errors"]) + sum(
        int(item.get("critical_errors", 0)) for item in reviewed
    )
    minor = sum(int(item.get("minor_disagreements", 0)) for item in reviewed)
    minor_rate = minor / len(reviewed) if reviewed else 1.0
    per_task: dict[tuple[str, int], int] = {}
    summary_by_id = {item["episode_id"]: item for item in report["episode_summaries"]}
    for episode_id in required:
        summary = summary_by_id[episode_id]
        key = (str(summary["suite"]), int(summary["task_id"]))
        per_task[key] = per_task.get(key, 0) + 1
    task_review_gate = (
        all(
            count >= int(report["minimum_review_episodes_per_task"])
            for count in per_task.values()
        )
        and len(per_task) == 8
    )
    checks = {
        "critical_errors": critical <= int(report["critical_error_limit"]),
        "minor_disagreement_rate": minor_rate
        <= float(report["minor_disagreement_rate_limit"]),
        "minimum_review_total": len(reviewed)
        >= int(report["minimum_review_episodes_total"]),
        "minimum_review_per_task": task_review_gate,
        "privileged_input_exclusion": not report["privileged_fields_in_model_input"],
        "timeout_semantic": (report["timeout_is_terminal_failure"] is False),
        "future_outcome_backfill": (report["future_terminal_backfill"] is False),
    }
    validated = json.loads(json.dumps(report))
    validated.update(
        {
            "status": "pass" if all(checks.values()) else "fail",
            "reviewed_episodes": len(reviewed),
            "review_decisions": reviewed,
            "reviewer": review.get("reviewer", "unspecified"),
            "critical_errors": critical,
            "minor_disagreements": minor,
            "minor_disagreement_rate": minor_rate,
            "checks": checks,
        }
    )
    return validated


def main() -> None:
    """Validate a complete explicit review; never self-approve labels."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = resolve_repo_path(args.manifest)
    validated = validate_report(
        read_json(manifest_path),
        read_json(resolve_repo_path(args.review)),
    )
    output = (
        resolve_repo_path(args.output) if args.output is not None else manifest_path
    )
    write_json(output, validated)
    print(json.dumps(validated, sort_keys=True))
    if validated["status"] != "pass":
        raise RuntimeError("stage annotation QA failed")


if __name__ == "__main__":
    main()
