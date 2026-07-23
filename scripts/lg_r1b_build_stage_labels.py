"""Build task-specific LG-R1b labels and a structured QA review queue."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _lg_r1b_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    write_json,
)
from lg_r1_build_stage_labels import _read_jsonl

from latentguard.progress.serialization import write_jsonl_atomic
from latentguard.progress.stages.libero_registry import LiberoStageRegistry


def _review_queue(
    episodes: list[dict[str, Any]],
    *,
    minimum_per_task: int,
    success_per_failed_task: int,
    failure_per_failed_task: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for episode in episodes:
        key = str(episode["suite"]), int(episode["task_id"])
        grouped.setdefault(key, []).append(episode)
    selected: list[str] = []
    shortages: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(item["seed"]))
        successes = [item for item in ordered if bool(item["success"])]
        failures = [item for item in ordered if not bool(item["success"])]
        required = ordered[:minimum_per_task]
        if failures:
            required = (
                successes[:success_per_failed_task] + failures[:failure_per_failed_task]
            )
            if len(successes) < success_per_failed_task:
                shortages.append(
                    {
                        "task": list(key),
                        "outcome": "success",
                        "required": success_per_failed_task,
                        "available": len(successes),
                    }
                )
            if len(failures) < failure_per_failed_task:
                required.extend(failures)
        for item in required:
            episode_id = str(item["episode_id"])
            if episode_id not in selected:
                selected.append(episode_id)
    return selected, shortages


def _plateau_start(
    progress: list[float], minimum_steps: int, epsilon: float
) -> int | None:
    for end in range(minimum_steps, len(progress)):
        start = end - minimum_steps
        if max(progress[start : end + 1]) - min(progress[start : end + 1]) <= epsilon:
            return start
    return None


def main() -> None:
    """Regenerate deterministic labels without future-outcome backfill."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1b/stages.yaml"),
    )
    parser.add_argument("--rollout-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    registry_config = read_yaml(resolve_repo_path(Path(str(config["task_registry"]))))
    registry = LiberoStageRegistry(registry_config["tasks"])
    runtime = (
        resolve_repo_path(args.output_root)
        if args.output_root is not None
        else output_root()
    )
    rollout_path = (
        resolve_repo_path(args.rollout_manifest)
        if args.rollout_manifest is not None
        else runtime / "rollout_manifest.json"
    )
    rollout = read_json(rollout_path)
    if rollout.get("status") != "pass":
        raise ValueError("aggregate rollout manifest is incomplete")
    label_root = runtime / "stage_labels"
    label_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    epsilon = float(config["progress_epsilon"])
    plateau_steps = int(config["plateau_min_steps"])
    for episode in rollout["episodes"]:
        episode_id = str(episode["episode_id"])
        rows = _read_jsonl(runtime / "privileged_sidecars" / f"{episode_id}.jsonl")
        adapter = registry.resolve(str(episode["suite"]), int(episode["task_id"]))
        metadata = {
            "initial_privileged_state": rows[0]["initial_privileged_state"],
            "distance_thresholds": config["distance_thresholds"],
        }
        labels: list[dict[str, Any]] = []
        stages: list[int] = []
        progress: list[float] = []
        regressions: list[int] = []
        deterministic_mismatches = 0
        for row in rows:
            try:
                pre = adapter.label_step(row["pre_state"], metadata)
                repeated = adapter.label_step(row["pre_state"], metadata)
                post = adapter.label_step(row["post_state"], metadata)
                if pre != repeated:
                    deterministic_mismatches += 1
                if stages and (
                    pre.stage_id < stages[-1]
                    or pre.overall_progress < progress[-1] - epsilon
                ):
                    regressions.append(int(row["step_index"]))
                stages.append(pre.stage_id)
                progress.append(pre.overall_progress)
                labels.append(
                    {
                        "schema_version": "latentguard.lg_r1b.stage_step.v1",
                        "episode_id": episode_id,
                        "step_index": int(row["step_index"]),
                        "pre_label": asdict(pre),
                        "post_label": asdict(post),
                        "current_sample": {
                            "dataset_locator": episode["dataset_locator"],
                            "frame_index": int(row["step_index"]),
                            "instruction": episode["instruction"],
                            "model_input_keys": [
                                "observation.images.image",
                                "observation.images.image2",
                                "observation.state",
                                "task",
                            ],
                        },
                        "executed_action_window": {
                            "executed_action": row["executed_action"],
                            "action_mask": row["action_mask"],
                            "action_is_pad": row["action_is_pad"],
                        },
                        "stage_regression": bool(
                            regressions and regressions[-1] == int(row["step_index"])
                        ),
                        "privileged_fields_in_model_input": [],
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "episode_id": episode_id,
                        "step_index": row.get("step_index"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
        if not labels:
            continue
        write_jsonl_atomic(label_root / f"{episode_id}.jsonl", labels)
        plateau_start = _plateau_start(progress, plateau_steps, epsilon)
        first_relevant = regressions[0] if regressions else plateau_start
        final = labels[-1]["post_label"]
        if bool(episode["success"]) and (
            final["terminal_success"] is not True
            or float(final["overall_progress"]) != 1.0
        ):
            errors.append(
                {
                    "episode_id": episode_id,
                    "error": "terminal success is not progress 1",
                }
            )
        summaries.append(
            {
                "episode_id": episode_id,
                "suite": episode["suite"],
                "task_id": episode["task_id"],
                "seed": episode["seed"],
                "success": bool(episode["success"]),
                "termination_reason": episode["termination_reason"],
                "frame_count": len(labels),
                "stage_ids_observed": sorted(set(stages)),
                "minimum_progress": min(progress),
                "maximum_progress": max(progress),
                "regression_events": len(regressions),
                "regression_steps": regressions,
                "plateau_start_step": plateau_start,
                "first_failure_relevant_step": (
                    first_relevant if not bool(episode["success"]) else None
                ),
                "determinism_mismatches": deterministic_mismatches,
                "final_pre_label": labels[-1]["pre_label"],
                "final_post_label": final,
                "label_locator": f"stage_label_root/{episode_id}.jsonl",
            }
        )
    review, shortages = _review_queue(
        summaries,
        minimum_per_task=int(config["minimum_review_episodes_per_task"]),
        success_per_failed_task=int(config["failed_task_minimum_success_reviews"]),
        failure_per_failed_task=int(config["failed_task_minimum_failure_reviews"]),
    )
    report = {
        "schema_version": "latentguard.lg_r1b.stage_annotation_report.v1",
        "status": "pending_structured_review" if not errors else "fail",
        "runtime_identity": runtime_identity(),
        "rollout_manifest_sha256": sha256_path(rollout_path),
        "task_registry_sha256": rollout["task_registry_sha256"],
        "automated_critical_errors": len(errors),
        "critical_error_limit": int(config["critical_error_limit"]),
        "minor_disagreement_rate_limit": float(config["minor_disagreement_rate_limit"]),
        "deterministic_mismatch_limit": int(config["deterministic_mismatch_limit"]),
        "labeled_episodes": len(summaries),
        "labeled_steps": sum(int(item["frame_count"]) for item in summaries),
        "review_queue": review,
        "review_outcome_shortages": shortages,
        "minimum_review_episodes_per_task": int(
            config["minimum_review_episodes_per_task"]
        ),
        "failed_task_minimum_success_reviews": int(
            config["failed_task_minimum_success_reviews"]
        ),
        "failed_task_minimum_failure_reviews": int(
            config["failed_task_minimum_failure_reviews"]
        ),
        "episode_summaries": summaries,
        "errors": errors,
        "privileged_fields_in_model_input": [],
        "timeout_is_terminal_failure": False,
        "future_terminal_backfill": False,
        "stage_uses_step_index": False,
    }
    report_path = (
        resolve_repo_path(args.report)
        if args.report is not None
        else runtime / "stage_annotation_report.json"
    )
    write_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    if errors:
        raise RuntimeError(errors[0]["error"])


if __name__ == "__main__":
    main()
