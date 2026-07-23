"""Generate task-specific stage labels from privileged rollout sidecars."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _lg_r1_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    write_json,
)

from latentguard.progress.serialization import write_jsonl_atomic
from latentguard.progress.stages.libero_registry import LiberoStageRegistry


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"empty sidecar: {path}")
    return rows


def _review_queue(
    episodes: list[dict[str, Any]], *, per_task: int, total: int
) -> list[str]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for episode in episodes:
        key = (str(episode["suite"]), int(episode["task_id"]))
        grouped.setdefault(key, []).append(episode)
    chosen: list[str] = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda item: (
                bool(item["success"]),
                int(item["seed"]),
            ),
        )
        outcomes: list[dict[str, Any]] = []
        for success in (False, True):
            match = next(
                (item for item in items if bool(item["success"]) is success),
                None,
            )
            if match is not None:
                outcomes.append(match)
        for item in outcomes + items:
            episode_id = str(item["episode_id"])
            if episode_id not in chosen:
                chosen.append(episode_id)
            if (
                sum(candidate["episode_id"] in chosen for candidate in items)
                >= per_task
            ):
                break
    for episode in sorted(
        episodes, key=lambda item: (str(item["suite"]), int(item["seed"]))
    ):
        episode_id = str(episode["episode_id"])
        if episode_id not in chosen:
            chosen.append(episode_id)
        if len(chosen) >= total:
            break
    return chosen


def main() -> None:
    """Build deterministic labels and a pre-training QA review queue."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/stage_labels.yaml"),
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
    rollout_manifest_path = (
        resolve_repo_path(args.rollout_manifest)
        if args.rollout_manifest is not None
        else runtime / "rollout_manifest.json"
    )
    rollout_manifest = read_json(rollout_manifest_path)
    if rollout_manifest.get("status") != "pass":
        raise ValueError("rollout manifest is not complete")
    label_root = runtime / "stage_labels"
    label_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for episode in rollout_manifest["episodes"]:
        episode_id = str(episode["episode_id"])
        sidecar_path = runtime / "privileged_sidecars" / f"{episode_id}.jsonl"
        rows = _read_jsonl(sidecar_path)
        adapter = registry.resolve(str(episode["suite"]), int(episode["task_id"]))
        labels: list[dict[str, Any]] = []
        stage_ids: list[int] = []
        progress_values: list[float] = []
        regressions = 0
        determinism_mismatches = 0
        initial = rows[0]["initial_privileged_state"]
        metadata = {
            "initial_privileged_state": initial,
            "distance_thresholds": config["distance_thresholds"],
        }
        for row in rows:
            try:
                pre_label = adapter.label_step(row["pre_state"], metadata)
                pre_repeat = adapter.label_step(row["pre_state"], metadata)
                post_label = adapter.label_step(row["post_state"], metadata)
                if pre_label != pre_repeat:
                    determinism_mismatches += 1
                if stage_ids and (
                    pre_label.stage_id < stage_ids[-1]
                    or pre_label.overall_progress
                    < progress_values[-1] - float(config["progress_epsilon"])
                ):
                    regressions += 1
                stage_ids.append(pre_label.stage_id)
                progress_values.append(pre_label.overall_progress)
                labels.append(
                    {
                        "schema_version": ("latentguard.lg_r1.stage_step.v1"),
                        "episode_id": episode_id,
                        "step_index": int(row["step_index"]),
                        "pre_label": asdict(pre_label),
                        "post_label": asdict(post_label),
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
        if labels:
            write_jsonl_atomic(label_root / f"{episode_id}.jsonl", labels)
            final_post = labels[-1]["post_label"]
            success = bool(episode["success"])
            if success and (
                final_post["terminal_success"] is not True
                or final_post["overall_progress"] != 1.0
            ):
                errors.append(
                    {
                        "episode_id": episode_id,
                        "error": "terminal success label is not progress 1",
                    }
                )
            summaries.append(
                {
                    "episode_id": episode_id,
                    "suite": episode["suite"],
                    "task_id": episode["task_id"],
                    "seed": episode["seed"],
                    "success": success,
                    "frame_count": len(labels),
                    "stage_ids_observed": sorted(set(stage_ids)),
                    "minimum_progress": min(progress_values),
                    "maximum_progress": max(progress_values),
                    "regression_events": regressions,
                    "determinism_mismatches": (determinism_mismatches),
                    "final_pre_label": labels[-1]["pre_label"],
                    "final_post_label": final_post,
                    "label_locator": (f"stage_label_root/{episode_id}.jsonl"),
                }
            )
    review = _review_queue(
        summaries,
        per_task=int(config["minimum_review_episodes_per_task"]),
        total=int(config["minimum_review_episodes_total"]),
    )
    report = {
        "schema_version": ("latentguard.lg_r1.stage_annotation_report.v1"),
        "status": "pending_human_review" if not errors else "fail",
        "runtime_identity": runtime_identity(),
        "rollout_runtime_identity": rollout_manifest["runtime_identity"],
        "rollout_manifest_sha256": sha256_path(rollout_manifest_path),
        "task_registry_sha256": rollout_manifest["task_registry_sha256"],
        "automated_critical_errors": len(errors),
        "critical_error_limit": int(config["critical_error_limit"]),
        "minor_disagreement_rate_limit": float(config["minor_disagreement_rate_limit"]),
        "labeled_episodes": len(summaries),
        "labeled_steps": sum(int(item["frame_count"]) for item in summaries),
        "review_queue": review,
        "minimum_review_episodes_per_task": int(
            config["minimum_review_episodes_per_task"]
        ),
        "minimum_review_episodes_total": int(config["minimum_review_episodes_total"]),
        "episode_summaries": summaries,
        "errors": errors,
        "privileged_fields_in_model_input": [],
        "timeout_is_terminal_failure": False,
        "future_terminal_backfill": False,
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
