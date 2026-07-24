"""Build deterministic episode- and seed-grouped LG-R2a folds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from _lg_r2a_common import (
    output_root,
    r1c_root,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    source_root,
    write_json,
)

from latentguard.action_conditioning.splits import (
    EpisodeRecord,
    assign_grouped_folds,
    validate_grouped_folds,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--r1c-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Assign folds and bind their episode/window membership."""

    args = parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    if args.dry_run:
        print({"status": "dry_run", "folds": config["folds"]})
        return
    source = source_root(args.source_root)
    reward = r1c_root(args.r1c_root)
    output = output_root(args.output_dir)
    current = read_jsonl(source / "progress_dataset" / "current_samples.jsonl")
    cache = np.load(
        reward / "vlajepa-probe" / "frozen_qwen_features.npz", allow_pickle=False
    )
    cache_counts = Counter(
        str(sample_id).rsplit(":", 1)[0] for sample_id in cache["sample_id"].tolist()
    )
    by_episode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in current:
        by_episode[str(row["episode_id"])].append(row)
    failures = {
        str(row["episode_id"]): row
        for row in read_json(resolve_repo_path(config["failure_registry"]))["episodes"]
    }
    records = []
    for episode_id, rows in sorted(by_episode.items()):
        first = rows[0]
        mean_progress = float(np.mean([float(row["overall_progress"]) for row in rows]))
        failure = failures.get(episode_id)
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                seed=int(first["seed"]),
                task=f"{first['suite']}/task{int(first['task_id'])}",
                success=bool(first["episode_success"]),
                taxonomy=(
                    str(failure["primary_taxonomy"]) if failure is not None else None
                ),
                progress_bin=min(4, int(mean_progress * 5)),
                windows=int(cache_counts[episode_id]),
            )
        )
    assignments = assign_grouped_folds(
        records,
        folds=int(config["folds"]),
        assignment_seed=int(config["assignment_seed"]),
    )
    validation = validate_grouped_folds(
        records, assignments, folds=int(config["folds"])
    )
    folds = int(config["folds"])
    manifest = {
        "schema_version": "latentguard.lg_r2a.cv_split_manifest.v1",
        "status": "pass",
        "algorithm": (
            "deterministic greedy seed-group assignment; failures first; "
            "then taxonomy, task, and cached-window balancing"
        ),
        "assignment_seed": int(config["assignment_seed"]),
        "group_unit": "episode_and_seed",
        "fold_count": folds,
        "protocol": [
            {
                "outer_fold": fold,
                "test_fold": fold,
                "validation_fold": (fold + 1) % folds,
                "train_folds": [
                    value
                    for value in range(folds)
                    if value not in {fold, (fold + 1) % folds}
                ],
                "test_consumption": "exactly_once_after_validation_selection",
            }
            for fold in range(folds)
        ],
        "validation": validation,
        "assignments": [
            {
                "episode_id": record.episode_id,
                "seed": record.seed,
                "task": record.task,
                "success": record.success,
                "taxonomy": record.taxonomy,
                "progress_bin": record.progress_bin,
                "cached_windows": record.windows,
                "fold": assignments[record.episode_id],
            }
            for record in records
        ],
        "original_split": {
            "preserved": True,
            "role": "compatibility reporting only",
            "used_as_unique_supervised_conclusion": False,
        },
        "held_out_failure_task": {
            "status": "not_pre_registered_for_gate",
            "role": "exploratory_only_if_support_allows",
        },
    }
    write_json(output / "cv_split_manifest.json", manifest)
    print(
        {
            "status": "pass",
            "fold_failures": [fold["failures"] for fold in validation["folds"]],
        }
    )


if __name__ == "__main__":
    main()
