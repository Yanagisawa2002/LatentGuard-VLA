"""Freeze and validate the outcome-independent LG-R1b task/seed registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1b_common import (
    read_yaml,
    repo_root,
    resolve_repo_path,
    sha256_path,
    write_json,
)

from latentguard.progress.lg_r1b import build_phase_jobs, validate_registry


def build_manifest(config_path: Path) -> dict[str, Any]:
    """Return a content-bound registry manifest without outcome fields."""

    registry = read_yaml(config_path)
    validation = validate_registry(registry)
    phase_jobs = {
        phase: build_phase_jobs(registry, phase)
        for phase in validation["phase_episode_counts"]
    }
    return {
        "schema_version": "latentguard.lg_r1b.task_seed_registry.v1",
        "status": "pass",
        "frozen_before_rollout": True,
        "registry_id": registry["registry_id"],
        "source_config": str(config_path.relative_to(repo_root()).as_posix()),
        "source_config_sha256": sha256_path(config_path),
        "baseline_commit": registry["baseline_commit"],
        "source_policy": registry["source_policy"],
        "frozen_sarm": registry["frozen_sarm"],
        "sealed_final_range": registry["sealed_final_range"],
        "adaptation_task_split": registry["adaptation_task_split"],
        "tasks": registry["tasks"],
        "phase_episode_counts": {
            phase: len(jobs) for phase, jobs in phase_jobs.items()
        },
        "validation": validation,
        "outcome_fields_present": False,
        "policy_training_authorized": False,
        "action_corruption_authorized": False,
        "failure_head_authorized": False,
        "intervention_authorized": False,
    }


def main() -> None:
    """Write the frozen public registry before remote execution."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1b/task_registry.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1b/task_seed_registry.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    payload = build_manifest(resolve_repo_path(args.config))
    if not args.validate_only:
        write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
