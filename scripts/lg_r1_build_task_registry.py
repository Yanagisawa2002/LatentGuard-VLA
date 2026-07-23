"""Freeze and validate the LG-R1 multi-task seed registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1_common import (
    read_yaml,
    repo_root,
    resolve_repo_path,
    sha256_path,
    write_json,
)

from latentguard.progress.rollout import (
    build_rollout_jobs,
    validate_seed_isolation,
)


def build_manifest(config_path: Path) -> dict[str, Any]:
    """Build the outcome-free public task and seed manifest."""

    registry = read_yaml(config_path)
    validate_seed_isolation(registry)
    pilot = build_rollout_jobs(registry, include_extension=False)
    all_jobs = build_rollout_jobs(registry, include_extension=True)
    tasks = registry.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 8:
        raise ValueError("LG-R1 registry must freeze exactly eight tasks")
    suites = {str(task["suite"]) for task in tasks}
    if len(suites) != 4:
        raise ValueError("LG-R1 registry must freeze exactly four suites")
    source_policy = registry.get("source_policy")
    if not isinstance(source_policy, dict):
        raise ValueError("source_policy must be a mapping")
    return {
        "schema_version": "latentguard.lg_r1.task_seed_registry.v1",
        "status": "pass",
        "frozen_before_rollout": registry.get("frozen_before_rollout") is True,
        "registry_id": registry.get("registry_id"),
        "source_config": str(config_path.relative_to(repo_root()).as_posix()),
        "source_config_sha256": sha256_path(config_path),
        "source_policy": source_policy,
        "task_count": len(tasks),
        "suite_count": len(suites),
        "tasks": tasks,
        "seed_schedule": registry["seed_schedule"],
        "pilot_episode_count": len(pilot),
        "extended_episode_count": len(all_jobs),
        "historical_seed_overlap": [],
        "sealed_final_seed_overlap": [],
        "outcome_fields_present": False,
    }


def main() -> None:
    """Validate and serialize the frozen task/seed design."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/task_registry.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1/task_seed_registry.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = resolve_repo_path(args.config)
    payload = build_manifest(config)
    if not args.validate_only:
        write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
