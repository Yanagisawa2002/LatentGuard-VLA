"""Frozen LG-R1b task, pilot, failure-window, and promotion contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FINAL_SEEDS = frozenset(range(900000, 900100))
FAILURE_TAXONOMY = (
    "HORIZON_EXHAUSTION",
    "STAGNATION",
    "STAGE_REGRESSION",
    "WRONG_OBJECT",
    "WRONG_TARGET",
    "MISSED_GRASP",
    "OBJECT_DROP",
    "FAILED_PLACEMENT",
    "FAILED_OPEN_CLOSE",
    "SUBGOAL_ORDER_ERROR",
    "FALSE_PROGRESS",
    "ENVIRONMENT_ERROR",
    "UNKNOWN",
)


@dataclass(frozen=True)
class PilotGateInput:
    """Outcome summary for one complete, uniformly sampled pilot group."""

    valid_episodes: int
    expected_episodes: int
    natural_failures: int
    infrastructure_errors: int


@dataclass(frozen=True)
class LGR2GateInput:
    """LG-R1b evidence required before any later failure-head milestone."""

    valid_total_rollouts: int
    tasks: int
    suites: int
    natural_failed_episodes: int
    failed_tasks: int
    failure_windows: int
    matched_success_windows: int
    stage_annotation_passed: bool
    held_out_zero_shot_complete: bool
    episode_leakage: int
    seed_leakage: int
    processor_identity_completeness: float
    checkpoint_identity_completeness: float
    infrastructure_failures_excluded: bool
    failure_categories: int
    held_out_test_failures: int


def task_key(task: dict[str, Any]) -> tuple[str, int]:
    """Return one task's stable suite/id identity."""

    return str(task["suite"]), int(task["task_id"])


def build_phase_jobs(
    registry: dict[str, Any],
    phase: str,
) -> list[dict[str, Any]]:
    """Expand task-local seeds for one frozen phase without outcomes."""

    tasks = registry.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("task registry tasks must be a list")
    jobs: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or phase not in task.get("phases", []):
            continue
        seed_map = task.get("seeds")
        if not isinstance(seed_map, dict):
            raise ValueError(f"task {task_key(task)} is missing seeds")
        seeds = seed_map.get(phase)
        if not isinstance(seeds, list) or not all(
            isinstance(seed, int) for seed in seeds
        ):
            raise ValueError(f"task {task_key(task)} phase {phase} has invalid seeds")
        for seed in seeds:
            episode_id = f"{task['suite']}-task{int(task['task_id'])}-seed{int(seed)}"
            if episode_id in episode_ids:
                raise ValueError(f"duplicate episode identity: {episode_id}")
            episode_ids.add(episode_id)
            jobs.append(
                {
                    "episode_id": episode_id,
                    "suite": str(task["suite"]),
                    "task_id": int(task["task_id"]),
                    "task_name": str(task["task_name"]),
                    "instruction": str(task["instruction"]),
                    "episode_horizon": int(task["episode_horizon"]),
                    "seed": int(seed),
                    "phase": phase,
                    "group": str(task["group"]),
                }
            )
    return jobs


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on task history, seed isolation, horizon, and split drift."""

    if registry.get("frozen_before_rollout") is not True:
        raise ValueError("task registry must be frozen before rollout")
    tasks = registry.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 16:
        raise ValueError("LG-R1b must pre-register exactly 16 tasks")
    keys = [task_key(task) for task in tasks if isinstance(task, dict)]
    if len(keys) != len(tasks) or len(set(keys)) != len(keys):
        raise ValueError("LG-R1b task identities must be unique")
    history_raw = registry.get("historical_task_keys")
    if not isinstance(history_raw, list):
        raise ValueError("historical_task_keys must be a list")
    history = {(str(item["suite"]), int(item["task_id"])) for item in history_raw}
    overlap = sorted(set(keys) & history)
    if overlap:
        raise ValueError(f"held-out tasks overlap historical tasks: {overlap}")
    expected_horizons = registry.get("standard_horizons")
    if not isinstance(expected_horizons, dict):
        raise ValueError("standard_horizons must be a mapping")
    all_seeds: dict[int, tuple[tuple[str, int], str]] = {}
    phase_counts: dict[str, int] = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("task entries must be mappings")
        suite = str(task["suite"])
        if int(task["episode_horizon"]) != int(expected_horizons[suite]):
            raise ValueError(f"standard horizon drift for {task_key(task)}")
        if task.get("zero_shot_held_out") is not True:
            raise ValueError(f"new task is not zero-shot held-out: {task_key(task)}")
        if task.get("synthetic_failure_allowed") is not False:
            raise ValueError(
                f"synthetic failure not explicitly prohibited: {task_key(task)}"
            )
        phases = task.get("phases")
        seeds = task.get("seeds")
        if not isinstance(phases, list) or not isinstance(seeds, dict):
            raise ValueError(f"invalid phase/seed contract for {task_key(task)}")
        for phase in phases:
            values = seeds.get(phase)
            if not isinstance(values, list) or not values:
                raise ValueError(f"missing seeds for {task_key(task)} phase {phase}")
            phase_counts[str(phase)] = phase_counts.get(str(phase), 0) + len(values)
            for seed in values:
                if not isinstance(seed, int):
                    raise ValueError("rollout seeds must be integers")
                if seed in FINAL_SEEDS:
                    raise ValueError(f"sealed final seed accessed: {seed}")
                prior = all_seeds.setdefault(seed, (task_key(task), str(phase)))
                if prior != (task_key(task), str(phase)):
                    raise ValueError(f"seed {seed} reused across experiments")
    historical_seeds = set(int(seed) for seed in registry["historical_seeds"])
    if historical_seeds & set(all_seeds):
        raise ValueError("LG-R1b seeds overlap historical seeds")
    adaptation = registry.get("adaptation_task_split")
    if not isinstance(adaptation, dict):
        raise ValueError("adaptation_task_split must be frozen")
    split_sets: dict[str, set[tuple[str, int]]] = {}
    for split in ("train", "validation", "test"):
        raw = adaptation.get(split)
        if not isinstance(raw, list):
            raise ValueError(f"adaptation split {split} must be a list")
        split_sets[split] = {(str(item["suite"]), int(item["task_id"])) for item in raw}
    if (
        split_sets["train"] & split_sets["validation"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["validation"] & split_sets["test"]
    ):
        raise ValueError("adaptation task splits overlap")
    if set().union(*split_sets.values()) != {
        key for key in keys if key in {task_key(task) for task in tasks[:8]}
    }:
        raise ValueError("adaptation split must partition all primary tasks")
    return {
        "task_count": len(tasks),
        "groups": sorted({str(task["group"]) for task in tasks}),
        "phase_episode_counts": dict(sorted(phase_counts.items())),
        "historical_task_overlap": [],
        "historical_seed_overlap": [],
        "sealed_final_seed_overlap": [],
        "cross_experiment_seed_reuse": [],
    }


def evaluate_pilot_gate(evidence: PilotGateInput) -> dict[str, Any]:
    """Select the exact pre-registered branch after a complete pilot."""

    if evidence.valid_episodes != evidence.expected_episodes:
        status = "PILOT_INCOMPLETE"
        action = "stop_and_repair_infrastructure"
    elif evidence.infrastructure_errors:
        status = "PILOT_INFRASTRUCTURE_ERROR"
        action = "stop_and_repair_infrastructure"
    elif evidence.natural_failures >= 10:
        status = "PILOT_A"
        action = "uniform_primary_expansion"
    elif evidence.natural_failures >= 3:
        status = "PILOT_B"
        action = "uniform_primary_expansion_to_20_40_per_task"
    else:
        status = "PILOT_C"
        action = "activate_preregistered_secondary_group"
    return {
        "schema_version": "latentguard.lg_r1b.pilot_gate.v1",
        "input": asdict(evidence),
        "status": status,
        "next_action": action,
    }


def evaluate_standard_failure_yield(
    *,
    valid_episodes: int,
    task_groups: int,
    natural_failures: int,
) -> dict[str, Any]:
    """Stop at the frozen two-group, 200-episode low-yield boundary."""

    stop = valid_episodes >= 200 and task_groups >= 2 and natural_failures < 10
    return {
        "schema_version": "latentguard.lg_r1b.standard_failure_yield.v1",
        "input": {
            "valid_episodes": valid_episodes,
            "task_groups": task_groups,
            "natural_failures": natural_failures,
        },
        "status": (
            "STANDARD_LIBERO_FAILURE_YIELD_TOO_LOW"
            if stop
            else "continue_preregistered_protocol"
        ),
        "stop_adding_libero_episodes": stop,
    }


def evaluate_lg_r2_gate(evidence: LGR2GateInput) -> dict[str, Any]:
    """Evaluate full and limited LG-R2 gates without inferring missing evidence."""

    common = {
        "tasks": evidence.tasks >= 8,
        "suites": evidence.suites >= 2,
        "failed_tasks": evidence.failed_tasks >= 3,
        "stage_annotation": evidence.stage_annotation_passed,
        "held_out_zero_shot": evidence.held_out_zero_shot_complete,
        "episode_leakage": evidence.episode_leakage == 0,
        "seed_leakage": evidence.seed_leakage == 0,
        "processor_identity": evidence.processor_identity_completeness == 1.0,
        "checkpoint_identity": evidence.checkpoint_identity_completeness == 1.0,
        "infrastructure_exclusion": evidence.infrastructure_failures_excluded,
        "failure_diversity": evidence.failure_categories >= 2,
        "held_out_test_failures": evidence.held_out_test_failures >= 1,
    }
    full = {
        **common,
        "valid_total_rollouts": evidence.valid_total_rollouts >= 250,
        "natural_failed_episodes": evidence.natural_failed_episodes >= 20,
        "failure_windows": evidence.failure_windows >= 250,
        "matched_success_windows": evidence.matched_success_windows >= 250,
    }
    limited = {
        **common,
        "natural_failed_episodes": evidence.natural_failed_episodes >= 10,
        "failure_windows": evidence.failure_windows >= 150,
    }
    full_pass = all(full.values())
    limited_pass = not full_pass and all(limited.values())
    return {
        "schema_version": "latentguard.lg_r1b.lg_r2_gate.v1",
        "input": asdict(evidence),
        "full_checks": full,
        "limited_exception_checks": limited,
        "LG_R2_AUTHORIZED": full_pass,
        "LG_R2_LIMITED_AUTHORIZED": limited_pass,
        "status": (
            "pass"
            if full_pass
            else "limited_exception"
            if limited_pass
            else "FAILURE_DATA_GATE_NOT_MET"
        ),
    }
