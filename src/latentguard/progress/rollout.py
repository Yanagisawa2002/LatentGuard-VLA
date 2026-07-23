"""Frozen rollout scheduling and atomic completion records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from latentguard.progress.serialization import read_json, write_json_atomic


@dataclass(frozen=True)
class RolloutJob:
    """One immutable on-policy episode request."""

    episode_id: str
    suite: str
    task_id: int
    task_name: str
    instruction: str
    episode_horizon: int
    seed: int
    phase: str


def build_rollout_jobs(
    registry: dict[str, Any],
    *,
    include_extension: bool,
) -> list[RolloutJob]:
    """Expand the frozen task and seed registry without using outcomes."""

    seed_schedule = registry.get("seed_schedule")
    tasks = registry.get("tasks")
    if not isinstance(seed_schedule, dict) or not isinstance(tasks, list):
        raise ValueError("task registry is missing seed_schedule or tasks")
    phases = ["pilot"]
    if include_extension:
        phases.append("extension")
    jobs: list[RolloutJob] = []
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("task registry entries must be mappings")
        for phase in phases:
            seeds = seed_schedule.get(phase)
            if not isinstance(seeds, list) or not all(
                isinstance(seed, int) for seed in seeds
            ):
                raise ValueError(f"seed schedule {phase} must be an integer list")
            for seed in seeds:
                episode_id = f"{task['suite']}-task{int(task['task_id'])}-seed{seed}"
                if episode_id in seen:
                    raise ValueError(f"duplicate rollout job: {episode_id}")
                seen.add(episode_id)
                jobs.append(
                    RolloutJob(
                        episode_id=episode_id,
                        suite=str(task["suite"]),
                        task_id=int(task["task_id"]),
                        task_name=str(task["task_name"]),
                        instruction=str(task["instruction"]),
                        episode_horizon=int(task["episode_horizon"]),
                        seed=seed,
                        phase=phase,
                    )
                )
    return jobs


def validate_seed_isolation(registry: dict[str, Any]) -> None:
    """Reject overlap among pilot, extension, history, and sealed seeds."""

    schedule = registry.get("seed_schedule")
    if not isinstance(schedule, dict):
        raise ValueError("task registry is missing seed_schedule")
    groups: dict[str, set[int]] = {}
    for name in ("pilot", "extension", "historical_development"):
        values = schedule.get(name)
        if not isinstance(values, list) or not all(
            isinstance(value, int) for value in values
        ):
            raise ValueError(f"{name} seeds must be an integer list")
        groups[name] = set(values)
        if len(groups[name]) != len(values):
            raise ValueError(f"{name} contains duplicate seeds")
    sealed = schedule.get("sealed_final_range")
    if not isinstance(sealed, dict):
        raise ValueError("sealed_final_range must be a mapping")
    start = int(sealed["start"])
    end = int(sealed["end"])
    if end < start:
        raise ValueError("sealed final seed range is reversed")
    groups["sealed_final"] = set(range(start, end + 1))
    names = list(groups)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = groups[left_name] & groups[right_name]
            if overlap:
                raise ValueError(
                    f"seed groups {left_name} and {right_name} overlap: "
                    f"{sorted(overlap)}"
                )


def completed_episode_ids(completion_root: Path) -> set[str]:
    """Return only valid atomic completion records."""

    if not completion_root.exists():
        return set()
    completed: set[str] = set()
    for marker in completion_root.glob("*.json"):
        payload = read_json(marker)
        episode_id = payload.get("episode_id")
        if payload.get("status") != "complete" or not isinstance(episode_id, str):
            raise ValueError(f"invalid completion marker: {marker}")
        if marker.stem != episode_id:
            raise ValueError(f"completion marker identity mismatch: {marker}")
        if episode_id in completed:
            raise ValueError(f"duplicate completion marker: {episode_id}")
        completed.add(episode_id)
    return completed


def write_completion_marker(
    completion_root: Path,
    job: RolloutJob,
    *,
    frame_count: int,
    success: bool,
    termination_reason: str,
    dataset_locator: str,
    sidecar_locator: str,
) -> Path:
    """Atomically mark a fully finalized episode as resumable."""

    if frame_count < 1 or frame_count > job.episode_horizon:
        raise ValueError("frame_count is outside the episode horizon")
    if success and termination_reason != "success":
        raise ValueError("successful episodes must terminate as success")
    if not success and termination_reason == "success":
        raise ValueError("failed episodes cannot terminate as success")
    payload = {
        "schema_version": "latentguard.lg_r1.episode_completion.v1",
        "status": "complete",
        **asdict(job),
        "frame_count": frame_count,
        "success": success,
        "termination_reason": termination_reason,
        "dataset_locator": dataset_locator,
        "sidecar_locator": sidecar_locator,
    }
    path = completion_root / f"{job.episode_id}.json"
    write_json_atomic(path, payload)
    return path


def action_step_record(
    *,
    step_index: int,
    generated_chunk: list[list[float]],
    chunk_offset: int,
    selected_policy_action: list[float],
    executed_action: list[float],
) -> dict[str, Any]:
    """Bind an executed action to its unmodified generated policy chunk."""

    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if len(generated_chunk) != 7 or any(len(action) != 7 for action in generated_chunk):
        raise ValueError("generated VLA-JEPA chunk must have shape (7, 7)")
    if chunk_offset < 0 or chunk_offset >= len(generated_chunk):
        raise ValueError("chunk_offset is outside generated chunk")
    if len(selected_policy_action) != 7 or len(executed_action) != 7:
        raise ValueError("selected and executed actions must have shape (7,)")
    if selected_policy_action != generated_chunk[chunk_offset]:
        raise ValueError("selected policy action differs from generated chunk")
    return {
        "step_index": step_index,
        "generated_action_chunk": generated_chunk,
        "generated_chunk_offset": chunk_offset,
        "selected_policy_action": selected_policy_action,
        "executed_action": executed_action,
        "action_mask": None,
        "action_is_pad": False,
    }
