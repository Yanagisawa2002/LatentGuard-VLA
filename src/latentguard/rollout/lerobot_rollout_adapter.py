"""LatentGuard manifest binding for official LeRobot 0.6 rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LeRobotRolloutIdentity:
    """Immutable policy/processor identity attached to every recorded episode."""

    episode_id: str
    suite: str
    task: str
    task_id: int
    seed: int
    instruction: str
    checkpoint_revision: str
    processor_revision: str
    policy_configuration_sha256: str


@dataclass(frozen=True)
class LeRobotRolloutResult:
    """Episode-level outcome and recording counters."""

    success: bool
    termination_reason: str
    frame_count: int
    policy_queries: int
    action_mask_semantic: str
    optimizer_steps: int = 0
    backward_calls: int = 0


def build_rollout_manifest_entry(
    identity: LeRobotRolloutIdentity,
    result: LeRobotRolloutResult,
    *,
    observation_keys: list[str],
    dataset_path: str,
    timestamps: dict[str, str],
    progress: float | None = None,
) -> dict[str, Any]:
    """Build one complete manifest entry without inferring missing evidence."""

    if result.frame_count < 1:
        raise ValueError("recorded rollout must contain at least one frame")
    if not observation_keys:
        raise ValueError("recorded rollout must declare observation keys")
    return {
        "identity": asdict(identity),
        "result": asdict(result),
        "observation_keys": list(observation_keys),
        "dataset_path": dataset_path,
        "timestamps": dict(timestamps),
        "progress": progress,
    }
