"""Strict closed-loop evaluation summaries for native PickCube ACT policies."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from statistics import mean
from typing import NoReturn, cast

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PickCubeActEvaluationError(ValueError):
    """Raised when policy evaluation evidence is incomplete or malformed."""


class PickCubeActCheckpointTier(StrEnum):
    """Outcome-bound checkpoint classifications frozen before final evaluation."""

    PRIMARY_ACCEPTED = "PRIMARY_ACCEPTED"
    SECONDARY_COMPATIBLE = "SECONDARY_COMPATIBLE"
    COMPATIBLE_BUT_WEAK = "COMPATIBLE_BUT_WEAK"
    REJECTED = "REJECTED"


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActEvaluationError(f"{context}: {reason}")


def _finite(value: float, context: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(context, "expected finite number")
    result = float(value)
    if result < minimum:
        _fail(context, f"expected value >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class PickCubeActEpisodeEvaluation:
    """Complete result of one independently reset closed-loop policy episode."""

    seed: int
    success: bool
    termination_category: str
    action_count: int
    grasp_success: bool
    post_grasp_drop: bool
    release_failure: bool
    workspace_violation: bool
    simulator_error: bool
    action_contract_violation: bool
    action_content_digest: str
    action_smoothness: float | None
    query_latencies_seconds: tuple[float, ...]
    peak_gpu_memory_allocated_bytes: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            _fail("episode.seed", "expected uint32")
        for name in (
            "success",
            "grasp_success",
            "post_grasp_drop",
            "release_failure",
            "workspace_violation",
            "simulator_error",
            "action_contract_violation",
        ):
            if type(getattr(self, name)) is not bool:
                _fail(f"episode.{name}", "expected boolean")
        if (
            not isinstance(self.termination_category, str)
            or not self.termination_category
            or self.termination_category != self.termination_category.strip()
        ):
            _fail("episode.termination_category", "expected canonical text")
        if type(self.action_count) is not int or not 0 <= self.action_count <= 50:
            _fail("episode.action_count", "expected native horizon count in [0, 50]")
        if self.success and (
            self.simulator_error
            or self.workspace_violation
            or self.action_contract_violation
            or self.termination_category != "success"
        ):
            _fail("episode.success", "success conflicts with failure evidence")
        if self.post_grasp_drop and not self.grasp_success:
            _fail("episode.post_grasp_drop", "drop requires a prior grasp")
        if self.action_smoothness is not None:
            _finite(self.action_smoothness, "episode.action_smoothness")
        if any(
            _finite(value, "episode.query_latency") < 0.0
            for value in self.query_latencies_seconds
        ):
            _fail("episode.query_latencies_seconds", "latency cannot be negative")
        if type(self.peak_gpu_memory_allocated_bytes) is not int or (
            self.peak_gpu_memory_allocated_bytes < 0
        ):
            _fail("episode.peak_gpu_memory_allocated_bytes", "expected non-negative")
        if _DIGEST_RE.fullmatch(self.action_content_digest) is None:
            _fail("episode.action_content_digest", "expected SHA-256 digest")

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-native episode record."""
        payload = asdict(self)
        payload["query_latencies_seconds"] = list(self.query_latencies_seconds)
        return payload


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the fixed two-sided 95% Wilson binomial interval."""
    if (
        type(successes) is not int
        or type(total) is not int
        or not 0 <= successes <= total
    ):
        _fail("Wilson interval", "counts are invalid")
    if total < 1:
        _fail("Wilson interval", "total must be positive")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_checkpoint_evaluation(
    *,
    checkpoint_id: str,
    checkpoint_digest: str,
    episodes: Sequence[PickCubeActEpisodeEvaluation],
    seed_start: int,
) -> Mapping[str, object]:
    """Aggregate an exact contiguous independent-seed evaluation ledger."""
    if not checkpoint_id or checkpoint_id != checkpoint_id.strip():
        _fail("checkpoint_id", "expected canonical text")
    if not checkpoint_digest.startswith("sha256:") or len(checkpoint_digest) != 71:
        _fail("checkpoint_digest", "expected SHA-256 digest")
    if not episodes:
        _fail("episodes", "evaluation ledger is empty")
    expected_seeds = list(range(seed_start, seed_start + len(episodes)))
    observed_seeds = [episode.seed for episode in episodes]
    if observed_seeds != expected_seeds or len(set(observed_seeds)) != len(episodes):
        _fail("episodes", "seed schedule is not exact, ordered, and independent")
    success_count = sum(episode.success for episode in episodes)
    latencies = [
        latency for episode in episodes for latency in episode.query_latencies_seconds
    ]
    smoothness = [
        episode.action_smoothness
        for episode in episodes
        if episode.action_smoothness is not None
    ]
    interval = wilson_interval(success_count, len(episodes))
    taxonomy = Counter(episode.termination_category for episode in episodes)
    return {
        "action_smoothness_mean_l2": mean(smoothness) if smoothness else None,
        "action_contract_violation_count": sum(
            episode.action_contract_violation for episode in episodes
        ),
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_id": checkpoint_id,
        "episode_count": len(episodes),
        "episodes": [episode.to_mapping() for episode in episodes],
        "grasp_success_count": sum(episode.grasp_success for episode in episodes),
        "mean_episode_length": mean(episode.action_count for episode in episodes),
        "peak_gpu_memory_allocated_bytes": max(
            episode.peak_gpu_memory_allocated_bytes for episode in episodes
        ),
        "policy_query_count": len(latencies),
        "policy_query_latency_mean_seconds": mean(latencies) if latencies else None,
        "policy_query_latency_p50_seconds": _percentile(latencies, 0.50),
        "policy_query_latency_p95_seconds": _percentile(latencies, 0.95),
        "post_grasp_drop_count": sum(episode.post_grasp_drop for episode in episodes),
        "release_failure_count": sum(episode.release_failure for episode in episodes),
        "schema_version": "pickcube-native-act-checkpoint-evaluation-v1",
        "seed_end_exclusive": seed_start + len(episodes),
        "seed_start": seed_start,
        "simulator_error_count": sum(episode.simulator_error for episode in episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "success_rate_95_percent_wilson": list(interval),
        "terminal_failure_count": len(episodes) - success_count,
        "termination_taxonomy": dict(sorted(taxonomy.items())),
        "timeout_count": taxonomy["timeout"],
        "workspace_violation_count": sum(
            episode.workspace_violation for episode in episodes
        ),
    }


def classify_final_checkpoint(
    summary: Mapping[str, object],
    *,
    assets_complete: bool,
    action_contract_complete: bool,
    reproducible: bool,
    primary_success_rate: float,
    secondary_success_rate: float,
) -> PickCubeActCheckpointTier:
    """Apply frozen final-only thresholds without repairing evidence."""
    if primary_success_rate != 0.75 or secondary_success_rate != 0.50:
        _fail("classification", "acceptance thresholds changed")
    episode_count = summary.get("episode_count")
    success_rate = summary.get("success_rate")
    simulator_errors = summary.get("simulator_error_count")
    workspace_violations = summary.get("workspace_violation_count")
    action_contract_violations = summary.get("action_contract_violation_count")
    if (
        type(episode_count) is not int
        or episode_count < 100
        or type(success_rate) not in (int, float)
        or not math.isfinite(float(cast(int | float, success_rate)))
        or type(simulator_errors) is not int
        or type(workspace_violations) is not int
        or type(action_contract_violations) is not int
    ):
        _fail("classification", "final evaluation evidence is incomplete")
    if (
        not assets_complete
        or not action_contract_complete
        or not reproducible
        or simulator_errors != 0
        or workspace_violations != 0
        or action_contract_violations != 0
    ):
        return PickCubeActCheckpointTier.REJECTED
    rate = float(cast(int | float, success_rate))
    if rate >= primary_success_rate:
        return PickCubeActCheckpointTier.PRIMARY_ACCEPTED
    if rate >= secondary_success_rate:
        return PickCubeActCheckpointTier.SECONDARY_COMPATIBLE
    return PickCubeActCheckpointTier.COMPATIBLE_BUT_WEAK


__all__ = [
    "PickCubeActCheckpointTier",
    "PickCubeActEpisodeEvaluation",
    "PickCubeActEvaluationError",
    "classify_final_checkpoint",
    "summarize_checkpoint_evaluation",
    "wilson_interval",
]
