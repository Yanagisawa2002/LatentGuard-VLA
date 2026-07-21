"""CPU-only native ACT closed-loop evaluation contract tests."""

from __future__ import annotations

import pytest

from latentguard.policies.act.evaluation import (
    PickCubeActCheckpointTier,
    PickCubeActEpisodeEvaluation,
    PickCubeActEvaluationError,
    classify_final_checkpoint,
    summarize_checkpoint_evaluation,
    wilson_interval,
)

_DIGEST = "sha256:" + "b" * 64


def _episode(seed: int, *, success: bool) -> PickCubeActEpisodeEvaluation:
    return PickCubeActEpisodeEvaluation(
        seed=seed,
        success=success,
        termination_category="success" if success else "timeout",
        action_count=40 if success else 50,
        grasp_success=success,
        post_grasp_drop=False,
        release_failure=False,
        workspace_violation=False,
        simulator_error=False,
        action_contract_violation=False,
        action_content_digest=_DIGEST,
        action_smoothness=0.2,
        query_latencies_seconds=(0.01, 0.02),
        peak_gpu_memory_allocated_bytes=1024,
    )


def test_summary_requires_exact_independent_seed_schedule() -> None:
    with pytest.raises(PickCubeActEvaluationError, match="seed schedule"):
        summarize_checkpoint_evaluation(
            checkpoint_id="best",
            checkpoint_digest=_DIGEST,
            episodes=(_episode(4, success=True), _episode(6, success=False)),
            seed_start=4,
        )


def test_summary_and_final_tiers_use_frozen_thresholds() -> None:
    episodes = tuple(_episode(seed, success=seed < 75) for seed in range(100))
    summary = summarize_checkpoint_evaluation(
        checkpoint_id="best",
        checkpoint_digest=_DIGEST,
        episodes=episodes,
        seed_start=0,
    )
    assert summary["success_rate"] == 0.75
    assert summary["timeout_count"] == 25
    assert (
        classify_final_checkpoint(
            summary,
            assets_complete=True,
            action_contract_complete=True,
            reproducible=True,
            primary_success_rate=0.75,
            secondary_success_rate=0.50,
        )
        is PickCubeActCheckpointTier.PRIMARY_ACCEPTED
    )
    assert (
        classify_final_checkpoint(
            summary,
            assets_complete=True,
            action_contract_complete=True,
            reproducible=False,
            primary_success_rate=0.75,
            secondary_success_rate=0.50,
        )
        is PickCubeActCheckpointTier.REJECTED
    )


def test_wilson_interval_is_bounded() -> None:
    lower, upper = wilson_interval(75, 100)
    assert 0.65 < lower < 0.75 < upper < 0.85
