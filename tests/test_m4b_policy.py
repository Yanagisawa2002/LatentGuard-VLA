"""CPU-only data-cycling, promotion, and statistical-policy tests for M4B."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from latentguard.visual_training.data import (
    TRAIN_DOMAIN_IDS,
    VisualTrainingDataError,
    domain_index_for_epoch,
    require_development_training_dataset,
)
from latentguard.visual_training.evaluation import _summary, _trajectory_bootstrap
from latentguard.visual_training.external import _checkpoint_matches_external_freeze
from latentguard.visual_training.training import (
    DomainValidationResultV1,
    VisualTrainingError,
    VisualTrainingResultV1,
    create_seed_zero_promotion_record,
)


def _result(model_id: str, score: float, *, seed: int = 0) -> VisualTrainingResultV1:
    domains = tuple(
        DomainValidationResultV1(
            domain_id=domain,
            failure_auprc=score,
            corrupted_only_failure_auprc=score,
            pairwise_concordance=score,
            brier_score=1.0 - score,
            candidate_count=24,
        )
        for domain in TRAIN_DOMAIN_IDS
    )
    return VisualTrainingResultV1(
        model_id=model_id,
        seed=seed,
        selected_epoch=2,
        completed_epochs=3,
        global_steps=9,
        mean_corrupted_only_failure_auprc=score,
        worst_domain_corrupted_only_failure_auprc=score,
        domain_results=domains,
        total_parameters=10,
        trainable_parameters=10,
        peak_gpu_memory_bytes=0,
        duration_seconds=1.0,
        best_checkpoint_digest="sha256:" + "a" * 64,
        resumed=False,
        zero_work_resume=False,
    )


def test_domain_cycle_is_deterministic_and_complete() -> None:
    """Every candidate visits all three allowed domains once per three epochs."""
    first = tuple(domain_index_for_epoch("sample-a", epoch=i, seed=7) for i in range(3))
    second = tuple(
        domain_index_for_epoch("sample-a", epoch=i, seed=7) for i in range(3)
    )
    assert first == second
    assert set(first) == {0, 1, 2}


def test_non_development_object_is_rejected() -> None:
    """Only the exact accepted development dataset can enter training."""
    with pytest.raises(VisualTrainingDataError, match="expected M3A"):
        require_development_training_dataset(object())


def test_seed_zero_promotion_reuses_winner_and_caps_families() -> None:
    """Validation promotes no more than the winner, direct baseline, and ablation."""
    results = (
        _result("random_resnet18_multiview_action", 0.55),
        _result("frozen_resnet18_singleview_action", 0.60),
        _result("frozen_resnet18_multiview_action", 0.65),
        _result("frozen_resnet18_multiview_action_distilled", 0.70),
    )
    record = create_seed_zero_promotion_record(results)

    assert record.validation_winner == "frozen_resnet18_multiview_action_distilled"
    assert "frozen_resnet18_multiview_action" in record.promoted_model_ids
    assert len(record.promoted_model_ids) <= 3
    assert all(item.seed == 0 for item in record.screening_results)
    assert record.as_mapping()["test_opened"] is False

    with pytest.raises(VisualTrainingError, match="exactly four"):
        create_seed_zero_promotion_record(results[:3])


def test_three_seed_statistics_and_trajectory_bootstrap_are_deterministic() -> None:
    """Aggregate and paired intervals use exactly three seeds and trajectories."""
    assert _summary([0.2, 0.4, 0.6]) == {
        "maximum": 0.6,
        "mean": pytest.approx(0.4),
        "median": 0.4,
        "minimum": 0.2,
        "standard_deviation": pytest.approx(np.std([0.2, 0.4, 0.6])),
    }
    first = np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    second = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    trajectories = ("t0", "t0", "t1", "t1")
    one = _trajectory_bootstrap(first, second, trajectories)
    two = _trajectory_bootstrap(first, second, trajectories)
    assert one == two
    assert one["bootstrap_samples"] == 2000
    assert one["resampling_unit"] == "source_trajectory"


def test_external_checkpoint_binds_training_sha_from_freeze() -> None:
    """A later feature-cache execution SHA does not invalidate frozen weights."""
    training_sha = "a" * 40
    later_cache_execution_sha = "b" * 40
    digest = "sha256:" + "c" * 64
    backbone = "sha256:" + "d" * 64
    model_config = "sha256:" + "e" * 64
    metadata = SimpleNamespace(
        progress=SimpleNamespace(
            complete=True,
            binding=SimpleNamespace(
                seed=0,
                model_config_digest=model_config,
                git_sha=training_sha,
                backbone_digest=backbone,
            ),
        )
    )

    assert training_sha != later_cache_execution_sha
    assert _checkpoint_matches_external_freeze(
        metadata,
        seed=0,
        model_config_digest=model_config,
        observed_checkpoint_digest=digest,
        expected_checkpoint_digest=digest,
        freeze_git_sha=training_sha,
        feature_backbone_digest=backbone,
    )
    assert not _checkpoint_matches_external_freeze(
        metadata,
        seed=0,
        model_config_digest=model_config,
        observed_checkpoint_digest=digest,
        expected_checkpoint_digest=digest,
        freeze_git_sha=later_cache_execution_sha,
        feature_backbone_digest=backbone,
    )
