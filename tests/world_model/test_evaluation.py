"""Candidate grouping and metric-contract tests."""

from __future__ import annotations

from latentguard.world_model.evaluation import (
    CandidatePrediction,
    candidate_ranking_metrics,
    compare_candidate_selectors,
)


def test_candidate_group_ranking() -> None:
    """Top-1 and pairwise metrics compare only within shared anchors."""

    result = candidate_ranking_metrics(
        (
            CandidatePrediction("a", "a-good", "policy_generated", 0.9, True),
            CandidatePrediction("a", "a-bad", "synthetic_corruption", 0.1, False),
            CandidatePrediction("b", "b-bad", "synthetic_corruption", 0.8, False),
            CandidatePrediction("b", "b-good", "synthetic_corruption", 0.7, True),
        )
    )
    assert result["all"] == {
        "group_count": 2,
        "oracle_regret": 0.5,
        "pairwise_accuracy": 0.5,
        "pairwise_support": 2,
        "top1_success_rate": 0.5,
    }


def test_fair_three_model_comparison_requires_identical_candidates() -> None:
    """Future-latent added value is computed only on the same candidates."""

    outcomes = (
        CandidatePrediction("a", "good", "synthetic_corruption", 0.9, True),
        CandidatePrediction("a", "bad", "synthetic_corruption", 0.1, False),
    )
    result = compare_candidate_selectors(
        {
            "direct_verifier": outcomes,
            "outcome_only": outcomes,
            "wm_v0": outcomes,
        }
    )
    assert result["does_future_latent_improve_selection"] == "tied_on_point_estimate"
