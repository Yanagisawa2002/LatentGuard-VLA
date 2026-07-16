"""CPU-only fixtures for M3B candidate, group, slice, and coverage metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from latentguard.training.metrics import (
    GroupCandidate,
    MetricStatus,
    average_precision_score,
    evaluate_binary_metrics,
    evaluate_coverage_risk,
    evaluate_group_ranking,
    evaluate_slices,
    roc_auc_score,
)


def test_binary_metrics_match_known_failure_positive_fixture() -> None:
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.4, 0.35, 0.8], dtype=np.float64)

    report = evaluate_binary_metrics(targets, probabilities, reliability_bin_count=5)

    assert report.failure_prevalence.value == pytest.approx(0.5)
    assert report.roc_auc.value == pytest.approx(0.75)
    assert report.failure_auprc.value == pytest.approx(5.0 / 6.0)
    assert report.success_auprc.value == pytest.approx(5.0 / 6.0)
    assert report.brier_score.value == pytest.approx(0.158125)
    assert report.balanced_accuracy.value == pytest.approx(0.75)
    assert report.precision.value == pytest.approx(1.0)
    assert report.recall.value == pytest.approx(0.5)
    assert report.specificity.value == pytest.approx(1.0)
    assert report.f1.value == pytest.approx(2.0 / 3.0)
    assert report.confusion_matrix.to_dict() == {
        "false_negative": 1,
        "false_positive": 0,
        "true_negative": 2,
        "true_positive": 1,
    }
    assert sum(item.count for item in report.reliability_bins) == 4


def test_single_class_and_zero_denominator_metrics_are_explicit_not_nan() -> None:
    targets = np.zeros(3, dtype=np.int64)
    probabilities = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)

    report = evaluate_binary_metrics(targets, probabilities)

    assert report.roc_auc.status is MetricStatus.SINGLE_CLASS
    assert report.failure_auprc.status is MetricStatus.SINGLE_CLASS
    assert report.recall.status is MetricStatus.ZERO_DENOMINATOR
    assert report.balanced_accuracy.status is MetricStatus.SINGLE_CLASS
    encoded = json.dumps(report.to_dict(), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_probability_and_target_contracts_reject_malformed_values() -> None:
    with pytest.raises(ValueError, match="boolean or integer"):
        evaluate_binary_metrics(np.asarray([0.0, 1.0]), np.asarray([0.2, 0.8]))
    with pytest.raises(ValueError, match="finite"):
        evaluate_binary_metrics(np.asarray([0, 1]), np.asarray([0.2, np.nan]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_binary_metrics(np.asarray([0, 1]), np.asarray([0.2, 1.2]))


def test_tied_ranking_scores_are_order_invariant() -> None:
    targets = np.asarray([1, 0, 1, 0], dtype=np.int64)
    probabilities = np.full(4, 0.5, dtype=np.float64)

    assert roc_auc_score(targets, probabilities).value == pytest.approx(0.5)
    assert average_precision_score(targets, probabilities).value == pytest.approx(0.5)
    permutation = np.asarray([3, 1, 2, 0])
    assert average_precision_score(
        targets[permutation], probabilities[permutation]
    ).value == pytest.approx(0.5)


def _group_candidate(
    sample_id: str,
    group_id: str,
    trajectory: str,
    candidate_type: str,
    failure: int,
    probability: float,
) -> GroupCandidate:
    return GroupCandidate(
        sample_id=sample_id,
        group_id=group_id,
        source_trajectory_id=trajectory,
        candidate_type=candidate_type,
        failure_target=failure,
        failure_probability=probability,
    )


def test_group_ranking_reports_eligibility_random_expectation_and_source_margin() -> (
    None
):
    candidates = (
        _group_candidate("s0", "g0", "t0", "source", 0, 0.05),
        _group_candidate("c0", "g0", "t0", "corrupted", 0, 0.10),
        _group_candidate("c1", "g0", "t0", "corrupted", 1, 0.80),
        _group_candidate("s1", "g1", "t1", "source", 0, 0.05),
        _group_candidate("c2", "g1", "t1", "corrupted", 0, 0.60),
        _group_candidate("c3", "g1", "t1", "corrupted", 0, 0.70),
    )

    report = evaluate_group_ranking(candidates)

    assert report.group_count == 2
    assert report.corrupted_top1_eligible_group_count == 2
    assert report.corrupted_top1_success.value == pytest.approx(1.0)
    assert report.random_choice_success_expectation.value == pytest.approx(0.75)
    assert report.both_class_group_count == 1
    assert report.pairwise_comparison_count == 1
    assert report.pairwise_success_over_failure_concordance.value == pytest.approx(1.0)
    assert report.corrupted_top1_failure_rate.value == pytest.approx(0.0)
    assert report.mean_reciprocal_rank_first_success.value == pytest.approx(1.0)
    assert report.source_margin_group_count == 1
    assert report.mean_source_success_margin_over_failed_corruptions.value == (
        pytest.approx(0.75)
    )


def test_group_pairwise_ties_receive_half_credit_and_ties_use_sample_id() -> None:
    candidates = (
        _group_candidate("s", "g", "t", "source", 0, 0.1),
        _group_candidate("z-success", "g", "t", "corrupted", 0, 0.5),
        _group_candidate("a-failure", "g", "t", "corrupted", 1, 0.5),
    )

    report = evaluate_group_ranking(candidates)

    assert report.pairwise_success_over_failure_concordance.value == pytest.approx(0.5)
    assert report.corrupted_top1_failure_rate.value == pytest.approx(1.0)
    assert report.mean_reciprocal_rank_first_success.value == pytest.approx(0.5)


def test_coverage_risk_uses_deterministic_prefix_semantics() -> None:
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)

    report = evaluate_coverage_risk(
        targets, probabilities, requested_coverages=(1.0, 0.5)
    )

    assert report.area_under_risk_coverage_curve.value == pytest.approx(5.0 / 24.0)
    half = report.points[1]
    assert half.threshold == pytest.approx(0.2)
    assert half.observed_coverage == pytest.approx(0.5)
    assert half.retained_failure_rate.value == pytest.approx(0.0)
    assert half.retained_success_rate.value == pytest.approx(1.0)
    assert half.rejected_failure_recall.value == pytest.approx(1.0)


def test_coverage_boundary_ties_are_all_retained_and_actual_coverage_is_reported() -> (
    None
):
    report = evaluate_coverage_risk(
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        np.asarray([0.1, 0.2, 0.2, 0.9], dtype=np.float64),
        requested_coverages=(0.5,),
    )

    assert report.points[0].threshold == pytest.approx(0.2)
    assert report.points[0].observed_coverage == pytest.approx(0.75)


def test_slice_metrics_share_one_threshold_and_mark_single_class_slices() -> None:
    reports = evaluate_slices(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        {
            "candidate_type": ["source", "source", "corrupted", "corrupted"],
            "severity": [None, None, "severe", "severe"],
        },
        threshold=0.7,
    )

    assert {report.metrics.threshold for report in reports} == {0.7}
    assert any(
        report.value == "source"
        and report.metrics.roc_auc.status is MetricStatus.SINGLE_CLASS
        for report in reports
    )
    assert any(report.value == "<not_applicable>" for report in reports)
