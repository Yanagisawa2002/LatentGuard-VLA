from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from latentguard.control.config import load_bootstrap_configuration
from latentguard.control.metrics import (
    EpisodeMetricRowV1,
    aggregate_episode_metrics,
    intervention_diagnostics,
    paired_trajectory_bootstrap,
)
from latentguard.control.models import EpisodeState
from latentguard.m4c_cli import _predeclared_bootstrap_comparisons


def row(source: str, selector: str, success: bool, steps: int) -> EpisodeMetricRowV1:
    return EpisodeMetricRowV1(
        source_trajectory_id=source,
        selector_id=selector,
        visual_domain="canonical" if selector == "visual" else "not_applicable",
        outcome=EpisodeState.SUCCESS if success else EpisodeState.HORIZON_EXHAUSTED,
        executed_control_steps=steps,
        decision_count=steps // 4,
        intervention_count=1 if selector == "visual" else 0,
        decisions_before_first_intervention=0 if selector == "visual" else None,
        selected_probability_sum=0.1 if selector == "visual" else 0.0,
        selected_probability_count=1 if selector == "visual" else 0,
        risk_margin_sum=0.2 if selector == "visual" else 0.0,
        risk_margin_count=1 if selector == "visual" else 0,
        tail_fallback_count=1,
        recovery_count=0,
        execution_seconds=1.0,
    )


def test_aggregate_keeps_horizon_separate_from_task_failure() -> None:
    report = aggregate_episode_metrics(
        (row("a", "visual", True, 8), row("b", "visual", False, 12))
    )
    assert report["success_rate"] == 0.5
    assert report["horizon_exhaustion_rate"] == 0.5
    assert report["task_failure_rate"] == 0.0


def test_paired_bootstrap_resamples_trajectory_identity() -> None:
    primary = tuple(row(str(index), "visual", index > 0, 8) for index in range(4))
    comparator = tuple(row(str(index), "random", False, 12) for index in range(4))
    configuration = load_bootstrap_configuration(
        Path("configs/control/m4c/bootstrap-v1.json")
    )
    first = paired_trajectory_bootstrap(
        primary, comparator, configuration=configuration
    )
    second = paired_trajectory_bootstrap(
        primary, comparator, configuration=configuration
    )
    assert first == second
    assert first["paired_source_trajectory_count"] == 4
    assert first["intervals"]["episode_success_rate_difference"]["estimate"] == 0.75


def test_intervention_diagnostics_disclaim_causality() -> None:
    report = intervention_diagnostics(
        (row("a", "visual", True, 8), row("b", "random", False, 12))
    )
    assert report["causal_interpretation_permitted"] is False
    assert report["success_conditional_on_intervention"] == 1.0
    assert report["success_conditional_on_no_intervention"] == 0.0
    assert report["decisions_before_first_intervention"]["mean"] == 0.0


def test_predeclared_bootstrap_covers_all_primary_and_domain_comparisons() -> None:
    sources = ("a", "b")
    grouped: dict[tuple[str, str], tuple[EpisodeMetricRowV1, ...]] = {}
    for selector in (
        "fixed_primary_v1",
        "deterministic_random_v1",
        "action_only_ensemble_v1",
        "privileged_structured_ensemble_v1",
    ):
        grouped[(selector, "not_applicable")] = tuple(
            row(source, selector, False, 12) for source in sources
        )
    for selector in (
        "direct_visual_ensemble_v1",
        "distilled_visual_ensemble_v1",
    ):
        for domain in (
            "canonical",
            "strong_camera_shift",
            "strong_lighting_shift",
        ):
            grouped[(selector, domain)] = tuple(
                replace(
                    row(source, "visual", True, 8),
                    selector_id=selector,
                    visual_domain=domain,
                )
                for source in sources
            )
    configuration = load_bootstrap_configuration(
        Path("configs/control/m4c/bootstrap-v1.json")
    )
    report = _predeclared_bootstrap_comparisons(grouped, bootstrap=configuration)
    assert len(report) == 17
    assert "distilled_visual_ensemble_v1/canonical_vs_deterministic_random_v1" in report
    assert (
        "distilled_visual_ensemble_v1/canonical_vs_"
        "direct_visual_ensemble_v1/canonical" in report
    )
    assert "distilled_visual_ensemble_v1/canonical_vs_strong_camera_shift" in report
