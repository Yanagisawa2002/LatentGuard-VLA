from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from latentguard.control.config import load_closed_loop_configuration
from latentguard.control.fallback import (
    ConservativeOverrideGateV1,
    FallbackShieldError,
    FallbackShieldSelectorV1,
    GateProfileV1,
    NominalScenario,
    designate_nominal,
    load_fallback_decision,
    load_fault_schedule,
    load_gate_profiles,
)
from latentguard.control.fallback_benchmark import (
    build_fallback_benchmark_schedule,
    load_gate_selection,
    run_fallback_benchmark_schedule,
)
from latentguard.control.fallback_config import (
    load_fallback_selector_matrix,
    load_m4d_source_seed_configuration,
)
from latentguard.control.fallback_metrics import (
    FallbackEpisodeMetricRowV1,
    aggregate_fallback_metrics,
    paired_fallback_bootstrap,
    select_gate_profiles,
)
from latentguard.control.models import (
    EpisodeState,
    SourcePlanIdentityV1,
    content_digest,
)
from latentguard.control.risk import CandidateRiskScoresV1
from latentguard.control.runner import (
    BoundSourcePlanV1,
    RuntimeBoundaryV1,
    RuntimeStepResultV1,
    run_closed_loop_episode,
    simple_pool_from_actions,
)
from latentguard.control.serialization import (
    ClosedLoopEpisodeStore,
    write_exclusive_json,
)


def digest(label: str) -> str:
    return content_digest({"label": label})


def source_plan(source_id: str = "m4d-source") -> BoundSourcePlanV1:
    actions = np.zeros((36, 8), dtype=np.float64)
    identity = SourcePlanIdentityV1(
        source_trajectory_id=source_id,
        split_group_id=f"split-{source_id}",
        reset_seed=430000,
        source_action_digest=digest(f"actions-{source_id}"),
        initial_state_digest=digest(f"state-{source_id}"),
        complete_state_tree_digests=(digest(f"state-{source_id}"),),
        independent_replay_success=True,
        planner_identity="fake-planner",
        compatibility_identity=digest("compatibility"),
    )
    return BoundSourcePlanV1(identity, actions, "state-0")


def boundary(step: int = 0) -> RuntimeBoundaryV1:
    return RuntimeBoundaryV1(
        state_reference=f"state-{step}",
        state_digest=digest(f"state-{step}"),
        verifier_state=np.full(38, step, dtype=np.float32),
    )


def pool(source_id: str = "m4d-source", decision: int = 0):
    source = source_plan(source_id)
    base = source.actions[:16]
    candidates = tuple(
        np.asarray(base + (index + 1) * 0.001, dtype=np.float64) for index in range(8)
    )
    return simple_pool_from_actions(
        source,
        boundary(),
        decision_ordinal=decision,
        nominal_plan_index=0,
        candidate_actions=candidates,
        candidate_pool_configuration_digest=(
            "sha256:b6a532550ba27e3ecc6acf16ccae0138fce97dd0fd8533724bd74dce2c3a714f"
        ),
    )


def test_checked_in_m4d_configs_and_exact_workloads() -> None:
    root = Path("configs/control/m4d")
    schedule = load_fault_schedule(root / "fault-schedule-v1.json")
    profiles = load_gate_profiles(root / "gate-profiles-v1.json")
    development = load_m4d_source_seed_configuration(root / "development-seeds-v1.json")
    final = load_m4d_source_seed_configuration(root / "final-seeds-v1.json")
    matrix = load_fallback_selector_matrix(root / "selector-matrix-v1.json")
    sources = (source_plan("one"), source_plan("two"))

    assert schedule.fixed_primary_probability == 0.70
    assert len(profiles.profiles) == 3
    assert development.requested_successes == 12
    assert final.requested_successes == 60
    assert (
        len(build_fallback_benchmark_schedule(sources, matrix, phase="development"))
        == 64
    )
    selected = {
        "gated_action_only": "conservative",
        "gated_direct_visual": "balanced",
        "gated_distilled_visual": "responsive",
        "gated_privileged_structured": "conservative",
    }
    assert (
        len(
            build_fallback_benchmark_schedule(
                sources,
                matrix,
                phase="final",
                selected_profiles=selected,
            )
        )
        == 52
    )


def test_gate_profiles_reproduce_accepted_m4b_validation_derivation() -> None:
    report = json.loads(
        Path(
            "reports/m4b/20260718T_m4b-full_61f05e5_seed0/internal/"
            "frozen_resnet18_multiview_action.json"
        ).read_text(encoding="utf-8")
    )
    thresholds = report["thresholds"]
    balanced = np.asarray(
        [item["maximum_balanced_accuracy"]["threshold"] for item in thresholds],
        dtype=np.float64,
    )
    recall = np.asarray(
        [item["target_failure_recall"]["threshold"] for item in thresholds],
        dtype=np.float64,
    )
    margins = np.abs(balanced - recall)
    uncertainty_proxy = np.abs(recall - np.median(recall))
    profiles = load_gate_profiles(Path("configs/control/m4d/gate-profiles-v1.json"))
    for profile, quantile in zip(profiles.profiles, (0.75, 0.50, 0.25), strict=True):
        assert profile.nominal_risk_threshold == float(
            np.quantile(balanced, quantile, method="linear")
        )
        assert profile.improvement_margin_threshold == float(
            np.quantile(margins, quantile, method="linear")
        )
    assert profiles.profiles[0].nominal_uncertainty_threshold == float(
        np.quantile(uncertainty_proxy, 0.25, method="linear")
    )
    assert profiles.profiles[1].nominal_uncertainty_threshold == float(
        np.quantile(uncertainty_proxy, 0.50, method="linear")
    )
    assert profiles.profiles[2].nominal_uncertainty_threshold == float(
        np.quantile(uncertainty_proxy, 0.75, method="linear")
    )
    assert profiles.validation_prediction_digests == tuple(
        item["validation_prediction_digest"] for item in thresholds
    )


def test_m4d_config_unknown_fields_fail_closed(tmp_path: Path) -> None:
    source = Path("configs/control/m4d/fault-schedule-v1.json")
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    target = tmp_path / "bad.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(FallbackShieldError, match="unexpected or missing"):
        load_fault_schedule(target)


def test_nominal_schedule_is_content_bound_and_selector_independent() -> None:
    schedule = load_fault_schedule(Path("configs/control/m4d/fault-schedule-v1.json"))
    candidate_pool = pool()
    clean = designate_nominal(
        candidate_pool, scenario=NominalScenario.CLEAN, schedule=schedule
    )
    fault_a = designate_nominal(
        candidate_pool, scenario=NominalScenario.FAULT_INJECTED, schedule=schedule
    )
    fault_b = designate_nominal(
        candidate_pool, scenario=NominalScenario.FAULT_INJECTED, schedule=schedule
    )
    assert clean.definition_ordinal == 0
    assert fault_a == fault_b
    assert fault_a.candidate_id in candidate_pool.ordered_candidate_ids

    counts = {"clean": 0, "moderate_or_shifted": 0, "severe": 0}
    for index in range(4000):
        item = designate_nominal(
            pool(f"source-{index}"),
            scenario=NominalScenario.FAULT_INJECTED,
            schedule=schedule,
        )
        counts[item.fault_class] += 1
    assert 0.66 < counts["clean"] / 4000 < 0.74
    assert 0.12 < counts["moderate_or_shifted"] / 4000 < 0.18
    assert 0.12 < counts["severe"] / 4000 < 0.18


def test_conservative_gate_requires_all_conditions_and_stable_ties() -> None:
    profile = GateProfileV1(
        profile_id="balanced",
        nominal_risk_threshold=0.5,
        improvement_margin_threshold=0.2,
        nominal_uncertainty_threshold=0.1,
        alternative_uncertainty_threshold=0.1,
        conservatism_ordinal=1,
        quantile_semantic="test_quantiles_v1",
    )
    scores = CandidateRiskScoresV1(
        risks={"nominal": 0.8, "candidate-b": 0.2, "candidate-a": 0.2},
        uncertainties={"nominal": 0.05, "candidate-b": 0.05, "candidate-a": 0.05},
        ensemble_identity=digest("ensemble"),
    )
    decision = ConservativeOverrideGateV1(profile).decide("nominal", scores)
    assert decision.override is True
    assert decision.selected_candidate_id == "candidate-a"

    uncertain = CandidateRiskScoresV1(
        risks=scores.risks,
        uncertainties={"nominal": 0.11, "candidate-b": 0.05, "candidate-a": 0.05},
        ensemble_identity=digest("ensemble"),
    )
    assert (
        ConservativeOverrideGateV1(profile)
        .decide("nominal", uncertain)
        .selected_candidate_id
        == "nominal"
    )


@dataclass
class FakeScorer:
    selector_id: str = "fake-risk-ensemble"
    visual: bool = False
    calls: int = 0

    def score_candidates(self, candidate_pool, current_boundary):
        del current_boundary
        assert not hasattr(candidate_pool, "fault_class")
        assert not hasattr(candidate_pool, "injected_fault")
        self.calls += 1
        risks = {
            candidate_id: 0.8 if index == 0 else 0.1 + index * 0.01
            for index, candidate_id in enumerate(candidate_pool.ordered_candidate_ids)
        }
        return CandidateRiskScoresV1(
            risks=risks,
            uncertainties={candidate_id: 0.01 for candidate_id in risks},
            ensemble_identity=digest("fake-risk-ensemble"),
        )


class FakePoolFactory:
    def build(
        self,
        source,
        current_boundary,
        *,
        decision_ordinal,
        nominal_plan_index,
    ):
        base = source.actions[nominal_plan_index : nominal_plan_index + 16]
        candidates = tuple(
            np.asarray(base + (index + 1) * 0.001, dtype=np.float64)
            for index in range(8)
        )
        return simple_pool_from_actions(
            source,
            current_boundary,
            decision_ordinal=decision_ordinal,
            nominal_plan_index=nominal_plan_index,
            candidate_actions=candidates,
            candidate_pool_configuration_digest=(
                "sha256:b6a532550ba27e3ecc6acf16ccae0138fce97dd0fd8533724bd74dce2c3a714f"
            ),
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.step = 0

    def start(self, source, *, visual_domain):
        del source
        assert visual_domain in {
            "not_applicable",
            "canonical",
            "strong_camera_shift",
            "strong_lighting_shift",
        }
        self.step = 0
        return boundary(self.step)

    def restore_boundary(
        self,
        source,
        *,
        state_reference,
        expected_state_digest,
        visual_domain,
    ):
        del source, visual_domain
        self.step = int(state_reference.rsplit("-", 1)[1])
        result = boundary(self.step)
        assert result.state_digest == expected_state_digest
        return result

    def execute(self, actions, *, maximum_control_steps_remaining, visual_domain):
        del visual_domain
        executed = min(len(actions), maximum_control_steps_remaining)
        self.step += executed
        return RuntimeStepResultV1(
            next_boundary=boundary(self.step),
            executed_step_count=executed,
            task_evidence_digest=digest(f"evidence-{self.step}"),
            outcome=EpisodeState.SUCCESS,
        )

    def close(self) -> None:
        pass


def test_runner_persists_shield_decision_before_execution_and_zero_work_resume(
    tmp_path: Path,
) -> None:
    scorer = FakeScorer()
    selector = FallbackShieldSelectorV1(
        policy_id="gated_action_only__balanced",
        scenario=NominalScenario.CLEAN,
        schedule=load_fault_schedule(
            Path("configs/control/m4d/fault-schedule-v1.json")
        ),
        mode="gated",
        scorer=scorer,
        gate_profile=load_gate_profiles(
            Path("configs/control/m4d/gate-profiles-v1.json")
        ).profile("balanced"),
    )
    store = ClosedLoopEpisodeStore(tmp_path / "episode")
    first = run_closed_loop_episode(
        source_plan(),
        selector=selector,
        visual_domain="not_applicable",
        config=load_closed_loop_configuration(
            Path("configs/control/m4c/closed-loop-v1.json")
        ),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(),
        store=store,
    )
    assert first.episode.state is EpisodeState.SUCCESS
    sidecar = load_fallback_decision(
        store.boundary_dir(0) / "fallback-shield-decision.json"
    )
    assert sidecar.decision_record_digest == store.load_decision(0).content_digest
    assert sidecar.outcomes_available_during_selection is False
    assert sidecar.gate_decision == "override"
    assert sidecar.intervention_type.value == "override_to_other_alternative"
    calls = scorer.calls
    second = run_closed_loop_episode(
        source_plan(),
        selector=selector,
        visual_domain="not_applicable",
        config=load_closed_loop_configuration(
            Path("configs/control/m4c/closed-loop-v1.json")
        ),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(),
        store=store,
    )
    assert second.zero_work_resume is True
    assert scorer.calls == calls


def test_shield_sidecar_survives_crash_recovery_without_rescoring(
    tmp_path: Path,
) -> None:
    scorer = FakeScorer()
    selector = FallbackShieldSelectorV1(
        policy_id="gated_action_only__balanced",
        scenario=NominalScenario.CLEAN,
        schedule=load_fault_schedule(
            Path("configs/control/m4d/fault-schedule-v1.json")
        ),
        mode="gated",
        scorer=scorer,
        gate_profile=load_gate_profiles(
            Path("configs/control/m4d/gate-profiles-v1.json")
        ).profile("balanced"),
    )
    store = ClosedLoopEpisodeStore(tmp_path / "recover")

    def interrupt(phase: str, ordinal: int) -> None:
        if phase == "after_selection" and ordinal == 0:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_closed_loop_episode(
            source_plan(),
            selector=selector,
            visual_domain="not_applicable",
            config=load_closed_loop_configuration(
                Path("configs/control/m4c/closed-loop-v1.json")
            ),
            candidate_factory=FakePoolFactory(),
            runtime=FakeRuntime(),
            store=store,
            interruption_hook=interrupt,
        )
    before = load_fallback_decision(
        store.boundary_dir(0) / "fallback-shield-decision.json"
    )
    result = run_closed_loop_episode(
        source_plan(),
        selector=selector,
        visual_domain="not_applicable",
        config=load_closed_loop_configuration(
            Path("configs/control/m4c/closed-loop-v1.json")
        ),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(),
        store=store,
    )
    after = load_fallback_decision(
        store.boundary_dir(0) / "fallback-shield-decision.json"
    )
    assert result.episode.state is EpisodeState.SUCCESS
    assert result.episode.recovery_count == 1
    assert before.as_mapping() == after.as_mapping()
    assert scorer.calls == 1


def metric_row(
    source_id: str, success: bool, intervention: int
) -> FallbackEpisodeMetricRowV1:
    return FallbackEpisodeMetricRowV1(
        source_trajectory_id=source_id,
        policy_id="policy",
        scenario="fault_injected",
        visual_domain="not_applicable",
        outcome=EpisodeState.SUCCESS if success else EpisodeState.HORIZON_EXHAUSTED,
        executed_control_steps=20,
        decision_count=5,
        intervention_count=intervention,
        fallback_count=intervention,
        alternative_count=0,
        injected_fault_count=2,
        injected_fault_override_count=min(intervention, 2),
        injected_fault_fallback_count=min(intervention, 2),
        clean_boundary_count=3,
        clean_override_count=0,
        fault_exposure_before_first_override=0,
        nominal_risks=(0.2, 0.8),
        injected_labels=(0, 1),
        execution_seconds=1.0,
    )


def test_paired_bootstrap_is_deterministic_and_source_level() -> None:
    primary = [metric_row(f"s-{i}", i % 2 == 0, 1) for i in range(8)]
    comparator = [metric_row(f"s-{i}", False, 2) for i in range(8)]
    first = paired_fallback_bootstrap(
        primary, comparator, resamples=2000, confidence_level=0.95, seed=9
    )
    second = paired_fallback_bootstrap(
        primary, comparator, resamples=2000, confidence_level=0.95, seed=9
    )
    assert first == second
    assert first["sampling_unit"] == "source_trajectory_v1"


def test_clean_false_override_and_fault_recall_metrics_are_distinct() -> None:
    rows = [metric_row("s-1", True, 1), metric_row("s-2", False, 2)]
    metrics = aggregate_fallback_metrics(rows)
    assert metrics["injected_fault_override_recall"] == 0.75
    assert metrics["false_override_rate"] == 0.0
    assert metrics["intervention_rate"] == 0.3
    assert metrics["fault_recovery_rate"] == 0.5


def test_gate_selection_uses_clean_constraint_then_cost_tiebreak() -> None:
    metrics: dict[str, dict[str, object]] = {
        "always_fixed_primary_fallback/none/clean/not_applicable": {"success_rate": 1.0}
    }
    for policy in (
        "gated_action_only",
        "gated_direct_visual",
        "gated_distilled_visual",
        "gated_privileged_structured",
    ):
        domain = "canonical" if "visual" in policy else "not_applicable"
        for profile, fault_success, intervention in (
            ("conservative", 0.7, 0.1),
            ("balanced", 0.8, 0.2),
            ("responsive", 0.8, 0.3),
        ):
            metrics[f"{policy}/{profile}/clean/{domain}"] = {
                "outcome_counts": {
                    "unsafe": 0,
                    "execution_error": 0,
                },
                "success_rate": 1.0,
            }
            metrics[f"{policy}/{profile}/fault_injected/{domain}"] = {
                "fallback_invocation_rate": intervention,
                "intervention_rate": intervention,
                "success_rate": fault_success,
            }
    selected = select_gate_profiles(
        metrics,
        gate_profiles_digest=digest("profiles"),
        development_manifest_digest=digest("development"),
    )
    entries = selected["selected_profiles"]
    assert isinstance(entries, dict)
    assert all(entry["selected_profile_id"] == "balanced" for entry in entries.values())


def development_metrics() -> dict[str, dict[str, object]]:
    """Return a complete synthetic development aggregate inventory."""

    metrics: dict[str, dict[str, object]] = {
        "always_fixed_primary_fallback/none/clean/not_applicable": {"success_rate": 1.0}
    }
    for policy in (
        "gated_action_only",
        "gated_direct_visual",
        "gated_distilled_visual",
        "gated_privileged_structured",
    ):
        domain = "canonical" if "visual" in policy else "not_applicable"
        for profile in ("conservative", "balanced", "responsive"):
            metrics[f"{policy}/{profile}/clean/{domain}"] = {
                "outcome_counts": {"unsafe": 0, "execution_error": 0},
                "success_rate": 1.0,
            }
            metrics[f"{policy}/{profile}/fault_injected/{domain}"] = {
                "fallback_invocation_rate": 0.1,
                "intervention_rate": 0.1,
                "success_rate": 1.0,
            }
    return metrics


def test_gate_selection_freeze_rejects_content_drift(tmp_path: Path) -> None:
    record = select_gate_profiles(
        development_metrics(),
        gate_profiles_digest=digest("profiles"),
        development_manifest_digest=digest("development"),
    )
    path = write_exclusive_json(tmp_path / "selection.json", record)
    selected, _, profile_digest = load_gate_selection(path)
    assert len(selected) == 4
    assert profile_digest == digest("profiles")
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["gate_profiles_digest"] = digest("changed")
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content digest changed"):
        load_gate_selection(path)


def test_complete_fake_development_final_and_zero_work_resume(
    tmp_path: Path,
) -> None:
    matrix = load_fallback_selector_matrix(
        Path("configs/control/m4d/selector-matrix-v1.json")
    )
    profiles = load_gate_profiles(Path("configs/control/m4d/gate-profiles-v1.json"))
    schedule = load_fault_schedule(Path("configs/control/m4d/fault-schedule-v1.json"))
    scorers = {
        "action_only_ensemble_v1": FakeScorer("action_only_ensemble_v1", False),
        "direct_visual_ensemble_v1": FakeScorer("direct_visual_ensemble_v1", True),
        "distilled_visual_ensemble_v1": FakeScorer(
            "distilled_visual_ensemble_v1", True
        ),
        "privileged_structured_ensemble_v1": FakeScorer(
            "privileged_structured_ensemble_v1", False
        ),
    }
    common = {
        "matrix": matrix,
        "gate_profiles": profiles,
        "fault_schedule": schedule,
        "config": load_closed_loop_configuration(
            Path("configs/control/m4c/closed-loop-v1.json")
        ),
        "scorers": scorers,
        "candidate_factory": FakePoolFactory(),
        "runtime_factory": FakeRuntime,
    }
    development = run_fallback_benchmark_schedule(
        (source_plan(),),
        phase="development",
        selected_profiles=None,
        output_root=tmp_path / "development",
        **common,
    )
    assert development.planned_episode_count == 32
    selected = {
        "gated_action_only": "conservative",
        "gated_direct_visual": "balanced",
        "gated_distilled_visual": "responsive",
        "gated_privileged_structured": "conservative",
    }
    final = run_fallback_benchmark_schedule(
        (source_plan(),),
        phase="final",
        selected_profiles=selected,
        output_root=tmp_path / "final",
        **common,
    )
    assert final.planned_episode_count == 26
    resumed = run_fallback_benchmark_schedule(
        (source_plan(),),
        phase="final",
        selected_profiles=selected,
        output_root=tmp_path / "final",
        **common,
    )
    assert resumed.zero_work_resume is True
