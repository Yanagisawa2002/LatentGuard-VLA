from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from latentguard.selection.blind_input import BlindCandidateGroupV1
from latentguard.selection.bootstrap import paired_trajectory_selector_bootstrap
from latentguard.selection.checkpoint_bundle import BundleLoadingDiagnosticsV1
from latentguard.selection.ensemble import (
    EnsembleEndToEndProfileV1,
    EnsembleInferenceDiagnosticsV1,
    EnsembleInferenceResultV1,
)
from latentguard.selection.metrics import CandidateOutcomeV1
from latentguard.selection.models import (
    CandidateDistribution,
    CandidateGroupV1,
    CandidateRefV1,
    SelectorDecisionV1,
    SourceTrajectoryIdentityV1,
    array_content_digest,
)
from latentguard.selection.reporting import (
    build_predeclared_interpretation,
    evaluate_selector_suite,
    render_human_readable_review,
)
from latentguard.selection.selectors import (
    AbstentionPolicyV1,
    SelectorError,
    action_magnitude_decision,
    deterministic_random_decision,
    fit_validation_ensemble_abstention_policies,
    learned_ensemble_decision,
    oracle_analysis_decision,
)
from latentguard.training.baselines import ActionMagnitudeBaselineV1

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_DIGEST_D = "sha256:" + "d" * 64


def _group(
    ordinal: int,
    *,
    trajectory_id: str | None = None,
) -> CandidateGroupV1:
    trajectory = SourceTrajectoryIdentityV1(
        source_trajectory_id=trajectory_id or f"trajectory-{ordinal}",
        source_seed=1000 + ordinal,
        split_group_id=f"split-{trajectory_id or ordinal}",
        complete_state_digests=(f"sha256:{ordinal + 1:064x}",),
    )
    candidates = tuple(
        CandidateRefV1(
            proposal_id=f"proposal-{ordinal}-{index}",
            configuration_ordinal=index,
            distribution=(
                CandidateDistribution.ID_LIKE
                if index < 4
                else CandidateDistribution.SHIFTED
            ),
            corruption_type="test_corruption",
            severity_id=f"severity-{index}",
            seed=ordinal * 100 + index,
            action_chunk=np.full((16, 8), (index + 1) / 20, dtype=np.float32),
            action_mask=np.ones(16, dtype=np.bool_),
        )
        for index in range(8)
    )
    return CandidateGroupV1(
        anchor_id=f"anchor-{ordinal}",
        trajectory=trajectory,
        state_content_digest=_DIGEST_A,
        verifier_state_content_digest=_DIGEST_B,
        continuation_identity=_DIGEST_C,
        source_action_prefix_digest=array_content_digest(
            np.zeros((16, 8), dtype=np.float32)
        ),
        state_vector=np.full(38, ordinal, dtype=np.float32),
        continuation_actions=np.zeros((2, 8), dtype=np.float32),
        candidates=candidates,
    )


def _blind(group: CandidateGroupV1) -> BlindCandidateGroupV1:
    return BlindCandidateGroupV1(
        group_id=group.group_id,
        candidate_ids=group.proposal_ids,
        state_vector=group.state_vector,
        action_chunks=np.stack(
            [candidate.action_chunk for candidate in group.candidates], axis=0
        ),
        action_masks=np.stack(
            [candidate.action_mask for candidate in group.candidates], axis=0
        ),
    )


def _inference(group: BlindCandidateGroupV1) -> EnsembleInferenceResultV1:
    probabilities = np.linspace(0.1, 0.8, 8, dtype=np.float64)
    per_seed = np.stack([probabilities] * 5)
    diagnostics = EnsembleInferenceDiagnosticsV1(
        bundle_loading=BundleLoadingDiagnosticsV1(
            device="cpu",
            total_seconds=0.0,
            per_seed_model_seconds=(0.0,) * 5,
        ),
        end_to_end_profile=EnsembleEndToEndProfileV1(
            device="cpu",
            candidate_count=8,
            group_durations_seconds=(0.008,) * 5,
        ),
    )
    return EnsembleInferenceResultV1(
        candidate_ids=group.candidate_ids,
        seed_order=(0, 1, 2, 3, 4),
        per_seed_raw_logits=np.zeros((5, 8), dtype=np.float64),
        per_seed_calibrated_failure_probabilities=per_seed,
        ensemble_failure_probabilities=np.mean(per_seed, axis=0, dtype=np.float64),
        ranking=group.candidate_ids,
        diagnostics=diagnostics,
    )


def _outcomes(
    groups: tuple[CandidateGroupV1, ...],
) -> tuple[CandidateOutcomeV1, ...]:
    values: list[CandidateOutcomeV1] = []
    for group in groups:
        for index, candidate in enumerate(group.candidates):
            values.append(
                CandidateOutcomeV1(
                    proposal_id=candidate.proposal_id,
                    group_id=group.group_id,
                    source_trajectory_id=group.source_trajectory_id,
                    distribution=candidate.distribution.value,
                    evidence_id=f"evidence-{group.anchor_id}-{index}",
                    status="conclusive",
                    success=index == 0,
                    unsafe=False,
                    simulator_replay_verified=True,
                    label_strength="strong",
                    state_component_count=70,
                )
            )
    return tuple(values)


def _fixed_decision(
    group: CandidateGroupV1,
    selector_id: str,
    selected_index: int,
) -> SelectorDecisionV1:
    selected = group.proposal_ids[selected_index]
    ranking = (selected,) + tuple(
        item for item in group.proposal_ids if item != selected
    )
    return SelectorDecisionV1(
        selector_id=selector_id,
        group_id=group.group_id,
        selected_proposal_id=selected,
        ranking=ranking,
        predicted_failure_probabilities={},
        abstained=False,
    )


def test_random_and_action_magnitude_are_deterministic_and_label_free() -> None:
    full_group = _group(1)
    group = _blind(full_group)
    random_first = deterministic_random_decision(group, seed=73)
    random_second = deterministic_random_decision(group, seed=73)
    assert random_first == random_second
    assert random_first.predicted_failure_probabilities == {}

    baseline = ActionMagnitudeBaselineV1(
        dataset_digest=_DIGEST_A,
        training_split_digest=_DIGEST_B,
        training_sample_count=10,
        valid_action_step_count=160,
        minimum_standard_deviation=1e-6,
        action_mean=np.zeros(8, dtype=np.float64),
        feature_mean=np.zeros(3, dtype=np.float64),
        feature_standard_deviation=np.ones(3, dtype=np.float64),
    )
    magnitude = action_magnitude_decision(
        group,
        baseline,
        expected_baseline_digest=baseline.content_digest,
    )
    assert magnitude.selected_proposal_id == group.candidate_ids[0]
    assert magnitude.ranking == group.candidate_ids
    with pytest.raises(SelectorError, match="frozen identity"):
        action_magnitude_decision(
            group,
            baseline,
            expected_baseline_digest=_DIGEST_D,
        )
    with pytest.raises(SelectorError, match="BlindCandidateGroupV1"):
        deterministic_random_decision(full_group, seed=73)  # type: ignore[arg-type]


def test_validation_policies_require_five_seed_predictions_and_use_group_minima() -> (
    None
):
    base = np.asarray(
        [
            0.10,
            0.20,
            0.30,
            0.40,
            0.55,
            0.65,
            0.75,
            0.85,
            0.80,
            0.70,
            0.60,
            0.50,
            0.10,
            0.20,
            0.30,
            0.40,
        ],
        dtype=np.float64,
    )
    per_seed = np.stack([base + offset for offset in (0.0, 0.01, 0.02, 0.03, 0.04)])
    targets = np.asarray([0, 0, 0, 0, 1, 1, 1, 1] * 2, dtype=np.int64)
    group_ids = ("validation-group-a",) * 8 + ("validation-group-b",) * 8
    candidate_ids = tuple(f"validation-candidate-{index}" for index in range(16))
    first = fit_validation_ensemble_abstention_policies(
        per_seed,
        targets,
        group_ids,
        candidate_ids,
        split="validation",
        verifier_bundle_digest=_DIGEST_A,
        validation_split_digest=_DIGEST_B,
    )
    second = fit_validation_ensemble_abstention_policies(
        per_seed,
        targets,
        group_ids,
        candidate_ids,
        split="validation",
        verifier_bundle_digest=_DIGEST_A,
        validation_split_digest=_DIGEST_B,
    )
    assert len(first) == 6
    assert [item.content_digest for item in first] == [
        item.content_digest for item in second
    ]
    assert first[0].policy_id == "maximum_validation_balanced_accuracy"
    assert first[1].policy_id == "target_validation_failure_recall"
    assert [item.target_coverage for item in first[2:]] == [0.9, 0.8, 0.7, 0.5]
    with pytest.raises(SelectorError, match="five-by-candidate"):
        fit_validation_ensemble_abstention_policies(
            np.mean(per_seed, axis=0),
            targets,
            group_ids,
            candidate_ids,
            split="validation",
            verifier_bundle_digest=_DIGEST_A,
            validation_split_digest=_DIGEST_B,
        )
    with pytest.raises(SelectorError, match="only on validation"):
        fit_validation_ensemble_abstention_policies(
            per_seed,
            targets,
            group_ids,
            candidate_ids,
            split="test",
            verifier_bundle_digest=_DIGEST_A,
            validation_split_digest=_DIGEST_B,
        )


def test_validation_thresholds_fit_all_candidates_when_group_minima_are_one_class() -> (
    None
):
    one_group = np.asarray(
        [0.05, 0.80, 0.70, 0.60, 0.40, 0.30, 0.20, 0.10],
        dtype=np.float64,
    )
    base = np.concatenate((one_group, one_group + 0.01))
    per_seed = np.stack([base + offset for offset in (0.0, 0.001, 0.002, 0.003, 0.004)])
    targets = np.asarray([0, 1, 1, 1, 0, 0, 0, 0] * 2, dtype=np.int64)
    group_ids = ("validation-group-a",) * 8 + ("validation-group-b",) * 8
    candidate_ids = tuple(f"validation-candidate-{index}" for index in range(16))

    policies = fit_validation_ensemble_abstention_policies(
        per_seed,
        targets,
        group_ids,
        candidate_ids,
        split="validation",
        verifier_bundle_digest=_DIGEST_A,
        validation_split_digest=_DIGEST_B,
    )

    assert policies[0].validation_balanced_accuracy == pytest.approx(1.0)
    assert policies[0].achieved_failure_recall == pytest.approx(1.0)
    assert policies[0].achieved_validation_coverage == pytest.approx(1.0)
    assert policies[1].achieved_failure_recall == pytest.approx(1.0)
    assert policies[1].target_met is True
    assert all(policy.validation_group_count == 2 for policy in policies)


def test_learned_abstention_is_strict_below_and_oracle_is_post_outcome() -> None:
    group = _group(2)
    blinded = _blind(group)
    inference = _inference(blinded)
    policy = AbstentionPolicyV1(
        policy_id="maximum_validation_balanced_accuracy",
        threshold=0.1,
        validation_prediction_digest=_DIGEST_A,
        verifier_bundle_digest=_DIGEST_B,
        validation_split_digest=_DIGEST_C,
        validation_group_count=10,
        achieved_validation_coverage=0.5,
        achieved_failure_recall=0.9,
        validation_balanced_accuracy=0.8,
    )
    abstained = learned_ensemble_decision(
        blinded,
        inference,
        selector_id="temporal-abstained",
        abstention_policy=policy,
    )
    assert abstained.abstained
    assert abstained.selected_proposal_id is None
    assert abstained.abstention_policy_id == policy.content_digest
    executed = learned_ensemble_decision(
        blinded,
        inference,
        selector_id="temporal-full",
    )
    assert not executed.abstained
    assert executed.selected_proposal_id == blinded.candidate_ids[0]

    outcomes = list(_outcomes((group,)))
    outcomes[0] = replace(outcomes[0], success=False)
    outcomes[3] = replace(outcomes[3], success=True)
    oracle = oracle_analysis_decision(group, outcomes)
    assert oracle.selected_proposal_id == group.proposal_ids[3]
    with pytest.raises(SelectorError, match="all eight"):
        oracle_analysis_decision(group, outcomes[:-1])


def test_bootstrap_resamples_trajectories_and_reporting_preserves_comparisons() -> None:
    groups = (
        _group(10, trajectory_id="trajectory-a"),
        _group(11, trajectory_id="trajectory-a"),
        _group(12, trajectory_id="trajectory-b"),
        _group(13, trajectory_id="trajectory-b"),
    )
    outcomes = _outcomes(groups)
    temporal = tuple(_fixed_decision(item, "temporal", 0) for item in groups)
    joint = tuple(_fixed_decision(item, "joint", 0) for item in groups)
    action = tuple(_fixed_decision(item, "action-only", 0) for item in groups)
    random = tuple(_fixed_decision(item, "random", 1) for item in groups)
    coverage_70 = tuple(
        _fixed_decision(item, "temporal-coverage-70", 0) for item in groups
    )

    first = paired_trajectory_selector_bootstrap(
        temporal,
        random,
        outcomes,
        resamples=2000,
        seed=101,
    )
    second = paired_trajectory_selector_bootstrap(
        temporal,
        random,
        outcomes,
        resamples=2000,
        seed=101,
    )
    assert first.to_dict() == second.to_dict()
    assert first.trajectory_count == 2
    observed = {item.metric: item.observed_difference for item in first.intervals}
    assert observed == {
        "selected_success_rate": 1.0,
        "task_failure_rate": -1.0,
        "oracle_success_regret": -1.0,
    }
    assert all(item.valid_resamples == 2000 for item in first.intervals)

    suite = evaluate_selector_suite(
        {
            "action-only": action,
            "joint": joint,
            "random": random,
            "temporal": temporal,
            "temporal-coverage-70": coverage_70,
        },
        outcomes,
        random_selector_id="random",
        learned_selector_ids=(
            "action-only",
            "joint",
            "temporal",
            "temporal-coverage-70",
        ),
        temporal_selector_id="temporal",
        joint_selector_id="joint",
        bootstrap_resamples=2000,
        bootstrap_seed=211,
    )
    payload = suite.to_dict()
    assert suite.group_count == 4
    assert suite.source_trajectory_count == 2
    assert len(suite.bootstrap_reports) == 5
    assert payload["random_selector_id"] == "random"
    assert [item.selector_id for item in suite.selector_reports] == [
        "action-only",
        "joint",
        "random",
        "temporal",
        "temporal-coverage-70",
    ]
    interpretation = build_predeclared_interpretation(
        suite,
        temporal_selector_id="temporal",
        joint_selector_id="joint",
        random_selector_id="random",
        coverage_70_selector_id="temporal-coverage-70",
        strong_replay_evidence_verified=True,
    )
    assert interpretation["online_selection_quality_targets_met"] is True
    assert (
        interpretation["joint_mlp_statistically_distinguishable_on_selected_success"]
        is False
    )
    assert interpretation["temporal_superiority_claim_supported"] is False
    assert (
        interpretation["result_disposition"]
        == "quality_targets_met_without_full_outcome_tuning"
    )
    review = render_human_readable_review(
        suite,
        interpretation,
        mode="full",
        git_sha="a" * 40,
        selection_manifest_digest=_DIGEST_A,
        candidate_pool_digest=_DIGEST_B,
        replay_evidence_digest=_DIGEST_C,
        full_pool_outcome_digest=_DIGEST_D,
        cpu_latency_report_digest=_DIGEST_A,
        gpu_latency_report_digest=_DIGEST_B,
    )
    assert "# M3C blind candidate-selection review" in review
    assert "| `temporal` |" in review
    assert "Trajectory-level paired bootstrap" in review
    assert "No VLM, LangMani" in review
    assert "raw_logit" not in review

    negative = build_predeclared_interpretation(
        suite,
        temporal_selector_id="temporal",
        joint_selector_id="joint",
        random_selector_id="random",
        coverage_70_selector_id="temporal-coverage-70",
        strong_replay_evidence_verified=False,
    )
    assert negative["online_selection_quality_targets_met"] is False
    assert "negative_result_preserved" in str(negative["result_disposition"])
