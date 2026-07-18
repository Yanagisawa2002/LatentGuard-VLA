from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentguard.control.config import (
    load_bootstrap_configuration,
    load_candidate_pool_binding,
    load_closed_loop_configuration,
    load_selector_matrix,
    load_source_seed_schedule,
)
from latentguard.control.models import (
    ClosedLoopCandidatePoolV1,
    ClosedLoopCandidateV1,
    ClosedLoopDecisionRecordV1,
    ClosedLoopModelError,
    SourcePlanIdentityV1,
    array_digest,
    content_digest,
    validate_source_plan_disjointness,
)
from latentguard.control.selectors import (
    build_fixed_visual_batch,
    consume_fixed_visual_features,
    deterministic_random_output,
    fixed_primary_output,
)
from latentguard.control.serialization import (
    load_candidate_pool,
    load_decision,
    save_candidate_pool,
    save_decision,
)


def digest(label: str) -> str:
    return content_digest({"label": label})


def candidate_pool() -> ClosedLoopCandidatePoolV1:
    source = np.zeros((16, 8), dtype=np.float64)
    candidates = tuple(
        ClosedLoopCandidateV1(
            candidate_id=f"candidate-{index}",
            definition_ordinal=index,
            action_chunk=np.full((16, 8), index + 1.0, dtype=np.float64),
            action_mask=np.ones(16, dtype=np.bool_),
            definition_identity=digest(f"definition-{index}"),
        )
        for index in range(8)
    )
    return ClosedLoopCandidatePoolV1(
        source_trajectory_id="trajectory-1",
        decision_ordinal=0,
        nominal_plan_index=0,
        pre_decision_state_digest=digest("state"),
        source_action_prefix_digest=array_digest(source),
        candidate_pool_configuration_digest=digest("configuration"),
        candidates=candidates,
    )


def test_checked_in_m4c_configs_are_strict_and_cross_bound() -> None:
    root = Path("configs/control/m4c")
    selectors = load_selector_matrix(root / "selector-matrix-v1.json")
    candidates = load_candidate_pool_binding(root / "candidate-pool-binding-v1.json")
    closed = load_closed_loop_configuration(root / "closed-loop-v1.json")
    seeds = load_source_seed_schedule(root / "source-seed-schedule-v1.json")
    bootstrap = load_bootstrap_configuration(root / "bootstrap-v1.json")

    assert closed.selector_matrix_identity == selectors.content_digest
    assert closed.candidate_pool_identity == candidates.content_digest
    assert closed.candidate_horizon == 16
    assert closed.execution_stride == 4
    assert closed.fixed_visual_batch_size == 128
    assert seeds.full_requested_successes == 60
    assert bootstrap.resamples == 2000


def test_candidate_pool_and_decision_round_trip(tmp_path: Path) -> None:
    pool = candidate_pool()
    fixed = fixed_primary_output(pool, primary_definition_ordinal=2)
    record = ClosedLoopDecisionRecordV1(
        episode_execution_id="episode-1",
        source_trajectory_id=pool.source_trajectory_id,
        selector_id=fixed.selector_id,
        visual_domain="not_applicable",
        decision_ordinal=0,
        nominal_plan_index=0,
        pre_decision_state_digest=pool.pre_decision_state_digest,
        candidate_pool_digest=pool.content_digest,
        ordered_candidate_ids=pool.ordered_candidate_ids,
        selector_scores=tuple(
            (key, fixed.scores[key]) for key in pool.ordered_candidate_ids
        ),
        deterministic_ranking=fixed.ranking,
        selected_candidate_id=fixed.ranking[0],
        selected_predicted_failure_probability=None,
        execution_stride=4,
        checkpoint_ensemble_identity=fixed.checkpoint_ensemble_identity,
        visual_packet_identity=None,
    )
    pool_path = save_candidate_pool(pool, tmp_path / "pool.json")
    decision_path = save_decision(record, tmp_path / "decision.json")

    assert load_candidate_pool(pool_path).content_digest == pool.content_digest
    assert load_decision(decision_path).content_digest == record.content_digest


def test_candidate_pool_rejects_exact_source() -> None:
    pool = candidate_pool()
    source = np.zeros((16, 8), dtype=np.float64)
    candidates = list(pool.candidates)
    candidates[0] = ClosedLoopCandidateV1(
        candidate_id="exact-source",
        definition_ordinal=0,
        action_chunk=source,
        action_mask=np.ones(16, dtype=np.bool_),
        definition_identity=digest("source"),
    )
    with pytest.raises(ClosedLoopModelError, match="exact source"):
        ClosedLoopCandidatePoolV1(
            source_trajectory_id=pool.source_trajectory_id,
            decision_ordinal=0,
            nominal_plan_index=0,
            pre_decision_state_digest=pool.pre_decision_state_digest,
            source_action_prefix_digest=array_digest(source),
            candidate_pool_configuration_digest=pool.candidate_pool_configuration_digest,
            candidates=tuple(candidates),
        )


def test_fixed_and_random_selectors_share_exact_pool() -> None:
    pool = candidate_pool()
    fixed = fixed_primary_output(pool)
    random_a = deterministic_random_output(pool)
    random_b = deterministic_random_output(pool)
    assert set(fixed.ranking) == set(pool.ordered_candidate_ids)
    assert random_a.ranking == random_b.ranking
    assert dict(random_a.scores) == dict(random_b.scores)


def test_fixed_batch_128_consumes_only_three_real_slots() -> None:
    images = np.arange(3 * 224 * 224 * 3, dtype=np.uint8).reshape(3, 224, 224, 3)
    batch = build_fixed_visual_batch(images)
    assert batch.shape == (128, 224, 224, 3)
    for index in range(128):
        assert np.array_equal(batch[index], images[index % 3])
    features = np.arange(128 * 512, dtype=np.float32).reshape(128, 512)
    consumed = consume_fixed_visual_features(features)
    assert np.array_equal(consumed, features[:3])
    features[3:] = -1.0
    assert np.array_equal(
        consumed, np.arange(3 * 512, dtype=np.float32).reshape(3, 512)
    )


def test_source_plan_disjointness_checks_all_identity_axes() -> None:
    plan = SourcePlanIdentityV1(
        source_trajectory_id="trajectory-new",
        split_group_id="split-new",
        reset_seed=420000,
        source_action_digest=digest("actions"),
        initial_state_digest=digest("state"),
        complete_state_tree_digests=(digest("state"), digest("terminal")),
        independent_replay_success=True,
        planner_identity="official-pickcube-planner",
        compatibility_identity=digest("compatibility"),
    )
    validate_source_plan_disjointness(
        (plan,),
        excluded_trajectory_ids=(),
        excluded_split_group_ids=(),
        excluded_state_tree_digests=(),
    )
    with pytest.raises(ClosedLoopModelError, match="state-tree digest overlap"):
        validate_source_plan_disjointness(
            (plan,),
            excluded_trajectory_ids=(),
            excluded_split_group_ids=(),
            excluded_state_tree_digests=(digest("terminal"),),
        )


def test_source_plan_inventory_preserves_repeated_physical_states() -> None:
    repeated = digest("repeated-state")
    plan = SourcePlanIdentityV1(
        source_trajectory_id="trajectory-with-hold",
        split_group_id="split-with-hold",
        reset_seed=420001,
        source_action_digest=digest("held-actions"),
        initial_state_digest=repeated,
        complete_state_tree_digests=(repeated, repeated, digest("next-state")),
        independent_replay_success=True,
        planner_identity="official-pickcube-planner",
        compatibility_identity=digest("compatibility"),
    )
    assert plan.complete_state_tree_digests[:2] == (repeated, repeated)
    validate_source_plan_disjointness(
        (plan,),
        excluded_trajectory_ids=(),
        excluded_split_group_ids=(),
        excluded_state_tree_digests=(),
    )
