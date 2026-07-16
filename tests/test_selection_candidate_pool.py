"""CPU-only tests for strict blind M3C candidate pools."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latentguard.corruptions.config import load_corruption_plan
from latentguard.integrations.maniskill_pickcube.session import (
    PickCubeReplayActionContract,
)
from latentguard.models import ActionChunk
from latentguard.selection.candidate_pool import (
    CandidatePoolBuildError,
    CandidatePoolSourceV1,
    build_candidate_pool,
)
from latentguard.selection.configuration import (
    CandidateDefinitionV1,
    CandidatePoolConfigurationError,
    CandidatePoolConfigurationV1,
    load_candidate_pool_configuration,
)
from latentguard.selection.models import (
    BlindSelectionManifestV1,
    CandidateDistribution,
    SelectionModelError,
    SelectorDecisionV1,
    SourceExclusionInventoryV1,
    SourceTrajectoryIdentityV1,
    array_content_digest,
)
from latentguard.selection.replay_dataset import build_replay_corruption_dataset
from latentguard.selection.serialization import (
    CandidatePoolSerializationError,
    load_candidate_pool,
    save_candidate_pool,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _configuration(*, base_seed: int = 271828) -> CandidatePoolConfigurationV1:
    raw = (
        (
            CandidateDistribution.ID_LIKE,
            "additive_gaussian_noise",
            {"target_fields": ["arm"], "standard_deviation": 0.005, "mean": 0.0},
            "mild_arm_gaussian_0p005",
        ),
        (
            CandidateDistribution.ID_LIKE,
            "constant_bias",
            {"target_fields": ["arm"], "bias": 0.01},
            "moderate_arm_bias_0p01",
        ),
        (
            CandidateDistribution.ID_LIKE,
            "additive_gaussian_noise",
            {"target_fields": ["arm"], "standard_deviation": 0.05, "mean": 0.0},
            "severe_arm_gaussian_0p05",
        ),
        (
            CandidateDistribution.ID_LIKE,
            "segment_hold",
            {"start_step": 1, "end_step": 16, "target_indices": list(range(8))},
            "severe_all_hold_1_16",
        ),
        (
            CandidateDistribution.SHIFTED,
            "additive_gaussian_noise",
            {"target_fields": ["arm"], "standard_deviation": 0.02, "mean": 0.0},
            "shifted_arm_gaussian_0p02",
        ),
        (
            CandidateDistribution.SHIFTED,
            "constant_bias",
            {"target_fields": ["arm"], "bias": 0.05},
            "shifted_arm_bias_0p05",
        ),
        (
            CandidateDistribution.SHIFTED,
            "segment_hold",
            {"start_step": 4, "end_step": 12, "target_indices": list(range(8))},
            "shifted_all_hold_4_12",
        ),
        (
            CandidateDistribution.SHIFTED,
            "local_temporal_permutation",
            {"start_step": 2, "end_step": 10, "target_fields": ["arm"]},
            "shifted_arm_permutation_2_10",
        ),
    )
    return CandidatePoolConfigurationV1(
        action_contract_digest=_digest("a"),
        base_seed=base_seed,
        candidates=tuple(
            CandidateDefinitionV1(
                configuration_ordinal=index,
                distribution=distribution,
                corruption_type=corruption_type,
                parameters=parameters,
                window_start=0,
                window_end=16,
                severity_id=severity,
            )
            for index, (
                distribution,
                corruption_type,
                parameters,
                severity,
            ) in enumerate(raw)
        ),
    )


def _trajectory() -> SourceTrajectoryIdentityV1:
    return SourceTrajectoryIdentityV1(
        source_trajectory_id="m3c-trajectory-10000",
        source_seed=10000,
        split_group_id="m3c-split-group-10000",
        complete_state_digests=(_digest("b"), _digest("c"), _digest("b")),
    )


def _actions() -> np.ndarray:
    time = np.linspace(-0.2, 0.2, 20, dtype=np.float64)[:, None]
    dimensions = np.linspace(0.0, 0.07, 8, dtype=np.float64)[None, :]
    return np.ascontiguousarray(time + dimensions)


def _source(*, actions: np.ndarray | None = None) -> CandidatePoolSourceV1:
    return CandidatePoolSourceV1(
        anchor_id="m3c-anchor-10000-0",
        source_episode_id="m3c-source-episode-10000-0",
        source_candidate_id="m3c-source-candidate-10000-0",
        source_policy_id="official-pickcube-solver-v1",
        source_task_id="PickCube-v1",
        trajectory=_trajectory(),
        state_tree_digest=_digest("b"),
        state_content_digest=_digest("d"),
        verifier_state_content_digest=_digest("e"),
        continuation_identity=_digest("f"),
        state_vector=np.linspace(-1.0, 1.0, 38, dtype=np.float32),
        source_remaining_action=ActionChunk(
            actions=_actions() if actions is None else actions,
            coordinate_frame="unspecified",
            control_period_s=0.05,
        ),
    )


def _contract(*, bound: float = 10.0) -> PickCubeReplayActionContract:
    return PickCubeReplayActionContract(
        total_dimension=8,
        environment_numpy_dtype="<f4",
        lower_bounds=np.full(8, -bound, dtype=np.float32),
        upper_bounds=np.full(8, bound, dtype=np.float32),
        environment_shape=(8,),
        coordinate_frame="unspecified",
        control_period_s=0.05,
    )


def _exclusions(**changes: object) -> SourceExclusionInventoryV1:
    values: dict[str, object] = {
        "dataset_digest": _digest("1"),
        "anchor_manifest_digest": _digest("2"),
        "reset_seeds": (0, 1),
        "source_trajectory_ids": ("m3a-trajectory-0",),
        "split_group_ids": ("m3a-split-group-0",),
        "complete_state_digests": (_digest("3"), _digest("3")),
    }
    values.update(changes)
    return SourceExclusionInventoryV1(**values)  # type: ignore[arg-type]


def _pool():
    plan = load_corruption_plan(
        Path("configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json")
    )
    return build_candidate_pool(
        (_source(),),
        configuration=_configuration(),
        action_layout=plan.action_layout,
        action_contract=_contract(),
        exclusion_inventory=_exclusions(),
    )


def test_pool_has_exact_4_plus_4_candidates_and_unchanged_continuation() -> None:
    source = _source()
    pool = _pool()
    assert len(pool.groups) == 1
    group = pool.groups[0]
    assert len(group.candidates) == 8
    assert len(pool.proposal_ids) == 8
    assert [item.configuration_ordinal for item in group.candidates] == list(range(8))
    assert (
        sum(
            item.distribution is CandidateDistribution.ID_LIKE
            for item in group.candidates
        )
        == 4
    )
    assert (
        sum(
            item.distribution is CandidateDistribution.SHIFTED
            for item in group.candidates
        )
        == 4
    )
    np.testing.assert_array_equal(
        group.continuation_actions,
        source.source_remaining_action.actions[16:],
    )
    source_digest = array_content_digest(source.source_remaining_action.actions[:16])
    assert group.source_action_prefix_digest == source_digest
    assert all(
        array_content_digest(item.action_chunk) != source_digest
        for item in group.candidates
    )


def test_generation_is_deterministic_and_base_seed_changes_identity() -> None:
    first = _pool()
    second = _pool()
    assert second.content_digest == first.content_digest
    assert second.proposal_ids == first.proposal_ids
    plan = load_corruption_plan(
        Path("configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json")
    )
    changed = build_candidate_pool(
        (_source(),),
        configuration=_configuration(base_seed=271829),
        action_layout=plan.action_layout,
        action_contract=_contract(),
        exclusion_inventory=_exclusions(),
    )
    assert changed.content_digest != first.content_digest
    assert changed.proposal_ids != first.proposal_ids


def test_multi_anchor_pool_bridges_to_existing_global_replay_ledger() -> None:
    plan = load_corruption_plan(
        Path("configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json")
    )
    first = _source()
    second_trajectory = replace(
        _trajectory(),
        source_trajectory_id="m3c-trajectory-10001",
        source_seed=10001,
        split_group_id="m3c-split-group-10001",
        complete_state_digests=(_digest("4"), _digest("5")),
    )
    second = replace(
        first,
        anchor_id="m3c-anchor-10001-0",
        source_episode_id="m3c-source-episode-10001-0",
        source_candidate_id="m3c-source-candidate-10001-0",
        trajectory=second_trajectory,
        state_tree_digest=_digest("4"),
        state_content_digest=_digest("6"),
        verifier_state_content_digest=_digest("7"),
        continuation_identity=_digest("8"),
    )
    configuration = _configuration()
    pool = build_candidate_pool(
        (first, second),
        configuration=configuration,
        action_layout=plan.action_layout,
        action_contract=_contract(),
        exclusion_inventory=_exclusions(),
    )
    replay = build_replay_corruption_dataset(
        (first, second),
        pool,
        configuration,
        plan.action_layout,
        source_dataset_id="m3c-source-fixture",
    )
    assert tuple(item.generation_ordinal for item in replay.proposals) == tuple(
        range(16)
    )
    assert tuple(item.proposal_id for item in replay.proposals) == pool.proposal_ids


def test_source_equivalent_noop_rejects_entire_anchor() -> None:
    plan = load_corruption_plan(
        Path("configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json")
    )
    with pytest.raises(CandidatePoolBuildError, match="source-equivalent no-op"):
        build_candidate_pool(
            (_source(actions=np.zeros((20, 8), dtype=np.float64)),),
            configuration=_configuration(),
            action_layout=plan.action_layout,
            action_contract=_contract(),
            exclusion_inventory=_exclusions(),
        )


def test_out_of_bounds_candidate_is_rejected_without_clipping() -> None:
    source_actions = _actions() * 0.01
    plan = load_corruption_plan(
        Path("configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json")
    )
    with pytest.raises(CandidatePoolBuildError, match="out_of_bounds"):
        build_candidate_pool(
            (_source(actions=source_actions),),
            configuration=_configuration(),
            action_layout=plan.action_layout,
            action_contract=_contract(bound=0.009),
            exclusion_inventory=_exclusions(),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reset_seeds", (10000,), "reset seed"),
        ("source_trajectory_ids", ("m3c-trajectory-10000",), "trajectory ID"),
        ("split_group_ids", ("m3c-split-group-10000",), "split-group"),
        ("complete_state_digests", (_digest("b"),), "state-tree digest"),
    ],
)
def test_every_m3a_source_overlap_axis_is_rejected(
    field: str, value: object, match: str
) -> None:
    plan = load_corruption_plan(
        Path("configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json")
    )
    with pytest.raises(SelectionModelError, match=match):
        build_candidate_pool(
            (_source(),),
            configuration=_configuration(),
            action_layout=plan.action_layout,
            action_contract=_contract(),
            exclusion_inventory=_exclusions(**{field: value}),
        )


def test_configuration_round_trip_is_strict_and_content_bound(tmp_path: Path) -> None:
    configuration = _configuration()
    path = tmp_path / "candidate-pool-v1.json"
    path.write_text(
        json.dumps(configuration.as_mapping(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_candidate_pool_configuration(path)
    assert loaded.content_digest == configuration.content_digest
    raw = configuration.as_mapping()
    raw["candidates"] = list(raw["candidates"])[:7]  # type: ignore[arg-type]
    path.unlink()
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CandidatePoolConfigurationError, match="exactly 8"):
        load_candidate_pool_configuration(path)


def test_candidate_pool_serialization_round_trip_and_tampering(tmp_path: Path) -> None:
    pool = _pool()
    output = save_candidate_pool(pool, tmp_path / "pool")
    loaded = load_candidate_pool(output, expected_content_digest=pool.content_digest)
    assert loaded.content_digest == pool.content_digest
    assert loaded.proposal_ids == pool.proposal_ids
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    candidate_path = (
        output / manifest["groups"][0]["candidates"][0]["action_chunk"]["path"]
    )
    data = bytearray(candidate_path.read_bytes())
    data[-1] ^= 1
    candidate_path.write_bytes(data)
    with pytest.raises(CandidatePoolSerializationError, match="file digest mismatch"):
        load_candidate_pool(output)


def test_bundle_rejects_unmanifested_files(tmp_path: Path) -> None:
    output = save_candidate_pool(_pool(), tmp_path / "pool")
    (output / "outcome.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CandidatePoolSerializationError, match="inventory differs"):
        load_candidate_pool(output)


def test_selection_manifest_is_blind_and_timestamp_is_nonsemantic() -> None:
    group = _pool().groups[0]
    ranking = tuple(sorted(group.proposal_ids))
    decision = SelectorDecisionV1(
        selector_id="temporal-five-seed-ensemble-v1",
        group_id=group.group_id,
        selected_proposal_id=ranking[0],
        ranking=ranking,
        predicted_failure_probabilities={
            proposal_id: index / 10.0 for index, proposal_id in enumerate(ranking)
        },
        abstained=False,
    )
    arguments = {
        "source_set_digest": _digest("4"),
        "candidate_pool_digest": _digest("5"),
        "candidate_pool_configuration_digest": _digest("6"),
        "selector_configuration_digest": _digest("7"),
        "verifier_bundle_digests": (("temporal", _digest("8")),),
        "input_allowlist_digest": _digest("9"),
        "selections": (decision,),
    }
    first = BlindSelectionManifestV1(
        selection_timestamp_utc="2026-07-16T08:00:00+00:00", **arguments
    )
    second = BlindSelectionManifestV1(
        selection_timestamp_utc="2026-07-16T08:01:00+00:00", **arguments
    )
    assert first.content_digest == second.content_digest
    assert first.outcomes_available_during_selection is False
    assert first.outcome_input_paths == ()
    assert BlindSelectionManifestV1.from_mapping(first.as_mapping()) == first
    tampered = first.as_mapping()
    tampered["outcomes_available_during_selection"] = True
    with pytest.raises(SelectionModelError, match="must be false"):
        BlindSelectionManifestV1.from_mapping(tampered)


def test_abstention_cannot_be_converted_to_a_selected_failure() -> None:
    ranking = _pool().groups[0].proposal_ids
    with pytest.raises(SelectionModelError, match="cannot select"):
        SelectorDecisionV1(
            selector_id="temporal-abstention-v1",
            group_id=_pool().groups[0].group_id,
            selected_proposal_id=ranking[0],
            ranking=ranking,
            predicted_failure_probabilities={},
            abstained=True,
            abstention_policy_id="validation-balanced-accuracy-v1",
        )
