"""CPU-only tests for M3C source admission and replay bridging."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import latentguard.selection.source as source_bridge
from latentguard.corruptions.config import load_corruption_plan
from latentguard.integrations.maniskill_pickcube.session import (
    PickCubeReplayActionContract,
)
from latentguard.models import ActionChunk
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.candidate_pool import (
    CandidatePoolSourceV1,
    build_candidate_pool,
)
from latentguard.selection.configuration import load_candidate_pool_configuration
from latentguard.selection.models import (
    SelectionModelError,
    SourceExclusionInventoryV1,
    SourceTrajectoryIdentityV1,
)
from latentguard.selection.replay_dataset import (
    SelectionReplayDatasetError,
    build_replay_corruption_dataset,
)
from latentguard.selection.source import (
    SelectionSourceError,
    build_m3a_exclusion_inventory,
    load_candidate_pool_sources,
    require_disjoint_source_sets,
)
from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _actions(*, offset: float = 0.0) -> np.ndarray:
    time = np.linspace(-0.2, 0.2, 20, dtype=np.float64)[:, None]
    dimensions = np.linspace(0.0, 0.07, 8, dtype=np.float64)[None, :]
    return np.ascontiguousarray(time + dimensions + offset)


def _remaining_action_digest(actions: np.ndarray) -> str:
    payload = {
        "action_byte_digest": (
            f"sha256:{hashlib.sha256(actions.tobytes(order='C')).hexdigest()}"
        ),
        "action_dtype": actions.dtype.str,
        "action_shape": list(actions.shape),
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _candidate_source(
    *,
    tag: str,
    source_seed: int,
    trajectory_id: str,
    split_group_id: str,
    state_digests: tuple[str, ...],
    offset: float = 0.0,
) -> CandidatePoolSourceV1:
    return CandidatePoolSourceV1(
        anchor_id=f"m3c-anchor-{tag}",
        source_episode_id=f"m3c-source-episode-{tag}",
        source_candidate_id=f"m3c-source-candidate-{tag}",
        source_policy_id="official-pickcube-solver-v1",
        source_task_id="PickCube-v1",
        trajectory=SourceTrajectoryIdentityV1(
            source_trajectory_id=trajectory_id,
            source_seed=source_seed,
            split_group_id=split_group_id,
            complete_state_digests=state_digests,
        ),
        state_tree_digest=state_digests[0],
        state_content_digest=_digest("d"),
        verifier_state_content_digest=_digest("e"),
        continuation_identity=_digest("f"),
        state_vector=np.linspace(-1.0, 1.0, 38, dtype=np.float32),
        source_remaining_action=ActionChunk(
            actions=_actions(offset=offset),
            coordinate_frame="unspecified",
            control_period_s=0.05,
        ),
    )


def _m3a_assignments() -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(
            source_seed=2,
            source_trajectory_id="m3a-trajectory-2",
            split_group_id="m3a-split-2",
            state_digests=(_digest("a"), _digest("b")),
        ),
        SimpleNamespace(
            source_seed=3,
            source_trajectory_id="m3a-trajectory-3",
            split_group_id="m3a-split-3",
            state_digests=(_digest("b"), _digest("c")),
        ),
    )


def _patch_m3a_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_digest: str = ACCEPTED_M3A_DATASET_DIGEST,
) -> None:
    dataset = SimpleNamespace(
        content_digest=dataset_digest,
        split_assignments=_m3a_assignments(),
    )
    manifest = SimpleNamespace(content_digest=_digest("d"))
    monkeypatch.setattr(
        source_bridge, "load_action_verifier_dataset", lambda _: dataset
    )
    monkeypatch.setattr(source_bridge, "load_anchor_manifest", lambda _: manifest)


def _patch_projection_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    numeric_component_count: int = 70,
    verifier_dimension: int = 38,
    baseline_success: bool = True,
) -> None:
    trajectory_id = "m3c-trajectory-10000"
    split_group_id = "m3c-split-10000"
    source_episode_id = "m3c-source-episode-10000"
    source_candidate_id = "m3c-source-candidate-10000"
    archive_episode_id = "m3c-archive-episode-10000"
    state_digest = _digest("1")
    state_content_digest = _digest("2")
    verifier_content_digest = _digest("3")
    verifier_schema_digest = _digest("4")
    archive_digest = _digest("5")
    action = ActionChunk(
        actions=_actions(),
        coordinate_frame="unspecified",
        control_period_s=0.05,
    )
    verifier_state = SimpleNamespace(
        values=np.linspace(-1.0, 1.0, verifier_dimension, dtype=np.float32),
        content_digest=verifier_content_digest,
        schema_digest=verifier_schema_digest,
    )
    state = SimpleNamespace(
        state_digest=state_digest,
        content_digest=state_content_digest,
        numeric_component_count=numeric_component_count,
        verifier_state=verifier_state,
    )
    archive_episode = SimpleNamespace(
        episode_id=archive_episode_id,
        source_trajectory_id=trajectory_id,
        seed=10000,
        states=(state,),
    )
    archive = SimpleNamespace(
        content_digest=archive_digest,
        episodes=(archive_episode,),
    )
    anchor = SimpleNamespace(
        anchor_id="m3c-anchor-10000-0",
        source_trajectory_id=trajectory_id,
        source_seed=10000,
        split_group_id=split_group_id,
        state_index=0,
    )
    evidence = SimpleNamespace(
        evidence_id="m3c-baseline-evidence-10000-0",
        state_restoration_verified=True,
        complete_state_comparison=True,
        compared_component_count=70,
        maximum_absolute_error=1.0e-7,
        verifier_state_restoration_verified=True,
        verifier_state_compared_component_count=38,
        verifier_state_maximum_absolute_error=1.0e-7,
        complete_action_execution=True,
        executed_action_count=20,
        expected_action_count=20,
        official_terminal_success=baseline_success,
        simulator_replay_verified=True,
    )
    record = SimpleNamespace(
        anchor=anchor,
        baseline_evidence_id=evidence.evidence_id,
        source_episode_id=source_episode_id,
        source_archive_episode_id=archive_episode_id,
        source_candidate_id=source_candidate_id,
        source_state_digest=state_digest,
        source_state_content_digest=state_content_digest,
        verifier_state_content_digest=verifier_content_digest,
        verifier_state_schema_digest=verifier_schema_digest,
        source_remaining_action_digest=_remaining_action_digest(action.actions),
        continuation_identity=_digest("6"),
    )
    manifest = SimpleNamespace(
        source_archive_content_digest=archive_digest,
        records=(record,),
        baseline_evidence=(evidence,),
        trajectory_state_digests={trajectory_id: (state_digest,)},
    )
    source_candidate = SimpleNamespace(
        candidate_id=source_candidate_id,
        action=action,
    )
    source_episode = SimpleNamespace(
        episode_id=source_episode_id,
        source_policy_id="official-pickcube-solver-v1",
        task_id="PickCube-v1",
        candidates=(source_candidate,),
    )
    monkeypatch.setattr(source_bridge, "load_state_indexed_archive", lambda _: archive)
    monkeypatch.setattr(source_bridge, "load_anchor_manifest", lambda _: manifest)
    monkeypatch.setattr(source_bridge, "load_episodes", lambda _: (source_episode,))
    monkeypatch.setattr(source_bridge, "validate_episodes", lambda _: None)


def _exclusions() -> SourceExclusionInventoryV1:
    return SourceExclusionInventoryV1(
        dataset_digest=_digest("7"),
        anchor_manifest_digest=_digest("8"),
        reset_seeds=(0, 1),
        source_trajectory_ids=("m3a-trajectory-0",),
        split_group_ids=("m3a-split-0",),
        complete_state_digests=(_digest("9"),),
    )


def _contract() -> PickCubeReplayActionContract:
    return PickCubeReplayActionContract(
        total_dimension=8,
        environment_numpy_dtype="<f4",
        lower_bounds=np.full(8, -10.0, dtype=np.float32),
        upper_bounds=np.full(8, 10.0, dtype=np.float32),
        environment_shape=(8,),
        coordinate_frame="unspecified",
        control_period_s=0.05,
    )


def test_m3a_exclusion_inventory_binds_the_accepted_complete_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_m3a_loaders(monkeypatch)

    inventory = build_m3a_exclusion_inventory(Path("dataset"), Path("anchors"))

    assert inventory.dataset_digest == ACCEPTED_M3A_DATASET_DIGEST
    assert inventory.anchor_manifest_digest == _digest("d")
    assert inventory.reset_seeds == (2, 3)
    assert inventory.source_trajectory_ids == (
        "m3a-trajectory-2",
        "m3a-trajectory-3",
    )
    assert inventory.split_group_ids == ("m3a-split-2", "m3a-split-3")
    assert inventory.complete_state_digests == (
        _digest("a"),
        _digest("b"),
        _digest("c"),
    )


def test_m3a_exclusion_inventory_rejects_an_unaccepted_dataset_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_m3a_loaders(monkeypatch, dataset_digest=_digest("0"))

    with pytest.raises(SelectionSourceError, match="accepted dataset"):
        build_m3a_exclusion_inventory(Path("dataset"), Path("anchors"))


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"source_seed": 2}, "reset seed"),
        ({"source_trajectory_id": "m3a-trajectory-2"}, "trajectory ID"),
        ({"split_group_id": "m3a-split-2"}, "split-group"),
        ({"complete_state_digests": (_digest("a"),)}, "state-tree digest"),
    ],
)
def test_built_m3a_inventory_rejects_overlap_on_every_axis(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    match: str,
) -> None:
    _patch_m3a_loaders(monkeypatch)
    inventory = build_m3a_exclusion_inventory(Path("dataset"), Path("anchors"))
    trajectory = SourceTrajectoryIdentityV1(
        source_trajectory_id="m3c-trajectory-10000",
        source_seed=10000,
        split_group_id="m3c-split-10000",
        complete_state_digests=(_digest("f"),),
    )

    with pytest.raises(SelectionModelError, match=match):
        inventory.require_disjoint(replace(trajectory, **changes))


def test_load_candidate_pool_sources_projects_only_verified_70_38_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_projection_loaders(monkeypatch)

    sources = load_candidate_pool_sources(
        Path("sources"),
        Path("archive"),
        Path("anchors"),
    )

    assert len(sources) == 1
    source = sources[0]
    assert source.trajectory.source_seed == 10000
    assert source.state_tree_digest == _digest("1")
    assert source.state_vector.shape == (38,)
    assert source.state_vector.dtype == np.dtype("<f4")
    assert not source.state_vector.flags.writeable
    np.testing.assert_array_equal(source.source_remaining_action.actions, _actions())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"numeric_component_count": 69}, "exactly 70"),
        ({"verifier_dimension": 37}, "exact 38-component"),
        ({"baseline_success": False}, "baseline does not satisfy"),
    ],
)
def test_load_candidate_pool_sources_rejects_incomplete_fixed_scope(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    match: str,
) -> None:
    _patch_projection_loaders(monkeypatch, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(SelectionSourceError, match=match):
        load_candidate_pool_sources(
            Path("sources"),
            Path("archive"),
            Path("anchors"),
        )


def test_smoke_and_full_source_sets_accept_all_four_disjoint_axes() -> None:
    smoke = _candidate_source(
        tag="smoke",
        source_seed=10000,
        trajectory_id="m3c-smoke-trajectory",
        split_group_id="m3c-smoke-split",
        state_digests=(_digest("a"),),
    )
    full = _candidate_source(
        tag="full",
        source_seed=20000,
        trajectory_id="m3c-full-trajectory",
        split_group_id="m3c-full-split",
        state_digests=(_digest("b"),),
    )

    require_disjoint_source_sets((smoke,), (full,))


@pytest.mark.parametrize(
    ("source_seed", "trajectory_id", "split_group_id", "state_digest", "match"),
    [
        (10000, "m3c-full-trajectory", "m3c-full-split", _digest("b"), "seeds"),
        (20000, "m3c-smoke-trajectory", "m3c-full-split", _digest("b"), "trajectory"),
        (20000, "m3c-full-trajectory", "m3c-smoke-split", _digest("b"), "split"),
        (20000, "m3c-full-trajectory", "m3c-full-split", _digest("a"), "state"),
    ],
)
def test_smoke_and_full_source_sets_reject_each_overlap_axis(
    source_seed: int,
    trajectory_id: str,
    split_group_id: str,
    state_digest: str,
    match: str,
) -> None:
    smoke = _candidate_source(
        tag="smoke",
        source_seed=10000,
        trajectory_id="m3c-smoke-trajectory",
        split_group_id="m3c-smoke-split",
        state_digests=(_digest("a"),),
    )
    full = _candidate_source(
        tag="full",
        source_seed=source_seed,
        trajectory_id=trajectory_id,
        split_group_id=split_group_id,
        state_digests=(state_digest,),
    )

    with pytest.raises(SelectionSourceError, match=match):
        require_disjoint_source_sets((smoke,), (full,))


def _two_anchor_pool():
    first = _candidate_source(
        tag="first",
        source_seed=10000,
        trajectory_id="m3c-trajectory-first",
        split_group_id="m3c-split-first",
        state_digests=(_digest("a"),),
    )
    second = _candidate_source(
        tag="second",
        source_seed=10001,
        trajectory_id="m3c-trajectory-second",
        split_group_id="m3c-split-second",
        state_digests=(_digest("b"),),
        offset=0.1,
    )
    plan = load_corruption_plan(
        Path("configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json")
    )
    configuration = load_candidate_pool_configuration(
        Path("configs/selection/m3c/candidate-pool-v1.json")
    )
    pool = build_candidate_pool(
        (first, second),
        configuration=configuration,
        action_layout=plan.action_layout,
        action_contract=_contract(),
        exclusion_inventory=_exclusions(),
    )
    return (first, second), pool, configuration, plan.action_layout


def test_replay_bridge_preserves_multi_anchor_global_ordinals_and_pool_identity() -> (
    None
):
    sources, pool, configuration, action_layout = _two_anchor_pool()

    replay = build_replay_corruption_dataset(
        sources,
        pool,
        configuration,
        action_layout,
        source_dataset_id="m3c-source-fixture",
    )

    assert tuple(item.generation_ordinal for item in replay.proposals) == tuple(
        range(16)
    )
    assert tuple(item.proposal_id for item in replay.proposals) == pool.proposal_ids
    assert (
        tuple(item.source_episode_id for item in replay.proposals[:8])
        == (sources[0].source_episode_id,) * 8
    )
    assert (
        tuple(item.source_episode_id for item in replay.proposals[8:])
        == (sources[1].source_episode_id,) * 8
    )


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("source_set_digest", "source inventory"),
        ("action_contract_digest", "action contract"),
    ],
)
def test_replay_bridge_rejects_tampered_pool_identity(field: str, match: str) -> None:
    sources, pool, configuration, action_layout = _two_anchor_pool()
    tampered = replace(pool, **{field: _digest("0")})

    with pytest.raises(SelectionReplayDatasetError, match=match):
        build_replay_corruption_dataset(
            sources,
            tampered,
            configuration,
            action_layout,
            source_dataset_id="m3c-source-fixture",
        )


def test_replay_bridge_rejects_group_content_detached_from_source() -> None:
    sources, pool, configuration, action_layout = _two_anchor_pool()
    changed_group = replace(pool.groups[0], state_content_digest=_digest("0"))
    tampered = replace(pool, groups=(changed_group, *pool.groups[1:]))

    with pytest.raises(SelectionReplayDatasetError, match="source content binding"):
        build_replay_corruption_dataset(
            sources,
            tampered,
            configuration,
            action_layout,
            source_dataset_id="m3c-source-fixture",
        )
