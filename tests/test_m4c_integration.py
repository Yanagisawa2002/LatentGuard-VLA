"""CPU-only M4C integration and persistence contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latentguard.control.benchmark import (
    load_episode_execution_seconds,
    record_episode_execution_seconds,
)
from latentguard.control.models import SourcePlanIdentityV1, content_digest
from latentguard.control.runner import BoundSourcePlanV1, RuntimeBoundaryV1
from latentguard.integrations.maniskill_pickcube.closed_loop import (
    PickCubeClosedLoopCandidateFactory,
)
from latentguard.integrations.maniskill_pickcube.closed_loop_state_store import (
    ClosedLoopStateStore,
    ClosedLoopStateStoreError,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_maniskill_pickcube_action_layout,
)
from latentguard.integrations.maniskill_pickcube.session import (
    PickCubeReplayActionContract,
)
from latentguard.selection.configuration import load_candidate_pool_configuration


def _digest(label: str) -> str:
    return content_digest({"label": label})


def test_content_addressed_state_store_round_trip_and_tamper(tmp_path: Path) -> None:
    store = ClosedLoopStateStore(tmp_path / "states")
    state = {
        "agent": {
            "qpos": np.arange(9, dtype=np.float32),
            "qvel": np.zeros(9, dtype=np.float64),
        },
        "elapsed": np.asarray([4], dtype=np.int64),
    }
    reference, digest = store.save(state)
    assert reference == f"m4c-state:{digest}"
    assert store.save(state) == (reference, digest)
    loaded = store.load(reference)
    assert store.save(loaded) == (reference, digest)

    manifest = tmp_path / "states" / digest.removeprefix("sha256:") / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["state_digest"] = _digest("tampered")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ClosedLoopStateStoreError, match="identity differs"):
        store.load(reference)


def test_m4c_candidate_factory_reuses_exact_eight_m3c_definitions() -> None:
    layout = load_maniskill_pickcube_action_layout(
        Path("configs/integrations/maniskill_pickcube/action-layout-v1.json")
    )
    configuration = load_candidate_pool_configuration(
        Path("configs/selection/m3c/candidate-pool-v1.json")
    )
    contract = PickCubeReplayActionContract(
        total_dimension=8,
        environment_numpy_dtype="<f4",
        lower_bounds=np.full(8, -1.0, dtype=np.float32),
        upper_bounds=np.full(8, 1.0, dtype=np.float32),
        environment_shape=(1, 8),
        coordinate_frame=layout.coordinate_frame,
        control_period_s=layout.control_period_s,
    )
    factory = PickCubeClosedLoopCandidateFactory(
        configuration=configuration,
        action_layout=layout.m1_action_layout,
        action_contract=contract,
        action_contract_digest=layout.action_contract_digest,
    )
    actions = np.linspace(-0.1, 0.1, 40 * 8, dtype=np.float64).reshape(40, 8)
    source = BoundSourcePlanV1(
        identity=SourcePlanIdentityV1(
            source_trajectory_id="m4c-test-source",
            split_group_id="m4c-test-source",
            reset_seed=420000,
            source_action_digest=_digest("actions"),
            initial_state_digest=_digest("state"),
            complete_state_tree_digests=(_digest("state"),),
            independent_replay_success=True,
            planner_identity="official-pickcube-solver",
            compatibility_identity=_digest("compatibility"),
        ),
        actions=actions,
        initial_state_reference="m4c-test-state",
    )
    boundary = RuntimeBoundaryV1(
        state_reference="m4c-test-state",
        state_digest=_digest("state"),
        verifier_state=np.zeros(38, dtype=np.float32),
    )
    pool = factory.build(source, boundary, decision_ordinal=0, nominal_plan_index=0)
    assert len(pool.candidates) == 8
    assert tuple(item.definition_ordinal for item in pool.candidates) == tuple(range(8))
    source_prefix = actions[:16].tobytes(order="C")
    assert all(
        item.action_chunk.tobytes(order="C") != source_prefix
        for item in pool.candidates
    )
    repeated = factory.build(source, boundary, decision_ordinal=0, nominal_plan_index=0)
    assert repeated.content_digest == pool.content_digest


def test_runtime_wall_time_sidecar_is_identity_bound(tmp_path: Path) -> None:
    record_episode_execution_seconds(
        tmp_path,
        episode_execution_id="episode-1",
        execution_seconds=1.25,
    )
    assert (
        load_episode_execution_seconds(
            tmp_path, expected_episode_execution_id="episode-1"
        )
        == 1.25
    )
    with pytest.raises(ValueError, match="identity differs"):
        load_episode_execution_seconds(
            tmp_path, expected_episode_execution_id="episode-2"
        )
