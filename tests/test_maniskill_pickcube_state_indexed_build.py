from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from latentguard.integrations.maniskill_pickcube.anchors import PickCubeStateAnchor
from latentguard.integrations.maniskill_pickcube.source_import import (
    MANISKILL_PICKCUBE_TASK_ID,
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_UNSAFE_SEMANTIC,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    ANCHOR_SOURCE_TRANSFORMATION,
    PICKCUBE_STATE_COMPARISON_SEMANTIC,
    PICKCUBE_STATE_COMPARISON_TOLERANCE,
    AnchorBaselineEvidenceV1,
    PickCubeActionControlContractV1,
    SourceContinuationIdentityV1,
    StateIndexedBuildError,
    assert_anchor_manifest_unchanged,
    build_source_continuation_identity,
    build_state_indexed_anchor_sources,
    load_anchor_manifest,
    save_anchor_manifest,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    build_pickcube_verifier_state_v1,
)
from latentguard.models import LabelSource, LabelStrength

_COMPATIBILITY = f"sha256:{'a' * 64}"


def _task(index: int, action_count: int) -> PickCubeTaskSnapshotV1:
    return PickCubeTaskSnapshotV1(
        success=index == action_count,
        is_obj_placed=index == action_count,
        is_robot_static=index == action_count,
        is_grasped=index >= 2,
        cube_center_z=0.02,
        cube_to_goal_distance=float(action_count - index) * 0.1,
        tcp_to_cube_distance=float(max(2 - index, 0)) * 0.1,
    )


def _state(index: int, action_count: int) -> PickCubeIndexedStateV1:
    terminal = index == action_count
    return PickCubeIndexedStateV1(
        state_index=index,
        source_action_index=index,
        tree={
            "actors": np.array([index, index + 0.25], dtype=np.float32),
            "articulations": {
                "qpos": np.array([index * 0.1, -index * 0.1], dtype=np.float64)
            },
        },
        task_snapshot=_task(index, action_count),
        restored_task_snapshot=_task(index, action_count),
        verifier_state=build_pickcube_verifier_state_v1(
            joint_names=("joint_a", "joint_b"),
            qpos=np.array([index, -index], dtype=np.float32),
            qvel=np.array([0.1, -0.1], dtype=np.float32),
            tcp_position=np.array([0.0, 0.0, 0.1 + index], dtype=np.float32),
            tcp_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            cube_position=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            cube_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            goal_position=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            is_grasped=index >= 2,
            is_obj_placed=terminal,
            is_robot_static=terminal,
        ),
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_trajectory_id="mspc-trajectory-source-a",
    )


def _source_episode() -> PickCubeStateIndexedEpisodeV1:
    action_count = 8
    actions = np.arange(action_count * 2, dtype=np.float32).reshape(action_count, 2)
    actions /= 32.0
    return PickCubeStateIndexedEpisodeV1(
        episode_id="mspc-sequence-episode-a",
        source_trajectory_id="mspc-trajectory-source-a",
        source_policy_identity="maniskill/3.0.1/official-pickcube-solver-v1",
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_actions=actions,
        states=tuple(_state(index, action_count) for index in range(action_count + 1)),
    )


def _archive() -> PickCubeStateIndexedArchiveV1:
    return PickCubeStateIndexedArchiveV1(episodes=(_source_episode(),))


def _contract() -> PickCubeActionControlContractV1:
    return PickCubeActionControlContractV1(
        coordinate_frame="joint_position",
        control_period_s=0.05,
        action_dtype="<f4",
        action_dimension=2,
    )


@dataclass
class _FakeBaselineValidator:
    fail_state_index: int | None = None
    weak_state_index: int | None = None
    seen_remainders: list[NDArray[Any]] | None = None
    seen_archive_digests: list[str] | None = None

    def validate_anchor_baseline(
        self,
        *,
        archive_content_digest: str,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        anchor: PickCubeStateAnchor,
        source_remaining_actions: NDArray[Any],
        continuation: SourceContinuationIdentityV1,
        action_control_contract: PickCubeActionControlContractV1,
    ) -> AnchorBaselineEvidenceV1:
        assert archive_content_digest.startswith("sha256:")
        if self.seen_archive_digests is not None:
            self.seen_archive_digests.append(archive_content_digest)
        assert action_control_contract.content_digest == (
            continuation.action_control_contract_digest
        )
        if self.seen_remainders is not None:
            self.seen_remainders.append(np.array(source_remaining_actions, copy=True))
        if anchor.state_index == self.fail_state_index:
            raise RuntimeError("runtime details must not enter exclusion evidence")
        strong = anchor.state_index != self.weak_state_index
        return AnchorBaselineEvidenceV1(
            anchor_id=anchor.anchor_id,
            source_archive_episode_id=source_episode.episode_id,
            source_trajectory_id=source_episode.source_trajectory_id,
            source_seed=source_episode.seed,
            state_index=anchor.state_index,
            source_state_digest=source_state.state_digest,
            compatibility_identity=source_episode.compatibility_identity,
            continuation_identity=continuation.content_digest,
            expected_action_count=anchor.remaining_horizon,
            executed_action_count=anchor.remaining_horizon,
            complete_action_execution=True,
            state_restoration_verified=True,
            complete_state_comparison=True,
            compared_component_count=source_state.numeric_component_count,
            maximum_absolute_error=1.1920929e-7,
            verifier_state_restoration_verified=True,
            verifier_state_compared_component_count=(
                source_state.verifier_state.values.size
            ),
            verifier_state_maximum_absolute_error=1.1920929e-7,
            comparison_semantic=PICKCUBE_STATE_COMPARISON_SEMANTIC,
            comparison_tolerance=PICKCUBE_STATE_COMPARISON_TOLERANCE,
            official_terminal_success=True,
            official_object_placed=True,
            official_robot_static=True,
            progress_before=0.0,
            progress_after_candidate=0.0,
            progress_delta=0.0,
            terminal_progress=1.0,
            progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
            terminal_unsafe=False,
            unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
            label_source=LabelSource.SIMULATOR,
            label_strength=(LabelStrength.STRONG if strong else LabelStrength.WEAK),
            simulator_replay_verified=strong,
        )


def test_continuation_identity_binds_exact_suffix_and_control_contract() -> None:
    source = _source_episode()
    contract = _contract()
    identity = build_source_continuation_identity(
        source_episode=source,
        anchor_state_index=1,
        candidate_horizon=3,
        action_control_contract=contract,
    )
    expected = source.source_actions[4:]

    assert identity.continuation_start_index == 4
    assert identity.action_shape == expected.shape
    assert identity.action_dtype == expected.dtype.str
    assert identity.action_byte_digest == (
        f"sha256:{__import__('hashlib').sha256(expected.tobytes()).hexdigest()}"
    )
    assert identity.action_control_contract_digest == contract.content_digest

    changed_actions = np.array(source.source_actions, copy=True)
    changed_actions[4, 0] += np.float32(0.01)
    changed = replace(source, source_actions=changed_actions)
    changed_identity = build_source_continuation_identity(
        source_episode=changed,
        anchor_state_index=1,
        candidate_horizon=3,
        action_control_contract=contract,
    )
    assert changed_identity.continuation_id != identity.continuation_id

    changed_contract = replace(contract, control_period_s=0.1)
    contract_identity = build_source_continuation_identity(
        source_episode=source,
        anchor_state_index=1,
        candidate_horizon=3,
        action_control_contract=changed_contract,
    )
    assert contract_identity.continuation_id != identity.continuation_id


def test_builder_creates_deterministic_one_observation_full_remainder_sources() -> None:
    archive = _archive()
    validator = _FakeBaselineValidator(seen_remainders=[])
    first = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=validator,
        candidate_horizon=3,
        maximum_anchors_per_trajectory=3,
    )
    second = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=3,
    )

    assert len(first.source_episodes) == 3
    assert first.manifest.content_digest == second.manifest.content_digest
    assert first.manifest.trajectory_state_digests == {
        archive.episodes[0].source_trajectory_id: tuple(
            state.state_digest for state in archive.episodes[0].states
        )
    }
    assert tuple(item.episode_id for item in first.source_episodes) == tuple(
        item.episode_id for item in second.source_episodes
    )
    for built, record in zip(
        first.source_episodes, first.manifest.records, strict=True
    ):
        anchor = record.anchor
        expected = archive.episodes[0].source_actions[anchor.state_index :]
        assert built.task_id == MANISKILL_PICKCUBE_TASK_ID
        assert len(built.observations.frames) == 1
        assert built.observations.frames[0].timestamp_s == 0.0
        assert built.observations.frames[0].cameras == ()
        assert np.array_equal(
            built.observations.frames[0].robot_state,
            archive.episodes[0].states[anchor.state_index].verifier_state.values,
        )
        assert len(built.candidates) == 1
        assert np.array_equal(built.candidates[0].action.actions, expected)
        assert built.candidates[0].action.actions.shape[0] == anchor.remaining_horizon
        provenance = built.candidates[0].provenance
        assert provenance.source_episode_id == built.episode_id
        assert provenance.transformation_type == ANCHOR_SOURCE_TRANSFORMATION
        assert provenance.split_group_id == anchor.split_group_id
        assert provenance.transformation_parameters["anchor_id"] == anchor.anchor_id
        assert provenance.transformation_parameters["continuation_identity"] == (
            record.continuation.content_digest
        )
        assert record.continuation_identity.startswith("sha256:")
        assert provenance.transformation_parameters["baseline_evidence_id"] == (
            record.baseline_evidence_id
        )


def test_builder_trajectory_limit_preserves_full_archive_binding() -> None:
    archive = _archive()

    result = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=1,
        source_trajectory_limit=1,
    )

    assert len(result.source_episodes) == 1
    assert result.manifest.source_archive_content_digest == archive.content_digest


def test_trajectory_state_inventory_deduplicates_a_revisited_state_tree() -> None:
    source = _source_episode()
    states = list(source.states)
    states[1] = replace(states[1], tree=states[0].tree)
    revisited = replace(source, states=tuple(states))
    archive = PickCubeStateIndexedArchiveV1(episodes=(revisited,))

    result = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=1,
    )

    inventory = result.manifest.trajectory_state_digests[revisited.source_trajectory_id]
    assert len(inventory) == len(revisited.states)
    assert inventory[0] == inventory[1]
    assert len(set(inventory)) == len(revisited.states) - 1


def test_builder_reuses_one_archive_digest_and_rechecks_only_at_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive()
    expected_digest = archive.content_digest
    original_getter = PickCubeStateIndexedArchiveV1.content_digest.fget
    assert original_getter is not None
    access_count = 0

    def counted_content_digest(instance: PickCubeStateIndexedArchiveV1) -> str:
        nonlocal access_count
        access_count += 1
        return original_getter(instance)

    monkeypatch.setattr(
        PickCubeStateIndexedArchiveV1,
        "content_digest",
        property(counted_content_digest),
    )
    seen_archive_digests: list[str] = []
    result = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(
            seen_archive_digests=seen_archive_digests
        ),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=3,
    )

    assert access_count == 2
    assert result.manifest.source_archive_content_digest == expected_digest
    assert set(seen_archive_digests) == {expected_digest}
    assert all(
        episode.candidates[0].provenance.transformation_parameters[
            "source_archive_content_digest"
        ]
        == expected_digest
        for episode in result.source_episodes
    )


def test_failed_or_non_strong_baseline_is_excluded_without_source_label() -> None:
    archive = _archive()
    scheduled = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=3,
    )
    state_indices = [record.anchor.state_index for record in scheduled.manifest.records]
    result = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(
            fail_state_index=state_indices[0], weak_state_index=state_indices[1]
        ),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=3,
    )

    assert len(result.source_episodes) == 1
    assert len(result.manifest.records) == 1
    assert len(result.manifest.baseline_evidence) == 1
    assert {item.reason_code for item in result.manifest.exclusions} == {
        "baseline_validator_error",
        "baseline_not_strong_simulator_evidence",
    }
    assert all(
        "runtime details" not in item.reason_code for item in result.manifest.exclusions
    )


def test_anchor_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    result = build_state_indexed_anchor_sources(
        _archive(),
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=2,
    )
    output = tmp_path / "anchors"
    saved = save_anchor_manifest(result.manifest, output)
    loaded = load_anchor_manifest(output)

    assert saved == output / "manifest.json"
    assert loaded.content_digest == result.manifest.content_digest
    assert loaded.trajectory_state_digests == result.manifest.trajectory_state_digests
    assert len(list((output / "evidence").glob("*.json"))) == 2
    assert loaded.baseline_evidence[0].progress_before == 0.0
    assert loaded.baseline_evidence[0].progress_after_candidate == 0.0
    assert loaded.baseline_evidence[0].progress_delta == 0.0
    assert_anchor_manifest_unchanged(output, result.manifest.content_digest)

    manifest_path = output / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest_payload = json.loads(original_manifest)
    trajectory_id = next(iter(manifest_payload["trajectory_state_digests"]))
    manifest_payload["trajectory_state_digests"][trajectory_id][-1] = (
        f"sha256:{'f' * 64}"
    )
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(StateIndexedBuildError, match="content digest mismatch"):
        load_anchor_manifest(output)
    manifest_path.write_bytes(original_manifest)

    evidence_path = output / "evidence" / "000000.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["maximum_absolute_error"] = 9e-7
    evidence_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(StateIndexedBuildError, match="content identity"):
        load_anchor_manifest(output)


def test_anchor_manifest_rejects_extra_inventory_and_nonempty_save(
    tmp_path: Path,
) -> None:
    result = build_state_indexed_anchor_sources(
        _archive(),
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=1,
    )
    output = tmp_path / "anchors"
    save_anchor_manifest(result.manifest, output)
    with pytest.raises(StateIndexedBuildError, match="non-empty"):
        save_anchor_manifest(result.manifest, output)

    (output / "evidence" / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StateIndexedBuildError, match="inventory"):
        load_anchor_manifest(output)


def test_baseline_evidence_rejects_inconsistent_binary_progress_delta() -> None:
    archive = _archive()
    source = archive.episodes[0]
    anchor_result = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=3,
        maximum_anchors_per_trajectory=1,
    )
    evidence = anchor_result.manifest.baseline_evidence[0]

    with pytest.raises(StateIndexedBuildError, match="after minus before"):
        replace(evidence, progress_after_candidate=1.0, progress_delta=0.0)
    with pytest.raises(StateIndexedBuildError, match="binary completion"):
        replace(evidence, progress_before=0.5)
    assert source.states[evidence.state_index].task_snapshot.success is False
