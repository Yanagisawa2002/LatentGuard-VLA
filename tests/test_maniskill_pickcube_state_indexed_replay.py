"""CPU-only content-binding tests for M3A state-indexed PickCube replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray

from latentguard.corruptions.generation import generate_episode_proposals
from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import CorruptionDataset
from latentguard.corruptions.transforms import ConstantBias, WindowScopedCorruption
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.integrations.maniskill_pickcube import (
    state_indexed_archive as state_indexed_archive_module,
)
from latentguard.integrations.maniskill_pickcube import (
    state_indexed_replay as state_indexed_replay_module,
)
from latentguard.integrations.maniskill_pickcube.adapter import (
    ManiSkillPickCubeAdapter,
    ManiSkillPickCubeSemanticIdentity,
    TrustedManiSkillRuntimeAttestation,
    resolve_maniskill_pickcube_configuration,
)
from latentguard.integrations.maniskill_pickcube.anchors import PickCubeStateAnchor
from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
    StateIndexedArchiveError,
    assert_state_indexed_archive_unchanged,
    load_state_indexed_archive,
    save_state_indexed_archive,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    PICKCUBE_STATE_COMPARISON_SEMANTIC,
    PICKCUBE_STATE_COMPARISON_TOLERANCE,
    AnchorBaselineEvidenceV1,
    PickCubeActionControlContractV1,
    SourceContinuationIdentityV1,
    build_state_indexed_anchor_sources,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_replay import (
    STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY,
    BoundStateIndexedArchiveReferenceStore,
    ManiSkillPickCubeStateIndexedCaseProvider,
    ManiSkillPickCubeStateIndexedReplayError,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_UNSAFE_SEMANTIC,
    PickCubeTaskKeyContract,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    build_pickcube_verifier_state_v1,
)
from latentguard.models import (
    ActionChunk,
    Episode,
    LabelSource,
    LabelStrength,
)
from latentguard.replay.base import ReplayInvalidContextError
from latentguard.replay.models import (
    ReplayCase,
    ReplayExecutionRole,
    ReplayStateReference,
)
from latentguard.replay.source import ReplaySourceBinding

_COMPATIBILITY = f"sha256:{'a' * 64}"
_CANDIDATE_HORIZON = 3
_ACTION_COUNT = 8


def _task(index: int) -> PickCubeTaskSnapshotV1:
    return PickCubeTaskSnapshotV1(
        success=index == _ACTION_COUNT,
        is_obj_placed=index == _ACTION_COUNT,
        is_robot_static=index == _ACTION_COUNT,
        is_grasped=index >= 2,
        cube_center_z=0.02,
        cube_to_goal_distance=float(_ACTION_COUNT - index) * 0.1,
        tcp_to_cube_distance=float(max(2 - index, 0)) * 0.1,
    )


def _state(index: int) -> PickCubeIndexedStateV1:
    terminal = index == _ACTION_COUNT
    return PickCubeIndexedStateV1(
        state_index=index,
        source_action_index=index,
        tree={
            "actors": np.array([index, index + 0.25], dtype=np.float32),
            "articulations": {
                "qpos": np.array([index * 0.1, -index * 0.1], dtype=np.float64)
            },
        },
        task_snapshot=_task(index),
        restored_task_snapshot=_task(index),
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
        source_trajectory_id="mspc-trajectory-state-indexed-a",
    )


def _archive() -> PickCubeStateIndexedArchiveV1:
    actions = np.arange(_ACTION_COUNT * 2, dtype=np.float32).reshape(_ACTION_COUNT, 2)
    actions /= np.float32(32.0)
    episode = PickCubeStateIndexedEpisodeV1(
        episode_id="mspc-sequence-state-indexed-a",
        source_trajectory_id="mspc-trajectory-state-indexed-a",
        source_policy_identity="maniskill/official-pickcube-solver-v1",
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_actions=actions,
        states=tuple(_state(index) for index in range(_ACTION_COUNT + 1)),
    )
    return PickCubeStateIndexedArchiveV1(episodes=(episode,))


def _action_contract() -> PickCubeReplayActionContract:
    return PickCubeReplayActionContract(
        total_dimension=2,
        environment_numpy_dtype=np.dtype(np.float32).str,
        lower_bounds=np.array([-1.0, -1.0], dtype=np.float32),
        upper_bounds=np.array([1.0, 1.0], dtype=np.float32),
        environment_shape=(1, 2),
        coordinate_frame="panda_pd_joint_pos_v1",
        control_period_s=0.02,
    )


def _control_contract() -> PickCubeActionControlContractV1:
    return PickCubeActionControlContractV1(
        coordinate_frame="panda_pd_joint_pos_v1",
        control_period_s=0.02,
        action_dtype=np.dtype(np.float32).str,
        action_dimension=2,
    )


def _task_keys() -> PickCubeTaskKeyContract:
    return PickCubeTaskKeyContract(
        success="success",
        object_placed="is_obj_placed",
        robot_static="is_robot_static",
        grasped="is_grasped",
    )


@dataclass
class _FakeBaselineValidator:
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
        assert source_remaining_actions.shape[0] == anchor.remaining_horizon
        assert continuation.action_control_contract_digest == (
            action_control_contract.content_digest
        )
        candidate_state = source_episode.states[
            anchor.state_index + anchor.candidate_horizon
        ]
        progress_after = float(candidate_state.task_snapshot.success)
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
            progress_before=float(source_state.task_snapshot.success),
            progress_after_candidate=progress_after,
            progress_delta=progress_after - float(source_state.task_snapshot.success),
            terminal_progress=1.0,
            progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
            terminal_unsafe=False,
            unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
            label_source=LabelSource.SIMULATOR,
            label_strength=LabelStrength.STRONG,
            simulator_replay_verified=True,
        )


def _layout() -> ActionLayout:
    return ActionLayout(
        action_dim=2,
        fields=(
            ActionField(
                name="joints",
                indices=(0, 1),
                semantic=ActionSemantic.AUXILIARY,
            ),
        ),
    )


def _anchor_source(archive: PickCubeStateIndexedArchiveV1) -> Episode:
    built = build_state_indexed_anchor_sources(
        archive,
        action_control_contract=_control_contract(),
        baseline_validator=_FakeBaselineValidator(),
        candidate_horizon=_CANDIDATE_HORIZON,
        maximum_anchors_per_trajectory=6,
    )
    return next(
        episode
        for episode in built.source_episodes
        if episode.candidates[0].provenance.transformation_parameters[
            "anchor_state_index"
        ]
        > 0
    )


def _semantic_configuration() -> tuple[
    ManiSkillPickCubeSemanticIdentity,
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
    PickCubeTaskKeyContract,
    str,
]:
    identity = ManiSkillPickCubeSemanticIdentity(
        compatibility_identity=_COMPATIBILITY,
        solver_identity="solver-sha256-" + "6" * 64,
        task_implementation_identity="task-sha256-" + "7" * 64,
        controller_configuration_identity="controller-sha256-" + "8" * 64,
        action_layout_digest="sha256:" + "9" * 64,
    )
    settings = ManiSkillPickCubeEnvironmentSettings(
        obs_mode="none",
        state_tolerance=1e-6,
    )
    action_contract = _action_contract()
    task_keys = _task_keys()
    resolved = resolve_maniskill_pickcube_configuration(
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_keys=task_keys,
    )
    return (
        identity,
        settings,
        action_contract,
        task_keys,
        compute_configuration_digest(resolved),
    )


@dataclass(frozen=True)
class _ReplayFixture:
    archive: PickCubeStateIndexedArchiveV1
    source_episode: Episode
    proposal: CorruptedActionProposal
    binding: ReplaySourceBinding
    provider: ManiSkillPickCubeStateIndexedCaseProvider
    identity: ManiSkillPickCubeSemanticIdentity
    settings: ManiSkillPickCubeEnvironmentSettings
    action_contract: PickCubeReplayActionContract
    task_keys: PickCubeTaskKeyContract
    configuration_digest: str


def _fixture(
    *,
    bias: float = 0.2,
    window_end: int = _CANDIDATE_HORIZON,
) -> _ReplayFixture:
    archive = _archive()
    source_episode = _anchor_source(archive)
    result = generate_episode_proposals(
        source_episode,
        _layout(),
        (
            WindowScopedCorruption(
                inner=ConstantBias(bias=bias, target_indices=(0,)),
                window_start=0,
                window_end=window_end,
                severity_id="mild_cpu_fixture",
            ),
        ),
        base_seed=23,
        candidate_limit=1,
        strict_applicability=True,
    )
    dataset = CorruptionDataset(
        source_dataset_id="mspc-anchor-m0-state-indexed-v1",
        action_layout=_layout(),
        proposals=result.proposals,
    )
    binding = ReplaySourceBinding.from_datasets((source_episode,), dataset)
    identity, settings, action_contract, task_keys, configuration_digest = (
        _semantic_configuration()
    )
    provider = ManiSkillPickCubeStateIndexedCaseProvider(
        binding,
        archive,
        adapter_configuration_digest=configuration_digest,
        compatibility_identity=_COMPATIBILITY,
        action_contract=action_contract,
        candidate_horizon=_CANDIDATE_HORIZON,
    )
    return _ReplayFixture(
        archive=archive,
        source_episode=source_episode,
        proposal=result.proposals[0],
        binding=binding,
        provider=provider,
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_keys=task_keys,
        configuration_digest=configuration_digest,
    )


def _binding(
    source_episode: Episode,
    proposal: CorruptedActionProposal,
) -> ReplaySourceBinding:
    dataset = CorruptionDataset(
        source_dataset_id="mspc-anchor-m0-state-indexed-v1",
        action_layout=_layout(),
        proposals=(proposal,),
    )
    return ReplaySourceBinding.from_datasets((source_episode,), dataset)


def _replace_source_parameter(
    episode: Episode,
    key: str,
    value: object,
) -> Episode:
    candidate = episode.candidates[0]
    parameters = dict(candidate.provenance.transformation_parameters)
    parameters[key] = value
    provenance = replace(
        candidate.provenance,
        transformation_parameters=parameters,
    )
    return replace(
        episode,
        candidates=(replace(candidate, provenance=provenance),),
    )


def test_middle_index_case_binds_exact_archive_suffix_and_state_store(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    replay_case = fixture.provider.resolve_case(fixture.proposal)
    reference = replay_case.state_reference
    state_index = reference.state_index

    assert state_index is not None and state_index > 0
    assert reference.state_key is None
    assert reference.source_reference_id == fixture.archive.episodes[0].episode_id
    assert reference.expected_state_digest == (
        fixture.archive.episodes[0].states[state_index].state_digest
    )
    np.testing.assert_array_equal(
        replay_case.original_action.actions,
        fixture.archive.episodes[0].source_actions[state_index:],
    )
    assert replay_case.transformed_action.actions[_CANDIDATE_HORIZON:].tobytes(
        order="C"
    ) == replay_case.original_action.actions[_CANDIDATE_HORIZON:].tobytes(order="C")
    assert reference.metadata["candidate_horizon_steps"] == _CANDIDATE_HORIZON
    assert reference.metadata["anchor_state_index"] == state_index
    assert reference.metadata["source_trajectory_id"] == (
        fixture.archive.episodes[0].source_trajectory_id
    )
    assert str(reference.metadata["continuation_identity"]).startswith("sha256:")
    assert reference.metadata[STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY] == (
        fixture.archive.content_digest
    )
    assert reference.metadata["source_reset_seed"] == fixture.archive.episodes[0].seed

    archive_dir = tmp_path / "state-indexed"
    save_state_indexed_archive(fixture.archive, archive_dir)
    store = BoundStateIndexedArchiveReferenceStore(
        archive_dir,
        load_state_indexed_archive(archive_dir),
    )
    loaded = store.load_reference_state(reference)

    assert loaded.source_reference_id == reference.source_reference_id
    assert loaded.state_key == "state_index"
    assert loaded.state_index == state_index
    assert loaded.state_digest == reference.expected_state_digest
    assert loaded.compared_component_count == (
        fixture.archive.episodes[0].states[state_index].numeric_component_count
    )


def test_provider_rejects_transformed_suffix_byte_tampering() -> None:
    fixture = _fixture()
    transformed = np.array(
        fixture.proposal.transformed_action.actions,
        copy=True,
        order="C",
    )
    transformed[_CANDIDATE_HORIZON, 0] += np.float32(0.25)
    tampered = replace(
        fixture.proposal,
        transformed_action=ActionChunk(
            actions=transformed,
            coordinate_frame=fixture.proposal.transformed_action.coordinate_frame,
            control_period_s=fixture.proposal.transformed_action.control_period_s,
        ),
    )

    with pytest.raises(ReplayInvalidContextError, match="byte-identical"):
        ManiSkillPickCubeStateIndexedCaseProvider(
            _binding(fixture.source_episode, tampered),
            fixture.archive,
            adapter_configuration_digest=fixture.configuration_digest,
            compatibility_identity=_COMPATIBILITY,
            action_contract=fixture.action_contract,
            candidate_horizon=_CANDIDATE_HORIZON,
        )


def test_provider_rejects_window_horizon_and_anchor_horizon_tampering() -> None:
    archive = _archive()
    source = _anchor_source(archive)
    identity, _, action_contract, _, configuration_digest = _semantic_configuration()
    assert identity.compatibility_identity == _COMPATIBILITY

    wrong_window = generate_episode_proposals(
        source,
        _layout(),
        (
            WindowScopedCorruption(
                inner=ConstantBias(bias=0.2, target_indices=(0,)),
                window_start=0,
                window_end=_CANDIDATE_HORIZON - 1,
                severity_id="mild_wrong_horizon",
            ),
        ),
        base_seed=23,
        candidate_limit=1,
        strict_applicability=True,
    ).proposals[0]
    with pytest.raises(ReplayInvalidContextError, match="window or severity"):
        ManiSkillPickCubeStateIndexedCaseProvider(
            _binding(source, wrong_window),
            archive,
            adapter_configuration_digest=configuration_digest,
            compatibility_identity=_COMPATIBILITY,
            action_contract=action_contract,
            candidate_horizon=_CANDIDATE_HORIZON,
        )

    valid = _fixture()
    tampered_source = _replace_source_parameter(
        valid.source_episode,
        "candidate_horizon",
        _CANDIDATE_HORIZON + 1,
    )
    with pytest.raises(ManiSkillPickCubeStateIndexedReplayError, match="provenance"):
        ManiSkillPickCubeStateIndexedCaseProvider(
            _binding(tampered_source, valid.proposal),
            valid.archive,
            adapter_configuration_digest=valid.configuration_digest,
            compatibility_identity=_COMPATIBILITY,
            action_contract=valid.action_contract,
            candidate_horizon=_CANDIDATE_HORIZON,
        )


def test_provider_rejects_tampered_continuation_identity() -> None:
    fixture = _fixture()
    tampered_source = _replace_source_parameter(
        fixture.source_episode,
        "continuation_identity",
        f"sha256:{'f' * 64}",
    )

    with pytest.raises(
        ManiSkillPickCubeStateIndexedReplayError,
        match="continuation_identity",
    ):
        ManiSkillPickCubeStateIndexedCaseProvider(
            _binding(tampered_source, fixture.proposal),
            fixture.archive,
            adapter_configuration_digest=fixture.configuration_digest,
            compatibility_identity=_COMPATIBILITY,
            action_contract=fixture.action_contract,
            candidate_horizon=_CANDIDATE_HORIZON,
        )


def test_repeated_case_resolution_skips_full_archive_and_binding_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    archive_digest = fixture.archive.content_digest
    archive_digest_calls = 0
    binding_gate_calls = 0

    def _counted_archive_digest(
        self: PickCubeStateIndexedArchiveV1,
    ) -> str:
        nonlocal archive_digest_calls
        archive_digest_calls += 1
        return archive_digest

    def _counted_binding_gate(self: ReplaySourceBinding) -> None:
        nonlocal binding_gate_calls
        binding_gate_calls += 1

    monkeypatch.setattr(
        PickCubeStateIndexedArchiveV1,
        "content_digest",
        property(_counted_archive_digest),
    )
    monkeypatch.setattr(
        ReplaySourceBinding,
        "assert_unchanged",
        _counted_binding_gate,
    )

    resolved = tuple(fixture.provider.resolve_case(fixture.proposal) for _ in range(5))

    assert len({replay_case.case_id for replay_case in resolved}) == 1
    assert archive_digest_calls == 0
    assert binding_gate_calls == 0


def test_archive_store_uses_strict_cached_lookup_without_full_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    reference = fixture.provider.resolve_case(fixture.proposal).state_reference
    archive_dir = tmp_path / "state-indexed"
    save_state_indexed_archive(fixture.archive, archive_dir)
    strictly_loaded = load_state_indexed_archive(archive_dir)
    store = BoundStateIndexedArchiveReferenceStore(
        archive_dir,
        strictly_loaded,
    )

    reload_calls = 0

    def _reject_full_reload(*args: object, **kwargs: object) -> NoReturn:
        nonlocal reload_calls
        reload_calls += 1
        raise AssertionError(f"hot state lookup reloaded the archive: {args}/{kwargs}")

    monkeypatch.setattr(
        state_indexed_archive_module,
        "load_state_indexed_archive",
        _reject_full_reload,
    )
    monkeypatch.setattr(
        state_indexed_archive_module,
        "assert_state_indexed_archive_unchanged",
        _reject_full_reload,
    )
    monkeypatch.setattr(
        state_indexed_replay_module,
        "load_state_indexed_archive",
        _reject_full_reload,
    )

    first = store.load_reference_state(reference)
    second = store.load_reference_state(reference)
    store.validate_reference(reference)

    assert first.state_digest == second.state_digest
    assert reload_calls == 0


def test_archive_store_rejects_reference_digest_tampering(tmp_path: Path) -> None:
    fixture = _fixture()
    reference = fixture.provider.resolve_case(fixture.proposal).state_reference
    archive_dir = tmp_path / "state-indexed"
    save_state_indexed_archive(fixture.archive, archive_dir)
    store = BoundStateIndexedArchiveReferenceStore(
        archive_dir,
        load_state_indexed_archive(archive_dir),
    )

    wrong_state_digest = replace(
        reference,
        expected_state_digest=f"sha256:{'f' * 64}",
    )
    with pytest.raises(ReplayInvalidContextError, match="state digest mismatch"):
        store.load_reference_state(wrong_state_digest)

    metadata = dict(reference.metadata)
    metadata[STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY] = f"sha256:{'e' * 64}"
    wrong_archive_digest = replace(reference, metadata=metadata)
    with pytest.raises(ReplayInvalidContextError, match="archive digest mismatch"):
        store.load_reference_state(wrong_archive_digest)


def test_cached_store_defers_runtime_disk_tamper_to_final_archive_gate(
    tmp_path: Path,
) -> None:
    fixture = _fixture()
    reference = fixture.provider.resolve_case(fixture.proposal).state_reference
    archive_dir = tmp_path / "state-indexed"
    save_state_indexed_archive(fixture.archive, archive_dir)
    store = BoundStateIndexedArchiveReferenceStore(
        archive_dir,
        load_state_indexed_archive(archive_dir),
    )

    array_path = next((archive_dir / "arrays").glob("*.npy"))
    payload = bytearray(array_path.read_bytes())
    payload[-1] ^= 1
    array_path.write_bytes(payload)

    assert store.load_reference_state(reference).state_digest == (
        reference.expected_state_digest
    )
    with pytest.raises(StateIndexedArchiveError):
        assert_state_indexed_archive_unchanged(
            archive_dir,
            fixture.archive.content_digest,
        )


class _NeverReferenceValidator:
    def validate_reference(self, reference: ReplayStateReference) -> None:
        raise AssertionError(f"bounds must fail before reference load: {reference}")


class _NeverSessionFactory:
    def create_session(
        self,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
    ) -> NoReturn:
        raise AssertionError(
            f"session must not be created for {replay_case.case_id}/{execution_role}"
        )


def test_out_of_bounds_corruption_is_deferred_until_adapter_resolve_case() -> None:
    fixture = _fixture(bias=2.0)
    fixture.provider.resolve_case(fixture.proposal)
    adapter = ManiSkillPickCubeAdapter(
        case_provider=fixture.provider,
        replay_bundle=fixture.provider.replay_bundle,
        identity=fixture.identity,
        settings=fixture.settings,
        action_contract=fixture.action_contract,
        task_key_contract=fixture.task_keys,
        reference_validator=_NeverReferenceValidator(),
        session_factory=_NeverSessionFactory(),
        trust_attestation=TrustedManiSkillRuntimeAttestation(
            compatibility_identity=fixture.identity.compatibility_identity,
            semantic_configuration_digest=fixture.configuration_digest,
            compatibility_probe_passed=True,
            dependency_versions_verified=True,
            action_contract_verified=True,
            state_api_verified=True,
            state_round_trip_verified=True,
            task_contract_verified=True,
            solver_source_verified=True,
            task_source_verified=True,
            real_simulator_runtime=True,
        ),
    )

    with pytest.raises(
        ReplayInvalidContextError,
        match="corrupted_action_out_of_bounds",
    ):
        adapter.resolve_case(fixture.proposal)


def test_replay_case_and_bundle_identity_are_path_independent(tmp_path: Path) -> None:
    fixture = _fixture()
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_state_indexed_archive(fixture.archive, first)
    save_state_indexed_archive(fixture.archive, second)

    first_store = BoundStateIndexedArchiveReferenceStore(
        first, load_state_indexed_archive(first)
    )
    second_store = BoundStateIndexedArchiveReferenceStore(
        second, load_state_indexed_archive(second)
    )
    replay_case = fixture.provider.resolve_case(fixture.proposal)

    assert fixture.provider.replay_bundle.bundle_digest.startswith("rpb-sha256-")
    assert replay_case.case_id.startswith("rpc-sha256-")
    assert first_store.load_reference_state(
        replay_case.state_reference
    ).state_digest == (
        second_store.load_reference_state(replay_case.state_reference).state_digest
    )
