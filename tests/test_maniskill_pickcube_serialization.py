"""CPU-only content-binding tests for PickCube replay serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    save_corruption_dataset,
)
from latentguard.corruptions.transforms import ConstantBias
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.integrations.maniskill_pickcube.adapter import (
    ManiSkillPickCubeSemanticIdentity,
    resolve_maniskill_pickcube_configuration,
)
from latentguard.integrations.maniskill_pickcube.archive import (
    ManiSkillReferenceArchive,
    ManiSkillReferenceEpisode,
    save_reference_archive,
)
from latentguard.integrations.maniskill_pickcube.compatibility import (
    ActionSpaceContract,
    CompatibilityBinding,
    CompatibilityReport,
    ControllerComponentContract,
    ControllerContract,
    ManiSkillCompatibilityError,
    ModuleSourceIdentity,
    OperationalRuntimeMetadata,
    ProbeCheckResult,
    RuntimeApiContract,
    StateRoundTripResult,
    validate_compatibility_report,
    write_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    ExpectedManiSkillPickCubeContract,
)
from latentguard.integrations.maniskill_pickcube.reporting import (
    PickCubeReportingError,
    build_collection_summary,
    build_incomplete_collection_summary,
    write_sanitized_report,
)
from latentguard.integrations.maniskill_pickcube.serialization import (
    BoundArchiveReferenceStore,
    ManiSkillPickCubeArtifactError,
    ManiSkillPickCubeCaseProvider,
    action_contract_from_compatibility,
    build_maniskill_pickcube_adapter,
    environment_settings_from_compatibility,
)
from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    SourceAttemptRecord,
    SourceCollectionResult,
)
from latentguard.integrations.maniskill_pickcube.source_import import (
    build_m0_source_episode,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
)
from latentguard.models import ActionChunk, Episode
from latentguard.replay.base import ReplayInvalidContextError
from latentguard.replay.models import StateComparisonSemantic
from latentguard.replay.registry import create_replay_adapter
from latentguard.replay.source import ReplaySourceBinding
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    save_episodes,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _compatibility_report() -> CompatibilityReport:
    arm = ControllerComponentContract(
        name="arm",
        indices=(0, 1),
        controller_type="fake.PDJointPosController",
        configuration_identity=_sha("1"),
    )
    gripper = ControllerComponentContract(
        name="gripper",
        indices=(2,),
        controller_type="fake.GripperController",
        configuration_identity=_sha("2"),
    )
    return CompatibilityReport(
        mani_skill_version="3.0.1",
        sapien_version="3.fake",
        mplib_version="0.fake",
        environment_id="PickCube-v1",
        robot_uid="panda",
        num_envs=1,
        reset_seed=0,
        observation_mode="none",
        control_mode="pd_joint_pos",
        sim_backend_request="gpu",
        resolved_sim_backend="physx_cuda",
        gpu_simulation=True,
        action_space=ActionSpaceContract(
            batched_shape=(1, 3),
            single_shape=(3,),
            dtype="float32",
            lower_bounds=(-1.0, -1.0, -1.0),
            upper_bounds=(1.0, 1.0, 1.0),
        ),
        control_frequency_hz=20.0,
        simulation_frequency_hz=100.0,
        runtime_apis=RuntimeApiContract(
            get_state_dict_callable=True,
            set_state_dict_callable=True,
            evaluate_callable=True,
        ),
        task_evaluator_keys=(
            "is_grasped",
            "is_obj_placed",
            "is_robot_static",
            "success",
        ),
        source_solver=ModuleSourceIdentity(
            module_name=(
                "mani_skill.examples.motionplanning.panda.solutions.pick_cube"
            ),
            source_sha256=_sha("3"),
        ),
        task_implementation=ModuleSourceIdentity(
            module_name="mani_skill.envs.tasks.tabletop.pick_cube",
            source_sha256=_sha("4"),
        ),
        controller=ControllerContract(
            components=(arm, gripper),
            configuration_identity=_sha("5"),
        ),
        state_tree_structure_digest=_sha("6"),
        state_round_trip=StateRoundTripResult(
            passed=True,
            expected_state_digest=_sha("7"),
            observed_state_digest=_sha("7"),
            compared_leaf_count=3,
            maximum_absolute_error=0.0,
            tolerance=1e-6,
        ),
        bounded_action_step=ProbeCheckResult(
            passed=True,
            detail="bounded action executed once",
        ),
        environment_close=ProbeCheckResult(
            passed=True,
            detail="environment closed cleanly",
        ),
        operational=OperationalRuntimeMetadata(
            python_version="3.11.fake",
            torch_version="2.fake",
            cuda_runtime_version="12.fake",
            gpu_model="Fake GPU",
            gpu_capability="9.0",
        ),
    )


def _expected_contract(
    report: CompatibilityReport,
    *,
    status: str = "verified",
) -> ExpectedManiSkillPickCubeContract:
    return ExpectedManiSkillPickCubeContract(
        contract_status=status,
        mani_skill_version=report.mani_skill_version,
        sapien_version=report.sapien_version,
        mplib_version=report.mplib_version,
        environment_id=report.environment_id,
        robot_uid=report.robot_uid,
        num_envs=report.num_envs,
        observation_mode=report.observation_mode,
        control_mode=report.control_mode,
        sim_backend_request=report.sim_backend_request,
        resolved_sim_backend=report.resolved_sim_backend,
        action_dimension=report.action_space.action_dimension,
        action_dtype=report.action_space.dtype,
        action_contract_digest=report.action_contract_digest,
        control_frequency_hz=report.control_frequency_hz,
        simulation_frequency_hz=report.simulation_frequency_hz,
        solver_sha256=report.source_solver.source_sha256,
        task_sha256=report.task_implementation.source_sha256,
        controller_configuration_identity=(report.controller.configuration_identity),
        state_tree_structure_digest=report.state_tree_structure_digest,
        compatibility_identity=report.compatibility_identity,
        state_round_trip_tolerance=report.state_round_trip.tolerance,
    )


def _compatibility_binding(*, trusted: bool = True) -> CompatibilityBinding:
    report = _compatibility_report()
    return validate_compatibility_report(
        report,
        _expected_contract(report, status="verified" if trusted else "probe_required"),
        require_trusted=trusted,
    )


def _reference_episode(
    binding: CompatibilityBinding,
    action_contract: PickCubeReplayActionContract,
    settings: ManiSkillPickCubeEnvironmentSettings,
    *,
    action_contract_override: dict[str, object] | None = None,
) -> ManiSkillReferenceEpisode:
    return ManiSkillReferenceEpisode(
        compatibility_identity=binding.report.compatibility_identity,
        source_solver_identity=binding.report.source_solver.to_dict(),
        environment_configuration=settings.as_mapping(),
        seed=17,
        source_actions=np.array([[0.1, 0.2, -0.3], [0.2, 0.1, -0.2]], dtype=np.float32),
        initial_state={
            "actors": {"cube": np.array([0.1, -0.2, 0.03], dtype=np.float32)},
            "agent": np.array([0.0, 0.1, 0.2], dtype=np.float32),
        },
        terminal_state={
            "actors": {"cube": np.array([0.0, 0.0, 0.04], dtype=np.float32)},
            "agent": np.array([0.2, 0.1, 0.0], dtype=np.float32),
        },
        terminal_task_evidence={
            "success": True,
            "is_obj_placed": True,
            "is_robot_static": True,
            "is_grasped": False,
            "cube_center_z": 0.04,
            "cube_to_goal_distance": 0.0,
        },
        initial_robot_state=np.array(
            [0.0, 0.1, 0.2, 0.01, 0.02, 0.03], dtype=np.float32
        ),
        robot_state_joint_names=("joint1", "joint2", "joint3"),
        robot_state_semantic="panda_active_joint_qpos_qvel_named_v1",
        action_contract=(
            action_contract.as_mapping()
            if action_contract_override is None
            else action_contract_override
        ),
        action_coordinate_frame=action_contract.coordinate_frame,
        control_period_s=action_contract.control_period_s,
        source_generation_success=True,
        independent_baseline_success=True,
    )


def _layout() -> ActionLayout:
    return ActionLayout(
        action_dim=3,
        fields=(
            ActionField(
                name="arm",
                indices=(0, 1),
                semantic=ActionSemantic.UNSPECIFIED,
                units="unspecified",
            ),
            ActionField(
                name="gripper",
                indices=(2,),
                semantic=ActionSemantic.UNSPECIFIED,
                units="unspecified",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Artifacts:
    compatibility: CompatibilityBinding
    settings: ManiSkillPickCubeEnvironmentSettings
    action_contract: PickCubeReplayActionContract
    archive: ManiSkillReferenceArchive
    source_episode: Episode
    source_binding: ReplaySourceBinding


def _artifacts(*, corruption_bias: float = 0.05) -> _Artifacts:
    compatibility = _compatibility_binding()
    settings = environment_settings_from_compatibility(compatibility)
    action_contract = action_contract_from_compatibility(
        compatibility,
        coordinate_frame="unspecified",
    )
    reference = _reference_episode(compatibility, action_contract, settings)
    archive = ManiSkillReferenceArchive((reference,))
    source_episode = build_m0_source_episode(reference)
    generated = generate_corruption_proposals(
        (source_episode,),
        _layout(),
        (
            ConstantBias(
                bias=corruption_bias,
                target_indices=(0,),
            ),
        ),
        base_seed=31,
    )
    corruption_dataset = CorruptionDataset(
        source_dataset_id=_sha("8"),
        action_layout=_layout(),
        proposals=generated.proposals,
    )
    source_binding = ReplaySourceBinding.from_datasets(
        (source_episode,), corruption_dataset
    )
    return _Artifacts(
        compatibility=compatibility,
        settings=settings,
        action_contract=action_contract,
        archive=archive,
        source_episode=source_episode,
        source_binding=source_binding,
    )


def _provider(artifacts: _Artifacts) -> ManiSkillPickCubeCaseProvider:
    identity = ManiSkillPickCubeSemanticIdentity.from_compatibility_binding(
        artifacts.compatibility,
        action_layout_digest=_sha("9"),
    )
    task_keys = PickCubeTaskKeyContract.from_compatibility_report(
        artifacts.compatibility.report
    )
    configuration = resolve_maniskill_pickcube_configuration(
        identity=identity,
        settings=artifacts.settings,
        action_contract=artifacts.action_contract,
        task_keys=task_keys,
    )
    return ManiSkillPickCubeCaseProvider(
        artifacts.source_binding,
        artifacts.archive,
        adapter_configuration_digest=compute_configuration_digest(configuration),
        compatibility_identity=(artifacts.compatibility.report.compatibility_identity),
        action_contract=artifacts.action_contract,
        environment_settings=artifacts.settings,
        source_solver_identity=(artifacts.compatibility.report.source_solver.to_dict()),
    )


def _flip_last_byte(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)


def test_contracts_are_derived_exactly_from_trusted_compatibility() -> None:
    binding = _compatibility_binding()

    action = action_contract_from_compatibility(
        binding, coordinate_frame="pd_joint_pos"
    )
    settings = environment_settings_from_compatibility(binding)

    assert action.total_dimension == 3
    assert action.environment_numpy_dtype == np.dtype(np.float32).str
    assert action.environment_shape == (1, 3)
    assert action.control_period_s == pytest.approx(0.05)
    np.testing.assert_array_equal(
        action.lower_bounds, np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    )
    assert settings.environment_id == "PickCube-v1"
    assert settings.obs_mode == "none"
    assert settings.control_mode == "pd_joint_pos"
    assert settings.sim_backend == "gpu"
    assert settings.state_tolerance == pytest.approx(1e-6)


def test_probe_only_binding_cannot_build_runtime_contracts() -> None:
    binding = _compatibility_binding(trusted=False)

    with pytest.raises(ManiSkillCompatibilityError, match="probe-only"):
        action_contract_from_compatibility(binding, coordinate_frame="pd_joint_pos")
    with pytest.raises(ManiSkillCompatibilityError, match="probe-only"):
        environment_settings_from_compatibility(binding)


def test_replay_case_and_bundle_bind_archive_without_runtime_paths() -> None:
    artifacts = _artifacts()
    provider = _provider(artifacts)
    proposal = artifacts.source_binding.corruption_dataset.proposals[0]

    replay_case = provider.resolve_case(proposal)
    reference = artifacts.archive.episodes[0]

    assert replay_case.state_reference.source_reference_id == reference.episode_id
    assert (
        replay_case.state_reference.expected_state_digest
        == reference.initial_state_digest
    )
    assert (
        replay_case.state_reference.comparison_semantic
        is StateComparisonSemantic.EXACT_DIGEST
    )
    assert (
        replay_case.state_reference.metadata["reference_archive_content_digest"]
        == artifacts.archive.content_digest
    )
    assert (
        provider.replay_bundle.metadata["reference_archive_content_digest"]
        == artifacts.archive.content_digest
    )
    assert "runtime_archive" not in str(replay_case.state_reference.metadata)


def test_adapter_and_bundle_identities_are_archive_path_independent(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    first = tmp_path / "first-runtime-archive"
    second = tmp_path / "second-runtime-archive"
    save_reference_archive(artifacts.archive, first)
    save_reference_archive(artifacts.archive, second)

    first_adapter = build_maniskill_pickcube_adapter(
        source_binding=artifacts.source_binding,
        archive_dir=first,
        compatibility_binding=artifacts.compatibility,
        action_layout_digest=_sha("9"),
        coordinate_frame=artifacts.action_contract.coordinate_frame,
    )
    second_adapter = build_maniskill_pickcube_adapter(
        source_binding=artifacts.source_binding,
        archive_dir=second,
        compatibility_binding=artifacts.compatibility,
        action_layout_digest=_sha("9"),
        coordinate_frame=artifacts.action_contract.coordinate_frame,
    )

    assert first_adapter.configuration_digest == second_adapter.configuration_digest
    assert first_adapter.replay_bundle.bundle_digest == (
        second_adapter.replay_bundle.bundle_digest
    )
    assert [case.case_id for case in first_adapter.replay_bundle.replay_cases] == [
        case.case_id for case in second_adapter.replay_bundle.replay_cases
    ]
    semantic_text = str(first_adapter.resolved_configuration())
    assert str(first) not in semantic_text
    assert str(second) not in semantic_text


def test_provider_rejects_changed_m0_content_even_when_ids_are_reused() -> None:
    artifacts = _artifacts()
    candidate = artifacts.source_episode.candidates[0]
    changed_action = ActionChunk(
        actions=candidate.action.actions + np.float32(0.01),
        coordinate_frame=candidate.action.coordinate_frame,
        control_period_s=candidate.action.control_period_s,
    )
    changed_source = replace(
        artifacts.source_episode,
        candidates=(replace(candidate, action=changed_action),),
    )
    changed_binding = ReplaySourceBinding.from_datasets(
        (changed_source,), artifacts.source_binding.corruption_dataset
    )
    changed_artifacts = replace(
        artifacts,
        source_episode=changed_source,
        source_binding=changed_binding,
    )

    with pytest.raises(
        ManiSkillPickCubeArtifactError, match="M0 source episode content differs"
    ):
        _provider(changed_artifacts)


def test_provider_rejects_out_of_bounds_corruption_before_bundle_creation() -> None:
    artifacts = _artifacts(corruption_bias=2.0)

    with pytest.raises(ReplayInvalidContextError, match="out_of_bounds"):
        _provider(artifacts)


def test_builder_rejects_archive_action_contract_drift(tmp_path: Path) -> None:
    artifacts = _artifacts()
    changed_reference = _reference_episode(
        artifacts.compatibility,
        artifacts.action_contract,
        artifacts.settings,
        action_contract_override={"schema_version": "changed"},
    )
    changed_archive = ManiSkillReferenceArchive((changed_reference,))
    changed_source = build_m0_source_episode(changed_reference)
    generated = generate_corruption_proposals(
        (changed_source,),
        _layout(),
        (ConstantBias(bias=0.05, target_indices=(0,)),),
        base_seed=31,
    )
    changed_binding = ReplaySourceBinding.from_datasets(
        (changed_source,),
        CorruptionDataset(
            source_dataset_id=_sha("8"),
            action_layout=_layout(),
            proposals=generated.proposals,
        ),
    )
    archive_dir = tmp_path / "changed-contract"
    save_reference_archive(changed_archive, archive_dir)

    with pytest.raises(ManiSkillPickCubeArtifactError, match="action contract differs"):
        build_maniskill_pickcube_adapter(
            source_binding=changed_binding,
            archive_dir=archive_dir,
            compatibility_binding=artifacts.compatibility,
            action_layout_digest=_sha("9"),
            coordinate_frame=artifacts.action_contract.coordinate_frame,
        )


def test_bound_store_rejects_full_archive_tampering(tmp_path: Path) -> None:
    artifacts = _artifacts()
    archive_dir = tmp_path / "runtime-archive"
    save_reference_archive(artifacts.archive, archive_dir)
    provider = _provider(artifacts)
    reference = provider.replay_bundle.replay_cases[0].state_reference
    store = BoundArchiveReferenceStore(
        runtime_archive_directory=archive_dir,
        archive_content_digest=artifacts.archive.content_digest,
    )
    _flip_last_byte(next((archive_dir / "arrays").glob("*.npy")))

    with pytest.raises(ReplayInvalidContextError, match="archive changed"):
        store.validate_reference(reference)


def test_adapter_resolve_revalidates_archive_after_construction(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    archive_dir = tmp_path / "runtime-archive"
    save_reference_archive(artifacts.archive, archive_dir)
    adapter = build_maniskill_pickcube_adapter(
        source_binding=artifacts.source_binding,
        archive_dir=archive_dir,
        compatibility_binding=artifacts.compatibility,
        action_layout_digest=_sha("9"),
        coordinate_frame=artifacts.action_contract.coordinate_frame,
    )
    proposal = artifacts.source_binding.corruption_dataset.proposals[0]
    adapter.resolve_case(proposal)
    _flip_last_byte(next((archive_dir / "arrays").glob("*.npy")))

    with pytest.raises(ReplayInvalidContextError, match="archive changed"):
        adapter.resolve_case(proposal)


def test_builder_rejects_invalid_action_layout_digest_before_runtime_use(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()

    with pytest.raises(ManiSkillPickCubeArtifactError, match="action layout digest"):
        build_maniskill_pickcube_adapter(
            source_binding=artifacts.source_binding,
            archive_dir=tmp_path / "unused",
            compatibility_binding=artifacts.compatibility,
            action_layout_digest="not-a-digest",
            coordinate_frame=artifacts.action_contract.coordinate_frame,
        )


def test_explicit_registry_factory_loads_paths_but_excludes_them_from_identity(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    source_dir = tmp_path / "source"
    corruption_dir = tmp_path / "corruption"
    archive_dir = tmp_path / "runtime-archive"
    expected_path = tmp_path / "expected.json"
    report_path = tmp_path / "report.json"
    layout_path = tmp_path / "layout.json"
    save_episodes((artifacts.source_episode,), source_dir)
    source_dataset_id = compute_episode_bundle_identifier(source_dir)
    save_corruption_dataset(
        CorruptionDataset(
            source_dataset_id=source_dataset_id,
            action_layout=artifacts.source_binding.corruption_dataset.action_layout,
            proposals=artifacts.source_binding.corruption_dataset.proposals,
        ),
        corruption_dir,
    )
    save_reference_archive(artifacts.archive, archive_dir)
    report = artifacts.compatibility.report
    expected_path.write_text(
        json.dumps(
            artifacts.compatibility.expected_contract.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_compatibility_report(report, report_path)
    layout = artifacts.source_binding.corruption_dataset.action_layout
    layout_path.write_text(
        json.dumps(
            {
                "action_contract_digest": report.action_contract_digest,
                "action_dim": layout.action_dim,
                "control_mode": report.control_mode,
                "control_period_s": report.control_period_s,
                "coordinate_frame": "unspecified",
                "description": layout.description,
                "fields": [
                    {
                        "description": field.description,
                        "indices": list(field.indices),
                        "metadata": dict(field.metadata),
                        "name": field.name,
                        "semantic": field.semantic.value,
                        "units": field.units,
                    }
                    for field in layout.fields
                ],
                "metadata": dict(layout.metadata),
                "schema_version": layout.schema_version,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    binding = ReplaySourceBinding.from_paths(source_dir, corruption_dir)
    configuration = {
        "action_layout_path": str(layout_path),
        "compatibility_report_path": str(report_path),
        "corruption_dataset_directory": str(corruption_dir),
        "expected_contract_path": str(expected_path),
        "runtime_archive_directory": str(archive_dir),
        "schema_version": "1.0",
        "source_dataset_directory": str(source_dir),
    }

    adapter = create_replay_adapter(
        "maniskill_pickcube_v1",
        configuration,
        binding,
    )

    assert adapter.adapter_id == "maniskill_pickcube_v1"
    resolved = str(adapter.resolved_configuration())
    for path in configuration.values():
        if isinstance(path, str) and path != "1.0":
            assert path not in resolved


def test_collection_reports_are_compact_sanitized_and_immutable(tmp_path: Path) -> None:
    artifacts = _artifacts()
    complete = SourceCollectionResult(
        requested_success_count=1,
        attempts=(SourceAttemptRecord(seed=17, accepted=True, failure_category=None),),
        archive=artifacts.archive,
    )
    summary = build_collection_summary(complete, source_dataset_id=_sha("a"))
    output = tmp_path / "collection-summary.json"

    write_sanitized_report(summary, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["accepted_source_count"] == 1
    assert payload["independent_baseline_success_count"] == 1
    assert "runtime-archive" not in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        write_sanitized_report(summary, output)

    incomplete = SourceCollectionResult(
        requested_success_count=2,
        attempts=(
            SourceAttemptRecord(
                seed=17,
                accepted=False,
                failure_category="FakeSolverFailure",
            ),
        ),
        archive=None,
    )
    failure = build_incomplete_collection_summary(
        incomplete,
        compatibility_identity=artifacts.compatibility.report.compatibility_identity,
    )
    assert failure["status"] == "incomplete"
    assert failure["failure_categories"] == {"FakeSolverFailure": 1}


def test_collection_report_writer_cannot_clobber_concurrent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    result = SourceCollectionResult(
        requested_success_count=1,
        attempts=(SourceAttemptRecord(seed=17, accepted=True, failure_category=None),),
        archive=artifacts.archive,
    )
    summary = build_collection_summary(result, source_dataset_id=_sha("a"))
    output = tmp_path / "collection-summary.json"
    concurrent_evidence = b"concurrent-writer-won\n"
    real_exists = Path.exists
    race_injected = False

    def inject_concurrent_writer(candidate: Path) -> bool:
        nonlocal race_injected
        observed = real_exists(candidate)
        if candidate == output.absolute() and not race_injected:
            assert not observed
            output.write_bytes(concurrent_evidence)
            race_injected = True
        return observed

    monkeypatch.setattr(Path, "exists", inject_concurrent_writer)

    with pytest.raises(PickCubeReportingError, match="already exists"):
        write_sanitized_report(summary, output)

    assert race_injected
    assert output.read_bytes() == concurrent_evidence
    assert not tuple(tmp_path.glob(f".{output.name}.tmp-*"))
