from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from latentguard.corruptions.layout import ActionLayout, ActionSemantic
from latentguard.integrations.maniskill_pickcube.availability import (
    ManiSkillAvailabilityError,
    discover_distribution_versions,
    load_runtime_modules,
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
    load_compatibility_report,
    validate_compatibility_report,
    write_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
    ExpectedManiSkillPickCubeContract,
    ManiSkillConfigurationError,
    compute_maniskill_pickcube_action_layout_digest,
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
    validate_maniskill_pickcube_action_layout_binding,
)
from latentguard.integrations.maniskill_pickcube.probe import (
    _state_round_trip_result,
    probe_maniskill_pickcube,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_CONTRACT = (
    _ROOT
    / "configs"
    / "integrations"
    / "maniskill_pickcube"
    / "expected-contract-v1.json"
)
_PROBE_REQUIREMENTS = _ROOT / "requirements" / "maniskill-pickcube-probe.txt"


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _report(
    *,
    mani_skill_version: str = "3.0.1",
    environment_id: str = "PickCube-v1",
    control_mode: str = "pd_joint_pos",
    action_dimension: int = 3,
    task_keys: tuple[str, ...] = (
        "is_grasped",
        "is_obj_placed",
        "is_robot_static",
        "success",
    ),
    runtime_apis: RuntimeApiContract | None = None,
    source_solver: ModuleSourceIdentity | None = None,
    task_implementation: ModuleSourceIdentity | None = None,
    gpu_simulation: bool = True,
    resolved_sim_backend: str = "physx_cuda",
    reset_seed: int = 0,
    operational: OperationalRuntimeMetadata | None = None,
) -> CompatibilityReport:
    arm_indices = tuple(range(action_dimension - 1))
    gripper_indices = (action_dimension - 1,)
    arm = ControllerComponentContract(
        name="arm",
        indices=arm_indices,
        controller_type="fake.PDJointPosController",
        configuration_identity=_sha("1"),
    )
    gripper = ControllerComponentContract(
        name="gripper",
        indices=gripper_indices,
        controller_type="fake.PassiveController",
        configuration_identity=_sha("2"),
    )
    return CompatibilityReport(
        mani_skill_version=mani_skill_version,
        sapien_version="3.fake",
        mplib_version="probe-observed.fake",
        environment_id=environment_id,
        robot_uid="panda",
        num_envs=1,
        reset_seed=reset_seed,
        observation_mode="none",
        control_mode=control_mode,
        sim_backend_request="gpu",
        resolved_sim_backend=resolved_sim_backend,
        gpu_simulation=gpu_simulation,
        action_space=ActionSpaceContract(
            batched_shape=(1, action_dimension),
            single_shape=(action_dimension,),
            dtype="float32",
            lower_bounds=(-1.0,) * action_dimension,
            upper_bounds=(1.0,) * action_dimension,
        ),
        control_frequency_hz=20.0,
        simulation_frequency_hz=100.0,
        runtime_apis=runtime_apis
        or RuntimeApiContract(
            get_state_dict_callable=True,
            set_state_dict_callable=True,
            evaluate_callable=True,
        ),
        task_evaluator_keys=task_keys,
        source_solver=source_solver
        or ModuleSourceIdentity(
            module_name=(
                "mani_skill.examples.motionplanning.panda.solutions.pick_cube"
            ),
            source_sha256=_sha("3"),
        ),
        task_implementation=task_implementation
        or ModuleSourceIdentity(
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
            complete_state_comparison=True,
            comparison_semantic=PICKCUBE_STATE_VERIFICATION_SEMANTIC,
            expected_state_digest=_sha("7"),
            observed_state_digest=_sha("7"),
            expected_structure_digest=_sha("6"),
            observed_structure_digest=_sha("6"),
            expected_leaf_count=5,
            observed_leaf_count=5,
            expected_numeric_component_count=11,
            observed_numeric_component_count=11,
            compared_numeric_component_count=11,
            maximum_absolute_error=0.0,
            tolerance=1e-6,
        ),
        bounded_action_step=ProbeCheckResult(
            passed=True,
            detail="bounded midpoint action executed once",
        ),
        environment_close=ProbeCheckResult(
            passed=True,
            detail="environment closed cleanly",
        ),
        operational=operational
        or OperationalRuntimeMetadata(
            python_version="3.11.9",
            torch_version="2.fake",
            cuda_runtime_version="12.fake",
            gpu_model="Fake GPU",
            gpu_capability="9.0",
        ),
    )


def _verified_contract(
    report: CompatibilityReport,
) -> ExpectedManiSkillPickCubeContract:
    return replace(
        load_expected_contract(_EXPECTED_CONTRACT),
        contract_status="verified",
        sapien_version=report.sapien_version,
        mplib_version=report.mplib_version,
        observation_mode=report.observation_mode,
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
    )


def _action_layout_document(
    report: CompatibilityReport | None = None,
) -> dict[str, object]:
    observed = report or _report()
    return {
        "schema_version": "1.0",
        "action_dim": 3,
        "fields": [
            {
                "name": "arm",
                "indices": [0, 1],
                "semantic": "unspecified",
                "units": "unspecified",
                "description": "Probe-verified arm controller component.",
                "metadata": {"controller_component": "arm"},
            },
            {
                "name": "gripper",
                "indices": [2],
                "semantic": "unspecified",
                "units": "unspecified",
                "description": None,
                "metadata": {"controller_component": "gripper"},
            },
        ],
        "description": "Locally reviewed fake probe action layout.",
        "metadata": {"review_status": "probe_verified"},
        "coordinate_frame": "unspecified",
        "control_mode": "pd_joint_pos",
        "control_period_s": observed.control_period_s,
        "action_contract_digest": observed.action_contract_digest,
    }


def _write_action_layout(
    path: Path,
    document: dict[str, object] | None = None,
    *,
    sort_keys: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document or _action_layout_document(), sort_keys=sort_keys),
        encoding="utf-8",
    )
    return path


def test_initial_contract_is_explicitly_probe_only_and_fail_closed() -> None:
    expected = load_expected_contract(_EXPECTED_CONTRACT)

    assert expected.mani_skill_version == "3.0.1"
    assert expected.contract_status == "probe_required"
    assert expected.action_dimension is None
    assert expected.mplib_version is None
    assert not expected.trusted_replay_ready
    with pytest.raises(ManiSkillConfigurationError, match="trusted replay"):
        expected.require_trusted_replay_ready()


def test_probe_requirements_pin_only_maniskill_before_discovery() -> None:
    package_lines = tuple(
        line.strip()
        for line in _PROBE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    assert package_lines == ("mani_skill==3.0.1",)


def test_exact_report_is_accepted_for_discovery_but_not_trusted() -> None:
    report = _report()
    binding = validate_compatibility_report(
        report,
        load_expected_contract(_EXPECTED_CONTRACT),
    )

    assert binding.report.compatibility_identity.startswith("sha256:")
    assert binding.report.action_space.action_dimension == 3
    assert binding.report.control_period_s == pytest.approx(0.05)
    assert not binding.trusted_replay_ready
    with pytest.raises(ManiSkillCompatibilityError, match="probe-only"):
        binding.require_trusted_replay_ready()


def test_fully_bound_report_can_authorize_trusted_replay() -> None:
    report = _report()
    expected = _verified_contract(report)

    binding = validate_compatibility_report(
        report,
        expected,
        require_trusted=True,
    )

    assert binding.trusted_replay_ready
    semantic = dict(binding.semantic_configuration())
    assert semantic["compatibility_identity"] == report.compatibility_identity
    assert semantic["action_contract_digest"] == report.action_contract_digest
    assert not any("path" in key for key in semantic)
    assert semantic["state_verification_semantic"] == (
        PICKCUBE_STATE_VERIFICATION_SEMANTIC
    )
    assert semantic["state_verification_tolerance"] == pytest.approx(1e-6)


def test_round_trip_accepts_non_equal_raw_digests_within_tolerance() -> None:
    expected_state = {"robot": np.array([1.0, -1.0], dtype=np.float32)}
    observed_state = {
        "robot": np.array(
            [np.nextafter(np.float32(1.0), np.float32(2.0)), -1.0],
            dtype=np.float32,
        )
    }

    round_trip = _state_round_trip_result(
        expected_state,
        observed_state,
        state_tolerance=1e-6,
    )
    report = replace(
        _report(),
        state_round_trip=round_trip,
        state_tree_structure_digest=round_trip.expected_structure_digest,
    )
    binding = validate_compatibility_report(
        report,
        load_expected_contract(_EXPECTED_CONTRACT),
    )

    assert round_trip.passed
    assert round_trip.complete_state_comparison
    assert round_trip.expected_structure_digest == round_trip.observed_structure_digest
    assert round_trip.compared_numeric_component_count == 2
    assert round_trip.expected_state_digest != round_trip.observed_state_digest
    assert round_trip.maximum_absolute_error == pytest.approx(1.1920929e-7)
    assert binding.report.state_round_trip == round_trip


def test_round_trip_rejects_over_tolerance_and_incomplete_structure() -> None:
    expected_state = {"robot": np.array([1.0, -1.0], dtype=np.float32)}
    over_tolerance = _state_round_trip_result(
        expected_state,
        {"robot": np.array([1.0 + 2e-6, -1.0], dtype=np.float32)},
        state_tolerance=1e-6,
    )
    wrong_structure = _state_round_trip_result(
        expected_state,
        {"robot": np.array([[1.0, -1.0]], dtype=np.float32)},
        state_tolerance=1e-6,
    )

    for round_trip in (over_tolerance, wrong_structure):
        assert not round_trip.passed
        with pytest.raises(ManiSkillCompatibilityError, match="round trip failed"):
            validate_compatibility_report(
                replace(
                    _report(),
                    state_round_trip=round_trip,
                    state_tree_structure_digest=(round_trip.expected_structure_digest),
                ),
                load_expected_contract(_EXPECTED_CONTRACT),
            )


def test_compatibility_rejects_claimed_pass_above_checked_in_tolerance() -> None:
    report = _report()

    with pytest.raises(ManiSkillCompatibilityError, match="must exactly reflect"):
        replace(
            report.state_round_trip,
            observed_state_digest=_sha("8"),
            maximum_absolute_error=2e-6,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"mani_skill_version": "3.0.0"}, "mani_skill_version"),
        ({"environment_id": "OtherTask-v1"}, "environment_id"),
        ({"control_mode": "pd_joint_delta_pos"}, "control_mode"),
        ({"gpu_simulation": False}, "GPU simulation"),
        ({"resolved_sim_backend": "physx_cpu"}, "GPU simulation"),
    ],
)
def test_fixed_scope_mismatches_are_rejected(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ManiSkillCompatibilityError, match=message):
        validate_compatibility_report(
            replace(_report(), **updates),
            load_expected_contract(_EXPECTED_CONTRACT),
        )


def test_python_311_is_required_for_the_real_integration_contract() -> None:
    report = _report()
    changed = replace(
        report,
        operational=replace(report.operational, python_version="3.12.4"),
    )

    with pytest.raises(ManiSkillCompatibilityError, match="Python 3.11"):
        validate_compatibility_report(
            changed,
            load_expected_contract(_EXPECTED_CONTRACT),
        )


def test_expected_contract_rejects_state_tolerance_drift() -> None:
    with pytest.raises(ManiSkillConfigurationError, match="authorized fixed"):
        replace(
            load_expected_contract(_EXPECTED_CONTRACT),
            state_round_trip_tolerance=5e-7,
        )


def test_checked_in_action_dimension_and_dtype_must_match_observation() -> None:
    report = _report()
    expected = replace(
        load_expected_contract(_EXPECTED_CONTRACT),
        action_dimension=report.action_space.action_dimension + 1,
    )
    with pytest.raises(ManiSkillCompatibilityError, match="action_dimension"):
        validate_compatibility_report(report, expected)

    expected = replace(
        load_expected_contract(_EXPECTED_CONTRACT),
        action_dtype="float64",
    )
    with pytest.raises(ManiSkillCompatibilityError, match="action_dtype"):
        validate_compatibility_report(report, expected)


def test_missing_state_api_and_task_keys_are_rejected() -> None:
    report = _report(
        runtime_apis=RuntimeApiContract(
            get_state_dict_callable=True,
            set_state_dict_callable=False,
            evaluate_callable=True,
        )
    )
    with pytest.raises(ManiSkillCompatibilityError, match="runtime_apis"):
        validate_compatibility_report(
            report, load_expected_contract(_EXPECTED_CONTRACT)
        )

    report = _report(task_keys=("is_obj_placed", "is_robot_static", "success"))
    with pytest.raises(ManiSkillCompatibilityError, match="is_grasped"):
        validate_compatibility_report(
            report, load_expected_contract(_EXPECTED_CONTRACT)
        )


def test_solver_and_task_module_identity_are_strict() -> None:
    with pytest.raises(ManiSkillCompatibilityError, match="solver module"):
        validate_compatibility_report(
            _report(
                source_solver=ModuleSourceIdentity(
                    module_name="fake.pick_cube",
                    source_sha256=_sha("3"),
                )
            ),
            load_expected_contract(_EXPECTED_CONTRACT),
        )

    with pytest.raises(ManiSkillCompatibilityError, match="task module"):
        validate_compatibility_report(
            _report(
                task_implementation=ModuleSourceIdentity(
                    module_name="fake.task",
                    source_sha256=_sha("4"),
                )
            ),
            load_expected_contract(_EXPECTED_CONTRACT),
        )


def test_solver_and_task_source_digests_are_content_bound() -> None:
    original = _report()
    changed_solver = _report(
        source_solver=replace(original.source_solver, source_sha256=_sha("a"))
    )
    changed_task = _report(
        task_implementation=replace(
            original.task_implementation,
            source_sha256=_sha("b"),
        )
    )

    assert changed_solver.compatibility_identity != original.compatibility_identity
    assert changed_task.compatibility_identity != original.compatibility_identity
    expected = _verified_contract(original)
    with pytest.raises(ManiSkillCompatibilityError, match="solver_sha256"):
        validate_compatibility_report(changed_solver, expected)
    with pytest.raises(ManiSkillCompatibilityError, match="task_sha256"):
        validate_compatibility_report(changed_task, expected)


def test_action_contract_binds_controller_components_and_bounds() -> None:
    original = _report()
    changed_component = replace(
        original.controller.components[0],
        configuration_identity=_sha("c"),
    )
    changed_controller = ControllerContract(
        components=(changed_component, original.controller.components[1]),
        configuration_identity=_sha("d"),
    )
    changed = replace(original, controller=changed_controller)

    assert changed.action_contract_digest != original.action_contract_digest
    assert changed.compatibility_identity != original.compatibility_identity
    with pytest.raises(ManiSkillCompatibilityError, match="finite interval"):
        ActionSpaceContract(
            batched_shape=(1, 2),
            single_shape=(2,),
            dtype="float32",
            lower_bounds=(-1.0, float("-inf")),
            upper_bounds=(1.0, 1.0),
        )


def test_operational_metadata_is_excluded_from_semantic_identity() -> None:
    first = _report()
    second = _report(
        operational=OperationalRuntimeMetadata(
            python_version="3.11.10",
            torch_version="2.other",
            cuda_runtime_version="13.fake",
            gpu_model="Different Fake GPU",
            gpu_capability="10.0",
        )
    )

    assert first.compatibility_identity == second.compatibility_identity
    assert first.to_dict()["operational"] != second.to_dict()["operational"]


def test_runtime_paths_cannot_enter_module_or_operational_identity() -> None:
    with pytest.raises(ValueError, match="runtime paths"):
        ModuleSourceIdentity(
            module_name=r"C:\private\solver.py",
            source_sha256=_sha("a"),
        )
    with pytest.raises(ManiSkillCompatibilityError, match="sanitized"):
        OperationalRuntimeMetadata(
            python_version="3.11",
            torch_version="2.fake",
            cuda_runtime_version="12.fake",
            gpu_model=r"C:\private\gpu.txt",
            gpu_capability="9.0",
        )


def test_report_round_trip_recomputes_identities_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "compatibility.json"
    report = _report()
    write_compatibility_report(report, output)

    reloaded = load_compatibility_report(output)
    assert reloaded == report
    assert reloaded.schema_version == "1.1"
    manifest = json.loads(output.read_text(encoding="utf-8"))
    manifest["source_solver"]["source_sha256"] = _sha("f")
    output.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManiSkillCompatibilityError, match="identity"):
        load_compatibility_report(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("comparison_semantic", "other_state_comparison"),
        ("observed_structure_digest", _sha("f")),
        ("compared_numeric_component_count", 10),
        ("observed_numeric_component_count", 12),
    ],
)
def test_report_loader_rejects_tampered_round_trip_coverage(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    output = tmp_path / f"tampered-{field}.json"
    write_compatibility_report(_report(), output)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    manifest["state_round_trip"][field] = value
    output.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ManiSkillCompatibilityError,
        match="must exactly reflect|unsupported PickCube runtime state-verification",
    ):
        load_compatibility_report(output)


def test_report_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "compatibility.json"
    write_compatibility_report(_report(), output)
    with pytest.raises(ManiSkillCompatibilityError, match="overwrite"):
        write_compatibility_report(_report(), output)


def test_report_writer_cannot_clobber_concurrent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "compatibility.json"
    concurrent_evidence = b"concurrent-writer-won\n"
    real_exists = Path.exists
    race_injected = False

    def inject_concurrent_writer(candidate: Path) -> bool:
        nonlocal race_injected
        observed = real_exists(candidate)
        if candidate == output and not race_injected:
            assert not observed
            output.write_bytes(concurrent_evidence)
            race_injected = True
        return observed

    monkeypatch.setattr(Path, "exists", inject_concurrent_writer)

    with pytest.raises(ManiSkillCompatibilityError, match="overwrite"):
        write_compatibility_report(_report(), output)

    assert race_injected
    assert output.read_bytes() == concurrent_evidence
    assert not tuple(tmp_path.glob(f".{output.name}.tmp-*"))


@dataclass
class _FakeProbeRuntime:
    report: CompatibilityReport
    calls: list[tuple[int, float]]

    def collect_report(
        self, *, reset_seed: int, state_tolerance: float
    ) -> CompatibilityReport:
        self.calls.append((reset_seed, state_tolerance))
        return self.report


def test_probe_uses_injected_fake_runtime_and_writes_sanitized_report(
    tmp_path: Path,
) -> None:
    fake = _FakeProbeRuntime(report=_report(reset_seed=17), calls=[])
    output = tmp_path / "compatibility.json"

    binding = probe_maniskill_pickcube(
        load_expected_contract(_EXPECTED_CONTRACT),
        runtime=fake,
        report_path=output,
        reset_seed=17,
    )

    assert fake.calls == [(17, 1e-6)]
    assert binding.report == load_compatibility_report(output)
    serialized = output.read_text(encoding="utf-8")
    assert "hostname" not in serialized
    assert "C:\\" not in serialized
    assert "/home/" not in serialized
    assert not binding.trusted_replay_ready


def test_probe_requires_fully_bound_expectation_when_trusted() -> None:
    fake = _FakeProbeRuntime(report=_report(), calls=[])
    with pytest.raises(ManiSkillCompatibilityError, match="probe-only"):
        probe_maniskill_pickcube(
            load_expected_contract(_EXPECTED_CONTRACT),
            runtime=fake,
            require_trusted=True,
        )

    report = _report()
    fake = _FakeProbeRuntime(report=report, calls=[])
    binding = probe_maniskill_pickcube(
        _verified_contract(report),
        runtime=fake,
        require_trusted=True,
    )
    assert binding.trusted_replay_ready


def test_expected_contract_loader_rejects_duplicate_and_wrong_fixed_fields(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(ManiSkillConfigurationError, match="duplicate"):
        load_expected_contract(duplicate)

    template = json.loads(_EXPECTED_CONTRACT.read_text(encoding="utf-8"))
    template["mani_skill_version"] = "3.0.0"
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ManiSkillConfigurationError, match="mani_skill_version"):
        load_expected_contract(wrong)


def test_verified_contract_cannot_retain_unresolved_fields() -> None:
    expected = load_expected_contract(_EXPECTED_CONTRACT)
    with pytest.raises(ManiSkillConfigurationError, match="unresolved"):
        replace(expected, contract_status="verified")


def test_distribution_discovery_checks_version_without_importing_modules() -> None:
    values = {
        "mani_skill": "3.0.1",
        "torch": "2.fake",
        "sapien": "3.fake",
        "mplib": "observed.fake",
        "gymnasium": "1.fake",
    }
    calls: list[str] = []

    def resolve(name: str) -> str:
        calls.append(name)
        return values[name]

    versions = discover_distribution_versions(version_resolver=resolve)

    assert versions.mani_skill == "3.0.1"
    assert versions.mplib == "observed.fake"
    assert calls == ["mani_skill", "torch", "sapien", "mplib", "gymnasium"]


def test_wrong_distribution_version_and_missing_module_are_descriptive() -> None:
    def wrong_version(name: str) -> str:
        return "3.0.0" if name == "mani_skill" else "fake"

    with pytest.raises(ManiSkillAvailabilityError, match="requires mani_skill==3.0.1"):
        discover_distribution_versions(version_resolver=wrong_version)

    def missing_module(name: str) -> ModuleType:
        if name == "mplib":
            raise ModuleNotFoundError(name)
        return ModuleType(name)

    with pytest.raises(ManiSkillAvailabilityError, match="mplib"):
        load_runtime_modules(module_importer=missing_module)


def test_runtime_module_loading_is_explicit_and_injected() -> None:
    calls: list[str] = []

    def load(name: str) -> ModuleType:
        calls.append(name)
        return ModuleType(name)

    modules = load_runtime_modules(module_importer=load)

    assert modules.mani_skill.__name__ == "mani_skill"
    assert calls == [
        "gymnasium",
        "mani_skill",
        "mani_skill.envs",
        "torch",
        "sapien",
        "mplib",
    ]


def test_core_import_does_not_import_optional_simulator_runtime() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_ROOT / "src")
    code = (
        "import sys; import latentguard; "
        "blocked=('mani_skill','sapien','mplib','torch'); "
        "assert not any(n == p or n.startswith(p + '.') "
        "for n in sys.modules for p in blocked)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_action_layout_loader_round_trip_is_path_independent_and_stable(
    tmp_path: Path,
) -> None:
    first = load_maniskill_pickcube_action_layout(
        _write_action_layout(tmp_path / "first" / "layout.json")
    )
    second = load_maniskill_pickcube_action_layout(
        _write_action_layout(
            tmp_path / "second" / "renamed.json",
            sort_keys=True,
        )
    )

    assert first.to_dict() == second.to_dict()
    assert first.action_layout_digest == second.action_layout_digest
    assert first.action_layout_digest.startswith("sha256:")
    assert first.action_layout_digest == (
        compute_maniskill_pickcube_action_layout_digest(first)
    )
    assert first.m1_action_layout.action_dim == 3
    assert all(
        field.semantic is ActionSemantic.UNSPECIFIED
        for field in first.m1_action_layout.fields
    )
    assert all(field.units == "unspecified" for field in first.m1_action_layout.fields)
    assert first.coordinate_frame == "unspecified"
    assert tuple(first.as_mapping()["fields"]) == tuple(second.as_mapping()["fields"])


def test_action_layout_loader_requires_exact_duplicate_free_fields(
    tmp_path: Path,
) -> None:
    missing = _action_layout_document()
    del missing["metadata"]
    with pytest.raises(ManiSkillConfigurationError, match="missing metadata"):
        load_maniskill_pickcube_action_layout(
            _write_action_layout(tmp_path / "missing.json", missing)
        )

    unexpected = _action_layout_document()
    unexpected["runtime_path"] = "/tmp/not-semantic"
    with pytest.raises(ManiSkillConfigurationError, match="unexpected runtime_path"):
        load_maniskill_pickcube_action_layout(
            _write_action_layout(tmp_path / "unexpected.json", unexpected)
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(ManiSkillConfigurationError, match="duplicate field"):
        load_maniskill_pickcube_action_layout(duplicate)


def test_action_layout_loader_rejects_nonfinite_and_runtime_path_data(
    tmp_path: Path,
) -> None:
    nonfinite = _action_layout_document()
    nonfinite["control_period_s"] = float("nan")
    with pytest.raises(ManiSkillConfigurationError, match="NaN|finite"):
        load_maniskill_pickcube_action_layout(
            _write_action_layout(tmp_path / "nonfinite.json", nonfinite)
        )

    runtime_path = _action_layout_document()
    runtime_path["metadata"] = {"review_source": r"C:\private\probe.json"}
    with pytest.raises(ManiSkillConfigurationError, match="path-independent"):
        load_maniskill_pickcube_action_layout(
            _write_action_layout(tmp_path / "runtime-path.json", runtime_path)
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document.update(action_dim=4),
            r"missing \[3\]",
        ),
        (
            lambda document: document["fields"][1].update(indices=[1]),
            "assigned to both",
        ),
        (
            lambda document: document["fields"][1].update(indices=[3]),
            "outside",
        ),
        (
            lambda document: document.update(action_dim=0),
            "positive integer",
        ),
        (
            lambda document: document["fields"][0].update(units=None),
            "units",
        ),
    ],
)
def test_action_layout_requires_complete_non_overlapping_index_coverage(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    document = _action_layout_document()
    assert callable(mutate)
    mutate(document)
    with pytest.raises(ManiSkillConfigurationError, match=message):
        load_maniskill_pickcube_action_layout(
            _write_action_layout(tmp_path / f"invalid-{message[:4]}.json", document)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("control_mode", "pd_joint_delta_pos", "control_mode"),
        ("control_period_s", 0.0, "control_period_s"),
        ("action_contract_digest", "sha256:ABC", "action_contract_digest"),
    ],
)
def test_action_layout_rejects_invalid_control_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    document = _action_layout_document()
    document[field] = value
    with pytest.raises(ManiSkillConfigurationError, match=message):
        load_maniskill_pickcube_action_layout(
            _write_action_layout(tmp_path / f"invalid-{field}.json", document)
        )


def test_action_layout_rejects_unknown_semantic(tmp_path: Path) -> None:
    document = _action_layout_document()
    document["fields"][0]["semantic"] = "joint_position"
    with pytest.raises(ManiSkillConfigurationError, match="unsupported semantic"):
        load_maniskill_pickcube_action_layout(
            _write_action_layout(tmp_path / "semantic.json", document)
        )


def test_action_layout_binding_accepts_only_exact_trusted_contract(
    tmp_path: Path,
) -> None:
    report = _report()
    trusted = validate_compatibility_report(
        report,
        _verified_contract(report),
        require_trusted=True,
    )
    layout = load_maniskill_pickcube_action_layout(
        _write_action_layout(
            tmp_path / "layout.json",
            _action_layout_document(report),
        )
    )

    digest = validate_maniskill_pickcube_action_layout_binding(
        layout,
        trusted,
        layout.m1_action_layout,
    )

    assert digest == layout.action_layout_digest


@pytest.mark.parametrize("mismatch", ("name", "indices", "order"))
def test_action_layout_binding_rejects_controller_component_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    report = _report()
    trusted = validate_compatibility_report(
        report,
        _verified_contract(report),
        require_trusted=True,
    )
    document = _action_layout_document(report)
    fields = document["fields"]
    assert isinstance(fields, list)
    assert all(isinstance(field, dict) for field in fields)
    if mismatch == "name":
        fields[0]["name"] = "renamed_arm"
    elif mismatch == "indices":
        fields[0]["indices"] = [0, 2]
        fields[1]["indices"] = [1]
    else:
        document["fields"] = list(reversed(fields))
    layout = load_maniskill_pickcube_action_layout(
        _write_action_layout(tmp_path / f"component-{mismatch}.json", document)
    )

    with pytest.raises(ManiSkillConfigurationError, match="controller.components"):
        validate_maniskill_pickcube_action_layout_binding(
            layout,
            trusted,
            layout.m1_action_layout,
        )


@pytest.mark.parametrize(
    "unattested_claim",
    ("semantic", "units", "coordinate_frame"),
)
def test_action_layout_binding_requires_unspecified_for_unattested_semantics(
    tmp_path: Path,
    unattested_claim: str,
) -> None:
    report = _report()
    trusted = validate_compatibility_report(
        report,
        _verified_contract(report),
        require_trusted=True,
    )
    document = _action_layout_document(report)
    fields = document["fields"]
    assert isinstance(fields, list)
    assert all(isinstance(field, dict) for field in fields)
    if unattested_claim == "semantic":
        fields[1]["semantic"] = "gripper"
    elif unattested_claim == "units":
        fields[1]["units"] = "normalized_command"
    else:
        document["coordinate_frame"] = "robot_base"
    layout = load_maniskill_pickcube_action_layout(
        _write_action_layout(tmp_path / f"claim-{unattested_claim}.json", document)
    )

    with pytest.raises(ManiSkillConfigurationError, match=unattested_claim):
        validate_maniskill_pickcube_action_layout_binding(
            layout,
            trusted,
            layout.m1_action_layout,
        )


def test_action_layout_binding_rejects_probe_only_binding(tmp_path: Path) -> None:
    report = _report()
    probe_only = validate_compatibility_report(
        report,
        load_expected_contract(_EXPECTED_CONTRACT),
    )
    layout = load_maniskill_pickcube_action_layout(
        _write_action_layout(tmp_path / "layout.json", _action_layout_document(report))
    )

    with pytest.raises(ManiSkillConfigurationError, match="not trusted"):
        validate_maniskill_pickcube_action_layout_binding(
            layout,
            probe_only,
            layout.m1_action_layout,
        )


def test_action_layout_binding_rejects_every_contract_mismatch(
    tmp_path: Path,
) -> None:
    report = _report()
    expected = _verified_contract(report)
    trusted = validate_compatibility_report(report, expected, require_trusted=True)
    exact = load_maniskill_pickcube_action_layout(
        _write_action_layout(
            tmp_path / "exact.json",
            _action_layout_document(report),
        )
    )

    dimension_document = _action_layout_document(report)
    dimension_document["action_dim"] = 2
    dimension_document["fields"][0]["indices"] = [0]
    dimension_document["fields"][1]["indices"] = [1]
    wrong_dimension = load_maniskill_pickcube_action_layout(
        _write_action_layout(tmp_path / "dimension.json", dimension_document)
    )
    with pytest.raises(ManiSkillConfigurationError, match="action_dimension"):
        validate_maniskill_pickcube_action_layout_binding(
            wrong_dimension,
            trusted,
            wrong_dimension.m1_action_layout,
        )

    digest_document = _action_layout_document(report)
    digest_document["action_contract_digest"] = _sha("f")
    wrong_digest = load_maniskill_pickcube_action_layout(
        _write_action_layout(tmp_path / "digest.json", digest_document)
    )
    with pytest.raises(ManiSkillConfigurationError, match="action_contract_digest"):
        validate_maniskill_pickcube_action_layout_binding(
            wrong_digest,
            trusted,
            wrong_digest.m1_action_layout,
        )

    period_document = _action_layout_document(report)
    period_document["control_period_s"] = 0.1
    wrong_period = load_maniskill_pickcube_action_layout(
        _write_action_layout(tmp_path / "period.json", period_document)
    )
    with pytest.raises(ManiSkillConfigurationError, match="control_period_s"):
        validate_maniskill_pickcube_action_layout_binding(
            wrong_period,
            trusted,
            wrong_period.m1_action_layout,
        )

    synthetic_control_mismatch = CompatibilityBinding(
        report=replace(report, control_mode="pd_joint_delta_pos"),
        expected_contract=expected,
        unresolved_fields=(),
    )
    with pytest.raises(ManiSkillConfigurationError, match="control_mode"):
        validate_maniskill_pickcube_action_layout_binding(
            exact,
            synthetic_control_mismatch,
            exact.m1_action_layout,
        )

    changed_m1_layout = replace(
        exact.m1_action_layout,
        description="Different local corruption layout.",
    )
    assert isinstance(changed_m1_layout, ActionLayout)
    with pytest.raises(ManiSkillConfigurationError, match="M1 ActionLayout"):
        validate_maniskill_pickcube_action_layout_binding(
            exact,
            trusted,
            changed_m1_layout,
        )
