"""CPU-only command-surface tests for M4A visual data workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import pytest

from latentguard import cli
from latentguard.m4a_cli import (
    ACCEPTED_M3C_EXECUTION_GIT_SHA,
    M4A_COMMANDS,
    M4ARenderCommandResult,
    M4AStaticScope,
    _load_static_scope,
    _run_render_m3a_visual_dataset,
    _run_render_m3c_external_visual_dataset,
    add_m4a_subparsers,
    run_m4a_command,
)
from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST
from latentguard.vision_data import (
    SourceCollection,
    VisualActionVerifierSampleV1,
    VisualCandidateBindingV1,
    VisualDatasetSplit,
    VisualObservationPacketV1,
    VisualTaskProjectionV1,
    VisualViewRecordV1,
    assigned_render_domain_ids,
    build_visual_development_dataset,
    build_visual_external_dataset,
    load_camera_rig_configuration,
    load_render_domain_configuration,
)
from latentguard.vision_data.serialization import (
    prepare_npy_image,
    save_visual_dataset,
)

_CONFIG_ROOT = Path("configs/vision/m4a")


def _digest(label: str) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_m4a_subparsers(subparsers)
    return parser


def _scope(
    *,
    compatibility: str = _digest("compatibility"),
    calibration_dtype: str = "float64",
) -> M4AStaticScope:
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        CALIBRATION_COMPARISON_SEMANTIC,
        VISUAL_RENDERER_SEMANTIC_VERSION,
    )

    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    report = SimpleNamespace(compatibility_identity=compatibility)
    binding = SimpleNamespace(report=report)
    pixel_views = tuple(
        SimpleNamespace(
            changed_pixel_count=0,
            maximum_per_channel_absolute_difference=0,
            mean_absolute_pixel_difference=0.0,
        )
        for _ in range(3)
    )
    visual = SimpleNamespace(
        visual_compatibility_identity=_digest("visual-compatibility"),
        renderer_semantic_version=VISUAL_RENDERER_SEMANTIC_VERSION,
        repeated_render=SimpleNamespace(
            views=pixel_views,
            spatially_stable=True,
        ),
        fresh_environment_render=SimpleNamespace(
            views=pixel_views,
            spatially_stable=True,
        ),
        renderer_api=SimpleNamespace(
            calibration_comparison_semantic=CALIBRATION_COMPARISON_SEMANTIC,
            runtime_intrinsics_dtype=calibration_dtype,
            runtime_extrinsics_dtype=calibration_dtype,
        ),
    )
    return M4AStaticScope(
        compatibility_binding=binding,  # type: ignore[arg-type]
        action_layout=SimpleNamespace(coordinate_frame="pd_joint_delta_pos"),  # type: ignore[arg-type]
        camera_rig=rig,  # type: ignore[arg-type]
        render_domains=domains,  # type: ignore[arg-type]
        visual_probe_report=visual,  # type: ignore[arg-type]
    )


class _Backend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def render_m3a(self, **kwargs: object) -> M4ARenderCommandResult:
        assert kwargs["source_binding"] is not None
        self.calls.append("m3a")
        return M4ARenderCommandResult(
            dataset_digest=_digest("m3a-visual"),
            packet_count=3,
            image_count=9,
            candidate_count=9,
            rendered_packet_count=3,
        )

    def render_m3c(self, **kwargs: object) -> M4ARenderCommandResult:
        assert kwargs["source_binding"] is not None
        self.calls.append("m3c")
        return M4ARenderCommandResult(
            dataset_digest=_digest("m3c-visual"),
            packet_count=3,
            image_count=9,
            candidate_count=8,
            rendered_packet_count=3,
        )


def _render_namespace(tmp_path: Path, *, command: str) -> argparse.Namespace:
    common = {
        "command": command,
        "compatibility_report": tmp_path / "compatibility.json",
        "expected_contract": tmp_path / "expected.json",
        "action_layout": tmp_path / "layout.json",
        "camera_rig": _CONFIG_ROOT / "camera-rig-v1.json",
        "render_domains": _CONFIG_ROOT / "render-domains-v1.json",
        "visual_probe_report": tmp_path / "visual-probe.json",
        "output_root": tmp_path / "output",
        "seed": 271828,
        "runtime_archive_dir": tmp_path / "archive",
        "anchor_manifest_dir": tmp_path / "anchors",
        "trajectory_limit": 1,
        "anchor_limit": 1,
        "domain_limit": 1,
        "resume": False,
        "retry_execution_errors": False,
        "dry_run": False,
    }
    if command == "render-m3a-visual-dataset":
        common.update(
            dataset_dir=tmp_path / "dataset",
            acceptance_report=tmp_path / "acceptance.json",
        )
    else:
        common.update(
            source_dir=tmp_path / "source",
            candidate_pool_dir=tmp_path / "pool",
            blind_manifest=tmp_path / "blind.json",
            bound_result=tmp_path / "bound.json",
            corruption_dir=tmp_path / "corruptions",
            selected_evaluation_dir=tmp_path / "selected",
            remainder_evaluation_dir=tmp_path / "remainder",
            expected_execution_git_sha=ACCEPTED_M3C_EXECUTION_GIT_SHA,
        )
    return argparse.Namespace(**common)


def _task_projection() -> VisualTaskProjectionV1:
    return VisualTaskProjectionV1(
        success=False,
        is_obj_placed=False,
        is_robot_static=True,
        is_grasped=False,
        cube_center_z=0.02,
        cube_to_goal_distance=0.1,
        tcp_to_cube_distance=0.2,
    )


def _visual_bundle(tmp_path: Path) -> tuple[Path, object, Path]:
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        VISUAL_RENDERER_SEMANTIC_VERSION,
        build_pickcube_visual_render_plan,
    )
    from latentguard.vision_data.pipeline import (
        VisualPacketJobV1,
        VisualRenderJobInventoryV1,
        derive_render_seed,
        save_render_job_inventory,
    )
    from latentguard.vision_data.rendering import initialize_render_ledger

    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    task = _task_projection()
    packets: list[VisualObservationPacketV1] = []
    jobs: list[VisualPacketJobV1] = []
    images: dict[str, np.ndarray] = {}
    base_render_seed = 271828
    for domain_index, domain_id in enumerate(
        assigned_render_domain_ids(
            SourceCollection.M3A_DEVELOPMENT, VisualDatasetSplit.TRAIN
        )
    ):
        render_seed = derive_render_seed("anchor-0", domain_id, base_render_seed)
        resolved_plan = build_pickcube_visual_render_plan(
            rig,
            domains.domain(domain_id),
            domains,
            render_seed,
        )
        views: list[VisualViewRecordV1] = []
        for camera_index, camera in enumerate(resolved_plan.cameras):
            image = np.full(
                (224, 224, 3),
                domain_index * 10 + camera_index,
                dtype=np.uint8,
            )
            prepared = prepare_npy_image(image)
            reference = f"images/{domain_id}/{camera.camera_id}.npy"
            images[reference] = image
            views.append(
                VisualViewRecordV1(
                    camera_id=camera.camera_id,
                    image_reference=reference,
                    pixel_sha256=prepared.pixel_sha256,
                    npy_sha256=prepared.npy_sha256,
                    dtype="uint8",
                    shape=(224, 224, 3),
                    intrinsics=tuple(
                        tuple(float(value) for value in row)
                        for row in camera.intrinsics
                    ),  # type: ignore[arg-type]
                    intrinsics_dtype=camera.intrinsics.dtype.name,
                    extrinsics=tuple(
                        tuple(float(value) for value in row)
                        for row in camera.extrinsics
                    ),  # type: ignore[arg-type]
                    extrinsics_dtype=camera.extrinsics.dtype.name,
                    camera_configuration_digest=(camera.camera_configuration_digest),
                    state_before_render_digest=_digest("state"),
                    state_after_render_digest=_digest("state"),
                    compared_state_component_count=70,
                    maximum_state_error=0.0,
                    task_projection_before=task,
                    task_projection_after=task,
                )
            )
        packets.append(
            VisualObservationPacketV1(
                source_collection=SourceCollection.M3A_DEVELOPMENT,
                source_trajectory_id="trajectory-0",
                anchor_id="anchor-0",
                split=VisualDatasetSplit.TRAIN,
                split_group_id="split-group-0",
                state_reference_id="state-reference-0",
                expected_state_digest=_digest("state"),
                verifier_state_semantic="PickCubeVerifierStateV1",
                verifier_state_digest=_digest("verifier-state"),
                verifier_state_component_count=38,
                verifier_state_maximum_absolute_error=0.0,
                elapsed_simulation_steps_before=0,
                elapsed_simulation_steps_after=0,
                environment_close_passed=True,
                camera_rig_id=rig.rig_id,
                camera_rig_digest=rig.content_digest,
                render_domain_id=domain_id,
                render_domain_digest=domains.domain(domain_id).content_digest,
                render_seed=render_seed,
                views=tuple(views),
                visual_compatibility_identity=_digest("visual-compatibility"),
                pickcube_compatibility_identity=_digest("compatibility"),
                renderer_semantic_version=VISUAL_RENDERER_SEMANTIC_VERSION,
            )
        )
        jobs.append(
            VisualPacketJobV1.create(
                job_ordinal=domain_index,
                source_collection=SourceCollection.M3A_DEVELOPMENT,
                source_trajectory_id="trajectory-0",
                source_reset_seed=17,
                anchor_id="anchor-0",
                split=VisualDatasetSplit.TRAIN,
                split_group_id="split-group-0",
                state_reference_id="state-reference-0",
                expected_state_digest=_digest("state"),
                verifier_state_semantic="PickCubeVerifierStateV1",
                verifier_state_digest=_digest("verifier-state"),
                camera_rig_id=rig.rig_id,
                camera_rig_digest=rig.rig_digest,
                render_domain_id=domain_id,
                render_domain_digest=domains.domain(domain_id).domain_digest,
                base_render_seed=base_render_seed,
                camera_configuration_digests=tuple(
                    camera.camera_configuration_digest
                    for camera in resolved_plan.cameras
                ),  # type: ignore[arg-type]
                visual_compatibility_identity=_digest("visual-compatibility"),
                pickcube_compatibility_identity=_digest("compatibility"),
                renderer_semantic_version=VISUAL_RENDERER_SEMANTIC_VERSION,
            )
        )
    packet_values = tuple(packets)
    binding = VisualCandidateBindingV1(
        candidate_sample_id="candidate-0",
        candidate_group_id="candidate-group-0",
        anchor_id="anchor-0",
        source_collection=SourceCollection.M3A_DEVELOPMENT,
        packet_ids=tuple(item.packet_id for item in packet_values),  # type: ignore[arg-type]
    )
    samples = tuple(
        VisualActionVerifierSampleV1(
            packet_id=packet.packet_id,
            candidate_sample_id="candidate-0",
            task_id="maniskill/PickCube-v1",
            canonical_task_text="Pick up the cube and place it at the goal.",
            candidate_action_chunk_reference="compact/candidate-0.npy",
            action_mask_reference="compact/candidate-0-mask.npy",
            final_success=True,
            final_unsafe=False,
            failure_events=(),
            evidence_id="evidence-0",
            source_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
            candidate_dataset_digest=_digest("candidate-dataset"),
            visual_dataset_digest=_digest("pending"),
            source_collection=SourceCollection.M3A_DEVELOPMENT,
        )
        for packet in packet_values
    )
    dataset = build_visual_development_dataset(
        source_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
        split_digest=_digest("split"),
        evidence_digest=_digest("evidence"),
        source_compatibility_identity=_digest("compatibility"),
        packets=packet_values,
        candidate_bindings=(binding,),
        samples=samples,
        camera_rig_digest=rig.content_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        full_target=False,
    )
    root = tmp_path / "visual-dataset"
    save_visual_dataset(dataset, images, root)
    render_root = tmp_path / "render-root"
    inventory = VisualRenderJobInventoryV1.create(
        source_identity_digest=_digest("source-identity"),
        camera_rig_digest=rig.rig_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        base_render_seed=base_render_seed,
        jobs=jobs,
    )
    initialize_render_ledger(
        render_root,
        run_id="m4a-test-render",
        source_collection=inventory.source_collection.value,
        source_identity_digest=inventory.source_identity_digest,
        camera_rig_digest=inventory.camera_rig_digest,
        render_domain_configuration_digest=(
            inventory.render_domain_configuration_digest
        ),
        visual_compatibility_identity=inventory.visual_compatibility_identity,
        packet_ids=inventory.packet_ids,
    )
    save_render_job_inventory(inventory, render_root)
    return root, dataset, render_root


def _external_dataset_from_development(development: object) -> object:
    """Build a disjoint partial external dataset for validate/report CLI tests."""

    from dataclasses import replace

    source_set = _digest("external-source-set")
    candidate_pool = _digest("external-candidate-pool")
    compatibility = _digest("external-compatibility")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    external_domain_ids = assigned_render_domain_ids(
        SourceCollection.M3C_EXTERNAL,
        VisualDatasetSplit.EXTERNAL,
    )
    packets: list[VisualObservationPacketV1] = []
    for packet_index, source_packet in enumerate(development.packets):  # type: ignore[attr-defined]
        views: list[VisualViewRecordV1] = []
        state_digest = _digest("external-state")
        for view_index, source_view in enumerate(source_packet.views):
            repeated = packet_index == 0 and view_index in {0, 1}
            image_label = (
                "external-repeated-image"
                if repeated
                else f"external-image-{packet_index}-{view_index}"
            )
            views.append(
                replace(
                    source_view,
                    image_reference=(
                        f"images/external-{packet_index}-{view_index}.npy"
                    ),
                    pixel_sha256=_digest(image_label),
                    npy_sha256=_digest(f"npy-{image_label}"),
                    state_before_render_digest=state_digest,
                    state_after_render_digest=state_digest,
                )
            )
        packets.append(
            replace(
                source_packet,
                source_collection=SourceCollection.M3C_EXTERNAL,
                source_trajectory_id="external-trajectory-0",
                anchor_id="external-anchor-0",
                split=VisualDatasetSplit.EXTERNAL,
                split_group_id="external-split-group-0",
                state_reference_id=_digest("external-state-reference"),
                expected_state_digest=state_digest,
                verifier_state_digest=_digest("external-verifier-state"),
                render_domain_id=external_domain_ids[packet_index],
                render_domain_digest=(
                    domains.domain(external_domain_ids[packet_index]).domain_digest
                ),
                views=tuple(views),
                pickcube_compatibility_identity=compatibility,
            )
        )
    packet_ids = tuple(packet.packet_id for packet in packets)
    source_binding = development.candidate_bindings[0]  # type: ignore[attr-defined]
    binding = replace(
        source_binding,
        candidate_sample_id="external-candidate-0",
        candidate_group_id="external-group-0",
        anchor_id="external-anchor-0",
        source_collection=SourceCollection.M3C_EXTERNAL,
        packet_ids=packet_ids,
    )
    samples = tuple(
        replace(
            development.samples[index],  # type: ignore[attr-defined]
            packet_id=packet.packet_id,
            candidate_sample_id="external-candidate-0",
            candidate_action_chunk_reference="compact/external-candidate.npy",
            action_mask_reference="compact/external-mask.npy",
            evidence_id="external-evidence-0",
            source_dataset_digest=source_set,
            candidate_dataset_digest=candidate_pool,
            visual_dataset_digest=_digest("pending-external-visual"),
            source_collection=SourceCollection.M3C_EXTERNAL,
        )
        for index, packet in enumerate(packets)
    )
    return build_visual_external_dataset(
        source_set_identity=source_set,
        candidate_pool_identity=candidate_pool,
        blind_manifest_digest=_digest("external-blind-manifest"),
        full_outcome_digest=_digest("external-full-outcome"),
        source_compatibility_identity=compatibility,
        packets=tuple(packets),
        candidate_bindings=(binding,),
        samples=samples,
        camera_rig_digest=development.camera_rig_digest,  # type: ignore[attr-defined]
        render_domain_configuration_digest=(
            development.render_domain_configuration_digest  # type: ignore[attr-defined]
        ),
        visual_compatibility_identity=(
            development.visual_compatibility_identity  # type: ignore[attr-defined]
        ),
        full_target=False,
    )


def test_registers_exact_commands_and_strict_required_artifacts() -> None:
    parser = _parser()
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    assert set(action.choices) == set(M4A_COMMANDS)
    probe = action.choices["probe-maniskill-pickcube-visual"]
    probe_required = {
        option
        for item in probe._actions
        if item.required
        for option in item.option_strings
    }
    assert "--output-root" in probe_required
    assert "--output" not in {
        option for item in probe._actions for option in item.option_strings
    }
    m3c = action.choices["render-m3c-external-visual-dataset"]
    required = {
        option
        for item in m3c._actions
        if item.required
        for option in item.option_strings
    }
    assert {
        "--candidate-pool-dir",
        "--blind-manifest",
        "--bound-result",
        "--selected-evaluation-dir",
        "--remainder-evaluation-dir",
        "--visual-probe-report",
    }.issubset(required)
    assert "--expected-execution-git-sha" not in {
        option for item in m3c._actions for option in item.option_strings
    }
    validate = action.choices["validate-visual-verifier-dataset"]
    validate_required = {
        option
        for item in validate._actions
        if item.required
        for option in item.option_strings
    }
    assert {
        "--dataset-dir",
        "--compatibility-report",
        "--visual-probe-report",
    }.issubset(validate_required)
    assert "--m3c-expected-execution-git-sha" not in {
        option for item in validate._actions for option in item.option_strings
    }
    report_action = next(
        item for item in validate._actions if "--report-dir" in item.option_strings
    )
    assert report_action.required is False
    with pytest.raises(SystemExit):
        parser.parse_args(["render-m3a-visual-dataset", "--output-root", "out"])


@pytest.mark.parametrize("command", tuple(sorted(M4A_COMMANDS)))
def test_every_m4a_help_path_is_cpu_only(command: str) -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main([command, "--help"])
    assert captured.value.code == 0


@pytest.mark.parametrize("trusted", (True, False))
def test_probe_routes_an_exact_archived_state_and_publishes_atomic_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    trusted: bool,
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.integrations.maniskill_pickcube import (
        state_indexed_archive,
        visual_probe,
        visual_rendering,
    )

    state = object()
    episode = SimpleNamespace(states=(state,))
    runtime = SimpleNamespace(collect_sessions=lambda **_: ())

    def require_trusted() -> None:
        if not trusted:
            raise ValueError("exact pixel gate failed")

    report = SimpleNamespace(
        visual_compatibility_identity=_digest("visual"),
        probe_source=object(),
        require_trusted_visual_generation_ready=require_trusted,
    )
    output = tmp_path / "probe-output"
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_args, **_kwargs: _scope())
    monkeypatch.setattr(
        state_indexed_archive,
        "load_state_indexed_archive",
        lambda _: SimpleNamespace(episodes=(episode,)),
    )
    monkeypatch.setattr(
        state_indexed_archive, "find_state_indexed_episode", lambda *_: episode
    )
    monkeypatch.setattr(
        visual_rendering, "build_pickcube_visual_render_plan", lambda *_: object()
    )

    def fake_probe(**kwargs: object) -> object:
        assert kwargs["runtime"] is runtime
        assert kwargs["source_state"] is state
        assert kwargs["require_trusted_generation"] is False
        assert kwargs["report_path"] is None
        return report

    monkeypatch.setattr(visual_probe, "probe_maniskill_pickcube_visual", fake_probe)
    monkeypatch.setattr(
        visual_probe,
        "write_visual_compatibility_report",
        lambda _report, path: Path(path).write_text("{}\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        visual_probe,
        "load_visual_compatibility_report",
        lambda *_args, **_kwargs: report,
    )
    fake_manifest = object()
    monkeypatch.setattr(
        m4a,
        "_build_operational_run_manifest",
        lambda **_kwargs: fake_manifest,
    )
    import latentguard.vision_data.run_manifest as run_manifest_module

    monkeypatch.setattr(
        run_manifest_module,
        "save_m4a_run_manifest",
        lambda _manifest, path: Path(path).write_text("{}\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        run_manifest_module,
        "load_m4a_run_manifest",
        lambda _path: fake_manifest,
    )
    args = argparse.Namespace(
        command="probe-maniskill-pickcube-visual",
        runtime_archive_dir=tmp_path / "archive",
        episode_id="episode-0",
        state_index=0,
        render_seed=7,
        output_root=output,
    )
    result = run_m4a_command(args, probe_runtime=runtime)  # type: ignore[arg-type]
    assert result == (0 if trusted else 1)
    assert (output / "visual-compatibility-report.json").is_file()
    assert (output / "run-manifest.json").is_file()
    captured = capsys.readouterr()
    if trusted:
        assert "state_integrity_verified=true" in captured.out
    else:
        assert "exact pixel gate failed" in captured.err


def test_bounded_m3a_render_routes_fake_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.vision_data import source_binding

    args = _render_namespace(tmp_path, command="render-m3a-visual-dataset")
    source = SimpleNamespace(
        compatibility_identity=_digest("compatibility"),
        dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
    )
    observed: dict[str, object] = {}

    def load_source(
        dataset_dir: Path,
        runtime_archive_dir: Path,
        anchor_manifest_dir: Path,
        *,
        acceptance_report: Path,
    ) -> object:
        observed.update(
            dataset_dir=dataset_dir,
            runtime_archive_dir=runtime_archive_dir,
            anchor_manifest_dir=anchor_manifest_dir,
            acceptance_report=acceptance_report,
        )
        return source

    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_args, **_kwargs: _scope())
    monkeypatch.setattr(
        source_binding, "load_m3a_development_source_binding", load_source
    )
    backend = _Backend()
    assert _run_render_m3a_visual_dataset(args, backend=backend) == 0  # type: ignore[arg-type]
    assert backend.calls == ["m3a"]
    assert observed == {
        "dataset_dir": args.dataset_dir,
        "runtime_archive_dir": args.runtime_archive_dir,
        "anchor_manifest_dir": args.anchor_manifest_dir,
        "acceptance_report": args.acceptance_report,
    }
    assert "packets=3 images=9 candidates=9" in capsys.readouterr().out


def test_m3a_validation_routes_runtime_archive_through_accepted_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.vision_data import source_binding

    args = argparse.Namespace(
        development_render_root=tmp_path / "render",
        m3a_runtime_archive_dir=tmp_path / "archive",
        m3a_source_dataset_dir=tmp_path / "dataset",
        m3a_anchor_manifest_dir=tmp_path / "anchors",
        m3a_acceptance_report=tmp_path / "acceptance.json",
        allow_partial=False,
    )
    observed: dict[str, object] = {}

    def capture_binding(
        dataset_dir: Path,
        runtime_archive_dir: Path,
        anchor_manifest_dir: Path,
        *,
        acceptance_report: Path,
    ) -> NoReturn:
        observed.update(
            dataset_dir=dataset_dir,
            runtime_archive_dir=runtime_archive_dir,
            anchor_manifest_dir=anchor_manifest_dir,
            acceptance_report=acceptance_report,
        )
        raise RuntimeError("accepted binding preflight reached")

    monkeypatch.setattr(
        source_binding,
        "load_m3a_development_source_binding",
        capture_binding,
    )
    with pytest.raises(RuntimeError, match="accepted binding preflight reached"):
        m4a._validate_m3a_visual_source(object(), args, _scope())  # type: ignore[arg-type]
    assert observed == {
        "dataset_dir": args.m3a_source_dataset_dir,
        "runtime_archive_dir": args.m3a_runtime_archive_dir,
        "anchor_manifest_dir": args.m3a_anchor_manifest_dir,
        "acceptance_report": args.m3a_acceptance_report,
    }


def test_bounded_m3c_render_routes_fake_backend_and_archive_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.vision_data import source_binding

    args = _render_namespace(tmp_path, command="render-m3c-external-visual-dataset")
    source = SimpleNamespace(
        simulator_compatibility_identity=_digest("compatibility"),
        source_set_digest=_digest("source-set"),
    )
    observed: list[object] = []
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_args, **_kwargs: _scope())
    monkeypatch.setattr(
        source_binding, "load_m3c_external_source_binding", lambda *_a, **_k: source
    )
    monkeypatch.setattr(
        source_binding,
        "require_m3c_anchor_archive_binding",
        lambda *values: observed.extend(values),
    )
    backend = _Backend()
    assert _run_render_m3c_external_visual_dataset(args, backend=backend) == 0  # type: ignore[arg-type]
    assert backend.calls == ["m3c"]
    assert observed[0] is source
    assert "candidates=8" in capsys.readouterr().out


def test_dry_run_preflights_sources_but_does_not_call_backend_or_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.vision_data import source_binding

    args = _render_namespace(tmp_path, command="render-m3a-visual-dataset")
    args.dry_run = True
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_args, **_kwargs: _scope())
    monkeypatch.setattr(
        source_binding,
        "load_m3a_development_source_binding",
        lambda *_a, **_k: SimpleNamespace(
            compatibility_identity=_digest("compatibility"),
            dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
        ),
    )
    backend = _Backend()
    assert _run_render_m3a_visual_dataset(args, backend=backend) == 0  # type: ignore[arg-type]
    assert backend.calls == ["m3a"]
    assert not args.output_root.exists()


def test_resume_retry_contract_and_compatibility_mismatch_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.vision_data import source_binding

    args = _render_namespace(tmp_path, command="render-m3a-visual-dataset")
    args.retry_execution_errors = True
    assert run_m4a_command(args, render_backend=_Backend()) == 1  # type: ignore[arg-type]
    assert "requires --resume" in capsys.readouterr().err
    args.retry_execution_errors = False
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_args, **_kwargs: _scope())
    monkeypatch.setattr(
        source_binding,
        "load_m3a_development_source_binding",
        lambda *_a, **_k: SimpleNamespace(
            compatibility_identity=_digest("other-runtime"),
            dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
        ),
    )
    assert run_m4a_command(args, render_backend=_Backend()) == 1  # type: ignore[arg-type]
    assert "compatibility identity differs" in capsys.readouterr().err


def test_raw_probe_and_render_roots_are_rejected_inside_git_checkout() -> None:
    import latentguard.m4a_cli as m4a

    raw_root = Path.cwd() / ".tmp" / "m4a-raw-must-stay-external"
    with pytest.raises(ValueError, match="outside Git"):
        m4a._validate_probe_output_root(
            argparse.Namespace(
                command="probe-maniskill-pickcube-visual",
                output_root=raw_root,
            )
        )
    with pytest.raises(ValueError, match="outside Git"):
        m4a._validate_render_controls(
            argparse.Namespace(
                command="render-m3a-visual-dataset",
                output_root=raw_root,
                resume=False,
                retry_execution_errors=False,
            )
        )


def test_static_scope_rejects_post_probe_camera_configuration_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.vision_data.configuration as vision_configuration
    from latentguard.integrations.maniskill_pickcube import (
        compatibility,
        configuration,
        visual_probe,
    )

    binding = SimpleNamespace(
        report=SimpleNamespace(compatibility_identity=_digest("c"))
    )
    layout = SimpleNamespace(m1_action_layout=object())
    rig = SimpleNamespace(rig_digest=_digest("rig"))
    domains = SimpleNamespace(content_digest=_digest("domains"))
    probe = SimpleNamespace(
        pickcube_compatibility_identity=_digest("c"),
        camera_rig_digest=_digest("drifted-rig"),
        render_domain_configuration_digest=_digest("domains"),
    )
    monkeypatch.setattr(configuration, "load_expected_contract", lambda _: object())
    monkeypatch.setattr(compatibility, "load_compatibility_report", lambda _: object())
    monkeypatch.setattr(
        compatibility, "validate_compatibility_report", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        configuration, "load_maniskill_pickcube_action_layout", lambda _: layout
    )
    monkeypatch.setattr(
        configuration,
        "validate_maniskill_pickcube_action_layout_binding",
        lambda *_: None,
    )
    monkeypatch.setattr(
        vision_configuration, "load_camera_rig_configuration", lambda _: rig
    )
    monkeypatch.setattr(
        vision_configuration, "load_render_domain_configuration", lambda _: domains
    )
    monkeypatch.setattr(
        visual_probe, "load_visual_compatibility_report", lambda *_a, **_k: probe
    )
    args = argparse.Namespace(
        expected_contract=tmp_path / "expected.json",
        compatibility_report=tmp_path / "compatibility.json",
        action_layout=tmp_path / "layout.json",
        camera_rig=tmp_path / "rig.json",
        render_domains=tmp_path / "domains.json",
        visual_probe_report=tmp_path / "probe.json",
    )
    with pytest.raises(ValueError, match="camera rig digest drifted"):
        _load_static_scope(args, require_visual_probe=True)


def test_render_plan_binds_seed_resolved_camera_shift_and_full_resume_inventory() -> (
    None
):
    import latentguard.m4a_cli as m4a
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        build_pickcube_visual_render_plan,
    )

    scope = _scope()
    state = SimpleNamespace(
        content_digest=_digest("state-content"),
        state_digest=_digest("state"),
        verifier_state=SimpleNamespace(
            semantic="PickCubeVerifierStateV1",
            content_digest=_digest("verifier"),
        ),
    )
    context = m4a._RenderAnchorContext(
        source_collection=SourceCollection.M3A_DEVELOPMENT,
        split=VisualDatasetSplit.TRAIN,
        source_trajectory_id="trajectory-0",
        source_reset_seed=19,
        split_group_id="split-group-0",
        anchor_id="anchor-0",
        episode=object(),
        state=state,
        candidate_group=object(),
        candidates=(object(),),
    )
    args = argparse.Namespace(seed=271828, domain_limit=1, dry_run=True)
    source_binding = SimpleNamespace(
        dataset_digest=_digest("dataset"),
        source_archive_digest=_digest("archive"),
        anchor_manifest_digest=_digest("manifest"),
        split_digest=_digest("split"),
        evidence_digest=_digest("evidence"),
        acceptance_report_digest=_digest("acceptance"),
        compatibility_identity=_digest("compatibility"),
    )
    plan = m4a._build_render_plan(
        args,
        scope,
        source_binding,
        (context,),
        SimpleNamespace(content_digest=_digest("source-model")),
        _digest("archive"),
        _digest("manifest"),
    )
    assert len(plan.inventory.jobs) == 3
    assert len(plan.selected_packet_ids) == 1
    shifted_job = next(
        job
        for job in plan.inventory.jobs
        if job.render_domain_id == "mild_camera_shift"
    )
    shifted_plan = build_pickcube_visual_render_plan(
        scope.camera_rig,
        scope.render_domains.domain("mild_camera_shift"),
        scope.render_domains,
        shifted_job.render_seed,
    )
    expected_digests = tuple(
        camera.camera_configuration_digest for camera in shifted_plan.cameras
    )
    base_digests = tuple(
        camera.camera_configuration_digest for camera in scope.camera_rig.cameras
    )
    assert shifted_job.camera_configuration_digests == expected_digests
    assert shifted_job.camera_configuration_digests != base_digests
    one_domain = m4a._PickCubeVisualDatasetRenderer._execute(
        args,
        scope,
        source_binding,
        plan,
        external=False,
    )
    assert (one_domain.packet_count, one_domain.image_count) == (1, 3)

    args.domain_limit = 2
    two_domain_plan = m4a._build_render_plan(
        args,
        scope,
        source_binding,
        (context,),
        SimpleNamespace(content_digest=_digest("source-model")),
        _digest("archive"),
        _digest("manifest"),
    )
    two_domains = m4a._PickCubeVisualDatasetRenderer._execute(
        args,
        scope,
        source_binding,
        two_domain_plan,
        external=False,
    )
    assert (two_domains.packet_count, two_domains.image_count) == (2, 6)


def test_operational_manifest_binds_clean_single_gpu_runtime_and_exact_resume() -> None:
    import latentguard.m4a_cli as m4a

    scope = _scope()
    compatibility_report = scope.compatibility_binding.report
    compatibility_report.mani_skill_version = "3.0.1"
    compatibility_report.sapien_version = "3.0.0.b1"
    compatibility_report.operational = SimpleNamespace(
        python_version="3.11.13",
        torch_version="2.7.1+cu128",
        cuda_runtime_version="12.8",
        gpu_model="NVIDIA GeForce RTX 5090",
    )
    visual = scope.visual_probe_report
    visual.mani_skill_version = "3.0.1"  # type: ignore[union-attr]
    visual.sapien_version = "3.0.0.b1"  # type: ignore[union-attr]
    visual.torch_version = "2.7.1+cu128"  # type: ignore[union-attr]
    visual.cuda_runtime_version = "12.8"  # type: ignore[union-attr]
    visual.gpu_model = "NVIDIA GeForce RTX 5090"  # type: ignore[union-attr]
    visual.renderer_api.renderer_backend = "sapien-vulkan"  # type: ignore[union-attr]

    def metadata_runner(arguments: tuple[str, ...], _cwd: Path | None) -> str:
        if arguments[:2] == ("git", "status"):
            return ""
        if arguments[:2] == ("git", "rev-parse"):
            return "a" * 40
        if arguments[:2] == ("git", "symbolic-ref"):
            return "codex/m4a-multiview-visual-dataset"
        if arguments[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 5090, 575.64.03"
        raise AssertionError(f"unexpected metadata command: {arguments!r}")

    args = argparse.Namespace(command="render-m3a-visual-dataset")
    manifest = m4a._build_operational_run_manifest(
        args=args,
        scope=scope,
        source_binding=SimpleNamespace(dataset_digest=_digest("accepted-m3a")),
        run_id="m4a-m3a-test",
        external=False,
        render_seed=271828,
        canonical_source_identity_digest=_digest("canonical-source-identity"),
        render_job_inventory_digest=_digest("render-jobs"),
        selected_packet_inventory_digest=_digest("selected-packets"),
        runner=metadata_runner,
        started_at="2026-07-18T00:00:00Z",
        launch_argv=(
            "latentguard",
            "render-m3a-visual-dataset",
            "--output-root",
            r"D:\private\raw-images",
        ),
        hostname="m4a-worker-01",
        python_version="3.11.13",
        installed_versions={
            "cuda": "12.8",
            "maniskill": "3.0.1",
            "pytorch": "2.7.1+cu128",
            "sapien": "3.0.0.b1",
        },
    )
    assert manifest.git_sha == "a" * 40
    assert manifest.environment.gpu_driver_version == "575.64.03"
    assert manifest.source_identity.m3a_dataset_digest == _digest("accepted-m3a")
    assert manifest.source_identity.canonical_source_identity_digest == _digest(
        "canonical-source-identity"
    )
    assert all("private" not in argument for argument in manifest.launch_argv)

    resumed = m4a._build_operational_run_manifest(
        args=args,
        scope=scope,
        source_binding=SimpleNamespace(dataset_digest=_digest("accepted-m3a")),
        run_id="m4a-m3a-test",
        external=False,
        render_seed=271828,
        canonical_source_identity_digest=_digest("canonical-source-identity"),
        render_job_inventory_digest=_digest("render-jobs"),
        selected_packet_inventory_digest=_digest("selected-packets"),
        prior_manifest=manifest,
        runner=metadata_runner,
        launch_argv=("different", "resume", "command"),
        hostname="m4a-worker-01",
        python_version="3.11.13",
        installed_versions={
            "cuda": "12.8",
            "maniskill": "3.0.1",
            "pytorch": "2.7.1+cu128",
            "sapien": "3.0.0.b1",
        },
    )
    assert resumed == manifest


def test_operational_manifest_rejects_multiple_or_wrong_gpu_models() -> None:
    import latentguard.m4a_cli as m4a

    with pytest.raises(ValueError, match="exactly one visible GPU"):
        m4a._single_gpu_operational_identity(
            lambda _arguments, _cwd: (
                "NVIDIA GeForce RTX 5090, 575.64.03\nNVIDIA GeForce RTX 5090, 575.64.03"
            )
        )
    with pytest.raises(ValueError, match="requires one RTX 5090"):
        m4a._single_gpu_operational_identity(
            lambda _arguments, _cwd: "NVIDIA A100, 575.64.03"
        )


def test_operational_git_identity_rejects_untracked_checkout_content() -> None:
    import latentguard.m4a_cli as m4a

    observed: list[tuple[str, ...]] = []

    def metadata_runner(arguments: tuple[str, ...], _cwd: Path | None) -> str:
        observed.append(arguments)
        if arguments == ("git", "status", "--porcelain"):
            return "?? src/latentguard/untracked_runtime.py"
        raise AssertionError("Git identity continued after a dirty status")

    with pytest.raises(ValueError, match="Git checkout is modified"):
        m4a._git_operational_identity(metadata_runner)
    assert observed == [("git", "status", "--porcelain")]


def _callback_pipeline_fixture() -> tuple[object, object, M4AStaticScope]:
    import latentguard.m4a_cli as m4a
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        VISUAL_RENDERER_SEMANTIC_VERSION,
        build_pickcube_visual_render_plan,
    )
    from latentguard.vision_data.pipeline import (
        VisualPacketJobV1,
        VisualRenderJobInventoryV1,
    )

    scope = _scope()
    domain = scope.render_domains.domain("canonical")
    render_plan = build_pickcube_visual_render_plan(
        scope.camera_rig, domain, scope.render_domains, 271828
    )
    job = VisualPacketJobV1.create(
        job_ordinal=0,
        source_collection=SourceCollection.M3A_DEVELOPMENT,
        source_trajectory_id="trajectory-0",
        source_reset_seed=19,
        anchor_id="anchor-0",
        split=VisualDatasetSplit.TRAIN,
        split_group_id="split-group-0",
        state_reference_id=_digest("state-content"),
        expected_state_digest=_digest("state"),
        verifier_state_semantic="PickCubeVerifierStateV1",
        verifier_state_digest=_digest("verifier"),
        camera_rig_id=scope.camera_rig.rig_id,
        camera_rig_digest=scope.camera_rig.rig_digest,
        render_domain_id=domain.domain_id,
        render_domain_digest=domain.domain_digest,
        base_render_seed=271828,
        camera_configuration_digests=tuple(
            camera.camera_configuration_digest for camera in render_plan.cameras
        ),  # type: ignore[arg-type]
        visual_compatibility_identity=(
            scope.visual_probe_report.visual_compatibility_identity  # type: ignore[union-attr]
        ),
        pickcube_compatibility_identity=(
            scope.compatibility_binding.report.compatibility_identity
        ),
        renderer_semantic_version=VISUAL_RENDERER_SEMANTIC_VERSION,
    )
    inventory = VisualRenderJobInventoryV1.create(
        source_identity_digest=_digest("source"),
        camera_rig_digest=scope.camera_rig.rig_digest,
        render_domain_configuration_digest=scope.render_domains.content_digest,
        visual_compatibility_identity=job.visual_compatibility_identity,
        base_render_seed=271828,
        jobs=(job,),
    )
    context = m4a._RenderAnchorContext(
        source_collection=SourceCollection.M3A_DEVELOPMENT,
        split=VisualDatasetSplit.TRAIN,
        source_trajectory_id="trajectory-0",
        source_reset_seed=19,
        split_group_id="split-group-0",
        anchor_id="anchor-0",
        episode=object(),
        state=SimpleNamespace(content_digest=_digest("state-content")),
        candidate_group=object(),
        candidates=(object(),),
    )
    callback_plan = SimpleNamespace(context_by_packet_id={job.packet_id: context})
    return inventory, callback_plan, scope


def test_production_execute_persists_zero_work_resume_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.m4a_cli as m4a
    import latentguard.vision_data.pipeline as pipeline_module
    import latentguard.vision_data.run_manifest as run_manifest_module
    import latentguard.vision_data.serialization as serialization_module
    from latentguard.training.reporting import load_strict_report
    from latentguard.vision_data.rendering import ZeroWorkRenderResumeV1
    from latentguard.vision_data.reporting import M4A_RENDER_RESUME_REPORT_FILENAME

    inventory, callback_plan, scope = _callback_pipeline_fixture()
    job = inventory.jobs[0]  # type: ignore[attr-defined]
    context = callback_plan.context_by_packet_id[job.packet_id]  # type: ignore[attr-defined]
    plan = m4a._PreparedRenderPlan(
        inventory=inventory,
        contexts=(context,),
        context_by_packet_id={job.packet_id: context},
        selected_packet_ids=(job.packet_id,),
        source_model=object(),
        archive_digest=_digest("archive"),
        manifest_digest=_digest("manifest"),
        all_required_domains=True,
    )
    output = tmp_path / "complete-render"
    output.mkdir()
    proof = ZeroWorkRenderResumeV1(
        run_id="m4a-m3a-test",
        packet_count=1,
        ledger_entry_count=1,
        ledger_content_digest=_digest("ledger"),
    )
    pipeline_result = SimpleNamespace(
        packets=(object(),),
        packet_count=1,
        image_count=3,
        rendered_packet_ids=(),
        zero_work_proof=proof,
    )
    monkeypatch.setattr(
        run_manifest_module, "load_m4a_run_manifest", lambda _path: object()
    )
    monkeypatch.setattr(
        m4a, "_build_operational_run_manifest", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_visual_render_pipeline",
        lambda *_args, **_kwargs: pipeline_result,
    )
    monkeypatch.setattr(m4a, "_pipeline_callback", lambda *_args: object())
    dataset = SimpleNamespace(content_digest=_digest("visual-dataset"))
    monkeypatch.setattr(m4a, "_build_m3a_visual_dataset", lambda *_a, **_k: dataset)
    monkeypatch.setattr(m4a, "_packet_images", lambda _result: {})
    monkeypatch.setattr(
        serialization_module, "save_visual_dataset", lambda *_args: output / "dataset"
    )

    result = m4a._PickCubeVisualDatasetRenderer._execute(
        argparse.Namespace(
            dry_run=False,
            resume=True,
            output_root=output,
            seed=271828,
            retry_execution_errors=False,
            trajectory_limit=None,
            anchor_limit=None,
            domain_limit=None,
        ),
        scope,
        SimpleNamespace(dataset_digest=_digest("accepted-m3a")),
        plan,
        external=False,
    )

    assert result.zero_work_resume_verified is True
    report = load_strict_report(
        output / M4A_RENDER_RESUME_REPORT_FILENAME,
        expected_report_type="m4a_render_resume_v1",
    )
    assert report.payload["zero_duplicate_work"] is True
    assert report.payload["source_identity_digest"] == inventory.source_identity_digest  # type: ignore[attr-defined]


def test_callback_invalid_context_is_persisted_as_nonretryable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.integrations.maniskill_pickcube.visual_session import (
        ManiSkillVisualInvalidContextError,
    )
    from latentguard.vision_data.pipeline import (
        InvalidVisualRenderPacketError,
        run_visual_render_pipeline,
    )
    from latentguard.vision_data.rendering import (
        RenderLedgerState,
        load_render_ledger,
    )

    inventory, plan, scope = _callback_pipeline_fixture()

    def invalid_context(**_kwargs: object) -> NoReturn:
        raise ManiSkillVisualInvalidContextError("restored state digest mismatch")

    monkeypatch.setattr(
        m4a,
        "_create_visual_session",
        lambda _scope: SimpleNamespace(render_state=invalid_context),
    )
    output = tmp_path / "invalid-context"
    with pytest.raises(InvalidVisualRenderPacketError):
        run_visual_render_pipeline(
            inventory,  # type: ignore[arg-type]
            output,
            run_id="m4a-invalid-context-test",
            render_callback=m4a._pipeline_callback(plan, scope),  # type: ignore[arg-type]
        )
    ledger = load_render_ledger(output)
    entry = ledger.entries[-1]
    assert entry.state is RenderLedgerState.INVALID
    assert not entry.retry_eligible
    assert entry.error_type == "InvalidVisualRenderPacketError"


def test_callback_operational_failure_remains_retryable_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.m4a_cli as m4a
    from latentguard.vision_data.pipeline import (
        VisualRenderExecutionError,
        run_visual_render_pipeline,
    )
    from latentguard.vision_data.rendering import (
        RenderLedgerState,
        load_render_ledger,
    )

    inventory, plan, scope = _callback_pipeline_fixture()

    def unavailable(**_kwargs: object) -> NoReturn:
        raise RuntimeError("renderer device unavailable")

    monkeypatch.setattr(
        m4a,
        "_create_visual_session",
        lambda _scope: SimpleNamespace(render_state=unavailable),
    )
    output = tmp_path / "operational-error"
    with pytest.raises(VisualRenderExecutionError):
        run_visual_render_pipeline(
            inventory,  # type: ignore[arg-type]
            output,
            run_id="m4a-operational-error-test",
            render_callback=m4a._pipeline_callback(plan, scope),  # type: ignore[arg-type]
        )
    ledger = load_render_ledger(output)
    entry = ledger.entries[-1]
    assert entry.state is RenderLedgerState.EXECUTION_ERROR
    assert entry.retry_eligible
    assert entry.error_type == "RuntimeError"


def test_validate_and_inspect_strict_serialized_partial_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a
    import latentguard.vision_data.dataset as visual_dataset

    root, dataset, render_root = _visual_bundle(tmp_path)
    scope = _scope()
    assert m4a._validate_render_inventory(
        dataset,
        render_root,
        scope,
        expected_source_identity_digest=_digest("source-identity"),
        expected_reset_seeds_by_anchor={"anchor-0": 17},
    ) == (17,)
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_a, **_k: scope)
    monkeypatch.setattr(
        m4a, "_validate_m3a_visual_source", lambda *_args, **_kwargs: (17,)
    )
    monkeypatch.setattr(
        visual_dataset,
        "validate_visual_verifier_dataset",
        lambda *_args, **_kwargs: None,
    )
    validate_args = argparse.Namespace(
        command="validate-visual-verifier-dataset",
        dataset_dir=[root],
        allow_partial=False,
        report_dir=tmp_path / "compact-reports",
    )
    assert run_m4a_command(validate_args) == 0
    assert "images=9" in capsys.readouterr().out
    report_dir = validate_args.report_dir
    assert {item.name for item in report_dir.iterdir()} == {
        "camera-rig-summary.json",
        "compact-retrieval-manifest.json",
        "development-domain-assignment.json",
        "development-dataset-summary.json",
        "development-image-digest-summary.json",
        "development-state-integrity.json",
        "render-determinism.json",
        "render-domain-summary.json",
    }
    from latentguard.training.reporting import load_strict_report

    integrity = load_strict_report(
        report_dir / "development-state-integrity.json",
        expected_report_type="m4a_render_state_integrity_v1",
    )
    assert integrity.payload["passed"] is True
    retrieval = load_strict_report(
        report_dir / "compact-retrieval-manifest.json",
        expected_report_type="m4a_compact_retrieval_manifest_v1",
    )
    assert retrieval.payload["raw_images_included"] is False
    assert retrieval.payload["raw_actions_included"] is False
    assert retrieval.payload["raw_state_archives_included"] is False

    packet = dataset.packets[0]  # type: ignore[attr-defined]
    inspect_args = argparse.Namespace(
        command="inspect-visual-packet",
        packet_dir=None,
        dataset_dir=root,
        packet_id=packet.packet_id,
        output=None,
    )
    assert run_m4a_command(inspect_args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["packet_id"] == packet.packet_id
    assert summary["image_count"] == 3
    assert summary["state_integrity_verified"] is True


def test_validation_retrieves_only_a_live_bound_zero_work_resume_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentguard.m4a_cli as m4a
    import latentguard.vision_data.dataset as visual_dataset
    from latentguard.vision_data.pipeline import load_render_job_inventory
    from latentguard.vision_data.rendering import (
        finish_render_attempt,
        load_render_ledger,
        plan_render_resume,
        start_render_attempt,
        update_render_ledger,
        verify_zero_work_render_resume,
    )
    from latentguard.vision_data.reporting import persist_render_resume_report

    root, _dataset, render_root = _visual_bundle(tmp_path)
    inventory = load_render_job_inventory(render_root)
    ledger = load_render_ledger(render_root)
    for index, packet_id in enumerate(inventory.packet_ids):
        rendering = start_render_attempt(
            ledger,
            packet_id,
            started_at=f"2026-07-18T12:00:{index:02d}Z",
        )
        update_render_ledger(ledger, rendering, render_root)
        ledger = load_render_ledger(render_root)
        complete = finish_render_attempt(
            ledger,
            packet_id,
            packet_content_digest=_digest(f"packet-{index}"),
            image_inventory_digest=_digest(f"images-{index}"),
            finished_at=f"2026-07-18T12:01:{index:02d}Z",
        )
        update_render_ledger(ledger, complete, render_root)
        ledger = load_render_ledger(render_root)
    unchanged, resume_plan = plan_render_resume(ledger)
    proof = verify_zero_work_render_resume(ledger, unchanged, resume_plan)
    persist_render_resume_report(
        proof,
        render_root,
        source_identity_digest=inventory.source_identity_digest,
        camera_rig_digest=inventory.camera_rig_digest,
        render_domain_configuration_digest=(
            inventory.render_domain_configuration_digest
        ),
    )

    scope = _scope()
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_a, **_k: scope)
    monkeypatch.setattr(
        m4a, "_validate_m3a_visual_source", lambda *_args, **_kwargs: (17,)
    )
    monkeypatch.setattr(
        visual_dataset,
        "validate_visual_verifier_dataset",
        lambda *_args, **_kwargs: None,
    )
    report_dir = tmp_path / "reports-with-resume"
    args = argparse.Namespace(
        command="validate-visual-verifier-dataset",
        dataset_dir=[root],
        development_render_root=render_root,
        allow_partial=False,
        report_dir=report_dir,
    )

    assert run_m4a_command(args) == 0
    from latentguard.training.reporting import load_strict_report

    resume = load_strict_report(
        report_dir / "development-resume.json",
        expected_report_type="m4a_render_resume_v1",
    )
    assert resume.payload["zero_duplicate_work"] is True
    assert resume.payload["ledger_content_digest"] == ledger.content_digest


def test_validate_two_datasets_publishes_complete_compact_report_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a
    import latentguard.vision_data.dataset as visual_dataset
    from latentguard.training.reporting import load_strict_report

    _, development, _ = _visual_bundle(tmp_path)
    external = _external_dataset_from_development(development)
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_a, **_k: _scope())
    monkeypatch.setattr(
        m4a,
        "_load_validation_roots",
        lambda _paths: (development, external),
    )
    monkeypatch.setattr(
        m4a,
        "_validate_m3a_visual_source",
        lambda *_args, **_kwargs: (17,),
    )
    monkeypatch.setattr(
        m4a,
        "_validate_m3c_visual_source",
        lambda *_args, **_kwargs: (29,),
    )
    monkeypatch.setattr(
        visual_dataset,
        "validate_visual_verifier_dataset",
        lambda *_args, **_kwargs: None,
    )
    report_dir = tmp_path / "dual-compact-reports"
    args = argparse.Namespace(
        command="validate-visual-verifier-dataset",
        dataset_dir=[tmp_path / "development", tmp_path / "external"],
        allow_partial=False,
        report_dir=report_dir,
    )

    assert run_m4a_command(args) == 0
    assert "cross_dataset_leakage=passed" in capsys.readouterr().out
    assert {item.name for item in report_dir.iterdir()} == {
        "camera-rig-summary.json",
        "compact-retrieval-manifest.json",
        "cross-dataset-leakage.json",
        "development-domain-assignment.json",
        "development-dataset-summary.json",
        "development-image-digest-summary.json",
        "development-state-integrity.json",
        "external-domain-assignment.json",
        "external-dataset-summary.json",
        "external-image-digest-summary.json",
        "external-state-integrity.json",
        "external-training-prohibition.json",
        "render-determinism.json",
        "render-domain-summary.json",
    }
    leakage = load_strict_report(
        report_dir / "cross-dataset-leakage.json",
        expected_report_type="m4a_cross_dataset_leakage_v1",
    )
    assert leakage.payload["cross_dataset_leakage_absent"] is True
    assert leakage.payload["repeated_images_within_development"] == []
    assert leakage.payload["repeated_images_within_external"] == [
        _digest("external-repeated-image")
    ]
    prohibition = load_strict_report(
        report_dir / "external-training-prohibition.json",
        expected_report_type="m4a_external_training_prohibition_v1",
    )
    assert prohibition.payload["training_loader_rejection_verified"] is True
    retrieval = load_strict_report(
        report_dir / "compact-retrieval-manifest.json",
        expected_report_type="m4a_compact_retrieval_manifest_v1",
    )
    assert retrieval.payload["report_file_count"] == 13
    assert all(item.suffix == ".json" for item in report_dir.iterdir())
    serialized = "\n".join(
        item.read_text(encoding="utf-8") for item in report_dir.iterdir()
    )
    assert "candidate_action_chunk_reference" not in serialized
    assert "state_reference_id" not in serialized
    assert "image_reference" not in serialized


def test_partial_validation_cannot_publish_acceptance_reports_before_source_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a

    monkeypatch.setattr(
        m4a,
        "_load_static_scope",
        lambda *_args, **_kwargs: pytest.fail(
            "partial report publication reached source loading"
        ),
    )
    report_dir = tmp_path / "compact-reports"
    args = argparse.Namespace(
        command="validate-visual-verifier-dataset",
        dataset_dir=[tmp_path / "partial-dataset"],
        allow_partial=True,
        report_dir=report_dir,
    )

    assert run_m4a_command(args) == 1
    assert "cannot publish acceptance reports" in capsys.readouterr().err
    assert not report_dir.exists()


def test_validate_rejects_existing_or_any_input_overlapping_report_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a

    source_root = tmp_path / "dataset"
    source_root.mkdir()
    runtime_root = tmp_path / "runtime-archive"
    runtime_root.mkdir()
    existing = tmp_path / "existing-reports"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    monkeypatch.setattr(
        m4a,
        "_load_static_scope",
        lambda *_a, **_k: pytest.fail("unsafe report path reached source loading"),
    )
    future_report_root = tmp_path / "future-report-root"
    for report_dir, extra_paths, message in (
        (existing, {}, "must be absent"),
        (
            source_root / "reports",
            {},
            "must not overlap any input artifact",
        ),
        (
            runtime_root / "reports",
            {"m3a_runtime_archive_dir": runtime_root},
            "must not overlap any input artifact",
        ),
        (
            future_report_root,
            {"m3c_source_dir": future_report_root / "future-source"},
            "must not overlap any input artifact",
        ),
    ):
        args = argparse.Namespace(
            command="validate-visual-verifier-dataset",
            dataset_dir=[source_root],
            allow_partial=True,
            report_dir=report_dir,
            **extra_paths,
        )
        assert run_m4a_command(args) == 1
        assert message in capsys.readouterr().err
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert not (source_root / "reports").exists()
    assert not (runtime_root / "reports").exists()
    assert not future_report_root.exists()


def test_inspect_output_rejects_source_overlap_and_linked_parent_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.vision_data.serialization as visual_serialization

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    packet_root = tmp_path / "packet"
    packet_root.mkdir()
    monkeypatch.setattr(
        visual_serialization,
        "load_visual_dataset",
        lambda _path: pytest.fail("overlapping output reached dataset loading"),
    )
    monkeypatch.setattr(
        visual_serialization,
        "load_visual_packet",
        lambda _path: pytest.fail("overlapping output reached packet loading"),
    )
    for args in (
        argparse.Namespace(
            command="inspect-visual-packet",
            packet_dir=None,
            dataset_dir=dataset_root,
            packet_id="vop-sha256-" + "a" * 64,
            output=dataset_root / "summary.json",
        ),
        argparse.Namespace(
            command="inspect-visual-packet",
            packet_dir=packet_root,
            dataset_dir=None,
            packet_id=None,
            output=packet_root / "summary.json",
        ),
    ):
        assert run_m4a_command(args) == 1
        assert "must not overlap any input artifact" in capsys.readouterr().err
        assert not args.output.exists()

    linked_parent = tmp_path / "linked-parent"
    real_link_guard = visual_serialization._is_link_or_junction
    monkeypatch.setattr(
        visual_serialization,
        "_is_link_or_junction",
        lambda path: Path(path) == linked_parent or real_link_guard(Path(path)),
    )
    linked_args = argparse.Namespace(
        command="inspect-visual-packet",
        packet_dir=packet_root,
        dataset_dir=None,
        packet_id=None,
        output=linked_parent / "summary.json",
    )
    assert run_m4a_command(linked_args) == 1
    assert "parent cannot traverse a link or junction" in capsys.readouterr().err
    assert not linked_parent.exists()


def test_validate_preserves_original_root_for_link_and_junction_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentguard.m4a_cli as m4a
    import latentguard.vision_data.serialization as visual_serialization

    root, dataset, _ = _visual_bundle(tmp_path)
    supplied = root.parent / "lexical-parent" / ".." / root.name
    original_absolute = supplied.absolute()
    assert original_absolute != original_absolute.resolve()
    observed: list[Path] = []

    def guarded_loader(path: Path) -> object:
        observed.append(path)
        return dataset

    monkeypatch.setattr(
        visual_serialization,
        "load_visual_dataset",
        guarded_loader,
    )
    assert m4a._load_validation_roots([supplied]) == (dataset,)
    assert observed == [original_absolute]


def test_validate_fails_closed_without_independent_source_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a

    root, _, _ = _visual_bundle(tmp_path)
    monkeypatch.setattr(m4a, "_load_static_scope", lambda *_a, **_k: _scope())
    args = argparse.Namespace(
        command="validate-visual-verifier-dataset",
        dataset_dir=[root],
        allow_partial=True,
    )
    assert run_m4a_command(args) == 1
    assert "missing required source inputs" in capsys.readouterr().err


def test_external_trust_roots_reject_coherently_rehashed_calibration_tamper(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import latentguard.m4a_cli as m4a

    _, dataset, render_root = _visual_bundle(tmp_path)
    packets = list(dataset.packets)  # type: ignore[attr-defined]
    views = list(packets[0].views)
    intrinsics = [list(row) for row in views[0].intrinsics]
    intrinsics[0][0] += 0.25
    views[0] = replace(
        views[0],
        intrinsics=tuple(tuple(row) for row in intrinsics),
    )
    packets[0] = replace(packets[0], views=tuple(views))
    tampered = build_visual_development_dataset(
        source_dataset_digest=dataset.source_dataset_digest,  # type: ignore[attr-defined]
        split_digest=dataset.split_digest,  # type: ignore[attr-defined]
        evidence_digest=dataset.evidence_digest,  # type: ignore[attr-defined]
        source_compatibility_identity=(
            dataset.source_compatibility_identity  # type: ignore[attr-defined]
        ),
        packets=tuple(packets),
        candidate_bindings=dataset.candidate_bindings,  # type: ignore[attr-defined]
        samples=dataset.samples,  # type: ignore[attr-defined]
        camera_rig_digest=dataset.camera_rig_digest,  # type: ignore[attr-defined]
        render_domain_configuration_digest=(
            dataset.render_domain_configuration_digest  # type: ignore[attr-defined]
        ),
        visual_compatibility_identity=(
            dataset.visual_compatibility_identity  # type: ignore[attr-defined]
        ),
        full_target=False,
    )
    assert tampered.content_digest != dataset.content_digest  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="front_oblique.intrinsics"):
        m4a._validate_render_inventory(
            tampered,
            render_root,
            _scope(),
            expected_source_identity_digest=_digest("source-identity"),
            expected_reset_seeds_by_anchor={"anchor-0": 17},
        )


def test_float32_probe_rejects_rehashed_float16_calibration_alias(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import latentguard.m4a_cli as m4a

    _, dataset, render_root = _visual_bundle(tmp_path)
    packets = []
    for packet_index, packet in enumerate(dataset.packets):  # type: ignore[attr-defined]
        views = []
        for view_index, view in enumerate(packet.views):
            intrinsics = np.asarray(view.intrinsics, dtype=np.float32)
            extrinsics = np.asarray(view.extrinsics, dtype=np.float32)
            if packet_index == 0 and view_index == 0:
                float16_intrinsics = intrinsics.astype(np.float16).astype(np.float64)
                assert not np.array_equal(
                    intrinsics.astype(np.float64), float16_intrinsics
                )
                serialized_intrinsics = float16_intrinsics
            else:
                serialized_intrinsics = intrinsics.astype(np.float64)
            views.append(
                replace(
                    view,
                    intrinsics=tuple(
                        tuple(float(value) for value in row)
                        for row in serialized_intrinsics
                    ),
                    intrinsics_dtype="float32",
                    extrinsics=tuple(
                        tuple(float(value) for value in row)
                        for row in extrinsics.astype(np.float64)
                    ),
                    extrinsics_dtype="float32",
                )
            )
        packets.append(replace(packet, views=tuple(views)))
    tampered = build_visual_development_dataset(
        source_dataset_digest=dataset.source_dataset_digest,  # type: ignore[attr-defined]
        split_digest=dataset.split_digest,  # type: ignore[attr-defined]
        evidence_digest=dataset.evidence_digest,  # type: ignore[attr-defined]
        source_compatibility_identity=(
            dataset.source_compatibility_identity  # type: ignore[attr-defined]
        ),
        packets=tuple(packets),
        candidate_bindings=dataset.candidate_bindings,  # type: ignore[attr-defined]
        samples=dataset.samples,  # type: ignore[attr-defined]
        camera_rig_digest=dataset.camera_rig_digest,  # type: ignore[attr-defined]
        render_domain_configuration_digest=(
            dataset.render_domain_configuration_digest  # type: ignore[attr-defined]
        ),
        visual_compatibility_identity=(
            dataset.visual_compatibility_identity  # type: ignore[attr-defined]
        ),
        full_target=False,
    )
    assert tampered.content_digest != dataset.content_digest  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="front_oblique.intrinsics"):
        m4a._validate_render_inventory(
            tampered,
            render_root,
            _scope(calibration_dtype="float32"),
            expected_source_identity_digest=_digest("source-identity"),
            expected_reset_seeds_by_anchor={"anchor-0": 17},
        )


def test_external_trust_roots_reject_checked_in_camera_rig_drift(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import latentguard.m4a_cli as m4a

    _, dataset, render_root = _visual_bundle(tmp_path)
    scope = _scope()
    cameras = list(scope.camera_rig.cameras)
    cameras[0] = replace(cameras[0], near=cameras[0].near + 0.001)
    drifted_scope = replace(
        scope,
        camera_rig=replace(scope.camera_rig, cameras=tuple(cameras)),
    )
    with pytest.raises(ValueError, match="dataset differs in camera_rig_digest"):
        m4a._validate_render_inventory(
            dataset,
            render_root,
            drifted_scope,
            expected_source_identity_digest=_digest("source-identity"),
            expected_reset_seeds_by_anchor={"anchor-0": 17},
        )


def test_source_bound_render_inventory_rejects_rehashed_reset_seed_tamper(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import latentguard.m4a_cli as m4a
    from latentguard.vision_data.pipeline import (
        VisualRenderJobInventoryV1,
        load_render_job_inventory,
        save_render_job_inventory,
    )

    _, dataset, render_root = _visual_bundle(tmp_path)
    inventory = load_render_job_inventory(render_root)
    jobs = list(inventory.jobs)
    jobs[0] = replace(jobs[0], source_reset_seed=18)
    tampered_root = tmp_path / "tampered-render-root"
    tampered_root.mkdir()
    save_render_job_inventory(
        VisualRenderJobInventoryV1.create(
            source_identity_digest=inventory.source_identity_digest,
            camera_rig_digest=inventory.camera_rig_digest,
            render_domain_configuration_digest=(
                inventory.render_domain_configuration_digest
            ),
            visual_compatibility_identity=(inventory.visual_compatibility_identity),
            base_render_seed=inventory.base_render_seed,
            jobs=jobs,
        ),
        tampered_root,
    )
    with pytest.raises(ValueError, match="source reset seed differs"):
        m4a._validate_render_inventory(
            dataset,
            tampered_root,
            _scope(),
            expected_source_identity_digest=_digest("source-identity"),
            expected_reset_seeds_by_anchor={"anchor-0": 17},
        )


def test_interrupt_is_reported_as_resumable_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import latentguard.m4a_cli as m4a

    def interrupt(_: argparse.Namespace) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(m4a, "_run_validate_visual_verifier_dataset", interrupt)
    args = argparse.Namespace(
        command="validate-visual-verifier-dataset",
        dataset_dir=[tmp_path],
        allow_partial=True,
    )
    assert run_m4a_command(args) == 130
    assert "may be resumed" in capsys.readouterr().err
