from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST
from latentguard.vision_data import (
    M4A_VISUAL_MODEL_INPUT_CONTRACT,
    PICKCUBE_CANONICAL_TASK_TEXT,
    PICKCUBE_VISUAL_TASK_ID,
    SourceCollection,
    VisionConfigurationError,
    VisionDataValidationError,
    VisualActionVerifierSampleV1,
    VisualCandidateBindingV1,
    VisualDatasetError,
    VisualDatasetSplit,
    VisualObservationPacketV1,
    VisualTaskProjectionV1,
    VisualVerifierDevelopmentDatasetV1,
    VisualViewRecordV1,
    assigned_render_domain_ids,
    build_visual_development_dataset,
    build_visual_external_dataset,
    compute_visual_packet_identifier_from_fields,
    load_camera_rig_configuration,
    load_render_domain_configuration,
    require_training_allowed,
)
from latentguard.vision_data.reporting import (
    build_domain_assignment_report,
    build_image_digest_summary_report,
    build_state_integrity_report_from_dataset,
    build_visual_dataset_summary_report,
)

_CONFIG_ROOT = Path("configs/vision/m4a")
_CANONICAL_TASK_TEXT = PICKCUBE_CANONICAL_TASK_TEXT


def _digest(label: str) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


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


def _packet(
    *,
    collection: SourceCollection = SourceCollection.M3A_DEVELOPMENT,
    split: VisualDatasetSplit = VisualDatasetSplit.TRAIN,
    domain_id: str | None = None,
    render_seed: int = 7,
) -> VisualObservationPacketV1:
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    selected_domain = (
        assigned_render_domain_ids(collection, split)[0]
        if domain_id is None
        else domain_id
    )
    task = _task_projection()
    state_digest = _digest("state")
    views = tuple(
        VisualViewRecordV1(
            camera_id=camera.camera_id,
            image_reference=(
                f"images/{selected_domain}/{camera.camera_id}-{render_seed}.npy"
            ),
            pixel_sha256=_digest(
                f"pixel-{selected_domain}-{camera.camera_id}-{render_seed}"
            ),
            npy_sha256=_digest(
                f"npy-{selected_domain}-{camera.camera_id}-{render_seed}"
            ),
            dtype="uint8",
            shape=(224, 224, 3),
            intrinsics=camera.intrinsics,
            intrinsics_dtype="float64",
            extrinsics=camera.extrinsics,
            extrinsics_dtype="float64",
            runtime_extrinsics_digest=_digest(f"runtime-extrinsics-{camera.camera_id}"),
            expected_runtime_extrinsics_digest=_digest(
                f"runtime-extrinsics-{camera.camera_id}"
            ),
            camera_configuration_digest=camera.content_digest,
            state_before_render_digest=state_digest,
            state_after_render_digest=state_digest,
            compared_state_component_count=70,
            maximum_state_error=0.0,
            task_projection_before=task,
            task_projection_after=task,
        )
        for camera in rig.cameras
    )
    return VisualObservationPacketV1(
        source_collection=collection,
        source_trajectory_id="trajectory-0",
        anchor_id="anchor-0",
        split=split,
        split_group_id="split-group-0",
        state_reference_id="state-reference-0",
        expected_state_digest=state_digest,
        verifier_state_semantic="PickCubeVerifierStateV1",
        verifier_state_digest=_digest("verifier-state"),
        verifier_state_component_count=38,
        verifier_state_maximum_absolute_error=0.0,
        elapsed_simulation_steps_before=0,
        elapsed_simulation_steps_after=0,
        environment_close_passed=True,
        camera_rig_id=rig.rig_id,
        camera_rig_digest=rig.content_digest,
        render_domain_id=selected_domain,
        render_domain_digest=domains.domain(selected_domain).content_digest,
        render_seed=render_seed,
        views=views,
        visual_compatibility_identity=_digest("visual-compatibility"),
        pickcube_compatibility_identity=_digest("pickcube-compatibility"),
        renderer_semantic_version="sapien-renderer-v1",
    )


def _packet_set(
    collection: SourceCollection = SourceCollection.M3A_DEVELOPMENT,
    split: VisualDatasetSplit = VisualDatasetSplit.TRAIN,
) -> tuple[VisualObservationPacketV1, ...]:
    return tuple(
        _packet(
            collection=collection,
            split=split,
            domain_id=domain_id,
            render_seed=index,
        )
        for index, domain_id in enumerate(
            assigned_render_domain_ids(collection, split), start=1
        )
    )


def _binding(
    packets: tuple[VisualObservationPacketV1, ...],
    collection: SourceCollection,
) -> VisualCandidateBindingV1:
    return VisualCandidateBindingV1(
        candidate_sample_id="candidate-0",
        candidate_group_id="candidate-group-0",
        anchor_id="anchor-0",
        source_collection=collection,
        packet_ids=(
            packets[0].packet_id,
            packets[1].packet_id,
            packets[2].packet_id,
        ),
    )


def _samples(
    packets: tuple[VisualObservationPacketV1, ...],
    collection: SourceCollection,
    *,
    source_dataset_digest: str,
) -> tuple[VisualActionVerifierSampleV1, ...]:
    return tuple(
        VisualActionVerifierSampleV1(
            packet_id=packet.packet_id,
            candidate_sample_id="candidate-0",
            task_id=PICKCUBE_VISUAL_TASK_ID,
            canonical_task_text=_CANONICAL_TASK_TEXT,
            candidate_action_chunk_reference="compact/actions/candidate-0.npy",
            action_mask_reference="compact/actions/candidate-0-mask.npy",
            final_success=True,
            final_unsafe=False,
            failure_events=(),
            evidence_id="evidence-0",
            source_dataset_digest=source_dataset_digest,
            candidate_dataset_digest=_digest("candidate-dataset"),
            visual_dataset_digest=_digest("pending-visual-dataset"),
            source_collection=collection,
        )
        for packet in packets
    )


def test_packet_plan_identity_matches_complete_packet_and_is_deterministic() -> None:
    packet = _packet()

    planned = compute_visual_packet_identifier_from_fields(
        source_trajectory_id=packet.source_trajectory_id,
        anchor_id=packet.anchor_id,
        expected_state_digest=packet.expected_state_digest,
        camera_rig_digest=packet.camera_rig_digest,
        render_domain_digest=packet.render_domain_digest,
        render_seed=packet.render_seed,
        camera_configuration_digests=tuple(
            view.camera_configuration_digest for view in packet.views
        ),
        visual_compatibility_identity=packet.visual_compatibility_identity,
        schema_version=packet.schema_version,
    )

    assert packet.packet_id == planned
    assert packet.packet_id == _packet().packet_id
    assert packet.content_digest == _packet().content_digest
    assert replace(packet, render_seed=8).packet_id != packet.packet_id


def test_packet_semantic_identity_excludes_runtime_reference_and_pixels() -> None:
    packet = _packet()
    changed_views = tuple(
        replace(
            view,
            image_reference=f"relocated/{view.camera_id}.npy",
            pixel_sha256=_digest(f"changed-pixels-{view.camera_id}"),
            npy_sha256=_digest(f"changed-npy-{view.camera_id}"),
        )
        for view in packet.views
    )
    changed = replace(packet, views=changed_views)

    assert changed.packet_id == packet.packet_id
    assert changed.content_digest != packet.content_digest


def test_packet_rejects_order_version_and_identity_tampering() -> None:
    packet = _packet()
    with pytest.raises(VisionDataValidationError, match="ordered three views"):
        replace(packet, views=tuple(reversed(packet.views)))
    with pytest.raises(VisionDataValidationError, match="unsupported version"):
        replace(packet, schema_version="2.0")

    changed = packet.as_mapping()
    changed["packet_id"] = f"vop-sha256-{'0' * 64}"
    with pytest.raises(VisionConfigurationError, match="content mismatch"):
        VisualObservationPacketV1.from_mapping(changed)

    changed = packet.as_mapping()
    views = changed["views"]
    assert isinstance(views, list)
    assert isinstance(views[0], dict)
    views[0]["pixel_sha256"] = "0" * 64
    with pytest.raises(VisionDataValidationError, match="sha256"):
        VisualObservationPacketV1.from_mapping(changed)

    changed = packet.as_mapping()
    views = changed["views"]
    assert isinstance(views, list)
    assert isinstance(views[0], dict)
    views[0]["dtype"] = "|u1"
    with pytest.raises(VisionDataValidationError, match="uint8"):
        VisualObservationPacketV1.from_mapping(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("verifier_state_component_count", 37, "expected 38"),
        (
            "verifier_state_maximum_absolute_error",
            1e-12,
            "exact verifier-state equality",
        ),
        (
            "elapsed_simulation_steps_after",
            1,
            "advanced the environment step counter",
        ),
        ("elapsed_simulation_steps_before", True, "expected integer"),
        ("environment_close_passed", False, "successful environment close"),
    ],
)
def test_packet_integrity_evidence_round_trips_and_rejects_tampering(
    field: str, value: object, message: str
) -> None:
    packet = _packet()
    assert VisualObservationPacketV1.from_mapping(packet.as_mapping()) == packet

    changed = packet.as_mapping()
    changed[field] = value
    with pytest.raises(ValueError, match=message):
        VisualObservationPacketV1.from_mapping(changed)


def test_packet_rejects_unbound_calibration_dtype() -> None:
    packet = _packet()
    changed = packet.as_mapping()
    views = changed["views"]
    assert isinstance(views, list)
    assert isinstance(views[0], dict)
    views[0]["intrinsics_dtype"] = "float128"
    with pytest.raises(VisionDataValidationError, match="calibration dtype"):
        VisualObservationPacketV1.from_mapping(changed)


def test_packet_integrity_evidence_is_content_bound_but_not_runtime_identity() -> None:
    packet = _packet()
    later_elapsed_boundary = replace(
        packet,
        elapsed_simulation_steps_before=4,
        elapsed_simulation_steps_after=4,
    )

    assert later_elapsed_boundary.packet_id == packet.packet_id
    assert later_elapsed_boundary.content_digest != packet.content_digest


def test_packet_models_are_mutation_safe() -> None:
    packet = _packet()
    assert isinstance(packet.views, tuple)
    assert isinstance(packet.views[0].shape, tuple)
    with pytest.raises(FrozenInstanceError):
        packet.render_seed = 99  # type: ignore[misc]


def test_visual_sample_requires_fixed_pickcube_task_identity_and_text() -> None:
    packet = _packet()
    sample = _samples(
        (packet, packet, packet),
        SourceCollection.M3A_DEVELOPMENT,
        source_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
    )[0]

    with pytest.raises(VisionDataValidationError, match="fixed maniskill/PickCube"):
        replace(sample, task_id="other-task")
    with pytest.raises(VisionDataValidationError, match="fixed canonical task text"):
        replace(sample, canonical_task_text="Move the cube.")


def test_development_dataset_stamps_and_strictly_checks_sample_backrefs() -> None:
    packets = _packet_set()
    binding = _binding(packets, SourceCollection.M3A_DEVELOPMENT)
    samples = _samples(
        packets,
        SourceCollection.M3A_DEVELOPMENT,
        source_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
    )
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    dataset = build_visual_development_dataset(
        source_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
        split_digest=_digest("split"),
        evidence_digest=_digest("evidence"),
        source_compatibility_identity=_digest("source-compatibility"),
        packets=packets,
        candidate_bindings=(binding,),
        samples=samples,
        camera_rig_digest=rig.content_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        full_target=False,
    )

    assert all(
        sample.visual_dataset_digest == dataset.content_digest
        for sample in dataset.samples
    )
    reloaded = VisualVerifierDevelopmentDatasetV1.from_mapping(dataset.as_mapping())
    assert reloaded.content_digest == dataset.content_digest
    integrity_report = build_state_integrity_report_from_dataset(reloaded)
    assert integrity_report.payload["passed"] is True
    assert integrity_report.payload["packet_count"] == 3
    assert integrity_report.payload["visual_dataset_digest"] == dataset.content_digest
    summary = build_visual_dataset_summary_report(reloaded)
    image_summary = build_image_digest_summary_report(reloaded)
    assignment = build_domain_assignment_report(reloaded)
    assert (
        summary.payload["image_inventory_digest"]
        == image_summary.payload["image_inventory_digest"]
    )
    assert summary.payload["split_domain_packet_counts"] == {
        "train": {
            "canonical": 1,
            "mild_camera_shift": 1,
            "mild_lighting_shift": 1,
        }
    }
    assert image_summary.payload["image_count"] == 9
    assert image_summary.payload["raw_image_bytes_included"] is False
    assert assignment.payload["assignment_validation_passed"] is True
    assert (
        assignment.payload["split_domain_packet_counts"]
        == summary.payload["split_domain_packet_counts"]
    )

    changed = dataset.as_mapping()
    changed_samples = changed["samples"]
    assert isinstance(changed_samples, list)
    assert isinstance(changed_samples[0], dict)
    changed_samples[0]["visual_dataset_digest"] = _digest("wrong-dataset")
    with pytest.raises(ValueError, match="backref mismatch"):
        VisualVerifierDevelopmentDatasetV1.from_mapping(changed)


def test_external_dataset_is_explicitly_evaluation_only() -> None:
    packets = _packet_set(SourceCollection.M3C_EXTERNAL, VisualDatasetSplit.EXTERNAL)
    binding = _binding(packets, SourceCollection.M3C_EXTERNAL)
    samples = _samples(
        packets,
        SourceCollection.M3C_EXTERNAL,
        source_dataset_digest=_digest("m3c-source-set"),
    )
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    dataset = build_visual_external_dataset(
        source_set_identity=_digest("m3c-source-set"),
        candidate_pool_identity=_digest("m3c-candidate-pool"),
        blind_manifest_digest=_digest("m3c-blind-manifest"),
        full_outcome_digest=_digest("m3c-full-outcome"),
        source_compatibility_identity=_digest("m3c-compatibility"),
        packets=packets,
        candidate_bindings=(binding,),
        samples=samples,
        camera_rig_digest=rig.content_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        full_target=False,
    )

    assert dataset.training_allowed is False
    assert all(
        sample.visual_dataset_digest == dataset.content_digest
        for sample in dataset.samples
    )
    with pytest.raises(VisualDatasetError, match="evaluation-only"):
        require_training_allowed(dataset)


def test_visual_student_input_allowlist_is_exact_and_disjoint() -> None:
    contract = M4A_VISUAL_MODEL_INPUT_CONTRACT
    assert contract.required_student_inputs == (
        "ordered_rgb_views",
        "candidate_action_chunk",
        "action_mask",
    )
    assert contract.conditional_student_inputs == ("canonical_task_text",)
    assert "render_domain_id" in contract.reporting_only_fields
    assert "camera_id" in contract.reporting_only_fields
    assert "verifier_state" in contract.privileged_teacher_fields
    assert "complete_simulator_state" in contract.privileged_teacher_fields
    inventories = (
        set(contract.required_student_inputs),
        set(contract.conditional_student_inputs),
        set(contract.reporting_only_fields),
        set(contract.privileged_teacher_fields),
    )
    assert all(
        not left.intersection(right)
        for index, left in enumerate(inventories)
        for right in inventories[index + 1 :]
    )
