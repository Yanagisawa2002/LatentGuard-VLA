from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray

from latentguard.integrations.maniskill_pickcube.visual_rendering import (
    VISUAL_RENDERER_SEMANTIC_VERSION,
    build_pickcube_visual_render_plan,
)
from latentguard.vision_data import (
    MILD_CAMERA_DOMAIN_ID,
    RENDER_SEED_DERIVATION,
    SourceCollection,
    VisualDatasetSplit,
    VisualObservationPacketV1,
    VisualTaskProjectionV1,
    VisualViewRecordV1,
    assigned_render_domain_ids,
    load_camera_rig_configuration,
    load_render_domain_configuration,
    prepare_npy_image,
    save_visual_packet,
)
from latentguard.vision_data.pipeline import (
    InvalidVisualRenderPacketError,
    VisualPacketJobV1,
    VisualRenderExecutionError,
    VisualRenderJobInventoryV1,
    VisualRenderPipelineError,
    derive_render_seed,
    load_render_job_inventory,
    run_visual_render_pipeline,
    save_render_job_inventory,
)
from latentguard.vision_data.rendering import (
    RenderLedgerError,
    RenderLedgerState,
    load_render_ledger,
)
from latentguard.vision_data.run_manifest import (
    M4A_RUN_MANIFEST_FILENAME,
    M4AOperationalEnvironmentV1,
    M4AOperationalRunManifestV1,
    M4AOperationalSourceIdentityV1,
    M4ARunManifestError,
    compute_selected_packet_inventory_digest,
    load_m4a_run_manifest,
)
from latentguard.vision_data.serialization import VisualSerializationError

_CONFIG_ROOT = Path("configs/vision/m4a")
_RUN_ID = "m4a-cpu-pipeline-test"


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _clock() -> str:
    return "2026-07-18T00:00:00.000000Z"


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


def _render_plan(job: VisualPacketJobV1) -> object:
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    return build_pickcube_visual_render_plan(
        rig,
        domains.domain(job.render_domain_id),
        domains,
        job.render_seed,
    )


def _inventory(
    *,
    source_collection: SourceCollection = SourceCollection.M3A_DEVELOPMENT,
    source_identity_digest: str | None = None,
) -> VisualRenderJobInventoryV1:
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    base_seed = 271828
    split = (
        VisualDatasetSplit.EXTERNAL
        if source_collection is SourceCollection.M3C_EXTERNAL
        else VisualDatasetSplit.TRAIN
    )
    jobs: list[VisualPacketJobV1] = []
    for ordinal, domain_id in enumerate(
        assigned_render_domain_ids(source_collection, split)
    ):
        domain = domains.domain(domain_id)
        render_seed = derive_render_seed("anchor-0", domain_id, base_seed)
        plan = build_pickcube_visual_render_plan(
            rig,
            domain,
            domains,
            render_seed,
        )
        jobs.append(
            VisualPacketJobV1.create(
                job_ordinal=ordinal,
                source_collection=source_collection,
                source_trajectory_id="trajectory-0",
                source_reset_seed=17,
                anchor_id="anchor-0",
                split=split,
                split_group_id="split-group-0",
                state_reference_id="states/trajectory-0/anchor-0.npz",
                expected_state_digest=_digest("source-state"),
                verifier_state_semantic="PickCubeVerifierStateV1",
                verifier_state_digest=_digest("verifier-state"),
                camera_rig_id=rig.rig_id,
                camera_rig_digest=rig.content_digest,
                render_domain_id=domain_id,
                render_domain_digest=domain.content_digest,
                base_render_seed=base_seed,
                camera_configuration_digests=tuple(
                    camera.camera_configuration_digest for camera in plan.cameras
                ),
                visual_compatibility_identity=_digest("visual-compatibility"),
                pickcube_compatibility_identity=_digest("pickcube-compatibility"),
                renderer_semantic_version=VISUAL_RENDERER_SEMANTIC_VERSION,
            )
        )
    return VisualRenderJobInventoryV1.create(
        source_identity_digest=(
            _digest("source-identity")
            if source_identity_digest is None
            else source_identity_digest
        ),
        camera_rig_digest=rig.content_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        base_render_seed=base_seed,
        jobs=jobs,
    )


def _operational_manifest(
    hostname: str = "m4a-worker-01",
    *,
    inventory: VisualRenderJobInventoryV1 | None = None,
    source_identity: M4AOperationalSourceIdentityV1 | None = None,
    render_seed: int | None = None,
    selected_packet_ids: tuple[str, ...] | None = None,
) -> M4AOperationalRunManifestV1:
    effective_inventory = _inventory() if inventory is None else inventory
    effective_source_identity = source_identity
    if effective_source_identity is None:
        effective_source_identity = (
            M4AOperationalSourceIdentityV1.for_m3c(
                _digest("accepted-m3c-source-set"),
                _digest("accepted-m3c-candidate-pool"),
                canonical_source_identity_digest=(
                    effective_inventory.source_identity_digest
                ),
            )
            if effective_inventory.source_collection is SourceCollection.M3C_EXTERNAL
            else M4AOperationalSourceIdentityV1.for_m3a(
                _digest("accepted-m3a-dataset"),
                canonical_source_identity_digest=(
                    effective_inventory.source_identity_digest
                ),
            )
        )
    selected = (
        effective_inventory.packet_ids
        if selected_packet_ids is None
        else selected_packet_ids
    )
    return M4AOperationalRunManifestV1.create(
        run_id=_RUN_ID,
        command=effective_source_identity.required_command,
        git_sha="a" * 40,
        branch="codex/m4a-multiview-visual-dataset",
        started_at="2026-07-18T00:00:00Z",
        render_seed=(
            effective_inventory.base_render_seed if render_seed is None else render_seed
        ),
        launch_argv=("latentguard", "render-m3a-visual-dataset", "--seed", "271828"),
        environment=M4AOperationalEnvironmentV1.create(
            hostname=hostname,
            python_version="3.11.13",
            gpu_model="NVIDIA GeForce RTX 5090",
            gpu_driver_version="575.64.03",
            cuda_version="12.8",
            pytorch_version="2.7.1+cu128",
            maniskill_version="3.0.1",
            sapien_version="3.0.0.b1",
            renderer_backend="sapien-vulkan",
        ),
        visual_compatibility_identity=effective_inventory.visual_compatibility_identity,
        camera_rig_digest=effective_inventory.camera_rig_digest,
        render_domain_configuration_digest=(
            effective_inventory.render_domain_configuration_digest
        ),
        source_identity=effective_source_identity,
        render_job_inventory_digest=effective_inventory.content_digest,
        selected_packet_inventory_digest=compute_selected_packet_inventory_digest(
            selected
        ),
    )


def test_render_seed_semantic_is_shared_and_fail_closed() -> None:
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    assert domains.seed_derivation == RENDER_SEED_DERIVATION
    assert derive_render_seed(
        "anchor-0", "canonical", 271828, RENDER_SEED_DERIVATION
    ) == derive_render_seed("anchor-0", "canonical", 271828)

    with pytest.raises(VisualRenderPipelineError, match="unsupported semantic"):
        derive_render_seed(
            "anchor-0", "canonical", 271828, "claimed-but-unimplemented-v999"
        )
    with pytest.raises(VisualRenderPipelineError, match="seed derivation"):
        replace(
            _inventory().jobs[0],
            seed_derivation="claimed-but-unimplemented-v999",
        )


@dataclass(frozen=True, slots=True)
class _Prepared:
    packet: VisualObservationPacketV1
    images: Mapping[str, NDArray[Any]]


def _prepared(job: VisualPacketJobV1) -> _Prepared:
    plan = _render_plan(job)
    task_projection = _task_projection()
    runtime_state_digest = _digest(f"runtime-state-{job.packet_id}")
    images: dict[str, NDArray[Any]] = {}
    views: list[VisualViewRecordV1] = []
    for index, camera in enumerate(plan.cameras):
        image = np.full(
            (224, 224, 3),
            (job.job_ordinal * 41 + index * 17 + 3) % 256,
            dtype=np.uint8,
        )
        prepared = prepare_npy_image(image)
        reference = f"images/{job.packet_id}/{camera.camera_id}.npy"
        images[reference] = prepared.image
        views.append(
            VisualViewRecordV1(
                camera_id=camera.camera_id,
                image_reference=reference,
                pixel_sha256=prepared.pixel_sha256,
                npy_sha256=prepared.npy_sha256,
                dtype="uint8",
                shape=(224, 224, 3),
                intrinsics=tuple(
                    tuple(float(item) for item in row)
                    for row in camera.intrinsics.tolist()
                ),
                intrinsics_dtype=camera.intrinsics.dtype.name,
                extrinsics=tuple(
                    tuple(float(item) for item in row)
                    for row in camera.extrinsics.tolist()
                ),
                extrinsics_dtype=camera.extrinsics.dtype.name,
                runtime_extrinsics_digest=_digest(
                    f"runtime-extrinsics-{camera.camera_id}"
                ),
                expected_runtime_extrinsics_digest=_digest(
                    f"runtime-extrinsics-{camera.camera_id}"
                ),
                camera_configuration_digest=(camera.camera_configuration_digest),
                state_before_render_digest=runtime_state_digest,
                state_after_render_digest=runtime_state_digest,
                compared_state_component_count=70,
                maximum_state_error=0.0,
                task_projection_before=task_projection,
                task_projection_after=task_projection,
            )
        )
    packet = VisualObservationPacketV1(
        source_collection=job.source_collection,
        source_trajectory_id=job.source_trajectory_id,
        anchor_id=job.anchor_id,
        split=job.split,
        split_group_id=job.split_group_id,
        state_reference_id=job.state_reference_id,
        expected_state_digest=job.expected_state_digest,
        verifier_state_semantic=job.verifier_state_semantic,
        verifier_state_digest=job.verifier_state_digest,
        verifier_state_component_count=38,
        verifier_state_maximum_absolute_error=0.0,
        elapsed_simulation_steps_before=0,
        elapsed_simulation_steps_after=0,
        environment_close_passed=True,
        camera_rig_id=job.camera_rig_id,
        camera_rig_digest=job.camera_rig_digest,
        render_domain_id=job.render_domain_id,
        render_domain_digest=job.render_domain_digest,
        render_seed=job.render_seed,
        views=tuple(views),
        visual_compatibility_identity=job.visual_compatibility_identity,
        pickcube_compatibility_identity=job.pickcube_compatibility_identity,
        renderer_semantic_version=job.renderer_semantic_version,
    )
    return _Prepared(packet=packet, images=images)


def _latest_state(output_dir: Path, packet_id: str) -> RenderLedgerState:
    entries = tuple(
        entry
        for entry in load_render_ledger(output_dir).entries
        if entry.packet_id == packet_id
    )
    return entries[-1].state


def test_seed_and_job_inventory_are_deterministic_and_self_bound(
    tmp_path: Path,
) -> None:
    first = _inventory()
    second = _inventory()

    assert first == second
    assert first.content_digest == second.content_digest
    assert len(set(first.packet_ids)) == 3
    assert all(0 <= job.render_seed < 2**32 for job in first.jobs)
    assert tuple(job.job_ordinal for job in first.jobs) == (0, 1, 2)

    output = tmp_path / "manifest"
    output.mkdir()
    save_render_job_inventory(first, output)
    assert load_render_job_inventory(output) == first


def test_resume_rejects_immutable_manifest_drift(tmp_path: Path) -> None:
    inventory = _inventory()
    output = tmp_path / "render"
    run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        timestamp_factory=_clock,
    )
    drifted = replace(
        inventory,
        source_identity_digest=_digest("different-source-identity"),
    )

    with pytest.raises(VisualRenderPipelineError, match="job inventory content drift"):
        run_visual_render_pipeline(
            drifted,
            output,
            run_id=_RUN_ID,
            render_callback=_prepared,
            resume=True,
            timestamp_factory=_clock,
        )


def test_pipeline_transactionally_binds_operational_manifest_and_resume(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    manifest = _operational_manifest()
    output = tmp_path / "operational-manifest"
    first = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        timestamp_factory=_clock,
        operational_manifest=manifest,
    )
    assert first.complete
    manifest_path = output / M4A_RUN_MANIFEST_FILENAME
    assert load_m4a_run_manifest(manifest_path) == manifest
    manifest_bytes = manifest_path.read_bytes()
    manifest_mtime = manifest_path.stat().st_mtime_ns

    resumed = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=lambda _job: pytest.fail("zero-work resume rendered a packet"),
        resume=True,
        timestamp_factory=_clock,
        operational_manifest=manifest,
    )
    assert resumed.zero_work_proof is not None
    assert resumed.rendered_packet_ids == ()
    assert manifest_path.read_bytes() == manifest_bytes
    assert manifest_path.stat().st_mtime_ns == manifest_mtime

    with pytest.raises(M4ARunManifestError, match="operational identity drift"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=_prepared,
            resume=True,
            timestamp_factory=_clock,
            operational_manifest=_operational_manifest("m4a-worker-02"),
        )
    assert manifest_path.read_bytes() == manifest_bytes


def test_pipeline_manifest_failure_leaves_no_published_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import latentguard.vision_data.run_manifest as run_manifest_module

    output = tmp_path / "manifest-publication-failure"

    def fail_manifest(*_args: object, **_kwargs: object) -> NoReturn:
        raise M4ARunManifestError("simulated manifest failure")

    monkeypatch.setattr(
        run_manifest_module,
        "save_m4a_run_manifest",
        fail_manifest,
    )
    with pytest.raises(M4ARunManifestError, match="simulated manifest failure"):
        run_visual_render_pipeline(
            _inventory(),
            output,
            run_id=_RUN_ID,
            render_callback=_prepared,
            timestamp_factory=_clock,
            operational_manifest=_operational_manifest(),
        )
    assert not output.exists()
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize("drift", ("seed", "selection"))
def test_pipeline_rejects_manifest_plan_drift_before_publication(
    tmp_path: Path, drift: str
) -> None:
    inventory = _inventory()
    output = tmp_path / f"manifest-{drift}-drift"
    manifest = (
        _operational_manifest(render_seed=271829)
        if drift == "seed"
        else _operational_manifest()
    )
    selected = inventory.packet_ids if drift == "seed" else (inventory.packet_ids[0],)
    with pytest.raises(VisualRenderPipelineError, match="render-plan binding drift"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=_prepared,
            selected_packet_ids=selected,
            timestamp_factory=_clock,
            operational_manifest=manifest,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "source_collection",
    (SourceCollection.M3A_DEVELOPMENT, SourceCollection.M3C_EXTERNAL),
)
def test_pipeline_rejects_same_collection_different_source_identity_digest(
    tmp_path: Path,
    source_collection: SourceCollection,
) -> None:
    inventory = _inventory(
        source_collection=source_collection,
        source_identity_digest=_digest("inventory-canonical-source"),
    )
    drifted_source = (
        M4AOperationalSourceIdentityV1.for_m3c(
            _digest("accepted-m3c-source-set"),
            _digest("accepted-m3c-candidate-pool"),
            canonical_source_identity_digest=_digest("different-canonical-source"),
        )
        if source_collection is SourceCollection.M3C_EXTERNAL
        else M4AOperationalSourceIdentityV1.for_m3a(
            _digest("accepted-m3a-dataset"),
            canonical_source_identity_digest=_digest("different-canonical-source"),
        )
    )
    manifest = _operational_manifest(
        inventory=inventory,
        source_identity=drifted_source,
    )
    output = tmp_path / f"manifest-{source_collection.value}-source-drift"

    with pytest.raises(VisualRenderPipelineError, match="source_identity_digest"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=_prepared,
            timestamp_factory=_clock,
            operational_manifest=manifest,
        )
    assert not output.exists()


def test_direct_ledger_load_rejects_junction_or_reparse_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentguard.vision_data.serialization as visual_serialization

    output = tmp_path / "render-ledger-link-guard"
    run_visual_render_pipeline(
        _inventory(),
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        timestamp_factory=_clock,
    )
    real_guard = visual_serialization._is_link_or_junction
    monkeypatch.setattr(
        visual_serialization,
        "_is_link_or_junction",
        lambda path: Path(path) == output or real_guard(Path(path)),
    )
    with pytest.raises(RenderLedgerError, match="regular bundle"):
        load_render_ledger(output)


def test_crash_after_packet_save_recovers_without_duplicate_render(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    job = inventory.jobs[0]
    output = tmp_path / "render"

    def crash_after_save(selected_job: VisualPacketJobV1) -> _Prepared:
        prepared = _prepared(selected_job)
        save_visual_packet(
            prepared.packet,
            prepared.images,
            output / "packets" / selected_job.packet_id,
        )
        raise KeyboardInterrupt("simulated process loss")

    with pytest.raises(KeyboardInterrupt, match="simulated process loss"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=crash_after_save,
            selected_packet_ids=(job.packet_id,),
            timestamp_factory=_clock,
        )
    assert _latest_state(output, job.packet_id) is RenderLedgerState.RENDERING

    callback_count = 0

    def must_not_render(selected_job: VisualPacketJobV1) -> _Prepared:
        nonlocal callback_count
        callback_count += 1
        return _prepared(selected_job)

    recovered = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=must_not_render,
        resume=True,
        selected_packet_ids=(job.packet_id,),
        timestamp_factory=_clock,
    )
    assert callback_count == 0
    assert recovered.recovered_packet_ids == (job.packet_id,)
    assert recovered.rendered_packet_ids == ()
    assert recovered.packet_count == 1
    assert not recovered.complete
    assert _latest_state(output, job.packet_id) is RenderLedgerState.COMPLETE


def test_process_loss_partial_staging_is_safely_removed_and_rerendered(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    job = inventory.jobs[0]
    output = tmp_path / "render"
    staging = output / "packets" / f".{job.packet_id}.staging-deadbeef"

    def crash_in_transaction(_: VisualPacketJobV1) -> _Prepared:
        (staging / "images").mkdir(parents=True)
        (staging / "images" / "partial.npy").write_bytes(b"partial")
        raise KeyboardInterrupt("simulated process termination")

    with pytest.raises(KeyboardInterrupt, match="process termination"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=crash_in_transaction,
            selected_packet_ids=(job.packet_id,),
            timestamp_factory=_clock,
        )
    assert staging.is_dir()
    assert _latest_state(output, job.packet_id) is RenderLedgerState.RENDERING

    resumed = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        resume=True,
        selected_packet_ids=(job.packet_id,),
        timestamp_factory=_clock,
    )
    assert not staging.exists()
    assert resumed.rendered_packet_ids == (job.packet_id,)
    assert resumed.recovered_packet_ids == (job.packet_id,)
    assert _latest_state(output, job.packet_id) is RenderLedgerState.COMPLETE


def test_execution_error_requires_explicit_retry_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    job = inventory.jobs[0]
    output = tmp_path / "render"

    def fail_operationally(_: VisualPacketJobV1) -> _Prepared:
        raise RuntimeError("renderer device unavailable")

    with pytest.raises(VisualRenderExecutionError, match="execution failed"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=fail_operationally,
            selected_packet_ids=(job.packet_id,),
            timestamp_factory=_clock,
        )
    ledger = load_render_ledger(output)
    first = tuple(entry for entry in ledger.entries if entry.packet_id == job.packet_id)
    assert len(first) == 1
    assert first[0].state is RenderLedgerState.EXECUTION_ERROR
    assert first[0].retry_eligible
    assert first[0].error_type == "RuntimeError"

    without_retry = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=fail_operationally,
        resume=True,
        selected_packet_ids=(job.packet_id,),
        timestamp_factory=_clock,
    )
    assert without_retry.rendered_packet_ids == ()
    assert _latest_state(output, job.packet_id) is RenderLedgerState.EXECUTION_ERROR

    retried = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        resume=True,
        retry_execution_errors=True,
        selected_packet_ids=(job.packet_id,),
        timestamp_factory=_clock,
    )
    attempts = tuple(
        entry for entry in retried.ledger.entries if entry.packet_id == job.packet_id
    )
    assert tuple(entry.state for entry in attempts) == (
        RenderLedgerState.EXECUTION_ERROR,
        RenderLedgerState.COMPLETE,
    )
    assert attempts[0].error_message == "renderer device unavailable"
    assert retried.retried_packet_ids == (job.packet_id,)


def test_invalid_state_contract_is_nonretryable(tmp_path: Path) -> None:
    inventory = _inventory()
    job = inventory.jobs[0]
    output = tmp_path / "render"

    def invalid_state(_: VisualPacketJobV1) -> _Prepared:
        raise InvalidVisualRenderPacketError("restored state changed during render")

    with pytest.raises(InvalidVisualRenderPacketError, match="invalid rendered"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=invalid_state,
            selected_packet_ids=(job.packet_id,),
            timestamp_factory=_clock,
        )
    ledger = load_render_ledger(output)
    entry = tuple(item for item in ledger.entries if item.packet_id == job.packet_id)[
        -1
    ]
    assert entry.state is RenderLedgerState.INVALID
    assert not entry.retry_eligible
    assert entry.error_type == "InvalidVisualRenderPacketError"
    assert "restored state changed" in (entry.error_message or "")

    resumed = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        resume=True,
        retry_execution_errors=True,
        selected_packet_ids=(job.packet_id,),
        timestamp_factory=_clock,
    )
    assert resumed.rendered_packet_ids == ()
    assert _latest_state(output, job.packet_id) is RenderLedgerState.INVALID


@pytest.mark.parametrize(
    "mutation",
    ("source", "state", "camera", "domain"),
)
def test_packet_source_state_camera_and_domain_drift_become_invalid(
    tmp_path: Path,
    mutation: str,
) -> None:
    inventory = _inventory()
    job = inventory.jobs[0]
    output = tmp_path / mutation

    def render_drifted(selected_job: VisualPacketJobV1) -> _Prepared:
        prepared = _prepared(selected_job)
        changes: dict[str, object]
        if mutation == "source":
            changes = {"source_trajectory_id": "trajectory-drift"}
        elif mutation == "state":
            changes = {"expected_state_digest": _digest("state-drift")}
        elif mutation == "camera":
            changes = {"camera_rig_digest": _digest("camera-rig-drift")}
        else:
            changes = {"render_domain_digest": _digest("domain-drift")}
        return replace(prepared, packet=replace(prepared.packet, **changes))

    with pytest.raises(InvalidVisualRenderPacketError):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=render_drifted,
            selected_packet_ids=(job.packet_id,),
            timestamp_factory=_clock,
        )
    assert _latest_state(output, job.packet_id) is RenderLedgerState.INVALID


def test_existing_packet_pixel_tamper_is_rejected_on_zero_work_resume(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    output = tmp_path / "render"
    completed = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        timestamp_factory=_clock,
    )
    packet = completed.packets[0]
    bundle = completed.packet_bundle_roots[0]
    image_path = bundle.joinpath(*PurePosixPath(packet.views[0].image_reference).parts)
    payload = bytearray(image_path.read_bytes())
    payload[-1] ^= 0x01
    image_path.write_bytes(payload)

    with pytest.raises(VisualSerializationError, match="digest"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=_prepared,
            resume=True,
            timestamp_factory=_clock,
        )


def test_orphan_packet_bundle_blocks_zero_work_proof(tmp_path: Path) -> None:
    inventory = _inventory()
    output = tmp_path / "render"
    run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        timestamp_factory=_clock,
    )
    orphan_id = f"vop-sha256-{'f' * 64}"
    assert orphan_id not in inventory.packet_ids
    (output / "packets" / orphan_id).mkdir()

    with pytest.raises(VisualRenderPipelineError, match="unknown packet directory"):
        run_visual_render_pipeline(
            inventory,
            output,
            run_id=_RUN_ID,
            render_callback=_prepared,
            resume=True,
            timestamp_factory=_clock,
        )


def test_partial_domain_run_resumes_to_full_then_proves_zero_work(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    output = tmp_path / "render"
    calls: list[str] = []

    def render(job: VisualPacketJobV1) -> _Prepared:
        calls.append(job.packet_id)
        return _prepared(job)

    first = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=render,
        selected_packet_ids=(inventory.packet_ids[0],),
        timestamp_factory=_clock,
    )
    assert first.planned_packet_count == 3
    assert first.packet_count == 1
    assert first.image_count == 3
    assert not first.complete
    assert first.zero_work_proof is None
    assert tuple(
        _latest_state(output, packet_id) for packet_id in inventory.packet_ids
    ) == (
        RenderLedgerState.COMPLETE,
        RenderLedgerState.PENDING,
        RenderLedgerState.PENDING,
    )

    finished = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=render,
        resume=True,
        timestamp_factory=_clock,
    )
    assert finished.complete
    assert finished.packet_count == 3
    assert finished.image_count == 9
    assert calls == list(inventory.packet_ids)
    before = (output / "render-ledger.json").read_bytes()

    def forbidden(_: VisualPacketJobV1) -> _Prepared:
        raise AssertionError("zero-work resume invoked renderer")

    zero_work = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=forbidden,
        resume=True,
        timestamp_factory=_clock,
    )
    assert zero_work.complete
    assert zero_work.zero_work_proof is not None
    assert zero_work.rendered_packet_ids == ()
    assert zero_work.recovered_packet_ids == ()
    assert zero_work.retried_packet_ids == ()
    assert (output / "render-ledger.json").read_bytes() == before


def test_camera_shift_packet_uses_seed_resolved_calibration(tmp_path: Path) -> None:
    inventory = _inventory()
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    shifted = inventory.jobs[1]
    assert shifted.render_domain_id == MILD_CAMERA_DOMAIN_ID
    assert shifted.camera_configuration_digests != tuple(
        camera.camera_configuration_digest for camera in rig.cameras
    )

    result = run_visual_render_pipeline(
        inventory,
        tmp_path / "render",
        run_id=_RUN_ID,
        render_callback=_prepared,
        selected_packet_ids=(shifted.packet_id,),
        timestamp_factory=_clock,
    )
    assert result.packets[0].packet_id == shifted.packet_id
    assert (
        tuple(view.camera_configuration_digest for view in result.packets[0].views)
        == shifted.camera_configuration_digests
    )


def test_pipeline_records_exact_packet_timings_and_zero_work_reuse(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    manifest = _operational_manifest(inventory=inventory)
    clock_values = iter((100, 110, 200, 220, 300, 330))
    output = tmp_path / "timed-render"
    first = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=_prepared,
        operational_manifest=manifest,
        monotonic_ns_factory=lambda: next(clock_values),
    )
    assert first.rendered_packet_ids == inventory.packet_ids
    assert first.reused_packet_ids == ()
    assert first.packet_render_durations_ns == tuple(
        zip(inventory.packet_ids, (10, 20, 30), strict=True)
    )

    resumed = run_visual_render_pipeline(
        inventory,
        output,
        run_id=_RUN_ID,
        render_callback=lambda _job: pytest.fail("zero-work resume rendered"),
        operational_manifest=manifest,
        resume=True,
        monotonic_ns_factory=lambda: pytest.fail("zero-work resume read render clock"),
    )
    assert resumed.rendered_packet_ids == ()
    assert resumed.reused_packet_ids == inventory.packet_ids
    assert resumed.packet_render_durations_ns == ()
    assert resumed.zero_work_proof is not None
