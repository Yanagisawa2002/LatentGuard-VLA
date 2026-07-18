from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import latentguard.vision_data.serialization as visual_serialization
from latentguard.vision_data import (
    SourceCollection,
    VisualDatasetSplit,
    VisualObservationPacketV1,
    VisualTaskProjectionV1,
    VisualViewRecordV1,
    load_camera_rig_configuration,
    load_render_domain_configuration,
    load_visual_image,
    load_visual_packet,
    prepare_npy_image,
    save_visual_packet,
)
from latentguard.vision_data.serialization import (
    VisualSerializationError,
    serialize_npy_image,
    validate_image_reference,
)

_CONFIG_ROOT = Path("configs/vision/m4a")


def _digest(label: str) -> str:
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


def _packet_and_images() -> tuple[
    VisualObservationPacketV1, Mapping[str, NDArray[Any]]
]:
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")
    domain = domains.domain("canonical")
    task = _task_projection()
    state_digest = _digest("runtime-state")
    images: dict[str, NDArray[Any]] = {}
    views: list[VisualViewRecordV1] = []
    for index, camera in enumerate(rig.cameras):
        image = np.full((224, 224, 3), index * 31 + 7, dtype=np.uint8)
        prepared = prepare_npy_image(image)
        reference = f"images/packet/{camera.camera_id}.npy"
        images[reference] = prepared.image
        views.append(
            VisualViewRecordV1(
                camera_id=camera.camera_id,
                image_reference=reference,
                pixel_sha256=prepared.pixel_sha256,
                npy_sha256=prepared.npy_sha256,
                dtype="uint8",
                shape=(224, 224, 3),
                intrinsics=camera.intrinsics,
                intrinsics_dtype="float64",
                extrinsics=camera.extrinsics,
                extrinsics_dtype="float64",
                runtime_extrinsics_digest=_digest(
                    f"runtime-extrinsics-{camera.camera_id}"
                ),
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
        )
    return (
        VisualObservationPacketV1(
            source_collection=SourceCollection.M3A_DEVELOPMENT,
            source_trajectory_id="trajectory-0",
            anchor_id="anchor-0",
            split=VisualDatasetSplit.TRAIN,
            split_group_id="split-group-0",
            state_reference_id="states/trajectory-0/anchor-0.npz",
            expected_state_digest=_digest("source-state"),
            verifier_state_semantic="PickCubeVerifierStateV1",
            verifier_state_digest=_digest("verifier-state"),
            verifier_state_component_count=38,
            verifier_state_maximum_absolute_error=0.0,
            elapsed_simulation_steps_before=0,
            elapsed_simulation_steps_after=0,
            environment_close_passed=True,
            camera_rig_id=rig.rig_id,
            camera_rig_digest=rig.content_digest,
            render_domain_id=domain.domain_id,
            render_domain_digest=domain.content_digest,
            render_seed=7,
            views=tuple(views),
            visual_compatibility_identity=_digest("visual-compatibility"),
            pickcube_compatibility_identity=_digest("pickcube-compatibility"),
            renderer_semantic_version="sapien-renderer-v1",
        ),
        images,
    )


def _first_image_path(bundle: Path, packet: VisualObservationPacketV1) -> Path:
    return bundle.joinpath(*PurePosixPath(packet.views[0].image_reference).parts)


def test_exact_npy_round_trip_always_disables_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, images = _packet_and_images()
    bundle = tmp_path / "packet"
    save_visual_packet(packet, images, bundle)
    real_load = np.load
    observed_allow_pickle: list[object] = []

    def guarded_load(*args: object, **kwargs: object) -> object:
        observed_allow_pickle.append(kwargs.get("allow_pickle"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(visual_serialization.np, "load", guarded_load)
    reloaded = load_visual_packet(bundle)
    assert isinstance(reloaded, VisualObservationPacketV1)
    assert reloaded.content_digest == packet.content_digest
    for reference, expected in images.items():
        observed = load_visual_image(bundle, reference)
        assert np.array_equal(observed, expected)
    assert observed_allow_pickle
    assert set(observed_allow_pickle) == {False}

    with pytest.raises(VisualSerializationError, match="uint8"):
        serialize_npy_image(np.empty((224, 224, 3), dtype=object))


@pytest.mark.parametrize(
    "reference",
    (
        "../escape.npy",
        "images/../escape.npy",
        "/absolute.npy",
        "C:/absolute.npy",
        "images\\windows.npy",
    ),
)
def test_path_traversal_and_absolute_references_are_rejected(reference: str) -> None:
    with pytest.raises(VisualSerializationError):
        validate_image_reference(reference)


def test_link_guard_and_hard_linked_image_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, images = _packet_and_images()
    bundle = tmp_path / "packet"
    save_visual_packet(packet, images, bundle)
    image_path = _first_image_path(bundle, packet)
    real_guard = visual_serialization._is_link_or_junction
    monkeypatch.setattr(
        visual_serialization,
        "_is_link_or_junction",
        lambda path: Path(path) == image_path or real_guard(Path(path)),
    )
    with pytest.raises(VisualSerializationError, match="single-linked image"):
        load_visual_packet(bundle)
    monkeypatch.setattr(visual_serialization, "_is_link_or_junction", real_guard)

    outside = tmp_path / "outside.npy"
    outside.write_bytes(image_path.read_bytes())
    image_path.unlink()
    os.link(outside, image_path)
    with pytest.raises(VisualSerializationError, match="single-linked image"):
        load_visual_packet(bundle)


def test_inventory_orphan_duplicate_and_image_tamper_are_rejected(
    tmp_path: Path,
) -> None:
    packet, images = _packet_and_images()

    orphan_bundle = tmp_path / "orphan"
    save_visual_packet(packet, images, orphan_bundle)
    (orphan_bundle / "unexpected.bin").write_bytes(b"orphan")
    with pytest.raises(VisualSerializationError, match="file inventory differs"):
        load_visual_packet(orphan_bundle)

    duplicate_bundle = tmp_path / "duplicate"
    save_visual_packet(packet, images, duplicate_bundle)
    manifest_path = duplicate_bundle / "packet.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["image_inventory"].append(dict(manifest["image_inventory"][0]))
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(VisualSerializationError, match="duplicate image reference"):
        load_visual_packet(duplicate_bundle)

    tampered_bundle = tmp_path / "tampered"
    save_visual_packet(packet, images, tampered_bundle)
    image_path = _first_image_path(tampered_bundle, packet)
    payload = bytearray(image_path.read_bytes())
    payload[-1] ^= 0x01
    image_path.write_bytes(payload)
    with pytest.raises(VisualSerializationError, match="NPY file digest mismatch"):
        load_visual_packet(tampered_bundle)


def test_same_pixels_with_alternate_npy_encoding_are_rejected(tmp_path: Path) -> None:
    packet, images = _packet_and_images()
    bundle = tmp_path / "alternate-npy"
    save_visual_packet(packet, images, bundle)
    reference = packet.views[0].image_reference
    image_path = _first_image_path(bundle, packet)
    original_payload = image_path.read_bytes()

    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        images[reference],
        version=(2, 0),
        allow_pickle=False,
    )
    alternate_payload = stream.getvalue()
    assert alternate_payload != original_payload
    assert np.array_equal(
        np.load(io.BytesIO(alternate_payload), allow_pickle=False),
        images[reference],
    )
    image_path.write_bytes(alternate_payload)

    manifest_path = bundle / "packet.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["image_inventory"]:
        if item["path"] == reference:
            item["npy_sha256"] = (
                "sha256:" + hashlib.sha256(alternate_payload).hexdigest()
            )
            break
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        VisualSerializationError,
        match="authoritative NPY digest differs from view record",
    ):
        load_visual_packet(bundle)


def test_interrupted_transaction_leaves_no_partial_packet_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, images = _packet_and_images()
    bundle = tmp_path / "packet"
    real_write = visual_serialization._write_new_bytes
    write_count = 0

    def interrupt_manifest(path: Path, payload: bytes) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 4:
            raise KeyboardInterrupt("simulated serialization interruption")
        real_write(path, payload)

    monkeypatch.setattr(
        visual_serialization,
        "_write_new_bytes",
        interrupt_manifest,
    )
    with pytest.raises(KeyboardInterrupt, match="serialization interruption"):
        save_visual_packet(packet, images, bundle)
    assert not bundle.exists()
    assert tuple(tmp_path.glob(".packet.staging-*")) == ()

    monkeypatch.setattr(visual_serialization, "_write_new_bytes", real_write)
    assert save_visual_packet(packet, images, bundle) == bundle
    assert load_visual_packet(bundle).content_digest == packet.content_digest
