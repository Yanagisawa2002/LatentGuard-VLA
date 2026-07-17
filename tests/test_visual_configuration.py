from __future__ import annotations

import copy
from pathlib import Path

import pytest

from latentguard.vision_data import (
    CANONICAL_DOMAIN_ID,
    MILD_CAMERA_DOMAIN_ID,
    MILD_LIGHTING_DOMAIN_ID,
    PICKCUBE_CAMERA_IDS,
    RENDER_SEED_DERIVATION,
    STRONG_CAMERA_DOMAIN_ID,
    STRONG_LIGHTING_DOMAIN_ID,
    CameraPoseV1,
    PickCubeMultiViewRigV1,
    RenderDomainConfigurationV1,
    SourceCollection,
    VisionConfigurationError,
    VisionDataValidationError,
    VisualDatasetSplit,
    assigned_render_domain_ids,
    load_camera_rig_configuration,
    load_render_domain_configuration,
)
from latentguard.vision_data.configuration import load_strict_json_mapping

_CONFIG_ROOT = Path("configs/vision/m4a")


def _rig_mapping() -> dict[str, object]:
    return copy.deepcopy(
        load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json").as_mapping()
    )


def _domain_mapping() -> dict[str, object]:
    return copy.deepcopy(
        load_render_domain_configuration(
            _CONFIG_ROOT / "render-domains-v1.json"
        ).as_mapping()
    )


def test_checked_in_visual_configuration_is_strict_and_complete() -> None:
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")

    assert tuple(camera.camera_id for camera in rig.cameras) == PICKCUBE_CAMERA_IDS
    assert all((camera.width, camera.height) == (224, 224) for camera in rig.cameras)
    assert tuple(domain.domain_id for domain in domains.domains) == (
        CANONICAL_DOMAIN_ID,
        MILD_CAMERA_DOMAIN_ID,
        MILD_LIGHTING_DOMAIN_ID,
        STRONG_CAMERA_DOMAIN_ID,
        STRONG_LIGHTING_DOMAIN_ID,
    )
    assert rig.content_digest.startswith("sha256:")
    assert domains.content_digest.startswith("sha256:")


def test_configuration_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    rig = _rig_mapping()
    rig["unexpected"] = True
    with pytest.raises(VisionConfigurationError, match="unknown"):
        PickCubeMultiViewRigV1.from_mapping(rig)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}',
        encoding="utf-8",
    )
    with pytest.raises(VisionConfigurationError, match="duplicate.*field"):
        load_strict_json_mapping(duplicate)


def test_configuration_rejects_nonfinite_pose_planes_and_resolution() -> None:
    with pytest.raises(VisionDataValidationError, match="finite"):
        CameraPoseV1(
            position=(float("nan"), 0.0, 0.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    with pytest.raises(VisionDataValidationError, match="normalized"):
        CameraPoseV1(
            position=(0.0, 0.0, 0.0),
            quaternion_wxyz=(2.0, 0.0, 0.0, 0.0),
        )

    invalid_planes = _rig_mapping()
    cameras = invalid_planes["cameras"]
    assert isinstance(cameras, list)
    assert isinstance(cameras[0], dict)
    cameras[0]["near"] = cameras[0]["far"]
    with pytest.raises(VisionDataValidationError, match="near/far"):
        PickCubeMultiViewRigV1.from_mapping(invalid_planes)

    invalid_resolution = _rig_mapping()
    cameras = invalid_resolution["cameras"]
    assert isinstance(cameras, list)
    assert isinstance(cameras[0], dict)
    cameras[0]["width"] = 225
    with pytest.raises(VisionDataValidationError, match="224 x 224"):
        PickCubeMultiViewRigV1.from_mapping(invalid_resolution)


def test_configuration_rejects_fov_and_intrinsics_projection_drift() -> None:
    changed_fov = _rig_mapping()
    cameras = changed_fov["cameras"]
    assert isinstance(cameras, list) and isinstance(cameras[0], dict)
    cameras[0]["fov_y_degrees"] = 120.0
    with pytest.raises(VisionDataValidationError, match="vertical FOV and f_y"):
        PickCubeMultiViewRigV1.from_mapping(changed_fov)

    changed_fy = _rig_mapping()
    cameras = changed_fy["cameras"]
    assert isinstance(cameras, list) and isinstance(cameras[0], dict)
    intrinsics = cameras[0]["intrinsics"]
    assert isinstance(intrinsics, list) and isinstance(intrinsics[1], list)
    intrinsics[1][1] = 194.5
    with pytest.raises(VisionDataValidationError, match="vertical FOV and f_y"):
        PickCubeMultiViewRigV1.from_mapping(changed_fy)

    changed_fx = _rig_mapping()
    cameras = changed_fx["cameras"]
    assert isinstance(cameras, list) and isinstance(cameras[0], dict)
    intrinsics = cameras[0]["intrinsics"]
    assert isinstance(intrinsics, list) and isinstance(intrinsics[0], list)
    intrinsics[0][0] = 194.5
    with pytest.raises(VisionDataValidationError, match="width/height aspect"):
        PickCubeMultiViewRigV1.from_mapping(changed_fx)


def test_configuration_rejects_duplicate_cameras_and_invalid_domain_ranges() -> None:
    duplicate_camera = _rig_mapping()
    cameras = duplicate_camera["cameras"]
    assert isinstance(cameras, list)
    assert isinstance(cameras[0], dict)
    assert isinstance(cameras[1], dict)
    cameras[1]["camera_id"] = cameras[0]["camera_id"]
    with pytest.raises(VisionDataValidationError, match="ordered three-view"):
        PickCubeMultiViewRigV1.from_mapping(duplicate_camera)

    invalid_range = _domain_mapping()
    domains = invalid_range["domains"]
    assert isinstance(domains, list)
    assert isinstance(domains[1], dict)
    domains[1]["camera_translation_min"] = [1.0, 0.0, 0.0]
    domains[1]["camera_translation_max"] = [0.0, 0.0, 0.0]
    with pytest.raises(VisionDataValidationError, match="minimum exceeds maximum"):
        RenderDomainConfigurationV1.from_mapping(invalid_range)


def test_configuration_rejects_domain_scope_drift() -> None:
    changed = _domain_mapping()
    domains = changed["domains"]
    assert isinstance(domains, list)
    assert isinstance(domains[1], dict)
    domains[1]["allowed_splits"] = ["train", "validation", "test"]
    with pytest.raises(VisionDataValidationError, match="authorization scope"):
        RenderDomainConfigurationV1.from_mapping(changed)


def test_configuration_rejects_unimplemented_seed_derivation_semantic() -> None:
    changed = _domain_mapping()
    changed["seed_derivation"] = "claimed-but-unimplemented-v999"

    with pytest.raises(VisionDataValidationError, match="seed-derivation semantic"):
        RenderDomainConfigurationV1.from_mapping(changed)

    checked_in = load_render_domain_configuration(
        _CONFIG_ROOT / "render-domains-v1.json"
    )
    assert checked_in.seed_derivation == RENDER_SEED_DERIVATION


@pytest.mark.parametrize(
    ("collection", "split", "expected"),
    [
        (
            SourceCollection.M3A_DEVELOPMENT,
            VisualDatasetSplit.TRAIN,
            (
                CANONICAL_DOMAIN_ID,
                MILD_CAMERA_DOMAIN_ID,
                MILD_LIGHTING_DOMAIN_ID,
            ),
        ),
        (
            SourceCollection.M3A_DEVELOPMENT,
            VisualDatasetSplit.VALIDATION,
            (
                CANONICAL_DOMAIN_ID,
                MILD_CAMERA_DOMAIN_ID,
                MILD_LIGHTING_DOMAIN_ID,
            ),
        ),
        (
            SourceCollection.M3A_DEVELOPMENT,
            VisualDatasetSplit.TEST,
            (
                CANONICAL_DOMAIN_ID,
                STRONG_CAMERA_DOMAIN_ID,
                STRONG_LIGHTING_DOMAIN_ID,
            ),
        ),
        (
            SourceCollection.M3C_EXTERNAL,
            VisualDatasetSplit.EXTERNAL,
            (
                CANONICAL_DOMAIN_ID,
                STRONG_CAMERA_DOMAIN_ID,
                STRONG_LIGHTING_DOMAIN_ID,
            ),
        ),
    ],
)
def test_domain_assignment_is_fixed_before_rendering(
    collection: SourceCollection,
    split: VisualDatasetSplit,
    expected: tuple[str, str, str],
) -> None:
    assert assigned_render_domain_ids(collection, split) == expected


def test_m3c_is_external_and_m3a_external_is_rejected() -> None:
    with pytest.raises(VisionConfigurationError, match="external split"):
        assigned_render_domain_ids(
            SourceCollection.M3C_EXTERNAL, VisualDatasetSplit.TEST
        )
    with pytest.raises(VisionConfigurationError, match="cannot use external"):
        assigned_render_domain_ids(
            SourceCollection.M3A_DEVELOPMENT, VisualDatasetSplit.EXTERNAL
        )
