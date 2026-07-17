"""Fail-closed validation for simulator-independent M4A visual contracts."""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import NoReturn, cast

from latentguard.models import FailureEvent
from latentguard.validation import DataValidationError, validate_failure_event
from latentguard.vision_data.cameras import (
    CAMERA_RIG_SCHEMA_VERSION,
    CAMERA_SCHEMA_VERSION,
    M4A_IMAGE_HEIGHT,
    M4A_IMAGE_WIDTH,
    M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE,
    PICKCUBE_CAMERA_IDS,
    PICKCUBE_MULTIVIEW_RIG_ID,
    PICKCUBE_MULTIVIEW_RIG_SEMANTIC_VERSION,
    CameraConfigurationV1,
    CameraPoseV1,
    PickCubeMultiViewRigV1,
    expected_pinhole_focal_lengths,
)
from latentguard.vision_data.domains import (
    CANONICAL_DOMAIN_ID,
    M4A_COLOR_FORMAT,
    M4A_IMAGE_RESOLUTION,
    MILD_CAMERA_DOMAIN_ID,
    MILD_LIGHTING_DOMAIN_ID,
    RENDER_DOMAIN_CONFIGURATION_ID,
    RENDER_DOMAIN_CONFIGURATION_SCHEMA_VERSION,
    RENDER_DOMAIN_IDS,
    RENDER_DOMAIN_SCHEMA_VERSION,
    RENDER_SEED_DERIVATION,
    STRONG_CAMERA_DOMAIN_ID,
    STRONG_LIGHTING_DOMAIN_ID,
    LightingConfigurationV1,
    RenderDomainConfigurationV1,
    RenderDomainV1,
    assigned_render_domain_ids,
)
from latentguard.vision_data.models import (
    PICKCUBE_CANONICAL_TASK_TEXT,
    PICKCUBE_VISUAL_TASK_ID,
    VISION_DATA_SCHEMA_VERSION,
    VISUAL_MODEL_INPUT_CONTRACT_SCHEMA_VERSION,
    VISUAL_TASK_PROJECTION_SCHEMA_VERSION,
    SourceCollection,
    VisualActionVerifierSampleV1,
    VisualCandidateBindingV1,
    VisualDatasetSplit,
    VisualModelInputContractV1,
    VisualTaskProjectionV1,
)

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_PACKET_ID = re.compile(r"^vop-sha256-[0-9a-f]{64}$")


class VisionDataValidationError(ValueError):
    """Raised when a visual model violates the fixed M4A contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisionDataValidationError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail(context, "expected canonical text")
    return value


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        _fail(context, "expected sha256:<64 lowercase hexadecimal characters>")
    return value


def _finite(value: object, context: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        _fail(context, "expected finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        _fail(
            context,
            "expected finite non-negative number"
            if nonnegative
            else "expected finite number",
        )
    return result


def _matrix_finite(
    value: tuple[tuple[float, ...], ...], shape: tuple[int, int], context: str
) -> None:
    if len(value) != shape[0] or any(len(row) != shape[1] for row in value):
        _fail(context, f"expected shape {shape}")
    for row_index, row in enumerate(value):
        for column_index, item in enumerate(row):
            _finite(item, f"{context}[{row_index}][{column_index}]")


def _safe_reference(value: object, context: str, *, npy: bool = False) -> str:
    text = _text(value, context)
    pure = PurePosixPath(text)
    if (
        "\\" in text
        or pure.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
        or (npy and pure.suffix != ".npy")
    ):
        _fail(context, "expected safe relative POSIX reference")
    return text


def validate_task_projection(value: VisualTaskProjectionV1) -> None:
    """Validate complete finite PickCube public task facts."""

    if value.schema_version != VISUAL_TASK_PROJECTION_SCHEMA_VERSION:
        _fail("VisualTaskProjectionV1.schema_version", "unsupported version")
    for field in ("success", "is_obj_placed", "is_robot_static", "is_grasped"):
        if type(getattr(value, field)) is not bool:
            _fail(f"VisualTaskProjectionV1.{field}", "expected boolean")
    _finite(value.cube_center_z, "VisualTaskProjectionV1.cube_center_z")
    for field in ("cube_to_goal_distance", "tcp_to_cube_distance"):
        _finite(
            getattr(value, field),
            f"VisualTaskProjectionV1.{field}",
            nonnegative=True,
        )


def validate_camera_pose(value: CameraPoseV1) -> None:
    """Validate a finite normalized world-frame pose."""

    if len(value.position) != 3 or len(value.quaternion_wxyz) != 4:
        _fail("CameraPoseV1", "expected position[3] and quaternion_wxyz[4]")
    for index, item in enumerate(value.position):
        _finite(item, f"CameraPoseV1.position[{index}]")
    quaternion = tuple(
        _finite(item, f"CameraPoseV1.quaternion_wxyz[{index}]")
        for index, item in enumerate(value.quaternion_wxyz)
    )
    norm = math.sqrt(sum(item * item for item in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        _fail("CameraPoseV1.quaternion_wxyz", "must be normalized")


def validate_camera_configuration(value: CameraConfigurationV1) -> None:
    """Validate one exact M4A camera configuration."""

    if value.schema_version != CAMERA_SCHEMA_VERSION:
        _fail("CameraConfigurationV1.schema_version", "unsupported version")
    if value.camera_id not in PICKCUBE_CAMERA_IDS:
        _fail("CameraConfigurationV1.camera_id", "unsupported camera")
    if value.width != M4A_IMAGE_WIDTH or value.height != M4A_IMAGE_HEIGHT:
        _fail("CameraConfigurationV1.resolution", "M4A requires 224 x 224")
    near = _finite(value.near, "CameraConfigurationV1.near", nonnegative=True)
    far = _finite(value.far, "CameraConfigurationV1.far", nonnegative=True)
    if near <= 0.0 or far <= near:
        _fail("CameraConfigurationV1.near/far", "require 0 < near < far")
    fov = _finite(value.fov_y_degrees, "CameraConfigurationV1.fov_y_degrees")
    if not 0.0 < fov < 180.0:
        _fail("CameraConfigurationV1.fov_y_degrees", "must be in (0, 180)")
    if not isinstance(value.world_pose, CameraPoseV1):
        _fail("CameraConfigurationV1.world_pose", "expected CameraPoseV1")
    _matrix_finite(value.intrinsics, (3, 3), "CameraConfigurationV1.intrinsics")
    _matrix_finite(value.extrinsics, (4, 4), "CameraConfigurationV1.extrinsics")
    if value.intrinsics[0][0] <= 0.0 or value.intrinsics[1][1] <= 0.0:
        _fail("CameraConfigurationV1.intrinsics", "focal lengths must be positive")
    expected_fx, expected_fy = expected_pinhole_focal_lengths(
        value.width, value.height, fov
    )
    if not math.isclose(
        value.intrinsics[1][1],
        expected_fy,
        rel_tol=0.0,
        abs_tol=M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE,
    ):
        _fail(
            "CameraConfigurationV1.fov_y_degrees/intrinsics",
            "vertical FOV and f_y describe different projections",
        )
    if not math.isclose(
        value.intrinsics[0][0],
        expected_fx,
        rel_tol=0.0,
        abs_tol=M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE,
    ):
        _fail(
            "CameraConfigurationV1.intrinsics",
            "f_x does not preserve the declared width/height aspect",
        )
    if value.intrinsics[2] != (0.0, 0.0, 1.0):
        _fail("CameraConfigurationV1.intrinsics", "last row must be [0,0,1]")
    if value.extrinsics[3] != (0.0, 0.0, 0.0, 1.0):
        _fail("CameraConfigurationV1.extrinsics", "last row must be [0,0,0,1]")
    _digest(value.content_digest, "CameraConfigurationV1.content_digest")


def validate_camera_rig(value: PickCubeMultiViewRigV1) -> None:
    """Validate the one fixed ordered three-view M4A camera rig."""

    if value.schema_version != CAMERA_RIG_SCHEMA_VERSION:
        _fail("PickCubeMultiViewRigV1.schema_version", "unsupported version")
    if value.rig_id != PICKCUBE_MULTIVIEW_RIG_ID:
        _fail("PickCubeMultiViewRigV1.rig_id", "unsupported rig")
    if value.semantic_version != PICKCUBE_MULTIVIEW_RIG_SEMANTIC_VERSION:
        _fail("PickCubeMultiViewRigV1.semantic_version", "unsupported version")
    if tuple(camera.camera_id for camera in value.cameras) != PICKCUBE_CAMERA_IDS:
        _fail("PickCubeMultiViewRigV1.cameras", "expected fixed ordered three-view rig")
    if any(not isinstance(camera, CameraConfigurationV1) for camera in value.cameras):
        _fail("PickCubeMultiViewRigV1.cameras", "invalid camera model")
    digests = tuple(camera.content_digest for camera in value.cameras)
    if len(set(digests)) != len(digests):
        _fail("PickCubeMultiViewRigV1.cameras", "duplicate configurations")
    _digest(value.content_digest, "PickCubeMultiViewRigV1.content_digest")


def validate_lighting_configuration(value: LightingConfigurationV1) -> None:
    """Validate finite, non-negative lighting without material randomization."""

    _finite(
        value.ambient_intensity,
        "LightingConfigurationV1.ambient_intensity",
        nonnegative=True,
    )
    _finite(
        value.key_intensity,
        "LightingConfigurationV1.key_intensity",
        nonnegative=True,
    )
    if len(value.key_color_rgb) != 3 or len(value.key_direction) != 3:
        _fail("LightingConfigurationV1", "expected RGB and direction vectors")
    for index, item in enumerate(value.key_color_rgb):
        component = _finite(item, f"LightingConfigurationV1.key_color_rgb[{index}]")
        if not 0.0 <= component <= 1.0:
            _fail(
                "LightingConfigurationV1.key_color_rgb", "components must be in [0,1]"
            )
    direction = tuple(
        _finite(item, f"LightingConfigurationV1.key_direction[{index}]")
        for index, item in enumerate(value.key_direction)
    )
    if math.sqrt(sum(item * item for item in direction)) == 0.0:
        _fail("LightingConfigurationV1.key_direction", "must be non-zero")


def validate_render_domain(value: RenderDomainV1) -> None:
    """Validate one bounded domain and its authorized collection/split scope."""

    if value.schema_version != RENDER_DOMAIN_SCHEMA_VERSION:
        _fail("RenderDomainV1.schema_version", "unsupported version")
    if value.domain_id not in RENDER_DOMAIN_IDS:
        _fail("RenderDomainV1.domain_id", "unsupported domain")
    if _SEMVER.fullmatch(value.semantic_version) is None:
        _fail("RenderDomainV1.semantic_version", "expected semantic version")
    for minimum, maximum, context in (
        (
            value.camera_translation_min,
            value.camera_translation_max,
            "RenderDomainV1.camera_translation",
        ),
        (
            value.camera_rotation_rpy_min_degrees,
            value.camera_rotation_rpy_max_degrees,
            "RenderDomainV1.camera_rotation_rpy_degrees",
        ),
    ):
        if len(minimum) != 3 or len(maximum) != 3:
            _fail(context, "expected three-axis ranges")
        for index, (lower, upper) in enumerate(zip(minimum, maximum, strict=True)):
            if _finite(lower, f"{context}.min[{index}]") > _finite(
                upper, f"{context}.max[{index}]"
            ):
                _fail(context, "minimum exceeds maximum")
    if not isinstance(value.lighting, LightingConfigurationV1):
        _fail("RenderDomainV1.lighting", "invalid lighting")
    if not value.allowed_collections or len(set(value.allowed_collections)) != len(
        value.allowed_collections
    ):
        _fail("RenderDomainV1.allowed_collections", "invalid inventory")
    if not value.allowed_splits or len(set(value.allowed_splits)) != len(
        value.allowed_splits
    ):
        _fail("RenderDomainV1.allowed_splits", "invalid inventory")
    if any(
        not isinstance(item, SourceCollection) for item in value.allowed_collections
    ):
        _fail("RenderDomainV1.allowed_collections", "invalid collection")
    if any(not isinstance(item, VisualDatasetSplit) for item in value.allowed_splits):
        _fail("RenderDomainV1.allowed_splits", "invalid split")
    _digest(value.content_digest, "RenderDomainV1.content_digest")


def validate_render_domain_configuration(value: RenderDomainConfigurationV1) -> None:
    """Validate the frozen five-domain M4A rendering configuration."""

    if value.schema_version != RENDER_DOMAIN_CONFIGURATION_SCHEMA_VERSION:
        _fail("RenderDomainConfigurationV1.schema_version", "unsupported version")
    if value.configuration_id != RENDER_DOMAIN_CONFIGURATION_ID:
        _fail(
            "RenderDomainConfigurationV1.configuration_id", "unsupported configuration"
        )
    if value.seed_derivation != RENDER_SEED_DERIVATION:
        _fail(
            "RenderDomainConfigurationV1.seed_derivation",
            "unsupported deterministic seed-derivation semantic",
        )
    _text(
        value.shader_configuration, "RenderDomainConfigurationV1.shader_configuration"
    )
    if value.image_resolution != M4A_IMAGE_RESOLUTION:
        _fail("RenderDomainConfigurationV1.image_resolution", "M4A requires [224,224]")
    if value.color_format != M4A_COLOR_FORMAT:
        _fail("RenderDomainConfigurationV1.color_format", "M4A requires rgb_uint8")
    if tuple(domain.domain_id for domain in value.domains) != RENDER_DOMAIN_IDS:
        _fail(
            "RenderDomainConfigurationV1.domains", "expected fixed ordered five domains"
        )
    expected_scopes = {
        CANONICAL_DOMAIN_ID: (
            (SourceCollection.M3A_DEVELOPMENT, SourceCollection.M3C_EXTERNAL),
            tuple(VisualDatasetSplit),
        ),
        MILD_CAMERA_DOMAIN_ID: (
            (SourceCollection.M3A_DEVELOPMENT,),
            (VisualDatasetSplit.TRAIN, VisualDatasetSplit.VALIDATION),
        ),
        MILD_LIGHTING_DOMAIN_ID: (
            (SourceCollection.M3A_DEVELOPMENT,),
            (VisualDatasetSplit.TRAIN, VisualDatasetSplit.VALIDATION),
        ),
        STRONG_CAMERA_DOMAIN_ID: (
            (SourceCollection.M3A_DEVELOPMENT, SourceCollection.M3C_EXTERNAL),
            (VisualDatasetSplit.TEST, VisualDatasetSplit.EXTERNAL),
        ),
        STRONG_LIGHTING_DOMAIN_ID: (
            (SourceCollection.M3A_DEVELOPMENT, SourceCollection.M3C_EXTERNAL),
            (VisualDatasetSplit.TEST, VisualDatasetSplit.EXTERNAL),
        ),
    }
    for domain in value.domains:
        collections, splits = expected_scopes[domain.domain_id]
        if domain.allowed_collections != collections or domain.allowed_splits != splits:
            _fail(
                f"RenderDomainConfigurationV1.{domain.domain_id}",
                "authorization scope differs",
            )
    _digest(value.content_digest, "RenderDomainConfigurationV1.content_digest")


def validate_domain_assignment(
    source_collection: SourceCollection,
    split: VisualDatasetSplit,
    domain_ids: tuple[str, ...],
) -> None:
    """Require the exact pre-render domain assignment for one split."""

    if tuple(domain_ids) != assigned_render_domain_ids(source_collection, split):
        _fail(
            "VisualDomainAssignment", "domain inventory differs from frozen assignment"
        )


def validate_view_record(value: object) -> None:
    """Validate exact RGB metadata and complete state-integrity evidence."""

    from latentguard.vision_data.packet import (
        M4A_STATE_COMPARISON_TOLERANCE,
        M4A_STATE_COMPONENT_COUNT,
        VISUAL_VIEW_SCHEMA_VERSION,
        VisualViewRecordV1,
    )

    if not isinstance(value, VisualViewRecordV1):
        _fail("VisualViewRecordV1", "invalid model")
    if value.schema_version != VISUAL_VIEW_SCHEMA_VERSION:
        _fail("VisualViewRecordV1.schema_version", "unsupported version")
    if value.camera_id not in PICKCUBE_CAMERA_IDS:
        _fail("VisualViewRecordV1.camera_id", "unsupported camera")
    _safe_reference(
        value.image_reference, "VisualViewRecordV1.image_reference", npy=True
    )
    _digest(value.pixel_sha256, "VisualViewRecordV1.pixel_sha256")
    _digest(value.npy_sha256, "VisualViewRecordV1.npy_sha256")
    if value.dtype != "uint8" or value.shape != (224, 224, 3):
        _fail("VisualViewRecordV1.image", "expected uint8 [224,224,3]")
    allowed_calibration_dtypes = {"float16", "float32", "float64"}
    if value.intrinsics_dtype not in allowed_calibration_dtypes:
        _fail(
            "VisualViewRecordV1.intrinsics_dtype",
            "expected an observed floating calibration dtype",
        )
    if value.extrinsics_dtype not in allowed_calibration_dtypes:
        _fail(
            "VisualViewRecordV1.extrinsics_dtype",
            "expected an observed floating calibration dtype",
        )
    _matrix_finite(value.intrinsics, (3, 3), "VisualViewRecordV1.intrinsics")
    _matrix_finite(value.extrinsics, (4, 4), "VisualViewRecordV1.extrinsics")
    for field in (
        "camera_configuration_digest",
        "state_before_render_digest",
        "state_after_render_digest",
    ):
        _digest(getattr(value, field), f"VisualViewRecordV1.{field}")
    if value.compared_state_component_count != M4A_STATE_COMPONENT_COUNT:
        _fail("VisualViewRecordV1.compared_state_component_count", "expected 70")
    error = _finite(
        value.maximum_state_error,
        "VisualViewRecordV1.maximum_state_error",
        nonnegative=True,
    )
    if error > M4A_STATE_COMPARISON_TOLERANCE:
        _fail("VisualViewRecordV1.maximum_state_error", "exceeds 1e-6")
    if value.state_before_render_digest != value.state_after_render_digest:
        _fail("VisualViewRecordV1.state_integrity", "rendering changed complete state")
    if value.task_projection_before != value.task_projection_after:
        _fail("VisualViewRecordV1.task_projection", "rendering changed task projection")


def validate_visual_packet(value: object) -> None:
    """Validate one three-view packet and its semantic identity."""

    from latentguard.vision_data.packet import (
        M4A_VERIFIER_STATE_COMPARISON_TOLERANCE,
        M4A_VERIFIER_STATE_COMPONENT_COUNT,
        VISUAL_PACKET_SCHEMA_VERSION,
        VisualObservationPacketV1,
    )

    if not isinstance(value, VisualObservationPacketV1):
        _fail("VisualObservationPacketV1", "invalid model")
    if value.schema_version != VISUAL_PACKET_SCHEMA_VERSION:
        _fail("VisualObservationPacketV1.schema_version", "unsupported version")
    if not isinstance(value.source_collection, SourceCollection) or not isinstance(
        value.split, VisualDatasetSplit
    ):
        _fail("VisualObservationPacketV1", "invalid collection or split")
    assigned_render_domain_ids(value.source_collection, value.split)
    if value.render_domain_id not in assigned_render_domain_ids(
        value.source_collection, value.split
    ):
        _fail("VisualObservationPacketV1.render_domain_id", "not authorized for split")
    for field in (
        "source_trajectory_id",
        "anchor_id",
        "split_group_id",
        "state_reference_id",
        "verifier_state_semantic",
        "camera_rig_id",
        "render_domain_id",
        "renderer_semantic_version",
    ):
        _text(getattr(value, field), f"VisualObservationPacketV1.{field}")
    for field in (
        "expected_state_digest",
        "verifier_state_digest",
        "camera_rig_digest",
        "render_domain_digest",
        "visual_compatibility_identity",
        "pickcube_compatibility_identity",
    ):
        _digest(getattr(value, field), f"VisualObservationPacketV1.{field}")
    if (
        type(value.verifier_state_component_count) is not int
        or value.verifier_state_component_count != M4A_VERIFIER_STATE_COMPONENT_COUNT
    ):
        _fail(
            "VisualObservationPacketV1.verifier_state_component_count",
            "expected 38",
        )
    verifier_error = _finite(
        value.verifier_state_maximum_absolute_error,
        "VisualObservationPacketV1.verifier_state_maximum_absolute_error",
        nonnegative=True,
    )
    if verifier_error != M4A_VERIFIER_STATE_COMPARISON_TOLERANCE:
        _fail(
            "VisualObservationPacketV1.verifier_state_maximum_absolute_error",
            "expected exact verifier-state equality",
        )
    for field in (
        "elapsed_simulation_steps_before",
        "elapsed_simulation_steps_after",
    ):
        elapsed = getattr(value, field)
        if type(elapsed) is not int or elapsed < 0:
            _fail(f"VisualObservationPacketV1.{field}", "expected non-negative integer")
    if value.elapsed_simulation_steps_before != value.elapsed_simulation_steps_after:
        _fail(
            "VisualObservationPacketV1.elapsed_simulation_steps",
            "rendering advanced the environment step counter",
        )
    if (
        type(value.environment_close_passed) is not bool
        or not value.environment_close_passed
    ):
        _fail(
            "VisualObservationPacketV1.environment_close_passed",
            "expected successful environment close",
        )
    if type(value.render_seed) is not int or not 0 <= value.render_seed < 2**64:
        _fail("VisualObservationPacketV1.render_seed", "expected uint64")
    if tuple(view.camera_id for view in value.views) != PICKCUBE_CAMERA_IDS:
        _fail("VisualObservationPacketV1.views", "expected fixed ordered three views")
    if len({view.image_reference for view in value.views}) != 3:
        _fail("VisualObservationPacketV1.views", "duplicate image references")
    if len({view.camera_configuration_digest for view in value.views}) != 3:
        _fail("VisualObservationPacketV1.views", "duplicate camera configurations")
    if len({view.intrinsics_dtype for view in value.views}) != 1:
        _fail(
            "VisualObservationPacketV1.views",
            "intrinsics dtype differs across views",
        )
    if len({view.extrinsics_dtype for view in value.views}) != 1:
        _fail(
            "VisualObservationPacketV1.views",
            "extrinsics dtype differs across views",
        )
    before = {view.state_before_render_digest for view in value.views}
    after = {view.state_after_render_digest for view in value.views}
    projections = {view.task_projection_before.content_digest for view in value.views}
    if len(before) != 1 or len(after) != 1 or len(projections) != 1:
        _fail("VisualObservationPacketV1.views", "render boundary differs across views")
    if _PACKET_ID.fullmatch(value.packet_id) is None:
        _fail("VisualObservationPacketV1.packet_id", "invalid derived identity")


def validate_candidate_binding(value: VisualCandidateBindingV1) -> None:
    """Validate one unique candidate-to-three-packet binding."""

    if value.schema_version != VISION_DATA_SCHEMA_VERSION:
        _fail("VisualCandidateBindingV1.schema_version", "unsupported version")
    for field in ("candidate_sample_id", "candidate_group_id", "anchor_id"):
        _text(getattr(value, field), f"VisualCandidateBindingV1.{field}")
    if not isinstance(value.source_collection, SourceCollection):
        _fail("VisualCandidateBindingV1.source_collection", "invalid collection")
    if len(value.packet_ids) != 3 or len(set(value.packet_ids)) != 3:
        _fail("VisualCandidateBindingV1.packet_ids", "expected three unique packet IDs")
    if any(_PACKET_ID.fullmatch(packet_id) is None for packet_id in value.packet_ids):
        _fail("VisualCandidateBindingV1.packet_ids", "invalid packet identity")


def validate_visual_sample(value: VisualActionVerifierSampleV1) -> None:
    """Validate a reference-only expanded sample with no duplicated actions."""

    if value.schema_version != VISION_DATA_SCHEMA_VERSION:
        _fail("VisualActionVerifierSampleV1.schema_version", "unsupported version")
    if _PACKET_ID.fullmatch(value.packet_id) is None:
        _fail("VisualActionVerifierSampleV1.packet_id", "invalid packet identity")
    for field in (
        "candidate_sample_id",
        "evidence_id",
    ):
        _text(getattr(value, field), f"VisualActionVerifierSampleV1.{field}")
    if value.task_id != PICKCUBE_VISUAL_TASK_ID:
        _fail(
            "VisualActionVerifierSampleV1.task_id",
            "expected fixed maniskill/PickCube-v1",
        )
    if value.canonical_task_text != PICKCUBE_CANONICAL_TASK_TEXT:
        _fail(
            "VisualActionVerifierSampleV1.canonical_task_text",
            "differs from the fixed canonical task text",
        )
    _safe_reference(
        value.candidate_action_chunk_reference,
        "VisualActionVerifierSampleV1.candidate_action_chunk_reference",
    )
    _safe_reference(
        value.action_mask_reference,
        "VisualActionVerifierSampleV1.action_mask_reference",
    )
    for field in ("final_success", "final_unsafe"):
        if type(getattr(value, field)) is not bool:
            _fail(f"VisualActionVerifierSampleV1.{field}", "expected boolean")
    for field in (
        "source_dataset_digest",
        "candidate_dataset_digest",
        "visual_dataset_digest",
    ):
        _digest(getattr(value, field), f"VisualActionVerifierSampleV1.{field}")
    if not isinstance(value.source_collection, SourceCollection):
        _fail("VisualActionVerifierSampleV1.source_collection", "invalid collection")
    for event in value.failure_events:
        if not isinstance(event, FailureEvent):
            _fail("VisualActionVerifierSampleV1.failure_events", "invalid event")
        try:
            validate_failure_event(event, value.evidence_id)
        except DataValidationError as exc:
            _fail("VisualActionVerifierSampleV1.failure_events", str(exc))


def validate_input_contract(value: VisualModelInputContractV1) -> None:
    """Require the exact M4A allowlist and disjoint metadata classifications."""

    if value.schema_version != VISUAL_MODEL_INPUT_CONTRACT_SCHEMA_VERSION:
        _fail("VisualModelInputContractV1.schema_version", "unsupported version")
    expected = (
        (value.required_student_inputs, value.REQUIRED),
        (value.conditional_student_inputs, value.CONDITIONAL),
        (value.reporting_only_fields, value.REPORTING),
        (value.privileged_teacher_fields, value.PRIVILEGED),
    )
    if any(observed != required for observed, required in expected):
        _fail("VisualModelInputContractV1", "field policy differs from M4A contract")
    inventories = [set(observed) for observed, _ in expected]
    for index, left in enumerate(inventories):
        if any(left & right for right in inventories[index + 1 :]):
            _fail("VisualModelInputContractV1", "field classifications overlap")


__all__ = [
    "VisionDataValidationError",
    "validate_camera_configuration",
    "validate_camera_pose",
    "validate_camera_rig",
    "validate_candidate_binding",
    "validate_domain_assignment",
    "validate_input_contract",
    "validate_lighting_configuration",
    "validate_render_domain",
    "validate_render_domain_configuration",
    "validate_task_projection",
    "validate_view_record",
    "validate_visual_packet",
    "validate_visual_sample",
]
