"""Lazy ManiSkill/SAPIEN multi-view RGB rendering for PickCube.

This module owns every simulator-native camera, pose, shader, light, and tensor
object used by M4A.  Importing it never imports ManiSkill, SAPIEN, or Torch.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from latentguard.vision_data.cameras import (
    M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE,
    CameraConfigurationV1,
    CameraPoseV1,
    Matrix4,
    PickCubeMultiViewRigV1,
    expected_pinhole_focal_lengths,
    opencv_world_to_camera_extrinsics,
)
from latentguard.vision_data.domains import (
    RenderDomainConfigurationV1,
    RenderDomainV1,
)

VISUAL_RENDERER_SEMANTIC_VERSION = "maniskill_pickcube_multiview_rgb_v3"
GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC = (
    "sapien_render_system_3_0_one_group_per_runtime_camera_v1"
)
VISUAL_CAMERA_RESOLUTION_SEMANTIC = "world_pose_rpy_pcg64_per_camera_v1"
CAMERA_POSE_APPLICATION_SEMANTIC = "explicit_world_pose_wxyz_v1"
LIGHTING_APPLICATION_SEMANTIC = "ambient_plus_directional_key_v1"
RGB_CHANNEL_ORDER = "RGB"
RGB_COLOR_SPACE_ASSUMPTION = "sRGB_after_versioned_mani_skill_color_conversion"
VERTICAL_ORIENTATION_CONVENTION = "top_row_first_as_returned_by_mani_skill"
FLOAT_COLOR_TO_RGB_UINT8_SEMANTIC = (
    "mani_skill_camera_finite_unit_rgb_times_255_uint8_v1"
)
UINT8_COLOR_TO_RGB_UINT8_SEMANTIC = "drop_alpha_identity_uint8_v1"
CALIBRATION_COMPARISON_SEMANTIC = (
    "exact_intrinsics_after_runtime_cast_and_exact_pose_bound_device_"
    "recomputed_extrinsics_v1"
)
EXTRINSIC_3X4_TO_4X4_SEMANTIC = (
    "mani_skill_public_opencv_world_to_camera_3x4_to_homogeneous_4x4_v1"
)
EXTRINSIC_4X4_SEMANTIC = "public_homogeneous_world_to_camera_4x4_identity_v1"
RUNTIME_CAMERA_POSE_VERIFICATION_SEMANTIC = (
    "exact_world_pose_float32_cuda_7_components_v1"
)
RUNTIME_PUBLIC_EXTRINSIC_DERIVATION_SEMANTIC = (
    "independent_maniskill_pose_inverse_ros_to_opencv_float32_cuda_matmul_bitwise_v1"
)
PACKET_EXTRINSIC_CANONICALIZATION_SEMANTIC = (
    "content_bound_ideal_opencv_plan_cast_to_runtime_dtype_v1"
)
RUNTIME_CALIBRATION_EVIDENCE_SEMANTIC = (
    "camera_plan_bound_public_vs_independent_expected_digest_v1"
)
UNDERLYING_CAMERA_POSE_VERIFICATION_SEMANTIC = (
    "unmounted_sapien_render_camera_component_local_pose_float32_numpy_"
    "plan_and_pre_post_bitwise_v1"
)
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ManiSkillVisualRenderingError(RuntimeError):
    """Raised when a renderer API or rendered view violates the visual contract."""


class ManiSkillVisualContractError(ManiSkillVisualRenderingError):
    """Raised for deterministic renderer-output or calibration mismatches."""


def _finite_tuple(value: object, size: int, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, Iterable):
        raise ManiSkillVisualRenderingError(
            f"{field_name} must contain {size} finite numbers"
        )
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ManiSkillVisualRenderingError(
            f"{field_name} must contain {size} finite numbers"
        ) from exc
    if len(values) != size or not all(math.isfinite(item) for item in values):
        raise ManiSkillVisualRenderingError(
            f"{field_name} must contain {size} finite numbers"
        )
    return values


def _freeze_array(
    value: object,
    *,
    field_name: str,
    dtype: np.dtype[Any] | None = None,
    shape: tuple[int, ...] | None = None,
) -> NDArray[Any]:
    candidate = value
    for method_name in ("detach", "cpu"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    array = np.asarray(candidate)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ManiSkillVisualRenderingError(f"{field_name} must be numeric")
    if dtype is not None and array.dtype != dtype:
        raise ManiSkillVisualRenderingError(
            f"{field_name} must have dtype {dtype}, got {array.dtype}"
        )
    if shape is not None and array.shape != shape:
        raise ManiSkillVisualRenderingError(
            f"{field_name} must have shape {shape}, got {array.shape}"
        )
    if not bool(np.all(np.isfinite(array))):
        raise ManiSkillVisualRenderingError(f"{field_name} must be finite")
    copied = np.array(array, copy=True, order="C", subok=False)
    immutable = copied.tobytes(order="C")
    return np.frombuffer(immutable, dtype=copied.dtype).reshape(copied.shape)


@dataclass(frozen=True, slots=True)
class VisualCameraRenderPlan:
    """One fully resolved project-owned world camera configuration."""

    camera_id: str
    width: int
    height: int
    near: float
    far: float
    fov_y_degrees: float
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    intrinsics: NDArray[Any]
    extrinsics: NDArray[Any]
    camera_configuration_digest: str

    def __post_init__(self) -> None:
        """Validate the fixed RGB camera contract without importing SAPIEN."""
        if (
            not isinstance(self.camera_id, str)
            or not self.camera_id
            or self.camera_id != self.camera_id.strip()
        ):
            raise ManiSkillVisualRenderingError("camera_id must be canonical text")
        if type(self.width) is not int or type(self.height) is not int:
            raise ManiSkillVisualRenderingError("camera resolution must be integral")
        if (self.width, self.height) != (224, 224):
            raise ManiSkillVisualRenderingError("M4A cameras must be exactly 224x224")
        if (
            type(self.near) not in (int, float)
            or type(self.far) not in (int, float)
            or not math.isfinite(float(self.near))
            or not math.isfinite(float(self.far))
            or not 0.0 < float(self.near) < float(self.far)
        ):
            raise ManiSkillVisualRenderingError("camera near/far planes are invalid")
        if (
            type(self.fov_y_degrees) not in (int, float)
            or not math.isfinite(float(self.fov_y_degrees))
            or not 0.0 < float(self.fov_y_degrees) < 180.0
        ):
            raise ManiSkillVisualRenderingError("camera field of view is invalid")
        position = cast(
            tuple[float, float, float],
            _finite_tuple(self.position, 3, field_name="camera position"),
        )
        quaternion = cast(
            tuple[float, float, float, float],
            _finite_tuple(self.quaternion_wxyz, 4, field_name="camera quaternion"),
        )
        norm = math.sqrt(sum(item * item for item in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ManiSkillVisualRenderingError("camera quaternion must be normalized")
        intrinsics = _freeze_array(
            self.intrinsics,
            field_name="camera intrinsics",
            shape=(3, 3),
        )
        expected_fx, expected_fy = expected_pinhole_focal_lengths(
            self.width, self.height, float(self.fov_y_degrees)
        )
        if not math.isclose(
            float(intrinsics[0, 0]),
            expected_fx,
            rel_tol=0.0,
            abs_tol=M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE,
        ) or not math.isclose(
            float(intrinsics[1, 1]),
            expected_fy,
            rel_tol=0.0,
            abs_tol=M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE,
        ):
            raise ManiSkillVisualRenderingError(
                "camera FOV, intrinsics, and image aspect describe "
                "different projections"
            )
        extrinsics = _freeze_array(
            self.extrinsics,
            field_name="camera extrinsics",
            shape=(4, 4),
        )
        expected_extrinsics = np.asarray(
            opencv_world_to_camera_extrinsics(position, quaternion),
            dtype=extrinsics.dtype,
        )
        if (
            extrinsics.shape != expected_extrinsics.shape
            or extrinsics.dtype != expected_extrinsics.dtype
            or extrinsics.tobytes(order="C") != expected_extrinsics.tobytes(order="C")
        ):
            raise ManiSkillVisualRenderingError(
                "camera extrinsics must exactly derive from the declared world pose"
            )
        if (
            not isinstance(self.camera_configuration_digest, str)
            or not self.camera_configuration_digest.startswith("sha256:")
            or len(self.camera_configuration_digest) != 71
        ):
            raise ManiSkillVisualRenderingError(
                "camera configuration digest must be sha256-prefixed"
            )
        object.__setattr__(self, "near", float(self.near))
        object.__setattr__(self, "far", float(self.far))
        object.__setattr__(self, "fov_y_degrees", float(self.fov_y_degrees))
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "extrinsics", extrinsics)


@dataclass(frozen=True, slots=True)
class VisualLightingRenderPlan:
    """One resolved render-only lighting configuration."""

    ambient_intensity: float
    key_intensity: float
    key_color_rgb: tuple[float, float, float]
    key_direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Require finite non-negative light values and a non-zero direction."""
        for name in ("ambient_intensity", "key_intensity"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ManiSkillVisualRenderingError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, float(value))
        color = _finite_tuple(self.key_color_rgb, 3, field_name="key light color")
        direction = _finite_tuple(
            self.key_direction, 3, field_name="key light direction"
        )
        if any(item < 0.0 for item in color):
            raise ManiSkillVisualRenderingError("key light color must be non-negative")
        if math.sqrt(sum(item * item for item in direction)) <= 0.0:
            raise ManiSkillVisualRenderingError("key light direction must be non-zero")
        object.__setattr__(self, "key_color_rgb", color)
        object.__setattr__(self, "key_direction", direction)


@dataclass(frozen=True, slots=True)
class PickCubeVisualRenderPlan:
    """Resolved three-view plan consumed only by the integration renderer."""

    domain_id: str
    render_seed: int
    shader_configuration: str
    cameras: tuple[VisualCameraRenderPlan, ...]
    lighting: VisualLightingRenderPlan

    def __post_init__(self) -> None:
        """Freeze ordering and enforce the fixed M4A view inventory."""
        if (
            not isinstance(self.domain_id, str)
            or not self.domain_id
            or self.domain_id != self.domain_id.strip()
        ):
            raise ManiSkillVisualRenderingError("render domain ID is invalid")
        if type(self.render_seed) is not int or not 0 <= self.render_seed < 2**32:
            raise ManiSkillVisualRenderingError("render seed must be uint32")
        if (
            not isinstance(self.shader_configuration, str)
            or not self.shader_configuration
            or self.shader_configuration != self.shader_configuration.strip()
        ):
            raise ManiSkillVisualRenderingError("shader configuration is invalid")
        cameras = tuple(self.cameras)
        expected_ids = ("front_oblique", "overhead", "side_oblique")
        if tuple(camera.camera_id for camera in cameras) != expected_ids:
            raise ManiSkillVisualRenderingError(
                "M4A render plan must contain the ordered three-view rig"
            )
        object.__setattr__(self, "cameras", cameras)


@dataclass(frozen=True, slots=True)
class RuntimeCalibrationEvidenceV1:
    """Persistent digest evidence for one independently verified public matrix."""

    camera_id: str
    camera_configuration_digest: str
    runtime_extrinsics_digest: str
    expected_runtime_extrinsics_digest: str
    semantic: str = RUNTIME_CALIBRATION_EVIDENCE_SEMANTIC

    def __post_init__(self) -> None:
        if self.camera_id not in ("front_oblique", "overhead", "side_oblique"):
            raise ManiSkillVisualRenderingError(
                "runtime calibration evidence camera is unsupported"
            )
        for name in (
            "camera_configuration_digest",
            "runtime_extrinsics_digest",
            "expected_runtime_extrinsics_digest",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or _SHA256_DIGEST_PATTERN.fullmatch(value) is None
            ):
                raise ManiSkillVisualRenderingError(
                    f"runtime calibration evidence {name} must be a lowercase SHA-256"
                )
        if self.runtime_extrinsics_digest != self.expected_runtime_extrinsics_digest:
            raise ManiSkillVisualRenderingError(
                "runtime calibration evidence does not match independent expectation"
            )
        if self.semantic != RUNTIME_CALIBRATION_EVIDENCE_SEMANTIC:
            raise ManiSkillVisualRenderingError(
                "runtime calibration evidence semantic is unsupported"
            )

    def as_mapping(self) -> MappingProxyType[str, str]:
        """Return the strict JSON-native evidence mapping."""

        return MappingProxyType(
            {
                "camera_configuration_digest": self.camera_configuration_digest,
                "camera_id": self.camera_id,
                "expected_runtime_extrinsics_digest": (
                    self.expected_runtime_extrinsics_digest
                ),
                "runtime_extrinsics_digest": self.runtime_extrinsics_digest,
                "semantic": self.semantic,
            }
        )


@dataclass(frozen=True, slots=True, eq=False)
class RenderedVisualView:
    """One detached RGB result with canonical and raw runtime calibration."""

    camera_id: str
    rgb: NDArray[Any]
    intrinsics: NDArray[Any]
    extrinsics: NDArray[Any]
    runtime_extrinsics: NDArray[Any]
    expected_runtime_extrinsics_digest: str
    camera_configuration_digest: str
    runtime_extrinsics_digest: str = field(init=False)

    def __post_init__(self) -> None:
        """Detach runtime tensors and require exact M4A image/calibration shapes."""
        rgb = _freeze_array(
            self.rgb,
            field_name=f"{self.camera_id} RGB",
            dtype=np.dtype(np.uint8),
            shape=(224, 224, 3),
        )
        intrinsics = _freeze_array(
            self.intrinsics,
            field_name=f"{self.camera_id} intrinsics",
            shape=(3, 3),
        )
        extrinsics = _freeze_array(
            self.extrinsics,
            field_name=f"{self.camera_id} canonical extrinsics",
            shape=(4, 4),
        )
        runtime_extrinsics = _freeze_array(
            self.runtime_extrinsics,
            field_name=f"{self.camera_id} runtime extrinsics",
            shape=(4, 4),
        )
        if extrinsics.dtype != runtime_extrinsics.dtype:
            raise ManiSkillVisualRenderingError(
                f"{self.camera_id} canonical and runtime extrinsics dtypes differ"
            )
        runtime_extrinsics_digest = runtime_calibration_array_digest(runtime_extrinsics)
        if (
            not isinstance(self.expected_runtime_extrinsics_digest, str)
            or self.expected_runtime_extrinsics_digest != runtime_extrinsics_digest
        ):
            raise ManiSkillVisualRenderingError(
                f"{self.camera_id} runtime extrinsics lack matching verified evidence"
            )
        if (
            not isinstance(self.camera_configuration_digest, str)
            or not self.camera_configuration_digest.startswith("sha256:")
            or len(self.camera_configuration_digest) != 71
        ):
            raise ManiSkillVisualRenderingError(
                "rendered camera configuration digest must be sha256-prefixed"
            )
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "extrinsics", extrinsics)
        object.__setattr__(self, "runtime_extrinsics", runtime_extrinsics)
        object.__setattr__(self, "runtime_extrinsics_digest", runtime_extrinsics_digest)

    @property
    def runtime_calibration_evidence(self) -> RuntimeCalibrationEvidenceV1:
        """Return the exact proof record for persistent compatibility evidence."""

        return RuntimeCalibrationEvidenceV1(
            camera_id=self.camera_id,
            camera_configuration_digest=self.camera_configuration_digest,
            runtime_extrinsics_digest=self.runtime_extrinsics_digest,
            expected_runtime_extrinsics_digest=(
                self.expected_runtime_extrinsics_digest
            ),
        )


@dataclass(frozen=True, slots=True)
class VisualRendererApiObservation:
    """Sanitized dynamic API facts observed from the installed renderer."""

    renderer_backend: str
    scene_type: str
    camera_type: str
    shader_configuration: str
    camera_configuration_api: str
    camera_group_initialization_semantic: str
    camera_group_texture_names: tuple[str, ...]
    camera_group_count: int
    underlying_camera_count_per_group: int
    camera_groups_ready: bool
    world_camera_pose_representation: str
    sensor_update_calls: tuple[str, ...]
    image_dtype: str = "uint8"
    image_channel_order: str = RGB_CHANNEL_ORDER
    vertical_orientation: str = VERTICAL_ORIENTATION_CONVENTION
    color_space_assumption: str = RGB_COLOR_SPACE_ASSUMPTION
    camera_intrinsics_available: bool = True
    camera_extrinsics_available: bool = True
    rendering_requires_sensor_update_calls: bool = False
    calibration_comparison_semantic: str = CALIBRATION_COMPARISON_SEMANTIC
    raw_color_texture_dtype: str = "unobserved"
    rgb_conversion_semantic: str = "unobserved"
    runtime_intrinsics_dtype: str = "unobserved"
    runtime_extrinsics_dtype: str = "unobserved"
    raw_extrinsic_matrix_shape: str = "unobserved"
    runtime_extrinsics_semantic: str = "unobserved"
    runtime_camera_pose_dtype: str = "unobserved"
    raw_camera_pose_shape: str = "unobserved"
    runtime_camera_pose_device_type: str = "unobserved"
    runtime_pose_component_count: int = 0
    runtime_underlying_pose_type: str = "unobserved"
    runtime_underlying_position_type: str = "unobserved"
    runtime_underlying_quaternion_type: str = "unobserved"
    runtime_underlying_pose_dtype: str = "unobserved"
    raw_underlying_position_shape: str = "unobserved"
    raw_underlying_quaternion_shape: str = "unobserved"
    runtime_underlying_pose_component_count: int = 0
    underlying_camera_pose_pre_post_bitwise: bool = False
    runtime_extrinsic_component_count: int = 0
    runtime_pose_verification_semantic: str = RUNTIME_CAMERA_POSE_VERIFICATION_SEMANTIC
    runtime_public_extrinsic_derivation_semantic: str = (
        RUNTIME_PUBLIC_EXTRINSIC_DERIVATION_SEMANTIC
    )
    packet_extrinsic_canonicalization_semantic: str = (
        PACKET_EXTRINSIC_CANONICALIZATION_SEMANTIC
    )
    underlying_camera_pose_verification_semantic: str = (
        UNDERLYING_CAMERA_POSE_VERIFICATION_SEMANTIC
    )

    def __post_init__(self) -> None:
        """Require a compact, sanitized, fixed RGB API observation."""
        for name in (
            "renderer_backend",
            "scene_type",
            "camera_type",
            "shader_configuration",
            "camera_configuration_api",
            "camera_group_initialization_semantic",
            "calibration_comparison_semantic",
            "world_camera_pose_representation",
            "image_dtype",
            "image_channel_order",
            "vertical_orientation",
            "color_space_assumption",
            "raw_color_texture_dtype",
            "rgb_conversion_semantic",
            "runtime_intrinsics_dtype",
            "runtime_extrinsics_dtype",
            "raw_extrinsic_matrix_shape",
            "runtime_extrinsics_semantic",
            "runtime_camera_pose_dtype",
            "raw_camera_pose_shape",
            "runtime_camera_pose_device_type",
            "runtime_underlying_pose_type",
            "runtime_underlying_position_type",
            "runtime_underlying_quaternion_type",
            "runtime_underlying_pose_dtype",
            "raw_underlying_position_shape",
            "raw_underlying_quaternion_shape",
            "runtime_pose_verification_semantic",
            "runtime_public_extrinsic_derivation_semantic",
            "packet_extrinsic_canonicalization_semantic",
            "underlying_camera_pose_verification_semantic",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or len(value) > 512
                or any(ord(character) < 32 for character in value)
                or "/" in value
                or "\\" in value
                or "0x" in value.lower()
            ):
                raise ManiSkillVisualRenderingError(
                    f"renderer API {name} must be sanitized text"
                )
        if (
            self.camera_group_initialization_semantic
            != GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API camera-group initialization semantic is unsupported"
            )
        texture_names = tuple(self.camera_group_texture_names)
        if (
            not texture_names
            or "Color" not in texture_names
            or len(texture_names) != len(set(texture_names))
            or any(
                not isinstance(name, str)
                or not name
                or name != name.strip()
                or len(name) > 128
                or any(ord(character) < 32 for character in name)
                for name in texture_names
            )
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API camera-group texture names are invalid"
            )
        if type(self.camera_group_count) is not int or self.camera_group_count != 3:
            raise ManiSkillVisualRenderingError(
                "renderer API must initialize exactly three camera groups"
            )
        if (
            type(self.underlying_camera_count_per_group) is not int
            or self.underlying_camera_count_per_group != 1
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API must bind exactly one underlying camera per group"
            )
        if type(self.camera_groups_ready) is not bool or not self.camera_groups_ready:
            raise ManiSkillVisualRenderingError(
                "renderer API camera groups must be ready"
            )
        valid_color_contracts = {
            ("unobserved", "unobserved"),
            ("uint8", UINT8_COLOR_TO_RGB_UINT8_SEMANTIC),
            ("float16", FLOAT_COLOR_TO_RGB_UINT8_SEMANTIC),
            ("float32", FLOAT_COLOR_TO_RGB_UINT8_SEMANTIC),
            ("float64", FLOAT_COLOR_TO_RGB_UINT8_SEMANTIC),
        }
        if (
            self.raw_color_texture_dtype,
            self.rgb_conversion_semantic,
        ) not in valid_color_contracts:
            raise ManiSkillVisualRenderingError(
                "renderer API raw Color conversion contract is unsupported"
            )
        valid_extrinsic_contracts = {
            ("unobserved", "unobserved"),
            ("[1,3,4]", EXTRINSIC_3X4_TO_4X4_SEMANTIC),
        }
        if (
            self.raw_extrinsic_matrix_shape,
            self.runtime_extrinsics_semantic,
        ) not in valid_extrinsic_contracts:
            raise ManiSkillVisualRenderingError(
                "renderer API runtime extrinsic conversion is unsupported"
            )
        allowed_calibration_dtypes = {
            "unobserved",
            "float16",
            "float32",
            "float64",
        }
        if (
            self.runtime_intrinsics_dtype not in allowed_calibration_dtypes
            or self.runtime_extrinsics_dtype not in allowed_calibration_dtypes
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API runtime calibration dtype is unsupported"
            )
        if (self.runtime_intrinsics_dtype == "unobserved") != (
            self.runtime_extrinsics_dtype == "unobserved"
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API runtime calibration dtypes must be observed together"
            )
        for name in (
            "runtime_pose_component_count",
            "runtime_underlying_pose_component_count",
            "runtime_extrinsic_component_count",
        ):
            if type(getattr(self, name)) is not int:
                raise ManiSkillVisualRenderingError(
                    f"renderer API {name} must be an integer"
                )
        pose_observation = (
            self.runtime_camera_pose_dtype,
            self.raw_camera_pose_shape,
            self.runtime_camera_pose_device_type,
            self.runtime_pose_component_count,
            self.runtime_underlying_pose_type,
            self.runtime_underlying_position_type,
            self.runtime_underlying_quaternion_type,
            self.runtime_underlying_pose_dtype,
            self.raw_underlying_position_shape,
            self.raw_underlying_quaternion_shape,
            self.runtime_underlying_pose_component_count,
            self.underlying_camera_pose_pre_post_bitwise,
            self.runtime_extrinsic_component_count,
        )
        if pose_observation not in {
            (
                "unobserved",
                "unobserved",
                "unobserved",
                0,
                "unobserved",
                "unobserved",
                "unobserved",
                "unobserved",
                "unobserved",
                "unobserved",
                0,
                False,
                0,
            ),
            (
                "float32",
                "[1,7]",
                "cuda",
                7,
                "sapien.pysapien.Pose",
                "numpy.ndarray",
                "numpy.ndarray",
                "float32",
                "[3]",
                "[4]",
                7,
                True,
                12,
            ),
        }:
            raise ManiSkillVisualRenderingError(
                "renderer API runtime pose/extrinsic evidence is unsupported"
            )
        calibration_observed = self.runtime_intrinsics_dtype != "unobserved"
        if any(
            observed != calibration_observed
            for observed in (
                self.raw_color_texture_dtype != "unobserved",
                self.raw_extrinsic_matrix_shape != "unobserved",
                self.runtime_camera_pose_dtype != "unobserved",
            )
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API image, pose, and calibration facts must be "
                "observed together"
            )
        if self.runtime_extrinsics_dtype not in {"unobserved", "float32"}:
            raise ManiSkillVisualRenderingError(
                "renderer API public extrinsics must use pinned float32"
            )
        if (
            self.runtime_pose_verification_semantic
            != RUNTIME_CAMERA_POSE_VERIFICATION_SEMANTIC
            or self.runtime_public_extrinsic_derivation_semantic
            != RUNTIME_PUBLIC_EXTRINSIC_DERIVATION_SEMANTIC
            or self.packet_extrinsic_canonicalization_semantic
            != PACKET_EXTRINSIC_CANONICALIZATION_SEMANTIC
            or self.underlying_camera_pose_verification_semantic
            != UNDERLYING_CAMERA_POSE_VERIFICATION_SEMANTIC
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API calibration evidence semantic is unsupported"
            )
        calls = tuple(self.sensor_update_calls)
        if not calls or len(calls) != len(set(calls)):
            raise ManiSkillVisualRenderingError(
                "renderer API sensor calls must be non-empty and unique"
            )
        if any(
            not isinstance(call, str)
            or not call
            or call != call.strip()
            or len(call) > 512
            or any(ord(character) < 32 for character in call)
            for call in calls
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API sensor calls must be sanitized text"
            )
        if (
            self.calibration_comparison_semantic != CALIBRATION_COMPARISON_SEMANTIC
            or self.image_dtype != "uint8"
            or self.image_channel_order != RGB_CHANNEL_ORDER
            or self.vertical_orientation != VERTICAL_ORIENTATION_CONVENTION
            or self.color_space_assumption != RGB_COLOR_SPACE_ASSUMPTION
        ):
            raise ManiSkillVisualRenderingError(
                "renderer API image contract differs from fixed M4A RGB"
            )
        for name in (
            "camera_intrinsics_available",
            "camera_extrinsics_available",
            "rendering_requires_sensor_update_calls",
            "underlying_camera_pose_pre_post_bitwise",
        ):
            if type(getattr(self, name)) is not bool:
                raise ManiSkillVisualRenderingError(
                    f"renderer API {name} must be boolean"
                )
        object.__setattr__(self, "sensor_update_calls", calls)
        object.__setattr__(self, "camera_group_texture_names", texture_names)

    def as_mapping(self) -> MappingProxyType[str, object]:
        """Return a JSON-native sanitized observation."""
        return MappingProxyType(
            {
                "calibration_comparison_semantic": (
                    self.calibration_comparison_semantic
                ),
                "camera_configuration_api": self.camera_configuration_api,
                "camera_group_count": self.camera_group_count,
                "camera_group_initialization_semantic": (
                    self.camera_group_initialization_semantic
                ),
                "camera_group_texture_names": list(self.camera_group_texture_names),
                "camera_groups_ready": self.camera_groups_ready,
                "camera_extrinsics_available": self.camera_extrinsics_available,
                "camera_intrinsics_available": self.camera_intrinsics_available,
                "camera_type": self.camera_type,
                "color_space_assumption": self.color_space_assumption,
                "image_channel_order": self.image_channel_order,
                "image_dtype": self.image_dtype,
                "renderer_backend": self.renderer_backend,
                "rendering_requires_sensor_update_calls": (
                    self.rendering_requires_sensor_update_calls
                ),
                "raw_color_texture_dtype": self.raw_color_texture_dtype,
                "raw_extrinsic_matrix_shape": self.raw_extrinsic_matrix_shape,
                "raw_camera_pose_shape": self.raw_camera_pose_shape,
                "raw_underlying_position_shape": self.raw_underlying_position_shape,
                "raw_underlying_quaternion_shape": (
                    self.raw_underlying_quaternion_shape
                ),
                "rgb_conversion_semantic": self.rgb_conversion_semantic,
                "runtime_extrinsics_dtype": self.runtime_extrinsics_dtype,
                "runtime_extrinsics_semantic": self.runtime_extrinsics_semantic,
                "runtime_intrinsics_dtype": self.runtime_intrinsics_dtype,
                "runtime_camera_pose_device_type": (
                    self.runtime_camera_pose_device_type
                ),
                "runtime_camera_pose_dtype": self.runtime_camera_pose_dtype,
                "runtime_extrinsic_component_count": (
                    self.runtime_extrinsic_component_count
                ),
                "runtime_pose_component_count": self.runtime_pose_component_count,
                "runtime_underlying_pose_component_count": (
                    self.runtime_underlying_pose_component_count
                ),
                "runtime_underlying_pose_dtype": self.runtime_underlying_pose_dtype,
                "runtime_underlying_pose_type": self.runtime_underlying_pose_type,
                "runtime_underlying_position_type": (
                    self.runtime_underlying_position_type
                ),
                "runtime_underlying_quaternion_type": (
                    self.runtime_underlying_quaternion_type
                ),
                "underlying_camera_pose_pre_post_bitwise": (
                    self.underlying_camera_pose_pre_post_bitwise
                ),
                "runtime_pose_verification_semantic": (
                    self.runtime_pose_verification_semantic
                ),
                "runtime_public_extrinsic_derivation_semantic": (
                    self.runtime_public_extrinsic_derivation_semantic
                ),
                "packet_extrinsic_canonicalization_semantic": (
                    self.packet_extrinsic_canonicalization_semantic
                ),
                "underlying_camera_pose_verification_semantic": (
                    self.underlying_camera_pose_verification_semantic
                ),
                "scene_type": self.scene_type,
                "sensor_update_calls": list(self.sensor_update_calls),
                "shader_configuration": self.shader_configuration,
                "vertical_orientation": self.vertical_orientation,
                "underlying_camera_count_per_group": (
                    self.underlying_camera_count_per_group
                ),
                "world_camera_pose_representation": (
                    self.world_camera_pose_representation
                ),
            }
        )


@runtime_checkable
class PickCubeVisualRenderHandle(Protocol):
    """Prepared cameras that may render repeatedly without reconfiguration."""

    @property
    def api_observation(self) -> VisualRendererApiObservation:
        """Return the observed dynamic API contract."""
        ...

    def render_views(self) -> tuple[RenderedVisualView, ...]:
        """Render the ordered three views without a simulation step."""
        ...


@runtime_checkable
class PickCubeVisualRenderer(Protocol):
    """Injectable renderer boundary used by CPU-only fake tests."""

    def prepare(
        self, environment: object, plan: PickCubeVisualRenderPlan
    ) -> PickCubeVisualRenderHandle:
        """Configure render-only cameras and lighting once."""
        ...


@dataclass(slots=True)
class _InstalledRenderHandle:
    scene: object
    cameras: tuple[tuple[VisualCameraRenderPlan, object], ...]
    _api_observation: VisualRendererApiObservation

    @property
    def api_observation(self) -> VisualRendererApiObservation:
        return self._api_observation

    def render_views(self) -> tuple[RenderedVisualView, ...]:
        """Update only render transforms, capture, and detach exact RGB bytes."""
        update_render = getattr(self.scene, "update_render", None)
        if not callable(update_render):
            raise ManiSkillVisualRenderingError("scene.update_render is unavailable")
        update_render(update_sensors=False, update_human_render_cameras=False)
        results: list[RenderedVisualView] = []
        for plan, camera in self.cameras:
            take_picture = getattr(camera, "take_picture", None)
            get_picture = getattr(camera, "get_picture", None)
            get_intrinsic = getattr(camera, "get_intrinsic_matrix", None)
            get_extrinsic = getattr(camera, "get_extrinsic_matrix", None)
            if (
                not callable(take_picture)
                or not callable(get_picture)
                or not callable(get_intrinsic)
                or not callable(get_extrinsic)
            ):
                raise ManiSkillVisualRenderingError(
                    f"camera {plan.camera_id!r} lacks a required public render API"
                )
            before = _read_verified_runtime_calibration(
                plan=plan,
                camera=camera,
                get_intrinsic=get_intrinsic,
                get_extrinsic=get_extrinsic,
            )
            take_picture()
            textures = get_picture(["Color"])
            if not isinstance(textures, Sequence) or len(textures) != 1:
                raise ManiSkillVisualRenderingError(
                    f"camera {plan.camera_id!r} returned an invalid Color texture"
                )
            after = _read_verified_runtime_calibration(
                plan=plan,
                camera=camera,
                get_intrinsic=get_intrinsic,
                get_extrinsic=get_extrinsic,
            )
            _require_array_bits_equal(
                before.intrinsics,
                after.intrinsics,
                field_name=f"{plan.camera_id} pre/post-capture intrinsics",
            )
            _require_array_bits_equal(
                before.raw_public_extrinsics,
                after.raw_public_extrinsics,
                field_name=f"{plan.camera_id} pre/post-capture public extrinsics",
            )
            _require_array_bits_equal(
                before.underlying_world_pose,
                after.underlying_world_pose,
                field_name=f"{plan.camera_id} pre/post-capture underlying pose",
            )
            color = _runtime_array(textures[0])
            if color.shape == (1, 224, 224, 4):
                color = color[0]
            if color.shape != (224, 224, 4):
                raise ManiSkillVisualRenderingError(
                    f"camera {plan.camera_id!r} Color must be [1,224,224,4]"
                )
            rgb, raw_color_dtype, rgb_conversion = _convert_color_to_rgb_uint8(color)
            observation = replace(
                self._api_observation,
                raw_color_texture_dtype=raw_color_dtype,
                rgb_conversion_semantic=rgb_conversion,
                runtime_intrinsics_dtype=after.intrinsics.dtype.name,
                runtime_extrinsics_dtype=after.extrinsics.dtype.name,
                raw_extrinsic_matrix_shape=after.raw_extrinsic_shape,
                runtime_extrinsics_semantic=after.extrinsic_semantic,
                runtime_camera_pose_dtype=after.runtime_pose_dtype,
                raw_camera_pose_shape=after.raw_pose_shape,
                runtime_camera_pose_device_type=after.runtime_pose_device_type,
                runtime_pose_component_count=7,
                runtime_underlying_pose_type=after.runtime_underlying_pose_type,
                runtime_underlying_position_type=(
                    after.runtime_underlying_position_type
                ),
                runtime_underlying_quaternion_type=(
                    after.runtime_underlying_quaternion_type
                ),
                runtime_underlying_pose_dtype=after.runtime_underlying_pose_dtype,
                raw_underlying_position_shape="[3]",
                raw_underlying_quaternion_shape="[4]",
                runtime_underlying_pose_component_count=7,
                underlying_camera_pose_pre_post_bitwise=True,
                runtime_extrinsic_component_count=12,
            )
            if self._api_observation.raw_color_texture_dtype != "unobserved" and (
                observation != self._api_observation
            ):
                raise ManiSkillVisualRenderingError(
                    "renderer raw image or calibration API changed between views"
                )
            self._api_observation = observation
            results.append(
                RenderedVisualView(
                    camera_id=plan.camera_id,
                    rgb=rgb,
                    intrinsics=after.intrinsics,
                    extrinsics=np.asarray(
                        plan.extrinsics, dtype=after.extrinsics.dtype
                    ),
                    runtime_extrinsics=after.extrinsics,
                    expected_runtime_extrinsics_digest=(
                        after.expected_runtime_extrinsics_digest
                    ),
                    camera_configuration_digest=plan.camera_configuration_digest,
                )
            )
        return tuple(results)


@dataclass(frozen=True, slots=True)
class _VerifiedRuntimeCalibration:
    """One exact live camera-calibration read bound to its planned world pose."""

    intrinsics: NDArray[Any]
    extrinsics: NDArray[Any]
    raw_public_extrinsics: NDArray[Any]
    raw_extrinsic_shape: str
    extrinsic_semantic: str
    runtime_pose_dtype: str
    raw_pose_shape: str
    runtime_pose_device_type: str
    underlying_world_pose: NDArray[Any]
    runtime_underlying_pose_type: str
    runtime_underlying_position_type: str
    runtime_underlying_quaternion_type: str
    runtime_underlying_pose_dtype: str
    expected_runtime_extrinsics_digest: str


def _read_verified_runtime_calibration(
    *,
    plan: VisualCameraRenderPlan,
    camera: object,
    get_intrinsic: Any,
    get_extrinsic: Any,
) -> _VerifiedRuntimeCalibration:
    """Read and independently verify one pinned GPU camera calibration."""

    if getattr(camera, "mount", object()) is not None:
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} must remain an unmounted world camera"
        )
    render_cameras = getattr(camera, "_render_cameras", None)
    if not isinstance(render_cameras, list) or len(render_cameras) != 1:
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} must retain one underlying camera"
        )
    underlying_before = _read_verified_underlying_world_pose(
        plan=plan,
        render_camera=render_cameras[0],
    )
    get_global_pose = getattr(camera, "get_global_pose", None)
    if not callable(get_global_pose):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} lacks its global-pose API"
        )
    runtime_pose = get_global_pose()
    if _qualified_type_name(runtime_pose) != "mani_skill.utils.structs.pose.Pose":
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} returned an unsupported pose type"
        )
    raw_pose_tensor = getattr(runtime_pose, "raw_pose", None)
    raw_pose, pose_device_type, pose_device = _require_pinned_runtime_tensor(
        raw_pose_tensor,
        shape=(1, 7),
        field_name=f"{plan.camera_id} runtime world pose",
    )
    new_tensor = getattr(raw_pose_tensor, "new_tensor", None)
    clone = getattr(raw_pose_tensor, "clone", None)
    if not callable(new_tensor) or not callable(clone):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} pose tensor lacks independent creation APIs"
        )
    expected_pose_tensor = new_tensor(((*plan.position, *plan.quaternion_wxyz),))
    expected_pose_raw, expected_device_type, expected_device = (
        _require_pinned_runtime_tensor(
            expected_pose_tensor,
            shape=(1, 7),
            field_name=f"{plan.camera_id} independently cast plan pose",
        )
    )
    if expected_device_type != pose_device_type or expected_device != pose_device:
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} plan pose used a different CUDA device"
        )
    _require_array_bits_equal(
        raw_pose,
        expected_pose_raw,
        field_name=f"{plan.camera_id} runtime world pose",
    )
    pose_type = type(runtime_pose)
    create_pose = getattr(pose_type, "create", None)
    if not callable(create_pose):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} pose type lacks independent creation"
        )
    independent_pose = create_pose(
        expected_pose_tensor.clone(),
        device=getattr(raw_pose_tensor, "device", None),
    )
    if type(independent_pose) is not pose_type:
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} independent pose type changed"
        )
    inverse = getattr(independent_pose, "inv", None)
    if not callable(inverse):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} independent pose lacks inversion"
        )
    inverse_pose = inverse()
    to_matrix = getattr(inverse_pose, "to_transformation_matrix", None)
    if not callable(to_matrix):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} inverse pose lacks matrix conversion"
        )
    inverse_matrix_tensor = to_matrix()
    _, inverse_device_type, inverse_device = _require_pinned_runtime_tensor(
        inverse_matrix_tensor,
        shape=(1, 4, 4),
        field_name=f"{plan.camera_id} independent inverse pose matrix",
    )
    axis_tensor = new_tensor(
        (
            (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    _, axis_device_type, axis_device = _require_pinned_runtime_tensor(
        axis_tensor,
        shape=(4, 4),
        field_name=f"{plan.camera_id} ROS-to-OpenCV axis matrix",
    )
    if (
        inverse_device_type != pose_device_type
        or axis_device_type != pose_device_type
        or inverse_device != pose_device
        or axis_device != pose_device
    ):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} derivation crossed CUDA devices"
        )
    transpose = getattr(axis_tensor, "T", None)
    if transpose is None:
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} axis matrix lacks transpose"
        )
    expected_public_tensor = (transpose @ inverse_matrix_tensor)[:, :3, :4]
    expected_public, expected_public_device_type, expected_public_device = (
        _require_pinned_runtime_tensor(
            expected_public_tensor,
            shape=(1, 3, 4),
            field_name=f"{plan.camera_id} independently derived public extrinsics",
        )
    )
    expected_extrinsics, _, _ = _single_extrinsic_matrix(expected_public_tensor)
    public_tensor = get_extrinsic()
    public_raw, public_device_type, public_device = _require_pinned_runtime_tensor(
        public_tensor,
        shape=(1, 3, 4),
        field_name=f"{plan.camera_id} public extrinsics",
    )
    if (
        expected_public_device_type != public_device_type
        or expected_public_device != public_device
        or public_device != pose_device
    ):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} public extrinsics crossed CUDA devices"
        )
    _require_array_bits_equal(
        public_raw,
        expected_public,
        field_name=f"{plan.camera_id} public extrinsics derivation",
    )
    intrinsics = _single_matrix(get_intrinsic(), (3, 3), "intrinsics")
    _require_runtime_calibration_matches_plan(
        intrinsics,
        plan.intrinsics,
        field_name=f"{plan.camera_id} intrinsics",
    )
    extrinsics, raw_shape, semantic = _single_extrinsic_matrix(public_tensor)
    underlying_after = _read_verified_underlying_world_pose(
        plan=plan,
        render_camera=render_cameras[0],
    )
    _require_array_bits_equal(
        underlying_before,
        underlying_after,
        field_name=f"{plan.camera_id} underlying pose across calibration reads",
    )
    return _VerifiedRuntimeCalibration(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        raw_public_extrinsics=public_raw,
        raw_extrinsic_shape=raw_shape,
        extrinsic_semantic=semantic,
        runtime_pose_dtype=raw_pose.dtype.name,
        raw_pose_shape="[1,7]",
        runtime_pose_device_type=pose_device_type,
        underlying_world_pose=underlying_after,
        runtime_underlying_pose_type="sapien.pysapien.Pose",
        runtime_underlying_position_type="numpy.ndarray",
        runtime_underlying_quaternion_type="numpy.ndarray",
        runtime_underlying_pose_dtype=underlying_after.dtype.name,
        expected_runtime_extrinsics_digest=runtime_calibration_array_digest(
            expected_extrinsics
        ),
    )


def _read_verified_underlying_world_pose(
    *,
    plan: VisualCameraRenderPlan,
    render_camera: object,
) -> NDArray[Any]:
    """Read the uncached SAPIEN component pose and bind all seven float32 bits."""

    get_local_pose = getattr(render_camera, "get_local_pose", None)
    if not callable(get_local_pose):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} underlying component lacks local pose"
        )
    native_pose = get_local_pose()
    if _qualified_type_name(native_pose) != "sapien.pysapien.Pose":
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} underlying pose type is unsupported"
        )
    raw_position = getattr(native_pose, "p", None)
    raw_quaternion = getattr(native_pose, "q", None)
    if (
        _qualified_type_name(raw_position) != "numpy.ndarray"
        or _qualified_type_name(raw_quaternion) != "numpy.ndarray"
    ):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} underlying pose arrays are unsupported"
        )
    if not bool(
        getattr(getattr(raw_position, "flags", None), "c_contiguous", False)
    ) or not bool(
        getattr(getattr(raw_quaternion, "flags", None), "c_contiguous", False)
    ):
        raise ManiSkillVisualContractError(
            f"camera {plan.camera_id!r} underlying pose arrays must be C-contiguous"
        )
    position = _freeze_array(
        raw_position,
        field_name=f"{plan.camera_id} underlying position",
        dtype=np.dtype(np.float32),
        shape=(3,),
    )
    quaternion = _freeze_array(
        raw_quaternion,
        field_name=f"{plan.camera_id} underlying quaternion",
        dtype=np.dtype(np.float32),
        shape=(4,),
    )
    _require_array_bits_equal(
        position,
        np.asarray(plan.position, dtype=np.float32),
        field_name=f"{plan.camera_id} underlying world position",
    )
    _require_array_bits_equal(
        quaternion,
        np.asarray(plan.quaternion_wxyz, dtype=np.float32),
        field_name=f"{plan.camera_id} underlying world quaternion",
    )
    return _freeze_array(
        np.concatenate((position, quaternion)),
        field_name=f"{plan.camera_id} underlying world pose",
        dtype=np.dtype(np.float32),
        shape=(7,),
    )


@dataclass(frozen=True, slots=True)
class LazyManiSkillPickCubeVisualRenderer:
    """Production public-camera bridge with imports delayed until ``prepare``."""

    module_importer: Any = field(default=importlib.import_module, repr=False)

    def prepare(
        self, environment: object, plan: PickCubeVisualRenderPlan
    ) -> PickCubeVisualRenderHandle:
        """Add fixed world cameras and render-only lighting to one fresh scene."""
        try:
            sapien = self.module_importer("sapien")
            render_module = self.module_importer("mani_skill.render")
        except Exception as exc:
            raise ManiSkillVisualRenderingError(
                "ManiSkill visual renderer dependencies are unavailable"
            ) from exc
        base = getattr(environment, "unwrapped", environment)
        scene = getattr(base, "scene", None)
        add_camera = getattr(scene, "add_camera", None)
        can_render = getattr(scene, "can_render", None)
        if scene is None or not callable(add_camera) or not callable(can_render):
            raise ManiSkillVisualRenderingError(
                "PickCube scene lacks the public camera/render capability API"
            )
        if not bool(can_render()):
            raise ManiSkillVisualRenderingError("PickCube scene cannot render")
        set_shader_pack = getattr(render_module, "set_shader_pack", None)
        prebuilt_shader_configs = getattr(
            render_module, "PREBUILT_SHADER_CONFIGS", None
        )
        shader_config_type = getattr(render_module, "ShaderConfig", None)
        pose_type = getattr(sapien, "Pose", None)
        if (
            not callable(set_shader_pack)
            or not isinstance(prebuilt_shader_configs, Mapping)
            or not isinstance(shader_config_type, type)
            or not callable(pose_type)
        ):
            raise ManiSkillVisualRenderingError(
                "installed renderer lacks the prebuilt shader or world-pose APIs"
            )
        shader_config = prebuilt_shader_configs.get(plan.shader_configuration)
        if not isinstance(shader_config, shader_config_type):
            raise ManiSkillVisualRenderingError(
                "configured prebuilt shader is unavailable or has an unexpected type"
            )
        render_system = getattr(render_module, "SAPIEN_RENDER_SYSTEM", None)
        if render_system != "3.0":
            raise ManiSkillVisualRenderingError(
                "installed renderer does not use the pinned SAPIEN render system 3.0"
            )
        raw_texture_names = getattr(shader_config, "texture_names", None)
        if not isinstance(raw_texture_names, Mapping):
            raise ManiSkillVisualRenderingError(
                "configured shader lacks a texture-name mapping"
            )
        texture_names = tuple(raw_texture_names.keys())
        if (
            not texture_names
            or "Color" not in texture_names
            or len(texture_names) != len(set(texture_names))
            or any(
                not isinstance(name, str)
                or not name
                or name != name.strip()
                or len(name) > 128
                or any(ord(character) < 32 for character in name)
                for name in texture_names
            )
        ):
            raise ManiSkillVisualRenderingError(
                "configured shader has an invalid ordered texture inventory"
            )
        scene_num_envs = getattr(scene, "num_envs", None)
        if (
            getattr(scene, "gpu_sim_enabled", None) is not True
            or getattr(scene, "parallel_in_single_scene", None) is not False
            or type(scene_num_envs) is not int
            or scene_num_envs != 1
        ):
            raise ManiSkillVisualRenderingError(
                "visual cameras require pinned single-environment GPU scene semantics"
            )
        try:
            set_shader_pack(shader_config)
        except Exception as exc:
            raise ManiSkillVisualRenderingError(
                "could not apply the configured prebuilt shader"
            ) from exc
        cameras: list[tuple[VisualCameraRenderPlan, object]] = []
        for camera in plan.cameras:
            pose = pose_type(camera.position, camera.quaternion_wxyz)
            try:
                runtime_camera = add_camera(
                    name=camera.camera_id,
                    pose=pose,
                    width=camera.width,
                    height=camera.height,
                    near=camera.near,
                    far=camera.far,
                    fovy=None,
                    intrinsic=np.array(camera.intrinsics, copy=True),
                )
            except Exception as exc:
                raise ManiSkillVisualRenderingError(
                    f"could not add world camera {camera.camera_id!r}"
                ) from exc
            cameras.append((camera, runtime_camera))
        self._apply_lighting(scene, plan.lighting)
        self._initialize_gpu_camera_groups(
            scene=scene,
            cameras=cameras,
            texture_names=texture_names,
        )
        backend = getattr(scene, "backend", None)
        observation = VisualRendererApiObservation(
            renderer_backend=_safe_runtime_name(backend),
            scene_type=_qualified_type_name(scene),
            camera_type=_qualified_type_name(cameras[0][1]),
            shader_configuration=plan.shader_configuration,
            camera_configuration_api=(
                "ManiSkillScene.add_camera(intrinsic,world_pose)+"
                "RenderSystemGroup.create_camera_group(RenderCamera._render_cameras)"
            ),
            camera_group_initialization_semantic=(
                GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC
            ),
            camera_group_texture_names=texture_names,
            camera_group_count=len(cameras),
            underlying_camera_count_per_group=1,
            camera_groups_ready=True,
            world_camera_pose_representation="sapien.Pose(position,quaternion_wxyz)",
            sensor_update_calls=(
                "scene.update_render(update_sensors=False,"
                "update_human_render_cameras=False) for group initialization",
                "render_system_group.create_camera_group("
                "camera._render_cameras,shader_texture_names)",
                "scene.update_render(update_sensors=False,"
                "update_human_render_cameras=False) for capture",
                "camera.take_picture()",
                "camera.get_picture(['Color'])",
            ),
        )
        return _InstalledRenderHandle(scene, tuple(cameras), observation)

    @staticmethod
    def _initialize_gpu_camera_groups(
        *,
        scene: object,
        cameras: Sequence[tuple[VisualCameraRenderPlan, object]],
        texture_names: tuple[str, ...],
    ) -> None:
        """Bind late-added GPU cameras to pinned SAPIEN 3.0 render groups."""
        update_render = getattr(scene, "update_render", None)
        camera_groups = getattr(scene, "camera_groups", None)
        if not callable(update_render) or not isinstance(camera_groups, MutableMapping):
            raise ManiSkillVisualRenderingError(
                "GPU scene lacks camera-group initialization APIs"
            )
        camera_ids = tuple(plan.camera_id for plan, _ in cameras)
        if any(camera_id in camera_groups for camera_id in camera_ids):
            raise ManiSkillVisualRenderingError(
                "GPU scene already contains an M4A camera-group identity"
            )
        prepared_cameras: list[tuple[object, list[object]]] = []
        missing = object()
        for plan, camera in cameras:
            existing_camera_group = getattr(camera, "camera_group", missing)
            if existing_camera_group is missing or existing_camera_group is not None:
                raise ManiSkillVisualRenderingError(
                    f"camera {plan.camera_id!r} lacks an unbound GPU camera-group slot"
                )
            render_cameras = getattr(camera, "_render_cameras", None)
            if not isinstance(render_cameras, list) or len(render_cameras) != 1:
                raise ManiSkillVisualRenderingError(
                    f"camera {plan.camera_id!r} must expose exactly one pinned "
                    "underlying render camera"
                )
            prepared_cameras.append((camera, render_cameras))
        try:
            update_render(
                update_sensors=False,
                update_human_render_cameras=False,
            )
        except Exception as exc:
            raise ManiSkillVisualRenderingError(
                "could not initialize the GPU render system group"
            ) from exc
        render_system_group = getattr(scene, "render_system_group", None)
        create_camera_group = getattr(render_system_group, "create_camera_group", None)
        if not callable(create_camera_group):
            raise ManiSkillVisualRenderingError(
                "GPU render system lacks the camera-group factory"
            )
        created_groups: list[object] = []
        for (plan, _), (_, render_cameras) in zip(
            cameras, prepared_cameras, strict=True
        ):
            try:
                group = create_camera_group(render_cameras, list(texture_names))
            except Exception as exc:
                raise ManiSkillVisualRenderingError(
                    f"could not create GPU camera group for {plan.camera_id!r}"
                ) from exc
            if (
                group is None
                or any(group is existing for existing in created_groups)
                or not callable(getattr(group, "take_picture", None))
                or not callable(getattr(group, "get_picture_cuda", None))
            ):
                raise ManiSkillVisualRenderingError(
                    f"GPU camera group for {plan.camera_id!r} is invalid"
                )
            created_groups.append(group)
        assigned: list[tuple[str, object]] = []
        try:
            for (plan, camera), group in zip(cameras, created_groups, strict=True):
                runtime_camera = cast(Any, camera)
                runtime_camera.camera_group = group
                if getattr(camera, "camera_group", None) is not group:
                    raise ManiSkillVisualRenderingError(
                        f"GPU camera group assignment for {plan.camera_id!r} failed"
                    )
                assigned.append((plan.camera_id, camera))
                camera_groups[plan.camera_id] = group
                if camera_groups.get(plan.camera_id) is not group:
                    raise ManiSkillVisualRenderingError(
                        f"GPU camera group registry for {plan.camera_id!r} failed"
                    )
        except Exception as exc:
            for camera_id, camera in assigned:
                camera_groups.pop(camera_id, None)
                try:
                    runtime_camera = cast(Any, camera)
                    runtime_camera.camera_group = None
                except Exception:
                    pass
            if isinstance(exc, ManiSkillVisualRenderingError):
                raise
            raise ManiSkillVisualRenderingError(
                "could not bind GPU camera groups to runtime cameras"
            ) from exc

    @staticmethod
    def _apply_lighting(scene: object, plan: VisualLightingRenderPlan) -> None:
        set_ambient = getattr(scene, "set_ambient_light", None)
        add_directional = getattr(scene, "add_directional_light", None)
        if not callable(set_ambient) or not callable(add_directional):
            raise ManiSkillVisualRenderingError(
                "PickCube scene lacks required render-only lighting APIs"
            )
        ambient = [plan.ambient_intensity] * 3
        key_color = [item * plan.key_intensity for item in plan.key_color_rgb]
        try:
            set_ambient(ambient)
            add_directional(
                direction=list(plan.key_direction),
                color=key_color,
                shadow=False,
            )
        except Exception as exc:
            raise ManiSkillVisualRenderingError(
                "could not apply render-only lighting before GPU group initialization"
            ) from exc


def build_pickcube_visual_render_plan(
    rig: PickCubeMultiViewRigV1,
    domain: RenderDomainV1,
    configuration: RenderDomainConfigurationV1,
    render_seed: int,
) -> PickCubeVisualRenderPlan:
    """Resolve one deterministic project-owned rig/domain render plan."""
    if type(render_seed) is not int or not 0 <= render_seed < 2**32:
        raise ManiSkillVisualRenderingError("render seed must be uint32")
    configured_domain = configuration.domain(domain.domain_id)
    if configured_domain.content_digest != domain.content_digest:
        raise ManiSkillVisualRenderingError(
            "render domain differs from its frozen configuration"
        )
    generator = np.random.Generator(np.random.PCG64(render_seed))
    camera_offsets_are_zero = all(
        value == 0.0
        for vector in (
            domain.camera_translation_min,
            domain.camera_translation_max,
            domain.camera_rotation_rpy_min_degrees,
            domain.camera_rotation_rpy_max_degrees,
        )
        for value in vector
    )
    cameras: list[VisualCameraRenderPlan] = []
    for base in rig.cameras:
        if camera_offsets_are_zero:
            position = base.world_pose.position
            quaternion = base.world_pose.quaternion_wxyz
            extrinsics = np.asarray(base.extrinsics, dtype=np.float64)
            camera_digest = base.camera_configuration_digest
        else:
            translation = _uniform_vector(
                generator,
                domain.camera_translation_min,
                domain.camera_translation_max,
            )
            rotation_rpy = _uniform_vector(
                generator,
                domain.camera_rotation_rpy_min_degrees,
                domain.camera_rotation_rpy_max_degrees,
            )
            position = cast(
                tuple[float, float, float],
                tuple(
                    float(value + offset)
                    for value, offset in zip(
                        base.world_pose.position, translation, strict=True
                    )
                ),
            )
            delta = _quaternion_from_rpy_degrees(rotation_rpy)
            quaternion = _normalized_quaternion(
                _quaternion_multiply(delta, base.world_pose.quaternion_wxyz)
            )
            extrinsics = _world_to_camera_extrinsics(position, quaternion)
            resolved_camera = CameraConfigurationV1(
                camera_id=base.camera_id,
                width=base.width,
                height=base.height,
                near=base.near,
                far=base.far,
                fov_y_degrees=base.fov_y_degrees,
                world_pose=CameraPoseV1(
                    position=position,
                    quaternion_wxyz=quaternion,
                ),
                intrinsics=base.intrinsics,
                extrinsics=cast(
                    Matrix4,
                    tuple(
                        tuple(float(item) for item in row)
                        for row in extrinsics.tolist()
                    ),
                ),
            )
            camera_digest = resolved_camera.camera_configuration_digest
        cameras.append(
            VisualCameraRenderPlan(
                camera_id=base.camera_id,
                width=base.width,
                height=base.height,
                near=base.near,
                far=base.far,
                fov_y_degrees=base.fov_y_degrees,
                position=position,
                quaternion_wxyz=quaternion,
                intrinsics=np.asarray(base.intrinsics, dtype=np.float64),
                extrinsics=extrinsics,
                camera_configuration_digest=camera_digest,
            )
        )
    return PickCubeVisualRenderPlan(
        domain_id=domain.domain_id,
        render_seed=render_seed,
        shader_configuration=configuration.shader_configuration,
        cameras=tuple(cameras),
        lighting=VisualLightingRenderPlan(
            ambient_intensity=domain.lighting.ambient_intensity,
            key_intensity=domain.lighting.key_intensity,
            key_color_rgb=domain.lighting.key_color_rgb,
            key_direction=domain.lighting.key_direction,
        ),
    )


def _uniform_vector(
    generator: np.random.Generator,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> tuple[float, float, float]:
    sampled = generator.uniform(
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    return (float(sampled[0]), float(sampled[1]), float(sampled[2]))


def _quaternion_from_rpy_degrees(
    rpy_degrees: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(value) for value in rpy_degrees)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return _normalized_quaternion(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _normalized_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(item * item for item in value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ManiSkillVisualRenderingError("resolved camera quaternion is invalid")
    return cast(tuple[float, float, float, float], tuple(item / norm for item in value))


def _world_to_camera_extrinsics(
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    return np.asarray(
        opencv_world_to_camera_extrinsics(position, quaternion_wxyz),
        dtype=np.float64,
    )


def _runtime_array(value: object) -> NDArray[Any]:
    candidate = value
    for method_name in ("detach", "cpu"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    return np.asarray(candidate)


def _require_pinned_runtime_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> tuple[NDArray[Any], str, object]:
    """Require one finite float32 CUDA torch tensor and detach exact bytes."""

    if _qualified_type_name(value) != "torch.Tensor":
        raise ManiSkillVisualContractError(f"runtime {field_name} must be torch.Tensor")
    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    if device_type != "cuda":
        raise ManiSkillVisualContractError(
            f"runtime {field_name} must remain on a CUDA device"
        )
    array = _runtime_array(value)
    if array.dtype != np.dtype(np.float32):
        raise ManiSkillVisualContractError(
            f"runtime {field_name} must use pinned float32"
        )
    return (
        _freeze_array(array, field_name=f"runtime {field_name}", shape=shape),
        device_type,
        device,
    )


def _require_array_bits_equal(
    observed: NDArray[Any],
    expected: NDArray[Any],
    *,
    field_name: str,
) -> None:
    """Require identical shape, dtype, and C-order numeric bit pattern."""

    if (
        observed.shape != expected.shape
        or observed.dtype != expected.dtype
        or observed.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ManiSkillVisualContractError(
            f"runtime {field_name} differs at the bit level"
        )


def runtime_calibration_array_digest(value: NDArray[Any]) -> str:
    """Content-bind one exact runtime calibration array without runtime paths."""

    array = np.asarray(value)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ManiSkillVisualRenderingError(
            "runtime calibration digest requires a numeric array"
        )
    digest = hashlib.sha256()
    digest.update(b"latentguard-runtime-calibration-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _convert_color_to_rgb_uint8(
    color: NDArray[Any],
) -> tuple[NDArray[np.uint8], str, str]:
    raw_dtype = color.dtype.name
    if color.dtype == np.dtype(np.uint8):
        return (
            np.array(color[..., :3], copy=True, order="C"),
            raw_dtype,
            UINT8_COLOR_TO_RGB_UINT8_SEMANTIC,
        )
    if not np.issubdtype(color.dtype, np.floating) or color.dtype.itemsize not in {
        2,
        4,
        8,
    }:
        raise ManiSkillVisualContractError(
            f"renderer Color texture dtype {color.dtype} is unsupported"
        )
    if not bool(np.all(np.isfinite(color))):
        raise ManiSkillVisualContractError("renderer Color texture is non-finite")
    rgb = color[..., :3]
    if bool(np.any(rgb < 0.0)) or bool(np.any(rgb > 1.0)):
        raise ManiSkillVisualContractError(
            "renderer floating Color texture is outside the authorized [0,1] range"
        )
    converted = (rgb * 255.0).astype(np.uint8, copy=False)
    return (
        np.array(converted, copy=True, order="C"),
        raw_dtype,
        FLOAT_COLOR_TO_RGB_UINT8_SEMANTIC,
    )


def _require_runtime_calibration_matches_plan(
    observed: NDArray[Any],
    planned: NDArray[Any],
    *,
    field_name: str,
) -> None:
    """Require exact planned calibration after the runtime's numeric cast."""

    if not np.issubdtype(observed.dtype, np.floating):
        raise ManiSkillVisualContractError(
            f"runtime {field_name} must use a floating dtype"
        )
    expected = np.asarray(planned, dtype=observed.dtype)
    try:
        _require_array_bits_equal(observed, expected, field_name=field_name)
    except ManiSkillVisualContractError as exc:
        raise ManiSkillVisualContractError(
            f"runtime {field_name} differs from the content-bound camera plan"
        ) from exc


def _single_matrix(
    value: object, shape: tuple[int, int], field_name: str
) -> NDArray[Any]:
    array = _runtime_array(value)
    if array.shape == (1, *shape):
        array = array[0]
    return _freeze_array(array, field_name=field_name, shape=shape)


def _single_extrinsic_matrix(
    value: object,
) -> tuple[NDArray[Any], str, str]:
    array = _runtime_array(value)
    raw_shape = array.shape
    if array.shape in {(1, 3, 4), (1, 4, 4)}:
        array = array[0]
    if array.shape == (3, 4):
        homogeneous = np.eye(4, dtype=array.dtype)
        homogeneous[:3, :] = array
        return (
            _freeze_array(homogeneous, field_name="runtime extrinsics", shape=(4, 4)),
            "[1,3,4]" if raw_shape == (1, 3, 4) else "[3,4]",
            EXTRINSIC_3X4_TO_4X4_SEMANTIC,
        )
    if array.shape == (4, 4):
        return (
            _freeze_array(array, field_name="runtime extrinsics", shape=(4, 4)),
            "[1,4,4]" if raw_shape == (1, 4, 4) else "[4,4]",
            EXTRINSIC_4X4_SEMANTIC,
        )
    raise ManiSkillVisualRenderingError(
        f"runtime extrinsics must be [1,3,4] or [1,4,4], got {array.shape}"
    )


def _qualified_type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_runtime_name(value: object) -> str:
    text = str(value)
    if not text or any(marker in text for marker in ("/", "\\", "@", "0x")):
        return _qualified_type_name(value)
    return text


__all__ = [
    "CALIBRATION_COMPARISON_SEMANTIC",
    "CAMERA_POSE_APPLICATION_SEMANTIC",
    "EXTRINSIC_3X4_TO_4X4_SEMANTIC",
    "EXTRINSIC_4X4_SEMANTIC",
    "FLOAT_COLOR_TO_RGB_UINT8_SEMANTIC",
    "GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC",
    "LIGHTING_APPLICATION_SEMANTIC",
    "PACKET_EXTRINSIC_CANONICALIZATION_SEMANTIC",
    "LazyManiSkillPickCubeVisualRenderer",
    "ManiSkillVisualContractError",
    "ManiSkillVisualRenderingError",
    "PickCubeVisualRenderHandle",
    "PickCubeVisualRenderPlan",
    "PickCubeVisualRenderer",
    "RGB_CHANNEL_ORDER",
    "RGB_COLOR_SPACE_ASSUMPTION",
    "RUNTIME_CALIBRATION_EVIDENCE_SEMANTIC",
    "RUNTIME_CAMERA_POSE_VERIFICATION_SEMANTIC",
    "RUNTIME_PUBLIC_EXTRINSIC_DERIVATION_SEMANTIC",
    "UNDERLYING_CAMERA_POSE_VERIFICATION_SEMANTIC",
    "RenderedVisualView",
    "RuntimeCalibrationEvidenceV1",
    "UINT8_COLOR_TO_RGB_UINT8_SEMANTIC",
    "VERTICAL_ORIENTATION_CONVENTION",
    "VISUAL_CAMERA_RESOLUTION_SEMANTIC",
    "VISUAL_RENDERER_SEMANTIC_VERSION",
    "VisualCameraRenderPlan",
    "VisualLightingRenderPlan",
    "VisualRendererApiObservation",
    "build_pickcube_visual_render_plan",
    "runtime_calibration_array_digest",
]
