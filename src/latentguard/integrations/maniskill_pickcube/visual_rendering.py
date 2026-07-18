"""Lazy ManiSkill/SAPIEN multi-view RGB rendering for PickCube.

This module owns every simulator-native camera, pose, shader, light, and tensor
object used by M4A.  Importing it never imports ManiSkill, SAPIEN, or Torch.
"""

from __future__ import annotations

import importlib
import math
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
)
from latentguard.vision_data.domains import (
    RenderDomainConfigurationV1,
    RenderDomainV1,
)

VISUAL_RENDERER_SEMANTIC_VERSION = "maniskill_pickcube_multiview_rgb_v2"
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
CALIBRATION_COMPARISON_SEMANTIC = "exact_after_runtime_dtype_cast_v1"
EXTRINSIC_3X4_TO_4X4_SEMANTIC = (
    "mani_skill_public_opencv_world_to_camera_3x4_to_homogeneous_4x4_v1"
)
EXTRINSIC_4X4_SEMANTIC = "public_homogeneous_world_to_camera_4x4_identity_v1"


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
        position = _finite_tuple(self.position, 3, field_name="camera position")
        quaternion = _finite_tuple(
            self.quaternion_wxyz, 4, field_name="camera quaternion"
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


@dataclass(frozen=True, slots=True, eq=False)
class RenderedVisualView:
    """One detached authoritative RGB result and runtime calibration."""

    camera_id: str
    rgb: NDArray[Any]
    intrinsics: NDArray[Any]
    extrinsics: NDArray[Any]
    camera_configuration_digest: str

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
            field_name=f"{self.camera_id} extrinsics",
            shape=(4, 4),
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
            ("[3,4]", EXTRINSIC_3X4_TO_4X4_SEMANTIC),
            ("[4,4]", EXTRINSIC_4X4_SEMANTIC),
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
                "rgb_conversion_semantic": self.rgb_conversion_semantic,
                "runtime_extrinsics_dtype": self.runtime_extrinsics_dtype,
                "runtime_extrinsics_semantic": self.runtime_extrinsics_semantic,
                "runtime_intrinsics_dtype": self.runtime_intrinsics_dtype,
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
            take_picture()
            textures = get_picture(["Color"])
            if not isinstance(textures, Sequence) or len(textures) != 1:
                raise ManiSkillVisualRenderingError(
                    f"camera {plan.camera_id!r} returned an invalid Color texture"
                )
            color = _runtime_array(textures[0])
            if color.shape == (1, 224, 224, 4):
                color = color[0]
            if color.shape != (224, 224, 4):
                raise ManiSkillVisualRenderingError(
                    f"camera {plan.camera_id!r} Color must be [1,224,224,4]"
                )
            rgb, raw_color_dtype, rgb_conversion = _convert_color_to_rgb_uint8(color)
            intrinsics = _single_matrix(get_intrinsic(), (3, 3), "intrinsics")
            (
                extrinsics,
                raw_extrinsic_shape,
                extrinsic_semantic,
            ) = _single_extrinsic_matrix(get_extrinsic())
            _require_runtime_calibration_matches_plan(
                intrinsics,
                plan.intrinsics,
                field_name=f"{plan.camera_id} intrinsics",
            )
            _require_runtime_calibration_matches_plan(
                extrinsics,
                plan.extrinsics,
                field_name=f"{plan.camera_id} extrinsics",
            )
            observation = replace(
                self._api_observation,
                raw_color_texture_dtype=raw_color_dtype,
                rgb_conversion_semantic=rgb_conversion,
                runtime_intrinsics_dtype=intrinsics.dtype.name,
                runtime_extrinsics_dtype=extrinsics.dtype.name,
                raw_extrinsic_matrix_shape=raw_extrinsic_shape,
                runtime_extrinsics_semantic=extrinsic_semantic,
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
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    camera_configuration_digest=plan.camera_configuration_digest,
                )
            )
        return tuple(results)


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
    w, x, y, z = quaternion_wxyz
    camera_to_world = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    world_to_camera = camera_to_world.T
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = world_to_camera
    extrinsics[:3, 3] = -world_to_camera @ np.asarray(position, dtype=np.float64)
    return extrinsics


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
    if observed.shape != expected.shape or not np.array_equal(observed, expected):
        raise ManiSkillVisualContractError(
            f"runtime {field_name} differs from the content-bound camera plan"
        )


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
    if array.shape in {(1, 3, 4), (1, 4, 4)}:
        array = array[0]
    if array.shape == (3, 4):
        homogeneous = np.eye(4, dtype=array.dtype)
        homogeneous[:3, :] = array
        return (
            _freeze_array(homogeneous, field_name="runtime extrinsics", shape=(4, 4)),
            "[3,4]",
            EXTRINSIC_3X4_TO_4X4_SEMANTIC,
        )
    if array.shape == (4, 4):
        return (
            _freeze_array(array, field_name="runtime extrinsics", shape=(4, 4)),
            "[4,4]",
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
    "LazyManiSkillPickCubeVisualRenderer",
    "ManiSkillVisualContractError",
    "ManiSkillVisualRenderingError",
    "PickCubeVisualRenderHandle",
    "PickCubeVisualRenderPlan",
    "PickCubeVisualRenderer",
    "RGB_CHANNEL_ORDER",
    "RGB_COLOR_SPACE_ASSUMPTION",
    "RenderedVisualView",
    "UINT8_COLOR_TO_RGB_UINT8_SEMANTIC",
    "VERTICAL_ORIENTATION_CONVENTION",
    "VISUAL_CAMERA_RESOLUTION_SEMANTIC",
    "VISUAL_RENDERER_SEMANTIC_VERSION",
    "VisualCameraRenderPlan",
    "VisualLightingRenderPlan",
    "VisualRendererApiObservation",
    "build_pickcube_visual_render_plan",
]
