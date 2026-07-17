"""Project-owned fixed three-view camera-rig contracts for M4A."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from latentguard.vision_data.configuration import (
    require_exact_fields,
    require_integer,
    require_list,
    require_mapping,
    require_matrix,
    require_number,
    require_text,
    require_vector,
)

CAMERA_SCHEMA_VERSION = "1.0"
CAMERA_RIG_SCHEMA_VERSION = "1.0"
PICKCUBE_MULTIVIEW_RIG_ID = "pickcube_multiview_rig_v1"
PICKCUBE_MULTIVIEW_RIG_SEMANTIC_VERSION = "1.0.0"
PICKCUBE_CAMERA_IDS = ("front_oblique", "overhead", "side_oblique")
M4A_IMAGE_WIDTH = 224
M4A_IMAGE_HEIGHT = 224
M4A_IMAGE_CHANNELS = 3
M4A_IMAGE_DTYPE = "uint8"
M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE = 1e-9

Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Matrix4 = tuple[Vector4, Vector4, Vector4, Vector4]


def expected_pinhole_focal_lengths(
    width: int, height: int, fov_y_degrees: float
) -> tuple[float, float]:
    """Return square-pixel focal lengths implied by vertical FOV and aspect."""

    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or type(fov_y_degrees) not in (int, float)
        or not math.isfinite(float(fov_y_degrees))
        or not 0.0 < float(fov_y_degrees) < 180.0
    ):
        raise ValueError("pinhole projection requires positive dimensions and FOV")
    half_tangent_y = math.tan(math.radians(float(fov_y_degrees)) / 2.0)
    aspect = float(width) / float(height)
    half_tangent_x = aspect * half_tangent_y
    focal_y = float(height) / (2.0 * half_tangent_y)
    focal_x = float(width) / (2.0 * half_tangent_x)
    return focal_x, focal_y


@dataclass(frozen=True, slots=True)
class CameraPoseV1:
    """Explicit world-frame camera position and WXYZ quaternion."""

    position: Vector3
    quaternion_wxyz: Vector4

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", tuple(self.position))
        object.__setattr__(self, "quaternion_wxyz", tuple(self.quaternion_wxyz))
        from latentguard.vision_data.validation import validate_camera_pose

        validate_camera_pose(self)

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-ready world-pose fields."""

        return {
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
        }

    @classmethod
    def from_mapping(cls, value: object) -> CameraPoseV1:
        """Decode an exact-field world-pose mapping."""

        item = require_mapping(value, "CameraPoseV1")
        require_exact_fields(item, {"position", "quaternion_wxyz"}, "CameraPoseV1")
        return cls(
            position=cast(
                Vector3, require_vector(item["position"], 3, "CameraPoseV1.position")
            ),
            quaternion_wxyz=cast(
                Vector4,
                require_vector(
                    item["quaternion_wxyz"], 4, "CameraPoseV1.quaternion_wxyz"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class CameraConfigurationV1:
    """One fixed world camera with explicit projection matrices."""

    camera_id: str
    width: int
    height: int
    near: float
    far: float
    fov_y_degrees: float
    world_pose: CameraPoseV1
    intrinsics: Matrix3
    extrinsics: Matrix4
    schema_version: str = CAMERA_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "camera_id",
            "width",
            "height",
            "near",
            "far",
            "fov_y_degrees",
            "world_pose",
            "intrinsics",
            "extrinsics",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "intrinsics", tuple(tuple(row) for row in self.intrinsics)
        )
        object.__setattr__(
            self, "extrinsics", tuple(tuple(row) for row in self.extrinsics)
        )
        from latentguard.vision_data.validation import validate_camera_configuration

        validate_camera_configuration(self)

    @property
    def content_digest(self) -> str:
        """Return the path-independent camera configuration digest."""

        from latentguard.vision_data.identity import compute_model_content_digest

        return compute_model_content_digest(
            self, context="CameraConfigurationV1Content"
        )

    @property
    def camera_configuration_digest(self) -> str:
        """Alias used by visual packet records."""

        return self.content_digest

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-ready camera fields."""

        return {
            "camera_id": self.camera_id,
            "extrinsics": [list(row) for row in self.extrinsics],
            "far": self.far,
            "fov_y_degrees": self.fov_y_degrees,
            "height": self.height,
            "intrinsics": [list(row) for row in self.intrinsics],
            "near": self.near,
            "width": self.width,
            "world_pose": self.world_pose.as_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> CameraConfigurationV1:
        """Decode one exact-field camera mapping."""

        item = require_mapping(value, "CameraConfigurationV1")
        require_exact_fields(item, cls._FIELDS, "CameraConfigurationV1")
        return cls(
            camera_id=require_text(
                item["camera_id"], "CameraConfigurationV1.camera_id"
            ),
            width=require_integer(
                item["width"], "CameraConfigurationV1.width", minimum=1
            ),
            height=require_integer(
                item["height"], "CameraConfigurationV1.height", minimum=1
            ),
            near=require_number(item["near"], "CameraConfigurationV1.near"),
            far=require_number(item["far"], "CameraConfigurationV1.far"),
            fov_y_degrees=require_number(
                item["fov_y_degrees"], "CameraConfigurationV1.fov_y_degrees"
            ),
            world_pose=CameraPoseV1.from_mapping(item["world_pose"]),
            intrinsics=cast(
                Matrix3,
                require_matrix(
                    item["intrinsics"], 3, 3, "CameraConfigurationV1.intrinsics"
                ),
            ),
            extrinsics=cast(
                Matrix4,
                require_matrix(
                    item["extrinsics"], 4, 4, "CameraConfigurationV1.extrinsics"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class PickCubeMultiViewRigV1:
    """Versioned fixed front, overhead, and side PickCube camera rig."""

    rig_id: str
    semantic_version: str
    cameras: tuple[CameraConfigurationV1, ...]
    schema_version: str = CAMERA_RIG_SCHEMA_VERSION

    _FIELDS = frozenset({"schema_version", "rig_id", "semantic_version", "cameras"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", tuple(self.cameras))
        from latentguard.vision_data.validation import validate_camera_rig

        validate_camera_rig(self)

    @property
    def content_digest(self) -> str:
        """Return the complete path-independent rig identity."""

        from latentguard.vision_data.identity import compute_model_content_digest

        return compute_model_content_digest(
            self, context="PickCubeMultiViewRigV1Content"
        )

    @property
    def rig_digest(self) -> str:
        """Alias used by packet and dataset bindings."""

        return self.content_digest

    def as_mapping(self) -> dict[str, object]:
        """Return the checked-in camera-rig configuration schema."""

        return {
            "cameras": [camera.as_mapping() for camera in self.cameras],
            "rig_id": self.rig_id,
            "schema_version": self.schema_version,
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> PickCubeMultiViewRigV1:
        """Decode the strict checked-in camera-rig configuration."""

        item = require_mapping(value, "PickCubeMultiViewRigV1")
        require_exact_fields(item, cls._FIELDS, "PickCubeMultiViewRigV1")
        cameras = tuple(
            CameraConfigurationV1.from_mapping(camera)
            for camera in require_list(
                item["cameras"], "PickCubeMultiViewRigV1.cameras"
            )
        )
        return cls(
            rig_id=require_text(item["rig_id"], "PickCubeMultiViewRigV1.rig_id"),
            semantic_version=require_text(
                item["semantic_version"], "PickCubeMultiViewRigV1.semantic_version"
            ),
            cameras=cameras,
            schema_version=require_text(
                item["schema_version"], "PickCubeMultiViewRigV1.schema_version"
            ),
        )


__all__ = [
    "CAMERA_RIG_SCHEMA_VERSION",
    "CAMERA_SCHEMA_VERSION",
    "M4A_IMAGE_CHANNELS",
    "M4A_IMAGE_DTYPE",
    "M4A_IMAGE_HEIGHT",
    "M4A_IMAGE_WIDTH",
    "M4A_PINHOLE_PROJECTION_ABSOLUTE_TOLERANCE",
    "PICKCUBE_CAMERA_IDS",
    "PICKCUBE_MULTIVIEW_RIG_ID",
    "PICKCUBE_MULTIVIEW_RIG_SEMANTIC_VERSION",
    "CameraConfigurationV1",
    "CameraPoseV1",
    "Matrix3",
    "Matrix4",
    "PickCubeMultiViewRigV1",
    "Vector3",
    "Vector4",
    "expected_pinhole_focal_lengths",
]
