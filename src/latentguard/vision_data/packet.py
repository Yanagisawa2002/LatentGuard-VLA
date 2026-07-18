"""Content-bound visual packet and per-view integrity records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from latentguard.vision_data.cameras import Matrix3, Matrix4
from latentguard.vision_data.configuration import (
    VisionConfigurationError,
    require_exact_fields,
    require_integer,
    require_list,
    require_mapping,
    require_matrix,
    require_number,
    require_text,
)
from latentguard.vision_data.models import (
    VISION_DATA_SCHEMA_VERSION,
    SourceCollection,
    VisualDatasetSplit,
    VisualTaskProjectionV1,
)

VISUAL_VIEW_SCHEMA_VERSION = "1.1"
VISUAL_PACKET_SCHEMA_VERSION = VISION_DATA_SCHEMA_VERSION
M4A_STATE_COMPONENT_COUNT = 70
M4A_VERIFIER_STATE_COMPONENT_COUNT = 38
M4A_STATE_COMPARISON_TOLERANCE = 1e-6
M4A_VERIFIER_STATE_COMPARISON_TOLERANCE = 0.0


def _require_boolean(value: object, context: str) -> bool:
    """Require a JSON boolean without truthiness coercion."""

    if type(value) is not bool:
        raise VisionConfigurationError(f"{context}: expected boolean")
    return value


@dataclass(frozen=True, slots=True)
class VisualViewRecordV1:
    """One exact authoritative RGB view and render-integrity record."""

    camera_id: str
    image_reference: str
    pixel_sha256: str
    npy_sha256: str
    dtype: str
    shape: tuple[int, int, int]
    intrinsics: Matrix3
    intrinsics_dtype: str
    extrinsics: Matrix4
    extrinsics_dtype: str
    runtime_extrinsics_digest: str
    expected_runtime_extrinsics_digest: str
    camera_configuration_digest: str
    state_before_render_digest: str
    state_after_render_digest: str
    compared_state_component_count: int
    maximum_state_error: float
    task_projection_before: VisualTaskProjectionV1
    task_projection_after: VisualTaskProjectionV1
    schema_version: str = VISUAL_VIEW_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "camera_id",
            "image_reference",
            "pixel_sha256",
            "npy_sha256",
            "dtype",
            "shape",
            "intrinsics",
            "intrinsics_dtype",
            "extrinsics",
            "extrinsics_dtype",
            "runtime_extrinsics_digest",
            "expected_runtime_extrinsics_digest",
            "camera_configuration_digest",
            "state_before_render_digest",
            "state_after_render_digest",
            "compared_state_component_count",
            "maximum_state_error",
            "task_projection_before",
            "task_projection_after",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(
            self, "intrinsics", tuple(tuple(row) for row in self.intrinsics)
        )
        object.__setattr__(
            self, "extrinsics", tuple(tuple(row) for row in self.extrinsics)
        )
        from latentguard.vision_data.validation import validate_view_record

        validate_view_record(self)

    @property
    def content_digest(self) -> str:
        """Return the complete view record digest including pixel content."""

        from latentguard.vision_data.identity import compute_visual_view_content_digest

        return compute_visual_view_content_digest(self)

    def as_mapping(self) -> dict[str, object]:
        """Return canonical JSON-ready view fields."""

        return {
            "camera_configuration_digest": self.camera_configuration_digest,
            "camera_id": self.camera_id,
            "compared_state_component_count": self.compared_state_component_count,
            "dtype": self.dtype,
            "extrinsics": [list(row) for row in self.extrinsics],
            "extrinsics_dtype": self.extrinsics_dtype,
            "expected_runtime_extrinsics_digest": (
                self.expected_runtime_extrinsics_digest
            ),
            "image_reference": self.image_reference,
            "intrinsics": [list(row) for row in self.intrinsics],
            "intrinsics_dtype": self.intrinsics_dtype,
            "maximum_state_error": self.maximum_state_error,
            "npy_sha256": self.npy_sha256,
            "pixel_sha256": self.pixel_sha256,
            "runtime_extrinsics_digest": self.runtime_extrinsics_digest,
            "schema_version": self.schema_version,
            "shape": list(self.shape),
            "state_after_render_digest": self.state_after_render_digest,
            "state_before_render_digest": self.state_before_render_digest,
            "task_projection_after": self.task_projection_after.as_mapping(),
            "task_projection_before": self.task_projection_before.as_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> VisualViewRecordV1:
        """Decode one exact-field view record."""

        item = require_mapping(value, "VisualViewRecordV1")
        require_exact_fields(item, cls._FIELDS, "VisualViewRecordV1")
        raw_shape = require_list(item["shape"], "VisualViewRecordV1.shape")
        if len(raw_shape) != 3:
            raise VisionConfigurationError(
                "VisualViewRecordV1.shape: expected 3 values"
            )
        return cls(
            camera_id=require_text(item["camera_id"], "VisualViewRecordV1.camera_id"),
            image_reference=require_text(
                item["image_reference"], "VisualViewRecordV1.image_reference"
            ),
            pixel_sha256=require_text(
                item["pixel_sha256"], "VisualViewRecordV1.pixel_sha256"
            ),
            npy_sha256=require_text(
                item["npy_sha256"], "VisualViewRecordV1.npy_sha256"
            ),
            dtype=require_text(item["dtype"], "VisualViewRecordV1.dtype"),
            shape=cast(
                tuple[int, int, int],
                tuple(
                    require_integer(
                        entry, f"VisualViewRecordV1.shape[{index}]", minimum=1
                    )
                    for index, entry in enumerate(raw_shape)
                ),
            ),
            intrinsics=cast(
                Matrix3,
                require_matrix(
                    item["intrinsics"], 3, 3, "VisualViewRecordV1.intrinsics"
                ),
            ),
            intrinsics_dtype=require_text(
                item["intrinsics_dtype"], "VisualViewRecordV1.intrinsics_dtype"
            ),
            extrinsics=cast(
                Matrix4,
                require_matrix(
                    item["extrinsics"], 4, 4, "VisualViewRecordV1.extrinsics"
                ),
            ),
            extrinsics_dtype=require_text(
                item["extrinsics_dtype"], "VisualViewRecordV1.extrinsics_dtype"
            ),
            runtime_extrinsics_digest=require_text(
                item["runtime_extrinsics_digest"],
                "VisualViewRecordV1.runtime_extrinsics_digest",
            ),
            expected_runtime_extrinsics_digest=require_text(
                item["expected_runtime_extrinsics_digest"],
                "VisualViewRecordV1.expected_runtime_extrinsics_digest",
            ),
            camera_configuration_digest=require_text(
                item["camera_configuration_digest"],
                "VisualViewRecordV1.camera_configuration_digest",
            ),
            state_before_render_digest=require_text(
                item["state_before_render_digest"],
                "VisualViewRecordV1.state_before_render_digest",
            ),
            state_after_render_digest=require_text(
                item["state_after_render_digest"],
                "VisualViewRecordV1.state_after_render_digest",
            ),
            compared_state_component_count=require_integer(
                item["compared_state_component_count"],
                "VisualViewRecordV1.compared_state_component_count",
                minimum=1,
            ),
            maximum_state_error=require_number(
                item["maximum_state_error"], "VisualViewRecordV1.maximum_state_error"
            ),
            task_projection_before=VisualTaskProjectionV1.from_mapping(
                item["task_projection_before"]
            ),
            task_projection_after=VisualTaskProjectionV1.from_mapping(
                item["task_projection_after"]
            ),
            schema_version=require_text(
                item["schema_version"], "VisualViewRecordV1.schema_version"
            ),
        )


@dataclass(frozen=True, slots=True)
class VisualObservationPacketV1:
    """Three ordered RGB views rendered from one verified pre-action state."""

    source_collection: SourceCollection
    source_trajectory_id: str
    anchor_id: str
    split: VisualDatasetSplit
    split_group_id: str
    state_reference_id: str
    expected_state_digest: str
    verifier_state_semantic: str
    verifier_state_digest: str
    verifier_state_component_count: int
    verifier_state_maximum_absolute_error: float
    elapsed_simulation_steps_before: int
    elapsed_simulation_steps_after: int
    environment_close_passed: bool
    camera_rig_id: str
    camera_rig_digest: str
    render_domain_id: str
    render_domain_digest: str
    render_seed: int
    views: tuple[VisualViewRecordV1, ...]
    visual_compatibility_identity: str
    pickcube_compatibility_identity: str
    renderer_semantic_version: str
    schema_version: str = VISUAL_PACKET_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "packet_id",
            "source_collection",
            "source_trajectory_id",
            "anchor_id",
            "split",
            "split_group_id",
            "state_reference_id",
            "expected_state_digest",
            "verifier_state_semantic",
            "verifier_state_digest",
            "verifier_state_component_count",
            "verifier_state_maximum_absolute_error",
            "elapsed_simulation_steps_before",
            "elapsed_simulation_steps_after",
            "environment_close_passed",
            "camera_rig_id",
            "camera_rig_digest",
            "render_domain_id",
            "render_domain_digest",
            "render_seed",
            "views",
            "visual_compatibility_identity",
            "pickcube_compatibility_identity",
            "renderer_semantic_version",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", tuple(self.views))
        from latentguard.vision_data.validation import validate_visual_packet

        validate_visual_packet(self)

    @property
    def packet_id(self) -> str:
        """Return the semantic packet ID excluding produced image content."""

        from latentguard.vision_data.identity import compute_visual_packet_identifier

        return compute_visual_packet_identifier(self)

    @property
    def image_content_digests(self) -> tuple[str, ...]:
        """Return exact ordered raw-pixel content digests."""

        return tuple(view.pixel_sha256 for view in self.views)

    @property
    def content_digest(self) -> str:
        """Bind packet semantics and every exact ordered view record."""

        from latentguard.vision_data.identity import canonical_sha256

        return canonical_sha256(
            self.as_mapping(), context="VisualObservationPacketV1Content"
        )

    def as_mapping(self) -> dict[str, object]:
        """Return canonical packet fields including content-derived packet ID."""

        return {
            "anchor_id": self.anchor_id,
            "camera_rig_digest": self.camera_rig_digest,
            "camera_rig_id": self.camera_rig_id,
            "expected_state_digest": self.expected_state_digest,
            "elapsed_simulation_steps_after": self.elapsed_simulation_steps_after,
            "elapsed_simulation_steps_before": self.elapsed_simulation_steps_before,
            "environment_close_passed": self.environment_close_passed,
            "packet_id": self.packet_id,
            "pickcube_compatibility_identity": self.pickcube_compatibility_identity,
            "render_domain_digest": self.render_domain_digest,
            "render_domain_id": self.render_domain_id,
            "render_seed": self.render_seed,
            "renderer_semantic_version": self.renderer_semantic_version,
            "schema_version": self.schema_version,
            "source_collection": self.source_collection.value,
            "source_trajectory_id": self.source_trajectory_id,
            "split": self.split.value,
            "split_group_id": self.split_group_id,
            "state_reference_id": self.state_reference_id,
            "verifier_state_digest": self.verifier_state_digest,
            "verifier_state_component_count": self.verifier_state_component_count,
            "verifier_state_maximum_absolute_error": (
                self.verifier_state_maximum_absolute_error
            ),
            "verifier_state_semantic": self.verifier_state_semantic,
            "views": [view.as_mapping() for view in self.views],
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }

    @classmethod
    def from_mapping(cls, value: object) -> VisualObservationPacketV1:
        """Decode and independently validate one strict packet mapping."""

        item = require_mapping(value, "VisualObservationPacketV1")
        require_exact_fields(item, cls._FIELDS, "VisualObservationPacketV1")
        try:
            collection = SourceCollection(
                require_text(
                    item["source_collection"],
                    "VisualObservationPacketV1.source_collection",
                )
            )
            split = VisualDatasetSplit(
                require_text(item["split"], "VisualObservationPacketV1.split")
            )
        except ValueError as exc:
            raise VisionConfigurationError(
                "VisualObservationPacketV1: unsupported collection or split"
            ) from exc
        packet = cls(
            source_collection=collection,
            source_trajectory_id=require_text(
                item["source_trajectory_id"],
                "VisualObservationPacketV1.source_trajectory_id",
            ),
            anchor_id=require_text(
                item["anchor_id"], "VisualObservationPacketV1.anchor_id"
            ),
            split=split,
            split_group_id=require_text(
                item["split_group_id"], "VisualObservationPacketV1.split_group_id"
            ),
            state_reference_id=require_text(
                item["state_reference_id"],
                "VisualObservationPacketV1.state_reference_id",
            ),
            expected_state_digest=require_text(
                item["expected_state_digest"],
                "VisualObservationPacketV1.expected_state_digest",
            ),
            verifier_state_semantic=require_text(
                item["verifier_state_semantic"],
                "VisualObservationPacketV1.verifier_state_semantic",
            ),
            verifier_state_digest=require_text(
                item["verifier_state_digest"],
                "VisualObservationPacketV1.verifier_state_digest",
            ),
            verifier_state_component_count=require_integer(
                item["verifier_state_component_count"],
                "VisualObservationPacketV1.verifier_state_component_count",
                minimum=1,
            ),
            verifier_state_maximum_absolute_error=require_number(
                item["verifier_state_maximum_absolute_error"],
                "VisualObservationPacketV1.verifier_state_maximum_absolute_error",
            ),
            elapsed_simulation_steps_before=require_integer(
                item["elapsed_simulation_steps_before"],
                "VisualObservationPacketV1.elapsed_simulation_steps_before",
            ),
            elapsed_simulation_steps_after=require_integer(
                item["elapsed_simulation_steps_after"],
                "VisualObservationPacketV1.elapsed_simulation_steps_after",
            ),
            environment_close_passed=_require_boolean(
                item["environment_close_passed"],
                "VisualObservationPacketV1.environment_close_passed",
            ),
            camera_rig_id=require_text(
                item["camera_rig_id"], "VisualObservationPacketV1.camera_rig_id"
            ),
            camera_rig_digest=require_text(
                item["camera_rig_digest"],
                "VisualObservationPacketV1.camera_rig_digest",
            ),
            render_domain_id=require_text(
                item["render_domain_id"],
                "VisualObservationPacketV1.render_domain_id",
            ),
            render_domain_digest=require_text(
                item["render_domain_digest"],
                "VisualObservationPacketV1.render_domain_digest",
            ),
            render_seed=require_integer(
                item["render_seed"], "VisualObservationPacketV1.render_seed"
            ),
            views=tuple(
                VisualViewRecordV1.from_mapping(view)
                for view in require_list(
                    item["views"], "VisualObservationPacketV1.views"
                )
            ),
            visual_compatibility_identity=require_text(
                item["visual_compatibility_identity"],
                "VisualObservationPacketV1.visual_compatibility_identity",
            ),
            pickcube_compatibility_identity=require_text(
                item["pickcube_compatibility_identity"],
                "VisualObservationPacketV1.pickcube_compatibility_identity",
            ),
            renderer_semantic_version=require_text(
                item["renderer_semantic_version"],
                "VisualObservationPacketV1.renderer_semantic_version",
            ),
            schema_version=require_text(
                item["schema_version"], "VisualObservationPacketV1.schema_version"
            ),
        )
        stored_id = require_text(
            item["packet_id"], "VisualObservationPacketV1.packet_id"
        )
        if stored_id != packet.packet_id:
            raise VisionConfigurationError(
                "VisualObservationPacketV1.packet_id: content mismatch"
            )
        return packet


__all__ = [
    "M4A_STATE_COMPARISON_TOLERANCE",
    "M4A_STATE_COMPONENT_COUNT",
    "M4A_VERIFIER_STATE_COMPARISON_TOLERANCE",
    "M4A_VERIFIER_STATE_COMPONENT_COUNT",
    "VISUAL_PACKET_SCHEMA_VERSION",
    "VISUAL_VIEW_SCHEMA_VERSION",
    "VisualObservationPacketV1",
    "VisualViewRecordV1",
]
