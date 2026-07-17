"""Simulator-independent immutable models for M4A visual data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from latentguard.models import FailureEvent

if TYPE_CHECKING:
    from latentguard.vision_data.packet import VisualObservationPacketV1

VISION_DATA_SCHEMA_VERSION = "1.0"
VISUAL_TASK_PROJECTION_SCHEMA_VERSION = "1.0"
VISUAL_MODEL_INPUT_CONTRACT_SCHEMA_VERSION = "1.0"
VISUAL_DATASET_SCHEMA_VERSION = "1.0"
PICKCUBE_VISUAL_TASK_ID = "maniskill/PickCube-v1"
PICKCUBE_CANONICAL_TASK_TEXT = "Pick up the cube and place it at the goal."


class SourceCollection(StrEnum):
    """Content-bound source collection used by one visual artifact."""

    M3A_DEVELOPMENT = "m3a_development"
    M3C_EXTERNAL = "m3c_external"


class VisualDatasetSplit(StrEnum):
    """M3A splits plus the distinct M3C evaluation-only partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class VisualTaskProjectionV1:
    """Serializable PickCube public task projection at the render boundary."""

    success: bool
    is_obj_placed: bool
    is_robot_static: bool
    is_grasped: bool
    cube_center_z: float
    cube_to_goal_distance: float
    tcp_to_cube_distance: float
    schema_version: str = VISUAL_TASK_PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        from latentguard.vision_data.validation import validate_task_projection

        validate_task_projection(self)

    @property
    def content_digest(self) -> str:
        """Return the path-independent public-task content digest."""

        from latentguard.vision_data.identity import compute_model_content_digest

        return compute_model_content_digest(
            self, context="VisualTaskProjectionV1Content"
        )

    def as_mapping(self) -> dict[str, object]:
        """Return canonical JSON-ready fields."""

        return {
            "cube_center_z": self.cube_center_z,
            "cube_to_goal_distance": self.cube_to_goal_distance,
            "is_grasped": self.is_grasped,
            "is_obj_placed": self.is_obj_placed,
            "is_robot_static": self.is_robot_static,
            "schema_version": self.schema_version,
            "success": self.success,
            "tcp_to_cube_distance": self.tcp_to_cube_distance,
        }

    @classmethod
    def from_mapping(cls, value: object) -> VisualTaskProjectionV1:
        """Decode one exact-field public task projection."""

        from latentguard.vision_data.configuration import (
            VisionConfigurationError,
            require_exact_fields,
            require_mapping,
            require_number,
            require_text,
        )

        item = require_mapping(value, "VisualTaskProjectionV1")
        expected = {
            "cube_center_z",
            "cube_to_goal_distance",
            "is_grasped",
            "is_obj_placed",
            "is_robot_static",
            "schema_version",
            "success",
            "tcp_to_cube_distance",
        }
        require_exact_fields(item, expected, "VisualTaskProjectionV1")

        def boolean(name: str) -> bool:
            observed = item[name]
            if type(observed) is not bool:
                raise VisionConfigurationError(
                    f"VisualTaskProjectionV1.{name}: expected boolean"
                )
            return observed

        return cls(
            success=boolean("success"),
            is_obj_placed=boolean("is_obj_placed"),
            is_robot_static=boolean("is_robot_static"),
            is_grasped=boolean("is_grasped"),
            cube_center_z=require_number(
                item["cube_center_z"], "VisualTaskProjectionV1.cube_center_z"
            ),
            cube_to_goal_distance=require_number(
                item["cube_to_goal_distance"],
                "VisualTaskProjectionV1.cube_to_goal_distance",
            ),
            tcp_to_cube_distance=require_number(
                item["tcp_to_cube_distance"],
                "VisualTaskProjectionV1.tcp_to_cube_distance",
            ),
            schema_version=require_text(
                item["schema_version"], "VisualTaskProjectionV1.schema_version"
            ),
        )


@dataclass(frozen=True, slots=True)
class VisualCandidateBindingV1:
    """One unique candidate bound to three ordered domain packets."""

    candidate_sample_id: str
    candidate_group_id: str
    anchor_id: str
    source_collection: SourceCollection
    packet_ids: tuple[str, str, str]
    schema_version: str = VISION_DATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_ids", tuple(self.packet_ids))
        from latentguard.vision_data.validation import validate_candidate_binding

        validate_candidate_binding(self)

    @property
    def binding_id(self) -> str:
        """Return the content-derived candidate binding identifier."""

        from latentguard.vision_data.identity import (
            compute_visual_candidate_binding_identifier,
        )

        return compute_visual_candidate_binding_identifier(self)

    def as_mapping(self) -> dict[str, object]:
        """Return canonical JSON-ready binding fields."""

        return {
            "anchor_id": self.anchor_id,
            "candidate_group_id": self.candidate_group_id,
            "candidate_sample_id": self.candidate_sample_id,
            "packet_ids": list(self.packet_ids),
            "schema_version": self.schema_version,
            "source_collection": self.source_collection.value,
        }

    @classmethod
    def from_mapping(cls, value: object) -> VisualCandidateBindingV1:
        """Decode one exact-field candidate-to-packet binding."""

        from latentguard.vision_data.configuration import (
            VisionConfigurationError,
            require_exact_fields,
            require_list,
            require_mapping,
            require_text,
        )

        item = require_mapping(value, "VisualCandidateBindingV1")
        require_exact_fields(
            item,
            {
                "anchor_id",
                "candidate_group_id",
                "candidate_sample_id",
                "packet_ids",
                "schema_version",
                "source_collection",
            },
            "VisualCandidateBindingV1",
        )
        try:
            collection = SourceCollection(
                require_text(
                    item["source_collection"],
                    "VisualCandidateBindingV1.source_collection",
                )
            )
        except ValueError as exc:
            raise VisionConfigurationError(
                "VisualCandidateBindingV1.source_collection: unsupported value"
            ) from exc
        packet_ids = require_list(
            item["packet_ids"], "VisualCandidateBindingV1.packet_ids"
        )
        if len(packet_ids) != 3:
            raise VisionConfigurationError(
                "VisualCandidateBindingV1.packet_ids: expected three values"
            )
        return cls(
            candidate_sample_id=require_text(
                item["candidate_sample_id"],
                "VisualCandidateBindingV1.candidate_sample_id",
            ),
            candidate_group_id=require_text(
                item["candidate_group_id"],
                "VisualCandidateBindingV1.candidate_group_id",
            ),
            anchor_id=require_text(
                item["anchor_id"], "VisualCandidateBindingV1.anchor_id"
            ),
            source_collection=collection,
            packet_ids=(
                require_text(packet_ids[0], "VisualCandidateBindingV1.packet_ids[0]"),
                require_text(packet_ids[1], "VisualCandidateBindingV1.packet_ids[1]"),
                require_text(packet_ids[2], "VisualCandidateBindingV1.packet_ids[2]"),
            ),
            schema_version=require_text(
                item["schema_version"], "VisualCandidateBindingV1.schema_version"
            ),
        )


def _failure_event_mapping(event: FailureEvent) -> dict[str, object]:
    return {
        "description": event.description,
        "failure_type": event.failure_type,
        "probability": event.probability,
        "schema_version": event.schema_version,
        "timestamp_s": event.timestamp_s,
    }


@dataclass(frozen=True, slots=True)
class VisualActionVerifierSampleV1:
    """Expanded single-packet example referencing an existing candidate."""

    packet_id: str
    candidate_sample_id: str
    task_id: str
    canonical_task_text: str
    candidate_action_chunk_reference: str
    action_mask_reference: str
    final_success: bool
    final_unsafe: bool
    failure_events: tuple[FailureEvent, ...]
    evidence_id: str
    source_dataset_digest: str
    candidate_dataset_digest: str
    visual_dataset_digest: str
    source_collection: SourceCollection
    schema_version: str = VISION_DATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure_events", tuple(self.failure_events))
        from latentguard.vision_data.validation import validate_visual_sample

        validate_visual_sample(self)

    @property
    def sample_id(self) -> str:
        """Return the content-derived expanded sample identifier."""

        from latentguard.vision_data.identity import compute_visual_sample_identifier

        return compute_visual_sample_identifier(self)

    def as_mapping(self) -> dict[str, object]:
        """Return references and reporting labels without action arrays."""

        return {
            "action_mask_reference": self.action_mask_reference,
            "candidate_action_chunk_reference": (self.candidate_action_chunk_reference),
            "candidate_dataset_digest": self.candidate_dataset_digest,
            "candidate_sample_id": self.candidate_sample_id,
            "canonical_task_text": self.canonical_task_text,
            "evidence_id": self.evidence_id,
            "failure_events": [
                _failure_event_mapping(event) for event in self.failure_events
            ],
            "final_success": self.final_success,
            "final_unsafe": self.final_unsafe,
            "packet_id": self.packet_id,
            "schema_version": self.schema_version,
            "source_collection": self.source_collection.value,
            "source_dataset_digest": self.source_dataset_digest,
            "task_id": self.task_id,
            "visual_dataset_digest": self.visual_dataset_digest,
        }

    def with_visual_dataset_digest(self, digest: str) -> VisualActionVerifierSampleV1:
        """Return a copy content-bound to the complete visual dataset."""

        return replace(self, visual_dataset_digest=digest)

    @classmethod
    def from_mapping(cls, value: object) -> VisualActionVerifierSampleV1:
        """Decode one exact-field reference-only visual sample."""

        from latentguard.vision_data.configuration import (
            VisionConfigurationError,
            require_exact_fields,
            require_list,
            require_mapping,
            require_number,
            require_text,
        )

        item = require_mapping(value, "VisualActionVerifierSampleV1")
        require_exact_fields(
            item,
            {
                "action_mask_reference",
                "candidate_action_chunk_reference",
                "candidate_dataset_digest",
                "candidate_sample_id",
                "canonical_task_text",
                "evidence_id",
                "failure_events",
                "final_success",
                "final_unsafe",
                "packet_id",
                "schema_version",
                "source_collection",
                "source_dataset_digest",
                "task_id",
                "visual_dataset_digest",
            },
            "VisualActionVerifierSampleV1",
        )

        def boolean(name: str) -> bool:
            observed = item[name]
            if type(observed) is not bool:
                raise VisionConfigurationError(
                    f"VisualActionVerifierSampleV1.{name}: expected boolean"
                )
            return bool(observed)

        def optional_number(value: object, context: str) -> float | None:
            if value is None:
                return None
            return require_number(value, context)

        def optional_text(value: object, context: str) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise VisionConfigurationError(f"{context}: expected text or null")
            return value

        events: list[FailureEvent] = []
        for index, raw_event in enumerate(
            require_list(
                item["failure_events"],
                "VisualActionVerifierSampleV1.failure_events",
            )
        ):
            context = f"VisualActionVerifierSampleV1.failure_events[{index}]"
            event = require_mapping(raw_event, context)
            require_exact_fields(
                event,
                {
                    "description",
                    "failure_type",
                    "probability",
                    "schema_version",
                    "timestamp_s",
                },
                context,
            )
            events.append(
                FailureEvent(
                    failure_type=require_text(
                        event["failure_type"], f"{context}.failure_type"
                    ),
                    timestamp_s=optional_number(
                        event["timestamp_s"], f"{context}.timestamp_s"
                    ),
                    probability=optional_number(
                        event["probability"], f"{context}.probability"
                    ),
                    description=optional_text(
                        event["description"], f"{context}.description"
                    ),
                    schema_version=require_text(
                        event["schema_version"], f"{context}.schema_version"
                    ),
                )
            )
        try:
            collection = SourceCollection(
                require_text(
                    item["source_collection"],
                    "VisualActionVerifierSampleV1.source_collection",
                )
            )
        except ValueError as exc:
            raise VisionConfigurationError(
                "VisualActionVerifierSampleV1.source_collection: unsupported value"
            ) from exc
        return cls(
            packet_id=require_text(
                item["packet_id"], "VisualActionVerifierSampleV1.packet_id"
            ),
            candidate_sample_id=require_text(
                item["candidate_sample_id"],
                "VisualActionVerifierSampleV1.candidate_sample_id",
            ),
            task_id=require_text(
                item["task_id"], "VisualActionVerifierSampleV1.task_id"
            ),
            canonical_task_text=require_text(
                item["canonical_task_text"],
                "VisualActionVerifierSampleV1.canonical_task_text",
            ),
            candidate_action_chunk_reference=require_text(
                item["candidate_action_chunk_reference"],
                "VisualActionVerifierSampleV1.candidate_action_chunk_reference",
            ),
            action_mask_reference=require_text(
                item["action_mask_reference"],
                "VisualActionVerifierSampleV1.action_mask_reference",
            ),
            final_success=boolean("final_success"),
            final_unsafe=boolean("final_unsafe"),
            failure_events=tuple(events),
            evidence_id=require_text(
                item["evidence_id"], "VisualActionVerifierSampleV1.evidence_id"
            ),
            source_dataset_digest=require_text(
                item["source_dataset_digest"],
                "VisualActionVerifierSampleV1.source_dataset_digest",
            ),
            candidate_dataset_digest=require_text(
                item["candidate_dataset_digest"],
                "VisualActionVerifierSampleV1.candidate_dataset_digest",
            ),
            visual_dataset_digest=require_text(
                item["visual_dataset_digest"],
                "VisualActionVerifierSampleV1.visual_dataset_digest",
            ),
            source_collection=collection,
            schema_version=require_text(
                item["schema_version"],
                "VisualActionVerifierSampleV1.schema_version",
            ),
        )


def _visual_dataset_content_digest(
    *,
    metadata: dict[str, object],
    packets: tuple[VisualObservationPacketV1, ...],
    candidate_bindings: tuple[VisualCandidateBindingV1, ...],
    samples: tuple[VisualActionVerifierSampleV1, ...],
) -> str:
    """Bind complete packet pixels and samples without a digest self-loop."""

    from latentguard.vision_data.identity import canonical_sha256

    sample_mappings: list[dict[str, object]] = []
    for sample in samples:
        item = sample.as_mapping()
        del item["visual_dataset_digest"]
        sample_mappings.append(item)
    return canonical_sha256(
        {
            **metadata,
            "candidate_bindings": [
                binding.as_mapping() for binding in candidate_bindings
            ],
            "packets": [packet.as_mapping() for packet in packets],
            "samples": sample_mappings,
        },
        context="M4AVisualVerifierDatasetV1Content",
    )


def _dataset_digest_text(value: str, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{context}: expected lowercase sha256 digest")


def _stamp_visual_samples(
    samples: tuple[VisualActionVerifierSampleV1, ...], digest: str
) -> tuple[VisualActionVerifierSampleV1, ...]:
    return tuple(sample.with_visual_dataset_digest(digest) for sample in samples)


@dataclass(frozen=True, slots=True, eq=False)
class VisualVerifierDevelopmentDatasetV1:
    """Complete M3A-derived visual dataset with shared packet references."""

    source_collection: SourceCollection
    source_dataset_digest: str
    split_digest: str
    evidence_digest: str
    source_compatibility_identity: str
    packet_manifest_digest: str
    packets: tuple[VisualObservationPacketV1, ...]
    candidate_bindings: tuple[VisualCandidateBindingV1, ...]
    samples: tuple[VisualActionVerifierSampleV1, ...]
    camera_rig_digest: str
    render_domain_configuration_digest: str
    visual_compatibility_identity: str
    training_allowed: bool = True
    schema_version: str = VISUAL_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze collections and reject source, digest, or backref drift."""

        try:
            collection = SourceCollection(self.source_collection)
        except ValueError as exc:
            raise ValueError("development dataset: invalid source collection") from exc
        object.__setattr__(self, "source_collection", collection)
        object.__setattr__(self, "packets", tuple(self.packets))
        object.__setattr__(self, "candidate_bindings", tuple(self.candidate_bindings))
        object.__setattr__(self, "samples", tuple(self.samples))
        if collection is not SourceCollection.M3A_DEVELOPMENT:
            raise ValueError("development dataset: source collection must be M3A")
        if self.training_allowed is not True:
            raise ValueError("development dataset: training_allowed must be true")
        if self.schema_version != VISUAL_DATASET_SCHEMA_VERSION:
            raise ValueError("development dataset: unsupported schema version")
        from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST

        if self.source_dataset_digest != ACCEPTED_M3A_DATASET_DIGEST:
            raise ValueError("development dataset: source is not accepted M3A")
        for name in (
            "source_dataset_digest",
            "split_digest",
            "evidence_digest",
            "source_compatibility_identity",
            "packet_manifest_digest",
            "camera_rig_digest",
            "render_domain_configuration_digest",
            "visual_compatibility_identity",
        ):
            _dataset_digest_text(getattr(self, name), f"development dataset {name}")
        expected = self.content_digest
        if any(sample.visual_dataset_digest != expected for sample in self.samples):
            raise ValueError("development dataset: sample dataset backref mismatch")
        if any(
            sample.source_collection is not collection for sample in self.samples
        ) or any(
            binding.source_collection is not collection
            for binding in self.candidate_bindings
        ):
            raise ValueError("development dataset: child source collection mismatch")
        from latentguard.vision_data.dataset import validate_visual_verifier_dataset

        validate_visual_verifier_dataset(self, full_target=False)

    def _content_metadata(self) -> dict[str, object]:
        return {
            "camera_rig_digest": self.camera_rig_digest,
            "evidence_digest": self.evidence_digest,
            "packet_manifest_digest": self.packet_manifest_digest,
            "render_domain_configuration_digest": (
                self.render_domain_configuration_digest
            ),
            "schema_version": self.schema_version,
            "source_collection": self.source_collection.value,
            "source_compatibility_identity": self.source_compatibility_identity,
            "source_dataset_digest": self.source_dataset_digest,
            "split_digest": self.split_digest,
            "training_allowed": self.training_allowed,
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete digest including exact ordered image records."""

        return _visual_dataset_content_digest(
            metadata=self._content_metadata(),
            packets=self.packets,
            candidate_bindings=self.candidate_bindings,
            samples=self.samples,
        )

    def as_mapping(self) -> dict[str, object]:
        """Return complete serializable content and its verified digest."""

        return {
            **self._content_metadata(),
            "candidate_bindings": [
                binding.as_mapping() for binding in self.candidate_bindings
            ],
            "content_digest": self.content_digest,
            "packets": [packet.as_mapping() for packet in self.packets],
            "samples": [sample.as_mapping() for sample in self.samples],
        }

    @classmethod
    def create(
        cls,
        *,
        source_dataset_digest: str,
        split_digest: str,
        evidence_digest: str,
        source_compatibility_identity: str,
        packet_manifest_digest: str,
        packets: tuple[VisualObservationPacketV1, ...],
        candidate_bindings: tuple[VisualCandidateBindingV1, ...],
        samples: tuple[VisualActionVerifierSampleV1, ...],
        camera_rig_digest: str,
        render_domain_configuration_digest: str,
        visual_compatibility_identity: str,
    ) -> VisualVerifierDevelopmentDatasetV1:
        """Compute the final digest, stamp sample backrefs, and construct."""

        packet_values = tuple(packets)
        binding_values = tuple(candidate_bindings)
        sample_values = tuple(samples)
        metadata = {
            "camera_rig_digest": camera_rig_digest,
            "evidence_digest": evidence_digest,
            "packet_manifest_digest": packet_manifest_digest,
            "render_domain_configuration_digest": (render_domain_configuration_digest),
            "schema_version": VISUAL_DATASET_SCHEMA_VERSION,
            "source_collection": SourceCollection.M3A_DEVELOPMENT.value,
            "source_compatibility_identity": source_compatibility_identity,
            "source_dataset_digest": source_dataset_digest,
            "split_digest": split_digest,
            "training_allowed": True,
            "visual_compatibility_identity": visual_compatibility_identity,
        }
        digest = _visual_dataset_content_digest(
            metadata=metadata,
            packets=packet_values,
            candidate_bindings=binding_values,
            samples=sample_values,
        )
        return cls(
            source_collection=SourceCollection.M3A_DEVELOPMENT,
            source_dataset_digest=source_dataset_digest,
            split_digest=split_digest,
            evidence_digest=evidence_digest,
            source_compatibility_identity=source_compatibility_identity,
            packet_manifest_digest=packet_manifest_digest,
            packets=packet_values,
            candidate_bindings=binding_values,
            samples=_stamp_visual_samples(sample_values, digest),
            camera_rig_digest=camera_rig_digest,
            render_domain_configuration_digest=render_domain_configuration_digest,
            visual_compatibility_identity=visual_compatibility_identity,
        )

    @classmethod
    def from_mapping(cls, value: object) -> VisualVerifierDevelopmentDatasetV1:
        """Decode and validate an exact-field development dataset."""

        return _development_dataset_from_mapping(value)


@dataclass(frozen=True, slots=True, eq=False)
class VisualVerifierExternalDatasetV1:
    """M3C-derived external visual dataset prohibited from training."""

    source_collection: SourceCollection
    source_set_identity: str
    candidate_pool_identity: str
    blind_manifest_digest: str
    full_outcome_digest: str
    source_compatibility_identity: str
    packet_manifest_digest: str
    packets: tuple[VisualObservationPacketV1, ...]
    candidate_bindings: tuple[VisualCandidateBindingV1, ...]
    samples: tuple[VisualActionVerifierSampleV1, ...]
    camera_rig_digest: str
    render_domain_configuration_digest: str
    visual_compatibility_identity: str
    training_allowed: bool = False
    schema_version: str = VISUAL_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze collections and enforce evaluation-only content binding."""

        try:
            collection = SourceCollection(self.source_collection)
        except ValueError as exc:
            raise ValueError("external dataset: invalid source collection") from exc
        object.__setattr__(self, "source_collection", collection)
        object.__setattr__(self, "packets", tuple(self.packets))
        object.__setattr__(self, "candidate_bindings", tuple(self.candidate_bindings))
        object.__setattr__(self, "samples", tuple(self.samples))
        if collection is not SourceCollection.M3C_EXTERNAL:
            raise ValueError("external dataset: source collection must be M3C")
        if self.training_allowed is not False:
            raise ValueError("external dataset: training_allowed must be false")
        if self.schema_version != VISUAL_DATASET_SCHEMA_VERSION:
            raise ValueError("external dataset: unsupported schema version")
        for name in (
            "source_set_identity",
            "candidate_pool_identity",
            "blind_manifest_digest",
            "full_outcome_digest",
            "source_compatibility_identity",
            "packet_manifest_digest",
            "camera_rig_digest",
            "render_domain_configuration_digest",
            "visual_compatibility_identity",
        ):
            _dataset_digest_text(getattr(self, name), f"external dataset {name}")
        expected = self.content_digest
        if any(sample.visual_dataset_digest != expected for sample in self.samples):
            raise ValueError("external dataset: sample dataset backref mismatch")
        if any(
            sample.source_collection is not collection for sample in self.samples
        ) or any(
            binding.source_collection is not collection
            for binding in self.candidate_bindings
        ):
            raise ValueError("external dataset: child source collection mismatch")
        from latentguard.vision_data.dataset import validate_visual_verifier_dataset

        validate_visual_verifier_dataset(self, full_target=False)

    def _content_metadata(self) -> dict[str, object]:
        return {
            "blind_manifest_digest": self.blind_manifest_digest,
            "camera_rig_digest": self.camera_rig_digest,
            "candidate_pool_identity": self.candidate_pool_identity,
            "full_outcome_digest": self.full_outcome_digest,
            "packet_manifest_digest": self.packet_manifest_digest,
            "render_domain_configuration_digest": (
                self.render_domain_configuration_digest
            ),
            "schema_version": self.schema_version,
            "source_collection": self.source_collection.value,
            "source_compatibility_identity": self.source_compatibility_identity,
            "source_set_identity": self.source_set_identity,
            "training_allowed": self.training_allowed,
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete external visual dataset digest."""

        return _visual_dataset_content_digest(
            metadata=self._content_metadata(),
            packets=self.packets,
            candidate_bindings=self.candidate_bindings,
            samples=self.samples,
        )

    def as_mapping(self) -> dict[str, object]:
        """Return complete evaluation-only content and verified digest."""

        return {
            **self._content_metadata(),
            "candidate_bindings": [
                binding.as_mapping() for binding in self.candidate_bindings
            ],
            "content_digest": self.content_digest,
            "packets": [packet.as_mapping() for packet in self.packets],
            "samples": [sample.as_mapping() for sample in self.samples],
        }

    @classmethod
    def create(
        cls,
        *,
        source_set_identity: str,
        candidate_pool_identity: str,
        blind_manifest_digest: str,
        full_outcome_digest: str,
        source_compatibility_identity: str,
        packet_manifest_digest: str,
        packets: tuple[VisualObservationPacketV1, ...],
        candidate_bindings: tuple[VisualCandidateBindingV1, ...],
        samples: tuple[VisualActionVerifierSampleV1, ...],
        camera_rig_digest: str,
        render_domain_configuration_digest: str,
        visual_compatibility_identity: str,
    ) -> VisualVerifierExternalDatasetV1:
        """Compute the final digest, stamp sample backrefs, and construct."""

        packet_values = tuple(packets)
        binding_values = tuple(candidate_bindings)
        sample_values = tuple(samples)
        metadata = {
            "blind_manifest_digest": blind_manifest_digest,
            "camera_rig_digest": camera_rig_digest,
            "candidate_pool_identity": candidate_pool_identity,
            "full_outcome_digest": full_outcome_digest,
            "packet_manifest_digest": packet_manifest_digest,
            "render_domain_configuration_digest": (render_domain_configuration_digest),
            "schema_version": VISUAL_DATASET_SCHEMA_VERSION,
            "source_collection": SourceCollection.M3C_EXTERNAL.value,
            "source_compatibility_identity": source_compatibility_identity,
            "source_set_identity": source_set_identity,
            "training_allowed": False,
            "visual_compatibility_identity": visual_compatibility_identity,
        }
        digest = _visual_dataset_content_digest(
            metadata=metadata,
            packets=packet_values,
            candidate_bindings=binding_values,
            samples=sample_values,
        )
        return cls(
            source_collection=SourceCollection.M3C_EXTERNAL,
            source_set_identity=source_set_identity,
            candidate_pool_identity=candidate_pool_identity,
            blind_manifest_digest=blind_manifest_digest,
            full_outcome_digest=full_outcome_digest,
            source_compatibility_identity=source_compatibility_identity,
            packet_manifest_digest=packet_manifest_digest,
            packets=packet_values,
            candidate_bindings=binding_values,
            samples=_stamp_visual_samples(sample_values, digest),
            camera_rig_digest=camera_rig_digest,
            render_domain_configuration_digest=render_domain_configuration_digest,
            visual_compatibility_identity=visual_compatibility_identity,
        )

    @classmethod
    def from_mapping(cls, value: object) -> VisualVerifierExternalDatasetV1:
        """Decode and validate an exact-field external dataset."""

        return _external_dataset_from_mapping(value)


def _dataset_boolean(value: object, context: str) -> bool:
    from latentguard.vision_data.configuration import VisionConfigurationError

    if type(value) is not bool:
        raise VisionConfigurationError(f"{context}: expected boolean")
    return bool(value)


def _development_dataset_from_mapping(
    value: object,
) -> VisualVerifierDevelopmentDatasetV1:
    from latentguard.vision_data.configuration import (
        require_exact_fields,
        require_list,
        require_mapping,
        require_text,
    )
    from latentguard.vision_data.packet import VisualObservationPacketV1

    item = require_mapping(value, "VisualVerifierDevelopmentDatasetV1")
    require_exact_fields(
        item,
        {
            "camera_rig_digest",
            "candidate_bindings",
            "content_digest",
            "evidence_digest",
            "packet_manifest_digest",
            "packets",
            "render_domain_configuration_digest",
            "samples",
            "schema_version",
            "source_collection",
            "source_compatibility_identity",
            "source_dataset_digest",
            "split_digest",
            "training_allowed",
            "visual_compatibility_identity",
        },
        "VisualVerifierDevelopmentDatasetV1",
    )
    dataset = VisualVerifierDevelopmentDatasetV1(
        source_collection=SourceCollection(
            require_text(
                item["source_collection"],
                "VisualVerifierDevelopmentDatasetV1.source_collection",
            )
        ),
        source_dataset_digest=require_text(
            item["source_dataset_digest"],
            "VisualVerifierDevelopmentDatasetV1.source_dataset_digest",
        ),
        split_digest=require_text(
            item["split_digest"], "VisualVerifierDevelopmentDatasetV1.split_digest"
        ),
        evidence_digest=require_text(
            item["evidence_digest"],
            "VisualVerifierDevelopmentDatasetV1.evidence_digest",
        ),
        source_compatibility_identity=require_text(
            item["source_compatibility_identity"],
            "VisualVerifierDevelopmentDatasetV1.source_compatibility_identity",
        ),
        packet_manifest_digest=require_text(
            item["packet_manifest_digest"],
            "VisualVerifierDevelopmentDatasetV1.packet_manifest_digest",
        ),
        packets=tuple(
            VisualObservationPacketV1.from_mapping(packet)
            for packet in require_list(
                item["packets"], "VisualVerifierDevelopmentDatasetV1.packets"
            )
        ),
        candidate_bindings=tuple(
            VisualCandidateBindingV1.from_mapping(binding)
            for binding in require_list(
                item["candidate_bindings"],
                "VisualVerifierDevelopmentDatasetV1.candidate_bindings",
            )
        ),
        samples=tuple(
            VisualActionVerifierSampleV1.from_mapping(sample)
            for sample in require_list(
                item["samples"], "VisualVerifierDevelopmentDatasetV1.samples"
            )
        ),
        camera_rig_digest=require_text(
            item["camera_rig_digest"],
            "VisualVerifierDevelopmentDatasetV1.camera_rig_digest",
        ),
        render_domain_configuration_digest=require_text(
            item["render_domain_configuration_digest"],
            "VisualVerifierDevelopmentDatasetV1.render_domain_configuration_digest",
        ),
        visual_compatibility_identity=require_text(
            item["visual_compatibility_identity"],
            "VisualVerifierDevelopmentDatasetV1.visual_compatibility_identity",
        ),
        training_allowed=_dataset_boolean(
            item["training_allowed"],
            "VisualVerifierDevelopmentDatasetV1.training_allowed",
        ),
        schema_version=require_text(
            item["schema_version"],
            "VisualVerifierDevelopmentDatasetV1.schema_version",
        ),
    )
    stored = require_text(
        item["content_digest"],
        "VisualVerifierDevelopmentDatasetV1.content_digest",
    )
    if stored != dataset.content_digest:
        from latentguard.vision_data.configuration import VisionConfigurationError

        raise VisionConfigurationError(
            "VisualVerifierDevelopmentDatasetV1.content_digest: content mismatch"
        )
    return dataset


def _external_dataset_from_mapping(value: object) -> VisualVerifierExternalDatasetV1:
    from latentguard.vision_data.configuration import (
        require_exact_fields,
        require_list,
        require_mapping,
        require_text,
    )
    from latentguard.vision_data.packet import VisualObservationPacketV1

    item = require_mapping(value, "VisualVerifierExternalDatasetV1")
    require_exact_fields(
        item,
        {
            "blind_manifest_digest",
            "camera_rig_digest",
            "candidate_bindings",
            "candidate_pool_identity",
            "content_digest",
            "full_outcome_digest",
            "packet_manifest_digest",
            "packets",
            "render_domain_configuration_digest",
            "samples",
            "schema_version",
            "source_collection",
            "source_compatibility_identity",
            "source_set_identity",
            "training_allowed",
            "visual_compatibility_identity",
        },
        "VisualVerifierExternalDatasetV1",
    )
    dataset = VisualVerifierExternalDatasetV1(
        source_collection=SourceCollection(
            require_text(
                item["source_collection"],
                "VisualVerifierExternalDatasetV1.source_collection",
            )
        ),
        source_set_identity=require_text(
            item["source_set_identity"],
            "VisualVerifierExternalDatasetV1.source_set_identity",
        ),
        candidate_pool_identity=require_text(
            item["candidate_pool_identity"],
            "VisualVerifierExternalDatasetV1.candidate_pool_identity",
        ),
        blind_manifest_digest=require_text(
            item["blind_manifest_digest"],
            "VisualVerifierExternalDatasetV1.blind_manifest_digest",
        ),
        full_outcome_digest=require_text(
            item["full_outcome_digest"],
            "VisualVerifierExternalDatasetV1.full_outcome_digest",
        ),
        source_compatibility_identity=require_text(
            item["source_compatibility_identity"],
            "VisualVerifierExternalDatasetV1.source_compatibility_identity",
        ),
        packet_manifest_digest=require_text(
            item["packet_manifest_digest"],
            "VisualVerifierExternalDatasetV1.packet_manifest_digest",
        ),
        packets=tuple(
            VisualObservationPacketV1.from_mapping(packet)
            for packet in require_list(
                item["packets"], "VisualVerifierExternalDatasetV1.packets"
            )
        ),
        candidate_bindings=tuple(
            VisualCandidateBindingV1.from_mapping(binding)
            for binding in require_list(
                item["candidate_bindings"],
                "VisualVerifierExternalDatasetV1.candidate_bindings",
            )
        ),
        samples=tuple(
            VisualActionVerifierSampleV1.from_mapping(sample)
            for sample in require_list(
                item["samples"], "VisualVerifierExternalDatasetV1.samples"
            )
        ),
        camera_rig_digest=require_text(
            item["camera_rig_digest"],
            "VisualVerifierExternalDatasetV1.camera_rig_digest",
        ),
        render_domain_configuration_digest=require_text(
            item["render_domain_configuration_digest"],
            "VisualVerifierExternalDatasetV1.render_domain_configuration_digest",
        ),
        visual_compatibility_identity=require_text(
            item["visual_compatibility_identity"],
            "VisualVerifierExternalDatasetV1.visual_compatibility_identity",
        ),
        training_allowed=_dataset_boolean(
            item["training_allowed"],
            "VisualVerifierExternalDatasetV1.training_allowed",
        ),
        schema_version=require_text(
            item["schema_version"],
            "VisualVerifierExternalDatasetV1.schema_version",
        ),
    )
    stored = require_text(
        item["content_digest"], "VisualVerifierExternalDatasetV1.content_digest"
    )
    if stored != dataset.content_digest:
        from latentguard.vision_data.configuration import VisionConfigurationError

        raise VisionConfigurationError(
            "VisualVerifierExternalDatasetV1.content_digest: content mismatch"
        )
    return dataset


@dataclass(frozen=True, slots=True)
class VisualModelInputContractV1:
    """Explicit separation of deployable, reporting, and privileged fields."""

    required_student_inputs: tuple[str, ...]
    conditional_student_inputs: tuple[str, ...]
    reporting_only_fields: tuple[str, ...]
    privileged_teacher_fields: tuple[str, ...]
    schema_version: str = VISUAL_MODEL_INPUT_CONTRACT_SCHEMA_VERSION

    REQUIRED: ClassVar[tuple[str, ...]] = (
        "ordered_rgb_views",
        "candidate_action_chunk",
        "action_mask",
    )
    CONDITIONAL: ClassVar[tuple[str, ...]] = ("canonical_task_text",)
    REPORTING: ClassVar[tuple[str, ...]] = (
        "render_domain_id",
        "camera_id",
        "source_collection",
        "split",
        "candidate_type",
        "corruption_family",
        "severity",
        "anchor_reason",
        "trajectory_id",
        "packet_id",
        "evidence_id",
        "outcome_fields",
    )
    PRIVILEGED: ClassVar[tuple[str, ...]] = (
        "verifier_state",
        "complete_simulator_state",
        "object_pose",
        "goal_position",
        "robot_joint_state",
        "task_snapshot_fields",
    )

    def __post_init__(self) -> None:
        for field in (
            "required_student_inputs",
            "conditional_student_inputs",
            "reporting_only_fields",
            "privileged_teacher_fields",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.schema_version != VISUAL_MODEL_INPUT_CONTRACT_SCHEMA_VERSION:
            raise ValueError("VisualModelInputContractV1: unsupported version")
        expected = (
            (self.required_student_inputs, self.REQUIRED),
            (self.conditional_student_inputs, self.CONDITIONAL),
            (self.reporting_only_fields, self.REPORTING),
            (self.privileged_teacher_fields, self.PRIVILEGED),
        )
        if any(observed != required for observed, required in expected):
            raise ValueError("VisualModelInputContractV1: field policy differs")
        inventories = tuple(observed for observed, _ in expected)
        if any(
            set(left).intersection(right)
            for index, left in enumerate(inventories)
            for right in inventories[index + 1 :]
        ):
            raise ValueError("VisualModelInputContractV1: field policies overlap")

    @property
    def content_digest(self) -> str:
        """Return the fixed field-policy identity."""

        from latentguard.vision_data.identity import compute_model_content_digest

        return compute_model_content_digest(
            self, context="VisualModelInputContractV1Content"
        )

    def as_mapping(self) -> dict[str, object]:
        """Return canonical field-policy metadata."""

        return {
            "conditional_student_inputs": list(self.conditional_student_inputs),
            "privileged_teacher_fields": list(self.privileged_teacher_fields),
            "reporting_only_fields": list(self.reporting_only_fields),
            "required_student_inputs": list(self.required_student_inputs),
            "schema_version": self.schema_version,
        }


M4A_VISUAL_MODEL_INPUT_CONTRACT = VisualModelInputContractV1(
    required_student_inputs=VisualModelInputContractV1.REQUIRED,
    conditional_student_inputs=VisualModelInputContractV1.CONDITIONAL,
    reporting_only_fields=VisualModelInputContractV1.REPORTING,
    privileged_teacher_fields=VisualModelInputContractV1.PRIVILEGED,
)


__all__ = [
    "M4A_VISUAL_MODEL_INPUT_CONTRACT",
    "PICKCUBE_CANONICAL_TASK_TEXT",
    "PICKCUBE_VISUAL_TASK_ID",
    "SourceCollection",
    "VISION_DATA_SCHEMA_VERSION",
    "VISUAL_DATASET_SCHEMA_VERSION",
    "VISUAL_MODEL_INPUT_CONTRACT_SCHEMA_VERSION",
    "VISUAL_TASK_PROJECTION_SCHEMA_VERSION",
    "VisualActionVerifierSampleV1",
    "VisualCandidateBindingV1",
    "VisualDatasetSplit",
    "VisualModelInputContractV1",
    "VisualTaskProjectionV1",
    "VisualVerifierDevelopmentDatasetV1",
    "VisualVerifierExternalDatasetV1",
]
