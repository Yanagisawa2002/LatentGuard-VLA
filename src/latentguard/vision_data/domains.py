"""Versioned M4A visual domains and fixed split assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from latentguard.vision_data.configuration import (
    VisionConfigurationError,
    require_exact_fields,
    require_integer,
    require_list,
    require_mapping,
    require_number,
    require_text,
    require_text_tuple,
    require_vector,
)
from latentguard.vision_data.models import SourceCollection, VisualDatasetSplit

RENDER_DOMAIN_SCHEMA_VERSION = "1.0"
RENDER_DOMAIN_CONFIGURATION_SCHEMA_VERSION = "1.0"
RENDER_DOMAIN_CONFIGURATION_ID = "pickcube_render_domains_v1"
RENDER_SEED_DERIVATION = "sha256_anchor_domain_base_seed_v1"
M4A_COLOR_FORMAT = "rgb_uint8"
M4A_IMAGE_RESOLUTION = (224, 224)

CANONICAL_DOMAIN_ID = "canonical"
MILD_CAMERA_DOMAIN_ID = "mild_camera_shift"
MILD_LIGHTING_DOMAIN_ID = "mild_lighting_shift"
STRONG_CAMERA_DOMAIN_ID = "strong_camera_shift"
STRONG_LIGHTING_DOMAIN_ID = "strong_lighting_shift"
RENDER_DOMAIN_IDS = (
    CANONICAL_DOMAIN_ID,
    MILD_CAMERA_DOMAIN_ID,
    MILD_LIGHTING_DOMAIN_ID,
    STRONG_CAMERA_DOMAIN_ID,
    STRONG_LIGHTING_DOMAIN_ID,
)

Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class LightingConfigurationV1:
    """Bounded deterministic lighting parameters without material changes."""

    ambient_intensity: float
    key_intensity: float
    key_color_rgb: Vector3
    key_direction: Vector3

    _FIELDS = frozenset(
        {"ambient_intensity", "key_intensity", "key_color_rgb", "key_direction"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_color_rgb", tuple(self.key_color_rgb))
        object.__setattr__(self, "key_direction", tuple(self.key_direction))
        from latentguard.vision_data.validation import validate_lighting_configuration

        validate_lighting_configuration(self)

    def as_mapping(self) -> dict[str, object]:
        """Return JSON-ready lighting parameters."""

        return {
            "ambient_intensity": self.ambient_intensity,
            "key_color_rgb": list(self.key_color_rgb),
            "key_direction": list(self.key_direction),
            "key_intensity": self.key_intensity,
        }

    @classmethod
    def from_mapping(cls, value: object) -> LightingConfigurationV1:
        """Decode one strict lighting object."""

        item = require_mapping(value, "LightingConfigurationV1")
        require_exact_fields(item, cls._FIELDS, "LightingConfigurationV1")
        return cls(
            ambient_intensity=require_number(
                item["ambient_intensity"],
                "LightingConfigurationV1.ambient_intensity",
            ),
            key_intensity=require_number(
                item["key_intensity"], "LightingConfigurationV1.key_intensity"
            ),
            key_color_rgb=cast(
                Vector3,
                require_vector(
                    item["key_color_rgb"], 3, "LightingConfigurationV1.key_color_rgb"
                ),
            ),
            key_direction=cast(
                Vector3,
                require_vector(
                    item["key_direction"], 3, "LightingConfigurationV1.key_direction"
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderDomainV1:
    """One deterministic camera or lighting render domain."""

    domain_id: str
    semantic_version: str
    camera_translation_min: Vector3
    camera_translation_max: Vector3
    camera_rotation_rpy_min_degrees: Vector3
    camera_rotation_rpy_max_degrees: Vector3
    lighting: LightingConfigurationV1
    allowed_collections: tuple[SourceCollection, ...]
    allowed_splits: tuple[VisualDatasetSplit, ...]
    schema_version: str = RENDER_DOMAIN_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "domain_id",
            "semantic_version",
            "camera_translation_min",
            "camera_translation_max",
            "camera_rotation_rpy_min_degrees",
            "camera_rotation_rpy_max_degrees",
            "lighting",
            "allowed_collections",
            "allowed_splits",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "camera_translation_min",
            "camera_translation_max",
            "camera_rotation_rpy_min_degrees",
            "camera_rotation_rpy_max_degrees",
            "allowed_collections",
            "allowed_splits",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        from latentguard.vision_data.validation import validate_render_domain

        validate_render_domain(self)

    @property
    def content_digest(self) -> str:
        """Return the complete semantic render-domain identity."""

        from latentguard.vision_data.identity import compute_model_content_digest

        return compute_model_content_digest(self, context="RenderDomainV1Content")

    @property
    def domain_digest(self) -> str:
        """Alias used by packet bindings."""

        return self.content_digest

    def as_mapping(self) -> dict[str, object]:
        """Return the checked-in domain schema."""

        return {
            "allowed_collections": [item.value for item in self.allowed_collections],
            "allowed_splits": [item.value for item in self.allowed_splits],
            "camera_rotation_rpy_max_degrees": list(
                self.camera_rotation_rpy_max_degrees
            ),
            "camera_rotation_rpy_min_degrees": list(
                self.camera_rotation_rpy_min_degrees
            ),
            "camera_translation_max": list(self.camera_translation_max),
            "camera_translation_min": list(self.camera_translation_min),
            "domain_id": self.domain_id,
            "lighting": self.lighting.as_mapping(),
            "semantic_version": self.semantic_version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> RenderDomainV1:
        """Decode one strict render-domain object."""

        item = require_mapping(value, "RenderDomainV1")
        require_exact_fields(item, cls._FIELDS, "RenderDomainV1")
        try:
            collections = tuple(
                SourceCollection(text)
                for text in require_text_tuple(
                    item["allowed_collections"], "RenderDomainV1.allowed_collections"
                )
            )
            splits = tuple(
                VisualDatasetSplit(text)
                for text in require_text_tuple(
                    item["allowed_splits"], "RenderDomainV1.allowed_splits"
                )
            )
        except ValueError as exc:
            raise VisionConfigurationError(
                "RenderDomainV1: unsupported collection or split"
            ) from exc
        return cls(
            domain_id=require_text(item["domain_id"], "RenderDomainV1.domain_id"),
            semantic_version=require_text(
                item["semantic_version"], "RenderDomainV1.semantic_version"
            ),
            camera_translation_min=cast(
                Vector3,
                require_vector(
                    item["camera_translation_min"],
                    3,
                    "RenderDomainV1.camera_translation_min",
                ),
            ),
            camera_translation_max=cast(
                Vector3,
                require_vector(
                    item["camera_translation_max"],
                    3,
                    "RenderDomainV1.camera_translation_max",
                ),
            ),
            camera_rotation_rpy_min_degrees=cast(
                Vector3,
                require_vector(
                    item["camera_rotation_rpy_min_degrees"],
                    3,
                    "RenderDomainV1.camera_rotation_rpy_min_degrees",
                ),
            ),
            camera_rotation_rpy_max_degrees=cast(
                Vector3,
                require_vector(
                    item["camera_rotation_rpy_max_degrees"],
                    3,
                    "RenderDomainV1.camera_rotation_rpy_max_degrees",
                ),
            ),
            lighting=LightingConfigurationV1.from_mapping(item["lighting"]),
            allowed_collections=collections,
            allowed_splits=splits,
        )


@dataclass(frozen=True, slots=True)
class RenderDomainConfigurationV1:
    """Frozen five-domain configuration used before any external rendering."""

    configuration_id: str
    seed_derivation: str
    shader_configuration: str
    image_resolution: tuple[int, int]
    color_format: str
    domains: tuple[RenderDomainV1, ...]
    schema_version: str = RENDER_DOMAIN_CONFIGURATION_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "schema_version",
            "configuration_id",
            "seed_derivation",
            "shader_configuration",
            "image_resolution",
            "color_format",
            "domains",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_resolution", tuple(self.image_resolution))
        object.__setattr__(self, "domains", tuple(self.domains))
        from latentguard.vision_data.validation import (
            validate_render_domain_configuration,
        )

        validate_render_domain_configuration(self)

    @property
    def content_digest(self) -> str:
        """Return the frozen complete render-configuration identity."""

        from latentguard.vision_data.identity import compute_model_content_digest

        return compute_model_content_digest(
            self, context="RenderDomainConfigurationV1Content"
        )

    def as_mapping(self) -> dict[str, object]:
        """Return the checked-in render-domain configuration schema."""

        return {
            "color_format": self.color_format,
            "configuration_id": self.configuration_id,
            "domains": [domain.as_mapping() for domain in self.domains],
            "image_resolution": list(self.image_resolution),
            "schema_version": self.schema_version,
            "seed_derivation": self.seed_derivation,
            "shader_configuration": self.shader_configuration,
        }

    @classmethod
    def from_mapping(cls, value: object) -> RenderDomainConfigurationV1:
        """Decode the strict checked-in five-domain configuration."""

        item = require_mapping(value, "RenderDomainConfigurationV1")
        require_exact_fields(item, cls._FIELDS, "RenderDomainConfigurationV1")
        resolution_items = require_list(
            item["image_resolution"], "RenderDomainConfigurationV1.image_resolution"
        )
        if len(resolution_items) != 2:
            raise VisionConfigurationError(
                "RenderDomainConfigurationV1.image_resolution: expected two values"
            )
        resolution = cast(
            tuple[int, int],
            tuple(
                require_integer(
                    entry,
                    f"RenderDomainConfigurationV1.image_resolution[{index}]",
                    minimum=1,
                )
                for index, entry in enumerate(resolution_items)
            ),
        )
        return cls(
            configuration_id=require_text(
                item["configuration_id"],
                "RenderDomainConfigurationV1.configuration_id",
            ),
            seed_derivation=require_text(
                item["seed_derivation"],
                "RenderDomainConfigurationV1.seed_derivation",
            ),
            shader_configuration=require_text(
                item["shader_configuration"],
                "RenderDomainConfigurationV1.shader_configuration",
            ),
            image_resolution=resolution,
            color_format=require_text(
                item["color_format"], "RenderDomainConfigurationV1.color_format"
            ),
            domains=tuple(
                RenderDomainV1.from_mapping(domain)
                for domain in require_list(
                    item["domains"], "RenderDomainConfigurationV1.domains"
                )
            ),
            schema_version=require_text(
                item["schema_version"],
                "RenderDomainConfigurationV1.schema_version",
            ),
        )

    def domain(self, domain_id: str) -> RenderDomainV1:
        """Return one configured domain by its stable ID."""

        for domain in self.domains:
            if domain.domain_id == domain_id:
                return domain
        raise VisionConfigurationError(f"unknown render domain {domain_id!r}")


def assigned_render_domain_ids(
    source_collection: SourceCollection, split: VisualDatasetSplit
) -> tuple[str, str, str]:
    """Return the frozen three-domain assignment before rendering."""

    if source_collection is SourceCollection.M3C_EXTERNAL:
        if split is not VisualDatasetSplit.EXTERNAL:
            raise VisionConfigurationError("M3C visual data must use external split")
        return (
            CANONICAL_DOMAIN_ID,
            STRONG_CAMERA_DOMAIN_ID,
            STRONG_LIGHTING_DOMAIN_ID,
        )
    if split in (VisualDatasetSplit.TRAIN, VisualDatasetSplit.VALIDATION):
        return (
            CANONICAL_DOMAIN_ID,
            MILD_CAMERA_DOMAIN_ID,
            MILD_LIGHTING_DOMAIN_ID,
        )
    if split is VisualDatasetSplit.TEST:
        return (
            CANONICAL_DOMAIN_ID,
            STRONG_CAMERA_DOMAIN_ID,
            STRONG_LIGHTING_DOMAIN_ID,
        )
    raise VisionConfigurationError("M3A development data cannot use external split")


__all__ = [
    "CANONICAL_DOMAIN_ID",
    "M4A_COLOR_FORMAT",
    "M4A_IMAGE_RESOLUTION",
    "MILD_CAMERA_DOMAIN_ID",
    "MILD_LIGHTING_DOMAIN_ID",
    "RENDER_DOMAIN_CONFIGURATION_ID",
    "RENDER_DOMAIN_CONFIGURATION_SCHEMA_VERSION",
    "RENDER_DOMAIN_IDS",
    "RENDER_DOMAIN_SCHEMA_VERSION",
    "RENDER_SEED_DERIVATION",
    "STRONG_CAMERA_DOMAIN_ID",
    "STRONG_LIGHTING_DOMAIN_ID",
    "LightingConfigurationV1",
    "RenderDomainConfigurationV1",
    "RenderDomainV1",
    "assigned_render_domain_ids",
]
