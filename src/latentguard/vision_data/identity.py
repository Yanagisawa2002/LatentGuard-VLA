"""Canonical identities for M4A visual data contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

if TYPE_CHECKING:
    from latentguard.vision_data.models import (
        VisualActionVerifierSampleV1,
        VisualCandidateBindingV1,
    )
    from latentguard.vision_data.packet import (
        VisualObservationPacketV1,
        VisualViewRecordV1,
    )


class _MappingModel(Protocol):
    def as_mapping(self) -> dict[str, object]:
        """Return canonical JSON-ready content."""


def canonical_sha256(payload: object, *, context: str, prefix: str = "sha256:") -> str:
    """Hash strict path-safe canonical JSON with SHA-256."""

    encoded = canonical_json_bytes(payload, context=context)
    return f"{prefix}{hashlib.sha256(encoded).hexdigest()}"


def array_payload(value: NDArray[np.generic]) -> dict[str, object]:
    """Describe exact C-order array bytes, dtype, and shape."""

    contiguous = np.ascontiguousarray(value)
    return {
        "content_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }


def compute_model_content_digest(model: _MappingModel, *, context: str) -> str:
    """Return a conventional digest for one mapping-backed model."""

    return canonical_sha256(model.as_mapping(), context=context)


def compute_visual_packet_identifier_from_fields(
    *,
    source_trajectory_id: str,
    anchor_id: str,
    expected_state_digest: str,
    camera_rig_digest: str,
    render_domain_digest: str,
    render_seed: int,
    camera_configuration_digests: Sequence[str],
    visual_compatibility_identity: str,
    schema_version: str,
) -> str:
    """Compute a packet ID before any pixels or task records exist."""

    camera_digests = tuple(camera_configuration_digests)
    if len(camera_digests) != 3:
        raise ValueError("packet identity requires exactly three camera digests")
    payload = {
        "anchor_id": anchor_id,
        "camera_configuration_digests": list(camera_digests),
        "camera_rig_digest": camera_rig_digest,
        "expected_state_digest": expected_state_digest,
        "render_domain_digest": render_domain_digest,
        "render_seed": render_seed,
        "schema_version": schema_version,
        "source_trajectory_id": source_trajectory_id,
        "visual_compatibility_identity": visual_compatibility_identity,
    }
    return canonical_sha256(
        payload,
        context="VisualObservationPacketV1Identity",
        prefix="vop-sha256-",
    )


def compute_visual_packet_identifier(packet: VisualObservationPacketV1) -> str:
    """Bind packet semantics without binding runtime paths or produced pixels."""

    return compute_visual_packet_identifier_from_fields(
        source_trajectory_id=packet.source_trajectory_id,
        anchor_id=packet.anchor_id,
        expected_state_digest=packet.expected_state_digest,
        camera_rig_digest=packet.camera_rig_digest,
        render_domain_digest=packet.render_domain_digest,
        render_seed=packet.render_seed,
        camera_configuration_digests=tuple(
            view.camera_configuration_digest for view in packet.views
        ),
        visual_compatibility_identity=packet.visual_compatibility_identity,
        schema_version=packet.schema_version,
    )


def compute_visual_view_content_digest(view: VisualViewRecordV1) -> str:
    """Bind one view record including exact pixel and NPY file digests."""

    return canonical_sha256(view.as_mapping(), context="VisualViewRecordV1Content")


def compute_visual_candidate_binding_identifier(
    binding: VisualCandidateBindingV1,
) -> str:
    """Bind one candidate to its ordered three-domain packet inventory."""

    return canonical_sha256(
        binding.as_mapping(),
        context="VisualCandidateBindingV1Identity",
        prefix="vcb-sha256-",
    )


def compute_visual_sample_identifier(sample: VisualActionVerifierSampleV1) -> str:
    """Bind one expanded single-packet visual verifier example."""

    return canonical_sha256(
        sample.as_mapping(),
        context="VisualActionVerifierSampleV1Identity",
        prefix="vas-sha256-",
    )


__all__ = [
    "array_payload",
    "canonical_sha256",
    "compute_model_content_digest",
    "compute_visual_candidate_binding_identifier",
    "compute_visual_packet_identifier",
    "compute_visual_packet_identifier_from_fields",
    "compute_visual_sample_identifier",
    "compute_visual_view_content_digest",
]
