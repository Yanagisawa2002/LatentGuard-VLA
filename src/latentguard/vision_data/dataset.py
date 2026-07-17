"""Factories and cross-reference validation for M4A visual datasets."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import NoReturn, cast

from latentguard.replay.identity import canonical_json_bytes

M3A_DEVELOPMENT_PACKET_COUNT = 1080
M3A_DEVELOPMENT_IMAGE_COUNT = 3240
M3A_CANDIDATE_BINDING_COUNT = 3240
M3A_EXPANDED_SAMPLE_COUNT = 9720
M3C_EXTERNAL_PACKET_COUNT = 1080
M3C_EXTERNAL_IMAGE_COUNT = 3240
M3C_CANDIDATE_BINDING_COUNT = 2880
M3C_EXPANDED_SAMPLE_COUNT = 8640


class VisualDatasetError(ValueError):
    """Raised when visual packet, candidate, or split binding is inconsistent."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualDatasetError(f"{context}: {reason}")


def _digest_payload(value: object, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _enum_text(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def compute_packet_manifest_digest(packets: Sequence[object]) -> str:
    """Bind the exact ordered packet identity inventory."""

    values = tuple(packets)
    if not values:
        _fail("packet manifest", "must not be empty")
    packet_ids = tuple(
        _identifier(getattr(item, "packet_id", None), "packet ID") for item in values
    )
    if len(packet_ids) != len(set(packet_ids)):
        _fail("packet manifest", "packet IDs must be unique")
    content = []
    for packet in values:
        digest = getattr(packet, "content_digest", None)
        if not isinstance(digest, str):
            _fail("packet manifest", "packet must expose its full content digest")
        content.append(_identifier(digest, "packet content digest"))
    return _digest_payload(
        {
            "packet_content_digests": content,
            "packet_ids": list(packet_ids),
            "schema_version": "m4a_packet_manifest_v1",
        },
        "M4APacketManifestV1",
    )


def _packet_indexes(
    packets: Sequence[object],
) -> tuple[dict[str, object], dict[str, tuple[object, ...]]]:
    by_id: dict[str, object] = {}
    by_anchor: dict[str, list[object]] = defaultdict(list)
    for packet in packets:
        packet_id = _identifier(getattr(packet, "packet_id", None), "packet ID")
        anchor_id = _identifier(getattr(packet, "anchor_id", None), "anchor ID")
        if packet_id in by_id:
            _fail("packets", "duplicate packet ID")
        by_id[packet_id] = packet
        by_anchor[anchor_id].append(packet)
    return by_id, {key: tuple(value) for key, value in by_anchor.items()}


def _validate_packet_groups(packets: Sequence[object], *, external: bool) -> None:
    from latentguard.vision_data.domains import assigned_render_domain_ids
    from latentguard.vision_data.models import SourceCollection, VisualDatasetSplit

    _, by_anchor = _packet_indexes(packets)
    expected_splits = {"external"} if external else {"train", "validation", "test"}
    expected_collection = (
        SourceCollection.M3C_EXTERNAL if external else SourceCollection.M3A_DEVELOPMENT
    )
    trajectory_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    for anchor_id, variants in by_anchor.items():
        if len(variants) != 3:
            _fail(anchor_id, "expected exactly three render-domain packets")
        split_values = {_enum_text(getattr(item, "split", "")) for item in variants}
        group_values = {getattr(item, "split_group_id", None) for item in variants}
        trajectory_values = {
            getattr(item, "source_trajectory_id", None) for item in variants
        }
        domains = tuple(
            _enum_text(getattr(item, "render_domain_id", "")) for item in variants
        )
        collections = {getattr(item, "source_collection", None) for item in variants}
        if len(split_values) != 1 or not split_values.issubset(expected_splits):
            _fail(anchor_id, "visual variants differ in split")
        if len(group_values) != 1 or len(trajectory_values) != 1:
            _fail(anchor_id, "visual variants differ in trajectory identity")
        if collections != {expected_collection}:
            _fail(anchor_id, "visual variants differ in source collection")
        if len(set(domains)) != 3:
            _fail(anchor_id, "render-domain IDs must be unique")
        split = next(iter(split_values))
        try:
            split_enum = VisualDatasetSplit(split)
        except ValueError as exc:
            raise VisualDatasetError(f"{anchor_id}: unsupported split") from exc
        required_domains = assigned_render_domain_ids(expected_collection, split_enum)
        if set(domains) != set(required_domains):
            _fail(anchor_id, "render-domain inventory differs from frozen assignment")
        trajectory_id = _identifier(
            next(iter(trajectory_values)), "packet trajectory ID"
        )
        group_id = _identifier(next(iter(group_values)), "packet split-group ID")
        prior_trajectory_split = trajectory_splits.setdefault(trajectory_id, split)
        prior_group_split = group_splits.setdefault(group_id, split)
        if prior_trajectory_split != split or prior_group_split != split:
            _fail(anchor_id, "trajectory or split group crosses dataset splits")


def _validate_candidate_bindings(
    packets: Sequence[object], candidate_bindings: Sequence[object]
) -> None:
    by_id, by_anchor = _packet_indexes(packets)
    seen: set[str] = set()
    anchor_group_ids: dict[str, str] = {}
    for binding in candidate_bindings:
        candidate_id = _identifier(
            getattr(binding, "candidate_sample_id", None), "candidate sample ID"
        )
        if candidate_id in seen:
            _fail("candidate bindings", "candidate appears more than once")
        seen.add(candidate_id)
        anchor_id = _identifier(getattr(binding, "anchor_id", None), "binding anchor")
        group_id = _identifier(
            getattr(binding, "candidate_group_id", None), "candidate group ID"
        )
        existing_group = anchor_group_ids.setdefault(anchor_id, group_id)
        if existing_group != group_id:
            _fail(candidate_id, "anchor is bound to multiple candidate groups")
        packet_ids = tuple(getattr(binding, "packet_ids", ()))
        if len(packet_ids) != 3 or len(set(packet_ids)) != 3:
            _fail(candidate_id, "must bind exactly three unique packets")
        anchor_packets = by_anchor.get(anchor_id, ())
        by_domain = {
            _identifier(getattr(item, "render_domain_id", None), "render domain ID"): (
                _identifier(getattr(item, "packet_id", None), "packet ID")
            )
            for item in anchor_packets
        }
        if anchor_packets:
            from latentguard.vision_data.domains import assigned_render_domain_ids
            from latentguard.vision_data.models import (
                SourceCollection,
                VisualDatasetSplit,
            )

            first = anchor_packets[0]
            required_domains = assigned_render_domain_ids(
                cast(SourceCollection, getattr(first, "source_collection", None)),
                cast(VisualDatasetSplit, getattr(first, "split", None)),
            )
            expected = tuple(by_domain[domain] for domain in required_domains)
        else:
            expected = ()
        if packet_ids != expected or any(item not in by_id for item in packet_ids):
            _fail(candidate_id, "packet inventory differs from anchor variants")
    if len(set(anchor_group_ids.values())) != len(anchor_group_ids):
        _fail("candidate bindings", "candidate group is reused across anchors")


def _validate_samples(
    packets: Sequence[object],
    candidate_bindings: Sequence[object],
    samples: Sequence[object],
) -> None:
    packet_ids = {
        _identifier(getattr(item, "packet_id", None), "packet ID") for item in packets
    }
    bindings = {
        _identifier(
            getattr(item, "candidate_sample_id", None), "candidate sample ID"
        ): tuple(getattr(item, "packet_ids", ()))
        for item in candidate_bindings
    }
    observed: Counter[tuple[str, str]] = Counter()
    candidate_payloads: dict[str, dict[str, object]] = {}
    for sample in samples:
        packet_id = _identifier(getattr(sample, "packet_id", None), "sample packet ID")
        candidate_id = _identifier(
            getattr(sample, "candidate_sample_id", None), "sample candidate ID"
        )
        if packet_id not in packet_ids or packet_id not in bindings.get(
            candidate_id, ()
        ):
            _fail(candidate_id, "sample packet is not candidate-bound")
        observed[(candidate_id, packet_id)] += 1
        mapping_method = getattr(sample, "as_mapping", None)
        if not callable(mapping_method):
            _fail(candidate_id, "sample must expose as_mapping()")
        payload = mapping_method()
        if not isinstance(payload, dict):
            _fail(candidate_id, "sample mapping must be an object")
        comparable = dict(payload)
        comparable.pop("packet_id", None)
        comparable.pop("visual_dataset_digest", None)
        prior = candidate_payloads.setdefault(candidate_id, comparable)
        if prior != comparable:
            _fail(candidate_id, "expanded packet samples disagree on candidate data")
    expected = {
        (candidate_id, packet_id)
        for candidate_id, values in bindings.items()
        for packet_id in values
    }
    if set(observed) != expected or any(count != 1 for count in observed.values()):
        _fail("visual samples", "expanded candidate/packet cross-product differs")


def validate_visual_verifier_dataset(
    dataset: object, *, full_target: bool = True
) -> None:
    """Validate packet grouping, candidate references, mode, and fixed counts."""

    packets = tuple(getattr(dataset, "packets", ()))
    bindings = tuple(getattr(dataset, "candidate_bindings", ()))
    samples = tuple(getattr(dataset, "samples", ()))
    external = type(dataset).__name__ == "VisualVerifierExternalDatasetV1"
    if type(dataset).__name__ not in {
        "VisualVerifierDevelopmentDatasetV1",
        "VisualVerifierExternalDatasetV1",
    }:
        _fail("visual dataset", "unsupported model")
    if getattr(dataset, "training_allowed", None) is not (not external):
        _fail("visual dataset", "training permission differs from source collection")
    expected_manifest = compute_packet_manifest_digest(packets)
    if getattr(dataset, "packet_manifest_digest", None) != expected_manifest:
        _fail("visual dataset", "packet manifest digest mismatch")
    _validate_packet_groups(packets, external=external)
    _validate_candidate_bindings(packets, bindings)
    _validate_samples(packets, bindings, samples)
    if not full_target:
        return
    expected = (
        (
            M3C_EXTERNAL_PACKET_COUNT,
            M3C_CANDIDATE_BINDING_COUNT,
            M3C_EXPANDED_SAMPLE_COUNT,
        )
        if external
        else (
            M3A_DEVELOPMENT_PACKET_COUNT,
            M3A_CANDIDATE_BINDING_COUNT,
            M3A_EXPANDED_SAMPLE_COUNT,
        )
    )
    if (len(packets), len(bindings), len(samples)) != expected:
        _fail("visual dataset", "full-target packet/binding/sample counts differ")
    expected_image_count = (
        M3C_EXTERNAL_IMAGE_COUNT if external else M3A_DEVELOPMENT_IMAGE_COUNT
    )
    image_count = sum(len(tuple(getattr(packet, "views", ()))) for packet in packets)
    if image_count != expected_image_count:
        _fail("visual dataset", "full-target image count differs")
    if not external:
        split_counts = Counter(
            _enum_text(getattr(item, "split", "")) for item in packets
        )
        if split_counts != {"train": 864, "validation": 108, "test": 108}:
            _fail(
                "development dataset", "packet split counts differ from 48/6/6 target"
            )
        trajectory_splits: dict[str, set[str]] = defaultdict(set)
        trajectory_anchors: dict[str, set[str]] = defaultdict(set)
        for packet in packets:
            trajectory = _identifier(
                getattr(packet, "source_trajectory_id", None), "trajectory ID"
            )
            trajectory_splits[_enum_text(getattr(packet, "split", None))].add(
                trajectory
            )
            trajectory_anchors[trajectory].add(
                _identifier(getattr(packet, "anchor_id", None), "anchor ID")
            )
        if {
            split: len(trajectories)
            for split, trajectories in trajectory_splits.items()
        } != {"train": 48, "validation": 6, "test": 6}:
            _fail("development dataset", "trajectory split counts differ from 48/6/6")
        if any(len(anchors) != 6 for anchors in trajectory_anchors.values()):
            _fail("development dataset", "each trajectory must contribute six anchors")
    else:
        trajectory_anchors = defaultdict(set)
        for packet in packets:
            trajectory_anchors[
                _identifier(
                    getattr(packet, "source_trajectory_id", None), "trajectory ID"
                )
            ].add(_identifier(getattr(packet, "anchor_id", None), "anchor ID"))
        if len(trajectory_anchors) != 60 or any(
            len(anchors) != 6 for anchors in trajectory_anchors.values()
        ):
            _fail("external dataset", "expected 60 trajectories with six anchors each")
    bindings_by_anchor = Counter(
        _identifier(getattr(binding, "anchor_id", None), "anchor ID")
        for binding in bindings
    )
    expected_per_anchor = 8 if external else 9
    if len(bindings_by_anchor) != 360 or set(bindings_by_anchor.values()) != {
        expected_per_anchor
    }:
        _fail("visual dataset", "candidate group size or anchor count differs")


def build_visual_development_dataset(
    *,
    source_dataset_digest: str,
    split_digest: str,
    evidence_digest: str,
    source_compatibility_identity: str,
    packets: Sequence[object],
    candidate_bindings: Sequence[object],
    samples: Sequence[object],
    camera_rig_digest: str,
    render_domain_configuration_digest: str,
    visual_compatibility_identity: str,
    full_target: bool = True,
) -> object:
    """Construct and validate the M3A-derived development visual dataset."""

    from latentguard.vision_data.models import (
        VisualActionVerifierSampleV1,
        VisualCandidateBindingV1,
        VisualVerifierDevelopmentDatasetV1,
    )
    from latentguard.vision_data.packet import VisualObservationPacketV1

    values = tuple(packets)
    dataset = VisualVerifierDevelopmentDatasetV1.create(
        source_dataset_digest=source_dataset_digest,
        split_digest=split_digest,
        evidence_digest=evidence_digest,
        source_compatibility_identity=source_compatibility_identity,
        packet_manifest_digest=compute_packet_manifest_digest(values),
        packets=cast(tuple[VisualObservationPacketV1, ...], values),
        candidate_bindings=cast(
            tuple[VisualCandidateBindingV1, ...], tuple(candidate_bindings)
        ),
        samples=cast(tuple[VisualActionVerifierSampleV1, ...], tuple(samples)),
        camera_rig_digest=camera_rig_digest,
        render_domain_configuration_digest=render_domain_configuration_digest,
        visual_compatibility_identity=visual_compatibility_identity,
    )
    validate_visual_verifier_dataset(dataset, full_target=full_target)
    return dataset


def build_visual_external_dataset(
    *,
    source_set_identity: str,
    candidate_pool_identity: str,
    blind_manifest_digest: str,
    full_outcome_digest: str,
    source_compatibility_identity: str,
    packets: Sequence[object],
    candidate_bindings: Sequence[object],
    samples: Sequence[object],
    camera_rig_digest: str,
    render_domain_configuration_digest: str,
    visual_compatibility_identity: str,
    full_target: bool = True,
) -> object:
    """Construct and validate the training-prohibited M3C external dataset."""

    from latentguard.vision_data.models import (
        VisualActionVerifierSampleV1,
        VisualCandidateBindingV1,
        VisualVerifierExternalDatasetV1,
    )
    from latentguard.vision_data.packet import VisualObservationPacketV1

    values = tuple(packets)
    dataset = VisualVerifierExternalDatasetV1.create(
        source_set_identity=source_set_identity,
        candidate_pool_identity=candidate_pool_identity,
        blind_manifest_digest=blind_manifest_digest,
        full_outcome_digest=full_outcome_digest,
        source_compatibility_identity=source_compatibility_identity,
        packet_manifest_digest=compute_packet_manifest_digest(values),
        packets=cast(tuple[VisualObservationPacketV1, ...], values),
        candidate_bindings=cast(
            tuple[VisualCandidateBindingV1, ...], tuple(candidate_bindings)
        ),
        samples=cast(tuple[VisualActionVerifierSampleV1, ...], tuple(samples)),
        camera_rig_digest=camera_rig_digest,
        render_domain_configuration_digest=render_domain_configuration_digest,
        visual_compatibility_identity=visual_compatibility_identity,
    )
    validate_visual_verifier_dataset(dataset, full_target=full_target)
    return dataset


def require_training_allowed(dataset: object) -> None:
    """Fail closed when an evaluation-only dataset reaches a training loader."""

    if getattr(dataset, "training_allowed", None) is not True:
        _fail("training loader", "visual dataset is evaluation-only")


__all__ = [
    "M3A_CANDIDATE_BINDING_COUNT",
    "M3A_DEVELOPMENT_IMAGE_COUNT",
    "M3A_DEVELOPMENT_PACKET_COUNT",
    "M3A_EXPANDED_SAMPLE_COUNT",
    "M3C_CANDIDATE_BINDING_COUNT",
    "M3C_EXPANDED_SAMPLE_COUNT",
    "M3C_EXTERNAL_IMAGE_COUNT",
    "M3C_EXTERNAL_PACKET_COUNT",
    "VisualDatasetError",
    "build_visual_development_dataset",
    "build_visual_external_dataset",
    "compute_packet_manifest_digest",
    "require_training_allowed",
    "validate_visual_verifier_dataset",
]
