from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace

import pytest

from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST
from latentguard.vision_data import (
    M3A_CANDIDATE_BINDING_COUNT,
    M3A_DEVELOPMENT_IMAGE_COUNT,
    M3A_DEVELOPMENT_PACKET_COUNT,
    M3A_EXPANDED_SAMPLE_COUNT,
    M3C_CANDIDATE_BINDING_COUNT,
    M3C_EXPANDED_SAMPLE_COUNT,
    M3C_EXTERNAL_IMAGE_COUNT,
    M3C_EXTERNAL_PACKET_COUNT,
    PICKCUBE_CANONICAL_TASK_TEXT,
    PICKCUBE_VISUAL_TASK_ID,
    SourceCollection,
    VisualActionVerifierSampleV1,
    VisualCandidateBindingV1,
    VisualDatasetError,
    VisualDatasetSplit,
    assigned_render_domain_ids,
    build_visual_development_dataset,
    build_visual_external_dataset,
    load_camera_rig_configuration,
    load_render_domain_configuration,
    validate_visual_verifier_dataset,
)
from latentguard.vision_data.source_binding import visual_candidate_array_reference


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _packet_id(label: str) -> str:
    return f"vop-sha256-{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class _CountView:
    pixel_sha256: str


@dataclass(frozen=True, slots=True)
class _CountPacket:
    packet_id: str
    content_digest: str
    source_collection: SourceCollection
    source_trajectory_id: str
    anchor_id: str
    split: VisualDatasetSplit
    split_group_id: str
    render_domain_id: str
    views: tuple[_CountView, ...]

    def as_mapping(self) -> dict[str, object]:
        """Return compact canonical content for count-focused fake packets."""

        return {
            "anchor_id": self.anchor_id,
            "content_digest": self.content_digest,
            "packet_id": self.packet_id,
            "render_domain_id": self.render_domain_id,
            "source_collection": self.source_collection.value,
            "source_trajectory_id": self.source_trajectory_id,
            "split": self.split.value,
            "split_group_id": self.split_group_id,
            "views": [view.pixel_sha256 for view in self.views],
        }


def _development_split(trajectory_index: int) -> VisualDatasetSplit:
    if trajectory_index < 48:
        return VisualDatasetSplit.TRAIN
    if trajectory_index < 54:
        return VisualDatasetSplit.VALIDATION
    return VisualDatasetSplit.TEST


def _full_components(
    collection: SourceCollection,
) -> tuple[
    tuple[_CountPacket, ...],
    tuple[VisualCandidateBindingV1, ...],
    tuple[VisualActionVerifierSampleV1, ...],
]:
    packets: list[_CountPacket] = []
    bindings: list[VisualCandidateBindingV1] = []
    samples: list[VisualActionVerifierSampleV1] = []
    candidate_count = 9 if collection is SourceCollection.M3A_DEVELOPMENT else 8
    source_digest = (
        ACCEPTED_M3A_DATASET_DIGEST
        if collection is SourceCollection.M3A_DEVELOPMENT
        else _digest("m3c-source-set")
    )
    candidate_digest = (
        ACCEPTED_M3A_DATASET_DIGEST
        if collection is SourceCollection.M3A_DEVELOPMENT
        else _digest("m3c-candidate-pool")
    )
    for trajectory_index in range(60):
        split = (
            _development_split(trajectory_index)
            if collection is SourceCollection.M3A_DEVELOPMENT
            else VisualDatasetSplit.EXTERNAL
        )
        trajectory_id = f"{collection.value}-trajectory-{trajectory_index:02d}"
        split_group_id = f"{collection.value}-split-{trajectory_index:02d}"
        domains = assigned_render_domain_ids(collection, split)
        for anchor_index in range(6):
            anchor_id = f"{trajectory_id}-anchor-{anchor_index}"
            anchor_packets: list[_CountPacket] = []
            for domain_id in domains:
                identity = f"{anchor_id}-{domain_id}"
                packet = _CountPacket(
                    packet_id=_packet_id(identity),
                    content_digest=_digest(f"packet-content-{identity}"),
                    source_collection=collection,
                    source_trajectory_id=trajectory_id,
                    anchor_id=anchor_id,
                    split=split,
                    split_group_id=split_group_id,
                    render_domain_id=domain_id,
                    views=(
                        _CountView(_digest(f"pixel-{identity}-0")),
                        _CountView(_digest(f"pixel-{identity}-1")),
                        _CountView(_digest(f"pixel-{identity}-2")),
                    ),
                )
                packets.append(packet)
                anchor_packets.append(packet)
            packet_ids = tuple(packet.packet_id for packet in anchor_packets)
            assert len(packet_ids) == 3
            candidate_group_id = f"{anchor_id}-candidate-group"
            for candidate_index in range(candidate_count):
                candidate_id = f"{anchor_id}-candidate-{candidate_index}"
                binding = VisualCandidateBindingV1(
                    candidate_sample_id=candidate_id,
                    candidate_group_id=candidate_group_id,
                    anchor_id=anchor_id,
                    source_collection=collection,
                    packet_ids=packet_ids,
                )
                bindings.append(binding)
                for packet_id in packet_ids:
                    samples.append(
                        VisualActionVerifierSampleV1(
                            packet_id=packet_id,
                            candidate_sample_id=candidate_id,
                            task_id=PICKCUBE_VISUAL_TASK_ID,
                            canonical_task_text=PICKCUBE_CANONICAL_TASK_TEXT,
                            candidate_action_chunk_reference=(
                                visual_candidate_array_reference(
                                    candidate_id, "action-chunk"
                                )
                            ),
                            action_mask_reference=visual_candidate_array_reference(
                                candidate_id, "action-mask"
                            ),
                            final_success=candidate_index == 0,
                            final_unsafe=False,
                            failure_events=(),
                            evidence_id=f"evidence-{candidate_id}",
                            source_dataset_digest=source_digest,
                            candidate_dataset_digest=candidate_digest,
                            visual_dataset_digest=_digest("pending-visual-dataset"),
                            source_collection=collection,
                        )
                    )
    return tuple(packets), tuple(bindings), tuple(samples)


def test_complete_m3a_target_counts_and_48_6_6_split_are_accepted() -> None:
    packets, bindings, samples = _full_components(SourceCollection.M3A_DEVELOPMENT)
    rig = load_camera_rig_configuration("configs/vision/m4a/camera-rig-v1.json")
    domains = load_render_domain_configuration(
        "configs/vision/m4a/render-domains-v1.json"
    )

    dataset = build_visual_development_dataset(
        source_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
        split_digest=_digest("m3a-split"),
        evidence_digest=_digest("m3a-evidence"),
        source_compatibility_identity=_digest("m3a-compatibility"),
        packets=packets,
        candidate_bindings=bindings,
        samples=samples,
        camera_rig_digest=rig.content_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        full_target=True,
    )
    validate_visual_verifier_dataset(dataset, full_target=True)
    assert len(dataset.packets) == M3A_DEVELOPMENT_PACKET_COUNT
    assert sum(len(packet.views) for packet in dataset.packets) == (
        M3A_DEVELOPMENT_IMAGE_COUNT
    )
    assert len(dataset.candidate_bindings) == M3A_CANDIDATE_BINDING_COUNT
    assert len(dataset.samples) == M3A_EXPANDED_SAMPLE_COUNT
    assert Counter(packet.split for packet in dataset.packets) == {
        VisualDatasetSplit.TRAIN: 864,
        VisualDatasetSplit.VALIDATION: 108,
        VisualDatasetSplit.TEST: 108,
    }


def test_complete_m3c_external_target_is_evaluation_only_and_accepted() -> None:
    packets, bindings, samples = _full_components(SourceCollection.M3C_EXTERNAL)
    rig = load_camera_rig_configuration("configs/vision/m4a/camera-rig-v1.json")
    domains = load_render_domain_configuration(
        "configs/vision/m4a/render-domains-v1.json"
    )

    dataset = build_visual_external_dataset(
        source_set_identity=_digest("m3c-source-set"),
        candidate_pool_identity=_digest("m3c-candidate-pool"),
        blind_manifest_digest=_digest("m3c-blind-manifest"),
        full_outcome_digest=_digest("m3c-full-outcome"),
        source_compatibility_identity=_digest("m3c-compatibility"),
        packets=packets,
        candidate_bindings=bindings,
        samples=samples,
        camera_rig_digest=rig.content_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        full_target=True,
    )
    validate_visual_verifier_dataset(dataset, full_target=True)
    assert dataset.training_allowed is False
    assert len(dataset.packets) == M3C_EXTERNAL_PACKET_COUNT
    assert sum(len(packet.views) for packet in dataset.packets) == (
        M3C_EXTERNAL_IMAGE_COUNT
    )
    assert len(dataset.candidate_bindings) == M3C_CANDIDATE_BINDING_COUNT
    assert len(dataset.samples) == M3C_EXPANDED_SAMPLE_COUNT
    assert {packet.split for packet in dataset.packets} == {VisualDatasetSplit.EXTERNAL}


def test_full_target_rejects_one_missing_authoritative_image() -> None:
    packets, bindings, samples = _full_components(SourceCollection.M3A_DEVELOPMENT)
    packets = (
        replace(packets[0], views=packets[0].views[:2]),
        *packets[1:],
    )
    rig = load_camera_rig_configuration("configs/vision/m4a/camera-rig-v1.json")
    domains = load_render_domain_configuration(
        "configs/vision/m4a/render-domains-v1.json"
    )
    dataset = build_visual_development_dataset(
        source_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
        split_digest=_digest("m3a-split"),
        evidence_digest=_digest("m3a-evidence"),
        source_compatibility_identity=_digest("m3a-compatibility"),
        packets=packets,
        candidate_bindings=bindings,
        samples=samples,
        camera_rig_digest=rig.content_digest,
        render_domain_configuration_digest=domains.content_digest,
        visual_compatibility_identity=_digest("visual-compatibility"),
        full_target=False,
    )

    with pytest.raises(VisualDatasetError, match="image count differs"):
        validate_visual_verifier_dataset(dataset, full_target=True)
