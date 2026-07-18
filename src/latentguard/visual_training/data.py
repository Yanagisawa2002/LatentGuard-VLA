"""Metadata-isolated M4B data projection and deterministic domain cycling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from latentguard.action_verifier import CandidateType, DatasetSplit
from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
from latentguard.vision_data.models import (
    SourceCollection,
    VisualActionVerifierSampleV1,
    VisualDatasetSplit,
    VisualVerifierDevelopmentDatasetV1,
    VisualVerifierExternalDatasetV1,
)

DEVELOPMENT_VISUAL_DATASET_DIGEST = (
    "sha256:f79be2b357a9de68e1deb80132f7915717158bec506884554536882e3adbd339"
)
EXTERNAL_VISUAL_DATASET_DIGEST = (
    "sha256:62e762acf6a441579c494e1aab6e873e868594cead40f256be2d809abcbeccef"
)
TRAIN_DOMAIN_IDS = ("canonical", "mild_camera_shift", "mild_lighting_shift")
TEST_DOMAIN_IDS = ("canonical", "strong_camera_shift", "strong_lighting_shift")


class VisualTrainingDataError(ValueError):
    """Raised when visual data crosses a permission or integrity boundary."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualTrainingDataError(f"{context}: {reason}")


def _freeze(value: NDArray[Any]) -> NDArray[Any]:
    result = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(result.tobytes(order="C"), dtype=result.dtype).reshape(
        result.shape
    )


@dataclass(frozen=True, slots=True, eq=False)
class VisualStudentExampleV1:
    """Deployable inputs and target only; no privileged or reporting metadata."""

    action_chunk: NDArray[Any]
    action_mask: NDArray[Any]
    failure_target: int
    sample_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action_chunk, np.ndarray)
            or self.action_chunk.dtype != np.dtype("<f8")
            or self.action_chunk.shape != (16, 8)
            or not bool(np.all(np.isfinite(self.action_chunk)))
        ):
            _fail(
                "VisualStudentExampleV1.action_chunk", "expected finite float64 [16,8]"
            )
        if (
            not isinstance(self.action_mask, np.ndarray)
            or self.action_mask.dtype != np.dtype(np.bool_)
            or self.action_mask.shape != (16,)
            or not bool(np.any(self.action_mask))
        ):
            _fail("VisualStudentExampleV1.action_mask", "expected nonempty bool [16]")
        if type(self.failure_target) is not int or self.failure_target not in (0, 1):
            _fail("VisualStudentExampleV1.failure_target", "expected binary int")
        if type(self.sample_index) is not int or self.sample_index < 0:
            _fail("VisualStudentExampleV1.sample_index", "expected non-negative int")
        object.__setattr__(self, "action_chunk", _freeze(self.action_chunk))
        object.__setattr__(self, "action_mask", _freeze(self.action_mask))


@dataclass(frozen=True, slots=True)
class VisualReportingMetadataV1:
    """Non-deployable join metadata retained exclusively for reporting."""

    sample_index: int
    candidate_sample_id: str
    candidate_group_id: str
    anchor_id: str
    split: DatasetSplit
    source_trajectory_id: str
    candidate_type: CandidateType
    packet_ids_by_domain: MappingProxyType[str, str]


@dataclass(frozen=True, slots=True)
class VisualActionTrainingDatasetV1:
    """Accepted M3A student projection with a separate packet/reporting join."""

    visual_dataset_digest: str
    structured_dataset_digest: str
    split_digest: str
    examples: tuple[VisualStudentExampleV1, ...]
    reporting: tuple[VisualReportingMetadataV1, ...]
    split_indices: MappingProxyType[DatasetSplit, tuple[int, ...]]

    def __post_init__(self) -> None:
        if len(self.examples) != 3_240 or len(self.reporting) != 3_240:
            _fail("VisualActionTrainingDatasetV1", "expected 3,240 aligned candidates")
        if any(item.sample_index != i for i, item in enumerate(self.examples)):
            _fail("VisualActionTrainingDatasetV1", "example ordering changed")
        if any(item.sample_index != i for i, item in enumerate(self.reporting)):
            _fail("VisualActionTrainingDatasetV1", "reporting ordering changed")

    def indices_for_split(self, split: DatasetSplit) -> tuple[int, ...]:
        """Return exact preserved M3A sample indices for one split."""
        return self.split_indices[split]


def domain_index_for_epoch(candidate_sample_id: str, *, epoch: int, seed: int) -> int:
    """Select one of three domains once per epoch with complete 3-epoch cycling."""
    if not isinstance(candidate_sample_id, str) or not candidate_sample_id:
        _fail("domain cycling", "candidate_sample_id is required")
    if type(epoch) is not int or epoch < 0 or type(seed) is not int or seed < 0:
        _fail("domain cycling", "epoch and seed must be non-negative integers")
    base = int.from_bytes(
        hashlib.sha256(f"{candidate_sample_id}\0{seed}".encode()).digest()[:8],
        "big",
    ) % len(TRAIN_DOMAIN_IDS)
    return (base + epoch) % len(TRAIN_DOMAIN_IDS)


def packet_id_for_epoch(
    metadata: VisualReportingMetadataV1, *, epoch: int, seed: int
) -> str:
    """Resolve a deterministic training packet without using outcome metadata."""
    domain = TRAIN_DOMAIN_IDS[
        domain_index_for_epoch(metadata.candidate_sample_id, epoch=epoch, seed=seed)
    ]
    try:
        return metadata.packet_ids_by_domain[domain]
    except KeyError as exc:
        raise VisualTrainingDataError(
            f"domain cycling: missing packet for {domain}"
        ) from exc


def require_development_training_dataset(
    dataset: object,
) -> VisualVerifierDevelopmentDatasetV1:
    """Reject every external or permission-disabled visual dataset at training entry."""
    if not isinstance(dataset, VisualVerifierDevelopmentDatasetV1):
        if isinstance(dataset, VisualVerifierExternalDatasetV1):
            _fail("visual training dataset", "M3C external data is evaluation-only")
        _fail("visual training dataset", "expected M3A development dataset")
    if dataset.source_collection is not SourceCollection.M3A_DEVELOPMENT or (
        dataset.training_allowed is not True
    ):
        _fail("visual training dataset", "training permission is absent")
    if dataset.content_digest != DEVELOPMENT_VISUAL_DATASET_DIGEST:
        _fail("visual training dataset", "accepted M3A visual digest differs")
    return dataset


def build_visual_action_training_dataset(
    visual_dataset: object,
    structured_dataset: AcceptedActionVerifierDatasetV1,
) -> VisualActionTrainingDatasetV1:
    """Join accepted labels/actions to visual packets without exposing state inputs."""
    visual = require_development_training_dataset(visual_dataset)
    if not isinstance(structured_dataset, AcceptedActionVerifierDatasetV1):
        _fail("structured dataset", "expected accepted M3A projection")
    if visual.source_dataset_digest != structured_dataset.dataset_digest:
        _fail("visual/structured join", "structured dataset digest differs")
    if visual.split_digest != structured_dataset.split_digest:
        _fail("visual/structured join", "split digest differs")

    packets = {packet.packet_id: packet for packet in visual.packets}
    bindings = {item.candidate_sample_id: item for item in visual.candidate_bindings}
    samples_by_candidate: dict[str, list[VisualActionVerifierSampleV1]] = {}
    for sample in visual.samples:
        samples_by_candidate.setdefault(sample.candidate_sample_id, []).append(sample)
    if len(bindings) != len(structured_dataset) or len(samples_by_candidate) != len(
        bindings
    ):
        _fail("visual/structured join", "candidate coverage differs")

    examples: list[VisualStudentExampleV1] = []
    reporting: list[VisualReportingMetadataV1] = []
    split_indices: dict[DatasetSplit, list[int]] = {split: [] for split in DatasetSplit}
    split_map = {
        DatasetSplit.TRAIN: VisualDatasetSplit.TRAIN,
        DatasetSplit.VALIDATION: VisualDatasetSplit.VALIDATION,
        DatasetSplit.TEST: VisualDatasetSplit.TEST,
    }
    for index, (source, report) in enumerate(
        zip(structured_dataset.examples, structured_dataset.reporting, strict=True)
    ):
        binding = bindings.get(report.sample_id)
        if binding is None:
            _fail("visual/structured join", "candidate ID is missing")
        packet_values = tuple(packets[item] for item in binding.packet_ids)
        expected_domains = (
            TRAIN_DOMAIN_IDS
            if report.dataset_split is not DatasetSplit.TEST
            else TEST_DOMAIN_IDS
        )
        domains = tuple(item.render_domain_id for item in packet_values)
        if domains != expected_domains:
            _fail("visual/structured join", "packet domain order differs")
        if any(
            item.split is not split_map[report.dataset_split]
            or item.source_trajectory_id != report.source_trajectory
            or item.anchor_id != report.anchor_id
            for item in packet_values
        ):
            _fail("visual/structured join", "packet provenance differs")
        sample_rows = samples_by_candidate[report.sample_id]
        if len(sample_rows) != 3 or any(
            item.final_success != (source.failure_target == 0) for item in sample_rows
        ):
            _fail("visual/structured join", "candidate target binding differs")
        examples.append(
            VisualStudentExampleV1(
                action_chunk=source.action_chunk,
                action_mask=source.action_mask,
                failure_target=source.failure_target,
                sample_index=index,
            )
        )
        reporting.append(
            VisualReportingMetadataV1(
                sample_index=index,
                candidate_sample_id=report.sample_id,
                candidate_group_id=report.group_id,
                anchor_id=report.anchor_id,
                split=report.dataset_split,
                source_trajectory_id=report.source_trajectory,
                candidate_type=report.candidate_type,
                packet_ids_by_domain=MappingProxyType(
                    dict(zip(domains, binding.packet_ids, strict=True))
                ),
            )
        )
        split_indices[report.dataset_split].append(index)
    return VisualActionTrainingDatasetV1(
        visual_dataset_digest=visual.content_digest,
        structured_dataset_digest=structured_dataset.dataset_digest,
        split_digest=structured_dataset.split_digest,
        examples=tuple(examples),
        reporting=tuple(reporting),
        split_indices=MappingProxyType(
            {key: tuple(value) for key, value in split_indices.items()}
        ),
    )


__all__ = [
    "DEVELOPMENT_VISUAL_DATASET_DIGEST",
    "EXTERNAL_VISUAL_DATASET_DIGEST",
    "TEST_DOMAIN_IDS",
    "TRAIN_DOMAIN_IDS",
    "VisualActionTrainingDatasetV1",
    "VisualReportingMetadataV1",
    "VisualStudentExampleV1",
    "VisualTrainingDataError",
    "build_visual_action_training_dataset",
    "domain_index_for_epoch",
    "packet_id_for_epoch",
    "require_development_training_dataset",
]
