"""Immutable model-ready contracts for action-verifier datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.models import FailureEvent, LabelSource, LabelStrength

ACTION_VERIFIER_SCHEMA_VERSION = "1.0"
"""Logical schema version for the M3A action-verifier dataset."""

FIXED_ACTION_CHUNK_HORIZON = 16
"""Reference candidate horizon required by M3A."""


def _freeze_array(value: NDArray[Any]) -> NDArray[Any]:
    """Return a detached C-order array backed by immutable bytes."""

    detached = np.array(value, copy=True, order="C", subok=False)
    if detached.dtype.hasobject:
        detached.setflags(write=False)
        return detached
    immutable = detached.tobytes(order="C")
    return np.frombuffer(immutable, dtype=detached.dtype).reshape(detached.shape)


class DatasetSplit(StrEnum):
    """Source-trajectory-level dataset partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class CandidateType(StrEnum):
    """Whether a verifier candidate is original or corrupted."""

    SOURCE = "source"
    CORRUPTED = "corrupted"


@dataclass(frozen=True, slots=True, eq=False)
class ActionVerifierSampleV1:
    """One compact, strongly labeled action-verification training sample."""

    anchor_id: str
    source_trajectory_id: str
    source_seed: int
    split_group_id: str
    dataset_split: DatasetSplit
    task_id: str
    instruction: str
    state_content_digest: str
    state_vector_semantic: str
    state_vector_schema_digest: str
    state_vector: NDArray[Any]
    candidate_action_chunk: NDArray[Any]
    action_mask: NDArray[Any]
    continuation_identity: str
    candidate_type: CandidateType
    proposal_id: str | None
    corruption_type: str | None
    severity_id: str | None
    final_task_success: bool
    progress_semantic: str | None
    progress_before: float | None
    progress_after_candidate_chunk: float | None
    progress_delta: float | None
    final_unsafe: bool
    failure_events: tuple[FailureEvent, ...]
    strong_simulator_evidence_id: str
    evidence_dataset_digest: str
    source_dataset_digest: str
    corruption_dataset_digest: str
    compatibility_identity: str
    adapter_version: str
    label_source: LabelSource
    label_strength: LabelStrength
    simulator_replay_verified: bool
    schema_version: str = ACTION_VERIFIER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach arrays and validate the complete training-sample contract."""

        if isinstance(self.state_vector, np.ndarray):
            object.__setattr__(self, "state_vector", _freeze_array(self.state_vector))
        if isinstance(self.candidate_action_chunk, np.ndarray):
            object.__setattr__(
                self,
                "candidate_action_chunk",
                _freeze_array(self.candidate_action_chunk),
            )
        if isinstance(self.action_mask, np.ndarray):
            object.__setattr__(self, "action_mask", _freeze_array(self.action_mask))
        object.__setattr__(self, "failure_events", tuple(self.failure_events))
        from latentguard.action_verifier.validation import (
            validate_action_verifier_sample,
        )

        validate_action_verifier_sample(self)

    @property
    def sample_id(self) -> str:
        """Return the canonical identifier derived from all sample content."""

        from latentguard.action_verifier.identity import (
            compute_action_verifier_sample_identifier,
        )

        return compute_action_verifier_sample_identifier(self)


@dataclass(frozen=True, slots=True)
class ActionVerifierCandidateGroupV1:
    """One anchor with one source sample and all evaluated corruptions."""

    anchor_id: str
    source_trajectory_id: str
    source_seed: int
    split_group_id: str
    dataset_split: DatasetSplit
    task_id: str
    state_content_digest: str
    state_vector_semantic: str
    continuation_identity: str
    source_sample_id: str
    corrupted_sample_ids: tuple[str, ...]
    baseline_evidence_id: str
    schema_version: str = ACTION_VERIFIER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze sample references and validate group-level identity."""

        object.__setattr__(
            self, "corrupted_sample_ids", tuple(self.corrupted_sample_ids)
        )
        from latentguard.action_verifier.validation import (
            validate_action_verifier_candidate_group,
        )

        validate_action_verifier_candidate_group(self)

    @property
    def group_id(self) -> str:
        """Return the canonical group identifier."""

        from latentguard.action_verifier.identity import (
            compute_candidate_group_identifier,
        )

        return compute_candidate_group_identifier(self)


@dataclass(frozen=True, slots=True)
class TrajectorySplitAssignmentV1:
    """One source trajectory and every leakage-sensitive identity assigned together."""

    split_policy_id: str
    source_trajectory_id: str
    source_seed: int
    split_group_id: str
    dataset_split: DatasetSplit
    state_digests: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    proposal_ids: tuple[str, ...]
    schema_version: str = ACTION_VERIFIER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze inventories and validate deterministic assignment identity."""

        object.__setattr__(self, "state_digests", tuple(self.state_digests))
        object.__setattr__(self, "anchor_ids", tuple(self.anchor_ids))
        object.__setattr__(self, "proposal_ids", tuple(self.proposal_ids))
        from latentguard.action_verifier.validation import (
            validate_trajectory_split_assignment,
        )

        validate_trajectory_split_assignment(self)

    @property
    def assignment_id(self) -> str:
        """Return the canonical trajectory-assignment identifier."""

        from latentguard.action_verifier.identity import (
            compute_split_assignment_identifier,
        )

        return compute_split_assignment_identifier(self)


@dataclass(frozen=True, slots=True, eq=False)
class ActionVerifierDatasetV1:
    """Complete compact dataset, candidate groups, and trajectory split bindings."""

    state_vector_semantic: str
    state_vector_schema_digest: str
    state_vector_dimension: int
    action_dimension: int
    split_policy_id: str
    samples: tuple[ActionVerifierSampleV1, ...]
    candidate_groups: tuple[ActionVerifierCandidateGroupV1, ...]
    split_assignments: tuple[TrajectorySplitAssignmentV1, ...]
    chunk_horizon: int = FIXED_ACTION_CHUNK_HORIZON
    schema_version: str = ACTION_VERIFIER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze ordered collections and validate every cross-reference."""

        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "candidate_groups", tuple(self.candidate_groups))
        object.__setattr__(self, "split_assignments", tuple(self.split_assignments))
        from latentguard.action_verifier.validation import (
            validate_action_verifier_dataset,
        )

        validate_action_verifier_dataset(self)

    @property
    def content_digest(self) -> str:
        """Return the path-independent digest of all ordered dataset content."""

        from latentguard.action_verifier.identity import (
            compute_action_verifier_dataset_digest,
        )

        return compute_action_verifier_dataset_digest(self)


__all__ = [
    "ACTION_VERIFIER_SCHEMA_VERSION",
    "FIXED_ACTION_CHUNK_HORIZON",
    "ActionVerifierCandidateGroupV1",
    "ActionVerifierDatasetV1",
    "ActionVerifierSampleV1",
    "CandidateType",
    "DatasetSplit",
    "TrajectorySplitAssignmentV1",
]
