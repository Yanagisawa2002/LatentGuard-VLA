"""Canonical path-independent identities for action-verifier content."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.action_verifier.models import (
    ActionVerifierCandidateGroupV1,
    ActionVerifierDatasetV1,
    ActionVerifierSampleV1,
    DatasetSplit,
    TrajectorySplitAssignmentV1,
)
from latentguard.models import FailureEvent
from latentguard.replay.identity import canonical_json_bytes

SAMPLE_ID_PREFIX = "avs-sha256-"
GROUP_ID_PREFIX = "avg-sha256-"
ASSIGNMENT_ID_PREFIX = "avt-sha256-"
SPLIT_POLICY_ID_PREFIX = "avp-sha256-"
DATASET_DIGEST_PREFIX = "sha256:"


def _digest(payload: object, *, prefix: str, context: str) -> str:
    encoded = canonical_json_bytes(payload, context=context)
    return f"{prefix}{hashlib.sha256(encoded).hexdigest()}"


def _array_payload(array: NDArray[Any]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "content_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }


def _failure_event_payload(event: FailureEvent) -> dict[str, object]:
    return {
        "description": event.description,
        "failure_type": event.failure_type,
        "probability": event.probability,
        "schema_version": event.schema_version,
        "timestamp_s": event.timestamp_s,
    }


def _sample_payload(sample: ActionVerifierSampleV1) -> dict[str, object]:
    return {
        "action_mask": _array_payload(sample.action_mask),
        "adapter_version": sample.adapter_version,
        "anchor_id": sample.anchor_id,
        "candidate_action_chunk": _array_payload(sample.candidate_action_chunk),
        "candidate_type": sample.candidate_type.value,
        "compatibility_identity": sample.compatibility_identity,
        "continuation_identity": sample.continuation_identity,
        "corruption_dataset_digest": sample.corruption_dataset_digest,
        "corruption_type": sample.corruption_type,
        "dataset_split": sample.dataset_split.value,
        "evidence_dataset_digest": sample.evidence_dataset_digest,
        "failure_events": [
            _failure_event_payload(event) for event in sample.failure_events
        ],
        "final_task_success": sample.final_task_success,
        "final_unsafe": sample.final_unsafe,
        "instruction": sample.instruction,
        "label_source": sample.label_source.value,
        "label_strength": sample.label_strength.value,
        "progress_after_candidate_chunk": sample.progress_after_candidate_chunk,
        "progress_before": sample.progress_before,
        "progress_delta": sample.progress_delta,
        "progress_semantic": sample.progress_semantic,
        "proposal_id": sample.proposal_id,
        "schema_version": sample.schema_version,
        "severity_id": sample.severity_id,
        "simulator_replay_verified": sample.simulator_replay_verified,
        "source_dataset_digest": sample.source_dataset_digest,
        "source_seed": sample.source_seed,
        "source_trajectory_id": sample.source_trajectory_id,
        "split_group_id": sample.split_group_id,
        "state_content_digest": sample.state_content_digest,
        "state_vector": _array_payload(sample.state_vector),
        "state_vector_schema_digest": sample.state_vector_schema_digest,
        "state_vector_semantic": sample.state_vector_semantic,
        "strong_simulator_evidence_id": sample.strong_simulator_evidence_id,
        "task_id": sample.task_id,
    }


def compute_action_verifier_sample_identifier(
    sample: ActionVerifierSampleV1,
) -> str:
    """Hash every sample field except its supplied identifier."""

    return _digest(
        _sample_payload(sample),
        prefix=SAMPLE_ID_PREFIX,
        context="ActionVerifierSampleV1Identity",
    )


def _group_payload(group: ActionVerifierCandidateGroupV1) -> dict[str, object]:
    return {
        "anchor_id": group.anchor_id,
        "baseline_evidence_id": group.baseline_evidence_id,
        "continuation_identity": group.continuation_identity,
        "corrupted_sample_ids": list(group.corrupted_sample_ids),
        "dataset_split": group.dataset_split.value,
        "schema_version": group.schema_version,
        "source_sample_id": group.source_sample_id,
        "source_seed": group.source_seed,
        "source_trajectory_id": group.source_trajectory_id,
        "split_group_id": group.split_group_id,
        "state_content_digest": group.state_content_digest,
        "state_vector_semantic": group.state_vector_semantic,
        "task_id": group.task_id,
    }


def compute_candidate_group_identifier(
    group: ActionVerifierCandidateGroupV1,
) -> str:
    """Hash the exact state, continuation, and ordered sample membership."""

    return _digest(
        _group_payload(group),
        prefix=GROUP_ID_PREFIX,
        context="ActionVerifierCandidateGroupV1Identity",
    )


def compute_split_policy_identifier(
    *,
    source_trajectory_ids: Sequence[str],
    split_counts: Mapping[DatasetSplit, int],
    split_seed: int,
) -> str:
    """Hash a resolved split policy and its complete trajectory population."""

    payload = {
        "schema_version": "1.0",
        "semantic": "source_trajectory_hash_order_v1",
        "source_trajectory_ids": sorted(source_trajectory_ids),
        "split_counts": {
            split.value: split_counts.get(split, 0) for split in DatasetSplit
        },
        "split_seed": split_seed,
    }
    return _digest(
        payload,
        prefix=SPLIT_POLICY_ID_PREFIX,
        context="TrajectorySplitPolicyV1Identity",
    )


def _assignment_payload(
    assignment: TrajectorySplitAssignmentV1,
) -> dict[str, object]:
    return {
        "anchor_ids": list(assignment.anchor_ids),
        "dataset_split": assignment.dataset_split.value,
        "proposal_ids": list(assignment.proposal_ids),
        "schema_version": assignment.schema_version,
        "source_seed": assignment.source_seed,
        "source_trajectory_id": assignment.source_trajectory_id,
        "split_group_id": assignment.split_group_id,
        "split_policy_id": assignment.split_policy_id,
        "state_digests": list(assignment.state_digests),
    }


def compute_split_assignment_identifier(
    assignment: TrajectorySplitAssignmentV1,
) -> str:
    """Hash one complete leakage-sensitive trajectory assignment."""

    return _digest(
        _assignment_payload(assignment),
        prefix=ASSIGNMENT_ID_PREFIX,
        context="TrajectorySplitAssignmentV1Identity",
    )


def compute_action_verifier_dataset_digest(
    dataset: ActionVerifierDatasetV1,
) -> str:
    """Hash every logical field and array independent of filesystem layout."""

    payload = {
        "action_dimension": dataset.action_dimension,
        "candidate_groups": [
            {"group_id": group.group_id, **_group_payload(group)}
            for group in dataset.candidate_groups
        ],
        "chunk_horizon": dataset.chunk_horizon,
        "samples": [
            {"sample_id": sample.sample_id, **_sample_payload(sample)}
            for sample in dataset.samples
        ],
        "schema_version": dataset.schema_version,
        "split_assignments": [
            {
                "assignment_id": assignment.assignment_id,
                **_assignment_payload(assignment),
            }
            for assignment in dataset.split_assignments
        ],
        "split_policy_id": dataset.split_policy_id,
        "state_vector_dimension": dataset.state_vector_dimension,
        "state_vector_schema_digest": dataset.state_vector_schema_digest,
        "state_vector_semantic": dataset.state_vector_semantic,
    }
    return _digest(
        payload,
        prefix=DATASET_DIGEST_PREFIX,
        context="ActionVerifierDatasetV1Content",
    )


__all__ = [
    "ASSIGNMENT_ID_PREFIX",
    "DATASET_DIGEST_PREFIX",
    "GROUP_ID_PREFIX",
    "SAMPLE_ID_PREFIX",
    "SPLIT_POLICY_ID_PREFIX",
    "compute_action_verifier_dataset_digest",
    "compute_action_verifier_sample_identifier",
    "compute_candidate_group_identifier",
    "compute_split_assignment_identifier",
    "compute_split_policy_identifier",
]
