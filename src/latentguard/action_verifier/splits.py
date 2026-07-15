"""Deterministic source-trajectory splits and hard leakage validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from latentguard.action_verifier.identity import compute_split_policy_identifier
from latentguard.action_verifier.models import (
    DatasetSplit,
    TrajectorySplitAssignmentV1,
)
from latentguard.action_verifier.validation import ActionVerifierValidationError
from latentguard.replay.identity import canonical_json_bytes

FULL_SPLIT_COUNTS: Mapping[DatasetSplit, int] = MappingProxyType(
    {
        DatasetSplit.TRAIN: 48,
        DatasetSplit.VALIDATION: 6,
        DatasetSplit.TEST: 6,
    }
)
"""Required 60-trajectory M3A split quotas."""


class SplitLeakageError(ActionVerifierValidationError):
    """Raised when any source-bound identity appears in multiple splits."""


@dataclass(frozen=True, slots=True)
class TrajectorySplitSourceV1:
    """Leakage-sensitive source inventory before a split is assigned."""

    source_trajectory_id: str
    source_seed: int
    split_group_id: str
    state_digests: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    proposal_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze and validate the complete pre-split inventory."""

        object.__setattr__(self, "state_digests", tuple(self.state_digests))
        object.__setattr__(self, "anchor_ids", tuple(self.anchor_ids))
        object.__setattr__(self, "proposal_ids", tuple(self.proposal_ids))
        for field in ("source_trajectory_id", "split_group_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ActionVerifierValidationError(
                    f"TrajectorySplitSourceV1.{field}: expected canonical text"
                )
        if type(self.source_seed) is not int or not 0 <= self.source_seed < 2**32:
            raise ActionVerifierValidationError(
                "TrajectorySplitSourceV1.source_seed: expected uint32"
            )
        for field in ("state_digests", "anchor_ids", "proposal_ids"):
            values = getattr(self, field)
            if not values:
                raise ActionVerifierValidationError(
                    f"TrajectorySplitSourceV1.{field}: inventory must not be empty"
                )
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise ActionVerifierValidationError(
                    f"TrajectorySplitSourceV1.{field}: must be sorted and unique"
                )


def _normalized_split_counts(
    split_counts: Mapping[DatasetSplit, int] | None,
    trajectory_count: int,
) -> Mapping[DatasetSplit, int]:
    supplied = FULL_SPLIT_COUNTS if split_counts is None else split_counts
    if not isinstance(supplied, Mapping) or any(
        not isinstance(key, DatasetSplit) for key in supplied
    ):
        raise ActionVerifierValidationError(
            "TrajectorySplitPolicy.split_counts: keys must be DatasetSplit values"
        )
    unexpected = set(supplied) - set(DatasetSplit)
    if unexpected:
        raise ActionVerifierValidationError(
            "TrajectorySplitPolicy.split_counts: unsupported split"
        )
    normalized: dict[DatasetSplit, int] = {}
    for split in DatasetSplit:
        count = supplied.get(split, 0)
        if type(count) is not int or count < 0:
            raise ActionVerifierValidationError(
                "TrajectorySplitPolicy.split_counts: counts must be "
                "non-negative integers"
            )
        normalized[split] = count
    if sum(normalized.values()) != trajectory_count:
        raise ActionVerifierValidationError(
            "TrajectorySplitPolicy.split_counts: quotas must exactly cover trajectories"
        )
    return MappingProxyType(normalized)


def _trajectory_order_key(source_trajectory_id: str, split_seed: int) -> str:
    payload = {
        "schema_version": "1.0",
        "semantic": "source_trajectory_hash_order_v1",
        "source_trajectory_id": source_trajectory_id,
        "split_seed": split_seed,
    }
    return hashlib.sha256(
        canonical_json_bytes(payload, context="TrajectorySplitOrderV1")
    ).hexdigest()


def assign_trajectory_splits(
    sources: Sequence[TrajectorySplitSourceV1],
    *,
    split_counts: Mapping[DatasetSplit, int] | None = None,
    split_seed: int = 0,
) -> tuple[TrajectorySplitAssignmentV1, ...]:
    """Assign exact quotas using stable identity hashes, never input order."""

    records = tuple(sources)
    if not records:
        raise ActionVerifierValidationError(
            "TrajectorySplitPolicy.sources: at least one trajectory is required"
        )
    if not all(isinstance(record, TrajectorySplitSourceV1) for record in records):
        raise ActionVerifierValidationError(
            "TrajectorySplitPolicy.sources: expected TrajectorySplitSourceV1 values"
        )
    if type(split_seed) is not int or not 0 <= split_seed < 2**64:
        raise ActionVerifierValidationError(
            "TrajectorySplitPolicy.split_seed: expected an integer in [0, 2**64)"
        )
    trajectory_ids = tuple(record.source_trajectory_id for record in records)
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ActionVerifierValidationError(
            "TrajectorySplitPolicy.sources: duplicate source trajectory IDs"
        )
    counts = _normalized_split_counts(split_counts, len(records))
    policy_id = compute_split_policy_identifier(
        source_trajectory_ids=trajectory_ids,
        split_counts=counts,
        split_seed=split_seed,
    )
    ordered = sorted(
        records,
        key=lambda record: (
            _trajectory_order_key(record.source_trajectory_id, split_seed),
            record.source_trajectory_id,
        ),
    )
    assigned: list[TrajectorySplitAssignmentV1] = []
    offset = 0
    for split in DatasetSplit:
        end = offset + counts[split]
        for record in ordered[offset:end]:
            assigned.append(
                TrajectorySplitAssignmentV1(
                    split_policy_id=policy_id,
                    source_trajectory_id=record.source_trajectory_id,
                    source_seed=record.source_seed,
                    split_group_id=record.split_group_id,
                    dataset_split=split,
                    state_digests=record.state_digests,
                    anchor_ids=record.anchor_ids,
                    proposal_ids=record.proposal_ids,
                )
            )
        offset = end
    result = tuple(sorted(assigned, key=lambda item: item.source_trajectory_id))
    validate_no_split_leakage(result)
    return result


def validate_no_split_leakage(
    assignments: Sequence[TrajectorySplitAssignmentV1],
) -> None:
    """Reject six source-bound identity classes whenever they cross splits."""

    values = tuple(assignments)
    if not values:
        raise SplitLeakageError("SplitLeakage.assignments: must not be empty")
    seen: dict[str, dict[object, DatasetSplit]] = {
        "source_trajectory_id": {},
        "source_seed": {},
        "state_digest": {},
        "anchor_id": {},
        "proposal_id": {},
        "split_group_id": {},
    }

    def record(category: str, token: object, split: DatasetSplit) -> None:
        prior = seen[category].get(token)
        if prior is not None and prior is not split:
            raise SplitLeakageError(
                f"SplitLeakage.{category}: {token!r} appears in "
                f"{prior.value!r} and {split.value!r}"
            )
        seen[category][token] = split

    for assignment in values:
        if not isinstance(assignment, TrajectorySplitAssignmentV1):
            raise SplitLeakageError(
                "SplitLeakage.assignments: expected TrajectorySplitAssignmentV1"
            )
        split = assignment.dataset_split
        record("source_trajectory_id", assignment.source_trajectory_id, split)
        record("source_seed", assignment.source_seed, split)
        record("split_group_id", assignment.split_group_id, split)
        for digest in assignment.state_digests:
            record("state_digest", digest, split)
        for anchor_id in assignment.anchor_ids:
            record("anchor_id", anchor_id, split)
        for proposal_id in assignment.proposal_ids:
            record("proposal_id", proposal_id, split)


__all__ = [
    "FULL_SPLIT_COUNTS",
    "SplitLeakageError",
    "TrajectorySplitSourceV1",
    "assign_trajectory_splits",
    "validate_no_split_leakage",
]
