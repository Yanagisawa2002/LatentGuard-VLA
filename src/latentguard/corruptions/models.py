"""Mutation-safe models for unlabeled, single-source corruption proposals."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from latentguard.models import ActionChunk, JsonScalar
from latentguard.validation import DataValidationError, validate_action_chunk

CORRUPTION_SCHEMA_VERSION = "1.0"
"""Schema version used by M1 corruption proposals and datasets."""

PROPOSAL_ID_PREFIX = "cap-sha256-"
"""Stable prefix for deterministic corruption action proposal identifiers."""

SUPPORTED_CORRUPTION_SCHEMA_VERSIONS = frozenset({CORRUPTION_SCHEMA_VERSION})
"""Corruption schema versions supported by this release."""

ResolvedParameterScalar: TypeAlias = JsonScalar
ResolvedParameterValue: TypeAlias = (
    ResolvedParameterScalar | tuple[ResolvedParameterScalar, ...]
)
ResolvedParameters: TypeAlias = Mapping[str, ResolvedParameterValue]


class CorruptedActionProposalError(ValueError):
    """Raised when an unlabeled corruption proposal is malformed."""


def _require_text(value: object, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise CorruptedActionProposalError(f"CorruptedActionProposal.{field}: required")


def _validate_scalar(value: object, context: str) -> ResolvedParameterScalar:
    if value is not None and type(value) not in (str, int, float, bool):
        raise CorruptedActionProposalError(
            f"{context}: expected a JSON scalar, got {type(value).__name__}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise CorruptedActionProposalError(f"{context}: must be finite")
    return value  # type: ignore[return-value]


def _freeze_parameters(value: ResolvedParameters) -> ResolvedParameters:
    if not isinstance(value, Mapping):
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.resolved_parameters: must be a mapping"
        )
    frozen: dict[str, ResolvedParameterValue] = {}
    for key, item in value.items():
        _require_text(key, "resolved_parameters.key")
        if isinstance(item, tuple):
            frozen[key] = tuple(
                _validate_scalar(element, f"resolved_parameters.{key}")
                for element in item
            )
        else:
            frozen[key] = _validate_scalar(item, f"resolved_parameters.{key}")
    return MappingProxyType(frozen)


def compute_proposal_identifier(
    *,
    source_episode_id: str,
    source_candidate_id: str,
    corruption_name: str,
    resolved_parameters: ResolvedParameters,
    seed: int,
    generation_ordinal: int,
) -> str:
    """Compute the canonical SHA-256 identifier required by the M1 contract."""
    for field, value in (
        ("source_episode_id", source_episode_id),
        ("source_candidate_id", source_candidate_id),
        ("corruption_name", corruption_name),
    ):
        _require_text(value, field)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.seed: must be an integer in [0, 2**64)"
        )
    if type(generation_ordinal) is not int or generation_ordinal < 0:
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.generation_ordinal: must be a non-negative integer"
        )
    parameters = _freeze_parameters(resolved_parameters)
    payload = {
        "corruption_name": corruption_name,
        "generation_ordinal": generation_ordinal,
        "resolved_parameters": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in parameters.items()
        },
        "seed": seed,
        "source_candidate_id": source_candidate_id,
        "source_episode_id": source_episode_id,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (
        f"{PROPOSAL_ID_PREFIX}{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"
    )


@dataclass(frozen=True, slots=True, eq=False)
class CorruptedActionProposal:
    """One unlabeled transformed action with complete single-source provenance."""

    proposal_id: str
    source_episode_id: str
    source_candidate_id: str
    source_policy_id: str
    source_task_id: str
    split_group_id: str
    transformed_action: ActionChunk
    corruption_type: str
    resolved_parameters: ResolvedParameters
    seed: int
    generation_ordinal: int
    schema_version: str = CORRUPTION_SCHEMA_VERSION
    notes: str | None = None

    def __post_init__(self) -> None:
        """Detach the transformed array and freeze resolved parameters."""
        if isinstance(self.transformed_action, ActionChunk):
            action = self.transformed_action
            object.__setattr__(
                self,
                "transformed_action",
                ActionChunk(
                    actions=action.actions,
                    coordinate_frame=action.coordinate_frame,
                    control_period_s=action.control_period_s,
                    schema_version=action.schema_version,
                ),
            )
        object.__setattr__(
            self, "resolved_parameters", _freeze_parameters(self.resolved_parameters)
        )
        validate_corrupted_action_proposal(self)


def validate_corrupted_action_proposal(proposal: CorruptedActionProposal) -> None:
    """Validate an unlabeled proposal and its complete provenance fields."""
    for field in (
        "proposal_id",
        "source_episode_id",
        "source_candidate_id",
        "source_policy_id",
        "source_task_id",
        "split_group_id",
        "corruption_type",
    ):
        _require_text(getattr(proposal, field), field)
    _require_text(proposal.notes, "notes", optional=True)
    if proposal.schema_version not in SUPPORTED_CORRUPTION_SCHEMA_VERSIONS:
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.schema_version: unsupported version "
            f"{proposal.schema_version!r}; supported: {CORRUPTION_SCHEMA_VERSION}"
        )
    if type(proposal.seed) is not int or not 0 <= proposal.seed < 2**64:
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.seed: must be an integer in [0, 2**64)"
        )
    if type(proposal.generation_ordinal) is not int or proposal.generation_ordinal < 0:
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.generation_ordinal: must be a non-negative integer"
        )
    expected_identifier = compute_proposal_identifier(
        source_episode_id=proposal.source_episode_id,
        source_candidate_id=proposal.source_candidate_id,
        corruption_name=proposal.corruption_type,
        resolved_parameters=proposal.resolved_parameters,
        seed=proposal.seed,
        generation_ordinal=proposal.generation_ordinal,
    )
    if proposal.proposal_id != expected_identifier:
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.proposal_id: deterministic identifier mismatch; "
            f"expected {expected_identifier!r}, got {proposal.proposal_id!r}"
        )
    if not isinstance(proposal.transformed_action, ActionChunk):
        raise CorruptedActionProposalError(
            "CorruptedActionProposal.transformed_action: expected ActionChunk"
        )
    try:
        validate_action_chunk(proposal.transformed_action, proposal.proposal_id)
    except DataValidationError as exc:
        raise CorruptedActionProposalError(
            f"CorruptedActionProposal.transformed_action: {exc}"
        ) from exc
    _freeze_parameters(proposal.resolved_parameters)
