"""Deterministic generation of unlabeled single-source action proposals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.base import (
    ActionCorruption,
    CorruptionApplicabilityError,
)
from latentguard.corruptions.layout import ActionLayout
from latentguard.corruptions.models import (
    CorruptedActionProposal,
    CorruptedActionProposalError,
    ResolvedParameterValue,
    compute_proposal_identifier,
)
from latentguard.models import Episode
from latentguard.validation import validate_episodes

_MAX_SEED = 2**64
_DERIVED_SEED_MASK = 2**63 - 1


class CorruptionGenerationError(ValueError):
    """Raised when deterministic proposal generation cannot continue safely."""


@dataclass(frozen=True, slots=True)
class ApplicabilitySkip:
    """Structured record of one explicitly skipped corruption application."""

    corruption_name: str
    source_episode_id: str
    source_candidate_id: str
    reason: str
    action_shape: tuple[int, ...]
    layout_action_dim: int
    configuration_ordinal: int

    def __str__(self) -> str:
        """Return a concise audit representation."""
        return (
            f"corruption={self.corruption_name} "
            f"episode={self.source_episode_id} "
            f"candidate={self.source_candidate_id} "
            f"config_ordinal={self.configuration_ordinal} "
            f"action_shape={self.action_shape} "
            f"layout_action_dim={self.layout_action_dim} "
            f"reason={self.reason}"
        )


class StrictApplicabilityError(CorruptionGenerationError):
    """Raised when strict generation encounters a non-applicable corruption."""

    def __init__(self, skip: ApplicabilitySkip) -> None:
        """Retain the structured skip that caused strict generation to fail."""
        self.skip = skip
        super().__init__(f"strict applicability failure: {skip}")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Ordered proposals, explicit skips, and deterministic generation counts."""

    proposals: tuple[CorruptedActionProposal, ...]
    skips: tuple[ApplicabilitySkip, ...]
    source_episode_count: int
    source_candidate_count: int
    attempted_transformation_count: int
    corruption_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze collections and validate internally consistent counts."""
        proposals = tuple(self.proposals)
        skips = tuple(self.skips)
        counts = MappingProxyType(dict(self.corruption_counts))
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "skips", skips)
        object.__setattr__(self, "corruption_counts", counts)
        if self.source_episode_count < 0 or self.source_candidate_count < 0:
            raise CorruptionGenerationError("source counts must be non-negative")
        if self.attempted_transformation_count != len(proposals) + len(skips):
            raise CorruptionGenerationError(
                "attempted transformation count must equal proposals plus skips"
            )
        if sum(counts.values()) != len(proposals):
            raise CorruptionGenerationError(
                "corruption counts must sum to the proposal count"
            )
        if any(count < 0 for count in counts.values()):
            raise CorruptionGenerationError("corruption counts must be non-negative")


def derive_corruption_seed(
    *,
    base_seed: int,
    source_episode_id: str,
    source_candidate_id: str,
    corruption_name: str,
    configuration_ordinal: int,
) -> int:
    """Derive a stable 63-bit seed from canonical generation inputs."""
    _validate_seed(base_seed, "base_seed")
    _validate_ordinal(configuration_ordinal, "configuration_ordinal")
    payload = {
        "base_seed": base_seed,
        "configuration_ordinal": configuration_ordinal,
        "corruption_name": corruption_name,
        "source_candidate_id": source_candidate_id,
        "source_episode_id": source_episode_id,
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big") & _DERIVED_SEED_MASK


def build_proposal_identifier(
    *,
    source_episode_id: str,
    source_candidate_id: str,
    corruption_name: str,
    resolved_parameters: Mapping[str, ResolvedParameterValue],
    seed: int,
    generation_ordinal: int,
) -> str:
    """Build a path-independent SHA-256 proposal identifier."""
    _validate_seed(seed, "seed")
    _validate_ordinal(generation_ordinal, "generation_ordinal")
    try:
        return compute_proposal_identifier(
            source_episode_id=source_episode_id,
            source_candidate_id=source_candidate_id,
            corruption_name=corruption_name,
            resolved_parameters=resolved_parameters,
            seed=seed,
            generation_ordinal=generation_ordinal,
        )
    except CorruptedActionProposalError as exc:
        raise CorruptionGenerationError(str(exc)) from exc


def generate_episode_proposals(
    episode: Episode,
    layout: ActionLayout,
    corruptions: Sequence[ActionCorruption],
    *,
    base_seed: int,
    candidate_limit: int | None = None,
    proposal_limit: int | None = None,
    strict_applicability: bool = False,
) -> GenerationResult:
    """Generate ordered proposals from one validated M0 episode."""
    return generate_corruption_proposals(
        (episode,),
        layout,
        corruptions,
        base_seed=base_seed,
        candidate_limit=candidate_limit,
        proposal_limit=proposal_limit,
        strict_applicability=strict_applicability,
    )


def generate_corruption_proposals(
    episodes: Sequence[Episode],
    layout: ActionLayout,
    corruptions: Sequence[ActionCorruption],
    *,
    base_seed: int,
    candidate_limit: int | None = None,
    proposal_limit: int | None = None,
    strict_applicability: bool = False,
) -> GenerationResult:
    """Apply ordered configurations to ordered source candidates deterministically."""
    source_episodes = tuple(episodes)
    definitions = tuple(corruptions)
    validate_episodes(source_episodes)
    _validate_seed(base_seed, "base_seed")
    _validate_optional_limit(candidate_limit, "candidate_limit")
    _validate_optional_limit(proposal_limit, "proposal_limit")
    if type(strict_applicability) is not bool:
        raise CorruptionGenerationError("strict_applicability must be a boolean")
    if not definitions:
        raise CorruptionGenerationError("corruptions must not be empty")
    for index, corruption in enumerate(definitions):
        if not isinstance(corruption, ActionCorruption):
            raise CorruptionGenerationError(
                f"corruptions[{index}] does not implement ActionCorruption"
            )

    proposals: list[CorruptedActionProposal] = []
    skips: list[ApplicabilitySkip] = []
    selected_candidates = 0
    attempted = 0
    stop = False

    for episode in source_episodes:
        if stop:
            break
        for candidate in episode.candidates:
            if candidate_limit is not None and selected_candidates >= candidate_limit:
                stop = True
                break
            if proposal_limit is not None and len(proposals) >= proposal_limit:
                stop = True
                break
            selected_candidates += 1
            source_snapshot = candidate.action.actions.tobytes(order="C")
            for configuration_ordinal, corruption in enumerate(definitions):
                if proposal_limit is not None and len(proposals) >= proposal_limit:
                    stop = True
                    break
                seed = derive_corruption_seed(
                    base_seed=base_seed,
                    source_episode_id=episode.episode_id,
                    source_candidate_id=candidate.candidate_id,
                    corruption_name=corruption.name,
                    configuration_ordinal=configuration_ordinal,
                )
                attempted += 1
                try:
                    resolved = corruption.resolved_parameters_for(
                        candidate.action, layout
                    )
                    transformed = corruption.apply(candidate.action, layout, seed=seed)
                except CorruptionApplicabilityError as error:
                    skip = ApplicabilitySkip(
                        corruption_name=corruption.name,
                        source_episode_id=episode.episode_id,
                        source_candidate_id=candidate.candidate_id,
                        reason=error.reason,
                        action_shape=tuple(candidate.action.actions.shape),
                        layout_action_dim=layout.action_dim,
                        configuration_ordinal=configuration_ordinal,
                    )
                    if strict_applicability:
                        raise StrictApplicabilityError(skip) from error
                    skips.append(skip)
                    continue

                _validate_transformed_action(
                    candidate.action.actions,
                    transformed.actions,
                    source_snapshot=source_snapshot,
                    corruption_name=corruption.name,
                    resolved_parameters=resolved,
                )
                generation_ordinal = len(proposals)
                proposal_id = build_proposal_identifier(
                    source_episode_id=episode.episode_id,
                    source_candidate_id=candidate.candidate_id,
                    corruption_name=corruption.name,
                    resolved_parameters=resolved,
                    seed=seed,
                    generation_ordinal=generation_ordinal,
                )
                proposals.append(
                    CorruptedActionProposal(
                        proposal_id=proposal_id,
                        source_episode_id=episode.episode_id,
                        source_candidate_id=candidate.candidate_id,
                        source_policy_id=episode.source_policy_id,
                        source_task_id=episode.task_id,
                        split_group_id=episode.split_group_id,
                        transformed_action=transformed,
                        corruption_type=corruption.name,
                        resolved_parameters=resolved,
                        seed=seed,
                        generation_ordinal=generation_ordinal,
                        notes="unlabeled M1 action corruption proposal",
                    )
                )

            if candidate.action.actions.tobytes(order="C") != source_snapshot:
                raise CorruptionGenerationError(
                    f"{candidate.candidate_id}: source action changed during generation"
                )

    counts = Counter(proposal.corruption_type for proposal in proposals)
    return GenerationResult(
        proposals=tuple(proposals),
        skips=tuple(skips),
        source_episode_count=len(source_episodes),
        source_candidate_count=selected_candidates,
        attempted_transformation_count=attempted,
        corruption_counts=dict(sorted(counts.items())),
    )


def _validate_transformed_action(
    source: NDArray[Any],
    transformed: NDArray[Any],
    *,
    source_snapshot: bytes,
    corruption_name: str,
    resolved_parameters: Mapping[str, ResolvedParameterValue],
) -> None:
    if source.tobytes(order="C") != source_snapshot:
        raise CorruptionGenerationError(
            f"{corruption_name}: transformed the source action in place"
        )
    if transformed.shape != source.shape:
        raise CorruptionGenerationError(
            f"{corruption_name}: changed action shape from {source.shape} "
            f"to {transformed.shape}"
        )
    if transformed.dtype != source.dtype:
        raise CorruptionGenerationError(
            f"{corruption_name}: changed action dtype from {source.dtype} "
            f"to {transformed.dtype}"
        )
    if np.shares_memory(source, transformed):
        raise CorruptionGenerationError(
            f"{corruption_name}: transformed action still shares source memory"
        )
    _validate_declared_window_immutability(
        source,
        transformed,
        corruption_name=corruption_name,
        resolved_parameters=resolved_parameters,
    )


def _validate_declared_window_immutability(
    source: NDArray[Any],
    transformed: NDArray[Any],
    *,
    corruption_name: str,
    resolved_parameters: Mapping[str, ResolvedParameterValue],
) -> None:
    window_fields = {"window_start", "window_end", "severity_id"}
    present = window_fields.intersection(resolved_parameters)
    if not present:
        return
    if present != window_fields:
        missing = ", ".join(sorted(window_fields - present))
        raise CorruptionGenerationError(
            f"{corruption_name}: resolved window contract is incomplete; missing "
            f"{missing}"
        )
    start = resolved_parameters["window_start"]
    end = resolved_parameters["window_end"]
    severity_id = resolved_parameters["severity_id"]
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end > source.shape[0]
    ):
        raise CorruptionGenerationError(
            f"{corruption_name}: resolved action window is outside the source horizon"
        )
    if (
        not isinstance(severity_id, str)
        or not severity_id
        or severity_id != severity_id.strip()
        or any(
            ord(character) < 32 or ord(character) == 127 for character in severity_id
        )
    ):
        raise CorruptionGenerationError(
            f"{corruption_name}: resolved severity_id is not canonical"
        )
    if source[:start].tobytes(order="C") != transformed[:start].tobytes(
        order="C"
    ) or source[end:].tobytes(order="C") != transformed[end:].tobytes(order="C"):
        raise CorruptionGenerationError(
            f"{corruption_name}: transformed action changed bytes outside declared "
            f"window [{start}, {end})"
        )


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CorruptionGenerationError(
            f"identifier inputs are not canonical JSON values: {exc}"
        ) from exc
    return serialized.encode("utf-8")


def _validate_seed(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value < _MAX_SEED:
        raise CorruptionGenerationError(
            f"{name} must be an integer in [0, {_MAX_SEED})"
        )


def _validate_ordinal(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise CorruptionGenerationError(f"{name} must be a non-negative integer")


def _validate_optional_limit(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise CorruptionGenerationError(f"{name} must be a positive integer or null")
