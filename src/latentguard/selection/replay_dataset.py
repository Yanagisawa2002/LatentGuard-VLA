"""Bridge immutable M3C pools to the existing M1/M2 replay dataset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

import numpy as np

from latentguard.corruptions.generation import build_proposal_identifier
from latentguard.corruptions.layout import ActionLayout
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import CorruptionDataset
from latentguard.models import ActionChunk
from latentguard.selection.candidate_pool import (
    CandidatePoolSourceV1,
    compute_source_set_digest,
)
from latentguard.selection.configuration import CandidatePoolConfigurationV1
from latentguard.selection.models import (
    CANDIDATE_COUNT,
    CandidatePoolV1,
    array_content_digest,
)


class SelectionReplayDatasetError(ValueError):
    """Raised when a blind pool cannot reproduce its exact replay proposals."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionReplayDatasetError(f"{context}: {reason}")


def build_replay_corruption_dataset(
    sources: Sequence[CandidatePoolSourceV1],
    candidate_pool: CandidatePoolV1,
    configuration: CandidatePoolConfigurationV1,
    action_layout: ActionLayout,
    *,
    source_dataset_id: str,
) -> CorruptionDataset:
    """Recreate full candidate+continuation actions with stable pool identities."""

    values = tuple(sources)
    if len(values) != len(candidate_pool.groups):
        _fail("source inventory", "count differs from candidate groups")
    if (
        candidate_pool.candidate_pool_configuration_digest
        != configuration.content_digest
    ):
        _fail("configuration", "digest differs from candidate pool")
    if candidate_pool.action_contract_digest != configuration.action_contract_digest:
        _fail("action contract", "digest differs from candidate pool")
    if candidate_pool.source_set_digest != compute_source_set_digest(values):
        _fail("source inventory", "content digest differs from candidate pool")
    if action_layout.action_dim != 8:
        _fail("action layout", "M3C requires action dimension 8")
    source_by_anchor = {item.anchor_id: item for item in values}
    if len(source_by_anchor) != len(values):
        _fail("source inventory", "duplicate anchor")
    proposals: list[CorruptedActionProposal] = []
    for group_index, group in enumerate(candidate_pool.groups):
        source = source_by_anchor.get(group.anchor_id)
        if source is None:
            _fail("candidate group", "source anchor is absent")
        if source.trajectory.content_digest != group.trajectory.content_digest:
            _fail("candidate group", "trajectory identity differs")
        source_prefix = np.ascontiguousarray(
            source.source_remaining_action.actions[:16]
        )
        source_continuation = np.ascontiguousarray(
            source.source_remaining_action.actions[16:]
        )
        if (
            group.state_content_digest != source.state_content_digest
            or group.verifier_state_content_digest
            != source.verifier_state_content_digest
            or group.continuation_identity != source.continuation_identity
            or group.source_action_prefix_digest != array_content_digest(source_prefix)
            or array_content_digest(group.state_vector)
            != array_content_digest(source.state_vector)
            or group.continuation_actions.tobytes(order="C")
            != source_continuation.tobytes(order="C")
        ):
            _fail("candidate group", "source content binding differs")
        for candidate in group.candidates:
            ordinal = candidate.configuration_ordinal
            definition = configuration.candidates[ordinal]
            corruption = definition.build_corruption()
            resolved = corruption.resolved_parameters_for(
                source.source_remaining_action, action_layout
            )
            transformed = corruption.apply(
                source.source_remaining_action,
                action_layout,
                seed=candidate.seed,
            )
            if transformed.actions[:16].tobytes(
                order="C"
            ) != candidate.action_chunk.tobytes(order="C") or transformed.actions[
                16:
            ].tobytes(order="C") != group.continuation_actions.tobytes(order="C"):
                _fail(
                    f"candidate {candidate.proposal_id}",
                    "pool actions do not reproduce from the frozen transformation",
                )
            full_actions = np.concatenate(
                (candidate.action_chunk, group.continuation_actions), axis=0
            )
            action = ActionChunk(
                actions=np.ascontiguousarray(full_actions),
                coordinate_frame=source.source_remaining_action.coordinate_frame,
                control_period_s=source.source_remaining_action.control_period_s,
                schema_version=source.source_remaining_action.schema_version,
            )
            generation_ordinal = group_index * CANDIDATE_COUNT + ordinal
            expected_id = build_proposal_identifier(
                source_episode_id=source.source_episode_id,
                source_candidate_id=source.source_candidate_id,
                corruption_name=corruption.name,
                resolved_parameters=resolved,
                seed=candidate.seed,
                generation_ordinal=generation_ordinal,
            )
            if expected_id != candidate.proposal_id:
                _fail(
                    f"candidate {candidate.proposal_id}",
                    "proposal identity differs from the replay ordinal",
                )
            proposals.append(
                CorruptedActionProposal(
                    proposal_id=candidate.proposal_id,
                    source_episode_id=source.source_episode_id,
                    source_candidate_id=source.source_candidate_id,
                    source_policy_id=source.source_policy_id,
                    source_task_id=source.source_task_id,
                    split_group_id=source.trajectory.split_group_id,
                    transformed_action=action,
                    corruption_type=corruption.name,
                    resolved_parameters=resolved,
                    seed=candidate.seed,
                    generation_ordinal=generation_ordinal,
                    notes="unlabeled M3C blind candidate proposal",
                )
            )
    result = CorruptionDataset(
        source_dataset_id=source_dataset_id,
        action_layout=action_layout,
        proposals=tuple(proposals),
    )
    if tuple(item.proposal_id for item in result.proposals) != (
        candidate_pool.proposal_ids
    ):
        _fail("replay dataset", "proposal inventory differs from candidate pool")
    return result


__all__ = [
    "SelectionReplayDatasetError",
    "build_replay_corruption_dataset",
]
