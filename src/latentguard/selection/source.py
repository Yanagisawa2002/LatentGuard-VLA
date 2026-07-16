"""Strict M3A exclusion and M3C source projection for candidate pools."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from latentguard.action_verifier import load_action_verifier_dataset
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    PickCubeStateIndexedArchiveV1,
    load_state_indexed_archive,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    PickCubeAnchorManifestV1,
    load_anchor_manifest,
)
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.candidate_pool import CandidatePoolSourceV1
from latentguard.selection.models import (
    SourceExclusionInventoryV1,
    SourceTrajectoryIdentityV1,
)
from latentguard.serialization import load_episodes
from latentguard.training.dataset import (
    ACCEPTED_M3A_DATASET_DIGEST,
    load_accepted_action_verifier_dataset,
)
from latentguard.validation import validate_episodes


class SelectionSourceError(ValueError):
    """Raised when source or exclusion content is incomplete or overlapping."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionSourceError(f"{context}: {reason}")


def _remaining_action_digest(actions: NDArray[Any]) -> str:
    payload = {
        "action_byte_digest": (
            f"sha256:{hashlib.sha256(actions.tobytes(order='C')).hexdigest()}"
        ),
        "action_dtype": actions.dtype.str,
        "action_shape": list(actions.shape),
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def build_m3a_exclusion_inventory(
    dataset_directory: Path,
    anchor_manifest_directory: Path,
    *,
    acceptance_report: Path | None = None,
) -> SourceExclusionInventoryV1:
    """Bind every accepted M3A seed, trajectory, split group, and state digest."""

    anchor_manifest = load_anchor_manifest(anchor_manifest_directory)
    dataset = load_action_verifier_dataset(dataset_directory)
    if dataset.content_digest != ACCEPTED_M3A_DATASET_DIGEST:
        _fail("M3A dataset", "content digest differs from the accepted dataset")
    if acceptance_report is not None:
        reasons = {
            record.anchor.anchor_id: record.anchor.selection_reason
            for record in anchor_manifest.records
        }
        accepted = load_accepted_action_verifier_dataset(
            dataset_directory,
            full_target_report=acceptance_report,
            anchor_selection_reasons=reasons,
            anchor_manifest_content_digest=anchor_manifest.content_digest,
        )
        if accepted.dataset_digest != dataset.content_digest:
            _fail("M3A dataset", "accepted projection identity differs")
    assignments = tuple(dataset.split_assignments)
    return SourceExclusionInventoryV1(
        dataset_digest=dataset.content_digest,
        anchor_manifest_digest=anchor_manifest.content_digest,
        reset_seeds=tuple(sorted({item.source_seed for item in assignments})),
        source_trajectory_ids=tuple(
            sorted({item.source_trajectory_id for item in assignments})
        ),
        split_group_ids=tuple(sorted({item.split_group_id for item in assignments})),
        complete_state_digests=tuple(
            sorted(
                {
                    digest
                    for assignment in assignments
                    for digest in assignment.state_digests
                }
            )
        ),
    )


def _validate_fixed_scope(
    archive: PickCubeStateIndexedArchiveV1,
    manifest: PickCubeAnchorManifestV1,
) -> None:
    for episode in archive.episodes:
        expected = manifest.trajectory_state_digests.get(episode.source_trajectory_id)
        observed = tuple(state.state_digest for state in episode.states)
        if expected is None or tuple(expected) != observed:
            _fail("source archive", "complete trajectory state inventory differs")
        for state in episode.states:
            if state.numeric_component_count != 70:
                _fail("source archive", "M3C requires exactly 70 state components")
            if state.verifier_state.values.dtype != np.dtype(
                "<f4"
            ) or state.verifier_state.values.shape != (38,):
                _fail("source archive", "M3C requires exact 38-component state")
    evidence_by_id = {item.evidence_id: item for item in manifest.baseline_evidence}
    for record in manifest.records:
        evidence = evidence_by_id[record.baseline_evidence_id]
        if (
            not evidence.state_restoration_verified
            or not evidence.complete_state_comparison
            or evidence.compared_component_count != 70
            or evidence.maximum_absolute_error > 1e-6
            or not evidence.verifier_state_restoration_verified
            or evidence.verifier_state_compared_component_count != 38
            or evidence.verifier_state_maximum_absolute_error > 1e-6
            or not evidence.complete_action_execution
            or evidence.executed_action_count != evidence.expected_action_count
            or not evidence.official_terminal_success
            or not evidence.simulator_replay_verified
        ):
            _fail(
                f"anchor {record.anchor.anchor_id}",
                "baseline does not satisfy the fixed 70/38 strong replay gate",
            )


def load_candidate_pool_sources(
    source_directory: Path,
    runtime_archive_directory: Path,
    anchor_manifest_directory: Path,
    *,
    trajectory_limit: int | None = None,
) -> tuple[CandidatePoolSourceV1, ...]:
    """Project outcome-free M3C source anchors after strict baseline validation."""

    if trajectory_limit is not None and (
        type(trajectory_limit) is not int or trajectory_limit <= 0
    ):
        _fail("trajectory_limit", "expected a positive integer")
    archive = load_state_indexed_archive(runtime_archive_directory)
    manifest = load_anchor_manifest(anchor_manifest_directory)
    if manifest.source_archive_content_digest != archive.content_digest:
        _fail("source binding", "anchor manifest references another archive")
    _validate_fixed_scope(archive, manifest)
    episodes = load_episodes(source_directory)
    validate_episodes(episodes)
    by_source_episode = {item.episode_id: item for item in episodes}
    by_archive_episode = {item.episode_id: item for item in archive.episodes}
    selected_trajectories = tuple(
        sorted({record.anchor.source_trajectory_id for record in manifest.records})
    )
    if trajectory_limit is not None:
        selected_trajectories = selected_trajectories[:trajectory_limit]
    selected = set(selected_trajectories)
    results: list[CandidatePoolSourceV1] = []
    for record in manifest.records:
        anchor = record.anchor
        if anchor.source_trajectory_id not in selected:
            continue
        source_episode = by_source_episode.get(record.source_episode_id)
        archived_episode = by_archive_episode.get(record.source_archive_episode_id)
        if source_episode is None or archived_episode is None:
            _fail("source binding", "anchor source episode is absent")
        candidates = {item.candidate_id: item for item in source_episode.candidates}
        source_candidate = candidates.get(record.source_candidate_id)
        if source_candidate is None:
            _fail("source binding", "anchor source candidate is absent")
        if (
            archived_episode.source_trajectory_id != anchor.source_trajectory_id
            or archived_episode.seed != anchor.source_seed
        ):
            _fail("source binding", "archive trajectory identity differs")
        state = archived_episode.states[anchor.state_index]
        complete_states = tuple(
            manifest.trajectory_state_digests[anchor.source_trajectory_id]
        )
        if (
            state.state_digest != record.source_state_digest
            or state.content_digest != record.source_state_content_digest
            or state.verifier_state.content_digest
            != record.verifier_state_content_digest
            or state.verifier_state.schema_digest != record.verifier_state_schema_digest
            or _remaining_action_digest(source_candidate.action.actions)
            != record.source_remaining_action_digest
        ):
            _fail("source binding", "anchor content differs from source records")
        trajectory = SourceTrajectoryIdentityV1(
            source_trajectory_id=anchor.source_trajectory_id,
            source_seed=anchor.source_seed,
            split_group_id=anchor.split_group_id,
            complete_state_digests=complete_states,
        )
        results.append(
            CandidatePoolSourceV1(
                anchor_id=anchor.anchor_id,
                source_episode_id=source_episode.episode_id,
                source_candidate_id=source_candidate.candidate_id,
                source_policy_id=source_episode.source_policy_id,
                source_task_id=source_episode.task_id,
                trajectory=trajectory,
                state_tree_digest=state.state_digest,
                state_content_digest=state.content_digest,
                verifier_state_content_digest=state.verifier_state.content_digest,
                continuation_identity=record.continuation_identity,
                state_vector=state.verifier_state.values,
                source_remaining_action=source_candidate.action,
            )
        )
    if not results:
        _fail("candidate sources", "no accepted anchor remained")
    expected_per_trajectory: Mapping[str, int] = {
        trajectory: sum(
            record.anchor.source_trajectory_id == trajectory
            for record in manifest.records
        )
        for trajectory in selected_trajectories
    }
    observed_per_trajectory = {
        trajectory: sum(
            item.trajectory.source_trajectory_id == trajectory for item in results
        )
        for trajectory in selected_trajectories
    }
    if observed_per_trajectory != expected_per_trajectory:
        _fail("candidate sources", "anchor inventory changed during projection")
    return tuple(results)


def require_disjoint_source_sets(
    first: Sequence[CandidatePoolSourceV1],
    second: Sequence[CandidatePoolSourceV1],
) -> None:
    """Reject smoke/full overlap across every trajectory-level identity field."""

    def inventories(
        values: Sequence[CandidatePoolSourceV1],
    ) -> tuple[set[int], set[str], set[str], set[str]]:
        trajectories = {item.trajectory for item in values}
        return (
            {item.source_seed for item in trajectories},
            {item.source_trajectory_id for item in trajectories},
            {item.split_group_id for item in trajectories},
            {digest for item in trajectories for digest in item.complete_state_digests},
        )

    left = inventories(first)
    right = inventories(second)
    if left[0].intersection(right[0]):
        _fail("source-set disjointness", "overlapping reset seeds")
    if left[1].intersection(right[1]):
        _fail("source-set disjointness", "overlapping trajectory IDs")
    if left[2].intersection(right[2]):
        _fail("source-set disjointness", "overlapping split groups")
    if left[3].intersection(right[3]):
        _fail("source-set disjointness", "overlapping state digests")


__all__ = [
    "SelectionSourceError",
    "build_m3a_exclusion_inventory",
    "load_candidate_pool_sources",
    "require_disjoint_source_sets",
]
