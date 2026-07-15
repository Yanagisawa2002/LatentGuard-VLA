"""Import independently verified PickCube references into the M0 data model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

from latentguard.models import (
    ActionChunk,
    CandidateAction,
    Episode,
    LabelSource,
    LabelStrength,
    ObservationFrame,
    ObservationHistory,
    OutcomeLabel,
    SampleProvenance,
)
from latentguard.replay.identity import canonical_json_bytes
from latentguard.serialization import save_episodes
from latentguard.validation import (
    DataValidationError,
    validate_episode,
    validate_episodes,
)

from .archive import (
    ManiSkillReferenceArchive,
    ManiSkillReferenceEpisode,
    ReferenceArchiveError,
    compute_reference_episode_content_digest,
    validate_reference_archive,
    validate_reference_episode,
)
from .state_tree import STATE_TREE_SEMANTIC

MANISKILL_PICKCUBE_TASK_ID = "maniskill/PickCube-v1"
OFFICIAL_PICKCUBE_SOURCE_POLICY_ID = (
    "maniskill/3.0.1/official-panda-motionplanning/solvePickCube"
)
PICKCUBE_PROGRESS_SEMANTIC = "pickcube_binary_completion_v0"
PICKCUBE_UNSAFE_SEMANTIC = "pickcube_cube_center_below_world_zero_v0"
SOURCE_IMPORT_TRANSFORMATION = "maniskill_official_source_import_v1"


class M0SourceImportError(ValueError):
    """Raised when a runtime reference cannot be represented honestly in M0."""


def build_m0_source_episode(reference: ManiSkillReferenceEpisode) -> Episode:
    """Build one strong M0 source episode after independent baseline success."""
    try:
        validate_reference_episode(reference)
    except ReferenceArchiveError as exc:
        raise M0SourceImportError(f"M0 source import: {exc}") from exc
    before_digest = compute_reference_episode_content_digest(reference)
    _require_accepted_reference(reference)

    split_group_id = _split_group_identifier(reference.source_trajectory_id)
    candidate_id = _source_candidate_identifier(reference)
    joint_names_digest = _joint_names_digest(reference.robot_state_joint_names)
    observation_id = f"{reference.episode_id}-initial-observation"
    history_id = f"{reference.episode_id}-history"
    outcome = OutcomeLabel(
        success=True,
        progress=1.0,
        unsafe=False,
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
    )
    provenance = SampleProvenance(
        source_episode_id=reference.episode_id,
        source_policy_id=OFFICIAL_PICKCUBE_SOURCE_POLICY_ID,
        source_task_id=MANISKILL_PICKCUBE_TASK_ID,
        transformation_type=SOURCE_IMPORT_TRANSFORMATION,
        transformation_parameters={
            "compatibility_identity": reference.compatibility_identity,
            "source_trajectory_id": reference.source_trajectory_id,
            "source_action_digest": reference.source_action_digest,
            "initial_state_digest": reference.initial_state_digest,
            "state_semantic": STATE_TREE_SEMANTIC,
            "robot_state_semantic": reference.robot_state_semantic,
            "robot_state_joint_names_digest": joint_names_digest,
            "progress_semantic": PICKCUBE_PROGRESS_SEMANTIC,
            "unsafe_semantic": PICKCUBE_UNSAFE_SEMANTIC,
            "independent_baseline_success": True,
        },
        seed=reference.seed,
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
        split_group_id=split_group_id,
    )
    episode = Episode(
        episode_id=reference.episode_id,
        task_id=MANISKILL_PICKCUBE_TASK_ID,
        source_policy_id=OFFICIAL_PICKCUBE_SOURCE_POLICY_ID,
        instruction="Pick up the cube and place it at the goal.",
        observations=ObservationHistory(
            history_id=history_id,
            frames=(
                ObservationFrame(
                    observation_id=observation_id,
                    timestamp_s=0.0,
                    robot_state=reference.initial_robot_state,
                    cameras=(),
                ),
            ),
        ),
        candidates=(
            CandidateAction(
                candidate_id=candidate_id,
                action=ActionChunk(
                    actions=reference.source_actions,
                    coordinate_frame=reference.action_coordinate_frame,
                    control_period_s=reference.control_period_s,
                ),
                outcome=outcome,
                provenance=provenance,
            ),
        ),
        split_group_id=split_group_id,
    )
    try:
        validate_episode(episode)
    except DataValidationError as exc:
        raise M0SourceImportError(f"M0 source import: {exc}") from exc
    after_digest = compute_reference_episode_content_digest(reference)
    if after_digest != before_digest:
        raise M0SourceImportError(
            "M0 source import: reference content changed during import"
        )
    return episode


def build_m0_source_episodes(
    archive: ManiSkillReferenceArchive,
) -> tuple[Episode, ...]:
    """Import every accepted archive episode while preserving deterministic order."""
    try:
        validate_reference_archive(archive)
    except ReferenceArchiveError as exc:
        raise M0SourceImportError(f"M0 source import: {exc}") from exc
    before_digest = archive.content_digest
    episodes = tuple(
        build_m0_source_episode(reference) for reference in archive.episodes
    )
    try:
        validate_episodes(episodes)
    except DataValidationError as exc:
        raise M0SourceImportError(f"M0 source import: {exc}") from exc
    if archive.content_digest != before_digest:
        raise M0SourceImportError(
            "M0 source import: archive content changed during import"
        )
    return episodes


def save_m0_source_dataset(
    archive: ManiSkillReferenceArchive, output_dir: Path
) -> Path:
    """Build and transactionally save the finalized M0 source dataset."""
    episodes = build_m0_source_episodes(archive)
    try:
        return save_episodes(episodes, output_dir)
    except (OSError, TypeError, ValueError) as exc:
        raise M0SourceImportError(
            f"M0 source import: could not save source dataset: {exc}"
        ) from exc


def _require_accepted_reference(reference: ManiSkillReferenceEpisode) -> None:
    if not reference.source_generation_success:
        raise M0SourceImportError(
            "M0 source import: source generation did not report success"
        )
    if not reference.independent_baseline_success:
        raise M0SourceImportError(
            "M0 source import: independent baseline replay is required"
        )
    evidence = reference.terminal_task_evidence
    if evidence["success"] is not True:
        raise M0SourceImportError(
            "M0 source import: official terminal success must be true"
        )
    if evidence["is_obj_placed"] is not True:
        raise M0SourceImportError(
            "M0 source import: official object-placement evidence must be true"
        )
    if evidence["is_robot_static"] is not True:
        raise M0SourceImportError(
            "M0 source import: official robot-static evidence must be true"
        )
    cube_center_z = cast(int | float, evidence["cube_center_z"])
    if float(cube_center_z) < 0.0:
        raise M0SourceImportError(
            "M0 source import: cube is below world z=0 under the narrow unsafe semantic"
        )


def _source_candidate_identifier(reference: ManiSkillReferenceEpisode) -> str:
    payload = {
        "episode_id": reference.episode_id,
        "source_action_digest": reference.source_action_digest,
        "source_policy_id": OFFICIAL_PICKCUBE_SOURCE_POLICY_ID,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"mspc-source-candidate-{digest}"


def _split_group_identifier(source_trajectory_id: str) -> str:
    digest = hashlib.sha256(source_trajectory_id.encode("utf-8")).hexdigest()
    return f"mspc-split-{digest}"


def _joint_names_digest(names: tuple[str, ...]) -> str:
    payload = {"ordered_joint_names": list(names)}
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "MANISKILL_PICKCUBE_TASK_ID",
    "OFFICIAL_PICKCUBE_SOURCE_POLICY_ID",
    "PICKCUBE_PROGRESS_SEMANTIC",
    "PICKCUBE_UNSAFE_SEMANTIC",
    "SOURCE_IMPORT_TRANSFORMATION",
    "M0SourceImportError",
    "build_m0_source_episode",
    "build_m0_source_episodes",
    "save_m0_source_dataset",
]
