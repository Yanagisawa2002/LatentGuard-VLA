"""Content-bound resolution of M1 proposals to immutable M0 source actions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.models import (
    CorruptedActionProposal,
    validate_corrupted_action_proposal,
)
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    load_corruption_dataset,
    validate_corruption_dataset,
)
from latentguard.evaluation.serialization import (
    compute_corruption_dataset_content_digest,
)
from latentguard.models import (
    ActionChunk,
    CameraFrame,
    CandidateAction,
    Episode,
    FailureEvent,
    JsonScalar,
    ObservationFrame,
    ObservationHistory,
    OutcomeLabel,
    SampleProvenance,
)
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    load_episodes,
)
from latentguard.validation import validate_episodes

SOURCE_CONTENT_DIGEST_VERSION = 1
"""Canonical logical M0 content-digest version used by replay binding."""


class ReplaySourceBindingError(ValueError):
    """Raised when M0 and M1 data cannot form one exact replay binding."""


class ReplaySourceMutationError(ReplaySourceBindingError):
    """Raised when bound source or corruption content changes after binding."""


def _json_scalar(value: object, context: str) -> JsonScalar:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, int, bool):
        return cast(JsonScalar, value)
    if type(value) is float and math.isfinite(value):
        return value
    raise ReplaySourceBindingError(
        f"{context}: expected a finite canonical JSON scalar"
    )


def _array_descriptor(array: NDArray[Any]) -> dict[str, object]:
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ReplaySourceBindingError(
            "EpisodeContent.ndarray: expected a non-object NumPy array"
        )
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _action_payload(action: ActionChunk) -> dict[str, object]:
    return {
        "actions": _array_descriptor(action.actions),
        "coordinate_frame": action.coordinate_frame,
        "control_period_s": _json_scalar(
            action.control_period_s, "EpisodeContent.ActionChunk.control_period_s"
        ),
        "schema_version": action.schema_version,
    }


def _failure_payload(failure: FailureEvent) -> dict[str, object]:
    return {
        "failure_type": failure.failure_type,
        "timestamp_s": _json_scalar(
            failure.timestamp_s, "EpisodeContent.FailureEvent.timestamp_s"
        ),
        "probability": _json_scalar(
            failure.probability, "EpisodeContent.FailureEvent.probability"
        ),
        "description": failure.description,
        "schema_version": failure.schema_version,
    }


def _outcome_payload(outcome: OutcomeLabel) -> dict[str, object]:
    return {
        "success": outcome.success,
        "progress": _json_scalar(
            outcome.progress, "EpisodeContent.OutcomeLabel.progress"
        ),
        "unsafe": outcome.unsafe,
        "label_source": outcome.label_source.value,
        "label_strength": outcome.label_strength.value,
        "simulator_replay_verified": outcome.simulator_replay_verified,
        "success_probability": _json_scalar(
            outcome.success_probability,
            "EpisodeContent.OutcomeLabel.success_probability",
        ),
        "unsafe_probability": _json_scalar(
            outcome.unsafe_probability,
            "EpisodeContent.OutcomeLabel.unsafe_probability",
        ),
        "failure_events": [
            _failure_payload(failure) for failure in outcome.failure_events
        ],
        "schema_version": outcome.schema_version,
    }


def _provenance_payload(provenance: SampleProvenance) -> dict[str, object]:
    return {
        "source_episode_id": provenance.source_episode_id,
        "source_policy_id": provenance.source_policy_id,
        "source_task_id": provenance.source_task_id,
        "transformation_type": provenance.transformation_type,
        "transformation_parameters": {
            key: _json_scalar(
                value,
                f"EpisodeContent.SampleProvenance.transformation_parameters.{key}",
            )
            for key, value in provenance.transformation_parameters.items()
        },
        "seed": _json_scalar(provenance.seed, "EpisodeContent.SampleProvenance.seed"),
        "label_source": provenance.label_source.value,
        "label_strength": provenance.label_strength.value,
        "simulator_replay_verified": provenance.simulator_replay_verified,
        "split_group_id": provenance.split_group_id,
        "schema_version": provenance.schema_version,
    }


def _candidate_payload(candidate: CandidateAction) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "action": _action_payload(candidate.action),
        "outcome": _outcome_payload(candidate.outcome),
        "provenance": _provenance_payload(candidate.provenance),
        "schema_version": candidate.schema_version,
    }


def _camera_payload(camera: CameraFrame) -> dict[str, object]:
    return {
        "camera_id": camera.camera_id,
        "rgb": _array_descriptor(camera.rgb),
        "depth": None if camera.depth is None else _array_descriptor(camera.depth),
        "intrinsics": (
            None if camera.intrinsics is None else _array_descriptor(camera.intrinsics)
        ),
        "extrinsics": (
            None if camera.extrinsics is None else _array_descriptor(camera.extrinsics)
        ),
        "schema_version": camera.schema_version,
    }


def _observation_payload(frame: ObservationFrame) -> dict[str, object]:
    return {
        "observation_id": frame.observation_id,
        "timestamp_s": _json_scalar(
            frame.timestamp_s, "EpisodeContent.ObservationFrame.timestamp_s"
        ),
        "robot_state": _array_descriptor(frame.robot_state),
        "cameras": [_camera_payload(camera) for camera in frame.cameras],
        "schema_version": frame.schema_version,
    }


def _history_payload(history: ObservationHistory) -> dict[str, object]:
    return {
        "history_id": history.history_id,
        "frames": [_observation_payload(frame) for frame in history.frames],
        "schema_version": history.schema_version,
    }


def _episode_payload(episode: Episode) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "source_policy_id": episode.source_policy_id,
        "instruction": episode.instruction,
        "observations": _history_payload(episode.observations),
        "candidates": [
            _candidate_payload(candidate) for candidate in episode.candidates
        ],
        "split_group_id": episode.split_group_id,
        "schema_version": episode.schema_version,
    }


def _canonical_json_bytes(value: object, context: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ReplaySourceBindingError(
            f"{context}: could not encode canonical JSON: {exc}"
        ) from exc


def compute_episode_content_digest(episodes: Sequence[Episode]) -> str:
    """Hash every validated logical M0 field and array without runtime paths."""

    episode_tuple = tuple(episodes)
    validate_episodes(episode_tuple)
    payload = {
        "digest_version": SOURCE_CONTENT_DIGEST_VERSION,
        "format": "latentguard-episode-content",
        "episodes": [_episode_payload(episode) for episode in episode_tuple],
    }
    return (
        "sha256:"
        + hashlib.sha256(_canonical_json_bytes(payload, "EpisodeContent")).hexdigest()
    )


def _proposal_payload(proposal: CorruptedActionProposal) -> dict[str, object]:
    return {
        "proposal_id": proposal.proposal_id,
        "source_episode_id": proposal.source_episode_id,
        "source_candidate_id": proposal.source_candidate_id,
        "source_policy_id": proposal.source_policy_id,
        "source_task_id": proposal.source_task_id,
        "split_group_id": proposal.split_group_id,
        "transformed_action": _action_payload(proposal.transformed_action),
        "corruption_type": proposal.corruption_type,
        "resolved_parameters": {
            key: (
                [_json_scalar(item, f"Proposal.{key}") for item in value]
                if isinstance(value, tuple)
                else _json_scalar(value, f"Proposal.{key}")
            )
            for key, value in proposal.resolved_parameters.items()
        },
        "seed": proposal.seed,
        "generation_ordinal": proposal.generation_ordinal,
        "schema_version": proposal.schema_version,
        "notes": proposal.notes,
    }


def _clone_action(action: ActionChunk) -> ActionChunk:
    return ActionChunk(
        actions=np.array(action.actions, copy=True, order="C", subok=False),
        coordinate_frame=action.coordinate_frame,
        control_period_s=action.control_period_s,
        schema_version=action.schema_version,
    )


@dataclass(frozen=True, slots=True, eq=False)
class BoundActionPair:
    """One detached original/transformed action pair with complete provenance."""

    proposal_id: str
    source_dataset_id: str
    source_dataset_digest: str
    corruption_dataset_digest: str
    source_episode_id: str
    source_candidate_id: str
    source_policy_id: str
    source_task_id: str
    split_group_id: str
    generation_ordinal: int
    original_action: ActionChunk
    transformed_action: ActionChunk

    def __post_init__(self) -> None:
        """Detach both action arrays from the bound M0 and M1 datasets."""

        if isinstance(self.original_action, ActionChunk):
            object.__setattr__(
                self, "original_action", _clone_action(self.original_action)
            )
        if isinstance(self.transformed_action, ActionChunk):
            object.__setattr__(
                self, "transformed_action", _clone_action(self.transformed_action)
            )


class ReplaySourceBinding:
    """Validated immutable lookup joining one M0 and one M1 dataset."""

    __slots__ = (
        "_candidate_index",
        "_corruption_dataset",
        "_corruption_dataset_digest",
        "_corruption_path",
        "_episode_index",
        "_episodes",
        "_proposal_index",
        "_source_bundle_identifier",
        "_source_dataset_digest",
        "_source_path",
    )

    def __init__(
        self,
        episodes: tuple[Episode, ...],
        corruption_dataset: CorruptionDataset,
        *,
        source_dataset_digest: str,
        corruption_dataset_digest: str,
        source_bundle_identifier: str | None,
        source_path: Path | None,
        corruption_path: Path | None,
    ) -> None:
        self._episodes = episodes
        self._corruption_dataset = corruption_dataset
        self._source_dataset_digest = source_dataset_digest
        self._corruption_dataset_digest = corruption_dataset_digest
        self._source_bundle_identifier = source_bundle_identifier
        self._source_path = source_path
        self._corruption_path = corruption_path
        self._episode_index = MappingProxyType(
            {episode.episode_id: episode for episode in episodes}
        )
        self._candidate_index = MappingProxyType(
            {
                (episode.episode_id, candidate.candidate_id): candidate
                for episode in episodes
                for candidate in episode.candidates
            }
        )
        self._proposal_index = MappingProxyType(
            {
                proposal.proposal_id: proposal
                for proposal in corruption_dataset.proposals
            }
        )

    @classmethod
    def from_paths(cls, source_dir: Path, corruption_dir: Path) -> ReplaySourceBinding:
        """Safely load, independently digest, and bind M0 and M1 bundles."""

        source_path = Path(source_dir).absolute()
        corruption_path = Path(corruption_dir).absolute()
        episodes = load_episodes(source_path)
        source_bundle_identifier = compute_episode_bundle_identifier(source_path)
        corruption_dataset = load_corruption_dataset(corruption_path)
        return cls.from_datasets(
            episodes,
            corruption_dataset,
            source_bundle_identifier=source_bundle_identifier,
            source_path=source_path,
            corruption_path=corruption_path,
        )

    @classmethod
    def from_datasets(
        cls,
        episodes: Sequence[Episode],
        corruption_dataset: CorruptionDataset,
        *,
        source_dataset_digest: str | None = None,
        corruption_dataset_digest: str | None = None,
        source_bundle_identifier: str | None = None,
        source_path: Path | None = None,
        corruption_path: Path | None = None,
    ) -> ReplaySourceBinding:
        """Bind loaded models while independently recomputing logical digests."""

        episode_tuple = tuple(episodes)
        validate_episodes(episode_tuple)
        validate_corruption_dataset(corruption_dataset)
        computed_source_digest = compute_episode_content_digest(episode_tuple)
        computed_corruption_digest = compute_corruption_dataset_content_digest(
            corruption_dataset
        )
        if (
            source_dataset_digest is not None
            and source_dataset_digest != computed_source_digest
        ):
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.source_dataset_digest: supplied digest does "
                "not match complete M0 content"
            )
        if (
            corruption_dataset_digest is not None
            and corruption_dataset_digest != computed_corruption_digest
        ):
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.corruption_dataset_digest: supplied digest does "
                "not match complete M1 content"
            )
        if (
            source_bundle_identifier is not None
            and source_bundle_identifier != corruption_dataset.source_dataset_id
        ):
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.source_dataset_id: M1 source identifier does "
                "not match the independently hashed M0 bundle"
            )
        binding = cls(
            episode_tuple,
            corruption_dataset,
            source_dataset_digest=computed_source_digest,
            corruption_dataset_digest=computed_corruption_digest,
            source_bundle_identifier=source_bundle_identifier,
            source_path=source_path,
            corruption_path=corruption_path,
        )
        for proposal in corruption_dataset.proposals:
            binding._resolve_validated(proposal)
        return binding

    @property
    def episodes(self) -> tuple[Episode, ...]:
        """Return the immutable ordered source episode collection."""

        return self._episodes

    @property
    def corruption_dataset(self) -> CorruptionDataset:
        """Return the immutable bound corruption dataset."""

        return self._corruption_dataset

    @property
    def source_dataset_id(self) -> str:
        """Return the M1 foreign-key identity for the source M0 bundle."""

        return self._corruption_dataset.source_dataset_id

    @property
    def source_dataset_digest(self) -> str:
        """Return the path-independent digest of every logical M0 value."""

        return self._source_dataset_digest

    @property
    def source_bundle_identifier(self) -> str | None:
        """Return the independently verified historical M1 file identity, if known."""

        return self._source_bundle_identifier

    @property
    def corruption_dataset_digest(self) -> str:
        """Return the complete logical M1 dataset digest."""

        return self._corruption_dataset_digest

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        """Return bound proposal identifiers in M1 generation order."""

        return tuple(
            proposal.proposal_id for proposal in self._corruption_dataset.proposals
        )

    @property
    def source_episode_count(self) -> int:
        """Return the number of bound M0 episodes."""

        return len(self._episodes)

    @property
    def source_candidate_count(self) -> int:
        """Return the number of bound M0 candidate actions."""

        return len(self._candidate_index)

    @property
    def proposal_count(self) -> int:
        """Return the number of bound M1 proposals."""

        return len(self._proposal_index)

    def _assert_models_unchanged(self) -> None:
        try:
            current_source = compute_episode_content_digest(self._episodes)
            current_corruption = compute_corruption_dataset_content_digest(
                self._corruption_dataset
            )
        except (TypeError, ValueError) as exc:
            raise ReplaySourceMutationError(
                f"ReplaySourceBinding: bound model became invalid: {exc}"
            ) from exc
        if current_source != self._source_dataset_digest:
            raise ReplaySourceMutationError(
                "ReplaySourceBinding: source M0 content changed after binding"
            )
        if current_corruption != self._corruption_dataset_digest:
            raise ReplaySourceMutationError(
                "ReplaySourceBinding: corruption M1 content changed after binding"
            )

    def assert_unchanged(self) -> None:
        """Revalidate in-memory and, when available, on-disk bound contents."""

        self._assert_models_unchanged()
        if self._source_path is not None:
            try:
                current_episodes = load_episodes(self._source_path)
                current_content_digest = compute_episode_content_digest(
                    current_episodes
                )
                current_identifier = compute_episode_bundle_identifier(
                    self._source_path
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ReplaySourceMutationError(
                    "ReplaySourceBinding: serialized M0 bundle is no longer "
                    f"loadable after binding: {exc}"
                ) from exc
            if current_content_digest != self._source_dataset_digest:
                raise ReplaySourceMutationError(
                    "ReplaySourceBinding: serialized M0 logical content changed "
                    "after binding"
                )
            if current_identifier != self._source_bundle_identifier:
                raise ReplaySourceMutationError(
                    "ReplaySourceBinding: serialized M0 bundle changed after binding"
                )
        if self._corruption_path is not None:
            try:
                current_dataset = load_corruption_dataset(self._corruption_path)
                current_digest = compute_corruption_dataset_content_digest(
                    current_dataset
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ReplaySourceMutationError(
                    "ReplaySourceBinding: serialized M1 bundle is no longer "
                    f"loadable after binding: {exc}"
                ) from exc
            if current_digest != self._corruption_dataset_digest:
                raise ReplaySourceMutationError(
                    "ReplaySourceBinding: serialized M1 bundle changed after binding"
                )

    def _resolve_validated(
        self, proposal: CorruptedActionProposal
    ) -> tuple[Episode, CandidateAction, CorruptedActionProposal]:
        try:
            bound_proposal = self._proposal_index[proposal.proposal_id]
        except (KeyError, TypeError) as exc:
            raise ReplaySourceBindingError(
                f"ReplaySourceBinding.proposal_id: unknown proposal "
                f"{getattr(proposal, 'proposal_id', '<missing>')!r}"
            ) from exc
        if _canonical_json_bytes(
            _proposal_payload(proposal), "ReplaySourceBinding.proposal"
        ) != _canonical_json_bytes(
            _proposal_payload(bound_proposal), "ReplaySourceBinding.bound_proposal"
        ):
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.proposal: supplied proposal content does not "
                "match the bound M1 proposal"
            )
        try:
            episode = self._episode_index[proposal.source_episode_id]
        except KeyError as exc:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.source_episode_id: proposal references a "
                f"missing source episode {proposal.source_episode_id!r}"
            ) from exc
        key = (proposal.source_episode_id, proposal.source_candidate_id)
        try:
            candidate = self._candidate_index[key]
        except KeyError as exc:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.source_candidate_id: proposal references a "
                f"missing source candidate {proposal.source_candidate_id!r} in "
                f"episode {proposal.source_episode_id!r}"
            ) from exc
        mismatches = [
            name
            for name, left, right in (
                ("source_task_id", proposal.source_task_id, episode.task_id),
                (
                    "source_policy_id",
                    proposal.source_policy_id,
                    episode.source_policy_id,
                ),
                ("split_group_id", proposal.split_group_id, episode.split_group_id),
            )
            if left != right
        ]
        if mismatches:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.provenance: proposal/source mismatch in "
                + ", ".join(mismatches)
            )
        original = candidate.action
        transformed = proposal.transformed_action
        if original.actions.shape[0] != transformed.actions.shape[0]:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.horizon: source and transformed horizons "
                "do not match"
            )
        if original.actions.shape[1] != transformed.actions.shape[1]:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.action_dim: source and transformed action "
                "dimensions do not match"
            )
        if original.actions.shape != transformed.actions.shape:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.shape: source and transformed shapes do "
                "not match"
            )
        if original.actions.dtype != transformed.actions.dtype:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.dtype: source and transformed dtypes do "
                "not match"
            )
        if (
            original.actions.shape[1]
            != self._corruption_dataset.action_layout.action_dim
        ):
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.action_dim: source action does not match "
                "the M1 action layout"
            )
        if original.schema_version != transformed.schema_version:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.schema_version: source and transformed "
                "schemas do not match"
            )
        if original.control_period_s != transformed.control_period_s:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.control_period_s: source and transformed "
                "control periods do not match"
            )
        if original.coordinate_frame != transformed.coordinate_frame:
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action.coordinate_frame: source and transformed "
                "coordinate frames do not match"
            )
        if np.shares_memory(original.actions, transformed.actions):
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.action: source and transformed arrays must not "
                "share memory"
            )
        return episode, candidate, bound_proposal

    def resolve(self, proposal: CorruptedActionProposal) -> BoundActionPair:
        """Resolve one exact M1 proposal to a detached validated source action pair."""

        self._validate_resolution_proposal(proposal)
        self.assert_unchanged()
        pair = self._build_bound_pair(proposal)
        self.assert_unchanged()
        return pair

    def resolve_prevalidated(
        self, proposal: CorruptedActionProposal
    ) -> BoundActionPair:
        """Resolve after the caller gates a batch with ``assert_unchanged``.

        This fast path validates the complete supplied proposal and its M0/M1
        relationship, but deliberately does not rehash or reload either complete
        dataset. Callers must invoke :meth:`assert_unchanged` immediately before
        and after the batch containing all such resolutions.
        """

        self._validate_resolution_proposal(proposal)
        return self._build_bound_pair(proposal)

    @staticmethod
    def _validate_resolution_proposal(proposal: object) -> None:
        if not isinstance(proposal, CorruptedActionProposal):
            raise ReplaySourceBindingError(
                "ReplaySourceBinding.proposal: expected CorruptedActionProposal"
            )
        validate_corrupted_action_proposal(proposal)

    def _build_bound_pair(self, proposal: CorruptedActionProposal) -> BoundActionPair:
        episode, candidate, bound_proposal = self._resolve_validated(proposal)
        return BoundActionPair(
            proposal_id=bound_proposal.proposal_id,
            source_dataset_id=self.source_dataset_id,
            source_dataset_digest=self._source_dataset_digest,
            corruption_dataset_digest=self._corruption_dataset_digest,
            source_episode_id=episode.episode_id,
            source_candidate_id=candidate.candidate_id,
            source_policy_id=episode.source_policy_id,
            source_task_id=episode.task_id,
            split_group_id=episode.split_group_id,
            generation_ordinal=bound_proposal.generation_ordinal,
            original_action=candidate.action,
            transformed_action=bound_proposal.transformed_action,
        )


__all__ = [
    "SOURCE_CONTENT_DIGEST_VERSION",
    "BoundActionPair",
    "ReplaySourceBinding",
    "ReplaySourceBindingError",
    "ReplaySourceMutationError",
    "compute_episode_content_digest",
]
