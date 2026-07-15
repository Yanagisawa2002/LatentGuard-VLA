"""Build content-bound M3A replay cases from state-indexed PickCube sources."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.models import (
    ActionChunk,
    CandidateAction,
    Episode,
    LabelSource,
    LabelStrength,
)
from latentguard.replay.base import ReplayInvalidContextError
from latentguard.replay.identity import (
    canonical_json_bytes,
    compute_replay_bundle_digest,
    compute_replay_case_identifier,
)
from latentguard.replay.models import (
    ReplayBundle,
    ReplayCase,
    ReplayStateReference,
    ReplayTaskReference,
    StateComparisonSemantic,
)
from latentguard.replay.source import BoundActionPair, ReplaySourceBinding

from .adapter import (
    MANISKILL_PICKCUBE_ADAPTER_ID,
    MANISKILL_PICKCUBE_ADAPTER_VERSION,
    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
    ManiSkillPickCubeAdapter,
    ManiSkillPickCubeSemanticIdentity,
    TrustedManiSkillRuntimeAttestation,
    resolve_maniskill_pickcube_configuration,
)
from .anchors import (
    DEFAULT_CANDIDATE_HORIZON,
    DEFAULT_MAX_ANCHORS_PER_TRAJECTORY,
    PickCubeAnchorStateFacts,
    PickCubeStateAnchor,
    select_pickcube_state_anchors,
)
from .compatibility import CompatibilityBinding
from .serialization import (
    ManiSkillPickCubeArtifactError,
    action_contract_from_compatibility,
    environment_settings_from_compatibility,
)
from .session import (
    PICKCUBE_SOURCE_RESET_SEED_METADATA_KEY,
    CanonicalPickCubeStateTreeComparator,
    LazyManiSkillPickCubeRuntime,
    LoadedReferenceState,
    ManiSkillPickCubeSessionFactory,
    PickCubeReplayActionContract,
)
from .source_import import MANISKILL_PICKCUBE_TASK_ID
from .state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    StateIndexedArchiveError,
    load_state_indexed_archive,
)
from .state_indexed_build import (
    ANCHOR_SOURCE_TRANSFORMATION,
    PickCubeActionControlContractV1,
    SourceContinuationIdentityV1,
    StateIndexedBuildError,
    build_source_continuation_identity,
)
from .state_tree import STATE_TREE_SEMANTIC
from .task_evidence import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_TASK_CONTRACT_VERSION,
    PICKCUBE_TASK_ID,
    PICKCUBE_UNSAFE_SEMANTIC,
    PickCubeTaskKeyContract,
)

STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY = (
    "state_indexed_archive_content_digest"
)
"""State-reference metadata key for the complete T+1 archive digest."""

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONFIGURATION_DIGEST_PATTERN = re.compile(r"^cfg-sha256-[0-9a-f]{64}$")


class ManiSkillPickCubeStateIndexedReplayError(ManiSkillPickCubeArtifactError):
    """Raised when anchor M0/M1 data disagrees with the T+1 source archive."""


@dataclass(frozen=True, slots=True)
class _BoundAnchorSource:
    source_episode: Episode
    source_candidate: CandidateAction
    archive_episode: PickCubeStateIndexedEpisodeV1
    source_state: PickCubeIndexedStateV1
    anchor: PickCubeStateAnchor
    continuation: SourceContinuationIdentityV1
    baseline_evidence_id: str
    source_remaining_action_digest: str


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ManiSkillPickCubeStateIndexedReplayError(
            f"{field} must be a lowercase sha256 content digest"
        )
    return value


def _require_configuration_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or _CONFIGURATION_DIGEST_PATTERN.fullmatch(value) is None
    ):
        raise ManiSkillPickCubeStateIndexedReplayError(
            "adapter configuration digest must use the cfg-sha256 identity format"
        )
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ManiSkillPickCubeStateIndexedReplayError(
            f"{field} must be non-empty canonical text"
        )
    return value


def _require_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ManiSkillPickCubeStateIndexedReplayError(
            f"{field} must be an integer at least {minimum}"
        )
    return value


def _parameter(
    parameters: Mapping[str, object], name: str, *, field: str | None = None
) -> object:
    try:
        return parameters[name]
    except KeyError as exc:
        raise ManiSkillPickCubeStateIndexedReplayError(
            f"anchor M0 provenance requires {field or name}"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _source_remaining_action_digest(actions: NDArray[Any]) -> str:
    payload = {
        "action_byte_digest": _sha256_bytes(actions.tobytes(order="C")),
        "action_dtype": actions.dtype.str,
        "action_shape": list(actions.shape),
    }
    return _sha256_bytes(canonical_json_bytes(payload))


def _action_control_contract(
    action_contract: PickCubeReplayActionContract,
    *,
    action_dtype: str,
) -> PickCubeActionControlContractV1:
    try:
        return PickCubeActionControlContractV1(
            coordinate_frame=action_contract.coordinate_frame,
            control_period_s=action_contract.control_period_s,
            action_dtype=np.dtype(action_dtype).str,
            action_dimension=action_contract.total_dimension,
        )
    except StateIndexedBuildError as exc:
        raise ManiSkillPickCubeStateIndexedReplayError(
            "replay and anchor action-control contracts are incompatible"
        ) from exc


def _anchor_facts(
    episode: PickCubeStateIndexedEpisodeV1,
) -> tuple[PickCubeAnchorStateFacts, ...]:
    return tuple(
        PickCubeAnchorStateFacts(
            state_index=state.state_index,
            success=state.task_snapshot.success,
            grasped=state.task_snapshot.is_grasped,
            object_placed=state.task_snapshot.is_obj_placed,
            robot_static=state.task_snapshot.is_robot_static,
            cube_to_goal_distance=state.task_snapshot.cube_to_goal_distance,
            tcp_to_cube_distance=state.task_snapshot.tcp_to_cube_distance,
            complete=True,
        )
        for state in episode.states
    )


def _selected_anchors(
    episode: PickCubeStateIndexedEpisodeV1,
    *,
    candidate_horizon: int,
) -> Mapping[int, PickCubeStateAnchor]:
    anchors = select_pickcube_state_anchors(
        source_trajectory_id=episode.source_trajectory_id,
        source_seed=episode.seed,
        action_count=int(episode.source_actions.shape[0]),
        state_facts=_anchor_facts(episode),
        candidate_horizon=candidate_horizon,
        maximum_anchor_count=DEFAULT_MAX_ANCHORS_PER_TRAJECTORY,
    )
    return MappingProxyType({anchor.state_index: anchor for anchor in anchors})


def _static_action_check(
    action: ActionChunk,
    action_contract: PickCubeReplayActionContract,
    *,
    role: str,
) -> None:
    """Check structure and finiteness while deliberately deferring value bounds."""
    if action.actions.shape[1:] != action_contract.row_shape:
        raise ReplayInvalidContextError(f"{role}_action_shape_mismatch")
    if action.actions.dtype.hasobject or not np.issubdtype(
        action.actions.dtype, np.floating
    ):
        raise ReplayInvalidContextError(f"{role}_action_dtype_not_floating")
    if action.coordinate_frame != action_contract.coordinate_frame:
        raise ReplayInvalidContextError(f"{role}_action_coordinate_frame_mismatch")
    if action.control_period_s != action_contract.control_period_s:
        raise ReplayInvalidContextError(f"{role}_action_control_period_mismatch")
    if not bool(np.all(np.isfinite(action.actions))):
        raise ReplayInvalidContextError(f"{role}_action_non_finite")


def _actions_equal(left: ActionChunk, right: ActionChunk) -> bool:
    return bool(
        left.coordinate_frame == right.coordinate_frame
        and left.control_period_s == right.control_period_s
        and left.schema_version == right.schema_version
        and left.actions.dtype == right.actions.dtype
        and left.actions.shape == right.actions.shape
        and left.actions.tobytes(order="C") == right.actions.tobytes(order="C")
    )


def _arrays_equal(left: NDArray[Any], right: NDArray[Any]) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


class ManiSkillPickCubeStateIndexedCaseProvider:
    """Resolve anchor M0/M1 pairs to exact intermediate-state replay cases."""

    def __init__(
        self,
        source_binding: ReplaySourceBinding,
        archive: PickCubeStateIndexedArchiveV1,
        *,
        adapter_configuration_digest: str,
        compatibility_identity: str,
        action_contract: PickCubeReplayActionContract,
        candidate_horizon: int = DEFAULT_CANDIDATE_HORIZON,
    ) -> None:
        """Bind every anchor, suffix, proposal, and state identity in memory."""
        if not isinstance(source_binding, ReplaySourceBinding):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "state-indexed provider requires ReplaySourceBinding"
            )
        if not isinstance(archive, PickCubeStateIndexedArchiveV1):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "state-indexed provider requires PickCubeStateIndexedArchiveV1"
            )
        if not isinstance(action_contract, PickCubeReplayActionContract):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "state-indexed provider requires PickCubeReplayActionContract"
            )
        _require_configuration_digest(adapter_configuration_digest)
        _require_digest(compatibility_identity, field="compatibility identity")
        _require_integer(candidate_horizon, field="candidate horizon", minimum=1)
        source_binding.assert_unchanged()
        archive_digest = archive.content_digest
        archive_by_id = MappingProxyType(
            {
                archive_episode.episode_id: archive_episode
                for archive_episode in archive.episodes
            }
        )
        for archive_episode in archive.episodes:
            if archive_episode.compatibility_identity != compatibility_identity:
                raise ManiSkillPickCubeStateIndexedReplayError(
                    "state-indexed archive compatibility identity mismatch"
                )

        control_contract = _action_control_contract(
            action_contract,
            action_dtype=archive.episodes[0].source_actions.dtype.str,
        )
        bound_anchors: dict[tuple[str, str], _BoundAnchorSource] = {}
        anchor_ids: set[str] = set()
        selected_by_episode = {
            archive_episode.episode_id: _selected_anchors(
                archive_episode, candidate_horizon=candidate_horizon
            )
            for archive_episode in archive.episodes
        }
        for source_episode in source_binding.episodes:
            bound = self._bind_anchor_source(
                source_episode,
                archive_by_id=archive_by_id,
                selected_by_episode=selected_by_episode,
                archive_content_digest=archive_digest,
                compatibility_identity=compatibility_identity,
                candidate_horizon=candidate_horizon,
                action_contract=action_contract,
                control_contract=control_contract,
            )
            if bound.anchor.anchor_id in anchor_ids:
                raise ManiSkillPickCubeStateIndexedReplayError(
                    "anchor M0 dataset contains a duplicate anchor identity"
                )
            anchor_ids.add(bound.anchor.anchor_id)
            key = (source_episode.episode_id, bound.source_candidate.candidate_id)
            bound_anchors[key] = bound

        self._source_binding = source_binding
        self._archive_digest = archive_digest
        self._bound_anchors = MappingProxyType(bound_anchors)
        self._action_contract = action_contract
        self._control_contract = control_contract
        self._candidate_horizon = candidate_horizon
        proposals = source_binding.corruption_dataset.proposals
        prevalidated_pairs = tuple(
            source_binding.resolve_prevalidated(proposal) for proposal in proposals
        )
        cases = tuple(
            self._build_case(proposal, pair)
            for proposal, pair in zip(proposals, prevalidated_pairs, strict=True)
        )
        self._cases = MappingProxyType(
            {replay_case.proposal_id: replay_case for replay_case in cases}
        )
        metadata: Mapping[str, object] = MappingProxyType(
            {
                "candidate_horizon_steps": candidate_horizon,
                "compatibility_identity": compatibility_identity,
                "real_simulator_reference": True,
                STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY: archive_digest,
                "state_semantic": STATE_TREE_SEMANTIC,
            }
        )
        bundle_digest = compute_replay_bundle_digest(
            source_dataset_id=source_binding.source_dataset_id,
            source_dataset_digest=source_binding.source_dataset_digest,
            corruption_dataset_digest=source_binding.corruption_dataset_digest,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            adapter_configuration_digest=adapter_configuration_digest,
            replay_cases=cases,
            metadata=metadata,
        )
        self._bundle = ReplayBundle(
            source_dataset_id=source_binding.source_dataset_id,
            source_dataset_digest=source_binding.source_dataset_digest,
            corruption_dataset_digest=source_binding.corruption_dataset_digest,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            adapter_configuration_digest=adapter_configuration_digest,
            replay_cases=cases,
            bundle_digest=bundle_digest,
            metadata=metadata,
        )
        source_binding.assert_unchanged()

    @property
    def provider_id(self) -> str:
        """Return the stable real-adapter identity."""
        return MANISKILL_PICKCUBE_ADAPTER_ID

    @property
    def provider_version(self) -> str:
        """Return the stable real-adapter semantic version."""
        return MANISKILL_PICKCUBE_ADAPTER_VERSION

    @property
    def replay_bundle(self) -> ReplayBundle:
        """Return the content-bound state-indexed replay bundle."""
        return self._bundle

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        """Revalidate anchor/source/continuation content and return its case."""
        pair = self._source_binding.resolve_prevalidated(proposal)
        current = self._build_case(proposal, pair)
        canonical = self._cases.get(proposal.proposal_id)
        if canonical is None or canonical.case_id != current.case_id:
            raise ReplayInvalidContextError(
                "state-indexed proposal is not present in the replay bundle"
            )
        return canonical

    def _bind_anchor_source(
        self,
        episode: Episode,
        *,
        archive_by_id: Mapping[str, PickCubeStateIndexedEpisodeV1],
        selected_by_episode: Mapping[str, Mapping[int, PickCubeStateAnchor]],
        archive_content_digest: str,
        compatibility_identity: str,
        candidate_horizon: int,
        action_contract: PickCubeReplayActionContract,
        control_contract: PickCubeActionControlContractV1,
    ) -> _BoundAnchorSource:
        if len(episode.candidates) != 1:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 episode must contain exactly one source candidate"
            )
        if episode.task_id != MANISKILL_PICKCUBE_TASK_ID:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 task identity mismatch"
            )
        candidate = episode.candidates[0]
        provenance = candidate.provenance
        if provenance.transformation_type != ANCHOR_SOURCE_TRANSFORMATION:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 transformation identity mismatch"
            )
        parameters = provenance.transformation_parameters
        archive_episode_id = _require_text(
            _parameter(parameters, "source_archive_episode_id"),
            field="source archive episode ID",
        )
        try:
            archive_episode = archive_by_id[archive_episode_id]
        except KeyError as exc:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 references an unknown state-indexed archive episode"
            ) from exc
        state_index = _require_integer(
            _parameter(parameters, "anchor_state_index"), field="anchor state index"
        )
        try:
            source_state = archive_episode.states[state_index]
        except IndexError as exc:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor state index is outside the source trajectory"
            ) from exc
        try:
            anchor = selected_by_episode[archive_episode_id][state_index]
        except KeyError as exc:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 does not reference a deterministic eligible anchor"
            ) from exc
        expected_continuation = build_source_continuation_identity(
            source_episode=archive_episode,
            anchor_state_index=state_index,
            candidate_horizon=candidate_horizon,
            action_control_contract=control_contract,
        )
        continuation_identity = _require_digest(
            _parameter(parameters, "continuation_identity"),
            field="continuation identity",
        )
        expected_parameters: tuple[tuple[str, object], ...] = (
            ("action_control_contract_digest", control_contract.content_digest),
            ("anchor_id", anchor.anchor_id),
            ("anchor_selection_reason", anchor.selection_reason),
            ("anchor_state_index", anchor.state_index),
            ("candidate_horizon", anchor.candidate_horizon),
            ("compatibility_identity", compatibility_identity),
            ("continuation_identity", expected_continuation.content_digest),
            ("progress_semantic", PICKCUBE_PROGRESS_SEMANTIC),
            ("remaining_horizon", anchor.remaining_horizon),
            ("source_archive_content_digest", archive_content_digest),
            ("source_archive_episode_content_digest", archive_episode.content_digest),
            ("source_archive_episode_id", archive_episode.episode_id),
            (
                "source_remaining_action_digest",
                _source_remaining_action_digest(
                    archive_episode.source_actions[state_index:]
                ),
            ),
            ("source_state_content_digest", source_state.content_digest),
            ("source_state_digest", source_state.state_digest),
            ("source_trajectory_id", archive_episode.source_trajectory_id),
            ("unsafe_semantic", PICKCUBE_UNSAFE_SEMANTIC),
            (
                "verifier_state_content_digest",
                source_state.verifier_state.content_digest,
            ),
            (
                "verifier_state_schema_digest",
                source_state.verifier_state.schema_digest,
            ),
        )
        mismatches = tuple(
            name
            for name, expected in expected_parameters
            if _parameter(parameters, name) != expected
        )
        if mismatches or continuation_identity != expected_continuation.content_digest:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 provenance mismatch: "
                + ", ".join(mismatches or ("continuation_identity",))
            )
        baseline_evidence_id = _require_text(
            _parameter(parameters, "baseline_evidence_id"),
            field="baseline evidence ID",
        )
        if (
            episode.source_policy_id != archive_episode.source_policy_identity
            or provenance.source_policy_id != archive_episode.source_policy_identity
            or provenance.source_task_id != MANISKILL_PICKCUBE_TASK_ID
            or provenance.source_episode_id != episode.episode_id
            or provenance.seed != archive_episode.seed
            or provenance.split_group_id != anchor.split_group_id
            or episode.split_group_id != anchor.split_group_id
        ):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 source identity or split provenance mismatch"
            )
        if (
            candidate.outcome.success is not True
            or candidate.outcome.progress != 1.0
            or candidate.outcome.unsafe is not False
            or candidate.outcome.label_source is not LabelSource.SIMULATOR
            or candidate.outcome.label_strength is not LabelStrength.STRONG
            or candidate.outcome.simulator_replay_verified is not True
            or provenance.label_source is not LabelSource.SIMULATOR
            or provenance.label_strength is not LabelStrength.STRONG
            or provenance.simulator_replay_verified is not True
        ):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 baseline outcome is not strong simulator evidence"
            )
        if len(episode.observations.frames) != 1:
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 must contain one state-vector observation"
            )
        observation = episode.observations.frames[0]
        if (
            observation.timestamp_s != 0.0
            or observation.cameras
            or not _arrays_equal(
                observation.robot_state, source_state.verifier_state.values
            )
        ):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 state-vector observation differs from the archive"
            )
        remaining = ActionChunk(
            actions=archive_episode.source_actions[state_index:],
            coordinate_frame=action_contract.coordinate_frame,
            control_period_s=action_contract.control_period_s,
        )
        if not _actions_equal(candidate.action, remaining):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "anchor M0 original action differs from archive a[t:T]"
            )
        _static_action_check(candidate.action, action_contract, role="anchor_source")
        return _BoundAnchorSource(
            source_episode=episode,
            source_candidate=candidate,
            archive_episode=archive_episode,
            source_state=source_state,
            anchor=anchor,
            continuation=expected_continuation,
            baseline_evidence_id=baseline_evidence_id,
            source_remaining_action_digest=_source_remaining_action_digest(
                candidate.action.actions
            ),
        )

    def _build_case(
        self,
        proposal: CorruptedActionProposal,
        pair: BoundActionPair,
    ) -> ReplayCase:
        try:
            anchor = self._bound_anchors[
                (pair.source_episode_id, pair.source_candidate_id)
            ]
        except KeyError as exc:
            raise ReplayInvalidContextError(
                "proposal does not reference a bound state-indexed anchor"
            ) from exc
        self._validate_pair(pair, proposal, anchor)
        state = anchor.source_state
        archive_episode = anchor.archive_episode
        reference_metadata = {
            "action_control_contract_digest": self._control_contract.content_digest,
            "anchor_id": anchor.anchor.anchor_id,
            "anchor_selection_reason": anchor.anchor.selection_reason,
            "anchor_state_index": anchor.anchor.state_index,
            "baseline_evidence_id": anchor.baseline_evidence_id,
            "candidate_horizon_steps": anchor.anchor.candidate_horizon,
            "compatibility_identity": archive_episode.compatibility_identity,
            "continuation_display_id": anchor.continuation.continuation_id,
            "continuation_identity": anchor.continuation.content_digest,
            "numeric_component_count": state.numeric_component_count,
            "remaining_horizon_steps": anchor.anchor.remaining_horizon,
            PICKCUBE_SOURCE_RESET_SEED_METADATA_KEY: archive_episode.seed,
            "source_archive_episode_content_digest": archive_episode.content_digest,
            "source_policy_identity": archive_episode.source_policy_identity,
            "source_remaining_action_digest": anchor.source_remaining_action_digest,
            "source_state_content_digest": state.content_digest,
            "source_trajectory_id": archive_episode.source_trajectory_id,
            STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY: self._archive_digest,
            "state_semantic": STATE_TREE_SEMANTIC,
            "state_structure_digest": state.structure_digest,
            "state_verification_maximum_absolute_tolerance": (
                PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
            ),
            "state_verification_semantic": PICKCUBE_STATE_VERIFICATION_SEMANTIC,
            "verifier_state_content_digest": state.verifier_state.content_digest,
            "verifier_state_schema_digest": state.verifier_state.schema_digest,
        }
        state_reference = ReplayStateReference(
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            source_reference_id=archive_episode.episode_id,
            expected_state_digest=state.state_digest,
            comparison_semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
            state_index=state.state_index,
            metadata=reference_metadata,
        )
        task_reference = ReplayTaskReference(
            task_id=PICKCUBE_TASK_ID,
            task_contract_version=PICKCUBE_TASK_CONTRACT_VERSION,
            metadata={"environment_id": "PickCube-v1", "robot_uid": "panda"},
        )
        case_id = compute_replay_case_identifier(
            proposal_id=pair.proposal_id,
            source_dataset_id=pair.source_dataset_id,
            source_dataset_digest=pair.source_dataset_digest,
            corruption_dataset_digest=pair.corruption_dataset_digest,
            source_episode_id=pair.source_episode_id,
            source_candidate_id=pair.source_candidate_id,
            split_group_id=pair.split_group_id,
            original_action=pair.original_action,
            transformed_action=pair.transformed_action,
            state_reference=state_reference,
            task_reference=task_reference,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
            unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
        )
        return ReplayCase(
            case_id=case_id,
            proposal_id=pair.proposal_id,
            source_dataset_id=pair.source_dataset_id,
            source_dataset_digest=pair.source_dataset_digest,
            corruption_dataset_digest=pair.corruption_dataset_digest,
            source_episode_id=pair.source_episode_id,
            source_candidate_id=pair.source_candidate_id,
            split_group_id=pair.split_group_id,
            original_action=pair.original_action,
            transformed_action=pair.transformed_action,
            state_reference=state_reference,
            task_reference=task_reference,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
            unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
        )

    def _validate_pair(
        self,
        pair: BoundActionPair,
        proposal: CorruptedActionProposal,
        anchor: _BoundAnchorSource,
    ) -> None:
        _static_action_check(
            pair.original_action, self._action_contract, role="baseline"
        )
        _static_action_check(
            pair.transformed_action, self._action_contract, role="corrupted"
        )
        if not _actions_equal(pair.original_action, anchor.source_candidate.action):
            raise ReplayInvalidContextError(
                "state-indexed baseline differs from bound anchor M0 source"
            )
        if (
            pair.source_episode_id != anchor.source_episode.episode_id
            or pair.source_candidate_id != anchor.source_candidate.candidate_id
            or pair.source_policy_id != anchor.archive_episode.source_policy_identity
            or pair.source_task_id != MANISKILL_PICKCUBE_TASK_ID
            or pair.split_group_id != anchor.anchor.split_group_id
        ):
            raise ReplayInvalidContextError(
                "state-indexed proposal source provenance mismatch"
            )
        parameters = proposal.resolved_parameters
        if (
            parameters.get("window_start") != 0
            or parameters.get("window_end") != anchor.anchor.candidate_horizon
            or not isinstance(parameters.get("severity_id"), str)
            or not parameters.get("severity_id")
        ):
            raise ReplayInvalidContextError(
                "state-indexed proposal window or severity contract mismatch"
            )
        horizon = anchor.anchor.candidate_horizon
        source_suffix = pair.original_action.actions[horizon:]
        transformed_suffix = pair.transformed_action.actions[horizon:]
        if not _arrays_equal(source_suffix, transformed_suffix):
            raise ReplayInvalidContextError(
                "state-indexed source continuation is not byte-identical"
            )
        current_continuation = build_source_continuation_identity(
            source_episode=anchor.archive_episode,
            anchor_state_index=anchor.anchor.state_index,
            candidate_horizon=horizon,
            action_control_contract=self._control_contract,
        )
        if current_continuation.content_digest != anchor.continuation.content_digest:
            raise ReplayInvalidContextError(
                "state-indexed continuation identity changed after binding"
            )


@dataclass(frozen=True, slots=True)
class _BoundIndexedState:
    """Immutable cached binding for one already-validated archive state."""

    source_reference_id: str
    source_reset_seed: int
    state: PickCubeIndexedStateV1
    state_digest: str
    expected_metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BoundStateIndexedArchiveReferenceStore:
    """Resolve state references from one strictly loaded immutable T+1 archive."""

    runtime_archive_directory: Path
    archive: PickCubeStateIndexedArchiveV1
    _archive_content_digest: str = field(init=False, repr=False, compare=False)
    _state_lookup: Mapping[tuple[str, int], _BoundIndexedState] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Detach the runtime path and index the strictly validated archive."""
        object.__setattr__(
            self,
            "runtime_archive_directory",
            Path(self.runtime_archive_directory).absolute(),
        )
        if not isinstance(self.archive, PickCubeStateIndexedArchiveV1):
            raise ManiSkillPickCubeStateIndexedReplayError(
                "state-indexed reference store requires a strictly loaded archive"
            )
        archive_content_digest = _require_digest(
            self.archive.content_digest,
            field="state-indexed archive content digest",
        )
        object.__setattr__(
            self,
            "_archive_content_digest",
            archive_content_digest,
        )
        state_lookup: dict[tuple[str, int], _BoundIndexedState] = {}
        for episode in self.archive.episodes:
            episode_content_digest = episode.content_digest
            for state in episode.states:
                key = (episode.episode_id, state.state_index)
                if key in state_lookup:
                    raise ManiSkillPickCubeStateIndexedReplayError(
                        "state-indexed archive contains a duplicate state selector"
                    )
                state_digest = state.state_digest
                expected_metadata: Mapping[str, object] = MappingProxyType(
                    {
                        "anchor_state_index": state.state_index,
                        "compatibility_identity": episode.compatibility_identity,
                        "numeric_component_count": state.numeric_component_count,
                        PICKCUBE_SOURCE_RESET_SEED_METADATA_KEY: episode.seed,
                        "source_archive_episode_content_digest": (
                            episode_content_digest
                        ),
                        "source_policy_identity": episode.source_policy_identity,
                        "source_state_content_digest": state.content_digest,
                        "source_trajectory_id": episode.source_trajectory_id,
                        "state_semantic": STATE_TREE_SEMANTIC,
                        "state_structure_digest": state.structure_digest,
                        "state_verification_maximum_absolute_tolerance": (
                            PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
                        ),
                        "state_verification_semantic": (
                            PICKCUBE_STATE_VERIFICATION_SEMANTIC
                        ),
                        "verifier_state_content_digest": (
                            state.verifier_state.content_digest
                        ),
                        "verifier_state_schema_digest": (
                            state.verifier_state.schema_digest
                        ),
                    }
                )
                state_lookup[key] = _BoundIndexedState(
                    source_reference_id=episode.episode_id,
                    source_reset_seed=episode.seed,
                    state=state,
                    state_digest=state_digest,
                    expected_metadata=expected_metadata,
                )
        object.__setattr__(self, "_state_lookup", MappingProxyType(state_lookup))

    @property
    def archive_content_digest(self) -> str:
        """Return the exact digest of the cached, strictly loaded archive."""
        return self._archive_content_digest

    def load_reference_state(
        self, reference: ReplayStateReference
    ) -> LoadedReferenceState:
        """Validate and resolve one state without reloading the complete archive."""
        if (
            reference.adapter_id != MANISKILL_PICKCUBE_ADAPTER_ID
            or reference.adapter_version != MANISKILL_PICKCUBE_ADAPTER_VERSION
        ):
            raise ReplayInvalidContextError(
                "PickCube indexed replay reference adapter identity mismatch"
            )
        if (
            reference.comparison_semantic
            is not StateComparisonSemantic.NUMERIC_TOLERANCE
        ):
            raise ReplayInvalidContextError(
                "PickCube indexed replay reference comparison semantic mismatch"
            )
        if reference.state_key is not None or reference.state_index is None:
            raise ReplayInvalidContextError(
                "PickCube state-indexed replay requires a state_index selector"
            )
        if (
            reference.metadata.get(STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY)
            != self.archive_content_digest
        ):
            raise ReplayInvalidContextError(
                "PickCube state-indexed replay archive digest mismatch"
            )
        try:
            bound = self._state_lookup[
                (reference.source_reference_id, reference.state_index)
            ]
        except KeyError as exc:
            raise ReplayInvalidContextError(
                "PickCube indexed replay reference selects an unknown archive state"
            ) from exc
        state = bound.state
        if bound.state_digest != reference.expected_state_digest:
            raise ReplayInvalidContextError(
                "PickCube indexed replay reference state digest mismatch"
            )
        mismatches = tuple(
            name
            for name, expected in bound.expected_metadata.items()
            if reference.metadata.get(name) != expected
        )
        if mismatches:
            raise ReplayInvalidContextError(
                "PickCube indexed state metadata mismatch: " + ", ".join(mismatches)
            )
        return LoadedReferenceState(
            source_reference_id=bound.source_reference_id,
            source_reset_seed=bound.source_reset_seed,
            state_key="state_index",
            state_digest=bound.state_digest,
            state_tree=state.tree,
            compared_component_count=state.numeric_component_count,
            state_index=state.state_index,
        )

    def validate_reference(self, reference: ReplayStateReference) -> None:
        """Reject a reference not bound to the cached immutable archive state."""
        self.load_reference_state(reference)


def build_state_indexed_maniskill_pickcube_adapter(
    *,
    source_binding: ReplaySourceBinding,
    archive_dir: Path,
    compatibility_binding: CompatibilityBinding,
    action_layout_digest: str,
    coordinate_frame: str,
    candidate_horizon: int = DEFAULT_CANDIDATE_HORIZON,
) -> ManiSkillPickCubeAdapter:
    """Construct the trusted PickCube adapter using indexed archive references."""
    compatibility_binding.require_trusted_replay_ready()
    _require_digest(action_layout_digest, field="action layout digest")
    try:
        archive = load_state_indexed_archive(archive_dir)
    except StateIndexedArchiveError as exc:
        raise ManiSkillPickCubeStateIndexedReplayError(str(exc)) from exc
    identity = ManiSkillPickCubeSemanticIdentity.from_compatibility_binding(
        compatibility_binding,
        action_layout_digest=action_layout_digest,
    )
    settings = environment_settings_from_compatibility(compatibility_binding)
    action_contract = action_contract_from_compatibility(
        compatibility_binding,
        coordinate_frame=coordinate_frame,
    )
    task_keys = PickCubeTaskKeyContract.from_compatibility_report(
        compatibility_binding.report
    )
    resolved = resolve_maniskill_pickcube_configuration(
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_keys=task_keys,
    )
    configuration_digest = compute_configuration_digest(resolved)
    provider = ManiSkillPickCubeStateIndexedCaseProvider(
        source_binding,
        archive,
        adapter_configuration_digest=configuration_digest,
        compatibility_identity=compatibility_binding.report.compatibility_identity,
        action_contract=action_contract,
        candidate_horizon=candidate_horizon,
    )
    state_store = BoundStateIndexedArchiveReferenceStore(
        runtime_archive_directory=Path(archive_dir),
        archive=archive,
    )
    session_factory = ManiSkillPickCubeSessionFactory(
        settings=settings,
        action_contract=action_contract,
        task_key_contract=task_keys,
        state_loader=state_store,
        state_comparator=CanonicalPickCubeStateTreeComparator(),
        runtime=LazyManiSkillPickCubeRuntime(),
    )
    attestation = TrustedManiSkillRuntimeAttestation.from_compatibility_binding(
        compatibility_binding,
        semantic_configuration_digest=configuration_digest,
    )
    return ManiSkillPickCubeAdapter(
        case_provider=provider,
        replay_bundle=provider.replay_bundle,
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_key_contract=task_keys,
        reference_validator=state_store,
        session_factory=session_factory,
        trust_attestation=attestation,
    )


build_maniskill_pickcube_state_indexed_adapter = (
    build_state_indexed_maniskill_pickcube_adapter
)
"""Compatibility alias with the simulator name before the replay mode."""


__all__ = [
    "STATE_INDEXED_ARCHIVE_CONTENT_DIGEST_METADATA_KEY",
    "BoundStateIndexedArchiveReferenceStore",
    "ManiSkillPickCubeStateIndexedCaseProvider",
    "ManiSkillPickCubeStateIndexedReplayError",
    "build_maniskill_pickcube_state_indexed_adapter",
    "build_state_indexed_maniskill_pickcube_adapter",
]
