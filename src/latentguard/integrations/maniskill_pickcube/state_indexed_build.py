"""Build strongly gated M0 anchor sources from state-indexed PickCube archives.

This module is deliberately CPU-only.  Runtime simulator replay is injected
through :class:`AnchorBaselineValidator`; the builder validates the returned
evidence before a source episode can enter a dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

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
from latentguard.validation import (
    DataValidationError,
    validate_episode,
    validate_episodes,
)

from .anchors import (
    DEFAULT_CANDIDATE_HORIZON,
    DEFAULT_MAX_ANCHORS_PER_TRAJECTORY,
    PickCubeAnchorStateFacts,
    PickCubeStateAnchor,
    select_pickcube_state_anchors,
)
from .archive import (
    ReferenceArchiveError,
    _publish_staging_directory,
    _reject_duplicate_json_fields,
    _reject_json_constant,
    _require_empty_destination,
    _require_regular_unlinked_file,
    _require_safe_root,
)
from .source_import import (
    MANISKILL_PICKCUBE_TASK_ID,
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_UNSAFE_SEMANTIC,
)
from .state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
)

ANCHOR_MANIFEST_FORMAT = "latentguard-maniskill-pickcube-anchor-source-manifest"
ANCHOR_MANIFEST_VERSION = 1
ANCHOR_MANIFEST_SCHEMA_VERSION = "1.1"
ANCHOR_MANIFEST_NAME = "manifest.json"
ANCHOR_EVIDENCE_DIRECTORY = "evidence"
ANCHOR_BASELINE_EVIDENCE_SCHEMA_VERSION = "1.1"
ANCHOR_SOURCE_RECORD_SCHEMA_VERSION = "1.0"
ANCHOR_EXCLUSION_SCHEMA_VERSION = "1.0"
CONTINUATION_IDENTITY_SCHEMA_VERSION = "1.0"
ACTION_CONTROL_CONTRACT_SCHEMA_VERSION = "1.0"
ANCHOR_SOURCE_TRANSFORMATION = "maniskill_pickcube_state_anchor_v1"
PICKCUBE_INSTRUCTION = "Pick up the cube and place it at the goal."
PICKCUBE_CONTROLLER_MODE = "pd_joint_pos"
PICKCUBE_STATE_COMPARISON_SEMANTIC = "tolerance_verified_full_state_v1"
PICKCUBE_STATE_COMPARISON_TOLERANCE = 1e-6

_BASELINE_VALIDATOR_REASON_CODES = frozenset(
    {
        "baseline_integrity_error",
        "baseline_invalid_context",
        "baseline_runtime_error",
        "baseline_task_failure",
        "baseline_validator_error",
    }
)

_DIGEST_PREFIX = "sha256:"
_DIGEST_LENGTH = len(_DIGEST_PREFIX) + 64


class StateIndexedBuildError(ValueError):
    """Raised when anchor sources or their evidence fail closed validation."""


class UnsupportedAnchorManifestVersionError(StateIndexedBuildError):
    """Raised when an anchor manifest uses an unsupported version."""


def _canonical_text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise StateIndexedBuildError(f"{context} must be non-empty canonical text")
    return value


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or not value.startswith(_DIGEST_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise StateIndexedBuildError(f"{context} must be a lowercase sha256 digest")
    return value


def _strict_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise StateIndexedBuildError(f"{context} must be boolean")
    return bool(value)


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StateIndexedBuildError(f"{context} must be an integer at least {minimum}")
    return int(value)


def _finite(value: object, context: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise StateIndexedBuildError(f"{context} must be finite")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise StateIndexedBuildError(f"{context} must be {qualifier}")
    return result


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _content_digest(payload: Mapping[str, object]) -> str:
    return _sha256_bytes(canonical_json_bytes(payload))


def _array_byte_digest(value: NDArray[Any]) -> str:
    return _sha256_bytes(value.tobytes(order="C"))


def _freeze_floating_actions(
    value: object, *, context: str, allow_empty_horizon: bool
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise StateIndexedBuildError(f"{context} must be a numpy.ndarray")
    if value.dtype.hasobject or not np.issubdtype(value.dtype, np.floating):
        raise StateIndexedBuildError(f"{context} must use a floating dtype")
    if (
        value.ndim != 2
        or value.shape[1] <= 0
        or (not allow_empty_horizon and value.shape[0] <= 0)
    ):
        qualifier = "non-negative" if allow_empty_horizon else "positive"
        raise StateIndexedBuildError(
            f"{context} must have {qualifier} [horizon, action_dim] shape"
        )
    if not bool(np.all(np.isfinite(value))):
        raise StateIndexedBuildError(f"{context} must contain only finite values")
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


@dataclass(frozen=True, slots=True)
class PickCubeActionControlContractV1:
    """Fixed action interpretation used by every built source candidate."""

    coordinate_frame: str
    control_period_s: float
    action_dtype: str
    action_dimension: int
    controller_mode: str = PICKCUBE_CONTROLLER_MODE
    schema_version: str = ACTION_CONTROL_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the exact controller, numeric type, dimension, and period."""
        _canonical_text(self.coordinate_frame, "action coordinate frame")
        if self.controller_mode != PICKCUBE_CONTROLLER_MODE:
            raise StateIndexedBuildError("action controller mode must be pd_joint_pos")
        period = _finite(
            self.control_period_s, "action control period", nonnegative=True
        )
        if period == 0.0:
            raise StateIndexedBuildError("action control period must be positive")
        if self.schema_version != ACTION_CONTROL_CONTRACT_SCHEMA_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                "unsupported action control contract schema version"
            )
        if not isinstance(self.action_dtype, str):
            raise StateIndexedBuildError("action dtype must be canonical text")
        try:
            dtype = np.dtype(self.action_dtype)
        except TypeError as exc:
            raise StateIndexedBuildError("action dtype is invalid") from exc
        if dtype.hasobject or not np.issubdtype(dtype, np.floating):
            raise StateIndexedBuildError("action dtype must be floating")
        if dtype.str != self.action_dtype:
            raise StateIndexedBuildError(
                f"action dtype must use canonical spelling {dtype.str!r}"
            )
        _integer(self.action_dimension, "action dimension", minimum=1)

    @property
    def content_digest(self) -> str:
        """Return the path-independent fixed action-contract digest."""
        return _content_digest(self.as_mapping())

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical JSON-ready action-contract fields."""
        return {
            "action_dimension": self.action_dimension,
            "action_dtype": self.action_dtype,
            "control_period_s": self.control_period_s,
            "controller_mode": self.controller_mode,
            "coordinate_frame": self.coordinate_frame,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class SourceContinuationIdentityV1:
    """Content identity for the byte-exact fixed suffix ``a[t+H:T]``."""

    source_policy_identity: str
    source_trajectory_id: str
    compatibility_identity: str
    anchor_state_index: int
    candidate_horizon: int
    continuation_start_index: int
    trajectory_action_count: int
    action_dtype: str
    action_shape: tuple[int, int]
    action_byte_digest: str
    action_control_contract_digest: str
    schema_version: str = CONTINUATION_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate all indices and exact suffix byte metadata."""
        _canonical_text(self.source_policy_identity, "continuation source policy")
        _canonical_text(self.source_trajectory_id, "continuation source trajectory")
        _digest(self.compatibility_identity, "continuation compatibility identity")
        _integer(self.anchor_state_index, "continuation anchor state index")
        _integer(self.candidate_horizon, "continuation candidate horizon", minimum=1)
        expected_start = self.anchor_state_index + self.candidate_horizon
        if self.continuation_start_index != expected_start:
            raise StateIndexedBuildError(
                "continuation start must equal anchor state index plus "
                "candidate horizon"
            )
        _integer(
            self.trajectory_action_count,
            "continuation trajectory action count",
            minimum=self.continuation_start_index,
        )
        if (
            not isinstance(self.action_shape, tuple)
            or len(self.action_shape) != 2
            or any(type(value) is not int for value in self.action_shape)
            or self.action_shape[0]
            != self.trajectory_action_count - self.continuation_start_index
            or self.action_shape[1] <= 0
        ):
            raise StateIndexedBuildError("continuation action shape is inconsistent")
        try:
            dtype = np.dtype(self.action_dtype)
        except TypeError as exc:
            raise StateIndexedBuildError(
                "continuation action dtype is invalid"
            ) from exc
        if dtype.str != self.action_dtype or not np.issubdtype(dtype, np.floating):
            raise StateIndexedBuildError(
                "continuation action dtype must be canonical floating dtype"
            )
        _digest(self.action_byte_digest, "continuation action byte digest")
        _digest(
            self.action_control_contract_digest,
            "continuation action control contract digest",
        )
        if self.schema_version != CONTINUATION_IDENTITY_SCHEMA_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                "unsupported continuation identity schema version"
            )

    @property
    def continuation_id(self) -> str:
        """Return the deterministic fixed-continuation identifier."""
        digest = hashlib.sha256(canonical_json_bytes(self.as_mapping())).hexdigest()
        return f"mspc-continuation-{digest}"

    @property
    def content_digest(self) -> str:
        """Return a conventional digest for evidence and manifest binding."""
        return _content_digest(self.as_mapping())

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical JSON-ready continuation identity fields."""
        return {
            "action_byte_digest": self.action_byte_digest,
            "action_control_contract_digest": self.action_control_contract_digest,
            "action_dtype": self.action_dtype,
            "action_shape": list(self.action_shape),
            "anchor_state_index": self.anchor_state_index,
            "candidate_horizon": self.candidate_horizon,
            "compatibility_identity": self.compatibility_identity,
            "continuation_start_index": self.continuation_start_index,
            "schema_version": self.schema_version,
            "source_policy_identity": self.source_policy_identity,
            "source_trajectory_id": self.source_trajectory_id,
            "trajectory_action_count": self.trajectory_action_count,
        }


def build_source_continuation_identity(
    *,
    source_episode: PickCubeStateIndexedEpisodeV1,
    anchor_state_index: int,
    candidate_horizon: int,
    action_control_contract: PickCubeActionControlContractV1,
) -> SourceContinuationIdentityV1:
    """Bind ``a[t+H:T]`` by exact dtype, shape, bytes, and source semantics."""
    if not isinstance(source_episode, PickCubeStateIndexedEpisodeV1):
        raise StateIndexedBuildError("continuation requires a state-indexed episode")
    _require_episode_action_contract(source_episode, action_control_contract)
    _integer(anchor_state_index, "continuation anchor state index")
    _integer(candidate_horizon, "continuation candidate horizon", minimum=1)
    start = anchor_state_index + candidate_horizon
    action_count = int(source_episode.source_actions.shape[0])
    if start > action_count:
        raise StateIndexedBuildError("continuation start exceeds source trajectory")
    continuation = _freeze_floating_actions(
        source_episode.source_actions[start:],
        context="source continuation",
        allow_empty_horizon=True,
    )
    return SourceContinuationIdentityV1(
        source_policy_identity=source_episode.source_policy_identity,
        source_trajectory_id=source_episode.source_trajectory_id,
        compatibility_identity=source_episode.compatibility_identity,
        anchor_state_index=anchor_state_index,
        candidate_horizon=candidate_horizon,
        continuation_start_index=start,
        trajectory_action_count=action_count,
        action_dtype=continuation.dtype.str,
        action_shape=cast(tuple[int, int], continuation.shape),
        action_byte_digest=_array_byte_digest(continuation),
        action_control_contract_digest=action_control_contract.content_digest,
    )


@dataclass(frozen=True, slots=True)
class AnchorBaselineEvidenceV1:
    """One independent remaining-trajectory baseline result for one anchor."""

    anchor_id: str
    source_archive_episode_id: str
    source_trajectory_id: str
    source_seed: int
    state_index: int
    source_state_digest: str
    compatibility_identity: str
    continuation_identity: str
    expected_action_count: int
    executed_action_count: int
    complete_action_execution: bool
    state_restoration_verified: bool
    complete_state_comparison: bool
    compared_component_count: int
    maximum_absolute_error: float
    verifier_state_restoration_verified: bool
    verifier_state_compared_component_count: int
    verifier_state_maximum_absolute_error: float
    comparison_semantic: str
    comparison_tolerance: float
    official_terminal_success: bool
    official_object_placed: bool
    official_robot_static: bool
    progress_before: float
    progress_after_candidate: float
    progress_delta: float
    terminal_progress: float
    progress_semantic: str
    terminal_unsafe: bool
    unsafe_semantic: str
    label_source: LabelSource
    label_strength: LabelStrength
    simulator_replay_verified: bool
    schema_version: str = ANCHOR_BASELINE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate evidence types while allowing failed results to be excluded."""
        for value, context in (
            (self.anchor_id, "baseline anchor ID"),
            (self.source_archive_episode_id, "baseline source archive episode ID"),
            (self.source_trajectory_id, "baseline source trajectory ID"),
            (self.continuation_identity, "baseline continuation identity"),
            (self.comparison_semantic, "baseline comparison semantic"),
            (self.progress_semantic, "baseline progress semantic"),
            (self.unsafe_semantic, "baseline unsafe semantic"),
        ):
            _canonical_text(value, context)
        _integer(self.source_seed, "baseline source seed")
        if self.source_seed >= 2**32:
            raise StateIndexedBuildError("baseline source seed must be uint32")
        _integer(self.state_index, "baseline state index")
        _digest(self.source_state_digest, "baseline source state digest")
        _digest(self.compatibility_identity, "baseline compatibility identity")
        _integer(
            self.expected_action_count, "baseline expected action count", minimum=1
        )
        _integer(self.executed_action_count, "baseline executed action count")
        _integer(
            self.compared_component_count,
            "baseline compared component count",
            minimum=1,
        )
        _integer(
            self.verifier_state_compared_component_count,
            "baseline verifier-state compared component count",
            minimum=1,
        )
        for name in (
            "complete_action_execution",
            "state_restoration_verified",
            "complete_state_comparison",
            "verifier_state_restoration_verified",
            "official_terminal_success",
            "official_object_placed",
            "official_robot_static",
            "terminal_unsafe",
            "simulator_replay_verified",
        ):
            _strict_bool(getattr(self, name), f"baseline {name}")
        error = _finite(
            self.maximum_absolute_error,
            "baseline maximum absolute error",
            nonnegative=True,
        )
        tolerance = _finite(
            self.comparison_tolerance,
            "baseline comparison tolerance",
            nonnegative=True,
        )
        _finite(
            self.verifier_state_maximum_absolute_error,
            "baseline verifier-state maximum absolute error",
            nonnegative=True,
        )
        if tolerance == 0.0 and error != 0.0:
            raise StateIndexedBuildError(
                "nonzero baseline error cannot use zero comparison tolerance"
            )
        for name in (
            "progress_before",
            "progress_after_candidate",
            "terminal_progress",
        ):
            progress = _finite(getattr(self, name), f"baseline {name}")
            if progress not in (0.0, 1.0):
                raise StateIndexedBuildError(
                    f"baseline {name} must use binary completion values"
                )
        delta = _finite(self.progress_delta, "baseline progress delta")
        expected_delta = self.progress_after_candidate - self.progress_before
        if delta != expected_delta:
            raise StateIndexedBuildError(
                "baseline progress delta must exactly equal after minus before"
            )
        if not isinstance(self.label_source, LabelSource):
            raise StateIndexedBuildError("baseline label source is invalid")
        if not isinstance(self.label_strength, LabelStrength):
            raise StateIndexedBuildError("baseline label strength is invalid")
        if self.schema_version != ANCHOR_BASELINE_EVIDENCE_SCHEMA_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                "unsupported anchor baseline evidence schema version"
            )

    @property
    def content_digest(self) -> str:
        """Return the complete content digest for this baseline result."""
        return _content_digest(self.as_mapping())

    @property
    def evidence_id(self) -> str:
        """Return the deterministic evidence identifier."""
        return f"mspc-anchor-baseline-{self.content_digest[7:]}"

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical JSON-ready evidence fields."""
        return {
            "anchor_id": self.anchor_id,
            "compared_component_count": self.compared_component_count,
            "comparison_semantic": self.comparison_semantic,
            "comparison_tolerance": self.comparison_tolerance,
            "compatibility_identity": self.compatibility_identity,
            "complete_action_execution": self.complete_action_execution,
            "complete_state_comparison": self.complete_state_comparison,
            "continuation_identity": self.continuation_identity,
            "executed_action_count": self.executed_action_count,
            "expected_action_count": self.expected_action_count,
            "label_source": self.label_source.value,
            "label_strength": self.label_strength.value,
            "maximum_absolute_error": self.maximum_absolute_error,
            "official_object_placed": self.official_object_placed,
            "official_robot_static": self.official_robot_static,
            "official_terminal_success": self.official_terminal_success,
            "progress_after_candidate": self.progress_after_candidate,
            "progress_before": self.progress_before,
            "progress_delta": self.progress_delta,
            "progress_semantic": self.progress_semantic,
            "schema_version": self.schema_version,
            "simulator_replay_verified": self.simulator_replay_verified,
            "source_archive_episode_id": self.source_archive_episode_id,
            "source_seed": self.source_seed,
            "source_state_digest": self.source_state_digest,
            "source_trajectory_id": self.source_trajectory_id,
            "state_index": self.state_index,
            "state_restoration_verified": self.state_restoration_verified,
            "terminal_progress": self.terminal_progress,
            "terminal_unsafe": self.terminal_unsafe,
            "unsafe_semantic": self.unsafe_semantic,
            "verifier_state_compared_component_count": (
                self.verifier_state_compared_component_count
            ),
            "verifier_state_maximum_absolute_error": (
                self.verifier_state_maximum_absolute_error
            ),
            "verifier_state_restoration_verified": (
                self.verifier_state_restoration_verified
            ),
        }


@runtime_checkable
class AnchorBaselineValidator(Protocol):
    """Injected boundary for real or fake remaining-trajectory validation."""

    def validate_anchor_baseline(
        self,
        *,
        archive_content_digest: str,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        anchor: PickCubeStateAnchor,
        source_remaining_actions: NDArray[Any],
        continuation: SourceContinuationIdentityV1,
        action_control_contract: PickCubeActionControlContractV1,
    ) -> AnchorBaselineEvidenceV1:
        """Replay and describe one source remainder without fabricating success."""
        ...


@dataclass(frozen=True, slots=True)
class PickCubeAnchorSourceRecordV1:
    """Manifest record binding one accepted anchor, source, and baseline."""

    anchor: PickCubeStateAnchor
    source_archive_episode_id: str
    source_archive_episode_content_digest: str
    source_state_digest: str
    source_state_content_digest: str
    verifier_state_schema_digest: str
    verifier_state_content_digest: str
    source_remaining_action_digest: str
    source_episode_id: str
    source_candidate_id: str
    continuation: SourceContinuationIdentityV1
    baseline_evidence_id: str
    baseline_evidence_content_digest: str
    schema_version: str = ANCHOR_SOURCE_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate every accepted record identity and digest."""
        if not isinstance(self.anchor, PickCubeStateAnchor):
            raise StateIndexedBuildError("anchor record requires PickCubeStateAnchor")
        for value, context in (
            (self.source_archive_episode_id, "anchor source archive episode ID"),
            (self.source_episode_id, "anchor M0 source episode ID"),
            (self.source_candidate_id, "anchor source candidate ID"),
            (self.baseline_evidence_id, "anchor baseline evidence ID"),
        ):
            _canonical_text(value, context)
        for value, context in (
            (
                self.source_archive_episode_content_digest,
                "anchor source archive episode digest",
            ),
            (self.source_state_digest, "anchor source state digest"),
            (self.source_state_content_digest, "anchor source state content digest"),
            (self.verifier_state_schema_digest, "anchor verifier schema digest"),
            (self.verifier_state_content_digest, "anchor verifier content digest"),
            (self.source_remaining_action_digest, "anchor remaining action digest"),
            (
                self.baseline_evidence_content_digest,
                "anchor baseline evidence content digest",
            ),
        ):
            _digest(value, context)
        if not isinstance(self.continuation, SourceContinuationIdentityV1):
            raise StateIndexedBuildError(
                "anchor record requires SourceContinuationIdentityV1"
            )
        if (
            self.anchor.source_trajectory_id != self.continuation.source_trajectory_id
            or self.anchor.state_index != self.continuation.anchor_state_index
            or self.anchor.candidate_horizon != self.continuation.candidate_horizon
            or self.anchor.remaining_horizon
            != self.continuation.trajectory_action_count - self.anchor.state_index
        ):
            raise StateIndexedBuildError(
                "anchor record continuation metadata differs from its anchor"
            )
        if self.schema_version != ANCHOR_SOURCE_RECORD_SCHEMA_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                "unsupported anchor source record schema version"
            )

    @property
    def continuation_identity(self) -> str:
        """Return the sha256 continuation identity consumed by datasets."""
        return self.continuation.content_digest

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical JSON-ready accepted-anchor fields."""
        anchor = self.anchor
        return {
            "anchor": {
                "anchor_id": anchor.anchor_id,
                "candidate_horizon": anchor.candidate_horizon,
                "remaining_horizon": anchor.remaining_horizon,
                "selection_reason": anchor.selection_reason,
                "source_seed": anchor.source_seed,
                "source_trajectory_id": anchor.source_trajectory_id,
                "split_group_id": anchor.split_group_id,
                "state_index": anchor.state_index,
            },
            "baseline_evidence_content_digest": self.baseline_evidence_content_digest,
            "baseline_evidence_id": self.baseline_evidence_id,
            "continuation": dict(self.continuation.as_mapping()),
            "continuation_id": self.continuation.continuation_id,
            "continuation_identity": self.continuation_identity,
            "schema_version": self.schema_version,
            "source_archive_episode_content_digest": (
                self.source_archive_episode_content_digest
            ),
            "source_archive_episode_id": self.source_archive_episode_id,
            "source_candidate_id": self.source_candidate_id,
            "source_episode_id": self.source_episode_id,
            "source_remaining_action_digest": self.source_remaining_action_digest,
            "source_state_content_digest": self.source_state_content_digest,
            "source_state_digest": self.source_state_digest,
            "verifier_state_content_digest": self.verifier_state_content_digest,
            "verifier_state_schema_digest": self.verifier_state_schema_digest,
        }


@dataclass(frozen=True, slots=True)
class AnchorBaselineExclusionV1:
    """Deterministic reason that a scheduled anchor did not become training data."""

    anchor_id: str
    source_archive_episode_id: str
    source_trajectory_id: str
    state_index: int
    reason_code: str
    schema_version: str = ANCHOR_EXCLUSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate compact exclusion provenance without runtime exception text."""
        for value, context in (
            (self.anchor_id, "excluded anchor ID"),
            (self.source_archive_episode_id, "excluded source archive episode ID"),
            (self.source_trajectory_id, "excluded source trajectory ID"),
            (self.reason_code, "excluded anchor reason"),
        ):
            _canonical_text(value, context)
        _integer(self.state_index, "excluded anchor state index")
        if self.schema_version != ANCHOR_EXCLUSION_SCHEMA_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                "unsupported anchor exclusion schema version"
            )

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical JSON-ready exclusion fields."""
        return {
            "anchor_id": self.anchor_id,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
            "source_archive_episode_id": self.source_archive_episode_id,
            "source_trajectory_id": self.source_trajectory_id,
            "state_index": self.state_index,
        }


@dataclass(frozen=True, slots=True)
class PickCubeAnchorManifestV1:
    """Compact anchor manifest with one independent evidence object per anchor."""

    source_archive_content_digest: str
    candidate_horizon: int
    maximum_anchors_per_trajectory: int
    action_control_contract: PickCubeActionControlContractV1
    records: tuple[PickCubeAnchorSourceRecordV1, ...]
    baseline_evidence: tuple[AnchorBaselineEvidenceV1, ...]
    trajectory_state_digests: Mapping[str, tuple[str, ...]]
    exclusions: tuple[AnchorBaselineExclusionV1, ...] = ()
    serialization_version: int = ANCHOR_MANIFEST_VERSION
    schema_version: str = ANCHOR_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze ordering and enforce one-to-one accepted evidence references."""
        _digest(self.source_archive_content_digest, "anchor source archive digest")
        _integer(self.candidate_horizon, "anchor candidate horizon", minimum=1)
        _integer(
            self.maximum_anchors_per_trajectory,
            "anchor maximum per trajectory",
            minimum=1,
        )
        if not isinstance(
            self.action_control_contract, PickCubeActionControlContractV1
        ):
            raise StateIndexedBuildError("anchor manifest action contract is invalid")
        records = tuple(self.records)
        evidence = tuple(self.baseline_evidence)
        exclusions = tuple(self.exclusions)
        if not isinstance(self.trajectory_state_digests, Mapping) or not (
            self.trajectory_state_digests
        ):
            raise StateIndexedBuildError(
                "anchor manifest requires complete trajectory state inventories"
            )
        state_inventories: dict[str, tuple[str, ...]] = {}
        for trajectory_id in sorted(self.trajectory_state_digests):
            _canonical_text(trajectory_id, "state-inventory source trajectory ID")
            state_digests = tuple(self.trajectory_state_digests[trajectory_id])
            if not state_digests:
                raise StateIndexedBuildError(
                    "trajectory state digest inventory must not be empty"
                )
            for value in state_digests:
                _digest(value, "trajectory state digest inventory")
            state_inventories[trajectory_id] = state_digests
        if any(not isinstance(item, PickCubeAnchorSourceRecordV1) for item in records):
            raise StateIndexedBuildError("anchor manifest contains invalid records")
        if any(not isinstance(item, AnchorBaselineEvidenceV1) for item in evidence):
            raise StateIndexedBuildError("anchor manifest contains invalid evidence")
        if any(not isinstance(item, AnchorBaselineExclusionV1) for item in exclusions):
            raise StateIndexedBuildError("anchor manifest contains invalid exclusions")
        for values, context in (
            ((record.anchor.anchor_id for record in records), "accepted anchor IDs"),
            ((record.source_episode_id for record in records), "source episode IDs"),
            ((record.source_candidate_id for record in records), "candidate IDs"),
            ((item.evidence_id for item in evidence), "baseline evidence IDs"),
            ((item.anchor_id for item in exclusions), "excluded anchor IDs"),
        ):
            materialized = tuple(values)
            if len(materialized) != len(set(materialized)):
                raise StateIndexedBuildError(f"anchor manifest has duplicate {context}")
        evidence_by_id = {item.evidence_id: item for item in evidence}
        if len(records) != len(evidence):
            raise StateIndexedBuildError(
                "anchor manifest requires exactly one baseline evidence per record"
            )
        for record in records:
            item = evidence_by_id.get(record.baseline_evidence_id)
            if (
                item is None
                or item.content_digest != record.baseline_evidence_content_digest
            ):
                raise StateIndexedBuildError(
                    "anchor record baseline evidence reference does not resolve"
                )
            if item.anchor_id != record.anchor.anchor_id:
                raise StateIndexedBuildError(
                    "anchor record and baseline evidence identify different anchors"
                )
            _validate_manifest_accepted_evidence(record, item)
            if (
                record.anchor.candidate_horizon != self.candidate_horizon
                or record.continuation.action_control_contract_digest
                != self.action_control_contract.content_digest
            ):
                raise StateIndexedBuildError(
                    "anchor record differs from the manifest build contract"
                )
            trajectory_states = state_inventories.get(
                record.anchor.source_trajectory_id
            )
            if (
                trajectory_states is None
                or len(trajectory_states)
                != record.continuation.trajectory_action_count + 1
                or trajectory_states[record.anchor.state_index]
                != record.source_state_digest
            ):
                raise StateIndexedBuildError(
                    "anchor source state index is not exactly bound in its complete "
                    "T+1 trajectory inventory"
                )
        accepted = {record.anchor.anchor_id for record in records}
        excluded = {item.anchor_id for item in exclusions}
        if accepted & excluded:
            raise StateIndexedBuildError(
                "an anchor cannot be both accepted and excluded"
            )
        if self.serialization_version != ANCHOR_MANIFEST_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                "unsupported anchor manifest serialization version"
            )
        if self.schema_version != ANCHOR_MANIFEST_SCHEMA_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                "unsupported anchor manifest schema version"
            )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "baseline_evidence", evidence)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(
            self,
            "trajectory_state_digests",
            MappingProxyType(state_inventories),
        )

    @property
    def content_digest(self) -> str:
        """Return the deterministic aggregate anchor/evidence digest."""
        payload = {
            "action_control_contract": dict(self.action_control_contract.as_mapping()),
            "action_control_contract_digest": (
                self.action_control_contract.content_digest
            ),
            "candidate_horizon": self.candidate_horizon,
            "evidence_content_digests": [
                item.content_digest for item in self.baseline_evidence
            ],
            "exclusions": [dict(item.as_mapping()) for item in self.exclusions],
            "format": ANCHOR_MANIFEST_FORMAT,
            "maximum_anchors_per_trajectory": self.maximum_anchors_per_trajectory,
            "records": [dict(record.as_mapping()) for record in self.records],
            "schema_version": self.schema_version,
            "serialization_version": self.serialization_version,
            "source_archive_content_digest": self.source_archive_content_digest,
            "trajectory_state_digests": {
                trajectory_id: list(digests)
                for trajectory_id, digests in self.trajectory_state_digests.items()
            },
        }
        return _content_digest(payload)


@dataclass(frozen=True, slots=True)
class PickCubeAnchorBuildResult:
    """Accepted M0 source episodes plus their compact provenance manifest."""

    source_episodes: tuple[Episode, ...]
    manifest: PickCubeAnchorManifestV1

    def __post_init__(self) -> None:
        """Require a one-to-one ordering between M0 episodes and records."""
        episodes = tuple(self.source_episodes)
        try:
            validate_episodes(episodes)
        except DataValidationError as exc:
            raise StateIndexedBuildError(f"anchor M0 source dataset: {exc}") from exc
        if tuple(episode.episode_id for episode in episodes) != tuple(
            record.source_episode_id for record in self.manifest.records
        ):
            raise StateIndexedBuildError(
                "anchor source episode ordering differs from manifest records"
            )
        object.__setattr__(self, "source_episodes", episodes)


def _validate_manifest_accepted_evidence(
    record: PickCubeAnchorSourceRecordV1,
    evidence: AnchorBaselineEvidenceV1,
) -> None:
    """Fail closed if an accepted manifest entry is not a strong baseline."""
    anchor = record.anchor
    if (
        evidence.source_archive_episode_id != record.source_archive_episode_id
        or evidence.source_trajectory_id != anchor.source_trajectory_id
        or evidence.source_seed != anchor.source_seed
        or evidence.state_index != anchor.state_index
        or evidence.source_state_digest != record.source_state_digest
        or evidence.compatibility_identity != record.continuation.compatibility_identity
        or evidence.continuation_identity != record.continuation_identity
    ):
        raise StateIndexedBuildError(
            "accepted anchor evidence identity differs from its record"
        )
    if (
        evidence.expected_action_count != anchor.remaining_horizon
        or evidence.executed_action_count != anchor.remaining_horizon
        or not evidence.complete_action_execution
    ):
        raise StateIndexedBuildError(
            "accepted anchor evidence has incomplete baseline execution"
        )
    if (
        evidence.comparison_semantic != PICKCUBE_STATE_COMPARISON_SEMANTIC
        or evidence.comparison_tolerance != PICKCUBE_STATE_COMPARISON_TOLERANCE
        or not evidence.state_restoration_verified
        or not evidence.complete_state_comparison
        or evidence.maximum_absolute_error > PICKCUBE_STATE_COMPARISON_TOLERANCE
        or not evidence.verifier_state_restoration_verified
        or evidence.verifier_state_compared_component_count <= 0
        or evidence.verifier_state_maximum_absolute_error
        > PICKCUBE_STATE_COMPARISON_TOLERANCE
    ):
        raise StateIndexedBuildError(
            "accepted anchor evidence has unverified state restoration"
        )
    if (
        evidence.label_source is not LabelSource.SIMULATOR
        or evidence.label_strength is not LabelStrength.STRONG
        or not evidence.simulator_replay_verified
    ):
        raise StateIndexedBuildError(
            "accepted anchor evidence is not strong simulator evidence"
        )
    if (
        not evidence.official_terminal_success
        or not evidence.official_object_placed
        or not evidence.official_robot_static
    ):
        raise StateIndexedBuildError(
            "accepted anchor evidence does not end in official task success"
        )
    if (
        evidence.progress_semantic != PICKCUBE_PROGRESS_SEMANTIC
        or evidence.progress_before != 0.0
        or evidence.terminal_progress != 1.0
        or evidence.unsafe_semantic != PICKCUBE_UNSAFE_SEMANTIC
    ):
        raise StateIndexedBuildError(
            "accepted anchor evidence has incompatible task semantics"
        )


def build_state_indexed_anchor_sources(
    archive: PickCubeStateIndexedArchiveV1,
    *,
    action_control_contract: PickCubeActionControlContractV1,
    baseline_validator: AnchorBaselineValidator,
    candidate_horizon: int = DEFAULT_CANDIDATE_HORIZON,
    maximum_anchors_per_trajectory: int = DEFAULT_MAX_ANCHORS_PER_TRAJECTORY,
    source_trajectory_limit: int | None = None,
) -> PickCubeAnchorBuildResult:
    """Schedule anchors and admit only independently successful strong baselines."""
    if not isinstance(archive, PickCubeStateIndexedArchiveV1):
        raise StateIndexedBuildError("expected PickCubeStateIndexedArchiveV1")
    if not isinstance(action_control_contract, PickCubeActionControlContractV1):
        raise StateIndexedBuildError("expected PickCubeActionControlContractV1")
    if not isinstance(baseline_validator, AnchorBaselineValidator):
        raise StateIndexedBuildError(
            "baseline validator must implement validate_anchor_baseline"
        )
    _integer(candidate_horizon, "anchor candidate horizon", minimum=1)
    _integer(
        maximum_anchors_per_trajectory,
        "anchor maximum per trajectory",
        minimum=1,
    )
    if source_trajectory_limit is not None:
        _integer(
            source_trajectory_limit,
            "anchor source trajectory limit",
            minimum=1,
        )
    archive_digest = archive.content_digest
    records: list[PickCubeAnchorSourceRecordV1] = []
    evidence_items: list[AnchorBaselineEvidenceV1] = []
    exclusions: list[AnchorBaselineExclusionV1] = []
    source_episodes: list[Episode] = []

    selected_source_episodes = (
        archive.episodes
        if source_trajectory_limit is None
        else archive.episodes[:source_trajectory_limit]
    )
    for source_episode in selected_source_episodes:
        _require_episode_action_contract(source_episode, action_control_contract)
        anchors = select_pickcube_state_anchors(
            source_trajectory_id=source_episode.source_trajectory_id,
            source_seed=source_episode.seed,
            action_count=int(source_episode.source_actions.shape[0]),
            state_facts=_anchor_facts(source_episode),
            candidate_horizon=candidate_horizon,
            maximum_anchor_count=maximum_anchors_per_trajectory,
        )
        for anchor in anchors:
            state = source_episode.states[anchor.state_index]
            remaining = _freeze_floating_actions(
                source_episode.source_actions[anchor.state_index :],
                context="anchor source remainder",
                allow_empty_horizon=False,
            )
            continuation = build_source_continuation_identity(
                source_episode=source_episode,
                anchor_state_index=anchor.state_index,
                candidate_horizon=anchor.candidate_horizon,
                action_control_contract=action_control_contract,
            )
            try:
                evidence = baseline_validator.validate_anchor_baseline(
                    archive_content_digest=archive_digest,
                    source_episode=source_episode,
                    source_state=state,
                    anchor=anchor,
                    source_remaining_actions=remaining,
                    continuation=continuation,
                    action_control_contract=action_control_contract,
                )
            except Exception as exc:
                reason_code = getattr(exc, "reason_code", None)
                if reason_code not in _BASELINE_VALIDATOR_REASON_CODES:
                    reason_code = "baseline_validator_error"
                exclusions.append(
                    _exclusion(source_episode, anchor, cast(str, reason_code))
                )
                continue
            reason = _baseline_rejection_reason(
                evidence=evidence,
                source_episode=source_episode,
                source_state=state,
                anchor=anchor,
                continuation=continuation,
            )
            if reason is not None:
                exclusions.append(_exclusion(source_episode, anchor, reason))
                continue
            assert isinstance(evidence, AnchorBaselineEvidenceV1)
            m0_episode = _build_anchor_m0_episode(
                archive_content_digest=archive_digest,
                source_episode=source_episode,
                state=state,
                anchor=anchor,
                remaining=remaining,
                continuation=continuation,
                action_control_contract=action_control_contract,
                evidence=evidence,
            )
            record = _build_anchor_record(
                source_episode=source_episode,
                state=state,
                anchor=anchor,
                remaining=remaining,
                continuation=continuation,
                evidence=evidence,
                m0_episode=m0_episode,
            )
            source_episodes.append(m0_episode)
            records.append(record)
            evidence_items.append(evidence)

    if archive.content_digest != archive_digest:
        raise StateIndexedBuildError("source archive changed during anchor build")
    manifest = PickCubeAnchorManifestV1(
        source_archive_content_digest=archive_digest,
        candidate_horizon=candidate_horizon,
        maximum_anchors_per_trajectory=maximum_anchors_per_trajectory,
        action_control_contract=action_control_contract,
        records=tuple(records),
        baseline_evidence=tuple(evidence_items),
        trajectory_state_digests={
            episode.source_trajectory_id: tuple(
                state.state_digest for state in episode.states
            )
            for episode in selected_source_episodes
        },
        exclusions=tuple(exclusions),
    )
    return PickCubeAnchorBuildResult(tuple(source_episodes), manifest)


def _anchor_facts(
    source_episode: PickCubeStateIndexedEpisodeV1,
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
        for state in source_episode.states
    )


def _require_episode_action_contract(
    source_episode: PickCubeStateIndexedEpisodeV1,
    contract: PickCubeActionControlContractV1,
) -> None:
    actions = source_episode.source_actions
    if actions.dtype.str != contract.action_dtype:
        raise StateIndexedBuildError(
            "source action dtype differs from the fixed action control contract"
        )
    if actions.shape[1] != contract.action_dimension:
        raise StateIndexedBuildError(
            "source action dimension differs from the fixed action control contract"
        )


def _baseline_rejection_reason(
    *,
    evidence: object,
    source_episode: PickCubeStateIndexedEpisodeV1,
    source_state: PickCubeIndexedStateV1,
    anchor: PickCubeStateAnchor,
    continuation: SourceContinuationIdentityV1,
) -> str | None:
    if not isinstance(evidence, AnchorBaselineEvidenceV1):
        return "baseline_evidence_invalid"
    if (
        evidence.anchor_id != anchor.anchor_id
        or evidence.source_archive_episode_id != source_episode.episode_id
        or evidence.source_trajectory_id != source_episode.source_trajectory_id
        or evidence.source_seed != source_episode.seed
        or evidence.state_index != anchor.state_index
        or evidence.source_state_digest != source_state.state_digest
        or evidence.compatibility_identity != source_episode.compatibility_identity
        or evidence.continuation_identity != continuation.content_digest
    ):
        return "baseline_identity_mismatch"
    if (
        evidence.expected_action_count != anchor.remaining_horizon
        or evidence.executed_action_count != anchor.remaining_horizon
        or not evidence.complete_action_execution
    ):
        return "baseline_execution_incomplete"
    if (
        evidence.comparison_semantic != PICKCUBE_STATE_COMPARISON_SEMANTIC
        or evidence.comparison_tolerance != PICKCUBE_STATE_COMPARISON_TOLERANCE
        or not evidence.state_restoration_verified
        or not evidence.complete_state_comparison
        or evidence.compared_component_count != source_state.numeric_component_count
        or evidence.maximum_absolute_error > PICKCUBE_STATE_COMPARISON_TOLERANCE
        or not evidence.verifier_state_restoration_verified
        or evidence.verifier_state_compared_component_count
        != int(source_state.verifier_state.values.size)
        or evidence.verifier_state_maximum_absolute_error
        > PICKCUBE_STATE_COMPARISON_TOLERANCE
    ):
        return "baseline_restoration_unverified"
    if (
        evidence.label_source is not LabelSource.SIMULATOR
        or evidence.label_strength is not LabelStrength.STRONG
        or not evidence.simulator_replay_verified
    ):
        return "baseline_not_strong_simulator_evidence"
    if (
        not evidence.official_terminal_success
        or not evidence.official_object_placed
        or not evidence.official_robot_static
    ):
        return "baseline_terminal_failure"
    if evidence.unsafe_semantic != PICKCUBE_UNSAFE_SEMANTIC:
        return "baseline_unsafe_semantic_mismatch"
    if (
        evidence.progress_semantic != PICKCUBE_PROGRESS_SEMANTIC
        or evidence.progress_before != 0.0
        or evidence.terminal_progress != 1.0
    ):
        return "baseline_progress_semantic_mismatch"
    return None


def _exclusion(
    source_episode: PickCubeStateIndexedEpisodeV1,
    anchor: PickCubeStateAnchor,
    reason: str,
) -> AnchorBaselineExclusionV1:
    return AnchorBaselineExclusionV1(
        anchor_id=anchor.anchor_id,
        source_archive_episode_id=source_episode.episode_id,
        source_trajectory_id=source_episode.source_trajectory_id,
        state_index=anchor.state_index,
        reason_code=reason,
    )


def _stable_identifier(prefix: str, payload: Mapping[str, object]) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _remaining_action_digest(remaining: NDArray[Any]) -> str:
    return _content_digest(
        {
            "action_byte_digest": _array_byte_digest(remaining),
            "action_dtype": remaining.dtype.str,
            "action_shape": list(remaining.shape),
        }
    )


def _anchor_episode_identifier(
    *,
    archive_content_digest: str,
    source_episode: PickCubeStateIndexedEpisodeV1,
    state: PickCubeIndexedStateV1,
    anchor: PickCubeStateAnchor,
    remaining: NDArray[Any],
    continuation: SourceContinuationIdentityV1,
    action_control_contract: PickCubeActionControlContractV1,
) -> str:
    return _stable_identifier(
        "mspc-anchor-episode",
        {
            "action_control_contract_digest": action_control_contract.content_digest,
            "anchor_id": anchor.anchor_id,
            "continuation_identity": continuation.content_digest,
            "source_archive_content_digest": archive_content_digest,
            "source_archive_episode_content_digest": source_episode.content_digest,
            "source_remaining_action_digest": _remaining_action_digest(remaining),
            "source_state_content_digest": state.content_digest,
        },
    )


def _build_anchor_m0_episode(
    *,
    archive_content_digest: str,
    source_episode: PickCubeStateIndexedEpisodeV1,
    state: PickCubeIndexedStateV1,
    anchor: PickCubeStateAnchor,
    remaining: NDArray[Any],
    continuation: SourceContinuationIdentityV1,
    action_control_contract: PickCubeActionControlContractV1,
    evidence: AnchorBaselineEvidenceV1,
) -> Episode:
    episode_id = _anchor_episode_identifier(
        archive_content_digest=archive_content_digest,
        source_episode=source_episode,
        state=state,
        anchor=anchor,
        remaining=remaining,
        continuation=continuation,
        action_control_contract=action_control_contract,
    )
    candidate_id = _stable_identifier(
        "mspc-anchor-source-candidate",
        {
            "episode_id": episode_id,
            "source_policy_identity": source_episode.source_policy_identity,
            "source_remaining_action_digest": _remaining_action_digest(remaining),
        },
    )
    parameters: dict[str, str | int | float | bool | None] = {
        "action_control_contract_digest": action_control_contract.content_digest,
        "anchor_id": anchor.anchor_id,
        "anchor_selection_reason": anchor.selection_reason,
        "anchor_state_index": anchor.state_index,
        "baseline_evidence_id": evidence.evidence_id,
        "candidate_horizon": anchor.candidate_horizon,
        "compatibility_identity": source_episode.compatibility_identity,
        "continuation_identity": continuation.content_digest,
        "progress_semantic": evidence.progress_semantic,
        "remaining_horizon": anchor.remaining_horizon,
        "source_archive_content_digest": archive_content_digest,
        "source_archive_episode_content_digest": source_episode.content_digest,
        "source_archive_episode_id": source_episode.episode_id,
        "source_remaining_action_digest": _remaining_action_digest(remaining),
        "source_state_content_digest": state.content_digest,
        "source_state_digest": state.state_digest,
        "source_trajectory_id": source_episode.source_trajectory_id,
        "unsafe_semantic": evidence.unsafe_semantic,
        "verifier_state_content_digest": state.verifier_state.content_digest,
        "verifier_state_schema_digest": state.verifier_state.schema_digest,
    }
    outcome = OutcomeLabel(
        success=evidence.official_terminal_success,
        progress=evidence.terminal_progress,
        unsafe=evidence.terminal_unsafe,
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
    )
    episode = Episode(
        episode_id=episode_id,
        task_id=MANISKILL_PICKCUBE_TASK_ID,
        source_policy_id=source_episode.source_policy_identity,
        instruction=PICKCUBE_INSTRUCTION,
        observations=ObservationHistory(
            history_id=f"{episode_id}-history",
            frames=(
                ObservationFrame(
                    observation_id=f"{episode_id}-state",
                    timestamp_s=0.0,
                    robot_state=state.verifier_state.values,
                    cameras=(),
                ),
            ),
        ),
        candidates=(
            CandidateAction(
                candidate_id=candidate_id,
                action=ActionChunk(
                    actions=remaining,
                    coordinate_frame=action_control_contract.coordinate_frame,
                    control_period_s=action_control_contract.control_period_s,
                ),
                outcome=outcome,
                provenance=SampleProvenance(
                    source_episode_id=episode_id,
                    source_policy_id=source_episode.source_policy_identity,
                    source_task_id=MANISKILL_PICKCUBE_TASK_ID,
                    transformation_type=ANCHOR_SOURCE_TRANSFORMATION,
                    transformation_parameters=parameters,
                    seed=source_episode.seed,
                    label_source=LabelSource.SIMULATOR,
                    label_strength=LabelStrength.STRONG,
                    simulator_replay_verified=True,
                    split_group_id=anchor.split_group_id,
                ),
            ),
        ),
        split_group_id=anchor.split_group_id,
    )
    try:
        validate_episode(episode)
    except DataValidationError as exc:
        raise StateIndexedBuildError(f"anchor M0 source episode: {exc}") from exc
    return episode


def _build_anchor_record(
    *,
    source_episode: PickCubeStateIndexedEpisodeV1,
    state: PickCubeIndexedStateV1,
    anchor: PickCubeStateAnchor,
    remaining: NDArray[Any],
    continuation: SourceContinuationIdentityV1,
    evidence: AnchorBaselineEvidenceV1,
    m0_episode: Episode,
) -> PickCubeAnchorSourceRecordV1:
    return PickCubeAnchorSourceRecordV1(
        anchor=anchor,
        source_archive_episode_id=source_episode.episode_id,
        source_archive_episode_content_digest=source_episode.content_digest,
        source_state_digest=state.state_digest,
        source_state_content_digest=state.content_digest,
        verifier_state_schema_digest=state.verifier_state.schema_digest,
        verifier_state_content_digest=state.verifier_state.content_digest,
        source_remaining_action_digest=_remaining_action_digest(remaining),
        source_episode_id=m0_episode.episode_id,
        source_candidate_id=m0_episode.candidates[0].candidate_id,
        continuation=continuation,
        baseline_evidence_id=evidence.evidence_id,
        baseline_evidence_content_digest=evidence.content_digest,
    )


def _evidence_file_name(index: int) -> str:
    return f"{ANCHOR_EVIDENCE_DIRECTORY}/{index:06d}.json"


def save_anchor_manifest(manifest: PickCubeAnchorManifestV1, output_dir: Path) -> Path:
    """Transactionally save a strict manifest and one JSON file per evidence."""
    if not isinstance(manifest, PickCubeAnchorManifestV1):
        raise StateIndexedBuildError("expected PickCubeAnchorManifestV1")
    destination = Path(output_dir).absolute()
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        _require_empty_destination(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'anchors'}.staging-",
                dir=destination.parent,
            )
        )
        evidence_dir = staging / ANCHOR_EVIDENCE_DIRECTORY
        evidence_dir.mkdir()
        descriptors: list[dict[str, object]] = []
        for index, evidence in enumerate(manifest.baseline_evidence):
            relative = _evidence_file_name(index)
            evidence_payload = dict(evidence.as_mapping())
            evidence_payload["content_digest"] = evidence.content_digest
            evidence_payload["evidence_id"] = evidence.evidence_id
            _write_json_exclusive(staging / relative, evidence_payload)
            descriptors.append(
                {
                    "content_digest": evidence.content_digest,
                    "evidence_id": evidence.evidence_id,
                    "path": relative,
                }
            )
        payload: dict[str, object] = {
            "accepted_anchor_count": len(manifest.records),
            "action_control_contract": dict(
                manifest.action_control_contract.as_mapping()
            ),
            "action_control_contract_digest": (
                manifest.action_control_contract.content_digest
            ),
            "candidate_horizon": manifest.candidate_horizon,
            "evidence_count": len(manifest.baseline_evidence),
            "evidence_files": descriptors,
            "exclusion_count": len(manifest.exclusions),
            "exclusions": [dict(item.as_mapping()) for item in manifest.exclusions],
            "format": ANCHOR_MANIFEST_FORMAT,
            "manifest_content_digest": manifest.content_digest,
            "maximum_anchors_per_trajectory": (manifest.maximum_anchors_per_trajectory),
            "records": [dict(record.as_mapping()) for record in manifest.records],
            "schema_version": manifest.schema_version,
            "serialization_version": manifest.serialization_version,
            "source_archive_content_digest": manifest.source_archive_content_digest,
            "trajectory_state_digests": {
                trajectory_id: list(digests)
                for trajectory_id, digests in manifest.trajectory_state_digests.items()
            },
        }
        _write_json_exclusive(staging / ANCHOR_MANIFEST_NAME, payload)
        _publish_staging_directory(staging, destination)
    except StateIndexedBuildError as exc:
        primary_error = exc
        raise
    except ReferenceArchiveError as exc:
        wrapped = StateIndexedBuildError(f"anchor manifest: {exc}")
        primary_error = wrapped
        raise wrapped from exc
    except (OSError, TypeError, ValueError) as exc:
        wrapped = StateIndexedBuildError(
            f"anchor manifest could not save transactionally: {exc}"
        )
        primary_error = wrapped
        raise wrapped from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if staging is not None and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        f"anchor manifest staging cleanup also failed: {cleanup_error}"
                    )
                else:
                    raise StateIndexedBuildError(
                        "could not clean anchor manifest staging directory"
                    ) from cleanup_error
    return destination / ANCHOR_MANIFEST_NAME


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")


def load_anchor_manifest(output_dir: Path) -> PickCubeAnchorManifestV1:
    """Reload and independently validate the strict anchor/evidence inventory."""
    try:
        requested = Path(output_dir).absolute()
        _require_safe_root(requested)
        root = requested.resolve()
        manifest_path = root / ANCHOR_MANIFEST_NAME
        _require_regular_unlinked_file(manifest_path, "AnchorManifest.manifest")
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_fields,
        )
        item = _mapping(raw, "AnchorManifest.manifest")
        _exact_fields(
            item,
            {
                "accepted_anchor_count",
                "action_control_contract",
                "action_control_contract_digest",
                "candidate_horizon",
                "evidence_count",
                "evidence_files",
                "exclusion_count",
                "exclusions",
                "format",
                "manifest_content_digest",
                "maximum_anchors_per_trajectory",
                "records",
                "schema_version",
                "serialization_version",
                "source_archive_content_digest",
                "trajectory_state_digests",
            },
            "AnchorManifest.manifest",
        )
        if (
            _text_field(item, "format", "AnchorManifest.manifest")
            != ANCHOR_MANIFEST_FORMAT
        ):
            raise UnsupportedAnchorManifestVersionError(
                "unsupported anchor manifest format"
            )
        version = _int_field(item, "serialization_version", "AnchorManifest.manifest")
        if version != ANCHOR_MANIFEST_VERSION:
            raise UnsupportedAnchorManifestVersionError(
                f"unsupported anchor manifest version {version!r}"
            )
        contract = _decode_action_contract(
            _field(item, "action_control_contract", "AnchorManifest.manifest"),
            "AnchorManifest.action_control_contract",
        )
        expected_contract_digest = _digest(
            _field(item, "action_control_contract_digest", "AnchorManifest.manifest"),
            "AnchorManifest.action_control_contract_digest",
        )
        if contract.content_digest != expected_contract_digest:
            raise StateIndexedBuildError("anchor action contract digest mismatch")
        descriptors = _list_field(item, "evidence_files", "AnchorManifest.manifest")
        evidence_count = _int_field(item, "evidence_count", "AnchorManifest.manifest")
        if evidence_count != len(descriptors):
            raise StateIndexedBuildError("anchor evidence count mismatch")
        evidence = tuple(
            _load_evidence_descriptor(root, descriptor, index)
            for index, descriptor in enumerate(descriptors)
        )
        records_raw = _list_field(item, "records", "AnchorManifest.manifest")
        records = tuple(
            _decode_record(value, f"AnchorManifest.records[{index}]")
            for index, value in enumerate(records_raw)
        )
        if _int_field(item, "accepted_anchor_count", "AnchorManifest.manifest") != len(
            records
        ):
            raise StateIndexedBuildError("anchor accepted record count mismatch")
        exclusions_raw = _list_field(item, "exclusions", "AnchorManifest.manifest")
        exclusions = tuple(
            _decode_exclusion(value, f"AnchorManifest.exclusions[{index}]")
            for index, value in enumerate(exclusions_raw)
        )
        if _int_field(item, "exclusion_count", "AnchorManifest.manifest") != len(
            exclusions
        ):
            raise StateIndexedBuildError("anchor exclusion count mismatch")
        manifest = PickCubeAnchorManifestV1(
            source_archive_content_digest=_digest(
                _field(
                    item, "source_archive_content_digest", "AnchorManifest.manifest"
                ),
                "AnchorManifest.source_archive_content_digest",
            ),
            candidate_horizon=_int_field(
                item, "candidate_horizon", "AnchorManifest.manifest"
            ),
            maximum_anchors_per_trajectory=_int_field(
                item,
                "maximum_anchors_per_trajectory",
                "AnchorManifest.manifest",
            ),
            action_control_contract=contract,
            records=records,
            baseline_evidence=evidence,
            trajectory_state_digests=_decode_trajectory_state_digests(
                _field(item, "trajectory_state_digests", "AnchorManifest.manifest")
            ),
            exclusions=exclusions,
            serialization_version=version,
            schema_version=_text_field(
                item, "schema_version", "AnchorManifest.manifest"
            ),
        )
        expected_manifest_digest = _digest(
            _field(item, "manifest_content_digest", "AnchorManifest.manifest"),
            "AnchorManifest.manifest_content_digest",
        )
        if manifest.content_digest != expected_manifest_digest:
            raise StateIndexedBuildError("anchor manifest content digest mismatch")
        _validate_manifest_inventory(root, evidence_count)
        return manifest
    except StateIndexedBuildError:
        raise
    except ReferenceArchiveError as exc:
        raise StateIndexedBuildError(f"anchor manifest: {exc}") from exc
    except Exception as exc:
        raise StateIndexedBuildError(
            f"anchor manifest could not be read safely: {exc}"
        ) from exc


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StateIndexedBuildError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise StateIndexedBuildError(f"{context} has a non-string field")
    return cast(Mapping[str, object], value)


def _decode_trajectory_state_digests(
    value: object,
) -> Mapping[str, tuple[str, ...]]:
    context = "AnchorManifest.trajectory_state_digests"
    item = _mapping(value, context)
    if not item:
        raise StateIndexedBuildError(f"{context} must not be empty")
    decoded: dict[str, tuple[str, ...]] = {}
    for trajectory_id in sorted(item):
        _canonical_text(trajectory_id, f"{context} key")
        raw_digests = item[trajectory_id]
        if not isinstance(raw_digests, list):
            raise StateIndexedBuildError(f"{context}.{trajectory_id} must be a list")
        decoded[trajectory_id] = tuple(
            _digest(digest, f"{context}.{trajectory_id}[{index}]")
            for index, digest in enumerate(raw_digests)
        )
    return decoded


def _exact_fields(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise StateIndexedBuildError(
            f"{context} fields differ; missing={missing}, unexpected={unexpected}"
        )


def _field(value: Mapping[str, object], name: str, context: str) -> object:
    try:
        return value[name]
    except KeyError as exc:
        raise StateIndexedBuildError(f"{context}.{name} is required") from exc


def _text_field(value: Mapping[str, object], name: str, context: str) -> str:
    return _canonical_text(_field(value, name, context), f"{context}.{name}")


def _int_field(value: Mapping[str, object], name: str, context: str) -> int:
    return _integer(_field(value, name, context), f"{context}.{name}")


def _bool_field(value: Mapping[str, object], name: str, context: str) -> bool:
    return _strict_bool(_field(value, name, context), f"{context}.{name}")


def _float_field(value: Mapping[str, object], name: str, context: str) -> float:
    return _finite(_field(value, name, context), f"{context}.{name}")


def _list_field(value: Mapping[str, object], name: str, context: str) -> list[object]:
    result = _field(value, name, context)
    if not isinstance(result, list):
        raise StateIndexedBuildError(f"{context}.{name} must be a JSON list")
    return result


def _decode_action_contract(
    value: object, context: str
) -> PickCubeActionControlContractV1:
    item = _mapping(value, context)
    _exact_fields(
        item,
        {
            "action_dimension",
            "action_dtype",
            "control_period_s",
            "controller_mode",
            "coordinate_frame",
            "schema_version",
        },
        context,
    )
    return PickCubeActionControlContractV1(
        coordinate_frame=_text_field(item, "coordinate_frame", context),
        control_period_s=_float_field(item, "control_period_s", context),
        action_dtype=_text_field(item, "action_dtype", context),
        action_dimension=_int_field(item, "action_dimension", context),
        controller_mode=_text_field(item, "controller_mode", context),
        schema_version=_text_field(item, "schema_version", context),
    )


def _decode_continuation(value: object, context: str) -> SourceContinuationIdentityV1:
    item = _mapping(value, context)
    _exact_fields(
        item,
        {
            "action_byte_digest",
            "action_control_contract_digest",
            "action_dtype",
            "action_shape",
            "anchor_state_index",
            "candidate_horizon",
            "compatibility_identity",
            "continuation_start_index",
            "schema_version",
            "source_policy_identity",
            "source_trajectory_id",
            "trajectory_action_count",
        },
        context,
    )
    shape = _list_field(item, "action_shape", context)
    if len(shape) != 2 or any(type(value) is not int for value in shape):
        raise StateIndexedBuildError(f"{context}.action_shape is invalid")
    return SourceContinuationIdentityV1(
        source_policy_identity=_text_field(item, "source_policy_identity", context),
        source_trajectory_id=_text_field(item, "source_trajectory_id", context),
        compatibility_identity=_digest(
            _field(item, "compatibility_identity", context),
            f"{context}.compatibility_identity",
        ),
        anchor_state_index=_int_field(item, "anchor_state_index", context),
        candidate_horizon=_int_field(item, "candidate_horizon", context),
        continuation_start_index=_int_field(item, "continuation_start_index", context),
        trajectory_action_count=_int_field(item, "trajectory_action_count", context),
        action_dtype=_text_field(item, "action_dtype", context),
        action_shape=(cast(int, shape[0]), cast(int, shape[1])),
        action_byte_digest=_digest(
            _field(item, "action_byte_digest", context),
            f"{context}.action_byte_digest",
        ),
        action_control_contract_digest=_digest(
            _field(item, "action_control_contract_digest", context),
            f"{context}.action_control_contract_digest",
        ),
        schema_version=_text_field(item, "schema_version", context),
    )


def _decode_anchor(value: object, context: str) -> PickCubeStateAnchor:
    item = _mapping(value, context)
    _exact_fields(
        item,
        {
            "anchor_id",
            "candidate_horizon",
            "remaining_horizon",
            "selection_reason",
            "source_seed",
            "source_trajectory_id",
            "split_group_id",
            "state_index",
        },
        context,
    )
    return PickCubeStateAnchor(
        anchor_id=_text_field(item, "anchor_id", context),
        state_index=_int_field(item, "state_index", context),
        selection_reason=_text_field(item, "selection_reason", context),
        source_trajectory_id=_text_field(item, "source_trajectory_id", context),
        source_seed=_int_field(item, "source_seed", context),
        split_group_id=_text_field(item, "split_group_id", context),
        remaining_horizon=_int_field(item, "remaining_horizon", context),
        candidate_horizon=_int_field(item, "candidate_horizon", context),
    )


def _decode_record(value: object, context: str) -> PickCubeAnchorSourceRecordV1:
    item = _mapping(value, context)
    _exact_fields(
        item,
        {
            "anchor",
            "baseline_evidence_content_digest",
            "baseline_evidence_id",
            "continuation",
            "continuation_id",
            "continuation_identity",
            "schema_version",
            "source_archive_episode_content_digest",
            "source_archive_episode_id",
            "source_candidate_id",
            "source_episode_id",
            "source_remaining_action_digest",
            "source_state_content_digest",
            "source_state_digest",
            "verifier_state_content_digest",
            "verifier_state_schema_digest",
        },
        context,
    )
    continuation = _decode_continuation(
        _field(item, "continuation", context), f"{context}.continuation"
    )
    if _text_field(item, "continuation_id", context) != continuation.continuation_id:
        raise StateIndexedBuildError(f"{context}.continuation_id content mismatch")
    if (
        _digest(
            _field(item, "continuation_identity", context),
            f"{context}.continuation_identity",
        )
        != continuation.content_digest
    ):
        raise StateIndexedBuildError(
            f"{context}.continuation_identity content mismatch"
        )
    return PickCubeAnchorSourceRecordV1(
        anchor=_decode_anchor(_field(item, "anchor", context), f"{context}.anchor"),
        source_archive_episode_id=_text_field(
            item, "source_archive_episode_id", context
        ),
        source_archive_episode_content_digest=_digest(
            _field(item, "source_archive_episode_content_digest", context),
            f"{context}.source_archive_episode_content_digest",
        ),
        source_state_digest=_digest(
            _field(item, "source_state_digest", context),
            f"{context}.source_state_digest",
        ),
        source_state_content_digest=_digest(
            _field(item, "source_state_content_digest", context),
            f"{context}.source_state_content_digest",
        ),
        verifier_state_schema_digest=_digest(
            _field(item, "verifier_state_schema_digest", context),
            f"{context}.verifier_state_schema_digest",
        ),
        verifier_state_content_digest=_digest(
            _field(item, "verifier_state_content_digest", context),
            f"{context}.verifier_state_content_digest",
        ),
        source_remaining_action_digest=_digest(
            _field(item, "source_remaining_action_digest", context),
            f"{context}.source_remaining_action_digest",
        ),
        source_episode_id=_text_field(item, "source_episode_id", context),
        source_candidate_id=_text_field(item, "source_candidate_id", context),
        continuation=continuation,
        baseline_evidence_id=_text_field(item, "baseline_evidence_id", context),
        baseline_evidence_content_digest=_digest(
            _field(item, "baseline_evidence_content_digest", context),
            f"{context}.baseline_evidence_content_digest",
        ),
        schema_version=_text_field(item, "schema_version", context),
    )


def _decode_exclusion(value: object, context: str) -> AnchorBaselineExclusionV1:
    item = _mapping(value, context)
    _exact_fields(
        item,
        {
            "anchor_id",
            "reason_code",
            "schema_version",
            "source_archive_episode_id",
            "source_trajectory_id",
            "state_index",
        },
        context,
    )
    return AnchorBaselineExclusionV1(
        anchor_id=_text_field(item, "anchor_id", context),
        source_archive_episode_id=_text_field(
            item, "source_archive_episode_id", context
        ),
        source_trajectory_id=_text_field(item, "source_trajectory_id", context),
        state_index=_int_field(item, "state_index", context),
        reason_code=_text_field(item, "reason_code", context),
        schema_version=_text_field(item, "schema_version", context),
    )


def _decode_evidence(value: object, context: str) -> AnchorBaselineEvidenceV1:
    item = _mapping(value, context)
    expected = {
        "anchor_id",
        "compared_component_count",
        "comparison_semantic",
        "comparison_tolerance",
        "compatibility_identity",
        "complete_action_execution",
        "complete_state_comparison",
        "continuation_identity",
        "executed_action_count",
        "expected_action_count",
        "label_source",
        "label_strength",
        "maximum_absolute_error",
        "official_object_placed",
        "official_robot_static",
        "official_terminal_success",
        "progress_after_candidate",
        "progress_before",
        "progress_delta",
        "progress_semantic",
        "schema_version",
        "simulator_replay_verified",
        "source_archive_episode_id",
        "source_seed",
        "source_state_digest",
        "source_trajectory_id",
        "state_index",
        "state_restoration_verified",
        "terminal_progress",
        "terminal_unsafe",
        "unsafe_semantic",
        "verifier_state_compared_component_count",
        "verifier_state_maximum_absolute_error",
        "verifier_state_restoration_verified",
    }
    _exact_fields(item, expected, context)
    try:
        label_source = LabelSource(_text_field(item, "label_source", context))
        label_strength = LabelStrength(_text_field(item, "label_strength", context))
    except ValueError as exc:
        raise StateIndexedBuildError(f"{context} has invalid label metadata") from exc
    return AnchorBaselineEvidenceV1(
        anchor_id=_text_field(item, "anchor_id", context),
        source_archive_episode_id=_text_field(
            item, "source_archive_episode_id", context
        ),
        source_trajectory_id=_text_field(item, "source_trajectory_id", context),
        source_seed=_int_field(item, "source_seed", context),
        state_index=_int_field(item, "state_index", context),
        source_state_digest=_digest(
            _field(item, "source_state_digest", context),
            f"{context}.source_state_digest",
        ),
        compatibility_identity=_digest(
            _field(item, "compatibility_identity", context),
            f"{context}.compatibility_identity",
        ),
        continuation_identity=_text_field(item, "continuation_identity", context),
        expected_action_count=_int_field(item, "expected_action_count", context),
        executed_action_count=_int_field(item, "executed_action_count", context),
        complete_action_execution=_bool_field(
            item, "complete_action_execution", context
        ),
        state_restoration_verified=_bool_field(
            item, "state_restoration_verified", context
        ),
        complete_state_comparison=_bool_field(
            item, "complete_state_comparison", context
        ),
        compared_component_count=_int_field(item, "compared_component_count", context),
        maximum_absolute_error=_float_field(item, "maximum_absolute_error", context),
        verifier_state_restoration_verified=_bool_field(
            item, "verifier_state_restoration_verified", context
        ),
        verifier_state_compared_component_count=_int_field(
            item, "verifier_state_compared_component_count", context
        ),
        verifier_state_maximum_absolute_error=_float_field(
            item, "verifier_state_maximum_absolute_error", context
        ),
        comparison_semantic=_text_field(item, "comparison_semantic", context),
        comparison_tolerance=_float_field(item, "comparison_tolerance", context),
        official_terminal_success=_bool_field(
            item, "official_terminal_success", context
        ),
        official_object_placed=_bool_field(item, "official_object_placed", context),
        official_robot_static=_bool_field(item, "official_robot_static", context),
        progress_before=_float_field(item, "progress_before", context),
        progress_after_candidate=_float_field(
            item, "progress_after_candidate", context
        ),
        progress_delta=_float_field(item, "progress_delta", context),
        terminal_progress=_float_field(item, "terminal_progress", context),
        progress_semantic=_text_field(item, "progress_semantic", context),
        terminal_unsafe=_bool_field(item, "terminal_unsafe", context),
        unsafe_semantic=_text_field(item, "unsafe_semantic", context),
        label_source=label_source,
        label_strength=label_strength,
        simulator_replay_verified=_bool_field(
            item, "simulator_replay_verified", context
        ),
        schema_version=_text_field(item, "schema_version", context),
    )


def _load_evidence_descriptor(
    root: Path, value: object, index: int
) -> AnchorBaselineEvidenceV1:
    context = f"AnchorManifest.evidence_files[{index}]"
    item = _mapping(value, context)
    _exact_fields(item, {"content_digest", "evidence_id", "path"}, context)
    expected_path = _evidence_file_name(index)
    if _text_field(item, "path", context) != expected_path:
        raise StateIndexedBuildError(f"{context}.path is not canonical")
    path = root / Path(*expected_path.split("/"))
    _require_regular_unlinked_file(path, context)
    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_json_fields,
    )
    evidence_payload = _mapping(raw, context)
    expected_fields = set(AnchorBaselineEvidenceV1.__dataclass_fields__) | {
        "content_digest",
        "evidence_id",
    }
    _exact_fields(evidence_payload, expected_fields, context)
    evidence = _decode_evidence(
        {
            key: value
            for key, value in evidence_payload.items()
            if key not in {"content_digest", "evidence_id"}
        },
        context,
    )
    expected_digest = _digest(
        _field(item, "content_digest", context), f"{context}.content_digest"
    )
    stored_digest = _digest(
        _field(evidence_payload, "content_digest", context),
        f"{context}.stored_content_digest",
    )
    expected_id = _text_field(item, "evidence_id", context)
    stored_id = _text_field(evidence_payload, "evidence_id", context)
    if (
        evidence.content_digest != expected_digest
        or stored_digest != expected_digest
        or evidence.evidence_id != expected_id
        or stored_id != expected_id
    ):
        raise StateIndexedBuildError(f"{context} content identity mismatch")
    return evidence


def _validate_manifest_inventory(root: Path, evidence_count: int) -> None:
    evidence_dir = root / ANCHOR_EVIDENCE_DIRECTORY
    try:
        if (
            evidence_dir.is_symlink()
            or not evidence_dir.is_dir()
            or evidence_dir.resolve() != evidence_dir.absolute()
        ):
            raise StateIndexedBuildError(
                "anchor evidence directory is missing or unsafe"
            )
        expected_files = {ANCHOR_MANIFEST_NAME} | {
            _evidence_file_name(index) for index in range(evidence_count)
        }
        observed_files: set[str] = set()
        observed_directories: set[str] = set()
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise StateIndexedBuildError(
                    f"anchor manifest inventory contains unsafe link {relative!r}"
                )
            if path.is_dir():
                observed_directories.add(relative)
            elif path.is_file():
                _require_regular_unlinked_file(path, "AnchorManifest.inventory")
                observed_files.add(relative)
            else:
                raise StateIndexedBuildError(
                    f"anchor manifest inventory contains unsafe entry {relative!r}"
                )
        if observed_directories != {ANCHOR_EVIDENCE_DIRECTORY}:
            raise StateIndexedBuildError(
                "anchor manifest directory inventory does not exactly match"
            )
        if observed_files != expected_files:
            raise StateIndexedBuildError(
                "anchor manifest file inventory does not exactly match"
            )
    except StateIndexedBuildError:
        raise
    except (OSError, ReferenceArchiveError) as exc:
        raise StateIndexedBuildError(
            f"anchor manifest inventory could not be validated: {exc}"
        ) from exc


def assert_anchor_manifest_unchanged(
    output_dir: Path, expected_content_digest: str
) -> None:
    """Reload an anchor manifest and reject any evidence or inventory drift."""
    expected = _digest(expected_content_digest, "expected anchor manifest digest")
    if load_anchor_manifest(output_dir).content_digest != expected:
        raise StateIndexedBuildError("anchor manifest changed after binding")


# Short aliases keep call sites readable without weakening the versioned models.
ActionControlContractV1 = PickCubeActionControlContractV1
ContinuationIdentityV1 = SourceContinuationIdentityV1
AnchorSourceRecordV1 = PickCubeAnchorSourceRecordV1
AnchorManifestV1 = PickCubeAnchorManifestV1
AnchorBuildResult = PickCubeAnchorBuildResult
build_state_indexed_sources = build_state_indexed_anchor_sources


__all__ = [
    "ACTION_CONTROL_CONTRACT_SCHEMA_VERSION",
    "ANCHOR_BASELINE_EVIDENCE_SCHEMA_VERSION",
    "ANCHOR_EVIDENCE_DIRECTORY",
    "ANCHOR_MANIFEST_FORMAT",
    "ANCHOR_MANIFEST_NAME",
    "ANCHOR_MANIFEST_SCHEMA_VERSION",
    "ANCHOR_MANIFEST_VERSION",
    "ANCHOR_SOURCE_TRANSFORMATION",
    "PICKCUBE_STATE_COMPARISON_SEMANTIC",
    "PICKCUBE_STATE_COMPARISON_TOLERANCE",
    "ActionControlContractV1",
    "AnchorBaselineEvidenceV1",
    "AnchorBaselineExclusionV1",
    "AnchorBaselineValidator",
    "AnchorBuildResult",
    "AnchorManifestV1",
    "AnchorSourceRecordV1",
    "ContinuationIdentityV1",
    "PickCubeActionControlContractV1",
    "PickCubeAnchorBuildResult",
    "PickCubeAnchorManifestV1",
    "PickCubeAnchorSourceRecordV1",
    "SourceContinuationIdentityV1",
    "StateIndexedBuildError",
    "UnsupportedAnchorManifestVersionError",
    "assert_anchor_manifest_unchanged",
    "build_source_continuation_identity",
    "build_state_indexed_anchor_sources",
    "build_state_indexed_sources",
    "load_anchor_manifest",
    "save_anchor_manifest",
]
