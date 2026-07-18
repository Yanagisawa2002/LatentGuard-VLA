"""Versioned models for receding-horizon blind action shielding."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

CONTROL_SCHEMA_VERSION = "1.0"
ACTION_HORIZON = 16
ACTION_DIMENSION = 8
CANDIDATE_COUNT = 8
VERIFIER_STATE_DIMENSION = 38
FIXED_VISUAL_BATCH_SIZE = 128


class ClosedLoopModelError(ValueError):
    """Raised when closed-loop content is malformed or semantically unsafe."""


def _fail(context: str, reason: str) -> NoReturn:
    raise ClosedLoopModelError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        _fail(context, "expected canonical non-empty text")
    return value


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase sha256 content digest")
    return value


def content_digest(payload: object, *, context: str = "ClosedLoopContentV1") -> str:
    """Return a deterministic SHA-256 digest for canonical JSON content."""

    encoded = canonical_json_bytes(payload, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def array_digest(value: NDArray[Any]) -> str:
    """Bind dtype, shape, and exact C-order bytes of one numeric array."""

    if not isinstance(value, np.ndarray):
        _fail("array", "expected numpy.ndarray")
    descriptor = {
        "bytes_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
    }
    return content_digest(descriptor, context="ClosedLoopArrayV1")


def _frozen_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    context: str,
    finite: bool = True,
) -> NDArray[Any]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
    ):
        _fail(context, f"expected dtype {dtype.str} and shape {shape}")
    if finite and not bool(np.all(np.isfinite(value))):
        _fail(context, "all values must be finite")
    detached = np.array(value, copy=True, order="C", subok=False)
    detached.setflags(write=False)
    return detached


class BoundaryState(StrEnum):
    """Transactional lifecycle for one decision boundary."""

    PENDING = "pending"
    SELECTED = "selected"
    EXECUTING = "executing"
    COMPLETE = "complete"
    EXECUTION_ERROR = "execution_error"


class EpisodeState(StrEnum):
    """Terminal and non-terminal lifecycle for one control episode."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    UNSAFE = "unsafe"
    HORIZON_EXHAUSTED = "horizon_exhausted"
    EXECUTION_ERROR = "execution_error"

    @property
    def terminal(self) -> bool:
        """Return whether no further control work is permitted."""

        return self not in {EpisodeState.PENDING, EpisodeState.RUNNING}


@dataclass(frozen=True, slots=True)
class SourcePlanIdentityV1:
    """Content-bound identity of one independently replayed nominal plan."""

    source_trajectory_id: str
    split_group_id: str
    reset_seed: int
    source_action_digest: str
    initial_state_digest: str
    complete_state_tree_digests: tuple[str, ...]
    independent_replay_success: bool
    planner_identity: str
    compatibility_identity: str
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate disjointness-relevant source identity fields."""

        for name in ("source_trajectory_id", "split_group_id", "planner_identity"):
            _text(getattr(self, name), f"SourcePlanIdentityV1.{name}")
        for name in (
            "source_action_digest",
            "initial_state_digest",
            "compatibility_identity",
        ):
            _digest(getattr(self, name), f"SourcePlanIdentityV1.{name}")
        if type(self.reset_seed) is not int or not 0 <= self.reset_seed < 2**32:
            _fail("SourcePlanIdentityV1.reset_seed", "expected uint32 seed")
        states = tuple(self.complete_state_tree_digests)
        if not states:
            _fail(
                "SourcePlanIdentityV1.complete_state_tree_digests",
                "expected non-empty ordered inventory",
            )
        for value in states:
            _digest(value, "SourcePlanIdentityV1.complete_state_tree_digests")
        if self.initial_state_digest not in states:
            _fail("SourcePlanIdentityV1", "initial state is absent from inventory")
        if self.independent_replay_success is not True:
            _fail("SourcePlanIdentityV1", "independent replay must have succeeded")
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("SourcePlanIdentityV1.schema_version", "unsupported version")
        object.__setattr__(self, "complete_state_tree_digests", states)

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-ready source identity content."""

        return {
            "compatibility_identity": self.compatibility_identity,
            "complete_state_tree_digests": list(self.complete_state_tree_digests),
            "independent_replay_success": True,
            "initial_state_digest": self.initial_state_digest,
            "planner_identity": self.planner_identity,
            "reset_seed": self.reset_seed,
            "schema_version": self.schema_version,
            "source_action_digest": self.source_action_digest,
            "source_trajectory_id": self.source_trajectory_id,
            "split_group_id": self.split_group_id,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent source plan digest."""

        return content_digest(self.as_mapping(), context="SourcePlanIdentityV1")


@dataclass(frozen=True, slots=True, eq=False)
class ClosedLoopCandidateV1:
    """One fixed-length candidate chunk with no future outcome fields."""

    candidate_id: str
    definition_ordinal: int
    action_chunk: NDArray[Any]
    action_mask: NDArray[Any]
    definition_identity: str
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach arrays and require a complete sixteen-action input."""

        _text(self.candidate_id, "ClosedLoopCandidateV1.candidate_id")
        _digest(self.definition_identity, "ClosedLoopCandidateV1.definition_identity")
        if (
            type(self.definition_ordinal) is not int
            or not 0 <= self.definition_ordinal < 8
        ):
            _fail("ClosedLoopCandidateV1.definition_ordinal", "expected 0..7")
        actions = _frozen_array(
            self.action_chunk,
            dtype=np.dtype("<f8"),
            shape=(ACTION_HORIZON, ACTION_DIMENSION),
            context="ClosedLoopCandidateV1.action_chunk",
        )
        mask = _frozen_array(
            self.action_mask,
            dtype=np.dtype(np.bool_),
            shape=(ACTION_HORIZON,),
            context="ClosedLoopCandidateV1.action_mask",
            finite=False,
        )
        if not bool(np.all(mask)):
            _fail("ClosedLoopCandidateV1.action_mask", "M4C forbids padded inputs")
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("ClosedLoopCandidateV1.schema_version", "unsupported version")
        object.__setattr__(self, "action_chunk", actions)
        object.__setattr__(self, "action_mask", mask)

    def identity_mapping(self) -> dict[str, object]:
        """Return identity content without serializing raw actions."""

        return {
            "action_chunk_digest": array_digest(self.action_chunk),
            "action_mask_digest": array_digest(self.action_mask),
            "candidate_id": self.candidate_id,
            "definition_identity": self.definition_identity,
            "definition_ordinal": self.definition_ordinal,
            "schema_version": self.schema_version,
        }

    @property
    def content_digest(self) -> str:
        """Return the exact candidate identity."""

        return content_digest(self.identity_mapping(), context="ClosedLoopCandidateV1")


@dataclass(frozen=True, slots=True, eq=False)
class ClosedLoopCandidatePoolV1:
    """Eight identical ordered candidates supplied to every selector."""

    source_trajectory_id: str
    decision_ordinal: int
    nominal_plan_index: int
    pre_decision_state_digest: str
    source_action_prefix_digest: str
    candidate_pool_configuration_digest: str
    candidates: tuple[ClosedLoopCandidateV1, ...]
    exact_source_excluded: bool = True
    outcomes_available: bool = False
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require eight unique, ordinal-ordered, source-distinct candidates."""

        _text(
            self.source_trajectory_id, "ClosedLoopCandidatePoolV1.source_trajectory_id"
        )
        for name in (
            "pre_decision_state_digest",
            "source_action_prefix_digest",
            "candidate_pool_configuration_digest",
        ):
            _digest(getattr(self, name), f"ClosedLoopCandidatePoolV1.{name}")
        for name in ("decision_ordinal", "nominal_plan_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(f"ClosedLoopCandidatePoolV1.{name}", "expected non-negative int")
        candidates = tuple(self.candidates)
        if len(candidates) != CANDIDATE_COUNT or any(
            not isinstance(item, ClosedLoopCandidateV1) for item in candidates
        ):
            _fail("ClosedLoopCandidatePoolV1.candidates", "expected exactly eight")
        if tuple(item.definition_ordinal for item in candidates) != tuple(range(8)):
            _fail("ClosedLoopCandidatePoolV1.candidates", "ordinals must be 0..7")
        if len({item.candidate_id for item in candidates}) != 8:
            _fail(
                "ClosedLoopCandidatePoolV1.candidates", "candidate IDs must be unique"
            )
        if any(
            array_digest(item.action_chunk) == self.source_action_prefix_digest
            for item in candidates
        ):
            _fail("ClosedLoopCandidatePoolV1", "exact source candidate is prohibited")
        if (
            self.exact_source_excluded is not True
            or self.outcomes_available is not False
        ):
            _fail(
                "ClosedLoopCandidatePoolV1",
                "pool must be outcome-free and source-excluded",
            )
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("ClosedLoopCandidatePoolV1.schema_version", "unsupported version")
        object.__setattr__(self, "candidates", candidates)

    @property
    def ordered_candidate_ids(self) -> tuple[str, ...]:
        """Return candidate IDs in fixed definition order."""

        return tuple(item.candidate_id for item in self.candidates)

    def identity_mapping(self) -> dict[str, object]:
        """Return path-independent pool identity content."""

        return {
            "candidate_content_digests": [
                item.content_digest for item in self.candidates
            ],
            "candidate_pool_configuration_digest": (
                self.candidate_pool_configuration_digest
            ),
            "decision_ordinal": self.decision_ordinal,
            "exact_source_excluded": True,
            "nominal_plan_index": self.nominal_plan_index,
            "outcomes_available": False,
            "pre_decision_state_digest": self.pre_decision_state_digest,
            "schema_version": self.schema_version,
            "source_action_prefix_digest": self.source_action_prefix_digest,
            "source_trajectory_id": self.source_trajectory_id,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete ordered pool digest."""

        return content_digest(
            self.identity_mapping(), context="ClosedLoopCandidatePoolV1"
        )


@dataclass(frozen=True, slots=True)
class ClosedLoopDecisionRecordV1:
    """Immutable outcome-blind selection finalized before action execution."""

    episode_execution_id: str
    source_trajectory_id: str
    selector_id: str
    visual_domain: str
    decision_ordinal: int
    nominal_plan_index: int
    pre_decision_state_digest: str
    candidate_pool_digest: str
    ordered_candidate_ids: tuple[str, ...]
    selector_scores: tuple[tuple[str, float], ...]
    deterministic_ranking: tuple[str, ...]
    selected_candidate_id: str
    selected_predicted_failure_probability: float | None
    execution_stride: int
    checkpoint_ensemble_identity: str
    visual_packet_identity: str | None
    outcomes_available_during_selection: bool = False
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject missing candidates, non-finite scores, and future outcomes."""

        for name in (
            "episode_execution_id",
            "source_trajectory_id",
            "selector_id",
            "visual_domain",
        ):
            _text(getattr(self, name), f"ClosedLoopDecisionRecordV1.{name}")
        for name in (
            "pre_decision_state_digest",
            "candidate_pool_digest",
            "checkpoint_ensemble_identity",
        ):
            _digest(getattr(self, name), f"ClosedLoopDecisionRecordV1.{name}")
        if self.visual_packet_identity is not None:
            _digest(
                self.visual_packet_identity,
                "ClosedLoopDecisionRecordV1.visual_packet_identity",
            )
        for name in ("decision_ordinal", "nominal_plan_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(f"ClosedLoopDecisionRecordV1.{name}", "expected non-negative int")
        if type(self.execution_stride) is not int or self.execution_stride <= 0:
            _fail(
                "ClosedLoopDecisionRecordV1.execution_stride", "expected positive int"
            )
        candidates = tuple(self.ordered_candidate_ids)
        ranking = tuple(self.deterministic_ranking)
        if (
            len(candidates) != 8
            or len(set(candidates)) != 8
            or set(ranking) != set(candidates)
        ):
            _fail(
                "ClosedLoopDecisionRecordV1", "candidate and ranking inventories differ"
            )
        if self.selected_candidate_id != ranking[0]:
            _fail("ClosedLoopDecisionRecordV1", "selection must equal ranking[0]")
        scores = tuple(self.selector_scores)
        if tuple(key for key, _ in scores) != candidates:
            _fail(
                "ClosedLoopDecisionRecordV1.selector_scores",
                "must follow candidate order",
            )
        for key, value in scores:
            _text(key, "ClosedLoopDecisionRecordV1.selector_scores.key")
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                _fail(
                    "ClosedLoopDecisionRecordV1.selector_scores",
                    "scores must be finite",
                )
        probability = self.selected_predicted_failure_probability
        if probability is not None and (
            type(probability) not in (int, float)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            _fail("ClosedLoopDecisionRecordV1", "invalid selected probability")
        if self.outcomes_available_during_selection is not False:
            _fail("ClosedLoopDecisionRecordV1", "outcomes must be unavailable")
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("ClosedLoopDecisionRecordV1.schema_version", "unsupported version")
        object.__setattr__(self, "ordered_candidate_ids", candidates)
        object.__setattr__(self, "deterministic_ranking", ranking)
        object.__setattr__(self, "selector_scores", scores)

    def semantic_mapping(self) -> dict[str, object]:
        """Return immutable selection content with no outcome or later-state fields."""

        return {
            "candidate_pool_digest": self.candidate_pool_digest,
            "checkpoint_ensemble_identity": self.checkpoint_ensemble_identity,
            "decision_ordinal": self.decision_ordinal,
            "deterministic_ranking": list(self.deterministic_ranking),
            "episode_execution_id": self.episode_execution_id,
            "execution_stride": self.execution_stride,
            "nominal_plan_index": self.nominal_plan_index,
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "outcomes_available_during_selection": False,
            "pre_decision_state_digest": self.pre_decision_state_digest,
            "schema_version": self.schema_version,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_predicted_failure_probability": (
                self.selected_predicted_failure_probability
            ),
            "selector_id": self.selector_id,
            "selector_scores": [[key, value] for key, value in self.selector_scores],
            "source_trajectory_id": self.source_trajectory_id,
            "visual_domain": self.visual_domain,
            "visual_packet_identity": self.visual_packet_identity,
        }

    @property
    def content_digest(self) -> str:
        """Return the immutable decision identity."""

        return content_digest(
            self.semantic_mapping(), context="ClosedLoopDecisionRecordV1"
        )

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON content including the verified digest."""

        return {**self.semantic_mapping(), "content_digest": self.content_digest}


@dataclass(frozen=True, slots=True)
class BoundaryLedgerEntryV1:
    """Transactional state of one decision boundary."""

    decision_ordinal: int
    nominal_plan_index: int
    state: BoundaryState
    pre_state_reference: str
    pre_state_digest: str
    candidate_pool_digest: str | None = None
    decision_record_digest: str | None = None
    executed_step_count: int = 0
    post_state_reference: str | None = None
    post_state_digest: str | None = None
    observed_task_evidence_digest: str | None = None
    recovery_count: int = 0
    selection_event_ordinal: int | None = None
    execution_event_ordinal: int | None = None
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Enforce publication chronology for each lifecycle state."""

        if type(self.decision_ordinal) is not int or self.decision_ordinal < 0:
            _fail("BoundaryLedgerEntryV1.decision_ordinal", "expected non-negative int")
        if type(self.nominal_plan_index) is not int or self.nominal_plan_index < 0:
            _fail(
                "BoundaryLedgerEntryV1.nominal_plan_index", "expected non-negative int"
            )
        if not isinstance(self.state, BoundaryState):
            _fail("BoundaryLedgerEntryV1.state", "invalid state")
        _text(self.pre_state_reference, "BoundaryLedgerEntryV1.pre_state_reference")
        _digest(self.pre_state_digest, "BoundaryLedgerEntryV1.pre_state_digest")
        for name in (
            "candidate_pool_digest",
            "decision_record_digest",
            "post_state_digest",
            "observed_task_evidence_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _digest(value, f"BoundaryLedgerEntryV1.{name}")
        if self.post_state_reference is not None:
            _text(
                self.post_state_reference, "BoundaryLedgerEntryV1.post_state_reference"
            )
        for name in ("executed_step_count", "recovery_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(f"BoundaryLedgerEntryV1.{name}", "expected non-negative int")
        if self.state is not BoundaryState.PENDING and (
            self.candidate_pool_digest is None or self.decision_record_digest is None
        ):
            _fail("BoundaryLedgerEntryV1", "selected states require pool and decision")
        if (
            self.state in {BoundaryState.SELECTED, BoundaryState.EXECUTING}
            and self.executed_step_count != 0
        ):
            _fail("BoundaryLedgerEntryV1", "unpublished execution cannot report steps")
        if self.state is BoundaryState.COMPLETE and (
            self.executed_step_count <= 0
            or self.post_state_reference is None
            or self.post_state_digest is None
            or self.observed_task_evidence_digest is None
        ):
            _fail(
                "BoundaryLedgerEntryV1",
                "complete boundary is missing post-execution evidence",
            )
        if self.selection_event_ordinal is not None and (
            type(self.selection_event_ordinal) is not int
            or self.selection_event_ordinal < 0
        ):
            _fail("BoundaryLedgerEntryV1.selection_event_ordinal", "invalid ordinal")
        if self.execution_event_ordinal is not None:
            if (
                type(self.execution_event_ordinal) is not int
                or self.execution_event_ordinal < 0
            ):
                _fail(
                    "BoundaryLedgerEntryV1.execution_event_ordinal", "invalid ordinal"
                )
            if (
                self.selection_event_ordinal is None
                or self.execution_event_ordinal <= self.selection_event_ordinal
            ):
                _fail("BoundaryLedgerEntryV1", "selection must precede execution")
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("BoundaryLedgerEntryV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-ready ledger content."""

        return {
            "candidate_pool_digest": self.candidate_pool_digest,
            "decision_ordinal": self.decision_ordinal,
            "decision_record_digest": self.decision_record_digest,
            "executed_step_count": self.executed_step_count,
            "execution_event_ordinal": self.execution_event_ordinal,
            "nominal_plan_index": self.nominal_plan_index,
            "observed_task_evidence_digest": self.observed_task_evidence_digest,
            "post_state_digest": self.post_state_digest,
            "post_state_reference": self.post_state_reference,
            "pre_state_digest": self.pre_state_digest,
            "pre_state_reference": self.pre_state_reference,
            "recovery_count": self.recovery_count,
            "schema_version": self.schema_version,
            "selection_event_ordinal": self.selection_event_ordinal,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class ClosedLoopEpisodeRecordV1:
    """Complete transactional ledger and outcome for one selector episode."""

    episode_execution_id: str
    source_plan_digest: str
    source_trajectory_id: str
    selector_id: str
    visual_domain: str
    state: EpisodeState
    boundaries: tuple[BoundaryLedgerEntryV1, ...]
    executed_control_steps: int
    tail_fallback_count: int
    final_nominal_plan_index: int
    started_event_ordinal: int
    completed_event_ordinal: int | None = None
    execution_error_type: str | None = None
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate ordered boundaries and distinct terminal reasons."""

        for name in (
            "episode_execution_id",
            "source_trajectory_id",
            "selector_id",
            "visual_domain",
        ):
            _text(getattr(self, name), f"ClosedLoopEpisodeRecordV1.{name}")
        _digest(self.source_plan_digest, "ClosedLoopEpisodeRecordV1.source_plan_digest")
        if not isinstance(self.state, EpisodeState):
            _fail("ClosedLoopEpisodeRecordV1.state", "invalid state")
        boundaries = tuple(self.boundaries)
        if tuple(item.decision_ordinal for item in boundaries) != tuple(
            range(len(boundaries))
        ):
            _fail("ClosedLoopEpisodeRecordV1.boundaries", "must be ordinal ordered")
        for name in (
            "executed_control_steps",
            "tail_fallback_count",
            "final_nominal_plan_index",
            "started_event_ordinal",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(f"ClosedLoopEpisodeRecordV1.{name}", "expected non-negative int")
        if self.executed_control_steps != sum(
            item.executed_step_count for item in boundaries
        ):
            _fail(
                "ClosedLoopEpisodeRecordV1", "executed step total differs from ledger"
            )
        if self.state.terminal != (self.completed_event_ordinal is not None):
            _fail("ClosedLoopEpisodeRecordV1", "terminal completion ordinal mismatch")
        if (
            self.completed_event_ordinal is not None
            and self.completed_event_ordinal <= self.started_event_ordinal
        ):
            _fail("ClosedLoopEpisodeRecordV1", "completion must follow start")
        if self.execution_error_type is not None:
            _text(
                self.execution_error_type,
                "ClosedLoopEpisodeRecordV1.execution_error_type",
            )
        if (self.state is EpisodeState.EXECUTION_ERROR) != (
            self.execution_error_type is not None
        ):
            _fail(
                "ClosedLoopEpisodeRecordV1", "execution error classification mismatch"
            )
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("ClosedLoopEpisodeRecordV1.schema_version", "unsupported version")
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def recovery_count(self) -> int:
        """Return total exact-boundary recoveries."""

        return sum(item.recovery_count for item in self.boundaries)

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-ready episode content."""

        return {
            "boundaries": [item.as_mapping() for item in self.boundaries],
            "completed_event_ordinal": self.completed_event_ordinal,
            "episode_execution_id": self.episode_execution_id,
            "executed_control_steps": self.executed_control_steps,
            "execution_error_type": self.execution_error_type,
            "final_nominal_plan_index": self.final_nominal_plan_index,
            "schema_version": self.schema_version,
            "selector_id": self.selector_id,
            "source_plan_digest": self.source_plan_digest,
            "source_trajectory_id": self.source_trajectory_id,
            "started_event_ordinal": self.started_event_ordinal,
            "state": self.state.value,
            "tail_fallback_count": self.tail_fallback_count,
            "visual_domain": self.visual_domain,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete episode-ledger digest."""

        return content_digest(self.as_mapping(), context="ClosedLoopEpisodeRecordV1")


@dataclass(frozen=True, slots=True)
class SelectorOutputV1:
    """Outcome-free scores and deterministic ranking from one selector."""

    selector_id: str
    scores: Mapping[str, float]
    ranking: tuple[str, ...]
    checkpoint_ensemble_identity: str
    probabilities: bool

    def __post_init__(self) -> None:
        """Freeze and validate a complete finite eight-candidate output."""

        _text(self.selector_id, "SelectorOutputV1.selector_id")
        _digest(
            self.checkpoint_ensemble_identity,
            "SelectorOutputV1.checkpoint_ensemble_identity",
        )
        values = dict(self.scores)
        ranking = tuple(self.ranking)
        if len(values) != 8 or len(ranking) != 8 or set(values) != set(ranking):
            _fail("SelectorOutputV1", "expected one score for each ranked candidate")
        for key, value in values.items():
            _text(key, "SelectorOutputV1.scores.key")
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                _fail("SelectorOutputV1.scores", "scores must be finite")
            if self.probabilities and not 0.0 <= float(value) <= 1.0:
                _fail("SelectorOutputV1.scores", "probability must be in [0, 1]")
        if self.probabilities not in (True, False):
            _fail("SelectorOutputV1.probabilities", "expected boolean")
        object.__setattr__(self, "scores", MappingProxyType(values))
        object.__setattr__(self, "ranking", ranking)


def validate_source_plan_disjointness(
    plans: Sequence[SourcePlanIdentityV1],
    *,
    excluded_trajectory_ids: Sequence[str],
    excluded_split_group_ids: Sequence[str],
    excluded_state_tree_digests: Sequence[str],
) -> None:
    """Reject overlap within M4C or against accepted M3A/M3C identities."""

    values = tuple(plans)
    if not values or any(not isinstance(item, SourcePlanIdentityV1) for item in values):
        _fail("source plans", "expected non-empty SourcePlanIdentityV1 inventory")
    trajectory_ids = tuple(item.source_trajectory_id for item in values)
    split_ids = tuple(item.split_group_id for item in values)
    reset_seeds = tuple(item.reset_seed for item in values)
    state_digests = tuple(
        digest for item in values for digest in item.complete_state_tree_digests
    )
    for entries, context in (
        (trajectory_ids, "trajectory IDs"),
        (split_ids, "split-group IDs"),
        (reset_seeds, "reset seeds"),
    ):
        if len(entries) != len(set(entries)):
            _fail("source plans", f"duplicate {context}")
    overlaps = (
        (set(trajectory_ids) & set(excluded_trajectory_ids), "trajectory ID"),
        (set(split_ids) & set(excluded_split_group_ids), "split-group ID"),
        (set(state_digests) & set(excluded_state_tree_digests), "state-tree digest"),
    )
    for overlap, context in overlaps:
        if overlap:
            _fail("source plans", f"excluded {context} overlap")


__all__ = [
    "ACTION_DIMENSION",
    "ACTION_HORIZON",
    "CANDIDATE_COUNT",
    "CONTROL_SCHEMA_VERSION",
    "FIXED_VISUAL_BATCH_SIZE",
    "VERIFIER_STATE_DIMENSION",
    "BoundaryLedgerEntryV1",
    "BoundaryState",
    "ClosedLoopCandidatePoolV1",
    "ClosedLoopCandidateV1",
    "ClosedLoopDecisionRecordV1",
    "ClosedLoopEpisodeRecordV1",
    "ClosedLoopModelError",
    "EpisodeState",
    "SelectorOutputV1",
    "SourcePlanIdentityV1",
    "array_digest",
    "content_digest",
    "validate_source_plan_disjointness",
]
