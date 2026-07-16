"""Immutable content contracts for blind candidate selection."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

SELECTION_SCHEMA_VERSION = "1.0"
CANDIDATE_COUNT = 8
STATE_DIMENSION = 38
ACTION_HORIZON = 16
ACTION_DIMENSION = 8

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GROUP_ID_PREFIX = "m3c-group-sha256-"


class SelectionModelError(ValueError):
    """Raised when blind-selection content is malformed or inconsistent."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionModelError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail(context, "expected canonical non-empty text")
    return value


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(context, f"expected integer >= {minimum}")
    return value


def _finite(value: object, context: str) -> float:
    if type(value) not in (int, float):
        _fail(context, "expected a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        _fail(context, "expected a finite number")
    return number


def _freeze_array(
    value: object,
    *,
    shape: tuple[int, ...] | None,
    context: str,
    floating: bool = False,
    boolean: bool = False,
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        _fail(context, "expected numpy.ndarray")
    if shape is not None and value.shape != shape:
        _fail(context, f"expected shape {shape}, observed {value.shape}")
    if floating and (
        value.dtype.hasobject or not np.issubdtype(value.dtype, np.floating)
    ):
        _fail(context, "expected a non-object floating dtype")
    if boolean and value.dtype != np.dtype(np.bool_):
        _fail(context, "expected bool dtype")
    if floating and not bool(np.all(np.isfinite(value))):
        _fail(context, "values must be finite")
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


def array_content_digest(value: NDArray[Any]) -> str:
    """Return a dtype-, shape-, and byte-bound array digest."""

    contiguous = np.ascontiguousarray(value)
    payload = {
        "content_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _content_digest(payload: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


class CandidateDistribution(StrEnum):
    """Predeclared candidate distribution used only for reporting."""

    ID_LIKE = "id_like"
    SHIFTED = "shifted"


@dataclass(frozen=True, slots=True)
class SourceTrajectoryIdentityV1:
    """One M3C source trajectory and its complete state-tree inventory."""

    source_trajectory_id: str
    source_seed: int
    split_group_id: str
    complete_state_digests: tuple[str, ...]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze and validate the complete trajectory identity."""

        _text(
            self.source_trajectory_id, "SourceTrajectoryIdentityV1.source_trajectory_id"
        )
        _integer(self.source_seed, "SourceTrajectoryIdentityV1.source_seed")
        if self.source_seed >= 2**32:
            _fail("SourceTrajectoryIdentityV1.source_seed", "expected uint32")
        _text(self.split_group_id, "SourceTrajectoryIdentityV1.split_group_id")
        states = tuple(self.complete_state_digests)
        if not states:
            _fail(
                "SourceTrajectoryIdentityV1.complete_state_digests", "must not be empty"
            )
        for value in states:
            _digest(value, "SourceTrajectoryIdentityV1.complete_state_digests")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("SourceTrajectoryIdentityV1.schema_version", "unsupported version")
        object.__setattr__(self, "complete_state_digests", states)

    def as_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-native trajectory content."""

        return {
            "complete_state_digests": list(self.complete_state_digests),
            "schema_version": self.schema_version,
            "source_seed": self.source_seed,
            "source_trajectory_id": self.source_trajectory_id,
            "split_group_id": self.split_group_id,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent trajectory identity."""

        return _content_digest(self.as_mapping())


@dataclass(frozen=True, slots=True)
class SourceExclusionInventoryV1:
    """Complete M3A source identities prohibited from M3C evaluation."""

    dataset_digest: str
    anchor_manifest_digest: str
    reset_seeds: tuple[int, ...]
    source_trajectory_ids: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    complete_state_digests: tuple[str, ...]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Canonicalize inventories and reject duplicates or partial identities."""

        _digest(self.dataset_digest, "SourceExclusionInventoryV1.dataset_digest")
        _digest(
            self.anchor_manifest_digest,
            "SourceExclusionInventoryV1.anchor_manifest_digest",
        )
        seeds = tuple(sorted(self.reset_seeds))
        trajectories = tuple(sorted(self.source_trajectory_ids))
        split_groups = tuple(sorted(self.split_group_ids))
        states = tuple(sorted(set(self.complete_state_digests)))
        for name, values in (
            ("reset_seeds", seeds),
            ("source_trajectory_ids", trajectories),
            ("split_group_ids", split_groups),
        ):
            if not values:
                _fail(f"SourceExclusionInventoryV1.{name}", "must not be empty")
            if len(values) != len(set(values)):
                _fail(
                    f"SourceExclusionInventoryV1.{name}", "duplicates are unsupported"
                )
        if not states:
            _fail(
                "SourceExclusionInventoryV1.complete_state_digests",
                "must not be empty",
            )
        for seed in seeds:
            _integer(seed, "SourceExclusionInventoryV1.reset_seeds")
            if seed >= 2**32:
                _fail("SourceExclusionInventoryV1.reset_seeds", "expected uint32")
        for value in trajectories:
            _text(value, "SourceExclusionInventoryV1.source_trajectory_ids")
        for value in split_groups:
            _text(value, "SourceExclusionInventoryV1.split_group_ids")
        for value in states:
            _digest(value, "SourceExclusionInventoryV1.complete_state_digests")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("SourceExclusionInventoryV1.schema_version", "unsupported version")
        object.__setattr__(self, "reset_seeds", seeds)
        object.__setattr__(self, "source_trajectory_ids", trajectories)
        object.__setattr__(self, "split_group_ids", split_groups)
        object.__setattr__(self, "complete_state_digests", states)

    def as_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-native exclusion content."""

        return {
            "anchor_manifest_digest": self.anchor_manifest_digest,
            "complete_state_digests": list(self.complete_state_digests),
            "dataset_digest": self.dataset_digest,
            "reset_seeds": list(self.reset_seeds),
            "schema_version": self.schema_version,
            "source_trajectory_ids": list(self.source_trajectory_ids),
            "split_group_ids": list(self.split_group_ids),
        }

    @property
    def content_digest(self) -> str:
        """Return the exclusion-inventory identity."""

        return _content_digest(self.as_mapping())

    def require_disjoint(self, trajectory: SourceTrajectoryIdentityV1) -> None:
        """Reject any M3C trajectory identity overlapping accepted M3A content."""

        if trajectory.source_seed in self.reset_seeds:
            _fail("source disjointness", "reset seed overlaps M3A")
        if trajectory.source_trajectory_id in self.source_trajectory_ids:
            _fail("source disjointness", "source trajectory ID overlaps M3A")
        if trajectory.split_group_id in self.split_group_ids:
            _fail("source disjointness", "split-group ID overlaps M3A")
        overlap = set(trajectory.complete_state_digests).intersection(
            self.complete_state_digests
        )
        if overlap:
            _fail("source disjointness", "complete state-tree digest overlaps M3A")


@dataclass(frozen=True, slots=True, eq=False)
class CandidateRefV1:
    """One unlabeled selectable candidate prefix with opaque proposal identity."""

    proposal_id: str
    configuration_ordinal: int
    distribution: CandidateDistribution
    corruption_type: str
    severity_id: str
    seed: int
    action_chunk: NDArray[Any]
    action_mask: NDArray[Any]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach arrays and validate the fixed M3C inference shape."""

        _text(self.proposal_id, "CandidateRefV1.proposal_id")
        ordinal = _integer(
            self.configuration_ordinal,
            "CandidateRefV1.configuration_ordinal",
        )
        if ordinal >= CANDIDATE_COUNT:
            _fail("CandidateRefV1.configuration_ordinal", "must be smaller than 8")
        try:
            distribution = CandidateDistribution(self.distribution)
        except (TypeError, ValueError) as exc:
            raise SelectionModelError(
                "CandidateRefV1.distribution: unsupported value"
            ) from exc
        _text(self.corruption_type, "CandidateRefV1.corruption_type")
        _text(self.severity_id, "CandidateRefV1.severity_id")
        _integer(self.seed, "CandidateRefV1.seed")
        if self.seed >= 2**64:
            _fail("CandidateRefV1.seed", "must be smaller than 2**64")
        actions = _freeze_array(
            self.action_chunk,
            shape=(ACTION_HORIZON, ACTION_DIMENSION),
            context="CandidateRefV1.action_chunk",
            floating=True,
        )
        mask = _freeze_array(
            self.action_mask,
            shape=(ACTION_HORIZON,),
            context="CandidateRefV1.action_mask",
            boolean=True,
        )
        if not bool(np.all(mask)):
            _fail(
                "CandidateRefV1.action_mask",
                "fixed 16-step candidates must be complete",
            )
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("CandidateRefV1.schema_version", "unsupported version")
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "action_chunk", actions)
        object.__setattr__(self, "action_mask", mask)

    def identity_mapping(self) -> dict[str, object]:
        """Return path-independent content without exposing action values."""

        return {
            "action_chunk_digest": array_content_digest(self.action_chunk),
            "action_mask_digest": array_content_digest(self.action_mask),
            "configuration_ordinal": self.configuration_ordinal,
            "corruption_type": self.corruption_type,
            "distribution": self.distribution.value,
            "proposal_id": self.proposal_id,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "severity_id": self.severity_id,
        }

    @property
    def content_digest(self) -> str:
        """Return the exact candidate identity."""

        return _content_digest(self.identity_mapping())


@dataclass(frozen=True, slots=True, eq=False)
class CandidateGroupV1:
    """One anchor state, fixed continuation, and exactly eight candidates."""

    anchor_id: str
    trajectory: SourceTrajectoryIdentityV1
    state_content_digest: str
    verifier_state_content_digest: str
    continuation_identity: str
    source_action_prefix_digest: str
    state_vector: NDArray[Any]
    continuation_actions: NDArray[Any]
    candidates: tuple[CandidateRefV1, ...]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze arrays and enforce exact candidate cardinality and identities."""

        _text(self.anchor_id, "CandidateGroupV1.anchor_id")
        if not isinstance(self.trajectory, SourceTrajectoryIdentityV1):
            _fail("CandidateGroupV1.trajectory", "expected SourceTrajectoryIdentityV1")
        for name in (
            "state_content_digest",
            "verifier_state_content_digest",
            "continuation_identity",
            "source_action_prefix_digest",
        ):
            _digest(getattr(self, name), f"CandidateGroupV1.{name}")
        state = _freeze_array(
            self.state_vector,
            shape=(STATE_DIMENSION,),
            context="CandidateGroupV1.state_vector",
            floating=True,
        )
        continuation = _freeze_array(
            self.continuation_actions,
            shape=None,
            context="CandidateGroupV1.continuation_actions",
            floating=True,
        )
        if continuation.ndim != 2 or continuation.shape[1:] != (ACTION_DIMENSION,):
            _fail(
                "CandidateGroupV1.continuation_actions",
                "expected shape [N, 8]",
            )
        candidates = tuple(self.candidates)
        if len(candidates) != CANDIDATE_COUNT or any(
            not isinstance(item, CandidateRefV1) for item in candidates
        ):
            _fail("CandidateGroupV1.candidates", "expected exactly 8 candidates")
        proposal_ids = tuple(item.proposal_id for item in candidates)
        ordinals = tuple(item.configuration_ordinal for item in candidates)
        if len(set(proposal_ids)) != CANDIDATE_COUNT:
            _fail("CandidateGroupV1.candidates", "proposal IDs must be unique")
        if set(ordinals) != set(range(CANDIDATE_COUNT)):
            _fail("CandidateGroupV1.candidates", "ordinals must be exactly 0..7")
        if tuple(sorted(ordinals)) != ordinals:
            _fail("CandidateGroupV1.candidates", "candidates must be ordinal ordered")
        if (
            sum(
                item.distribution is CandidateDistribution.ID_LIKE
                for item in candidates
            )
            != 4
        ):
            _fail(
                "CandidateGroupV1.candidates",
                "expected exactly four ID-like candidates",
            )
        if (
            sum(
                item.distribution is CandidateDistribution.SHIFTED
                for item in candidates
            )
            != 4
        ):
            _fail(
                "CandidateGroupV1.candidates",
                "expected exactly four shifted candidates",
            )
        for item in candidates:
            if (
                array_content_digest(item.action_chunk)
                == self.source_action_prefix_digest
            ):
                _fail(
                    "CandidateGroupV1.candidates",
                    "source-equivalent candidate is prohibited",
                )
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("CandidateGroupV1.schema_version", "unsupported version")
        object.__setattr__(self, "state_vector", state)
        object.__setattr__(self, "continuation_actions", continuation)
        object.__setattr__(self, "candidates", candidates)

    @property
    def source_trajectory_id(self) -> str:
        """Return the source trajectory identifier without exposing it to inference."""

        return self.trajectory.source_trajectory_id

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        """Return the complete ordered candidate inventory."""

        return tuple(item.proposal_id for item in self.candidates)

    def identity_mapping(self) -> dict[str, object]:
        """Return path-independent group content without raw vectors or actions."""

        return {
            "anchor_id": self.anchor_id,
            "candidate_content_digests": [
                item.content_digest for item in self.candidates
            ],
            "continuation_actions_digest": array_content_digest(
                self.continuation_actions
            ),
            "continuation_identity": self.continuation_identity,
            "schema_version": self.schema_version,
            "source_action_prefix_digest": self.source_action_prefix_digest,
            "state_content_digest": self.state_content_digest,
            "state_vector_digest": array_content_digest(self.state_vector),
            "trajectory": self.trajectory.as_mapping(),
            "verifier_state_content_digest": self.verifier_state_content_digest,
        }

    @property
    def group_id(self) -> str:
        """Return the deterministic candidate-group identifier."""

        digest = hashlib.sha256(
            canonical_json_bytes(self.identity_mapping())
        ).hexdigest()
        return f"{_GROUP_ID_PREFIX}{digest}"


@dataclass(frozen=True, slots=True, eq=False)
class CandidatePoolV1:
    """A content-bound set of identical candidate groups for every selector."""

    source_set_digest: str
    candidate_pool_configuration_digest: str
    action_contract_digest: str
    exclusion_inventory_digest: str
    groups: tuple[CandidateGroupV1, ...]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze groups and enforce unique anchors, groups, and proposal IDs."""

        for name in (
            "source_set_digest",
            "candidate_pool_configuration_digest",
            "action_contract_digest",
            "exclusion_inventory_digest",
        ):
            _digest(getattr(self, name), f"CandidatePoolV1.{name}")
        groups = tuple(self.groups)
        if not groups or any(not isinstance(item, CandidateGroupV1) for item in groups):
            _fail(
                "CandidatePoolV1.groups",
                "expected non-empty CandidateGroupV1 inventory",
            )
        for values, context in (
            ((item.anchor_id for item in groups), "anchor IDs"),
            ((item.group_id for item in groups), "group IDs"),
            (
                (proposal for item in groups for proposal in item.proposal_ids),
                "proposal IDs",
            ),
        ):
            materialized = tuple(values)
            if len(materialized) != len(set(materialized)):
                _fail("CandidatePoolV1.groups", f"duplicate {context}")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("CandidatePoolV1.schema_version", "unsupported version")
        object.__setattr__(self, "groups", groups)

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        """Return every proposal ID in deterministic pool order."""

        return tuple(
            proposal for group in self.groups for proposal in group.proposal_ids
        )

    def identity_mapping(self) -> dict[str, object]:
        """Return the complete path-independent pool identity payload."""

        return {
            "action_contract_digest": self.action_contract_digest,
            "candidate_pool_configuration_digest": (
                self.candidate_pool_configuration_digest
            ),
            "exclusion_inventory_digest": self.exclusion_inventory_digest,
            "group_ids": [item.group_id for item in self.groups],
            "schema_version": self.schema_version,
            "source_set_digest": self.source_set_digest,
        }

    @property
    def content_digest(self) -> str:
        """Return the deterministic candidate-pool identity."""

        return _content_digest(self.identity_mapping())


@dataclass(frozen=True, slots=True)
class SelectorDecisionV1:
    """One blind selector decision for one exact candidate group."""

    selector_id: str
    group_id: str
    selected_proposal_id: str | None
    ranking: tuple[str, ...]
    predicted_failure_probabilities: Mapping[str, float]
    abstained: bool
    abstention_policy_id: str | None = None
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate exact ranking membership and abstention semantics."""

        _text(self.selector_id, "SelectorDecisionV1.selector_id")
        _text(self.group_id, "SelectorDecisionV1.group_id")
        ranking = tuple(self.ranking)
        if len(ranking) != CANDIDATE_COUNT or len(set(ranking)) != CANDIDATE_COUNT:
            _fail("SelectorDecisionV1.ranking", "expected 8 unique proposal IDs")
        for value in ranking:
            _text(value, "SelectorDecisionV1.ranking")
        if type(self.abstained) is not bool:
            _fail("SelectorDecisionV1.abstained", "expected a boolean")
        if self.abstained:
            if self.selected_proposal_id is not None:
                _fail("SelectorDecisionV1", "abstention cannot select a proposal")
            if self.abstention_policy_id is None:
                _fail("SelectorDecisionV1", "abstention requires a frozen policy ID")
        else:
            if self.selected_proposal_id != ranking[0]:
                _fail("SelectorDecisionV1", "selected proposal must be ranking[0]")
        if self.abstention_policy_id is not None:
            _text(self.abstention_policy_id, "SelectorDecisionV1.abstention_policy_id")
        probabilities = dict(self.predicted_failure_probabilities)
        if probabilities and set(probabilities) != set(ranking):
            _fail(
                "SelectorDecisionV1.predicted_failure_probabilities",
                "must be empty or cover the exact ranking inventory",
            )
        for proposal_id, probability in probabilities.items():
            _text(proposal_id, "SelectorDecisionV1.predicted_failure_probabilities.key")
            number = _finite(
                probability, "SelectorDecisionV1.predicted_failure_probabilities"
            )
            if not 0.0 <= number <= 1.0:
                _fail(
                    "SelectorDecisionV1.predicted_failure_probabilities",
                    "expected values in [0, 1]",
                )
            probabilities[proposal_id] = number
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("SelectorDecisionV1.schema_version", "unsupported version")
        object.__setattr__(self, "ranking", ranking)
        object.__setattr__(
            self,
            "predicted_failure_probabilities",
            MappingProxyType(dict(sorted(probabilities.items()))),
        )

    def as_mapping(self) -> dict[str, object]:
        """Return safe manifest content with no model inputs or outcomes."""

        return {
            "abstained": self.abstained,
            "abstention_policy_id": self.abstention_policy_id,
            "group_id": self.group_id,
            "predicted_failure_probabilities": dict(
                self.predicted_failure_probabilities
            ),
            "ranking": list(self.ranking),
            "schema_version": self.schema_version,
            "selected_proposal_id": self.selected_proposal_id,
            "selector_id": self.selector_id,
        }


@dataclass(frozen=True, slots=True)
class BlindSelectionManifestV1:
    """Immutable Stage-A selections created without outcome capabilities."""

    source_set_digest: str
    candidate_pool_digest: str
    candidate_pool_configuration_digest: str
    selector_configuration_digest: str
    verifier_bundle_digests: tuple[tuple[str, str], ...]
    selection_timestamp_utc: str
    input_allowlist_digest: str
    selections: tuple[SelectorDecisionV1, ...]
    outcomes_available_during_selection: bool = False
    outcome_input_paths: tuple[str, ...] = ()
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate blind capability isolation and exact selection inventories."""

        for name in (
            "source_set_digest",
            "candidate_pool_digest",
            "candidate_pool_configuration_digest",
            "selector_configuration_digest",
            "input_allowlist_digest",
        ):
            _digest(getattr(self, name), f"BlindSelectionManifestV1.{name}")
        _text(
            self.selection_timestamp_utc,
            "BlindSelectionManifestV1.selection_timestamp_utc",
        )
        bundles = tuple(self.verifier_bundle_digests)
        if not bundles or len({name for name, _ in bundles}) != len(bundles):
            _fail(
                "BlindSelectionManifestV1.verifier_bundle_digests",
                "keys must be unique",
            )
        for name, digest in bundles:
            _text(name, "BlindSelectionManifestV1.verifier_bundle_digests.key")
            _digest(digest, "BlindSelectionManifestV1.verifier_bundle_digests.value")
        if tuple(sorted(bundles)) != bundles:
            _fail(
                "BlindSelectionManifestV1.verifier_bundle_digests", "must be key sorted"
            )
        selections = tuple(self.selections)
        if not selections or any(
            not isinstance(item, SelectorDecisionV1) for item in selections
        ):
            _fail("BlindSelectionManifestV1.selections", "must not be empty")
        keys = tuple((item.selector_id, item.group_id) for item in selections)
        if len(keys) != len(set(keys)):
            _fail(
                "BlindSelectionManifestV1.selections",
                "duplicate selector/group decision",
            )
        if self.outcomes_available_during_selection is not False:
            _fail(
                "BlindSelectionManifestV1.outcomes_available_during_selection",
                "must be false",
            )
        if tuple(self.outcome_input_paths):
            _fail("BlindSelectionManifestV1.outcome_input_paths", "must be empty")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("BlindSelectionManifestV1.schema_version", "unsupported version")
        object.__setattr__(self, "verifier_bundle_digests", bundles)
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "outcome_input_paths", ())

    def _semantic_mapping(self) -> dict[str, object]:
        return {
            "candidate_pool_configuration_digest": (
                self.candidate_pool_configuration_digest
            ),
            "candidate_pool_digest": self.candidate_pool_digest,
            "input_allowlist_digest": self.input_allowlist_digest,
            "outcome_input_paths": [],
            "outcomes_available_during_selection": False,
            "schema_version": self.schema_version,
            "selections": [item.as_mapping() for item in self.selections],
            "selector_configuration_digest": self.selector_configuration_digest,
            "source_set_digest": self.source_set_digest,
            "verifier_bundle_digests": [
                list(item) for item in self.verifier_bundle_digests
            ],
        }

    @property
    def content_digest(self) -> str:
        """Return semantic Stage-A identity, deliberately excluding timestamp."""

        return _content_digest(self._semantic_mapping())

    def as_mapping(self) -> dict[str, object]:
        """Return the strict manifest including non-semantic audit timestamp."""

        return {
            **self._semantic_mapping(),
            "content_digest": self.content_digest,
            "selection_timestamp_utc": self.selection_timestamp_utc,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> BlindSelectionManifestV1:
        """Load exact manifest fields and reject content drift or outcomes."""

        expected = {
            "candidate_pool_configuration_digest",
            "candidate_pool_digest",
            "content_digest",
            "input_allowlist_digest",
            "outcome_input_paths",
            "outcomes_available_during_selection",
            "schema_version",
            "selection_timestamp_utc",
            "selections",
            "selector_configuration_digest",
            "source_set_digest",
            "verifier_bundle_digests",
        }
        if set(value) != expected:
            _fail("BlindSelectionManifestV1", "unexpected or missing fields")
        raw_bundles = value["verifier_bundle_digests"]
        raw_selections = value["selections"]
        if not isinstance(raw_bundles, list) or not isinstance(raw_selections, list):
            _fail("BlindSelectionManifestV1", "expected list inventories")
        bundles: list[tuple[str, str]] = []
        for item in raw_bundles:
            if not isinstance(item, list) or len(item) != 2:
                _fail(
                    "BlindSelectionManifestV1.verifier_bundle_digests", "invalid entry"
                )
            bundles.append((cast(str, item[0]), cast(str, item[1])))
        decisions: list[SelectorDecisionV1] = []
        for raw in raw_selections:
            if not isinstance(raw, Mapping):
                _fail("BlindSelectionManifestV1.selections", "invalid entry")
            exact = {
                "abstained",
                "abstention_policy_id",
                "group_id",
                "predicted_failure_probabilities",
                "ranking",
                "schema_version",
                "selected_proposal_id",
                "selector_id",
            }
            if (
                set(raw) != exact
                or not isinstance(raw["ranking"], list)
                or not isinstance(raw["predicted_failure_probabilities"], Mapping)
            ):
                _fail("BlindSelectionManifestV1.selections", "unexpected fields")
            decisions.append(
                SelectorDecisionV1(
                    selector_id=cast(str, raw["selector_id"]),
                    group_id=cast(str, raw["group_id"]),
                    selected_proposal_id=cast(str | None, raw["selected_proposal_id"]),
                    ranking=tuple(cast(Sequence[str], raw["ranking"])),
                    predicted_failure_probabilities=cast(
                        Mapping[str, float], raw["predicted_failure_probabilities"]
                    ),
                    abstained=cast(bool, raw["abstained"]),
                    abstention_policy_id=cast(str | None, raw["abstention_policy_id"]),
                    schema_version=cast(str, raw["schema_version"]),
                )
            )
        outcome_paths = value["outcome_input_paths"]
        if not isinstance(outcome_paths, list):
            _fail("BlindSelectionManifestV1.outcome_input_paths", "expected list")
        result = cls(
            source_set_digest=cast(str, value["source_set_digest"]),
            candidate_pool_digest=cast(str, value["candidate_pool_digest"]),
            candidate_pool_configuration_digest=cast(
                str, value["candidate_pool_configuration_digest"]
            ),
            selector_configuration_digest=cast(
                str, value["selector_configuration_digest"]
            ),
            verifier_bundle_digests=tuple(bundles),
            selection_timestamp_utc=cast(str, value["selection_timestamp_utc"]),
            input_allowlist_digest=cast(str, value["input_allowlist_digest"]),
            selections=tuple(decisions),
            outcomes_available_during_selection=cast(
                bool, value["outcomes_available_during_selection"]
            ),
            outcome_input_paths=tuple(cast(Sequence[str], outcome_paths)),
            schema_version=cast(str, value["schema_version"]),
        )
        if value["content_digest"] != result.content_digest:
            _fail("BlindSelectionManifestV1.content_digest", "content changed")
        return result


__all__ = [
    "ACTION_DIMENSION",
    "ACTION_HORIZON",
    "CANDIDATE_COUNT",
    "SELECTION_SCHEMA_VERSION",
    "STATE_DIMENSION",
    "BlindSelectionManifestV1",
    "CandidateDistribution",
    "CandidateGroupV1",
    "CandidatePoolV1",
    "CandidateRefV1",
    "SelectionModelError",
    "SelectorDecisionV1",
    "SourceExclusionInventoryV1",
    "SourceTrajectoryIdentityV1",
    "array_content_digest",
]
