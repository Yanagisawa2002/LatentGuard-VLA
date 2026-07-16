"""Outcome-isolated Stage-A selection and immutable Stage-B/C result binding."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.blind_input import (
    BlindCandidateGroupV1,
    BlindCandidatePoolV1,
)
from latentguard.selection.manifest import (
    BLIND_SELECTION_MANIFEST_FILENAME,
    BlindSelectionManifestEnvelopeV1,
    initialize_blind_selection_output_root,
    load_blind_selection_manifest,
    save_blind_selection_manifest,
)
from latentguard.selection.metrics import CandidateOutcomeV1
from latentguard.selection.models import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    CANDIDATE_COUNT,
    SELECTION_SCHEMA_VERSION,
    STATE_DIMENSION,
    BlindSelectionManifestV1,
    CandidatePoolV1,
    SelectionModelError,
    SelectorDecisionV1,
)

BLIND_SELECTOR_INPUT_FIELDS = (
    "action_chunks",
    "action_masks",
    "candidate_ids",
    "state_vectors",
)
BLIND_SELECTOR_INPUT_ALLOWLIST_SEMANTIC = (
    "versioned_blind_pool_state_action_mask_and_opaque_identity_only_v1"
)
BLIND_STAGE_A_GROUP_FIELDS = (
    "action_chunks",
    "action_masks",
    "candidate_ids",
    "group_id",
    "state_vector",
)
BLIND_STAGE_A_POOL_BINDING_FIELDS = (
    "action_contract_digest",
    "candidate_pool_configuration_digest",
    "full_candidate_pool_digest",
    "source_set_digest",
)
EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS = (
    "action_only_mlp",
    "state_action_mlp",
    "temporal_state_action_verifier",
)
EXPECTED_STAGE_A_SELECTOR_IDS = (
    "deterministic_random_v1",
    "frozen_m3b_action_magnitude_v1",
    "action_only_ensemble_v1",
    "state_action_mlp_ensemble_v1",
    "temporal_ensemble_v1",
    "temporal_ensemble_abstention_maximum_validation_balanced_accuracy_v1",
    "temporal_ensemble_abstention_target_validation_failure_recall_v1",
    "temporal_ensemble_abstention_target_validation_coverage_90_v1",
    "temporal_ensemble_abstention_target_validation_coverage_80_v1",
    "temporal_ensemble_abstention_target_validation_coverage_70_v1",
    "temporal_ensemble_abstention_target_validation_coverage_50_v1",
)
BOUND_SELECTION_RESULT_SCHEMA_VERSION = "1.0"
BOUND_SELECTION_RESULT_FILENAME = "bound-selection-result.json"
MAX_BOUND_SELECTION_RESULT_BYTES = 32 * 1024 * 1024
ORACLE_SELECTOR_ID = "oracle_analysis_only_v1"
FAKE_BLIND_SMOKE_SCHEMA_VERSION = "1.0"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,9})?Z$"
)


class BlindProtocolError(ValueError):
    """Raised when blind isolation or post-selection binding is violated."""


def _fail(context: str, reason: str) -> NoReturn:
    raise BlindProtocolError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")
    return value


def _content_digest(value: object, *, context: str) -> str:
    try:
        encoded = canonical_json_bytes(value, context=context)
    except ReplayValidationError as exc:
        raise BlindProtocolError(str(exc)) from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _utc_timestamp(value: object, context: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        _fail(context, "expected a canonical RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise BlindProtocolError(f"{context}: invalid calendar timestamp") from exc
    if parsed.tzinfo != UTC:
        _fail(context, "expected UTC")
    return parsed


def blind_selector_input_allowlist_digest() -> str:
    """Return the frozen identity of fields available to every selector."""

    return _content_digest(
        {
            "allowed_fields": list(BLIND_SELECTOR_INPUT_FIELDS),
            "blinded_group_fields": list(BLIND_STAGE_A_GROUP_FIELDS),
            "blinded_pool_binding_fields": list(BLIND_STAGE_A_POOL_BINDING_FIELDS),
            "schema_version": SELECTION_SCHEMA_VERSION,
            "semantic": BLIND_SELECTOR_INPUT_ALLOWLIST_SEMANTIC,
        },
        context="BlindSelectorInputAllowlistV1",
    )


@dataclass(frozen=True, slots=True, eq=False)
class BlindSelectorInputV1:
    """Detached selector input containing no provenance, metadata, or outcomes."""

    candidate_ids: tuple[str, ...]
    state_vectors: NDArray[Any]
    action_chunks: NDArray[Any]
    action_masks: NDArray[Any]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Enforce the exact eight-candidate deployment input contract."""

        ids = tuple(self.candidate_ids)
        if (
            len(ids) != CANDIDATE_COUNT
            or len(set(ids)) != CANDIDATE_COUNT
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in ids
            )
        ):
            _fail("BlindSelectorInputV1.candidate_ids", "expected 8 unique IDs")
        contracts = (
            (
                self.state_vectors,
                (CANDIDATE_COUNT, STATE_DIMENSION),
                "state_vectors",
                True,
            ),
            (
                self.action_chunks,
                (CANDIDATE_COUNT, ACTION_HORIZON, ACTION_DIMENSION),
                "action_chunks",
                True,
            ),
            (
                self.action_masks,
                (CANDIDATE_COUNT, ACTION_HORIZON),
                "action_masks",
                False,
            ),
        )
        for value, shape, name, floating in contracts:
            if not isinstance(value, np.ndarray) or value.shape != shape:
                _fail(f"BlindSelectorInputV1.{name}", f"expected array shape {shape}")
            if floating:
                if value.dtype.hasobject or not np.issubdtype(value.dtype, np.floating):
                    _fail(f"BlindSelectorInputV1.{name}", "expected floating dtype")
                if not bool(np.all(np.isfinite(value))):
                    _fail(f"BlindSelectorInputV1.{name}", "values must be finite")
            elif value.dtype != np.dtype(np.bool_):
                _fail(f"BlindSelectorInputV1.{name}", "expected bool dtype")
        if not bool(np.all(self.action_masks)):
            _fail(
                "BlindSelectorInputV1.action_masks",
                "all fixed 16-step candidates must be complete",
            )
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("BlindSelectorInputV1.schema_version", "unsupported version")
        object.__setattr__(self, "candidate_ids", ids)
        for name in ("state_vectors", "action_chunks", "action_masks"):
            value = np.array(getattr(self, name), copy=True, order="C", subok=False)
            frozen = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(
                value.shape
            )
            object.__setattr__(self, name, frozen)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> BlindSelectorInputV1:
        """Accept only the four allowlisted fields and reject metadata channels."""

        if set(value) != set(BLIND_SELECTOR_INPUT_FIELDS):
            _fail(
                "BlindSelectorInputV1",
                "input fields differ from the frozen allowlist",
            )
        raw_ids = value["candidate_ids"]
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            _fail("BlindSelectorInputV1.candidate_ids", "expected sequence")
        return cls(
            candidate_ids=tuple(cast(Sequence[str], raw_ids)),
            state_vectors=cast(NDArray[Any], value["state_vectors"]),
            action_chunks=cast(NDArray[Any], value["action_chunks"]),
            action_masks=cast(NDArray[Any], value["action_masks"]),
        )

    @classmethod
    def from_candidate_group(cls, group: object) -> BlindSelectorInputV1:
        """Project a candidate group onto the exact selector allowlist."""

        if isinstance(group, BlindCandidateGroupV1):
            return cls(
                candidate_ids=group.candidate_ids,
                state_vectors=np.repeat(
                    group.state_vector[None, :], CANDIDATE_COUNT, axis=0
                ),
                action_chunks=group.action_chunks,
                action_masks=group.action_masks,
            )

        try:
            candidates = tuple(cast(Any, group).candidates)
            state = cast(Any, group).state_vector
        except (AttributeError, TypeError) as exc:
            raise BlindProtocolError(
                "candidate group: missing fixed selection fields"
            ) from exc
        if len(candidates) != CANDIDATE_COUNT or not isinstance(state, np.ndarray):
            _fail("candidate group", "invalid candidate inventory or state")
        return cls(
            candidate_ids=tuple(item.proposal_id for item in candidates),
            state_vectors=np.repeat(state[None, :], CANDIDATE_COUNT, axis=0),
            action_chunks=np.stack([item.action_chunk for item in candidates], axis=0),
            action_masks=np.stack([item.action_mask for item in candidates], axis=0),
        )

    def as_mapping(self) -> Mapping[str, object]:
        """Return only the frozen fields available to selector implementations."""

        return MappingProxyType(
            {
                "action_chunks": self.action_chunks,
                "action_masks": self.action_masks,
                "candidate_ids": self.candidate_ids,
                "state_vectors": self.state_vectors,
            }
        )


def _selection_snapshot_digest(manifest: BlindSelectionManifestV1) -> str:
    return _content_digest(
        {
            "schema_version": manifest.schema_version,
            "selections": [item.as_mapping() for item in manifest.selections],
        },
        context="BlindSelectionSnapshotV1",
    )


def _validate_selection_manifest_inventory(
    manifest: BlindSelectionManifestV1,
    *,
    source_set_digest: str,
    candidate_pool_digest: str,
    candidate_pool_configuration_digest: str,
    group_candidates: Mapping[str, tuple[str, ...]],
) -> None:
    if not isinstance(manifest, BlindSelectionManifestV1):
        _fail("manifest", "expected BlindSelectionManifestV1")
    if (
        manifest.source_set_digest != source_set_digest
        or manifest.candidate_pool_digest != candidate_pool_digest
        or manifest.candidate_pool_configuration_digest
        != candidate_pool_configuration_digest
    ):
        _fail("selection manifest", "source or candidate-pool identity changed")
    if manifest.input_allowlist_digest != blind_selector_input_allowlist_digest():
        _fail("selection manifest", "selector input allowlist changed")
    if (
        manifest.outcomes_available_during_selection is not False
        or manifest.outcome_input_paths != ()
    ):
        _fail("selection manifest", "outcome capability appeared in Stage A")
    bundle_keys = tuple(name for name, _ in manifest.verifier_bundle_digests)
    if bundle_keys != EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS:
        _fail(
            "selection manifest",
            "verifier bundle inventory must be the exact three frozen architectures",
        )
    group_proposals = dict(group_candidates)
    selector_groups: defaultdict[str, set[str]] = defaultdict(set)
    ordered_keys: list[tuple[str, str]] = []
    for decision in manifest.selections:
        proposals = group_proposals.get(decision.group_id)
        if proposals is None:
            _fail("selection manifest", "decision references unknown candidate group")
        if set(decision.ranking) != set(proposals):
            _fail("selection manifest", "ranking differs from exact group inventory")
        if not decision.abstained and decision.selected_proposal_id not in proposals:
            _fail("selection manifest", "selected proposal is outside its group")
        selector_groups[decision.selector_id].add(decision.group_id)
        ordered_keys.append((decision.selector_id, decision.group_id))
    expected_groups = set(group_proposals)
    if tuple(sorted(selector_groups)) != tuple(sorted(EXPECTED_STAGE_A_SELECTOR_IDS)):
        _fail(
            "selection manifest",
            "selector inventory must contain the exact eleven Stage-A selectors",
        )
    if any(groups != expected_groups for groups in selector_groups.values()):
        _fail(
            "selection manifest",
            "selector inventory must cover every pool group exactly once",
        )
    expected_decision_count = len(EXPECTED_STAGE_A_SELECTOR_IDS) * len(expected_groups)
    if len(ordered_keys) != expected_decision_count or len(set(ordered_keys)) != len(
        ordered_keys
    ):
        _fail(
            "selection manifest",
            "selector/group decisions must form one exact duplicate-free product",
        )
    if ordered_keys != sorted(ordered_keys):
        _fail("selection manifest", "decisions must be selector/group sorted")


def validate_selection_manifest_against_pool(
    manifest: BlindSelectionManifestV1,
    candidate_pool: CandidatePoolV1,
) -> None:
    """Require the exact Stage-A inventory to bind the complete full pool."""

    if not isinstance(candidate_pool, CandidatePoolV1):
        _fail("candidate_pool", "expected CandidatePoolV1")
    _validate_selection_manifest_inventory(
        manifest,
        source_set_digest=candidate_pool.source_set_digest,
        candidate_pool_digest=candidate_pool.content_digest,
        candidate_pool_configuration_digest=(
            candidate_pool.candidate_pool_configuration_digest
        ),
        group_candidates={
            group.group_id: group.proposal_ids for group in candidate_pool.groups
        },
    )


def validate_selection_manifest_against_blind_pool(
    manifest: BlindSelectionManifestV1,
    candidate_pool: BlindCandidatePoolV1,
) -> None:
    """Require the exact Stage-A inventory to bind only a blinded input pool."""

    if not isinstance(candidate_pool, BlindCandidatePoolV1):
        _fail("candidate_pool", "expected BlindCandidatePoolV1")
    _validate_selection_manifest_inventory(
        manifest,
        source_set_digest=candidate_pool.source_set_digest,
        candidate_pool_digest=candidate_pool.full_candidate_pool_digest,
        candidate_pool_configuration_digest=(
            candidate_pool.candidate_pool_configuration_digest
        ),
        group_candidates={
            group.group_id: group.candidate_ids for group in candidate_pool.groups
        },
    )


def finalize_blind_selection(
    candidate_pool: CandidatePoolV1,
    selections: Sequence[SelectorDecisionV1],
    *,
    selector_configuration_digest: str,
    verifier_bundle_digests: Mapping[str, str] | Sequence[tuple[str, str]],
    selection_timestamp_utc: str,
) -> BlindSelectionManifestEnvelopeV1:
    """Finalize Stage A without accepting any evidence, replay, or outcome path."""

    if not isinstance(candidate_pool, CandidatePoolV1):
        _fail("candidate_pool", "expected CandidatePoolV1")
    _digest(selector_configuration_digest, "selector_configuration_digest")
    _utc_timestamp(selection_timestamp_utc, "selection_timestamp_utc")
    if isinstance(verifier_bundle_digests, Mapping):
        bundle_items = tuple(verifier_bundle_digests.items())
    else:
        bundle_items = tuple(verifier_bundle_digests)
    if not bundle_items:
        _fail("verifier_bundle_digests", "must not be empty")
    decisions = tuple(selections)
    if not decisions or any(
        not isinstance(item, SelectorDecisionV1) for item in decisions
    ):
        _fail("selections", "expected non-empty SelectorDecisionV1 sequence")
    decisions = tuple(
        sorted(decisions, key=lambda item: (item.selector_id, item.group_id))
    )
    try:
        manifest = BlindSelectionManifestV1(
            source_set_digest=candidate_pool.source_set_digest,
            candidate_pool_digest=candidate_pool.content_digest,
            candidate_pool_configuration_digest=(
                candidate_pool.candidate_pool_configuration_digest
            ),
            selector_configuration_digest=selector_configuration_digest,
            verifier_bundle_digests=tuple(sorted(bundle_items)),
            selection_timestamp_utc=selection_timestamp_utc,
            input_allowlist_digest=blind_selector_input_allowlist_digest(),
            selections=decisions,
            outcomes_available_during_selection=False,
            outcome_input_paths=(),
        )
    except (SelectionModelError, TypeError, ValueError) as exc:
        raise BlindProtocolError(f"selection manifest: {exc}") from exc
    validate_selection_manifest_against_pool(manifest, candidate_pool)
    return BlindSelectionManifestEnvelopeV1.create(manifest)


def finalize_blind_selection_from_blind_pool(
    candidate_pool: BlindCandidatePoolV1,
    selections: Sequence[SelectorDecisionV1],
    *,
    selector_configuration_digest: str,
    verifier_bundle_digests: Mapping[str, str] | Sequence[tuple[str, str]],
    selection_timestamp_utc: str,
) -> BlindSelectionManifestEnvelopeV1:
    """Finalize Stage A using only the serialized blinded candidate pool."""

    if not isinstance(candidate_pool, BlindCandidatePoolV1):
        _fail("candidate_pool", "expected BlindCandidatePoolV1")
    _digest(selector_configuration_digest, "selector_configuration_digest")
    _utc_timestamp(selection_timestamp_utc, "selection_timestamp_utc")
    if isinstance(verifier_bundle_digests, Mapping):
        bundle_items = tuple(verifier_bundle_digests.items())
    else:
        bundle_items = tuple(verifier_bundle_digests)
    if not bundle_items:
        _fail("verifier_bundle_digests", "must not be empty")
    decisions = tuple(selections)
    if not decisions or any(
        not isinstance(item, SelectorDecisionV1) for item in decisions
    ):
        _fail("selections", "expected non-empty SelectorDecisionV1 sequence")
    decisions = tuple(
        sorted(decisions, key=lambda item: (item.selector_id, item.group_id))
    )
    try:
        manifest = BlindSelectionManifestV1(
            source_set_digest=candidate_pool.source_set_digest,
            candidate_pool_digest=candidate_pool.full_candidate_pool_digest,
            candidate_pool_configuration_digest=(
                candidate_pool.candidate_pool_configuration_digest
            ),
            selector_configuration_digest=selector_configuration_digest,
            verifier_bundle_digests=tuple(sorted(bundle_items)),
            selection_timestamp_utc=selection_timestamp_utc,
            input_allowlist_digest=blind_selector_input_allowlist_digest(),
            selections=decisions,
            outcomes_available_during_selection=False,
            outcome_input_paths=(),
        )
    except (SelectionModelError, TypeError, ValueError) as exc:
        raise BlindProtocolError(f"selection manifest: {exc}") from exc
    validate_selection_manifest_against_blind_pool(manifest, candidate_pool)
    return BlindSelectionManifestEnvelopeV1.create(manifest)


def run_blind_selection_stage_a(
    candidate_pool: CandidatePoolV1,
    selections: Sequence[SelectorDecisionV1],
    *,
    selector_configuration_digest: str,
    verifier_bundle_digests: Mapping[str, str] | Sequence[tuple[str, str]],
    selection_timestamp_utc: str,
    output_root: Path,
) -> tuple[Path, BlindSelectionManifestEnvelopeV1]:
    """Create one manifest in a previously empty Stage-A-only output root."""

    root = initialize_blind_selection_output_root(output_root)
    envelope = finalize_blind_selection(
        candidate_pool,
        selections,
        selector_configuration_digest=selector_configuration_digest,
        verifier_bundle_digests=verifier_bundle_digests,
        selection_timestamp_utc=selection_timestamp_utc,
    )
    path = save_blind_selection_manifest(
        root / BLIND_SELECTION_MANIFEST_FILENAME,
        envelope,
    )
    return path, envelope


def run_blind_selection_stage_a_from_blind_pool(
    candidate_pool: BlindCandidatePoolV1,
    selections: Sequence[SelectorDecisionV1],
    *,
    selector_configuration_digest: str,
    verifier_bundle_digests: Mapping[str, str] | Sequence[tuple[str, str]],
    selection_timestamp_utc: str,
    output_root: Path,
) -> tuple[Path, BlindSelectionManifestEnvelopeV1]:
    """Persist Stage A from blinded inputs into a previously empty output root."""

    root = initialize_blind_selection_output_root(output_root)
    envelope = finalize_blind_selection_from_blind_pool(
        candidate_pool,
        selections,
        selector_configuration_digest=selector_configuration_digest,
        verifier_bundle_digests=verifier_bundle_digests,
        selection_timestamp_utc=selection_timestamp_utc,
    )
    path = save_blind_selection_manifest(
        root / BLIND_SELECTION_MANIFEST_FILENAME,
        envelope,
    )
    return path, envelope


def _outcome_mapping(value: CandidateOutcomeV1) -> dict[str, object]:
    return {
        "distribution": value.distribution,
        "evidence_id": value.evidence_id,
        "execution_error": value.execution_error,
        "group_id": value.group_id,
        "label_strength": value.label_strength,
        "proposal_id": value.proposal_id,
        "simulator_replay_verified": value.simulator_replay_verified,
        "source_trajectory_id": value.source_trajectory_id,
        "state_component_count": value.state_component_count,
        "status": value.status,
        "success": value.success,
        "unsafe": value.unsafe,
    }


def compute_full_pool_outcome_digest(
    outcomes: Sequence[CandidateOutcomeV1],
) -> str:
    """Bind the complete ordered candidate-outcome inventory exactly."""

    values = tuple(outcomes)
    if not values or any(not isinstance(item, CandidateOutcomeV1) for item in values):
        _fail("full pool outcomes", "expected non-empty CandidateOutcomeV1 sequence")
    return _content_digest(
        {
            "outcomes": [_outcome_mapping(item) for item in values],
            "schema_version": SELECTION_SCHEMA_VERSION,
        },
        context="CompleteCandidatePoolOutcomesV1",
    )


def _validate_complete_outcomes(
    candidate_pool: CandidatePoolV1,
    outcomes: Sequence[CandidateOutcomeV1],
) -> tuple[CandidateOutcomeV1, ...]:
    values = tuple(outcomes)
    if len(values) != len(candidate_pool.proposal_ids):
        _fail("full pool outcomes", "must contain one result per pool proposal")
    by_id: dict[str, CandidateOutcomeV1] = {}
    group_by_proposal = {
        candidate.proposal_id: (group.group_id, group.source_trajectory_id)
        for group in candidate_pool.groups
        for candidate in group.candidates
    }
    distribution_by_proposal = {
        candidate.proposal_id: candidate.distribution.value
        for group in candidate_pool.groups
        for candidate in group.candidates
    }
    evidence_ids: set[str] = set()
    for outcome in values:
        expected = group_by_proposal.get(outcome.proposal_id)
        if expected is None:
            _fail("full pool outcomes", "evidence references an unknown proposal")
        if outcome.proposal_id in by_id or outcome.evidence_id in evidence_ids:
            _fail("full pool outcomes", "duplicate proposal or evidence identity")
        if (
            outcome.group_id != expected[0]
            or outcome.source_trajectory_id != expected[1]
            or outcome.distribution != distribution_by_proposal[outcome.proposal_id]
        ):
            _fail("full pool outcomes", "proposal provenance differs from pool")
        by_id[outcome.proposal_id] = outcome
        evidence_ids.add(outcome.evidence_id)
    if set(by_id) != set(candidate_pool.proposal_ids):
        _fail("full pool outcomes", "outcome inventory differs from pool")
    return tuple(by_id[item] for item in candidate_pool.proposal_ids)


@dataclass(frozen=True, slots=True)
class BoundSelectionResultV1:
    """Final Stage-B/C identities bound to the immutable Stage-A artifact."""

    selection_manifest_digest: str
    selection_manifest_envelope_digest: str
    selection_snapshot_digest: str
    replay_evidence_digest: str
    full_pool_outcome_digest: str
    source_set_digest: str
    candidate_pool_digest: str
    verifier_bundle_digests: tuple[tuple[str, str], ...]
    evidence_proposal_ids: tuple[str, ...]
    full_outcome_proposal_ids: tuple[str, ...]
    simulator_compatibility_identity: str
    git_sha: str
    outcomes_generated_at_utc: str
    schema_version: str = BOUND_SELECTION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate a complete, duplicate-free, content-bound result inventory."""

        for name in (
            "selection_manifest_digest",
            "selection_manifest_envelope_digest",
            "selection_snapshot_digest",
            "replay_evidence_digest",
            "full_pool_outcome_digest",
            "source_set_digest",
            "candidate_pool_digest",
            "simulator_compatibility_identity",
        ):
            _digest(getattr(self, name), f"BoundSelectionResultV1.{name}")
        bundles = tuple(self.verifier_bundle_digests)
        if (
            not bundles
            or bundles != tuple(sorted(bundles))
            or len({name for name, _ in bundles}) != len(bundles)
        ):
            _fail("BoundSelectionResultV1.verifier_bundle_digests", "invalid inventory")
        for name, digest in bundles:
            if not isinstance(name, str) or not name or name != name.strip():
                _fail("BoundSelectionResultV1.verifier_bundle_digests", "invalid key")
            _digest(digest, "BoundSelectionResultV1.verifier_bundle_digests")
        evidence = tuple(self.evidence_proposal_ids)
        outcomes = tuple(self.full_outcome_proposal_ids)
        if (
            not evidence
            or len(evidence) != len(set(evidence))
            or evidence != outcomes
            or any(not isinstance(item, str) or not item for item in evidence)
        ):
            _fail(
                "BoundSelectionResultV1",
                "evidence and outcomes must cover one identical ordered inventory",
            )
        if (
            not isinstance(self.git_sha, str)
            or _GIT_SHA_RE.fullmatch(self.git_sha) is None
        ):
            _fail("BoundSelectionResultV1.git_sha", "expected full 40-hex Git SHA")
        _utc_timestamp(
            self.outcomes_generated_at_utc,
            "BoundSelectionResultV1.outcomes_generated_at_utc",
        )
        if self.schema_version != BOUND_SELECTION_RESULT_SCHEMA_VERSION:
            _fail("BoundSelectionResultV1.schema_version", "unsupported version")
        object.__setattr__(self, "verifier_bundle_digests", bundles)
        object.__setattr__(self, "evidence_proposal_ids", evidence)
        object.__setattr__(self, "full_outcome_proposal_ids", outcomes)

    def _content_mapping(self) -> dict[str, object]:
        return {
            "candidate_pool_digest": self.candidate_pool_digest,
            "evidence_proposal_ids": list(self.evidence_proposal_ids),
            "full_outcome_proposal_ids": list(self.full_outcome_proposal_ids),
            "full_pool_outcome_digest": self.full_pool_outcome_digest,
            "git_sha": self.git_sha,
            "outcomes_generated_at_utc": self.outcomes_generated_at_utc,
            "replay_evidence_digest": self.replay_evidence_digest,
            "schema_version": self.schema_version,
            "selection_manifest_digest": self.selection_manifest_digest,
            "selection_manifest_envelope_digest": (
                self.selection_manifest_envelope_digest
            ),
            "selection_snapshot_digest": self.selection_snapshot_digest,
            "simulator_compatibility_identity": self.simulator_compatibility_identity,
            "source_set_digest": self.source_set_digest,
            "verifier_bundle_digests": [
                list(item) for item in self.verifier_bundle_digests
            ],
        }

    @property
    def content_digest(self) -> str:
        """Return the exact final result identity."""

        return _content_digest(
            self._content_mapping(),
            context="BoundSelectionResultV1",
        )

    def as_mapping(self) -> dict[str, object]:
        """Return strict serialized result fields and digest."""

        return {**self._content_mapping(), "content_digest": self.content_digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> BoundSelectionResultV1:
        """Decode exact fields and reject any result-content drift."""

        expected = {
            "candidate_pool_digest",
            "content_digest",
            "evidence_proposal_ids",
            "full_outcome_proposal_ids",
            "full_pool_outcome_digest",
            "git_sha",
            "outcomes_generated_at_utc",
            "replay_evidence_digest",
            "schema_version",
            "selection_manifest_digest",
            "selection_manifest_envelope_digest",
            "selection_snapshot_digest",
            "simulator_compatibility_identity",
            "source_set_digest",
            "verifier_bundle_digests",
        }
        if set(value) != expected:
            _fail("BoundSelectionResultV1", "unexpected or missing fields")
        raw_bundles = value["verifier_bundle_digests"]
        raw_evidence = value["evidence_proposal_ids"]
        raw_outcomes = value["full_outcome_proposal_ids"]
        if (
            not isinstance(raw_bundles, list)
            or not isinstance(raw_evidence, list)
            or not isinstance(raw_outcomes, list)
        ):
            _fail("BoundSelectionResultV1", "expected list inventories")
        bundles: list[tuple[str, str]] = []
        for item in raw_bundles:
            if not isinstance(item, list) or len(item) != 2:
                _fail("BoundSelectionResultV1.verifier_bundle_digests", "invalid entry")
            bundles.append((cast(str, item[0]), cast(str, item[1])))
        result = cls(
            selection_manifest_digest=cast(str, value["selection_manifest_digest"]),
            selection_manifest_envelope_digest=cast(
                str, value["selection_manifest_envelope_digest"]
            ),
            selection_snapshot_digest=cast(str, value["selection_snapshot_digest"]),
            replay_evidence_digest=cast(str, value["replay_evidence_digest"]),
            full_pool_outcome_digest=cast(str, value["full_pool_outcome_digest"]),
            source_set_digest=cast(str, value["source_set_digest"]),
            candidate_pool_digest=cast(str, value["candidate_pool_digest"]),
            verifier_bundle_digests=tuple(bundles),
            evidence_proposal_ids=tuple(cast(Sequence[str], raw_evidence)),
            full_outcome_proposal_ids=tuple(cast(Sequence[str], raw_outcomes)),
            simulator_compatibility_identity=cast(
                str, value["simulator_compatibility_identity"]
            ),
            git_sha=cast(str, value["git_sha"]),
            outcomes_generated_at_utc=cast(str, value["outcomes_generated_at_utc"]),
            schema_version=cast(str, value["schema_version"]),
        )
        if value["content_digest"] != result.content_digest:
            _fail("BoundSelectionResultV1.content_digest", "result content changed")
        return result


def validate_bound_selection_result(
    result: BoundSelectionResultV1,
    selection_envelope: BlindSelectionManifestEnvelopeV1,
    candidate_pool: CandidatePoolV1,
) -> None:
    """Reject changed Stage-A choices, pool, bundles, or replay inventory."""

    if not isinstance(result, BoundSelectionResultV1):
        _fail("result", "expected BoundSelectionResultV1")
    if not isinstance(selection_envelope, BlindSelectionManifestEnvelopeV1):
        _fail("selection_envelope", "invalid manifest envelope")
    validate_selection_manifest_against_pool(
        selection_envelope.manifest, candidate_pool
    )
    manifest = selection_envelope.manifest
    if (
        result.selection_manifest_digest != manifest.content_digest
        or result.selection_manifest_envelope_digest
        != selection_envelope.manifest_envelope_digest
        or result.selection_snapshot_digest != _selection_snapshot_digest(manifest)
    ):
        _fail("bound result", "selected candidate or ranking changed after Stage A")
    if (
        result.source_set_digest != candidate_pool.source_set_digest
        or result.candidate_pool_digest != candidate_pool.content_digest
        or result.verifier_bundle_digests != manifest.verifier_bundle_digests
    ):
        _fail("bound result", "source, pool, or checkpoint identity changed")
    if (
        result.evidence_proposal_ids != candidate_pool.proposal_ids
        or result.full_outcome_proposal_ids != candidate_pool.proposal_ids
    ):
        _fail("bound result", "full-pool replay inventory is mismatched")
    if _utc_timestamp(
        result.outcomes_generated_at_utc,
        "bound result outcomes_generated_at_utc",
    ) <= _utc_timestamp(
        manifest.selection_timestamp_utc,
        "selection manifest selection_timestamp_utc",
    ):
        _fail("bound result", "outcomes do not postdate finalized blind selection")


def bind_complete_pool_result(
    selection_envelope: BlindSelectionManifestEnvelopeV1,
    candidate_pool: CandidatePoolV1,
    outcomes: Sequence[CandidateOutcomeV1],
    *,
    replay_evidence_digest: str,
    simulator_compatibility_identity: str,
    git_sha: str,
    outcomes_generated_at_utc: str,
    expected_full_pool_outcome_digest: str | None = None,
) -> BoundSelectionResultV1:
    """Bind complete post-selection evidence without permitting Stage-A mutation."""

    if not isinstance(selection_envelope, BlindSelectionManifestEnvelopeV1):
        _fail("selection_envelope", "invalid manifest envelope")
    validate_selection_manifest_against_pool(
        selection_envelope.manifest, candidate_pool
    )
    ordered_outcomes = _validate_complete_outcomes(candidate_pool, outcomes)
    full_digest = compute_full_pool_outcome_digest(ordered_outcomes)
    if expected_full_pool_outcome_digest is not None:
        if (
            _digest(
                expected_full_pool_outcome_digest,
                "expected_full_pool_outcome_digest",
            )
            != full_digest
        ):
            _fail("full pool outcomes", "persisted outcome digest mismatches content")
    _digest(replay_evidence_digest, "replay_evidence_digest")
    _digest(simulator_compatibility_identity, "simulator_compatibility_identity")
    manifest = selection_envelope.manifest
    result = BoundSelectionResultV1(
        selection_manifest_digest=manifest.content_digest,
        selection_manifest_envelope_digest=(
            selection_envelope.manifest_envelope_digest
        ),
        selection_snapshot_digest=_selection_snapshot_digest(manifest),
        replay_evidence_digest=replay_evidence_digest,
        full_pool_outcome_digest=full_digest,
        source_set_digest=candidate_pool.source_set_digest,
        candidate_pool_digest=candidate_pool.content_digest,
        verifier_bundle_digests=manifest.verifier_bundle_digests,
        evidence_proposal_ids=candidate_pool.proposal_ids,
        full_outcome_proposal_ids=candidate_pool.proposal_ids,
        simulator_compatibility_identity=simulator_compatibility_identity,
        git_sha=git_sha,
        outcomes_generated_at_utc=outcomes_generated_at_utc,
    )
    validate_bound_selection_result(result, selection_envelope, candidate_pool)
    return result


def create_oracle_decisions(
    candidate_pool: CandidatePoolV1,
    outcomes: Sequence[CandidateOutcomeV1],
    *,
    full_pool_outcome_digest: str,
) -> tuple[SelectorDecisionV1, ...]:
    """Create analysis-only oracle choices after exact complete outcomes exist."""

    ordered = _validate_complete_outcomes(candidate_pool, outcomes)
    observed_digest = compute_full_pool_outcome_digest(ordered)
    if _digest(full_pool_outcome_digest, "full_pool_outcome_digest") != observed_digest:
        _fail("oracle", "full-pool outcome content is unavailable or mismatched")
    by_id = {item.proposal_id: item for item in ordered}
    decisions: list[SelectorDecisionV1] = []
    for group in candidate_pool.groups:
        members = [by_id[item] for item in group.proposal_ids]
        if any(item.status != "conclusive" for item in members):
            _fail("oracle", "requires complete conclusive full-pool outcomes")
        ranking = tuple(
            item.proposal_id
            for item in sorted(
                members,
                key=lambda item: (item.success is not True, item.proposal_id),
            )
        )
        decisions.append(
            SelectorDecisionV1(
                selector_id=ORACLE_SELECTOR_ID,
                group_id=group.group_id,
                selected_proposal_id=ranking[0],
                ranking=ranking,
                predicted_failure_probabilities={},
                abstained=False,
            )
        )
    return tuple(decisions)


def _atomic_new_json(path: Path, value: object) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        _fail("bound result", "refusing to overwrite existing path")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        _fail("bound result", "parent must be a non-symlink directory")
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, destination)
        Path(temporary_name).unlink()
        temporary_name = None
    except OSError as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise BlindProtocolError(f"bound result: atomic create failed: {exc}") from exc
    return destination


def save_bound_selection_result(path: Path, result: BoundSelectionResultV1) -> Path:
    """Persist a new strict Stage-B/C result artifact."""

    if not isinstance(result, BoundSelectionResultV1):
        _fail("result", "expected BoundSelectionResultV1")
    return _atomic_new_json(Path(path), result.as_mapping())


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("bound result JSON", f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail("bound result JSON", f"non-finite constant {value!r}")


def load_bound_selection_result(
    path: Path,
    *,
    selection_envelope: BlindSelectionManifestEnvelopeV1,
    candidate_pool: CandidatePoolV1,
) -> BoundSelectionResultV1:
    """Load a result and verify it against the original Stage-A artifact and pool."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail("bound result", "expected regular non-symlink file")
    try:
        stat = source.stat()
        if stat.st_nlink != 1:
            _fail("bound result", "hard-linked files are unsupported")
        if stat.st_size <= 0 or stat.st_size > MAX_BOUND_SELECTION_RESULT_BYTES:
            _fail("bound result", "file size is outside safe bounds")
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except BlindProtocolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlindProtocolError(f"bound result: could not read safely: {exc}") from exc
    if not isinstance(raw, Mapping):
        _fail("bound result", "expected JSON object")
    result = BoundSelectionResultV1.from_mapping(cast(Mapping[str, object], raw))
    validate_bound_selection_result(result, selection_envelope, candidate_pool)
    return result


@dataclass(frozen=True, slots=True)
class FakeBlindProtocolSmokeReportV1:
    """Infrastructure-only proof that blind artifact gates compose end to end."""

    selection_manifest_digest: str
    selection_manifest_envelope_digest: str
    bound_result_digest: str
    full_pool_outcome_digest: str
    candidate_count: int
    selector_count: int
    oracle_decision_count: int
    physical_simulator_evidence: bool = False
    schema_version: str = FAKE_BLIND_SMOKE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Prevent fake smoke output from being represented as physical evidence."""

        for name in (
            "selection_manifest_digest",
            "selection_manifest_envelope_digest",
            "bound_result_digest",
            "full_pool_outcome_digest",
        ):
            _digest(getattr(self, name), f"FakeBlindProtocolSmokeReportV1.{name}")
        if (
            type(self.candidate_count) is not int
            or self.candidate_count <= 0
            or type(self.selector_count) is not int
            or self.selector_count <= 0
            or type(self.oracle_decision_count) is not int
            or self.oracle_decision_count <= 0
        ):
            _fail("FakeBlindProtocolSmokeReportV1", "counts must be positive")
        if self.physical_simulator_evidence is not False:
            _fail(
                "FakeBlindProtocolSmokeReportV1.physical_simulator_evidence",
                "fake smoke is infrastructure-only",
            )
        if self.schema_version != FAKE_BLIND_SMOKE_SCHEMA_VERSION:
            _fail(
                "FakeBlindProtocolSmokeReportV1.schema_version",
                "unsupported version",
            )


def run_fake_blind_protocol_smoke(
    output_root: Path,
    candidate_pool: CandidatePoolV1,
    selections: Sequence[SelectorDecisionV1],
    outcomes: Sequence[CandidateOutcomeV1],
    *,
    selector_configuration_digest: str,
    verifier_bundle_digests: Mapping[str, str] | Sequence[tuple[str, str]],
    selection_timestamp_utc: str,
    outcomes_generated_at_utc: str,
    replay_evidence_digest: str,
    simulator_compatibility_identity: str,
    git_sha: str,
) -> FakeBlindProtocolSmokeReportV1:
    """Exercise Stage A, post-selection binding, oracle timing, and strict reloads."""

    root = initialize_blind_selection_output_root(output_root)
    stage_a_root = root / "stage-a"
    manifest_path, envelope = run_blind_selection_stage_a(
        candidate_pool,
        selections,
        selector_configuration_digest=selector_configuration_digest,
        verifier_bundle_digests=verifier_bundle_digests,
        selection_timestamp_utc=selection_timestamp_utc,
        output_root=stage_a_root,
    )
    loaded_envelope = load_blind_selection_manifest(manifest_path)
    result = bind_complete_pool_result(
        loaded_envelope,
        candidate_pool,
        outcomes,
        replay_evidence_digest=replay_evidence_digest,
        simulator_compatibility_identity=simulator_compatibility_identity,
        git_sha=git_sha,
        outcomes_generated_at_utc=outcomes_generated_at_utc,
    )
    oracle = create_oracle_decisions(
        candidate_pool,
        outcomes,
        full_pool_outcome_digest=result.full_pool_outcome_digest,
    )
    result_root = root / "stage-bc"
    result_root.mkdir(exist_ok=False)
    result_path = save_bound_selection_result(
        result_root / BOUND_SELECTION_RESULT_FILENAME,
        result,
    )
    loaded_result = load_bound_selection_result(
        result_path,
        selection_envelope=loaded_envelope,
        candidate_pool=candidate_pool,
    )
    return FakeBlindProtocolSmokeReportV1(
        selection_manifest_digest=loaded_envelope.semantic_digest,
        selection_manifest_envelope_digest=loaded_envelope.envelope_digest,
        bound_result_digest=loaded_result.content_digest,
        full_pool_outcome_digest=loaded_result.full_pool_outcome_digest,
        candidate_count=len(candidate_pool.proposal_ids),
        selector_count=len({item.selector_id for item in selections}),
        oracle_decision_count=len(oracle),
    )


__all__ = [
    "BLIND_SELECTOR_INPUT_ALLOWLIST_SEMANTIC",
    "BLIND_SELECTOR_INPUT_FIELDS",
    "BLIND_STAGE_A_GROUP_FIELDS",
    "BLIND_STAGE_A_POOL_BINDING_FIELDS",
    "BOUND_SELECTION_RESULT_FILENAME",
    "BOUND_SELECTION_RESULT_SCHEMA_VERSION",
    "ORACLE_SELECTOR_ID",
    "EXPECTED_STAGE_A_SELECTOR_IDS",
    "EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS",
    "BlindProtocolError",
    "BlindSelectorInputV1",
    "BoundSelectionResultV1",
    "FakeBlindProtocolSmokeReportV1",
    "bind_complete_pool_result",
    "blind_selector_input_allowlist_digest",
    "compute_full_pool_outcome_digest",
    "create_oracle_decisions",
    "finalize_blind_selection",
    "finalize_blind_selection_from_blind_pool",
    "load_bound_selection_result",
    "run_blind_selection_stage_a",
    "run_blind_selection_stage_a_from_blind_pool",
    "run_fake_blind_protocol_smoke",
    "save_bound_selection_result",
    "validate_bound_selection_result",
    "validate_selection_manifest_against_pool",
    "validate_selection_manifest_against_blind_pool",
]
