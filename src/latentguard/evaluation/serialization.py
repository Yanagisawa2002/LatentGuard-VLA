"""Versioned, crash-safe serialization for evaluation runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, TypeVar, cast

from latentguard.corruptions.serialization import (
    CorruptionDataset,
    load_corruption_dataset,
    validate_corruption_dataset,
)
from latentguard.evaluation.models import (
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EvaluationEvidence,
    EvaluationIdentityError,
    EvaluationStatus,
    compute_configuration_digest,
    validate_evaluator_identity,
)
from latentguard.evaluation.security import is_sanitized_operational_text
from latentguard.evaluation.validation import validate_evaluation_evidence
from latentguard.models import (
    CURRENT_SCHEMA_VERSION,
    FailureEvent,
    JsonScalar,
    LabelSource,
    LabelStrength,
)

MANIFEST_NAME = "manifest.json"
EVALUATION_SERIALIZATION_FORMAT = "latentguard-evaluation-dataset"
EVALUATION_SERIALIZATION_VERSION = 1
EVALUATION_DATASET_SCHEMA_VERSION = "1.0"
LEDGER_SCHEMA_VERSION = "1.0"
RUN_MANIFEST_SCHEMA_VERSION = "1.0"
RUN_ID_PREFIX = "evr-sha256-"

_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^evr-sha256-[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_MAX_SEED = 2**64
_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


class EvaluationSerializationError(ValueError):
    """Raised when an evaluation bundle is malformed, unsafe, or conflicting."""


class UnsupportedEvaluationSerializationVersionError(EvaluationSerializationError):
    """Raised when an evaluation serialization version is unsupported."""


class LedgerState(StrEnum):
    """Durable state for one proposal attempt."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INDETERMINATE = "indeterminate"
    INVALID = "invalid"
    SKIPPED = "skipped"
    EXECUTION_ERROR = "execution_error"


class RunState(StrEnum):
    """Lifecycle state for an evaluation dataset."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class RunEnvironment:
    """Sanitized, non-identity environment metadata for a local run."""

    git_commit_sha: str | None
    git_branch: str | None
    python_version: str
    numpy_version: str
    platform: str
    launch_command: str


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Operational provenance for one resumable evaluation run."""

    environment: RunEnvironment
    evaluator_id: str
    evaluator_version: str
    evaluator_configuration_digest: str
    source_corruption_dataset_digest: str
    source_dataset_id: str
    seed: int
    started_at: str
    finished_at: str | None
    final_state: RunState
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One durable attempt state and its sanitized operational error."""

    proposal_id: str
    evidence_id: str | None
    attempt_ordinal: int
    evaluator_id: str
    evaluator_version: str
    evaluator_configuration_digest: str
    evaluation_seed: int
    state: LedgerState
    error_type: str | None
    error_message: str | None
    retry_eligible: bool
    started_at: str | None
    finished_at: str | None
    schema_version: str = LEDGER_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Deterministic attempt-level status counts for one run."""

    conclusive: int
    indeterminate: int
    invalid: int
    skipped: int
    execution_error: int
    projected_outcome_labels: int
    retried_attempts: int
    total_attempts: int


def _freeze_json_value(value: object, context: str) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise EvaluationSerializationError(f"{context}: floats must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or key != key.strip():
                raise EvaluationSerializationError(
                    f"{context}: keys must be non-empty strings without whitespace"
                )
            frozen[key] = _freeze_json_value(item, f"{context}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise EvaluationSerializationError(
        f"{context}: unsupported JSON value {type(value).__name__}"
    )


def _freeze_configuration(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json_value(value, "EvaluationDataset.resolved_configuration")
    if not isinstance(frozen, Mapping):
        raise EvaluationSerializationError(
            "EvaluationDataset.resolved_configuration: expected an object"
        )
    return cast(Mapping[str, object], frozen)


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Immutable evidence, ledger, summary, and provenance for one run."""

    source_corruption_dataset_digest: str
    source_dataset_id: str
    evaluator_id: str
    evaluator_version: str
    resolved_evaluator_configuration: Mapping[str, object]
    evaluator_configuration_digest: str
    run_id: str
    base_seed: int
    max_proposals: int | None
    selected_proposal_ids: tuple[str, ...]
    evidence: tuple[EvaluationEvidence, ...]
    ledger: tuple[LedgerEntry, ...]
    summary: EvaluationSummary
    artifact_references: tuple[str, ...]
    run_state: RunState
    run_manifest: RunManifest
    schema_version: str = EVALUATION_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze collections and validate all internal relationships."""
        object.__setattr__(
            self,
            "resolved_evaluator_configuration",
            _freeze_configuration(self.resolved_evaluator_configuration),
        )
        object.__setattr__(
            self, "selected_proposal_ids", tuple(self.selected_proposal_ids)
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "ledger", tuple(self.ledger))
        object.__setattr__(self, "artifact_references", tuple(self.artifact_references))
        validate_evaluation_dataset(self)


_TERMINAL_LEDGER_STATES = frozenset(
    {
        LedgerState.COMPLETED,
        LedgerState.INDETERMINATE,
        LedgerState.INVALID,
        LedgerState.SKIPPED,
        LedgerState.EXECUTION_ERROR,
    }
)

_STATUS_TO_LEDGER_STATE = {
    EvaluationStatus.CONCLUSIVE: LedgerState.COMPLETED,
    EvaluationStatus.INDETERMINATE: LedgerState.INDETERMINATE,
    EvaluationStatus.INVALID: LedgerState.INVALID,
    EvaluationStatus.SKIPPED: LedgerState.SKIPPED,
    EvaluationStatus.EXECUTION_ERROR: LedgerState.EXECUTION_ERROR,
}


def compute_evaluation_seed(
    *,
    base_seed: int,
    proposal_id: str,
    evaluator_id: str,
    evaluator_version: str,
    evaluator_configuration_digest: str,
    attempt_ordinal: int,
) -> int:
    """Derive one stable unsigned 64-bit seed from canonical identity inputs."""
    _validate_seed(base_seed, "base_seed")
    _validate_ordinal(attempt_ordinal, "attempt_ordinal")
    payload = {
        "attempt_ordinal": attempt_ordinal,
        "base_seed": base_seed,
        "evaluator_configuration_digest": _required_text(
            evaluator_configuration_digest,
            "evaluator_configuration_digest",
        ),
        "evaluator_id": _required_text(evaluator_id, "evaluator_id"),
        "evaluator_version": _required_text(evaluator_version, "evaluator_version"),
        "proposal_id": _required_text(proposal_id, "proposal_id"),
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_run_identifier(
    *,
    source_corruption_dataset_digest: str,
    source_dataset_id: str,
    evaluator_id: str,
    evaluator_version: str,
    evaluator_configuration_digest: str,
    base_seed: int,
    max_proposals: int | None,
    selected_proposal_ids: Sequence[str],
) -> str:
    """Compute the path-independent identity of an exact evaluation plan."""
    _validate_seed(base_seed, "base_seed")
    _validate_max_proposals(max_proposals)
    try:
        validate_evaluator_identity(
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
        )
    except EvaluationIdentityError as exc:
        raise EvaluationSerializationError(str(exc)) from exc
    proposal_ids = tuple(selected_proposal_ids)
    for index, proposal_id in enumerate(proposal_ids):
        _required_text(proposal_id, f"selected_proposal_ids[{index}]")
    if len(set(proposal_ids)) != len(proposal_ids):
        raise EvaluationSerializationError(
            "selected_proposal_ids: duplicate proposal identifiers are unsupported"
        )
    payload = {
        "base_seed": base_seed,
        "evaluator_configuration_digest": _required_text(
            evaluator_configuration_digest, "evaluator_configuration_digest"
        ),
        "evaluator_id": _required_text(evaluator_id, "evaluator_id"),
        "evaluator_version": _required_text(evaluator_version, "evaluator_version"),
        "max_proposals": max_proposals,
        "selected_proposal_ids": list(proposal_ids),
        "source_corruption_dataset_digest": _required_text(
            source_corruption_dataset_digest,
            "source_corruption_dataset_digest",
        ),
        "source_dataset_id": _required_text(source_dataset_id, "source_dataset_id"),
    }
    hexadecimal = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"{RUN_ID_PREFIX}{hexadecimal}"


def compute_corruption_dataset_content_digest(
    dataset: CorruptionDataset,
) -> str:
    """Hash all validated M1 logical metadata and transformed action contents."""
    validate_corruption_dataset(dataset)
    layout = dataset.action_layout
    proposals: list[dict[str, object]] = []
    for proposal in dataset.proposals:
        action = proposal.transformed_action
        proposals.append(
            {
                "proposal_id": proposal.proposal_id,
                "source_episode_id": proposal.source_episode_id,
                "source_candidate_id": proposal.source_candidate_id,
                "source_policy_id": proposal.source_policy_id,
                "source_task_id": proposal.source_task_id,
                "split_group_id": proposal.split_group_id,
                "corruption_type": proposal.corruption_type,
                "resolved_parameters": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in proposal.resolved_parameters.items()
                },
                "seed": proposal.seed,
                "generation_ordinal": proposal.generation_ordinal,
                "schema_version": proposal.schema_version,
                "notes": proposal.notes,
                "transformed_action": {
                    "coordinate_frame": action.coordinate_frame,
                    "control_period_s": action.control_period_s,
                    "schema_version": action.schema_version,
                    "dtype": action.actions.dtype.str,
                    "shape": list(action.actions.shape),
                    "content_sha256": hashlib.sha256(
                        action.actions.tobytes(order="C")
                    ).hexdigest(),
                },
            }
        )
    payload = {
        "schema_version": dataset.schema_version,
        "source_dataset_id": dataset.source_dataset_id,
        "action_layout": {
            "action_dim": layout.action_dim,
            "schema_version": layout.schema_version,
            "description": layout.description,
            "metadata": dict(layout.metadata),
            "fields": [
                {
                    "name": field.name,
                    "indices": list(field.indices),
                    "semantic": field.semantic.value,
                    "units": field.units,
                    "description": field.description,
                    "metadata": dict(field.metadata),
                }
                for field in layout.fields
            ],
        },
        "proposals": proposals,
    }
    return f"sha256:{hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()}"


def compute_corruption_dataset_digest(path: Path) -> str:
    """Hash one validated M1 bundle independently of its absolute location."""
    return compute_corruption_dataset_content_digest(
        load_corruption_dataset(Path(path).absolute())
    )


def validate_evaluation_dataset(
    dataset: EvaluationDataset,
    corruption_dataset: CorruptionDataset | None = None,
    expected_corruption_digest: str | None = None,
) -> None:
    """Validate identities, attempts, evidence links, state, and source references."""
    if not isinstance(dataset, EvaluationDataset):
        raise EvaluationSerializationError(
            "EvaluationDataset: expected an EvaluationDataset model"
        )
    if dataset.schema_version != EVALUATION_DATASET_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            "EvaluationDataset.schema_version: unsupported version "
            f"{dataset.schema_version!r}"
        )
    if not isinstance(dataset.run_state, RunState):
        raise EvaluationSerializationError(
            "EvaluationDataset.run_state: expected a RunState value"
        )
    if (
        _SHA256_DIGEST_PATTERN.fullmatch(dataset.source_corruption_dataset_digest)
        is None
    ):
        raise EvaluationSerializationError(
            "EvaluationDataset.source_corruption_dataset_digest: expected "
            "sha256:<64 lowercase hex characters>"
        )
    if (
        expected_corruption_digest is not None
        and dataset.source_corruption_dataset_digest != expected_corruption_digest
    ):
        raise EvaluationSerializationError(
            "EvaluationDataset.source_corruption_dataset_digest: source bundle "
            "does not match the persisted run"
        )
    if corruption_dataset is not None:
        actual_corruption_digest = compute_corruption_dataset_content_digest(
            corruption_dataset
        )
        if actual_corruption_digest != dataset.source_corruption_dataset_digest:
            raise EvaluationSerializationError(
                "EvaluationDataset.source_corruption_dataset_digest: in-memory "
                "corruption dataset contents do not match the persisted run"
            )
    for field in ("source_dataset_id", "evaluator_id", "evaluator_version"):
        _required_text(getattr(dataset, field), field)
    try:
        validate_evaluator_identity(
            evaluator_id=dataset.evaluator_id,
            evaluator_version=dataset.evaluator_version,
        )
    except EvaluationIdentityError as exc:
        raise EvaluationSerializationError(str(exc)) from exc
    _validate_seed(dataset.base_seed, "base_seed")
    _validate_max_proposals(dataset.max_proposals)
    expected_configuration_digest = compute_configuration_digest(
        dataset.resolved_evaluator_configuration
    )
    if dataset.evaluator_configuration_digest != expected_configuration_digest:
        raise EvaluationSerializationError(
            "EvaluationDataset.evaluator_configuration_digest: deterministic "
            "configuration digest mismatch"
        )
    selected = dataset.selected_proposal_ids
    if any(not isinstance(item, str) or not item for item in selected):
        raise EvaluationSerializationError(
            "EvaluationDataset.selected_proposal_ids: expected non-empty strings"
        )
    if len(set(selected)) != len(selected):
        raise EvaluationSerializationError(
            "EvaluationDataset.selected_proposal_ids: duplicate identifiers"
        )
    expected_run_id = compute_run_identifier(
        source_corruption_dataset_digest=dataset.source_corruption_dataset_digest,
        source_dataset_id=dataset.source_dataset_id,
        evaluator_id=dataset.evaluator_id,
        evaluator_version=dataset.evaluator_version,
        evaluator_configuration_digest=dataset.evaluator_configuration_digest,
        base_seed=dataset.base_seed,
        max_proposals=dataset.max_proposals,
        selected_proposal_ids=selected,
    )
    if (
        dataset.run_id != expected_run_id
        or _RUN_ID_PATTERN.fullmatch(dataset.run_id) is None
    ):
        raise EvaluationSerializationError(
            "EvaluationDataset.run_id: deterministic run identifier mismatch"
        )
    _validate_run_manifest(dataset)
    proposal_lookup = _validate_source_references(dataset, corruption_dataset)
    _validate_evidence_and_ledger(dataset, proposal_lookup)
    expected_artifacts = tuple(
        sorted(
            {
                reference
                for evidence in dataset.evidence
                for reference in evidence.artifact_references
            }
        )
    )
    if dataset.artifact_references != expected_artifacts:
        raise EvaluationSerializationError(
            "EvaluationDataset.artifact_references: must be the sorted unique "
            "references from evidence"
        )
    from latentguard.evaluation.reporting import build_evaluation_summary

    _validate_summary(dataset.summary)
    expected_summary = build_evaluation_summary(dataset.evidence, dataset.ledger)
    if dataset.summary != expected_summary:
        raise EvaluationSerializationError(
            "EvaluationDataset.summary: does not match recomputed ledger counts"
        )
    unfinished = any(
        entry.state in {LedgerState.PENDING, LedgerState.RUNNING}
        for entry in dataset.ledger
    )
    if dataset.run_state is RunState.COMPLETE and unfinished:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_state: complete runs cannot contain "
            "unfinished attempts"
        )
    if dataset.run_state in {RunState.COMPLETE, RunState.INTERRUPTED} and (
        dataset.run_manifest.finished_at is None
    ):
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.finished_at: complete and interrupted "
            "runs require a timestamp"
        )
    if (
        dataset.run_state is RunState.IN_PROGRESS
        and dataset.run_manifest.finished_at is not None
    ):
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.finished_at: in-progress runs must use null"
        )
    _validate_attempt_timestamp_bounds(dataset)


def _validate_summary(summary: EvaluationSummary) -> None:
    if not isinstance(summary, EvaluationSummary):
        raise EvaluationSerializationError(
            "EvaluationDataset.summary: expected EvaluationSummary"
        )
    for field in (
        "conclusive",
        "indeterminate",
        "invalid",
        "skipped",
        "execution_error",
        "projected_outcome_labels",
        "retried_attempts",
        "total_attempts",
    ):
        value = getattr(summary, field)
        if type(value) is not int or value < 0:
            raise EvaluationSerializationError(
                f"EvaluationDataset.summary.{field}: expected a non-negative integer"
            )


def _validate_attempt_timestamp_bounds(dataset: EvaluationDataset) -> None:
    run_start = _validate_timestamp(
        dataset.run_manifest.started_at, "run_manifest.started_at"
    )
    run_finish = (
        None
        if dataset.run_manifest.finished_at is None
        else _validate_timestamp(
            dataset.run_manifest.finished_at, "run_manifest.finished_at"
        )
    )
    for index, entry in enumerate(dataset.ledger):
        if entry.started_at is not None:
            started = _validate_timestamp(
                entry.started_at, f"ledger[{index}].started_at"
            )
            if started < run_start:
                raise EvaluationSerializationError(
                    f"EvaluationDataset.ledger[{index}].started_at: precedes run start"
                )
            if run_finish is not None and started > run_finish:
                raise EvaluationSerializationError(
                    f"EvaluationDataset.ledger[{index}].started_at: exceeds run finish"
                )
        if entry.finished_at is not None and run_finish is not None:
            finished = _validate_timestamp(
                entry.finished_at, f"ledger[{index}].finished_at"
            )
            if finished > run_finish:
                raise EvaluationSerializationError(
                    f"EvaluationDataset.ledger[{index}].finished_at: exceeds run finish"
                )


def _validate_source_references(
    dataset: EvaluationDataset,
    corruption_dataset: CorruptionDataset | None,
) -> Mapping[str, object]:
    if corruption_dataset is None:
        return MappingProxyType({})
    validate_corruption_dataset(corruption_dataset)
    if corruption_dataset.source_dataset_id != dataset.source_dataset_id:
        raise EvaluationSerializationError(
            "EvaluationDataset.source_dataset_id: does not match corruption dataset"
        )
    proposals = corruption_dataset.proposals
    selected_count = (
        len(proposals)
        if dataset.max_proposals is None
        else min(dataset.max_proposals, len(proposals))
    )
    expected_ids = tuple(item.proposal_id for item in proposals[:selected_count])
    if dataset.selected_proposal_ids != expected_ids:
        raise EvaluationSerializationError(
            "EvaluationDataset.selected_proposal_ids: do not match the stable "
            "generation-order prefix of the corruption dataset"
        )
    return MappingProxyType(
        {item.proposal_id: item for item in proposals[:selected_count]}
    )


def _validate_evidence_and_ledger(
    dataset: EvaluationDataset, proposal_lookup: Mapping[str, object]
) -> None:
    selected_order = {
        proposal_id: index
        for index, proposal_id in enumerate(dataset.selected_proposal_ids)
    }
    evidence_by_key: dict[tuple[str, int], EvaluationEvidence] = {}
    evidence_ids: set[str] = set()
    evidence_order: list[tuple[int, int]] = []
    for index, evidence in enumerate(dataset.evidence):
        try:
            validate_evaluation_evidence(evidence)
        except (TypeError, ValueError) as exc:
            raise EvaluationSerializationError(
                f"EvaluationDataset.evidence[{index}]: {exc}"
            ) from exc
        if evidence.status is EvaluationStatus.EXECUTION_ERROR:
            _validate_sanitized_execution_error_evidence(evidence, index)
        if evidence.proposal_id not in selected_order:
            raise EvaluationSerializationError(
                f"EvaluationDataset.evidence[{index}].proposal_id: unknown reference"
            )
        key = (evidence.proposal_id, evidence.attempt_ordinal)
        if key in evidence_by_key or evidence.evidence_id in evidence_ids:
            raise EvaluationSerializationError(
                "EvaluationDataset.evidence: duplicate attempt or evidence identifier"
            )
        evidence_by_key[key] = evidence
        evidence_ids.add(evidence.evidence_id)
        evidence_order.append(
            (selected_order[evidence.proposal_id], evidence.attempt_ordinal)
        )
        if evidence.source_dataset_id != dataset.source_dataset_id:
            raise EvaluationSerializationError(
                f"EvaluationDataset.evidence[{index}].source_dataset_id: mismatch"
            )
        if (
            evidence.evaluator_id != dataset.evaluator_id
            or evidence.evaluator_version != dataset.evaluator_version
            or evidence.evaluator_configuration_digest
            != dataset.evaluator_configuration_digest
        ):
            raise EvaluationSerializationError(
                f"EvaluationDataset.evidence[{index}]: evaluator identity mismatch"
            )
        expected_seed = compute_evaluation_seed(
            base_seed=dataset.base_seed,
            proposal_id=evidence.proposal_id,
            evaluator_id=dataset.evaluator_id,
            evaluator_version=dataset.evaluator_version,
            evaluator_configuration_digest=dataset.evaluator_configuration_digest,
            attempt_ordinal=evidence.attempt_ordinal,
        )
        if evidence.evaluation_seed != expected_seed:
            raise EvaluationSerializationError(
                f"EvaluationDataset.evidence[{index}].evaluation_seed: mismatch"
            )
        source = proposal_lookup.get(evidence.proposal_id)
        if source is not None:
            for field in (
                "source_episode_id",
                "source_candidate_id",
                "split_group_id",
            ):
                if getattr(evidence, field) != getattr(source, field):
                    raise EvaluationSerializationError(
                        f"EvaluationDataset.evidence[{index}].{field}: source mismatch"
                    )
    if evidence_order != sorted(evidence_order):
        raise EvaluationSerializationError(
            "EvaluationDataset.evidence: records are not in deterministic order"
        )

    ledger_by_key: dict[tuple[str, int], LedgerEntry] = {}
    attempts_by_proposal: dict[str, list[LedgerEntry]] = defaultdict(list)
    ledger_order: list[tuple[int, int]] = []
    for index, entry in enumerate(dataset.ledger):
        _validate_ledger_entry(entry, dataset, index)
        if entry.proposal_id not in selected_order:
            raise EvaluationSerializationError(
                f"EvaluationDataset.ledger[{index}].proposal_id: unknown reference"
            )
        key = (entry.proposal_id, entry.attempt_ordinal)
        if key in ledger_by_key:
            raise EvaluationSerializationError(
                "EvaluationDataset.ledger: duplicate proposal attempt"
            )
        ledger_by_key[key] = entry
        attempts_by_proposal[entry.proposal_id].append(entry)
        ledger_order.append((selected_order[entry.proposal_id], entry.attempt_ordinal))
        expected_seed = compute_evaluation_seed(
            base_seed=dataset.base_seed,
            proposal_id=entry.proposal_id,
            evaluator_id=dataset.evaluator_id,
            evaluator_version=dataset.evaluator_version,
            evaluator_configuration_digest=dataset.evaluator_configuration_digest,
            attempt_ordinal=entry.attempt_ordinal,
        )
        if entry.evaluation_seed != expected_seed:
            raise EvaluationSerializationError(
                f"EvaluationDataset.ledger[{index}].evaluation_seed: mismatch"
            )
        linked_evidence = evidence_by_key.get(key)
        if entry.state in {LedgerState.PENDING, LedgerState.RUNNING}:
            if linked_evidence is not None or entry.evidence_id is not None:
                raise EvaluationSerializationError(
                    f"EvaluationDataset.ledger[{index}]: unfinished attempts cannot "
                    "reference evidence"
                )
        else:
            if (
                linked_evidence is None
                or entry.evidence_id != linked_evidence.evidence_id
            ):
                raise EvaluationSerializationError(
                    f"EvaluationDataset.ledger[{index}]: terminal attempt must "
                    "reference its evidence"
                )
            if _STATUS_TO_LEDGER_STATE[linked_evidence.status] is not entry.state:
                raise EvaluationSerializationError(
                    f"EvaluationDataset.ledger[{index}].state: evidence status mismatch"
                )
    if ledger_order != sorted(ledger_order):
        raise EvaluationSerializationError(
            "EvaluationDataset.ledger: entries are not in deterministic order"
        )
    if set(evidence_by_key) - set(ledger_by_key):
        raise EvaluationSerializationError(
            "EvaluationDataset.evidence: record has no matching ledger entry"
        )
    for proposal_id in dataset.selected_proposal_ids:
        attempts = attempts_by_proposal.get(proposal_id, [])
        if not attempts:
            raise EvaluationSerializationError(
                f"EvaluationDataset.ledger[{proposal_id!r}]: selected proposals "
                "must contain an attempt-zero ledger entry"
            )
        ordinals = [entry.attempt_ordinal for entry in attempts]
        if ordinals != list(range(len(attempts))):
            raise EvaluationSerializationError(
                f"EvaluationDataset.ledger[{proposal_id!r}]: attempt ordinals must "
                "start at zero and remain contiguous"
            )
        for prior, current in zip(attempts, attempts[1:], strict=False):
            if (
                prior.state is not LedgerState.EXECUTION_ERROR
                or not prior.retry_eligible
            ):
                raise EvaluationSerializationError(
                    f"EvaluationDataset.ledger[{proposal_id!r}]: attempt "
                    f"{current.attempt_ordinal} does not follow a retry-eligible error"
                )
            if prior.finished_at is not None and current.started_at is not None:
                prior_finished = _validate_timestamp(
                    prior.finished_at,
                    f"ledger[{proposal_id!r}][{prior.attempt_ordinal}].finished_at",
                )
                current_started = _validate_timestamp(
                    current.started_at,
                    f"ledger[{proposal_id!r}][{current.attempt_ordinal}].started_at",
                )
                if current_started < prior_finished:
                    raise EvaluationSerializationError(
                        f"EvaluationDataset.ledger[{proposal_id!r}]: retry attempt "
                        "cannot start before the prior attempt finishes"
                    )


def _validate_ledger_entry(
    entry: LedgerEntry, dataset: EvaluationDataset, index: int
) -> None:
    context = f"EvaluationDataset.ledger[{index}]"
    if not isinstance(entry, LedgerEntry):
        raise EvaluationSerializationError(f"{context}: expected LedgerEntry")
    if entry.schema_version != LEDGER_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            f"{context}.schema_version: unsupported version {entry.schema_version!r}"
        )
    _required_text(entry.proposal_id, f"{context}.proposal_id")
    _validate_ordinal(entry.attempt_ordinal, f"{context}.attempt_ordinal")
    _validate_seed(entry.evaluation_seed, f"{context}.evaluation_seed")
    if (
        entry.evaluator_id != dataset.evaluator_id
        or entry.evaluator_version != dataset.evaluator_version
        or entry.evaluator_configuration_digest
        != dataset.evaluator_configuration_digest
    ):
        raise EvaluationSerializationError(f"{context}: evaluator identity mismatch")
    if not isinstance(entry.state, LedgerState):
        raise EvaluationSerializationError(f"{context}.state: unsupported value")
    if type(entry.retry_eligible) is not bool:
        raise EvaluationSerializationError(
            f"{context}.retry_eligible: expected boolean"
        )
    started = (
        None
        if entry.started_at is None
        else _validate_timestamp(entry.started_at, f"{context}.started_at")
    )
    finished = (
        None
        if entry.finished_at is None
        else _validate_timestamp(entry.finished_at, f"{context}.finished_at")
    )
    if started is not None and finished is not None and finished < started:
        raise EvaluationSerializationError(
            f"{context}.finished_at: cannot precede started_at"
        )
    if entry.state is LedgerState.PENDING:
        if (
            any(
                value is not None
                for value in (
                    entry.evidence_id,
                    entry.error_type,
                    entry.error_message,
                    entry.started_at,
                    entry.finished_at,
                )
            )
            or entry.retry_eligible
        ):
            raise EvaluationSerializationError(
                f"{context}: pending entries cannot contain execution results"
            )
    elif entry.state is LedgerState.RUNNING:
        if (
            entry.started_at is None
            or entry.evidence_id is not None
            or entry.finished_at is not None
            or entry.error_type is not None
            or entry.error_message is not None
            or entry.retry_eligible
        ):
            raise EvaluationSerializationError(
                f"{context}: running entry fields are inconsistent"
            )
    else:
        if (
            entry.evidence_id is None
            or entry.started_at is None
            or entry.finished_at is None
        ):
            raise EvaluationSerializationError(
                f"{context}: terminal entries require evidence and timestamps"
            )
        if entry.state is LedgerState.EXECUTION_ERROR:
            if (
                entry.error_type is None
                or entry.error_message is None
                or not entry.retry_eligible
            ):
                raise EvaluationSerializationError(
                    f"{context}: execution errors require sanitized error fields "
                    "and retry"
                )
            error_type = _required_text(entry.error_type, f"{context}.error_type")
            error_message = _required_text(
                entry.error_message, f"{context}.error_message"
            )
            if _ERROR_TYPE_PATTERN.fullmatch(error_type) is None:
                raise EvaluationSerializationError(
                    f"{context}.error_type: expected a concise exception type"
                )
            if not is_sanitized_operational_text(error_message):
                raise EvaluationSerializationError(
                    f"{context}.error_message: expected concise sanitized text"
                )
        elif (
            entry.error_type is not None
            or entry.error_message is not None
            or entry.retry_eligible
        ):
            raise EvaluationSerializationError(
                f"{context}: non-error terminal entries cannot contain errors"
            )


def _validate_sanitized_execution_error_evidence(
    evidence: EvaluationEvidence, index: int
) -> None:
    context = f"EvaluationDataset.evidence[{index}]"
    for field in ("termination_reason", "notes"):
        value = getattr(evidence, field)
        if value is None:
            continue
        if not is_sanitized_operational_text(value):
            raise EvaluationSerializationError(
                f"{context}.{field}: expected concise sanitized text"
            )
    for key, value in evidence.metrics.items():
        if not is_sanitized_operational_text(key):
            raise EvaluationSerializationError(
                f"{context}.metrics[{key!r}]: expected a sanitized diagnostic key"
            )
        if isinstance(value, str) and not is_sanitized_operational_text(value):
            raise EvaluationSerializationError(
                f"{context}.metrics[{key!r}]: expected concise sanitized text"
            )


def _validate_run_manifest(dataset: EvaluationDataset) -> None:
    manifest = dataset.run_manifest
    if not isinstance(manifest, RunManifest):
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest: expected RunManifest"
        )
    if manifest.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            "EvaluationDataset.run_manifest.schema_version: unsupported version"
        )
    if not isinstance(manifest.final_state, RunState):
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.final_state: expected a RunState value"
        )
    if manifest.final_state is not dataset.run_state:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.final_state: does not match run_state"
        )
    duplicate_fields = (
        (manifest.evaluator_id, dataset.evaluator_id, "evaluator_id"),
        (manifest.evaluator_version, dataset.evaluator_version, "evaluator_version"),
        (
            manifest.evaluator_configuration_digest,
            dataset.evaluator_configuration_digest,
            "evaluator_configuration_digest",
        ),
        (
            manifest.source_corruption_dataset_digest,
            dataset.source_corruption_dataset_digest,
            "source_corruption_dataset_digest",
        ),
        (manifest.source_dataset_id, dataset.source_dataset_id, "source_dataset_id"),
        (manifest.seed, dataset.base_seed, "seed"),
    )
    for actual, expected, field in duplicate_fields:
        if actual != expected:
            raise EvaluationSerializationError(
                f"EvaluationDataset.run_manifest.{field}: top-level mismatch"
            )
    _validate_seed(manifest.seed, "run_manifest.seed")
    started = _validate_timestamp(manifest.started_at, "run_manifest.started_at")
    finished = (
        None
        if manifest.finished_at is None
        else _validate_timestamp(manifest.finished_at, "run_manifest.finished_at")
    )
    if finished is not None and finished < started:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.finished_at: cannot precede started_at"
        )
    environment = manifest.environment
    if not isinstance(environment, RunEnvironment):
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.environment: expected RunEnvironment"
        )
    if environment.git_commit_sha is not None and (
        not isinstance(environment.git_commit_sha, str)
        or _GIT_SHA_PATTERN.fullmatch(environment.git_commit_sha) is None
    ):
        raise EvaluationSerializationError(
            "RunEnvironment.git_commit_sha: expected a full lowercase SHA or null"
        )
    for field in (
        "python_version",
        "numpy_version",
        "platform",
        "launch_command",
    ):
        value = _required_text(getattr(environment, field), f"RunEnvironment.{field}")
        if not is_sanitized_operational_text(value):
            raise EvaluationSerializationError(
                f"EvaluationDataset.RunEnvironment.{field}: expected sanitized text"
            )
    if environment.git_branch is not None:
        branch = _required_text(environment.git_branch, "RunEnvironment.git_branch")
        if not is_sanitized_operational_text(branch):
            raise EvaluationSerializationError(
                "EvaluationDataset.RunEnvironment.git_branch: expected sanitized text"
            )


def save_evaluation_dataset(dataset: EvaluationDataset, output_dir: Path) -> Path:
    """Transactionally create a new evaluation bundle in an absent/empty path."""
    validate_evaluation_dataset(dataset)
    destination = Path(output_dir).absolute()
    _require_empty_destination(destination)
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'evaluation'}.staging-",
                dir=destination.parent,
            )
        )
        _write_new_manifest(staging / MANIFEST_NAME, _encode_dataset(dataset))
        _publish_staging_bundle(staging, destination)
    except EvaluationSerializationError as exc:
        primary_error = exc
        raise
    except (OSError, TypeError, ValueError) as exc:
        wrapped = EvaluationSerializationError(
            f"EvaluationDataset: could not save transactionally: {exc}"
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
                        f"Staging cleanup also failed: {cleanup_error}"
                    )
                else:
                    raise EvaluationSerializationError(
                        "EvaluationDataset: could not clean staging directory"
                    ) from cleanup_error
    return Path(output_dir) / MANIFEST_NAME


def update_evaluation_dataset(dataset: EvaluationDataset, output_dir: Path) -> Path:
    """Atomically replace one manifest while preserving its last valid version."""
    validate_evaluation_dataset(dataset)
    root = Path(output_dir).absolute()
    _validate_bundle_root(root)
    current = load_evaluation_dataset(root)
    if current.run_id != dataset.run_id:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_id: refusing to overwrite a different run"
        )
    validate_evaluation_update(current, dataset)
    payload = _json_payload(_encode_dataset(dataset))
    try:
        stream = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{root.name}.manifest-update-",
            suffix=".json",
            dir=root.parent,
            delete=False,
        )
        temporary = Path(stream.name)
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset: could not create manifest update: {exc}"
        ) from exc
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / MANIFEST_NAME)
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset: atomic manifest update failed: {exc}"
        ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return Path(output_dir) / MANIFEST_NAME


def validate_evaluation_update(
    previous: EvaluationDataset, updated: EvaluationDataset
) -> None:
    """Require one same-run update to preserve history and advance monotonically."""
    validate_evaluation_dataset(previous)
    validate_evaluation_dataset(updated)
    if previous.run_id != updated.run_id:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_id: refusing to overwrite a different run"
        )

    immutable_fields = (
        "source_corruption_dataset_digest",
        "source_dataset_id",
        "evaluator_id",
        "evaluator_version",
        "resolved_evaluator_configuration",
        "evaluator_configuration_digest",
        "base_seed",
        "max_proposals",
        "selected_proposal_ids",
        "schema_version",
    )
    changed = [
        field
        for field in immutable_fields
        if getattr(previous, field) != getattr(updated, field)
    ]
    if changed:
        raise EvaluationSerializationError(
            "EvaluationDataset update changes immutable fields: " + ", ".join(changed)
        )
    if previous.run_manifest.environment != updated.run_manifest.environment:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.environment: cannot change during a run"
        )
    if previous.run_manifest.started_at != updated.run_manifest.started_at:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_manifest.started_at: cannot change during a run"
        )

    previous_evidence = {
        (item.proposal_id, item.attempt_ordinal): item for item in previous.evidence
    }
    updated_evidence = {
        (item.proposal_id, item.attempt_ordinal): item for item in updated.evidence
    }
    for key, prior_evidence in previous_evidence.items():
        current_evidence = updated_evidence.get(key)
        if current_evidence is None:
            raise EvaluationSerializationError(
                f"EvaluationDataset.evidence[{key!r}]: prior evidence cannot be removed"
            )
        if _json_payload(_encode_evidence(prior_evidence)) != _json_payload(
            _encode_evidence(current_evidence)
        ):
            raise EvaluationSerializationError(
                f"EvaluationDataset.evidence[{key!r}]: conflicting prior evidence"
            )

    previous_ledger = {
        (entry.proposal_id, entry.attempt_ordinal): entry for entry in previous.ledger
    }
    updated_ledger = {
        (entry.proposal_id, entry.attempt_ordinal): entry for entry in updated.ledger
    }
    for key, prior_entry in previous_ledger.items():
        current_entry = updated_ledger.get(key)
        if current_entry is None:
            raise EvaluationSerializationError(
                f"EvaluationDataset.ledger[{key!r}]: prior attempts cannot be removed"
            )
        _validate_ledger_transition(prior_entry, current_entry, key)

    new_keys = set(updated_ledger) - set(previous_ledger)
    for key in new_keys:
        if updated_ledger[key].state is not LedgerState.PENDING:
            raise EvaluationSerializationError(
                f"EvaluationDataset.ledger[{key!r}]: new retry attempts must be pending"
            )

    allowed_states = {
        RunState.IN_PROGRESS: {
            RunState.IN_PROGRESS,
            RunState.COMPLETE,
            RunState.INTERRUPTED,
        },
        RunState.INTERRUPTED: {
            RunState.INTERRUPTED,
            RunState.IN_PROGRESS,
            RunState.COMPLETE,
        },
        RunState.COMPLETE: {RunState.COMPLETE, RunState.IN_PROGRESS},
    }
    if updated.run_state not in allowed_states[previous.run_state]:
        raise EvaluationSerializationError(
            "EvaluationDataset.run_state: invalid lifecycle transition "
            f"{previous.run_state.value!r} -> {updated.run_state.value!r}"
        )
    if previous.run_state is RunState.COMPLETE:
        if updated.run_state is RunState.COMPLETE:
            if _json_payload(_encode_dataset(previous)) != _json_payload(
                _encode_dataset(updated)
            ):
                raise EvaluationSerializationError(
                    "EvaluationDataset: a complete run is immutable"
                )
        elif not new_keys:
            raise EvaluationSerializationError(
                "EvaluationDataset: reopening a complete run requires an explicit retry"
            )
    if (
        previous.run_state is RunState.INTERRUPTED
        and updated.run_state is RunState.INTERRUPTED
    ):
        if _json_payload(_encode_dataset(previous)) != _json_payload(
            _encode_dataset(updated)
        ):
            raise EvaluationSerializationError(
                "EvaluationDataset: an interrupted run cannot mutate without resuming"
            )


def _validate_ledger_transition(
    previous: LedgerEntry,
    updated: LedgerEntry,
    key: tuple[str, int],
) -> None:
    if previous.state is LedgerState.PENDING:
        allowed = {LedgerState.PENDING, LedgerState.RUNNING}
    elif previous.state is LedgerState.RUNNING:
        allowed = {LedgerState.RUNNING, *_TERMINAL_LEDGER_STATES}
    else:
        allowed = {previous.state}
    if updated.state not in allowed:
        raise EvaluationSerializationError(
            f"EvaluationDataset.ledger[{key!r}]: invalid state transition "
            f"{previous.state.value!r} -> {updated.state.value!r}"
        )
    if previous.state is updated.state:
        if _json_payload(_encode_ledger_entry(previous)) != _json_payload(
            _encode_ledger_entry(updated)
        ):
            raise EvaluationSerializationError(
                f"EvaluationDataset.ledger[{key!r}]: existing state cannot be mutated"
            )
        return
    if previous.state is LedgerState.RUNNING and (
        updated.started_at != previous.started_at
    ):
        raise EvaluationSerializationError(
            f"EvaluationDataset.ledger[{key!r}].started_at: recovery must preserve it"
        )


def load_evaluation_dataset(
    output_dir: Path,
    corruption_dataset: CorruptionDataset | None = None,
    expected_corruption_digest: str | None = None,
) -> EvaluationDataset:
    """Load strict JSON without pickle and validate every durable relationship."""
    root = Path(output_dir).absolute()
    _validate_bundle_root(root)
    manifest_path = root / MANIFEST_NAME
    _require_regular_unlinked_file(manifest_path, "EvaluationDataset.manifest")
    try:
        raw = cast(
            object,
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
        dataset = _decode_dataset(raw)
        validate_evaluation_dataset(
            dataset,
            corruption_dataset=corruption_dataset,
            expected_corruption_digest=expected_corruption_digest,
        )
        return dataset
    except EvaluationSerializationError:
        raise
    except Exception as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset.manifest: could not load safely: {exc}"
        ) from exc


_TOP_LEVEL_FIELDS = frozenset(
    {
        "format",
        "serialization_version",
        "dataset_schema_version",
        "source_corruption_dataset_digest",
        "source_dataset_id",
        "evaluator_id",
        "evaluator_version",
        "resolved_evaluator_configuration",
        "evaluator_configuration_digest",
        "run_id",
        "base_seed",
        "max_proposals",
        "selected_proposal_ids",
        "evidence",
        "ledger",
        "summary",
        "artifact_references",
        "run_state",
        "run_manifest",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "proposal_id",
        "source_dataset_id",
        "source_episode_id",
        "source_candidate_id",
        "split_group_id",
        "evaluator_id",
        "evaluator_version",
        "evaluator_configuration_digest",
        "evaluation_seed",
        "attempt_ordinal",
        "status",
        "success",
        "progress_before",
        "progress_after",
        "progress_delta",
        "unsafe",
        "failure_events",
        "termination_reason",
        "replayed_control_steps",
        "metrics",
        "artifact_references",
        "label_source",
        "label_strength",
        "simulator_replay_verified",
        "notes",
        "schema_version",
    }
)
_FAILURE_FIELDS = frozenset(
    {
        "failure_type",
        "timestamp_s",
        "probability",
        "description",
        "schema_version",
    }
)
_LEDGER_FIELDS = frozenset(
    {
        "proposal_id",
        "evidence_id",
        "attempt_ordinal",
        "evaluator_id",
        "evaluator_version",
        "evaluator_configuration_digest",
        "evaluation_seed",
        "state",
        "error_type",
        "error_message",
        "retry_eligible",
        "started_at",
        "finished_at",
        "schema_version",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "conclusive",
        "indeterminate",
        "invalid",
        "skipped",
        "execution_error",
        "projected_outcome_labels",
        "retried_attempts",
        "total_attempts",
    }
)
_RUN_MANIFEST_FIELDS = frozenset(
    {
        "environment",
        "evaluator_id",
        "evaluator_version",
        "evaluator_configuration_digest",
        "source_corruption_dataset_digest",
        "source_dataset_id",
        "seed",
        "started_at",
        "finished_at",
        "final_state",
        "schema_version",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "git_commit_sha",
        "git_branch",
        "python_version",
        "numpy_version",
        "platform",
        "launch_command",
    }
)


def _encode_dataset(dataset: EvaluationDataset) -> dict[str, object]:
    return {
        "format": EVALUATION_SERIALIZATION_FORMAT,
        "serialization_version": EVALUATION_SERIALIZATION_VERSION,
        "dataset_schema_version": dataset.schema_version,
        "source_corruption_dataset_digest": dataset.source_corruption_dataset_digest,
        "source_dataset_id": dataset.source_dataset_id,
        "evaluator_id": dataset.evaluator_id,
        "evaluator_version": dataset.evaluator_version,
        "resolved_evaluator_configuration": _thaw_json_value(
            dataset.resolved_evaluator_configuration
        ),
        "evaluator_configuration_digest": dataset.evaluator_configuration_digest,
        "run_id": dataset.run_id,
        "base_seed": dataset.base_seed,
        "max_proposals": dataset.max_proposals,
        "selected_proposal_ids": list(dataset.selected_proposal_ids),
        "evidence": [_encode_evidence(item) for item in dataset.evidence],
        "ledger": [_encode_ledger_entry(item) for item in dataset.ledger],
        "summary": {
            "conclusive": dataset.summary.conclusive,
            "indeterminate": dataset.summary.indeterminate,
            "invalid": dataset.summary.invalid,
            "skipped": dataset.summary.skipped,
            "execution_error": dataset.summary.execution_error,
            "projected_outcome_labels": dataset.summary.projected_outcome_labels,
            "retried_attempts": dataset.summary.retried_attempts,
            "total_attempts": dataset.summary.total_attempts,
        },
        "artifact_references": list(dataset.artifact_references),
        "run_state": dataset.run_state.value,
        "run_manifest": _encode_run_manifest(dataset.run_manifest),
    }


def _encode_evidence(evidence: EvaluationEvidence) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "proposal_id": evidence.proposal_id,
        "source_dataset_id": evidence.source_dataset_id,
        "source_episode_id": evidence.source_episode_id,
        "source_candidate_id": evidence.source_candidate_id,
        "split_group_id": evidence.split_group_id,
        "evaluator_id": evidence.evaluator_id,
        "evaluator_version": evidence.evaluator_version,
        "evaluator_configuration_digest": evidence.evaluator_configuration_digest,
        "evaluation_seed": evidence.evaluation_seed,
        "attempt_ordinal": evidence.attempt_ordinal,
        "status": evidence.status.value,
        "success": evidence.success,
        "progress_before": evidence.progress_before,
        "progress_after": evidence.progress_after,
        "progress_delta": evidence.progress_delta,
        "unsafe": evidence.unsafe,
        "failure_events": [
            {
                "failure_type": item.failure_type,
                "timestamp_s": item.timestamp_s,
                "probability": item.probability,
                "description": item.description,
                "schema_version": item.schema_version,
            }
            for item in evidence.failure_events
        ],
        "termination_reason": evidence.termination_reason,
        "replayed_control_steps": evidence.replayed_control_steps,
        "metrics": dict(evidence.metrics),
        "artifact_references": list(evidence.artifact_references),
        "label_source": (
            evidence.label_source.value if evidence.label_source is not None else None
        ),
        "label_strength": (
            evidence.label_strength.value
            if evidence.label_strength is not None
            else None
        ),
        "simulator_replay_verified": evidence.simulator_replay_verified,
        "notes": evidence.notes,
        "schema_version": evidence.schema_version,
    }


def compute_evaluation_evidence_content_digest(
    evidence: EvaluationEvidence,
) -> str:
    """Hash every persisted evidence field, including task values and metrics."""

    validate_evaluation_evidence(evidence)
    encoded = _canonical_json_bytes(_encode_evidence(evidence))
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def compute_evaluation_dataset_content_digest(dataset: EvaluationDataset) -> str:
    """Hash the complete path-independent persisted run, ledger, and evidence."""

    validate_evaluation_dataset(dataset)
    encoded = _canonical_json_bytes(_encode_dataset(dataset))
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _encode_ledger_entry(entry: LedgerEntry) -> dict[str, object]:
    return {
        "proposal_id": entry.proposal_id,
        "evidence_id": entry.evidence_id,
        "attempt_ordinal": entry.attempt_ordinal,
        "evaluator_id": entry.evaluator_id,
        "evaluator_version": entry.evaluator_version,
        "evaluator_configuration_digest": entry.evaluator_configuration_digest,
        "evaluation_seed": entry.evaluation_seed,
        "state": entry.state.value,
        "error_type": entry.error_type,
        "error_message": entry.error_message,
        "retry_eligible": entry.retry_eligible,
        "started_at": entry.started_at,
        "finished_at": entry.finished_at,
        "schema_version": entry.schema_version,
    }


def _encode_run_manifest(manifest: RunManifest) -> dict[str, object]:
    environment = manifest.environment
    return {
        "environment": {
            "git_commit_sha": environment.git_commit_sha,
            "git_branch": environment.git_branch,
            "python_version": environment.python_version,
            "numpy_version": environment.numpy_version,
            "platform": environment.platform,
            "launch_command": environment.launch_command,
        },
        "evaluator_id": manifest.evaluator_id,
        "evaluator_version": manifest.evaluator_version,
        "evaluator_configuration_digest": manifest.evaluator_configuration_digest,
        "source_corruption_dataset_digest": manifest.source_corruption_dataset_digest,
        "source_dataset_id": manifest.source_dataset_id,
        "seed": manifest.seed,
        "started_at": manifest.started_at,
        "finished_at": manifest.finished_at,
        "final_state": manifest.final_state.value,
        "schema_version": manifest.schema_version,
    }


def _decode_dataset(value: object) -> EvaluationDataset:
    context = "EvaluationDataset.manifest"
    item = _mapping(value, context)
    _require_exact_fields(item, _TOP_LEVEL_FIELDS, context)
    if _string(item, "format", context) != EVALUATION_SERIALIZATION_FORMAT:
        raise EvaluationSerializationError(f"{context}.format: unsupported format")
    version = _integer(item, "serialization_version", context)
    if version != EVALUATION_SERIALIZATION_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            f"{context}.serialization_version: unsupported version {version!r}"
        )
    schema_version = _string(item, "dataset_schema_version", context)
    if schema_version != EVALUATION_DATASET_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            f"{context}.dataset_schema_version: unsupported version {schema_version!r}"
        )
    raw_configuration = _mapping(
        _field(item, "resolved_evaluator_configuration", context),
        f"{context}.resolved_evaluator_configuration",
    )
    configuration = cast(
        Mapping[str, object],
        _freeze_json_value(raw_configuration, f"{context}.resolved_configuration"),
    )
    return EvaluationDataset(
        source_corruption_dataset_digest=_string(
            item, "source_corruption_dataset_digest", context
        ),
        source_dataset_id=_string(item, "source_dataset_id", context),
        evaluator_id=_string(item, "evaluator_id", context),
        evaluator_version=_string(item, "evaluator_version", context),
        resolved_evaluator_configuration=configuration,
        evaluator_configuration_digest=_string(
            item, "evaluator_configuration_digest", context
        ),
        run_id=_string(item, "run_id", context),
        base_seed=_integer(item, "base_seed", context),
        max_proposals=_optional_integer(item, "max_proposals", context),
        selected_proposal_ids=tuple(
            _string_value(raw, f"{context}.selected_proposal_ids[{index}]")
            for index, raw in enumerate(_list(item, "selected_proposal_ids", context))
        ),
        evidence=tuple(
            _decode_evidence(raw, index)
            for index, raw in enumerate(_list(item, "evidence", context))
        ),
        ledger=tuple(
            _decode_ledger_entry(raw, index)
            for index, raw in enumerate(_list(item, "ledger", context))
        ),
        summary=_decode_summary(_field(item, "summary", context)),
        artifact_references=tuple(
            _string_value(raw, f"{context}.artifact_references[{index}]")
            for index, raw in enumerate(_list(item, "artifact_references", context))
        ),
        run_state=_enum_value(
            RunState, _field(item, "run_state", context), f"{context}.run_state"
        ),
        run_manifest=_decode_run_manifest(_field(item, "run_manifest", context)),
        schema_version=schema_version,
    )


def _decode_evidence(value: object, index: int) -> EvaluationEvidence:
    context = f"EvaluationDataset.evidence[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _EVIDENCE_FIELDS, context)
    schema_version = _string(item, "schema_version", context)
    if schema_version != EVALUATION_EVIDENCE_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            f"{context}.schema_version: unsupported version {schema_version!r}"
        )
    failures = tuple(
        _decode_failure_event(raw, context, failure_index)
        for failure_index, raw in enumerate(_list(item, "failure_events", context))
    )
    metrics = _decode_scalar_mapping(
        _field(item, "metrics", context), f"{context}.metrics"
    )
    return EvaluationEvidence(
        evidence_id=_string(item, "evidence_id", context),
        proposal_id=_string(item, "proposal_id", context),
        source_dataset_id=_string(item, "source_dataset_id", context),
        source_episode_id=_string(item, "source_episode_id", context),
        source_candidate_id=_string(item, "source_candidate_id", context),
        split_group_id=_string(item, "split_group_id", context),
        evaluator_id=_string(item, "evaluator_id", context),
        evaluator_version=_string(item, "evaluator_version", context),
        evaluator_configuration_digest=_string(
            item, "evaluator_configuration_digest", context
        ),
        evaluation_seed=_integer(item, "evaluation_seed", context),
        attempt_ordinal=_integer(item, "attempt_ordinal", context),
        status=_enum_value(
            EvaluationStatus, _field(item, "status", context), f"{context}.status"
        ),
        success=_optional_boolean(item, "success", context),
        progress_before=_optional_number(item, "progress_before", context),
        progress_after=_optional_number(item, "progress_after", context),
        progress_delta=_optional_number(item, "progress_delta", context),
        unsafe=_optional_boolean(item, "unsafe", context),
        failure_events=failures,
        termination_reason=_optional_string(item, "termination_reason", context),
        replayed_control_steps=_integer(item, "replayed_control_steps", context),
        metrics=metrics,
        artifact_references=tuple(
            _string_value(raw, f"{context}.artifact_references[{ref_index}]")
            for ref_index, raw in enumerate(_list(item, "artifact_references", context))
        ),
        label_source=_optional_enum(
            LabelSource,
            _field(item, "label_source", context),
            f"{context}.label_source",
        ),
        label_strength=_optional_enum(
            LabelStrength,
            _field(item, "label_strength", context),
            f"{context}.label_strength",
        ),
        simulator_replay_verified=_boolean(item, "simulator_replay_verified", context),
        notes=_optional_string(item, "notes", context),
        schema_version=schema_version,
    )


def _decode_failure_event(
    value: object, evidence_context: str, index: int
) -> FailureEvent:
    context = f"{evidence_context}.failure_events[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _FAILURE_FIELDS, context)
    schema_version = _string(item, "schema_version", context)
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            f"{context}.schema_version: unsupported version {schema_version!r}"
        )
    return FailureEvent(
        failure_type=_string(item, "failure_type", context),
        timestamp_s=_optional_number(item, "timestamp_s", context),
        probability=_optional_number(item, "probability", context),
        description=_optional_string(item, "description", context),
        schema_version=schema_version,
    )


def _decode_ledger_entry(value: object, index: int) -> LedgerEntry:
    context = f"EvaluationDataset.ledger[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _LEDGER_FIELDS, context)
    schema_version = _string(item, "schema_version", context)
    if schema_version != LEDGER_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            f"{context}.schema_version: unsupported version {schema_version!r}"
        )
    return LedgerEntry(
        proposal_id=_string(item, "proposal_id", context),
        evidence_id=_optional_string(item, "evidence_id", context),
        attempt_ordinal=_integer(item, "attempt_ordinal", context),
        evaluator_id=_string(item, "evaluator_id", context),
        evaluator_version=_string(item, "evaluator_version", context),
        evaluator_configuration_digest=_string(
            item, "evaluator_configuration_digest", context
        ),
        evaluation_seed=_integer(item, "evaluation_seed", context),
        state=_enum_value(
            LedgerState, _field(item, "state", context), f"{context}.state"
        ),
        error_type=_optional_string(item, "error_type", context),
        error_message=_optional_string(item, "error_message", context),
        retry_eligible=_boolean(item, "retry_eligible", context),
        started_at=_optional_string(item, "started_at", context),
        finished_at=_optional_string(item, "finished_at", context),
        schema_version=schema_version,
    )


def _decode_summary(value: object) -> EvaluationSummary:
    context = "EvaluationDataset.summary"
    item = _mapping(value, context)
    _require_exact_fields(item, _SUMMARY_FIELDS, context)
    summary = EvaluationSummary(
        conclusive=_integer(item, "conclusive", context),
        indeterminate=_integer(item, "indeterminate", context),
        invalid=_integer(item, "invalid", context),
        skipped=_integer(item, "skipped", context),
        execution_error=_integer(item, "execution_error", context),
        projected_outcome_labels=_integer(item, "projected_outcome_labels", context),
        retried_attempts=_integer(item, "retried_attempts", context),
        total_attempts=_integer(item, "total_attempts", context),
    )
    if any(
        value < 0
        for value in (
            summary.conclusive,
            summary.indeterminate,
            summary.invalid,
            summary.skipped,
            summary.execution_error,
            summary.projected_outcome_labels,
            summary.retried_attempts,
            summary.total_attempts,
        )
    ):
        raise EvaluationSerializationError(f"{context}: counts must be non-negative")
    return summary


def _decode_run_manifest(value: object) -> RunManifest:
    context = "EvaluationDataset.run_manifest"
    item = _mapping(value, context)
    _require_exact_fields(item, _RUN_MANIFEST_FIELDS, context)
    environment_value = _mapping(
        _field(item, "environment", context), f"{context}.environment"
    )
    _require_exact_fields(
        environment_value, _ENVIRONMENT_FIELDS, f"{context}.environment"
    )
    environment = RunEnvironment(
        git_commit_sha=_optional_string(
            environment_value, "git_commit_sha", f"{context}.environment"
        ),
        git_branch=_optional_string(
            environment_value, "git_branch", f"{context}.environment"
        ),
        python_version=_string(
            environment_value, "python_version", f"{context}.environment"
        ),
        numpy_version=_string(
            environment_value, "numpy_version", f"{context}.environment"
        ),
        platform=_string(environment_value, "platform", f"{context}.environment"),
        launch_command=_string(
            environment_value, "launch_command", f"{context}.environment"
        ),
    )
    schema_version = _string(item, "schema_version", context)
    if schema_version != RUN_MANIFEST_SCHEMA_VERSION:
        raise UnsupportedEvaluationSerializationVersionError(
            f"{context}.schema_version: unsupported version {schema_version!r}"
        )
    return RunManifest(
        environment=environment,
        evaluator_id=_string(item, "evaluator_id", context),
        evaluator_version=_string(item, "evaluator_version", context),
        evaluator_configuration_digest=_string(
            item, "evaluator_configuration_digest", context
        ),
        source_corruption_dataset_digest=_string(
            item, "source_corruption_dataset_digest", context
        ),
        source_dataset_id=_string(item, "source_dataset_id", context),
        seed=_integer(item, "seed", context),
        started_at=_string(item, "started_at", context),
        finished_at=_optional_string(item, "finished_at", context),
        final_state=_enum_value(
            RunState, _field(item, "final_state", context), f"{context}.final_state"
        ),
        schema_version=schema_version,
    )


def _decode_scalar_mapping(value: object, context: str) -> Mapping[str, JsonScalar]:
    item = _mapping(value, context)
    result: dict[str, JsonScalar] = {}
    for key, scalar in item.items():
        if not key:
            raise EvaluationSerializationError(f"{context}: keys must be non-empty")
        if scalar is None or type(scalar) in (str, int, bool):
            result[key] = cast(JsonScalar, scalar)
        elif type(scalar) is float and math.isfinite(scalar):
            result[key] = scalar
        else:
            raise EvaluationSerializationError(
                f"{context}.{key}: expected a finite JSON scalar"
            )
    return MappingProxyType(result)


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            _thaw_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationSerializationError(
            f"canonical identity inputs are not JSON values: {exc}"
        ) from exc


def _json_payload(value: object) -> str:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset.manifest: could not encode JSON: {exc}"
        ) from exc


def _write_new_manifest(path: Path, value: object) -> None:
    payload = _json_payload(value)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _require_empty_destination(destination: Path) -> None:
    try:
        unsafe = destination.is_symlink() or destination.resolve() != destination
    except OSError as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset.output_dir: could not inspect destination: {exc}"
        ) from exc
    if unsafe:
        raise EvaluationSerializationError(
            "EvaluationDataset.output_dir: symbolic links and junctions are unsupported"
        )
    if not destination.exists():
        return
    if not destination.is_dir():
        raise EvaluationSerializationError(
            "EvaluationDataset.output_dir: must be a directory"
        )
    try:
        next(destination.iterdir())
    except StopIteration:
        return
    except OSError as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset.output_dir: could not inspect destination: {exc}"
        ) from exc
    raise EvaluationSerializationError(
        "EvaluationDataset.output_dir: destination must be absent or empty"
    )


def _validate_bundle_root(root: Path) -> None:
    try:
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise EvaluationSerializationError(
                "EvaluationDataset.output_dir: missing or unsafe dataset directory"
            )
        entries = {entry.name: entry for entry in root.iterdir()}
    except EvaluationSerializationError:
        raise
    except OSError as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset.output_dir: could not inspect bundle: {exc}"
        ) from exc
    if set(entries) != {MANIFEST_NAME}:
        raise EvaluationSerializationError(
            "EvaluationDataset.output_dir: bundle inventory is incomplete or has extras"
        )


def _require_regular_unlinked_file(path: Path, context: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise EvaluationSerializationError(
                f"{context}: missing or unsafe regular file"
            )
        if path.stat().st_nlink != 1:
            raise EvaluationSerializationError(f"{context}: hard links are unsafe")
    except EvaluationSerializationError:
        raise
    except OSError as exc:
        raise EvaluationSerializationError(
            f"{context}: could not inspect: {exc}"
        ) from exc


def _publish_staging_bundle(staging: Path, destination: Path) -> None:
    existed = destination.exists()
    if existed:
        _require_empty_destination(destination)
        destination.rmdir()
    try:
        staging.replace(destination)
    except OSError:
        if existed and not destination.exists():
            destination.mkdir()
        raise


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationSerializationError(
            f"EvaluationDataset.{field}: expected a non-empty string without whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise EvaluationSerializationError(
            f"EvaluationDataset.{field}: control characters are unsupported"
        )
    return value


def _validate_seed(value: object, field: str) -> None:
    if type(value) is not int or not 0 <= value < _MAX_SEED:
        raise EvaluationSerializationError(
            f"EvaluationDataset.{field}: expected an integer in [0, {_MAX_SEED})"
        )


def _validate_ordinal(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise EvaluationSerializationError(
            f"EvaluationDataset.{field}: expected a non-negative integer"
        )


def _validate_max_proposals(value: object) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise EvaluationSerializationError(
            "EvaluationDataset.max_proposals: expected a positive integer or null"
        )


def _validate_timestamp(value: object, field: str) -> datetime:
    text = _required_text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvaluationSerializationError(
            f"EvaluationDataset.{field}: expected an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise EvaluationSerializationError(
            f"EvaluationDataset.{field}: timestamp must include a timezone"
        )
    return parsed


def _require_exact_fields(
    item: Mapping[str, object], expected: Collection[str], context: str
) -> None:
    missing = sorted(set(expected) - set(item))
    unexpected = sorted(set(item) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise EvaluationSerializationError(
            f"{context}: invalid fields ({'; '.join(details)})"
        )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluationSerializationError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _field(item: Mapping[str, object], field: str, context: str) -> object:
    if field not in item:
        raise EvaluationSerializationError(f"{context}.{field}: missing required field")
    return item[field]


def _string_value(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise EvaluationSerializationError(f"{context}: expected a string")
    return value


def _string(item: Mapping[str, object], field: str, context: str) -> str:
    return _string_value(_field(item, field, context), f"{context}.{field}")


def _optional_string(
    item: Mapping[str, object], field: str, context: str
) -> str | None:
    value = _field(item, field, context)
    if value is None:
        return None
    return _string_value(value, f"{context}.{field}")


def _integer(item: Mapping[str, object], field: str, context: str) -> int:
    value = _field(item, field, context)
    if type(value) is not int:
        raise EvaluationSerializationError(f"{context}.{field}: expected an integer")
    return value


def _optional_integer(
    item: Mapping[str, object], field: str, context: str
) -> int | None:
    value = _field(item, field, context)
    if value is None:
        return None
    if type(value) is not int:
        raise EvaluationSerializationError(
            f"{context}.{field}: expected an integer or null"
        )
    return value


def _number_value(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise EvaluationSerializationError(f"{context}: expected a finite number")
    try:
        number = float(cast(int | float, value))
    except OverflowError as exc:
        raise EvaluationSerializationError(f"{context}: number is too large") from exc
    if not math.isfinite(number):
        raise EvaluationSerializationError(f"{context}: expected a finite number")
    if isinstance(value, int) and int(number) != value:
        raise EvaluationSerializationError(
            f"{context}: integer cannot be represented exactly as a float"
        )
    return number


def _optional_number(
    item: Mapping[str, object], field: str, context: str
) -> float | None:
    value = _field(item, field, context)
    if value is None:
        return None
    return _number_value(value, f"{context}.{field}")


def _boolean(item: Mapping[str, object], field: str, context: str) -> bool:
    value = _field(item, field, context)
    if type(value) is not bool:
        raise EvaluationSerializationError(f"{context}.{field}: expected a boolean")
    return value


def _optional_boolean(
    item: Mapping[str, object], field: str, context: str
) -> bool | None:
    value = _field(item, field, context)
    if value is None:
        return None
    if type(value) is not bool:
        raise EvaluationSerializationError(
            f"{context}.{field}: expected a boolean or null"
        )
    return value


def _list(item: Mapping[str, object], field: str, context: str) -> list[object]:
    value = _field(item, field, context)
    if not isinstance(value, list):
        raise EvaluationSerializationError(f"{context}.{field}: expected an array")
    return cast(list[object], value)


def _enum_value(enum_type: type[_StrEnumT], value: object, context: str) -> _StrEnumT:
    if not isinstance(value, str):
        raise EvaluationSerializationError(f"{context}: expected a string enum")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise EvaluationSerializationError(
            f"{context}: unsupported value {value!r}"
        ) from exc


def _optional_enum(
    enum_type: type[_StrEnumT], value: object, context: str
) -> _StrEnumT | None:
    if value is None:
        return None
    return _enum_value(enum_type, value, context)


def _reject_json_constant(value: str) -> NoReturn:
    raise EvaluationSerializationError(
        f"EvaluationDataset.manifest: non-finite constant {value!r} is unsupported"
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationSerializationError(f"JSON object: duplicate field {key!r}")
        result[key] = value
    return result


__all__ = [
    "EVALUATION_DATASET_SCHEMA_VERSION",
    "EVALUATION_SERIALIZATION_FORMAT",
    "EVALUATION_SERIALIZATION_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "EvaluationDataset",
    "EvaluationSerializationError",
    "EvaluationSummary",
    "LedgerEntry",
    "LedgerState",
    "RunEnvironment",
    "RunManifest",
    "RunState",
    "UnsupportedEvaluationSerializationVersionError",
    "compute_corruption_dataset_content_digest",
    "compute_corruption_dataset_digest",
    "compute_evaluation_dataset_content_digest",
    "compute_evaluation_evidence_content_digest",
    "compute_evaluation_seed",
    "compute_run_identifier",
    "load_evaluation_dataset",
    "save_evaluation_dataset",
    "update_evaluation_dataset",
    "validate_evaluation_update",
    "validate_evaluation_dataset",
]
