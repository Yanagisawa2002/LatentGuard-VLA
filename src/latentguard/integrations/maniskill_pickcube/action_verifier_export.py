"""Export strongly verified PickCube replay evidence into compact verifier data."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from latentguard.action_verifier import (
    FIXED_ACTION_CHUNK_HORIZON,
    ActionVerifierCandidateGroupV1,
    ActionVerifierDatasetV1,
    ActionVerifierSampleV1,
    CandidateType,
    DatasetSplit,
    TrajectorySplitSourceV1,
    assign_trajectory_splits,
    load_action_verifier_dataset,
    validate_action_verifier_dataset,
    validate_action_verifier_evidence_references,
)
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    load_corruption_dataset,
    validate_corruption_dataset,
)
from latentguard.evaluation.models import (
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
)
from latentguard.evaluation.serialization import (
    EvaluationDataset,
    RunState,
    compute_corruption_dataset_content_digest,
    load_evaluation_dataset,
    validate_evaluation_dataset,
)
from latentguard.integrations.maniskill_pickcube.adapter import (
    MANISKILL_PICKCUBE_ADAPTER_ID,
    MANISKILL_PICKCUBE_ADAPTER_VERSION,
)
from latentguard.integrations.maniskill_pickcube.source_import import (
    MANISKILL_PICKCUBE_TASK_ID,
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_UNSAFE_SEMANTIC,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    ANCHOR_SOURCE_TRANSFORMATION,
    PICKCUBE_STATE_COMPARISON_SEMANTIC,
    PICKCUBE_STATE_COMPARISON_TOLERANCE,
    AnchorBaselineEvidenceV1,
    PickCubeAnchorManifestV1,
    PickCubeAnchorSourceRecordV1,
    load_anchor_manifest,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_DTYPE,
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
)
from latentguard.models import CandidateAction, Episode, LabelSource, LabelStrength
from latentguard.replay.identity import canonical_json_bytes
from latentguard.replay.models import ReplayTrustTier, StateComparisonSemantic
from latentguard.replay.source import compute_episode_content_digest
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    load_episodes,
)
from latentguard.validation import DataValidationError, validate_episodes

ACTION_VERIFIER_EXPORT_SCHEMA_VERSION = "1.0"
ACTION_VERIFIER_SUMMARY_NAME = "summary.json"
ACTION_VERIFIER_VALIDATION_NAME = "validation.json"

_PREFIX_PROGRESS_SEMANTIC = "pickcube_candidate_progress_semantic"
_PREFIX_PROGRESS_BEFORE = "pickcube_progress_before_candidate"
_PREFIX_PROGRESS_AFTER = "pickcube_progress_after_candidate"
_PREFIX_PROGRESS_DELTA = "pickcube_progress_delta_candidate"
_PREFIX_EVALUATED = "pickcube_prefix_evaluated_before_continuation"
_PREFIX_HORIZON = "pickcube_candidate_horizon_steps"


class ActionVerifierExportError(ValueError):
    """Raised when source, corruption, or evidence bindings fail closed."""


def _freeze_counts(value: Mapping[str, int], context: str) -> Mapping[str, int]:
    frozen: dict[str, int] = {}
    for key, count in sorted(value.items()):
        if not isinstance(key, str) or not key or key != key.strip():
            raise ActionVerifierExportError(f"{context}: invalid count key")
        if type(count) is not int or count < 0:
            raise ActionVerifierExportError(f"{context}.{key}: invalid count")
        frozen[key] = count
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ActionVerifierExportSummaryV1:
    """Compact deterministic accounting for one verifier export."""

    source_trajectory_count: int
    manifest_anchor_count: int
    included_anchor_count: int
    anchors_without_conclusive_corruption: int
    total_proposal_count: int
    selected_proposal_count: int
    evaluation_attempt_count: int
    source_sample_count: int
    corrupted_sample_count: int
    corrupted_success_count: int
    corrupted_failure_count: int
    corrupted_unsafe_count: int
    candidate_group_count: int
    dataset_sample_count: int
    state_vector_dimension: int
    action_dimension: int
    chunk_horizon: int
    progress_semantic: str
    evidence_reference_count: int
    source_dataset_id: str
    source_dataset_digest: str
    corruption_dataset_digest: str
    evidence_dataset_digest: str
    dataset_content_digest: str
    split_trajectory_counts: Mapping[str, int]
    split_sample_counts: Mapping[str, int]
    evaluation_status_counts: Mapping[str, int]
    exclusion_reason_counts: Mapping[str, int]
    corruption_type_counts: Mapping[str, int]
    severity_counts: Mapping[str, int]
    schema_version: str = ACTION_VERIFIER_EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze count mappings and validate exact class accounting."""

        for field in (
            "source_trajectory_count",
            "manifest_anchor_count",
            "included_anchor_count",
            "anchors_without_conclusive_corruption",
            "total_proposal_count",
            "selected_proposal_count",
            "evaluation_attempt_count",
            "source_sample_count",
            "corrupted_sample_count",
            "corrupted_success_count",
            "corrupted_failure_count",
            "corrupted_unsafe_count",
            "candidate_group_count",
            "dataset_sample_count",
            "state_vector_dimension",
            "action_dimension",
            "chunk_horizon",
            "evidence_reference_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ActionVerifierExportError(f"summary.{field}: invalid count")
        if self.corrupted_success_count + self.corrupted_failure_count != (
            self.corrupted_sample_count
        ):
            raise ActionVerifierExportError("summary corrupted class counts disagree")
        if self.source_sample_count != self.included_anchor_count:
            raise ActionVerifierExportError("summary source/anchor counts disagree")
        if self.candidate_group_count != self.included_anchor_count:
            raise ActionVerifierExportError("summary group/anchor counts disagree")
        if self.dataset_sample_count != (
            self.source_sample_count + self.corrupted_sample_count
        ):
            raise ActionVerifierExportError("summary dataset sample count disagrees")
        if self.schema_version != ACTION_VERIFIER_EXPORT_SCHEMA_VERSION:
            raise ActionVerifierExportError("unsupported export summary schema")
        for field in (
            "split_trajectory_counts",
            "split_sample_counts",
            "evaluation_status_counts",
            "exclusion_reason_counts",
            "corruption_type_counts",
            "severity_counts",
        ):
            object.__setattr__(
                self,
                field,
                _freeze_counts(cast(Mapping[str, int], getattr(self, field)), field),
            )

    def as_mapping(self) -> Mapping[str, object]:
        """Return a JSON-ready compact summary."""

        return MappingProxyType(
            {
                "action_dimension": self.action_dimension,
                "anchors_without_conclusive_corruption": (
                    self.anchors_without_conclusive_corruption
                ),
                "candidate_group_count": self.candidate_group_count,
                "chunk_horizon": self.chunk_horizon,
                "corrupted_failure_count": self.corrupted_failure_count,
                "corrupted_sample_count": self.corrupted_sample_count,
                "corrupted_success_count": self.corrupted_success_count,
                "corrupted_unsafe_count": self.corrupted_unsafe_count,
                "corruption_dataset_digest": self.corruption_dataset_digest,
                "corruption_type_counts": dict(self.corruption_type_counts),
                "dataset_content_digest": self.dataset_content_digest,
                "dataset_sample_count": self.dataset_sample_count,
                "evaluation_attempt_count": self.evaluation_attempt_count,
                "evaluation_status_counts": dict(self.evaluation_status_counts),
                "evidence_dataset_digest": self.evidence_dataset_digest,
                "evidence_reference_count": self.evidence_reference_count,
                "exclusion_reason_counts": dict(self.exclusion_reason_counts),
                "included_anchor_count": self.included_anchor_count,
                "manifest_anchor_count": self.manifest_anchor_count,
                "progress_semantic": self.progress_semantic,
                "schema_version": self.schema_version,
                "selected_proposal_count": self.selected_proposal_count,
                "severity_counts": dict(self.severity_counts),
                "source_dataset_digest": self.source_dataset_digest,
                "source_dataset_id": self.source_dataset_id,
                "source_sample_count": self.source_sample_count,
                "source_trajectory_count": self.source_trajectory_count,
                "split_sample_counts": dict(self.split_sample_counts),
                "split_trajectory_counts": dict(self.split_trajectory_counts),
                "state_vector_dimension": self.state_vector_dimension,
                "total_proposal_count": self.total_proposal_count,
            }
        )


@dataclass(frozen=True, slots=True)
class ActionVerifierValidationReportV1:
    """Independent validation result for in-memory or reloaded verifier data."""

    validation_scope: str
    valid: bool
    leakage_valid: bool
    evidence_references_valid: bool
    sample_count: int
    source_sample_count: int
    corrupted_sample_count: int
    corrupted_success_count: int
    corrupted_failure_count: int
    candidate_group_count: int
    source_trajectory_count: int
    evidence_reference_count: int
    split_trajectory_counts: Mapping[str, int]
    split_sample_counts: Mapping[str, int]
    evidence_dataset_digest: str
    dataset_content_digest: str
    schema_version: str = ACTION_VERIFIER_EXPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze report mappings and reject internally inconsistent reports."""

        if self.validation_scope not in {"in_memory", "serialized_reload"}:
            raise ActionVerifierExportError("validation report scope is unsupported")
        if self.valid is not True or self.leakage_valid is not True:
            raise ActionVerifierExportError("only passing validation reports are valid")
        if self.evidence_references_valid is not True:
            raise ActionVerifierExportError("evidence references were not validated")
        if self.schema_version != ACTION_VERIFIER_EXPORT_SCHEMA_VERSION:
            raise ActionVerifierExportError("unsupported validation report schema")
        object.__setattr__(
            self,
            "split_trajectory_counts",
            _freeze_counts(self.split_trajectory_counts, "split_trajectory_counts"),
        )
        object.__setattr__(
            self,
            "split_sample_counts",
            _freeze_counts(self.split_sample_counts, "split_sample_counts"),
        )

    def as_mapping(self) -> Mapping[str, object]:
        """Return a JSON-ready validation report."""

        return MappingProxyType(
            {
                "candidate_group_count": self.candidate_group_count,
                "corrupted_failure_count": self.corrupted_failure_count,
                "corrupted_sample_count": self.corrupted_sample_count,
                "corrupted_success_count": self.corrupted_success_count,
                "dataset_content_digest": self.dataset_content_digest,
                "evidence_dataset_digest": self.evidence_dataset_digest,
                "evidence_reference_count": self.evidence_reference_count,
                "evidence_references_valid": self.evidence_references_valid,
                "leakage_valid": self.leakage_valid,
                "sample_count": self.sample_count,
                "schema_version": self.schema_version,
                "source_sample_count": self.source_sample_count,
                "source_trajectory_count": self.source_trajectory_count,
                "split_sample_counts": dict(self.split_sample_counts),
                "split_trajectory_counts": dict(self.split_trajectory_counts),
                "valid": self.valid,
                "validation_scope": self.validation_scope,
            }
        )


@dataclass(frozen=True, slots=True)
class ActionVerifierExportResultV1:
    """Dataset plus compact accounting and exact evidence inventory."""

    dataset: ActionVerifierDatasetV1
    summary: ActionVerifierExportSummaryV1
    validation_report: ActionVerifierValidationReportV1
    available_evidence_ids: tuple[str, ...]
    evidence_dataset_digest: str

    def __post_init__(self) -> None:
        """Require deterministic foreign-key inventory and matching digests."""

        evidence_ids = tuple(self.available_evidence_ids)
        if evidence_ids != tuple(sorted(evidence_ids)) or len(evidence_ids) != len(
            set(evidence_ids)
        ):
            raise ActionVerifierExportError(
                "available evidence identifiers must be sorted and unique"
            )
        if self.summary.dataset_content_digest != self.dataset.content_digest:
            raise ActionVerifierExportError("summary dataset digest mismatch")
        if self.evidence_dataset_digest != self.summary.evidence_dataset_digest:
            raise ActionVerifierExportError("summary evidence digest mismatch")
        object.__setattr__(self, "available_evidence_ids", evidence_ids)


@dataclass(frozen=True, slots=True)
class _EligibleCorruption:
    proposal: CorruptedActionProposal
    evidence: EvaluationEvidence
    progress_semantic: str
    progress_before: float
    progress_after: float
    progress_delta: float
    severity_id: str


@dataclass(frozen=True, slots=True)
class _EligibleAnchor:
    record: PickCubeAnchorSourceRecordV1
    episode: Episode
    source_candidate: CandidateAction
    baseline: AnchorBaselineEvidenceV1
    corruptions: tuple[_EligibleCorruption, ...]


def _failure_event_payload(event: object) -> Mapping[str, object]:
    try:
        return {
            "description": cast(Any, event).description,
            "failure_type": cast(Any, event).failure_type,
            "probability": cast(Any, event).probability,
            "schema_version": cast(Any, event).schema_version,
            "timestamp_s": cast(Any, event).timestamp_s,
        }
    except AttributeError as exc:
        raise ActionVerifierExportError(
            "evidence contains an invalid failure event"
        ) from exc


def _evaluation_evidence_payload(evidence: EvaluationEvidence) -> Mapping[str, object]:
    """Return path-independent task and execution evidence content."""

    return {
        "attempt_ordinal": evidence.attempt_ordinal,
        "evaluation_seed": evidence.evaluation_seed,
        "evaluator_configuration_digest": evidence.evaluator_configuration_digest,
        "evaluator_id": evidence.evaluator_id,
        "evaluator_version": evidence.evaluator_version,
        "evidence_id": evidence.evidence_id,
        "failure_events": [
            dict(_failure_event_payload(event)) for event in evidence.failure_events
        ],
        "label_source": (
            None if evidence.label_source is None else evidence.label_source.value
        ),
        "label_strength": (
            None if evidence.label_strength is None else evidence.label_strength.value
        ),
        "metrics": dict(evidence.metrics),
        "progress_after": evidence.progress_after,
        "progress_before": evidence.progress_before,
        "progress_delta": evidence.progress_delta,
        "proposal_id": evidence.proposal_id,
        "replayed_control_steps": evidence.replayed_control_steps,
        "schema_version": evidence.schema_version,
        "simulator_replay_verified": evidence.simulator_replay_verified,
        "source_candidate_id": evidence.source_candidate_id,
        "source_dataset_id": evidence.source_dataset_id,
        "source_episode_id": evidence.source_episode_id,
        "split_group_id": evidence.split_group_id,
        "status": evidence.status.value,
        "success": evidence.success,
        "termination_reason": evidence.termination_reason,
        "unsafe": evidence.unsafe,
    }


def compute_action_verifier_evidence_dataset_digest(
    anchor_manifest: PickCubeAnchorManifestV1,
    evaluation_dataset: EvaluationDataset,
) -> str:
    """Bind all baseline and replay evidence without runtime paths or timestamps."""

    if not isinstance(anchor_manifest, PickCubeAnchorManifestV1):
        raise ActionVerifierExportError("expected PickCubeAnchorManifestV1")
    if not isinstance(evaluation_dataset, EvaluationDataset):
        raise ActionVerifierExportError("expected EvaluationDataset")
    payload = {
        "anchor_manifest_content_digest": anchor_manifest.content_digest,
        "evaluation": {
            "evaluator_configuration_digest": (
                evaluation_dataset.evaluator_configuration_digest
            ),
            "evaluator_id": evaluation_dataset.evaluator_id,
            "evaluator_version": evaluation_dataset.evaluator_version,
            "evidence": [
                dict(_evaluation_evidence_payload(evidence))
                for evidence in evaluation_dataset.evidence
            ],
            "run_id": evaluation_dataset.run_id,
            "selected_proposal_ids": list(evaluation_dataset.selected_proposal_ids),
            "source_corruption_dataset_digest": (
                evaluation_dataset.source_corruption_dataset_digest
            ),
            "source_dataset_id": evaluation_dataset.source_dataset_id,
        },
        "schema_version": ACTION_VERIFIER_EXPORT_SCHEMA_VERSION,
        "semantic": "pickcube_action_verifier_evidence_binding_v1",
    }
    hexadecimal = hashlib.sha256(
        canonical_json_bytes(payload, context="ActionVerifierEvidenceBindingV1")
    ).hexdigest()
    return f"sha256:{hexadecimal}"


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ActionVerifierExportError(f"{context}: expected a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ActionVerifierExportError(f"{context}: expected canonical text")
    return value


def _number(value: object, context: str, *, normalized: bool = False) -> float:
    if type(value) not in (int, float):
        raise ActionVerifierExportError(f"{context}: expected a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ActionVerifierExportError(f"{context}: expected a finite number")
    if normalized and not 0.0 <= number <= 1.0:
        raise ActionVerifierExportError(f"{context}: expected a value in [0, 1]")
    return number


def _integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise ActionVerifierExportError(f"{context}: expected an integer")
    return value


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _remaining_action_digest(actions: np.ndarray[Any, Any]) -> str:
    payload = {
        "action_byte_digest": _sha256_bytes(actions.tobytes(order="C")),
        "action_dtype": actions.dtype.str,
        "action_shape": list(actions.shape),
    }
    return _sha256_bytes(
        canonical_json_bytes(payload, context="PickCubeRemainingActionDigest")
    )


def _verifier_state_content_digest(
    values: np.ndarray[Any, Any], schema_digest: str
) -> str:
    payload = {
        "content_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "dtype": values.dtype.str,
        "schema_digest": schema_digest,
        "semantic": PICKCUBE_VERIFIER_STATE_SEMANTIC,
        "shape": values.shape,
    }
    return _sha256_bytes(
        canonical_json_bytes(payload, context="PickCubeVerifierStateV1Content")
    )


def _configuration_contract(
    evaluation_dataset: EvaluationDataset,
    *,
    source_dataset_id: str,
    source_dataset_digest: str,
    corruption_dataset_digest: str,
) -> tuple[str, str]:
    configuration = _mapping(
        evaluation_dataset.resolved_evaluator_configuration,
        "EvaluationDataset.resolved_evaluator_configuration",
    )
    expected = {
        "adapter_id": MANISKILL_PICKCUBE_ADAPTER_ID,
        "corruption_dataset_digest": corruption_dataset_digest,
        "source_dataset_digest": source_dataset_digest,
        "source_dataset_id": source_dataset_id,
    }
    for field, expected_value in expected.items():
        if configuration.get(field) != expected_value:
            raise ActionVerifierExportError(
                f"evaluator configuration {field} does not match export input"
            )
    adapter_version = _text(
        configuration.get("adapter_version"),
        "evaluator configuration adapter_version",
    )
    if adapter_version != MANISKILL_PICKCUBE_ADAPTER_VERSION:
        raise ActionVerifierExportError(
            "evaluator configuration adapter_version is outside the fixed M3A scope"
        )
    fixed_evaluator_fields: tuple[tuple[str, object], ...] = (
        ("exact_state_verification_required", True),
        ("label_source", LabelSource.SIMULATOR.value),
        ("maximum_label_strength", LabelStrength.STRONG.value),
        ("progress_semantic", PICKCUBE_PROGRESS_SEMANTIC),
        ("simulator_verification_allowed", True),
        (
            "state_verification_semantic",
            StateComparisonSemantic.NUMERIC_TOLERANCE.value,
        ),
        ("state_verification_tolerance", PICKCUBE_STATE_COMPARISON_TOLERANCE),
        ("trust_tier", ReplayTrustTier.EXACT_SIMULATOR.value),
        ("unsafe_semantic", PICKCUBE_UNSAFE_SEMANTIC),
    )
    for contract_field, contract_expected in fixed_evaluator_fields:
        if configuration.get(contract_field) != contract_expected:
            raise ActionVerifierExportError(
                "evaluator configuration "
                f"{contract_field} is outside the fixed M3A scope"
            )
    adapter_configuration = _mapping(
        configuration.get("adapter_configuration"),
        "evaluator configuration adapter_configuration",
    )
    if configuration.get("adapter_configuration_digest") != (
        compute_configuration_digest(adapter_configuration)
    ):
        raise ActionVerifierExportError(
            "evaluator adapter configuration digest does not match its content"
        )
    for adapter_field, adapter_expected in (
        ("adapter_id", MANISKILL_PICKCUBE_ADAPTER_ID),
        ("adapter_version", MANISKILL_PICKCUBE_ADAPTER_VERSION),
        ("progress_semantic", PICKCUBE_PROGRESS_SEMANTIC),
        ("unsafe_semantic", PICKCUBE_UNSAFE_SEMANTIC),
    ):
        if adapter_configuration.get(adapter_field) != adapter_expected:
            raise ActionVerifierExportError(
                f"adapter configuration {adapter_field} is outside the fixed M3A scope"
            )
    state_verification = _mapping(
        adapter_configuration.get("state_verification"),
        "evaluator adapter state_verification",
    )
    for state_field, state_expected in (
        (
            "comparison_semantic",
            StateComparisonSemantic.NUMERIC_TOLERANCE.value,
        ),
        ("maximum_absolute_tolerance", PICKCUBE_STATE_COMPARISON_TOLERANCE),
        ("runtime_semantic", PICKCUBE_STATE_COMPARISON_SEMANTIC),
    ):
        if state_verification.get(state_field) != state_expected:
            raise ActionVerifierExportError(
                "adapter state-verification contract is outside the fixed M3A scope"
            )
    identity = _mapping(
        adapter_configuration.get("identity"),
        "evaluator adapter identity",
    )
    compatibility_identity = _text(
        identity.get("compatibility_identity"),
        "evaluator compatibility identity",
    )
    return adapter_version, compatibility_identity


def _final_evidence_by_proposal(
    evaluation_dataset: EvaluationDataset,
) -> Mapping[str, EvaluationEvidence]:
    attempts: dict[str, list[EvaluationEvidence]] = defaultdict(list)
    for evidence in evaluation_dataset.evidence:
        attempts[evidence.proposal_id].append(evidence)
    final: dict[str, EvaluationEvidence] = {}
    for proposal_id in evaluation_dataset.selected_proposal_ids:
        values = attempts.get(proposal_id)
        if not values:
            raise ActionVerifierExportError(
                f"selected proposal {proposal_id!r} has no terminal evidence"
            )
        ordered = sorted(values, key=lambda item: item.attempt_ordinal)
        if tuple(item.attempt_ordinal for item in ordered) != tuple(
            range(len(ordered))
        ):
            raise ActionVerifierExportError(
                f"proposal {proposal_id!r} evidence attempts are not contiguous"
            )
        final[proposal_id] = ordered[-1]
    return MappingProxyType(final)


def collect_action_verifier_evidence_ids(
    anchor_manifest: PickCubeAnchorManifestV1,
    evaluation_dataset: EvaluationDataset,
) -> tuple[str, ...]:
    """Return accepted baselines plus final strong simulator replay evidence IDs."""

    if not isinstance(anchor_manifest, PickCubeAnchorManifestV1):
        raise ActionVerifierExportError("expected PickCubeAnchorManifestV1")
    if not isinstance(evaluation_dataset, EvaluationDataset):
        raise ActionVerifierExportError("expected EvaluationDataset")
    final = _final_evidence_by_proposal(evaluation_dataset)
    identifiers = {
        evidence.evidence_id for evidence in anchor_manifest.baseline_evidence
    }
    identifiers.update(
        evidence.evidence_id
        for evidence in final.values()
        if evidence.status is EvaluationStatus.CONCLUSIVE
        and evidence.label_source is LabelSource.SIMULATOR
        and evidence.label_strength is LabelStrength.STRONG
        and evidence.simulator_replay_verified
    )
    return tuple(sorted(identifiers))


def _require_source_record_binding(
    record: PickCubeAnchorSourceRecordV1,
    episode: Episode,
) -> CandidateAction:
    if episode.episode_id != record.source_episode_id:
        raise ActionVerifierExportError("anchor record source episode ID mismatch")
    if episode.task_id != MANISKILL_PICKCUBE_TASK_ID:
        raise ActionVerifierExportError("anchor source task ID mismatch")
    if episode.split_group_id != record.anchor.split_group_id:
        raise ActionVerifierExportError("anchor source split group mismatch")
    if len(episode.observations.frames) != 1:
        raise ActionVerifierExportError("anchor source requires one observation")
    frame = episode.observations.frames[0]
    if frame.timestamp_s != 0.0 or frame.cameras:
        raise ActionVerifierExportError(
            "anchor source observation must be timestamp zero with zero cameras"
        )
    if frame.robot_state.dtype.str != PICKCUBE_VERIFIER_STATE_DTYPE:
        raise ActionVerifierExportError("anchor verifier state dtype mismatch")
    if (
        _verifier_state_content_digest(
            frame.robot_state,
            record.verifier_state_schema_digest,
        )
        != record.verifier_state_content_digest
    ):
        raise ActionVerifierExportError(
            "anchor verifier state bytes do not match the manifest"
        )
    if len(episode.candidates) != 1:
        raise ActionVerifierExportError("anchor source requires one candidate")
    candidate = episode.candidates[0]
    if candidate.candidate_id != record.source_candidate_id:
        raise ActionVerifierExportError("anchor source candidate ID mismatch")
    provenance = candidate.provenance
    if (
        provenance.transformation_type != ANCHOR_SOURCE_TRANSFORMATION
        or provenance.source_episode_id != episode.episode_id
        or provenance.split_group_id != record.anchor.split_group_id
        or provenance.seed != record.anchor.source_seed
    ):
        raise ActionVerifierExportError("anchor source provenance mismatch")
    parameters = provenance.transformation_parameters
    expected_parameters = {
        "anchor_id": record.anchor.anchor_id,
        "anchor_state_index": record.anchor.state_index,
        "baseline_evidence_id": record.baseline_evidence_id,
        "candidate_horizon": record.anchor.candidate_horizon,
        "continuation_identity": record.continuation_identity,
        "source_archive_episode_id": record.source_archive_episode_id,
        "source_state_content_digest": record.source_state_content_digest,
        "source_trajectory_id": record.anchor.source_trajectory_id,
        "verifier_state_schema_digest": record.verifier_state_schema_digest,
    }
    for field, expected in expected_parameters.items():
        if parameters.get(field) != expected:
            raise ActionVerifierExportError(
                f"anchor source provenance parameter {field} mismatch"
            )
    actions = candidate.action.actions
    if (
        actions.ndim != 2
        or actions.shape[0] != record.anchor.remaining_horizon
        or actions.shape[1] <= 0
        or actions.dtype.str != record.continuation.action_dtype
    ):
        raise ActionVerifierExportError("anchor source remaining action mismatch")
    if record.anchor.candidate_horizon != FIXED_ACTION_CHUNK_HORIZON:
        raise ActionVerifierExportError("M3A export requires candidate horizon 16")
    if actions.shape[0] < FIXED_ACTION_CHUNK_HORIZON:
        raise ActionVerifierExportError("anchor source is shorter than horizon 16")
    if _remaining_action_digest(actions) != record.source_remaining_action_digest:
        raise ActionVerifierExportError(
            "anchor remaining action bytes do not match the manifest"
        )
    continuation = np.ascontiguousarray(actions[FIXED_ACTION_CHUNK_HORIZON:])
    if (
        continuation.dtype.str != record.continuation.action_dtype
        or continuation.shape != record.continuation.action_shape
        or _sha256_bytes(continuation.tobytes(order="C"))
        != record.continuation.action_byte_digest
    ):
        raise ActionVerifierExportError(
            "anchor continuation bytes do not match the content-bound identity"
        )
    return candidate


def _require_proposal_binding(
    proposal: CorruptedActionProposal,
    *,
    episode: Episode,
    source_candidate: CandidateAction,
) -> str:
    if (
        proposal.source_episode_id != episode.episode_id
        or proposal.source_candidate_id != source_candidate.candidate_id
        or proposal.source_policy_id != episode.source_policy_id
        or proposal.source_task_id != episode.task_id
        or proposal.split_group_id != episode.split_group_id
    ):
        raise ActionVerifierExportError("corruption proposal source binding mismatch")
    parameters = proposal.resolved_parameters
    if (
        parameters.get("window_start") != 0
        or parameters.get("window_end") != FIXED_ACTION_CHUNK_HORIZON
    ):
        raise ActionVerifierExportError("corruption proposal window must be [0, 16)")
    severity_id = _text(
        parameters.get("severity_id"),
        f"proposal {proposal.proposal_id} severity_id",
    )
    source = source_candidate.action
    transformed = proposal.transformed_action
    if (
        transformed.actions.shape != source.actions.shape
        or transformed.actions.dtype != source.actions.dtype
        or transformed.coordinate_frame != source.coordinate_frame
        or transformed.control_period_s != source.control_period_s
    ):
        raise ActionVerifierExportError("corruption proposal action contract mismatch")
    if transformed.actions[FIXED_ACTION_CHUNK_HORIZON:].tobytes(order="C") != (
        source.actions[FIXED_ACTION_CHUNK_HORIZON:].tobytes(order="C")
    ):
        raise ActionVerifierExportError(
            "corruption proposal changed the fixed source continuation"
        )
    return severity_id


def _metric(
    evidence: EvaluationEvidence,
    name: str,
    *,
    expected_type: type[str] | type[bool] | type[int],
) -> object:
    if name not in evidence.metrics:
        raise ActionVerifierExportError(
            f"evidence {evidence.evidence_id} is missing metric {name}"
        )
    value = evidence.metrics[name]
    if expected_type is int:
        if type(value) is not int:
            raise ActionVerifierExportError(
                f"evidence metric {name} must be an integer"
            )
    elif not isinstance(value, expected_type):
        raise ActionVerifierExportError(f"evidence metric {name} has the wrong type")
    return value


def _eligible_corruption(
    proposal: CorruptedActionProposal,
    evidence: EvaluationEvidence,
    *,
    episode: Episode,
    source_candidate: CandidateAction,
    baseline: AnchorBaselineEvidenceV1,
) -> _EligibleCorruption:
    severity_id = _require_proposal_binding(
        proposal,
        episode=episode,
        source_candidate=source_candidate,
    )
    if evidence.status is not EvaluationStatus.CONCLUSIVE:
        raise ActionVerifierExportError("eligible evidence must be conclusive")
    if (
        evidence.label_source is not LabelSource.SIMULATOR
        or evidence.label_strength is not LabelStrength.STRONG
        or not evidence.simulator_replay_verified
        or type(evidence.success) is not bool
        or type(evidence.unsafe) is not bool
    ):
        raise ActionVerifierExportError(
            "conclusive training evidence must be strong simulator verified"
        )
    if (
        evidence.source_episode_id != episode.episode_id
        or evidence.source_candidate_id != source_candidate.candidate_id
        or evidence.split_group_id != episode.split_group_id
        or evidence.proposal_id != proposal.proposal_id
    ):
        raise ActionVerifierExportError("evaluation evidence source binding mismatch")

    semantic = _text(
        _metric(
            evidence,
            _PREFIX_PROGRESS_SEMANTIC,
            expected_type=str,
        ),
        f"evidence metric {_PREFIX_PROGRESS_SEMANTIC}",
    )
    if semantic != baseline.progress_semantic:
        raise ActionVerifierExportError(
            "candidate-prefix progress semantic differs from anchor baseline"
        )
    before = _number(
        evidence.metrics.get(_PREFIX_PROGRESS_BEFORE),
        f"evidence metric {_PREFIX_PROGRESS_BEFORE}",
        normalized=True,
    )
    after = _number(
        evidence.metrics.get(_PREFIX_PROGRESS_AFTER),
        f"evidence metric {_PREFIX_PROGRESS_AFTER}",
        normalized=True,
    )
    delta = _number(
        evidence.metrics.get(_PREFIX_PROGRESS_DELTA),
        f"evidence metric {_PREFIX_PROGRESS_DELTA}",
    )
    if not math.isclose(delta, after - before, rel_tol=0.0, abs_tol=1e-12):
        raise ActionVerifierExportError(
            "candidate-prefix progress delta does not equal after minus before"
        )
    if before != baseline.progress_before:
        raise ActionVerifierExportError(
            "corrupted and source baseline progress-before values differ"
        )
    if _metric(evidence, _PREFIX_EVALUATED, expected_type=bool) is not True:
        raise ActionVerifierExportError(
            "candidate-prefix evidence was not captured before continuation"
        )
    if _metric(evidence, _PREFIX_HORIZON, expected_type=int) != (
        FIXED_ACTION_CHUNK_HORIZON
    ):
        raise ActionVerifierExportError("candidate-prefix horizon is not 16")

    expected_steps = int(source_candidate.action.actions.shape[0])
    for name in (
        "replay_baseline_requested_steps",
        "replay_baseline_steps",
        "replay_corrupted_requested_steps",
        "replay_corrupted_steps",
    ):
        if _metric(evidence, name, expected_type=int) != expected_steps:
            raise ActionVerifierExportError(
                f"evidence metric {name} does not prove complete execution"
            )
    for role in ("baseline", "corrupted"):
        complete_name = f"replay_{role}_restoration_complete_state_comparison"
        if _metric(evidence, complete_name, expected_type=bool) is not True:
            raise ActionVerifierExportError(
                f"{role} restoration did not compare the complete state"
            )
        components = _integer(
            evidence.metrics.get(f"replay_{role}_restoration_compared_component_count"),
            f"{role} restoration component count",
        )
        if components != baseline.compared_component_count:
            raise ActionVerifierExportError(
                f"{role} restoration component inventory differs from baseline"
            )
        error = _number(
            evidence.metrics.get(f"replay_{role}_restoration_maximum_absolute_error"),
            f"{role} restoration maximum error",
        )
        if error < 0.0 or error > PICKCUBE_STATE_COMPARISON_TOLERANCE:
            raise ActionVerifierExportError(
                f"{role} restoration error exceeds the authorized tolerance"
            )
    return _EligibleCorruption(
        proposal=proposal,
        evidence=evidence,
        progress_semantic=semantic,
        progress_before=before,
        progress_after=after,
        progress_delta=delta,
        severity_id=severity_id,
    )


def _build_sample(
    *,
    anchor: _EligibleAnchor,
    dataset_split: DatasetSplit,
    source_dataset_digest: str,
    corruption_dataset_digest: str,
    evidence_dataset_digest: str,
    adapter_version: str,
    corruption: _EligibleCorruption | None,
) -> ActionVerifierSampleV1:
    record = anchor.record
    source = anchor.source_candidate
    source_actions = source.action.actions
    baseline = anchor.baseline
    if corruption is None:
        actions = source_actions[:FIXED_ACTION_CHUNK_HORIZON]
        candidate_type = CandidateType.SOURCE
        proposal_id = None
        corruption_type = None
        severity_id = None
        success = baseline.official_terminal_success
        progress_semantic = baseline.progress_semantic
        progress_before = baseline.progress_before
        progress_after = baseline.progress_after_candidate
        progress_delta = baseline.progress_delta
        unsafe = baseline.terminal_unsafe
        failure_events = anchor.source_candidate.outcome.failure_events
        evidence_id = baseline.evidence_id
        label_source = baseline.label_source
        label_strength = baseline.label_strength
        verified = baseline.simulator_replay_verified
    else:
        evidence = corruption.evidence
        actions = corruption.proposal.transformed_action.actions[
            :FIXED_ACTION_CHUNK_HORIZON
        ]
        candidate_type = CandidateType.CORRUPTED
        proposal_id = corruption.proposal.proposal_id
        corruption_type = corruption.proposal.corruption_type
        severity_id = corruption.severity_id
        success = cast(bool, evidence.success)
        progress_semantic = corruption.progress_semantic
        progress_before = corruption.progress_before
        progress_after = corruption.progress_after
        progress_delta = corruption.progress_delta
        unsafe = cast(bool, evidence.unsafe)
        failure_events = evidence.failure_events
        evidence_id = evidence.evidence_id
        label_source = cast(LabelSource, evidence.label_source)
        label_strength = cast(LabelStrength, evidence.label_strength)
        verified = evidence.simulator_replay_verified
    return ActionVerifierSampleV1(
        anchor_id=record.anchor.anchor_id,
        source_trajectory_id=record.anchor.source_trajectory_id,
        source_seed=record.anchor.source_seed,
        split_group_id=record.anchor.split_group_id,
        dataset_split=dataset_split,
        task_id=anchor.episode.task_id,
        instruction=anchor.episode.instruction,
        state_content_digest=record.source_state_digest,
        state_vector_semantic=PICKCUBE_VERIFIER_STATE_SEMANTIC,
        state_vector_schema_digest=record.verifier_state_schema_digest,
        state_vector=anchor.episode.observations.frames[0].robot_state,
        candidate_action_chunk=actions,
        action_mask=np.ones(FIXED_ACTION_CHUNK_HORIZON, dtype=np.bool_),
        continuation_identity=record.continuation_identity,
        candidate_type=candidate_type,
        proposal_id=proposal_id,
        corruption_type=corruption_type,
        severity_id=severity_id,
        final_task_success=success,
        progress_semantic=progress_semantic,
        progress_before=progress_before,
        progress_after_candidate_chunk=progress_after,
        progress_delta=progress_delta,
        final_unsafe=unsafe,
        failure_events=failure_events,
        strong_simulator_evidence_id=evidence_id,
        evidence_dataset_digest=evidence_dataset_digest,
        source_dataset_digest=source_dataset_digest,
        corruption_dataset_digest=corruption_dataset_digest,
        compatibility_identity=baseline.compatibility_identity,
        adapter_version=adapter_version,
        label_source=label_source,
        label_strength=label_strength,
        simulator_replay_verified=verified,
    )


def _count_by_split_assignments(
    dataset: ActionVerifierDatasetV1,
) -> tuple[Mapping[str, int], Mapping[str, int]]:
    trajectories = Counter(
        assignment.dataset_split.value for assignment in dataset.split_assignments
    )
    samples = Counter(sample.dataset_split.value for sample in dataset.samples)
    return (
        MappingProxyType(
            {split.value: trajectories.get(split.value, 0) for split in DatasetSplit}
        ),
        MappingProxyType(
            {split.value: samples.get(split.value, 0) for split in DatasetSplit}
        ),
    )


def validate_action_verifier_export(
    dataset: ActionVerifierDatasetV1,
    *,
    available_evidence_ids: Sequence[str],
    evidence_dataset_digest: str,
    expected_split_counts: Mapping[DatasetSplit, int] | None = None,
    validation_scope: str = "in_memory",
) -> ActionVerifierValidationReportV1:
    """Independently validate data, foreign keys, classes, and split quotas."""

    validate_action_verifier_dataset(dataset)
    validate_action_verifier_evidence_references(
        dataset,
        available_evidence_ids=available_evidence_ids,
        evidence_dataset_digest=evidence_dataset_digest,
    )
    trajectory_counts, sample_counts = _count_by_split_assignments(dataset)
    if expected_split_counts is not None:
        expected = {
            split.value: expected_split_counts.get(split, 0) for split in DatasetSplit
        }
        if dict(trajectory_counts) != expected:
            raise ActionVerifierExportError(
                "exported source-trajectory split counts do not match quotas"
            )
    source_samples = tuple(
        sample
        for sample in dataset.samples
        if sample.candidate_type is CandidateType.SOURCE
    )
    corrupted = tuple(
        sample
        for sample in dataset.samples
        if sample.candidate_type is CandidateType.CORRUPTED
    )
    successes = sum(sample.final_task_success for sample in corrupted)
    return ActionVerifierValidationReportV1(
        validation_scope=validation_scope,
        valid=True,
        leakage_valid=True,
        evidence_references_valid=True,
        sample_count=len(dataset.samples),
        source_sample_count=len(source_samples),
        corrupted_sample_count=len(corrupted),
        corrupted_success_count=successes,
        corrupted_failure_count=len(corrupted) - successes,
        candidate_group_count=len(dataset.candidate_groups),
        source_trajectory_count=len(dataset.split_assignments),
        evidence_reference_count=len(set(available_evidence_ids)),
        split_trajectory_counts=trajectory_counts,
        split_sample_counts=sample_counts,
        evidence_dataset_digest=evidence_dataset_digest,
        dataset_content_digest=dataset.content_digest,
    )


def _validate_independent_evidence_metrics(
    evidence: EvaluationEvidence,
    *,
    baseline: AnchorBaselineEvidenceV1,
    expected_steps: int,
) -> tuple[str, float, float, float]:
    if (
        evidence.status is not EvaluationStatus.CONCLUSIVE
        or evidence.label_source is not LabelSource.SIMULATOR
        or evidence.label_strength is not LabelStrength.STRONG
        or not evidence.simulator_replay_verified
        or type(evidence.success) is not bool
        or type(evidence.unsafe) is not bool
    ):
        raise ActionVerifierExportError(
            "referenced evidence is not conclusive strong simulator evidence"
        )
    semantic = _text(
        _metric(evidence, _PREFIX_PROGRESS_SEMANTIC, expected_type=str),
        f"evidence metric {_PREFIX_PROGRESS_SEMANTIC}",
    )
    before = _number(
        evidence.metrics.get(_PREFIX_PROGRESS_BEFORE),
        f"evidence metric {_PREFIX_PROGRESS_BEFORE}",
        normalized=True,
    )
    after = _number(
        evidence.metrics.get(_PREFIX_PROGRESS_AFTER),
        f"evidence metric {_PREFIX_PROGRESS_AFTER}",
        normalized=True,
    )
    delta = _number(
        evidence.metrics.get(_PREFIX_PROGRESS_DELTA),
        f"evidence metric {_PREFIX_PROGRESS_DELTA}",
    )
    if (
        semantic != baseline.progress_semantic
        or before != baseline.progress_before
        or not math.isclose(delta, after - before, rel_tol=0.0, abs_tol=1e-12)
        or _metric(evidence, _PREFIX_EVALUATED, expected_type=bool) is not True
        or _metric(evidence, _PREFIX_HORIZON, expected_type=int)
        != FIXED_ACTION_CHUNK_HORIZON
    ):
        raise ActionVerifierExportError(
            "referenced evidence prefix-progress contract does not match its anchor"
        )
    for name in (
        "replay_baseline_requested_steps",
        "replay_baseline_steps",
        "replay_corrupted_requested_steps",
        "replay_corrupted_steps",
    ):
        if _metric(evidence, name, expected_type=int) != expected_steps:
            raise ActionVerifierExportError(
                "referenced evidence does not prove complete continuation execution"
            )
    for role in ("baseline", "corrupted"):
        if (
            _metric(
                evidence,
                f"replay_{role}_restoration_complete_state_comparison",
                expected_type=bool,
            )
            is not True
        ):
            raise ActionVerifierExportError(
                "referenced evidence restoration is not a complete-state comparison"
            )
        count = _integer(
            evidence.metrics.get(f"replay_{role}_restoration_compared_component_count"),
            f"{role} restoration component count",
        )
        error = _number(
            evidence.metrics.get(f"replay_{role}_restoration_maximum_absolute_error"),
            f"{role} restoration maximum error",
        )
        if (
            count != baseline.compared_component_count
            or error < 0.0
            or error > PICKCUBE_STATE_COMPARISON_TOLERANCE
        ):
            raise ActionVerifierExportError(
                "referenced evidence restoration differs from the fixed contract"
            )
    return semantic, before, after, delta


def validate_action_verifier_evidence_bindings(
    dataset: ActionVerifierDatasetV1,
    *,
    anchor_manifest: PickCubeAnchorManifestV1,
    corruption_dataset: CorruptionDataset,
    evaluation_dataset: EvaluationDataset,
) -> None:
    """Rebind every compact sample to exact baseline or counterfactual evidence."""

    validate_action_verifier_dataset(dataset)
    try:
        validate_corruption_dataset(corruption_dataset)
        corruption_digest = compute_corruption_dataset_content_digest(
            corruption_dataset
        )
        validate_evaluation_dataset(
            evaluation_dataset,
            corruption_dataset=corruption_dataset,
            expected_corruption_digest=corruption_digest,
        )
    except (TypeError, ValueError) as exc:
        raise ActionVerifierExportError(
            f"independent evidence input validation failed: {exc}"
        ) from exc
    evidence_digest = compute_action_verifier_evidence_dataset_digest(
        anchor_manifest, evaluation_dataset
    )
    inventory = collect_action_verifier_evidence_ids(
        anchor_manifest, evaluation_dataset
    )
    validate_action_verifier_evidence_references(
        dataset,
        available_evidence_ids=inventory,
        evidence_dataset_digest=evidence_digest,
    )
    first = dataset.samples[0]
    adapter_version, compatibility_identity = _configuration_contract(
        evaluation_dataset,
        source_dataset_id=evaluation_dataset.source_dataset_id,
        source_dataset_digest=first.source_dataset_digest,
        corruption_dataset_digest=first.corruption_dataset_digest,
    )
    if corruption_digest != first.corruption_dataset_digest:
        raise ActionVerifierExportError(
            "compact dataset corruption digest differs from replay proposals"
        )
    records = {record.anchor.anchor_id: record for record in anchor_manifest.records}
    baselines = {
        evidence.evidence_id: evidence for evidence in anchor_manifest.baseline_evidence
    }
    proposals = {
        proposal.proposal_id: proposal for proposal in corruption_dataset.proposals
    }
    final_evidence = _final_evidence_by_proposal(evaluation_dataset)
    samples = {sample.sample_id: sample for sample in dataset.samples}
    expected_corrupted_evidence = {
        evidence.evidence_id
        for evidence in final_evidence.values()
        if evidence.status is EvaluationStatus.CONCLUSIVE
        and evidence.label_source is LabelSource.SIMULATOR
        and evidence.label_strength is LabelStrength.STRONG
        and evidence.simulator_replay_verified
    }
    referenced_corrupted_evidence = {
        sample.strong_simulator_evidence_id
        for sample in dataset.samples
        if sample.candidate_type is CandidateType.CORRUPTED
    }
    if referenced_corrupted_evidence != expected_corrupted_evidence:
        raise ActionVerifierExportError(
            "compact dataset does not exactly cover final strong replay evidence"
        )
    for group in dataset.candidate_groups:
        record = records.get(group.anchor_id)
        baseline = baselines.get(group.baseline_evidence_id)
        if record is None or baseline is None:
            raise ActionVerifierExportError(
                "candidate group does not resolve to accepted anchor evidence"
            )
        if (
            group.source_trajectory_id != record.anchor.source_trajectory_id
            or group.source_seed != record.anchor.source_seed
            or group.split_group_id != record.anchor.split_group_id
            or group.state_content_digest != record.source_state_digest
            or group.continuation_identity != record.continuation_identity
            or baseline.compatibility_identity != compatibility_identity
        ):
            raise ActionVerifierExportError(
                "candidate group identity differs from its accepted anchor"
            )
        source = samples[group.source_sample_id]
        if (
            source.strong_simulator_evidence_id != baseline.evidence_id
            or source.final_task_success != baseline.official_terminal_success
            or source.progress_semantic != baseline.progress_semantic
            or source.progress_before != baseline.progress_before
            or source.progress_after_candidate_chunk
            != baseline.progress_after_candidate
            or source.progress_delta != baseline.progress_delta
            or source.final_unsafe != baseline.terminal_unsafe
            or source.failure_events
            or source.adapter_version != adapter_version
        ):
            raise ActionVerifierExportError(
                "source sample does not reproduce its strong baseline evidence"
            )
        for sample_id in group.corrupted_sample_ids:
            sample = samples[sample_id]
            proposal = proposals.get(cast(str, sample.proposal_id))
            evidence = final_evidence.get(cast(str, sample.proposal_id))
            if proposal is None or evidence is None:
                raise ActionVerifierExportError(
                    "corrupted sample proposal or final evidence does not resolve"
                )
            if (
                proposal.source_episode_id != record.source_episode_id
                or proposal.source_candidate_id != record.source_candidate_id
                or proposal.source_task_id != sample.task_id
                or proposal.split_group_id != record.anchor.split_group_id
                or evidence.proposal_id != proposal.proposal_id
                or evidence.evidence_id != sample.strong_simulator_evidence_id
                or evidence.source_episode_id != record.source_episode_id
                or evidence.source_candidate_id != record.source_candidate_id
                or evidence.split_group_id != record.anchor.split_group_id
            ):
                raise ActionVerifierExportError(
                    "corrupted sample source/evidence foreign keys disagree"
                )
            severity = _text(
                proposal.resolved_parameters.get("severity_id"),
                "corruption proposal severity_id",
            )
            actions = proposal.transformed_action.actions
            continuation = np.ascontiguousarray(actions[FIXED_ACTION_CHUNK_HORIZON:])
            if (
                proposal.resolved_parameters.get("window_start") != 0
                or proposal.resolved_parameters.get("window_end")
                != FIXED_ACTION_CHUNK_HORIZON
                or actions.shape[0] != record.anchor.remaining_horizon
                or actions.shape[1] != dataset.action_dimension
                or actions[:FIXED_ACTION_CHUNK_HORIZON].dtype
                != sample.candidate_action_chunk.dtype
                or actions[:FIXED_ACTION_CHUNK_HORIZON].tobytes(order="C")
                != sample.candidate_action_chunk.tobytes(order="C")
                or continuation.dtype.str != record.continuation.action_dtype
                or continuation.shape != record.continuation.action_shape
                or _sha256_bytes(continuation.tobytes(order="C"))
                != record.continuation.action_byte_digest
                or sample.corruption_type != proposal.corruption_type
                or sample.severity_id != severity
            ):
                raise ActionVerifierExportError(
                    "corrupted sample bytes or continuation identity disagree"
                )
            semantic, before, after, delta = _validate_independent_evidence_metrics(
                evidence,
                baseline=baseline,
                expected_steps=record.anchor.remaining_horizon,
            )
            if (
                sample.final_task_success != evidence.success
                or sample.final_unsafe != evidence.unsafe
                or sample.failure_events != evidence.failure_events
                or sample.progress_semantic != semantic
                or sample.progress_before != before
                or sample.progress_after_candidate_chunk != after
                or sample.progress_delta != delta
                or sample.adapter_version != adapter_version
            ):
                raise ActionVerifierExportError(
                    "corrupted sample labels do not reproduce final replay evidence"
                )


def export_action_verifier_dataset(
    *,
    anchor_manifest: PickCubeAnchorManifestV1,
    source_episodes: Sequence[Episode],
    source_dataset_id: str,
    corruption_dataset: CorruptionDataset,
    evaluation_dataset: EvaluationDataset,
    split_counts: Mapping[DatasetSplit, int] | None = None,
    split_seed: int = 0,
) -> ActionVerifierExportResultV1:
    """Create deterministic compact samples from accepted exact replay evidence."""

    if not isinstance(anchor_manifest, PickCubeAnchorManifestV1):
        raise ActionVerifierExportError("expected PickCubeAnchorManifestV1")
    episodes = tuple(source_episodes)
    try:
        validate_episodes(episodes)
        validate_corruption_dataset(corruption_dataset)
        corruption_digest = compute_corruption_dataset_content_digest(
            corruption_dataset
        )
        validate_evaluation_dataset(
            evaluation_dataset,
            corruption_dataset=corruption_dataset,
            expected_corruption_digest=corruption_digest,
        )
    except (DataValidationError, TypeError, ValueError) as exc:
        raise ActionVerifierExportError(
            f"export input validation failed: {exc}"
        ) from exc
    if evaluation_dataset.run_state is not RunState.COMPLETE:
        raise ActionVerifierExportError(
            "action-verifier export requires a complete evaluation run"
        )
    source_dataset_id = _text(source_dataset_id, "source_dataset_id")
    if (
        corruption_dataset.source_dataset_id != source_dataset_id
        or evaluation_dataset.source_dataset_id != source_dataset_id
    ):
        raise ActionVerifierExportError("source dataset identity does not match inputs")
    if tuple(episode.episode_id for episode in episodes) != tuple(
        record.source_episode_id for record in anchor_manifest.records
    ):
        raise ActionVerifierExportError(
            "source M0 episode order does not match the anchor manifest"
        )
    source_digest = compute_episode_content_digest(episodes)
    adapter_version, compatibility_identity = _configuration_contract(
        evaluation_dataset,
        source_dataset_id=source_dataset_id,
        source_dataset_digest=source_digest,
        corruption_dataset_digest=corruption_digest,
    )
    evidence_digest = compute_action_verifier_evidence_dataset_digest(
        anchor_manifest, evaluation_dataset
    )

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    record_by_episode = {
        record.source_episode_id: record for record in anchor_manifest.records
    }
    baseline_by_id = {
        evidence.evidence_id: evidence for evidence in anchor_manifest.baseline_evidence
    }
    final_evidence = _final_evidence_by_proposal(evaluation_dataset)
    proposals_by_episode: dict[str, list[CorruptedActionProposal]] = defaultdict(list)
    for proposal in corruption_dataset.proposals:
        record = record_by_episode.get(proposal.source_episode_id)
        episode = episode_by_id.get(proposal.source_episode_id)
        if record is None or episode is None:
            raise ActionVerifierExportError(
                "corruption dataset references an unknown anchor source episode"
            )
        source_candidate = _require_source_record_binding(record, episode)
        _require_proposal_binding(
            proposal,
            episode=episode,
            source_candidate=source_candidate,
        )
        proposals_by_episode[proposal.source_episode_id].append(proposal)

    status_counts = Counter(
        evidence.status.value for evidence in evaluation_dataset.evidence
    )
    exclusion_counts: Counter[str] = Counter()
    eligible_anchors: list[_EligibleAnchor] = []
    eligible_evidence_ids: set[str] = set()
    for record in sorted(
        anchor_manifest.records, key=lambda item: item.anchor.anchor_id
    ):
        episode = episode_by_id[record.source_episode_id]
        source_candidate = _require_source_record_binding(record, episode)
        baseline = baseline_by_id.get(record.baseline_evidence_id)
        if baseline is None:
            raise ActionVerifierExportError(
                "anchor source record baseline evidence does not resolve"
            )
        if baseline.compatibility_identity != compatibility_identity:
            raise ActionVerifierExportError(
                "anchor compatibility identity differs from replay configuration"
            )
        eligible: list[_EligibleCorruption] = []
        for proposal in sorted(
            proposals_by_episode.get(episode.episode_id, ()),
            key=lambda item: item.proposal_id,
        ):
            evidence = final_evidence.get(proposal.proposal_id)
            if evidence is None:
                exclusion_counts["proposal_not_selected"] += 1
                continue
            if evidence.status is not EvaluationStatus.CONCLUSIVE:
                exclusion_counts[f"status_{evidence.status.value}"] += 1
                continue
            if (
                evidence.label_source is not LabelSource.SIMULATOR
                or evidence.label_strength is not LabelStrength.STRONG
                or not evidence.simulator_replay_verified
            ):
                exclusion_counts["not_strong_simulator_verified"] += 1
                continue
            item = _eligible_corruption(
                proposal,
                evidence,
                episode=episode,
                source_candidate=source_candidate,
                baseline=baseline,
            )
            eligible.append(item)
            eligible_evidence_ids.add(evidence.evidence_id)
        if not eligible:
            exclusion_counts["anchor_without_conclusive_corruption"] += 1
            continue
        eligible_anchors.append(
            _EligibleAnchor(
                record=record,
                episode=episode,
                source_candidate=source_candidate,
                baseline=baseline,
                corruptions=tuple(eligible),
            )
        )
    if not eligible_anchors:
        raise ActionVerifierExportError(
            "no anchor has conclusive strong simulator-verified corruptions"
        )

    anchors_by_trajectory: dict[str, list[_EligibleAnchor]] = defaultdict(list)
    for anchor in eligible_anchors:
        anchors_by_trajectory[anchor.record.anchor.source_trajectory_id].append(anchor)
    split_sources: list[TrajectorySplitSourceV1] = []
    for trajectory_id, anchors in sorted(anchors_by_trajectory.items()):
        split_sources.append(
            TrajectorySplitSourceV1(
                source_trajectory_id=trajectory_id,
                source_seed=anchors[0].record.anchor.source_seed,
                split_group_id=anchors[0].record.anchor.split_group_id,
                state_digests=tuple(
                    sorted({anchor.record.source_state_digest for anchor in anchors})
                ),
                anchor_ids=tuple(
                    sorted(anchor.record.anchor.anchor_id for anchor in anchors)
                ),
                proposal_ids=tuple(
                    sorted(
                        corruption.proposal.proposal_id
                        for anchor in anchors
                        for corruption in anchor.corruptions
                    )
                ),
            )
        )
    assignments = assign_trajectory_splits(
        tuple(split_sources), split_counts=split_counts, split_seed=split_seed
    )
    assignment_by_trajectory = {
        assignment.source_trajectory_id: assignment for assignment in assignments
    }
    samples: list[ActionVerifierSampleV1] = []
    groups: list[ActionVerifierCandidateGroupV1] = []
    for anchor in eligible_anchors:
        record = anchor.record
        assignment = assignment_by_trajectory[record.anchor.source_trajectory_id]
        source_sample = _build_sample(
            anchor=anchor,
            dataset_split=assignment.dataset_split,
            source_dataset_digest=source_digest,
            corruption_dataset_digest=corruption_digest,
            evidence_dataset_digest=evidence_digest,
            adapter_version=adapter_version,
            corruption=None,
        )
        corrupted_samples = tuple(
            _build_sample(
                anchor=anchor,
                dataset_split=assignment.dataset_split,
                source_dataset_digest=source_digest,
                corruption_dataset_digest=corruption_digest,
                evidence_dataset_digest=evidence_digest,
                adapter_version=adapter_version,
                corruption=corruption,
            )
            for corruption in anchor.corruptions
        )
        samples.extend((source_sample, *corrupted_samples))
        groups.append(
            ActionVerifierCandidateGroupV1(
                anchor_id=record.anchor.anchor_id,
                source_trajectory_id=record.anchor.source_trajectory_id,
                source_seed=record.anchor.source_seed,
                split_group_id=record.anchor.split_group_id,
                dataset_split=assignment.dataset_split,
                task_id=anchor.episode.task_id,
                state_content_digest=record.source_state_digest,
                state_vector_semantic=PICKCUBE_VERIFIER_STATE_SEMANTIC,
                continuation_identity=record.continuation_identity,
                source_sample_id=source_sample.sample_id,
                corrupted_sample_ids=tuple(
                    sorted(sample.sample_id for sample in corrupted_samples)
                ),
                baseline_evidence_id=record.baseline_evidence_id,
            )
        )

    first = samples[0]
    dataset = ActionVerifierDatasetV1(
        state_vector_semantic=PICKCUBE_VERIFIER_STATE_SEMANTIC,
        state_vector_schema_digest=first.state_vector_schema_digest,
        state_vector_dimension=int(first.state_vector.shape[0]),
        action_dimension=int(first.candidate_action_chunk.shape[1]),
        split_policy_id=assignments[0].split_policy_id,
        samples=tuple(sorted(samples, key=lambda item: item.sample_id)),
        candidate_groups=tuple(sorted(groups, key=lambda item: item.group_id)),
        split_assignments=assignments,
    )
    available_evidence_ids = collect_action_verifier_evidence_ids(
        anchor_manifest,
        evaluation_dataset,
    )
    if set(available_evidence_ids) != {
        *(evidence.evidence_id for evidence in anchor_manifest.baseline_evidence),
        *eligible_evidence_ids,
    }:
        raise ActionVerifierExportError(
            "strong simulator evidence inventory differs from exported samples"
        )
    validate_action_verifier_evidence_bindings(
        dataset,
        anchor_manifest=anchor_manifest,
        corruption_dataset=corruption_dataset,
        evaluation_dataset=evaluation_dataset,
    )
    report = validate_action_verifier_export(
        dataset,
        available_evidence_ids=available_evidence_ids,
        evidence_dataset_digest=evidence_digest,
        expected_split_counts=split_counts,
    )
    corrupted_samples = tuple(
        sample
        for sample in dataset.samples
        if sample.candidate_type is CandidateType.CORRUPTED
    )
    trajectory_counts, sample_counts = _count_by_split_assignments(dataset)
    corruption_counts = Counter(
        cast(str, sample.corruption_type) for sample in corrupted_samples
    )
    severity_counts = Counter(
        cast(str, sample.severity_id) for sample in corrupted_samples
    )
    progress_semantics = {sample.progress_semantic for sample in dataset.samples}
    if len(progress_semantics) != 1 or None in progress_semantics:
        raise ActionVerifierExportError("exported samples use inconsistent progress")
    corrupted_successes = sum(sample.final_task_success for sample in corrupted_samples)
    summary = ActionVerifierExportSummaryV1(
        source_trajectory_count=len(assignments),
        manifest_anchor_count=len(anchor_manifest.records),
        included_anchor_count=len(groups),
        anchors_without_conclusive_corruption=exclusion_counts[
            "anchor_without_conclusive_corruption"
        ],
        total_proposal_count=len(corruption_dataset.proposals),
        selected_proposal_count=len(evaluation_dataset.selected_proposal_ids),
        evaluation_attempt_count=len(evaluation_dataset.evidence),
        source_sample_count=len(groups),
        corrupted_sample_count=len(corrupted_samples),
        corrupted_success_count=corrupted_successes,
        corrupted_failure_count=len(corrupted_samples) - corrupted_successes,
        corrupted_unsafe_count=sum(sample.final_unsafe for sample in corrupted_samples),
        candidate_group_count=len(groups),
        dataset_sample_count=len(dataset.samples),
        state_vector_dimension=dataset.state_vector_dimension,
        action_dimension=dataset.action_dimension,
        chunk_horizon=dataset.chunk_horizon,
        progress_semantic=cast(str, next(iter(progress_semantics))),
        evidence_reference_count=len(available_evidence_ids),
        source_dataset_id=source_dataset_id,
        source_dataset_digest=source_digest,
        corruption_dataset_digest=corruption_digest,
        evidence_dataset_digest=evidence_digest,
        dataset_content_digest=dataset.content_digest,
        split_trajectory_counts=trajectory_counts,
        split_sample_counts=sample_counts,
        evaluation_status_counts=status_counts,
        exclusion_reason_counts=exclusion_counts,
        corruption_type_counts=corruption_counts,
        severity_counts=severity_counts,
    )
    return ActionVerifierExportResultV1(
        dataset=dataset,
        summary=summary,
        validation_report=report,
        available_evidence_ids=available_evidence_ids,
        evidence_dataset_digest=evidence_digest,
    )


def export_action_verifier_dataset_from_paths(
    *,
    anchor_manifest_dir: Path,
    source_episode_dir: Path,
    corruption_dataset_dir: Path,
    evaluation_dataset_dir: Path,
    split_counts: Mapping[DatasetSplit, int] | None = None,
    split_seed: int = 0,
) -> ActionVerifierExportResultV1:
    """Reload every source artifact before performing the deterministic export."""

    try:
        manifest = load_anchor_manifest(Path(anchor_manifest_dir))
        episodes = load_episodes(Path(source_episode_dir))
        source_dataset_id = compute_episode_bundle_identifier(Path(source_episode_dir))
        corruption = load_corruption_dataset(Path(corruption_dataset_dir))
        corruption_digest = compute_corruption_dataset_content_digest(corruption)
        evaluation = load_evaluation_dataset(
            Path(evaluation_dataset_dir),
            corruption_dataset=corruption,
            expected_corruption_digest=corruption_digest,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ActionVerifierExportError(
            f"could not reload export inputs safely: {exc}"
        ) from exc
    return export_action_verifier_dataset(
        anchor_manifest=manifest,
        source_episodes=episodes,
        source_dataset_id=source_dataset_id,
        corruption_dataset=corruption,
        evaluation_dataset=evaluation,
        split_counts=split_counts,
        split_seed=split_seed,
    )


def validate_serialized_action_verifier_export(
    dataset_dir: Path,
    *,
    available_evidence_ids: Sequence[str],
    evidence_dataset_digest: str,
    expected_split_counts: Mapping[DatasetSplit, int] | None = None,
) -> ActionVerifierValidationReportV1:
    """Reload a compact bundle and independently repeat every dataset gate."""

    try:
        dataset = load_action_verifier_dataset(Path(dataset_dir))
    except (OSError, TypeError, ValueError) as exc:
        raise ActionVerifierExportError(
            f"serialized action-verifier dataset failed reload: {exc}"
        ) from exc
    return validate_action_verifier_export(
        dataset,
        available_evidence_ids=available_evidence_ids,
        evidence_dataset_digest=evidence_dataset_digest,
        expected_split_counts=expected_split_counts,
        validation_scope="serialized_reload",
    )


def save_action_verifier_export_reports(
    result: ActionVerifierExportResultV1,
    output_dir: Path,
    *,
    validation_report: ActionVerifierValidationReportV1 | None = None,
) -> tuple[Path, Path]:
    """Transactionally publish compact summary and validation JSON reports."""

    if not isinstance(result, ActionVerifierExportResultV1):
        raise ActionVerifierExportError("expected ActionVerifierExportResultV1")
    report = (
        result.validation_report if validation_report is None else validation_report
    )
    if not isinstance(report, ActionVerifierValidationReportV1):
        raise ActionVerifierExportError("expected ActionVerifierValidationReportV1")
    if report.dataset_content_digest != result.dataset.content_digest:
        raise ActionVerifierExportError("validation report dataset digest mismatch")
    destination = Path(output_dir).absolute()
    try:
        if destination.is_symlink() or destination.resolve() != destination:
            raise ActionVerifierExportError(
                "report destination links and junctions are unsupported"
            )
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise ActionVerifierExportError(
                    "report destination must be absent or empty"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
    except ActionVerifierExportError:
        raise
    except OSError as exc:
        raise ActionVerifierExportError(
            f"could not inspect report destination: {exc}"
        ) from exc
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'reports'}.staging-",
                dir=destination.parent,
            )
        )
        for name, payload in (
            (ACTION_VERIFIER_SUMMARY_NAME, result.summary.as_mapping()),
            (ACTION_VERIFIER_VALIDATION_NAME, report.as_mapping()),
        ):
            with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        dict(payload),
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
    except ActionVerifierExportError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ActionVerifierExportError(
            f"could not publish action-verifier reports: {exc}"
        ) from exc
    finally:
        if "staging" in locals() and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return (
        destination / ACTION_VERIFIER_SUMMARY_NAME,
        destination / ACTION_VERIFIER_VALIDATION_NAME,
    )


__all__ = [
    "ACTION_VERIFIER_EXPORT_SCHEMA_VERSION",
    "ACTION_VERIFIER_SUMMARY_NAME",
    "ACTION_VERIFIER_VALIDATION_NAME",
    "ActionVerifierExportError",
    "ActionVerifierExportResultV1",
    "ActionVerifierExportSummaryV1",
    "ActionVerifierValidationReportV1",
    "collect_action_verifier_evidence_ids",
    "compute_action_verifier_evidence_dataset_digest",
    "export_action_verifier_dataset",
    "export_action_verifier_dataset_from_paths",
    "save_action_verifier_export_reports",
    "validate_action_verifier_evidence_bindings",
    "validate_action_verifier_export",
    "validate_serialized_action_verifier_export",
]
