"""Fail-closed validation for action-verifier samples, groups, and datasets."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import NoReturn, cast

import numpy as np

from latentguard.action_verifier.models import (
    ACTION_VERIFIER_SCHEMA_VERSION,
    FIXED_ACTION_CHUNK_HORIZON,
    ActionVerifierCandidateGroupV1,
    ActionVerifierDatasetV1,
    ActionVerifierSampleV1,
    CandidateType,
    DatasetSplit,
    TrajectorySplitAssignmentV1,
)
from latentguard.models import FailureEvent, LabelSource, LabelStrength
from latentguard.validation import DataValidationError, validate_failure_event

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SPLIT_POLICY_PATTERN = re.compile(r"^avp-sha256-[0-9a-f]{64}$")
_SAMPLE_ID_PATTERN = re.compile(r"^avs-sha256-[0-9a-f]{64}$")
_BASELINE_EVIDENCE_ID_PATTERN = re.compile(r"^mspc-anchor-baseline-[0-9a-f]{64}$")
_CORRUPTED_EVIDENCE_ID_PATTERN = re.compile(r"^evd-sha256-[0-9a-f]{64}$")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class ActionVerifierValidationError(ValueError):
    """Raised when compact verifier data is malformed or internally inconsistent."""


def _fail(context: str, field: str, reason: str) -> NoReturn:
    raise ActionVerifierValidationError(f"{context}.{field}: {reason}")


def _text(value: object, context: str, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail(context, field, "expected canonical non-empty text")


def _digest(value: object, context: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(context, field, "expected sha256:<64 lowercase hex characters>")


def _identifier(
    value: object, pattern: re.Pattern[str], context: str, field: str
) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(context, field, "expected a canonical content-derived identifier")


def _source_seed(value: object, context: str) -> None:
    if type(value) is not int or not 0 <= value < 2**32:
        _fail(context, "source_seed", "expected an integer in [0, 2**32)")


def _schema(value: object, context: str) -> None:
    if value != ACTION_VERIFIER_SCHEMA_VERSION:
        _fail(
            context,
            "schema_version",
            f"unsupported version {value!r}; supported: "
            f"{ACTION_VERIFIER_SCHEMA_VERSION}",
        )


def _finite_probability(value: object, context: str, field: str) -> float:
    if type(value) not in (int, float):
        _fail(context, field, "expected a finite normalized number")
    number = float(cast(int | float, value))
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        _fail(context, field, "expected a finite number in [0, 1]")
    return number


def validate_action_verifier_sample(sample: ActionVerifierSampleV1) -> None:
    """Validate one fixed-horizon, strongly simulator-verified sample."""

    context = "ActionVerifierSampleV1"
    if not isinstance(sample, ActionVerifierSampleV1):
        _fail(context, "value", "expected ActionVerifierSampleV1")
    _schema(sample.schema_version, context)
    for field in (
        "anchor_id",
        "source_trajectory_id",
        "split_group_id",
        "task_id",
        "instruction",
        "state_vector_semantic",
    ):
        _text(getattr(sample, field), context, field)
    _source_seed(sample.source_seed, context)
    if not isinstance(sample.dataset_split, DatasetSplit):
        _fail(context, "dataset_split", "expected DatasetSplit")
    for field in (
        "state_content_digest",
        "state_vector_schema_digest",
        "continuation_identity",
        "evidence_dataset_digest",
        "source_dataset_digest",
        "corruption_dataset_digest",
        "compatibility_identity",
    ):
        _digest(getattr(sample, field), context, field)
    if (
        not isinstance(sample.adapter_version, str)
        or _SEMANTIC_VERSION_PATTERN.fullmatch(sample.adapter_version) is None
    ):
        _fail(context, "adapter_version", "expected a semantic version")

    state = sample.state_vector
    if (
        not isinstance(state, np.ndarray)
        or state.ndim != 1
        or state.size == 0
        or state.dtype.hasobject
        or not np.issubdtype(state.dtype, np.floating)
        or not bool(np.all(np.isfinite(state)))
    ):
        _fail(
            context,
            "state_vector",
            "expected a non-empty finite rank-1 floating NumPy array",
        )
    chunk = sample.candidate_action_chunk
    if (
        not isinstance(chunk, np.ndarray)
        or chunk.ndim != 2
        or chunk.shape[0] != FIXED_ACTION_CHUNK_HORIZON
        or chunk.shape[1] <= 0
        or chunk.dtype.hasobject
        or not np.issubdtype(chunk.dtype, np.floating)
        or not bool(np.all(np.isfinite(chunk)))
    ):
        _fail(
            context,
            "candidate_action_chunk",
            "expected finite floating actions shaped [16, action_dim]",
        )
    mask = sample.action_mask
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.dtype(np.bool_)
        or mask.shape != (FIXED_ACTION_CHUNK_HORIZON,)
        or not bool(np.all(mask))
    ):
        _fail(context, "action_mask", "M3A requires a bool [16] mask of all true")

    if not isinstance(sample.candidate_type, CandidateType):
        _fail(context, "candidate_type", "expected CandidateType")
    corruption_fields = (
        ("proposal_id", sample.proposal_id),
        ("corruption_type", sample.corruption_type),
        ("severity_id", sample.severity_id),
    )
    if sample.candidate_type is CandidateType.SOURCE:
        _identifier(
            sample.strong_simulator_evidence_id,
            _BASELINE_EVIDENCE_ID_PATTERN,
            context,
            "strong_simulator_evidence_id",
        )
        if any(value is not None for _, value in corruption_fields):
            _fail(
                context,
                "candidate_type",
                "source samples cannot carry proposal or corruption fields",
            )
        if sample.final_task_success is not True:
            _fail(
                context,
                "final_task_success",
                "source candidates require a successful remaining-trajectory baseline",
            )
    else:
        _identifier(
            sample.strong_simulator_evidence_id,
            _CORRUPTED_EVIDENCE_ID_PATTERN,
            context,
            "strong_simulator_evidence_id",
        )
        for field, value in corruption_fields:
            _text(value, context, field)

    for field in ("final_task_success", "final_unsafe", "simulator_replay_verified"):
        if type(getattr(sample, field)) is not bool:
            _fail(context, field, "expected a boolean")
    if sample.label_source is not LabelSource.SIMULATOR:
        _fail(context, "label_source", "training samples require simulator evidence")
    if sample.label_strength is not LabelStrength.STRONG:
        _fail(context, "label_strength", "training samples require strong evidence")
    if not sample.simulator_replay_verified:
        _fail(
            context,
            "simulator_replay_verified",
            "training samples require verified simulator replay",
        )

    progress_values = (
        sample.progress_before,
        sample.progress_after_candidate_chunk,
        sample.progress_delta,
    )
    if sample.progress_semantic is None:
        if any(value is not None for value in progress_values):
            _fail(
                context,
                "progress_semantic",
                "progress values require an explicit semantic",
            )
    else:
        _text(sample.progress_semantic, context, "progress_semantic")
        if sample.progress_before is None:
            _fail(
                context,
                "progress_before",
                "a declared progress semantic requires before-candidate progress",
            )
    if sample.progress_before is not None:
        before = _finite_probability(sample.progress_before, context, "progress_before")
    else:
        before = None
    after_present = sample.progress_after_candidate_chunk is not None
    delta_present = sample.progress_delta is not None
    if after_present != delta_present:
        _fail(
            context,
            "progress_after_candidate_chunk",
            "after-candidate progress and delta must be present together",
        )
    if after_present:
        if before is None:
            _fail(
                context,
                "progress_before",
                "progress delta requires before-candidate progress",
            )
        after = _finite_probability(
            sample.progress_after_candidate_chunk,
            context,
            "progress_after_candidate_chunk",
        )
        delta = sample.progress_delta
        if type(delta) not in (int, float):
            _fail(context, "progress_delta", "expected a finite number")
        finite_delta = float(cast(int | float, delta))
        if not math.isfinite(finite_delta):
            _fail(context, "progress_delta", "expected a finite number")
        if not math.isclose(finite_delta, after - before, rel_tol=0.0, abs_tol=1e-12):
            _fail(
                context,
                "progress_delta",
                "must equal progress_after_candidate_chunk - progress_before",
            )

    if not isinstance(sample.failure_events, tuple):
        _fail(context, "failure_events", "expected an immutable tuple")
    for index, event in enumerate(sample.failure_events):
        if not isinstance(event, FailureEvent):
            _fail(context, f"failure_events[{index}]", "expected FailureEvent")
        try:
            validate_failure_event(event, sample.strong_simulator_evidence_id)
        except DataValidationError as exc:
            _fail(context, f"failure_events[{index}]", str(exc))

    # Force canonical identity construction during validation.
    _identifier(sample.sample_id, _SAMPLE_ID_PATTERN, context, "sample_id")


def validate_action_verifier_candidate_group(
    group: ActionVerifierCandidateGroupV1,
) -> None:
    """Validate one exact-state candidate group and ordered membership."""

    context = "ActionVerifierCandidateGroupV1"
    if not isinstance(group, ActionVerifierCandidateGroupV1):
        _fail(context, "value", "expected ActionVerifierCandidateGroupV1")
    _schema(group.schema_version, context)
    for field in (
        "anchor_id",
        "source_trajectory_id",
        "split_group_id",
        "task_id",
        "state_vector_semantic",
    ):
        _text(getattr(group, field), context, field)
    _identifier(group.source_sample_id, _SAMPLE_ID_PATTERN, context, "source_sample_id")
    _identifier(
        group.baseline_evidence_id,
        _BASELINE_EVIDENCE_ID_PATTERN,
        context,
        "baseline_evidence_id",
    )
    _source_seed(group.source_seed, context)
    if not isinstance(group.dataset_split, DatasetSplit):
        _fail(context, "dataset_split", "expected DatasetSplit")
    for field in ("state_content_digest", "continuation_identity"):
        _digest(getattr(group, field), context, field)
    if not group.corrupted_sample_ids:
        _fail(context, "corrupted_sample_ids", "must contain evaluated corruptions")
    if tuple(sorted(group.corrupted_sample_ids)) != group.corrupted_sample_ids:
        _fail(context, "corrupted_sample_ids", "must be sorted deterministically")
    if len(set(group.corrupted_sample_ids)) != len(group.corrupted_sample_ids):
        _fail(context, "corrupted_sample_ids", "duplicate sample IDs are invalid")
    if group.source_sample_id in group.corrupted_sample_ids:
        _fail(context, "source_sample_id", "source cannot also be corrupted")
    for index, sample_id in enumerate(group.corrupted_sample_ids):
        _identifier(
            sample_id,
            _SAMPLE_ID_PATTERN,
            context,
            f"corrupted_sample_ids[{index}]",
        )
    _text(group.group_id, context, "group_id")


def validate_trajectory_split_assignment(
    assignment: TrajectorySplitAssignmentV1,
) -> None:
    """Validate one complete trajectory-level split inventory."""

    context = "TrajectorySplitAssignmentV1"
    if not isinstance(assignment, TrajectorySplitAssignmentV1):
        _fail(context, "value", "expected TrajectorySplitAssignmentV1")
    _schema(assignment.schema_version, context)
    for field in ("source_trajectory_id", "split_group_id"):
        _text(getattr(assignment, field), context, field)
    if (
        not isinstance(assignment.split_policy_id, str)
        or _SPLIT_POLICY_PATTERN.fullmatch(assignment.split_policy_id) is None
    ):
        _fail(context, "split_policy_id", "expected avp-sha256- identity")
    _source_seed(assignment.source_seed, context)
    if not isinstance(assignment.dataset_split, DatasetSplit):
        _fail(context, "dataset_split", "expected DatasetSplit")
    for field_name, values, require_digest in (
        ("state_digests", assignment.state_digests, True),
        ("anchor_ids", assignment.anchor_ids, False),
        ("proposal_ids", assignment.proposal_ids, False),
    ):
        if not values:
            _fail(context, field_name, "inventory must not be empty")
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            _fail(context, field_name, "inventory must be sorted and unique")
        for index, value in enumerate(values):
            if require_digest:
                _digest(value, context, f"{field_name}[{index}]")
            else:
                _text(value, context, f"{field_name}[{index}]")
    _text(assignment.assignment_id, context, "assignment_id")


def _require_unique(values: Iterable[str], context: str) -> None:
    sequence = tuple(values)
    if len(sequence) != len(set(sequence)):
        _fail("ActionVerifierDatasetV1", context, "duplicate identifiers are invalid")


def validate_action_verifier_dataset(dataset: ActionVerifierDatasetV1) -> None:
    """Validate ordering, groups, evidence bindings, and split relationships."""

    context = "ActionVerifierDatasetV1"
    if not isinstance(dataset, ActionVerifierDatasetV1):
        _fail(context, "value", "expected ActionVerifierDatasetV1")
    _schema(dataset.schema_version, context)
    _text(dataset.state_vector_semantic, context, "state_vector_semantic")
    _digest(dataset.state_vector_schema_digest, context, "state_vector_schema_digest")
    if dataset.chunk_horizon != FIXED_ACTION_CHUNK_HORIZON:
        _fail(context, "chunk_horizon", "M3A requires exactly 16 actions")
    for field in ("state_vector_dimension", "action_dimension"):
        value = getattr(dataset, field)
        if type(value) is not int or value <= 0:
            _fail(context, field, "expected a positive integer")
    if (
        not isinstance(dataset.split_policy_id, str)
        or _SPLIT_POLICY_PATTERN.fullmatch(dataset.split_policy_id) is None
    ):
        _fail(context, "split_policy_id", "expected avp-sha256- identity")
    if (
        not dataset.samples
        or not dataset.candidate_groups
        or not dataset.split_assignments
    ):
        _fail(context, "content", "samples, groups, and split assignments are required")

    for sample in dataset.samples:
        validate_action_verifier_sample(sample)
    for group in dataset.candidate_groups:
        validate_action_verifier_candidate_group(group)
    for assignment in dataset.split_assignments:
        validate_trajectory_split_assignment(assignment)
    if tuple(sample.sample_id for sample in dataset.samples) != tuple(
        sorted(sample.sample_id for sample in dataset.samples)
    ):
        _fail(context, "samples", "must be sorted by sample_id")
    if tuple(group.group_id for group in dataset.candidate_groups) != tuple(
        sorted(group.group_id for group in dataset.candidate_groups)
    ):
        _fail(context, "candidate_groups", "must be sorted by group_id")
    if tuple(
        assignment.source_trajectory_id for assignment in dataset.split_assignments
    ) != tuple(
        sorted(
            assignment.source_trajectory_id for assignment in dataset.split_assignments
        )
    ):
        _fail(
            context,
            "split_assignments",
            "must be sorted by source_trajectory_id",
        )

    _require_unique((sample.sample_id for sample in dataset.samples), "samples")
    _require_unique(
        (sample.strong_simulator_evidence_id for sample in dataset.samples),
        "strong_simulator_evidence_ids",
    )
    _require_unique(
        (group.group_id for group in dataset.candidate_groups), "candidate_groups"
    )
    _require_unique(
        (group.anchor_id for group in dataset.candidate_groups), "anchor_ids"
    )
    _require_unique(
        (assignment.source_trajectory_id for assignment in dataset.split_assignments),
        "source_trajectory_ids",
    )
    if any(
        assignment.split_policy_id != dataset.split_policy_id
        for assignment in dataset.split_assignments
    ):
        _fail(context, "split_policy_id", "assignments use a different split policy")

    first = dataset.samples[0]
    shared_dataset_context: tuple[tuple[str, object], ...] = (
        ("evidence_dataset_digest", first.evidence_dataset_digest),
        ("source_dataset_digest", first.source_dataset_digest),
        ("corruption_dataset_digest", first.corruption_dataset_digest),
        ("compatibility_identity", first.compatibility_identity),
        ("adapter_version", first.adapter_version),
    )
    for sample in dataset.samples:
        if sample.state_vector_semantic != dataset.state_vector_semantic:
            _fail(context, "state_vector_semantic", "sample semantic mismatch")
        if sample.state_vector_schema_digest != dataset.state_vector_schema_digest:
            _fail(context, "state_vector_schema_digest", "sample schema mismatch")
        if sample.state_vector.shape != (dataset.state_vector_dimension,):
            _fail(context, "state_vector_dimension", "sample vector shape mismatch")
        if sample.candidate_action_chunk.shape != (
            dataset.chunk_horizon,
            dataset.action_dimension,
        ):
            _fail(context, "action_dimension", "sample action shape mismatch")
        if sample.state_vector.dtype != first.state_vector.dtype:
            _fail(context, "state_vector", "all vector dtypes must match")
        if sample.candidate_action_chunk.dtype != first.candidate_action_chunk.dtype:
            _fail(context, "candidate_action_chunk", "all action dtypes must match")
        for field, expected in shared_dataset_context:
            if getattr(sample, field) != expected:
                _fail(context, field, "all samples must share one bound context")

    samples = {sample.sample_id: sample for sample in dataset.samples}
    referenced: set[str] = set()
    for group in dataset.candidate_groups:
        source = samples.get(group.source_sample_id)
        if source is None or source.candidate_type is not CandidateType.SOURCE:
            _fail(context, "candidate_groups", "group source reference is invalid")
        corrupted = tuple(
            samples.get(sample_id) for sample_id in group.corrupted_sample_ids
        )
        if any(
            sample is None or sample.candidate_type is not CandidateType.CORRUPTED
            for sample in corrupted
        ):
            _fail(context, "candidate_groups", "group corruption reference is invalid")
        members = (source, *(sample for sample in corrupted if sample is not None))
        for sample in members:
            if sample.sample_id in referenced:
                _fail(context, "candidate_groups", "sample belongs to multiple groups")
            referenced.add(sample.sample_id)
            for field in (
                "anchor_id",
                "source_trajectory_id",
                "source_seed",
                "split_group_id",
                "dataset_split",
                "task_id",
                "state_content_digest",
                "state_vector_semantic",
                "continuation_identity",
            ):
                if getattr(sample, field) != getattr(group, field):
                    _fail(context, "candidate_groups", f"member {field} mismatch")
        for corrupted_sample in members[1:]:
            for field in (
                "instruction",
                "state_vector_schema_digest",
                "progress_semantic",
                "progress_before",
            ):
                if getattr(corrupted_sample, field) != getattr(source, field):
                    _fail(context, "candidate_groups", f"member {field} mismatch")
            if (
                corrupted_sample.state_vector.dtype != source.state_vector.dtype
                or corrupted_sample.state_vector.shape != source.state_vector.shape
                or corrupted_sample.state_vector.tobytes(order="C")
                != source.state_vector.tobytes(order="C")
            ):
                _fail(
                    context,
                    "candidate_groups",
                    "members must carry the same exact state vector",
                )
        if source.strong_simulator_evidence_id != group.baseline_evidence_id:
            _fail(context, "baseline_evidence_id", "does not match source sample")
    if referenced != set(samples):
        _fail(
            context, "candidate_groups", "every sample must belong to exactly one group"
        )

    assignments = {
        assignment.source_trajectory_id: assignment
        for assignment in dataset.split_assignments
    }
    if set(assignments) != {sample.source_trajectory_id for sample in dataset.samples}:
        _fail(context, "split_assignments", "trajectory coverage is not exact")
    for trajectory_id, assignment in assignments.items():
        trajectory_samples = tuple(
            sample
            for sample in dataset.samples
            if sample.source_trajectory_id == trajectory_id
        )
        trajectory_groups = tuple(
            group
            for group in dataset.candidate_groups
            if group.source_trajectory_id == trajectory_id
        )
        if any(
            sample.source_seed != assignment.source_seed
            or sample.split_group_id != assignment.split_group_id
            or sample.dataset_split is not assignment.dataset_split
            for sample in trajectory_samples
        ):
            _fail(context, "split_assignments", "sample assignment mismatch")
        if any(
            group.source_seed != assignment.source_seed
            or group.split_group_id != assignment.split_group_id
            or group.dataset_split is not assignment.dataset_split
            for group in trajectory_groups
        ):
            _fail(context, "split_assignments", "group assignment mismatch")
        expected_states = tuple(
            sorted({sample.state_content_digest for sample in trajectory_samples})
        )
        expected_anchors = tuple(
            sorted({group.anchor_id for group in trajectory_groups})
        )
        expected_proposals = tuple(
            sorted(
                sample.proposal_id
                for sample in trajectory_samples
                if sample.proposal_id is not None
            )
        )
        if assignment.state_digests != expected_states:
            _fail(context, "split_assignments", "state inventory mismatch")
        if assignment.anchor_ids != expected_anchors:
            _fail(context, "split_assignments", "anchor inventory mismatch")
        if assignment.proposal_ids != expected_proposals:
            _fail(context, "split_assignments", "proposal inventory mismatch")

    from latentguard.action_verifier.splits import validate_no_split_leakage

    validate_no_split_leakage(dataset.split_assignments)
    # Force deterministic digest construction as the final content-integrity check.
    _digest(dataset.content_digest, context, "content_digest")


def validate_action_verifier_evidence_references(
    dataset: ActionVerifierDatasetV1,
    *,
    available_evidence_ids: Iterable[str],
    evidence_dataset_digest: str,
) -> None:
    """Validate sample foreign keys against one content-bound evidence inventory."""

    validate_action_verifier_dataset(dataset)
    context = "ActionVerifierDatasetV1.evidence_references"
    _digest(evidence_dataset_digest, context, "evidence_dataset_digest")
    expected_digest = dataset.samples[0].evidence_dataset_digest
    if evidence_dataset_digest != expected_digest:
        _fail(
            context,
            "evidence_dataset_digest",
            "does not match the digest bound into every sample",
        )
    available = tuple(available_evidence_ids)
    if len(available) != len(set(available)):
        _fail(context, "available_evidence_ids", "duplicates are invalid")
    for index, evidence_id in enumerate(available):
        if not isinstance(evidence_id, str) or not (
            _BASELINE_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
            or _CORRUPTED_EVIDENCE_ID_PATTERN.fullmatch(evidence_id)
        ):
            _fail(
                context,
                f"available_evidence_ids[{index}]",
                "expected a baseline or corrupted evidence identifier",
            )
    referenced = {sample.strong_simulator_evidence_id for sample in dataset.samples}
    missing = sorted(referenced - set(available))
    if missing:
        _fail(
            context,
            "available_evidence_ids",
            "missing referenced evidence: " + ", ".join(missing),
        )


__all__ = [
    "ActionVerifierValidationError",
    "validate_action_verifier_candidate_group",
    "validate_action_verifier_dataset",
    "validate_action_verifier_evidence_references",
    "validate_action_verifier_sample",
    "validate_trajectory_split_assignment",
]
