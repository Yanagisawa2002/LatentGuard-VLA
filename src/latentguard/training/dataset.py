"""Strict, metadata-isolated loading for the accepted M3A verifier dataset."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.action_verifier import (
    ActionVerifierDatasetV1,
    CandidateType,
    DatasetSplit,
    load_action_verifier_dataset,
    validate_action_verifier_dataset,
    validate_no_split_leakage,
)
from latentguard.models import LabelSource, LabelStrength
from latentguard.replay.identity import canonical_json_bytes

ACCEPTED_M3A_DATASET_DIGEST = (
    "sha256:7847c9d0e09170531e13ba07531fabb3ea6f0aa6b0122298b733726d2055856d"
)
"""Exact accepted M3A full-dataset identity authorized for M3B."""

ACCEPTED_STATE_SEMANTIC = "PickCubeVerifierStateV1"
ACCEPTED_PROGRESS_SEMANTIC = "pickcube_binary_completion_v0"
ACCEPTED_STATE_DIMENSION = 38
ACCEPTED_ACTION_HORIZON = 16
ACCEPTED_ACTION_DIMENSION = 8
ACCEPTED_SAMPLE_COUNT = 3_240
ACCEPTED_GROUP_COUNT = 360
ACCEPTED_TRAJECTORY_COUNT = 60
ACCEPTED_SOURCE_SAMPLE_COUNT = 360
ACCEPTED_CORRUPTED_SAMPLE_COUNT = 2_880
ACCEPTED_CORRUPTED_SUCCESS_COUNT = 2_289
ACCEPTED_CORRUPTED_FAILURE_COUNT = 591

_EXPECTED_TRAJECTORIES = MappingProxyType(
    {
        DatasetSplit.TRAIN: 48,
        DatasetSplit.VALIDATION: 6,
        DatasetSplit.TEST: 6,
    }
)
_EXPECTED_SAMPLES = MappingProxyType(
    {
        DatasetSplit.TRAIN: 2_592,
        DatasetSplit.VALIDATION: 324,
        DatasetSplit.TEST: 324,
    }
)
_STATE_FLOAT32 = np.dtype("<f4")
_ACTION_FLOAT64 = np.dtype("<f8")
_BOOL = np.dtype(np.bool_)
_MAX_REPORT_BYTES = 1024 * 1024


class TrainingDatasetError(ValueError):
    """Raised when a dataset cannot cross the M3B training boundary."""


def _fail(context: str, reason: str) -> NoReturn:
    raise TrainingDatasetError(f"{context}: {reason}")


def _freeze_array(value: NDArray[Any]) -> NDArray[Any]:
    """Detach an array into immutable C-order byte-backed storage."""

    detached = np.array(value, copy=True, order="C", subok=False)
    raw = detached.tobytes(order="C")
    return np.frombuffer(raw, dtype=detached.dtype).reshape(detached.shape)


def _validate_model_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    context: str,
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        _fail(context, "expected numpy.ndarray")
    if value.dtype != dtype:
        _fail(context, f"expected dtype {dtype.str}, observed {value.dtype.str}")
    if value.shape != shape:
        _fail(context, f"expected shape {shape}, observed {value.shape}")
    if dtype != _BOOL and not bool(np.all(np.isfinite(value))):
        _fail(context, "all values must be finite")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class ActionVerifierModelExampleV1:
    """One exact allowlisted model example with no reporting metadata."""

    state_vector: NDArray[Any]
    action_chunk: NDArray[Any]
    action_mask: NDArray[Any]
    failure_target: int
    sample_index: int

    def __post_init__(self) -> None:
        """Validate fixed shapes and detach all model arrays."""

        state = _validate_model_array(
            self.state_vector,
            dtype=_STATE_FLOAT32,
            shape=(ACCEPTED_STATE_DIMENSION,),
            context="ActionVerifierModelExampleV1.state_vector",
        )
        actions = _validate_model_array(
            self.action_chunk,
            dtype=_ACTION_FLOAT64,
            shape=(ACCEPTED_ACTION_HORIZON, ACCEPTED_ACTION_DIMENSION),
            context="ActionVerifierModelExampleV1.action_chunk",
        )
        mask = _validate_model_array(
            self.action_mask,
            dtype=_BOOL,
            shape=(ACCEPTED_ACTION_HORIZON,),
            context="ActionVerifierModelExampleV1.action_mask",
        )
        if type(self.failure_target) is not int or self.failure_target not in (0, 1):
            _fail("ActionVerifierModelExampleV1.failure_target", "expected binary int")
        if type(self.sample_index) is not int or self.sample_index < 0:
            _fail(
                "ActionVerifierModelExampleV1.sample_index",
                "expected non-negative int",
            )
        object.__setattr__(self, "state_vector", _freeze_array(state))
        object.__setattr__(self, "action_chunk", _freeze_array(actions))
        object.__setattr__(self, "action_mask", _freeze_array(mask))


@dataclass(frozen=True, slots=True)
class ActionVerifierReportingMetadataV1:
    """Reporting-only fields joined by sample index outside the learned model."""

    sample_index: int
    group_index: int
    sample_id: str
    group_id: str
    anchor_id: str
    dataset_split: DatasetSplit
    candidate_type: CandidateType
    corruption_family: str | None
    severity: str | None
    source_trajectory: str
    anchor_selection_reason: str

    def __post_init__(self) -> None:
        """Reject malformed or partially populated reporting metadata."""

        for name in ("sample_index", "group_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(f"ActionVerifierReportingMetadataV1.{name}", "invalid index")
        if not isinstance(self.dataset_split, DatasetSplit):
            _fail("ActionVerifierReportingMetadataV1.dataset_split", "invalid split")
        if not isinstance(self.candidate_type, CandidateType):
            _fail("ActionVerifierReportingMetadataV1.candidate_type", "invalid type")
        for name in ("sample_id", "group_id", "anchor_id", "source_trajectory"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                _fail(
                    f"ActionVerifierReportingMetadataV1.{name}",
                    "expected canonical text",
                )
        optional = (self.corruption_family, self.severity, self.anchor_selection_reason)
        if any(
            value is not None
            and (not isinstance(value, str) or not value or value != value.strip())
            for value in optional
        ):
            _fail("ActionVerifierReportingMetadataV1", "invalid optional text")
        if self.candidate_type is CandidateType.SOURCE:
            if self.corruption_family is not None or self.severity is not None:
                _fail(
                    "ActionVerifierReportingMetadataV1",
                    "source metadata cannot carry corruption fields",
                )
        elif self.corruption_family is None or self.severity is None:
            _fail(
                "ActionVerifierReportingMetadataV1",
                "corrupted metadata requires family and severity",
            )


@dataclass(frozen=True, slots=True)
class AcceptedActionVerifierDatasetV1:
    """Validated model projection plus separately indexed reporting metadata."""

    dataset_digest: str
    split_digest: str
    training_split_digest: str
    acceptance_report_digest: str
    examples: tuple[ActionVerifierModelExampleV1, ...]
    reporting: tuple[ActionVerifierReportingMetadataV1, ...]
    split_indices: Mapping[DatasetSplit, tuple[int, ...]]
    group_members: Mapping[int, tuple[int, ...]]

    def __post_init__(self) -> None:
        """Freeze indexes and enforce exact projection/report alignment."""

        for name in (
            "dataset_digest",
            "split_digest",
            "training_split_digest",
            "acceptance_report_digest",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.startswith("sha256:")
                or len(value) != 71
            ):
                _fail(f"AcceptedActionVerifierDatasetV1.{name}", "invalid digest")
        examples = tuple(self.examples)
        reporting = tuple(self.reporting)
        if len(examples) != len(reporting) or not examples:
            _fail("AcceptedActionVerifierDatasetV1", "projection alignment differs")
        if any(item.sample_index != index for index, item in enumerate(examples)):
            _fail(
                "AcceptedActionVerifierDatasetV1.examples", "indices are not canonical"
            )
        if any(item.sample_index != index for index, item in enumerate(reporting)):
            _fail(
                "AcceptedActionVerifierDatasetV1.reporting", "indices are not canonical"
            )
        normalized_splits: dict[DatasetSplit, tuple[int, ...]] = {}
        covered: set[int] = set()
        for split in DatasetSplit:
            indices = tuple(self.split_indices.get(split, ()))
            if tuple(sorted(indices)) != indices or len(indices) != len(set(indices)):
                _fail("AcceptedActionVerifierDatasetV1.split_indices", "invalid order")
            if any(index < 0 or index >= len(examples) for index in indices):
                _fail("AcceptedActionVerifierDatasetV1.split_indices", "out of range")
            if any(reporting[index].dataset_split is not split for index in indices):
                _fail("AcceptedActionVerifierDatasetV1.split_indices", "split mismatch")
            covered.update(indices)
            normalized_splits[split] = indices
        if covered != set(range(len(examples))):
            _fail("AcceptedActionVerifierDatasetV1.split_indices", "coverage differs")
        normalized_groups: dict[int, tuple[int, ...]] = {}
        group_covered: set[int] = set()
        for group_index in sorted(self.group_members):
            members = tuple(self.group_members[group_index])
            if group_index != len(normalized_groups):
                _fail(
                    "AcceptedActionVerifierDatasetV1.group_members", "noncanonical key"
                )
            if tuple(sorted(members)) != members or not members:
                _fail(
                    "AcceptedActionVerifierDatasetV1.group_members", "invalid members"
                )
            if any(reporting[index].group_index != group_index for index in members):
                _fail("AcceptedActionVerifierDatasetV1.group_members", "group mismatch")
            group_covered.update(members)
            normalized_groups[group_index] = members
        if group_covered != set(range(len(examples))):
            _fail("AcceptedActionVerifierDatasetV1.group_members", "coverage differs")
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "reporting", reporting)
        object.__setattr__(self, "split_indices", MappingProxyType(normalized_splits))
        object.__setattr__(self, "group_members", MappingProxyType(normalized_groups))

    def __len__(self) -> int:
        """Return the exact accepted sample count."""

        return len(self.examples)

    def __getitem__(self, index: int) -> ActionVerifierModelExampleV1:
        """Return one metadata-free immutable model example."""

        return self.examples[index]

    def indices_for_split(self, split: DatasetSplit) -> tuple[int, ...]:
        """Return canonical sample indices for one preserved M3A split."""

        if not isinstance(split, DatasetSplit):
            _fail("AcceptedActionVerifierDatasetV1.indices_for_split", "invalid split")
        return self.split_indices[split]

    def reporting_for(self, sample_index: int) -> ActionVerifierReportingMetadataV1:
        """Join reporting metadata without adding it to the model example."""

        return self.reporting[sample_index]


def _digest_payload(payload: object, context: str) -> str:
    encoded = canonical_json_bytes(payload, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def compute_dataset_split_digest(dataset: ActionVerifierDatasetV1) -> str:
    """Bind the complete immutable M3A split and group membership."""

    group_index_by_sample: dict[str, int] = {}
    for group_index, group in enumerate(dataset.candidate_groups):
        for sample_id in (group.source_sample_id, *group.corrupted_sample_ids):
            group_index_by_sample[sample_id] = group_index
    payload = {
        "schema_version": "1.0",
        "semantic": "action_verifier_dataset_split_v1",
        "dataset_digest": dataset.content_digest,
        "split_policy_id": dataset.split_policy_id,
        "assignments": [
            {
                "assignment_id": assignment.assignment_id,
                "dataset_split": assignment.dataset_split.value,
                "source_trajectory_id": assignment.source_trajectory_id,
            }
            for assignment in dataset.split_assignments
        ],
        "samples": [
            {
                "dataset_split": sample.dataset_split.value,
                "group_index": group_index_by_sample[sample.sample_id],
                "sample_id": sample.sample_id,
            }
            for sample in dataset.samples
        ],
    }
    return _digest_payload(payload, "ActionVerifierDatasetSplitV1")


def compute_training_split_digest(
    dataset: ActionVerifierDatasetV1, *, split_digest: str | None = None
) -> str:
    """Bind exactly the training assignments, groups, and sample membership."""

    resolved_split_digest = (
        compute_dataset_split_digest(dataset) if split_digest is None else split_digest
    )
    train_assignments = [
        assignment
        for assignment in dataset.split_assignments
        if assignment.dataset_split is DatasetSplit.TRAIN
    ]
    train_samples = [
        sample
        for sample in dataset.samples
        if sample.dataset_split is DatasetSplit.TRAIN
    ]
    train_sample_ids = {sample.sample_id for sample in train_samples}
    train_groups = [
        group
        for group in dataset.candidate_groups
        if group.dataset_split is DatasetSplit.TRAIN
    ]
    if any(
        group.source_sample_id not in train_sample_ids
        or any(item not in train_sample_ids for item in group.corrupted_sample_ids)
        for group in train_groups
    ):
        _fail(
            "ActionVerifierTrainingSplitV1", "group membership crosses training split"
        )
    payload = {
        "schema_version": "1.0",
        "semantic": "action_verifier_training_split_v1",
        "dataset_digest": dataset.content_digest,
        "split_digest": resolved_split_digest,
        "assignments": [item.assignment_id for item in train_assignments],
        "groups": [item.group_id for item in train_groups],
        "samples": [item.sample_id for item in train_samples],
    }
    return _digest_payload(payload, "ActionVerifierTrainingSplitV1")


def _require_report_file(path: Path) -> bytes:
    requested = Path(path).absolute()
    try:
        if (
            requested.is_symlink()
            or not requested.is_file()
            or requested.resolve() != requested
            or requested.stat().st_nlink != 1
        ):
            _fail("M3AFullTargetReport", "expected one unlinked regular file")
        if requested.stat().st_size > _MAX_REPORT_BYTES:
            _fail("M3AFullTargetReport", "report is unexpectedly large")
        return requested.read_bytes()
    except TrainingDatasetError:
        raise
    except OSError as exc:
        raise TrainingDatasetError(
            f"M3AFullTargetReport: could not read: {exc}"
        ) from exc


def _reject_constant(value: str) -> NoReturn:
    _fail("M3AFullTargetReport", f"non-finite JSON constant {value!r}")


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("M3AFullTargetReport", f"duplicate field {key!r}")
        result[key] = value
    return result


def _report_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected JSON object")
    return cast(dict[str, object], value)


def _report_field(report: Mapping[str, object], name: str) -> object:
    if name not in report:
        _fail("M3AFullTargetReport", f"missing field {name!r}")
    return report[name]


def _require_report_value(
    report: Mapping[str, object], name: str, expected: object
) -> None:
    value = _report_field(report, name)
    if type(value) is not type(expected) or value != expected:
        _fail("M3AFullTargetReport", f"{name} does not match accepted value")


def _require_report_counts(
    report: Mapping[str, object],
    name: str,
    expected: Mapping[DatasetSplit, int],
) -> None:
    value = _report_mapping(_report_field(report, name), f"M3AFullTargetReport.{name}")
    expected_text = {split.value: count for split, count in expected.items()}
    if set(value) != set(expected_text) or any(
        type(value[key]) is not int or value[key] != count
        for key, count in expected_text.items()
    ):
        _fail("M3AFullTargetReport", f"{name} does not match accepted counts")


def validate_full_target_report(
    report_path: Path,
    *,
    expected_dataset_digest: str,
    expected_anchor_manifest_digest: str | None = None,
) -> str:
    """Validate the independent M3A full-target report and return its digest."""

    payload = _require_report_file(report_path)
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except TrainingDatasetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingDatasetError(
            f"M3AFullTargetReport: invalid UTF-8 JSON: {exc}"
        ) from exc
    report = _report_mapping(raw, "M3AFullTargetReport")
    required: Mapping[str, object] = {
        "schema_version": "1.0",
        "validation_scope": "serialized_reload",
        "valid": True,
        "full_target_required": True,
        "full_target_valid": True,
        "leakage_valid": True,
        "evidence_references_valid": True,
        "continuation_integrity_valid": True,
        "resume_idempotence_valid": True,
        "verifier_state_restoration_valid": True,
        "dataset_content_digest": expected_dataset_digest,
        "sample_count": ACCEPTED_SAMPLE_COUNT,
        "candidate_group_count": ACCEPTED_GROUP_COUNT,
        "source_trajectory_count": ACCEPTED_TRAJECTORY_COUNT,
        "source_sample_count": ACCEPTED_SOURCE_SAMPLE_COUNT,
        "corrupted_sample_count": ACCEPTED_CORRUPTED_SAMPLE_COUNT,
        "corrupted_success_count": ACCEPTED_CORRUPTED_SUCCESS_COUNT,
        "corrupted_failure_count": ACCEPTED_CORRUPTED_FAILURE_COUNT,
        "baseline_success_count": ACCEPTED_GROUP_COUNT,
        "evidence_reference_count": ACCEPTED_SAMPLE_COUNT,
        "continuation_proposal_count": ACCEPTED_CORRUPTED_SAMPLE_COUNT,
        "restoration_evidence_count": ACCEPTED_CORRUPTED_SAMPLE_COUNT,
        "evaluation_execution_error_count": 0,
        "evaluation_indeterminate_count": 0,
        "evaluation_invalid_count": 0,
        "state_vector_dimension": ACCEPTED_STATE_DIMENSION,
        "state_vector_semantic": ACCEPTED_STATE_SEMANTIC,
        "chunk_horizon": ACCEPTED_ACTION_HORIZON,
        "action_dimension": ACCEPTED_ACTION_DIMENSION,
        "progress_semantic": ACCEPTED_PROGRESS_SEMANTIC,
    }
    for name, expected in required.items():
        _require_report_value(report, name, expected)
    if expected_anchor_manifest_digest is not None:
        if (
            not isinstance(expected_anchor_manifest_digest, str)
            or not expected_anchor_manifest_digest.startswith("sha256:")
            or len(expected_anchor_manifest_digest) != 71
        ):
            _fail("M3AFullTargetReport", "invalid anchor manifest digest")
        _require_report_value(
            report,
            "anchor_manifest_content_digest",
            expected_anchor_manifest_digest,
        )
    _require_report_counts(report, "split_trajectory_counts", _EXPECTED_TRAJECTORIES)
    _require_report_counts(report, "split_sample_counts", _EXPECTED_SAMPLES)
    for name in (
        "restoration_maximum_absolute_error",
        "verifier_state_restoration_maximum_absolute_error",
    ):
        value = _report_field(report, name)
        if type(value) not in (int, float):
            _fail("M3AFullTargetReport", f"{name} must be finite")
        number = float(cast(int | float, value))
        if not math.isfinite(number) or number < 0.0 or number > 1e-6:
            _fail("M3AFullTargetReport", f"{name} exceeds authorized tolerance")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_accepted_dataset(
    dataset: ActionVerifierDatasetV1, *, expected_dataset_digest: str
) -> None:
    validate_action_verifier_dataset(dataset)
    validate_no_split_leakage(dataset.split_assignments)
    if dataset.content_digest != expected_dataset_digest:
        _fail("AcceptedM3ADataset.dataset_digest", "unexpected dataset content")
    if (
        dataset.state_vector_semantic != ACCEPTED_STATE_SEMANTIC
        or dataset.state_vector_dimension != ACCEPTED_STATE_DIMENSION
        or dataset.action_dimension != ACCEPTED_ACTION_DIMENSION
        or dataset.chunk_horizon != ACCEPTED_ACTION_HORIZON
    ):
        _fail("AcceptedM3ADataset.contract", "fixed dimensions or semantic changed")
    if (
        len(dataset.samples) != ACCEPTED_SAMPLE_COUNT
        or len(dataset.candidate_groups) != ACCEPTED_GROUP_COUNT
        or len(dataset.split_assignments) != ACCEPTED_TRAJECTORY_COUNT
    ):
        _fail("AcceptedM3ADataset.inventory", "full-target counts changed")
    trajectory_counts = Counter(
        item.dataset_split for item in dataset.split_assignments
    )
    sample_counts = Counter(item.dataset_split for item in dataset.samples)
    if dict(trajectory_counts) != dict(_EXPECTED_TRAJECTORIES):
        _fail("AcceptedM3ADataset.split", "trajectory counts changed")
    if dict(sample_counts) != dict(_EXPECTED_SAMPLES):
        _fail("AcceptedM3ADataset.split", "sample counts changed")
    source = [
        item for item in dataset.samples if item.candidate_type is CandidateType.SOURCE
    ]
    corrupted = [
        item
        for item in dataset.samples
        if item.candidate_type is CandidateType.CORRUPTED
    ]
    corrupted_successes = sum(item.final_task_success for item in corrupted)
    if (
        len(source) != ACCEPTED_SOURCE_SAMPLE_COUNT
        or len(corrupted) != ACCEPTED_CORRUPTED_SAMPLE_COUNT
        or corrupted_successes != ACCEPTED_CORRUPTED_SUCCESS_COUNT
        or len(corrupted) - corrupted_successes != ACCEPTED_CORRUPTED_FAILURE_COUNT
        or not all(item.final_task_success for item in source)
    ):
        _fail("AcceptedM3ADataset.labels", "accepted class inventory changed")
    group_counts = Counter(
        item.source_trajectory_id for item in dataset.candidate_groups
    )
    if set(group_counts.values()) != {6}:
        _fail("AcceptedM3ADataset.groups", "each trajectory must have six groups")
    if any(len(item.corrupted_sample_ids) != 8 for item in dataset.candidate_groups):
        _fail("AcceptedM3ADataset.groups", "each group must have eight corruptions")
    for index, sample in enumerate(dataset.samples):
        _validate_model_array(
            sample.state_vector,
            dtype=_STATE_FLOAT32,
            shape=(ACCEPTED_STATE_DIMENSION,),
            context=f"AcceptedM3ADataset.samples[{index}].state_vector",
        )
        _validate_model_array(
            sample.candidate_action_chunk,
            dtype=_ACTION_FLOAT64,
            shape=(ACCEPTED_ACTION_HORIZON, ACCEPTED_ACTION_DIMENSION),
            context=f"AcceptedM3ADataset.samples[{index}].candidate_action_chunk",
        )
        mask = _validate_model_array(
            sample.action_mask,
            dtype=_BOOL,
            shape=(ACCEPTED_ACTION_HORIZON,),
            context=f"AcceptedM3ADataset.samples[{index}].action_mask",
        )
        if not bool(np.all(mask)):
            _fail("AcceptedM3ADataset.action_mask", "M3A masks must all be true")
        if (
            sample.progress_semantic != ACCEPTED_PROGRESS_SEMANTIC
            or sample.label_source is not LabelSource.SIMULATOR
            or sample.label_strength is not LabelStrength.STRONG
            or not sample.simulator_replay_verified
        ):
            _fail(
                "AcceptedM3ADataset.evidence", "sample is not accepted strong evidence"
            )


def _validate_anchor_reasons(
    dataset: ActionVerifierDatasetV1,
    values: Mapping[str, str],
) -> Mapping[str, str]:
    anchor_ids = {group.anchor_id for group in dataset.candidate_groups}
    if not isinstance(values, Mapping) or set(values) != anchor_ids:
        _fail("AcceptedM3ADataset.anchor_selection_reasons", "coverage differs")
    normalized: dict[str, str] = {}
    for anchor_id in sorted(anchor_ids):
        reason = values[anchor_id]
        if not isinstance(reason, str) or not reason or reason != reason.strip():
            _fail("AcceptedM3ADataset.anchor_selection_reasons", "invalid reason")
        normalized[anchor_id] = reason
    return MappingProxyType(normalized)


def load_accepted_action_verifier_dataset(
    dataset_dir: Path,
    *,
    full_target_report: Path,
    anchor_selection_reasons: Mapping[str, str],
    anchor_manifest_content_digest: str,
    expected_dataset_digest: str = ACCEPTED_M3A_DATASET_DIGEST,
) -> AcceptedActionVerifierDatasetV1:
    """Load, independently gate, and project the fixed accepted M3A dataset.

    ``anchor_selection_reasons`` is reporting-only and must cover every anchor
    exactly; it is never retained in a model example or batch.
    """

    if (
        not isinstance(expected_dataset_digest, str)
        or not expected_dataset_digest.startswith("sha256:")
        or len(expected_dataset_digest) != 71
    ):
        _fail("AcceptedM3ADataset.expected_dataset_digest", "invalid digest")
    try:
        dataset = load_action_verifier_dataset(Path(dataset_dir))
    except (OSError, TypeError, ValueError) as exc:
        raise TrainingDatasetError(f"AcceptedM3ADataset: load failed: {exc}") from exc
    _validate_accepted_dataset(dataset, expected_dataset_digest=expected_dataset_digest)
    report_digest = validate_full_target_report(
        full_target_report,
        expected_dataset_digest=expected_dataset_digest,
        expected_anchor_manifest_digest=anchor_manifest_content_digest,
    )
    reasons = _validate_anchor_reasons(dataset, anchor_selection_reasons)

    group_index_by_sample: dict[str, int] = {}
    group_members: dict[int, tuple[int, ...]] = {}
    sample_index_by_id = {
        sample.sample_id: index for index, sample in enumerate(dataset.samples)
    }
    group_anchor: dict[int, str] = {}
    for group_index, group in enumerate(dataset.candidate_groups):
        ids = (group.source_sample_id, *group.corrupted_sample_ids)
        members = tuple(sorted(sample_index_by_id[item] for item in ids))
        group_members[group_index] = members
        group_anchor[group_index] = group.anchor_id
        for sample_id in ids:
            if sample_id in group_index_by_sample:
                _fail("AcceptedM3ADataset.groups", "sample appears in multiple groups")
            group_index_by_sample[sample_id] = group_index

    examples: list[ActionVerifierModelExampleV1] = []
    reporting: list[ActionVerifierReportingMetadataV1] = []
    split_indices: dict[DatasetSplit, list[int]] = {split: [] for split in DatasetSplit}
    for sample_index, sample in enumerate(dataset.samples):
        group_index = group_index_by_sample[sample.sample_id]
        examples.append(
            ActionVerifierModelExampleV1(
                state_vector=sample.state_vector,
                action_chunk=sample.candidate_action_chunk,
                action_mask=sample.action_mask,
                failure_target=1 - int(sample.final_task_success),
                sample_index=sample_index,
            )
        )
        reporting.append(
            ActionVerifierReportingMetadataV1(
                sample_index=sample_index,
                group_index=group_index,
                sample_id=sample.sample_id,
                group_id=dataset.candidate_groups[group_index].group_id,
                anchor_id=group_anchor[group_index],
                dataset_split=sample.dataset_split,
                candidate_type=sample.candidate_type,
                corruption_family=sample.corruption_type,
                severity=sample.severity_id,
                source_trajectory=sample.source_trajectory_id,
                anchor_selection_reason=reasons[group_anchor[group_index]],
            )
        )
        split_indices[sample.dataset_split].append(sample_index)
    split_digest = compute_dataset_split_digest(dataset)
    return AcceptedActionVerifierDatasetV1(
        dataset_digest=dataset.content_digest,
        split_digest=split_digest,
        training_split_digest=compute_training_split_digest(
            dataset, split_digest=split_digest
        ),
        acceptance_report_digest=report_digest,
        examples=tuple(examples),
        reporting=tuple(reporting),
        split_indices=MappingProxyType(
            {split: tuple(indices) for split, indices in split_indices.items()}
        ),
        group_members=MappingProxyType(group_members),
    )


__all__ = [
    "ACCEPTED_M3A_DATASET_DIGEST",
    "AcceptedActionVerifierDatasetV1",
    "ActionVerifierModelExampleV1",
    "ActionVerifierReportingMetadataV1",
    "TrainingDatasetError",
    "compute_dataset_split_digest",
    "compute_training_split_digest",
    "load_accepted_action_verifier_dataset",
    "validate_full_target_report",
]
