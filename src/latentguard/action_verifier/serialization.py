"""Safe compact JSON/stacked-NPY serialization for verifier datasets."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.action_verifier.models import (
    ACTION_VERIFIER_SCHEMA_VERSION,
    ActionVerifierCandidateGroupV1,
    ActionVerifierDatasetV1,
    ActionVerifierSampleV1,
    CandidateType,
    DatasetSplit,
    TrajectorySplitAssignmentV1,
)
from latentguard.action_verifier.validation import (
    ActionVerifierValidationError,
    validate_action_verifier_dataset,
)
from latentguard.models import FailureEvent, LabelSource, LabelStrength

MANIFEST_NAME = "manifest.json"
SERIALIZATION_FORMAT = "latentguard-action-verifier-dataset"
SERIALIZATION_VERSION = 1

_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)

_ARRAY_NAMES = (
    "state_vectors",
    "candidate_action_chunks",
    "action_masks",
)
_ARRAY_FILES = {
    "state_vectors": "state_vectors.npy",
    "candidate_action_chunks": "candidate_action_chunks.npy",
    "action_masks": "action_masks.npy",
}
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "serialization_version",
        "dataset_schema_version",
        "dataset_content_digest",
        "sample_count",
        "group_count",
        "trajectory_count",
        "array_count",
        "chunk_horizon",
        "action_dimension",
        "state_vector_dimension",
        "state_vector_semantic",
        "state_vector_schema_digest",
        "split_policy_id",
        "arrays",
        "samples",
        "candidate_groups",
        "split_assignments",
    }
)
_ARRAY_FIELDS = frozenset({"path", "dtype", "shape", "content_sha256"})
_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "array_index",
        "anchor_id",
        "source_trajectory_id",
        "source_seed",
        "split_group_id",
        "dataset_split",
        "task_id",
        "instruction",
        "state_content_digest",
        "state_vector_semantic",
        "state_vector_schema_digest",
        "continuation_identity",
        "candidate_type",
        "proposal_id",
        "corruption_type",
        "severity_id",
        "final_task_success",
        "progress_semantic",
        "progress_before",
        "progress_after_candidate_chunk",
        "progress_delta",
        "final_unsafe",
        "failure_events",
        "strong_simulator_evidence_id",
        "evidence_dataset_digest",
        "source_dataset_digest",
        "corruption_dataset_digest",
        "compatibility_identity",
        "adapter_version",
        "label_source",
        "label_strength",
        "simulator_replay_verified",
        "schema_version",
    }
)
_FAILURE_FIELDS = frozenset(
    {"failure_type", "timestamp_s", "probability", "description", "schema_version"}
)
_GROUP_FIELDS = frozenset(
    {
        "group_id",
        "anchor_id",
        "source_trajectory_id",
        "source_seed",
        "split_group_id",
        "dataset_split",
        "task_id",
        "state_content_digest",
        "state_vector_semantic",
        "continuation_identity",
        "source_sample_id",
        "corrupted_sample_ids",
        "baseline_evidence_id",
        "schema_version",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_id",
        "split_policy_id",
        "source_trajectory_id",
        "source_seed",
        "split_group_id",
        "dataset_split",
        "state_digests",
        "anchor_ids",
        "proposal_ids",
        "schema_version",
    }
)


class ActionVerifierSerializationError(ValueError):
    """Raised when a compact verifier bundle is malformed, unsafe, or changed."""


class UnsupportedActionVerifierSerializationVersionError(
    ActionVerifierSerializationError
):
    """Raised for unsupported file or logical schema versions."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_array(root: Path, name: str, array: NDArray[Any]) -> dict[str, object]:
    if array.dtype.hasobject:
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.arrays.{name}: object dtype is unsafe"
        )
    relative = PurePosixPath("arrays") / _ARRAY_FILES[name]
    destination = _safe_array_path(root, relative.as_posix(), writing=True)
    with destination.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return {
        "path": relative.as_posix(),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": _sha256_file(destination),
    }


def _read_array(root: Path, value: object, name: str) -> tuple[NDArray[Any], str]:
    context = f"ActionVerifierDataset.arrays.{name}"
    reference = _mapping(value, context)
    _require_exact_fields(reference, _ARRAY_FIELDS, context)
    relative = _string(reference, "path", context)
    expected_relative = f"arrays/{_ARRAY_FILES[name]}"
    if relative != expected_relative:
        raise ActionVerifierSerializationError(
            f"{context}.path: expected {expected_relative!r}"
        )
    dtype_text = _string(reference, "dtype", context)
    try:
        expected_dtype = np.dtype(dtype_text)
    except (TypeError, ValueError) as exc:
        raise ActionVerifierSerializationError(
            f"{context}.dtype: invalid dtype"
        ) from exc
    if expected_dtype.hasobject:
        raise ActionVerifierSerializationError(
            f"{context}.dtype: object dtype is unsafe"
        )
    shape = tuple(
        _nonnegative_integer(item, f"{context}.shape[{index}]")
        for index, item in enumerate(_list(reference, "shape", context))
    )
    expected_content = _string(reference, "content_sha256", context)
    if len(expected_content) != 64 or any(
        character not in "0123456789abcdef" for character in expected_content
    ):
        raise ActionVerifierSerializationError(
            f"{context}.content_sha256: expected 64 lowercase hex characters"
        )
    path = _safe_array_path(root, relative, writing=False)
    if _sha256_file(path) != expected_content:
        raise ActionVerifierSerializationError(
            f"{context}.content_sha256: file content mismatch"
        )
    try:
        with path.open("rb") as stream:
            loaded = np.load(stream, allow_pickle=False)
    except Exception as exc:
        raise ActionVerifierSerializationError(
            f"{context}: could not load safe NPY content"
        ) from exc
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise ActionVerifierSerializationError(f"{context}: expected one NPY array")
    if loaded.dtype != expected_dtype:
        raise ActionVerifierSerializationError(
            f"{context}.dtype: expected {expected_dtype}, got {loaded.dtype}"
        )
    if loaded.shape != shape:
        raise ActionVerifierSerializationError(
            f"{context}.shape: expected {shape}, got {loaded.shape}"
        )
    return loaded, relative


def _encode_failure(event: FailureEvent) -> dict[str, object]:
    return {
        "failure_type": event.failure_type,
        "timestamp_s": event.timestamp_s,
        "probability": event.probability,
        "description": event.description,
        "schema_version": event.schema_version,
    }


def _encode_sample(sample: ActionVerifierSampleV1, index: int) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "array_index": index,
        "anchor_id": sample.anchor_id,
        "source_trajectory_id": sample.source_trajectory_id,
        "source_seed": sample.source_seed,
        "split_group_id": sample.split_group_id,
        "dataset_split": sample.dataset_split.value,
        "task_id": sample.task_id,
        "instruction": sample.instruction,
        "state_content_digest": sample.state_content_digest,
        "state_vector_semantic": sample.state_vector_semantic,
        "state_vector_schema_digest": sample.state_vector_schema_digest,
        "continuation_identity": sample.continuation_identity,
        "candidate_type": sample.candidate_type.value,
        "proposal_id": sample.proposal_id,
        "corruption_type": sample.corruption_type,
        "severity_id": sample.severity_id,
        "final_task_success": sample.final_task_success,
        "progress_semantic": sample.progress_semantic,
        "progress_before": sample.progress_before,
        "progress_after_candidate_chunk": sample.progress_after_candidate_chunk,
        "progress_delta": sample.progress_delta,
        "final_unsafe": sample.final_unsafe,
        "failure_events": [_encode_failure(event) for event in sample.failure_events],
        "strong_simulator_evidence_id": sample.strong_simulator_evidence_id,
        "evidence_dataset_digest": sample.evidence_dataset_digest,
        "source_dataset_digest": sample.source_dataset_digest,
        "corruption_dataset_digest": sample.corruption_dataset_digest,
        "compatibility_identity": sample.compatibility_identity,
        "adapter_version": sample.adapter_version,
        "label_source": sample.label_source.value,
        "label_strength": sample.label_strength.value,
        "simulator_replay_verified": sample.simulator_replay_verified,
        "schema_version": sample.schema_version,
    }


def _encode_group(group: ActionVerifierCandidateGroupV1) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "anchor_id": group.anchor_id,
        "source_trajectory_id": group.source_trajectory_id,
        "source_seed": group.source_seed,
        "split_group_id": group.split_group_id,
        "dataset_split": group.dataset_split.value,
        "task_id": group.task_id,
        "state_content_digest": group.state_content_digest,
        "state_vector_semantic": group.state_vector_semantic,
        "continuation_identity": group.continuation_identity,
        "source_sample_id": group.source_sample_id,
        "corrupted_sample_ids": list(group.corrupted_sample_ids),
        "baseline_evidence_id": group.baseline_evidence_id,
        "schema_version": group.schema_version,
    }


def _encode_assignment(assignment: TrajectorySplitAssignmentV1) -> dict[str, object]:
    return {
        "assignment_id": assignment.assignment_id,
        "split_policy_id": assignment.split_policy_id,
        "source_trajectory_id": assignment.source_trajectory_id,
        "source_seed": assignment.source_seed,
        "split_group_id": assignment.split_group_id,
        "dataset_split": assignment.dataset_split.value,
        "state_digests": list(assignment.state_digests),
        "anchor_ids": list(assignment.anchor_ids),
        "proposal_ids": list(assignment.proposal_ids),
        "schema_version": assignment.schema_version,
    }


def save_action_verifier_dataset(
    dataset: ActionVerifierDatasetV1, output_dir: Path
) -> Path:
    """Validate and transactionally publish a new compact verifier bundle."""

    validate_action_verifier_dataset(dataset)
    destination = Path(output_dir).absolute()
    _require_empty_destination(destination)
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'verifier'}.staging-",
                dir=destination.parent,
            )
        )
        (staging / "arrays").mkdir(exist_ok=False)
        arrays = {
            "state_vectors": np.stack(
                [sample.state_vector for sample in dataset.samples], axis=0
            ),
            "candidate_action_chunks": np.stack(
                [sample.candidate_action_chunk for sample in dataset.samples], axis=0
            ),
            "action_masks": np.stack(
                [sample.action_mask for sample in dataset.samples], axis=0
            ),
        }
        references = {
            name: _write_array(staging, name, arrays[name]) for name in _ARRAY_NAMES
        }
        manifest: dict[str, object] = {
            "format": SERIALIZATION_FORMAT,
            "serialization_version": SERIALIZATION_VERSION,
            "dataset_schema_version": dataset.schema_version,
            "dataset_content_digest": dataset.content_digest,
            "sample_count": len(dataset.samples),
            "group_count": len(dataset.candidate_groups),
            "trajectory_count": len(dataset.split_assignments),
            "array_count": len(references),
            "chunk_horizon": dataset.chunk_horizon,
            "action_dimension": dataset.action_dimension,
            "state_vector_dimension": dataset.state_vector_dimension,
            "state_vector_semantic": dataset.state_vector_semantic,
            "state_vector_schema_digest": dataset.state_vector_schema_digest,
            "split_policy_id": dataset.split_policy_id,
            "arrays": references,
            "samples": [
                _encode_sample(sample, index)
                for index, sample in enumerate(dataset.samples)
            ],
            "candidate_groups": [
                _encode_group(group) for group in dataset.candidate_groups
            ],
            "split_assignments": [
                _encode_assignment(assignment)
                for assignment in dataset.split_assignments
            ],
        }
        payload = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        with (staging / MANIFEST_NAME).open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(payload + "\n")
        reloaded = load_action_verifier_dataset(staging)
        if reloaded.content_digest != dataset.content_digest:
            raise ActionVerifierSerializationError(
                "ActionVerifierDataset: staged round-trip content changed"
            )
        _publish_staging_directory(staging, destination)
    except ActionVerifierSerializationError as exc:
        primary_error = exc
        raise
    except (OSError, TypeError, ValueError) as exc:
        wrapped = ActionVerifierSerializationError(
            f"ActionVerifierDataset: could not save transactionally: {exc}"
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
                    raise ActionVerifierSerializationError(
                        "ActionVerifierDataset: could not clean staging directory"
                    ) from cleanup_error
    return destination / MANIFEST_NAME


def _load_action_verifier_dataset(output_dir: Path) -> ActionVerifierDatasetV1:
    """Implement strict reload after the public error-boundary wrapper."""

    requested = Path(output_dir).absolute()
    _require_safe_root(requested)
    root = requested.resolve()
    _validate_root_inventory(root)
    manifest_path = root / MANIFEST_NAME
    _require_regular_unlinked_file(manifest_path, "ActionVerifierDataset.manifest")
    try:
        raw = cast(
            object,
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
    except ActionVerifierSerializationError:
        raise
    except Exception as exc:
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.manifest: could not read safely: {exc}"
        ) from exc
    manifest = _mapping(raw, "ActionVerifierDataset.manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "ActionVerifierDataset.manifest")
    if (
        _string(manifest, "format", "ActionVerifierDataset.manifest")
        != SERIALIZATION_FORMAT
    ):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.manifest.format: unsupported format"
        )
    version = _integer(
        manifest, "serialization_version", "ActionVerifierDataset.manifest"
    )
    if version != SERIALIZATION_VERSION:
        raise UnsupportedActionVerifierSerializationVersionError(
            "ActionVerifierDataset.manifest.serialization_version: "
            f"unsupported version {version!r}"
        )
    schema = _string(
        manifest, "dataset_schema_version", "ActionVerifierDataset.manifest"
    )
    if schema != ACTION_VERIFIER_SCHEMA_VERSION:
        raise UnsupportedActionVerifierSerializationVersionError(
            "ActionVerifierDataset.manifest.dataset_schema_version: "
            f"unsupported version {schema!r}"
        )
    array_count = _integer(manifest, "array_count", "ActionVerifierDataset.manifest")
    if array_count != len(_ARRAY_NAMES):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.manifest.array_count: expected three stacked arrays"
        )
    encoded_arrays = _mapping(
        _field(manifest, "arrays", "ActionVerifierDataset.manifest"),
        "ActionVerifierDataset.arrays",
    )
    _require_exact_fields(
        encoded_arrays, frozenset(_ARRAY_NAMES), "ActionVerifierDataset.arrays"
    )
    arrays: dict[str, NDArray[Any]] = {}
    references: list[str] = []
    for name in _ARRAY_NAMES:
        array, relative = _read_array(root, encoded_arrays[name], name)
        arrays[name] = array
        references.append(relative)
    _validate_array_inventory(root, references)

    encoded_samples = _list(manifest, "samples", "ActionVerifierDataset.manifest")
    sample_count = _integer(manifest, "sample_count", "ActionVerifierDataset.manifest")
    if sample_count != len(encoded_samples):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.manifest.sample_count: does not match samples"
        )
    if any(array.shape[0] != sample_count for array in arrays.values()):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.arrays: leading dimension does not match samples"
        )
    samples = tuple(
        _decode_sample(value, arrays, index)
        for index, value in enumerate(encoded_samples)
    )
    groups = tuple(
        _decode_group(value, index)
        for index, value in enumerate(
            _list(manifest, "candidate_groups", "ActionVerifierDataset.manifest")
        )
    )
    assignments = tuple(
        _decode_assignment(value, index)
        for index, value in enumerate(
            _list(manifest, "split_assignments", "ActionVerifierDataset.manifest")
        )
    )
    if _integer(manifest, "group_count", "ActionVerifierDataset.manifest") != len(
        groups
    ):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.manifest.group_count: does not match groups"
        )
    if _integer(manifest, "trajectory_count", "ActionVerifierDataset.manifest") != len(
        assignments
    ):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.manifest.trajectory_count: "
            "does not match assignments"
        )
    dataset = ActionVerifierDatasetV1(
        state_vector_semantic=_string(
            manifest, "state_vector_semantic", "ActionVerifierDataset.manifest"
        ),
        state_vector_schema_digest=_string(
            manifest, "state_vector_schema_digest", "ActionVerifierDataset.manifest"
        ),
        state_vector_dimension=_integer(
            manifest, "state_vector_dimension", "ActionVerifierDataset.manifest"
        ),
        action_dimension=_integer(
            manifest, "action_dimension", "ActionVerifierDataset.manifest"
        ),
        split_policy_id=_string(
            manifest, "split_policy_id", "ActionVerifierDataset.manifest"
        ),
        samples=samples,
        candidate_groups=groups,
        split_assignments=assignments,
        chunk_horizon=_integer(
            manifest, "chunk_horizon", "ActionVerifierDataset.manifest"
        ),
        schema_version=schema,
    )
    expected_digest = _string(
        manifest, "dataset_content_digest", "ActionVerifierDataset.manifest"
    )
    if dataset.content_digest != expected_digest:
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.manifest.dataset_content_digest: content mismatch"
        )
    return dataset


def load_action_verifier_dataset(output_dir: Path) -> ActionVerifierDatasetV1:
    """Safely reload and fully validate a compact verifier bundle."""

    try:
        return _load_action_verifier_dataset(output_dir)
    except ActionVerifierSerializationError:
        raise
    except (ActionVerifierValidationError, OSError, ValueError) as exc:
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset: invalid serialized content: {exc}"
        ) from exc


def _decode_sample(
    value: object,
    arrays: Mapping[str, NDArray[Any]],
    index: int,
) -> ActionVerifierSampleV1:
    context = f"ActionVerifierDataset.samples[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _SAMPLE_FIELDS, context)
    array_index = _integer(item, "array_index", context)
    if array_index != index:
        raise ActionVerifierSerializationError(
            f"{context}.array_index: expected deterministic row {index}"
        )
    sample = ActionVerifierSampleV1(
        anchor_id=_string(item, "anchor_id", context),
        source_trajectory_id=_string(item, "source_trajectory_id", context),
        source_seed=_integer(item, "source_seed", context),
        split_group_id=_string(item, "split_group_id", context),
        dataset_split=_enum(item, "dataset_split", context, DatasetSplit),
        task_id=_string(item, "task_id", context),
        instruction=_string(item, "instruction", context),
        state_content_digest=_string(item, "state_content_digest", context),
        state_vector_semantic=_string(item, "state_vector_semantic", context),
        state_vector_schema_digest=_string(item, "state_vector_schema_digest", context),
        state_vector=arrays["state_vectors"][array_index],
        candidate_action_chunk=arrays["candidate_action_chunks"][array_index],
        action_mask=arrays["action_masks"][array_index],
        continuation_identity=_string(item, "continuation_identity", context),
        candidate_type=_enum(item, "candidate_type", context, CandidateType),
        proposal_id=_optional_string(item, "proposal_id", context),
        corruption_type=_optional_string(item, "corruption_type", context),
        severity_id=_optional_string(item, "severity_id", context),
        final_task_success=_boolean(item, "final_task_success", context),
        progress_semantic=_optional_string(item, "progress_semantic", context),
        progress_before=_optional_number(item, "progress_before", context),
        progress_after_candidate_chunk=_optional_number(
            item, "progress_after_candidate_chunk", context
        ),
        progress_delta=_optional_number(item, "progress_delta", context),
        final_unsafe=_boolean(item, "final_unsafe", context),
        failure_events=tuple(
            _decode_failure(event, context, failure_index)
            for failure_index, event in enumerate(
                _list(item, "failure_events", context)
            )
        ),
        strong_simulator_evidence_id=_string(
            item, "strong_simulator_evidence_id", context
        ),
        evidence_dataset_digest=_string(item, "evidence_dataset_digest", context),
        source_dataset_digest=_string(item, "source_dataset_digest", context),
        corruption_dataset_digest=_string(item, "corruption_dataset_digest", context),
        compatibility_identity=_string(item, "compatibility_identity", context),
        adapter_version=_string(item, "adapter_version", context),
        label_source=_enum(item, "label_source", context, LabelSource),
        label_strength=_enum(item, "label_strength", context, LabelStrength),
        simulator_replay_verified=_boolean(item, "simulator_replay_verified", context),
        schema_version=_string(item, "schema_version", context),
    )
    if sample.sample_id != _string(item, "sample_id", context):
        raise ActionVerifierSerializationError(f"{context}.sample_id: content mismatch")
    return sample


def _decode_failure(value: object, parent: str, index: int) -> FailureEvent:
    context = f"{parent}.failure_events[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _FAILURE_FIELDS, context)
    return FailureEvent(
        failure_type=_string(item, "failure_type", context),
        timestamp_s=_optional_number(item, "timestamp_s", context),
        probability=_optional_number(item, "probability", context),
        description=_optional_string(item, "description", context),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_group(value: object, index: int) -> ActionVerifierCandidateGroupV1:
    context = f"ActionVerifierDataset.candidate_groups[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _GROUP_FIELDS, context)
    group = ActionVerifierCandidateGroupV1(
        anchor_id=_string(item, "anchor_id", context),
        source_trajectory_id=_string(item, "source_trajectory_id", context),
        source_seed=_integer(item, "source_seed", context),
        split_group_id=_string(item, "split_group_id", context),
        dataset_split=_enum(item, "dataset_split", context, DatasetSplit),
        task_id=_string(item, "task_id", context),
        state_content_digest=_string(item, "state_content_digest", context),
        state_vector_semantic=_string(item, "state_vector_semantic", context),
        continuation_identity=_string(item, "continuation_identity", context),
        source_sample_id=_string(item, "source_sample_id", context),
        corrupted_sample_ids=tuple(
            _string_value(value, f"{context}.corrupted_sample_ids[{position}]")
            for position, value in enumerate(
                _list(item, "corrupted_sample_ids", context)
            )
        ),
        baseline_evidence_id=_string(item, "baseline_evidence_id", context),
        schema_version=_string(item, "schema_version", context),
    )
    if group.group_id != _string(item, "group_id", context):
        raise ActionVerifierSerializationError(f"{context}.group_id: content mismatch")
    return group


def _decode_assignment(value: object, index: int) -> TrajectorySplitAssignmentV1:
    context = f"ActionVerifierDataset.split_assignments[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _ASSIGNMENT_FIELDS, context)
    assignment = TrajectorySplitAssignmentV1(
        split_policy_id=_string(item, "split_policy_id", context),
        source_trajectory_id=_string(item, "source_trajectory_id", context),
        source_seed=_integer(item, "source_seed", context),
        split_group_id=_string(item, "split_group_id", context),
        dataset_split=_enum(item, "dataset_split", context, DatasetSplit),
        state_digests=_string_tuple(item, "state_digests", context),
        anchor_ids=_string_tuple(item, "anchor_ids", context),
        proposal_ids=_string_tuple(item, "proposal_ids", context),
        schema_version=_string(item, "schema_version", context),
    )
    if assignment.assignment_id != _string(item, "assignment_id", context):
        raise ActionVerifierSerializationError(
            f"{context}.assignment_id: content mismatch"
        )
    return assignment


def _safe_array_path(root: Path, relative: str, *, writing: bool) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or "\x00" in relative
        or "\\" in relative
        or pure.is_absolute()
        or len(pure.parts) != 2
        or pure.parts[0] != "arrays"
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.suffix != ".npy"
    ):
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.ndarray.path: unsafe path {relative!r}"
        )
    resolved_root = root.resolve()
    unresolved = resolved_root / Path(*pure.parts)
    candidate = unresolved.resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.ndarray.path: path escapes bundle {relative!r}"
        ) from exc
    if candidate != unresolved.absolute():
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.ndarray.path: links are unsafe {relative!r}"
        )
    if writing:
        if candidate.parent != resolved_root / "arrays":
            raise ActionVerifierSerializationError(
                f"ActionVerifierDataset.ndarray.path: unsafe parent {relative!r}"
            )
    else:
        _require_regular_unlinked_file(candidate, "ActionVerifierDataset.ndarray")
    return candidate


def _require_empty_destination(destination: Path) -> None:
    try:
        if destination.is_symlink() or destination.resolve() != destination:
            raise ActionVerifierSerializationError(
                "ActionVerifierDataset.output_dir: links and junctions are unsupported"
            )
    except ActionVerifierSerializationError:
        raise
    except OSError as exc:
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.output_dir: could not inspect destination: {exc}"
        ) from exc
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.output_dir: must be a directory"
        )
    try:
        next(destination.iterdir())
    except StopIteration:
        return
    raise ActionVerifierSerializationError(
        "ActionVerifierDataset.output_dir: destination must be absent or empty"
    )


def _require_safe_root(root: Path) -> None:
    try:
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise ActionVerifierSerializationError(
                "ActionVerifierDataset.output_dir: missing or unsafe real directory"
            )
    except ActionVerifierSerializationError:
        raise
    except OSError as exc:
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.output_dir: could not inspect root: {exc}"
        ) from exc


def _validate_root_inventory(root: Path) -> None:
    try:
        entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as exc:
        raise ActionVerifierSerializationError(
            f"ActionVerifierDataset.output_dir: could not inspect inventory: {exc}"
        ) from exc
    if set(entries) != {MANIFEST_NAME, "arrays"}:
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.output_dir: inventory is incomplete or has extras"
        )
    arrays_dir = entries["arrays"]
    if (
        arrays_dir.is_symlink()
        or not arrays_dir.is_dir()
        or arrays_dir.resolve() != arrays_dir.absolute()
    ):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.arrays: missing or unsafe real directory"
        )


def _validate_array_inventory(root: Path, references: Collection[str]) -> None:
    if len(references) != len(set(references)):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.arrays: duplicate references are invalid"
        )
    actual: set[str] = set()
    for path in (root / "arrays").iterdir():
        if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
            raise ActionVerifierSerializationError(
                "ActionVerifierDataset.arrays: unsafe or unexpected entry"
            )
        _require_regular_unlinked_file(path, "ActionVerifierDataset.ndarray")
        actual.add(path.relative_to(root).as_posix())
    if actual != set(references):
        raise ActionVerifierSerializationError(
            "ActionVerifierDataset.arrays: files do not exactly match manifest"
        )


def _require_regular_unlinked_file(path: Path, context: str) -> None:
    try:
        if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
            raise ActionVerifierSerializationError(
                f"{context}: missing or unsafe regular file"
            )
        if path.stat().st_nlink != 1:
            raise ActionVerifierSerializationError(f"{context}: hard links are unsafe")
    except ActionVerifierSerializationError:
        raise
    except OSError as exc:
        raise ActionVerifierSerializationError(
            f"{context}: could not inspect file: {exc}"
        ) from exc


def _publish_staging_directory(staging: Path, destination: Path) -> None:
    destination_existed = destination.exists()
    if destination_existed:
        _require_empty_destination(destination)
        destination.rmdir()
    try:
        staging.replace(destination)
    except OSError:
        if destination_existed and not destination.exists():
            destination.mkdir()
        raise


def _require_exact_fields(
    item: Mapping[str, object], expected: Collection[str], context: str
) -> None:
    missing = sorted(set(expected) - set(item))
    extra = sorted(set(item) - set(expected))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ActionVerifierSerializationError(
            f"{context}: invalid fields ({'; '.join(details)})"
        )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ActionVerifierSerializationError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _field(item: Mapping[str, object], field: str, context: str) -> object:
    if field not in item:
        raise ActionVerifierSerializationError(f"{context}.{field}: missing field")
    return item[field]


def _string_value(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ActionVerifierSerializationError(f"{context}: expected a string")
    return value


def _string(item: Mapping[str, object], field: str, context: str) -> str:
    return _string_value(_field(item, field, context), f"{context}.{field}")


def _optional_string(
    item: Mapping[str, object], field: str, context: str
) -> str | None:
    value = _field(item, field, context)
    return None if value is None else _string_value(value, f"{context}.{field}")


def _integer(item: Mapping[str, object], field: str, context: str) -> int:
    value = _field(item, field, context)
    if type(value) is not int:
        raise ActionVerifierSerializationError(f"{context}.{field}: expected integer")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ActionVerifierSerializationError(
            f"{context}: expected non-negative integer"
        )
    return value


def _boolean(item: Mapping[str, object], field: str, context: str) -> bool:
    value = _field(item, field, context)
    if type(value) is not bool:
        raise ActionVerifierSerializationError(f"{context}.{field}: expected boolean")
    return value


def _optional_number(
    item: Mapping[str, object], field: str, context: str
) -> float | None:
    value = _field(item, field, context)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ActionVerifierSerializationError(
            f"{context}.{field}: expected finite number or null"
        )
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ActionVerifierSerializationError(
            f"{context}.{field}: expected finite number or null"
        )
    return number


def _list(item: Mapping[str, object], field: str, context: str) -> list[object]:
    value = _field(item, field, context)
    if not isinstance(value, list):
        raise ActionVerifierSerializationError(f"{context}.{field}: expected an array")
    return cast(list[object], value)


def _string_tuple(
    item: Mapping[str, object], field: str, context: str
) -> tuple[str, ...]:
    return tuple(
        _string_value(value, f"{context}.{field}[{index}]")
        for index, value in enumerate(_list(item, field, context))
    )


def _enum(
    item: Mapping[str, object],
    field: str,
    context: str,
    enum_type: type[_StrEnumT],
) -> _StrEnumT:
    value = _string(item, field, context)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ActionVerifierSerializationError(
            f"{context}.{field}: unsupported value {value!r}"
        ) from exc


def _reject_json_constant(value: str) -> NoReturn:
    raise ActionVerifierSerializationError(
        f"ActionVerifierDataset.manifest: non-finite value {value!r} is unsupported"
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ActionVerifierSerializationError(
                f"ActionVerifierDataset.manifest: duplicate field {key!r}"
            )
        result[key] = value
    return result


__all__ = [
    "MANIFEST_NAME",
    "SERIALIZATION_FORMAT",
    "SERIALIZATION_VERSION",
    "ActionVerifierSerializationError",
    "UnsupportedActionVerifierSerializationVersionError",
    "load_action_verifier_dataset",
    "save_action_verifier_dataset",
]
