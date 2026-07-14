"""Safe, transactional serialization for unlabeled corruption proposals."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.layout import (
    ACTION_LAYOUT_SCHEMA_VERSION,
    ActionField,
    ActionLayout,
    ActionLayoutError,
    ActionSemantic,
    validate_action_layout,
)
from latentguard.corruptions.models import (
    CORRUPTION_SCHEMA_VERSION,
    CorruptedActionProposal,
    CorruptedActionProposalError,
    ResolvedParameterValue,
    validate_corrupted_action_proposal,
)
from latentguard.corruptions.registry import create_corruption
from latentguard.models import CURRENT_SCHEMA_VERSION, ActionChunk, JsonScalar
from latentguard.validation import DataValidationError, validate_action_chunk

MANIFEST_NAME = "manifest.json"
"""Filename used for a corruption dataset manifest."""

SERIALIZATION_FORMAT = "latentguard-corruption-dataset"
"""Stable format discriminator written to every manifest."""

SERIALIZATION_VERSION = 1
"""Serialization version supported by this release."""

CORRUPTION_DATASET_SCHEMA_VERSION = "1.0"
"""Logical corruption-dataset schema version supported by this release."""

_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "serialization_version",
        "dataset_schema_version",
        "source_dataset_id",
        "proposal_count",
        "array_count",
        "action_layout",
        "proposals",
    }
)
_LAYOUT_FIELDS = frozenset(
    {"action_dim", "fields", "schema_version", "description", "metadata"}
)
_ACTION_FIELD_FIELDS = frozenset(
    {"name", "indices", "semantic", "units", "description", "metadata"}
)
_PROPOSAL_FIELDS = frozenset(
    {
        "proposal_id",
        "source_episode_id",
        "source_candidate_id",
        "source_policy_id",
        "source_task_id",
        "split_group_id",
        "transformed_action",
        "corruption_type",
        "resolved_parameters",
        "seed",
        "generation_ordinal",
        "schema_version",
        "notes",
    }
)
_ACTION_FIELDS = frozenset(
    {"actions", "coordinate_frame", "control_period_s", "schema_version"}
)
_ARRAY_FIELDS = frozenset({"path", "dtype", "shape"})


class CorruptionSerializationError(ValueError):
    """Raised when a corruption dataset is malformed, unsafe, or invalid."""


class UnsupportedCorruptionSerializationVersionError(CorruptionSerializationError):
    """Raised when a dataset or contained model uses an unsupported version."""


@dataclass(frozen=True, slots=True)
class CorruptionDataset:
    """An immutable collection of single-source unlabeled action proposals."""

    source_dataset_id: str
    action_layout: ActionLayout
    proposals: tuple[CorruptedActionProposal, ...]
    schema_version: str = CORRUPTION_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze the proposal collection and validate the complete dataset."""
        object.__setattr__(self, "proposals", tuple(self.proposals))
        validate_corruption_dataset(self)


class _ArrayWriter:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.arrays_dir = self.root / "arrays"
        self.arrays_dir.mkdir(parents=True, exist_ok=False)
        self.count = 0

    def write(self, array: NDArray[Any]) -> dict[str, object]:
        if array.dtype.hasobject:
            raise CorruptionSerializationError(
                "CorruptionDataset.ndarray.dtype: object arrays are unsafe"
            )
        relative = PurePosixPath("arrays") / f"{self.count:06d}.npy"
        destination = _safe_array_path(self.root, relative.as_posix(), writing=True)
        with destination.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        self.count += 1
        return {
            "path": relative.as_posix(),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }


class _ArrayReader:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.references: list[str] = []


def validate_corruption_dataset(dataset: CorruptionDataset) -> None:
    """Validate dataset identity, layout, proposal provenance, and dimensions."""

    if not isinstance(dataset.source_dataset_id, str) or not (
        source_dataset_id := dataset.source_dataset_id.strip()
    ):
        raise CorruptionSerializationError(
            "CorruptionDataset.source_dataset_id: expected a non-empty string"
        )
    if source_dataset_id != dataset.source_dataset_id:
        raise CorruptionSerializationError(
            "CorruptionDataset.source_dataset_id: surrounding whitespace is invalid"
        )
    if any(character in dataset.source_dataset_id for character in ("\x00", "/", "\\")):
        raise CorruptionSerializationError(
            "CorruptionDataset.source_dataset_id: must be a path-independent identifier"
        )
    if dataset.schema_version != CORRUPTION_DATASET_SCHEMA_VERSION:
        raise UnsupportedCorruptionSerializationVersionError(
            "CorruptionDataset.schema_version: unsupported version "
            f"{dataset.schema_version!r}; supported: "
            f"{CORRUPTION_DATASET_SCHEMA_VERSION}"
        )
    if not isinstance(dataset.action_layout, ActionLayout):
        raise CorruptionSerializationError(
            "CorruptionDataset.action_layout: expected ActionLayout"
        )
    try:
        validate_action_layout(dataset.action_layout)
    except ActionLayoutError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.action_layout: {exc}"
        ) from exc

    proposal_ids: set[str] = set()
    for index, proposal in enumerate(dataset.proposals):
        context = f"CorruptionDataset.proposals[{index}]"
        if not isinstance(proposal, CorruptedActionProposal):
            raise CorruptionSerializationError(
                f"{context}: expected CorruptedActionProposal"
            )
        try:
            validate_corrupted_action_proposal(proposal)
            validate_action_chunk(proposal.transformed_action, proposal.proposal_id)
        except (CorruptedActionProposalError, DataValidationError) as exc:
            raise CorruptionSerializationError(f"{context}: {exc}") from exc
        if proposal.schema_version != CORRUPTION_SCHEMA_VERSION:
            raise UnsupportedCorruptionSerializationVersionError(
                f"{context}.schema_version: unsupported version "
                f"{proposal.schema_version!r}; supported: {CORRUPTION_SCHEMA_VERSION}"
            )
        if proposal.generation_ordinal != index:
            raise CorruptionSerializationError(
                f"{context}.generation_ordinal: expected stable output position "
                f"{index}, got {proposal.generation_ordinal}"
            )
        if proposal.proposal_id in proposal_ids:
            raise CorruptionSerializationError(
                f"{context}.proposal_id: duplicate identifier {proposal.proposal_id!r}"
            )
        proposal_ids.add(proposal.proposal_id)
        actions = proposal.transformed_action.actions
        if actions.ndim != 2 or actions.shape[1] != dataset.action_layout.action_dim:
            raise CorruptionSerializationError(
                f"{context}.transformed_action.actions: action dimension does not "
                f"match layout ({actions.shape!r} versus "
                f"{dataset.action_layout.action_dim})"
            )
        _validate_resolved_corruption(proposal, dataset.action_layout, context)


def _validate_resolved_corruption(
    proposal: CorruptedActionProposal,
    layout: ActionLayout,
    context: str,
) -> None:
    """Validate that serialized parameters exactly describe one built-in M1 type."""
    parameters = dict(proposal.resolved_parameters)
    factory_parameters = dict(parameters)
    if proposal.corruption_type == "temporal_field_shift":
        factory_parameters.pop("target_indices", None)
    try:
        corruption = create_corruption(proposal.corruption_type, factory_parameters)
        validation_action = _validation_source_action(proposal)
        expected = corruption.resolved_parameters_for(validation_action, layout)
    except (TypeError, ValueError) as exc:
        raise CorruptionSerializationError(
            f"{context}.corruption_type/resolved_parameters: {exc}"
        ) from exc
    if not _resolved_parameters_equal(expected, proposal.resolved_parameters):
        raise CorruptionSerializationError(
            f"{context}.resolved_parameters: values do not match the fully "
            f"resolved {proposal.corruption_type!r} configuration"
        )


def _validation_source_action(proposal: CorruptedActionProposal) -> ActionChunk:
    """Recover an integer-bias source when value-aware applicability requires it."""
    transformed = proposal.transformed_action
    if proposal.corruption_type != "constant_bias" or not np.issubdtype(
        transformed.actions.dtype, np.integer
    ):
        return transformed
    raw_indices = proposal.resolved_parameters.get("target_indices")
    raw_bias = proposal.resolved_parameters.get("bias")
    if not isinstance(raw_indices, tuple) or not isinstance(raw_bias, tuple):
        return transformed
    source = np.array(transformed.actions, copy=True, order="C")
    limits = np.iinfo(source.dtype)
    try:
        for index, bias in zip(raw_indices, raw_bias, strict=True):
            if type(index) is not int or type(bias) not in (int, float):
                return transformed
            delta = int(cast(int | float, bias))
            recovered = [int(value) - delta for value in source[:, index]]
            if any(value < limits.min or value > limits.max for value in recovered):
                return transformed
            source[:, index] = recovered
    except (IndexError, TypeError, ValueError):
        return transformed
    return ActionChunk(
        actions=source,
        coordinate_frame=transformed.coordinate_frame,
        control_period_s=transformed.control_period_s,
        schema_version=transformed.schema_version,
    )


def _resolved_parameters_equal(
    left: Mapping[str, ResolvedParameterValue],
    right: Mapping[str, ResolvedParameterValue],
) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if type(left_value) is not type(right_value):
            return False
        if isinstance(left_value, tuple):
            if not isinstance(right_value, tuple) or len(left_value) != len(
                right_value
            ):
                return False
            if any(
                type(left_item) is not type(right_item) or left_item != right_item
                for left_item, right_item in zip(left_value, right_value, strict=True)
            ):
                return False
        elif left_value != right_value:
            return False
    return True


def save_corruption_dataset(dataset: CorruptionDataset, output_dir: Path) -> Path:
    """Validate and transactionally save a corruption dataset to a new bundle."""

    validate_corruption_dataset(dataset)
    destination = Path(output_dir).absolute()
    _require_empty_destination(destination)
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'corruptions'}.staging-",
                dir=destination.parent,
            )
        )
        writer = _ArrayWriter(staging)
        encoded_proposals = [
            _encode_proposal(proposal, writer) for proposal in dataset.proposals
        ]
        manifest: dict[str, object] = {
            "format": SERIALIZATION_FORMAT,
            "serialization_version": SERIALIZATION_VERSION,
            "dataset_schema_version": dataset.schema_version,
            "source_dataset_id": dataset.source_dataset_id,
            "proposal_count": len(dataset.proposals),
            "array_count": writer.count,
            "action_layout": _encode_layout(dataset.action_layout),
            "proposals": encoded_proposals,
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
        _publish_staging_bundle(staging, destination)
    except CorruptionSerializationError as exc:
        primary_error = exc
        raise
    except (OSError, TypeError, ValueError) as exc:
        wrapped = CorruptionSerializationError(
            f"CorruptionDataset: could not save transactionally: {exc}"
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
                    raise CorruptionSerializationError(
                        "CorruptionDataset: could not clean staging directory"
                    ) from cleanup_error
    return Path(output_dir) / MANIFEST_NAME


def load_corruption_dataset(output_dir: Path) -> CorruptionDataset:
    """Load a corruption dataset without pickle and validate every proposal."""

    requested = Path(output_dir).absolute()
    try:
        unsafe_root = requested.is_symlink() or (
            requested.exists() and requested.resolve() != requested
        )
    except OSError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.output_dir: could not inspect bundle root: {exc}"
        ) from exc
    if unsafe_root:
        raise CorruptionSerializationError(
            "CorruptionDataset.output_dir: symbolic links and junctions are unsupported"
        )
    root = requested.resolve()
    _validate_bundle_root(root)
    manifest_path = root / MANIFEST_NAME
    _require_regular_unlinked_file(manifest_path, "CorruptionDataset.manifest")
    try:
        raw = cast(
            object,
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
    except CorruptionSerializationError:
        raise
    except Exception as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.manifest: could not read: {exc}"
        ) from exc

    manifest = _mapping(raw, "CorruptionDataset.manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "CorruptionDataset.manifest")
    if (
        _string(manifest, "format", "CorruptionDataset.manifest")
        != SERIALIZATION_FORMAT
    ):
        raise CorruptionSerializationError(
            "CorruptionDataset.manifest.format: unsupported format"
        )
    version = _integer(manifest, "serialization_version", "CorruptionDataset.manifest")
    if version != SERIALIZATION_VERSION:
        raise UnsupportedCorruptionSerializationVersionError(
            "CorruptionDataset.manifest.serialization_version: unsupported version "
            f"{version!r}; supported: {SERIALIZATION_VERSION}"
        )
    dataset_schema = _string(
        manifest, "dataset_schema_version", "CorruptionDataset.manifest"
    )
    if dataset_schema != CORRUPTION_DATASET_SCHEMA_VERSION:
        raise UnsupportedCorruptionSerializationVersionError(
            "CorruptionDataset.manifest.dataset_schema_version: unsupported version "
            f"{dataset_schema!r}; supported: {CORRUPTION_DATASET_SCHEMA_VERSION}"
        )
    encoded_proposals = _list(manifest, "proposals", "CorruptionDataset.manifest")
    proposal_count = _integer(manifest, "proposal_count", "CorruptionDataset.manifest")
    if proposal_count < 0 or proposal_count != len(encoded_proposals):
        raise CorruptionSerializationError(
            "CorruptionDataset.manifest.proposal_count: does not match proposals"
        )
    array_count = _integer(manifest, "array_count", "CorruptionDataset.manifest")
    if array_count < 0 or array_count != proposal_count:
        raise CorruptionSerializationError(
            "CorruptionDataset.manifest.array_count: must equal proposal_count"
        )

    reader = _ArrayReader(root)
    try:
        layout = _decode_layout(
            _field(manifest, "action_layout", "CorruptionDataset.manifest")
        )
        proposals = tuple(
            _decode_proposal(value, reader, index)
            for index, value in enumerate(encoded_proposals)
        )
        _validate_array_inventory(reader, array_count)
        return CorruptionDataset(
            source_dataset_id=_string(
                manifest, "source_dataset_id", "CorruptionDataset.manifest"
            ),
            action_layout=layout,
            proposals=proposals,
            schema_version=dataset_schema,
        )
    except (
        ActionLayoutError,
        CorruptedActionProposalError,
        DataValidationError,
    ) as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.manifest: invalid model data: {exc}"
        ) from exc


def _encode_layout(layout: ActionLayout) -> dict[str, object]:
    return {
        "action_dim": layout.action_dim,
        "fields": [
            {
                "name": field.name,
                "indices": list(field.indices),
                "semantic": field.semantic.value,
                "units": field.units,
                "description": field.description,
                "metadata": _encode_scalar_mapping(field.metadata),
            }
            for field in layout.fields
        ],
        "schema_version": layout.schema_version,
        "description": layout.description,
        "metadata": _encode_scalar_mapping(layout.metadata),
    }


def _encode_proposal(
    proposal: CorruptedActionProposal, writer: _ArrayWriter
) -> dict[str, object]:
    action = proposal.transformed_action
    return {
        "proposal_id": proposal.proposal_id,
        "source_episode_id": proposal.source_episode_id,
        "source_candidate_id": proposal.source_candidate_id,
        "source_policy_id": proposal.source_policy_id,
        "source_task_id": proposal.source_task_id,
        "split_group_id": proposal.split_group_id,
        "transformed_action": {
            "actions": writer.write(action.actions),
            "coordinate_frame": action.coordinate_frame,
            "control_period_s": action.control_period_s,
            "schema_version": action.schema_version,
        },
        "corruption_type": proposal.corruption_type,
        "resolved_parameters": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in proposal.resolved_parameters.items()
        },
        "seed": proposal.seed,
        "generation_ordinal": proposal.generation_ordinal,
        "schema_version": proposal.schema_version,
        "notes": proposal.notes,
    }


def _decode_layout(value: object) -> ActionLayout:
    context = "CorruptionDataset.action_layout"
    item = _mapping(value, context)
    _require_exact_fields(item, _LAYOUT_FIELDS, context)
    schema_version = _string(item, "schema_version", context)
    if schema_version != ACTION_LAYOUT_SCHEMA_VERSION:
        raise UnsupportedCorruptionSerializationVersionError(
            f"{context}.schema_version: unsupported version {schema_version!r}; "
            f"supported: {ACTION_LAYOUT_SCHEMA_VERSION}"
        )
    action_dim = _integer(item, "action_dim", context)
    fields = tuple(
        _decode_action_field(field, index)
        for index, field in enumerate(_list(item, "fields", context))
    )
    return ActionLayout(
        action_dim=action_dim,
        fields=fields,
        schema_version=schema_version,
        description=_optional_string(item, "description", context),
        metadata=_decode_scalar_mapping(
            _field(item, "metadata", context), f"{context}.metadata"
        ),
    )


def _decode_action_field(value: object, index: int) -> ActionField:
    context = f"CorruptionDataset.action_layout.fields[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _ACTION_FIELD_FIELDS, context)
    raw_indices = _list(item, "indices", context)
    indices: list[int] = []
    for item_index, raw_index in enumerate(raw_indices):
        if type(raw_index) is not int:
            raise CorruptionSerializationError(
                f"{context}.indices[{item_index}]: expected an integer"
            )
        indices.append(raw_index)
    semantic_text = _string(item, "semantic", context)
    try:
        semantic = ActionSemantic(semantic_text)
    except ValueError as exc:
        raise CorruptionSerializationError(
            f"{context}.semantic: unsupported value {semantic_text!r}"
        ) from exc
    return ActionField(
        name=_string(item, "name", context),
        indices=tuple(indices),
        semantic=semantic,
        units=_optional_string(item, "units", context),
        description=_optional_string(item, "description", context),
        metadata=_decode_scalar_mapping(
            _field(item, "metadata", context), f"{context}.metadata"
        ),
    )


def _decode_proposal(
    value: object, reader: _ArrayReader, index: int
) -> CorruptedActionProposal:
    context = f"CorruptionDataset.proposals[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _PROPOSAL_FIELDS, context)
    schema_version = _string(item, "schema_version", context)
    if schema_version != CORRUPTION_SCHEMA_VERSION:
        raise UnsupportedCorruptionSerializationVersionError(
            f"{context}.schema_version: unsupported version {schema_version!r}; "
            f"supported: {CORRUPTION_SCHEMA_VERSION}"
        )
    raw_parameters = _mapping(
        _field(item, "resolved_parameters", context),
        f"{context}.resolved_parameters",
    )
    parameters: dict[str, ResolvedParameterValue] = {}
    for key, raw_parameter in raw_parameters.items():
        parameters[key] = _decode_parameter(
            raw_parameter, f"{context}.resolved_parameters.{key}"
        )
    return CorruptedActionProposal(
        proposal_id=_string(item, "proposal_id", context),
        source_episode_id=_string(item, "source_episode_id", context),
        source_candidate_id=_string(item, "source_candidate_id", context),
        source_policy_id=_string(item, "source_policy_id", context),
        source_task_id=_string(item, "source_task_id", context),
        split_group_id=_string(item, "split_group_id", context),
        transformed_action=_decode_action(
            _field(item, "transformed_action", context), reader, context
        ),
        corruption_type=_string(item, "corruption_type", context),
        resolved_parameters=parameters,
        seed=_integer(item, "seed", context),
        generation_ordinal=_integer(item, "generation_ordinal", context),
        schema_version=schema_version,
        notes=_optional_string(item, "notes", context),
    )


def _decode_action(value: object, reader: _ArrayReader, context: str) -> ActionChunk:
    action_context = f"{context}.transformed_action"
    item = _mapping(value, action_context)
    _require_exact_fields(item, _ACTION_FIELDS, action_context)
    schema_version = _string(item, "schema_version", action_context)
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise UnsupportedCorruptionSerializationVersionError(
            f"{action_context}.schema_version: unsupported version "
            f"{schema_version!r}; supported: {CURRENT_SCHEMA_VERSION}"
        )
    return ActionChunk(
        actions=_read_array(
            _field(item, "actions", action_context), reader, action_context
        ),
        coordinate_frame=_string(item, "coordinate_frame", action_context),
        control_period_s=_number(item, "control_period_s", action_context),
        schema_version=schema_version,
    )


def _read_array(value: object, reader: _ArrayReader, context: str) -> NDArray[Any]:
    reference = _mapping(value, f"{context}.actions")
    _require_exact_fields(reference, _ARRAY_FIELDS, f"{context}.actions")
    relative = _string(reference, "path", f"{context}.actions")
    dtype_text = _string(reference, "dtype", f"{context}.actions")
    raw_shape = _list(reference, "shape", f"{context}.actions")
    shape: list[int] = []
    for index, raw_dimension in enumerate(raw_shape):
        if type(raw_dimension) is not int or raw_dimension < 0:
            raise CorruptionSerializationError(
                f"{context}.actions.shape[{index}]: expected a non-negative integer"
            )
        shape.append(raw_dimension)
    try:
        expected_dtype = np.dtype(dtype_text)
    except (TypeError, ValueError) as exc:
        raise CorruptionSerializationError(
            f"{context}.actions.dtype: invalid dtype"
        ) from exc
    if expected_dtype.hasobject:
        raise CorruptionSerializationError(
            f"{context}.actions.dtype: object dtype is unsafe"
        )
    path = _safe_array_path(reader.root, relative, writing=False)
    reader.references.append(relative)
    try:
        with path.open("rb") as stream:
            loaded = np.load(stream, allow_pickle=False)
    except Exception as exc:
        raise CorruptionSerializationError(
            f"{context}.actions: could not load safely: {exc}"
        ) from exc
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise CorruptionSerializationError(f"{context}.actions: expected one NPY array")
    if loaded.dtype != expected_dtype:
        raise CorruptionSerializationError(
            f"{context}.actions.dtype: expected {expected_dtype}, got {loaded.dtype}"
        )
    if loaded.shape != tuple(shape):
        raise CorruptionSerializationError(
            f"{context}.actions.shape: expected {tuple(shape)}, got {loaded.shape}"
        )
    return loaded


def _encode_scalar_mapping(value: Mapping[str, JsonScalar]) -> dict[str, JsonScalar]:
    return dict(value)


def _decode_scalar_mapping(value: object, context: str) -> Mapping[str, JsonScalar]:
    item = _mapping(value, context)
    decoded: dict[str, JsonScalar] = {}
    for key, scalar in item.items():
        decoded[key] = _json_scalar(scalar, f"{context}.{key}")
    return MappingProxyType(decoded)


def _decode_parameter(value: object, context: str) -> ResolvedParameterValue:
    if isinstance(value, list):
        return tuple(
            _json_scalar(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    return _json_scalar(value, context)


def _json_scalar(value: object, context: str) -> JsonScalar:
    if value is None or type(value) in (str, int, bool):
        return cast(JsonScalar, value)
    if type(value) is float and math.isfinite(value):
        return value
    raise CorruptionSerializationError(f"{context}: expected a finite JSON scalar")


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
        raise CorruptionSerializationError(
            f"CorruptionDataset.ndarray.path: unsafe path {relative!r}"
        )
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.ndarray.path: path escapes bundle {relative!r}"
        ) from exc
    if writing:
        if candidate.parent.resolve() != (resolved_root / "arrays"):
            raise CorruptionSerializationError(
                f"CorruptionDataset.ndarray.path: unsafe parent for {relative!r}"
            )
    else:
        _require_regular_unlinked_file(candidate, "CorruptionDataset.ndarray")
    return candidate


def _require_empty_destination(destination: Path) -> None:
    try:
        unsafe_destination = (
            destination.is_symlink() or destination.resolve() != destination
        )
    except OSError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.output_dir: could not inspect destination: {exc}"
        ) from exc
    if unsafe_destination:
        raise CorruptionSerializationError(
            "CorruptionDataset.output_dir: symbolic links and junctions are unsupported"
        )
    if not destination.exists():
        return
    if not destination.is_dir():
        raise CorruptionSerializationError(
            "CorruptionDataset.output_dir: must be a directory"
        )
    try:
        next(destination.iterdir())
    except StopIteration:
        return
    except OSError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.output_dir: could not inspect destination: {exc}"
        ) from exc
    raise CorruptionSerializationError(
        "CorruptionDataset.output_dir: destination must be absent or empty"
    )


def _validate_bundle_root(root: Path) -> None:
    if not root.is_dir():
        raise CorruptionSerializationError(
            "CorruptionDataset.output_dir: missing dataset directory"
        )
    expected = {MANIFEST_NAME, "arrays"}
    try:
        entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.output_dir: could not inspect dataset: {exc}"
        ) from exc
    if set(entries) != expected:
        raise CorruptionSerializationError(
            "CorruptionDataset.output_dir: bundle inventory is incomplete or has extras"
        )
    arrays_dir = entries["arrays"]
    try:
        if (
            arrays_dir.is_symlink()
            or not arrays_dir.is_dir()
            or arrays_dir.resolve() != arrays_dir.absolute()
        ):
            raise CorruptionSerializationError(
                "CorruptionDataset.arrays: missing or unsafe directory"
            )
    except CorruptionSerializationError:
        raise
    except OSError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.arrays: could not inspect directory: {exc}"
        ) from exc


def _require_regular_unlinked_file(path: Path, context: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise CorruptionSerializationError(
                f"{context}: missing or unsafe regular file"
            )
        if path.stat().st_nlink != 1:
            raise CorruptionSerializationError(f"{context}: hard links are unsafe")
    except CorruptionSerializationError:
        raise
    except OSError as exc:
        raise CorruptionSerializationError(
            f"{context}: could not inspect file: {exc}"
        ) from exc


def _publish_staging_bundle(staging: Path, destination: Path) -> None:
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


def _validate_array_inventory(reader: _ArrayReader, expected_count: int) -> None:
    if len(reader.references) != expected_count:
        raise CorruptionSerializationError(
            "CorruptionDataset.manifest.array_count: does not match references"
        )
    if len(set(reader.references)) != len(reader.references):
        raise CorruptionSerializationError(
            "CorruptionDataset.arrays: duplicate array references are unsupported"
        )
    arrays_dir = reader.root / "arrays"
    actual_files: set[str] = set()
    try:
        for path in arrays_dir.iterdir():
            if path.is_symlink() or not path.is_file():
                raise CorruptionSerializationError(
                    "CorruptionDataset.arrays: unexpected or unsafe entry"
                )
            if path.stat().st_nlink != 1:
                raise CorruptionSerializationError(
                    "CorruptionDataset.arrays: hard-linked files are unsafe"
                )
            actual_files.add(path.relative_to(reader.root).as_posix())
    except CorruptionSerializationError:
        raise
    except OSError as exc:
        raise CorruptionSerializationError(
            f"CorruptionDataset.arrays: could not inspect inventory: {exc}"
        ) from exc
    if actual_files != set(reader.references):
        raise CorruptionSerializationError(
            "CorruptionDataset.arrays: files do not exactly match manifest references"
        )


def _require_exact_fields(
    item: Mapping[str, object], expected: Collection[str], context: str
) -> None:
    expected_fields = set(expected)
    actual_fields = set(item)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise CorruptionSerializationError(
            f"{context}: invalid fields ({'; '.join(details)})"
        )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CorruptionSerializationError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _field(item: Mapping[str, object], field: str, context: str) -> object:
    if field not in item:
        raise CorruptionSerializationError(f"{context}.{field}: missing required field")
    return item[field]


def _string(item: Mapping[str, object], field: str, context: str) -> str:
    value = _field(item, field, context)
    if not isinstance(value, str):
        raise CorruptionSerializationError(f"{context}.{field}: expected a string")
    return value


def _optional_string(
    item: Mapping[str, object], field: str, context: str
) -> str | None:
    value = _field(item, field, context)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CorruptionSerializationError(
            f"{context}.{field}: expected a string or null"
        )
    return value


def _integer(item: Mapping[str, object], field: str, context: str) -> int:
    value = _field(item, field, context)
    if type(value) is not int:
        raise CorruptionSerializationError(f"{context}.{field}: expected an integer")
    return value


def _number(item: Mapping[str, object], field: str, context: str) -> float:
    value = _field(item, field, context)
    if type(value) not in (int, float):
        raise CorruptionSerializationError(
            f"{context}.{field}: expected a finite number"
        )
    try:
        number = float(cast(int | float, value))
    except OverflowError as exc:
        raise CorruptionSerializationError(
            f"{context}.{field}: number is outside the finite floating-point range"
        ) from exc
    if not math.isfinite(number):
        raise CorruptionSerializationError(
            f"{context}.{field}: expected a finite number"
        )
    if isinstance(value, int) and int(number) != value:
        raise CorruptionSerializationError(
            f"{context}.{field}: integer cannot be represented exactly as a float"
        )
    return number


def _list(item: Mapping[str, object], field: str, context: str) -> list[object]:
    value = _field(item, field, context)
    if not isinstance(value, list):
        raise CorruptionSerializationError(f"{context}.{field}: expected an array")
    return cast(list[object], value)


def _reject_json_constant(value: str) -> NoReturn:
    raise CorruptionSerializationError(
        f"CorruptionDataset.manifest: non-finite JSON constant {value!r} is unsupported"
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CorruptionSerializationError(f"JSON object: duplicate field {key!r}")
        result[key] = value
    return result
