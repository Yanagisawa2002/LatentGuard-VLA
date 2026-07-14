"""Versioned JSON-manifest and safe NumPy-array serialization."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.models import (
    CURRENT_SCHEMA_VERSION,
    ActionChunk,
    CameraFrame,
    CandidateAction,
    Episode,
    FailureEvent,
    JsonScalar,
    LabelSource,
    LabelStrength,
    ObservationFrame,
    ObservationHistory,
    OutcomeLabel,
    SampleProvenance,
)
from latentguard.validation import validate_episodes

MANIFEST_NAME = "manifest.json"
SERIALIZATION_FORMAT = "latentguard-episode-bundle"
SERIALIZATION_VERSION = 1
EnumType = TypeVar("EnumType", bound=Enum)

_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "serialization_version",
        "schema_version",
        "episode_count",
        "array_count",
        "episodes",
    }
)


class SerializationError(ValueError):
    """Raised when serialized episode data is malformed or unsafe."""


class UnsupportedSerializationVersionError(SerializationError):
    """Raised when a manifest or model schema version is unsupported."""


class _ArrayWriter:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.arrays_dir = self.root / "arrays"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write(self, array: NDArray[Any]) -> dict[str, object]:
        if array.dtype.hasobject:
            raise SerializationError("ndarray.dtype: object arrays are not supported")
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


def save_episodes(episodes: Sequence[Episode], output_dir: Path) -> Path:
    """Validate and transactionally save a new episode bundle.

    The destination must be absent or an empty real directory. Files are first
    written exclusively in a fresh sibling staging directory, then the complete
    bundle is published with one directory rename so failures cannot corrupt a
    previously valid bundle or follow pre-existing files.
    """

    episode_tuple = tuple(episodes)
    validate_episodes(episode_tuple)
    destination = Path(output_dir).absolute()
    _require_empty_destination(destination)
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'bundle'}.staging-",
                dir=destination.parent,
            )
        )
        writer = _ArrayWriter(staging)
        encoded = [_encode_value(episode, writer) for episode in episode_tuple]
        manifest: dict[str, object] = {
            "format": SERIALIZATION_FORMAT,
            "serialization_version": SERIALIZATION_VERSION,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "episode_count": len(episode_tuple),
            "array_count": writer.count,
            "episodes": encoded,
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
    except SerializationError as exc:
        primary_error = exc
        raise
    except (OSError, TypeError, ValueError) as exc:
        wrapped = SerializationError(
            f"EpisodeBundle: could not save transactionally: {exc}"
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
                    raise SerializationError(
                        "EpisodeBundle: could not clean staging directory"
                    ) from cleanup_error
    return Path(output_dir) / MANIFEST_NAME


def load_episodes(output_dir: Path) -> tuple[Episode, ...]:
    """Load episodes without pickle, reconstruct models, and validate all schemas."""

    requested = Path(output_dir).absolute()
    try:
        unsafe_root = requested.is_symlink() or (
            requested.exists() and requested.resolve() != requested
        )
    except OSError as exc:
        raise SerializationError(
            f"EpisodeBundle.output_dir: could not inspect bundle root: {exc}"
        ) from exc
    if unsafe_root:
        raise SerializationError(
            "EpisodeBundle.output_dir: symbolic links and junctions are unsupported"
        )
    root = requested.resolve()
    manifest_path = root / MANIFEST_NAME
    _require_regular_unlinked_file(manifest_path, "EpisodeBundle.manifest")
    try:
        raw = cast(
            object,
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError(
            f"EpisodeBundle.manifest: could not read: {exc}"
        ) from exc

    manifest = _mapping(raw, "EpisodeBundle.manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "EpisodeBundle.manifest")
    if _string(manifest, "format", "EpisodeBundle.manifest") != SERIALIZATION_FORMAT:
        raise SerializationError("EpisodeBundle.manifest.format: unsupported format")
    version = _integer(manifest, "serialization_version", "EpisodeBundle.manifest")
    if version != SERIALIZATION_VERSION:
        raise UnsupportedSerializationVersionError(
            "EpisodeBundle.manifest.serialization_version: "
            f"unsupported version {version!r}; supported: {SERIALIZATION_VERSION}"
        )
    _schema(manifest, "EpisodeBundle.manifest")
    encoded_episodes = _list(manifest, "episodes", "EpisodeBundle.manifest")
    expected_count = _integer(manifest, "episode_count", "EpisodeBundle.manifest")
    if expected_count != len(encoded_episodes):
        raise SerializationError(
            "EpisodeBundle.manifest.episode_count: does not match episodes length"
        )
    expected_array_count = _integer(manifest, "array_count", "EpisodeBundle.manifest")
    if expected_array_count < 0:
        raise SerializationError(
            "EpisodeBundle.manifest.array_count: must be non-negative"
        )
    reader = _ArrayReader(root)
    episodes = tuple(
        _decode_episode(value, reader, index)
        for index, value in enumerate(encoded_episodes)
    )
    _validate_array_inventory(reader, expected_array_count)
    validate_episodes(episodes)
    return episodes


def compute_episode_bundle_identifier(output_dir: Path) -> str:
    """Hash a validated M0 bundle independently of its absolute location.

    This is the public form of the file-content identity historically used by
    ``corrupt-data``.  The byte stream is intentionally unchanged: sorted
    POSIX relative paths, each path length and file size as unsigned 64-bit
    big-endian integers, followed by the exact file bytes.  Validation and
    link checks happen before hashing so an identity is never computed through
    an unsafe bundle entry.
    """

    requested = Path(output_dir).absolute()
    # Reuse the complete safe loader rather than assigning an identity to a
    # malformed bundle.  This also rejects an unsafe bundle root and every
    # referenced array before any bytes contribute to the digest.
    load_episodes(requested)
    try:
        entries = tuple(requested.rglob("*"))
        for entry in entries:
            if entry.is_symlink() or entry.resolve() != entry.absolute():
                raise SerializationError(
                    "EpisodeBundle.identity: symbolic links and junctions are "
                    "unsupported"
                )
        files = sorted(
            (path for path in entries if path.is_file()),
            key=lambda path: path.relative_to(requested).as_posix(),
        )
        if not files:
            raise SerializationError("EpisodeBundle.identity: bundle contains no files")
        digest = hashlib.sha256()
        for path in files:
            _require_regular_unlinked_file(path, "EpisodeBundle.identity.file")
            relative = path.relative_to(requested).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, byteorder="big"))
            digest.update(relative)
            size = path.stat().st_size
            digest.update(size.to_bytes(8, byteorder="big"))
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    except SerializationError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise SerializationError(
            f"EpisodeBundle.identity: could not compute safely: {exc}"
        ) from exc
    return f"sha256:{digest.hexdigest()}"


def _encode_value(value: object, writer: _ArrayWriter) -> object:
    if isinstance(value, np.ndarray):
        return {"__array__": writer.write(value)}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode_value(getattr(value, field.name), writer)
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        encoded: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError("mapping.key: expected a string")
            encoded[key] = _encode_value(item, writer)
        return encoded
    if isinstance(value, (tuple, list)):
        return [_encode_value(item, writer) for item in value]
    if isinstance(value, np.generic):
        scalar = value.item()
        if scalar is None or type(scalar) in (str, int, float, bool):
            return scalar
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise SerializationError(f"value: unsupported type {type(value).__name__}")


def _decode_episode(value: object, reader: _ArrayReader, index: int) -> Episode:
    context = f"Episode[index={index}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {
            "episode_id",
            "task_id",
            "source_policy_id",
            "instruction",
            "observations",
            "candidates",
            "split_group_id",
            "schema_version",
        },
        context,
    )
    _schema(item, context)
    episode_id = _string(item, "episode_id", context)
    return Episode(
        episode_id=episode_id,
        task_id=_string(item, "task_id", context),
        source_policy_id=_string(item, "source_policy_id", context),
        instruction=_string(item, "instruction", context),
        observations=_decode_history(
            _field(item, "observations", context), reader, episode_id
        ),
        candidates=tuple(
            _decode_candidate(candidate, reader, episode_id, candidate_index)
            for candidate_index, candidate in enumerate(
                _list(item, "candidates", context)
            )
        ),
        split_group_id=_string(item, "split_group_id", context),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_history(
    value: object, reader: _ArrayReader, episode_id: str
) -> ObservationHistory:
    context = f"ObservationHistory[id={episode_id}]"
    item = _mapping(value, context)
    _require_exact_fields(item, {"history_id", "frames", "schema_version"}, context)
    _schema(item, context)
    history_id = _string(item, "history_id", context)
    return ObservationHistory(
        history_id=history_id,
        frames=tuple(
            _decode_observation(frame, reader, history_id, index)
            for index, frame in enumerate(_list(item, "frames", context))
        ),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_observation(
    value: object, reader: _ArrayReader, history_id: str, index: int
) -> ObservationFrame:
    context = f"ObservationFrame[id={history_id}:{index}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {
            "observation_id",
            "timestamp_s",
            "robot_state",
            "cameras",
            "schema_version",
        },
        context,
    )
    _schema(item, context)
    observation_id = _string(item, "observation_id", context)
    return ObservationFrame(
        observation_id=observation_id,
        timestamp_s=_number(item, "timestamp_s", context),
        robot_state=_array(_field(item, "robot_state", context), reader, context),
        cameras=tuple(
            _decode_camera(camera, reader, observation_id, camera_index)
            for camera_index, camera in enumerate(_list(item, "cameras", context))
        ),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_camera(
    value: object, reader: _ArrayReader, observation_id: str, index: int
) -> CameraFrame:
    context = f"CameraFrame[id={observation_id}:{index}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {
            "camera_id",
            "rgb",
            "depth",
            "intrinsics",
            "extrinsics",
            "schema_version",
        },
        context,
    )
    _schema(item, context)
    return CameraFrame(
        camera_id=_string(item, "camera_id", context),
        rgb=_array(_field(item, "rgb", context), reader, context),
        depth=_optional_array(item, "depth", reader, context),
        intrinsics=_optional_array(item, "intrinsics", reader, context),
        extrinsics=_optional_array(item, "extrinsics", reader, context),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_candidate(
    value: object, reader: _ArrayReader, episode_id: str, index: int
) -> CandidateAction:
    context = f"CandidateAction[id={episode_id}:{index}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {"candidate_id", "action", "outcome", "provenance", "schema_version"},
        context,
    )
    _schema(item, context)
    candidate_id = _string(item, "candidate_id", context)
    return CandidateAction(
        candidate_id=candidate_id,
        action=_decode_action(_field(item, "action", context), reader, candidate_id),
        outcome=_decode_outcome(_field(item, "outcome", context), candidate_id),
        provenance=_decode_provenance(
            _field(item, "provenance", context), candidate_id
        ),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_action(
    value: object, reader: _ArrayReader, candidate_id: str
) -> ActionChunk:
    context = f"ActionChunk[id={candidate_id}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {"actions", "coordinate_frame", "control_period_s", "schema_version"},
        context,
    )
    _schema(item, context)
    return ActionChunk(
        actions=_array(_field(item, "actions", context), reader, context),
        coordinate_frame=_string(item, "coordinate_frame", context),
        control_period_s=_number(item, "control_period_s", context),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_outcome(value: object, candidate_id: str) -> OutcomeLabel:
    context = f"OutcomeLabel[id={candidate_id}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {
            "success",
            "progress",
            "unsafe",
            "label_source",
            "label_strength",
            "simulator_replay_verified",
            "success_probability",
            "unsafe_probability",
            "failure_events",
            "schema_version",
        },
        context,
    )
    _schema(item, context)
    return OutcomeLabel(
        success=_boolean(item, "success", context),
        progress=_number(item, "progress", context),
        unsafe=_boolean(item, "unsafe", context),
        label_source=_enum(item, "label_source", context, LabelSource),
        label_strength=_enum(item, "label_strength", context, LabelStrength),
        simulator_replay_verified=_boolean(item, "simulator_replay_verified", context),
        success_probability=_optional_number(item, "success_probability", context),
        unsafe_probability=_optional_number(item, "unsafe_probability", context),
        failure_events=tuple(
            _decode_failure(failure, candidate_id, index)
            for index, failure in enumerate(_list(item, "failure_events", context))
        ),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_failure(value: object, candidate_id: str, index: int) -> FailureEvent:
    context = f"FailureEvent[id={candidate_id}:{index}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {
            "failure_type",
            "timestamp_s",
            "probability",
            "description",
            "schema_version",
        },
        context,
    )
    _schema(item, context)
    return FailureEvent(
        failure_type=_string(item, "failure_type", context),
        timestamp_s=_optional_number(item, "timestamp_s", context),
        probability=_optional_number(item, "probability", context),
        description=_optional_string(item, "description", context),
        schema_version=_string(item, "schema_version", context),
    )


def _decode_provenance(value: object, candidate_id: str) -> SampleProvenance:
    context = f"SampleProvenance[id={candidate_id}]"
    item = _mapping(value, context)
    _require_exact_fields(
        item,
        {
            "source_episode_id",
            "source_policy_id",
            "source_task_id",
            "transformation_type",
            "transformation_parameters",
            "seed",
            "label_source",
            "label_strength",
            "simulator_replay_verified",
            "split_group_id",
            "schema_version",
        },
        context,
    )
    _schema(item, context)
    parameters_raw = _mapping(
        _field(item, "transformation_parameters", context),
        f"{context}.transformation_parameters",
    )
    parameters: dict[str, JsonScalar] = {}
    for key, parameter in parameters_raw.items():
        if parameter is not None and type(parameter) not in (str, int, float, bool):
            raise SerializationError(
                f"{context}.transformation_parameters.{key}: expected a JSON scalar"
            )
        if isinstance(parameter, float) and not math.isfinite(parameter):
            raise SerializationError(
                f"{context}.transformation_parameters.{key}: must be finite"
            )
        parameters[key] = cast(JsonScalar, parameter)
    return SampleProvenance(
        source_episode_id=_string(item, "source_episode_id", context),
        source_policy_id=_string(item, "source_policy_id", context),
        source_task_id=_string(item, "source_task_id", context),
        transformation_type=_string(item, "transformation_type", context),
        transformation_parameters=parameters,
        seed=_integer(item, "seed", context),
        label_source=_enum(item, "label_source", context, LabelSource),
        label_strength=_enum(item, "label_strength", context, LabelStrength),
        simulator_replay_verified=_boolean(item, "simulator_replay_verified", context),
        split_group_id=_string(item, "split_group_id", context),
        schema_version=_string(item, "schema_version", context),
    )


def _array(value: object, reader: _ArrayReader, context: str) -> NDArray[Any]:
    wrapper = _mapping(value, f"{context}.array")
    _require_exact_fields(wrapper, {"__array__"}, f"{context}.array")
    reference = _mapping(_field(wrapper, "__array__", context), f"{context}.array")
    _require_exact_fields(reference, {"path", "dtype", "shape"}, f"{context}.array")
    relative = _string(reference, "path", f"{context}.array")
    dtype_text = _string(reference, "dtype", f"{context}.array")
    shape_values = _list(reference, "shape", f"{context}.array")
    shape: list[int] = []
    for dimension in shape_values:
        if type(dimension) is not int or dimension < 0:
            raise SerializationError(f"{context}.array.shape: invalid dimension")
        shape.append(dimension)
    try:
        expected_dtype = np.dtype(dtype_text)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"{context}.array.dtype: invalid dtype") from exc
    if expected_dtype.hasobject:
        raise SerializationError(f"{context}.array.dtype: object dtype is unsafe")
    path = _safe_array_path(reader.root, relative, writing=False)
    reader.references.append(relative)
    try:
        with path.open("rb") as stream:
            loaded = np.load(stream, allow_pickle=False)
    except Exception as exc:
        raise SerializationError(
            f"{context}.array: could not load safely: {exc}"
        ) from exc
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise SerializationError(f"{context}.array: expected an NPY array")
    if loaded.dtype != expected_dtype:
        raise SerializationError(
            f"{context}.array.dtype: expected {expected_dtype}, got {loaded.dtype}"
        )
    if loaded.shape != tuple(shape):
        raise SerializationError(
            f"{context}.array.shape: expected {tuple(shape)}, got {loaded.shape}"
        )
    return loaded


def _optional_array(
    item: Mapping[str, object], field: str, reader: _ArrayReader, context: str
) -> NDArray[Any] | None:
    value = _field(item, field, context)
    return None if value is None else _array(value, reader, f"{context}.{field}")


def _safe_array_path(root: Path, relative: str, *, writing: bool) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or "\x00" in relative
        or "\\" in relative
        or pure.is_absolute()
        or pure.parts[0] != "arrays"
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.suffix != ".npy"
    ):
        raise SerializationError(f"ndarray.path: unsafe array path {relative!r}")
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise SerializationError(
            f"ndarray.path: path escapes bundle {relative!r}"
        ) from exc
    if writing:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    elif not candidate.is_file():
        raise SerializationError(f"ndarray.path: missing array file {relative!r}")
    else:
        try:
            link_count = candidate.stat().st_nlink
        except OSError as exc:
            raise SerializationError(
                f"ndarray.path: could not inspect array file {relative!r}"
            ) from exc
        if link_count != 1:
            raise SerializationError(
                f"ndarray.path: hard-linked array file is unsafe {relative!r}"
            )
    return candidate


def _require_empty_destination(destination: Path) -> None:
    try:
        unsafe_destination = (
            destination.is_symlink() or destination.resolve() != destination
        )
    except OSError as exc:
        raise SerializationError(
            f"EpisodeBundle.output_dir: could not inspect destination: {exc}"
        ) from exc
    if unsafe_destination:
        raise SerializationError(
            "EpisodeBundle.output_dir: symbolic links and junctions are unsupported"
        )
    if not destination.exists():
        return
    if not destination.is_dir():
        raise SerializationError("EpisodeBundle.output_dir: must be a directory")
    try:
        next(destination.iterdir())
    except StopIteration:
        return
    except OSError as exc:
        raise SerializationError(
            f"EpisodeBundle.output_dir: could not inspect destination: {exc}"
        ) from exc
    raise SerializationError(
        "EpisodeBundle.output_dir: destination must be absent or empty"
    )


def _require_regular_unlinked_file(path: Path, context: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise SerializationError(f"{context}: missing or unsafe regular file")
        if path.stat().st_nlink != 1:
            raise SerializationError(f"{context}: hard-linked files are unsafe")
    except SerializationError:
        raise
    except OSError as exc:
        raise SerializationError(f"{context}: could not inspect file: {exc}") from exc


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
    references = reader.references
    if len(references) != expected_count:
        raise SerializationError(
            "EpisodeBundle.manifest.array_count: does not match referenced arrays"
        )
    if len(set(references)) != len(references):
        raise SerializationError(
            "EpisodeBundle.manifest.arrays: duplicate array references are unsupported"
        )

    arrays_dir = reader.root / "arrays"
    if arrays_dir.is_symlink() or not arrays_dir.is_dir():
        raise SerializationError(
            "EpisodeBundle.arrays: missing or unsafe arrays directory"
        )
    actual_files: set[str] = set()
    for path in arrays_dir.rglob("*"):
        if path.is_symlink():
            raise SerializationError("EpisodeBundle.arrays: symbolic links are unsafe")
        if path.is_file():
            actual_files.add(path.relative_to(reader.root).as_posix())
    if actual_files != set(references):
        raise SerializationError(
            "EpisodeBundle.arrays: files do not exactly match manifest references"
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
        raise SerializationError(f"{context}: invalid fields ({'; '.join(details)})")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SerializationError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _field(item: Mapping[str, object], field: str, context: str) -> object:
    if field not in item:
        raise SerializationError(f"{context}.{field}: missing required field")
    return item[field]


def _string(item: Mapping[str, object], field: str, context: str) -> str:
    value = _field(item, field, context)
    if not isinstance(value, str):
        raise SerializationError(f"{context}.{field}: expected a string")
    return value


def _optional_string(
    item: Mapping[str, object], field: str, context: str
) -> str | None:
    value = _field(item, field, context)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SerializationError(f"{context}.{field}: expected a string or null")
    return value


def _integer(item: Mapping[str, object], field: str, context: str) -> int:
    value = _field(item, field, context)
    if type(value) is not int:
        raise SerializationError(f"{context}.{field}: expected an integer")
    return value


def _number(item: Mapping[str, object], field: str, context: str) -> float:
    value = _field(item, field, context)
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise SerializationError(f"{context}.{field}: expected a finite number")
    return float(cast(int | float, value))


def _optional_number(
    item: Mapping[str, object], field: str, context: str
) -> float | None:
    value = _field(item, field, context)
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise SerializationError(f"{context}.{field}: expected a finite number or null")
    return float(cast(int | float, value))


def _boolean(item: Mapping[str, object], field: str, context: str) -> bool:
    value = _field(item, field, context)
    if type(value) is not bool:
        raise SerializationError(f"{context}.{field}: expected a boolean")
    return value


def _list(item: Mapping[str, object], field: str, context: str) -> list[object]:
    value = _field(item, field, context)
    if not isinstance(value, list):
        raise SerializationError(f"{context}.{field}: expected an array")
    return cast(list[object], value)


def _enum(
    item: Mapping[str, object],
    field: str,
    context: str,
    enum_type: type[EnumType],
) -> EnumType:
    value = _string(item, field, context)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise SerializationError(
            f"{context}.{field}: unsupported enum value {value!r}"
        ) from exc


def _schema(item: Mapping[str, object], context: str) -> None:
    version = _string(item, "schema_version", context)
    if version != CURRENT_SCHEMA_VERSION:
        raise UnsupportedSerializationVersionError(
            f"{context}.schema_version: unsupported schema version {version!r}; "
            f"supported: {CURRENT_SCHEMA_VERSION}"
        )


def _reject_json_constant(value: str) -> NoReturn:
    raise SerializationError(
        f"EpisodeBundle.manifest: non-finite JSON constant {value!r} is unsupported"
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SerializationError(f"JSON object: duplicate field {key!r}")
        result[key] = value
    return result
