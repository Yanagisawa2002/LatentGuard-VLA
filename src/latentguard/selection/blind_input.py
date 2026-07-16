"""Strict outcome-free Stage-A candidate inputs and safe persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.models import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    CANDIDATE_COUNT,
    SELECTION_SCHEMA_VERSION,
    STATE_DIMENSION,
    CandidatePoolV1,
    array_content_digest,
)

BLIND_CANDIDATE_POOL_FORMAT = "latentguard-m3c-stage-a-candidate-input"
BLIND_CANDIDATE_POOL_SERIALIZATION_VERSION = 1
BLIND_CANDIDATE_POOL_MANIFEST_NAME = "manifest.json"
MAX_BLIND_CANDIDATE_POOL_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_BLIND_CANDIDATE_ARRAY_BYTES = 128 * 1024

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class BlindCandidateInputError(ValueError):
    """Raised when a Stage-A input exposes or drifts non-allowlisted content."""


def _fail(context: str, reason: str) -> NoReturn:
    raise BlindCandidateInputError(f"{context}: {reason}")


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
        _fail(context, "expected lowercase sha256 content digest")
    return value


def _content_digest(value: object, *, context: str) -> str:
    try:
        encoded = canonical_json_bytes(value, context=context)
    except ReplayValidationError as exc:
        raise BlindCandidateInputError(str(exc)) from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _freeze_array(
    value: object,
    *,
    shape: tuple[int, ...],
    floating: bool,
    context: str,
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        _fail(context, f"expected ndarray shape {shape}")
    if floating:
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.floating):
            _fail(context, "expected non-object floating dtype")
        if not bool(np.all(np.isfinite(value))):
            _fail(context, "all values must be finite")
    elif value.dtype != np.dtype(np.bool_):
        _fail(context, "expected bool dtype")
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


@dataclass(frozen=True, slots=True, eq=False)
class BlindCandidateGroupV1:
    """One opaque group containing only deployable Stage-A model inputs."""

    group_id: str
    candidate_ids: tuple[str, ...]
    state_vector: NDArray[Any]
    action_chunks: NDArray[Any]
    action_masks: NDArray[Any]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach arrays and enforce the exact fixed PickCube input contract."""

        _text(self.group_id, "BlindCandidateGroupV1.group_id")
        candidate_ids = tuple(self.candidate_ids)
        if len(candidate_ids) != CANDIDATE_COUNT or len(set(candidate_ids)) != (
            CANDIDATE_COUNT
        ):
            _fail(
                "BlindCandidateGroupV1.candidate_ids",
                "expected exactly eight unique opaque IDs",
            )
        for value in candidate_ids:
            _text(value, "BlindCandidateGroupV1.candidate_ids")
        state = _freeze_array(
            self.state_vector,
            shape=(STATE_DIMENSION,),
            floating=True,
            context="BlindCandidateGroupV1.state_vector",
        )
        if state.dtype != np.dtype("<f4"):
            _fail(
                "BlindCandidateGroupV1.state_vector",
                "expected exact little-endian float32 verifier state",
            )
        actions = _freeze_array(
            self.action_chunks,
            shape=(CANDIDATE_COUNT, ACTION_HORIZON, ACTION_DIMENSION),
            floating=True,
            context="BlindCandidateGroupV1.action_chunks",
        )
        masks = _freeze_array(
            self.action_masks,
            shape=(CANDIDATE_COUNT, ACTION_HORIZON),
            floating=False,
            context="BlindCandidateGroupV1.action_masks",
        )
        if not bool(np.all(masks)):
            _fail(
                "BlindCandidateGroupV1.action_masks",
                "all fixed-horizon candidates must be complete",
            )
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("BlindCandidateGroupV1.schema_version", "unsupported version")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "state_vector", state)
        object.__setattr__(self, "action_chunks", actions)
        object.__setattr__(self, "action_masks", masks)

    def identity_mapping(self) -> dict[str, object]:
        """Return the opaque IDs and exact deployable-array identities."""

        return {
            "action_chunks_digest": array_content_digest(self.action_chunks),
            "action_masks_digest": array_content_digest(self.action_masks),
            "candidate_ids": list(self.candidate_ids),
            "group_id": self.group_id,
            "schema_version": self.schema_version,
            "state_vector_digest": array_content_digest(self.state_vector),
        }

    @property
    def content_digest(self) -> str:
        """Return the exact, path-independent blinded-group identity."""

        return _content_digest(self.identity_mapping(), context="BlindCandidateGroupV1")


@dataclass(frozen=True, slots=True, eq=False)
class BlindCandidatePoolV1:
    """Complete outcome-free Stage-A pool with only required digest bindings."""

    source_set_digest: str
    candidate_pool_configuration_digest: str
    full_candidate_pool_digest: str
    action_contract_digest: str
    groups: tuple[BlindCandidateGroupV1, ...]
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate exact bindings and reject duplicate opaque identities."""

        for name in (
            "source_set_digest",
            "candidate_pool_configuration_digest",
            "full_candidate_pool_digest",
            "action_contract_digest",
        ):
            _digest(getattr(self, name), f"BlindCandidatePoolV1.{name}")
        groups = tuple(self.groups)
        if not groups or any(
            not isinstance(item, BlindCandidateGroupV1) for item in groups
        ):
            _fail(
                "BlindCandidatePoolV1.groups",
                "expected a non-empty blinded-group inventory",
            )
        group_ids = tuple(item.group_id for item in groups)
        candidate_ids = tuple(
            candidate_id for item in groups for candidate_id in item.candidate_ids
        )
        if len(group_ids) != len(set(group_ids)):
            _fail("BlindCandidatePoolV1.groups", "duplicate group ID")
        if len(candidate_ids) != len(set(candidate_ids)):
            _fail("BlindCandidatePoolV1.groups", "duplicate candidate ID")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("BlindCandidatePoolV1.schema_version", "unsupported version")
        object.__setattr__(self, "groups", groups)

    def identity_mapping(self) -> dict[str, object]:
        """Return the exact pool bindings without provenance or outcomes."""

        return {
            "action_contract_digest": self.action_contract_digest,
            "candidate_pool_configuration_digest": (
                self.candidate_pool_configuration_digest
            ),
            "full_candidate_pool_digest": self.full_candidate_pool_digest,
            "group_content_digests": [item.content_digest for item in self.groups],
            "schema_version": self.schema_version,
            "source_set_digest": self.source_set_digest,
        }

    @property
    def content_digest(self) -> str:
        """Return the exact blinded-pool identity."""

        return _content_digest(self.identity_mapping(), context="BlindCandidatePoolV1")

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Return ordered opaque group identities."""

        return tuple(item.group_id for item in self.groups)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Return ordered opaque candidate identities."""

        return tuple(
            candidate_id for item in self.groups for candidate_id in item.candidate_ids
        )

    @classmethod
    def from_candidate_pool(
        cls, candidate_pool: CandidatePoolV1
    ) -> BlindCandidatePoolV1:
        """Build the exact Stage-A projection of a complete candidate pool."""

        return build_blind_candidate_pool(candidate_pool)

    def validate_against_full_pool(self, candidate_pool: CandidatePoolV1) -> None:
        """Require this blinded input to match one exact complete pool."""

        validate_blind_candidate_pool_against_full_pool(self, candidate_pool)


def build_blind_candidate_pool(candidate_pool: CandidatePoolV1) -> BlindCandidatePoolV1:
    """Project a full pool into the only data structure admitted to Stage A."""

    if not isinstance(candidate_pool, CandidatePoolV1):
        _fail("candidate_pool", "expected CandidatePoolV1")
    groups = tuple(
        BlindCandidateGroupV1(
            group_id=group.group_id,
            candidate_ids=group.proposal_ids,
            state_vector=group.state_vector,
            action_chunks=np.stack(
                [candidate.action_chunk for candidate in group.candidates], axis=0
            ),
            action_masks=np.stack(
                [candidate.action_mask for candidate in group.candidates], axis=0
            ),
        )
        for group in candidate_pool.groups
    )
    blinded = BlindCandidatePoolV1(
        source_set_digest=candidate_pool.source_set_digest,
        candidate_pool_configuration_digest=(
            candidate_pool.candidate_pool_configuration_digest
        ),
        full_candidate_pool_digest=candidate_pool.content_digest,
        action_contract_digest=candidate_pool.action_contract_digest,
        groups=groups,
    )
    validate_blind_candidate_pool_against_full_pool(blinded, candidate_pool)
    return blinded


def validate_blind_candidate_pool_against_full_pool(
    blinded: BlindCandidatePoolV1,
    candidate_pool: CandidatePoolV1,
) -> None:
    """Reject any binding, group, candidate, state, action, mask, or order drift."""

    if not isinstance(blinded, BlindCandidatePoolV1):
        _fail("blinded", "expected BlindCandidatePoolV1")
    if not isinstance(candidate_pool, CandidatePoolV1):
        _fail("candidate_pool", "expected CandidatePoolV1")
    if (
        blinded.source_set_digest != candidate_pool.source_set_digest
        or blinded.candidate_pool_configuration_digest
        != candidate_pool.candidate_pool_configuration_digest
        or blinded.full_candidate_pool_digest != candidate_pool.content_digest
        or blinded.action_contract_digest != candidate_pool.action_contract_digest
    ):
        _fail("blinded pool", "full-pool digest binding differs")
    if len(blinded.groups) != len(candidate_pool.groups):
        _fail("blinded pool", "group count differs from full pool")
    for index, (blind_group, full_group) in enumerate(
        zip(blinded.groups, candidate_pool.groups, strict=True)
    ):
        expected = BlindCandidateGroupV1(
            group_id=full_group.group_id,
            candidate_ids=full_group.proposal_ids,
            state_vector=full_group.state_vector,
            action_chunks=np.stack(
                [candidate.action_chunk for candidate in full_group.candidates], axis=0
            ),
            action_masks=np.stack(
                [candidate.action_mask for candidate in full_group.candidates], axis=0
            ),
        )
        if blind_group.content_digest != expected.content_digest:
            _fail(f"blinded pool group {index}", "content differs from full pool")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _array_reference(
    root: Path, relative: str, value: NDArray[Any]
) -> dict[str, object]:
    destination = root / Path(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "content_digest": array_content_digest(value),
        "dtype": value.dtype.str,
        "file_digest": _sha256_file(destination),
        "path": relative,
        "shape": list(value.shape),
    }


def _manifest(pool: BlindCandidatePoolV1, root: Path) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for index, group in enumerate(pool.groups):
        prefix = f"arrays/group-{index:06d}"
        groups.append(
            {
                "actions": _array_reference(
                    root, f"{prefix}-actions.npy", group.action_chunks
                ),
                "candidate_ids": list(group.candidate_ids),
                "group_content_digest": group.content_digest,
                "group_id": group.group_id,
                "masks": _array_reference(
                    root, f"{prefix}-masks.npy", group.action_masks
                ),
                "schema_version": group.schema_version,
                "state": _array_reference(
                    root, f"{prefix}-state.npy", group.state_vector
                ),
            }
        )
    return {
        "action_contract_digest": pool.action_contract_digest,
        "blind_candidate_pool_content_digest": pool.content_digest,
        "candidate_pool_configuration_digest": (
            pool.candidate_pool_configuration_digest
        ),
        "format": BLIND_CANDIDATE_POOL_FORMAT,
        "full_candidate_pool_digest": pool.full_candidate_pool_digest,
        "groups": groups,
        "schema_version": pool.schema_version,
        "serialization_version": BLIND_CANDIDATE_POOL_SERIALIZATION_VERSION,
        "source_set_digest": pool.source_set_digest,
    }


def _manifest_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise BlindCandidateInputError(
            "blinded-pool manifest is not JSON-safe"
        ) from exc


def save_blind_candidate_pool(pool: BlindCandidatePoolV1, output_dir: Path) -> Path:
    """Atomically publish a new strict Stage-A input bundle."""

    if not isinstance(pool, BlindCandidatePoolV1):
        _fail("blinded pool", "expected BlindCandidatePoolV1")
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        existing = load_blind_candidate_pool(destination)
        if existing.content_digest != pool.content_digest:
            _fail("output directory", "already contains a different blinded pool")
        return destination
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        manifest = _manifest(pool, staging)
        path = staging / BLIND_CANDIDATE_POOL_MANIFEST_NAME
        with path.open("xb") as stream:
            stream.write(_manifest_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            staging.rename(destination)
        except FileExistsError:
            existing = load_blind_candidate_pool(destination)
            if existing.content_digest != pool.content_digest:
                _fail(
                    "output directory",
                    "concurrent publication contains different content",
                )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    reloaded = load_blind_candidate_pool(destination)
    if reloaded.content_digest != pool.content_digest:
        _fail("blinded pool", "published content failed exact reload")
    return destination


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("blinded-pool JSON", f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail("blinded-pool JSON", f"unsupported non-finite constant {value!r}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected an object")
    return cast(Mapping[str, object], value)


def _exact(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        _fail(context, "unexpected or missing fields")


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        _fail(context, "expected a list")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(context, f"expected integer >= {minimum}")
    return value


def _safe_array_path(root: Path, value: object, expected: str, context: str) -> Path:
    relative = _text(value, f"{context}.path")
    if relative != expected:
        _fail(context, "array path differs from its canonical location")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != ("arrays",):
        _fail(context, "array path must stay under arrays/")
    path = root / Path(*pure.parts)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            _fail(context, "expected regular single-linked array file")
        if not 0 < path.stat().st_size <= MAX_BLIND_CANDIDATE_ARRAY_BYTES:
            _fail(context, "array file size is outside safe bounds")
    except OSError as exc:
        raise BlindCandidateInputError(f"{context}: unsafe array file") from exc
    return path


def _read_array(
    root: Path,
    value: object,
    *,
    expected_path: str,
    expected_shape: tuple[int, ...],
    context: str,
    referenced: set[str],
) -> NDArray[Any]:
    item = _mapping(value, context)
    _exact(
        item,
        {"content_digest", "dtype", "file_digest", "path", "shape"},
        context,
    )
    path = _safe_array_path(root, item["path"], expected_path, context)
    relative = path.relative_to(root).as_posix()
    if relative in referenced:
        _fail(context, "array path is referenced more than once")
    referenced.add(relative)
    if _sha256_file(path) != _digest(item["file_digest"], f"{context}.file_digest"):
        _fail(context, "array file digest mismatch")
    try:
        with path.open("rb") as stream:
            loaded = np.load(stream, allow_pickle=False)
            if stream.read(1):
                _fail(context, "array file contains trailing bytes")
    except BlindCandidateInputError:
        raise
    except (OSError, ValueError) as exc:
        raise BlindCandidateInputError(f"{context}: could not load array") from exc
    if not isinstance(loaded, np.ndarray):
        _fail(context, "expected one ndarray")
    shape = tuple(
        _integer(element, f"{context}.shape")
        for element in _list(item["shape"], f"{context}.shape")
    )
    if (
        loaded.shape != expected_shape
        or shape != expected_shape
        or loaded.dtype.str != _text(item["dtype"], f"{context}.dtype")
    ):
        _fail(context, "array dtype or shape mismatch")
    if array_content_digest(loaded) != _digest(
        item["content_digest"], f"{context}.content_digest"
    ):
        _fail(context, "array content digest mismatch")
    return loaded


def _decode_group(
    root: Path,
    value: object,
    index: int,
    referenced: set[str],
) -> BlindCandidateGroupV1:
    context = f"BlindCandidatePool.groups[{index}]"
    item = _mapping(value, context)
    _exact(
        item,
        {
            "actions",
            "candidate_ids",
            "group_content_digest",
            "group_id",
            "masks",
            "schema_version",
            "state",
        },
        context,
    )
    prefix = f"arrays/group-{index:06d}"
    candidate_ids = tuple(
        _text(element, f"{context}.candidate_ids")
        for element in _list(item["candidate_ids"], f"{context}.candidate_ids")
    )
    group = BlindCandidateGroupV1(
        group_id=_text(item["group_id"], f"{context}.group_id"),
        candidate_ids=candidate_ids,
        state_vector=_read_array(
            root,
            item["state"],
            expected_path=f"{prefix}-state.npy",
            expected_shape=(STATE_DIMENSION,),
            context=f"{context}.state",
            referenced=referenced,
        ),
        action_chunks=_read_array(
            root,
            item["actions"],
            expected_path=f"{prefix}-actions.npy",
            expected_shape=(CANDIDATE_COUNT, ACTION_HORIZON, ACTION_DIMENSION),
            context=f"{context}.actions",
            referenced=referenced,
        ),
        action_masks=_read_array(
            root,
            item["masks"],
            expected_path=f"{prefix}-masks.npy",
            expected_shape=(CANDIDATE_COUNT, ACTION_HORIZON),
            context=f"{context}.masks",
            referenced=referenced,
        ),
        schema_version=_text(item["schema_version"], f"{context}.schema_version"),
    )
    if group.content_digest != _digest(
        item["group_content_digest"], f"{context}.group_content_digest"
    ):
        _fail(context, "group content digest mismatch")
    return group


def _validate_inventory(root: Path, referenced: set[str]) -> None:
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail("blinded-pool bundle", "symbolic links are unsupported")
        if path.is_file():
            if path.stat().st_nlink != 1:
                _fail("blinded-pool bundle", "hard-linked files are unsupported")
            observed.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            _fail("blinded-pool bundle", "unsupported filesystem entry")
    expected = {BLIND_CANDIDATE_POOL_MANIFEST_NAME, *referenced}
    if observed != expected:
        _fail("blinded-pool bundle", "file inventory differs from manifest")


def load_blind_candidate_pool(
    output_dir: Path,
    *,
    expected_content_digest: str | None = None,
) -> BlindCandidatePoolV1:
    """Load a strict Stage-A bundle and revalidate every file and digest."""

    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        _fail("blinded-pool root", "expected regular non-symlink directory")
    manifest_path = root / BLIND_CANDIDATE_POOL_MANIFEST_NAME
    try:
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.stat().st_nlink != 1
        ):
            _fail("blinded-pool manifest", "expected regular single-linked file")
        if (
            not 0
            < manifest_path.stat().st_size
            <= (MAX_BLIND_CANDIDATE_POOL_MANIFEST_BYTES)
        ):
            _fail("blinded-pool manifest", "file size is outside safe bounds")
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except BlindCandidateInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlindCandidateInputError(
            f"blinded-pool manifest: could not read safely: {exc}"
        ) from exc
    manifest = _mapping(raw, "BlindCandidatePool.manifest")
    _exact(
        manifest,
        {
            "action_contract_digest",
            "blind_candidate_pool_content_digest",
            "candidate_pool_configuration_digest",
            "format",
            "full_candidate_pool_digest",
            "groups",
            "schema_version",
            "serialization_version",
            "source_set_digest",
        },
        "BlindCandidatePool.manifest",
    )
    if manifest["format"] != BLIND_CANDIDATE_POOL_FORMAT:
        _fail("BlindCandidatePool.manifest.format", "unsupported format")
    if manifest["serialization_version"] != (
        BLIND_CANDIDATE_POOL_SERIALIZATION_VERSION
    ):
        _fail(
            "BlindCandidatePool.manifest.serialization_version", "unsupported version"
        )
    referenced: set[str] = set()
    groups = tuple(
        _decode_group(root, item, index, referenced)
        for index, item in enumerate(
            _list(manifest["groups"], "BlindCandidatePool.groups")
        )
    )
    _validate_inventory(root, referenced)
    pool = BlindCandidatePoolV1(
        source_set_digest=_digest(
            manifest["source_set_digest"], "BlindCandidatePool.source_set_digest"
        ),
        candidate_pool_configuration_digest=_digest(
            manifest["candidate_pool_configuration_digest"],
            "BlindCandidatePool.candidate_pool_configuration_digest",
        ),
        full_candidate_pool_digest=_digest(
            manifest["full_candidate_pool_digest"],
            "BlindCandidatePool.full_candidate_pool_digest",
        ),
        action_contract_digest=_digest(
            manifest["action_contract_digest"],
            "BlindCandidatePool.action_contract_digest",
        ),
        groups=groups,
        schema_version=_text(
            manifest["schema_version"], "BlindCandidatePool.schema_version"
        ),
    )
    stored = _digest(
        manifest["blind_candidate_pool_content_digest"],
        "BlindCandidatePool.content_digest",
    )
    if pool.content_digest != stored:
        _fail("BlindCandidatePool.content_digest", "manifest content changed")
    if expected_content_digest is not None:
        _digest(expected_content_digest, "expected_content_digest")
        if pool.content_digest != expected_content_digest:
            _fail(
                "BlindCandidatePool.content_digest", "does not match expected identity"
            )
    return pool


__all__ = [
    "BLIND_CANDIDATE_POOL_FORMAT",
    "BLIND_CANDIDATE_POOL_MANIFEST_NAME",
    "BLIND_CANDIDATE_POOL_SERIALIZATION_VERSION",
    "BlindCandidateGroupV1",
    "BlindCandidateInputError",
    "BlindCandidatePoolV1",
    "build_blind_candidate_pool",
    "load_blind_candidate_pool",
    "save_blind_candidate_pool",
    "validate_blind_candidate_pool_against_full_pool",
]
