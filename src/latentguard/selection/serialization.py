"""Safe manifest/NPY serialization for unlabeled M3C candidate pools."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.selection.models import (
    CandidateDistribution,
    CandidateGroupV1,
    CandidatePoolV1,
    CandidateRefV1,
    SourceTrajectoryIdentityV1,
    array_content_digest,
)

CANDIDATE_POOL_FORMAT = "latentguard-m3c-blind-candidate-pool"
CANDIDATE_POOL_SERIALIZATION_VERSION = 1
CANDIDATE_POOL_MANIFEST_NAME = "manifest.json"


class CandidatePoolSerializationError(ValueError):
    """Raised when a candidate-pool bundle is unsafe or content-drifted."""


def _fail(context: str, reason: str) -> NoReturn:
    raise CandidatePoolSerializationError(f"{context}: {reason}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _reject_constant(value: str) -> NoReturn:
    _fail("candidate-pool JSON", f"unsupported non-finite constant {value!r}")


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("candidate-pool JSON", f"duplicate field {key!r}")
        result[key] = value
    return result


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected an object")
    return cast(Mapping[str, object], value)


def _exact(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        _fail(context, "unexpected or missing fields")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(context, f"expected integer >= {minimum}")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        _fail(context, "expected a list")
    return value


def _array_reference(
    path: Path, relative: str, value: NDArray[Any]
) -> dict[str, object]:
    destination = path / Path(*PurePosixPath(relative).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "array_content_digest": array_content_digest(value),
        "dtype": value.dtype.str,
        "file_sha256": _sha256_file(destination),
        "path": relative,
        "shape": list(value.shape),
    }


def _candidate_mapping(
    root: Path,
    group_index: int,
    candidate: CandidateRefV1,
) -> dict[str, object]:
    prefix = (
        f"arrays/group-{group_index:06d}/candidate-{candidate.configuration_ordinal}"
    )
    return {
        "action_chunk": _array_reference(
            root, f"{prefix}-action.npy", candidate.action_chunk
        ),
        "action_mask": _array_reference(
            root, f"{prefix}-mask.npy", candidate.action_mask
        ),
        "candidate_content_digest": candidate.content_digest,
        "configuration_ordinal": candidate.configuration_ordinal,
        "corruption_type": candidate.corruption_type,
        "distribution": candidate.distribution.value,
        "proposal_id": candidate.proposal_id,
        "schema_version": candidate.schema_version,
        "seed": candidate.seed,
        "severity_id": candidate.severity_id,
    }


def _group_mapping(
    root: Path, index: int, group: CandidateGroupV1
) -> dict[str, object]:
    prefix = f"arrays/group-{index:06d}"
    return {
        "anchor_id": group.anchor_id,
        "candidates": [
            _candidate_mapping(root, index, candidate) for candidate in group.candidates
        ],
        "continuation_actions": _array_reference(
            root,
            f"{prefix}/continuation.npy",
            group.continuation_actions,
        ),
        "continuation_identity": group.continuation_identity,
        "group_id": group.group_id,
        "schema_version": group.schema_version,
        "source_action_prefix_digest": group.source_action_prefix_digest,
        "state_content_digest": group.state_content_digest,
        "state_vector": _array_reference(
            root, f"{prefix}/state.npy", group.state_vector
        ),
        "trajectory": group.trajectory.as_mapping(),
        "verifier_state_content_digest": group.verifier_state_content_digest,
    }


def _manifest(pool: CandidatePoolV1, root: Path) -> dict[str, object]:
    return {
        "action_contract_digest": pool.action_contract_digest,
        "candidate_pool_configuration_digest": pool.candidate_pool_configuration_digest,
        "candidate_pool_content_digest": pool.content_digest,
        "exclusion_inventory_digest": pool.exclusion_inventory_digest,
        "format": CANDIDATE_POOL_FORMAT,
        "groups": [
            _group_mapping(root, index, group)
            for index, group in enumerate(pool.groups)
        ],
        "schema_version": pool.schema_version,
        "serialization_version": CANDIDATE_POOL_SERIALIZATION_VERSION,
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
        raise CandidatePoolSerializationError(
            "candidate-pool manifest is not JSON-safe"
        ) from exc


def save_candidate_pool(pool: CandidatePoolV1, output_dir: Path) -> Path:
    """Publish a candidate pool atomically; repeated identical saves are idempotent."""

    if not isinstance(pool, CandidatePoolV1):
        _fail("candidate pool", "expected CandidatePoolV1")
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_candidate_pool(destination)
        if existing.content_digest != pool.content_digest:
            _fail("output directory", "already contains a different candidate pool")
        return destination
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        manifest = _manifest(pool, staging)
        manifest_path = staging / CANDIDATE_POOL_MANIFEST_NAME
        with manifest_path.open("xb") as stream:
            stream.write(_manifest_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            staging.rename(destination)
        except FileExistsError:
            existing = load_candidate_pool(destination)
            if existing.content_digest != pool.content_digest:
                _fail(
                    "output directory",
                    "concurrent publication contains different content",
                )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    reloaded = load_candidate_pool(destination)
    if reloaded.content_digest != pool.content_digest:
        _fail("candidate pool", "published content failed exact reload")
    return destination


def _safe_array_path(root: Path, value: object, context: str) -> Path:
    relative = _text(value, f"{context}.path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != ("arrays",):
        _fail(context, "array path must stay under arrays/")
    path = root / Path(*pure.parts)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            _fail(context, "expected regular single-linked array file")
    except OSError as exc:
        raise CandidatePoolSerializationError(f"{context}: unsafe array file") from exc
    return path


def _read_array(
    root: Path,
    value: object,
    context: str,
    referenced: set[str],
) -> NDArray[Any]:
    item = _mapping(value, context)
    _exact(
        item,
        {"array_content_digest", "dtype", "file_sha256", "path", "shape"},
        context,
    )
    path = _safe_array_path(root, item["path"], context)
    relative = path.relative_to(root).as_posix()
    if relative in referenced:
        _fail(context, "array path is referenced more than once")
    referenced.add(relative)
    if _sha256_file(path) != _text(item["file_sha256"], f"{context}.file_sha256"):
        _fail(context, "array file digest mismatch")
    try:
        with path.open("rb") as stream:
            loaded = np.load(stream, allow_pickle=False)
            if stream.read(1):
                _fail(context, "array file contains trailing bytes")
    except CandidatePoolSerializationError:
        raise
    except (OSError, ValueError) as exc:
        raise CandidatePoolSerializationError(
            f"{context}: could not load array"
        ) from exc
    if not isinstance(loaded, np.ndarray):
        _fail(context, "expected one ndarray")
    dtype = _text(item["dtype"], f"{context}.dtype")
    shape = tuple(
        _integer(element, f"{context}.shape")
        for element in _list(item["shape"], f"{context}.shape")
    )
    if loaded.dtype.str != dtype or loaded.shape != shape:
        _fail(context, "array dtype or shape mismatch")
    if array_content_digest(loaded) != _text(
        item["array_content_digest"], f"{context}.array_content_digest"
    ):
        _fail(context, "array content digest mismatch")
    return loaded


def _decode_trajectory(value: object, context: str) -> SourceTrajectoryIdentityV1:
    item = _mapping(value, context)
    _exact(
        item,
        {
            "complete_state_digests",
            "schema_version",
            "source_seed",
            "source_trajectory_id",
            "split_group_id",
        },
        context,
    )
    return SourceTrajectoryIdentityV1(
        source_trajectory_id=_text(
            item["source_trajectory_id"], f"{context}.source_trajectory_id"
        ),
        source_seed=_integer(item["source_seed"], f"{context}.source_seed"),
        split_group_id=_text(item["split_group_id"], f"{context}.split_group_id"),
        complete_state_digests=tuple(
            _text(element, f"{context}.complete_state_digests")
            for element in _list(
                item["complete_state_digests"], f"{context}.complete_state_digests"
            )
        ),
        schema_version=_text(item["schema_version"], f"{context}.schema_version"),
    )


def _decode_candidate(
    root: Path,
    value: object,
    context: str,
    referenced: set[str],
) -> CandidateRefV1:
    item = _mapping(value, context)
    _exact(
        item,
        {
            "action_chunk",
            "action_mask",
            "candidate_content_digest",
            "configuration_ordinal",
            "corruption_type",
            "distribution",
            "proposal_id",
            "schema_version",
            "seed",
            "severity_id",
        },
        context,
    )
    candidate = CandidateRefV1(
        proposal_id=_text(item["proposal_id"], f"{context}.proposal_id"),
        configuration_ordinal=_integer(
            item["configuration_ordinal"], f"{context}.configuration_ordinal"
        ),
        distribution=cast(CandidateDistribution, item["distribution"]),
        corruption_type=_text(item["corruption_type"], f"{context}.corruption_type"),
        severity_id=_text(item["severity_id"], f"{context}.severity_id"),
        seed=_integer(item["seed"], f"{context}.seed"),
        action_chunk=_read_array(
            root, item["action_chunk"], f"{context}.action_chunk", referenced
        ),
        action_mask=_read_array(
            root, item["action_mask"], f"{context}.action_mask", referenced
        ),
        schema_version=_text(item["schema_version"], f"{context}.schema_version"),
    )
    if candidate.content_digest != _text(
        item["candidate_content_digest"], f"{context}.candidate_content_digest"
    ):
        _fail(context, "candidate content digest mismatch")
    return candidate


def _decode_group(
    root: Path,
    value: object,
    index: int,
    referenced: set[str],
) -> CandidateGroupV1:
    context = f"CandidatePool.groups[{index}]"
    item = _mapping(value, context)
    _exact(
        item,
        {
            "anchor_id",
            "candidates",
            "continuation_actions",
            "continuation_identity",
            "group_id",
            "schema_version",
            "source_action_prefix_digest",
            "state_content_digest",
            "state_vector",
            "trajectory",
            "verifier_state_content_digest",
        },
        context,
    )
    candidates = tuple(
        _decode_candidate(root, raw, f"{context}.candidates[{ordinal}]", referenced)
        for ordinal, raw in enumerate(
            _list(item["candidates"], f"{context}.candidates")
        )
    )
    group = CandidateGroupV1(
        anchor_id=_text(item["anchor_id"], f"{context}.anchor_id"),
        trajectory=_decode_trajectory(item["trajectory"], f"{context}.trajectory"),
        state_content_digest=_text(
            item["state_content_digest"], f"{context}.state_content_digest"
        ),
        verifier_state_content_digest=_text(
            item["verifier_state_content_digest"],
            f"{context}.verifier_state_content_digest",
        ),
        continuation_identity=_text(
            item["continuation_identity"], f"{context}.continuation_identity"
        ),
        source_action_prefix_digest=_text(
            item["source_action_prefix_digest"],
            f"{context}.source_action_prefix_digest",
        ),
        state_vector=_read_array(
            root, item["state_vector"], f"{context}.state_vector", referenced
        ),
        continuation_actions=_read_array(
            root,
            item["continuation_actions"],
            f"{context}.continuation_actions",
            referenced,
        ),
        candidates=candidates,
        schema_version=_text(item["schema_version"], f"{context}.schema_version"),
    )
    if group.group_id != _text(item["group_id"], f"{context}.group_id"):
        _fail(context, "group content digest mismatch")
    return group


def _validate_inventory(root: Path, referenced: set[str]) -> None:
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail("candidate-pool bundle", "symbolic links are unsupported")
        if path.is_file():
            if path.stat().st_nlink != 1:
                _fail("candidate-pool bundle", "hard-linked files are unsupported")
            observed.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            _fail("candidate-pool bundle", "unsupported filesystem entry")
    expected = {CANDIDATE_POOL_MANIFEST_NAME, *referenced}
    if observed != expected:
        _fail("candidate-pool bundle", "file inventory differs from manifest")


def load_candidate_pool(
    output_dir: Path,
    *,
    expected_content_digest: str | None = None,
) -> CandidatePoolV1:
    """Load and fully revalidate a safe candidate-pool bundle."""

    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        _fail("candidate-pool root", "expected regular non-symlink directory")
    manifest_path = root / CANDIDATE_POOL_MANIFEST_NAME
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_nlink != 1
    ):
        _fail("candidate-pool manifest", "expected regular single-linked file")
    try:
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except CandidatePoolSerializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidatePoolSerializationError(
            f"candidate-pool manifest: could not read safely: {exc}"
        ) from exc
    manifest = _mapping(raw, "CandidatePool.manifest")
    _exact(
        manifest,
        {
            "action_contract_digest",
            "candidate_pool_configuration_digest",
            "candidate_pool_content_digest",
            "exclusion_inventory_digest",
            "format",
            "groups",
            "schema_version",
            "serialization_version",
            "source_set_digest",
        },
        "CandidatePool.manifest",
    )
    if manifest["format"] != CANDIDATE_POOL_FORMAT:
        _fail("CandidatePool.manifest.format", "unsupported format")
    if manifest["serialization_version"] != CANDIDATE_POOL_SERIALIZATION_VERSION:
        _fail("CandidatePool.manifest.serialization_version", "unsupported version")
    referenced: set[str] = set()
    groups = tuple(
        _decode_group(root, item, index, referenced)
        for index, item in enumerate(_list(manifest["groups"], "CandidatePool.groups"))
    )
    _validate_inventory(root, referenced)
    pool = CandidatePoolV1(
        source_set_digest=_text(
            manifest["source_set_digest"], "CandidatePool.source_set_digest"
        ),
        candidate_pool_configuration_digest=_text(
            manifest["candidate_pool_configuration_digest"],
            "CandidatePool.candidate_pool_configuration_digest",
        ),
        action_contract_digest=_text(
            manifest["action_contract_digest"], "CandidatePool.action_contract_digest"
        ),
        exclusion_inventory_digest=_text(
            manifest["exclusion_inventory_digest"],
            "CandidatePool.exclusion_inventory_digest",
        ),
        groups=groups,
        schema_version=_text(
            manifest["schema_version"], "CandidatePool.schema_version"
        ),
    )
    stored_digest = _text(
        manifest["candidate_pool_content_digest"],
        "CandidatePool.candidate_pool_content_digest",
    )
    if pool.content_digest != stored_digest:
        _fail("CandidatePool.content_digest", "manifest content changed")
    if (
        expected_content_digest is not None
        and pool.content_digest != expected_content_digest
    ):
        _fail("CandidatePool.content_digest", "does not match expected identity")
    return pool


__all__ = [
    "CANDIDATE_POOL_FORMAT",
    "CANDIDATE_POOL_MANIFEST_NAME",
    "CANDIDATE_POOL_SERIALIZATION_VERSION",
    "CandidatePoolSerializationError",
    "load_candidate_pool",
    "save_candidate_pool",
]
