"""Safe path-independent WM-v0 sample serialization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_schema import WorldModelSample, WorldModelSchemaError


@dataclass(frozen=True, slots=True)
class StoredSampleReference:
    """Content identities for one metadata/array sample pair."""

    sample_id: str
    metadata_digest: str
    arrays_digest: str


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return f"sha256:{value.hexdigest()}"


def write_world_model_sample(
    directory: Path, sample: WorldModelSample
) -> StoredSampleReference:
    """Write one sample atomically as JSON metadata plus non-pickled NPZ arrays."""

    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=True)
    sample_id = hashlib.sha256(
        f"{sample.episode_id}\0{sample.anchor_id}\0{sample.candidate_id}".encode()
    ).hexdigest()
    metadata_path = root / f"{sample_id}.json"
    arrays_path = root / f"{sample_id}.npz"
    metadata_payload = (
        json.dumps(sample.metadata_mapping(), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary_metadata = root / f".{sample_id}.json.tmp-{os.getpid()}"
    temporary_arrays = root / f".{sample_id}.npz.tmp-{os.getpid()}"
    try:
        temporary_metadata.write_bytes(metadata_payload)
        with temporary_arrays.open("wb") as stream:
            np.savez(stream, **sample.array_mapping())
            stream.flush()
            os.fsync(stream.fileno())
        if metadata_path.exists() or arrays_path.exists():
            existing = read_world_model_sample(root, sample_id)
            if existing.metadata_mapping() != sample.metadata_mapping() or any(
                not np.array_equal(existing.array_mapping()[name], value)
                for name, value in sample.array_mapping().items()
            ):
                raise WorldModelSchemaError(
                    f"sample {sample_id} already exists with different content"
                )
            return StoredSampleReference(
                sample_id, _digest(metadata_path), _digest(arrays_path)
            )
        temporary_metadata.replace(metadata_path)
        temporary_arrays.replace(arrays_path)
    finally:
        temporary_metadata.unlink(missing_ok=True)
        temporary_arrays.unlink(missing_ok=True)
    return StoredSampleReference(
        sample_id, _digest(metadata_path), _digest(arrays_path)
    )


def read_world_model_sample(directory: Path, sample_id: str) -> WorldModelSample:
    """Reload one stored sample with pickle disabled and strict field checks."""

    if len(sample_id) != 64 or any(
        character not in "0123456789abcdef" for character in sample_id
    ):
        raise WorldModelSchemaError("sample_id must be a lowercase SHA-256 hex value")
    root = Path(directory).absolute()
    try:
        metadata = json.loads((root / f"{sample_id}.json").read_text("utf-8"))
        with np.load(root / f"{sample_id}.npz", allow_pickle=False) as loaded:
            expected = {
                "action_chunk",
                "action_mask",
                "event_labels",
                "event_mask",
                "future_observations",
                "future_progress",
                "future_proprio",
                "observation_t",
                "progress_mask",
                "progress_t",
                "progress_t_present",
                "proprio_t",
            }
            if set(loaded.files) != expected:
                raise WorldModelSchemaError("stored sample array inventory differs")
            arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, WorldModelSchemaError):
            raise
        raise WorldModelSchemaError(
            f"could not load sample {sample_id}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise WorldModelSchemaError("stored sample metadata must be an object")
    return WorldModelSample.from_parts(metadata, arrays)
