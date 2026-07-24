"""Remote-only frozen reward-model inference utilities."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1_training import EpisodeDatasetCache, peak_cuda_memory, tensor_digest
from _lg_r1c_common import (
    file_identity,
    output_root,
    read_jsonl,
    sha256_path,
    write_json,
)


def model_root() -> Path:
    """Return the external exact-revision model snapshot root."""

    configured = os.environ.get("LG_R1C_MODEL_ROOT")
    root = Path(configured) if configured else output_root() / "model_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_exact_local_snapshot(destination: Path, *, revision: str) -> bool:
    if not destination.is_dir():
        return False
    files = [
        path
        for path in sorted(destination.iterdir())
        if path.is_file() and not path.name.startswith(".")
    ]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset({path.name for path in files}):
        return False
    metadata_root = destination / ".cache" / "huggingface" / "download"
    for path in files:
        metadata = metadata_root / f"{path.name}.metadata"
        if not metadata.is_file():
            return False
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0] != revision:
            return False
    return True


def snapshot_exact(
    *,
    role: str,
    model_id: str,
    revision: str,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    """Download one exact Hub revision outside Git."""

    if len(revision) != 40 or not all(
        character in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{role} requires a full lowercase commit revision")
    from huggingface_hub import snapshot_download

    destination = model_root() / f"{role}-{revision[:12]}"
    if _is_exact_local_snapshot(destination, revision=revision):
        return destination
    resolved = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
    )
    path = Path(resolved)
    if not path.is_dir():
        raise FileNotFoundError(f"snapshot download did not create {path}")
    if not _is_exact_local_snapshot(path, revision=revision):
        raise ValueError(f"{role} snapshot metadata is not bound to {revision}")
    return path


def validate_snapshot_files(
    snapshot: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on missing model files, sizes, or SHA-256 values."""

    validated: dict[str, Any] = {}
    for relative, identity in expected.items():
        if not isinstance(identity, dict):
            raise ValueError(f"invalid expected identity for {relative}")
        path = snapshot / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_identity(path, locator=relative)
        if int(actual["bytes"]) != int(identity["bytes"]):
            raise ValueError(f"model file size mismatch: {relative}")
        if actual["sha256"] != identity["sha256"]:
            raise ValueError(f"model file hash mismatch: {relative}")
        validated[relative] = actual
    return validated


def processor_identity(snapshot: Path) -> dict[str, Any]:
    """Hash every processor/tokenizer/config file in one snapshot."""

    suffixes = {".json", ".txt", ".model"}
    files = [
        path
        for path in sorted(snapshot.iterdir())
        if path.is_file() and path.suffix in suffixes
    ]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    present = {path.name for path in files}
    missing = sorted(required - present)
    if missing:
        raise FileNotFoundError(f"processor snapshot missing files: {missing}")
    digest = hashlib.sha256()
    identities = []
    for path in files:
        sha256 = sha256_path(path)
        digest.update(f"{path.name}:{sha256}\n".encode())
        identities.append(
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256,
            }
        )
    return {
        "combined_sha256": digest.hexdigest(),
        "files": identities,
    }


def load_windows(root: Path) -> list[dict[str, Any]]:
    """Load the frozen window stream after its compact manifest exists."""

    manifest = root / "window_manifest.json"
    windows = root / "reward_windows.jsonl"
    if not manifest.is_file() or not windows.is_file():
        raise FileNotFoundError("frozen window manifest and JSONL are required")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError("window manifest did not pass")
    if sha256_path(windows) != payload["windows_file"]["sha256"]:
        raise ValueError("reward window JSONL hash drift")
    rows = read_jsonl(windows)
    if len(rows) != int(payload["window_count"]):
        raise ValueError("reward window count drift")
    return rows


def load_completed_ids(path: Path, *, model_revision: str) -> set[str]:
    """Validate and return completed window identities for safe resume."""

    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid prediction row {line_number}")
            if row.get("model_revision") != model_revision:
                raise ValueError("prediction resume model revision mismatch")
            window_id = str(row["window_id"])
            if window_id in completed:
                raise ValueError(f"duplicate completed window: {window_id}")
            completed.add(window_id)
    return completed


def append_predictions(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append complete JSON lines and flush them for resumable inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_videos(
    cache: EpisodeDatasetCache,
    windows: Sequence[dict[str, Any]],
    *,
    frame_key: str,
    index_key: str,
) -> Any:
    """Load a `(B,T,C,H,W)` float tensor from the frozen episode datasets."""

    import torch

    videos = []
    for window in windows:
        frames = [
            cache.frame(str(window["episode_id"]), int(index))[frame_key]
            for index in window[index_key]
        ]
        videos.append(torch.stack(frames))
    return torch.stack(videos)


def videos_to_uint8_samples(
    videos: Any,
    instructions: Sequence[str],
) -> list[tuple[np.ndarray, str]]:
    """Convert batched LeRobot tensors to ROBOMETER processor samples."""

    samples = []
    for video, instruction in zip(videos, instructions, strict=True):
        array = video.detach().to("cpu").permute(0, 2, 3, 1).numpy()
        if (
            np.issubdtype(array.dtype, np.floating)
            and array.size
            and array.max() <= 1.0
        ):
            array = array * 255.0
        samples.append((np.clip(array, 0, 255).astype(np.uint8), instruction))
    return samples


def benchmark_summary(
    *,
    prediction_path: Path,
    model: Any,
    digest_before: str,
    elapsed_seconds: float,
    windows: int,
    model_identity: dict[str, Any],
    processor: dict[str, Any],
) -> dict[str, Any]:
    """Build an inference-only completion record."""

    digest_after = tensor_digest(model)
    if digest_after != digest_before:
        raise RuntimeError("frozen reward model parameters changed during inference")
    return {
        "status": "pass",
        "windows": windows,
        "elapsed_seconds": elapsed_seconds,
        "mean_latency_ms": (elapsed_seconds * 1000.0 / max(windows, 1)),
        "prediction_file": file_identity(
            prediction_path,
            locator=prediction_path.name,
        ),
        "model_identity": model_identity,
        "processor_identity": processor,
        "backbone_digest_before": digest_before,
        "backbone_digest_after": digest_after,
        "foundation_model_frozen": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "cuda_memory": peak_cuda_memory(),
    }


def monotonic_time() -> float:
    """Return a monotonic timer value."""

    return time.monotonic()


__all__ = [
    "EpisodeDatasetCache",
    "append_predictions",
    "benchmark_summary",
    "load_completed_ids",
    "load_videos",
    "load_windows",
    "model_root",
    "monotonic_time",
    "processor_identity",
    "snapshot_exact",
    "tensor_digest",
    "validate_snapshot_files",
    "videos_to_uint8_samples",
    "write_json",
]
