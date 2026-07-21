"""Content-bound frozen-feature cache for WM-v0 training."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data_schema import WorldModelSample
from .encoder import ObservationEncoder


@dataclass(frozen=True, slots=True, eq=False)
class EncodedWorldModelSample:
    """One raw sample projected through an immutable frozen encoder."""

    sample_id: str
    source_episode_id: str
    split_group_id: str
    anchor_id: str
    candidate_id: str
    split: str
    policy_source: str
    encoder_identity: str
    event_names: tuple[str, ...]
    current_latents: NDArray[np.float32]
    future_latents: NDArray[np.float32]
    proprio: NDArray[np.float32]
    actions: NDArray[np.float32]
    action_mask: NDArray[np.bool_]
    future_progress: NDArray[np.float32]
    progress_mask: NDArray[np.bool_]
    event_labels: NDArray[np.float32]
    event_mask: NDArray[np.bool_]
    terminal_success: float
    terminal_success_mask: bool

    def __post_init__(self) -> None:
        """Validate aligned finite features and masks."""

        if self.current_latents.ndim != 2 or self.future_latents.ndim != 3:
            raise ValueError("feature shapes must be [V,D] and [K,V,D]")
        if tuple(self.future_latents.shape[1:]) != tuple(self.current_latents.shape):
            raise ValueError("current and future latent shapes differ")
        if not bool(
            np.isfinite(self.current_latents).all()
            and np.isfinite(self.future_latents).all()
        ):
            raise ValueError("encoded features must be finite")


def encode_world_model_sample(
    sample: WorldModelSample,
    encoder: ObservationEncoder,
    *,
    sample_id: str,
) -> EncodedWorldModelSample:
    """Encode current and future views once with the same frozen identity."""

    current = encoder.encode(sample.observation_t[None, ...])[0]
    future = encoder.encode(sample.future_observations)
    return EncodedWorldModelSample(
        sample_id=sample_id,
        source_episode_id=sample.source_episode_id,
        split_group_id=sample.split_group_id,
        anchor_id=sample.anchor_id,
        candidate_id=sample.candidate_id,
        split=sample.split.value,
        policy_source=sample.policy_source.value,
        encoder_identity=encoder.identity,
        event_names=sample.event_names,
        current_latents=np.asarray(current, dtype=np.float32),
        future_latents=np.asarray(future, dtype=np.float32),
        proprio=np.asarray(sample.proprio_t, dtype=np.float32),
        actions=np.asarray(sample.action_chunk, dtype=np.float32),
        action_mask=np.asarray(sample.action_mask, dtype=np.bool_),
        future_progress=np.asarray(sample.future_progress, dtype=np.float32),
        progress_mask=np.asarray(sample.progress_mask, dtype=np.bool_),
        event_labels=np.asarray(sample.event_labels, dtype=np.float32),
        event_mask=np.asarray(sample.event_mask, dtype=np.bool_),
        terminal_success=(
            0.0 if sample.terminal_success is None else float(sample.terminal_success)
        ),
        terminal_success_mask=sample.terminal_success is not None,
    )


def write_feature_cache(
    directory: Path,
    samples: Iterable[tuple[str, WorldModelSample]],
    encoder: ObservationEncoder,
) -> Path:
    """Write immutable per-sample feature records and one compact index."""

    root = Path(directory).absolute()
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for sample_id, sample in samples:
        encoded = encode_world_model_sample(sample, encoder, sample_id=sample_id)
        path = root / f"{sample_id}.npz"
        temporary = root / f".{sample_id}.tmp-{os.getpid()}.npz"
        metadata = {
            "anchor_id": encoded.anchor_id,
            "candidate_id": encoded.candidate_id,
            "encoder_identity": encoded.encoder_identity,
            "event_names": list(encoded.event_names),
            "policy_source": encoded.policy_source,
            "sample_id": encoded.sample_id,
            "source_episode_id": encoded.source_episode_id,
            "split": encoded.split,
            "split_group_id": encoded.split_group_id,
            "terminal_success": encoded.terminal_success,
            "terminal_success_mask": encoded.terminal_success_mask,
        }
        payload = json.dumps(metadata, sort_keys=True, allow_nan=False)
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                metadata=np.asarray(payload),
                current_latents=encoded.current_latents,
                future_latents=encoded.future_latents,
                proprio=encoded.proprio,
                actions=encoded.actions,
                action_mask=encoded.action_mask,
                future_progress=encoded.future_progress,
                progress_mask=encoded.progress_mask,
                event_labels=encoded.event_labels,
                event_mask=encoded.event_mask,
            )
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                temporary.unlink()
                raise ValueError(f"feature cache {sample_id} already differs")
            temporary.unlink()
        else:
            temporary.replace(path)
        records.append(metadata)
    index_without_digest: dict[str, object] = {
        "encoder_identity": encoder.identity,
        "latent_dimension": encoder.latent_dimension,
        "records": sorted(records, key=lambda item: str(item["sample_id"])),
        "schema_version": "1.0",
    }
    digest = hashlib.sha256(
        json.dumps(
            index_without_digest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    index = dict(index_without_digest)
    index["content_digest"] = f"sha256:{digest}"
    destination = root / "index.json"
    destination.write_text(
        json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_feature_sample(path: Path) -> EncodedWorldModelSample:
    """Load one cached record with pickle disabled."""

    with np.load(Path(path), allow_pickle=False) as loaded:
        expected = {
            "action_mask",
            "actions",
            "current_latents",
            "event_labels",
            "event_mask",
            "future_latents",
            "future_progress",
            "metadata",
            "progress_mask",
            "proprio",
        }
        if set(loaded.files) != expected:
            raise ValueError("feature cache inventory differs")
        metadata_value: Any = json.loads(str(loaded["metadata"].item()))
        if not isinstance(metadata_value, dict):
            raise ValueError("feature metadata must be an object")
        metadata: dict[str, Any] = metadata_value
        return EncodedWorldModelSample(
            sample_id=str(metadata["sample_id"]),
            source_episode_id=str(metadata["source_episode_id"]),
            split_group_id=str(metadata["split_group_id"]),
            anchor_id=str(metadata["anchor_id"]),
            candidate_id=str(metadata["candidate_id"]),
            split=str(metadata["split"]),
            policy_source=str(metadata["policy_source"]),
            encoder_identity=str(metadata["encoder_identity"]),
            event_names=tuple(str(item) for item in metadata["event_names"]),
            current_latents=np.asarray(loaded["current_latents"], dtype=np.float32),
            future_latents=np.asarray(loaded["future_latents"], dtype=np.float32),
            proprio=np.asarray(loaded["proprio"], dtype=np.float32),
            actions=np.asarray(loaded["actions"], dtype=np.float32),
            action_mask=np.asarray(loaded["action_mask"], dtype=np.bool_),
            future_progress=np.asarray(loaded["future_progress"], dtype=np.float32),
            progress_mask=np.asarray(loaded["progress_mask"], dtype=np.bool_),
            event_labels=np.asarray(loaded["event_labels"], dtype=np.float32),
            event_mask=np.asarray(loaded["event_mask"], dtype=np.bool_),
            terminal_success=float(metadata["terminal_success"]),
            terminal_success_mask=bool(metadata["terminal_success_mask"]),
        )
