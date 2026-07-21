"""Episode-scoped PickCube ACT demonstrations and train-only statistics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.control.serialization import write_atomic_json
from latentguard.replay.identity import canonical_json_bytes


class PickCubeDemoDataError(ValueError):
    """Raised when a native-policy demonstration violates its contract."""


class PickCubeDemoSplit(StrEnum):
    """Episode-level split assigned deterministically from the scene seed."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeDemoDataError(f"{context}: {reason}")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def split_for_scene_seed(seed: int) -> PickCubeDemoSplit:
    """Assign 80/10/10 by scene seed without any frame-level randomization."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        _fail("split_for_scene_seed", "seed must be uint32")
    remainder = seed % 10
    if remainder == 0:
        return PickCubeDemoSplit.VALIDATION
    if remainder == 1:
        return PickCubeDemoSplit.TEST
    return PickCubeDemoSplit.TRAIN


def _finite_array(
    value: object,
    *,
    dtype: np.dtype[Any],
    shape_tail: tuple[int, ...],
    context: str,
) -> NDArray[Any]:
    array = np.array(value, copy=True, order="C", subok=False)
    if array.dtype != dtype:
        _fail(context, f"expected dtype {dtype.str}")
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        _fail(context, f"expected [T,{','.join(map(str, shape_tail))}]")
    if array.shape[0] < 1 or not np.all(np.isfinite(array)):
        _fail(context, "expected non-empty finite values")
    return array


@dataclass(frozen=True, slots=True)
class PickCubeDemoEpisode:
    """One complete successful expert episode with pre-action observations."""

    scene_seed: int
    rgb: NDArray[Any]
    proprioception: NDArray[Any]
    actions: NDArray[Any]
    phases: tuple[str, ...]
    compatibility_identity: str
    contract_digest: str
    camera_configuration_digest: str
    success: bool = True
    termination_reason: str = "official_pickcube_success"
    control_frequency_hz: float = 20.0
    schema_version: str = "pickcube-act-demo-episode-v1"

    def __post_init__(self) -> None:
        if type(self.scene_seed) is not int or not 0 <= self.scene_seed < 2**32:
            _fail("PickCubeDemoEpisode.scene_seed", "expected uint32")
        if self.schema_version != "pickcube-act-demo-episode-v1":
            _fail("PickCubeDemoEpisode.schema_version", "unsupported version")
        if (
            self.success is not True
            or self.termination_reason != "official_pickcube_success"
        ):
            _fail("PickCubeDemoEpisode", "training episodes must be complete successes")
        if self.control_frequency_hz != 20.0:
            _fail("PickCubeDemoEpisode.control_frequency_hz", "expected 20 Hz")
        for name, value in (
            ("compatibility_identity", self.compatibility_identity),
            ("contract_digest", self.contract_digest),
            ("camera_configuration_digest", self.camera_configuration_digest),
        ):
            if (
                not isinstance(value, str)
                or not value.startswith("sha256:")
                or len(value) != 71
            ):
                _fail(f"PickCubeDemoEpisode.{name}", "expected SHA-256 identity")
        rgb = np.array(self.rgb, copy=True, order="C", subok=False)
        if (
            rgb.dtype != np.dtype(np.uint8)
            or rgb.ndim != 4
            or rgb.shape[1:]
            != (
                224,
                224,
                3,
            )
        ):
            _fail("PickCubeDemoEpisode.rgb", "expected uint8[T,224,224,3]")
        if rgb.shape[0] < 1:
            _fail("PickCubeDemoEpisode.rgb", "episode must contain frames")
        state = _finite_array(
            self.proprioception,
            dtype=np.dtype(np.float32),
            shape_tail=(18,),
            context="PickCubeDemoEpisode.proprioception",
        )
        actions = np.array(self.actions, copy=True, order="C", subok=False)
        if (
            actions.ndim != 2
            or actions.shape[0] < 1
            or actions.shape[1:] != (8,)
            or actions.dtype.hasobject
            or not np.issubdtype(actions.dtype, np.floating)
            or not np.all(np.isfinite(actions))
        ):
            _fail(
                "PickCubeDemoEpisode.actions",
                "expected non-empty finite floating [T,8]",
            )
        phases = tuple(self.phases)
        if (
            rgb.shape[0] != state.shape[0]
            or state.shape[0] != actions.shape[0]
            or len(phases) != actions.shape[0]
        ):
            _fail("PickCubeDemoEpisode", "image/state/action/phase alignment differs")
        if any(not isinstance(phase, str) or not phase for phase in phases):
            _fail("PickCubeDemoEpisode.phases", "expected non-empty phase names")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "proprioception", state)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "phases", phases)

    @property
    def split(self) -> PickCubeDemoSplit:
        """Return the scene-seed split shared by every episode frame."""
        return split_for_scene_seed(self.scene_seed)

    @property
    def frame_count(self) -> int:
        """Return the exact number of aligned pre-action frames."""
        return int(self.actions.shape[0])

    @property
    def episode_id(self) -> str:
        """Return a content identity independent of storage location."""
        payload = {
            "camera_configuration_digest": self.camera_configuration_digest,
            "compatibility_identity": self.compatibility_identity,
            "contract_digest": self.contract_digest,
            "scene_seed": self.scene_seed,
            "schema_version": self.schema_version,
        }
        digest = hashlib.sha256(canonical_json_bytes(payload, context="DemoEpisode"))
        for array in (self.rgb, self.proprioception, self.actions):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        digest.update(canonical_json_bytes(list(self.phases), context="DemoPhases"))
        return f"pickcube-demo-{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class DemoEpisodeReference:
    """Strict relative file inventory for one persisted demonstration."""

    episode_id: str
    scene_seed: int
    split: PickCubeDemoSplit
    frame_count: int
    relative_directory: str
    rgb_sha256: str
    proprioception_sha256: str
    actions_sha256: str
    metadata_sha256: str

    def to_mapping(self) -> dict[str, object]:
        """Return the JSON-ready reference."""
        return {
            "actions_sha256": self.actions_sha256,
            "episode_id": self.episode_id,
            "frame_count": self.frame_count,
            "metadata_sha256": self.metadata_sha256,
            "proprioception_sha256": self.proprioception_sha256,
            "relative_directory": self.relative_directory,
            "rgb_sha256": self.rgb_sha256,
            "scene_seed": self.scene_seed,
            "split": self.split.value,
        }


def _safe_episode_directory(root: Path, episode_id: str) -> Path:
    if not episode_id.startswith("pickcube-demo-") or len(episode_id) != 78:
        _fail("episode_id", "malformed content identity")
    return root / "episodes" / episode_id


def save_demo_episode(root: Path, episode: PickCubeDemoEpisode) -> DemoEpisodeReference:
    """Commit one episode directory atomically and refuse duplicate publication."""
    dataset_root = Path(root).absolute()
    final = _safe_episode_directory(dataset_root, episode.episode_id)
    if final.exists() or final.is_symlink():
        _fail("save_demo_episode", "episode already exists")
    staging = dataset_root / ".staging" / episode.episode_id
    if staging.exists() or staging.is_symlink():
        _fail("save_demo_episode", "staging episode already exists")
    staging.mkdir(parents=True)
    try:
        np.save(staging / "rgb.npy", episode.rgb, allow_pickle=False)
        np.save(
            staging / "proprioception.npy",
            episode.proprioception,
            allow_pickle=False,
        )
        np.save(staging / "actions.npy", episode.actions, allow_pickle=False)
        metadata = {
            "camera_configuration_digest": episode.camera_configuration_digest,
            "action_dtype": episode.actions.dtype.str,
            "compatibility_identity": episode.compatibility_identity,
            "contract_digest": episode.contract_digest,
            "control_frequency_hz": episode.control_frequency_hz,
            "episode_id": episode.episode_id,
            "frame_count": episode.frame_count,
            "phases": list(episode.phases),
            "scene_seed": episode.scene_seed,
            "schema_version": episode.schema_version,
            "split": episode.split.value,
            "success": episode.success,
            "termination_reason": episode.termination_reason,
        }
        write_atomic_json(staging / "episode.json", metadata)
        (dataset_root / "episodes").mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_demo_episode_reference(dataset_root, final)


def _read_mapping(path: Path, *, context: str) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeDemoDataError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def load_demo_episode_reference(
    root: Path,
    episode_directory: Path,
) -> DemoEpisodeReference:
    """Verify one persisted episode inventory without loading image bytes."""
    dataset_root = Path(root).absolute()
    directory = Path(episode_directory).absolute()
    try:
        relative = directory.relative_to(dataset_root)
    except ValueError as exc:
        raise PickCubeDemoDataError("episode directory escapes dataset root") from exc
    if directory.is_symlink() or not directory.is_dir():
        _fail("load_demo_episode_reference", "expected unlinked directory")
    metadata_path = directory / "episode.json"
    metadata = _read_mapping(metadata_path, context="demo episode metadata")
    expected = {
        "camera_configuration_digest",
        "action_dtype",
        "compatibility_identity",
        "contract_digest",
        "control_frequency_hz",
        "episode_id",
        "frame_count",
        "phases",
        "scene_seed",
        "schema_version",
        "split",
        "success",
        "termination_reason",
    }
    if set(metadata) != expected:
        _fail("demo episode metadata", "field inventory mismatch")
    episode_id = metadata["episode_id"]
    scene_seed = metadata["scene_seed"]
    frame_count = metadata["frame_count"]
    split_value = metadata["split"]
    if not isinstance(episode_id, str) or episode_id != directory.name:
        _fail("demo episode metadata", "episode identity mismatch")
    if type(scene_seed) is not int or not 0 <= scene_seed < 2**32:
        _fail("demo episode metadata", "invalid scene seed")
    if type(frame_count) is not int or frame_count < 1:
        _fail("demo episode metadata", "invalid frame count")
    try:
        split = PickCubeDemoSplit(cast(str, split_value))
    except (TypeError, ValueError) as exc:
        raise PickCubeDemoDataError("demo episode metadata: invalid split") from exc
    if split is not split_for_scene_seed(scene_seed):
        _fail("demo episode metadata", "scene seed crossed split contract")
    relative_posix = PurePosixPath(*relative.parts).as_posix()
    return DemoEpisodeReference(
        episode_id=episode_id,
        scene_seed=scene_seed,
        split=split,
        frame_count=frame_count,
        relative_directory=relative_posix,
        rgb_sha256=_sha256_file(directory / "rgb.npy"),
        proprioception_sha256=_sha256_file(directory / "proprioception.npy"),
        actions_sha256=_sha256_file(directory / "actions.npy"),
        metadata_sha256=_sha256_file(metadata_path),
    )


def _load_npy(path: Path) -> NDArray[Any]:
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(value, np.ndarray):
        _fail("demo array", "expected NumPy array")
    return value


def read_demo_episode(
    root: Path, reference: DemoEpisodeReference
) -> PickCubeDemoEpisode:
    """Reload all arrays, verify hashes, and reconstruct the strict episode."""
    dataset_root = Path(root).absolute()
    directory = dataset_root.joinpath(
        *PurePosixPath(reference.relative_directory).parts
    )
    observed = load_demo_episode_reference(dataset_root, directory)
    if observed != reference:
        _fail("read_demo_episode", "episode file inventory changed")
    metadata = _read_mapping(directory / "episode.json", context="demo metadata")
    phases = metadata["phases"]
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        _fail("demo metadata phases", "expected array")
    episode = PickCubeDemoEpisode(
        scene_seed=reference.scene_seed,
        rgb=_load_npy(directory / "rgb.npy"),
        proprioception=_load_npy(directory / "proprioception.npy"),
        actions=_load_npy(directory / "actions.npy"),
        phases=tuple(cast(Sequence[str], phases)),
        compatibility_identity=cast(str, metadata["compatibility_identity"]),
        contract_digest=cast(str, metadata["contract_digest"]),
        camera_configuration_digest=cast(str, metadata["camera_configuration_digest"]),
        success=cast(bool, metadata["success"]),
        termination_reason=cast(str, metadata["termination_reason"]),
        control_frequency_hz=cast(float, metadata["control_frequency_hz"]),
        schema_version=cast(str, metadata["schema_version"]),
    )
    if episode.actions.dtype.str != metadata["action_dtype"]:
        _fail("read_demo_episode", "declared action dtype changed")
    if episode.episode_id != reference.episode_id:
        _fail("read_demo_episode", "array content identity changed")
    return episode


def collect_episode_references(root: Path) -> tuple[DemoEpisodeReference, ...]:
    """Load every committed episode in deterministic scene-seed order."""
    dataset_root = Path(root).absolute()
    episodes_root = dataset_root / "episodes"
    if not episodes_root.is_dir() or episodes_root.is_symlink():
        _fail("collect_episode_references", "episodes directory is missing")
    references = tuple(
        load_demo_episode_reference(dataset_root, directory)
        for directory in sorted(episodes_root.iterdir())
        if directory.is_dir() and not directory.is_symlink()
    )
    seeds = [reference.scene_seed for reference in references]
    if len(set(seeds)) != len(seeds):
        _fail("collect_episode_references", "duplicate scene seed")
    return tuple(sorted(references, key=lambda item: item.scene_seed))


def _array_statistics(values: NDArray[Any]) -> dict[str, object]:
    return {
        "maximum": np.max(values, axis=0).astype(np.float64).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "minimum": np.min(values, axis=0).astype(np.float64).tolist(),
        "standard_deviation": np.std(values, axis=0, dtype=np.float64).tolist(),
    }


def _image_statistics(episodes: Sequence[PickCubeDemoEpisode]) -> dict[str, object]:
    count = 0
    channel_sum = np.zeros((3,), dtype=np.float64)
    channel_square_sum = np.zeros((3,), dtype=np.float64)
    for episode in episodes:
        values = episode.rgb.astype(np.float64) / 255.0
        flattened = values.reshape(-1, 3)
        count += flattened.shape[0]
        channel_sum += np.sum(flattened, axis=0, dtype=np.float64)
        channel_square_sum += np.sum(flattened * flattened, axis=0, dtype=np.float64)
    if count < 1:
        _fail("image statistics", "no training pixels")
    mean = channel_sum / count
    variance = np.maximum(channel_square_sum / count - mean * mean, 0.0)
    return {
        "channel_mean": mean.tolist(),
        "channel_standard_deviation": np.sqrt(variance).tolist(),
        "input_dtype": "uint8",
        "input_range": [0, 255],
        "processor_conversion": "float32_divide_255_then_mean_std",
    }


def build_demo_dataset_reports(
    root: Path,
    *,
    contract_digest: str,
    expert_success_rate: float,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    """Reload all episodes and build manifest, quality, and train-only stats."""
    if not 0.0 <= expert_success_rate <= 1.0 or not math.isfinite(expert_success_rate):
        _fail("expert_success_rate", "expected finite ratio")
    references = collect_episode_references(root)
    if not references:
        _fail("build_demo_dataset_reports", "no episodes found")
    episodes = tuple(read_demo_episode(root, reference) for reference in references)
    if any(episode.contract_digest != contract_digest for episode in episodes):
        _fail("build_demo_dataset_reports", "episode contract digest mismatch")
    split_counts = Counter({split.value: 0 for split in PickCubeDemoSplit})
    split_counts.update(reference.split.value for reference in references)
    split_frames = Counter(
        {
            split.value: sum(
                reference.frame_count
                for reference in references
                if reference.split is split
            )
            for split in PickCubeDemoSplit
        }
    )
    train = tuple(
        episode
        for episode, reference in zip(episodes, references, strict=True)
        if reference.split is PickCubeDemoSplit.TRAIN
    )
    if not train:
        _fail("build_demo_dataset_reports", "training split is empty")
    train_state = np.concatenate([episode.proprioception for episode in train], axis=0)
    train_action = np.concatenate([episode.actions for episode in train], axis=0)
    phase_counts = Counter(phase for episode in episodes for phase in episode.phases)
    manifest_body: dict[str, object] = {
        "contract_digest": contract_digest,
        "episode_count": len(references),
        "episodes": [reference.to_mapping() for reference in references],
        "frame_count": sum(reference.frame_count for reference in references),
        "schema_version": "pickcube-act-dataset-manifest-v1",
        "split_episode_counts": dict(sorted(split_counts.items())),
        "split_frame_counts": dict(sorted(split_frames.items())),
        "split_semantic": "scene_seed_modulo_10_80_10_10_v1",
    }
    manifest_body["dataset_digest"] = _sha256_bytes(
        canonical_json_bytes(manifest_body, context="PickCubeActDatasetManifest")
    )
    quality: dict[str, object] = {
        "action_distribution": _array_statistics(
            np.concatenate([episode.actions for episode in episodes], axis=0)
        ),
        "average_episode_length": sum(item.frame_count for item in references)
        / len(references),
        "duplicate_episode_count": len(references)
        - len({episode.episode_id for episode in episodes}),
        "episode_count": len(references),
        "expert_success_rate": expert_success_rate,
        "frame_count": sum(item.frame_count for item in references),
        "gripper_distribution": _array_statistics(
            np.concatenate([episode.actions[:, 7:8] for episode in episodes], axis=0)
        ),
        "missing_observation_count": 0,
        "nonfinite_value_count": 0,
        "action_dtypes": sorted({episode.actions.dtype.str for episode in episodes}),
        "phase_frame_counts": dict(sorted(phase_counts.items())),
        "scene_seed_count": len({item.scene_seed for item in references}),
        "schema_version": "pickcube-act-data-quality-v1",
        "split_episode_counts": dict(sorted(split_counts.items())),
        "split_frame_counts": dict(sorted(split_frames.items())),
        "split_leakage_count": 0,
    }
    normalization_body: dict[str, object] = {
        "action": _array_statistics(train_action),
        "action_clipping": "prohibited_fail_closed",
        "gripper_special_handling": "none_beyond_shared_action_mean_std",
        "image": _image_statistics(train),
        "proprioception": _array_statistics(train_state),
        "schema_version": "pickcube-act-normalization-v1",
        "source_dataset_digest": manifest_body["dataset_digest"],
        "source_split": "train",
    }
    normalization_body["normalization_digest"] = _sha256_bytes(
        canonical_json_bytes(normalization_body, context="PickCubeActNormalization")
    )
    return (
        MappingProxyType(manifest_body),
        MappingProxyType(quality),
        MappingProxyType(normalization_body),
    )


def action_chunk_at(
    actions: NDArray[Any],
    index: int,
    chunk_size: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Build one future chunk and an exact padding mask without data repair."""
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1:] != (8,) or not np.all(np.isfinite(values)):
        _fail("action_chunk_at.actions", "expected finite [T,8]")
    if type(index) is not int or not 0 <= index < values.shape[0]:
        _fail("action_chunk_at.index", "index is outside episode")
    if type(chunk_size) is not int or chunk_size < 1:
        _fail("action_chunk_at.chunk_size", "expected positive integer")
    end = min(values.shape[0], index + chunk_size)
    valid = np.array(values[index:end], dtype=np.float32, copy=True)
    chunk = np.zeros((chunk_size, 8), dtype=np.float32)
    chunk[: valid.shape[0]] = valid
    is_pad = np.ones((chunk_size,), dtype=np.bool_)
    is_pad[: valid.shape[0]] = False
    return chunk, is_pad


__all__ = [
    "DemoEpisodeReference",
    "PickCubeDemoDataError",
    "PickCubeDemoEpisode",
    "PickCubeDemoSplit",
    "action_chunk_at",
    "build_demo_dataset_reports",
    "collect_episode_references",
    "load_demo_episode_reference",
    "read_demo_episode",
    "save_demo_episode",
    "split_for_scene_seed",
]
