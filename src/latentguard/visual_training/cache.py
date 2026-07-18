"""Safe content-bound feature and privileged-teacher caches for M4B."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.vision_data.serialization import load_visual_image
from latentguard.visual_training.backbone import FrozenBackboneRuntime

FEATURE_CACHE_FORMAT = "latentguard-m4b-feature-cache-v1"
TEACHER_CACHE_FORMAT = "latentguard-m4b-teacher-cache-v1"
FEATURE_COMPARISON_TOLERANCE = 0.0


class VisualCacheError(ValueError):
    """Raised when a cache is incomplete, unsafe, or content-drifted."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualCacheError(f"{context}: {reason}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _digest(value: object, context: str) -> str:
    payload = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with Path(path).open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True, slots=True)
class FeatureCacheEntryV1:
    """One unique packet/view row bound to authoritative image content."""

    packet_id: str
    camera_slot: int
    image_digest: str
    feature_row: int

    def as_mapping(self) -> dict[str, object]:
        """Return semantic row fields without runtime paths."""
        return {
            "camera_slot": self.camera_slot,
            "feature_row": self.feature_row,
            "image_digest": self.image_digest,
            "packet_id": self.packet_id,
        }


@dataclass(frozen=True, slots=True, eq=False)
class LoadedFeatureCacheV1:
    """Strictly verified feature cache and immutable lookup inventory."""

    git_sha: str
    dataset_digest: str
    source_dataset_digest: str
    split_digest: str
    backbone_digest: str
    weight_content_digest: str
    model_freeze_digest: str | None
    preprocessing_semantic: str
    entries: tuple[FeatureCacheEntryV1, ...]
    features: NDArray[np.float32]
    content_digest: str

    def feature(self, packet_id: str, camera_slot: int) -> NDArray[np.float32]:
        """Return one exact cached feature by semantic packet and ordered slot."""
        for item in self.entries:
            if item.packet_id == packet_id and item.camera_slot == camera_slot:
                return cast(NDArray[np.float32], self.features[item.feature_row])
        raise VisualCacheError("feature cache: packet/view key is absent")


def _feature_inventory(dataset: object) -> tuple[FeatureCacheEntryV1, ...]:
    packets = getattr(dataset, "packets", None)
    if not isinstance(packets, tuple) or not packets:
        _fail("feature cache dataset", "expected immutable packet inventory")
    entries: list[FeatureCacheEntryV1] = []
    seen: set[tuple[str, int]] = set()
    for packet in packets:
        views = tuple(getattr(packet, "views", ()))
        if len(views) != 3:
            _fail("feature cache dataset", "expected three ordered views")
        for slot, view in enumerate(views):
            key = (packet.packet_id, slot)
            if key in seen:
                _fail("feature cache dataset", "duplicate packet/view key")
            seen.add(key)
            entries.append(
                FeatureCacheEntryV1(
                    packet_id=packet.packet_id,
                    camera_slot=slot,
                    image_digest=view.pixel_sha256,
                    feature_row=len(entries),
                )
            )
    return tuple(entries)


def extract_feature_cache(
    dataset: object,
    dataset_root: Path,
    runtime: FrozenBackboneRuntime,
    output_dir: Path,
    *,
    git_sha: str,
    model_freeze_digest: str | None = None,
    batch_size: int = 64,
    limit_images: int | None = None,
    resume: bool = False,
) -> tuple[LoadedFeatureCacheV1, int]:
    """Extract each unique image once into a transactional safe NPY cache."""
    if type(batch_size) is not int or batch_size <= 0:
        _fail("feature cache", "batch_size must be positive")
    dataset_digest = getattr(dataset, "content_digest", None)
    source_dataset_digest = getattr(
        dataset,
        "source_dataset_digest",
        getattr(dataset, "candidate_pool_identity", None),
    )
    split_digest = getattr(
        dataset,
        "split_digest",
        getattr(dataset, "source_set_identity", None),
    )
    if (
        not isinstance(dataset_digest, str)
        or not isinstance(source_dataset_digest, str)
        or not isinstance(split_digest, str)
    ):
        _fail("feature cache dataset", "content or source identity is unavailable")
    if (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or any(item not in "0123456789abcdef" for item in git_sha)
    ):
        _fail("feature cache", "expected full lowercase Git SHA")
    inventory = _feature_inventory(dataset)
    if limit_images is not None:
        if type(limit_images) is not int or limit_images <= 0:
            _fail("feature cache", "limit_images must be positive")
        inventory = inventory[:limit_images]
    destination = Path(output_dir).absolute()
    if destination.exists():
        loaded = load_feature_cache(destination)
        if not resume:
            _fail("feature cache", "output exists; use --resume")
        if (
            loaded.dataset_digest != dataset_digest
            or loaded.source_dataset_digest != source_dataset_digest
            or loaded.split_digest != split_digest
            or loaded.backbone_digest != runtime.manifest.content_digest
            or loaded.git_sha != git_sha
            or loaded.model_freeze_digest != model_freeze_digest
            or loaded.entries != inventory
        ):
            _fail("feature cache", "resume identity differs")
        return loaded, 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(f".{destination.name}.partial")
    if work.exists() and not resume:
        _fail("feature cache", "partial output exists; use --resume")
    work.mkdir(exist_ok=True)
    array_path = work / "features.npy"
    ledger_path = work / "ledger.json"
    completed = 0
    if array_path.exists() or ledger_path.exists():
        if not (array_path.is_file() and ledger_path.is_file() and resume):
            _fail("feature cache", "partial inventory is incomplete")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict) or set(ledger) != {"completed", "identity"}:
            _fail("feature cache", "partial ledger is malformed")
        identity = _digest(
            {
                "backbone_digest": runtime.manifest.content_digest,
                "dataset_digest": dataset_digest,
                "git_sha": git_sha,
                "model_freeze_digest": model_freeze_digest,
                "source_dataset_digest": source_dataset_digest,
                "split_digest": split_digest,
                "entries": [item.as_mapping() for item in inventory],
            },
            "M4BFeatureCachePartialV1",
        )
        if ledger["identity"] != identity or type(ledger["completed"]) is not int:
            _fail("feature cache", "partial identity differs")
        completed = ledger["completed"]
        features = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            array_path, mode="r+", dtype="<f4", shape=(len(inventory), 512)
        )
    else:
        features = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            array_path, mode="w+", dtype="<f4", shape=(len(inventory), 512)
        )
    packets = {packet.packet_id: packet for packet in cast(Any, dataset).packets}
    identity = _digest(
        {
            "backbone_digest": runtime.manifest.content_digest,
            "dataset_digest": dataset_digest,
            "git_sha": git_sha,
            "model_freeze_digest": model_freeze_digest,
            "source_dataset_digest": source_dataset_digest,
            "split_digest": split_digest,
            "entries": [item.as_mapping() for item in inventory],
        },
        "M4BFeatureCachePartialV1",
    )
    extracted = 0
    for start in range(completed, len(inventory), batch_size):
        batch = inventory[start : start + batch_size]
        images = []
        for item in batch:
            view = packets[item.packet_id].views[item.camera_slot]
            image = load_visual_image(dataset_root, view.image_reference)
            if view.pixel_sha256 != item.image_digest:
                _fail("feature cache", "image binding changed")
            images.append(image)
        observed = runtime.extract(np.stack(images, axis=0))
        features[start : start + len(batch)] = observed
        features.flush()
        completed = start + len(batch)
        _write_json(ledger_path, {"completed": completed, "identity": identity})
        extracted += len(batch)
    del features
    features_digest = _sha256_file(array_path)
    manifest_body = {
        "backbone_digest": runtime.manifest.content_digest,
        "dataset_digest": dataset_digest,
        "git_sha": git_sha,
        "model_freeze_digest": model_freeze_digest,
        "source_dataset_digest": source_dataset_digest,
        "split_digest": split_digest,
        "entries": [item.as_mapping() for item in inventory],
        "feature_dtype": "float32",
        "feature_shape": [len(inventory), 512],
        "features_npy_digest": features_digest,
        "format": FEATURE_CACHE_FORMAT,
        "preprocessing_semantic": runtime.manifest.preprocessing_semantic,
        "weight_content_digest": runtime.manifest.weight_content_digest,
        "schema_version": "1.0",
    }
    manifest = {
        **manifest_body,
        "content_digest": _digest(manifest_body, "M4BFeatureCacheV1"),
    }
    _write_json(work / "manifest.json", manifest)
    ledger_path.unlink()
    work.rename(destination)
    return load_feature_cache(destination), extracted


def load_feature_cache(output_dir: Path) -> LoadedFeatureCacheV1:
    """Strictly reload a complete cache and reject any file or row drift."""
    root = Path(output_dir).absolute()
    if root.is_symlink() or not root.is_dir():
        _fail("feature cache", "expected regular directory")
    observed_files = {item.name for item in root.iterdir() if item.is_file()}
    if observed_files != {"features.npy", "manifest.json"}:
        _fail("feature cache", "file inventory differs")
    raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "backbone_digest",
        "content_digest",
        "dataset_digest",
        "entries",
        "feature_dtype",
        "feature_shape",
        "features_npy_digest",
        "format",
        "git_sha",
        "model_freeze_digest",
        "preprocessing_semantic",
        "schema_version",
        "source_dataset_digest",
        "split_digest",
        "weight_content_digest",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        _fail("feature cache manifest", "unexpected or missing fields")
    body = {key: value for key, value in raw.items() if key != "content_digest"}
    if (
        raw["format"] != FEATURE_CACHE_FORMAT
        or raw["schema_version"] != "1.0"
        or raw["content_digest"] != _digest(body, "M4BFeatureCacheV1")
    ):
        _fail("feature cache manifest", "identity differs")
    if _sha256_file(root / "features.npy") != raw["features_npy_digest"]:
        _fail("feature cache", "feature bytes changed")
    try:
        features = np.load(root / "features.npy", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise VisualCacheError(f"feature cache: invalid NPY: {exc}") from exc
    if (
        features.dtype != np.dtype("<f4")
        or list(features.shape) != raw["feature_shape"]
        or not bool(np.all(np.isfinite(features)))
    ):
        _fail("feature cache", "feature tensor contract differs")
    entries_raw = raw["entries"]
    if not isinstance(entries_raw, list):
        _fail("feature cache", "entry inventory is invalid")
    entries: list[FeatureCacheEntryV1] = []
    for item in entries_raw:
        if not isinstance(item, dict) or set(item) != {
            "camera_slot",
            "feature_row",
            "image_digest",
            "packet_id",
        }:
            _fail("feature cache", "entry fields differ")
        entries.append(FeatureCacheEntryV1(**item))
    if [item.feature_row for item in entries] != list(
        range(len(entries))
    ) or features.shape != (len(entries), 512):
        _fail("feature cache", "row inventory differs")
    frozen = np.frombuffer(features.tobytes(order="C"), dtype=np.dtype("<f4")).reshape(
        features.shape
    )
    return LoadedFeatureCacheV1(
        git_sha=raw["git_sha"],
        dataset_digest=raw["dataset_digest"],
        source_dataset_digest=raw["source_dataset_digest"],
        split_digest=raw["split_digest"],
        backbone_digest=raw["backbone_digest"],
        weight_content_digest=raw["weight_content_digest"],
        model_freeze_digest=raw["model_freeze_digest"],
        preprocessing_semantic=raw["preprocessing_semantic"],
        entries=tuple(entries),
        features=frozen,
        content_digest=raw["content_digest"],
    )


@dataclass(frozen=True, slots=True, eq=False)
class LoadedTeacherCacheV1:
    """Five-seed teacher outputs without privileged state or action arrays."""

    git_sha: str
    dataset_digest: str
    split_digest: str
    teacher_bundle_digest: str
    candidate_ids: tuple[str, ...]
    raw_logits: NDArray[np.float64]
    calibrated_probabilities: NDArray[np.float64]
    ensemble_probabilities: NDArray[np.float64]
    content_digest: str

    def training_probability(self, sample_index: int, *, is_training: bool) -> float:
        """Expose distillation targets only to the development training split."""
        if not is_training:
            _fail("teacher cache", "distillation access is training-only")
        return float(self.ensemble_probabilities[sample_index])


def save_teacher_cache(
    output_dir: Path,
    *,
    git_sha: str,
    dataset_digest: str,
    split_digest: str,
    teacher_bundle_digest: str,
    candidate_ids: tuple[str, ...],
    raw_logits: NDArray[Any],
    calibrated_probabilities: NDArray[Any],
    resume: bool = False,
) -> tuple[LoadedTeacherCacheV1, int]:
    """Transactionally publish deterministic five-seed teacher targets once."""
    count = len(candidate_ids)
    if (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or any(item not in "0123456789abcdef" for item in git_sha)
    ):
        _fail("teacher cache", "expected full lowercase Git SHA")
    raw = np.asarray(raw_logits, dtype=np.float64)
    calibrated = np.asarray(calibrated_probabilities, dtype=np.float64)
    if (
        raw.shape != (5, count)
        or calibrated.shape != (5, count)
        or not bool(np.all(np.isfinite(raw)))
        or not bool(np.all(np.isfinite(calibrated)))
        or not bool(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))
    ):
        _fail("teacher cache", "expected finite five-seed matrices")
    if len(set(candidate_ids)) != count:
        _fail("teacher cache", "candidate identities must be unique")
    destination = Path(output_dir).absolute()
    if destination.exists():
        if not resume:
            _fail("teacher cache", "output exists; use --resume")
        loaded = load_teacher_cache(destination)
        if (
            loaded.dataset_digest != dataset_digest
            or loaded.git_sha != git_sha
            or loaded.split_digest != split_digest
            or loaded.teacher_bundle_digest != teacher_bundle_digest
            or loaded.candidate_ids != candidate_ids
            or not np.array_equal(loaded.raw_logits, raw)
            or not np.array_equal(loaded.calibrated_probabilities, calibrated)
        ):
            _fail("teacher cache", "resume identity differs")
        return loaded, 0
    work = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    work.mkdir(parents=True)
    try:
        np.save(work / "raw_logits.npy", raw, allow_pickle=False)
        np.save(work / "calibrated_probabilities.npy", calibrated, allow_pickle=False)
        ensemble = np.mean(calibrated, axis=0, dtype=np.float64)
        np.save(work / "ensemble_probabilities.npy", ensemble, allow_pickle=False)
        body = {
            "calibrated_probabilities_digest": _sha256_file(
                work / "calibrated_probabilities.npy"
            ),
            "candidate_ids": list(candidate_ids),
            "dataset_digest": dataset_digest,
            "ensemble_probabilities_digest": _sha256_file(
                work / "ensemble_probabilities.npy"
            ),
            "format": TEACHER_CACHE_FORMAT,
            "git_sha": git_sha,
            "raw_logits_digest": _sha256_file(work / "raw_logits.npy"),
            "schema_version": "1.0",
            "seed_order": [0, 1, 2, 3, 4],
            "split_digest": split_digest,
            "teacher_bundle_digest": teacher_bundle_digest,
        }
        _write_json(
            work / "manifest.json",
            {**body, "content_digest": _digest(body, "M4BTeacherCacheV1")},
        )
        work.rename(destination)
    finally:
        if work.exists():
            shutil.rmtree(work)
    return load_teacher_cache(destination), count


def load_teacher_cache(output_dir: Path) -> LoadedTeacherCacheV1:
    """Strictly reload teacher targets and verify exact byte inventory."""
    root = Path(output_dir).absolute()
    names = {item.name for item in root.iterdir()} if root.is_dir() else set()
    expected_names = {
        "manifest.json",
        "raw_logits.npy",
        "calibrated_probabilities.npy",
        "ensemble_probabilities.npy",
    }
    if root.is_symlink() or names != expected_names:
        _fail("teacher cache", "file inventory differs")
    raw_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "calibrated_probabilities_digest",
        "candidate_ids",
        "content_digest",
        "dataset_digest",
        "ensemble_probabilities_digest",
        "format",
        "git_sha",
        "raw_logits_digest",
        "schema_version",
        "seed_order",
        "split_digest",
        "teacher_bundle_digest",
    }
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != required:
        _fail("teacher cache", "manifest field inventory differs")
    body = {
        key: value for key, value in raw_manifest.items() if key != "content_digest"
    }
    if (
        raw_manifest["format"] != TEACHER_CACHE_FORMAT
        or raw_manifest["schema_version"] != "1.0"
        or raw_manifest["seed_order"] != [0, 1, 2, 3, 4]
        or raw_manifest["content_digest"] != _digest(body, "M4BTeacherCacheV1")
    ):
        _fail("teacher cache", "manifest identity differs")
    files = {
        "raw_logits": "raw_logits_digest",
        "calibrated_probabilities": "calibrated_probabilities_digest",
        "ensemble_probabilities": "ensemble_probabilities_digest",
    }
    arrays: dict[str, NDArray[Any]] = {}
    for stem, field in files.items():
        path = root / f"{stem}.npy"
        if _sha256_file(path) != raw_manifest[field]:
            _fail("teacher cache", f"{stem} bytes changed")
        arrays[stem] = np.load(path, allow_pickle=False)
    ids = tuple(raw_manifest["candidate_ids"])
    count = len(ids)
    if (
        arrays["raw_logits"].shape != (5, count)
        or arrays["calibrated_probabilities"].shape != (5, count)
        or arrays["ensemble_probabilities"].shape != (count,)
        or any(
            value.dtype != np.dtype("<f8") or not bool(np.all(np.isfinite(value)))
            for value in arrays.values()
        )
        or not np.array_equal(
            np.mean(arrays["calibrated_probabilities"], axis=0, dtype=np.float64),
            arrays["ensemble_probabilities"],
        )
    ):
        _fail("teacher cache", "array contract differs")
    return LoadedTeacherCacheV1(
        git_sha=raw_manifest["git_sha"],
        dataset_digest=raw_manifest["dataset_digest"],
        split_digest=raw_manifest["split_digest"],
        teacher_bundle_digest=raw_manifest["teacher_bundle_digest"],
        candidate_ids=ids,
        raw_logits=cast(NDArray[np.float64], arrays["raw_logits"]),
        calibrated_probabilities=cast(
            NDArray[np.float64], arrays["calibrated_probabilities"]
        ),
        ensemble_probabilities=cast(
            NDArray[np.float64], arrays["ensemble_probabilities"]
        ),
        content_digest=raw_manifest["content_digest"],
    )


def verify_live_cache_equivalence(
    runtime: FrozenBackboneRuntime,
    images: NDArray[Any],
    cached_features: NDArray[Any],
) -> float:
    """Compare every live and cached component under the fixed exact contract."""
    live = runtime.extract(images)
    cached = np.asarray(cached_features, dtype=np.float32)
    if live.shape != cached.shape or not bool(np.all(np.isfinite(cached))):
        _fail("feature equivalence", "tensor inventory differs")
    maximum = float(np.max(np.abs(live.astype(np.float64) - cached.astype(np.float64))))
    if maximum > FEATURE_COMPARISON_TOLERANCE:
        _fail("feature equivalence", "fixed exact comparison failed")
    return maximum


__all__ = [
    "FEATURE_CACHE_FORMAT",
    "FEATURE_COMPARISON_TOLERANCE",
    "TEACHER_CACHE_FORMAT",
    "FeatureCacheEntryV1",
    "LoadedFeatureCacheV1",
    "LoadedTeacherCacheV1",
    "VisualCacheError",
    "extract_feature_cache",
    "load_feature_cache",
    "load_teacher_cache",
    "save_teacher_cache",
    "verify_live_cache_equivalence",
]
