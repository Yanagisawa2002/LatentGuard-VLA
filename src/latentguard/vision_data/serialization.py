"""Safe, lossless publication of authoritative multi-view RGB arrays.

The visual models deliberately contain references and digests rather than image
arrays.  This module owns the filesystem boundary: NPY bytes are written with
pickle disabled, every reference is relative and POSIX-normalized, and a bundle
is visible only after a complete staging directory has been renamed into place.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat as stat_module
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

VISUAL_BUNDLE_FORMAT = "latentguard-m4a-visual-dataset"
VISUAL_PACKET_FORMAT = "latentguard-m4a-visual-packet"
VISUAL_SERIALIZATION_VERSION = 1
VISUAL_MANIFEST_NAME = "manifest.json"
VISUAL_PACKET_MANIFEST_NAME = "packet.json"
RGB_SHAPE = (224, 224, 3)
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class VisualSerializationError(ValueError):
    """Raised when visual content is unsafe, incomplete, or content-drifted."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualSerializationError(f"{context}: {reason}")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _is_link_or_junction(path: Path) -> bool:
    """Detect symbolic links and Windows junction/reparse points cross-version."""

    candidate = Path(path)
    if candidate.is_symlink():
        return True
    is_junction = getattr(candidate, "is_junction", None)
    if callable(is_junction) and bool(is_junction()):
        return True
    try:
        attributes = getattr(candidate.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _require_plain_directory(path: Path, *, context: str) -> None:
    if _is_link_or_junction(path) or not path.is_dir():
        _fail(context, "expected regular non-link directory")


def _reject_link_ancestors(root: Path, path: Path, *, context: str) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if _is_link_or_junction(current):
            _fail(context, "path traverses a link or junction")


def sha256_file(path: Path) -> str:
    """Return the exact SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def validate_rgb_image(value: object, *, context: str = "RGB image") -> NDArray[Any]:
    """Return a detached canonical RGB array or reject it without repair."""

    if not isinstance(value, np.ndarray):
        _fail(context, "expected numpy.ndarray")
    if value.dtype != np.dtype("uint8") or value.shape != RGB_SHAPE:
        _fail(context, "expected uint8 [224, 224, 3]")
    if not value.flags.c_contiguous:
        # Contiguity is a serialization detail, not an image repair.  Copying
        # preserves every byte and provides one unambiguous C-order digest.
        value = np.ascontiguousarray(value)
    detached = np.array(value, copy=True, order="C", subok=False)
    detached.setflags(write=False)
    return detached


def compute_pixel_sha256(value: object) -> str:
    """Digest exact canonical C-order RGB pixel bytes."""

    image = validate_rgb_image(value)
    return _sha256_bytes(image.tobytes(order="C"))


def serialize_npy_image(value: object) -> bytes:
    """Serialize one RGB array as deterministic, pickle-free NPY bytes."""

    image = validate_rgb_image(value)
    stream = io.BytesIO()
    np.save(stream, image, allow_pickle=False)
    return stream.getvalue()


def compute_npy_sha256(value: object) -> str:
    """Digest the exact authoritative NPY file bytes for one RGB image."""

    return _sha256_bytes(serialize_npy_image(value))


@dataclass(frozen=True, slots=True)
class PreparedNpyImageV1:
    """Detached image bytes and both required exact digests."""

    image: NDArray[Any]
    pixel_sha256: str
    npy_sha256: str
    npy_bytes: bytes


def prepare_npy_image(value: object) -> PreparedNpyImageV1:
    """Validate and prepare an image once for model creation and publication."""

    image = validate_rgb_image(value)
    payload = serialize_npy_image(image)
    return PreparedNpyImageV1(
        image=image,
        pixel_sha256=_sha256_bytes(image.tobytes(order="C")),
        npy_sha256=_sha256_bytes(payload),
        npy_bytes=payload,
    )


def validate_image_reference(value: object, *, context: str = "image reference") -> str:
    """Require a canonical relative POSIX NPY reference without traversal."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail(context, "expected canonical relative POSIX path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix != ".npy"
    ):
        _fail(context, "must be a traversal-free relative .npy path")
    if pure.parts[0] not in {"images", "packets"}:
        _fail(context, "must stay under images/ or packets/")
    return value


def _safe_join(root: Path, reference: str, *, context: str) -> Path:
    relative = validate_image_reference(reference, context=context)
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        path.relative_to(root)
    except ValueError:
        _fail(context, "resolved path escapes bundle root")
    _reject_link_ancestors(root, path, context=context)
    return path


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("visual JSON", f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail("visual JSON", f"non-finite constant {value!r}")


def _json_bytes(value: Mapping[str, object]) -> bytes:
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
        raise VisualSerializationError("visual manifest is not JSON-safe") from exc


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected object")
    return cast(Mapping[str, object], value)


def _exact(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        _fail(context, "unexpected or missing fields")


def _model_mapping(value: object, context: str) -> dict[str, object]:
    method = getattr(value, "as_mapping", None)
    if not callable(method):
        method = getattr(value, "to_dict", None)
    if not callable(method):
        _fail(context, "model must provide as_mapping()")
    mapped = method()
    if not isinstance(mapped, Mapping):
        _fail(context, "model mapping must be an object")
    # Canonicalization also rejects runtime absolute paths and non-finite values.
    canonical = json.loads(canonical_json_bytes(mapped).decode("utf-8"))
    assert isinstance(canonical, dict)
    return cast(dict[str, object], canonical)


def _content_digest(value: object, context: str) -> str:
    digest = getattr(value, "content_digest", None)
    if not isinstance(digest, str):
        _fail(context, "model must expose content_digest")
    return digest


def _packet_views(packet: object) -> tuple[object, ...]:
    raw = getattr(packet, "views", None)
    if raw is None:
        raw = getattr(packet, "ordered_views", None)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        _fail("visual packet", "expected ordered views")
    views = tuple(raw)
    if len(views) != 3:
        _fail("visual packet", "expected exactly three ordered views")
    return views


def _view_reference(view: object) -> str:
    return validate_image_reference(
        getattr(view, "image_reference", None), context="visual view image_reference"
    )


def _view_digest(view: object, name: str) -> str:
    value = getattr(view, name, None)
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(f"visual view {name}", "expected lowercase SHA-256 digest")
    return value


def _images_for_packet(
    packet: object,
    images: Mapping[str, NDArray[Any]],
    *,
    allow_extra: bool = False,
) -> dict[str, PreparedNpyImageV1]:
    views = _packet_views(packet)
    by_camera = {
        cast(str, getattr(view, "camera_id", "")): _view_reference(view)
        for view in views
    }
    references = tuple(by_camera.values())
    if len(set(references)) != len(references):
        _fail("visual packet", "duplicate image references")
    result: dict[str, PreparedNpyImageV1] = {}
    for view in views:
        camera_id = getattr(view, "camera_id", None)
        reference = _view_reference(view)
        raw = images.get(reference)
        if raw is None and isinstance(camera_id, str):
            raw = images.get(camera_id)
        if raw is None:
            _fail(reference, "image bytes are missing")
        prepared = prepare_npy_image(raw)
        if prepared.pixel_sha256 != _view_digest(view, "pixel_sha256"):
            _fail(reference, "raw pixel digest differs from view record")
        if prepared.npy_sha256 != _view_digest(view, "npy_sha256"):
            _fail(reference, "NPY file digest differs from view record")
        dtype = getattr(view, "dtype", None)
        shape = tuple(getattr(view, "shape", ()))
        if dtype != "uint8" or shape != RGB_SHAPE:
            _fail(reference, "view record dtype or shape differs")
        result[reference] = prepared
    extra = set(images) - set(references) - set(by_camera)
    if extra and not allow_extra:
        _fail("visual packet", "unreferenced input images are unsupported")
    return result


def _image_inventory(
    prepared: Mapping[str, PreparedNpyImageV1],
) -> list[dict[str, object]]:
    return [
        {
            "dtype": "uint8",
            "npy_sha256": item.npy_sha256,
            "path": reference,
            "pixel_sha256": item.pixel_sha256,
            "shape": list(RGB_SHAPE),
        }
        for reference, item in sorted(prepared.items())
    ]


def _publish_staging(staging: Path, destination: Path) -> None:
    _require_plain_directory(staging, context="visual staging directory")
    _require_plain_directory(destination.parent, context="visual output parent")
    try:
        staging.rename(destination)
    except FileExistsError as exc:
        raise VisualSerializationError(
            "visual publication: destination appeared concurrently"
        ) from exc


def save_visual_packet(
    packet: object,
    images: Mapping[str, NDArray[Any]],
    output_dir: Path,
) -> Path:
    """Transactionally publish one complete three-view packet bundle."""

    destination = Path(output_dir)
    prepared = _images_for_packet(packet, images)
    expected_digest = _content_digest(packet, "visual packet")
    if destination.exists() or _is_link_or_junction(destination):
        loaded = load_visual_packet(destination)
        if _content_digest(loaded, "visual packet") != expected_digest:
            _fail("visual packet output", "contains a different packet")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_plain_directory(destination.parent, context="visual output parent")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        for reference, item in prepared.items():
            _write_new_bytes(
                _safe_join(staging, reference, context=reference), item.npy_bytes
            )
        manifest = {
            "format": VISUAL_PACKET_FORMAT,
            "image_inventory": _image_inventory(prepared),
            "packet": _model_mapping(packet, "visual packet"),
            "packet_content_digest": expected_digest,
            "serialization_version": VISUAL_SERIALIZATION_VERSION,
        }
        _write_new_bytes(staging / VISUAL_PACKET_MANIFEST_NAME, _json_bytes(manifest))
        _validate_packet_root(staging, packet)
        _publish_staging(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    loaded = load_visual_packet(destination)
    if _content_digest(loaded, "visual packet") != expected_digest:
        _fail("visual packet", "published packet failed exact reload")
    return destination


def _load_json(path: Path, *, context: str) -> Mapping[str, object]:
    if _is_link_or_junction(path) or not path.is_file() or path.stat().st_nlink != 1:
        _fail(context, "expected regular single-linked file")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except VisualSerializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualSerializationError(
            f"{context}: could not read safely: {exc}"
        ) from exc
    return _mapping(raw, context)


def _decode_packet(value: Mapping[str, object]) -> object:
    from latentguard.vision_data.packet import VisualObservationPacketV1

    decoder = getattr(VisualObservationPacketV1, "from_mapping", None)
    if not callable(decoder):
        decoder = getattr(VisualObservationPacketV1, "from_dict", None)
    if not callable(decoder):
        _fail("visual packet", "model must provide from_mapping()")
    return decoder(value)


def _verify_inventory(
    root: Path,
    raw_inventory: object,
    *,
    manifest_names: set[str],
    expected_npy_digests: Mapping[str, str],
) -> dict[str, NDArray[Any]]:
    _require_plain_directory(root, context="visual bundle root")
    if not isinstance(raw_inventory, list):
        _fail("image inventory", "expected list")
    images: dict[str, NDArray[Any]] = {}
    expected_files = set(manifest_names)
    for index, raw in enumerate(raw_inventory):
        item = _mapping(raw, f"image inventory[{index}]")
        _exact(
            item,
            {"dtype", "npy_sha256", "path", "pixel_sha256", "shape"},
            f"image inventory[{index}]",
        )
        reference = validate_image_reference(
            item["path"], context="image inventory path"
        )
        if reference in images:
            _fail("image inventory", "duplicate image reference")
        path = _safe_join(root, reference, context=reference)
        if (
            _is_link_or_junction(path)
            or not path.is_file()
            or path.stat().st_nlink != 1
        ):
            _fail(reference, "expected regular single-linked image")
        payload = path.read_bytes()
        actual_npy_digest = _sha256_bytes(payload)
        if actual_npy_digest != item["npy_sha256"]:
            _fail(reference, "NPY file digest mismatch")
        expected_npy_digest = expected_npy_digests.get(reference)
        if expected_npy_digest is None:
            _fail(reference, "image inventory is not referenced by a view")
        if actual_npy_digest != expected_npy_digest:
            _fail(reference, "authoritative NPY digest differs from view record")
        try:
            stream = io.BytesIO(payload)
            loaded = np.load(stream, allow_pickle=False)
            if stream.read(1):
                _fail(reference, "NPY contains trailing bytes")
        except VisualSerializationError:
            raise
        except (OSError, ValueError) as exc:
            raise VisualSerializationError(f"{reference}: invalid NPY: {exc}") from exc
        image = validate_rgb_image(loaded, context=reference)
        if item["dtype"] != "uint8" or item["shape"] != list(RGB_SHAPE):
            _fail(reference, "manifest dtype or shape differs")
        if compute_pixel_sha256(image) != item["pixel_sha256"]:
            _fail(reference, "raw pixel digest mismatch")
        images[reference] = image
        expected_files.add(reference)
    observed_files: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        for name in directory_names:
            child = current / name
            if _is_link_or_junction(child):
                _fail("visual bundle", "links and junctions are unsupported")
            if not stat_module.S_ISDIR(child.lstat().st_mode):
                _fail("visual bundle", "unsupported filesystem entry")
        for name in file_names:
            child = current / name
            if _is_link_or_junction(child):
                _fail("visual bundle", "links and junctions are unsupported")
            details = child.lstat()
            if not stat_module.S_ISREG(details.st_mode):
                _fail("visual bundle", "unsupported filesystem entry")
            if details.st_nlink != 1:
                _fail("visual bundle", "hard-linked files are unsupported")
            observed_files.add(child.relative_to(root).as_posix())
    if observed_files != expected_files:
        _fail("visual bundle", "file inventory differs from manifest")
    return images


def _validate_packet_root(root: Path, expected_packet: object | None = None) -> object:
    manifest = _load_json(root / VISUAL_PACKET_MANIFEST_NAME, context="packet manifest")
    _exact(
        manifest,
        {
            "format",
            "image_inventory",
            "packet",
            "packet_content_digest",
            "serialization_version",
        },
        "packet manifest",
    )
    if (
        manifest["format"] != VISUAL_PACKET_FORMAT
        or manifest["serialization_version"] != VISUAL_SERIALIZATION_VERSION
    ):
        _fail("packet manifest", "unsupported format or version")
    packet = _decode_packet(_mapping(manifest["packet"], "packet manifest.packet"))
    digest = _content_digest(packet, "visual packet")
    if digest != manifest["packet_content_digest"]:
        _fail("packet manifest", "packet content digest mismatch")
    if expected_packet is not None and digest != _content_digest(
        expected_packet, "expected visual packet"
    ):
        _fail("packet manifest", "packet differs from expected content")
    images = _verify_inventory(
        root,
        manifest["image_inventory"],
        manifest_names={VISUAL_PACKET_MANIFEST_NAME},
        expected_npy_digests={
            _view_reference(view): _view_digest(view, "npy_sha256")
            for view in _packet_views(packet)
        },
    )
    expected = _images_for_packet(packet, images)
    if set(expected) != set(images):
        _fail("packet manifest", "view and image inventories differ")
    return packet


def load_visual_packet(output_dir: Path) -> object:
    """Strictly reload one packet and validate every authoritative image byte."""

    root = Path(output_dir)
    if _is_link_or_junction(root) or not root.is_dir():
        _fail("visual packet root", "expected regular non-symlink directory")
    return _validate_packet_root(root)


def load_visual_image(bundle_root: Path, image_reference: str) -> NDArray[Any]:
    """Load one referenced authoritative RGB image with pickle disabled."""

    root = Path(bundle_root)
    _require_plain_directory(root, context="visual bundle root")
    path = _safe_join(root, image_reference, context="image reference")
    if _is_link_or_junction(path) or not path.is_file() or path.stat().st_nlink != 1:
        _fail("image reference", "expected regular single-linked image")
    with path.open("rb") as stream:
        try:
            value = np.load(stream, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise VisualSerializationError("image reference: invalid NPY") from exc
        if stream.read(1):
            _fail("image reference", "NPY contains trailing bytes")
    return validate_rgb_image(value, context="image reference")


def _dataset_packets(dataset: object) -> tuple[object, ...]:
    raw = getattr(dataset, "packets", None)
    if raw is None:
        raw = getattr(dataset, "packet_manifest", None)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        _fail("visual dataset", "expected packet inventory")
    packets = tuple(raw)
    if not packets:
        _fail("visual dataset", "packet inventory must not be empty")
    return packets


def _dataset_type(dataset: object) -> str:
    name = type(dataset).__name__
    if name == "VisualVerifierDevelopmentDatasetV1":
        return "development"
    if name == "VisualVerifierExternalDatasetV1":
        return "external"
    _fail("visual dataset", "unsupported dataset model")


def save_visual_dataset(
    dataset: object,
    images: Mapping[str, NDArray[Any]],
    output_dir: Path,
) -> Path:
    """Transactionally publish a complete visual dataset and exact inventory."""

    packets = _dataset_packets(dataset)
    expected_references: set[str] = set()
    prepared: dict[str, PreparedNpyImageV1] = {}
    for packet in packets:
        packet_images = _images_for_packet(packet, images, allow_extra=True)
        overlap = set(prepared).intersection(packet_images)
        if overlap:
            _fail("visual dataset", "duplicate image references across packets")
        prepared.update(packet_images)
        expected_references.update(packet_images)
    if set(images) != expected_references:
        # Dataset publication intentionally requires reference-keyed inputs so
        # camera IDs cannot collide across packets.
        _fail("visual dataset", "images must be keyed by exact image reference")
    destination = Path(output_dir)
    expected_digest = _content_digest(dataset, "visual dataset")
    if destination.exists() or _is_link_or_junction(destination):
        loaded = load_visual_dataset(destination)
        if _content_digest(loaded, "visual dataset") != expected_digest:
            _fail("visual dataset output", "contains a different dataset")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_plain_directory(destination.parent, context="visual output parent")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        for reference, item in prepared.items():
            _write_new_bytes(
                _safe_join(staging, reference, context=reference), item.npy_bytes
            )
        manifest = {
            "dataset": _model_mapping(dataset, "visual dataset"),
            "dataset_content_digest": expected_digest,
            "dataset_type": _dataset_type(dataset),
            "format": VISUAL_BUNDLE_FORMAT,
            "image_inventory": _image_inventory(prepared),
            "serialization_version": VISUAL_SERIALIZATION_VERSION,
        }
        _write_new_bytes(staging / VISUAL_MANIFEST_NAME, _json_bytes(manifest))
        _validate_dataset_root(staging, dataset)
        _publish_staging(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    loaded = load_visual_dataset(destination)
    if _content_digest(loaded, "visual dataset") != expected_digest:
        _fail("visual dataset", "published dataset failed exact reload")
    return destination


def _decode_dataset(kind: object, value: Mapping[str, object]) -> object:
    from latentguard.vision_data.models import (
        VisualVerifierDevelopmentDatasetV1,
        VisualVerifierExternalDatasetV1,
    )

    if kind == "development":
        return VisualVerifierDevelopmentDatasetV1.from_mapping(value)
    if kind == "external":
        return VisualVerifierExternalDatasetV1.from_mapping(value)
    _fail("visual dataset", "unsupported dataset type")


def _validate_dataset_root(
    root: Path, expected_dataset: object | None = None
) -> object:
    manifest = _load_json(root / VISUAL_MANIFEST_NAME, context="visual manifest")
    _exact(
        manifest,
        {
            "dataset",
            "dataset_content_digest",
            "dataset_type",
            "format",
            "image_inventory",
            "serialization_version",
        },
        "visual manifest",
    )
    if (
        manifest["format"] != VISUAL_BUNDLE_FORMAT
        or manifest["serialization_version"] != VISUAL_SERIALIZATION_VERSION
    ):
        _fail("visual manifest", "unsupported format or version")
    dataset = _decode_dataset(
        manifest["dataset_type"], _mapping(manifest["dataset"], "visual dataset")
    )
    digest = _content_digest(dataset, "visual dataset")
    if digest != manifest["dataset_content_digest"]:
        _fail("visual manifest", "dataset content digest mismatch")
    if expected_dataset is not None and digest != _content_digest(
        expected_dataset, "expected visual dataset"
    ):
        _fail("visual manifest", "dataset differs from expected content")
    references = tuple(
        _view_reference(view)
        for packet in _dataset_packets(dataset)
        for view in _packet_views(packet)
    )
    if len(references) != len(set(references)):
        _fail("visual manifest", "duplicate image references across packets")
    expected_npy_digests = {
        _view_reference(view): _view_digest(view, "npy_sha256")
        for packet in _dataset_packets(dataset)
        for view in _packet_views(packet)
    }
    images = _verify_inventory(
        root,
        manifest["image_inventory"],
        manifest_names={VISUAL_MANIFEST_NAME},
        expected_npy_digests=expected_npy_digests,
    )
    if set(references) != set(images):
        _fail("visual manifest", "packet and image inventories differ")
    for packet in _dataset_packets(dataset):
        _images_for_packet(packet, images, allow_extra=True)
    return dataset


def load_visual_dataset(output_dir: Path) -> object:
    """Strictly reload a visual dataset and verify all files and model digests."""

    root = Path(output_dir)
    if _is_link_or_junction(root) or not root.is_dir():
        _fail("visual dataset root", "expected regular non-symlink directory")
    return _validate_dataset_root(root)


__all__ = [
    "RGB_SHAPE",
    "VISUAL_BUNDLE_FORMAT",
    "VISUAL_MANIFEST_NAME",
    "VISUAL_PACKET_FORMAT",
    "VISUAL_PACKET_MANIFEST_NAME",
    "VISUAL_SERIALIZATION_VERSION",
    "PreparedNpyImageV1",
    "VisualSerializationError",
    "compute_npy_sha256",
    "compute_pixel_sha256",
    "load_visual_dataset",
    "load_visual_image",
    "load_visual_packet",
    "prepare_npy_image",
    "save_visual_dataset",
    "save_visual_packet",
    "serialize_npy_image",
    "sha256_file",
    "validate_image_reference",
    "validate_rgb_image",
]
