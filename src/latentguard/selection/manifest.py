"""Strict persistence envelope for immutable blind-selection manifests.

The manifest model deliberately excludes its audit timestamp from semantic
selection identity.  This module adds a second, strict envelope digest over the
entire serialized manifest so timestamp or envelope tampering is still detected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.models import (
    BlindSelectionManifestV1,
    SelectionModelError,
    SelectorDecisionV1,
)

BLIND_SELECTION_MANIFEST_ARTIFACT_TYPE = "latentguard_blind_selection_manifest_v1"
BLIND_SELECTION_MANIFEST_FILENAME = "blind-selection-manifest.json"
BLIND_SELECTION_ENVELOPE_SCHEMA_VERSION = "1.0"
MAX_BLIND_SELECTION_MANIFEST_BYTES = 32 * 1024 * 1024

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class BlindSelectionManifestError(ValueError):
    """Raised when a blind manifest envelope or persistence gate is invalid."""


def _fail(context: str, reason: str) -> NoReturn:
    raise BlindSelectionManifestError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")
    return value


def _content_digest(value: object, *, context: str) -> str:
    try:
        encoded = canonical_json_bytes(value, context=context)
    except ReplayValidationError as exc:
        raise BlindSelectionManifestError(str(exc)) from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class BlindSelectionManifestEnvelopeV1:
    """A semantic manifest plus a strict digest that also protects audit fields."""

    manifest: BlindSelectionManifestV1
    manifest_envelope_digest: str
    artifact_type: str = BLIND_SELECTION_MANIFEST_ARTIFACT_TYPE
    schema_version: str = BLIND_SELECTION_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject semantic, audit, type, or strict-envelope drift."""

        if not isinstance(self.manifest, BlindSelectionManifestV1):
            _fail(
                "BlindSelectionManifestEnvelopeV1.manifest",
                "expected BlindSelectionManifestV1",
            )
        if self.artifact_type != BLIND_SELECTION_MANIFEST_ARTIFACT_TYPE:
            _fail(
                "BlindSelectionManifestEnvelopeV1.artifact_type",
                "unsupported artifact type",
            )
        if self.schema_version != BLIND_SELECTION_ENVELOPE_SCHEMA_VERSION:
            _fail(
                "BlindSelectionManifestEnvelopeV1.schema_version",
                "unsupported version",
            )
        _digest(
            self.manifest_envelope_digest,
            "BlindSelectionManifestEnvelopeV1.manifest_envelope_digest",
        )
        if self.manifest_envelope_digest != self.expected_envelope_digest:
            _fail(
                "BlindSelectionManifestEnvelopeV1.manifest_envelope_digest",
                "strict manifest content changed",
            )

    @classmethod
    def create(
        cls, manifest: BlindSelectionManifestV1
    ) -> BlindSelectionManifestEnvelopeV1:
        """Create a strict audit envelope for one finalized Stage-A manifest."""

        if not isinstance(manifest, BlindSelectionManifestV1):
            _fail("manifest", "expected BlindSelectionManifestV1")
        payload = {
            "artifact_type": BLIND_SELECTION_MANIFEST_ARTIFACT_TYPE,
            "manifest": manifest.as_mapping(),
            "schema_version": BLIND_SELECTION_ENVELOPE_SCHEMA_VERSION,
        }
        return cls(
            manifest=manifest,
            manifest_envelope_digest=_content_digest(
                payload,
                context="BlindSelectionManifestEnvelopeV1",
            ),
        )

    @property
    def semantic_digest(self) -> str:
        """Return the timestamp-independent selection identity."""

        return self.manifest.content_digest

    @property
    def content_digest(self) -> str:
        """Expose the timestamp-independent digest to the replay view protocol."""

        return self.manifest.content_digest

    @property
    def candidate_pool_digest(self) -> str:
        """Expose the pre-outcome candidate-pool identity to replay gating."""

        return self.manifest.candidate_pool_digest

    @property
    def selections(self) -> tuple[SelectorDecisionV1, ...]:
        """Expose immutable blind decisions without discarding the envelope."""

        return self.manifest.selections

    @property
    def envelope_digest(self) -> str:
        """Return the strict digest, including the audit timestamp."""

        return self.manifest_envelope_digest

    @property
    def expected_envelope_digest(self) -> str:
        """Recompute the strict digest from the current complete manifest."""

        return _content_digest(
            {
                "artifact_type": self.artifact_type,
                "manifest": self.manifest.as_mapping(),
                "schema_version": self.schema_version,
            },
            context="BlindSelectionManifestEnvelopeV1",
        )

    def as_mapping(self) -> dict[str, object]:
        """Return the exact persisted envelope."""

        return {
            "artifact_type": self.artifact_type,
            "manifest": self.manifest.as_mapping(),
            "manifest_envelope_digest": self.manifest_envelope_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> BlindSelectionManifestEnvelopeV1:
        """Decode exact fields and validate both semantic and strict digests."""

        expected = {
            "artifact_type",
            "manifest",
            "manifest_envelope_digest",
            "schema_version",
        }
        if set(value) != expected:
            _fail(
                "BlindSelectionManifestEnvelopeV1",
                "unexpected or missing fields",
            )
        raw_manifest = value["manifest"]
        if not isinstance(raw_manifest, Mapping):
            _fail("BlindSelectionManifestEnvelopeV1.manifest", "expected object")
        try:
            manifest = BlindSelectionManifestV1.from_mapping(
                cast(Mapping[str, object], raw_manifest)
            )
        except SelectionModelError as exc:
            raise BlindSelectionManifestError(str(exc)) from exc
        return cls(
            manifest=manifest,
            manifest_envelope_digest=cast(str, value["manifest_envelope_digest"]),
            artifact_type=cast(str, value["artifact_type"]),
            schema_version=cast(str, value["schema_version"]),
        )


def initialize_blind_selection_output_root(path: Path) -> Path:
    """Create or validate an empty, non-linked Stage-A-only output directory."""

    root = Path(path)
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            _fail("blind selection output root", "expected a non-symlink directory")
        try:
            if next(root.iterdir(), None) is not None:
                _fail(
                    "blind selection output root",
                    "must be empty before Stage A",
                )
        except OSError as exc:
            raise BlindSelectionManifestError(
                f"blind selection output root: cannot inspect safely: {exc}"
            ) from exc
        return root
    try:
        root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise BlindSelectionManifestError(
            f"blind selection output root: cannot create safely: {exc}"
        ) from exc
    if root.is_symlink() or not root.is_dir():
        _fail("blind selection output root", "created path is not a safe directory")
    return root


def save_blind_selection_manifest(
    path: Path,
    value: BlindSelectionManifestEnvelopeV1 | BlindSelectionManifestV1,
) -> Path:
    """Persist a new strict manifest without overwriting any existing artifact."""

    destination = Path(path)
    envelope = (
        value
        if isinstance(value, BlindSelectionManifestEnvelopeV1)
        else BlindSelectionManifestEnvelopeV1.create(value)
    )
    if destination.exists() or destination.is_symlink():
        _fail("blind selection manifest", "refusing to overwrite an existing path")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        _fail("blind selection manifest", "parent must be a non-symlink directory")
    payload = json.dumps(
        envelope.as_mapping(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, destination)
        Path(temporary_name).unlink()
        temporary_name = None
    except OSError as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise BlindSelectionManifestError(
            f"blind selection manifest: atomic create failed: {exc}"
        ) from exc
    return destination


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("blind selection manifest JSON", f"duplicate field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail("blind selection manifest JSON", f"non-finite constant {value!r}")


def load_blind_selection_manifest(path: Path) -> BlindSelectionManifestEnvelopeV1:
    """Load a regular, single-linked manifest and verify every digest."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail("blind selection manifest", "expected regular non-symlink file")
    try:
        stat = source.stat()
        if stat.st_nlink != 1:
            _fail("blind selection manifest", "hard-linked files are unsupported")
        if stat.st_size <= 0 or stat.st_size > MAX_BLIND_SELECTION_MANIFEST_BYTES:
            _fail("blind selection manifest", "file size is outside safe bounds")
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except BlindSelectionManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlindSelectionManifestError(
            f"blind selection manifest: could not read safely: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        _fail("blind selection manifest", "expected JSON object")
    return BlindSelectionManifestEnvelopeV1.from_mapping(
        cast(Mapping[str, object], raw)
    )


def save_manifest_in_empty_root(
    output_root: Path,
    manifest: BlindSelectionManifestV1,
) -> tuple[Path, BlindSelectionManifestEnvelopeV1]:
    """Finalize an independent empty Stage-A root with exactly one manifest."""

    root = initialize_blind_selection_output_root(output_root)
    envelope = BlindSelectionManifestEnvelopeV1.create(manifest)
    path = save_blind_selection_manifest(
        root / BLIND_SELECTION_MANIFEST_FILENAME,
        envelope,
    )
    return path, envelope


__all__ = [
    "BLIND_SELECTION_ENVELOPE_SCHEMA_VERSION",
    "BLIND_SELECTION_MANIFEST_ARTIFACT_TYPE",
    "BLIND_SELECTION_MANIFEST_FILENAME",
    "BlindSelectionManifestEnvelopeV1",
    "BlindSelectionManifestError",
    "initialize_blind_selection_output_root",
    "load_blind_selection_manifest",
    "save_blind_selection_manifest",
    "save_manifest_in_empty_root",
]
