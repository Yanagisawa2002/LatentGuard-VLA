"""Fail-closed file identity validation for external model assets."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class IdentityValidationError(ValueError):
    """Raised when an external asset is missing or content-mismatched."""


@dataclass(frozen=True)
class FileIdentity:
    """Expected identity of one file beneath a bound snapshot root."""

    relative_path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.relative_path or Path(self.relative_path).is_absolute():
            raise IdentityValidationError(
                "relative_path must be a non-empty relative path"
            )
        if self.bytes < 0:
            raise IdentityValidationError("bytes must be non-negative")
        if len(self.sha256) != 64:
            raise IdentityValidationError(
                "sha256 must contain 64 hexadecimal characters"
            )
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise IdentityValidationError("sha256 must be hexadecimal") from exc


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest for a local file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_file_identities(
    root: Path,
    identities: Iterable[FileIdentity],
) -> tuple[Path, ...]:
    """Validate every declared file and return its resolved path in input order."""

    resolved_root = root.resolve(strict=True)
    validated: list[Path] = []
    for identity in identities:
        path = (resolved_root / identity.relative_path).resolve()
        if not path.is_relative_to(resolved_root):
            raise IdentityValidationError(
                f"asset path escapes snapshot root: {identity.relative_path}"
            )
        if not path.is_file():
            raise IdentityValidationError(
                f"required asset is missing: {identity.relative_path}"
            )
        actual_bytes = path.stat().st_size
        if actual_bytes != identity.bytes:
            raise IdentityValidationError(
                f"size mismatch for {identity.relative_path}: "
                f"expected {identity.bytes}, observed {actual_bytes}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != identity.sha256.lower():
            raise IdentityValidationError(
                f"sha256 mismatch for {identity.relative_path}: "
                f"expected {identity.sha256.lower()}, observed {actual_sha256}"
            )
        validated.append(path)
    return tuple(validated)


def validate_snapshot_manifest(
    manifest: dict[str, Any],
    snapshot_roots: dict[str, Path],
) -> tuple[Path, ...]:
    """Validate every model-manifest asset against its declared snapshot root."""

    if manifest.get("schema_version") != "latentguard.lg_r0.base_model_manifest.v1":
        raise IdentityValidationError("unsupported base-model manifest schema")
    if manifest.get("status") != "pass":
        raise IdentityValidationError("base-model manifest is not accepted")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise IdentityValidationError("base-model manifest has no assets")

    grouped: dict[str, list[FileIdentity]] = {}
    for item in assets:
        if not isinstance(item, dict):
            raise IdentityValidationError("base-model asset entry must be an object")
        repository = item.get("repository_or_model_id")
        if not isinstance(repository, str) or repository not in snapshot_roots:
            raise IdentityValidationError(
                f"base-model asset has an unbound repository: {repository!r}"
            )
        grouped.setdefault(repository, []).append(
            FileIdentity(
                relative_path=str(item.get("relative_path", "")),
                bytes=int(item.get("bytes", -1)),
                sha256=str(item.get("sha256", "")),
            )
        )

    if set(grouped) != set(snapshot_roots):
        missing = sorted(set(snapshot_roots) - set(grouped))
        raise IdentityValidationError(
            f"base-model manifest omits snapshot repositories: {missing}"
        )
    validated: list[Path] = []
    for repository in sorted(grouped):
        validated.extend(
            validate_file_identities(
                snapshot_roots[repository],
                grouped[repository],
            )
        )
    return tuple(validated)
