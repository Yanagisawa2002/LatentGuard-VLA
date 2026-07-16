"""Strict, immutable, sanitized JSON envelopes for compact M3B reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import canonical_json_bytes, canonical_json_value

REPORT_ENVELOPE_SCHEMA_VERSION = "1.0"
_REPORT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_v[0-9]+$")
_FORBIDDEN_SECRET_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "ssh_key",
        "access_token",
    }
)


class ReportValidationError(ValueError):
    """Raised when compact report content is malformed, unsafe, or changed."""


def _fail(context: str, reason: str) -> NoReturn:
    raise ReportValidationError(f"{context}: {reason}")


def _reject_json_constant(value: str) -> NoReturn:
    _fail("JSON", f"unsupported non-finite constant {value!r}")


def _reject_duplicate_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate object field {key!r}")
        result[key] = value
    return result


def _sanitize_keys(value: object, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = key.lower()
            if lowered in _FORBIDDEN_SECRET_KEYS:
                _fail(context, f"forbidden secret-bearing field {key!r}")
            _sanitize_keys(item, f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for index, item in enumerate(value):
            _sanitize_keys(item, f"{context}[{index}]")


def _digest_payload(value: object, *, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class StrictReportV1:
    """One path-independent report payload with a recomputable content digest."""

    report_type: str
    payload: Mapping[str, object]
    schema_version: str = REPORT_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach, sanitize, and freeze the JSON-native payload."""

        if _REPORT_TYPE_PATTERN.fullmatch(self.report_type) is None:
            _fail("report_type", "expected a versioned snake-case report type")
        if self.schema_version != REPORT_ENVELOPE_SCHEMA_VERSION:
            _fail("schema_version", "unsupported report envelope schema")
        try:
            canonical = canonical_json_value(
                self.payload,
                context=f"StrictReportV1.{self.report_type}.payload",
                reject_runtime_paths=True,
            )
        except ReplayValidationError as exc:
            raise ReportValidationError(str(exc)) from exc
        if not isinstance(canonical, dict):
            _fail("payload", "expected a JSON object")
        _sanitize_keys(canonical, "payload")
        object.__setattr__(self, "payload", MappingProxyType(canonical))

    @property
    def content_digest(self) -> str:
        """Return the path-independent report identity."""

        return _digest_payload(self._payload(), context="StrictReportV1")

    def _payload(self) -> dict[str, object]:
        return {
            "payload": dict(self.payload),
            "report_type": self.report_type,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a strict self-digesting JSON representation."""

        return {**self._payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StrictReportV1:
        """Load an envelope and reject fields or content-digest tampering."""

        expected = {"content_digest", "payload", "report_type", "schema_version"}
        if set(value) != expected:
            _fail("StrictReportV1", "unexpected or missing fields")
        report_type = value["report_type"]
        schema_version = value["schema_version"]
        payload = value["payload"]
        digest = value["content_digest"]
        if not isinstance(report_type, str) or not isinstance(schema_version, str):
            _fail("StrictReportV1", "report_type and schema_version must be text")
        if not isinstance(payload, Mapping):
            _fail("StrictReportV1.payload", "expected an object")
        if not isinstance(digest, str):
            _fail("StrictReportV1.content_digest", "expected text")
        result = cls(
            report_type=report_type,
            payload=payload,
            schema_version=schema_version,
        )
        if result.content_digest != digest:
            _fail("StrictReportV1.content_digest", "report content changed")
        return result


def _serialized_bytes(report: StrictReportV1) -> bytes:
    try:
        return (
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ReportValidationError("report could not be serialized") from exc


def load_strict_report(
    path: Path,
    *,
    expected_report_type: str | None = None,
    expected_content_digest: str | None = None,
) -> StrictReportV1:
    """Load a strict report with duplicate-field and non-finite rejection."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail("path", "expected a regular non-symlink report file")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportValidationError(f"could not read strict report: {exc}") from exc
    if not isinstance(value, Mapping):
        _fail("report", "expected a JSON object")
    report = StrictReportV1.from_dict(value)
    if expected_report_type is not None and report.report_type != expected_report_type:
        _fail("report_type", "does not match the expected report type")
    if (
        expected_content_digest is not None
        and report.content_digest != expected_content_digest
    ):
        _fail("content_digest", "does not match the expected report identity")
    return report


def save_strict_report(report: StrictReportV1, path: Path) -> Path:
    """Publish a report immutably; identical repeated publication is idempotent."""

    if not isinstance(report, StrictReportV1):
        _fail("report", "expected StrictReportV1")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _serialized_bytes(report)
    if destination.exists():
        existing = load_strict_report(
            destination,
            expected_report_type=report.report_type,
            expected_content_digest=report.content_digest,
        )
        if _serialized_bytes(existing) != content:
            _fail("path", "completed report path contains different bytes")
        return destination
    descriptor, staging_text = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    staging = Path(staging_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError:
            existing = load_strict_report(
                destination,
                expected_report_type=report.report_type,
                expected_content_digest=report.content_digest,
            )
            if _serialized_bytes(existing) != content:
                _fail("path", "concurrent report publication conflicts")
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    staging.unlink(missing_ok=True)
    return destination


__all__ = [
    "REPORT_ENVELOPE_SCHEMA_VERSION",
    "ReportValidationError",
    "StrictReportV1",
    "load_strict_report",
    "save_strict_report",
]
