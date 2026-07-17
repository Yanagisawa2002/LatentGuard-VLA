"""Persistent, monotonic rendering ledger and crash-safe resume semantics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, cast

from latentguard.evaluation.security import (
    is_sanitized_operational_text,
    sanitize_operational_text,
)
from latentguard.replay.identity import canonical_json_bytes

RENDER_LEDGER_FILENAME = "render-ledger.json"
RENDER_LEDGER_SCHEMA_VERSION = "1.0"


class RenderLedgerError(ValueError):
    """Raised when render state, resume inputs, or persisted history conflicts."""


class RenderLedgerState(StrEnum):
    """Durable state for one packet-render attempt."""

    PENDING = "pending"
    RENDERING = "rendering"
    COMPLETE = "complete"
    INVALID = "invalid"
    EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True, slots=True)
class RenderLedgerEntryV1:
    """One immutable attempt record for one expected visual packet."""

    packet_id: str
    attempt_ordinal: int
    state: RenderLedgerState
    packet_content_digest: str | None = None
    image_inventory_digest: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    retry_eligible: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    schema_version: str = RENDER_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.packet_id, "render ledger packet_id")
        if type(self.attempt_ordinal) is not int or self.attempt_ordinal < 0:
            _fail("render ledger attempt", "expected non-negative integer")
        try:
            state = RenderLedgerState(self.state)
        except (TypeError, ValueError) as exc:
            raise RenderLedgerError("render ledger state: unsupported value") from exc
        if self.schema_version != RENDER_LEDGER_SCHEMA_VERSION:
            _fail("render ledger entry", "unsupported schema version")
        if state is RenderLedgerState.PENDING:
            if (
                any(
                    value is not None
                    for value in (
                        self.packet_content_digest,
                        self.image_inventory_digest,
                        self.error_type,
                        self.error_message,
                        self.started_at,
                        self.finished_at,
                    )
                )
                or self.retry_eligible
            ):
                _fail("pending render entry", "must not contain result or audit fields")
        elif state is RenderLedgerState.RENDERING:
            if (
                self.started_at is None
                or self.finished_at is not None
                or any(
                    value is not None
                    for value in (
                        self.packet_content_digest,
                        self.image_inventory_digest,
                        self.error_type,
                        self.error_message,
                    )
                )
                or self.retry_eligible
            ):
                _fail("rendering entry", "requires start and no finish")
        elif state is RenderLedgerState.COMPLETE:
            _digest(self.packet_content_digest, "complete packet digest")
            _digest(self.image_inventory_digest, "complete image inventory digest")
            if self.started_at is None or self.finished_at is None:
                _fail("complete render entry", "requires start and finish")
            if self.error_type is not None or self.error_message is not None:
                _fail("complete render entry", "cannot contain an error")
            if self.retry_eligible:
                _fail("complete render entry", "cannot be retryable")
        else:
            if self.started_at is None or self.finished_at is None:
                _fail("failed render entry", "requires start and finish")
            _text(self.error_type, "render error type")
            _text(self.error_message, "render error message")
            if not is_sanitized_operational_text(self.error_type) or not (
                is_sanitized_operational_text(self.error_message)
            ):
                _fail("failed render entry", "error evidence is not sanitized")
            if (
                self.packet_content_digest is not None
                or self.image_inventory_digest is not None
            ):
                _fail("failed render entry", "cannot publish packet digests")
            if state is RenderLedgerState.INVALID and self.retry_eligible:
                _fail("invalid render entry", "invalid packets are not retryable")
            if state is RenderLedgerState.EXECUTION_ERROR and not self.retry_eligible:
                _fail("execution error entry", "must preserve retry eligibility")
        object.__setattr__(self, "state", state)

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-native attempt fields."""

        return {
            "attempt_ordinal": self.attempt_ordinal,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "finished_at": self.finished_at,
            "image_inventory_digest": self.image_inventory_digest,
            "packet_content_digest": self.packet_content_digest,
            "packet_id": self.packet_id,
            "retry_eligible": self.retry_eligible,
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class RenderLedgerV1:
    """Complete immutable run contract plus append-only packet attempt history."""

    run_id: str
    source_collection: str
    source_identity_digest: str
    camera_rig_digest: str
    render_domain_configuration_digest: str
    visual_compatibility_identity: str
    packet_ids: tuple[str, ...]
    entries: tuple[RenderLedgerEntryV1, ...]
    schema_version: str = RENDER_LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.run_id, "render ledger run_id")
        if self.source_collection not in {"m3a_development", "m3c_external"}:
            _fail("render ledger source_collection", "unsupported collection")
        for name in (
            "source_identity_digest",
            "camera_rig_digest",
            "render_domain_configuration_digest",
            "visual_compatibility_identity",
        ):
            _digest(getattr(self, name), f"render ledger {name}")
        packet_ids = tuple(self.packet_ids)
        if not packet_ids or len(packet_ids) != len(set(packet_ids)):
            _fail("render ledger packet IDs", "expected unique non-empty inventory")
        for packet_id in packet_ids:
            _text(packet_id, "render ledger packet ID")
        entries = tuple(self.entries)
        keys = tuple((item.packet_id, item.attempt_ordinal) for item in entries)
        if len(keys) != len(set(keys)) or any(
            item.packet_id not in packet_ids for item in entries
        ):
            _fail("render ledger entries", "duplicate or unknown attempt")
        for packet_id in packet_ids:
            ordinals = sorted(
                item.attempt_ordinal for item in entries if item.packet_id == packet_id
            )
            if ordinals != list(range(len(ordinals))):
                _fail(packet_id, "attempt ordinals must be contiguous from zero")
            if not ordinals:
                _fail(packet_id, "missing initial pending attempt")
        if self.schema_version != RENDER_LEDGER_SCHEMA_VERSION:
            _fail("render ledger", "unsupported schema version")
        object.__setattr__(self, "packet_ids", packet_ids)
        object.__setattr__(self, "entries", entries)

    @property
    def content_digest(self) -> str:
        """Return exact persisted ledger identity including operational evidence."""

        return _content_digest(self._content_mapping())

    def _content_mapping(self) -> dict[str, object]:
        return {
            "camera_rig_digest": self.camera_rig_digest,
            "entries": [item.as_mapping() for item in self.entries],
            "packet_ids": list(self.packet_ids),
            "render_domain_configuration_digest": (
                self.render_domain_configuration_digest
            ),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "source_collection": self.source_collection,
            "source_identity_digest": self.source_identity_digest,
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }

    def as_mapping(self) -> dict[str, object]:
        """Return strict self-digesting JSON content."""

        return {**self._content_mapping(), "content_digest": self.content_digest}


@dataclass(frozen=True, slots=True)
class RenderResumePlanV1:
    """Work selected by a strict resume invocation."""

    pending_packet_ids: tuple[str, ...]
    recovered_packet_ids: tuple[str, ...]
    retried_packet_ids: tuple[str, ...]
    complete_packet_ids: tuple[str, ...]

    @property
    def work_packet_ids(self) -> tuple[str, ...]:
        """Return deterministic pending, recovered, and explicit retry work."""

        selected = set(
            (
                *self.pending_packet_ids,
                *self.recovered_packet_ids,
                *self.retried_packet_ids,
            )
        )
        return tuple(
            item
            for inventory in (
                self.pending_packet_ids,
                self.recovered_packet_ids,
                self.retried_packet_ids,
            )
            for item in inventory
            if item in selected
        )


@dataclass(frozen=True, slots=True)
class ZeroWorkRenderResumeV1:
    """Proof that resuming a complete render performed no work or mutation."""

    run_id: str
    packet_count: int
    ledger_entry_count: int
    ledger_content_digest: str
    rendered_packets: int = 0
    recovered_packets: int = 0
    retried_packets: int = 0
    zero_duplicate_work: bool = True


def _fail(context: str, reason: str) -> NoReturn:
    raise RenderLedgerError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase sha256 digest")
    return value


def _content_digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def _reject_duplicate_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("render ledger", f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> NoReturn:
    _fail("render ledger", f"non-finite JSON constant {value!r}")


def initialize_render_ledger(
    output_dir: Path,
    *,
    run_id: str,
    source_collection: str,
    source_identity_digest: str,
    camera_rig_digest: str,
    render_domain_configuration_digest: str,
    visual_compatibility_identity: str,
    packet_ids: Sequence[str],
) -> RenderLedgerV1:
    """Create a new all-pending render ledger in an absent output directory."""

    values = tuple(packet_ids)
    ledger = RenderLedgerV1(
        run_id=run_id,
        source_collection=source_collection,
        source_identity_digest=source_identity_digest,
        camera_rig_digest=camera_rig_digest,
        render_domain_configuration_digest=render_domain_configuration_digest,
        visual_compatibility_identity=visual_compatibility_identity,
        packet_ids=values,
        entries=tuple(
            RenderLedgerEntryV1(
                packet_id=packet_id,
                attempt_ordinal=0,
                state=RenderLedgerState.PENDING,
            )
            for packet_id in values
        ),
    )
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        _fail("render output", "must be absent for a new render")
    root.mkdir(parents=True, exist_ok=False)
    try:
        _write_ledger_new(ledger, root / RENDER_LEDGER_FILENAME)
    except BaseException:
        try:
            root.rmdir()
        except OSError:
            pass
        raise
    return load_render_ledger(root)


def _write_ledger_new(ledger: RenderLedgerV1, path: Path) -> None:
    payload = (
        json.dumps(ledger.as_mapping(), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _decode_entry(value: Mapping[str, object]) -> RenderLedgerEntryV1:
    expected = {
        "attempt_ordinal",
        "error_message",
        "error_type",
        "finished_at",
        "image_inventory_digest",
        "packet_content_digest",
        "packet_id",
        "retry_eligible",
        "schema_version",
        "started_at",
        "state",
    }
    if set(value) != expected:
        _fail("render ledger entry", "unexpected or missing fields")
    packet_id = value["packet_id"]
    ordinal = value["attempt_ordinal"]
    state = value["state"]
    retry = value["retry_eligible"]
    schema = value["schema_version"]
    if not isinstance(packet_id, str) or type(ordinal) is not int:
        _fail("render ledger entry", "invalid packet ID or attempt ordinal")
    if not isinstance(state, str) or type(retry) is not bool:
        _fail("render ledger entry", "invalid state or retry flag")
    if not isinstance(schema, str):
        _fail("render ledger entry", "invalid schema version")
    optional_text: dict[str, str | None] = {}
    for name in (
        "error_message",
        "error_type",
        "finished_at",
        "image_inventory_digest",
        "packet_content_digest",
        "started_at",
    ):
        observed = value[name]
        if observed is not None and not isinstance(observed, str):
            _fail("render ledger entry", f"invalid {name}")
        optional_text[name] = observed
    return RenderLedgerEntryV1(
        packet_id=packet_id,
        attempt_ordinal=ordinal,
        state=RenderLedgerState(state),
        packet_content_digest=optional_text["packet_content_digest"],
        image_inventory_digest=optional_text["image_inventory_digest"],
        error_type=optional_text["error_type"],
        error_message=optional_text["error_message"],
        retry_eligible=retry,
        started_at=optional_text["started_at"],
        finished_at=optional_text["finished_at"],
        schema_version=schema,
    )


def load_render_ledger(output_dir: Path) -> RenderLedgerV1:
    """Strictly load and self-validate a persistent render ledger."""

    from latentguard.vision_data.serialization import _is_link_or_junction

    root = Path(output_dir)
    path = root / RENDER_LEDGER_FILENAME
    if (
        _is_link_or_junction(root)
        or not root.is_dir()
        or _is_link_or_junction(path)
        or not path.is_file()
    ):
        _fail("render ledger", "expected regular bundle and manifest")
    if path.stat().st_nlink != 1:
        _fail("render ledger", "hard-linked manifest is unsupported")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonfinite_constant,
        )
    except RenderLedgerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RenderLedgerError(f"render ledger: could not read safely: {exc}") from exc
    if not isinstance(raw, Mapping):
        _fail("render ledger", "expected JSON object")
    expected = {
        "camera_rig_digest",
        "content_digest",
        "entries",
        "packet_ids",
        "render_domain_configuration_digest",
        "run_id",
        "schema_version",
        "source_collection",
        "source_identity_digest",
        "visual_compatibility_identity",
    }
    if (
        set(raw) != expected
        or not isinstance(raw["entries"], list)
        or not isinstance(raw["packet_ids"], list)
    ):
        _fail("render ledger", "unexpected or missing fields")
    ledger = RenderLedgerV1(
        run_id=cast(str, raw["run_id"]),
        source_collection=cast(str, raw["source_collection"]),
        source_identity_digest=cast(str, raw["source_identity_digest"]),
        camera_rig_digest=cast(str, raw["camera_rig_digest"]),
        render_domain_configuration_digest=cast(
            str, raw["render_domain_configuration_digest"]
        ),
        visual_compatibility_identity=cast(str, raw["visual_compatibility_identity"]),
        packet_ids=tuple(cast(list[str], raw["packet_ids"])),
        entries=tuple(
            _decode_entry(cast(Mapping[str, object], item))
            for item in cast(list[object], raw["entries"])
            if isinstance(item, Mapping)
        ),
        schema_version=cast(str, raw["schema_version"]),
    )
    if len(ledger.entries) != len(cast(list[object], raw["entries"])):
        _fail("render ledger", "invalid entry")
    if ledger.content_digest != raw["content_digest"]:
        _fail("render ledger", "content digest mismatch")
    return ledger


def update_render_ledger(
    previous: RenderLedgerV1, updated: RenderLedgerV1, output_dir: Path
) -> None:
    """Atomically persist one monotonic same-run ledger update."""

    _validate_update(previous, updated)
    current = load_render_ledger(output_dir)
    if current.content_digest != previous.content_digest:
        _fail("render ledger update", "persisted ledger changed concurrently")
    path = Path(output_dir) / RENDER_LEDGER_FILENAME
    payload = (
        json.dumps(updated.as_mapping(), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=".render-ledger.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_update(previous: RenderLedgerV1, updated: RenderLedgerV1) -> None:
    immutable = (
        "run_id",
        "source_collection",
        "source_identity_digest",
        "camera_rig_digest",
        "render_domain_configuration_digest",
        "visual_compatibility_identity",
        "packet_ids",
        "schema_version",
    )
    changed = [
        name for name in immutable if getattr(previous, name) != getattr(updated, name)
    ]
    if changed:
        _fail("render ledger update", "immutable fields changed: " + ", ".join(changed))
    old = {(item.packet_id, item.attempt_ordinal): item for item in previous.entries}
    new = {(item.packet_id, item.attempt_ordinal): item for item in updated.entries}
    if not set(old).issubset(new):
        _fail("render ledger update", "prior attempts cannot be removed")
    for key, prior in old.items():
        current = new[key]
        allowed = {
            RenderLedgerState.PENDING: {
                RenderLedgerState.PENDING,
                RenderLedgerState.RENDERING,
            },
            RenderLedgerState.RENDERING: {
                RenderLedgerState.RENDERING,
                RenderLedgerState.COMPLETE,
                RenderLedgerState.INVALID,
                RenderLedgerState.EXECUTION_ERROR,
            },
        }.get(prior.state, {prior.state})
        if current.state not in allowed:
            _fail("render ledger update", f"invalid transition for {key!r}")
        if current.state is prior.state and current != prior:
            _fail("render ledger update", "an existing state cannot mutate")
        if (
            prior.state is RenderLedgerState.RENDERING
            and current.started_at != prior.started_at
        ):
            _fail("render ledger update", "recovery must preserve start time")
    for key in set(new) - set(old):
        packet_id, ordinal = key
        prior_attempts = [
            item for item in previous.entries if item.packet_id == packet_id
        ]
        if (
            new[key].state is not RenderLedgerState.PENDING
            or ordinal != len(prior_attempts)
            or not prior_attempts
            or prior_attempts[-1].state is not RenderLedgerState.EXECUTION_ERROR
        ):
            _fail("render ledger retry", "new attempts require prior execution error")


def _latest(ledger: RenderLedgerV1, packet_id: str) -> RenderLedgerEntryV1:
    return max(
        (item for item in ledger.entries if item.packet_id == packet_id),
        key=lambda item: item.attempt_ordinal,
    )


def start_render_attempt(
    ledger: RenderLedgerV1, packet_id: str, *, started_at: str
) -> RenderLedgerV1:
    """Advance one pending packet attempt to rendering before simulator work."""

    current = _latest(ledger, packet_id)
    if current.state is not RenderLedgerState.PENDING:
        _fail(packet_id, "latest attempt is not pending")
    replacement = replace(
        current,
        state=RenderLedgerState.RENDERING,
        started_at=_text(started_at, "started_at"),
    )
    return _replace_entry(ledger, replacement)


def finish_render_attempt(
    ledger: RenderLedgerV1,
    packet_id: str,
    *,
    packet_content_digest: str,
    image_inventory_digest: str,
    finished_at: str,
) -> RenderLedgerV1:
    """Mark a rendering attempt complete only after strict packet reload."""

    current = _latest(ledger, packet_id)
    if current.state is not RenderLedgerState.RENDERING:
        _fail(packet_id, "latest attempt is not rendering")
    replacement = replace(
        current,
        state=RenderLedgerState.COMPLETE,
        packet_content_digest=_digest(packet_content_digest, "packet digest"),
        image_inventory_digest=_digest(
            image_inventory_digest, "image inventory digest"
        ),
        finished_at=_text(finished_at, "finished_at"),
    )
    return _replace_entry(ledger, replacement)


def fail_render_attempt(
    ledger: RenderLedgerV1,
    packet_id: str,
    *,
    invalid: bool,
    error_type: str,
    error_message: str,
    finished_at: str,
) -> RenderLedgerV1:
    """Preserve invalid-context or operational-error evidence without labels."""

    current = _latest(ledger, packet_id)
    if current.state is not RenderLedgerState.RENDERING:
        _fail(packet_id, "latest attempt is not rendering")
    state = RenderLedgerState.INVALID if invalid else RenderLedgerState.EXECUTION_ERROR
    replacement = replace(
        current,
        state=state,
        error_type=sanitize_operational_text(_text(error_type, "error_type")),
        error_message=sanitize_operational_text(_text(error_message, "error_message")),
        retry_eligible=not invalid,
        finished_at=_text(finished_at, "finished_at"),
    )
    return _replace_entry(ledger, replacement)


def _replace_entry(
    ledger: RenderLedgerV1, entry: RenderLedgerEntryV1
) -> RenderLedgerV1:
    key = (entry.packet_id, entry.attempt_ordinal)
    return replace(
        ledger,
        entries=tuple(
            entry if (item.packet_id, item.attempt_ordinal) == key else item
            for item in ledger.entries
        ),
    )


def plan_render_resume(
    ledger: RenderLedgerV1,
    *,
    retry_execution_errors: bool = False,
    selected_packet_ids: Sequence[str] | None = None,
) -> tuple[RenderLedgerV1, RenderResumePlanV1]:
    """Detect drift-free pending/recovery work and append explicit retries."""

    selected = (
        set(ledger.packet_ids)
        if selected_packet_ids is None
        else set(selected_packet_ids)
    )
    if selected_packet_ids is not None:
        materialized = tuple(selected_packet_ids)
        expected_order = tuple(item for item in ledger.packet_ids if item in selected)
        if (
            not materialized
            or len(materialized) != len(selected)
            or materialized != expected_order
        ):
            _fail("render resume selection", "must be an ordered packet-ID subset")
    updated = ledger
    retried: list[str] = []
    if retry_execution_errors:
        additions = []
        for packet_id in ledger.packet_ids:
            current = _latest(ledger, packet_id)
            if (
                packet_id in selected
                and current.state is RenderLedgerState.EXECUTION_ERROR
            ):
                additions.append(
                    RenderLedgerEntryV1(
                        packet_id=packet_id,
                        attempt_ordinal=current.attempt_ordinal + 1,
                        state=RenderLedgerState.PENDING,
                    )
                )
                retried.append(packet_id)
        if additions:
            updated = replace(ledger, entries=(*ledger.entries, *additions))
    pending: list[str] = []
    recovered: list[str] = []
    complete: list[str] = []
    for packet_id in updated.packet_ids:
        state = _latest(updated, packet_id).state
        if state is RenderLedgerState.PENDING and packet_id not in retried:
            pending.append(packet_id)
        elif state is RenderLedgerState.RENDERING:
            recovered.append(packet_id)
        elif state is RenderLedgerState.COMPLETE:
            complete.append(packet_id)
    return updated, RenderResumePlanV1(
        pending_packet_ids=tuple(pending),
        recovered_packet_ids=tuple(recovered),
        retried_packet_ids=tuple(retried),
        complete_packet_ids=tuple(complete),
    )


def verify_zero_work_render_resume(
    before: RenderLedgerV1,
    after: RenderLedgerV1,
    plan: RenderResumePlanV1,
) -> ZeroWorkRenderResumeV1:
    """Require a completed resume to preserve every byte of ledger content."""

    if plan.work_packet_ids or before.content_digest != after.content_digest:
        _fail("render resume", "completed resume performed work or changed content")
    if len(plan.complete_packet_ids) != len(before.packet_ids):
        _fail("render resume", "ledger is not complete")
    return ZeroWorkRenderResumeV1(
        run_id=before.run_id,
        packet_count=len(before.packet_ids),
        ledger_entry_count=len(before.entries),
        ledger_content_digest=before.content_digest,
    )


__all__ = [
    "RENDER_LEDGER_FILENAME",
    "RENDER_LEDGER_SCHEMA_VERSION",
    "RenderLedgerEntryV1",
    "RenderLedgerError",
    "RenderLedgerState",
    "RenderLedgerV1",
    "RenderResumePlanV1",
    "ZeroWorkRenderResumeV1",
    "fail_render_attempt",
    "finish_render_attempt",
    "initialize_render_ledger",
    "load_render_ledger",
    "plan_render_resume",
    "start_render_attempt",
    "update_render_ledger",
    "verify_zero_work_render_resume",
]
