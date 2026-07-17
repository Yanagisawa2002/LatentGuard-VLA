"""Simulator-independent, ledger-driven visual packet rendering pipeline.

The callback boundary is deliberately project-owned: this module plans and
persists content-bound work, while simulator integrations only produce one
validated packet plus its three RGB arrays for a requested job.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast, runtime_checkable

from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.vision_data.domains import (
    RENDER_SEED_DERIVATION,
    assigned_render_domain_ids,
)
from latentguard.vision_data.identity import (
    compute_visual_packet_identifier_from_fields,
)
from latentguard.vision_data.models import SourceCollection, VisualDatasetSplit
from latentguard.vision_data.packet import (
    VISUAL_PACKET_SCHEMA_VERSION,
    VisualObservationPacketV1,
)
from latentguard.vision_data.rendering import (
    RenderLedgerEntryV1,
    RenderLedgerState,
    RenderLedgerV1,
    ZeroWorkRenderResumeV1,
    fail_render_attempt,
    finish_render_attempt,
    initialize_render_ledger,
    load_render_ledger,
    plan_render_resume,
    start_render_attempt,
    update_render_ledger,
    verify_zero_work_render_resume,
)
from latentguard.vision_data.run_manifest import (
    compute_selected_packet_inventory_digest,
)
from latentguard.vision_data.serialization import (
    VisualSerializationError,
    _is_link_or_junction,
    load_visual_packet,
    save_visual_packet,
)

if TYPE_CHECKING:
    from latentguard.vision_data.run_manifest import M4AOperationalRunManifestV1

RENDER_JOB_MANIFEST_FILENAME = "render-jobs.json"
RENDER_JOB_MANIFEST_FORMAT = "latentguard-m4a-render-jobs"
RENDER_JOB_MANIFEST_SERIALIZATION_VERSION = 1
RENDER_JOB_SCHEMA_VERSION = "1.0"
RENDER_JOB_INVENTORY_SCHEMA_VERSION = "1.0"
_PACKET_ID = re.compile(r"vop-sha256-[0-9a-f]{64}\Z")
_PACKET_STAGING = re.compile(r"\.(vop-sha256-[0-9a-f]{64})\.staging-[a-z0-9_]{8}\Z")


class VisualRenderPipelineError(RuntimeError):
    """Raised when persistent render planning or execution is inconsistent."""


class InvalidVisualRenderPacketError(VisualRenderPipelineError):
    """Raised when a callback result cannot be accepted as a visual packet."""


class VisualRenderExecutionError(VisualRenderPipelineError):
    """Raised after preserving one retryable operational render failure."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualRenderPipelineError(f"{context}: {reason}")


def _invalid(context: str, reason: str) -> NoReturn:
    raise InvalidVisualRenderPacketError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail(context, "expected canonical non-empty text")
    return value


def _logical_reference(value: object, context: str) -> str:
    text = _text(value, context)
    if "\\" in text or text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        _fail(context, "runtime absolute paths are unsupported")
    pure = PurePosixPath(text)
    if any(part in {"", ".", ".."} for part in pure.parts):
        _fail(context, "path traversal is unsupported")
    return text


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase SHA-256 digest")
    return value


def _uint64(value: object, context: str) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        _fail(context, "expected uint64")
    return value


def _uint32(value: object, context: str) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        _fail(context, "expected uint32")
    return value


def _content_digest(value: object, *, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def derive_render_seed(
    anchor_id: str,
    domain_id: str,
    base_seed: int,
    semantic: str = RENDER_SEED_DERIVATION,
) -> int:
    """Derive one deterministic uint32 render seed without global RNG state."""

    anchor = _text(anchor_id, "render seed anchor ID")
    domain = _text(domain_id, "render seed domain ID")
    base = _uint64(base_seed, "base render seed")
    if semantic != RENDER_SEED_DERIVATION:
        _fail("render seed derivation", "unsupported semantic")
    encoded = canonical_json_bytes(
        {
            "anchor_id": anchor,
            "base_seed": base,
            "domain_id": domain,
            "semantic": semantic,
        },
        context="M4ARenderSeedDerivationV1",
    )
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


@dataclass(frozen=True, slots=True)
class VisualPacketJobV1:
    """Immutable pre-render metadata for exactly one semantic packet."""

    job_ordinal: int
    packet_id: str
    source_collection: SourceCollection
    source_trajectory_id: str
    source_reset_seed: int
    anchor_id: str
    split: VisualDatasetSplit
    split_group_id: str
    state_reference_id: str
    expected_state_digest: str
    verifier_state_semantic: str
    verifier_state_digest: str
    camera_rig_id: str
    camera_rig_digest: str
    render_domain_id: str
    render_domain_digest: str
    render_seed: int
    seed_derivation: str
    camera_configuration_digests: tuple[str, str, str]
    visual_compatibility_identity: str
    pickcube_compatibility_identity: str
    renderer_semantic_version: str
    packet_schema_version: str = VISUAL_PACKET_SCHEMA_VERSION
    schema_version: str = RENDER_JOB_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.job_ordinal) is not int or self.job_ordinal < 0:
            _fail("render job ordinal", "expected non-negative integer")
        try:
            collection = SourceCollection(self.source_collection)
            split = VisualDatasetSplit(self.split)
        except (TypeError, ValueError) as exc:
            raise VisualRenderPipelineError(
                "render job: invalid source collection or split"
            ) from exc
        object.__setattr__(self, "source_collection", collection)
        object.__setattr__(self, "split", split)
        if self.render_domain_id not in assigned_render_domain_ids(collection, split):
            _fail("render job domain", "not authorized for collection and split")
        for name in (
            "source_trajectory_id",
            "anchor_id",
            "split_group_id",
            "verifier_state_semantic",
            "camera_rig_id",
            "render_domain_id",
            "renderer_semantic_version",
        ):
            _text(getattr(self, name), f"render job {name}")
        _logical_reference(self.state_reference_id, "render job state reference")
        _uint64(self.source_reset_seed, "render job source reset seed")
        _uint32(self.render_seed, "render job render seed")
        if self.seed_derivation != RENDER_SEED_DERIVATION:
            _fail("render job seed derivation", "unsupported semantic")
        for name in (
            "expected_state_digest",
            "verifier_state_digest",
            "camera_rig_digest",
            "render_domain_digest",
            "visual_compatibility_identity",
            "pickcube_compatibility_identity",
        ):
            _digest(getattr(self, name), f"render job {name}")
        camera_digests = tuple(self.camera_configuration_digests)
        if len(camera_digests) != 3 or len(set(camera_digests)) != 3:
            _fail("render job cameras", "expected three unique ordered digests")
        for digest in camera_digests:
            _digest(digest, "render job camera configuration digest")
        object.__setattr__(self, "camera_configuration_digests", camera_digests)
        if self.packet_schema_version != VISUAL_PACKET_SCHEMA_VERSION:
            _fail("render job packet schema", "unsupported version")
        if self.schema_version != RENDER_JOB_SCHEMA_VERSION:
            _fail("render job schema", "unsupported version")
        expected_packet_id = compute_visual_packet_identifier_from_fields(
            source_trajectory_id=self.source_trajectory_id,
            anchor_id=self.anchor_id,
            expected_state_digest=self.expected_state_digest,
            camera_rig_digest=self.camera_rig_digest,
            render_domain_digest=self.render_domain_digest,
            render_seed=self.render_seed,
            camera_configuration_digests=camera_digests,
            visual_compatibility_identity=self.visual_compatibility_identity,
            schema_version=self.packet_schema_version,
        )
        if (
            self.packet_id != expected_packet_id
            or _PACKET_ID.fullmatch(self.packet_id) is None
        ):
            _fail("render job packet ID", "differs from derived semantic identity")

    @classmethod
    def create(
        cls,
        *,
        job_ordinal: int,
        source_collection: SourceCollection,
        source_trajectory_id: str,
        source_reset_seed: int,
        anchor_id: str,
        split: VisualDatasetSplit,
        split_group_id: str,
        state_reference_id: str,
        expected_state_digest: str,
        verifier_state_semantic: str,
        verifier_state_digest: str,
        camera_rig_id: str,
        camera_rig_digest: str,
        render_domain_id: str,
        render_domain_digest: str,
        base_render_seed: int,
        camera_configuration_digests: tuple[str, str, str],
        visual_compatibility_identity: str,
        pickcube_compatibility_identity: str,
        renderer_semantic_version: str,
        seed_derivation: str = RENDER_SEED_DERIVATION,
    ) -> VisualPacketJobV1:
        """Create a job with derived render seed and semantic packet ID."""

        render_seed = derive_render_seed(
            anchor_id, render_domain_id, base_render_seed, seed_derivation
        )
        packet_id = compute_visual_packet_identifier_from_fields(
            source_trajectory_id=source_trajectory_id,
            anchor_id=anchor_id,
            expected_state_digest=expected_state_digest,
            camera_rig_digest=camera_rig_digest,
            render_domain_digest=render_domain_digest,
            render_seed=render_seed,
            camera_configuration_digests=camera_configuration_digests,
            visual_compatibility_identity=visual_compatibility_identity,
            schema_version=VISUAL_PACKET_SCHEMA_VERSION,
        )
        return cls(
            job_ordinal=job_ordinal,
            packet_id=packet_id,
            source_collection=source_collection,
            source_trajectory_id=source_trajectory_id,
            source_reset_seed=source_reset_seed,
            anchor_id=anchor_id,
            split=split,
            split_group_id=split_group_id,
            state_reference_id=state_reference_id,
            expected_state_digest=expected_state_digest,
            verifier_state_semantic=verifier_state_semantic,
            verifier_state_digest=verifier_state_digest,
            camera_rig_id=camera_rig_id,
            camera_rig_digest=camera_rig_digest,
            render_domain_id=render_domain_id,
            render_domain_digest=render_domain_digest,
            render_seed=render_seed,
            seed_derivation=seed_derivation,
            camera_configuration_digests=camera_configuration_digests,
            visual_compatibility_identity=visual_compatibility_identity,
            pickcube_compatibility_identity=pickcube_compatibility_identity,
            renderer_semantic_version=renderer_semantic_version,
        )

    @property
    def content_digest(self) -> str:
        """Return exact immutable job metadata identity."""

        return _content_digest(self.as_mapping(), context="M4AVisualPacketJobV1")

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-native job fields."""

        return {
            "anchor_id": self.anchor_id,
            "camera_configuration_digests": list(self.camera_configuration_digests),
            "camera_rig_digest": self.camera_rig_digest,
            "camera_rig_id": self.camera_rig_id,
            "expected_state_digest": self.expected_state_digest,
            "job_ordinal": self.job_ordinal,
            "packet_id": self.packet_id,
            "packet_schema_version": self.packet_schema_version,
            "pickcube_compatibility_identity": self.pickcube_compatibility_identity,
            "render_domain_digest": self.render_domain_digest,
            "render_domain_id": self.render_domain_id,
            "render_seed": self.render_seed,
            "renderer_semantic_version": self.renderer_semantic_version,
            "schema_version": self.schema_version,
            "seed_derivation": self.seed_derivation,
            "source_collection": self.source_collection.value,
            "source_reset_seed": self.source_reset_seed,
            "source_trajectory_id": self.source_trajectory_id,
            "split": self.split.value,
            "split_group_id": self.split_group_id,
            "state_reference_id": self.state_reference_id,
            "verifier_state_digest": self.verifier_state_digest,
            "verifier_state_semantic": self.verifier_state_semantic,
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }

    @classmethod
    def from_mapping(cls, value: object) -> VisualPacketJobV1:
        """Decode one exact-field persistent render job."""

        item = _mapping(value, "render job")
        expected = set(cls.__dataclass_fields__)
        if set(item) != expected:
            _fail("render job", "unexpected or missing fields")
        camera_values = _list(item["camera_configuration_digests"], "render cameras")
        if len(camera_values) != 3:
            _fail("render cameras", "expected exactly three values")
        try:
            collection = SourceCollection(_text(item["source_collection"], "source"))
            split = VisualDatasetSplit(_text(item["split"], "split"))
        except ValueError as exc:
            raise VisualRenderPipelineError("render job: invalid enum value") from exc
        return cls(
            job_ordinal=_integer(item["job_ordinal"], "render job ordinal"),
            packet_id=_text(item["packet_id"], "render job packet ID"),
            source_collection=collection,
            source_trajectory_id=_text(
                item["source_trajectory_id"], "render job trajectory"
            ),
            source_reset_seed=_integer(
                item["source_reset_seed"], "render job source reset seed"
            ),
            anchor_id=_text(item["anchor_id"], "render job anchor"),
            split=split,
            split_group_id=_text(item["split_group_id"], "render job split group"),
            state_reference_id=_text(
                item["state_reference_id"], "render job state reference"
            ),
            expected_state_digest=_text(
                item["expected_state_digest"], "render job state digest"
            ),
            verifier_state_semantic=_text(
                item["verifier_state_semantic"], "render job verifier semantic"
            ),
            verifier_state_digest=_text(
                item["verifier_state_digest"], "render job verifier digest"
            ),
            camera_rig_id=_text(item["camera_rig_id"], "render job rig ID"),
            camera_rig_digest=_text(item["camera_rig_digest"], "render job rig digest"),
            render_domain_id=_text(item["render_domain_id"], "render job domain ID"),
            render_domain_digest=_text(
                item["render_domain_digest"], "render job domain digest"
            ),
            render_seed=_integer(item["render_seed"], "render job seed"),
            seed_derivation=_text(
                item["seed_derivation"], "render job seed derivation"
            ),
            camera_configuration_digests=cast(
                tuple[str, str, str],
                tuple(_text(entry, "camera digest") for entry in camera_values),
            ),
            visual_compatibility_identity=_text(
                item["visual_compatibility_identity"], "visual compatibility"
            ),
            pickcube_compatibility_identity=_text(
                item["pickcube_compatibility_identity"], "PickCube compatibility"
            ),
            renderer_semantic_version=_text(
                item["renderer_semantic_version"], "renderer semantic"
            ),
            packet_schema_version=_text(item["packet_schema_version"], "packet schema"),
            schema_version=_text(item["schema_version"], "render job schema"),
        )


@dataclass(frozen=True, slots=True)
class VisualRenderJobInventoryV1:
    """Self-digesting deterministic job inventory bound beside the ledger."""

    source_collection: SourceCollection
    source_identity_digest: str
    camera_rig_digest: str
    render_domain_configuration_digest: str
    visual_compatibility_identity: str
    base_render_seed: int
    seed_derivation: str
    jobs: tuple[VisualPacketJobV1, ...]
    schema_version: str = RENDER_JOB_INVENTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            collection = SourceCollection(self.source_collection)
        except (TypeError, ValueError) as exc:
            raise VisualRenderPipelineError(
                "render job inventory: invalid source collection"
            ) from exc
        object.__setattr__(self, "source_collection", collection)
        for name in (
            "source_identity_digest",
            "camera_rig_digest",
            "render_domain_configuration_digest",
            "visual_compatibility_identity",
        ):
            _digest(getattr(self, name), f"render job inventory {name}")
        base_seed = _uint64(self.base_render_seed, "render job inventory base seed")
        if self.seed_derivation != RENDER_SEED_DERIVATION:
            _fail("render job inventory", "unsupported seed derivation")
        jobs = tuple(self.jobs)
        if not jobs:
            _fail("render job inventory", "must contain at least one job")
        if tuple(job.job_ordinal for job in jobs) != tuple(range(len(jobs))):
            _fail("render job inventory", "job ordinals or ordering differ")
        if len({job.packet_id for job in jobs}) != len(jobs):
            _fail("render job inventory", "packet IDs must be unique")
        for job in jobs:
            if (
                job.source_collection is not collection
                or job.camera_rig_digest != self.camera_rig_digest
                or job.visual_compatibility_identity
                != self.visual_compatibility_identity
                or job.seed_derivation != self.seed_derivation
                or job.render_seed
                != derive_render_seed(
                    job.anchor_id,
                    job.render_domain_id,
                    base_seed,
                    self.seed_derivation,
                )
            ):
                _fail(job.packet_id, "job differs from immutable run contract")
        if self.schema_version != RENDER_JOB_INVENTORY_SCHEMA_VERSION:
            _fail("render job inventory", "unsupported schema version")
        object.__setattr__(self, "jobs", jobs)

    @classmethod
    def create(
        cls,
        *,
        source_identity_digest: str,
        camera_rig_digest: str,
        render_domain_configuration_digest: str,
        visual_compatibility_identity: str,
        base_render_seed: int,
        jobs: Sequence[VisualPacketJobV1],
        seed_derivation: str = RENDER_SEED_DERIVATION,
    ) -> VisualRenderJobInventoryV1:
        """Create a deterministic inventory from caller-ordered jobs."""

        values = tuple(jobs)
        if not values:
            _fail("render job inventory", "must contain at least one job")
        return cls(
            source_collection=values[0].source_collection,
            source_identity_digest=source_identity_digest,
            camera_rig_digest=camera_rig_digest,
            render_domain_configuration_digest=render_domain_configuration_digest,
            visual_compatibility_identity=visual_compatibility_identity,
            base_render_seed=base_render_seed,
            seed_derivation=seed_derivation,
            jobs=values,
        )

    @property
    def packet_ids(self) -> tuple[str, ...]:
        """Return the exact deterministic ledger packet inventory."""

        return tuple(job.packet_id for job in self.jobs)

    def _content_mapping(self) -> dict[str, object]:
        return {
            "base_render_seed": self.base_render_seed,
            "camera_rig_digest": self.camera_rig_digest,
            "jobs": [job.as_mapping() for job in self.jobs],
            "render_domain_configuration_digest": (
                self.render_domain_configuration_digest
            ),
            "schema_version": self.schema_version,
            "seed_derivation": self.seed_derivation,
            "source_collection": self.source_collection.value,
            "source_identity_digest": self.source_identity_digest,
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete immutable pipeline run identity."""

        return _content_digest(
            self._content_mapping(), context="M4AVisualRenderJobInventoryV1"
        )

    def as_mapping(self) -> dict[str, object]:
        """Return strict self-digesting persistent inventory fields."""

        return {**self._content_mapping(), "content_digest": self.content_digest}

    @classmethod
    def from_mapping(cls, value: object) -> VisualRenderJobInventoryV1:
        """Decode and self-validate an exact-field render inventory."""

        item = _mapping(value, "render job inventory")
        expected = {
            "base_render_seed",
            "camera_rig_digest",
            "content_digest",
            "jobs",
            "render_domain_configuration_digest",
            "schema_version",
            "seed_derivation",
            "source_collection",
            "source_identity_digest",
            "visual_compatibility_identity",
        }
        if set(item) != expected:
            _fail("render job inventory", "unexpected or missing fields")
        try:
            collection = SourceCollection(
                _text(item["source_collection"], "render inventory source")
            )
        except ValueError as exc:
            raise VisualRenderPipelineError(
                "render job inventory: invalid source collection"
            ) from exc
        inventory = cls(
            source_collection=collection,
            source_identity_digest=_text(
                item["source_identity_digest"], "render inventory source digest"
            ),
            camera_rig_digest=_text(
                item["camera_rig_digest"], "render inventory rig digest"
            ),
            render_domain_configuration_digest=_text(
                item["render_domain_configuration_digest"],
                "render inventory domain configuration digest",
            ),
            visual_compatibility_identity=_text(
                item["visual_compatibility_identity"],
                "render inventory visual compatibility",
            ),
            base_render_seed=_integer(
                item["base_render_seed"], "render inventory base seed"
            ),
            seed_derivation=_text(
                item["seed_derivation"], "render inventory seed derivation"
            ),
            jobs=tuple(
                VisualPacketJobV1.from_mapping(job)
                for job in _list(item["jobs"], "render inventory jobs")
            ),
            schema_version=_text(item["schema_version"], "render inventory schema"),
        )
        if item["content_digest"] != inventory.content_digest:
            _fail("render job inventory", "content digest mismatch")
        return inventory


@runtime_checkable
class PreparedVisualPacket(Protocol):
    """Core packet and exact reference-keyed arrays returned by a renderer."""

    @property
    def packet(self) -> VisualObservationPacketV1:
        """Return the complete validated project-owned packet model."""
        ...

    @property
    def images(self) -> Mapping[str, NDArray[Any]]:
        """Return exact RGB arrays keyed by packet image reference."""
        ...


VisualPacketRenderCallback = Callable[[VisualPacketJobV1], PreparedVisualPacket]


@dataclass(frozen=True, slots=True)
class VisualRenderPipelineResultV1:
    """Completed packet stores and resume facts for one pipeline invocation."""

    inventory_content_digest: str
    ledger: RenderLedgerV1
    packet_bundle_roots: tuple[Path, ...]
    packets: tuple[VisualObservationPacketV1, ...]
    rendered_packet_ids: tuple[str, ...]
    recovered_packet_ids: tuple[str, ...]
    retried_packet_ids: tuple[str, ...]
    zero_work_proof: ZeroWorkRenderResumeV1 | None = None

    @property
    def planned_packet_count(self) -> int:
        """Return the immutable full job inventory size."""

        return len(self.ledger.packet_ids)

    @property
    def packet_count(self) -> int:
        """Return the complete planned packet count."""

        return len(self.packets)

    @property
    def image_count(self) -> int:
        """Return the authoritative image count across completed packets."""

        return sum(len(packet.views) for packet in self.packets)

    @property
    def complete(self) -> bool:
        """Return whether every planned packet is durably complete."""

        return self.packet_count == self.planned_packet_count


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        _fail(context, "expected JSON array")
    return cast(list[object], value)


def _integer(value: object, context: str) -> int:
    if type(value) is not int:
        _fail(context, "expected integer")
    return value


def _reject_duplicate_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("render jobs", f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail("render jobs", f"non-finite JSON constant {value!r}")


def save_render_job_inventory(
    inventory: VisualRenderJobInventoryV1, output_dir: Path
) -> Path:
    """Persist one strict inventory once, or verify identical existing content."""

    root = Path(output_dir)
    if _is_link_or_junction(root) or not root.is_dir():
        _fail("render jobs root", "expected regular non-link directory")
    path = root / RENDER_JOB_MANIFEST_FILENAME
    if path.exists() or _is_link_or_junction(path):
        observed = load_render_job_inventory(root)
        if observed.content_digest != inventory.content_digest:
            _fail("render jobs", "existing inventory differs")
        return path
    payload = {
        "format": RENDER_JOB_MANIFEST_FORMAT,
        "inventory": inventory.as_mapping(),
        "serialization_version": RENDER_JOB_MANIFEST_SERIALIZATION_VERSION,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def load_render_job_inventory(output_dir: Path) -> VisualRenderJobInventoryV1:
    """Strictly reload the self-digesting persistent job inventory."""

    root = Path(output_dir)
    path = root / RENDER_JOB_MANIFEST_FILENAME
    if (
        _is_link_or_junction(root)
        or not root.is_dir()
        or _is_link_or_junction(path)
        or not path.is_file()
        or path.stat().st_nlink != 1
    ):
        _fail("render jobs", "expected regular single-linked manifest")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonfinite,
        )
    except VisualRenderPipelineError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualRenderPipelineError(
            f"render jobs: could not read safely: {exc}"
        ) from exc
    manifest = _mapping(raw, "render jobs")
    expected = {"format", "inventory", "serialization_version"}
    if set(manifest) != expected:
        _fail("render jobs", "unexpected or missing fields")
    if (
        manifest["format"] != RENDER_JOB_MANIFEST_FORMAT
        or manifest["serialization_version"]
        != RENDER_JOB_MANIFEST_SERIALIZATION_VERSION
    ):
        _fail("render jobs", "unsupported format or version")
    return VisualRenderJobInventoryV1.from_mapping(manifest["inventory"])


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _latest_entry(ledger: RenderLedgerV1, packet_id: str) -> RenderLedgerEntryV1:
    return max(
        (entry for entry in ledger.entries if entry.packet_id == packet_id),
        key=lambda entry: entry.attempt_ordinal,
    )


def _packet_bundle_root(output_dir: Path, packet_id: str) -> Path:
    if _PACKET_ID.fullmatch(packet_id) is None:
        _fail("packet bundle", "invalid packet ID")
    return Path(output_dir) / "packets" / packet_id


def _packet_image_inventory_digest(packet: VisualObservationPacketV1) -> str:
    return _content_digest(
        {
            "images": [
                {
                    "camera_id": view.camera_id,
                    "image_reference": view.image_reference,
                    "npy_sha256": view.npy_sha256,
                    "pixel_sha256": view.pixel_sha256,
                }
                for view in packet.views
            ],
            "packet_id": packet.packet_id,
            "schema_version": "m4a_packet_image_inventory_v1",
        },
        context="M4APacketImageInventoryV1",
    )


def _validate_packet_for_job(
    packet: VisualObservationPacketV1, job: VisualPacketJobV1
) -> None:
    comparisons = {
        "packet_id": job.packet_id,
        "source_collection": job.source_collection,
        "source_trajectory_id": job.source_trajectory_id,
        "anchor_id": job.anchor_id,
        "split": job.split,
        "split_group_id": job.split_group_id,
        "state_reference_id": job.state_reference_id,
        "expected_state_digest": job.expected_state_digest,
        "verifier_state_semantic": job.verifier_state_semantic,
        "verifier_state_digest": job.verifier_state_digest,
        "camera_rig_id": job.camera_rig_id,
        "camera_rig_digest": job.camera_rig_digest,
        "render_domain_id": job.render_domain_id,
        "render_domain_digest": job.render_domain_digest,
        "render_seed": job.render_seed,
        "visual_compatibility_identity": job.visual_compatibility_identity,
        "pickcube_compatibility_identity": job.pickcube_compatibility_identity,
        "renderer_semantic_version": job.renderer_semantic_version,
        "schema_version": job.packet_schema_version,
    }
    changed = [
        name
        for name, expected in comparisons.items()
        if getattr(packet, name) != expected
    ]
    observed_cameras = tuple(view.camera_configuration_digest for view in packet.views)
    if observed_cameras != job.camera_configuration_digests:
        changed.append("camera_configuration_digests")
    if changed:
        _invalid(job.packet_id, "rendered packet differs in " + ", ".join(changed))


def _load_packet_for_job(
    bundle_root: Path,
    job: VisualPacketJobV1,
    complete_entry: RenderLedgerEntryV1 | None = None,
) -> VisualObservationPacketV1:
    loaded = load_visual_packet(bundle_root)
    if not isinstance(loaded, VisualObservationPacketV1):
        _invalid(job.packet_id, "packet loader returned an unsupported model")
    _validate_packet_for_job(loaded, job)
    if complete_entry is not None:
        if (
            complete_entry.packet_content_digest != loaded.content_digest
            or complete_entry.image_inventory_digest
            != _packet_image_inventory_digest(loaded)
        ):
            _invalid(job.packet_id, "complete ledger digests differ from packet store")
    return loaded


def _validate_ledger_contract(
    ledger: RenderLedgerV1,
    inventory: VisualRenderJobInventoryV1,
    run_id: str,
) -> None:
    expected = {
        "run_id": run_id,
        "source_collection": inventory.source_collection.value,
        "source_identity_digest": inventory.source_identity_digest,
        "camera_rig_digest": inventory.camera_rig_digest,
        "render_domain_configuration_digest": (
            inventory.render_domain_configuration_digest
        ),
        "visual_compatibility_identity": inventory.visual_compatibility_identity,
        "packet_ids": inventory.packet_ids,
    }
    changed = [
        name for name, value in expected.items() if getattr(ledger, name) != value
    ]
    if changed:
        _fail("render resume", "ledger contract drift: " + ", ".join(changed))


def _validate_operational_manifest_binding(
    manifest: M4AOperationalRunManifestV1,
    inventory: VisualRenderJobInventoryV1,
    run_id: str,
    selected_packet_ids: tuple[str, ...],
) -> None:
    """Bind operational evidence one-way to this exact render plan."""

    from latentguard.vision_data.run_manifest import M4AOperationalRunManifestV1

    if not isinstance(manifest, M4AOperationalRunManifestV1):
        _fail("render run manifest", "unexpected manifest model")
    expected = {
        "run_id": run_id,
        "camera_rig_digest": inventory.camera_rig_digest,
        "render_domain_configuration_digest": (
            inventory.render_domain_configuration_digest
        ),
        "visual_compatibility_identity": inventory.visual_compatibility_identity,
        "render_seed": inventory.base_render_seed,
        "render_job_inventory_digest": inventory.content_digest,
        "selected_packet_inventory_digest": (
            compute_selected_packet_inventory_digest(selected_packet_ids)
        ),
    }
    changed = [
        name for name, value in expected.items() if getattr(manifest, name) != value
    ]
    if manifest.source_identity.source_collection is not inventory.source_collection:
        changed.append("source_identity")
    if (
        manifest.source_identity.canonical_source_identity_digest
        != inventory.source_identity_digest
    ):
        changed.append("source_identity_digest")
    if manifest.source_identity.is_visual_probe:
        changed.append("source_variant")
    if changed:
        _fail(
            "render run manifest",
            "render-plan binding drift: " + ", ".join(sorted(set(changed))),
        )


def _initialize_pipeline_root(
    output_dir: Path,
    inventory: VisualRenderJobInventoryV1,
    run_id: str,
    operational_manifest: M4AOperationalRunManifestV1 | None,
) -> RenderLedgerV1:
    destination = Path(output_dir)
    if destination.exists() or _is_link_or_junction(destination):
        _fail("render output", "must be absent for a new run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_junction(destination.parent):
        _fail("render output", "parent cannot be a link or junction")
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.pipeline-staging-", dir=destination.parent
        )
    )
    staging = temporary_root / "run"
    try:
        initialize_render_ledger(
            staging,
            run_id=run_id,
            source_collection=inventory.source_collection.value,
            source_identity_digest=inventory.source_identity_digest,
            camera_rig_digest=inventory.camera_rig_digest,
            render_domain_configuration_digest=(
                inventory.render_domain_configuration_digest
            ),
            visual_compatibility_identity=inventory.visual_compatibility_identity,
            packet_ids=inventory.packet_ids,
        )
        save_render_job_inventory(inventory, staging)
        if operational_manifest is not None:
            from latentguard.vision_data.run_manifest import (
                M4A_RUN_MANIFEST_FILENAME,
                load_m4a_run_manifest,
                save_m4a_run_manifest,
            )

            save_m4a_run_manifest(
                operational_manifest,
                staging / M4A_RUN_MANIFEST_FILENAME,
            )
            if (
                load_m4a_run_manifest(staging / M4A_RUN_MANIFEST_FILENAME)
                != operational_manifest
            ):
                _fail("render run manifest", "staging reload changed content")
        staged_inventory = load_render_job_inventory(staging)
        if staged_inventory.content_digest != inventory.content_digest:
            _fail("render job inventory", "staging reload changed content")
        staged_ledger = load_render_ledger(staging)
        _validate_ledger_contract(staged_ledger, inventory, run_id)
        staging.rename(destination)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return load_render_ledger(destination)


def _load_complete_packets(
    output_dir: Path,
    inventory: VisualRenderJobInventoryV1,
    ledger: RenderLedgerV1,
) -> tuple[tuple[Path, ...], tuple[VisualObservationPacketV1, ...]]:
    roots: list[Path] = []
    packets: list[VisualObservationPacketV1] = []
    for job in inventory.jobs:
        entry = _latest_entry(ledger, job.packet_id)
        if entry.state is not RenderLedgerState.COMPLETE:
            continue
        root = _packet_bundle_root(output_dir, job.packet_id)
        packets.append(_load_packet_for_job(root, job, entry))
        roots.append(root)
    return tuple(roots), tuple(packets)


def _validate_packet_store_inventory(
    output_dir: Path,
    inventory: VisualRenderJobInventoryV1,
    ledger: RenderLedgerV1,
) -> None:
    """Reject packet-store entries not justified by the latest ledger states."""

    root = Path(output_dir)
    packets_root = root / "packets"
    jobs = {job.packet_id: job for job in inventory.jobs}
    complete = {
        packet_id
        for packet_id in ledger.packet_ids
        if _latest_entry(ledger, packet_id).state is RenderLedgerState.COMPLETE
    }
    if not packets_root.exists() and not _is_link_or_junction(packets_root):
        if complete:
            _fail("packet store inventory", "complete packet directory is missing")
        return
    if _is_link_or_junction(packets_root) or not packets_root.is_dir():
        _fail("packet store inventory", "packets root must be a regular directory")
    observed: set[str] = set()
    recoverable_rendering: set[str] = set()
    try:
        children = tuple(packets_root.iterdir())
    except OSError as exc:
        raise VisualRenderPipelineError(
            f"packet store inventory: could not enumerate safely: {exc}"
        ) from exc
    for child in children:
        if _is_link_or_junction(child):
            _fail("packet store inventory", "links and junctions are unsupported")
        try:
            mode = child.lstat().st_mode
        except OSError as exc:
            raise VisualRenderPipelineError(
                f"packet store inventory: could not inspect safely: {exc}"
            ) from exc
        if not stat_module.S_ISDIR(mode):
            _fail("packet store inventory", "unknown files are unsupported")
        packet_id = child.name
        job = jobs.get(packet_id)
        if job is None or _PACKET_ID.fullmatch(packet_id) is None:
            _fail("packet store inventory", "unknown packet directory")
        state = _latest_entry(ledger, packet_id).state
        if state is RenderLedgerState.COMPLETE:
            _load_packet_for_job(child, job, _latest_entry(ledger, packet_id))
        elif state is RenderLedgerState.RENDERING:
            _load_packet_for_job(child, job)
            recoverable_rendering.add(packet_id)
        else:
            _fail(
                "packet store inventory",
                f"packet directory is not allowed for ledger state {state.value}",
            )
        observed.add(packet_id)
    expected = complete | recoverable_rendering
    if observed != expected:
        _fail("packet store inventory", "directory inventory differs from ledger")


def _require_safe_staging_tree(path: Path, packets_root: Path) -> None:
    """Reject links or special files before deleting owned crash staging."""

    if _is_link_or_junction(path) or not path.is_dir():
        _fail("packet staging recovery", "staging entry must be a regular directory")
    staging_root = path.resolve(strict=True)
    expected_parent = packets_root.resolve(strict=True)
    if staging_root.parent != expected_parent:
        _fail("packet staging recovery", "staging path escaped the packet store")
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            raise VisualRenderPipelineError(
                f"packet staging recovery: could not enumerate safely: {exc}"
            ) from exc
        for child in children:
            if _is_link_or_junction(child):
                _fail("packet staging recovery", "links and junctions are unsupported")
            try:
                details = child.lstat()
                resolved = child.resolve(strict=True)
            except OSError as exc:
                raise VisualRenderPipelineError(
                    f"packet staging recovery: could not inspect safely: {exc}"
                ) from exc
            if not resolved.is_relative_to(staging_root):
                _fail("packet staging recovery", "staging child escaped its root")
            if stat_module.S_ISDIR(details.st_mode):
                pending.append(child)
            elif not stat_module.S_ISREG(details.st_mode) or details.st_nlink != 1:
                _fail(
                    "packet staging recovery",
                    "special or hard-linked staging files are unsupported",
                )


def _remove_interrupted_packet_staging(
    output_dir: Path,
    inventory: VisualRenderJobInventoryV1,
    ledger: RenderLedgerV1,
) -> None:
    """Remove only owned partial staging for ledger attempts left rendering."""

    packets_root = Path(output_dir) / "packets"
    if not packets_root.exists() and not _is_link_or_junction(packets_root):
        return
    if _is_link_or_junction(packets_root) or not packets_root.is_dir():
        _fail("packet staging recovery", "packets root must be a regular directory")
    jobs = {job.packet_id for job in inventory.jobs}
    for child in tuple(packets_root.iterdir()):
        match = _PACKET_STAGING.fullmatch(child.name)
        if match is None:
            continue
        packet_id = match.group(1)
        if (
            packet_id not in jobs
            or _latest_entry(ledger, packet_id).state is not RenderLedgerState.RENDERING
        ):
            _fail(
                "packet staging recovery",
                "staging entry is not bound to an interrupted render attempt",
            )
        _require_safe_staging_tree(child, packets_root)
        shutil.rmtree(child)


def run_visual_render_pipeline(
    inventory: VisualRenderJobInventoryV1,
    output_dir: Path,
    *,
    run_id: str,
    render_callback: VisualPacketRenderCallback,
    resume: bool = False,
    retry_execution_errors: bool = False,
    selected_packet_ids: Sequence[str] | None = None,
    timestamp_factory: Callable[[], str] = _timestamp,
    operational_manifest: M4AOperationalRunManifestV1 | None = None,
) -> VisualRenderPipelineResultV1:
    """Render or resume an exact inventory with durable per-packet evidence."""

    root = Path(output_dir)
    canonical_run_id = _text(run_id, "render pipeline run ID")
    selected = (
        inventory.packet_ids
        if selected_packet_ids is None
        else tuple(selected_packet_ids)
    )
    selected_set = set(selected)
    expected_selected = tuple(
        packet_id for packet_id in inventory.packet_ids if packet_id in selected_set
    )
    if (
        not selected
        or len(selected) != len(selected_set)
        or selected != expected_selected
    ):
        _fail("render pipeline selection", "must be an ordered packet-ID subset")
    compute_selected_packet_inventory_digest(selected)
    if operational_manifest is not None:
        _validate_operational_manifest_binding(
            operational_manifest,
            inventory,
            canonical_run_id,
            selected,
        )
    if resume:
        if _is_link_or_junction(root) or not root.is_dir():
            _fail("render resume", "requires an existing regular output root")
        if operational_manifest is not None:
            from latentguard.vision_data.run_manifest import (
                M4A_RUN_MANIFEST_FILENAME,
                require_m4a_run_manifest_resume_match,
            )

            require_m4a_run_manifest_resume_match(
                root / M4A_RUN_MANIFEST_FILENAME,
                operational_manifest,
            )
        persisted_inventory = load_render_job_inventory(root)
        if persisted_inventory.content_digest != inventory.content_digest:
            _fail("render resume", "job inventory content drift")
        ledger = load_render_ledger(root)
        _remove_interrupted_packet_staging(root, inventory, ledger)
    else:
        ledger = _initialize_pipeline_root(
            root,
            inventory,
            canonical_run_id,
            operational_manifest,
        )
    _validate_ledger_contract(ledger, inventory, canonical_run_id)
    before_plan = ledger
    updated, plan = plan_render_resume(
        ledger,
        retry_execution_errors=retry_execution_errors,
        selected_packet_ids=selected,
    )
    if updated.content_digest != ledger.content_digest:
        update_render_ledger(ledger, updated, root)
        ledger = load_render_ledger(root)
    selected_work = tuple(
        packet_id for packet_id in plan.work_packet_ids if packet_id in selected_set
    )
    if not selected_work:
        _validate_packet_store_inventory(root, inventory, ledger)
        roots, packets = _load_complete_packets(root, inventory, ledger)
        all_complete = len(packets) == len(inventory.jobs)
        proof = (
            verify_zero_work_render_resume(before_plan, ledger, plan)
            if all_complete
            else None
        )
        return VisualRenderPipelineResultV1(
            inventory_content_digest=inventory.content_digest,
            ledger=ledger,
            packet_bundle_roots=roots,
            packets=packets,
            rendered_packet_ids=(),
            recovered_packet_ids=(),
            retried_packet_ids=(),
            zero_work_proof=proof,
        )

    work = set(selected_work)
    rendered: list[str] = []
    for job in inventory.jobs:
        if job.packet_id not in work:
            continue
        current = _latest_entry(ledger, job.packet_id)
        began_as_rendering = current.state is RenderLedgerState.RENDERING
        if current.state is RenderLedgerState.PENDING:
            started = start_render_attempt(
                ledger, job.packet_id, started_at=timestamp_factory()
            )
            update_render_ledger(ledger, started, root)
            ledger = load_render_ledger(root)
        elif not began_as_rendering:
            _fail(job.packet_id, "resume selected a non-renderable attempt")
        bundle_root = _packet_bundle_root(root, job.packet_id)
        try:
            allow_existing = began_as_rendering or job.packet_id in set(
                plan.retried_packet_ids
            )
            if bundle_root.exists() or _is_link_or_junction(bundle_root):
                if not allow_existing:
                    _invalid(job.packet_id, "unexpected pre-existing packet store")
                packet = _load_packet_for_job(bundle_root, job)
            else:
                prepared = render_callback(job)
                if not isinstance(prepared, PreparedVisualPacket):
                    _invalid(job.packet_id, "renderer returned an unsupported result")
                _validate_packet_for_job(prepared.packet, job)
                save_visual_packet(prepared.packet, prepared.images, bundle_root)
                packet = _load_packet_for_job(bundle_root, job)
                rendered.append(job.packet_id)
            completed = finish_render_attempt(
                ledger,
                job.packet_id,
                packet_content_digest=packet.content_digest,
                image_inventory_digest=_packet_image_inventory_digest(packet),
                finished_at=timestamp_factory(),
            )
            update_render_ledger(ledger, completed, root)
            ledger = load_render_ledger(root)
        except (InvalidVisualRenderPacketError, VisualSerializationError) as exc:
            failed = fail_render_attempt(
                ledger,
                job.packet_id,
                invalid=True,
                error_type=type(exc).__name__,
                error_message=str(exc),
                finished_at=timestamp_factory(),
            )
            update_render_ledger(ledger, failed, root)
            raise InvalidVisualRenderPacketError(
                f"{job.packet_id}: invalid rendered packet"
            ) from exc
        except Exception as exc:
            failed = fail_render_attempt(
                ledger,
                job.packet_id,
                invalid=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
                finished_at=timestamp_factory(),
            )
            update_render_ledger(ledger, failed, root)
            raise VisualRenderExecutionError(
                f"{job.packet_id}: render execution failed"
            ) from exc

    ledger = load_render_ledger(root)
    _validate_packet_store_inventory(root, inventory, ledger)
    roots, packets = _load_complete_packets(root, inventory, ledger)
    return VisualRenderPipelineResultV1(
        inventory_content_digest=inventory.content_digest,
        ledger=ledger,
        packet_bundle_roots=roots,
        packets=packets,
        rendered_packet_ids=tuple(rendered),
        recovered_packet_ids=tuple(
            item for item in plan.recovered_packet_ids if item in selected_set
        ),
        retried_packet_ids=tuple(
            item for item in plan.retried_packet_ids if item in selected_set
        ),
    )


__all__ = [
    "RENDER_JOB_INVENTORY_SCHEMA_VERSION",
    "RENDER_JOB_MANIFEST_FILENAME",
    "RENDER_JOB_SCHEMA_VERSION",
    "RENDER_SEED_DERIVATION",
    "InvalidVisualRenderPacketError",
    "PreparedVisualPacket",
    "VisualPacketJobV1",
    "VisualPacketRenderCallback",
    "VisualRenderExecutionError",
    "VisualRenderJobInventoryV1",
    "VisualRenderPipelineError",
    "VisualRenderPipelineResultV1",
    "compute_selected_packet_inventory_digest",
    "derive_render_seed",
    "load_render_job_inventory",
    "run_visual_render_pipeline",
    "save_render_job_inventory",
]
