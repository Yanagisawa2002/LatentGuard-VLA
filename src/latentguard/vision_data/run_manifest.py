"""Strict operational run and environment evidence for M4A rendering.

This manifest deliberately remains outside packet and dataset semantic
identities.  Its complete digest protects reproducibility and resume evidence,
including host and wall-clock facts, without making those facts model inputs or
visual-content identity components.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn, cast

from latentguard.evaluation.security import (
    REDACTED_OPERATIONAL_COMMAND,
    is_sanitized_operational_text,
    sanitize_launch_command,
)
from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import canonical_json_bytes
from latentguard.vision_data.models import SourceCollection

M4A_RUN_MANIFEST_ARTIFACT_TYPE = "latentguard_m4a_operational_run_manifest_v1"
M4A_RUN_MANIFEST_FILENAME = "run-manifest.json"
M4A_RUN_MANIFEST_SCHEMA_VERSION = "1.0"
M4A_VISUAL_PROBE_SOURCE_SCHEMA_VERSION = "1.0"
MAX_M4A_RUN_MANIFEST_BYTES = 1024 * 1024
M4A_PROBE_COMMAND = "probe-maniskill-pickcube-visual"
M4A_RENDER_M3A_COMMAND = "render-m3a-visual-dataset"
M4A_RENDER_M3C_COMMAND = "render-m3c-external-visual-dataset"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class M4ARunManifestError(ValueError):
    """Raised when M4A operational evidence is unsafe or identity-drifted."""


def _fail(context: str, reason: str) -> NoReturn:
    raise M4ARunManifestError(f"{context}: {reason}")


def _safe_text(value: object, context: str) -> str:
    if not is_sanitized_operational_text(value):
        _fail(context, "expected sanitized non-sensitive operational text")
    return cast(str, value)


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        _fail(context, "expected lowercase sha256 digest")
    return value


def _git_sha(value: object, context: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        _fail(context, "expected full lowercase 40-hex Git SHA")
    return value


def _uint32(value: object, context: str) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        _fail(context, "expected uint32 integer")
    return value


def _timestamp(value: object, context: str) -> str:
    text = _safe_text(value, context)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise M4ARunManifestError(
            f"{context}: expected timezone-aware ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(context, "expected timezone-aware ISO-8601 timestamp")
    return text


def _content_digest(value: object, *, context: str) -> str:
    try:
        encoded = canonical_json_bytes(value, context=context)
    except ReplayValidationError as exc:
        raise M4ARunManifestError(str(exc)) from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _exact_fields(
    value: Mapping[str, object], expected: set[str], context: str
) -> Mapping[str, object]:
    if set(value) != expected or any(not isinstance(key, str) for key in value):
        _fail(context, "unexpected or missing fields")
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return value


def sanitize_m4a_launch_argv(arguments: Sequence[str]) -> tuple[str, ...]:
    """Return sequence-aware sanitized launch arguments for persistence."""

    if isinstance(arguments, (str, bytes, bytearray)) or not arguments:
        _fail("M4A launch argv", "expected a non-empty argument sequence")
    if any(not isinstance(argument, str) for argument in arguments):
        _fail("M4A launch argv", "arguments must be strings")
    command = sanitize_launch_command(tuple(arguments))
    if command == REDACTED_OPERATIONAL_COMMAND:
        return (command,)
    try:
        sanitized = tuple(shlex.split(command, posix=True))
    except ValueError as exc:  # pragma: no cover - sanitizer always quotes safely
        raise M4ARunManifestError(
            "M4A launch argv: sanitized command could not be decoded"
        ) from exc
    if not sanitized:
        _fail("M4A launch argv", "sanitization produced no arguments")
    for index, argument in enumerate(sanitized):
        _safe_text(argument, f"M4A launch argv[{index}]")
    return sanitized


def compute_selected_packet_inventory_digest(
    selected_packet_ids: Sequence[str],
) -> str:
    """Bind the exact ordered, duplicate-free packet selection for one run."""

    if isinstance(selected_packet_ids, (str, bytes, bytearray)):
        _fail("M4A selected packet inventory", "expected packet ID sequence")
    packet_ids = tuple(selected_packet_ids)
    if not packet_ids:
        _fail("M4A selected packet inventory", "cannot be empty")
    for index, packet_id in enumerate(packet_ids):
        _safe_text(packet_id, f"M4A selected packet inventory[{index}]")
    if len(packet_ids) != len(set(packet_ids)):
        _fail("M4A selected packet inventory", "duplicate packet IDs")
    return _content_digest(
        {
            "schema_version": "m4a_selected_packet_inventory_v1",
            "selected_packet_ids": list(packet_ids),
        },
        context="M4ASelectedPacketInventoryV1",
    )


@dataclass(frozen=True, slots=True)
class M4AOperationalEnvironmentV1:
    """Sanitized runtime facts recorded for reproducibility, never semantics."""

    hostname: str
    python_version: str
    gpu_model: str
    gpu_driver_version: str
    cuda_version: str
    pytorch_version: str
    maniskill_version: str
    sapien_version: str
    renderer_backend: str

    def __post_init__(self) -> None:
        """Reject paths, credentials, controls, and missing environment facts."""

        for name in (
            "hostname",
            "python_version",
            "gpu_model",
            "gpu_driver_version",
            "cuda_version",
            "pytorch_version",
            "maniskill_version",
            "sapien_version",
            "renderer_backend",
        ):
            _safe_text(getattr(self, name), f"M4A environment {name}")

    @classmethod
    def create(
        cls,
        *,
        hostname: object,
        python_version: object,
        gpu_model: object,
        gpu_driver_version: object,
        cuda_version: object,
        pytorch_version: object,
        maniskill_version: object,
        sapien_version: object,
        renderer_backend: object,
    ) -> M4AOperationalEnvironmentV1:
        """Validate raw runtime strings into a strict persisted environment."""

        values = {
            name: _safe_text(value, f"M4A environment {name}")
            for name, value in {
                "hostname": hostname,
                "python_version": python_version,
                "gpu_model": gpu_model,
                "gpu_driver_version": gpu_driver_version,
                "cuda_version": cuda_version,
                "pytorch_version": pytorch_version,
                "maniskill_version": maniskill_version,
                "sapien_version": sapien_version,
                "renderer_backend": renderer_backend,
            }.items()
        }
        return cls(**values)

    def as_mapping(self) -> dict[str, object]:
        """Return the exact environment inventory."""

        return {
            "cuda_version": self.cuda_version,
            "gpu_driver_version": self.gpu_driver_version,
            "gpu_model": self.gpu_model,
            "hostname": self.hostname,
            "maniskill_version": self.maniskill_version,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "renderer_backend": self.renderer_backend,
            "sapien_version": self.sapien_version,
        }

    @classmethod
    def from_mapping(cls, value: object) -> M4AOperationalEnvironmentV1:
        """Decode an exact-field sanitized environment inventory."""

        item = _exact_fields(
            _mapping(value, "M4A environment"),
            {
                "cuda_version",
                "gpu_driver_version",
                "gpu_model",
                "hostname",
                "maniskill_version",
                "python_version",
                "pytorch_version",
                "renderer_backend",
                "sapien_version",
            },
            "M4A environment",
        )
        return cls(
            hostname=cast(str, item["hostname"]),
            python_version=cast(str, item["python_version"]),
            gpu_model=cast(str, item["gpu_model"]),
            gpu_driver_version=cast(str, item["gpu_driver_version"]),
            cuda_version=cast(str, item["cuda_version"]),
            pytorch_version=cast(str, item["pytorch_version"]),
            maniskill_version=cast(str, item["maniskill_version"]),
            sapien_version=cast(str, item["sapien_version"]),
            renderer_backend=cast(str, item["renderer_backend"]),
        )


@dataclass(frozen=True, slots=True)
class M4AVisualProbeSourceIdentityV1:
    """Core-owned, content-bound archived-state identity for a visual probe."""

    source_archive_digest: str
    source_episode_id: str
    source_episode_content_digest: str
    source_trajectory_id: str
    source_reset_seed: int
    state_index: int
    state_content_digest: str
    expected_state_digest: str
    verifier_state_digest: str
    schema_version: str = M4A_VISUAL_PROBE_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject incomplete, unsafe, or weakly bound probe identities."""

        for name in (
            "source_archive_digest",
            "source_episode_content_digest",
            "state_content_digest",
            "expected_state_digest",
            "verifier_state_digest",
        ):
            _digest(getattr(self, name), f"M4A visual probe source {name}")
        for name in ("source_episode_id", "source_trajectory_id"):
            _safe_text(getattr(self, name), f"M4A visual probe source {name}")
        _uint32(self.source_reset_seed, "M4A visual probe source reset seed")
        if type(self.state_index) is not int or self.state_index < 0:
            _fail(
                "M4A visual probe source state index",
                "expected non-negative integer",
            )
        if self.schema_version != M4A_VISUAL_PROBE_SOURCE_SCHEMA_VERSION:
            _fail("M4A visual probe source", "unsupported schema version")

    def as_mapping(self) -> dict[str, object]:
        """Return exact path-independent archived-state evidence."""

        return {
            "expected_state_digest": self.expected_state_digest,
            "schema_version": self.schema_version,
            "source_archive_digest": self.source_archive_digest,
            "source_episode_content_digest": self.source_episode_content_digest,
            "source_episode_id": self.source_episode_id,
            "source_reset_seed": self.source_reset_seed,
            "source_trajectory_id": self.source_trajectory_id,
            "state_content_digest": self.state_content_digest,
            "state_index": self.state_index,
            "verifier_state_digest": self.verifier_state_digest,
        }

    @classmethod
    def from_mapping(cls, value: object) -> M4AVisualProbeSourceIdentityV1:
        """Decode exact integration evidence into the core identity model."""

        item = _exact_fields(
            _mapping(value, "M4A visual probe source"),
            {
                "expected_state_digest",
                "schema_version",
                "source_archive_digest",
                "source_episode_content_digest",
                "source_episode_id",
                "source_reset_seed",
                "source_trajectory_id",
                "state_content_digest",
                "state_index",
                "verifier_state_digest",
            },
            "M4A visual probe source",
        )
        return cls(
            source_archive_digest=cast(str, item["source_archive_digest"]),
            source_episode_id=cast(str, item["source_episode_id"]),
            source_episode_content_digest=cast(
                str, item["source_episode_content_digest"]
            ),
            source_trajectory_id=cast(str, item["source_trajectory_id"]),
            source_reset_seed=cast(int, item["source_reset_seed"]),
            state_index=cast(int, item["state_index"]),
            state_content_digest=cast(str, item["state_content_digest"]),
            expected_state_digest=cast(str, item["expected_state_digest"]),
            verifier_state_digest=cast(str, item["verifier_state_digest"]),
            schema_version=cast(str, item["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class M4AOperationalSourceIdentityV1:
    """One exact probe, M3A, or M3C source variant for operational evidence."""

    source_collection: SourceCollection | None
    canonical_source_identity_digest: str | None = None
    visual_probe_source: M4AVisualProbeSourceIdentityV1 | None = None
    m3a_dataset_digest: str | None = None
    m3c_source_set_digest: str | None = None
    m3c_candidate_pool_digest: str | None = None

    def __post_init__(self) -> None:
        """Require exactly the source identities appropriate to the variant."""

        if self.visual_probe_source is not None:
            if not isinstance(self.visual_probe_source, M4AVisualProbeSourceIdentityV1):
                _fail("M4A source identity", "invalid visual probe evidence")
            if self.source_collection is not None or any(
                value is not None
                for value in (
                    self.canonical_source_identity_digest,
                    self.m3a_dataset_digest,
                    self.m3c_source_set_digest,
                    self.m3c_candidate_pool_digest,
                )
            ):
                _fail(
                    "M4A source identity",
                    "visual probe cannot masquerade as a dataset source",
                )
            _safe_text(
                self.visual_probe_source.source_episode_id,
                "M4A source identity probe episode",
            )
            _safe_text(
                self.visual_probe_source.source_trajectory_id,
                "M4A source identity probe trajectory",
            )
            return
        if not isinstance(self.source_collection, SourceCollection):
            _fail("M4A source identity", "unsupported or missing source variant")
        _digest(
            self.canonical_source_identity_digest,
            "M4A source identity canonical source identity",
        )
        if self.source_collection is SourceCollection.M3A_DEVELOPMENT:
            _digest(self.m3a_dataset_digest, "M4A source identity M3A dataset")
            if (
                self.m3c_source_set_digest is not None
                or self.m3c_candidate_pool_digest is not None
            ):
                _fail("M4A source identity", "M3A cannot carry M3C identities")
            return
        if self.source_collection is SourceCollection.M3C_EXTERNAL:
            _digest(self.m3c_source_set_digest, "M4A source identity M3C source set")
            _digest(
                self.m3c_candidate_pool_digest,
                "M4A source identity M3C candidate pool",
            )
            if self.m3a_dataset_digest is not None:
                _fail("M4A source identity", "M3C cannot carry an M3A identity")
            return
        _fail("M4A source identity", "unsupported source collection")

    @property
    def is_visual_probe(self) -> bool:
        """Return whether this is archived-state probe evidence, not a dataset."""

        return self.visual_probe_source is not None

    @property
    def required_command(self) -> str:
        """Return the only command authorized for this source variant."""

        if self.is_visual_probe:
            return M4A_PROBE_COMMAND
        if self.source_collection is SourceCollection.M3A_DEVELOPMENT:
            return M4A_RENDER_M3A_COMMAND
        if self.source_collection is SourceCollection.M3C_EXTERNAL:
            return M4A_RENDER_M3C_COMMAND
        _fail("M4A source identity", "missing source variant")

    @classmethod
    def for_visual_probe(
        cls,
        evidence: M4AVisualProbeSourceIdentityV1 | Mapping[str, object],
    ) -> M4AOperationalSourceIdentityV1:
        """Create a probe-only source variant from exact archived-state evidence."""

        identity = (
            evidence
            if isinstance(evidence, M4AVisualProbeSourceIdentityV1)
            else M4AVisualProbeSourceIdentityV1.from_mapping(evidence)
        )
        return cls(source_collection=None, visual_probe_source=identity)

    @classmethod
    def for_m3a(
        cls,
        dataset_digest: str,
        *,
        canonical_source_identity_digest: str,
    ) -> M4AOperationalSourceIdentityV1:
        """Create the development source variant."""

        return cls(
            source_collection=SourceCollection.M3A_DEVELOPMENT,
            canonical_source_identity_digest=canonical_source_identity_digest,
            m3a_dataset_digest=dataset_digest,
        )

    @classmethod
    def for_m3c(
        cls,
        source_set_digest: str,
        candidate_pool_digest: str,
        *,
        canonical_source_identity_digest: str,
    ) -> M4AOperationalSourceIdentityV1:
        """Create the external evaluation source variant."""

        return cls(
            source_collection=SourceCollection.M3C_EXTERNAL,
            canonical_source_identity_digest=canonical_source_identity_digest,
            m3c_source_set_digest=source_set_digest,
            m3c_candidate_pool_digest=candidate_pool_digest,
        )

    def as_mapping(self) -> dict[str, object]:
        """Return exact variant-specific source identities."""

        if self.source_collection is SourceCollection.M3A_DEVELOPMENT:
            return {
                "canonical_source_identity_digest": (
                    self.canonical_source_identity_digest
                ),
                "m3a_dataset_digest": self.m3a_dataset_digest,
                "source_collection": self.source_collection.value,
            }
        if self.visual_probe_source is not None:
            return {
                "source_collection": "visual_probe",
                "visual_probe_source": dict(self.visual_probe_source.as_mapping()),
            }
        if self.source_collection is None:  # pragma: no cover - guarded at creation
            _fail("M4A source identity", "missing source variant")
        return {
            "canonical_source_identity_digest": self.canonical_source_identity_digest,
            "m3c_candidate_pool_digest": self.m3c_candidate_pool_digest,
            "m3c_source_set_digest": self.m3c_source_set_digest,
            "source_collection": self.source_collection.value,
        }

    @classmethod
    def from_mapping(cls, value: object) -> M4AOperationalSourceIdentityV1:
        """Decode exactly one source variant without repairing mixed fields."""

        item = _mapping(value, "M4A source identity")
        raw_collection = item.get("source_collection")
        if raw_collection == "visual_probe":
            exact = _exact_fields(
                item,
                {"source_collection", "visual_probe_source"},
                "M4A source identity",
            )
            return cls.for_visual_probe(
                _mapping(
                    exact["visual_probe_source"],
                    "M4A source identity visual probe source",
                )
            )
        try:
            collection = SourceCollection(cast(str, raw_collection))
        except (TypeError, ValueError) as exc:
            raise M4ARunManifestError(
                "M4A source identity: unsupported source collection"
            ) from exc
        if collection is SourceCollection.M3A_DEVELOPMENT:
            exact = _exact_fields(
                item,
                {
                    "canonical_source_identity_digest",
                    "m3a_dataset_digest",
                    "source_collection",
                },
                "M4A source identity",
            )
            return cls.for_m3a(
                cast(str, exact["m3a_dataset_digest"]),
                canonical_source_identity_digest=cast(
                    str, exact["canonical_source_identity_digest"]
                ),
            )
        if collection is SourceCollection.M3C_EXTERNAL:
            exact = _exact_fields(
                item,
                {
                    "canonical_source_identity_digest",
                    "m3c_candidate_pool_digest",
                    "m3c_source_set_digest",
                    "source_collection",
                },
                "M4A source identity",
            )
            return cls.for_m3c(
                cast(str, exact["m3c_source_set_digest"]),
                cast(str, exact["m3c_candidate_pool_digest"]),
                canonical_source_identity_digest=cast(
                    str, exact["canonical_source_identity_digest"]
                ),
            )
        _fail("M4A source identity", "unsupported source collection")


@dataclass(frozen=True, slots=True)
class M4AOperationalRunManifestV1:
    """Complete operational evidence excluded from visual semantic identities."""

    run_id: str
    command: str
    git_sha: str
    branch: str
    started_at: str
    render_seed: int
    launch_argv: tuple[str, ...]
    environment: M4AOperationalEnvironmentV1
    visual_compatibility_identity: str
    camera_rig_digest: str
    render_domain_configuration_digest: str
    source_identity: M4AOperationalSourceIdentityV1
    render_job_inventory_digest: str | None
    selected_packet_inventory_digest: str | None
    content_digest: str
    artifact_type: str = M4A_RUN_MANIFEST_ARTIFACT_TYPE
    schema_version: str = M4A_RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate all fields and the digest over complete operational content."""

        if self.artifact_type != M4A_RUN_MANIFEST_ARTIFACT_TYPE:
            _fail("M4A run manifest artifact_type", "unsupported artifact type")
        if self.schema_version != M4A_RUN_MANIFEST_SCHEMA_VERSION:
            _fail("M4A run manifest schema_version", "unsupported version")
        for name in ("run_id", "command", "branch"):
            _safe_text(getattr(self, name), f"M4A run manifest {name}")
        _git_sha(self.git_sha, "M4A run manifest git_sha")
        _timestamp(self.started_at, "M4A run manifest started_at")
        _uint32(self.render_seed, "M4A run manifest render_seed")
        if not isinstance(self.launch_argv, tuple) or not self.launch_argv:
            _fail("M4A run manifest launch_argv", "expected non-empty tuple")
        for index, argument in enumerate(self.launch_argv):
            _safe_text(argument, f"M4A run manifest launch_argv[{index}]")
        if not isinstance(self.environment, M4AOperationalEnvironmentV1):
            _fail("M4A run manifest environment", "invalid environment")
        if not isinstance(self.source_identity, M4AOperationalSourceIdentityV1):
            _fail("M4A run manifest source_identity", "invalid source identity")
        if self.command != self.source_identity.required_command:
            _fail(
                "M4A run manifest command",
                "does not match the exact source variant",
            )
        if self.source_identity.is_visual_probe:
            if (
                self.render_job_inventory_digest is not None
                or self.selected_packet_inventory_digest is not None
            ):
                _fail(
                    "M4A run manifest inventory",
                    "visual probe cannot carry render-job or packet inventory",
                )
        else:
            _digest(
                self.render_job_inventory_digest,
                "M4A run manifest render_job_inventory_digest",
            )
            _digest(
                self.selected_packet_inventory_digest,
                "M4A run manifest selected_packet_inventory_digest",
            )
        for name in (
            "visual_compatibility_identity",
            "camera_rig_digest",
            "render_domain_configuration_digest",
        ):
            _digest(getattr(self, name), f"M4A run manifest {name}")
        _digest(self.content_digest, "M4A run manifest content_digest")
        if self.content_digest != self.expected_content_digest:
            _fail("M4A run manifest content_digest", "complete content changed")

    @classmethod
    def create(
        cls,
        *,
        run_id: object,
        command: object,
        git_sha: str,
        branch: object,
        started_at: object,
        render_seed: int,
        launch_argv: Sequence[str],
        environment: M4AOperationalEnvironmentV1,
        visual_compatibility_identity: str,
        camera_rig_digest: str,
        render_domain_configuration_digest: str,
        source_identity: M4AOperationalSourceIdentityV1,
        render_job_inventory_digest: str | None,
        selected_packet_inventory_digest: str | None,
    ) -> M4AOperationalRunManifestV1:
        """Sanitize operational inputs and create a fully digested manifest."""

        if not isinstance(environment, M4AOperationalEnvironmentV1):
            _fail("M4A run manifest environment", "invalid environment")
        if not isinstance(source_identity, M4AOperationalSourceIdentityV1):
            _fail("M4A run manifest source_identity", "invalid source identity")
        fields: dict[str, object] = {
            "artifact_type": M4A_RUN_MANIFEST_ARTIFACT_TYPE,
            "branch": _safe_text(branch, "M4A run manifest branch"),
            "camera_rig_digest": camera_rig_digest,
            "command": _safe_text(command, "M4A run manifest command"),
            "environment": environment.as_mapping(),
            "git_sha": git_sha,
            "launch_argv": list(sanitize_m4a_launch_argv(launch_argv)),
            "render_job_inventory_digest": render_job_inventory_digest,
            "render_domain_configuration_digest": (render_domain_configuration_digest),
            "render_seed": render_seed,
            "run_id": _safe_text(run_id, "M4A run manifest run_id"),
            "schema_version": M4A_RUN_MANIFEST_SCHEMA_VERSION,
            "selected_packet_inventory_digest": selected_packet_inventory_digest,
            "source_identity": source_identity.as_mapping(),
            "started_at": _timestamp(started_at, "M4A run manifest started_at"),
            "visual_compatibility_identity": visual_compatibility_identity,
        }
        digest = _content_digest(fields, context="M4AOperationalRunManifestV1")
        return cls(
            run_id=cast(str, fields["run_id"]),
            command=cast(str, fields["command"]),
            git_sha=git_sha,
            branch=cast(str, fields["branch"]),
            started_at=cast(str, fields["started_at"]),
            render_seed=render_seed,
            launch_argv=tuple(cast(list[str], fields["launch_argv"])),
            environment=environment,
            visual_compatibility_identity=visual_compatibility_identity,
            camera_rig_digest=camera_rig_digest,
            render_domain_configuration_digest=render_domain_configuration_digest,
            source_identity=source_identity,
            render_job_inventory_digest=render_job_inventory_digest,
            selected_packet_inventory_digest=selected_packet_inventory_digest,
            content_digest=digest,
        )

    def _content_mapping(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_type,
            "branch": self.branch,
            "camera_rig_digest": self.camera_rig_digest,
            "command": self.command,
            "environment": self.environment.as_mapping(),
            "git_sha": self.git_sha,
            "launch_argv": list(self.launch_argv),
            "render_job_inventory_digest": self.render_job_inventory_digest,
            "render_domain_configuration_digest": (
                self.render_domain_configuration_digest
            ),
            "render_seed": self.render_seed,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "selected_packet_inventory_digest": (self.selected_packet_inventory_digest),
            "source_identity": self.source_identity.as_mapping(),
            "started_at": self.started_at,
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }

    @property
    def expected_content_digest(self) -> str:
        """Recompute the digest protecting every operational field."""

        return _content_digest(
            self._content_mapping(), context="M4AOperationalRunManifestV1"
        )

    def as_mapping(self) -> dict[str, object]:
        """Return the exact persisted manifest including its strict digest."""

        return {**self._content_mapping(), "content_digest": self.content_digest}

    @classmethod
    def from_mapping(cls, value: object) -> M4AOperationalRunManifestV1:
        """Decode exact fields and verify complete operational content."""

        item = _exact_fields(
            _mapping(value, "M4A run manifest"),
            {
                "artifact_type",
                "branch",
                "camera_rig_digest",
                "command",
                "content_digest",
                "environment",
                "git_sha",
                "launch_argv",
                "render_job_inventory_digest",
                "render_domain_configuration_digest",
                "render_seed",
                "run_id",
                "schema_version",
                "selected_packet_inventory_digest",
                "source_identity",
                "started_at",
                "visual_compatibility_identity",
            },
            "M4A run manifest",
        )
        raw_argv = item["launch_argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            _fail("M4A run manifest launch_argv", "expected non-empty JSON array")
        return cls(
            run_id=cast(str, item["run_id"]),
            command=cast(str, item["command"]),
            git_sha=cast(str, item["git_sha"]),
            branch=cast(str, item["branch"]),
            started_at=cast(str, item["started_at"]),
            render_seed=cast(int, item["render_seed"]),
            launch_argv=tuple(cast(list[str], raw_argv)),
            environment=M4AOperationalEnvironmentV1.from_mapping(item["environment"]),
            visual_compatibility_identity=cast(
                str, item["visual_compatibility_identity"]
            ),
            camera_rig_digest=cast(str, item["camera_rig_digest"]),
            render_domain_configuration_digest=cast(
                str, item["render_domain_configuration_digest"]
            ),
            source_identity=M4AOperationalSourceIdentityV1.from_mapping(
                item["source_identity"]
            ),
            render_job_inventory_digest=cast(
                str | None, item["render_job_inventory_digest"]
            ),
            selected_packet_inventory_digest=cast(
                str | None, item["selected_packet_inventory_digest"]
            ),
            content_digest=cast(str, item["content_digest"]),
            artifact_type=cast(str, item["artifact_type"]),
            schema_version=cast(str, item["schema_version"]),
        )


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("M4A run manifest JSON", "duplicate fields are unsupported")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail("M4A run manifest JSON", "non-finite constants are unsupported")


def save_m4a_run_manifest(manifest: M4AOperationalRunManifestV1, path: Path) -> Path:
    """Transactionally create a new operational manifest without overwriting."""

    if not isinstance(manifest, M4AOperationalRunManifestV1):
        _fail("M4A run manifest save", "expected M4AOperationalRunManifestV1")
    destination = Path(path)
    from latentguard.vision_data.serialization import _is_link_or_junction

    if destination.exists() or _is_link_or_junction(destination):
        _fail("M4A run manifest save", "refusing to overwrite an existing path")
    parent = destination.parent
    if _is_link_or_junction(parent) or not parent.is_dir():
        _fail("M4A run manifest save", "parent must be a regular directory")
    payload = (
        json.dumps(
            manifest.as_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary_name: str | None = None
    destination_created = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, destination)
        destination_created = True
        Path(temporary_name).unlink()
        temporary_name = None
    except OSError as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        if destination_created:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
        raise M4ARunManifestError(
            "M4A run manifest save: atomic create failed"
        ) from exc
    return destination


def load_m4a_run_manifest(path: Path) -> M4AOperationalRunManifestV1:
    """Strictly load a regular single-linked manifest and verify its digest."""

    source = Path(path)
    from latentguard.vision_data.serialization import _is_link_or_junction

    if (
        _is_link_or_junction(source)
        or _is_link_or_junction(source.parent)
        or not source.is_file()
    ):
        _fail("M4A run manifest load", "expected regular non-linked file")
    try:
        details = source.stat()
        if details.st_nlink != 1:
            _fail("M4A run manifest load", "hard-linked files are unsupported")
        if details.st_size <= 0 or details.st_size > MAX_M4A_RUN_MANIFEST_BYTES:
            _fail("M4A run manifest load", "file size is outside safe bounds")
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except M4ARunManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M4ARunManifestError(
            "M4A run manifest load: could not read safely"
        ) from exc
    return M4AOperationalRunManifestV1.from_mapping(raw)


def require_m4a_run_manifest_resume_match(
    path: Path, expected: M4AOperationalRunManifestV1
) -> M4AOperationalRunManifestV1:
    """Require an existing manifest to equal the requested resume exactly."""

    if not isinstance(expected, M4AOperationalRunManifestV1):
        _fail("M4A run manifest resume", "invalid expected manifest")
    existing = load_m4a_run_manifest(path)
    if existing != expected or existing.content_digest != expected.content_digest:
        _fail("M4A run manifest resume", "operational identity drift")
    return existing


def persist_m4a_run_manifest(
    manifest: M4AOperationalRunManifestV1,
    path: Path,
    *,
    resume: bool,
) -> Path:
    """Create a new manifest or validate an exact zero-write resume."""

    if resume:
        require_m4a_run_manifest_resume_match(path, manifest)
        return Path(path)
    return save_m4a_run_manifest(manifest, path)


__all__ = [
    "M4A_PROBE_COMMAND",
    "M4A_RENDER_M3A_COMMAND",
    "M4A_RENDER_M3C_COMMAND",
    "M4A_RUN_MANIFEST_ARTIFACT_TYPE",
    "M4A_RUN_MANIFEST_FILENAME",
    "M4A_RUN_MANIFEST_SCHEMA_VERSION",
    "M4A_VISUAL_PROBE_SOURCE_SCHEMA_VERSION",
    "MAX_M4A_RUN_MANIFEST_BYTES",
    "M4AOperationalEnvironmentV1",
    "M4AOperationalRunManifestV1",
    "M4AOperationalSourceIdentityV1",
    "M4AVisualProbeSourceIdentityV1",
    "M4ARunManifestError",
    "compute_selected_packet_inventory_digest",
    "load_m4a_run_manifest",
    "persist_m4a_run_manifest",
    "require_m4a_run_manifest_resume_match",
    "sanitize_m4a_launch_argv",
    "save_m4a_run_manifest",
]
