"""Deterministic M3B training-run identities and reproducibility manifests."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import StrEnum

from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.config import LossMode, ResolvedModelConfig, TrainingConfig

TRAINING_RUN_IDENTITY_SCHEMA_VERSION = "1.0"
TRAINING_RUN_MANIFEST_SCHEMA_VERSION = "1.0"
TRAINING_RUN_ID_PREFIX = "m3b-run-sha256-"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")


class TrainingManifestError(ValueError):
    """Raised when a training identity or manifest is incomplete or malformed."""


class TrainingRunStatus(StrEnum):
    """Immutable lifecycle states recorded for one run directory."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _canonical_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrainingManifestError(f"{context}: expected canonical non-empty text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TrainingManifestError(f"{context}: control characters are unsupported")
    return value


def _sha256(value: object, context: str) -> str:
    text = _canonical_text(value, context)
    if _SHA256_RE.fullmatch(text) is None:
        raise TrainingManifestError(
            f"{context}: expected sha256:<64 lowercase hexadecimal characters>"
        )
    return text


def _optional_text(value: object, context: str) -> str | None:
    return None if value is None else _canonical_text(value, context)


@dataclass(frozen=True, slots=True)
class TrainingRunIdentity:
    """Path- and host-independent identity of one semantic training run."""

    dataset_digest: str
    split_digest: str
    preprocessing_digest: str
    model_config_digest: str
    training_config_digest: str
    seed: int
    code_semantic_version: str
    schema_version: str = TRAINING_RUN_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate every semantic identity component."""

        if self.schema_version != TRAINING_RUN_IDENTITY_SCHEMA_VERSION:
            raise TrainingManifestError(
                "TrainingRunIdentity.schema_version: unsupported version "
                f"{self.schema_version!r}"
            )
        for name in (
            "dataset_digest",
            "split_digest",
            "preprocessing_digest",
            "model_config_digest",
            "training_config_digest",
        ):
            _sha256(getattr(self, name), f"TrainingRunIdentity.{name}")
        if type(self.seed) is not int or self.seed < 0:
            raise TrainingManifestError(
                "TrainingRunIdentity.seed: expected a non-negative integer"
            )
        _canonical_text(
            self.code_semantic_version,
            "TrainingRunIdentity.code_semantic_version",
        )

    def as_mapping(self) -> dict[str, object]:
        """Return exactly the semantic fields participating in run identity."""

        return {
            "schema_version": self.schema_version,
            "dataset_digest": self.dataset_digest,
            "split_digest": self.split_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "model_config_digest": self.model_config_digest,
            "training_config_digest": self.training_config_digest,
            "seed": self.seed,
            "code_semantic_version": self.code_semantic_version,
        }

    @property
    def run_id(self) -> str:
        """Return the canonical run identifier."""

        encoded = canonical_json_bytes(
            self.as_mapping(), context="M3BTrainingRunIdentity"
        )
        return f"{TRAINING_RUN_ID_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def compute_training_run_identifier(
    *,
    dataset_digest: str,
    split_digest: str,
    preprocessing_digest: str,
    model_config_digest: str,
    training_config_digest: str,
    seed: int,
    code_semantic_version: str,
) -> str:
    """Compute a run ID without accepting a path, hostname, or timestamp."""

    return TrainingRunIdentity(
        dataset_digest=dataset_digest,
        split_digest=split_digest,
        preprocessing_digest=preprocessing_digest,
        model_config_digest=model_config_digest,
        training_config_digest=training_config_digest,
        seed=seed,
        code_semantic_version=code_semantic_version,
    ).run_id


compute_run_identity = compute_training_run_identifier


@dataclass(frozen=True, slots=True)
class TrainingRunEnvironment:
    """Runtime facts recorded for reproducibility but excluded from run identity."""

    hostname: str
    python_version: str
    numpy_version: str
    pytorch_version: str
    cuda_version: str | None
    gpu_model: str | None
    gpu_driver_version: str | None
    cublas_workspace_config: str | None
    package_versions: tuple[str, ...]
    launch_command: tuple[str, ...]
    deterministic_algorithms: bool
    nondeterministic_operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate detached runtime metadata without hashing it into identity."""

        for name in (
            "hostname",
            "python_version",
            "numpy_version",
            "pytorch_version",
        ):
            _canonical_text(getattr(self, name), f"TrainingRunEnvironment.{name}")
        _optional_text(self.cuda_version, "TrainingRunEnvironment.cuda_version")
        _optional_text(self.gpu_model, "TrainingRunEnvironment.gpu_model")
        _optional_text(
            self.gpu_driver_version,
            "TrainingRunEnvironment.gpu_driver_version",
        )
        _optional_text(
            self.cublas_workspace_config,
            "TrainingRunEnvironment.cublas_workspace_config",
        )
        if not isinstance(self.package_versions, tuple) or not self.package_versions:
            raise TrainingManifestError(
                "TrainingRunEnvironment.package_versions: expected a non-empty tuple"
            )
        for index, package in enumerate(self.package_versions):
            _canonical_text(
                package, f"TrainingRunEnvironment.package_versions[{index}]"
            )
        if not isinstance(self.launch_command, tuple) or not self.launch_command:
            raise TrainingManifestError(
                "TrainingRunEnvironment.launch_command: expected a non-empty tuple"
            )
        for index, argument in enumerate(self.launch_command):
            _canonical_text(argument, f"TrainingRunEnvironment.launch_command[{index}]")
        if type(self.deterministic_algorithms) is not bool:
            raise TrainingManifestError(
                "TrainingRunEnvironment.deterministic_algorithms: expected a boolean"
            )
        if not isinstance(self.nondeterministic_operations, tuple):
            raise TrainingManifestError(
                "TrainingRunEnvironment.nondeterministic_operations: expected a tuple"
            )
        for index, operation in enumerate(self.nondeterministic_operations):
            _canonical_text(
                operation,
                f"TrainingRunEnvironment.nondeterministic_operations[{index}]",
            )

    def as_mapping(self) -> dict[str, object]:
        """Return runtime facts for the persisted run manifest."""

        return {
            "hostname": self.hostname,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "gpu_model": self.gpu_model,
            "gpu_driver_version": self.gpu_driver_version,
            "cublas_workspace_config": self.cublas_workspace_config,
            "package_versions": list(self.package_versions),
            "launch_command": list(self.launch_command),
            "deterministic_algorithms": self.deterministic_algorithms,
            "nondeterministic_operations": list(self.nondeterministic_operations),
        }


@dataclass(frozen=True, slots=True)
class TrainingRunManifest:
    """Reproducibility record binding code and runtime to one semantic run ID."""

    identity: TrainingRunIdentity
    acceptance_report_digest: str
    model_configuration: ResolvedModelConfig
    training_configuration: TrainingConfig
    positive_class_weight: float | None
    git_sha: str
    branch: str
    environment: TrainingRunEnvironment
    started_at_utc: str
    finished_at_utc: str | None
    status: TrainingRunStatus
    schema_version: str = TRAINING_RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate code and lifecycle metadata while preserving identity scope."""

        if self.schema_version != TRAINING_RUN_MANIFEST_SCHEMA_VERSION:
            raise TrainingManifestError(
                "TrainingRunManifest.schema_version: unsupported version "
                f"{self.schema_version!r}"
            )
        if not isinstance(self.identity, TrainingRunIdentity):
            raise TrainingManifestError(
                "TrainingRunManifest.identity: expected TrainingRunIdentity"
            )
        _sha256(
            self.acceptance_report_digest,
            "TrainingRunManifest.acceptance_report_digest",
        )
        if not isinstance(self.model_configuration, ResolvedModelConfig):
            raise TrainingManifestError(
                "TrainingRunManifest.model_configuration: invalid resolved model"
            )
        if not isinstance(self.training_configuration, TrainingConfig):
            raise TrainingManifestError(
                "TrainingRunManifest.training_configuration: invalid training config"
            )
        if (
            self.model_configuration.content_digest != self.identity.model_config_digest
            or self.training_configuration.content_digest
            != self.identity.training_config_digest
        ):
            raise TrainingManifestError(
                "TrainingRunManifest: resolved configuration digest mismatch"
            )
        if self.positive_class_weight is not None and (
            type(self.positive_class_weight) not in (int, float)
            or not math.isfinite(float(self.positive_class_weight))
            or float(self.positive_class_weight) <= 0.0
        ):
            raise TrainingManifestError(
                "TrainingRunManifest.positive_class_weight: invalid value"
            )
        if (
            self.training_configuration.loss_mode is LossMode.UNWEIGHTED_BCE
            and self.positive_class_weight is not None
        ) or (
            self.training_configuration.loss_mode is LossMode.CLASS_WEIGHTED_BCE
            and self.positive_class_weight is None
        ):
            raise TrainingManifestError(
                "TrainingRunManifest.positive_class_weight: loss-mode mismatch"
            )
        if not isinstance(self.environment, TrainingRunEnvironment):
            raise TrainingManifestError(
                "TrainingRunManifest.environment: expected TrainingRunEnvironment"
            )
        if _GIT_SHA_RE.fullmatch(self.git_sha) is None:
            raise TrainingManifestError(
                "TrainingRunManifest.git_sha: expected a full lowercase Git SHA"
            )
        _canonical_text(self.branch, "TrainingRunManifest.branch")
        _canonical_text(self.started_at_utc, "TrainingRunManifest.started_at_utc")
        _optional_text(self.finished_at_utc, "TrainingRunManifest.finished_at_utc")
        if not isinstance(self.status, TrainingRunStatus):
            raise TrainingManifestError(
                "TrainingRunManifest.status: expected TrainingRunStatus"
            )
        if (
            self.status is TrainingRunStatus.RUNNING
            and self.finished_at_utc is not None
        ):
            raise TrainingManifestError(
                "running manifests must not record a finish time"
            )
        if (
            self.status is not TrainingRunStatus.RUNNING
            and self.finished_at_utc is None
        ):
            raise TrainingManifestError(
                "completed or failed manifests require a finish time"
            )

    @property
    def run_id(self) -> str:
        """Return the semantic run ID, independent of environment fields."""

        return self.identity.run_id

    def as_mapping(self) -> dict[str, object]:
        """Return the complete persisted manifest payload."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "identity": self.identity.as_mapping(),
            "acceptance_report_digest": self.acceptance_report_digest,
            "model_configuration": self.model_configuration.as_mapping(),
            "training_configuration": self.training_configuration.as_mapping(),
            "positive_class_weight": self.positive_class_weight,
            "git_sha": self.git_sha,
            "branch": self.branch,
            "environment": self.environment.as_mapping(),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "status": self.status.value,
        }


__all__ = [
    "TRAINING_RUN_IDENTITY_SCHEMA_VERSION",
    "TRAINING_RUN_ID_PREFIX",
    "TRAINING_RUN_MANIFEST_SCHEMA_VERSION",
    "TrainingManifestError",
    "TrainingRunEnvironment",
    "TrainingRunIdentity",
    "TrainingRunManifest",
    "TrainingRunStatus",
    "compute_run_identity",
    "compute_training_run_identifier",
]
