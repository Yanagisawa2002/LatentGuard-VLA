"""Content-bound runtime package for one robot-policy checkpoint."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from latentguard.replay.identity import canonical_json_bytes, canonical_json_value

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class PolicyPackageError(ValueError):
    """Raised when a policy package is incomplete or content-invalid."""


class PolicyCompatibilityError(PolicyPackageError):
    """Raised when a complete package does not match a required contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PolicyPackageError(f"{context}: {reason}")


def _required_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected non-empty canonical text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail(context, "control characters are unsupported")
    return value


def _digest(value: object, context: str) -> str:
    text = _required_text(value, context)
    if _DIGEST_RE.fullmatch(text) is None:
        _fail(context, "expected sha256:<64 lowercase hexadecimal characters>")
    return text


def _relative_path(value: object, context: str) -> str:
    text = _required_text(value, context)
    if "\\" in text:
        _fail(context, "paths must use portable forward slashes")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        _fail(context, "expected a normalized package-relative path")
    return path.as_posix()


def _optional_path(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _relative_path(value, context)


def _optional_digest(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _digest(value, context)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    canonical = canonical_json_value(value, context=context)
    if not isinstance(canonical, dict):
        _fail(context, "expected JSON object")
    return cast(dict[str, object], canonical)


@dataclass(frozen=True)
class PolicyArtifactVerification:
    """Verified runtime inventory for a content-bound policy package."""

    policy_id: str
    verified_artifact_count: int
    package_digest: str
    artifact_digests: Mapping[str, str]


@dataclass(frozen=True)
class PolicyPackage:
    """Complete, path-portable definition of one executable robot policy."""

    policy_id: str
    policy_family: str
    source_commit: str
    checkpoint_path: str
    checkpoint_sha256: str
    model_config_path: str
    model_config_sha256: str
    preprocessor_path: str | None
    preprocessor_sha256: str | None
    postprocessor_path: str | None
    postprocessor_sha256: str | None
    normalization_path: str | None
    normalization_sha256: str | None
    observation_spec: Mapping[str, object]
    action_spec: Mapping[str, object]
    environment_spec: Mapping[str, object]
    action_horizon: int
    deterministic: bool
    acceptance_evidence: Mapping[str, object]
    schema_version: str = "policy-package-v1"

    def __post_init__(self) -> None:
        """Reject malformed or incomplete package definitions immediately."""
        _required_text(self.policy_id, "PolicyPackage.policy_id")
        _required_text(self.policy_family, "PolicyPackage.policy_family")
        if _COMMIT_RE.fullmatch(self.source_commit) is None:
            _fail("PolicyPackage.source_commit", "expected a full lowercase Git SHA")
        _relative_path(self.checkpoint_path, "PolicyPackage.checkpoint_path")
        _digest(self.checkpoint_sha256, "PolicyPackage.checkpoint_sha256")
        _relative_path(self.model_config_path, "PolicyPackage.model_config_path")
        _digest(self.model_config_sha256, "PolicyPackage.model_config_sha256")
        self._validate_optional_pair(
            self.preprocessor_path,
            self.preprocessor_sha256,
            "preprocessor",
        )
        self._validate_optional_pair(
            self.postprocessor_path,
            self.postprocessor_sha256,
            "postprocessor",
        )
        self._validate_optional_pair(
            self.normalization_path,
            self.normalization_sha256,
            "normalization",
        )
        for name, value in (
            ("observation_spec", self.observation_spec),
            ("action_spec", self.action_spec),
            ("environment_spec", self.environment_spec),
            ("acceptance_evidence", self.acceptance_evidence),
        ):
            if not _mapping(value, f"PolicyPackage.{name}"):
                _fail(f"PolicyPackage.{name}", "must not be empty")
        if type(self.action_horizon) is not int or self.action_horizon <= 0:
            _fail("PolicyPackage.action_horizon", "expected a positive integer")
        if type(self.deterministic) is not bool:
            _fail("PolicyPackage.deterministic", "expected boolean")
        if self.schema_version != "policy-package-v1":
            _fail("PolicyPackage.schema_version", "unsupported schema version")

    @staticmethod
    def _validate_optional_pair(
        path: str | None, digest: str | None, name: str
    ) -> None:
        if (path is None) != (digest is None):
            _fail(
                f"PolicyPackage.{name}",
                "path and digest must either both be present or both be absent",
            )
        if path is not None:
            _relative_path(path, f"PolicyPackage.{name}_path")
        if digest is not None:
            _digest(digest, f"PolicyPackage.{name}_sha256")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PolicyPackage:
        """Decode one strict JSON-native package definition."""
        required = {
            "acceptance_evidence",
            "action_horizon",
            "action_spec",
            "checkpoint_path",
            "checkpoint_sha256",
            "deterministic",
            "environment_spec",
            "model_config_path",
            "model_config_sha256",
            "normalization_path",
            "normalization_sha256",
            "observation_spec",
            "policy_family",
            "policy_id",
            "postprocessor_path",
            "postprocessor_sha256",
            "preprocessor_path",
            "preprocessor_sha256",
            "schema_version",
            "source_commit",
        }
        observed = set(value)
        if observed != required:
            missing = sorted(required - observed)
            extra = sorted(observed - required)
            _fail("PolicyPackage", f"field mismatch; missing={missing}, extra={extra}")
        action_horizon = value["action_horizon"]
        deterministic = value["deterministic"]
        if type(action_horizon) is not int:
            _fail("PolicyPackage.action_horizon", "expected integer")
        if type(deterministic) is not bool:
            _fail("PolicyPackage.deterministic", "expected boolean")
        return cls(
            policy_id=_required_text(value["policy_id"], "PolicyPackage.policy_id"),
            policy_family=_required_text(
                value["policy_family"], "PolicyPackage.policy_family"
            ),
            source_commit=_required_text(
                value["source_commit"], "PolicyPackage.source_commit"
            ),
            checkpoint_path=_relative_path(
                value["checkpoint_path"], "PolicyPackage.checkpoint_path"
            ),
            checkpoint_sha256=_digest(
                value["checkpoint_sha256"], "PolicyPackage.checkpoint_sha256"
            ),
            model_config_path=_relative_path(
                value["model_config_path"], "PolicyPackage.model_config_path"
            ),
            model_config_sha256=_digest(
                value["model_config_sha256"], "PolicyPackage.model_config_sha256"
            ),
            preprocessor_path=_optional_path(
                value["preprocessor_path"], "PolicyPackage.preprocessor_path"
            ),
            preprocessor_sha256=_optional_digest(
                value["preprocessor_sha256"], "PolicyPackage.preprocessor_sha256"
            ),
            postprocessor_path=_optional_path(
                value["postprocessor_path"], "PolicyPackage.postprocessor_path"
            ),
            postprocessor_sha256=_optional_digest(
                value["postprocessor_sha256"], "PolicyPackage.postprocessor_sha256"
            ),
            normalization_path=_optional_path(
                value["normalization_path"], "PolicyPackage.normalization_path"
            ),
            normalization_sha256=_optional_digest(
                value["normalization_sha256"], "PolicyPackage.normalization_sha256"
            ),
            observation_spec=_mapping(
                value["observation_spec"], "PolicyPackage.observation_spec"
            ),
            action_spec=_mapping(value["action_spec"], "PolicyPackage.action_spec"),
            environment_spec=_mapping(
                value["environment_spec"], "PolicyPackage.environment_spec"
            ),
            action_horizon=action_horizon,
            deterministic=deterministic,
            acceptance_evidence=_mapping(
                value["acceptance_evidence"], "PolicyPackage.acceptance_evidence"
            ),
            schema_version=_required_text(
                value["schema_version"], "PolicyPackage.schema_version"
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the complete path-portable JSON representation."""
        return {
            "acceptance_evidence": dict(self.acceptance_evidence),
            "action_horizon": self.action_horizon,
            "action_spec": dict(self.action_spec),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "deterministic": self.deterministic,
            "environment_spec": dict(self.environment_spec),
            "model_config_path": self.model_config_path,
            "model_config_sha256": self.model_config_sha256,
            "normalization_path": self.normalization_path,
            "normalization_sha256": self.normalization_sha256,
            "observation_spec": dict(self.observation_spec),
            "policy_family": self.policy_family,
            "policy_id": self.policy_id,
            "postprocessor_path": self.postprocessor_path,
            "postprocessor_sha256": self.postprocessor_sha256,
            "preprocessor_path": self.preprocessor_path,
            "preprocessor_sha256": self.preprocessor_sha256,
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
        }

    @property
    def package_digest(self) -> str:
        """Return a path-independent digest of the full package definition."""
        encoded = canonical_json_bytes(self.to_mapping(), context="PolicyPackageV1")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def require_runtime_components(self) -> None:
        """Reject missing processors or normalization without inventing defaults."""
        missing = [
            name
            for name, value in (
                ("preprocessor", self.preprocessor_path),
                ("postprocessor", self.postprocessor_path),
                ("normalization", self.normalization_path),
            )
            if value is None
        ]
        if missing:
            _fail("PolicyPackage.runtime_components", f"missing required {missing}")

    def verify_artifacts(self, package_root: Path) -> PolicyArtifactVerification:
        """Verify every required runtime file as a regular file with exact bytes."""
        self.require_runtime_components()
        root = Path(package_root).absolute()
        if not root.is_dir() or root.is_symlink():
            _fail("PolicyPackage.package_root", "expected regular package directory")
        artifact_pairs = (
            ("checkpoint", self.checkpoint_path, self.checkpoint_sha256),
            ("model_config", self.model_config_path, self.model_config_sha256),
            ("preprocessor", self.preprocessor_path, self.preprocessor_sha256),
            ("postprocessor", self.postprocessor_path, self.postprocessor_sha256),
            ("normalization", self.normalization_path, self.normalization_sha256),
        )
        observed: dict[str, str] = {}
        for name, relative, expected in artifact_pairs:
            if relative is None or expected is None:
                _fail(f"PolicyPackage.{name}", "required artifact is unbound")
            path = root.joinpath(*PurePosixPath(relative).parts)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise PolicyPackageError(
                    f"PolicyPackage.{name}: path escapes package root"
                ) from exc
            if not path.is_file() or path.is_symlink():
                _fail(f"PolicyPackage.{name}", "expected regular unlinked file")
            digest = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
            if digest != expected:
                _fail(
                    f"PolicyPackage.{name}",
                    f"content digest mismatch: expected {expected}, observed {digest}",
                )
            observed[name] = digest
        additional = self.acceptance_evidence.get("additional_artifacts")
        if additional is not None:
            if not isinstance(additional, Mapping) or not additional:
                _fail(
                    "PolicyPackage.additional_artifacts",
                    "expected non-empty artifact mapping",
                )
            occupied_paths = {
                relative for _, relative, _ in artifact_pairs if relative is not None
            }
            for name, raw in sorted(additional.items()):
                artifact_name = _required_text(
                    name, "PolicyPackage.additional_artifacts.name"
                )
                record = _mapping(
                    raw,
                    f"PolicyPackage.additional_artifacts.{artifact_name}",
                )
                if set(record) != {"path", "sha256"}:
                    _fail(
                        f"PolicyPackage.additional_artifacts.{artifact_name}",
                        "expected path and sha256 only",
                    )
                relative = _relative_path(
                    record["path"],
                    f"PolicyPackage.additional_artifacts.{artifact_name}.path",
                )
                expected = _digest(
                    record["sha256"],
                    f"PolicyPackage.additional_artifacts.{artifact_name}.sha256",
                )
                if relative in occupied_paths or artifact_name in observed:
                    _fail(
                        "PolicyPackage.additional_artifacts",
                        "artifact name or path is duplicate",
                    )
                path = root.joinpath(*PurePosixPath(relative).parts)
                if not path.is_file() or path.is_symlink():
                    _fail(
                        f"PolicyPackage.additional_artifacts.{artifact_name}",
                        "expected regular unlinked file",
                    )
                digest = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
                if digest != expected:
                    _fail(
                        f"PolicyPackage.additional_artifacts.{artifact_name}",
                        "content digest mismatch",
                    )
                occupied_paths.add(relative)
                observed[artifact_name] = digest
        return PolicyArtifactVerification(
            policy_id=self.policy_id,
            verified_artifact_count=len(observed),
            package_digest=self.package_digest,
            artifact_digests=observed,
        )

    def assert_compatible(
        self,
        *,
        observation_spec: Mapping[str, object],
        action_spec: Mapping[str, object],
        environment_spec: Mapping[str, object],
    ) -> None:
        """Require exact bound observation, action, and environment contracts."""
        required = {
            "observation_spec": _mapping(observation_spec, "required observation_spec"),
            "action_spec": _mapping(action_spec, "required action_spec"),
            "environment_spec": _mapping(environment_spec, "required environment_spec"),
        }
        observed = {
            "observation_spec": dict(self.observation_spec),
            "action_spec": dict(self.action_spec),
            "environment_spec": dict(self.environment_spec),
        }
        mismatches = [name for name in required if observed[name] != required[name]]
        if mismatches:
            raise PolicyCompatibilityError(
                "PolicyPackage.compatibility: exact contract mismatch for "
                + ", ".join(mismatches)
            )
