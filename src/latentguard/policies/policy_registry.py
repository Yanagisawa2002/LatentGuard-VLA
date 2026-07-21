"""Audited registry that exposes only verified compatible policy packages."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, cast

from latentguard.policies.policy_package import PolicyPackage
from latentguard.replay.identity import canonical_json_bytes, canonical_json_value


class PolicyRegistryError(ValueError):
    """Raised when an audited policy registry is malformed or unsafe to use."""


class PolicyCompatibilityStatus(StrEnum):
    """Closed set of policy-asset compatibility audit results."""

    ACCEPTED_COMPATIBLE = "ACCEPTED_COMPATIBLE"
    PRESENT_BUT_UNBOUND = "PRESENT_BUT_UNBOUND"
    INCOMPATIBLE_ENVIRONMENT = "INCOMPATIBLE_ENVIRONMENT"
    INCOMPLETE_ASSET = "INCOMPLETE_ASSET"
    UNVERIFIED = "UNVERIFIED"
    REJECTED = "REJECTED"


def _fail(context: str, reason: str) -> NoReturn:
    raise PolicyRegistryError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected non-empty canonical text")
    return value


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(context, "expected array of text values")
    result = tuple(
        _text(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        _fail(context, "duplicate values are unsupported")
    return result


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    canonical = canonical_json_value(value, context=context)
    if not isinstance(canonical, dict):
        _fail(context, "expected JSON object")
    return cast(dict[str, object], canonical)


@dataclass(frozen=True)
class PolicyRegistryEntry:
    """One complete audit record, whether accepted or rejected."""

    policy_id: str
    policy_name: str
    policy_family: str
    source_repository: str
    compatibility_status: PolicyCompatibilityStatus
    missing_assets: tuple[str, ...]
    compatibility_reasons: tuple[str, ...]
    known_evaluation_result: Mapping[str, object]
    audit_record: Mapping[str, object]
    package: PolicyPackage | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("policy_id", self.policy_id),
            ("policy_name", self.policy_name),
            ("policy_family", self.policy_family),
            ("source_repository", self.source_repository),
        ):
            _text(value, f"PolicyRegistryEntry.{name}")
        _mapping(self.known_evaluation_result, "known_evaluation_result")
        _mapping(self.audit_record, "audit_record")
        if self.compatibility_status is PolicyCompatibilityStatus.ACCEPTED_COMPATIBLE:
            if self.package is None:
                _fail(self.policy_id, "accepted entry requires a PolicyPackage")
            if self.missing_assets or self.compatibility_reasons:
                _fail(self.policy_id, "accepted entry cannot retain blockers")
        if self.package is not None and self.package.policy_id != self.policy_id:
            _fail(self.policy_id, "package policy_id does not match registry entry")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PolicyRegistryEntry:
        """Decode one strict registry audit entry."""
        expected = {
            "audit_record",
            "compatibility_reasons",
            "compatibility_status",
            "known_evaluation_result",
            "missing_assets",
            "package",
            "policy_family",
            "policy_id",
            "policy_name",
            "source_repository",
        }
        if set(value) != expected:
            _fail("PolicyRegistryEntry", "field mismatch")
        status_text = _text(value["compatibility_status"], "compatibility_status")
        try:
            status = PolicyCompatibilityStatus(status_text)
        except ValueError as exc:
            raise PolicyRegistryError(
                f"compatibility_status: unsupported value {status_text!r}"
            ) from exc
        package_value = value["package"]
        package = None
        if package_value is not None:
            package = PolicyPackage.from_mapping(_mapping(package_value, "package"))
        return cls(
            policy_id=_text(value["policy_id"], "policy_id"),
            policy_name=_text(value["policy_name"], "policy_name"),
            policy_family=_text(value["policy_family"], "policy_family"),
            source_repository=_text(value["source_repository"], "source_repository"),
            compatibility_status=status,
            missing_assets=_string_tuple(value["missing_assets"], "missing_assets"),
            compatibility_reasons=_string_tuple(
                value["compatibility_reasons"], "compatibility_reasons"
            ),
            known_evaluation_result=_mapping(
                value["known_evaluation_result"], "known_evaluation_result"
            ),
            audit_record=_mapping(value["audit_record"], "audit_record"),
            package=package,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical JSON-native audit entry."""
        return {
            "audit_record": dict(self.audit_record),
            "compatibility_reasons": list(self.compatibility_reasons),
            "compatibility_status": self.compatibility_status.value,
            "known_evaluation_result": dict(self.known_evaluation_result),
            "missing_assets": list(self.missing_assets),
            "package": None if self.package is None else self.package.to_mapping(),
            "policy_family": self.policy_family,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "source_repository": self.source_repository,
        }


@dataclass(frozen=True)
class PolicyRegistry:
    """Content-bound collection of accepted and rejected policy audits."""

    entries: tuple[PolicyRegistryEntry, ...]
    required_observation_spec: Mapping[str, object]
    required_action_spec: Mapping[str, object]
    required_environment_spec: Mapping[str, object]
    result: str
    schema_version: str = "wm-v0-d2-policy-registry-v1"

    def __post_init__(self) -> None:
        if not self.entries:
            _fail("PolicyRegistry.entries", "at least one audited entry is required")
        identifiers = [entry.policy_id for entry in self.entries]
        if len(set(identifiers)) != len(identifiers):
            _fail("PolicyRegistry.entries", "policy_id values must be unique")
        for name, value in (
            ("required_observation_spec", self.required_observation_spec),
            ("required_action_spec", self.required_action_spec),
            ("required_environment_spec", self.required_environment_spec),
        ):
            if not _mapping(value, name):
                _fail(name, "must not be empty")
        if self.result not in ("RESULT_A", "RESULT_B"):
            _fail("PolicyRegistry.result", "expected RESULT_A or RESULT_B")
        if self.schema_version != "wm-v0-d2-policy-registry-v1":
            _fail("PolicyRegistry.schema_version", "unsupported schema version")
        if self.result == "RESULT_A" and self.accepted_count < 1:
            _fail("PolicyRegistry.result", "RESULT_A requires an accepted policy")
        if self.result == "RESULT_B" and self.accepted_count != 0:
            _fail("PolicyRegistry.result", "RESULT_B cannot contain accepted policies")

    @property
    def accepted_count(self) -> int:
        """Return the number of explicitly accepted compatible entries."""
        return sum(
            entry.compatibility_status is PolicyCompatibilityStatus.ACCEPTED_COMPATIBLE
            for entry in self.entries
        )

    @property
    def registry_digest(self) -> str:
        """Return a canonical content digest of the registry body."""
        encoded = canonical_json_bytes(self.to_mapping(), context="PolicyRegistryV1")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PolicyRegistry:
        """Decode and validate one persisted D2 registry."""
        expected = {
            "entries",
            "required_action_spec",
            "required_environment_spec",
            "required_observation_spec",
            "result",
            "schema_version",
        }
        if set(value) != expected:
            _fail("PolicyRegistry", "field mismatch")
        entries_value = value["entries"]
        if not isinstance(entries_value, Sequence) or isinstance(
            entries_value, (str, bytes)
        ):
            _fail("PolicyRegistry.entries", "expected array")
        entries = tuple(
            PolicyRegistryEntry.from_mapping(_mapping(entry, f"entries[{index}]"))
            for index, entry in enumerate(entries_value)
        )
        return cls(
            entries=entries,
            required_observation_spec=_mapping(
                value["required_observation_spec"], "required_observation_spec"
            ),
            required_action_spec=_mapping(
                value["required_action_spec"], "required_action_spec"
            ),
            required_environment_spec=_mapping(
                value["required_environment_spec"], "required_environment_spec"
            ),
            result=_text(value["result"], "result"),
            schema_version=_text(value["schema_version"], "schema_version"),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the strict persisted registry body."""
        return {
            "entries": [entry.to_mapping() for entry in self.entries],
            "required_action_spec": dict(self.required_action_spec),
            "required_environment_spec": dict(self.required_environment_spec),
            "required_observation_spec": dict(self.required_observation_spec),
            "result": self.result,
            "schema_version": self.schema_version,
        }

    def accepted_package(self, policy_id: str, package_root: Path) -> PolicyPackage:
        """Return one accepted package only after exact contract and byte checks."""
        matches = [entry for entry in self.entries if entry.policy_id == policy_id]
        if not matches:
            _fail("PolicyRegistry.accepted_package", f"unknown policy_id {policy_id!r}")
        entry = matches[0]
        if (
            entry.compatibility_status
            is not PolicyCompatibilityStatus.ACCEPTED_COMPATIBLE
        ):
            _fail(
                "PolicyRegistry.accepted_package",
                f"{policy_id!r} has status {entry.compatibility_status.value}",
            )
        if entry.package is None:
            _fail("PolicyRegistry.accepted_package", "accepted package is missing")
        entry.package.assert_compatible(
            observation_spec=self.required_observation_spec,
            action_spec=self.required_action_spec,
            environment_spec=self.required_environment_spec,
        )
        entry.package.verify_artifacts(package_root)
        return entry.package
