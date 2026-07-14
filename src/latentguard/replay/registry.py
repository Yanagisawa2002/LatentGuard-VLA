"""Explicit duplicate-resistant replay adapter registry and safe configuration."""

from __future__ import annotations

import json
import stat
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from latentguard.evaluation.models import (
    EvaluationIdentityError,
    compute_configuration_digest,
    validate_evaluator_identity,
)
from latentguard.replay.base import ExactReplayAdapter, ReplayValidationError
from latentguard.replay.evaluator import (
    ExactStatePairedReplayEvaluator,
    create_exact_state_paired_replay_evaluator,
)
from latentguard.replay.fixture import (
    FIXTURE_ADAPTER_ID,
    create_deterministic_replay_fixture_adapter,
)
from latentguard.replay.identity import canonical_json_value
from latentguard.replay.models import ReplayBundle
from latentguard.replay.source import ReplaySourceBinding
from latentguard.replay.validation import validate_replay_trust_descriptor

ReplayAdapterFactory = Callable[
    [Mapping[str, object], ReplaySourceBinding], ExactReplayAdapter
]


class ReplayAdapterRegistryError(ValueError):
    """Base error for adapter lookup, configuration, and factory contracts."""


class DuplicateReplayAdapterError(ReplayAdapterRegistryError):
    """Raised when one explicit adapter name is registered more than once."""


class UnknownReplayAdapterError(ReplayAdapterRegistryError):
    """Raised when a CLI adapter name is absent from the explicit registry."""


def _reject_json_constant(value: str) -> NoReturn:
    raise ReplayAdapterRegistryError(
        f"ReplayAdapterConfiguration: non-finite JSON constant {value!r} is unsupported"
    )


def _reject_duplicate_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterConfiguration: duplicate field {key!r}"
            )
        result[key] = value
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def load_replay_adapter_configuration(path: Path) -> Mapping[str, object]:
    """Safely load one canonical JSON adapter configuration without code imports."""
    requested = Path(path).absolute()
    try:
        resolved = requested.resolve(strict=True)
        metadata = requested.stat()
    except OSError as exc:
        raise ReplayAdapterRegistryError(
            f"ReplayAdapterConfiguration.path: could not inspect file: {exc}"
        ) from exc
    if requested.is_symlink() or resolved != requested:
        raise ReplayAdapterRegistryError(
            "ReplayAdapterConfiguration.path: symbolic links and junctions are "
            "unsupported"
        )
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ReplayAdapterRegistryError(
            "ReplayAdapterConfiguration.path: expected one unlinked regular file"
        )
    try:
        raw = cast(
            object,
            json.loads(
                requested.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_fields,
            ),
        )
    except ReplayAdapterRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayAdapterRegistryError(
            f"ReplayAdapterConfiguration: could not read strict JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ReplayAdapterRegistryError(
            "ReplayAdapterConfiguration: top-level value must be an object"
        )
    try:
        compute_configuration_digest(raw)
    except EvaluationIdentityError as exc:
        raise ReplayAdapterRegistryError(str(exc)) from exc
    return cast(Mapping[str, object], _freeze_json(raw))


class ReplayAdapterRegistry:
    """Map stable explicit names to trusted adapter construction functions."""

    def __init__(
        self, entries: Iterable[tuple[str, ReplayAdapterFactory]] = ()
    ) -> None:
        """Create a fresh registry and reject duplicate initial entries."""
        self._factories: dict[str, ReplayAdapterFactory] = {}
        for name, factory in entries:
            self.register(name, factory)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in stable insertion order."""
        return tuple(self._factories)

    def register(self, name: str, factory: ReplayAdapterFactory) -> None:
        """Register one explicit adapter factory without dynamic import behavior."""
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or "\x00" in name
        ):
            raise ReplayAdapterRegistryError(
                "ReplayAdapterRegistry.name: expected a canonical non-empty string"
            )
        if not callable(factory):
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterRegistry[{name!r}].factory: expected a callable"
            )
        if name in self._factories:
            raise DuplicateReplayAdapterError(
                f"ReplayAdapterRegistry: duplicate adapter {name!r}"
            )
        self._factories[name] = factory

    def create(
        self,
        name: str,
        configuration: Mapping[str, object],
        source_binding: ReplaySourceBinding,
    ) -> ExactReplayAdapter:
        """Create and fully validate one explicitly registered adapter."""
        try:
            factory = self._factories[name]
        except (KeyError, TypeError) as exc:
            available = ", ".join(self.names) or "<none>"
            raise UnknownReplayAdapterError(
                f"unknown replay adapter {name!r}; available: {available}"
            ) from exc
        if not isinstance(configuration, Mapping):
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterRegistry[{name!r}].configuration: expected a mapping"
            )
        adapter = factory(MappingProxyType(dict(configuration)), source_binding)
        if not isinstance(adapter, ExactReplayAdapter):
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterRegistry[{name!r}]: factory returned an invalid adapter"
            )
        if adapter.adapter_id != name:
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterRegistry[{name!r}]: returned mismatched adapter ID "
                f"{adapter.adapter_id!r}"
            )
        try:
            validate_evaluator_identity(
                evaluator_id=adapter.adapter_id,
                evaluator_version=adapter.adapter_version,
            )
        except EvaluationIdentityError as exc:
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterRegistry[{name!r}]: invalid semantic identity: {exc}"
            ) from exc
        resolved = adapter.resolved_configuration()
        if not isinstance(resolved, Mapping):
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterRegistry[{name!r}]: resolved config must be a mapping"
            )
        try:
            canonical_json_value(
                resolved,
                context=f"ReplayAdapterRegistry[{name!r}].resolved_configuration",
                reject_runtime_paths=True,
            )
        except ReplayValidationError as exc:
            raise ReplayAdapterRegistryError(str(exc)) from exc
        expected_digest = compute_configuration_digest(resolved)
        if adapter.configuration_digest != expected_digest:
            raise ReplayAdapterRegistryError(
                f"ReplayAdapterRegistry[{name!r}]: configuration digest mismatch"
            )
        validate_replay_trust_descriptor(adapter.trust_descriptor())
        return adapter


def default_replay_adapter_registry() -> ReplayAdapterRegistry:
    """Return a fresh registry containing only the non-physical fixture adapter."""
    return ReplayAdapterRegistry(
        ((FIXTURE_ADAPTER_ID, create_deterministic_replay_fixture_adapter),)
    )


def create_replay_adapter(
    name: str,
    configuration: Mapping[str, object],
    source_binding: ReplaySourceBinding,
) -> ExactReplayAdapter:
    """Create one built-in adapter through the explicit default registry."""
    return default_replay_adapter_registry().create(name, configuration, source_binding)


def create_replay_evaluator(
    name: str,
    configuration: Mapping[str, object],
    source_binding: ReplaySourceBinding,
) -> ExactStatePairedReplayEvaluator:
    """Create the adapter, in-memory bundle, and M2A paired replay evaluator."""
    adapter = create_replay_adapter(name, configuration, source_binding)
    bundle = getattr(adapter, "replay_bundle", None)
    if not isinstance(bundle, ReplayBundle):
        raise ReplayAdapterRegistryError(
            f"ReplayAdapterRegistry[{name!r}]: adapter did not expose ReplayBundle"
        )
    return create_exact_state_paired_replay_evaluator(adapter, bundle)


__all__ = [
    "DuplicateReplayAdapterError",
    "ReplayAdapterFactory",
    "ReplayAdapterRegistry",
    "ReplayAdapterRegistryError",
    "UnknownReplayAdapterError",
    "create_replay_adapter",
    "create_replay_evaluator",
    "default_replay_adapter_registry",
    "load_replay_adapter_configuration",
]
