"""Safe evaluator configuration loading and duplicate-resistant registry lookup."""

from __future__ import annotations

import json
import stat
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from latentguard.evaluation.base import (
    EvaluationError,
    EvaluatorConfigurationError,
    ProposalEvaluator,
)
from latentguard.evaluation.models import (
    EvaluationIdentityError,
    compute_configuration_digest,
    validate_evaluator_identity,
)

EvaluatorFactory = Callable[[Mapping[str, object]], ProposalEvaluator]


class DuplicateEvaluatorError(EvaluationError):
    """Raised when an evaluator name is registered more than once."""


class UnknownEvaluatorError(EvaluationError):
    """Raised when an evaluator name is absent from the registry."""


def _raise_json_constant(value: str) -> NoReturn:
    raise EvaluatorConfigurationError(
        f"EvaluatorConfiguration: non-finite JSON constant {value!r} is not allowed"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorConfigurationError(
                f"EvaluatorConfiguration: duplicate object key {key!r}"
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


def load_evaluator_configuration(path: Path) -> Mapping[str, object]:
    """Load a safe immutable JSON object without evaluator-specific interpretation."""
    requested = Path(path).absolute()
    try:
        resolved = requested.resolve(strict=True)
        metadata = requested.stat()
    except OSError as exc:
        raise EvaluatorConfigurationError(
            f"EvaluatorConfiguration.path: could not inspect {requested}: {exc}"
        ) from exc
    if requested.is_symlink() or resolved != requested:
        raise EvaluatorConfigurationError(
            "EvaluatorConfiguration.path: symbolic links, junctions, and "
            "non-canonical paths are unsupported"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise EvaluatorConfigurationError(
            "EvaluatorConfiguration.path: expected a regular file"
        )
    if metadata.st_nlink != 1:
        raise EvaluatorConfigurationError(
            "EvaluatorConfiguration.path: hard-linked files are unsupported"
        )
    try:
        text = requested.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EvaluatorConfigurationError(
            f"EvaluatorConfiguration.path: could not read strict UTF-8 JSON: {exc}"
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_raise_json_constant,
        )
    except EvaluatorConfigurationError:
        raise
    except json.JSONDecodeError as exc:
        raise EvaluatorConfigurationError(
            "EvaluatorConfiguration: invalid JSON at "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise EvaluatorConfigurationError(
            "EvaluatorConfiguration: top-level JSON value must be an object"
        )
    try:
        compute_configuration_digest(value)
    except EvaluationIdentityError as exc:
        raise EvaluatorConfigurationError(str(exc)) from exc
    return cast(Mapping[str, object], _freeze_json(value))


class EvaluatorRegistry:
    """Map stable evaluator names to strict configuration factories."""

    def __init__(self, entries: Iterable[tuple[str, EvaluatorFactory]] = ()) -> None:
        """Create a registry and reject duplicate initial names."""
        self._factories: dict[str, EvaluatorFactory] = {}
        for name, factory in entries:
            self.register(name, factory)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered evaluator names in stable insertion order."""
        return tuple(self._factories)

    def register(self, name: str, factory: EvaluatorFactory) -> None:
        """Register one evaluator factory, rejecting empty and duplicate names."""
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or "\x00" in name
        ):
            raise EvaluatorConfigurationError(
                "EvaluatorRegistry.name: must be a non-empty string without "
                "surrounding whitespace"
            )
        if not callable(factory):
            raise EvaluatorConfigurationError(
                f"EvaluatorRegistry[{name!r}].factory: must be callable"
            )
        if name in self._factories:
            raise DuplicateEvaluatorError(
                f"EvaluatorRegistry: duplicate evaluator name {name!r}"
            )
        self._factories[name] = factory

    def create(
        self, name: str, configuration: Mapping[str, object]
    ) -> ProposalEvaluator:
        """Create a validated evaluator or fail clearly for an unknown name."""
        try:
            factory = self._factories[name]
        except (KeyError, TypeError) as exc:
            available = ", ".join(self.names) or "<none>"
            raise UnknownEvaluatorError(
                f"unknown evaluator {name!r}; available: {available}"
            ) from exc
        if not isinstance(configuration, Mapping):
            raise EvaluatorConfigurationError(
                f"{name}.configuration: expected a JSON object"
            )
        immutable_configuration = MappingProxyType(dict(configuration))
        evaluator = factory(immutable_configuration)
        if not isinstance(evaluator, ProposalEvaluator):
            raise EvaluatorConfigurationError(
                f"EvaluatorRegistry[{name!r}]: factory returned an object that "
                "does not implement ProposalEvaluator"
            )
        if evaluator.evaluator_id != name:
            raise EvaluatorConfigurationError(
                f"EvaluatorRegistry[{name!r}]: factory returned mismatched evaluator "
                f"ID {evaluator.evaluator_id!r}"
            )
        try:
            validate_evaluator_identity(
                evaluator_id=evaluator.evaluator_id,
                evaluator_version=evaluator.evaluator_version,
            )
        except EvaluationIdentityError as exc:
            raise EvaluatorConfigurationError(
                f"EvaluatorRegistry[{name!r}]: invalid evaluator identity: {exc}"
            ) from exc
        resolved = evaluator.resolved_configuration()
        if not isinstance(resolved, Mapping):
            raise EvaluatorConfigurationError(
                f"EvaluatorRegistry[{name!r}]: resolved configuration must be a mapping"
            )
        try:
            expected_digest = compute_configuration_digest(resolved)
        except EvaluationIdentityError as exc:
            raise EvaluatorConfigurationError(
                f"EvaluatorRegistry[{name!r}]: invalid resolved configuration: {exc}"
            ) from exc
        if evaluator.configuration_digest != expected_digest:
            raise EvaluatorConfigurationError(
                f"EvaluatorRegistry[{name!r}]: configuration digest mismatch; "
                f"expected {expected_digest!r}, got {evaluator.configuration_digest!r}"
            )
        return evaluator


def _create_fixture_evaluator(
    configuration: Mapping[str, object],
) -> ProposalEvaluator:
    from latentguard.evaluation.fixture import create_deterministic_fixture_evaluator

    return cast(
        ProposalEvaluator, create_deterministic_fixture_evaluator(configuration)
    )


def default_evaluator_registry() -> EvaluatorRegistry:
    """Return a fresh registry containing the deterministic M2A fixture."""
    return EvaluatorRegistry((("deterministic_fixture", _create_fixture_evaluator),))


def create_evaluator(
    name: str, configuration: Mapping[str, object]
) -> ProposalEvaluator:
    """Create one built-in evaluator from strict serialized configuration."""
    return default_evaluator_registry().create(name, configuration)


__all__ = [
    "DuplicateEvaluatorError",
    "EvaluatorFactory",
    "EvaluatorRegistry",
    "UnknownEvaluatorError",
    "create_evaluator",
    "default_evaluator_registry",
    "load_evaluator_configuration",
]
