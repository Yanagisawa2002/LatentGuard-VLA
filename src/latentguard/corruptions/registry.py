"""Duplicate-safe registry and strict factory lookup for action corruptions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from latentguard.corruptions.base import (
    ActionCorruption,
    CorruptionConfigurationError,
    CorruptionError,
)

CorruptionFactory = Callable[[Mapping[str, object]], ActionCorruption]


class DuplicateCorruptionError(CorruptionError):
    """Raised when a stable corruption name is registered more than once."""


class UnknownCorruptionError(CorruptionError):
    """Raised when configuration names a corruption absent from the registry."""


class CorruptionRegistry:
    """A typed registry mapping stable names to strict configuration factories."""

    def __init__(self, entries: Iterable[tuple[str, CorruptionFactory]] = ()) -> None:
        """Create a registry and reject duplicate initial names."""
        self._factories: dict[str, CorruptionFactory] = {}
        for name, factory in entries:
            self.register(name, factory)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in stable insertion order."""
        return tuple(self._factories)

    def register(self, name: str, factory: CorruptionFactory) -> None:
        """Register one factory, rejecting empty or duplicate names."""
        if not isinstance(name, str) or not name.strip():
            raise CorruptionConfigurationError(
                "CorruptionRegistry.name: must be a non-empty string"
            )
        if not callable(factory):
            raise CorruptionConfigurationError(
                f"CorruptionRegistry[{name!r}].factory: must be callable"
            )
        if name in self._factories:
            raise DuplicateCorruptionError(
                f"CorruptionRegistry: duplicate corruption name {name!r}"
            )
        self._factories[name] = factory

    def create(self, name: str, parameters: Mapping[str, object]) -> ActionCorruption:
        """Create a validated corruption or fail clearly for an unknown name."""
        try:
            factory = self._factories[name]
        except (KeyError, TypeError) as exc:
            available = ", ".join(self.names) or "<none>"
            raise UnknownCorruptionError(
                f"unknown corruption {name!r}; available: {available}"
            ) from exc
        if not isinstance(parameters, Mapping):
            raise CorruptionConfigurationError(f"{name}.parameters: expected an object")
        corruption = factory(parameters)
        if corruption.name != name:
            raise CorruptionConfigurationError(
                f"CorruptionRegistry[{name!r}]: factory returned "
                f"mismatched name {corruption.name!r}"
            )
        return corruption


def default_corruption_registry() -> CorruptionRegistry:
    """Return a fresh registry containing all M1 single-source corruptions."""
    from latentguard.corruptions.transforms import BUILTIN_CORRUPTION_FACTORIES

    return CorruptionRegistry(BUILTIN_CORRUPTION_FACTORIES.items())


def create_corruption(name: str, parameters: Mapping[str, object]) -> ActionCorruption:
    """Create one built-in corruption from strict serialized parameters."""
    return default_corruption_registry().create(name, parameters)
