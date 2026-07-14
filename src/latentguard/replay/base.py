"""Narrow typed protocols for simulator-independent exact paired replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from numpy.typing import NDArray

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.replay.models import (
    ReplayCase,
    ReplayExecutionRole,
    ReplayStateReference,
    ReplayTaskReference,
    ReplayTrustDescriptor,
    StateRestorationEvidence,
    TerminalTaskEvidence,
)


class ReplayError(RuntimeError):
    """Base error for exact replay contracts and execution."""


class ReplayValidationError(ReplayError, ValueError):
    """Raised when a replay model, identity, or adapter contract is malformed."""


class ReplayInvalidContextError(ReplayError):
    """Raised when a proposal cannot be compared in its supplied source context."""


class ReplayExecutionError(ReplayError):
    """Raised for replay infrastructure failures rather than task failures."""


@runtime_checkable
class ReplayCaseProvider(Protocol):
    """Resolve one corruption proposal to one fully validated replay case."""

    @property
    def provider_id(self) -> str:
        """Return the stable provider identifier."""
        ...

    @property
    def provider_version(self) -> str:
        """Return the provider semantic version."""
        ...

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        """Resolve one proposal without mutating source or corruption data."""
        ...


@runtime_checkable
class ReplayEnvironmentSession(Protocol):
    """Minimal isolated execution session needed by the generic replay core."""

    def restore_state(
        self, reference: ReplayStateReference
    ) -> StateRestorationEvidence:
        """Restore and verify one opaque content-bound state reference."""
        ...

    def evaluate_task(self, task: ReplayTaskReference) -> TerminalTaskEvidence:
        """Evaluate task evidence at the current session state."""
        ...

    def step_action(self, action: NDArray[Any]) -> None:
        """Execute exactly one validated, detached action row."""
        ...

    def close(self) -> None:
        """Release session resources."""
        ...


@runtime_checkable
class ExactReplayAdapter(Protocol):
    """Stable adapter boundary implemented by fixtures and future simulators."""

    @property
    def adapter_id(self) -> str:
        """Return the stable adapter registry identifier."""
        ...

    @property
    def adapter_version(self) -> str:
        """Return the adapter semantic version."""
        ...

    @property
    def configuration_digest(self) -> str:
        """Return the digest of the resolved semantic configuration."""
        ...

    def trust_descriptor(self) -> ReplayTrustDescriptor:
        """Return the immutable trust ceiling for adapter-produced evidence."""
        ...

    def resolved_configuration(self) -> Mapping[str, object]:
        """Return canonical semantic configuration without runtime metadata."""
        ...

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        """Resolve one proposal to its source action and exact-state reference."""
        ...

    def create_session(
        self,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
    ) -> ReplayEnvironmentSession:
        """Create a fresh isolated session for one paired-execution role."""
        ...


__all__ = [
    "ExactReplayAdapter",
    "ReplayCaseProvider",
    "ReplayEnvironmentSession",
    "ReplayError",
    "ReplayExecutionError",
    "ReplayInvalidContextError",
    "ReplayValidationError",
]
