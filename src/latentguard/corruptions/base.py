"""Typed corruption interface and shared explicit-applicability behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, NoReturn, Protocol, runtime_checkable

from numpy.typing import NDArray

from latentguard.corruptions.layout import ActionLayout
from latentguard.corruptions.models import ResolvedParameterValue
from latentguard.models import ActionChunk
from latentguard.validation import DataValidationError, validate_action_chunk


class CorruptionError(ValueError):
    """Base error for corruption configuration and application failures."""


class CorruptionConfigurationError(CorruptionError):
    """Raised when corruption parameters are malformed or ambiguous."""


class CorruptionApplicabilityError(CorruptionError):
    """Raised when valid parameters cannot apply to a particular action/layout."""

    def __init__(
        self,
        corruption_name: str,
        reason: str,
        *,
        action_shape: tuple[int, ...] | None = None,
        layout_action_dim: int | None = None,
    ) -> None:
        """Build a structured, descriptive applicability failure."""
        self.corruption_name = corruption_name
        self.reason = reason
        self.action_shape = action_shape
        self.layout_action_dim = layout_action_dim
        details: list[str] = []
        if action_shape is not None:
            details.append(f"action_shape={action_shape}")
        if layout_action_dim is not None:
            details.append(f"layout_action_dim={layout_action_dim}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{corruption_name}: not applicable: {reason}{suffix}")


@runtime_checkable
class ActionCorruption(Protocol):
    """Typed interface implemented by deterministic action corruptions."""

    @property
    def name(self) -> str:
        """Return the stable registry name."""
        ...

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return immutable, canonical configuration parameters."""
        ...

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve fields, dimensions, and horizon-dependent parameters."""
        ...

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Raise when this corruption cannot apply without changing parameters."""
        ...

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Return a deterministic, detached transformed action chunk."""
        ...


class BaseActionCorruption(ABC):
    """Shared validation and result construction for built-in corruptions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable registry name."""

    @property
    @abstractmethod
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return immutable, canonical configuration parameters."""

    @abstractmethod
    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve fields, dimensions, and horizon-dependent parameters."""

    @abstractmethod
    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Raise when this corruption is not applicable."""

    @abstractmethod
    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Return the deterministic transformed action."""

    def _validate_common(self, action: ActionChunk, layout: ActionLayout) -> None:
        try:
            validate_action_chunk(action)
        except DataValidationError as exc:
            self._not_applicable(f"invalid source action: {exc}", action, layout)
        if action.actions.shape[1] != layout.action_dim:
            self._not_applicable(
                "action dimension does not match declared layout", action, layout
            )

    def _validate_seed(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise CorruptionConfigurationError(
                f"{self.name}.seed: must be an integer in [0, 2**64)"
            )

    def _result(self, source: ActionChunk, actions: NDArray[Any]) -> ActionChunk:
        return ActionChunk(
            actions=actions,
            coordinate_frame=source.coordinate_frame,
            control_period_s=source.control_period_s,
            schema_version=source.schema_version,
        )

    def _not_applicable(
        self,
        reason: str,
        action: ActionChunk | None = None,
        layout: ActionLayout | None = None,
    ) -> NoReturn:
        raise CorruptionApplicabilityError(
            self.name,
            reason,
            action_shape=(tuple(action.actions.shape) if action is not None else None),
            layout_action_dim=(layout.action_dim if layout is not None else None),
        )
