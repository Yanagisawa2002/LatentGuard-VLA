"""Explicit, mutation-safe semantic layouts for action vectors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import StrEnum
from types import MappingProxyType

from latentguard.models import JsonScalar

ACTION_LAYOUT_SCHEMA_VERSION = "1.0"
"""Schema version used by M1 action layouts."""


class ActionLayoutError(ValueError):
    """Raised when an action layout is malformed or a selection is invalid."""


class ActionSemantic(StrEnum):
    """Declared meaning of an action field without assuming index positions."""

    TRANSLATION = "translation"
    ROTATION = "rotation"
    GRIPPER = "gripper"
    AUXILIARY = "auxiliary"
    UNSPECIFIED = "unspecified"


def _require_text(value: object, context: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ActionLayoutError(f"{context}: must be a non-empty string")


def _freeze_metadata(
    value: Mapping[str, JsonScalar], context: str
) -> Mapping[str, JsonScalar]:
    if not isinstance(value, Mapping):
        raise ActionLayoutError(f"{context}: must be a mapping")
    frozen: dict[str, JsonScalar] = {}
    for key, item in value.items():
        _require_text(key, f"{context}.key")
        if item is not None and type(item) not in (str, int, float, bool):
            raise ActionLayoutError(
                f"{context}.{key}: must be a JSON scalar, got {type(item).__name__}"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise ActionLayoutError(f"{context}.{key}: must be finite")
        frozen[key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ActionField:
    """One named semantic field mapped to explicit action-vector indices."""

    name: str
    indices: tuple[int, ...]
    semantic: ActionSemantic
    units: str | None = None
    description: str | None = None
    metadata: Mapping[str, JsonScalar] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach mutable inputs and validate the field definition."""
        try:
            indices = tuple(self.indices)
        except TypeError as exc:
            raise ActionLayoutError(
                f"ActionField[{self.name!r}].indices: must be an integer sequence"
            ) from exc
        object.__setattr__(self, "indices", indices)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, f"ActionField[{self.name!r}].metadata"),
        )
        _validate_action_field(self)


def _validate_action_field(action_field: ActionField) -> None:
    context = f"ActionField[{action_field.name!r}]"
    _require_text(action_field.name, f"{context}.name")
    if not isinstance(action_field.semantic, ActionSemantic):
        raise ActionLayoutError(
            f"{context}.semantic: expected ActionSemantic, "
            f"got {action_field.semantic!r}"
        )
    _require_text(action_field.units, f"{context}.units", optional=True)
    _require_text(action_field.description, f"{context}.description", optional=True)
    if not action_field.indices:
        raise ActionLayoutError(f"{context}.indices: must contain at least one index")
    seen: set[int] = set()
    for index in action_field.indices:
        if type(index) is not int:
            raise ActionLayoutError(
                f"{context}.indices: expected integers, got {index!r}"
            )
        if index < 0:
            raise ActionLayoutError(
                f"{context}.indices: index {index} must be non-negative"
            )
        if index in seen:
            raise ActionLayoutError(
                f"{context}.indices: duplicate index {index} within field"
            )
        seen.add(index)


@dataclass(frozen=True, slots=True)
class ActionLayout:
    """An explicit semantic partition of some or all action dimensions."""

    action_dim: int
    fields: tuple[ActionField, ...]
    schema_version: str = ACTION_LAYOUT_SCHEMA_VERSION
    description: str | None = None
    metadata: Mapping[str, JsonScalar] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach mutable inputs and validate all fields and index ownership."""
        try:
            fields = tuple(self.fields)
        except TypeError as exc:
            raise ActionLayoutError(
                "ActionLayout.fields: must be a field sequence"
            ) from exc
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "ActionLayout.metadata"),
        )
        validate_action_layout(self)

    def field(self, name: str) -> ActionField:
        """Return a named field or raise a descriptive selection error."""
        _require_text(name, "ActionLayout.field.name")
        for action_field in self.fields:
            if action_field.name == name:
                return action_field
        available = ", ".join(field.name for field in self.fields) or "<none>"
        raise ActionLayoutError(
            f"ActionLayout.field: unknown field {name!r}; available: {available}"
        )

    def indices_for_fields(self, names: Sequence[str]) -> tuple[int, ...]:
        """Resolve named fields to indices while preserving declared name order."""
        selected = tuple(names)
        if not selected:
            raise ActionLayoutError(
                "ActionLayout.indices_for_fields: field selection must not be empty"
            )
        if len(set(selected)) != len(selected):
            raise ActionLayoutError(
                "ActionLayout.indices_for_fields: duplicate field names are unsupported"
            )
        indices: list[int] = []
        for name in selected:
            indices.extend(self.field(name).indices)
        return tuple(indices)

    def resolve_selection(
        self,
        *,
        target_fields: Sequence[str] | None = None,
        target_indices: Sequence[int] | None = None,
        allow_all: bool = False,
    ) -> tuple[int, ...]:
        """Resolve exactly one field/index selection, or all dimensions if allowed."""
        if target_fields is not None and target_indices is not None:
            raise ActionLayoutError(
                "ActionLayout.selection: target_fields and target_indices are ambiguous"
            )
        if target_fields is not None:
            return self.indices_for_fields(target_fields)
        if target_indices is not None:
            indices = tuple(target_indices)
            if not indices:
                raise ActionLayoutError(
                    "ActionLayout.selection: target_indices must not be empty"
                )
            if len(set(indices)) != len(indices):
                raise ActionLayoutError(
                    "ActionLayout.selection: duplicate target indices are unsupported"
                )
            for index in indices:
                if type(index) is not int:
                    raise ActionLayoutError(
                        f"ActionLayout.selection: expected integer index, got {index!r}"
                    )
                if not 0 <= index < self.action_dim:
                    raise ActionLayoutError(
                        f"ActionLayout.selection: index {index} outside "
                        f"[0, {self.action_dim})"
                    )
            return indices
        if allow_all:
            return tuple(range(self.action_dim))
        raise ActionLayoutError(
            "ActionLayout.selection: provide target_fields or target_indices"
        )


def validate_action_layout(layout: ActionLayout) -> None:
    """Validate dimensions, field names, and disjoint explicit indices."""
    if type(layout.action_dim) is not int or layout.action_dim <= 0:
        raise ActionLayoutError(
            "ActionLayout.action_dim: must be a positive integer, got "
            f"{layout.action_dim!r}"
        )
    if layout.schema_version != ACTION_LAYOUT_SCHEMA_VERSION:
        raise ActionLayoutError(
            "ActionLayout.schema_version: unsupported version "
            f"{layout.schema_version!r}; supported: {ACTION_LAYOUT_SCHEMA_VERSION}"
        )
    _require_text(layout.description, "ActionLayout.description", optional=True)
    names: set[str] = set()
    owners: dict[int, str] = {}
    for action_field in layout.fields:
        if not isinstance(action_field, ActionField):
            raise ActionLayoutError(
                "ActionLayout.fields: expected ActionField values, got "
                f"{type(action_field).__name__}"
            )
        _validate_action_field(action_field)
        if action_field.name in names:
            raise ActionLayoutError(
                f"ActionLayout.fields: duplicate field name {action_field.name!r}"
            )
        names.add(action_field.name)
        for index in action_field.indices:
            if index >= layout.action_dim:
                raise ActionLayoutError(
                    "ActionLayout.fields"
                    f"[{action_field.name!r}].indices: index {index} "
                    f"outside [0, {layout.action_dim})"
                )
            if index in owners:
                raise ActionLayoutError(
                    f"ActionLayout.fields: index {index} assigned to both "
                    f"{owners[index]!r} and {action_field.name!r}"
                )
            owners[index] = action_field.name
