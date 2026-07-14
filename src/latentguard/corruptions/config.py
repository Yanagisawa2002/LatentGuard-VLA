"""Strict JSON configuration loading for the M1 corruption engine."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from latentguard.corruptions.base import ActionCorruption
from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.registry import create_corruption
from latentguard.models import JsonScalar

CORRUPTION_CONFIG_SCHEMA_VERSION = "1.0"
"""Configuration schema version supported by M1."""

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "action_layout", "corruptions"})
_LAYOUT_REQUIRED_FIELDS = frozenset({"action_dim", "fields", "schema_version"})
_LAYOUT_OPTIONAL_FIELDS = frozenset({"description", "metadata"})
_FIELD_REQUIRED_FIELDS = frozenset({"name", "indices", "semantic"})
_FIELD_OPTIONAL_FIELDS = frozenset({"units", "description", "metadata"})
_CORRUPTION_FIELDS = frozenset({"type", "parameters"})


class CorruptionPlanError(ValueError):
    """Raised when an M1 corruption configuration is malformed."""


@dataclass(frozen=True, slots=True)
class CorruptionPlan:
    """Validated action layout and ordered corruption definitions."""

    action_layout: ActionLayout
    corruptions: tuple[ActionCorruption, ...]
    schema_version: str = CORRUPTION_CONFIG_SCHEMA_VERSION


def load_corruption_plan(path: Path) -> CorruptionPlan:
    """Load a strict, duplicate-free JSON corruption configuration."""

    config_path = Path(path)
    try:
        if config_path.is_symlink() or not config_path.is_file():
            raise CorruptionPlanError(
                "CorruptionPlan.config: missing or unsafe regular file"
            )
        if config_path.stat().st_nlink != 1:
            raise CorruptionPlanError(
                "CorruptionPlan.config: hard-linked configuration is unsupported"
            )
        raw = cast(
            object,
            json.loads(
                config_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
    except CorruptionPlanError:
        raise
    except Exception as exc:
        raise CorruptionPlanError(
            f"CorruptionPlan.config: could not read safely: {exc}"
        ) from exc

    _validate_finite_json(raw, "CorruptionPlan.config")
    root = _mapping(raw, "CorruptionPlan.config")
    _require_exact_fields(root, _TOP_LEVEL_FIELDS, "CorruptionPlan.config")
    version = _string(root, "schema_version", "CorruptionPlan.config")
    if version != CORRUPTION_CONFIG_SCHEMA_VERSION:
        raise CorruptionPlanError(
            "CorruptionPlan.config.schema_version: unsupported version "
            f"{version!r}; supported: {CORRUPTION_CONFIG_SCHEMA_VERSION}"
        )

    layout = _parse_layout(_field(root, "action_layout", "CorruptionPlan.config"))
    definitions = _list(root, "corruptions", "CorruptionPlan.config")
    if not definitions:
        raise CorruptionPlanError(
            "CorruptionPlan.config.corruptions: must contain at least one definition"
        )
    corruptions = tuple(
        _parse_corruption(definition, layout, index)
        for index, definition in enumerate(definitions)
    )
    return CorruptionPlan(
        action_layout=layout,
        corruptions=corruptions,
        schema_version=version,
    )


def _parse_layout(value: object) -> ActionLayout:
    context = "CorruptionPlan.action_layout"
    item = _mapping(value, context)
    _require_fields(
        item,
        _LAYOUT_REQUIRED_FIELDS,
        _LAYOUT_OPTIONAL_FIELDS,
        context,
    )
    schema_version = _string(item, "schema_version", context)
    if schema_version != CORRUPTION_CONFIG_SCHEMA_VERSION:
        raise CorruptionPlanError(
            f"{context}.schema_version: unsupported version {schema_version!r}"
        )
    action_dim = _integer(item, "action_dim", context)
    fields = tuple(
        _parse_action_field(field, index)
        for index, field in enumerate(_list(item, "fields", context))
    )
    metadata = _metadata(item.get("metadata", {}), f"{context}.metadata")
    description = _optional_string(item.get("description"), f"{context}.description")
    try:
        return ActionLayout(
            action_dim=action_dim,
            fields=fields,
            schema_version=schema_version,
            description=description,
            metadata=metadata,
        )
    except (TypeError, ValueError) as exc:
        raise CorruptionPlanError(f"{context}: {exc}") from exc


def _parse_action_field(value: object, index: int) -> ActionField:
    context = f"CorruptionPlan.action_layout.fields[index={index}]"
    item = _mapping(value, context)
    _require_fields(
        item,
        _FIELD_REQUIRED_FIELDS,
        _FIELD_OPTIONAL_FIELDS,
        context,
    )
    semantic_text = _string(item, "semantic", context)
    try:
        semantic = ActionSemantic(semantic_text)
    except ValueError as exc:
        raise CorruptionPlanError(
            f"{context}.semantic: unsupported semantic {semantic_text!r}"
        ) from exc
    indices = tuple(
        _integer_value(value, f"{context}.indices[index={position}]")
        for position, value in enumerate(_list(item, "indices", context))
    )
    metadata = _metadata(item.get("metadata", {}), f"{context}.metadata")
    units = _optional_string(item.get("units"), f"{context}.units")
    description = _optional_string(item.get("description"), f"{context}.description")
    try:
        return ActionField(
            name=_string(item, "name", context),
            indices=indices,
            semantic=semantic,
            units=units,
            description=description,
            metadata=metadata,
        )
    except (TypeError, ValueError) as exc:
        raise CorruptionPlanError(f"{context}: {exc}") from exc


def _parse_corruption(
    value: object, layout: ActionLayout, index: int
) -> ActionCorruption:
    context = f"CorruptionPlan.corruptions[index={index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _CORRUPTION_FIELDS, context)
    name = _string(item, "type", context)
    parameters = _mapping(_field(item, "parameters", context), f"{context}.parameters")
    _validate_explicit_indices(parameters, layout, context)
    try:
        return create_corruption(name, parameters)
    except (TypeError, ValueError) as exc:
        raise CorruptionPlanError(f"{context}: {exc}") from exc


def _validate_explicit_indices(
    parameters: Mapping[str, object], layout: ActionLayout, context: str
) -> None:
    if "target_indices" not in parameters:
        return
    raw_indices = parameters["target_indices"]
    if not isinstance(raw_indices, list) or not raw_indices:
        raise CorruptionPlanError(
            f"{context}.parameters.target_indices: expected a non-empty array"
        )
    indices = [
        _integer_value(value, f"{context}.parameters.target_indices[index={index}]")
        for index, value in enumerate(raw_indices)
    ]
    if len(set(indices)) != len(indices):
        raise CorruptionPlanError(
            f"{context}.parameters.target_indices: duplicate indices are unsupported"
        )
    for index in indices:
        if not 0 <= index < layout.action_dim:
            raise CorruptionPlanError(
                f"{context}.parameters.target_indices: index {index} is outside "
                f"[0, {layout.action_dim})"
            )


def _metadata(value: object, context: str) -> Mapping[str, JsonScalar]:
    item = _mapping(value, context)
    result: dict[str, JsonScalar] = {}
    for key, metadata_value in item.items():
        if not key:
            raise CorruptionPlanError(f"{context}: keys must be non-empty strings")
        if metadata_value is not None and type(metadata_value) not in (
            str,
            int,
            float,
            bool,
        ):
            raise CorruptionPlanError(
                f"{context}.{key}: expected a JSON scalar or null"
            )
        result[key] = cast(JsonScalar, metadata_value)
    return result


def _validate_finite_json(value: object, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CorruptionPlanError(f"{context}: contains NaN or infinity")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{context}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{context}.{key}")


def _require_fields(
    item: Mapping[str, object],
    required: frozenset[str],
    optional: frozenset[str],
    context: str,
) -> None:
    missing = sorted(required - set(item))
    unexpected = sorted(set(item) - required - optional)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise CorruptionPlanError(f"{context}: invalid fields ({'; '.join(details)})")


def _require_exact_fields(
    item: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    _require_fields(item, expected, frozenset(), context)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CorruptionPlanError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _field(item: Mapping[str, object], name: str, context: str) -> object:
    if name not in item:
        raise CorruptionPlanError(f"{context}.{name}: missing required field")
    return item[name]


def _string(item: Mapping[str, object], name: str, context: str) -> str:
    value = _field(item, name, context)
    if not isinstance(value, str) or not value:
        raise CorruptionPlanError(f"{context}.{name}: expected a non-empty string")
    return value


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CorruptionPlanError(f"{context}: expected a non-empty string or null")
    return value


def _integer(item: Mapping[str, object], name: str, context: str) -> int:
    return _integer_value(_field(item, name, context), f"{context}.{name}")


def _integer_value(value: object, context: str) -> int:
    if type(value) is not int:
        raise CorruptionPlanError(f"{context}: expected an integer")
    return value


def _list(item: Mapping[str, object], name: str, context: str) -> list[object]:
    value = _field(item, name, context)
    if not isinstance(value, list):
        raise CorruptionPlanError(f"{context}.{name}: expected an array")
    return cast(list[object], value)


def _reject_json_constant(value: str) -> NoReturn:
    raise CorruptionPlanError(f"CorruptionPlan.config: {value!r} is unsupported")


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CorruptionPlanError(f"JSON object: duplicate field {key!r}")
        result[key] = value
    return result
