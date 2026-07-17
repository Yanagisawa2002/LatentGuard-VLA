"""Strict duplicate-free configuration loading for M4A visual data."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

MAX_CONFIGURATION_BYTES = 1024 * 1024


class VisionConfigurationError(ValueError):
    """Raised when a visual configuration is malformed or unsupported."""


def reject_json_constant(value: str) -> NoReturn:
    """Reject NaN and infinity during JSON parsing."""

    raise VisionConfigurationError(f"non-finite JSON constant {value!r}")


def reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate keys."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VisionConfigurationError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def require_mapping(value: object, context: str) -> Mapping[str, object]:
    """Require a string-keyed JSON object."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VisionConfigurationError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def require_exact_fields(
    value: Mapping[str, object], expected: Collection[str], context: str
) -> None:
    """Reject missing and unknown fields."""

    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise VisionConfigurationError(f"{context}: {'; '.join(details)}")


def require_text(value: object, context: str) -> str:
    """Require canonical non-empty text without control characters."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise VisionConfigurationError(f"{context}: expected canonical text")
    return value


def require_integer(value: object, context: str, *, minimum: int = 0) -> int:
    """Require an integer at least ``minimum`` without bool coercion."""

    if type(value) is not int or value < minimum:
        raise VisionConfigurationError(f"{context}: expected integer >= {minimum}")
    return value


def require_number(value: object, context: str) -> float:
    """Require a finite JSON number without bool coercion."""

    if type(value) not in (int, float):
        raise VisionConfigurationError(f"{context}: expected finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise VisionConfigurationError(f"{context}: expected finite number")
    return result


def require_list(value: object, context: str) -> list[object]:
    """Require one JSON list."""

    if not isinstance(value, list):
        raise VisionConfigurationError(f"{context}: expected an array")
    return cast(list[object], value)


def require_vector(value: object, size: int, context: str) -> tuple[float, ...]:
    """Require one fixed-size finite numeric vector."""

    items = require_list(value, context)
    if len(items) != size:
        raise VisionConfigurationError(f"{context}: expected {size} values")
    return tuple(
        require_number(item, f"{context}[{index}]") for index, item in enumerate(items)
    )


def require_matrix(
    value: object, rows: int, columns: int, context: str
) -> tuple[tuple[float, ...], ...]:
    """Require one fixed-size finite row-major matrix."""

    raw_rows = require_list(value, context)
    if len(raw_rows) != rows:
        raise VisionConfigurationError(f"{context}: expected {rows} rows")
    return tuple(
        require_vector(row, columns, f"{context}[{index}]")
        for index, row in enumerate(raw_rows)
    )


def require_text_tuple(value: object, context: str) -> tuple[str, ...]:
    """Require a duplicate-free ordered string array."""

    result = tuple(
        require_text(item, f"{context}[{index}]")
        for index, item in enumerate(require_list(value, context))
    )
    if len(result) != len(set(result)):
        raise VisionConfigurationError(f"{context}: duplicate values are invalid")
    return result


def load_strict_json_mapping(path: Path) -> Mapping[str, object]:
    """Load one bounded, regular, duplicate-free UTF-8 JSON object."""

    requested = Path(path).absolute()
    try:
        if (
            requested.is_symlink()
            or not requested.is_file()
            or requested.resolve() != requested
            or requested.stat().st_nlink != 1
        ):
            raise VisionConfigurationError("configuration must be an unlinked file")
        if requested.stat().st_size > MAX_CONFIGURATION_BYTES:
            raise VisionConfigurationError("configuration is unexpectedly large")
        raw = json.loads(
            requested.read_text(encoding="utf-8"),
            parse_constant=reject_json_constant,
            object_pairs_hook=reject_duplicate_fields,
        )
    except VisionConfigurationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisionConfigurationError(
            f"could not read strict configuration: {exc}"
        ) from exc
    return require_mapping(raw, "configuration")


def load_camera_rig_configuration(path: Path) -> object:
    """Load a strict :class:`PickCubeMultiViewRigV1` JSON file."""

    from latentguard.vision_data.cameras import PickCubeMultiViewRigV1

    return PickCubeMultiViewRigV1.from_mapping(load_strict_json_mapping(path))


def load_render_domain_configuration(path: Path) -> object:
    """Load a strict :class:`RenderDomainConfigurationV1` JSON file."""

    from latentguard.vision_data.domains import RenderDomainConfigurationV1

    return RenderDomainConfigurationV1.from_mapping(load_strict_json_mapping(path))


def sequence_mapping(value: Sequence[object]) -> list[object]:
    """Return a JSON-ready list from one immutable sequence."""

    return list(value)


__all__ = [
    "MAX_CONFIGURATION_BYTES",
    "VisionConfigurationError",
    "load_camera_rig_configuration",
    "load_render_domain_configuration",
    "load_strict_json_mapping",
    "reject_duplicate_fields",
    "reject_json_constant",
    "require_exact_fields",
    "require_integer",
    "require_list",
    "require_mapping",
    "require_matrix",
    "require_number",
    "require_text",
    "require_text_tuple",
    "require_vector",
]
