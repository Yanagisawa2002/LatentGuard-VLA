"""Classification of RoboLab recorded-config overlay skips."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class OverlaySkipCategory(StrEnum):
    """Stable categories emitted by the LG-RB0.1 upstream patch."""

    EXPECTED_RUNTIME_PRESERVATION = "EXPECTED_RUNTIME_PRESERVATION"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    INVALID_RECORDED_VALUE = "INVALID_RECORDED_VALUE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class OverlaySkipReport:
    """Categorized overlay skips with a fail-closed fatality decision."""

    expected_runtime_preservation: tuple[str, ...]
    schema_drift: tuple[str, ...]
    invalid_recorded_value: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def fatal(self) -> bool:
        """Return whether any unexpected or invalid skip occurred."""
        return bool(self.schema_drift or self.invalid_recorded_value or self.unknown)

    @property
    def all_skips(self) -> tuple[str, ...]:
        """Return every skip in stable category order."""
        return (
            self.expected_runtime_preservation
            + self.schema_drift
            + self.invalid_recorded_value
            + self.unknown
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this report for compact replay evidence."""
        return {
            "expected_runtime_preservation": list(self.expected_runtime_preservation),
            "schema_drift": list(self.schema_drift),
            "invalid_recorded_value": list(self.invalid_recorded_value),
            "unknown": list(self.unknown),
            "fatal": self.fatal,
            "unexpected_skip_count": (
                len(self.schema_drift)
                + len(self.invalid_recorded_value)
                + len(self.unknown)
            ),
        }


def classify_overlay_skips(skips: list[str]) -> OverlaySkipReport:
    """Parse exact patch categories and treat unclassified skips as fatal."""
    categorized: dict[OverlaySkipCategory, list[str]] = {
        category: [] for category in OverlaySkipCategory
    }
    for skip in skips:
        matched = False
        for category in (
            OverlaySkipCategory.EXPECTED_RUNTIME_PRESERVATION,
            OverlaySkipCategory.SCHEMA_DRIFT,
            OverlaySkipCategory.INVALID_RECORDED_VALUE,
        ):
            if skip.startswith(f"[{category.value}] "):
                categorized[category].append(skip)
                matched = True
                break
        if not matched:
            categorized[OverlaySkipCategory.UNKNOWN].append(skip)
    return OverlaySkipReport(
        expected_runtime_preservation=tuple(
            categorized[OverlaySkipCategory.EXPECTED_RUNTIME_PRESERVATION]
        ),
        schema_drift=tuple(categorized[OverlaySkipCategory.SCHEMA_DRIFT]),
        invalid_recorded_value=tuple(
            categorized[OverlaySkipCategory.INVALID_RECORDED_VALUE]
        ),
        unknown=tuple(categorized[OverlaySkipCategory.UNKNOWN]),
    )
