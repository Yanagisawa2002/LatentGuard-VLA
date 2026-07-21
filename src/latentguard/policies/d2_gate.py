"""Strict WM-v0 D2 pilot gate with no synthetic-candidate substitution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn, cast


class D2GateError(ValueError):
    """Raised when a D2 manifest cannot be interpreted safely."""


def _fail(field: str, reason: str) -> NoReturn:
    raise D2GateError(f"D2Gate.{field}: {reason}")


def _integer(manifest: Mapping[str, object], field: str) -> int:
    value = manifest.get(field)
    if type(value) is not int or value < 0:
        _fail(field, "expected non-negative integer")
    return value


def _ratio(manifest: Mapping[str, object], field: str) -> float:
    value = manifest.get(field)
    if type(value) not in (int, float):
        _fail(field, "expected finite ratio")
    result = float(cast(int | float, value))
    if not 0.0 <= result <= 1.0:
        _fail(field, "expected ratio in [0, 1]")
    return result


@dataclass(frozen=True)
class D2GateResult:
    """Named D2 checks and their fail-closed aggregate decision."""

    authorized: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        """Return a reviewable JSON-native result."""
        return {
            "authorized": self.authorized,
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
        }


def evaluate_d2_gate(manifest: Mapping[str, object]) -> D2GateResult:
    """Evaluate every fixed D2 pilot threshold without repairing input."""
    if manifest.get("schema_version") != "wm-v0-d2-dataset-manifest-v1":
        _fail("schema_version", "unsupported schema version")
    checks = {
        "accepted_compatible_policy_package_at_least_1": _integer(
            manifest, "accepted_compatible_policy_count"
        )
        >= 1,
        "policy_generated_candidate_ratio_exactly_100_percent": _ratio(
            manifest, "policy_generated_candidate_ratio"
        )
        == 1.0,
        "valid_sample_count_at_least_150": _integer(manifest, "valid_sample_count")
        >= 150,
        "independent_anchor_count_at_least_50": _integer(
            manifest, "independent_anchor_count"
        )
        >= 50,
        "terminal_success_count_at_least_25": _integer(
            manifest, "terminal_success_count"
        )
        >= 25,
        "terminal_failure_count_at_least_25": _integer(
            manifest, "terminal_failure_count"
        )
        >= 25,
        "terminalized_ratio_at_least_70_percent": _ratio(manifest, "terminalized_ratio")
        >= 0.70,
        "meaningfully_distinct_anchor_ratio_at_least_70_percent": _ratio(
            manifest, "meaningfully_distinct_anchor_ratio"
        )
        >= 0.70,
        "mixed_outcome_anchor_count_at_least_15": _integer(
            manifest, "mixed_outcome_anchor_count"
        )
        >= 15,
        "future_completeness_at_least_99_percent": _ratio(
            manifest, "future_completeness"
        )
        >= 0.99,
        "restoration_mismatch_zero": _integer(manifest, "restoration_mismatch_count")
        == 0,
        "split_leakage_zero": _integer(manifest, "split_leakage_count") == 0,
        "duplicate_identity_zero": _integer(manifest, "duplicate_identity_count") == 0,
        "policy_metadata_completeness_exactly_100_percent": _ratio(
            manifest, "policy_metadata_completeness"
        )
        == 1.0,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return D2GateResult(not failed, checks, failed)
