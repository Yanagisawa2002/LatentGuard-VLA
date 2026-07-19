"""Strict checked-in source and policy matrices for M4D."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from latentguard.control.models import content_digest


class FallbackConfigurationError(ValueError):
    """Raised when a checked-in M4D protocol file drifts."""


def _fail(context: str, reason: str) -> NoReturn:
    raise FallbackConfigurationError(f"{context}: {reason}")


def _read(path: Path, expected: set[str], context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON file")
    try:
        raw = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FallbackConfigurationError(f"{context}: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != expected:
        _fail(context, "unexpected or missing fields")
    return cast(Mapping[str, object], raw)


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected non-empty trimmed text")
    return value


@dataclass(frozen=True, slots=True)
class M4DSourceSeedConfigurationV1:
    """One disjoint successful-source collection range."""

    profile: str
    starting_seed: int
    requested_successes: int
    maximum_attempts: int
    prior_seed_ranges: tuple[str, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Require the exact development or final source count."""

        if self.profile not in {"development", "final"}:
            _fail("M4D source seeds.profile", "unsupported profile")
        expected = 12 if self.profile == "development" else 60
        if self.requested_successes != expected:
            _fail("M4D source seeds", f"{self.profile} requires {expected} sources")
        for name in ("starting_seed", "requested_successes", "maximum_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                _fail(f"M4D source seeds.{name}", "expected positive int")
        if self.maximum_attempts < self.requested_successes:
            _fail("M4D source seeds", "maximum attempts is too small")
        ranges = tuple(self.prior_seed_ranges)
        if not ranges or len(set(ranges)) != len(ranges):
            _fail("M4D source seeds", "prior ranges must be unique")
        for value in ranges:
            _text(value, "M4D source seeds.prior range")
        if self.schema_version != "1.0":
            _fail("M4D source seeds.schema_version", "unsupported version")
        object.__setattr__(self, "prior_seed_ranges", ranges)

    @property
    def content_digest(self) -> str:
        """Return the immutable seed-range identity."""

        return content_digest(
            {
                "maximum_attempts": self.maximum_attempts,
                "prior_seed_ranges": list(self.prior_seed_ranges),
                "profile": self.profile,
                "requested_successes": self.requested_successes,
                "schema_version": self.schema_version,
                "starting_seed": self.starting_seed,
            },
            context="M4DSourceSeedConfigurationV1",
        )


def load_m4d_source_seed_configuration(path: Path) -> M4DSourceSeedConfigurationV1:
    """Strictly load one M4D source range."""

    raw = _read(
        path,
        {
            "maximum_attempts",
            "prior_seed_ranges",
            "profile",
            "requested_successes",
            "schema_version",
            "starting_seed",
        },
        "M4D source seeds",
    )
    ranges = raw["prior_seed_ranges"]
    if not isinstance(ranges, list):
        _fail("M4D source seeds.prior ranges", "expected list")
    return M4DSourceSeedConfigurationV1(
        profile=cast(str, raw["profile"]),
        starting_seed=cast(int, raw["starting_seed"]),
        requested_successes=cast(int, raw["requested_successes"]),
        maximum_attempts=cast(int, raw["maximum_attempts"]),
        prior_seed_ranges=tuple(cast(Sequence[str], ranges)),
        schema_version=cast(str, raw["schema_version"]),
    )


@dataclass(frozen=True, slots=True)
class FallbackPolicyConfigurationV1:
    """One frozen policy role in the development or final matrix."""

    policy_id: str
    mode: str
    scorer_id: str | None
    visual: bool
    gate_profiles: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require scorer/gate applicability without duplicate variants."""

        _text(self.policy_id, "M4D policy.id")
        if self.mode not in {"accept_nominal", "always_fallback", "ungated", "gated"}:
            _fail("M4D policy.mode", "unsupported mode")
        if (self.mode in {"ungated", "gated"}) != (self.scorer_id is not None):
            _fail("M4D policy", "scorer applicability differs")
        if self.scorer_id is not None:
            _text(self.scorer_id, "M4D policy.scorer")
        profiles = tuple(self.gate_profiles)
        if self.mode == "gated":
            if profiles not in {
                ("conservative", "balanced", "responsive"),
                ("selected",),
            }:
                _fail("M4D policy", "gate profile inventory differs")
        elif profiles:
            _fail("M4D policy", "ungated policy declares profiles")
        object.__setattr__(self, "gate_profiles", profiles)


@dataclass(frozen=True, slots=True)
class FallbackSelectorMatrixV1:
    """Exact cost-aware M4D development and final policy matrices."""

    development_policies: tuple[FallbackPolicyConfigurationV1, ...]
    final_policies: tuple[FallbackPolicyConfigurationV1, ...]
    scenarios: tuple[str, ...]
    visual_domains: tuple[str, ...]
    development_visual_domains: tuple[str, ...]
    expected_final_episodes_per_source: int
    primary_policy_id: str
    distillation_ablation_policy_id: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Enforce 16 development variants and the 26-episode final matrix."""

        development = tuple(self.development_policies)
        final = tuple(self.final_policies)
        if len(development) != 8 or len({item.policy_id for item in development}) != 8:
            _fail("M4D matrix", "expected eight development policy definitions")
        expanded = sum(max(1, len(item.gate_profiles)) for item in development)
        if expanded != 16:
            _fail("M4D matrix", "development must expand to sixteen policies")
        if len(final) != 7 or len({item.policy_id for item in final}) != 7:
            _fail("M4D matrix", "expected seven final policies")
        if tuple(self.scenarios) != ("clean", "fault_injected"):
            _fail("M4D matrix", "scenario inventory differs")
        if tuple(self.visual_domains) != (
            "canonical",
            "strong_camera_shift",
            "strong_lighting_shift",
        ):
            _fail("M4D matrix", "visual-domain inventory differs")
        if tuple(self.development_visual_domains) != ("canonical",):
            _fail("M4D matrix", "development must use canonical visual data only")
        nonvisual = sum(not item.visual for item in final)
        visual = sum(item.visual for item in final)
        expected = len(self.scenarios) * (nonvisual + visual * len(self.visual_domains))
        if expected != 26 or self.expected_final_episodes_per_source != 26:
            _fail("M4D matrix", "final matrix must contain 26 episodes per source")
        if self.primary_policy_id != "gated_direct_visual":
            _fail("M4D matrix", "direct visual primary role differs")
        if self.distillation_ablation_policy_id != "gated_distilled_visual":
            _fail("M4D matrix", "distillation ablation role differs")
        if self.schema_version != "1.0":
            _fail("M4D matrix.schema_version", "unsupported version")
        object.__setattr__(self, "development_policies", development)
        object.__setattr__(self, "final_policies", final)
        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(self, "visual_domains", tuple(self.visual_domains))
        object.__setattr__(
            self, "development_visual_domains", tuple(self.development_visual_domains)
        )

    @property
    def content_digest(self) -> str:
        """Return the frozen role and workload identity."""

        return content_digest(
            {
                "development_policies": [
                    _policy_mapping(item) for item in self.development_policies
                ],
                "development_visual_domains": list(self.development_visual_domains),
                "expected_final_episodes_per_source": (
                    self.expected_final_episodes_per_source
                ),
                "final_policies": [
                    _policy_mapping(item) for item in self.final_policies
                ],
                "primary_policy_id": self.primary_policy_id,
                "distillation_ablation_policy_id": (
                    self.distillation_ablation_policy_id
                ),
                "scenarios": list(self.scenarios),
                "schema_version": self.schema_version,
                "visual_domains": list(self.visual_domains),
            },
            context="M4DSelectorMatrixV1",
        )


def _policy_mapping(item: FallbackPolicyConfigurationV1) -> dict[str, object]:
    return {
        "gate_profiles": list(item.gate_profiles),
        "mode": item.mode,
        "policy_id": item.policy_id,
        "scorer_id": item.scorer_id,
        "visual": item.visual,
    }


def load_fallback_selector_matrix(path: Path) -> FallbackSelectorMatrixV1:
    """Strictly load the exact M4D development and final matrices."""

    raw = _read(
        path,
        {
            "development_policies",
            "development_visual_domains",
            "distillation_ablation_policy_id",
            "expected_final_episodes_per_source",
            "final_policies",
            "primary_policy_id",
            "scenarios",
            "schema_version",
            "visual_domains",
        },
        "M4D selector matrix",
    )
    fields = {"gate_profiles", "mode", "policy_id", "scorer_id", "visual"}

    def policies(name: str) -> tuple[FallbackPolicyConfigurationV1, ...]:
        values = raw[name]
        if not isinstance(values, list):
            _fail(f"M4D selector matrix.{name}", "expected list")
        result: list[FallbackPolicyConfigurationV1] = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping) or set(value) != fields:
                _fail(f"M4D selector matrix.{name}[{index}]", "schema differs")
            entry = cast(Mapping[str, object], value)
            profiles = entry["gate_profiles"]
            if not isinstance(profiles, list):
                _fail(f"M4D selector matrix.{name}[{index}]", "profiles differ")
            result.append(
                FallbackPolicyConfigurationV1(
                    policy_id=cast(str, entry["policy_id"]),
                    mode=cast(str, entry["mode"]),
                    scorer_id=cast(str | None, entry["scorer_id"]),
                    visual=cast(bool, entry["visual"]),
                    gate_profiles=tuple(cast(Sequence[str], profiles)),
                )
            )
        return tuple(result)

    def strings(name: str) -> tuple[str, ...]:
        values = raw[name]
        if not isinstance(values, list):
            _fail(f"M4D selector matrix.{name}", "expected list")
        return tuple(cast(Sequence[str], values))

    return FallbackSelectorMatrixV1(
        development_policies=policies("development_policies"),
        final_policies=policies("final_policies"),
        scenarios=strings("scenarios"),
        visual_domains=strings("visual_domains"),
        development_visual_domains=strings("development_visual_domains"),
        expected_final_episodes_per_source=cast(
            int, raw["expected_final_episodes_per_source"]
        ),
        primary_policy_id=cast(str, raw["primary_policy_id"]),
        distillation_ablation_policy_id=cast(
            str, raw["distillation_ablation_policy_id"]
        ),
        schema_version=cast(str, raw["schema_version"]),
    )


__all__ = [
    "FallbackConfigurationError",
    "FallbackPolicyConfigurationV1",
    "FallbackSelectorMatrixV1",
    "M4DSourceSeedConfigurationV1",
    "load_fallback_selector_matrix",
    "load_m4d_source_seed_configuration",
]
