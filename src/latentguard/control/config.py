"""Strict checked-in configuration for the M4C control protocol."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from latentguard.control.models import (
    ACTION_HORIZON,
    CONTROL_SCHEMA_VERSION,
    FIXED_VISUAL_BATCH_SIZE,
    content_digest,
)


class ClosedLoopConfigurationError(ValueError):
    """Raised when M4C configuration is malformed or drifts from the contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise ClosedLoopConfigurationError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase sha256 content digest")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _load(path: Path, fields: set[str], context: str) -> Mapping[str, object]:
    try:
        raw = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClosedLoopConfigurationError(f"{context}: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != fields:
        _fail(context, "unexpected or missing fields")
    return cast(Mapping[str, object], raw)


@dataclass(frozen=True, slots=True)
class ClosedLoopConfigurationV1:
    """Frozen horizon, stride, tail, rendering, and termination semantics."""

    candidate_horizon: int
    execution_stride: int
    maximum_control_steps: int
    task_check_cadence: str
    tail_semantic: str
    visual_domains: tuple[str, ...]
    fixed_visual_batch_size: int
    fixed_batch_semantic: str
    candidate_pool_identity: str
    selector_matrix_identity: str
    state_restoration_semantic: str
    state_restoration_tolerance: float
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Enforce the authorized M4C control constants."""

        if self.candidate_horizon != ACTION_HORIZON or self.execution_stride != 4:
            _fail("ClosedLoopConfigurationV1", "M4C requires H=16 and E=4")
        if (
            type(self.maximum_control_steps) is not int
            or self.maximum_control_steps < 16
        ):
            _fail(
                "ClosedLoopConfigurationV1.maximum_control_steps",
                "expected integer >=16",
            )
        if self.task_check_cadence != "after_every_executed_action_v1":
            _fail(
                "ClosedLoopConfigurationV1.task_check_cadence", "unsupported semantic"
            )
        if self.tail_semantic != "last_full_window_then_nominal_residual_v1":
            _fail("ClosedLoopConfigurationV1.tail_semantic", "unsupported semantic")
        domains = tuple(self.visual_domains)
        if domains != ("canonical", "strong_camera_shift", "strong_lighting_shift"):
            _fail("ClosedLoopConfigurationV1.visual_domains", "unexpected domain order")
        if self.fixed_visual_batch_size != FIXED_VISUAL_BATCH_SIZE:
            _fail("ClosedLoopConfigurationV1.fixed_visual_batch_size", "must equal 128")
        if (
            self.fixed_batch_semantic
            != "three_real_round_robin_repeat_to_128_consume_first_three_v1"
        ):
            _fail(
                "ClosedLoopConfigurationV1.fixed_batch_semantic", "unsupported semantic"
            )
        for name in ("candidate_pool_identity", "selector_matrix_identity"):
            _digest(getattr(self, name), f"ClosedLoopConfigurationV1.{name}")
        if self.state_restoration_semantic != "tolerance_verified_full_state_v1":
            _fail(
                "ClosedLoopConfigurationV1.state_restoration_semantic",
                "unsupported semantic",
            )
        if type(self.state_restoration_tolerance) not in (
            int,
            float,
        ) or not math.isclose(
            float(self.state_restoration_tolerance), 1e-6, rel_tol=0.0, abs_tol=0.0
        ):
            _fail(
                "ClosedLoopConfigurationV1.state_restoration_tolerance",
                "must equal 1e-6",
            )
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("ClosedLoopConfigurationV1.schema_version", "unsupported version")
        object.__setattr__(self, "visual_domains", domains)

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON-ready configuration content."""

        return {
            "candidate_horizon": self.candidate_horizon,
            "candidate_pool_identity": self.candidate_pool_identity,
            "execution_stride": self.execution_stride,
            "fixed_batch_semantic": self.fixed_batch_semantic,
            "fixed_visual_batch_size": self.fixed_visual_batch_size,
            "maximum_control_steps": self.maximum_control_steps,
            "schema_version": self.schema_version,
            "selector_matrix_identity": self.selector_matrix_identity,
            "state_restoration_semantic": self.state_restoration_semantic,
            "state_restoration_tolerance": self.state_restoration_tolerance,
            "tail_semantic": self.tail_semantic,
            "task_check_cadence": self.task_check_cadence,
            "visual_domains": list(self.visual_domains),
        }

    @property
    def content_digest(self) -> str:
        """Return the deterministic protocol digest."""

        return content_digest(self.as_mapping(), context="ClosedLoopConfigurationV1")


@dataclass(frozen=True, slots=True)
class SelectorDefinitionV1:
    """One frozen selector role and accepted ensemble binding."""

    selector_id: str
    role: str
    input_semantic: str
    ensemble_size: int
    accepted_report_digest: str
    visual: bool

    def __post_init__(self) -> None:
        """Validate one predeclared selector without accepting outcome identities."""

        for name in ("selector_id", "role", "input_semantic"):
            _text(getattr(self, name), f"SelectorDefinitionV1.{name}")
        if type(self.ensemble_size) is not int or self.ensemble_size < 0:
            _fail("SelectorDefinitionV1.ensemble_size", "expected non-negative int")
        _digest(
            self.accepted_report_digest, "SelectorDefinitionV1.accepted_report_digest"
        )
        if type(self.visual) is not bool:
            _fail("SelectorDefinitionV1.visual", "expected boolean")

    def as_mapping(self) -> dict[str, object]:
        """Return strict selector identity content."""

        return {
            "accepted_report_digest": self.accepted_report_digest,
            "ensemble_size": self.ensemble_size,
            "input_semantic": self.input_semantic,
            "role": self.role,
            "selector_id": self.selector_id,
            "visual": self.visual,
        }


@dataclass(frozen=True, slots=True)
class SelectorMatrixConfigurationV1:
    """Exactly six predeclared selectors and the efficient domain schedule."""

    selectors: tuple[SelectorDefinitionV1, ...]
    nonvisual_episode_count_per_source: int
    visual_episode_count_per_source: int
    visual_domains: tuple[str, ...]
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require four nonvisual plus two three-domain visual selectors."""

        selectors = tuple(self.selectors)
        expected = (
            "fixed_primary_v1",
            "deterministic_random_v1",
            "action_only_ensemble_v1",
            "direct_visual_ensemble_v1",
            "distilled_visual_ensemble_v1",
            "privileged_structured_ensemble_v1",
        )
        if tuple(item.selector_id for item in selectors) != expected:
            _fail(
                "SelectorMatrixConfigurationV1.selectors", "unexpected selector order"
            )
        if sum(item.visual for item in selectors) != 2:
            _fail(
                "SelectorMatrixConfigurationV1.selectors",
                "expected two visual selectors",
            )
        if (
            self.nonvisual_episode_count_per_source != 4
            or self.visual_episode_count_per_source != 6
        ):
            _fail(
                "SelectorMatrixConfigurationV1",
                "expected 4 nonvisual and 6 visual executions",
            )
        domains = tuple(self.visual_domains)
        if domains != ("canonical", "strong_camera_shift", "strong_lighting_shift"):
            _fail("SelectorMatrixConfigurationV1.visual_domains", "unexpected domains")
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("SelectorMatrixConfigurationV1.schema_version", "unsupported version")
        object.__setattr__(self, "selectors", selectors)
        object.__setattr__(self, "visual_domains", domains)

    def as_mapping(self) -> dict[str, object]:
        """Return strict selector-matrix content."""

        return {
            "nonvisual_episode_count_per_source": (
                self.nonvisual_episode_count_per_source
            ),
            "schema_version": self.schema_version,
            "selectors": [item.as_mapping() for item in self.selectors],
            "visual_domains": list(self.visual_domains),
            "visual_episode_count_per_source": self.visual_episode_count_per_source,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete selector-matrix identity."""

        return content_digest(
            self.as_mapping(), context="SelectorMatrixConfigurationV1"
        )


@dataclass(frozen=True, slots=True)
class CandidatePoolBindingV1:
    """Accepted M3C candidate generator and action-contract binding."""

    candidate_pool_configuration_digest: str
    action_contract_digest: str
    candidate_count: int
    id_like_count: int
    shifted_count: int
    exact_source_excluded: bool
    no_clipping_or_repair: bool
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require the exact eight-candidate M3C semantic."""

        for name in ("candidate_pool_configuration_digest", "action_contract_digest"):
            _digest(getattr(self, name), f"CandidatePoolBindingV1.{name}")
        if (self.candidate_count, self.id_like_count, self.shifted_count) != (8, 4, 4):
            _fail("CandidatePoolBindingV1", "expected 8=4+4 candidates")
        if (
            self.exact_source_excluded is not True
            or self.no_clipping_or_repair is not True
        ):
            _fail(
                "CandidatePoolBindingV1", "source exclusion and no repair are mandatory"
            )
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("CandidatePoolBindingV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return strict binding content."""

        return {
            "action_contract_digest": self.action_contract_digest,
            "candidate_count": self.candidate_count,
            "candidate_pool_configuration_digest": (
                self.candidate_pool_configuration_digest
            ),
            "exact_source_excluded": True,
            "id_like_count": self.id_like_count,
            "no_clipping_or_repair": True,
            "schema_version": self.schema_version,
            "shifted_count": self.shifted_count,
        }

    @property
    def content_digest(self) -> str:
        """Return the candidate binding digest."""

        return content_digest(self.as_mapping(), context="CandidatePoolBindingV1")


@dataclass(frozen=True, slots=True)
class SourceSeedScheduleV1:
    """Disjoint deterministic smoke and untouched full seed ranges."""

    smoke_start_seed: int
    smoke_requested_successes: int
    smoke_maximum_attempts: int
    full_start_seed: int
    full_requested_successes: int
    full_maximum_attempts: int
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require separate six-source smoke and sixty-source full ranges."""

        values = (
            self.smoke_start_seed,
            self.smoke_requested_successes,
            self.smoke_maximum_attempts,
            self.full_start_seed,
            self.full_requested_successes,
            self.full_maximum_attempts,
        )
        if any(type(value) is not int or value < 0 for value in values):
            _fail("SourceSeedScheduleV1", "expected non-negative integers")
        if self.smoke_requested_successes != 6 or self.full_requested_successes != 60:
            _fail("SourceSeedScheduleV1", "expected smoke=6 and full=60")
        if self.smoke_maximum_attempts < 6 or self.full_maximum_attempts < 60:
            _fail("SourceSeedScheduleV1", "attempt bounds are too small")
        smoke = set(
            range(
                self.smoke_start_seed,
                self.smoke_start_seed + self.smoke_maximum_attempts,
            )
        )
        full = set(
            range(
                self.full_start_seed, self.full_start_seed + self.full_maximum_attempts
            )
        )
        if smoke & full:
            _fail("SourceSeedScheduleV1", "smoke and full ranges overlap")
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("SourceSeedScheduleV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return strict schedule content."""

        return {
            "full_maximum_attempts": self.full_maximum_attempts,
            "full_requested_successes": self.full_requested_successes,
            "full_start_seed": self.full_start_seed,
            "schema_version": self.schema_version,
            "smoke_maximum_attempts": self.smoke_maximum_attempts,
            "smoke_requested_successes": self.smoke_requested_successes,
            "smoke_start_seed": self.smoke_start_seed,
        }

    @property
    def content_digest(self) -> str:
        """Return the seed-schedule digest."""

        return content_digest(self.as_mapping(), context="SourceSeedScheduleV1")


@dataclass(frozen=True, slots=True)
class BootstrapConfigurationV1:
    """Trajectory-level deterministic bootstrap policy."""

    resamples: int
    seed: int
    confidence_level: float
    sampling_unit: str
    schema_version: str = CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require the predeclared 2,000-resample trajectory protocol."""

        if self.resamples != 2000 or type(self.seed) is not int or self.seed < 0:
            _fail(
                "BootstrapConfigurationV1",
                "expected 2000 resamples and non-negative seed",
            )
        if not math.isclose(
            float(self.confidence_level), 0.95, rel_tol=0.0, abs_tol=0.0
        ):
            _fail("BootstrapConfigurationV1.confidence_level", "must equal 0.95")
        if self.sampling_unit != "source_trajectory_v1":
            _fail("BootstrapConfigurationV1.sampling_unit", "unsupported semantic")
        if self.schema_version != CONTROL_SCHEMA_VERSION:
            _fail("BootstrapConfigurationV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return strict bootstrap content."""

        return {
            "confidence_level": self.confidence_level,
            "resamples": self.resamples,
            "sampling_unit": self.sampling_unit,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }

    @property
    def content_digest(self) -> str:
        """Return the bootstrap configuration digest."""

        return content_digest(self.as_mapping(), context="BootstrapConfigurationV1")


def load_closed_loop_configuration(path: Path) -> ClosedLoopConfigurationV1:
    """Strictly load the closed-loop configuration."""

    fields = {
        "candidate_horizon",
        "candidate_pool_identity",
        "execution_stride",
        "fixed_batch_semantic",
        "fixed_visual_batch_size",
        "maximum_control_steps",
        "schema_version",
        "selector_matrix_identity",
        "state_restoration_semantic",
        "state_restoration_tolerance",
        "tail_semantic",
        "task_check_cadence",
        "visual_domains",
    }
    raw = _load(path, fields, "closed-loop configuration")
    domains = raw["visual_domains"]
    if not isinstance(domains, list):
        _fail("closed-loop configuration.visual_domains", "expected list")
    return ClosedLoopConfigurationV1(
        candidate_horizon=cast(int, raw["candidate_horizon"]),
        execution_stride=cast(int, raw["execution_stride"]),
        maximum_control_steps=cast(int, raw["maximum_control_steps"]),
        task_check_cadence=cast(str, raw["task_check_cadence"]),
        tail_semantic=cast(str, raw["tail_semantic"]),
        visual_domains=tuple(cast(list[str], domains)),
        fixed_visual_batch_size=cast(int, raw["fixed_visual_batch_size"]),
        fixed_batch_semantic=cast(str, raw["fixed_batch_semantic"]),
        candidate_pool_identity=cast(str, raw["candidate_pool_identity"]),
        selector_matrix_identity=cast(str, raw["selector_matrix_identity"]),
        state_restoration_semantic=cast(str, raw["state_restoration_semantic"]),
        state_restoration_tolerance=cast(float, raw["state_restoration_tolerance"]),
        schema_version=cast(str, raw["schema_version"]),
    )


def load_selector_matrix(path: Path) -> SelectorMatrixConfigurationV1:
    """Strictly load exactly six selectors."""

    fields = {
        "nonvisual_episode_count_per_source",
        "schema_version",
        "selectors",
        "visual_domains",
        "visual_episode_count_per_source",
    }
    raw = _load(path, fields, "selector matrix")
    entries = raw["selectors"]
    domains = raw["visual_domains"]
    if not isinstance(entries, list) or not isinstance(domains, list):
        _fail("selector matrix", "selectors and domains must be lists")
    selectors: list[SelectorDefinitionV1] = []
    expected = {
        "accepted_report_digest",
        "ensemble_size",
        "input_semantic",
        "role",
        "selector_id",
        "visual",
    }
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping) or set(item) != expected:
            _fail(f"selector matrix.selectors[{index}]", "unexpected fields")
        selectors.append(SelectorDefinitionV1(**cast(dict[str, object], dict(item))))  # type: ignore[arg-type]
    return SelectorMatrixConfigurationV1(
        selectors=tuple(selectors),
        nonvisual_episode_count_per_source=cast(
            int, raw["nonvisual_episode_count_per_source"]
        ),
        visual_episode_count_per_source=cast(
            int, raw["visual_episode_count_per_source"]
        ),
        visual_domains=tuple(cast(list[str], domains)),
        schema_version=cast(str, raw["schema_version"]),
    )


def load_candidate_pool_binding(path: Path) -> CandidatePoolBindingV1:
    """Strictly load the accepted M3C candidate binding."""

    fields = {
        "action_contract_digest",
        "candidate_count",
        "candidate_pool_configuration_digest",
        "exact_source_excluded",
        "id_like_count",
        "no_clipping_or_repair",
        "schema_version",
        "shifted_count",
    }
    raw = _load(path, fields, "candidate-pool binding")
    return CandidatePoolBindingV1(**cast(dict[str, object], dict(raw)))  # type: ignore[arg-type]


def load_source_seed_schedule(path: Path) -> SourceSeedScheduleV1:
    """Strictly load smoke and untouched full seed ranges."""

    fields = {
        "full_maximum_attempts",
        "full_requested_successes",
        "full_start_seed",
        "schema_version",
        "smoke_maximum_attempts",
        "smoke_requested_successes",
        "smoke_start_seed",
    }
    raw = _load(path, fields, "source seed schedule")
    return SourceSeedScheduleV1(**cast(dict[str, object], dict(raw)))  # type: ignore[arg-type]


def load_bootstrap_configuration(path: Path) -> BootstrapConfigurationV1:
    """Strictly load trajectory bootstrap settings."""

    fields = {
        "confidence_level",
        "resamples",
        "sampling_unit",
        "schema_version",
        "seed",
    }
    raw = _load(path, fields, "bootstrap configuration")
    return BootstrapConfigurationV1(**cast(dict[str, object], dict(raw)))  # type: ignore[arg-type]


__all__ = [
    "BootstrapConfigurationV1",
    "CandidatePoolBindingV1",
    "ClosedLoopConfigurationError",
    "ClosedLoopConfigurationV1",
    "SelectorDefinitionV1",
    "SelectorMatrixConfigurationV1",
    "SourceSeedScheduleV1",
    "load_bootstrap_configuration",
    "load_candidate_pool_binding",
    "load_closed_loop_configuration",
    "load_selector_matrix",
    "load_source_seed_schedule",
]
