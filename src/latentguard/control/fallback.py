"""Conservative nominal-accepting fallback shield for M4D."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, Protocol, cast, runtime_checkable

from latentguard.control.models import (
    ClosedLoopCandidatePoolV1,
    ClosedLoopDecisionRecordV1,
    SelectorOutputV1,
    content_digest,
)
from latentguard.control.risk import CandidateRiskScoresV1
from latentguard.control.runner import RuntimeBoundaryV1
from latentguard.control.serialization import write_exclusive_json


class FallbackShieldError(ValueError):
    """Raised when M4D configuration or selection evidence is invalid."""


def _fail(context: str, reason: str) -> NoReturn:
    raise FallbackShieldError(f"{context}: {reason}")


def _read_exact(path: Path, fields: set[str], context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON file")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FallbackShieldError(f"{context}: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(context, "unexpected or missing fields")
    return cast(Mapping[str, object], value)


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected non-empty trimmed text")
    return value


def _digest(value: object, context: str) -> str:
    text = _text(value, context)
    if not text.startswith("sha256:") or len(text) != 71:
        _fail(context, "expected SHA-256 digest")
    return text


def _probability(value: object, context: str) -> float:
    if type(value) not in (int, float):
        _fail(context, "expected finite probability")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        _fail(context, "expected finite probability in [0,1]")
    return result


class NominalScenario(StrEnum):
    """Predeclared upstream nominal-action scenario."""

    CLEAN = "clean"
    FAULT_INJECTED = "fault_injected"


class InterventionType(StrEnum):
    """Mutually exclusive M4D selection categories."""

    ACCEPT_NOMINAL = "accept_nominal"
    OVERRIDE_TO_FIXED_FALLBACK = "override_to_fixed_fallback"
    OVERRIDE_TO_OTHER_ALTERNATIVE = "override_to_other_alternative"


@dataclass(frozen=True, slots=True)
class FaultScheduleConfigurationV1:
    """Fixed content-bound clean/fault nominal designation contract."""

    schedule_seed: int
    fixed_primary_ordinal: int
    moderate_shifted_ordinals: tuple[int, ...]
    severe_ordinals: tuple[int, ...]
    fixed_primary_probability: float
    moderate_shifted_probability: float
    severe_probability: float
    semantic: str
    candidate_pool_configuration_digest: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Require the reviewed 70/15/15 schedule over the eight candidates."""

        if type(self.schedule_seed) is not int or self.schedule_seed < 0:
            _fail("fault schedule.seed", "expected non-negative int")
        if self.fixed_primary_ordinal != 0:
            _fail("fault schedule.fixed primary", "must be accepted ordinal zero")
        moderate = tuple(self.moderate_shifted_ordinals)
        severe = tuple(self.severe_ordinals)
        if not moderate or not severe or set(moderate) & set(severe):
            _fail("fault schedule", "fault subsets must be non-empty and disjoint")
        if set(moderate) | set(severe) | {0} != set(range(8)):
            _fail("fault schedule", "candidate definition inventory differs")
        if (
            self.fixed_primary_probability != 0.70
            or self.moderate_shifted_probability != 0.15
            or self.severe_probability != 0.15
        ):
            _fail("fault schedule", "probabilities must be exactly 70/15/15")
        _text(self.semantic, "fault schedule.semantic")
        _digest(
            self.candidate_pool_configuration_digest,
            "fault schedule.candidate pool digest",
        )
        if self.schema_version != "1.0":
            _fail("fault schedule.schema_version", "unsupported version")
        object.__setattr__(self, "moderate_shifted_ordinals", moderate)
        object.__setattr__(self, "severe_ordinals", severe)

    @property
    def content_digest(self) -> str:
        """Return the path-independent schedule identity."""

        return content_digest(self.as_mapping(), context="M4DFaultScheduleV1")

    def as_mapping(self) -> dict[str, object]:
        """Return strict canonical configuration content."""

        return {
            "candidate_pool_configuration_digest": (
                self.candidate_pool_configuration_digest
            ),
            "fixed_primary_ordinal": self.fixed_primary_ordinal,
            "fixed_primary_probability": self.fixed_primary_probability,
            "moderate_shifted_ordinals": list(self.moderate_shifted_ordinals),
            "moderate_shifted_probability": self.moderate_shifted_probability,
            "schedule_seed": self.schedule_seed,
            "schema_version": self.schema_version,
            "semantic": self.semantic,
            "severe_ordinals": list(self.severe_ordinals),
            "severe_probability": self.severe_probability,
        }


def load_fault_schedule(path: Path) -> FaultScheduleConfigurationV1:
    """Strictly load the reviewed content-bound nominal schedule."""

    raw = _read_exact(
        path,
        {
            "candidate_pool_configuration_digest",
            "fixed_primary_ordinal",
            "fixed_primary_probability",
            "moderate_shifted_ordinals",
            "moderate_shifted_probability",
            "schedule_seed",
            "schema_version",
            "semantic",
            "severe_ordinals",
            "severe_probability",
        },
        "fault schedule",
    )
    moderate = raw["moderate_shifted_ordinals"]
    severe = raw["severe_ordinals"]
    if not isinstance(moderate, list) or not isinstance(severe, list):
        _fail("fault schedule", "ordinal subsets must be lists")
    return FaultScheduleConfigurationV1(
        schedule_seed=cast(int, raw["schedule_seed"]),
        fixed_primary_ordinal=cast(int, raw["fixed_primary_ordinal"]),
        moderate_shifted_ordinals=tuple(cast(Sequence[int], moderate)),
        severe_ordinals=tuple(cast(Sequence[int], severe)),
        fixed_primary_probability=_probability(
            raw["fixed_primary_probability"], "fault schedule.fixed probability"
        ),
        moderate_shifted_probability=_probability(
            raw["moderate_shifted_probability"],
            "fault schedule.moderate probability",
        ),
        severe_probability=_probability(
            raw["severe_probability"], "fault schedule.severe probability"
        ),
        semantic=cast(str, raw["semantic"]),
        candidate_pool_configuration_digest=cast(
            str, raw["candidate_pool_configuration_digest"]
        ),
        schema_version=cast(str, raw["schema_version"]),
    )


@dataclass(frozen=True, slots=True)
class NominalDesignationV1:
    """One pre-execution schedule designation with reporting-only fault label."""

    scenario: NominalScenario
    candidate_id: str
    definition_ordinal: int
    fault_class: str
    injected_fault: bool
    schedule_digest: str


def designate_nominal(
    pool: ClosedLoopCandidatePoolV1,
    *,
    scenario: NominalScenario,
    schedule: FaultScheduleConfigurationV1,
) -> NominalDesignationV1:
    """Choose the same nominal definition for every selector and visual domain."""

    if (
        pool.candidate_pool_configuration_digest
        != schedule.candidate_pool_configuration_digest
    ):
        _fail("nominal designation", "candidate configuration differs")
    if scenario is NominalScenario.CLEAN:
        ordinal = schedule.fixed_primary_ordinal
        fault_class = "clean"
    else:
        canonical = json.dumps(
            {
                "decision_ordinal": pool.decision_ordinal,
                "scenario_id": scenario.value,
                "schedule_seed": schedule.schedule_seed,
                "schedule_semantic": schedule.semantic,
                "source_trajectory_id": pool.source_trajectory_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).digest()
        bucket = int.from_bytes(digest[:8], "big") % 10000
        if bucket < 7000:
            ordinal = schedule.fixed_primary_ordinal
            fault_class = "clean"
        elif bucket < 8500:
            ordinal = schedule.moderate_shifted_ordinals[
                int.from_bytes(digest[8:16], "big")
                % len(schedule.moderate_shifted_ordinals)
            ]
            fault_class = "moderate_or_shifted"
        else:
            ordinal = schedule.severe_ordinals[
                int.from_bytes(digest[8:16], "big") % len(schedule.severe_ordinals)
            ]
            fault_class = "severe"
    candidate = next(
        (item for item in pool.candidates if item.definition_ordinal == ordinal), None
    )
    if candidate is None:
        _fail("nominal designation", "scheduled definition is absent")
    return NominalDesignationV1(
        scenario=scenario,
        candidate_id=candidate.candidate_id,
        definition_ordinal=ordinal,
        fault_class=fault_class,
        injected_fault=ordinal != schedule.fixed_primary_ordinal,
        schedule_digest=schedule.content_digest,
    )


@dataclass(frozen=True, slots=True)
class GateProfileV1:
    """One predeclared conservative override threshold profile."""

    profile_id: str
    nominal_risk_threshold: float
    improvement_margin_threshold: float
    nominal_uncertainty_threshold: float
    alternative_uncertainty_threshold: float
    conservatism_ordinal: int
    quantile_semantic: str

    def __post_init__(self) -> None:
        """Require bounded finite thresholds and a stable order."""

        _text(self.profile_id, "gate profile.id")
        _text(self.quantile_semantic, "gate profile.quantile semantic")
        for name in (
            "nominal_risk_threshold",
            "improvement_margin_threshold",
            "nominal_uncertainty_threshold",
            "alternative_uncertainty_threshold",
        ):
            _probability(getattr(self, name), f"gate profile.{name}")
        if (
            type(self.conservatism_ordinal) is not int
            or not 0 <= self.conservatism_ordinal <= 2
        ):
            _fail("gate profile.conservatism ordinal", "expected 0, 1, or 2")


@dataclass(frozen=True, slots=True)
class GateProfilesConfigurationV1:
    """Exactly three validation-derived profiles frozen before development."""

    profiles: tuple[GateProfileV1, ...]
    validation_prediction_digests: tuple[str, ...]
    derivation_semantic: str
    source_model_id: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Require conservative, balanced, and responsive exactly once."""

        profiles = tuple(self.profiles)
        if tuple(item.profile_id for item in profiles) != (
            "conservative",
            "balanced",
            "responsive",
        ):
            _fail("gate profiles", "expected exactly three ordered profiles")
        if tuple(item.conservatism_ordinal for item in profiles) != (2, 1, 0):
            _fail("gate profiles", "conservatism order differs")
        digests = tuple(self.validation_prediction_digests)
        if len(digests) != 3 or len(set(digests)) != 3:
            _fail("gate profiles", "expected three validation prediction digests")
        for value in digests:
            _digest(value, "gate profiles.validation prediction digest")
        _text(self.derivation_semantic, "gate profiles.derivation semantic")
        _text(self.source_model_id, "gate profiles.source model")
        if self.schema_version != "1.0":
            _fail("gate profiles.schema_version", "unsupported version")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "validation_prediction_digests", digests)

    @property
    def content_digest(self) -> str:
        """Return the immutable profile-set identity."""

        return content_digest(self.as_mapping(), context="M4DGateProfilesV1")

    def as_mapping(self) -> dict[str, object]:
        """Return canonical profile configuration content."""

        return {
            "derivation_semantic": self.derivation_semantic,
            "profiles": [
                {
                    "alternative_uncertainty_threshold": (
                        item.alternative_uncertainty_threshold
                    ),
                    "conservatism_ordinal": item.conservatism_ordinal,
                    "improvement_margin_threshold": (item.improvement_margin_threshold),
                    "nominal_risk_threshold": item.nominal_risk_threshold,
                    "nominal_uncertainty_threshold": (
                        item.nominal_uncertainty_threshold
                    ),
                    "profile_id": item.profile_id,
                    "quantile_semantic": item.quantile_semantic,
                }
                for item in self.profiles
            ],
            "schema_version": self.schema_version,
            "source_model_id": self.source_model_id,
            "validation_prediction_digests": list(self.validation_prediction_digests),
        }

    def profile(self, profile_id: str) -> GateProfileV1:
        """Return one profile by exact ID."""

        matches = [item for item in self.profiles if item.profile_id == profile_id]
        if len(matches) != 1:
            _fail("gate profiles", f"unknown profile {profile_id!r}")
        return matches[0]


def load_gate_profiles(path: Path) -> GateProfilesConfigurationV1:
    """Strictly load exactly three validation-derived gate profiles."""

    raw = _read_exact(
        path,
        {
            "derivation_semantic",
            "profiles",
            "schema_version",
            "source_model_id",
            "validation_prediction_digests",
        },
        "gate profiles",
    )
    entries = raw["profiles"]
    digests = raw["validation_prediction_digests"]
    if not isinstance(entries, list) or not isinstance(digests, list):
        _fail("gate profiles", "profiles and digests must be lists")
    fields = {
        "alternative_uncertainty_threshold",
        "conservatism_ordinal",
        "improvement_margin_threshold",
        "nominal_risk_threshold",
        "nominal_uncertainty_threshold",
        "profile_id",
        "quantile_semantic",
    }
    profiles: list[GateProfileV1] = []
    for index, value in enumerate(entries):
        if not isinstance(value, Mapping) or set(value) != fields:
            _fail(f"gate profiles[{index}]", "unexpected or missing fields")
        item = cast(Mapping[str, object], value)
        profiles.append(
            GateProfileV1(
                profile_id=cast(str, item["profile_id"]),
                nominal_risk_threshold=_probability(
                    item["nominal_risk_threshold"], "gate profile.risk"
                ),
                improvement_margin_threshold=_probability(
                    item["improvement_margin_threshold"], "gate profile.margin"
                ),
                nominal_uncertainty_threshold=_probability(
                    item["nominal_uncertainty_threshold"],
                    "gate profile.nominal uncertainty",
                ),
                alternative_uncertainty_threshold=_probability(
                    item["alternative_uncertainty_threshold"],
                    "gate profile.alternative uncertainty",
                ),
                conservatism_ordinal=cast(int, item["conservatism_ordinal"]),
                quantile_semantic=cast(str, item["quantile_semantic"]),
            )
        )
    return GateProfilesConfigurationV1(
        profiles=tuple(profiles),
        validation_prediction_digests=tuple(cast(Sequence[str], digests)),
        derivation_semantic=cast(str, raw["derivation_semantic"]),
        source_model_id=cast(str, raw["source_model_id"]),
        schema_version=cast(str, raw["schema_version"]),
    )


@dataclass(frozen=True, slots=True)
class ConservativeOverrideDecisionV1:
    """Pure output of the four-condition M4D override gate."""

    selected_candidate_id: str
    best_candidate_id: str
    override: bool
    nominal_risk: float
    best_risk: float
    improvement_margin: float
    nominal_uncertainty: float
    best_uncertainty: float


@dataclass(frozen=True, slots=True)
class ConservativeOverrideGateV1:
    """Accept nominal unless every reviewed risk and confidence gate passes."""

    profile: GateProfileV1

    def decide(
        self, nominal_candidate_id: str, scores: CandidateRiskScoresV1
    ) -> ConservativeOverrideDecisionV1:
        """Apply the conjunctive gate with stable candidate-ID tie breaking."""

        if nominal_candidate_id not in scores.risks:
            _fail("override gate", "nominal candidate is absent from scores")
        best = min(scores.risks, key=lambda item: (scores.risks[item], item))
        nominal_risk = scores.risks[nominal_candidate_id]
        best_risk = scores.risks[best]
        margin = nominal_risk - best_risk
        override = bool(
            best != nominal_candidate_id
            and nominal_risk >= self.profile.nominal_risk_threshold
            and margin >= self.profile.improvement_margin_threshold
            and scores.uncertainties[nominal_candidate_id]
            <= self.profile.nominal_uncertainty_threshold
            and scores.uncertainties[best]
            <= self.profile.alternative_uncertainty_threshold
        )
        return ConservativeOverrideDecisionV1(
            selected_candidate_id=best if override else nominal_candidate_id,
            best_candidate_id=best,
            override=override,
            nominal_risk=nominal_risk,
            best_risk=best_risk,
            improvement_margin=margin,
            nominal_uncertainty=scores.uncertainties[nominal_candidate_id],
            best_uncertainty=scores.uncertainties[best],
        )


@runtime_checkable
class CandidateRiskScorer(Protocol):
    """Frozen outcome-blind scorer used by a fallback policy."""

    selector_id: str
    visual: bool

    def score_candidates(
        self, pool: ClosedLoopCandidatePoolV1, boundary: RuntimeBoundaryV1
    ) -> CandidateRiskScoresV1:
        """Return risks and uncertainties without schedule labels."""
        ...


@dataclass(frozen=True, slots=True)
class FallbackShieldDecisionRecordV1:
    """Content-bound M4D decision sidecar persisted before execution."""

    decision_record_digest: str
    candidate_pool_digest: str
    source_trajectory_id: str
    policy_id: str
    scenario: str
    decision_ordinal: int
    nominal_plan_index: int
    nominal_candidate_id: str
    nominal_definition_ordinal: int
    fixed_fallback_candidate_id: str
    best_candidate_id: str
    selected_candidate_id: str
    nominal_risk: float | None
    best_risk: float | None
    improvement_margin: float | None
    nominal_uncertainty: float | None
    best_uncertainty: float | None
    gate_profile_id: str | None
    gate_decision: str
    intervention_type: InterventionType
    scorer_id: str | None
    scorer_ensemble_identity: str | None
    schedule_digest: str
    fault_class_reporting_only: str
    injected_fault_reporting_only: bool
    outcomes_available_during_selection: bool = False
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Authenticate complete scored records and explicit scorer-free references."""

        for name in (
            "decision_record_digest",
            "candidate_pool_digest",
            "schedule_digest",
        ):
            _digest(getattr(self, name), f"shield decision.{name}")
        for name in (
            "source_trajectory_id",
            "policy_id",
            "scenario",
            "nominal_candidate_id",
            "fixed_fallback_candidate_id",
            "best_candidate_id",
            "selected_candidate_id",
            "gate_decision",
            "fault_class_reporting_only",
        ):
            _text(getattr(self, name), f"shield decision.{name}")
        if self.scenario not in {item.value for item in NominalScenario}:
            _fail("shield decision.scenario", "unsupported scenario")
        for name in ("decision_ordinal", "nominal_plan_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(f"shield decision.{name}", "expected non-negative int")
        if not 0 <= self.nominal_definition_ordinal < 8:
            _fail("shield decision.nominal ordinal", "expected candidate ordinal")
        numeric = (
            self.nominal_risk,
            self.best_risk,
            self.improvement_margin,
            self.nominal_uncertainty,
            self.best_uncertainty,
        )
        if self.scorer_id is None:
            if any(value is not None for value in numeric):
                _fail("shield decision", "scorer-free reference has risk values")
            if self.scorer_ensemble_identity is not None:
                _fail("shield decision", "scorer-free reference has ensemble")
        else:
            _text(self.scorer_id, "shield decision.scorer")
            if self.scorer_ensemble_identity is None:
                _fail("shield decision", "scored decision is missing ensemble")
            _digest(self.scorer_ensemble_identity, "shield decision.ensemble")
            if any(value is None for value in numeric):
                _fail("shield decision", "scored decision is incomplete")
            for value in cast(tuple[float, ...], numeric):
                if not math.isfinite(value) or value < 0.0:
                    _fail("shield decision", "risk values must be finite/non-negative")
        if self.gate_profile_id is not None:
            _text(self.gate_profile_id, "shield decision.gate profile")
        if self.outcomes_available_during_selection is not False:
            _fail("shield decision", "outcomes must be unavailable")
        if self.schema_version != "1.0":
            _fail("shield decision.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return strict JSON content including the sidecar digest."""

        payload: dict[str, object] = {
            "best_candidate_id": self.best_candidate_id,
            "best_risk": self.best_risk,
            "best_uncertainty": self.best_uncertainty,
            "candidate_pool_digest": self.candidate_pool_digest,
            "decision_ordinal": self.decision_ordinal,
            "decision_record_digest": self.decision_record_digest,
            "fault_class_reporting_only": self.fault_class_reporting_only,
            "fixed_fallback_candidate_id": self.fixed_fallback_candidate_id,
            "gate_decision": self.gate_decision,
            "gate_profile_id": self.gate_profile_id,
            "improvement_margin": self.improvement_margin,
            "injected_fault_reporting_only": self.injected_fault_reporting_only,
            "intervention_type": self.intervention_type.value,
            "nominal_candidate_id": self.nominal_candidate_id,
            "nominal_definition_ordinal": self.nominal_definition_ordinal,
            "nominal_plan_index": self.nominal_plan_index,
            "nominal_risk": self.nominal_risk,
            "nominal_uncertainty": self.nominal_uncertainty,
            "outcomes_available_during_selection": False,
            "policy_id": self.policy_id,
            "scenario": self.scenario,
            "schedule_digest": self.schedule_digest,
            "schema_version": self.schema_version,
            "scorer_ensemble_identity": self.scorer_ensemble_identity,
            "scorer_id": self.scorer_id,
            "selected_candidate_id": self.selected_candidate_id,
            "source_trajectory_id": self.source_trajectory_id,
        }
        return {
            **payload,
            "content_digest": content_digest(
                payload, context="FallbackShieldDecisionRecordV1"
            ),
        }


def load_fallback_decision(path: Path) -> FallbackShieldDecisionRecordV1:
    """Strictly reload and authenticate one M4D decision sidecar."""

    fields = {
        "best_candidate_id",
        "best_risk",
        "best_uncertainty",
        "candidate_pool_digest",
        "content_digest",
        "decision_ordinal",
        "decision_record_digest",
        "fault_class_reporting_only",
        "fixed_fallback_candidate_id",
        "gate_decision",
        "gate_profile_id",
        "improvement_margin",
        "injected_fault_reporting_only",
        "intervention_type",
        "nominal_candidate_id",
        "nominal_definition_ordinal",
        "nominal_plan_index",
        "nominal_risk",
        "nominal_uncertainty",
        "outcomes_available_during_selection",
        "policy_id",
        "scenario",
        "schedule_digest",
        "schema_version",
        "scorer_ensemble_identity",
        "scorer_id",
        "selected_candidate_id",
        "source_trajectory_id",
    }
    raw = _read_exact(path, fields, "fallback shield decision")

    def optional_float(name: str) -> float | None:
        value = raw[name]
        if value is None:
            return None
        if type(value) not in (int, float):
            _fail(f"fallback shield decision.{name}", "expected number or null")
        return float(cast(int | float, value))

    result = FallbackShieldDecisionRecordV1(
        decision_record_digest=cast(str, raw["decision_record_digest"]),
        candidate_pool_digest=cast(str, raw["candidate_pool_digest"]),
        source_trajectory_id=cast(str, raw["source_trajectory_id"]),
        policy_id=cast(str, raw["policy_id"]),
        scenario=cast(str, raw["scenario"]),
        decision_ordinal=cast(int, raw["decision_ordinal"]),
        nominal_plan_index=cast(int, raw["nominal_plan_index"]),
        nominal_candidate_id=cast(str, raw["nominal_candidate_id"]),
        nominal_definition_ordinal=cast(int, raw["nominal_definition_ordinal"]),
        fixed_fallback_candidate_id=cast(str, raw["fixed_fallback_candidate_id"]),
        best_candidate_id=cast(str, raw["best_candidate_id"]),
        selected_candidate_id=cast(str, raw["selected_candidate_id"]),
        nominal_risk=optional_float("nominal_risk"),
        best_risk=optional_float("best_risk"),
        improvement_margin=optional_float("improvement_margin"),
        nominal_uncertainty=optional_float("nominal_uncertainty"),
        best_uncertainty=optional_float("best_uncertainty"),
        gate_profile_id=cast(str | None, raw["gate_profile_id"]),
        gate_decision=cast(str, raw["gate_decision"]),
        intervention_type=InterventionType(cast(str, raw["intervention_type"])),
        scorer_id=cast(str | None, raw["scorer_id"]),
        scorer_ensemble_identity=cast(str | None, raw["scorer_ensemble_identity"]),
        schedule_digest=cast(str, raw["schedule_digest"]),
        fault_class_reporting_only=cast(str, raw["fault_class_reporting_only"]),
        injected_fault_reporting_only=cast(bool, raw["injected_fault_reporting_only"]),
        outcomes_available_during_selection=cast(
            bool, raw["outcomes_available_during_selection"]
        ),
        schema_version=cast(str, raw["schema_version"]),
    )
    if raw["content_digest"] != result.as_mapping()["content_digest"]:
        _fail("fallback shield decision", "content digest changed")
    return result


@dataclass(slots=True)
class FallbackShieldSelectorV1:
    """Thin selector wrapper that preserves the M4C pool and execution runner."""

    policy_id: str
    scenario: NominalScenario
    schedule: FaultScheduleConfigurationV1
    mode: str
    scorer: CandidateRiskScorer | None = None
    gate_profile: GateProfileV1 | None = None
    _pending: FallbackShieldDecisionRecordV1 | None = None

    def __post_init__(self) -> None:
        """Validate the four supported selection modes and scorer applicability."""

        _text(self.policy_id, "fallback selector.policy")
        if self.mode not in {"accept_nominal", "always_fallback", "ungated", "gated"}:
            _fail("fallback selector.mode", "unsupported mode")
        if (self.mode in {"ungated", "gated"}) != (self.scorer is not None):
            _fail("fallback selector", "scorer applicability differs")
        if (self.mode == "gated") != (self.gate_profile is not None):
            _fail("fallback selector", "gate profile applicability differs")

    @property
    def selector_id(self) -> str:
        """Return an execution-safe identity binding policy, scenario, and profile."""

        profile = "none" if self.gate_profile is None else self.gate_profile.profile_id
        return f"m4d__{self.policy_id}__{self.scenario.value}__{profile}"

    @property
    def visual(self) -> bool:
        """Return whether the wrapped frozen scorer needs current RGB."""

        return False if self.scorer is None else self.scorer.visual

    def select(
        self, pool: ClosedLoopCandidatePoolV1, boundary: RuntimeBoundaryV1
    ) -> SelectorOutputV1:
        """Finalize an outcome-blind nominal decision and hold its sidecar."""

        designation = designate_nominal(
            pool, scenario=self.scenario, schedule=self.schedule
        )
        fixed = next(
            item
            for item in pool.candidates
            if item.definition_ordinal == self.schedule.fixed_primary_ordinal
        )
        scored: CandidateRiskScoresV1 | None = None
        gate_decision = "not_applicable"
        profile_id: str | None = None
        if self.mode == "accept_nominal":
            selected = designation.candidate_id
            best = selected
        elif self.mode == "always_fallback":
            selected = fixed.candidate_id
            best = selected
        else:
            assert self.scorer is not None
            # The scorer sees only the allowlisted boundary and unchanged pool.
            scored = self.scorer.score_candidates(pool, boundary)
            if set(scored.risks) != set(pool.ordered_candidate_ids):
                _fail("fallback selector", "scorer changed candidate inventory")
            best = min(scored.risks, key=lambda item: (scored.risks[item], item))
            if self.mode == "ungated":
                selected = best
            else:
                assert self.gate_profile is not None
                decision = ConservativeOverrideGateV1(self.gate_profile).decide(
                    designation.candidate_id, scored
                )
                selected = decision.selected_candidate_id
                best = decision.best_candidate_id
                gate_decision = "override" if decision.override else "accept_nominal"
                profile_id = self.gate_profile.profile_id
        if selected == designation.candidate_id:
            intervention = InterventionType.ACCEPT_NOMINAL
        elif selected == fixed.candidate_id:
            intervention = InterventionType.OVERRIDE_TO_FIXED_FALLBACK
        else:
            intervention = InterventionType.OVERRIDE_TO_OTHER_ALTERNATIVE
        if scored is None:
            scores = {
                candidate_id: float(index)
                for index, candidate_id in enumerate(pool.ordered_candidate_ids)
            }
            ensemble_identity = content_digest(
                {
                    "mode": self.mode,
                    "policy_id": self.policy_id,
                    "schema_version": "1.0",
                },
                context="M4DScorerFreeReferenceV1",
            )
            probabilities = False
            numeric: tuple[float | None, ...] = (None, None, None, None, None)
        else:
            scores = dict(scored.risks)
            ensemble_identity = scored.ensemble_identity
            probabilities = True
            numeric = (
                scored.risks[designation.candidate_id],
                scored.risks[best],
                scored.risks[designation.candidate_id] - scored.risks[best],
                scored.uncertainties[designation.candidate_id],
                scored.uncertainties[best],
            )
        ranking = (selected,) + tuple(
            candidate_id
            for candidate_id in sorted(scores, key=lambda item: (scores[item], item))
            if candidate_id != selected
        )
        self._pending = FallbackShieldDecisionRecordV1(
            decision_record_digest="sha256:" + "0" * 64,
            candidate_pool_digest=pool.content_digest,
            source_trajectory_id=pool.source_trajectory_id,
            policy_id=self.policy_id,
            scenario=self.scenario.value,
            decision_ordinal=pool.decision_ordinal,
            nominal_plan_index=pool.nominal_plan_index,
            nominal_candidate_id=designation.candidate_id,
            nominal_definition_ordinal=designation.definition_ordinal,
            fixed_fallback_candidate_id=fixed.candidate_id,
            best_candidate_id=best,
            selected_candidate_id=selected,
            nominal_risk=numeric[0],
            best_risk=numeric[1],
            improvement_margin=numeric[2],
            nominal_uncertainty=numeric[3],
            best_uncertainty=numeric[4],
            gate_profile_id=profile_id,
            gate_decision=gate_decision,
            intervention_type=intervention,
            scorer_id=None if self.scorer is None else self.scorer.selector_id,
            scorer_ensemble_identity=(
                None if self.scorer is None else ensemble_identity
            ),
            schedule_digest=designation.schedule_digest,
            fault_class_reporting_only=designation.fault_class,
            injected_fault_reporting_only=designation.injected_fault,
        )
        return SelectorOutputV1(
            selector_id=self.selector_id,
            scores=scores,
            ranking=ranking,
            checkpoint_ensemble_identity=ensemble_identity,
            probabilities=probabilities,
        )

    def persist_decision_context(
        self,
        decision: ClosedLoopDecisionRecordV1,
        pool: ClosedLoopCandidatePoolV1,
        boundary_directory: Path,
    ) -> None:
        """Persist the shield sidecar after base decision, before ledger execution."""

        del pool
        if self._pending is None:
            _fail("fallback selector", "no pending shield decision")
        record = replace(self._pending, decision_record_digest=decision.content_digest)
        if (
            record.decision_ordinal != decision.decision_ordinal
            or record.selected_candidate_id != decision.selected_candidate_id
            or record.candidate_pool_digest != decision.candidate_pool_digest
        ):
            _fail("fallback selector", "sidecar differs from base decision")
        write_exclusive_json(
            Path(boundary_directory) / "fallback-shield-decision.json",
            record.as_mapping(),
        )
        self._pending = None


__all__ = [
    "CandidateRiskScorer",
    "ConservativeOverrideDecisionV1",
    "ConservativeOverrideGateV1",
    "FallbackShieldDecisionRecordV1",
    "FallbackShieldError",
    "FallbackShieldSelectorV1",
    "FaultScheduleConfigurationV1",
    "GateProfileV1",
    "GateProfilesConfigurationV1",
    "InterventionType",
    "NominalDesignationV1",
    "NominalScenario",
    "designate_nominal",
    "load_fallback_decision",
    "load_fault_schedule",
    "load_gate_profiles",
]
