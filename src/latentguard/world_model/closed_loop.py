"""Conservative deployment gate for an optional WM-v0 closed-loop probe."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClosedLoopGateConfig:
    """Frozen risk, margin, uncertainty, and intervention-budget thresholds."""

    risk_threshold: float
    minimum_score_margin: float
    maximum_uncertainty: float
    maximum_interventions: int
    cooldown_decisions: int

    def __post_init__(self) -> None:
        """Validate a conservative finite threshold contract."""

        for name in (
            "risk_threshold",
            "minimum_score_margin",
            "maximum_uncertainty",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float")
        if self.risk_threshold > 1.0 or self.minimum_score_margin > 1.0:
            raise ValueError("risk and margin thresholds must not exceed one")
        if (
            type(self.maximum_interventions) is not int
            or self.maximum_interventions < 0
        ):
            raise ValueError("maximum_interventions must be non-negative")
        if type(self.cooldown_decisions) is not int or self.cooldown_decisions < 0:
            raise ValueError("cooldown_decisions must be non-negative")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Auditable decision that defaults to the primary action."""

    intervene: bool
    reason: str


def decide_intervention(
    *,
    primary_risk: float,
    best_alternative_risk: float,
    uncertainty: float,
    interventions_used: int,
    decisions_since_intervention: int,
    config: ClosedLoopGateConfig,
) -> GateDecision:
    """Intervene only when every frozen conservative condition passes."""

    values = (primary_risk, best_alternative_risk, uncertainty)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("closed-loop scores must be finite probabilities")
    if interventions_used >= config.maximum_interventions:
        return GateDecision(False, "intervention_budget_exhausted")
    if decisions_since_intervention < config.cooldown_decisions:
        return GateDecision(False, "cooldown_active")
    if primary_risk < config.risk_threshold:
        return GateDecision(False, "primary_risk_below_threshold")
    if primary_risk - best_alternative_risk < config.minimum_score_margin:
        return GateDecision(False, "alternative_margin_insufficient")
    if uncertainty > config.maximum_uncertainty:
        return GateDecision(False, "uncertainty_too_high")
    return GateDecision(True, "all_frozen_gate_conditions_passed")
