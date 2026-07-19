"""Outcome-blind ensemble risk scores shared by M4C and M4D control."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateRiskScoresV1:
    """Mean calibrated failure risks and per-candidate ensemble uncertainty."""

    risks: Mapping[str, float]
    uncertainties: Mapping[str, float]
    ensemble_identity: str

    def __post_init__(self) -> None:
        """Require identical finite candidate inventories and bounded risks."""

        risks = dict(self.risks)
        uncertainties = dict(self.uncertainties)
        if not risks or set(risks) != set(uncertainties):
            raise ValueError("risk and uncertainty candidate inventories differ")
        for candidate_id, value in risks.items():
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("risk score candidate ID is invalid")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("candidate risk must be finite in [0,1]")
            uncertainty = uncertainties[candidate_id]
            if not math.isfinite(uncertainty) or uncertainty < 0.0:
                raise ValueError(
                    "candidate uncertainty must be finite and non-negative"
                )
        if not isinstance(
            self.ensemble_identity, str
        ) or not self.ensemble_identity.startswith("sha256:"):
            raise ValueError("ensemble identity is invalid")
        object.__setattr__(self, "risks", risks)
        object.__setattr__(self, "uncertainties", uncertainties)


__all__ = ["CandidateRiskScoresV1"]
