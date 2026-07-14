"""Typed, simulator-independent evaluator interface and applicability contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.models import EvaluationEvidence, EvaluationStatus


class EvaluationError(ValueError):
    """Base error for evaluator configuration and contract failures."""


class EvaluatorConfigurationError(EvaluationError):
    """Raised when evaluator configuration is malformed or inconsistent."""


class ApplicabilityDecisionError(EvaluationError):
    """Raised when an evaluator returns an invalid applicability decision."""


@dataclass(frozen=True, slots=True)
class ApplicabilityDecision:
    """Explicitly mark a proposal applicable, skipped, or invalid."""

    status: EvaluationStatus | None
    reason: str | None

    def __post_init__(self) -> None:
        """Reject ambiguous applicability states without changing their meaning."""
        if self.status is None:
            if self.reason is not None:
                raise ApplicabilityDecisionError(
                    "ApplicabilityDecision.reason: applicable decisions must not "
                    "contain a skip or invalidity reason"
                )
            return
        if self.status not in {EvaluationStatus.SKIPPED, EvaluationStatus.INVALID}:
            raise ApplicabilityDecisionError(
                "ApplicabilityDecision.status: only skipped or invalid are allowed "
                "for non-applicable proposals"
            )
        if (
            not isinstance(self.reason, str)
            or not self.reason
            or self.reason != self.reason.strip()
        ):
            raise ApplicabilityDecisionError(
                "ApplicabilityDecision.reason: skipped and invalid decisions require "
                "a non-empty reason without surrounding whitespace"
            )

    @classmethod
    def applicable(cls) -> ApplicabilityDecision:
        """Return an applicable decision."""
        return cls(status=None, reason=None)

    @classmethod
    def skipped(cls, reason: str) -> ApplicabilityDecision:
        """Return an explicit policy or applicability skip."""
        return cls(status=EvaluationStatus.SKIPPED, reason=reason)

    @classmethod
    def invalid(cls, reason: str) -> ApplicabilityDecision:
        """Return an explicit invalid-source or invalid-proposal decision."""
        return cls(status=EvaluationStatus.INVALID, reason=reason)


@runtime_checkable
class ProposalEvaluator(Protocol):
    """Typed interface for one deterministic proposal evaluator."""

    @property
    def evaluator_id(self) -> str:
        """Return the stable evaluator registry identifier."""
        ...

    @property
    def evaluator_version(self) -> str:
        """Return the evaluator semantic version."""
        ...

    @property
    def configuration_digest(self) -> str:
        """Return the digest of the resolved evaluator configuration."""
        ...

    def resolved_configuration(self) -> Mapping[str, object]:
        """Return immutable, serializable resolved configuration values."""
        ...

    def check_applicability(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
    ) -> ApplicabilityDecision:
        """Return an explicit decision without mutating the proposal."""
        ...

    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        """Evaluate one applicable proposal and return evidence for this attempt."""
        ...


__all__ = [
    "ApplicabilityDecision",
    "ApplicabilityDecisionError",
    "EvaluationError",
    "EvaluatorConfigurationError",
    "ProposalEvaluator",
]
