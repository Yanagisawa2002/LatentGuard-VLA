"""Deterministic, CPU-only fixture evaluator for M2A infrastructure tests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, cast

import numpy as np

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.base import ApplicabilityDecision
from latentguard.evaluation.models import (
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.evaluation.validation import validate_evaluation_evidence
from latentguard.models import JsonScalar, LabelSource, LabelStrength

FIXTURE_CONFIGURATION_SCHEMA_VERSION = "1.0"
"""Configuration schema supported by the deterministic fixture evaluator."""

DETERMINISTIC_FIXTURE_EVALUATOR_ID = "deterministic_fixture"
"""Stable registry identifier for the fixture-only evaluator."""

DETERMINISTIC_FIXTURE_EVALUATOR_VERSION = "1.0.0"
"""Semantic version of the fixture evaluator's deterministic behavior."""

_CONFIGURATION_FIELDS = frozenset(
    {
        "schema_version",
        "success_mean_abs_threshold",
        "unsafe_max_abs_threshold",
        "skip_mean_abs_below",
        "indeterminate_temporal_variation_below",
        "execution_error_max_abs_above",
    }
)
_FIXTURE_NOTE = (
    "Synthetic fixture-only evidence; non-physical, not simulator replay, and "
    "unsuitable for research claims."
)


class FixtureConfigurationError(ValueError):
    """Raised when deterministic-fixture configuration is malformed."""


class FixtureEvaluationError(RuntimeError):
    """Controlled fixture failure used to exercise runner error handling."""


def _threshold(
    configuration: Mapping[str, object],
    name: str,
    *,
    optional: bool,
) -> float | None:
    value = configuration[name]
    if optional and value is None:
        return None
    if type(value) not in (int, float):
        suffix = " or null" if optional else ""
        raise FixtureConfigurationError(
            f"DeterministicFixture.configuration.{name}: expected a finite "
            f"non-negative number{suffix}"
        )
    try:
        resolved = float(cast(int | float, value))
    except (OverflowError, ValueError) as exc:
        raise FixtureConfigurationError(
            f"DeterministicFixture.configuration.{name}: value is outside the "
            "supported floating-point range"
        ) from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise FixtureConfigurationError(
            f"DeterministicFixture.configuration.{name}: expected a finite "
            "non-negative number"
        )
    if resolved == 0.0:
        resolved = 0.0
    return resolved


def _resolve_configuration(
    configuration: Mapping[str, object],
) -> Mapping[str, JsonScalar]:
    if not isinstance(configuration, Mapping):
        raise FixtureConfigurationError(
            "DeterministicFixture.configuration: expected an object"
        )
    actual_fields = set(configuration)
    missing = sorted(_CONFIGURATION_FIELDS - actual_fields)
    unexpected = sorted(actual_fields - _CONFIGURATION_FIELDS)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise FixtureConfigurationError(
            "DeterministicFixture.configuration: invalid fields ("
            + "; ".join(details)
            + ")"
        )
    version = configuration["schema_version"]
    if version != FIXTURE_CONFIGURATION_SCHEMA_VERSION:
        raise FixtureConfigurationError(
            "DeterministicFixture.configuration.schema_version: unsupported "
            f"version {version!r}; supported: "
            f"{FIXTURE_CONFIGURATION_SCHEMA_VERSION}"
        )
    resolved: dict[str, JsonScalar] = {
        "schema_version": FIXTURE_CONFIGURATION_SCHEMA_VERSION,
        "success_mean_abs_threshold": _threshold(
            configuration, "success_mean_abs_threshold", optional=False
        ),
        "unsafe_max_abs_threshold": _threshold(
            configuration, "unsafe_max_abs_threshold", optional=False
        ),
        "skip_mean_abs_below": _threshold(
            configuration, "skip_mean_abs_below", optional=True
        ),
        "indeterminate_temporal_variation_below": _threshold(
            configuration,
            "indeterminate_temporal_variation_below",
            optional=True,
        ),
        "execution_error_max_abs_above": _threshold(
            configuration, "execution_error_max_abs_above", optional=True
        ),
    }
    return MappingProxyType(resolved)


def _action_statistics(proposal: CorruptedActionProposal) -> tuple[float, float, float]:
    actions = proposal.transformed_action.actions
    absolute = np.abs(actions.astype(np.float64, copy=False))
    mean_absolute_magnitude = float(np.mean(absolute))
    maximum_absolute_magnitude = float(np.max(absolute))
    if actions.shape[0] < 2:
        mean_temporal_variation = 0.0
    else:
        differences = np.diff(actions.astype(np.float64, copy=False), axis=0)
        mean_temporal_variation = float(np.mean(np.abs(differences)))
    statistics = (
        mean_absolute_magnitude,
        maximum_absolute_magnitude,
        mean_temporal_variation,
    )
    if not all(math.isfinite(value) for value in statistics):
        raise FixtureEvaluationError(
            "deterministic fixture encountered non-finite action statistics"
        )
    return statistics


@dataclass(frozen=True, slots=True)
class DeterministicFixtureEvaluator:
    """Compute explicitly weak, non-physical evidence from action statistics."""

    configuration: Mapping[str, object]
    _resolved_configuration: Mapping[str, JsonScalar] = field(init=False, repr=False)
    _configuration_digest: str = field(init=False, repr=False)

    evaluator_id: ClassVar[str] = DETERMINISTIC_FIXTURE_EVALUATOR_ID
    evaluator_version: ClassVar[str] = DETERMINISTIC_FIXTURE_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        """Validate and detach the complete fixture configuration."""
        resolved = _resolve_configuration(self.configuration)
        object.__setattr__(self, "configuration", resolved)
        object.__setattr__(self, "_resolved_configuration", resolved)
        object.__setattr__(
            self,
            "_configuration_digest",
            compute_configuration_digest(resolved),
        )

    @property
    def configuration_digest(self) -> str:
        """Return the stable digest of the canonical resolved configuration."""
        return self._configuration_digest

    def resolved_configuration(self) -> Mapping[str, JsonScalar]:
        """Return an immutable canonical configuration mapping."""
        return self._resolved_configuration

    def check_applicability(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
    ) -> ApplicabilityDecision:
        """Apply the configured synthetic skip rule before evaluation."""
        del source_dataset_id
        mean_absolute_magnitude, _, _ = _action_statistics(proposal)
        threshold = self._resolved_configuration["skip_mean_abs_below"]
        if isinstance(threshold, float) and mean_absolute_magnitude < threshold:
            return ApplicabilityDecision.skipped(
                "synthetic fixture policy skipped proposal because mean absolute "
                f"magnitude {mean_absolute_magnitude:.6g} is below {threshold:.6g}"
            )
        return ApplicabilityDecision.applicable()

    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        """Return deterministic fixture evidence or a controlled fixture failure."""
        mean_absolute_magnitude, maximum_absolute_magnitude, temporal_variation = (
            _action_statistics(proposal)
        )
        error_threshold = self._resolved_configuration["execution_error_max_abs_above"]
        if (
            isinstance(error_threshold, float)
            and maximum_absolute_magnitude > error_threshold
        ):
            raise FixtureEvaluationError(
                "controlled deterministic fixture execution error: maximum "
                "absolute magnitude exceeded the configured synthetic threshold"
            )

        status = EvaluationStatus.CONCLUSIVE
        success: bool | None = None
        progress_before: float | None = None
        progress_after: float | None = None
        progress_delta: float | None = None
        unsafe: bool | None = None
        label_source: LabelSource | None = None
        label_strength: LabelStrength | None = None
        termination_reason = "synthetic fixture threshold evaluation complete"
        indeterminate_threshold = self._resolved_configuration[
            "indeterminate_temporal_variation_below"
        ]
        if (
            isinstance(indeterminate_threshold, float)
            and temporal_variation < indeterminate_threshold
        ):
            status = EvaluationStatus.INDETERMINATE
            termination_reason = (
                "synthetic fixture temporal-variation threshold left the task "
                "outcome indeterminate"
            )
        else:
            success_threshold = self._resolved_configuration[
                "success_mean_abs_threshold"
            ]
            unsafe_threshold = self._resolved_configuration["unsafe_max_abs_threshold"]
            assert isinstance(success_threshold, float)
            assert isinstance(unsafe_threshold, float)
            success = mean_absolute_magnitude <= success_threshold
            unsafe = maximum_absolute_magnitude >= unsafe_threshold
            progress_before = 0.0
            progress_after = 1.0 / (1.0 + mean_absolute_magnitude)
            progress_delta = progress_after - progress_before
            label_source = LabelSource.DETERMINISTIC_EVALUATOR
            label_strength = LabelStrength.WEAK

        evidence = EvaluationEvidence(
            evidence_id=compute_evidence_identifier(
                proposal_id=proposal.proposal_id,
                evaluator_id=self.evaluator_id,
                evaluator_version=self.evaluator_version,
                evaluator_configuration_digest=self.configuration_digest,
                evaluation_seed=evaluation_seed,
                attempt_ordinal=attempt_ordinal,
            ),
            proposal_id=proposal.proposal_id,
            source_dataset_id=source_dataset_id,
            source_episode_id=proposal.source_episode_id,
            source_candidate_id=proposal.source_candidate_id,
            split_group_id=proposal.split_group_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            evaluator_configuration_digest=self.configuration_digest,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
            status=status,
            success=success,
            progress_before=progress_before,
            progress_after=progress_after,
            progress_delta=progress_delta,
            unsafe=unsafe,
            failure_events=(),
            termination_reason=termination_reason,
            replayed_control_steps=0,
            metrics={
                "mean_absolute_magnitude": mean_absolute_magnitude,
                "maximum_absolute_magnitude": maximum_absolute_magnitude,
                "mean_temporal_variation": temporal_variation,
            },
            artifact_references=(),
            label_source=label_source,
            label_strength=label_strength,
            simulator_replay_verified=False,
            notes=_FIXTURE_NOTE,
            schema_version=EVALUATION_EVIDENCE_SCHEMA_VERSION,
        )
        validate_evaluation_evidence(evidence)
        return evidence


def create_deterministic_fixture_evaluator(
    configuration: Mapping[str, object],
) -> DeterministicFixtureEvaluator:
    """Create the strict built-in deterministic fixture evaluator."""
    return DeterministicFixtureEvaluator(configuration)


__all__ = [
    "DETERMINISTIC_FIXTURE_EVALUATOR_ID",
    "DETERMINISTIC_FIXTURE_EVALUATOR_VERSION",
    "FIXTURE_CONFIGURATION_SCHEMA_VERSION",
    "DeterministicFixtureEvaluator",
    "FixtureConfigurationError",
    "FixtureEvaluationError",
    "create_deterministic_fixture_evaluator",
]
