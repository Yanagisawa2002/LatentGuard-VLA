"""Outcome-blind selectors and validation-frozen abstention policies for M3C."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.blind_input import BlindCandidateGroupV1
from latentguard.selection.ensemble import EnsembleInferenceResultV1
from latentguard.selection.metrics import CandidateOutcomeV1
from latentguard.selection.models import (
    CANDIDATE_COUNT,
    SELECTION_SCHEMA_VERSION,
    CandidateGroupV1,
    SelectorDecisionV1,
)
from latentguard.training.baselines import ActionMagnitudeBaselineV1

EXPECTED_ENSEMBLE_SEEDS = (0, 1, 2, 3, 4)
RANDOM_SELECTOR_SEMANTIC = "sha256_per_candidate_uniform_ranking_v1"
ABSTENTION_SEMANTIC = "execute_when_group_min_failure_probability_lt_threshold_v1"
VALIDATION_ENSEMBLE_SEMANTIC = "arithmetic_mean_of_five_calibrated_probabilities_v1"


class SelectorError(ValueError):
    """Raised when a selector would violate the frozen blind protocol."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectorError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")
    return value


def _content_digest(value: object) -> str:
    encoded = canonical_json_bytes(value, context="M3CSelectorContentV1")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _probabilities(value: ArrayLike, context: str) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "fiu":
        _fail(context, "expected a rank-one numeric array")
    result = np.asarray(raw, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))) or not bool(
        np.all((result >= 0.0) & (result <= 1.0))
    ):
        _fail(context, "expected finite probabilities in [0, 1]")
    return result


def _binary_targets(value: ArrayLike, count: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (count,) or raw.dtype.kind not in "biu":
        _fail("failure_targets", "expected a matching rank-one binary array")
    result = np.asarray(raw, dtype=np.int64)
    if not bool(np.all((result == 0) | (result == 1))):
        _fail("failure_targets", "expected only zero and one")
    return result


@dataclass(frozen=True, slots=True)
class AbstentionPolicyV1:
    """One immutable validation-derived selector execution threshold."""

    policy_id: str
    threshold: float
    validation_prediction_digest: str
    verifier_bundle_digest: str
    validation_split_digest: str
    validation_group_count: int
    achieved_validation_coverage: float
    target_failure_recall: float | None = None
    achieved_failure_recall: float | None = None
    validation_balanced_accuracy: float | None = None
    target_coverage: float | None = None
    target_met: bool | None = None
    ensemble_semantic: str = VALIDATION_ENSEMBLE_SEMANTIC
    comparison_semantic: str = ABSTENTION_SEMANTIC
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the fixed policy and its validation-only bindings."""

        _text(self.policy_id, "AbstentionPolicyV1.policy_id")
        for name in (
            "validation_prediction_digest",
            "verifier_bundle_digest",
            "validation_split_digest",
        ):
            _digest(getattr(self, name), f"AbstentionPolicyV1.{name}")
        if (
            type(self.threshold) not in (int, float)
            or not math.isfinite(float(self.threshold))
            or not 0.0 <= float(self.threshold) <= 1.0
        ):
            _fail("AbstentionPolicyV1.threshold", "expected a finite value in [0, 1]")
        if (
            type(self.validation_group_count) is not int
            or self.validation_group_count <= 0
        ):
            _fail(
                "AbstentionPolicyV1.validation_group_count",
                "expected a positive integer",
            )
        for name in (
            "achieved_validation_coverage",
            "target_failure_recall",
            "achieved_failure_recall",
            "validation_balanced_accuracy",
            "target_coverage",
        ):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                _fail(
                    f"AbstentionPolicyV1.{name}", "expected null or a value in [0, 1]"
                )
        if self.policy_id == "maximum_validation_balanced_accuracy":
            if self.validation_balanced_accuracy is None or any(
                value is not None
                for value in (
                    self.target_failure_recall,
                    self.target_coverage,
                    self.target_met,
                )
            ):
                _fail("AbstentionPolicyV1", "invalid balanced-accuracy policy fields")
        elif self.policy_id == "target_validation_failure_recall":
            if (
                self.target_failure_recall is None
                or self.achieved_failure_recall is None
            ):
                _fail("AbstentionPolicyV1", "invalid target-recall policy fields")
            if self.target_coverage is not None or type(self.target_met) is not bool:
                _fail(
                    "AbstentionPolicyV1", "target-recall policy cannot target coverage"
                )
        elif self.policy_id.startswith("target_validation_coverage_"):
            if (
                self.target_coverage is None
                or type(self.target_met) is not bool
                or any(
                    value is not None
                    for value in (
                        self.target_failure_recall,
                        self.achieved_failure_recall,
                    )
                )
            ):
                _fail("AbstentionPolicyV1", "invalid target-coverage policy fields")
        else:
            _fail("AbstentionPolicyV1.policy_id", "unsupported policy")
        if self.ensemble_semantic != VALIDATION_ENSEMBLE_SEMANTIC:
            _fail("AbstentionPolicyV1.ensemble_semantic", "semantic changed")
        if self.comparison_semantic != ABSTENTION_SEMANTIC:
            _fail("AbstentionPolicyV1.comparison_semantic", "semantic changed")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("AbstentionPolicyV1.schema_version", "unsupported version")
        for name in (
            "threshold",
            "achieved_validation_coverage",
            "target_failure_recall",
            "achieved_failure_recall",
            "validation_balanced_accuracy",
            "target_coverage",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))

    def as_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-native policy content."""

        return {
            "achieved_failure_recall": self.achieved_failure_recall,
            "achieved_validation_coverage": self.achieved_validation_coverage,
            "comparison_semantic": self.comparison_semantic,
            "ensemble_semantic": self.ensemble_semantic,
            "policy_id": self.policy_id,
            "schema_version": self.schema_version,
            "target_coverage": self.target_coverage,
            "target_failure_recall": self.target_failure_recall,
            "target_met": self.target_met,
            "threshold": self.threshold,
            "validation_balanced_accuracy": self.validation_balanced_accuracy,
            "validation_group_count": self.validation_group_count,
            "validation_prediction_digest": self.validation_prediction_digest,
            "validation_split_digest": self.validation_split_digest,
            "verifier_bundle_digest": self.verifier_bundle_digest,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent frozen policy identity."""

        return _content_digest(self.as_mapping())

    def permits(self, selected_failure_probability: float) -> bool:
        """Return whether strict-below validation confidence permits execution."""

        if (
            type(selected_failure_probability) not in (int, float)
            or not math.isfinite(float(selected_failure_probability))
            or not 0.0 <= float(selected_failure_probability) <= 1.0
        ):
            _fail("selected_failure_probability", "expected a finite value in [0, 1]")
        return float(selected_failure_probability) < self.threshold


def _classification_counts(
    probabilities: NDArray[np.float64],
    targets: NDArray[np.int64],
    threshold: float,
) -> tuple[float, float]:
    predictions = probabilities >= threshold
    positives = targets == 1
    negatives = ~positives
    recall = float(np.mean(predictions[positives]))
    specificity = float(np.mean(~predictions[negatives]))
    return (recall + specificity) / 2.0, recall


def _threshold_candidates(probabilities: NDArray[np.float64]) -> tuple[float, ...]:
    unique = sorted({float(item) for item in probabilities}, reverse=True)
    maximum = unique[0]
    if maximum < 1.0:
        unique.insert(0, float(np.nextafter(maximum, 1.0)))
    return tuple(unique)


def _validation_prediction_digest(
    per_seed_probabilities: NDArray[np.float64],
    targets: NDArray[np.int64],
    group_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    verifier_bundle_digest: str,
    validation_split_digest: str,
) -> str:
    payload = {
        "ensemble_semantic": VALIDATION_ENSEMBLE_SEMANTIC,
        "failure_target_sha256": hashlib.sha256(targets.tobytes(order="C")).hexdigest(),
        "group_ids": list(group_ids),
        "candidate_ids": list(candidate_ids),
        "per_seed_probability_dtype": per_seed_probabilities.dtype.str,
        "per_seed_probability_sha256": hashlib.sha256(
            per_seed_probabilities.tobytes(order="C")
        ).hexdigest(),
        "seed_order": list(EXPECTED_ENSEMBLE_SEEDS),
        "validation_split_digest": validation_split_digest,
        "verifier_bundle_digest": verifier_bundle_digest,
    }
    return _content_digest(payload)


def fit_validation_ensemble_abstention_policies(
    per_seed_calibrated_failure_probabilities: ArrayLike,
    failure_targets: ArrayLike,
    validation_group_ids: Sequence[str],
    validation_candidate_ids: Sequence[str],
    *,
    split: str,
    verifier_bundle_digest: str,
    validation_split_digest: str,
    target_failure_recall: float = 0.90,
    target_coverages: Sequence[float] = (0.90, 0.80, 0.70, 0.50),
) -> tuple[AbstentionPolicyV1, ...]:
    """Fit ensemble policies from validation predictions, never seed thresholds.

    The function requires the complete five-by-candidate probability matrix and
    computes the arithmetic ensemble itself.  Accepting only the matrix makes it
    impossible to implement this contract by averaging five pre-fitted scalar
    thresholds.
    """

    if split != "validation":
        _fail("split", "abstention policies may be fitted only on validation")
    _digest(verifier_bundle_digest, "verifier_bundle_digest")
    _digest(validation_split_digest, "validation_split_digest")
    raw = np.asarray(per_seed_calibrated_failure_probabilities)
    if raw.ndim != 2 or raw.shape[0] != len(EXPECTED_ENSEMBLE_SEEDS):
        _fail(
            "per_seed_calibrated_failure_probabilities",
            "expected an exact five-by-candidate matrix",
        )
    if raw.dtype.kind not in "fiu":
        _fail("per_seed_calibrated_failure_probabilities", "expected numeric values")
    per_seed = np.asarray(raw, dtype=np.float64)
    if (
        per_seed.shape[1] == 0
        or not bool(np.all(np.isfinite(per_seed)))
        or not bool(np.all((per_seed >= 0.0) & (per_seed <= 1.0)))
    ):
        _fail(
            "per_seed_calibrated_failure_probabilities",
            "expected finite probabilities in [0, 1]",
        )
    targets = _binary_targets(failure_targets, per_seed.shape[1])
    if int(np.min(targets)) == int(np.max(targets)):
        _fail("failure_targets", "threshold fitting requires both validation classes")
    group_ids = tuple(validation_group_ids)
    if len(group_ids) != per_seed.shape[1]:
        _fail("validation_group_ids", "length differs from candidate inventory")
    for group_id in group_ids:
        _text(group_id, "validation_group_ids")
    candidate_ids = tuple(validation_candidate_ids)
    if len(candidate_ids) != per_seed.shape[1]:
        _fail("validation_candidate_ids", "length differs from candidate inventory")
    for candidate_id in candidate_ids:
        _text(candidate_id, "validation_candidate_ids")
    if len(set(candidate_ids)) != len(candidate_ids):
        _fail("validation_candidate_ids", "candidate IDs must be unique")
    counts = Counter(group_ids)
    if any(count != CANDIDATE_COUNT for count in counts.values()):
        _fail(
            "validation_group_ids", "every validation group must contain 8 candidates"
        )
    if (
        type(target_failure_recall) not in (int, float)
        or not math.isfinite(float(target_failure_recall))
        or not 0.0 < float(target_failure_recall) <= 1.0
    ):
        _fail("target_failure_recall", "expected a finite value in (0, 1]")
    coverage_targets = tuple(float(item) for item in target_coverages)
    if coverage_targets != (0.90, 0.80, 0.70, 0.50):
        _fail("target_coverages", "M3C requires exactly 90%, 80%, 70%, and 50%")

    ensemble = np.mean(per_seed, axis=0, dtype=np.float64)
    prediction_digest = _validation_prediction_digest(
        per_seed,
        targets,
        group_ids,
        candidate_ids,
        verifier_bundle_digest,
        validation_split_digest,
    )
    selected_indices = tuple(
        min(
            (
                index
                for index, candidate_group in enumerate(group_ids)
                if candidate_group == group_id
            ),
            key=lambda index: (float(ensemble[index]), candidate_ids[index]),
        )
        for group_id in sorted(counts)
    )
    group_minima = np.asarray(
        [ensemble[index] for index in selected_indices], dtype=np.float64
    )
    group_targets = np.asarray(
        [targets[index] for index in selected_indices], dtype=np.int64
    )
    if int(np.min(group_targets)) == int(np.max(group_targets)):
        _fail(
            "failure_targets",
            "selected validation group minima require both outcome classes",
        )
    candidates = _threshold_candidates(group_minima)
    evaluated = [
        (threshold, *_classification_counts(group_minima, group_targets, threshold))
        for threshold in candidates
    ]
    best_threshold, best_balanced, best_recall = max(
        evaluated, key=lambda item: (item[1], item[0])
    )
    satisfying = [item for item in evaluated if item[2] >= target_failure_recall]
    if satisfying:
        recall_threshold, recall_balanced, achieved_recall = max(
            satisfying, key=lambda item: item[0]
        )
    else:
        recall_threshold, recall_balanced, achieved_recall = max(
            evaluated, key=lambda item: (item[2], item[0])
        )

    def coverage(threshold: float) -> float:
        return float(np.mean(group_minima < threshold))

    policies: list[AbstentionPolicyV1] = [
        AbstentionPolicyV1(
            policy_id="maximum_validation_balanced_accuracy",
            threshold=best_threshold,
            validation_prediction_digest=prediction_digest,
            verifier_bundle_digest=verifier_bundle_digest,
            validation_split_digest=validation_split_digest,
            validation_group_count=group_minima.size,
            achieved_validation_coverage=coverage(best_threshold),
            achieved_failure_recall=best_recall,
            validation_balanced_accuracy=best_balanced,
        ),
        AbstentionPolicyV1(
            policy_id="target_validation_failure_recall",
            threshold=recall_threshold,
            validation_prediction_digest=prediction_digest,
            verifier_bundle_digest=verifier_bundle_digest,
            validation_split_digest=validation_split_digest,
            validation_group_count=group_minima.size,
            achieved_validation_coverage=coverage(recall_threshold),
            target_failure_recall=float(target_failure_recall),
            achieved_failure_recall=achieved_recall,
            validation_balanced_accuracy=recall_balanced,
            target_met=achieved_recall >= target_failure_recall,
        ),
    ]
    ordered_minima = np.sort(group_minima)
    for target in coverage_targets:
        rank = max(1, math.ceil(target * group_minima.size))
        boundary = float(ordered_minima[rank - 1])
        threshold = 1.0 if boundary == 1.0 else float(np.nextafter(boundary, 1.0))
        policies.append(
            AbstentionPolicyV1(
                policy_id=f"target_validation_coverage_{int(target * 100):02d}",
                threshold=threshold,
                validation_prediction_digest=prediction_digest,
                verifier_bundle_digest=verifier_bundle_digest,
                validation_split_digest=validation_split_digest,
                validation_group_count=group_minima.size,
                achieved_validation_coverage=coverage(threshold),
                target_coverage=target,
                target_met=coverage(threshold) >= target,
            )
        )
    return tuple(policies)


def deterministic_random_decision(
    group: BlindCandidateGroupV1,
    *,
    seed: int,
    selector_id: str = "deterministic_random_v1",
) -> SelectorDecisionV1:
    """Choose by stable SHA-256 ranks without mutable random-process state."""

    group_id, candidate_ids, _ = _selector_group_content(group)
    if type(seed) is not int or not 0 <= seed < 2**64:
        _fail("seed", "expected uint64")
    _text(selector_id, "selector_id")

    def rank_key(proposal_id: str) -> tuple[bytes, str]:
        encoded = canonical_json_bytes(
            {
                "group_id": group_id,
                "proposal_id": proposal_id,
                "seed": seed,
                "semantic": RANDOM_SELECTOR_SEMANTIC,
            },
            context="M3CDeterministicRandomSelectorV1",
        )
        return hashlib.sha256(encoded).digest(), proposal_id

    ranking = tuple(sorted(candidate_ids, key=rank_key))
    return SelectorDecisionV1(
        selector_id=selector_id,
        group_id=group_id,
        selected_proposal_id=ranking[0],
        ranking=ranking,
        predicted_failure_probabilities={},
        abstained=False,
    )


def _action_magnitude_probability(
    baseline: ActionMagnitudeBaselineV1,
    action_chunk: NDArray[np.generic],
) -> float:
    rows = np.asarray(action_chunk, dtype=np.float64)
    features = np.asarray(
        (
            float(np.mean(np.abs(rows))),
            float(np.max(np.abs(np.diff(rows, axis=0)))) if rows.shape[0] > 1 else 0.0,
            float(np.mean(np.abs(rows - baseline.action_mean))),
        ),
        dtype=np.float64,
    )
    score = float(
        np.mean(
            (features - baseline.feature_mean) / baseline.feature_standard_deviation
        )
    )
    if not math.isfinite(score):
        _fail("action-magnitude score", "non-finite result")
    probability = np.asarray(
        1.0 / (1.0 + np.exp(-np.clip(score, -60.0, 60.0))),
        dtype=np.float32,
    )
    return float(probability)


def _selector_group_content(
    group: BlindCandidateGroupV1,
) -> tuple[str, tuple[str, ...], NDArray[Any]]:
    """Return only opaque identity and deployable actions for blind selectors."""

    if isinstance(group, BlindCandidateGroupV1):
        return group.group_id, group.candidate_ids, group.action_chunks
    _fail("group", "expected BlindCandidateGroupV1")


def action_magnitude_decision(
    group: BlindCandidateGroupV1,
    baseline: ActionMagnitudeBaselineV1,
    *,
    expected_baseline_digest: str,
    selector_id: str = "frozen_m3b_action_magnitude_v1",
) -> SelectorDecisionV1:
    """Rank action chunks with the exact frozen, label-free M3B heuristic."""

    group_id, candidate_ids, action_chunks = _selector_group_content(group)
    if not isinstance(baseline, ActionMagnitudeBaselineV1):
        _fail("baseline", "expected ActionMagnitudeBaselineV1")
    _digest(expected_baseline_digest, "expected_baseline_digest")
    if baseline.content_digest != expected_baseline_digest:
        _fail("baseline", "fitted content digest differs from the frozen identity")
    _text(selector_id, "selector_id")
    probabilities = {
        candidate_id: _action_magnitude_probability(baseline, action_chunk)
        for candidate_id, action_chunk in zip(candidate_ids, action_chunks, strict=True)
    }
    ranking = tuple(sorted(candidate_ids, key=lambda item: (probabilities[item], item)))
    return SelectorDecisionV1(
        selector_id=selector_id,
        group_id=group_id,
        selected_proposal_id=ranking[0],
        ranking=ranking,
        predicted_failure_probabilities=probabilities,
        abstained=False,
    )


def learned_ensemble_decision(
    group: BlindCandidateGroupV1,
    inference: EnsembleInferenceResultV1,
    *,
    selector_id: str,
    abstention_policy: AbstentionPolicyV1 | None = None,
) -> SelectorDecisionV1:
    """Create one decision from label-free five-seed calibrated inference."""

    group_id, candidate_ids, _ = _selector_group_content(group)
    if not isinstance(inference, EnsembleInferenceResultV1):
        _fail("inference", "expected EnsembleInferenceResultV1")
    _text(selector_id, "selector_id")
    if set(inference.candidate_ids) != set(candidate_ids):
        _fail("inference", "candidate inventory differs from the exact group")
    probabilities = {
        proposal_id: inference.probability_for(proposal_id)
        for proposal_id in candidate_ids
    }
    if set(inference.ranking) != set(candidate_ids):
        _fail("inference", "ranking inventory differs from the exact group")
    selected = inference.ranking[0]
    abstained = abstention_policy is not None and not abstention_policy.permits(
        probabilities[selected]
    )
    return SelectorDecisionV1(
        selector_id=selector_id,
        group_id=group_id,
        selected_proposal_id=None if abstained else selected,
        ranking=inference.ranking,
        predicted_failure_probabilities=probabilities,
        abstained=abstained,
        abstention_policy_id=(
            None if abstention_policy is None else abstention_policy.content_digest
        ),
    )


def oracle_analysis_decision(
    group: CandidateGroupV1,
    complete_outcomes: Sequence[CandidateOutcomeV1],
    *,
    selector_id: str = "oracle_analysis_only_v1",
) -> SelectorDecisionV1:
    """Create the post-Stage-C oracle from eight complete strong outcomes."""

    if not isinstance(group, CandidateGroupV1):
        _fail("group", "expected CandidateGroupV1")
    outcomes = tuple(complete_outcomes)
    if len(outcomes) != CANDIDATE_COUNT:
        _fail("complete_outcomes", "oracle requires all eight outcomes")
    by_id = {item.proposal_id: item for item in outcomes}
    if len(by_id) != CANDIDATE_COUNT or set(by_id) != set(group.proposal_ids):
        _fail("complete_outcomes", "outcome inventory differs from the group")
    if any(
        item.group_id != group.group_id
        or item.source_trajectory_id != group.source_trajectory_id
        or item.status != "conclusive"
        or not item.simulator_replay_verified
        or item.label_strength != "strong"
        or item.state_component_count != 70
        for item in outcomes
    ):
        _fail("complete_outcomes", "oracle requires complete strong simulator evidence")
    ranking = tuple(
        sorted(
            group.proposal_ids,
            key=lambda item: (
                not bool(by_id[item].success),
                bool(by_id[item].unsafe),
                item,
            ),
        )
    )
    probabilities: Mapping[str, float] = {
        proposal_id: float(not bool(by_id[proposal_id].success))
        for proposal_id in group.proposal_ids
    }
    return SelectorDecisionV1(
        selector_id=selector_id,
        group_id=group.group_id,
        selected_proposal_id=ranking[0],
        ranking=ranking,
        predicted_failure_probabilities=probabilities,
        abstained=False,
    )


__all__ = [
    "ABSTENTION_SEMANTIC",
    "EXPECTED_ENSEMBLE_SEEDS",
    "RANDOM_SELECTOR_SEMANTIC",
    "VALIDATION_ENSEMBLE_SEMANTIC",
    "AbstentionPolicyV1",
    "SelectorError",
    "action_magnitude_decision",
    "deterministic_random_decision",
    "fit_validation_ensemble_abstention_policies",
    "learned_ensemble_decision",
    "oracle_analysis_decision",
]
