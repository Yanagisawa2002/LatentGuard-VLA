"""Typed LG-R1c reward and LG-R2 feature contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class CandidateTimeAvailability(StrEnum):
    """When a reward signal is available relative to candidate execution."""

    CURRENT_STATE_ONLY = "CURRENT_STATE_ONLY"
    EXECUTED_TRAJECTORY_REQUIRED = "EXECUTED_TRAJECTORY_REQUIRED"
    FUTURE_OBSERVATION_REQUIRED = "FUTURE_OBSERVATION_REQUIRED"
    CANDIDATE_ACTION_CONDITIONED = "CANDIDATE_ACTION_CONDITIONED"
    NOT_USABLE_FOR_PRE_EXECUTION_SELECTION = "NOT_USABLE_FOR_PRE_EXECUTION_SELECTION"


@dataclass(frozen=True)
class FeatureSpec:
    """One fully declared tensor in the LG-R2 feature contract."""

    tensor_name: str
    shape: tuple[str | int, ...]
    dtype: str
    pooling: str
    time_horizon: str
    online_computable: bool
    measured_latency_ms: float | None
    measured_peak_gpu_bytes: int | None
    depends_on_future_observation: bool
    candidate_time_availability: CandidateTimeAvailability
    candidate_time_usable: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True)
class FeatureSetContract:
    """One candidate input set for a future, separately authorized LG-R2."""

    name: str
    role: str
    features: tuple[FeatureSpec, ...]
    lg_r2_training_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "name": self.name,
            "role": self.role,
            "features": [feature.to_dict() for feature in self.features],
            "lg_r2_training_authorized": self.lg_r2_training_authorized,
        }


@dataclass(frozen=True)
class RewardGateThresholds:
    """Pre-registered task-agnostic reward promotion thresholds."""

    progress_spearman_min: float = 0.60
    pairwise_accuracy_min: float = 0.70
    task_macro_success_auroc_min: float = 0.75
    failure_auprc_floor: float = 0.20
    failure_prevalence_multiplier: float = 2.0
    failure_precision_min: float = 0.25
    failure_recall_min: float = 0.60
    failure_fpr_max: float = 0.35
    leave_task6_auprc_drop_max: float = 0.15


def validate_feature_contract(payload: dict[str, Any]) -> None:
    """Fail closed on missing feature sets, shapes, or candidate-time semantics."""

    if payload.get("schema_version") != "latentguard.lg_r1c.lg_r2_features.v1":
        raise ValueError("unexpected LG-R2 feature-contract schema")
    sets = payload.get("feature_sets")
    if not isinstance(sets, list):
        raise ValueError("feature_sets must be a list")
    expected = {"A", "B", "C", "D"}
    names = {
        str(item.get("name"))
        for item in sets
        if isinstance(item, dict) and item.get("name") is not None
    }
    if names != expected:
        raise ValueError(f"feature sets must be exactly {sorted(expected)}")
    allowed = {value.value for value in CandidateTimeAvailability}
    for feature_set in sets:
        if not isinstance(feature_set, dict):
            raise ValueError("feature-set entries must be objects")
        if feature_set.get("lg_r2_training_authorized") is not False:
            raise ValueError("LG-R1c must not authorize LG-R2 training")
        features = feature_set.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("every feature set must contain features")
        for feature in features:
            if not isinstance(feature, dict):
                raise ValueError("feature entries must be objects")
            shape = feature.get("shape")
            if not isinstance(shape, list) or not shape:
                raise ValueError("every feature requires a non-empty shape")
            availability = feature.get("candidate_time_availability")
            if availability not in allowed:
                raise ValueError(f"invalid candidate-time availability: {availability}")
            usable = feature.get("candidate_time_usable")
            if not isinstance(usable, bool):
                raise ValueError("candidate_time_usable must be boolean")
            if (
                availability
                in {
                    CandidateTimeAvailability.EXECUTED_TRAJECTORY_REQUIRED.value,
                    CandidateTimeAvailability.FUTURE_OBSERVATION_REQUIRED.value,
                    CandidateTimeAvailability.NOT_USABLE_FOR_PRE_EXECUTION_SELECTION.value,
                }
                and usable
            ):
                raise ValueError(
                    "post-execution features cannot be candidate-time usable"
                )


def evaluate_reward_gate(
    candidate: dict[str, float | None],
    *,
    prevalence: float,
    thresholds: RewardGateThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen LG-R1c promotion gate without imputing missing metrics."""

    limits = thresholds or RewardGateThresholds()
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("prevalence must lie in [0, 1]")
    required = {
        "progress_spearman",
        "pairwise_accuracy",
        "task_macro_success_auroc",
        "failure_auprc",
        "failure_precision_at_recall",
        "failure_recall",
        "failure_fpr_at_recall",
        "leave_task6_failure_auprc",
    }
    missing = sorted(key for key in required if candidate.get(key) is None)
    checks: dict[str, bool] = {}
    if not missing:
        values: dict[str, float] = {}
        for key in required:
            value = candidate.get(key)
            if value is None:
                raise AssertionError("missing metric escaped the fail-closed check")
            values[key] = float(value)
        failure_floor = max(
            limits.failure_prevalence_multiplier * prevalence,
            limits.failure_auprc_floor,
        )
        checks = {
            "progress_spearman": (
                values["progress_spearman"] >= limits.progress_spearman_min
            ),
            "pairwise_accuracy": (
                values["pairwise_accuracy"] >= limits.pairwise_accuracy_min
            ),
            "task_macro_success_auroc": (
                values["task_macro_success_auroc"]
                >= limits.task_macro_success_auroc_min
            ),
            "failure_auprc": values["failure_auprc"] >= failure_floor,
            "failure_precision": (
                values["failure_precision_at_recall"] >= limits.failure_precision_min
            ),
            "failure_recall": (values["failure_recall"] >= limits.failure_recall_min),
            "failure_fpr": (values["failure_fpr_at_recall"] <= limits.failure_fpr_max),
            "leave_task6_stability": (
                values["failure_auprc"] - values["leave_task6_failure_auprc"]
                <= limits.leave_task6_auprc_drop_max
            ),
        }
    passed = not missing and all(checks.values())
    return {
        "schema_version": "latentguard.lg_r1c.reward_gate.v1",
        "status": "pass" if passed else "not_promoted",
        "LG_R2_REWARD_BASELINE_AUTHORIZED": passed,
        "missing_metrics": missing,
        "checks": checks,
        "prevalence": prevalence,
        "required_failure_auprc": max(
            limits.failure_prevalence_multiplier * prevalence,
            limits.failure_auprc_floor,
        ),
        "thresholds": asdict(limits),
    }
