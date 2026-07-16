"""Validation-only temperature calibration and frozen threshold policies."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.metrics import (
    MetricStatus,
    MetricValue,
    evaluate_binary_metrics,
)

CALIBRATION_SCHEMA_VERSION = "1.0"
THRESHOLD_SCHEMA_VERSION = "1.0"
TEMPERATURE_METHOD = "bounded_log_temperature_golden_section_v1"
THRESHOLD_COMPARISON_SEMANTIC = "predict_failure_when_probability_gte_threshold_v1"


def _fail(context: str, reason: str) -> NoReturn:
    raise ValueError(f"{context}: {reason}")


def _digest(value: object, *, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _require_digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")
    return value


def _finite_logits(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "fiu":
        _fail("logits", "expected a rank-one numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        _fail("logits", "all values must be finite")
    return result


def _binary_targets(value: ArrayLike, expected_count: int) -> NDArray[np.int64]:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.shape[0] != expected_count
        or array.dtype.kind not in "biu"
    ):
        _fail("targets", "expected a matching rank-one boolean/integer array")
    result = np.asarray(array, dtype=np.int64)
    if not bool(np.all((result == 0) | (result == 1))):
        _fail("targets", "values must be zero or one")
    return result


def _finite_probabilities(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "fiu":
        _fail("probabilities", "expected a rank-one numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))) or not bool(
        np.all((result >= 0.0) & (result <= 1.0))
    ):
        _fail("probabilities", "values must be finite and lie in [0, 1]")
    return result


def logits_to_failure_probabilities(logits: ArrayLike) -> NDArray[np.float64]:
    """Convert finite logits to stable double-precision failure probabilities."""

    values = _finite_logits(logits)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def compute_validation_prediction_digest(
    logits: ArrayLike,
    targets: ArrayLike,
) -> str:
    """Bind one ordered validation logit/target vector without JSON float drift."""

    values = _finite_logits(logits)
    labels = _binary_targets(targets, values.size)
    payload = {
        "logits_dtype": values.dtype.str,
        "logits_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "sample_count": values.size,
        "semantic": "ordered_validation_logits_and_failure_targets_v1",
        "targets_dtype": labels.dtype.str,
        "targets_sha256": hashlib.sha256(labels.tobytes(order="C")).hexdigest(),
    }
    return _digest(payload, context="ValidationPredictionDigestV1")


@dataclass(frozen=True, slots=True)
class TemperatureCalibrationStateV1:
    """Content-bound scalar temperature fitted only on validation logits."""

    dataset_digest: str
    split_digest: str
    checkpoint_identity: str
    checkpoint_content_digest: str
    model_config_digest: str
    preprocessing_digest: str
    validation_prediction_digest: str
    temperature: float
    method: str = TEMPERATURE_METHOD
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate semantic identity and positive finite temperature."""

        _require_digest(self.dataset_digest, "dataset_digest")
        _require_digest(self.split_digest, "split_digest")
        _text(self.checkpoint_identity, "checkpoint_identity")
        _require_digest(self.checkpoint_content_digest, "checkpoint_content_digest")
        _require_digest(self.model_config_digest, "model_config_digest")
        _require_digest(self.preprocessing_digest, "preprocessing_digest")
        _require_digest(
            self.validation_prediction_digest, "validation_prediction_digest"
        )
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            _fail("temperature", "expected a positive finite scalar")
        if self.method != TEMPERATURE_METHOD:
            _fail("method", f"expected {TEMPERATURE_METHOD!r}")
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            _fail("schema_version", "unsupported calibration schema")

    @property
    def content_digest(self) -> str:
        """Return the path-independent calibration-state digest."""

        return _digest(self._payload(), context="TemperatureCalibrationStateV1")

    def _payload(self) -> dict[str, object]:
        return {
            "checkpoint_content_digest": self.checkpoint_content_digest,
            "checkpoint_identity": self.checkpoint_identity,
            "dataset_digest": self.dataset_digest,
            "method": self.method,
            "model_config_digest": self.model_config_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "schema_version": self.schema_version,
            "split_digest": self.split_digest,
            "temperature": self.temperature,
            "validation_prediction_digest": self.validation_prediction_digest,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a strict self-digesting JSON representation."""

        return {**self._payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TemperatureCalibrationStateV1:
        """Load strict calibration content and reject field or digest tampering."""

        expected = {
            "checkpoint_content_digest",
            "checkpoint_identity",
            "content_digest",
            "dataset_digest",
            "method",
            "model_config_digest",
            "preprocessing_digest",
            "schema_version",
            "split_digest",
            "temperature",
            "validation_prediction_digest",
        }
        if set(value) != expected:
            _fail("TemperatureCalibrationStateV1", "unexpected or missing fields")
        temperature = value["temperature"]
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            _fail("temperature", "expected a number")
        result = cls(
            dataset_digest=_text(value["dataset_digest"], "dataset_digest"),
            split_digest=_text(value["split_digest"], "split_digest"),
            checkpoint_identity=_text(
                value["checkpoint_identity"], "checkpoint_identity"
            ),
            checkpoint_content_digest=_text(
                value["checkpoint_content_digest"], "checkpoint_content_digest"
            ),
            model_config_digest=_text(
                value["model_config_digest"], "model_config_digest"
            ),
            preprocessing_digest=_text(
                value["preprocessing_digest"], "preprocessing_digest"
            ),
            validation_prediction_digest=_text(
                value["validation_prediction_digest"],
                "validation_prediction_digest",
            ),
            temperature=float(temperature),
            method=_text(value["method"], "method"),
            schema_version=_text(value["schema_version"], "schema_version"),
        )
        expected_digest = _require_digest(value["content_digest"], "content_digest")
        if result.content_digest != expected_digest:
            _fail("content_digest", "calibration content changed")
        return result

    def apply(self, logits: ArrayLike) -> NDArray[np.float64]:
        """Apply the frozen temperature and return failure probabilities."""

        return logits_to_failure_probabilities(
            _finite_logits(logits) / self.temperature
        )


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _number(value: object, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(context, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(context, "expected a finite number")
    return result


def _metric_value_from_dict(value: object, context: str) -> MetricValue:
    if not isinstance(value, Mapping):
        _fail(context, "expected an object")
    if set(value) != {"reason", "status", "value"}:
        _fail(context, "unexpected or missing fields")
    try:
        status = MetricStatus(_text(value["status"], f"{context}.status"))
    except ValueError as exc:
        raise ValueError(f"{context}.status: unsupported metric status") from exc
    raw_reason = value["reason"]
    reason = None if raw_reason is None else _text(raw_reason, f"{context}.reason")
    raw_metric = value["value"]
    metric = None if raw_metric is None else _number(raw_metric, f"{context}.value")
    return MetricValue(value=metric, status=status, reason=reason)


def _threshold_policy_from_dict(value: object, context: str) -> ThresholdPolicyResult:
    if not isinstance(value, Mapping):
        _fail(context, "expected an object")
    expected = {
        "achieved_failure_recall",
        "policy",
        "target_failure_recall",
        "target_met",
        "threshold",
        "validation_balanced_accuracy",
    }
    if set(value) != expected:
        _fail(context, "unexpected or missing fields")
    raw_target = value["target_failure_recall"]
    target = (
        None
        if raw_target is None
        else _number(raw_target, f"{context}.target_failure_recall")
    )
    raw_met = value["target_met"]
    if raw_met is not None and type(raw_met) is not bool:
        _fail(f"{context}.target_met", "expected a boolean or null")
    return ThresholdPolicyResult(
        policy=_text(value["policy"], f"{context}.policy"),
        threshold=_number(value["threshold"], f"{context}.threshold"),
        validation_balanced_accuracy=_metric_value_from_dict(
            value["validation_balanced_accuracy"],
            f"{context}.validation_balanced_accuracy",
        ),
        achieved_failure_recall=_metric_value_from_dict(
            value["achieved_failure_recall"],
            f"{context}.achieved_failure_recall",
        ),
        target_failure_recall=target,
        target_met=raw_met,
    )


def _binary_nll(
    logits: NDArray[np.float64],
    targets: NDArray[np.int64],
    log_temperature: float,
) -> float:
    temperature = math.exp(log_temperature)
    scaled = logits / temperature
    losses = (
        np.maximum(scaled, 0.0) - scaled * targets + np.log1p(np.exp(-np.abs(scaled)))
    )
    return float(np.mean(losses))


def fit_temperature_scaling(
    logits: ArrayLike,
    targets: ArrayLike,
    *,
    split: str,
    dataset_digest: str,
    split_digest: str,
    checkpoint_identity: str,
    checkpoint_content_digest: str,
    model_config_digest: str,
    preprocessing_digest: str,
) -> TemperatureCalibrationStateV1:
    """Fit one scalar temperature, refusing any split except validation."""

    if split != "validation":
        _fail("split", "temperature scaling may be fitted only on validation")
    values = _finite_logits(logits)
    labels = _binary_targets(targets, values.size)
    if values.size == 0 or int(np.min(labels)) == int(np.max(labels)):
        _fail("targets", "temperature fitting requires both validation classes")
    _require_digest(dataset_digest, "dataset_digest")
    _require_digest(split_digest, "split_digest")
    _text(checkpoint_identity, "checkpoint_identity")
    _require_digest(checkpoint_content_digest, "checkpoint_content_digest")
    _require_digest(model_config_digest, "model_config_digest")
    _require_digest(preprocessing_digest, "preprocessing_digest")
    lower = -8.0
    upper = 8.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = _binary_nll(values, labels, left)
    right_value = _binary_nll(values, labels, right)
    for _ in range(160):
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = _binary_nll(values, labels, left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = _binary_nll(values, labels, right)
    candidates = ((lower + upper) / 2.0, 0.0)
    best_log_temperature = min(
        candidates, key=lambda candidate: _binary_nll(values, labels, candidate)
    )
    return TemperatureCalibrationStateV1(
        dataset_digest=dataset_digest,
        split_digest=split_digest,
        checkpoint_identity=checkpoint_identity,
        checkpoint_content_digest=checkpoint_content_digest,
        model_config_digest=model_config_digest,
        preprocessing_digest=preprocessing_digest,
        validation_prediction_digest=compute_validation_prediction_digest(
            values, labels
        ),
        temperature=math.exp(best_log_temperature),
    )


@dataclass(frozen=True, slots=True)
class ThresholdPolicyResult:
    """One validation-selected deterministic threshold policy."""

    policy: str
    threshold: float
    validation_balanced_accuracy: MetricValue
    achieved_failure_recall: MetricValue
    target_failure_recall: float | None
    target_met: bool | None

    def __post_init__(self) -> None:
        """Validate policy-specific target fields and normalized threshold."""

        if self.policy not in {
            "maximum_validation_balanced_accuracy",
            "target_validation_failure_recall",
        }:
            _fail("policy", "unsupported threshold policy")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            _fail("threshold", "expected a finite value in [0, 1]")
        target_policy = self.policy == "target_validation_failure_recall"
        if target_policy:
            if (
                self.target_failure_recall is None
                or not math.isfinite(self.target_failure_recall)
                or not 0.0 < self.target_failure_recall <= 1.0
                or type(self.target_met) is not bool
            ):
                _fail("target_failure_recall", "invalid target policy state")
        elif self.target_failure_recall is not None or self.target_met is not None:
            _fail("target_failure_recall", "balanced-accuracy policy has no target")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native policy result."""

        return {
            "achieved_failure_recall": self.achieved_failure_recall.to_dict(),
            "policy": self.policy,
            "target_failure_recall": self.target_failure_recall,
            "target_met": self.target_met,
            "threshold": self.threshold,
            "validation_balanced_accuracy": (
                self.validation_balanced_accuracy.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class FrozenThresholdStateV1:
    """Two threshold policies fitted on one frozen validation prediction set."""

    dataset_digest: str
    split_digest: str
    calibration_digest: str
    validation_prediction_digest: str
    maximum_balanced_accuracy: ThresholdPolicyResult
    target_failure_recall: ThresholdPolicyResult
    comparison_semantic: str = THRESHOLD_COMPARISON_SEMANTIC
    schema_version: str = THRESHOLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate content bindings and the two required policies."""

        for context, value in (
            ("dataset_digest", self.dataset_digest),
            ("split_digest", self.split_digest),
            ("calibration_digest", self.calibration_digest),
            ("validation_prediction_digest", self.validation_prediction_digest),
        ):
            _require_digest(value, context)
        if self.maximum_balanced_accuracy.policy != (
            "maximum_validation_balanced_accuracy"
        ):
            _fail("maximum_balanced_accuracy", "wrong policy")
        if self.target_failure_recall.policy != "target_validation_failure_recall":
            _fail("target_failure_recall", "wrong policy")
        if self.comparison_semantic != THRESHOLD_COMPARISON_SEMANTIC:
            _fail("comparison_semantic", "unsupported threshold comparison")
        if self.schema_version != THRESHOLD_SCHEMA_VERSION:
            _fail("schema_version", "unsupported threshold schema")

    @property
    def content_digest(self) -> str:
        """Return the path-independent frozen-threshold digest."""

        return _digest(self._payload(), context="FrozenThresholdStateV1")

    def _payload(self) -> dict[str, object]:
        return {
            "calibration_digest": self.calibration_digest,
            "comparison_semantic": self.comparison_semantic,
            "dataset_digest": self.dataset_digest,
            "maximum_balanced_accuracy": self.maximum_balanced_accuracy.to_dict(),
            "schema_version": self.schema_version,
            "split_digest": self.split_digest,
            "target_failure_recall": self.target_failure_recall.to_dict(),
            "validation_prediction_digest": self.validation_prediction_digest,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a strict self-digesting JSON representation."""

        return {**self._payload(), "content_digest": self.content_digest}


def temperature_calibration_from_dict(
    value: Mapping[str, object],
) -> TemperatureCalibrationStateV1:
    """Strictly load a self-digesting scalar-temperature state."""

    return TemperatureCalibrationStateV1.from_dict(value)


def frozen_thresholds_from_dict(
    value: Mapping[str, object],
) -> FrozenThresholdStateV1:
    """Strictly load both frozen policies and reject content tampering."""

    expected = {
        "calibration_digest",
        "comparison_semantic",
        "content_digest",
        "dataset_digest",
        "maximum_balanced_accuracy",
        "schema_version",
        "split_digest",
        "target_failure_recall",
        "validation_prediction_digest",
    }
    if set(value) != expected:
        _fail("FrozenThresholdStateV1", "unexpected or missing fields")
    result = FrozenThresholdStateV1(
        dataset_digest=_text(value["dataset_digest"], "dataset_digest"),
        split_digest=_text(value["split_digest"], "split_digest"),
        calibration_digest=_text(value["calibration_digest"], "calibration_digest"),
        validation_prediction_digest=_text(
            value["validation_prediction_digest"],
            "validation_prediction_digest",
        ),
        maximum_balanced_accuracy=_threshold_policy_from_dict(
            value["maximum_balanced_accuracy"], "maximum_balanced_accuracy"
        ),
        target_failure_recall=_threshold_policy_from_dict(
            value["target_failure_recall"], "target_failure_recall"
        ),
        comparison_semantic=_text(value["comparison_semantic"], "comparison_semantic"),
        schema_version=_text(value["schema_version"], "schema_version"),
    )
    expected_digest = _require_digest(value["content_digest"], "content_digest")
    if result.content_digest != expected_digest:
        _fail("content_digest", "threshold content changed")
    return result


def _threshold_candidates(probabilities: NDArray[np.float64]) -> tuple[float, ...]:
    if probabilities.size == 0:
        return (0.5,)
    unique = sorted({float(value) for value in probabilities}, reverse=True)
    maximum = unique[0]
    if maximum < 1.0:
        no_positive = float(np.nextafter(maximum, 1.0))
        unique.insert(0, no_positive)
    return tuple(unique)


def fit_validation_thresholds(
    probabilities: ArrayLike,
    targets: ArrayLike,
    *,
    split: str,
    dataset_digest: str,
    split_digest: str,
    calibration_digest: str,
    validation_prediction_digest: str,
    target_failure_recall: float = 0.90,
) -> FrozenThresholdStateV1:
    """Fit both required policies, refusing any split except validation."""

    if split != "validation":
        _fail("split", "thresholds may be fitted only on validation")
    scores = _finite_probabilities(probabilities)
    labels = _binary_targets(targets, scores.size)
    if scores.size == 0:
        _fail("probabilities", "threshold fitting requires validation samples")
    for context, value in (
        ("dataset_digest", dataset_digest),
        ("split_digest", split_digest),
        ("calibration_digest", calibration_digest),
        ("validation_prediction_digest", validation_prediction_digest),
    ):
        _require_digest(value, context)
    if not math.isfinite(target_failure_recall) or not (
        0.0 < target_failure_recall <= 1.0
    ):
        _fail("target_failure_recall", "expected a finite value in (0, 1]")
    candidates = _threshold_candidates(scores)
    evaluated = [
        (threshold, evaluate_binary_metrics(labels, scores, threshold=threshold))
        for threshold in candidates
    ]
    defined_balanced = [
        item for item in evaluated if item[1].balanced_accuracy.value is not None
    ]
    if defined_balanced:
        best_threshold, best_report = max(
            defined_balanced,
            key=lambda item: (item[1].balanced_accuracy.value, item[0]),
        )
    else:
        best_threshold, best_report = evaluated[0]
    balanced_policy = ThresholdPolicyResult(
        policy="maximum_validation_balanced_accuracy",
        threshold=best_threshold,
        validation_balanced_accuracy=best_report.balanced_accuracy,
        achieved_failure_recall=best_report.recall,
        target_failure_recall=None,
        target_met=None,
    )
    satisfying = [
        item
        for item in evaluated
        if item[1].recall.value is not None
        and item[1].recall.value >= target_failure_recall
    ]
    if satisfying:
        recall_threshold, recall_report = max(satisfying, key=lambda item: item[0])
        target_met = True
    else:
        defined_recall = [
            item for item in evaluated if item[1].recall.value is not None
        ]
        if defined_recall:
            recall_threshold, recall_report = max(
                defined_recall,
                key=lambda item: (item[1].recall.value, item[0]),
            )
        else:
            recall_threshold, recall_report = evaluated[0]
        target_met = False
    recall_policy = ThresholdPolicyResult(
        policy="target_validation_failure_recall",
        threshold=recall_threshold,
        validation_balanced_accuracy=recall_report.balanced_accuracy,
        achieved_failure_recall=recall_report.recall,
        target_failure_recall=float(target_failure_recall),
        target_met=target_met,
    )
    return FrozenThresholdStateV1(
        dataset_digest=dataset_digest,
        split_digest=split_digest,
        calibration_digest=calibration_digest,
        validation_prediction_digest=validation_prediction_digest,
        maximum_balanced_accuracy=balanced_policy,
        target_failure_recall=recall_policy,
    )


def apply_threshold_policy(
    probabilities: ArrayLike,
    policy: ThresholdPolicyResult,
) -> NDArray[np.bool_]:
    """Apply a frozen policy using its recorded inclusive comparison semantic."""

    scores = _finite_probabilities(probabilities)
    return np.asarray(scores >= policy.threshold, dtype=np.bool_)


__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "FrozenThresholdStateV1",
    "TEMPERATURE_METHOD",
    "THRESHOLD_COMPARISON_SEMANTIC",
    "THRESHOLD_SCHEMA_VERSION",
    "TemperatureCalibrationStateV1",
    "ThresholdPolicyResult",
    "apply_threshold_policy",
    "compute_validation_prediction_digest",
    "fit_temperature_scaling",
    "fit_validation_thresholds",
    "frozen_thresholds_from_dict",
    "logits_to_failure_probabilities",
    "temperature_calibration_from_dict",
]
