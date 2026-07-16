"""Frozen-artifact evaluation for direct action-verifier logits."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.calibration import (
    FrozenThresholdStateV1,
    TemperatureCalibrationStateV1,
    logits_to_failure_probabilities,
)
from latentguard.training.metrics import (
    BinaryMetricReport,
    CoverageRiskReport,
    GroupCandidate,
    GroupRankingReport,
    SliceMetricReport,
    evaluate_binary_metrics,
    evaluate_coverage_risk,
    evaluate_group_ranking,
    evaluate_slices,
)

EVALUATION_SCHEMA_VERSION = "1.0"


def _fail(context: str, reason: str) -> NoReturn:
    raise ValueError(f"{context}: {reason}")


def _require_digest(value: str, context: str) -> None:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")


def _digest(value: object, *, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _logits(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "fiu":
        _fail("logits", "expected a rank-one numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        _fail("logits", "all values must be finite")
    return result


def _targets(value: ArrayLike, count: int) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != count or array.dtype.kind not in "biu":
        _fail("targets", "expected a matching rank-one boolean/integer array")
    result = np.asarray(array, dtype=np.int64)
    if not bool(np.all((result == 0) | (result == 1))):
        _fail("targets", "values must be zero or one")
    return result


@dataclass(frozen=True, slots=True)
class EvaluationMetadataV1:
    """Reporting-only metadata kept outside all learned-model inputs."""

    sample_index: int
    sample_id: str
    group_id: str
    source_trajectory_id: str
    split: str
    candidate_type: str
    corruption_family: str | None
    corruption_severity: str | None
    anchor_selection_reason: str

    def __post_init__(self) -> None:
        """Validate one metadata join record without inferring missing values."""

        if type(self.sample_index) is not int or self.sample_index < 0:
            _fail("sample_index", "expected a non-negative integer")
        for context, value in (
            ("sample_id", self.sample_id),
            ("group_id", self.group_id),
            ("source_trajectory_id", self.source_trajectory_id),
            ("split", self.split),
            ("candidate_type", self.candidate_type),
            ("anchor_selection_reason", self.anchor_selection_reason),
        ):
            if not value or value != value.strip():
                _fail(context, "expected canonical non-empty text")
        if self.split not in {"train", "validation", "test"}:
            _fail("split", "expected train, validation, or test")
        if self.candidate_type not in {"source", "corrupted"}:
            _fail("candidate_type", "expected source or corrupted")
        optional = (self.corruption_family, self.corruption_severity)
        if self.candidate_type == "source":
            if any(value is not None for value in optional):
                _fail(
                    "candidate_type",
                    "source metadata cannot carry corruption fields",
                )
        elif any(
            value is None or not value or value != value.strip() for value in optional
        ):
            _fail("candidate_type", "corrupted metadata requires family and severity")


@dataclass(frozen=True, slots=True)
class CompactPredictionV1:
    """Sanitized prediction record with no model inputs or runtime paths."""

    sample_id: str
    group_id: str
    source_trajectory_id: str
    split: str
    candidate_type: str
    failure_target: int
    raw_logit: float
    uncalibrated_failure_probability: float
    calibrated_failure_probability: float | None

    def __post_init__(self) -> None:
        """Reject malformed or non-finite compact prediction content."""

        if type(self.failure_target) is not int or self.failure_target not in {0, 1}:
            _fail("failure_target", "expected zero or one")
        for context, value in (
            ("raw_logit", self.raw_logit),
            (
                "uncalibrated_failure_probability",
                self.uncalibrated_failure_probability,
            ),
        ):
            if not math.isfinite(value):
                _fail(context, "expected a finite number")
        if not 0.0 <= self.uncalibrated_failure_probability <= 1.0:
            _fail("uncalibrated_failure_probability", "expected value in [0, 1]")
        if self.calibrated_failure_probability is not None and (
            not math.isfinite(self.calibrated_failure_probability)
            or not 0.0 <= self.calibrated_failure_probability <= 1.0
        ):
            _fail("calibrated_failure_probability", "expected value in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native prediction record."""

        return {
            "calibrated_failure_probability": self.calibrated_failure_probability,
            "candidate_type": self.candidate_type,
            "failure_target": self.failure_target,
            "group_id": self.group_id,
            "raw_logit": self.raw_logit,
            "sample_id": self.sample_id,
            "source_trajectory_id": self.source_trajectory_id,
            "split": self.split,
            "uncalibrated_failure_probability": (self.uncalibrated_failure_probability),
        }


@dataclass(frozen=True, slots=True)
class ProbabilityEvaluationReport:
    """Metrics for one uncalibrated or calibrated probability representation."""

    probability_semantic: str
    candidate_metrics: BinaryMetricReport
    corrupted_only_metrics: BinaryMetricReport
    group_ranking: GroupRankingReport
    candidate_coverage_risk: CoverageRiskReport
    corrupted_only_coverage_risk: CoverageRiskReport
    slices: tuple[SliceMetricReport, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native probability report."""

        return {
            "candidate_coverage_risk": self.candidate_coverage_risk.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "corrupted_only_coverage_risk": (
                self.corrupted_only_coverage_risk.to_dict()
            ),
            "corrupted_only_metrics": self.corrupted_only_metrics.to_dict(),
            "group_ranking": self.group_ranking.to_dict(),
            "probability_semantic": self.probability_semantic,
            "slices": [item.to_dict() for item in self.slices],
        }


@dataclass(frozen=True, slots=True)
class ThresholdEvaluationReport:
    """Test or validation metrics under one already frozen threshold."""

    policy: str
    threshold: float
    candidate_metrics: BinaryMetricReport
    corrupted_only_metrics: BinaryMetricReport

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native threshold report."""

        return {
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "corrupted_only_metrics": self.corrupted_only_metrics.to_dict(),
            "policy": self.policy,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class ActionVerifierEvaluationReportV1:
    """Complete immutable evaluation result for one checkpoint and split."""

    split: str
    dataset_digest: str
    split_digest: str
    checkpoint_identity: str
    checkpoint_content_digest: str
    model_config_digest: str
    preprocessing_digest: str
    selection_digest: str | None
    calibration_digest: str | None
    threshold_digest: str | None
    prediction_digest: str
    sample_count: int
    uncalibrated: ProbabilityEvaluationReport
    calibrated: ProbabilityEvaluationReport | None
    threshold_evaluations: tuple[ThresholdEvaluationReport, ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate content bindings and mandatory frozen test artifacts."""

        for context, value in (
            ("dataset_digest", self.dataset_digest),
            ("split_digest", self.split_digest),
            ("checkpoint_content_digest", self.checkpoint_content_digest),
            ("model_config_digest", self.model_config_digest),
            ("preprocessing_digest", self.preprocessing_digest),
            ("prediction_digest", self.prediction_digest),
        ):
            _require_digest(value, context)
        for context, optional_digest in (
            ("selection_digest", self.selection_digest),
            ("calibration_digest", self.calibration_digest),
            ("threshold_digest", self.threshold_digest),
        ):
            if optional_digest is not None:
                _require_digest(optional_digest, context)
        if self.split not in {"train", "validation", "test"}:
            _fail("split", "unsupported evaluation split")
        if not self.checkpoint_identity or self.checkpoint_identity != (
            self.checkpoint_identity.strip()
        ):
            _fail("checkpoint_identity", "expected canonical text")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            _fail("sample_count", "expected a positive integer")
        if self.split == "test" and (
            self.selection_digest is None
            or self.calibration_digest is None
            or self.threshold_digest is None
            or self.calibrated is None
            or len(self.threshold_evaluations) != 2
        ):
            _fail("test", "test evaluation requires all frozen validation artifacts")
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            _fail("schema_version", "unsupported evaluation schema")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native evaluation report."""

        return {
            "calibrated": (
                None if self.calibrated is None else self.calibrated.to_dict()
            ),
            "calibration_digest": self.calibration_digest,
            "checkpoint_content_digest": self.checkpoint_content_digest,
            "checkpoint_identity": self.checkpoint_identity,
            "dataset_digest": self.dataset_digest,
            "model_config_digest": self.model_config_digest,
            "prediction_digest": self.prediction_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "selection_digest": self.selection_digest,
            "split": self.split,
            "split_digest": self.split_digest,
            "threshold_digest": self.threshold_digest,
            "threshold_evaluations": [
                item.to_dict() for item in self.threshold_evaluations
            ],
            "uncalibrated": self.uncalibrated.to_dict(),
        }


def _probability_report(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    metadata: Sequence[EvaluationMetadataV1],
    *,
    semantic: str,
    reliability_bin_count: int,
) -> ProbabilityEvaluationReport:
    corrupted_mask = np.asarray(
        [item.candidate_type == "corrupted" for item in metadata], dtype=np.bool_
    )
    group_candidates = tuple(
        GroupCandidate(
            sample_id=item.sample_id,
            group_id=item.group_id,
            source_trajectory_id=item.source_trajectory_id,
            candidate_type=item.candidate_type,
            failure_target=int(labels[index]),
            failure_probability=float(probabilities[index]),
        )
        for index, item in enumerate(metadata)
    )
    slice_metadata: dict[str, Sequence[str | None]] = {
        "anchor_selection_reason": [item.anchor_selection_reason for item in metadata],
        "candidate_type": [item.candidate_type for item in metadata],
        "corruption_family": [item.corruption_family for item in metadata],
        "corruption_severity": [item.corruption_severity for item in metadata],
        "source_trajectory": [item.source_trajectory_id for item in metadata],
        "split": [item.split for item in metadata],
    }
    return ProbabilityEvaluationReport(
        probability_semantic=semantic,
        candidate_metrics=evaluate_binary_metrics(
            labels,
            probabilities,
            reliability_bin_count=reliability_bin_count,
        ),
        corrupted_only_metrics=evaluate_binary_metrics(
            labels[corrupted_mask],
            probabilities[corrupted_mask],
            reliability_bin_count=reliability_bin_count,
        ),
        group_ranking=evaluate_group_ranking(group_candidates),
        candidate_coverage_risk=evaluate_coverage_risk(labels, probabilities),
        corrupted_only_coverage_risk=evaluate_coverage_risk(
            labels[corrupted_mask], probabilities[corrupted_mask]
        ),
        slices=evaluate_slices(
            labels,
            probabilities,
            slice_metadata,
            reliability_bin_count=reliability_bin_count,
        ),
    )


def _prediction_digest(predictions: Sequence[CompactPredictionV1]) -> str:
    ordered = sorted(predictions, key=lambda item: item.sample_id)
    return _digest(
        {
            "predictions": [item.to_dict() for item in ordered],
            "semantic": "compact_action_verifier_predictions_v1",
        },
        context="CompactPredictionSetV1",
    )


def evaluate_action_verifier_logits(
    logits: ArrayLike,
    failure_targets: ArrayLike,
    metadata: Sequence[EvaluationMetadataV1],
    *,
    split: str,
    dataset_digest: str,
    split_digest: str,
    checkpoint_identity: str,
    checkpoint_content_digest: str,
    model_config_digest: str,
    preprocessing_digest: str,
    calibration: TemperatureCalibrationStateV1 | None = None,
    thresholds: FrozenThresholdStateV1 | None = None,
    selection_digest: str | None = None,
    reliability_bin_count: int = 10,
) -> tuple[ActionVerifierEvaluationReportV1, tuple[CompactPredictionV1, ...]]:
    """Evaluate logits without exposing any calibration or threshold fit path."""

    values = _logits(logits)
    labels = _targets(failure_targets, values.size)
    if len(metadata) != values.size or values.size == 0:
        _fail("metadata", "must match one or more predictions")
    if tuple(item.sample_index for item in metadata) != tuple(range(values.size)):
        _fail("metadata", "sample_index must exactly join the ordered predictions")
    if len({item.sample_id for item in metadata}) != len(metadata):
        _fail("metadata", "sample identifiers must be unique")
    if any(item.split != split for item in metadata):
        _fail("split", "metadata contains a different split")
    for context, digest_value in (
        ("dataset_digest", dataset_digest),
        ("split_digest", split_digest),
        ("checkpoint_content_digest", checkpoint_content_digest),
        ("model_config_digest", model_config_digest),
        ("preprocessing_digest", preprocessing_digest),
    ):
        _require_digest(digest_value, context)
    if selection_digest is not None:
        _require_digest(selection_digest, "selection_digest")
    uncalibrated_probabilities = logits_to_failure_probabilities(values)
    if calibration is None:
        calibrated_probabilities = None
        calibration_digest = None
    else:
        if (
            calibration.dataset_digest != dataset_digest
            or calibration.split_digest != split_digest
            or calibration.checkpoint_identity != checkpoint_identity
            or calibration.checkpoint_content_digest != checkpoint_content_digest
            or calibration.model_config_digest != model_config_digest
            or calibration.preprocessing_digest != preprocessing_digest
        ):
            _fail("calibration", "checkpoint or content binding mismatch")
        calibrated_probabilities = calibration.apply(values)
        calibration_digest = calibration.content_digest
    if thresholds is not None:
        if calibration is None:
            _fail("thresholds", "frozen thresholds require frozen calibration")
        if (
            thresholds.dataset_digest != dataset_digest
            or thresholds.split_digest != split_digest
            or thresholds.calibration_digest != calibration.content_digest
            or thresholds.validation_prediction_digest
            != calibration.validation_prediction_digest
        ):
            _fail("thresholds", "calibration or prediction binding mismatch")
        threshold_digest = thresholds.content_digest
    else:
        threshold_digest = None
    if split == "test" and (
        selection_digest is None or calibration is None or thresholds is None
    ):
        _fail(
            "test",
            "test evaluation requires frozen selection/calibration/thresholds",
        )
    predictions = tuple(
        CompactPredictionV1(
            sample_id=item.sample_id,
            group_id=item.group_id,
            source_trajectory_id=item.source_trajectory_id,
            split=item.split,
            candidate_type=item.candidate_type,
            failure_target=int(labels[index]),
            raw_logit=float(values[index]),
            uncalibrated_failure_probability=float(uncalibrated_probabilities[index]),
            calibrated_failure_probability=(
                None
                if calibrated_probabilities is None
                else float(calibrated_probabilities[index])
            ),
        )
        for index, item in enumerate(metadata)
    )
    uncalibrated_report = _probability_report(
        labels,
        uncalibrated_probabilities,
        metadata,
        semantic="uncalibrated_sigmoid_failure_probability_v1",
        reliability_bin_count=reliability_bin_count,
    )
    if calibrated_probabilities is None:
        calibrated_report = None
    else:
        calibrated_report = _probability_report(
            labels,
            calibrated_probabilities,
            metadata,
            semantic="validation_temperature_scaled_failure_probability_v1",
            reliability_bin_count=reliability_bin_count,
        )
    threshold_reports: list[ThresholdEvaluationReport] = []
    if thresholds is not None and calibrated_probabilities is not None:
        corrupted_mask = np.asarray(
            [item.candidate_type == "corrupted" for item in metadata], dtype=np.bool_
        )
        for policy in (
            thresholds.maximum_balanced_accuracy,
            thresholds.target_failure_recall,
        ):
            threshold_reports.append(
                ThresholdEvaluationReport(
                    policy=policy.policy,
                    threshold=policy.threshold,
                    candidate_metrics=evaluate_binary_metrics(
                        labels,
                        calibrated_probabilities,
                        threshold=policy.threshold,
                        reliability_bin_count=reliability_bin_count,
                    ),
                    corrupted_only_metrics=evaluate_binary_metrics(
                        labels[corrupted_mask],
                        calibrated_probabilities[corrupted_mask],
                        threshold=policy.threshold,
                        reliability_bin_count=reliability_bin_count,
                    ),
                )
            )
    report = ActionVerifierEvaluationReportV1(
        split=split,
        dataset_digest=dataset_digest,
        split_digest=split_digest,
        checkpoint_identity=checkpoint_identity,
        checkpoint_content_digest=checkpoint_content_digest,
        model_config_digest=model_config_digest,
        preprocessing_digest=preprocessing_digest,
        selection_digest=selection_digest,
        calibration_digest=calibration_digest,
        threshold_digest=threshold_digest,
        prediction_digest=_prediction_digest(predictions),
        sample_count=values.size,
        uncalibrated=uncalibrated_report,
        calibrated=calibrated_report,
        threshold_evaluations=tuple(threshold_reports),
    )
    return report, predictions


__all__ = [
    "ActionVerifierEvaluationReportV1",
    "CompactPredictionV1",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationMetadataV1",
    "ProbabilityEvaluationReport",
    "ThresholdEvaluationReport",
    "evaluate_action_verifier_logits",
]
