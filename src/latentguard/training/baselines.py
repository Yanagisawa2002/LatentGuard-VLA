"""Non-learned, training-split-only direct verifier baselines."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.action_verifier import DatasetSplit
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.dataset import (
    ACCEPTED_ACTION_DIMENSION,
    AcceptedActionVerifierDatasetV1,
    ActionVerifierModelExampleV1,
)

ACTION_MAGNITUDE_FEATURES = (
    "mean_absolute_action",
    "maximum_temporal_difference",
    "mean_absolute_training_mean_deviation",
)


class BaselineError(ValueError):
    """Raised when a non-learned baseline is fitted or applied unsafely."""


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _freeze_float64(
    value: NDArray[Any], *, shape: tuple[int, ...], context: str
) -> NDArray[Any]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("<f8")
        or value.shape != shape
        or not bool(np.all(np.isfinite(value)))
    ):
        raise BaselineError(f"{context}: expected finite float64 array {shape}")
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


@dataclass(frozen=True, slots=True)
class ConstantFailureBaselineV1:
    """One training-only constant failure-probability baseline."""

    baseline_type: str
    failure_probability: float
    training_failure_count: int
    training_sample_count: int
    dataset_digest: str
    training_split_digest: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate probability, training counts, and content bindings."""

        if self.baseline_type not in {"majority", "failure_prevalence"}:
            raise BaselineError("ConstantFailureBaselineV1: invalid baseline type")
        if (
            type(self.failure_probability) not in (int, float)
            or not math.isfinite(float(self.failure_probability))
            or not 0.0 <= float(self.failure_probability) <= 1.0
        ):
            raise BaselineError("ConstantFailureBaselineV1: invalid probability")
        if (
            type(self.training_sample_count) is not int
            or self.training_sample_count <= 0
            or type(self.training_failure_count) is not int
            or not 0 <= self.training_failure_count <= self.training_sample_count
        ):
            raise BaselineError("ConstantFailureBaselineV1: invalid training counts")
        if not _valid_digest(self.dataset_digest) or not _valid_digest(
            self.training_split_digest
        ):
            raise BaselineError("ConstantFailureBaselineV1: invalid content binding")
        if self.schema_version != "1.0":
            raise BaselineError("ConstantFailureBaselineV1: unsupported schema")
        object.__setattr__(self, "failure_probability", float(self.failure_probability))

    @property
    def content_digest(self) -> str:
        """Return the path-independent fitted-baseline identity."""

        payload = {
            "baseline_type": self.baseline_type,
            "dataset_digest": self.dataset_digest,
            "failure_probability": self.failure_probability,
            "schema_version": self.schema_version,
            "training_failure_count": self.training_failure_count,
            "training_sample_count": self.training_sample_count,
            "training_split_digest": self.training_split_digest,
        }
        encoded = canonical_json_bytes(payload, context="ConstantFailureBaselineV1")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def predict_failure_probability(self, sample_count: int) -> NDArray[Any]:
        """Return one explicit failure probability for every requested sample."""

        if type(sample_count) is not int or sample_count <= 0:
            raise BaselineError("constant baseline sample_count must be positive")
        values = np.full(sample_count, self.failure_probability, dtype=np.float32)
        return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype)


def fit_constant_baselines(
    dataset: AcceptedActionVerifierDatasetV1,
) -> tuple[ConstantFailureBaselineV1, ConstantFailureBaselineV1]:
    """Fit majority and failure-prevalence baselines from training targets only."""

    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        raise BaselineError("fit_constant_baselines: expected accepted dataset")
    indices = dataset.indices_for_split(DatasetSplit.TRAIN)
    targets = tuple(dataset[index].failure_target for index in indices)
    failures = sum(targets)
    count = len(targets)
    prevalence = failures / count
    majority_probability = 1.0 if failures > count - failures else 0.0
    return (
        ConstantFailureBaselineV1(
            baseline_type="majority",
            failure_probability=majority_probability,
            training_failure_count=failures,
            training_sample_count=count,
            dataset_digest=dataset.dataset_digest,
            training_split_digest=dataset.training_split_digest,
        ),
        ConstantFailureBaselineV1(
            baseline_type="failure_prevalence",
            failure_probability=prevalence,
            training_failure_count=failures,
            training_sample_count=count,
            dataset_digest=dataset.dataset_digest,
            training_split_digest=dataset.training_split_digest,
        ),
    )


def _valid_action_rows(example: ActionVerifierModelExampleV1) -> NDArray[Any]:
    rows = example.action_chunk[example.action_mask]
    if rows.shape[0] == 0:
        raise BaselineError("action-magnitude baseline requires a valid action step")
    result: NDArray[Any] = rows.astype(np.float64, copy=False)
    return result


def _action_features(
    example: ActionVerifierModelExampleV1, action_mean: NDArray[Any]
) -> NDArray[Any]:
    rows = _valid_action_rows(example)
    mean_absolute = float(np.mean(np.abs(rows)))
    maximum_difference = (
        0.0 if rows.shape[0] < 2 else float(np.max(np.abs(np.diff(rows, axis=0))))
    )
    mean_deviation = float(np.mean(np.abs(rows - action_mean)))
    return np.asarray(
        (mean_absolute, maximum_difference, mean_deviation), dtype=np.float64
    )


@dataclass(frozen=True, slots=True, eq=False)
class ActionMagnitudeBaselineV1:
    """Training-statistic action anomaly score with no corruption metadata."""

    dataset_digest: str
    training_split_digest: str
    training_sample_count: int
    valid_action_step_count: int
    minimum_standard_deviation: float
    action_mean: NDArray[Any]
    feature_mean: NDArray[Any]
    feature_standard_deviation: NDArray[Any]
    schema_version: str = "1.0"
    semantic: str = "training_action_magnitude_score_v1"

    def __post_init__(self) -> None:
        """Validate fitted train-only statistics and freeze them."""

        if not _valid_digest(self.dataset_digest) or not _valid_digest(
            self.training_split_digest
        ):
            raise BaselineError("ActionMagnitudeBaselineV1: invalid content binding")
        for name in ("training_sample_count", "valid_action_step_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise BaselineError(f"ActionMagnitudeBaselineV1.{name}: invalid count")
        if (
            type(self.minimum_standard_deviation) not in (int, float)
            or not math.isfinite(float(self.minimum_standard_deviation))
            or float(self.minimum_standard_deviation) <= 0.0
        ):
            raise BaselineError("ActionMagnitudeBaselineV1: invalid std floor")
        action_mean = _freeze_float64(
            self.action_mean,
            shape=(ACCEPTED_ACTION_DIMENSION,),
            context="ActionMagnitudeBaselineV1.action_mean",
        )
        feature_mean = _freeze_float64(
            self.feature_mean,
            shape=(len(ACTION_MAGNITUDE_FEATURES),),
            context="ActionMagnitudeBaselineV1.feature_mean",
        )
        feature_std = _freeze_float64(
            self.feature_standard_deviation,
            shape=(len(ACTION_MAGNITUDE_FEATURES),),
            context="ActionMagnitudeBaselineV1.feature_standard_deviation",
        )
        if bool(np.any(feature_std < float(self.minimum_standard_deviation))):
            raise BaselineError("ActionMagnitudeBaselineV1: std is below floor")
        if self.schema_version != "1.0" or self.semantic != (
            "training_action_magnitude_score_v1"
        ):
            raise BaselineError("ActionMagnitudeBaselineV1: unsupported schema")
        object.__setattr__(
            self, "minimum_standard_deviation", float(self.minimum_standard_deviation)
        )
        object.__setattr__(self, "action_mean", action_mean)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_standard_deviation", feature_std)

    @property
    def content_digest(self) -> str:
        """Return the fitted action-score identity."""

        payload = {
            "action_mean": self.action_mean.tolist(),
            "dataset_digest": self.dataset_digest,
            "feature_mean": self.feature_mean.tolist(),
            "feature_names": list(ACTION_MAGNITUDE_FEATURES),
            "feature_standard_deviation": self.feature_standard_deviation.tolist(),
            "minimum_standard_deviation": self.minimum_standard_deviation,
            "schema_version": self.schema_version,
            "semantic": self.semantic,
            "training_sample_count": self.training_sample_count,
            "training_split_digest": self.training_split_digest,
            "valid_action_step_count": self.valid_action_step_count,
        }
        encoded = canonical_json_bytes(payload, context="ActionMagnitudeBaselineV1")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def score(self, example: ActionVerifierModelExampleV1) -> float:
        """Return the deterministic monotonic standardized action score."""

        if not isinstance(example, ActionVerifierModelExampleV1):
            raise BaselineError("ActionMagnitudeBaselineV1.score: invalid example")
        features = _action_features(example, self.action_mean)
        score = float(
            np.mean((features - self.feature_mean) / self.feature_standard_deviation)
        )
        if not math.isfinite(score):
            raise BaselineError("ActionMagnitudeBaselineV1.score: non-finite score")
        return score

    def predict_failure_probability(
        self, examples: Sequence[ActionVerifierModelExampleV1]
    ) -> NDArray[Any]:
        """Map the monotonic diagnostic score through a fixed logistic transform."""

        values = tuple(examples)
        if not values:
            raise BaselineError("action-magnitude prediction requires examples")
        scores = np.asarray([self.score(item) for item in values], dtype=np.float64)
        clipped = np.clip(scores, -60.0, 60.0)
        probabilities = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
        return np.frombuffer(
            probabilities.tobytes(order="C"), dtype=probabilities.dtype
        )


def fit_action_magnitude_baseline(
    dataset: AcceptedActionVerifierDatasetV1,
    *,
    minimum_standard_deviation: float = 1e-6,
) -> ActionMagnitudeBaselineV1:
    """Fit action and feature statistics using preserved training samples only."""

    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        raise BaselineError("fit_action_magnitude_baseline: invalid dataset")
    if (
        type(minimum_standard_deviation) not in (int, float)
        or not math.isfinite(float(minimum_standard_deviation))
        or float(minimum_standard_deviation) <= 0.0
    ):
        raise BaselineError("fit_action_magnitude_baseline: invalid std floor")
    examples = tuple(
        dataset[index] for index in dataset.indices_for_split(DatasetSplit.TRAIN)
    )
    rows = np.concatenate([_valid_action_rows(item) for item in examples], axis=0)
    action_mean = np.mean(rows, axis=0, dtype=np.float64)
    features = np.stack([_action_features(item, action_mean) for item in examples])
    floor = float(minimum_standard_deviation)
    feature_std = np.maximum(np.std(features, axis=0, ddof=0), floor)
    return ActionMagnitudeBaselineV1(
        dataset_digest=dataset.dataset_digest,
        training_split_digest=dataset.training_split_digest,
        training_sample_count=len(examples),
        valid_action_step_count=rows.shape[0],
        minimum_standard_deviation=floor,
        action_mean=np.asarray(action_mean, dtype=np.float64),
        feature_mean=np.mean(features, axis=0, dtype=np.float64),
        feature_standard_deviation=np.asarray(feature_std, dtype=np.float64),
    )


__all__ = [
    "ACTION_MAGNITUDE_FEATURES",
    "ActionMagnitudeBaselineV1",
    "BaselineError",
    "ConstantFailureBaselineV1",
    "fit_action_magnitude_baseline",
    "fit_constant_baselines",
]
