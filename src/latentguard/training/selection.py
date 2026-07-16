"""Validation-only five-seed aggregation and architecture selection."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import NoReturn

import numpy as np

from latentguard.replay.identity import canonical_json_bytes

SELECTION_SCHEMA_VERSION = "1.0"
DEFAULT_MODEL_TYPES = (
    "state_only_mlp",
    "action_only_mlp",
    "state_action_mlp",
    "temporal_state_action_verifier",
)
DEFAULT_FIVE_SEEDS = (0, 1, 2, 3, 4)
SELECTION_PRIMARY_METRIC = "validation_corrupted_only_failure_auprc_mean"
SELECTION_TIE_BREAKERS = (
    "validation_corrupted_only_brier_score_mean",
    "validation_pairwise_concordance_mean",
    "lower_parameter_count",
)
_T_975_DF4 = 2.7764451051977987


def _fail(context: str, reason: str) -> NoReturn:
    raise ValueError(f"{context}: {reason}")


def _digest(value: object, *, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _require_digest(value: str, context: str) -> None:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")


def _canonical_text(value: str, context: str) -> None:
    if not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")


@dataclass(frozen=True, slots=True)
class SeedValidationResult:
    """Selection inputs from one seed, restricted to validation metrics."""

    model_type: str
    seed: int
    failure_auprc: float
    brier_score: float
    pairwise_concordance: float
    parameter_count: int
    model_config_digest: str
    training_config_digest: str
    checkpoint_identity: str
    checkpoint_content_digest: str
    checkpoint_kind: str
    checkpoint_epoch: int
    validation_prediction_digest: str
    split: str = "validation"

    def __post_init__(self) -> None:
        """Reject test metrics, non-finite values, and unbound identities."""

        _canonical_text(self.model_type, "model_type")
        if self.split != "validation":
            _fail("split", "model selection accepts validation results only")
        if type(self.seed) is not int or self.seed < 0:
            _fail("seed", "expected a non-negative integer")
        for context, value in (
            ("failure_auprc", self.failure_auprc),
            ("brier_score", self.brier_score),
            ("pairwise_concordance", self.pairwise_concordance),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                _fail(context, "expected a finite normalized metric")
        if type(self.parameter_count) is not int or self.parameter_count <= 0:
            _fail("parameter_count", "expected a positive integer")
        _require_digest(self.model_config_digest, "model_config_digest")
        _require_digest(self.training_config_digest, "training_config_digest")
        _canonical_text(self.checkpoint_identity, "checkpoint_identity")
        _require_digest(self.checkpoint_content_digest, "checkpoint_content_digest")
        if self.checkpoint_kind != "best":
            _fail("checkpoint_kind", "selection accepts best checkpoints only")
        if type(self.checkpoint_epoch) is not int or self.checkpoint_epoch < 0:
            _fail("checkpoint_epoch", "expected a non-negative integer")
        _require_digest(
            self.validation_prediction_digest,
            "validation_prediction_digest",
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native seed result."""

        return {
            "brier_score": self.brier_score,
            "checkpoint_content_digest": self.checkpoint_content_digest,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_kind": self.checkpoint_kind,
            "failure_auprc": self.failure_auprc,
            "model_config_digest": self.model_config_digest,
            "model_type": self.model_type,
            "pairwise_concordance": self.pairwise_concordance,
            "parameter_count": self.parameter_count,
            "seed": self.seed,
            "split": self.split,
            "training_config_digest": self.training_config_digest,
            "validation_prediction_digest": self.validation_prediction_digest,
        }


@dataclass(frozen=True, slots=True)
class AggregateStatistic:
    """Five-seed descriptive statistics and a documented t interval."""

    count: int
    mean: float
    standard_deviation: float
    median: float
    minimum: float
    maximum: float
    confidence_lower: float
    confidence_upper: float
    confidence_method: str

    def __post_init__(self) -> None:
        """Require finite internally ordered summary values."""

        values = (
            self.mean,
            self.standard_deviation,
            self.median,
            self.minimum,
            self.maximum,
            self.confidence_lower,
            self.confidence_upper,
        )
        if type(self.count) is not int or self.count != 5:
            _fail("count", "M3B architecture aggregation requires exactly five seeds")
        if any(not math.isfinite(value) for value in values):
            _fail("AggregateStatistic", "all values must be finite")
        if self.standard_deviation < 0.0 or self.minimum > self.maximum:
            _fail("AggregateStatistic", "invalid range or deviation")
        if not self.minimum <= self.median <= self.maximum:
            _fail("median", "must lie inside the observed range")
        if self.confidence_lower > self.confidence_upper:
            _fail("confidence", "lower bound exceeds upper bound")
        if self.confidence_method != "student_t_95_df4_v1":
            _fail("confidence_method", "unsupported five-seed interval method")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native aggregate."""

        return {
            "confidence_lower": self.confidence_lower,
            "confidence_method": self.confidence_method,
            "confidence_upper": self.confidence_upper,
            "count": self.count,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "standard_deviation": self.standard_deviation,
        }


def aggregate_five_seed_values(values: Sequence[float]) -> AggregateStatistic:
    """Aggregate exactly five finite values with a Student-t 95% interval."""

    if len(values) != 5 or any(not math.isfinite(value) for value in values):
        _fail("values", "expected exactly five finite values")
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1))
    half_width = _T_975_DF4 * standard_deviation / math.sqrt(5.0)
    return AggregateStatistic(
        count=5,
        mean=mean,
        standard_deviation=standard_deviation,
        median=float(median(values)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        confidence_lower=mean - half_width,
        confidence_upper=mean + half_width,
        confidence_method="student_t_95_df4_v1",
    )


@dataclass(frozen=True, slots=True)
class ArchitectureValidationAggregate:
    """Five-seed validation aggregate for one fixed architecture/configuration."""

    model_type: str
    seeds: tuple[int, ...]
    parameter_count: int
    model_config_digest: str
    training_config_digest: str
    failure_auprc: AggregateStatistic
    brier_score: AggregateStatistic
    pairwise_concordance: AggregateStatistic
    seed_results: tuple[SeedValidationResult, ...]

    def __post_init__(self) -> None:
        """Recompute every aggregate and reject cross-seed identity drift."""

        _canonical_text(self.model_type, "model_type")
        if self.seeds != tuple(sorted(set(self.seeds))) or len(self.seeds) != 5:
            _fail("seeds", "expected five sorted unique seeds")
        if tuple(item.seed for item in self.seed_results) != self.seeds:
            _fail("seed_results", "ordered seed inventory mismatch")
        if any(item.model_type != self.model_type for item in self.seed_results):
            _fail("seed_results", "model type changed across seeds")
        if any(
            item.parameter_count != self.parameter_count
            or item.model_config_digest != self.model_config_digest
            or item.training_config_digest != self.training_config_digest
            for item in self.seed_results
        ):
            _fail("seed_results", "configuration identity changed across seeds")
        if self.failure_auprc != aggregate_five_seed_values(
            [item.failure_auprc for item in self.seed_results]
        ):
            _fail("failure_auprc", "aggregate does not match seed results")
        if self.brier_score != aggregate_five_seed_values(
            [item.brier_score for item in self.seed_results]
        ):
            _fail("brier_score", "aggregate does not match seed results")
        if self.pairwise_concordance != aggregate_five_seed_values(
            [item.pairwise_concordance for item in self.seed_results]
        ):
            _fail("pairwise_concordance", "aggregate does not match seed results")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-native architecture aggregate."""

        return {
            "brier_score": self.brier_score.to_dict(),
            "failure_auprc": self.failure_auprc.to_dict(),
            "model_config_digest": self.model_config_digest,
            "model_type": self.model_type,
            "pairwise_concordance": self.pairwise_concordance.to_dict(),
            "parameter_count": self.parameter_count,
            "seed_results": [item.to_dict() for item in self.seed_results],
            "seeds": list(self.seeds),
            "training_config_digest": self.training_config_digest,
        }


def _selection_order_key(
    candidate: ArchitectureValidationAggregate,
) -> tuple[float, float, float, int, str]:
    """Return the fixed M3B validation-only architecture ordering key."""

    return (
        -candidate.failure_auprc.mean,
        candidate.brier_score.mean,
        -candidate.pairwise_concordance.mean,
        candidate.parameter_count,
        candidate.model_type,
    )


@dataclass(frozen=True, slots=True)
class SelectionRecordV1:
    """Immutable validation-only result of the fixed M3B selection protocol."""

    dataset_digest: str
    split_digest: str
    expected_seeds: tuple[int, ...]
    selected_model_type: str
    selected_model_config_digest: str
    selected_training_config_digest: str
    candidates: tuple[ArchitectureValidationAggregate, ...]
    primary_metric: str = SELECTION_PRIMARY_METRIC
    tie_breakers: tuple[str, ...] = SELECTION_TIE_BREAKERS
    schema_version: str = SELECTION_SCHEMA_VERSION
    frozen: bool = True

    def __post_init__(self) -> None:
        """Validate selected identity and frozen selection semantics."""

        _require_digest(self.dataset_digest, "dataset_digest")
        _require_digest(self.split_digest, "split_digest")
        _require_digest(
            self.selected_model_config_digest, "selected_model_config_digest"
        )
        _require_digest(
            self.selected_training_config_digest, "selected_training_config_digest"
        )
        if self.expected_seeds != tuple(sorted(set(self.expected_seeds))):
            _fail("expected_seeds", "must be sorted and unique")
        if len(self.expected_seeds) != 5:
            _fail("expected_seeds", "M3B requires exactly five seeds")
        if not self.frozen:
            _fail("frozen", "selection records must be frozen before test evaluation")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("schema_version", "unsupported selection schema")
        if self.primary_metric != SELECTION_PRIMARY_METRIC:
            _fail("primary_metric", "selection metric semantic changed")
        if self.tie_breakers != SELECTION_TIE_BREAKERS:
            _fail("tie_breakers", "selection tie-breaker semantics changed")
        if (
            not isinstance(self.candidates, tuple)
            or any(
                not isinstance(candidate, ArchitectureValidationAggregate)
                for candidate in self.candidates
            )
            or tuple(candidate.model_type for candidate in self.candidates)
            != DEFAULT_MODEL_TYPES
        ):
            _fail(
                "candidates",
                "must match the canonical four-architecture inventory and order",
            )
        if any(candidate.seeds != self.expected_seeds for candidate in self.candidates):
            _fail("candidates", "all candidates require the fixed seed inventory")
        if (
            len({candidate.training_config_digest for candidate in self.candidates})
            != 1
        ):
            _fail("candidates", "training protocol changed across architectures")
        selected = min(self.candidates, key=_selection_order_key)
        if self.selected_model_type != selected.model_type:
            _fail(
                "selected_model_type",
                "does not match the canonical validation aggregate winner",
            )
        if (
            selected.model_config_digest != self.selected_model_config_digest
            or selected.training_config_digest != self.selected_training_config_digest
        ):
            _fail("selected_model_type", "selected configuration binding mismatch")

    @property
    def content_digest(self) -> str:
        """Return the path-independent frozen-selection digest."""

        return _digest(self._payload(), context="SelectionRecordV1")

    def _payload(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "dataset_digest": self.dataset_digest,
            "expected_seeds": list(self.expected_seeds),
            "frozen": self.frozen,
            "primary_metric": self.primary_metric,
            "schema_version": self.schema_version,
            "selected_model_config_digest": self.selected_model_config_digest,
            "selected_model_type": self.selected_model_type,
            "selected_training_config_digest": (self.selected_training_config_digest),
            "split_digest": self.split_digest,
            "tie_breakers": list(self.tie_breakers),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a strict self-digesting JSON representation."""

        return {**self._payload(), "content_digest": self.content_digest}


def _aggregate_architecture(
    model_type: str,
    results: Sequence[SeedValidationResult],
    expected_seeds: tuple[int, ...],
) -> ArchitectureValidationAggregate:
    ordered = tuple(sorted(results, key=lambda item: item.seed))
    seeds = tuple(item.seed for item in ordered)
    if seeds != expected_seeds:
        _fail(model_type, "seed inventory does not match the fixed benchmark")
    if len({item.model_config_digest for item in ordered}) != 1:
        _fail(model_type, "model configuration changed across seeds")
    if len({item.training_config_digest for item in ordered}) != 1:
        _fail(model_type, "training configuration changed across seeds")
    if len({item.parameter_count for item in ordered}) != 1:
        _fail(model_type, "parameter count changed across seeds")
    return ArchitectureValidationAggregate(
        model_type=model_type,
        seeds=seeds,
        parameter_count=ordered[0].parameter_count,
        model_config_digest=ordered[0].model_config_digest,
        training_config_digest=ordered[0].training_config_digest,
        failure_auprc=aggregate_five_seed_values(
            [item.failure_auprc for item in ordered]
        ),
        brier_score=aggregate_five_seed_values([item.brier_score for item in ordered]),
        pairwise_concordance=aggregate_five_seed_values(
            [item.pairwise_concordance for item in ordered]
        ),
        seed_results=ordered,
    )


def select_model_architecture(
    results: Sequence[SeedValidationResult],
    *,
    dataset_digest: str,
    split_digest: str,
    expected_model_types: Sequence[str] = DEFAULT_MODEL_TYPES,
    expected_seeds: Sequence[int] = DEFAULT_FIVE_SEEDS,
) -> SelectionRecordV1:
    """Select from aggregated validation results without accepting test input."""

    _require_digest(dataset_digest, "dataset_digest")
    _require_digest(split_digest, "split_digest")
    models = tuple(expected_model_types)
    seeds = tuple(expected_seeds)
    if models != tuple(dict.fromkeys(models)) or not models:
        _fail("expected_model_types", "must be non-empty and unique")
    if seeds != tuple(sorted(set(seeds))) or len(seeds) != 5:
        _fail("expected_seeds", "must contain five sorted unique seeds")
    grouped: defaultdict[str, list[SeedValidationResult]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    for result in results:
        identity = (result.model_type, result.seed)
        if identity in identities:
            _fail("results", "duplicate model/seed result")
        identities.add(identity)
        grouped[result.model_type].append(result)
    if set(grouped) != set(models):
        _fail("results", "model inventory does not match the fixed benchmark")
    aggregates = tuple(
        _aggregate_architecture(model, grouped[model], seeds) for model in models
    )
    training_digests = {item.training_config_digest for item in aggregates}
    if len(training_digests) != 1:
        _fail("results", "all architectures must use the same training protocol")
    selected = min(aggregates, key=_selection_order_key)
    return SelectionRecordV1(
        dataset_digest=dataset_digest,
        split_digest=split_digest,
        expected_seeds=seeds,
        selected_model_type=selected.model_type,
        selected_model_config_digest=selected.model_config_digest,
        selected_training_config_digest=selected.training_config_digest,
        candidates=aggregates,
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected an object with text keys")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str):
        _fail(context, "expected text")
    _canonical_text(value, context)
    return value


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        _fail(context, "expected an integer")
    if value < (1 if positive else 0):
        _fail(context, "integer is outside the allowed range")
    return value


def _number(value: object, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(context, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(context, "expected a finite number")
    return result


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        _fail(context, "expected an array")
    return value


def _seed_result_from_dict(value: object, context: str) -> SeedValidationResult:
    item = _mapping(value, context)
    expected = {
        "brier_score",
        "checkpoint_content_digest",
        "checkpoint_epoch",
        "checkpoint_identity",
        "checkpoint_kind",
        "failure_auprc",
        "model_config_digest",
        "model_type",
        "pairwise_concordance",
        "parameter_count",
        "seed",
        "split",
        "training_config_digest",
        "validation_prediction_digest",
    }
    if set(item) != expected:
        _fail(context, "unexpected or missing fields")
    return SeedValidationResult(
        model_type=_text(item["model_type"], f"{context}.model_type"),
        seed=_integer(item["seed"], f"{context}.seed"),
        failure_auprc=_number(item["failure_auprc"], f"{context}.failure_auprc"),
        brier_score=_number(item["brier_score"], f"{context}.brier_score"),
        pairwise_concordance=_number(
            item["pairwise_concordance"], f"{context}.pairwise_concordance"
        ),
        parameter_count=_integer(
            item["parameter_count"], f"{context}.parameter_count", positive=True
        ),
        model_config_digest=_text(
            item["model_config_digest"], f"{context}.model_config_digest"
        ),
        training_config_digest=_text(
            item["training_config_digest"], f"{context}.training_config_digest"
        ),
        checkpoint_identity=_text(
            item["checkpoint_identity"], f"{context}.checkpoint_identity"
        ),
        checkpoint_content_digest=_text(
            item["checkpoint_content_digest"],
            f"{context}.checkpoint_content_digest",
        ),
        checkpoint_kind=_text(item["checkpoint_kind"], f"{context}.checkpoint_kind"),
        checkpoint_epoch=_integer(
            item["checkpoint_epoch"], f"{context}.checkpoint_epoch"
        ),
        validation_prediction_digest=_text(
            item["validation_prediction_digest"],
            f"{context}.validation_prediction_digest",
        ),
        split=_text(item["split"], f"{context}.split"),
    )


def _aggregate_statistic_from_dict(value: object, context: str) -> AggregateStatistic:
    item = _mapping(value, context)
    expected = {
        "confidence_lower",
        "confidence_method",
        "confidence_upper",
        "count",
        "maximum",
        "mean",
        "median",
        "minimum",
        "standard_deviation",
    }
    if set(item) != expected:
        _fail(context, "unexpected or missing fields")
    return AggregateStatistic(
        count=_integer(item["count"], f"{context}.count", positive=True),
        mean=_number(item["mean"], f"{context}.mean"),
        standard_deviation=_number(
            item["standard_deviation"], f"{context}.standard_deviation"
        ),
        median=_number(item["median"], f"{context}.median"),
        minimum=_number(item["minimum"], f"{context}.minimum"),
        maximum=_number(item["maximum"], f"{context}.maximum"),
        confidence_lower=_number(
            item["confidence_lower"], f"{context}.confidence_lower"
        ),
        confidence_upper=_number(
            item["confidence_upper"], f"{context}.confidence_upper"
        ),
        confidence_method=_text(
            item["confidence_method"], f"{context}.confidence_method"
        ),
    )


def _architecture_from_dict(
    value: object, context: str
) -> ArchitectureValidationAggregate:
    item = _mapping(value, context)
    expected = {
        "brier_score",
        "failure_auprc",
        "model_config_digest",
        "model_type",
        "pairwise_concordance",
        "parameter_count",
        "seed_results",
        "seeds",
        "training_config_digest",
    }
    if set(item) != expected:
        _fail(context, "unexpected or missing fields")
    raw_seeds = _list(item["seeds"], f"{context}.seeds")
    raw_results = _list(item["seed_results"], f"{context}.seed_results")
    return ArchitectureValidationAggregate(
        model_type=_text(item["model_type"], f"{context}.model_type"),
        seeds=tuple(
            _integer(seed, f"{context}.seeds[{index}]")
            for index, seed in enumerate(raw_seeds)
        ),
        parameter_count=_integer(
            item["parameter_count"], f"{context}.parameter_count", positive=True
        ),
        model_config_digest=_text(
            item["model_config_digest"], f"{context}.model_config_digest"
        ),
        training_config_digest=_text(
            item["training_config_digest"], f"{context}.training_config_digest"
        ),
        failure_auprc=_aggregate_statistic_from_dict(
            item["failure_auprc"], f"{context}.failure_auprc"
        ),
        brier_score=_aggregate_statistic_from_dict(
            item["brier_score"], f"{context}.brier_score"
        ),
        pairwise_concordance=_aggregate_statistic_from_dict(
            item["pairwise_concordance"], f"{context}.pairwise_concordance"
        ),
        seed_results=tuple(
            _seed_result_from_dict(result, f"{context}.seed_results[{index}]")
            for index, result in enumerate(raw_results)
        ),
    )


def selection_record_from_dict(value: Mapping[str, object]) -> SelectionRecordV1:
    """Strictly load a frozen selection record and reject aggregate tampering."""

    expected = {
        "candidates",
        "content_digest",
        "dataset_digest",
        "expected_seeds",
        "frozen",
        "primary_metric",
        "schema_version",
        "selected_model_config_digest",
        "selected_model_type",
        "selected_training_config_digest",
        "split_digest",
        "tie_breakers",
    }
    if set(value) != expected:
        _fail("SelectionRecordV1", "unexpected or missing fields")
    raw_candidates = _list(value["candidates"], "candidates")
    raw_seeds = _list(value["expected_seeds"], "expected_seeds")
    raw_ties = _list(value["tie_breakers"], "tie_breakers")
    raw_frozen = value["frozen"]
    if type(raw_frozen) is not bool:
        _fail("frozen", "expected a boolean")
    result = SelectionRecordV1(
        dataset_digest=_text(value["dataset_digest"], "dataset_digest"),
        split_digest=_text(value["split_digest"], "split_digest"),
        expected_seeds=tuple(
            _integer(seed, f"expected_seeds[{index}]")
            for index, seed in enumerate(raw_seeds)
        ),
        selected_model_type=_text(value["selected_model_type"], "selected_model_type"),
        selected_model_config_digest=_text(
            value["selected_model_config_digest"],
            "selected_model_config_digest",
        ),
        selected_training_config_digest=_text(
            value["selected_training_config_digest"],
            "selected_training_config_digest",
        ),
        candidates=tuple(
            _architecture_from_dict(item, f"candidates[{index}]")
            for index, item in enumerate(raw_candidates)
        ),
        primary_metric=_text(value["primary_metric"], "primary_metric"),
        tie_breakers=tuple(
            _text(item, f"tie_breakers[{index}]") for index, item in enumerate(raw_ties)
        ),
        schema_version=_text(value["schema_version"], "schema_version"),
        frozen=raw_frozen,
    )
    expected_digest = _text(value["content_digest"], "content_digest")
    _require_digest(expected_digest, "content_digest")
    if result.content_digest != expected_digest:
        _fail("content_digest", "selection content changed")
    return result


__all__ = [
    "AggregateStatistic",
    "ArchitectureValidationAggregate",
    "DEFAULT_FIVE_SEEDS",
    "DEFAULT_MODEL_TYPES",
    "SELECTION_PRIMARY_METRIC",
    "SELECTION_SCHEMA_VERSION",
    "SELECTION_TIE_BREAKERS",
    "SeedValidationResult",
    "SelectionRecordV1",
    "aggregate_five_seed_values",
    "selection_record_from_dict",
    "select_model_architecture",
]
