"""Strict, path-independent configuration contracts for M3B training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, cast

from latentguard.replay.identity import canonical_json_bytes

MODEL_CONFIG_SCHEMA_VERSION = "1.0"
TRAINING_CONFIG_SCHEMA_VERSION = "1.0"
BENCHMARK_CONFIG_SCHEMA_VERSION = "1.0"


class TrainingConfigurationError(ValueError):
    """Raised when an M3B model or training configuration is malformed."""


class ModelType(StrEnum):
    """Fixed learned baseline architectures in the M3B comparison."""

    STATE_ONLY_MLP = "state_only_mlp"
    ACTION_ONLY_MLP = "action_only_mlp"
    STATE_ACTION_MLP = "state_action_mlp"
    TEMPORAL_STATE_ACTION_VERIFIER = "temporal_state_action_verifier"


class ActivationName(StrEnum):
    """Allowlisted hidden-layer activation functions."""

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


class LossMode(StrEnum):
    """Supported binary failure-loss weighting semantics."""

    UNWEIGHTED_BCE = "unweighted_bce"
    CLASS_WEIGHTED_BCE = "class_weighted_bce"


_MODEL_FIELDS = frozenset(
    {
        "schema_version",
        "model_type",
        "state_dimension",
        "action_dimension",
        "action_horizon",
        "hidden_dimensions",
        "state_projection_dimension",
        "action_projection_dimension",
        "temporal_embedding_dimension",
        "transformer_layers",
        "attention_heads",
        "transformer_feedforward_dimension",
        "dropout",
        "activation",
    }
)
_TRAINING_FIELDS = frozenset(
    {
        "schema_version",
        "optimizer",
        "scheduler",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "gradient_clip_norm",
        "max_epochs",
        "max_steps",
        "limit_samples",
        "validation_interval_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "selection_metric",
        "loss_mode",
        "target_failure_recall",
        "minimum_standard_deviation",
        "checkpoint_interval_epochs",
        "deterministic_algorithms",
        "num_workers",
    }
)
_BENCHMARK_FIELDS = frozenset(
    {
        "schema_version",
        "model_types",
        "seeds",
        "bootstrap_seed",
        "bootstrap_replicates",
        "confidence_level",
    }
)


def _digest(value: Mapping[str, object], *, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _positive_integer(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingConfigurationError(f"{context}: expected a positive integer")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise TrainingConfigurationError(f"{context}: expected a non-negative integer")
    return value


def _optional_positive_integer(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, context)


def _finite_number(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise TrainingConfigurationError(f"{context}: expected a finite number")
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)):
        raise TrainingConfigurationError(f"{context}: expected a finite number")
    return float(numeric)


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrainingConfigurationError(
            f"{context}: expected canonical non-empty text"
        )
    return value


def _enum_value(enum_type: type[StrEnum], value: object, context: str) -> StrEnum:
    text = _nonempty_string(value, context)
    try:
        return enum_type(text)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise TrainingConfigurationError(
            f"{context}: unsupported value {text!r}; expected one of {choices}"
        ) from exc


def _optional_dimension(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, context)


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TrainingConfigurationError(f"{context}: expected a JSON array")
    result = tuple(
        _positive_integer(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise TrainingConfigurationError(f"{context}: must not be empty")
    return result


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Serializable architecture definition without runtime-dependent values."""

    model_type: ModelType
    state_dimension: int
    action_dimension: int
    action_horizon: int
    hidden_dimensions: tuple[int, ...]
    state_projection_dimension: int | None
    action_projection_dimension: int | None
    temporal_embedding_dimension: int | None
    transformer_layers: int | None
    attention_heads: int | None
    transformer_feedforward_dimension: int | None
    dropout: float
    activation: ActivationName
    schema_version: str = MODEL_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate dimensions and the fields applicable to one model type."""

        if self.schema_version != MODEL_CONFIG_SCHEMA_VERSION:
            raise TrainingConfigurationError(
                "ModelConfig.schema_version: unsupported version "
                f"{self.schema_version!r}"
            )
        for name in ("state_dimension", "action_dimension", "action_horizon"):
            _positive_integer(getattr(self, name), f"ModelConfig.{name}")
        if not isinstance(self.hidden_dimensions, tuple):
            raise TrainingConfigurationError(
                "ModelConfig.hidden_dimensions: expected an immutable tuple"
            )
        for index, width in enumerate(self.hidden_dimensions):
            _positive_integer(width, f"ModelConfig.hidden_dimensions[{index}]")
        if not 0.0 <= self.dropout < 1.0 or not math.isfinite(self.dropout):
            raise TrainingConfigurationError(
                "ModelConfig.dropout: expected a finite value in [0, 1)"
            )
        if not isinstance(self.activation, ActivationName):
            raise TrainingConfigurationError(
                "ModelConfig.activation: expected an allowlisted activation"
            )
        self._validate_model_specific_fields()

    def _validate_model_specific_fields(self) -> None:
        optional = {
            "state_projection_dimension": self.state_projection_dimension,
            "action_projection_dimension": self.action_projection_dimension,
            "temporal_embedding_dimension": self.temporal_embedding_dimension,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
            "transformer_feedforward_dimension": (
                self.transformer_feedforward_dimension
            ),
        }
        for name, value in optional.items():
            if value is not None:
                _positive_integer(value, f"ModelConfig.{name}")

        if self.model_type is ModelType.STATE_ONLY_MLP:
            required = {"state_projection_dimension"}
            allowed_hidden_counts = {2}
        elif self.model_type is ModelType.ACTION_ONLY_MLP:
            required = {"action_projection_dimension"}
            allowed_hidden_counts = {2, 3}
        elif self.model_type is ModelType.STATE_ACTION_MLP:
            required = {"state_projection_dimension", "action_projection_dimension"}
            allowed_hidden_counts = {2}
        elif self.model_type is ModelType.TEMPORAL_STATE_ACTION_VERIFIER:
            required = {
                "state_projection_dimension",
                "temporal_embedding_dimension",
                "transformer_layers",
                "attention_heads",
                "transformer_feedforward_dimension",
            }
            allowed_hidden_counts = {2}
        else:  # pragma: no cover - the enum type closes this branch
            raise TrainingConfigurationError(
                "ModelConfig.model_type: unsupported value"
            )

        present = {name for name, value in optional.items() if value is not None}
        unexpected = sorted(present - required)
        missing = sorted(required - present)
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise TrainingConfigurationError(
                "ModelConfig: model-specific fields are invalid ("
                + "; ".join(details)
                + ")"
            )
        if len(self.hidden_dimensions) not in allowed_hidden_counts:
            expected = "/".join(str(value) for value in sorted(allowed_hidden_counts))
            raise TrainingConfigurationError(
                "ModelConfig.hidden_dimensions: model requires "
                f"{expected} hidden blocks"
            )
        if self.model_type is ModelType.TEMPORAL_STATE_ACTION_VERIFIER:
            embedding = cast(int, self.temporal_embedding_dimension)
            heads = cast(int, self.attention_heads)
            if embedding % heads != 0:
                raise TrainingConfigurationError(
                    "ModelConfig: temporal embedding dimension must be divisible by "
                    "attention heads"
                )
            if self.activation is ActivationName.SILU:
                raise TrainingConfigurationError(
                    "ModelConfig.activation: temporal encoder supports relu or gelu"
                )

    def as_mapping(self) -> dict[str, object]:
        """Return the complete canonical architecture payload."""

        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type.value,
            "state_dimension": self.state_dimension,
            "action_dimension": self.action_dimension,
            "action_horizon": self.action_horizon,
            "hidden_dimensions": list(self.hidden_dimensions),
            "state_projection_dimension": self.state_projection_dimension,
            "action_projection_dimension": self.action_projection_dimension,
            "temporal_embedding_dimension": self.temporal_embedding_dimension,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
            "transformer_feedforward_dimension": (
                self.transformer_feedforward_dimension
            ),
            "dropout": self.dropout,
            "activation": self.activation.value,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent architecture-configuration digest."""

        return _digest(self.as_mapping(), context="M3BModelConfig")


@dataclass(frozen=True, slots=True)
class ResolvedModelConfig:
    """Model configuration bound to its exact trainable parameter count."""

    architecture: ModelConfig
    parameter_count: int

    def __post_init__(self) -> None:
        _positive_integer(self.parameter_count, "ResolvedModelConfig.parameter_count")

    def as_mapping(self) -> dict[str, object]:
        """Return all architecture fields plus the verified parameter count."""

        payload = self.architecture.as_mapping()
        payload["parameter_count"] = self.parameter_count
        return payload

    @property
    def content_digest(self) -> str:
        """Return the digest of architecture and verified parameter count."""

        return _digest(self.as_mapping(), context="M3BResolvedModelConfig")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Fixed optimizer, validation, and checkpoint-selection recipe."""

    optimizer: str
    scheduler: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_clip_norm: float
    max_epochs: int
    max_steps: int | None
    limit_samples: int | None
    validation_interval_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    selection_metric: str
    loss_mode: LossMode
    target_failure_recall: float
    minimum_standard_deviation: float
    checkpoint_interval_epochs: int
    deterministic_algorithms: bool
    num_workers: int
    schema_version: str = TRAINING_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject unsupported optimization semantics and invalid bounds."""

        if self.schema_version != TRAINING_CONFIG_SCHEMA_VERSION:
            raise TrainingConfigurationError(
                "TrainingConfig.schema_version: unsupported version "
                f"{self.schema_version!r}"
            )
        if self.optimizer != "adamw":
            raise TrainingConfigurationError(
                "TrainingConfig.optimizer: only 'adamw' is supported"
            )
        if self.scheduler != "none":
            raise TrainingConfigurationError(
                "TrainingConfig.scheduler: only 'none' is supported in M3B"
            )
        if self.selection_metric != "validation_corrupted_failure_auprc":
            raise TrainingConfigurationError(
                "TrainingConfig.selection_metric: must be validation-only "
                "corrupted failure AUPRC"
            )
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "minimum_standard_deviation",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise TrainingConfigurationError(
                    f"TrainingConfig.{name}: expected a finite positive value"
                )
        for name in ("weight_decay", "early_stopping_min_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise TrainingConfigurationError(
                    f"TrainingConfig.{name}: expected a finite non-negative value"
                )
        for name in (
            "batch_size",
            "max_epochs",
            "validation_interval_epochs",
            "early_stopping_patience",
            "checkpoint_interval_epochs",
        ):
            _positive_integer(getattr(self, name), f"TrainingConfig.{name}")
        if self.validation_interval_epochs != 1:
            raise TrainingConfigurationError(
                "TrainingConfig.validation_interval_epochs: M3B validates every epoch"
            )
        if self.checkpoint_interval_epochs != 1:
            raise TrainingConfigurationError(
                "TrainingConfig.checkpoint_interval_epochs: exact resume requires "
                "an epoch checkpoint"
            )
        if self.max_steps is not None:
            _positive_integer(self.max_steps, "TrainingConfig.max_steps")
        if self.limit_samples is not None:
            _positive_integer(self.limit_samples, "TrainingConfig.limit_samples")
        _nonnegative_integer(self.num_workers, "TrainingConfig.num_workers")
        if self.num_workers != 0:
            raise TrainingConfigurationError(
                "TrainingConfig.num_workers: M3B uses the deterministic in-process "
                "loader and requires zero workers"
            )
        if type(self.deterministic_algorithms) is not bool:
            raise TrainingConfigurationError(
                "TrainingConfig.deterministic_algorithms: expected a boolean"
            )
        if not isinstance(self.loss_mode, LossMode):
            raise TrainingConfigurationError(
                "TrainingConfig.loss_mode: expected a supported loss mode"
            )
        if not 0.0 < self.target_failure_recall <= 1.0:
            raise TrainingConfigurationError(
                "TrainingConfig.target_failure_recall: expected a value in (0, 1]"
            )

    def as_mapping(self) -> dict[str, object]:
        """Return a complete path-independent training configuration."""

        return {
            "schema_version": self.schema_version,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "gradient_clip_norm": self.gradient_clip_norm,
            "max_epochs": self.max_epochs,
            "max_steps": self.max_steps,
            "limit_samples": self.limit_samples,
            "validation_interval_epochs": self.validation_interval_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "selection_metric": self.selection_metric,
            "loss_mode": self.loss_mode.value,
            "target_failure_recall": self.target_failure_recall,
            "minimum_standard_deviation": self.minimum_standard_deviation,
            "checkpoint_interval_epochs": self.checkpoint_interval_epochs,
            "deterministic_algorithms": self.deterministic_algorithms,
            "num_workers": self.num_workers,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent semantic training digest."""

        return _digest(self.as_mapping(), context="M3BTrainingConfig")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Deterministic model matrix, seed set, and bootstrap protocol."""

    model_types: tuple[ModelType, ...]
    seeds: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float
    schema_version: str = BENCHMARK_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require the fixed four-model, five-seed M3B comparison."""

        if self.schema_version != BENCHMARK_CONFIG_SCHEMA_VERSION:
            raise TrainingConfigurationError(
                "BenchmarkConfig.schema_version: unsupported version "
                f"{self.schema_version!r}"
            )
        expected = tuple(ModelType)
        if self.model_types != expected:
            raise TrainingConfigurationError(
                "BenchmarkConfig.model_types: must list each fixed M3B model once "
                "in canonical order"
            )
        if self.seeds != tuple(sorted(set(self.seeds))) or len(self.seeds) != 5:
            raise TrainingConfigurationError(
                "BenchmarkConfig.seeds: expected five sorted unique seeds"
            )
        for index, seed in enumerate(self.seeds):
            _nonnegative_integer(seed, f"BenchmarkConfig.seeds[{index}]")
        _nonnegative_integer(self.bootstrap_seed, "BenchmarkConfig.bootstrap_seed")
        _positive_integer(
            self.bootstrap_replicates, "BenchmarkConfig.bootstrap_replicates"
        )
        if not 0.0 < self.confidence_level < 1.0:
            raise TrainingConfigurationError(
                "BenchmarkConfig.confidence_level: expected a value in (0, 1)"
            )

    def as_mapping(self) -> dict[str, object]:
        """Return the canonical benchmark payload."""

        return {
            "schema_version": self.schema_version,
            "model_types": [item.value for item in self.model_types],
            "seeds": list(self.seeds),
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "confidence_level": self.confidence_level,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent benchmark-protocol digest."""

        return _digest(self.as_mapping(), context="M3BBenchmarkConfig")


def model_config_from_mapping(value: Mapping[str, object]) -> ModelConfig:
    """Parse one strict model configuration mapping."""

    item = _exact_mapping(value, _MODEL_FIELDS, "ModelConfig")
    schema = _nonempty_string(item["schema_version"], "ModelConfig.schema_version")
    model_type = cast(
        ModelType, _enum_value(ModelType, item["model_type"], "ModelConfig.model_type")
    )
    activation = cast(
        ActivationName,
        _enum_value(ActivationName, item["activation"], "ModelConfig.activation"),
    )
    return ModelConfig(
        model_type=model_type,
        state_dimension=_positive_integer(
            item["state_dimension"], "ModelConfig.state_dimension"
        ),
        action_dimension=_positive_integer(
            item["action_dimension"], "ModelConfig.action_dimension"
        ),
        action_horizon=_positive_integer(
            item["action_horizon"], "ModelConfig.action_horizon"
        ),
        hidden_dimensions=_integer_tuple(
            item["hidden_dimensions"], "ModelConfig.hidden_dimensions"
        ),
        state_projection_dimension=_optional_dimension(
            item["state_projection_dimension"],
            "ModelConfig.state_projection_dimension",
        ),
        action_projection_dimension=_optional_dimension(
            item["action_projection_dimension"],
            "ModelConfig.action_projection_dimension",
        ),
        temporal_embedding_dimension=_optional_dimension(
            item["temporal_embedding_dimension"],
            "ModelConfig.temporal_embedding_dimension",
        ),
        transformer_layers=_optional_dimension(
            item["transformer_layers"], "ModelConfig.transformer_layers"
        ),
        attention_heads=_optional_dimension(
            item["attention_heads"], "ModelConfig.attention_heads"
        ),
        transformer_feedforward_dimension=_optional_dimension(
            item["transformer_feedforward_dimension"],
            "ModelConfig.transformer_feedforward_dimension",
        ),
        dropout=_finite_number(item["dropout"], "ModelConfig.dropout"),
        activation=activation,
        schema_version=schema,
    )


def training_config_from_mapping(value: Mapping[str, object]) -> TrainingConfig:
    """Parse one strict fixed-recipe training configuration mapping."""

    item = _exact_mapping(value, _TRAINING_FIELDS, "TrainingConfig")
    loss_mode = cast(
        LossMode,
        _enum_value(LossMode, item["loss_mode"], "TrainingConfig.loss_mode"),
    )
    max_steps = _optional_positive_integer(
        item["max_steps"], "TrainingConfig.max_steps"
    )
    limit_samples = _optional_positive_integer(
        item["limit_samples"], "TrainingConfig.limit_samples"
    )
    deterministic = item["deterministic_algorithms"]
    if type(deterministic) is not bool:
        raise TrainingConfigurationError(
            "TrainingConfig.deterministic_algorithms: expected a boolean"
        )
    return TrainingConfig(
        optimizer=_nonempty_string(item["optimizer"], "TrainingConfig.optimizer"),
        scheduler=_nonempty_string(item["scheduler"], "TrainingConfig.scheduler"),
        learning_rate=_finite_number(
            item["learning_rate"], "TrainingConfig.learning_rate"
        ),
        weight_decay=_finite_number(
            item["weight_decay"], "TrainingConfig.weight_decay"
        ),
        batch_size=_positive_integer(item["batch_size"], "TrainingConfig.batch_size"),
        gradient_clip_norm=_finite_number(
            item["gradient_clip_norm"], "TrainingConfig.gradient_clip_norm"
        ),
        max_epochs=_positive_integer(item["max_epochs"], "TrainingConfig.max_epochs"),
        max_steps=max_steps,
        limit_samples=limit_samples,
        validation_interval_epochs=_positive_integer(
            item["validation_interval_epochs"],
            "TrainingConfig.validation_interval_epochs",
        ),
        early_stopping_patience=_positive_integer(
            item["early_stopping_patience"],
            "TrainingConfig.early_stopping_patience",
        ),
        early_stopping_min_delta=_finite_number(
            item["early_stopping_min_delta"],
            "TrainingConfig.early_stopping_min_delta",
        ),
        selection_metric=_nonempty_string(
            item["selection_metric"], "TrainingConfig.selection_metric"
        ),
        loss_mode=loss_mode,
        target_failure_recall=_finite_number(
            item["target_failure_recall"], "TrainingConfig.target_failure_recall"
        ),
        minimum_standard_deviation=_finite_number(
            item["minimum_standard_deviation"],
            "TrainingConfig.minimum_standard_deviation",
        ),
        checkpoint_interval_epochs=_positive_integer(
            item["checkpoint_interval_epochs"],
            "TrainingConfig.checkpoint_interval_epochs",
        ),
        deterministic_algorithms=deterministic,
        num_workers=_nonnegative_integer(
            item["num_workers"], "TrainingConfig.num_workers"
        ),
        schema_version=_nonempty_string(
            item["schema_version"], "TrainingConfig.schema_version"
        ),
    )


def benchmark_config_from_mapping(value: Mapping[str, object]) -> BenchmarkConfig:
    """Parse the strict fixed M3B benchmark protocol."""

    item = _exact_mapping(value, _BENCHMARK_FIELDS, "BenchmarkConfig")
    raw_models = item["model_types"]
    if not isinstance(raw_models, list):
        raise TrainingConfigurationError(
            "BenchmarkConfig.model_types: expected a JSON array"
        )
    model_types = tuple(
        cast(
            ModelType,
            _enum_value(ModelType, model, f"BenchmarkConfig.model_types[{index}]"),
        )
        for index, model in enumerate(raw_models)
    )
    raw_seeds = item["seeds"]
    if not isinstance(raw_seeds, list):
        raise TrainingConfigurationError("BenchmarkConfig.seeds: expected a JSON array")
    seeds = tuple(
        _nonnegative_integer(seed, f"BenchmarkConfig.seeds[{index}]")
        for index, seed in enumerate(raw_seeds)
    )
    return BenchmarkConfig(
        model_types=model_types,
        seeds=seeds,
        bootstrap_seed=_nonnegative_integer(
            item["bootstrap_seed"], "BenchmarkConfig.bootstrap_seed"
        ),
        bootstrap_replicates=_positive_integer(
            item["bootstrap_replicates"], "BenchmarkConfig.bootstrap_replicates"
        ),
        confidence_level=_finite_number(
            item["confidence_level"], "BenchmarkConfig.confidence_level"
        ),
        schema_version=_nonempty_string(
            item["schema_version"], "BenchmarkConfig.schema_version"
        ),
    )


def load_model_config(path: Path) -> ModelConfig:
    """Load a duplicate-free, strict model JSON file."""

    return model_config_from_mapping(
        _mapping(_load_strict_json(path, "ModelConfig"), "ModelConfig")
    )


def load_training_config(path: Path) -> TrainingConfig:
    """Load a duplicate-free, strict training JSON file."""

    return training_config_from_mapping(
        _mapping(_load_strict_json(path, "TrainingConfig"), "TrainingConfig")
    )


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load a duplicate-free, strict benchmark JSON file."""

    return benchmark_config_from_mapping(
        _mapping(_load_strict_json(path, "BenchmarkConfig"), "BenchmarkConfig")
    )


def _load_strict_json(path: Path, context: str) -> object:
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise TrainingConfigurationError(
                f"{context}: missing or unsafe regular configuration file"
            )
        if source.stat().st_nlink != 1:
            raise TrainingConfigurationError(
                f"{context}: hard-linked configuration files are unsupported"
            )
        return cast(
            object,
            json.loads(
                source.read_text(encoding="utf-8"),
                parse_constant=lambda value: _reject_json_constant(value, context),
                object_pairs_hook=lambda pairs: _reject_duplicate_fields(
                    pairs, context
                ),
            ),
        )
    except TrainingConfigurationError:
        raise
    except Exception as exc:
        raise TrainingConfigurationError(
            f"{context}: could not read safely: {exc}"
        ) from exc


def _reject_json_constant(value: str, context: str) -> NoReturn:
    raise TrainingConfigurationError(
        f"{context}: non-finite JSON constant {value!r} is unsupported"
    )


def _reject_duplicate_fields(
    pairs: Sequence[tuple[str, object]], context: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingConfigurationError(f"{context}: duplicate JSON field {key!r}")
        result[key] = value
    return result


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TrainingConfigurationError(f"{context}: expected a JSON object")
    return cast(dict[str, object], value)


def _exact_mapping(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> Mapping[str, object]:
    item = _mapping(value, context)
    missing = sorted(expected - set(item))
    unexpected = sorted(set(item) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise TrainingConfigurationError(
            f"{context}: invalid fields (" + "; ".join(details) + ")"
        )
    return item


__all__ = [
    "BENCHMARK_CONFIG_SCHEMA_VERSION",
    "MODEL_CONFIG_SCHEMA_VERSION",
    "TRAINING_CONFIG_SCHEMA_VERSION",
    "ActivationName",
    "BenchmarkConfig",
    "LossMode",
    "ModelConfig",
    "ModelType",
    "ResolvedModelConfig",
    "TrainingConfig",
    "TrainingConfigurationError",
    "benchmark_config_from_mapping",
    "load_benchmark_config",
    "load_model_config",
    "load_training_config",
    "model_config_from_mapping",
    "training_config_from_mapping",
]
