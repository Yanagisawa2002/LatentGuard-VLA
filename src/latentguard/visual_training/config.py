"""Strict, content-bound M4B configuration models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from latentguard.replay.identity import canonical_json_bytes

M4B_SCHEMA_VERSION = "1.0"
M4B_MODEL_IDS = (
    "random_resnet18_multiview_action",
    "frozen_resnet18_singleview_action",
    "frozen_resnet18_multiview_action",
    "frozen_resnet18_multiview_action_distilled",
)
PRETRAINED_WEIGHT_ENUM = "ResNet18_Weights.IMAGENET1K_V1"
PREPROCESSING_SEMANTIC = "torchvision_resnet18_imagenet1k_v1_rgb224_v1"


class VisualTrainingConfigurationError(ValueError):
    """Raised when an M4B configuration differs from its reviewed contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualTrainingConfigurationError(f"{context}: {reason}")


def _digest_mapping(value: Mapping[str, object], context: str) -> str:
    payload = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _load_json(path: Path, context: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                _fail(context, f"duplicate field {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> NoReturn:
        _fail(context, f"non-finite constant {value!r}")

    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except VisualTrainingConfigurationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VisualTrainingConfigurationError(f"{context}: {exc}") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        _fail(context, "expected a JSON object")
    return cast(Mapping[str, object], raw)


def _exact(value: Mapping[str, object], fields: set[str], context: str) -> None:
    if set(value) != fields:
        _fail(context, "unexpected or missing fields")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected non-empty canonical text")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(context, f"expected integer >= {minimum}")
    return value


def _number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(context, "expected number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        _fail(context, f"expected finite number >= {minimum}")
    return result


def _boolean(value: object, context: str) -> bool:
    if type(value) is not bool:
        _fail(context, "expected boolean")
    return value


@dataclass(frozen=True, slots=True)
class BackboneConfigV1:
    """Reviewed official torchvision ResNet-18 feature contract."""

    family: str
    weight_enum: str
    expected_input_resolution: int
    normalization_mean: tuple[float, float, float]
    normalization_standard_deviation: tuple[float, float, float]
    output_feature_dimension: int
    preprocessing_semantic: str
    schema_version: str = M4B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.family != "torchvision_resnet18":
            _fail("BackboneConfigV1.family", "only torchvision ResNet-18 is allowed")
        if self.weight_enum != PRETRAINED_WEIGHT_ENUM:
            _fail("BackboneConfigV1.weight_enum", "weight identity changed")
        if (
            self.expected_input_resolution != 224
            or self.output_feature_dimension != 512
        ):
            _fail("BackboneConfigV1", "resolution or feature dimension changed")
        if self.normalization_mean != (0.485, 0.456, 0.406) or (
            self.normalization_standard_deviation != (0.229, 0.224, 0.225)
        ):
            _fail("BackboneConfigV1", "ImageNet normalization changed")
        if self.preprocessing_semantic != PREPROCESSING_SEMANTIC:
            _fail("BackboneConfigV1.preprocessing_semantic", "semantic changed")
        if self.schema_version != M4B_SCHEMA_VERSION:
            _fail("BackboneConfigV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return canonical JSON-ready fields."""
        return {
            "expected_input_resolution": self.expected_input_resolution,
            "family": self.family,
            "normalization_mean": list(self.normalization_mean),
            "normalization_standard_deviation": list(
                self.normalization_standard_deviation
            ),
            "output_feature_dimension": self.output_feature_dimension,
            "preprocessing_semantic": self.preprocessing_semantic,
            "schema_version": self.schema_version,
            "weight_enum": self.weight_enum,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent backbone configuration digest."""
        return _digest_mapping(self.as_mapping(), "M4BBackboneConfigV1")


@dataclass(frozen=True, slots=True)
class VisualModelConfigV1:
    """One of the four fixed M4B visual-action architectures."""

    model_id: str
    backbone_frozen: bool
    pretrained: bool
    view_count: int
    distillation_enabled: bool
    action_embedding_dimension: int
    action_attention_heads: int
    action_feedforward_dimension: int
    action_transformer_layers: int
    view_projection_dimension: int
    fusion_hidden_dimension: int
    dropout: float
    schema_version: str = M4B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.model_id not in M4B_MODEL_IDS:
            _fail("VisualModelConfigV1.model_id", "unsupported M4B model")
        expected = {
            M4B_MODEL_IDS[0]: (False, False, 3, False),
            M4B_MODEL_IDS[1]: (True, True, 1, False),
            M4B_MODEL_IDS[2]: (True, True, 3, False),
            M4B_MODEL_IDS[3]: (True, True, 3, True),
        }[self.model_id]
        if (
            self.backbone_frozen,
            self.pretrained,
            self.view_count,
            self.distillation_enabled,
        ) != expected:
            _fail("VisualModelConfigV1", "model ablation contract changed")
        for name in (
            "action_embedding_dimension",
            "action_attention_heads",
            "action_feedforward_dimension",
            "action_transformer_layers",
            "view_projection_dimension",
            "fusion_hidden_dimension",
        ):
            _integer(getattr(self, name), f"VisualModelConfigV1.{name}", minimum=1)
        if self.action_embedding_dimension % self.action_attention_heads != 0:
            _fail("VisualModelConfigV1", "action embedding must divide by heads")
        if not 0.0 <= self.dropout < 1.0:
            _fail("VisualModelConfigV1.dropout", "expected value in [0, 1)")
        if self.schema_version != M4B_SCHEMA_VERSION:
            _fail("VisualModelConfigV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return canonical architecture fields."""
        return {
            "action_attention_heads": self.action_attention_heads,
            "action_embedding_dimension": self.action_embedding_dimension,
            "action_feedforward_dimension": self.action_feedforward_dimension,
            "action_transformer_layers": self.action_transformer_layers,
            "backbone_frozen": self.backbone_frozen,
            "distillation_enabled": self.distillation_enabled,
            "dropout": self.dropout,
            "fusion_hidden_dimension": self.fusion_hidden_dimension,
            "model_id": self.model_id,
            "pretrained": self.pretrained,
            "schema_version": self.schema_version,
            "view_count": self.view_count,
            "view_projection_dimension": self.view_projection_dimension,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete reviewed architecture digest."""
        return _digest_mapping(self.as_mapping(), "M4BVisualModelConfigV1")


@dataclass(frozen=True, slots=True)
class VisualTrainingConfigV1:
    """Fixed optimizer, early-stopping, and distillation settings."""

    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    gradient_clip_norm: float
    hard_label_weight: float
    teacher_weight: float
    distillation_temperature: float
    target_failure_recall: float
    schema_version: str = M4B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_epochs", "early_stopping_patience"):
            _integer(getattr(self, name), f"VisualTrainingConfigV1.{name}", minimum=1)
        for name in (
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "hard_label_weight",
            "teacher_weight",
            "distillation_temperature",
        ):
            _number(getattr(self, name), f"VisualTrainingConfigV1.{name}")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            _fail(
                "VisualTrainingConfigV1", "learning rate and clipping must be positive"
            )
        if self.distillation_temperature <= 0.0:
            _fail("VisualTrainingConfigV1", "temperature must be positive")
        if not 0.0 < self.target_failure_recall <= 1.0:
            _fail("VisualTrainingConfigV1.target_failure_recall", "invalid recall")
        if self.schema_version != M4B_SCHEMA_VERSION:
            _fail("VisualTrainingConfigV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return canonical training fields."""
        return {
            "batch_size": self.batch_size,
            "distillation_temperature": self.distillation_temperature,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "hard_label_weight": self.hard_label_weight,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "schema_version": self.schema_version,
            "target_failure_recall": self.target_failure_recall,
            "teacher_weight": self.teacher_weight,
            "weight_decay": self.weight_decay,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete training configuration digest."""
        return _digest_mapping(self.as_mapping(), "M4BVisualTrainingConfigV1")


@dataclass(frozen=True, slots=True)
class SeedPolicyV1:
    """Validation-only staged seed escalation policy."""

    screen_seeds: tuple[int, ...]
    promoted_seeds: tuple[int, ...]
    maximum_promoted_families: int
    close_gap_threshold: float
    high_standard_deviation_threshold: float
    five_seed_requires_explicit_authorization: bool
    schema_version: str = M4B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.screen_seeds != (0,) or self.promoted_seeds != (0, 1, 2):
            _fail("SeedPolicyV1", "fixed M4B seed sets changed")
        if self.maximum_promoted_families != 3:
            _fail("SeedPolicyV1", "at most three families may be promoted")
        if self.close_gap_threshold != 0.02 or (
            self.high_standard_deviation_threshold != 0.03
        ):
            _fail("SeedPolicyV1", "escalation thresholds changed")
        if self.five_seed_requires_explicit_authorization is not True:
            _fail("SeedPolicyV1", "five-seed authorization gate is required")
        if self.schema_version != M4B_SCHEMA_VERSION:
            _fail("SeedPolicyV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return canonical seed-policy fields."""
        return {
            "close_gap_threshold": self.close_gap_threshold,
            "five_seed_requires_explicit_authorization": (
                self.five_seed_requires_explicit_authorization
            ),
            "high_standard_deviation_threshold": (
                self.high_standard_deviation_threshold
            ),
            "maximum_promoted_families": self.maximum_promoted_families,
            "promoted_seeds": list(self.promoted_seeds),
            "schema_version": self.schema_version,
            "screen_seeds": list(self.screen_seeds),
        }


@dataclass(frozen=True, slots=True)
class ExternalEvaluationConfigV1:
    """Frozen M3C visual-domain selection and bootstrap contract."""

    domains: tuple[str, ...]
    bootstrap_samples: int
    bootstrap_seed: int
    selection_semantic: str
    outcomes_available_during_selection: bool
    schema_version: str = M4B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.domains != (
            "canonical",
            "strong_camera_shift",
            "strong_lighting_shift",
        ):
            _fail("ExternalEvaluationConfigV1.domains", "domain order changed")
        if self.bootstrap_samples != 2000 or self.bootstrap_seed != 271828:
            _fail("ExternalEvaluationConfigV1", "bootstrap contract changed")
        if (
            self.selection_semantic
            != "minimum_three_seed_mean_calibrated_failure_probability_v1"
        ):
            _fail("ExternalEvaluationConfigV1", "selection semantic changed")
        if self.outcomes_available_during_selection is not False:
            _fail("ExternalEvaluationConfigV1", "selection must remain outcome-free")
        if self.schema_version != M4B_SCHEMA_VERSION:
            _fail("ExternalEvaluationConfigV1.schema_version", "unsupported version")

    def as_mapping(self) -> dict[str, object]:
        """Return exact external-evaluation configuration fields."""
        return {
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "domains": list(self.domains),
            "outcomes_available_during_selection": (
                self.outcomes_available_during_selection
            ),
            "schema_version": self.schema_version,
            "selection_semantic": self.selection_semantic,
        }

    @property
    def content_digest(self) -> str:
        """Return the complete external-evaluation configuration digest."""
        return _digest_mapping(self.as_mapping(), "M4BExternalEvaluationConfigV1")


def load_backbone_config(path: Path) -> BackboneConfigV1:
    """Load a strict official ResNet-18 contract."""
    raw = _load_json(path, "BackboneConfigV1")
    fields = {
        "expected_input_resolution",
        "family",
        "normalization_mean",
        "normalization_standard_deviation",
        "output_feature_dimension",
        "preprocessing_semantic",
        "schema_version",
        "weight_enum",
    }
    _exact(raw, fields, "BackboneConfigV1")
    mean = raw["normalization_mean"]
    std = raw["normalization_standard_deviation"]
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 3
        or len(std) != 3
    ):
        _fail("BackboneConfigV1", "normalization requires three components")
    return BackboneConfigV1(
        family=_text(raw["family"], "BackboneConfigV1.family"),
        weight_enum=_text(raw["weight_enum"], "BackboneConfigV1.weight_enum"),
        expected_input_resolution=_integer(
            raw["expected_input_resolution"],
            "BackboneConfigV1.expected_input_resolution",
            minimum=1,
        ),
        normalization_mean=cast(
            tuple[float, float, float],
            tuple(float(_number(x, "BackboneConfigV1.mean")) for x in mean),
        ),
        normalization_standard_deviation=cast(
            tuple[float, float, float],
            tuple(float(_number(x, "BackboneConfigV1.std")) for x in std),
        ),
        output_feature_dimension=_integer(
            raw["output_feature_dimension"],
            "BackboneConfigV1.output_feature_dimension",
            minimum=1,
        ),
        preprocessing_semantic=_text(
            raw["preprocessing_semantic"], "BackboneConfigV1.preprocessing_semantic"
        ),
        schema_version=_text(raw["schema_version"], "BackboneConfigV1.schema_version"),
    )


def load_visual_model_config(path: Path) -> VisualModelConfigV1:
    """Load one exact-field M4B model configuration."""
    raw = _load_json(path, "VisualModelConfigV1")
    fields = {
        "action_attention_heads",
        "action_embedding_dimension",
        "action_feedforward_dimension",
        "action_transformer_layers",
        "backbone_frozen",
        "distillation_enabled",
        "dropout",
        "fusion_hidden_dimension",
        "model_id",
        "pretrained",
        "schema_version",
        "view_count",
        "view_projection_dimension",
    }
    _exact(raw, fields, "VisualModelConfigV1")
    return VisualModelConfigV1(
        model_id=_text(raw["model_id"], "VisualModelConfigV1.model_id"),
        backbone_frozen=_boolean(
            raw["backbone_frozen"], "VisualModelConfigV1.backbone_frozen"
        ),
        pretrained=_boolean(raw["pretrained"], "VisualModelConfigV1.pretrained"),
        view_count=_integer(
            raw["view_count"], "VisualModelConfigV1.view_count", minimum=1
        ),
        distillation_enabled=_boolean(
            raw["distillation_enabled"], "VisualModelConfigV1.distillation_enabled"
        ),
        action_embedding_dimension=_integer(
            raw["action_embedding_dimension"],
            "VisualModelConfigV1.action_embedding_dimension",
            minimum=1,
        ),
        action_attention_heads=_integer(
            raw["action_attention_heads"],
            "VisualModelConfigV1.action_attention_heads",
            minimum=1,
        ),
        action_feedforward_dimension=_integer(
            raw["action_feedforward_dimension"],
            "VisualModelConfigV1.action_feedforward_dimension",
            minimum=1,
        ),
        action_transformer_layers=_integer(
            raw["action_transformer_layers"],
            "VisualModelConfigV1.action_transformer_layers",
            minimum=1,
        ),
        view_projection_dimension=_integer(
            raw["view_projection_dimension"],
            "VisualModelConfigV1.view_projection_dimension",
            minimum=1,
        ),
        fusion_hidden_dimension=_integer(
            raw["fusion_hidden_dimension"],
            "VisualModelConfigV1.fusion_hidden_dimension",
            minimum=1,
        ),
        dropout=_number(raw["dropout"], "VisualModelConfigV1.dropout"),
        schema_version=_text(
            raw["schema_version"], "VisualModelConfigV1.schema_version"
        ),
    )


def load_visual_training_config(path: Path) -> VisualTrainingConfigV1:
    """Load the reviewed fixed optimizer and loss contract."""
    raw = _load_json(path, "VisualTrainingConfigV1")
    fields = {
        "batch_size",
        "distillation_temperature",
        "early_stopping_patience",
        "gradient_clip_norm",
        "hard_label_weight",
        "learning_rate",
        "max_epochs",
        "schema_version",
        "target_failure_recall",
        "teacher_weight",
        "weight_decay",
    }
    _exact(raw, fields, "VisualTrainingConfigV1")
    return VisualTrainingConfigV1(
        batch_size=_integer(
            raw["batch_size"], "VisualTrainingConfigV1.batch_size", minimum=1
        ),
        learning_rate=_number(
            raw["learning_rate"], "VisualTrainingConfigV1.learning_rate"
        ),
        weight_decay=_number(
            raw["weight_decay"], "VisualTrainingConfigV1.weight_decay"
        ),
        max_epochs=_integer(
            raw["max_epochs"], "VisualTrainingConfigV1.max_epochs", minimum=1
        ),
        early_stopping_patience=_integer(
            raw["early_stopping_patience"],
            "VisualTrainingConfigV1.early_stopping_patience",
            minimum=1,
        ),
        gradient_clip_norm=_number(
            raw["gradient_clip_norm"], "VisualTrainingConfigV1.gradient_clip_norm"
        ),
        hard_label_weight=_number(
            raw["hard_label_weight"], "VisualTrainingConfigV1.hard_label_weight"
        ),
        teacher_weight=_number(
            raw["teacher_weight"], "VisualTrainingConfigV1.teacher_weight"
        ),
        distillation_temperature=_number(
            raw["distillation_temperature"],
            "VisualTrainingConfigV1.distillation_temperature",
        ),
        target_failure_recall=_number(
            raw["target_failure_recall"], "VisualTrainingConfigV1.target_failure_recall"
        ),
        schema_version=_text(
            raw["schema_version"], "VisualTrainingConfigV1.schema_version"
        ),
    )


def load_seed_policy(path: Path) -> SeedPolicyV1:
    """Load the fixed staged M4B seed policy."""
    raw = _load_json(path, "SeedPolicyV1")
    fields = {
        "close_gap_threshold",
        "five_seed_requires_explicit_authorization",
        "high_standard_deviation_threshold",
        "maximum_promoted_families",
        "promoted_seeds",
        "schema_version",
        "screen_seeds",
    }
    _exact(raw, fields, "SeedPolicyV1")
    screen = raw["screen_seeds"]
    promoted = raw["promoted_seeds"]
    if not isinstance(screen, list) or not isinstance(promoted, list):
        _fail("SeedPolicyV1", "seed fields must be arrays")
    return SeedPolicyV1(
        screen_seeds=tuple(_integer(x, "SeedPolicyV1.screen_seed") for x in screen),
        promoted_seeds=tuple(
            _integer(x, "SeedPolicyV1.promoted_seed") for x in promoted
        ),
        maximum_promoted_families=_integer(
            raw["maximum_promoted_families"],
            "SeedPolicyV1.maximum_promoted_families",
            minimum=1,
        ),
        close_gap_threshold=_number(
            raw["close_gap_threshold"], "SeedPolicyV1.close_gap_threshold"
        ),
        high_standard_deviation_threshold=_number(
            raw["high_standard_deviation_threshold"],
            "SeedPolicyV1.high_standard_deviation_threshold",
        ),
        five_seed_requires_explicit_authorization=_boolean(
            raw["five_seed_requires_explicit_authorization"],
            "SeedPolicyV1.five_seed_requires_explicit_authorization",
        ),
        schema_version=_text(raw["schema_version"], "SeedPolicyV1.schema_version"),
    )


def load_external_evaluation_config(path: Path) -> ExternalEvaluationConfigV1:
    """Load the fixed outcome-free M3C evaluation contract."""
    raw = _load_json(path, "ExternalEvaluationConfigV1")
    fields = {
        "bootstrap_samples",
        "bootstrap_seed",
        "domains",
        "outcomes_available_during_selection",
        "schema_version",
        "selection_semantic",
    }
    _exact(raw, fields, "ExternalEvaluationConfigV1")
    domains = raw["domains"]
    if not isinstance(domains, list):
        _fail("ExternalEvaluationConfigV1.domains", "expected array")
    return ExternalEvaluationConfigV1(
        domains=tuple(
            _text(item, "ExternalEvaluationConfigV1.domain") for item in domains
        ),
        bootstrap_samples=_integer(
            raw["bootstrap_samples"],
            "ExternalEvaluationConfigV1.bootstrap_samples",
            minimum=1,
        ),
        bootstrap_seed=_integer(
            raw["bootstrap_seed"], "ExternalEvaluationConfigV1.bootstrap_seed"
        ),
        selection_semantic=_text(
            raw["selection_semantic"],
            "ExternalEvaluationConfigV1.selection_semantic",
        ),
        outcomes_available_during_selection=_boolean(
            raw["outcomes_available_during_selection"],
            "ExternalEvaluationConfigV1.outcomes_available_during_selection",
        ),
        schema_version=_text(
            raw["schema_version"], "ExternalEvaluationConfigV1.schema_version"
        ),
    )


__all__ = [
    "M4B_MODEL_IDS",
    "M4B_SCHEMA_VERSION",
    "PREPROCESSING_SEMANTIC",
    "PRETRAINED_WEIGHT_ENUM",
    "BackboneConfigV1",
    "ExternalEvaluationConfigV1",
    "SeedPolicyV1",
    "VisualModelConfigV1",
    "VisualTrainingConfigV1",
    "VisualTrainingConfigurationError",
    "load_backbone_config",
    "load_external_evaluation_config",
    "load_seed_policy",
    "load_visual_model_config",
    "load_visual_training_config",
]
