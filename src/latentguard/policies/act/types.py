"""Dependency-free configuration and run identities for native PickCube ACT."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Self, cast

from latentguard.replay.identity import canonical_json_bytes

PICKCUBE_ACT_IMAGE_FEATURE = "observation.images.front_oblique"
PICKCUBE_ACT_STATE_FEATURE = "observation.state"
PICKCUBE_ACT_ACTION_FEATURE = "action"
PICKCUBE_ACT_LEROBOT_VERSION = "0.6.0"
PICKCUBE_ACT_BOUNDED_EXPERIMENT_SCHEMA = "pickcube-native-act-bounded-experiment-v1"
PICKCUBE_ACT_BOUNDED_CHECKPOINT_SCHEMA = "pickcube_act_bounded_v1"
PICKCUBE_ACT_GRASP_EXPERIMENT_SCHEMA = "pickcube-native-act-grasp-experiment-v1"


class PickCubeActConfigurationError(ValueError):
    """Raised when a training or model setting differs from the P0 contract."""


def _positive(value: int, name: str, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise PickCubeActConfigurationError(f"{name} must be a valid integer")


def _finite(value: float, name: str, *, positive: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PickCubeActConfigurationError(f"{name} must be finite")
    if positive and float(value) <= 0.0:
        raise PickCubeActConfigurationError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class PickCubeActModelConfig:
    """Standard single-view ACT architecture and action-chunk semantics."""

    image_feature_key: str = PICKCUBE_ACT_IMAGE_FEATURE
    state_feature_key: str = PICKCUBE_ACT_STATE_FEATURE
    action_feature_key: str = PICKCUBE_ACT_ACTION_FEATURE
    image_shape_chw: tuple[int, int, int] = (3, 224, 224)
    state_dimension: int = 18
    action_dimension: int = 8
    chunk_size: int = 16
    n_action_steps: int = 4
    n_obs_steps: int = 1
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    dropout: float = 0.1
    kl_weight: float = 10.0
    normalization_mode: str = "MEAN_STD"
    schema_version: str = "pickcube-native-act-model-v1"

    def __post_init__(self) -> None:
        expected_features = (
            PICKCUBE_ACT_IMAGE_FEATURE,
            PICKCUBE_ACT_STATE_FEATURE,
            PICKCUBE_ACT_ACTION_FEATURE,
        )
        if (
            self.image_feature_key,
            self.state_feature_key,
            self.action_feature_key,
        ) != expected_features:
            raise PickCubeActConfigurationError("ACT feature allowlist changed")
        if self.image_shape_chw != (3, 224, 224):
            raise PickCubeActConfigurationError("ACT image shape must be 3x224x224")
        if self.state_dimension != 18 or self.action_dimension != 8:
            raise PickCubeActConfigurationError("ACT state/action dimensions changed")
        for name in (
            "chunk_size",
            "n_action_steps",
            "n_obs_steps",
            "dim_model",
            "n_heads",
            "dim_feedforward",
            "n_encoder_layers",
            "n_decoder_layers",
            "latent_dim",
            "n_vae_encoder_layers",
        ):
            _positive(cast(int, getattr(self, name)), name)
        if self.n_action_steps > self.chunk_size or self.n_obs_steps != 1:
            raise PickCubeActConfigurationError("ACT temporal contract is invalid")
        if self.vision_backbone != "resnet18":
            raise PickCubeActConfigurationError("ACT backbone must be resnet18")
        if self.pretrained_backbone_weights is not None:
            raise PickCubeActConfigurationError("network downloads are prohibited")
        if self.normalization_mode != "MEAN_STD":
            raise PickCubeActConfigurationError("ACT normalization must be MEAN_STD")
        _finite(self.dropout, "dropout")
        _finite(self.kl_weight, "kl_weight", positive=True)
        if not 0.0 <= self.dropout < 1.0:
            raise PickCubeActConfigurationError("dropout must lie in [0,1)")
        if self.schema_version != "pickcube-native-act-model-v1":
            raise PickCubeActConfigurationError("model schema changed")

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-native complete architecture description."""
        value = asdict(self)
        value["image_shape_chw"] = list(self.image_shape_chw)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Decode one strict model configuration."""
        payload = dict(value)
        shape = payload.get("image_shape_chw")
        if isinstance(shape, list):
            payload["image_shape_chw"] = tuple(shape)
        return cls(**cast(Any, payload))


@dataclass(frozen=True, slots=True)
class PickCubeActOptimizationConfig:
    """Fixed optimizer, validation, checkpoint, and early-stop settings."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 10.0
    mixed_precision: str = "bfloat16"
    batch_size: int = 32
    dataloader_workers: int = 4
    training_steps: int = 20000
    checkpoint_interval: int = 2000
    validation_interval: int = 1000
    early_stopping_patience_evaluations: int = 5
    early_stopping_min_delta: float = 1e-4
    schema_version: str = "pickcube-native-act-optimization-v1"

    def __post_init__(self) -> None:
        if self.optimizer != "adamw" or self.mixed_precision != "bfloat16":
            raise PickCubeActConfigurationError("optimizer/precision contract changed")
        for name in (
            "learning_rate",
            "backbone_learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "early_stopping_min_delta",
        ):
            _finite(cast(float, getattr(self, name)), name)
        for name in (
            "batch_size",
            "dataloader_workers",
            "training_steps",
            "checkpoint_interval",
            "validation_interval",
            "early_stopping_patience_evaluations",
        ):
            _positive(
                cast(int, getattr(self, name)),
                name,
                allow_zero=name == "dataloader_workers",
            )
        if self.checkpoint_interval % self.validation_interval != 0:
            raise PickCubeActConfigurationError(
                "every checkpoint must have validation evidence"
            )
        if self.schema_version != "pickcube-native-act-optimization-v1":
            raise PickCubeActConfigurationError("optimization schema changed")

    def to_mapping(self) -> dict[str, object]:
        """Return all effective optimization settings."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Decode one strict optimization configuration."""
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True, slots=True)
class PickCubeActActionParameterizationConfig:
    """Exact P0.1 bounded-output semantics outside the unchanged ACT backbone."""

    parameterization_type: str = "affine_tanh_v1"
    transform_version: str = "affine_tanh_v1"
    checkpoint_schema_version: str = PICKCUBE_ACT_BOUNDED_CHECKPOINT_SCHEMA
    canonical_training_space: str = "box_minus_one_plus_one_v1"
    action_normalization_mode: str = "IDENTITY"
    eps: float = 1e-6
    continuous_dimensions: tuple[int, ...] = tuple(range(8))
    discrete_dimensions: tuple[int, ...] = ()
    gripper_convention: str = "continuous_normalized_mimic_target_v1"
    temporal_aggregation_mode: str = "disabled_receding_horizon_v1"
    schema_version: str = "pickcube-act-action-parameterization-v1"

    def __post_init__(self) -> None:
        if (
            self.parameterization_type != "affine_tanh_v1"
            or self.transform_version != "affine_tanh_v1"
            or self.checkpoint_schema_version != PICKCUBE_ACT_BOUNDED_CHECKPOINT_SCHEMA
            or self.canonical_training_space != "box_minus_one_plus_one_v1"
            or self.action_normalization_mode != "IDENTITY"
            or self.continuous_dimensions != tuple(range(8))
            or self.discrete_dimensions
            or self.gripper_convention != "continuous_normalized_mimic_target_v1"
            or self.temporal_aggregation_mode != "disabled_receding_horizon_v1"
            or self.schema_version != "pickcube-act-action-parameterization-v1"
        ):
            raise PickCubeActConfigurationError(
                "bounded action parameterization contract changed"
            )
        _finite(self.eps, "action parameterization eps", positive=True)
        if not 0.0 < self.eps < 0.5:
            raise PickCubeActConfigurationError(
                "action parameterization eps must lie in (0,0.5)"
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the stable JSON-native P0.1 action declaration."""
        value = asdict(self)
        value["continuous_dimensions"] = list(self.continuous_dimensions)
        value["discrete_dimensions"] = list(self.discrete_dimensions)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Decode one strict bounded action declaration."""
        payload = dict(value)
        for name in ("continuous_dimensions", "discrete_dimensions"):
            raw = payload.get(name)
            if isinstance(raw, list):
                payload[name] = tuple(raw)
        return cls(**cast(Any, payload))


@dataclass(frozen=True, slots=True)
class PickCubeActGraspSupervisionConfig:
    """Bounded P0.2 phase/gripper supervision without policy-input leakage."""

    temporal_target_offset: int = 0
    phase_weight_exponent: float = 0.5
    maximum_phase_weight: float = 3.0
    arm_loss_weight: float = 0.875
    gripper_loss_weight: float = 0.5
    transition_loss_weight: float = 0.125
    transition_weight_boost: float = 3.0
    close_threshold: float = -0.5
    open_threshold: float = 0.5
    schema_version: str = "pickcube-act-grasp-supervision-v1"

    def __post_init__(self) -> None:
        if (
            type(self.temporal_target_offset) is not int
            or not -2 <= (self.temporal_target_offset) <= 2
        ):
            raise PickCubeActConfigurationError(
                "temporal target offset must lie in [-2,2]"
            )
        for name in (
            "phase_weight_exponent",
            "maximum_phase_weight",
            "arm_loss_weight",
            "gripper_loss_weight",
            "transition_loss_weight",
            "transition_weight_boost",
            "close_threshold",
            "open_threshold",
        ):
            _finite(cast(float, getattr(self, name)), name)
        if not 0.0 <= self.phase_weight_exponent <= 1.0:
            raise PickCubeActConfigurationError(
                "phase weight exponent must lie in [0,1]"
            )
        if not 1.0 <= self.maximum_phase_weight <= 3.0:
            raise PickCubeActConfigurationError(
                "maximum phase weight must lie in [1,3]"
            )
        if not 0.0 < self.arm_loss_weight <= 1.0:
            raise PickCubeActConfigurationError("arm loss weight is invalid")
        if not 0.0 < self.gripper_loss_weight <= 1.0:
            raise PickCubeActConfigurationError("gripper loss weight is invalid")
        if not 0.0 <= self.transition_loss_weight <= 0.5:
            raise PickCubeActConfigurationError("transition loss weight is invalid")
        if not 1.0 <= self.transition_weight_boost <= 4.0:
            raise PickCubeActConfigurationError("transition boost is invalid")
        if not -1.0 < self.close_threshold < 0.0:
            raise PickCubeActConfigurationError("close threshold is invalid")
        if not 0.0 < self.open_threshold < 1.0:
            raise PickCubeActConfigurationError("open threshold is invalid")
        if self.schema_version != "pickcube-act-grasp-supervision-v1":
            raise PickCubeActConfigurationError("grasp supervision schema changed")

    def to_mapping(self) -> dict[str, object]:
        """Return the exact bounded loss and alignment declaration."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Decode one strict P0.2 supervision declaration."""
        return cls(**cast(Any, dict(value)))


@dataclass(frozen=True, slots=True)
class PickCubeActExperimentConfig:
    """Complete native ACT experiment declaration without runtime paths."""

    model: PickCubeActModelConfig
    optimization: PickCubeActOptimizationConfig
    seed: int = 0
    device: str = "cuda"
    deterministic: bool = True
    expected_episode_count: int = 500
    expected_split_counts: tuple[int, int, int] = (400, 50, 50)
    lerobot_version: str = PICKCUBE_ACT_LEROBOT_VERSION
    action_parameterization: PickCubeActActionParameterizationConfig | None = None
    grasp_supervision: PickCubeActGraspSupervisionConfig | None = None
    schema_version: str = "pickcube-native-act-experiment-v1"

    def __post_init__(self) -> None:
        _positive(self.seed, "seed", allow_zero=True)
        if self.device != "cuda" or self.deterministic is not True:
            raise PickCubeActConfigurationError("P0 requires deterministic CUDA")
        if self.expected_episode_count != 500 or self.expected_split_counts != (
            400,
            50,
            50,
        ):
            raise PickCubeActConfigurationError("dataset size/split contract changed")
        if self.lerobot_version != PICKCUBE_ACT_LEROBOT_VERSION:
            raise PickCubeActConfigurationError("LeRobot must be exactly 0.6.0")
        if self.grasp_supervision is not None:
            if self.action_parameterization is None:
                raise PickCubeActConfigurationError(
                    "grasp supervision requires bounded action parameterization"
                )
            if self.schema_version != PICKCUBE_ACT_GRASP_EXPERIMENT_SCHEMA:
                raise PickCubeActConfigurationError(
                    "grasp supervision requires its versioned experiment schema"
                )
        elif self.action_parameterization is None:
            if self.schema_version != "pickcube-native-act-experiment-v1":
                raise PickCubeActConfigurationError("experiment schema changed")
        elif self.schema_version != PICKCUBE_ACT_BOUNDED_EXPERIMENT_SCHEMA:
            raise PickCubeActConfigurationError(
                "bounded experiment requires its versioned schema"
            )

    @property
    def bounded(self) -> bool:
        """Report whether this is the P0.1 intrinsic output parameterization."""
        return self.action_parameterization is not None

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-native resolved experiment."""
        result: dict[str, object] = {
            "deterministic": self.deterministic,
            "device": self.device,
            "expected_episode_count": self.expected_episode_count,
            "expected_split_counts": list(self.expected_split_counts),
            "lerobot_version": self.lerobot_version,
            "model": self.model.to_mapping(),
            "optimization": self.optimization.to_mapping(),
            "schema_version": self.schema_version,
            "seed": self.seed,
        }
        if self.action_parameterization is not None:
            result["action_parameterization"] = (
                self.action_parameterization.to_mapping()
            )
        if self.grasp_supervision is not None:
            result["grasp_supervision"] = self.grasp_supervision.to_mapping()
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Decode one strict resolved experiment."""
        payload = dict(value)
        model = payload.pop("model", None)
        optimization = payload.pop("optimization", None)
        action_parameterization = payload.pop("action_parameterization", None)
        grasp_supervision = payload.pop("grasp_supervision", None)
        counts = payload.get("expected_split_counts")
        if not isinstance(model, Mapping) or not isinstance(optimization, Mapping):
            raise PickCubeActConfigurationError("experiment nested configs are missing")
        if isinstance(counts, list):
            payload["expected_split_counts"] = tuple(counts)
        return cls(
            model=PickCubeActModelConfig.from_mapping(model),
            optimization=PickCubeActOptimizationConfig.from_mapping(optimization),
            action_parameterization=(
                PickCubeActActionParameterizationConfig.from_mapping(
                    cast(Mapping[str, object], action_parameterization)
                )
                if isinstance(action_parameterization, Mapping)
                else None
            ),
            grasp_supervision=(
                PickCubeActGraspSupervisionConfig.from_mapping(
                    cast(Mapping[str, object], grasp_supervision)
                )
                if isinstance(grasp_supervision, Mapping)
                else None
            ),
            **cast(Any, payload),
        )


def build_training_identity(
    *,
    experiment: PickCubeActExperimentConfig,
    dataset_digest: str,
    normalization_digest: str,
    contract_digest: str,
    source_commit: str,
    action_transform: Mapping[str, object] | None = None,
    data_view: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Build the exact resume identity for one native ACT run."""
    body: dict[str, object] = {
        "contract_digest": contract_digest,
        "dataset_digest": dataset_digest,
        "experiment": experiment.to_mapping(),
        "normalization_digest": normalization_digest,
        "source_commit": source_commit,
    }
    if experiment.bounded:
        if action_transform is None:
            raise PickCubeActConfigurationError(
                "bounded training identity requires action transform"
            )
        body["action_transform"] = dict(action_transform)
    elif action_transform is not None:
        raise PickCubeActConfigurationError(
            "unbounded training identity cannot declare action transform"
        )
    if data_view is not None:
        body["data_view"] = dict(data_view)
    body["training_identity_digest"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(body, context="PickCubeActTrainingIdentity")
        ).hexdigest()
    )
    return body


def validate_bounded_final_authorization(
    report: Mapping[str, object], checkpoint_id: str
) -> None:
    """Fail closed unless development promoted this exact bounded checkpoint."""
    if (
        report.get("schema_version") != "pickcube-act-bounded-development-report-v1"
        or report.get("development_gate_passed") is not True
        or report.get("promoted_checkpoint_id") != checkpoint_id
    ):
        raise PickCubeActConfigurationError(
            "bounded final checkpoint lacks exact development authorization"
        )


__all__ = [
    "PICKCUBE_ACT_ACTION_FEATURE",
    "PICKCUBE_ACT_BOUNDED_CHECKPOINT_SCHEMA",
    "PICKCUBE_ACT_BOUNDED_EXPERIMENT_SCHEMA",
    "PICKCUBE_ACT_GRASP_EXPERIMENT_SCHEMA",
    "PICKCUBE_ACT_IMAGE_FEATURE",
    "PICKCUBE_ACT_LEROBOT_VERSION",
    "PICKCUBE_ACT_STATE_FEATURE",
    "PickCubeActConfigurationError",
    "PickCubeActActionParameterizationConfig",
    "PickCubeActExperimentConfig",
    "PickCubeActGraspSupervisionConfig",
    "PickCubeActModelConfig",
    "PickCubeActOptimizationConfig",
    "build_training_identity",
    "validate_bounded_final_authorization",
]
