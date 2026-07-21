"""Fair outcome-prediction baselines for the WM-v0 comparison."""

from __future__ import annotations

from typing import cast

try:
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("world-model baselines require the training extra") from exc

from latentguard.training.config import ActivationName, ModelConfig, ModelType
from latentguard.training.models import (
    TemporalStateActionVerifier,
    build_action_verifier_model,
)

from .model import OutcomeOnlyWorldModel, WorldModelConfig


class DirectVerifierSuccessBaseline(nn.Module):
    """Reuse the accepted M3B temporal architecture on the identical samples."""

    def __init__(self, verifier: TemporalStateActionVerifier) -> None:
        super().__init__()
        self.verifier = verifier

    def forward(self, proprio: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
        """Return success logits by sign-reversing the native failure logit."""

        return cast(Tensor, -self.verifier(proprio, actions, action_mask))


def build_direct_verifier_success_baseline(
    config: ModelConfig, *, seed: int
) -> DirectVerifierSuccessBaseline:
    """Build the current direct verifier without changing its architecture."""

    if config.model_type is not ModelType.TEMPORAL_STATE_ACTION_VERIFIER:
        raise ValueError("direct WM-v0 baseline requires the accepted temporal model")
    model = build_action_verifier_model(config, seed=seed)
    if not isinstance(model, TemporalStateActionVerifier):
        raise TypeError("direct verifier constructor returned the wrong architecture")
    return DirectVerifierSuccessBaseline(model)


def accepted_direct_verifier_config() -> ModelConfig:
    """Return the exact accepted M3B temporal architecture configuration."""

    return ModelConfig(
        model_type=ModelType.TEMPORAL_STATE_ACTION_VERIFIER,
        state_dimension=38,
        action_dimension=8,
        action_horizon=16,
        hidden_dimensions=(128, 128),
        state_projection_dimension=64,
        action_projection_dimension=None,
        temporal_embedding_dimension=128,
        transformer_layers=2,
        attention_heads=4,
        transformer_feedforward_dimension=256,
        dropout=0.1,
        activation=ActivationName.GELU,
    )


def build_matched_outcome_only_baseline(
    config: WorldModelConfig,
) -> OutcomeOnlyWorldModel:
    """Build the same WM-v0 backbone for outcome-only loss training."""

    return OutcomeOnlyWorldModel(config)
