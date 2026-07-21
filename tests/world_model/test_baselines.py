"""Fair direct and matched-backbone baseline tests."""

from __future__ import annotations

import torch

from latentguard.training.config import (
    ActivationName,
    ModelConfig,
    ModelType,
)
from latentguard.world_model.baselines import build_direct_verifier_success_baseline


def test_direct_verifier_baseline_preserves_current_input_contract() -> None:
    """The direct baseline remains the accepted 38-D, 16-by-8 architecture."""

    config = ModelConfig(
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
    baseline = build_direct_verifier_success_baseline(config, seed=0)
    result = baseline(
        torch.zeros(2, 38),
        torch.zeros(2, 16, 8),
        torch.ones(2, 16, dtype=torch.bool),
    )
    assert result.shape == (2,)
