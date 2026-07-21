"""CPU model shape, masking, and finite-loss smoke tests."""

from __future__ import annotations

import torch

from latentguard.world_model.losses import WorldModelTargets, world_model_loss
from latentguard.world_model.model import (
    ActionConditionedLatentWorldModel,
    WorldModelConfig,
)


def test_forward_shapes_and_finite_loss() -> None:
    """Variable prediction horizon preserves every public output shape."""

    config = WorldModelConfig(
        latent_dimension=12,
        proprio_dimension=5,
        action_dimension=3,
        action_horizon=4,
        prediction_horizon=3,
        view_count=2,
        event_count=4,
        model_dimension=16,
        attention_heads=4,
        transformer_layers=1,
        dropout=0.0,
    )
    model = ActionConditionedLatentWorldModel(config)
    output = model(
        torch.randn(2, 2, 12),
        torch.randn(2, 5),
        torch.randn(2, 4, 3),
        torch.tensor([[True, True, False, False], [True, True, True, True]]),
        prediction_horizon=2,
    )
    assert output.future_latents.shape == (2, 2, 2, 12)
    assert output.future_progress.shape == (2, 2)
    assert output.event_logits.shape == (2, 2, 4)
    assert output.success_logit.shape == (2,)
    assert output.uncertainty.shape == (2, 2)
    targets = WorldModelTargets(
        future_latents=torch.randn(2, 2, 2, 12),
        future_progress=torch.randn(2, 2),
        progress_mask=torch.tensor([[True, True], [False, True]]),
        event_labels=torch.zeros(2, 2, 4),
        event_mask=torch.ones(2, 2, 4, dtype=torch.bool),
        terminal_success=torch.tensor([0.0, 1.0]),
        terminal_success_mask=torch.tensor([True, True]),
    )
    loss = world_model_loss(output, targets)
    assert torch.isfinite(loss.total)
    loss.total.backward()
