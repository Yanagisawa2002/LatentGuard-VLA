"""CPU-only architecture and privileged-distillation tests for M4B."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from latentguard.visual_training.config import load_visual_model_config
from latentguard.visual_training.models import (
    build_visual_action_verifier,
    count_parameters,
    verify_cached_probability_equivalence,
    visual_distillation_loss,
)

_ROOT = Path("configs/training/m4b")


class _TinyImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(3, 512)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(images).flatten(1))


@pytest.mark.parametrize(
    ("config_name", "view_count"),
    (
        ("frozen-resnet18-singleview-action.json", 1),
        ("frozen-resnet18-multiview-action.json", 3),
        ("frozen-resnet18-multiview-distilled.json", 3),
    ),
)
def test_frozen_models_forward_backward_and_parameter_budget(
    config_name: str, view_count: int
) -> None:
    """All frozen-feature variants are trainable and remain below 3M parameters."""
    config = load_visual_model_config(_ROOT / config_name)
    model = build_visual_action_verifier(config, seed=0)
    features = torch.randn(2, view_count, 512)
    actions = torch.randn(2, 16, 8)
    mask = torch.ones(2, 16, dtype=torch.bool)

    logits = model(features, actions, mask)
    logits.sum().backward()
    _, trainable = count_parameters(model)

    assert logits.shape == (2,)
    assert trainable < 3_000_000
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_random_multiview_model_uses_raw_rgb() -> None:
    """The random-backbone baseline consumes RGB end to end, not cached features."""
    config = load_visual_model_config(_ROOT / "random-resnet18-multiview-action.json")
    model = build_visual_action_verifier(
        config, seed=0, image_encoder_factory=_TinyImageEncoder
    )
    logits = model(
        torch.randn(1, 3, 3, 224, 224),
        torch.randn(1, 16, 8),
        torch.ones(1, 16, dtype=torch.bool),
    )
    logits.sum().backward()

    assert logits.shape == (1,)
    assert model.image_encoder is not None
    assert any(
        parameter.grad is not None for parameter in model.image_encoder.parameters()
    )


def test_masked_action_components_do_not_change_prediction() -> None:
    """Masked continuation values never leak into the student prediction."""
    config = load_visual_model_config(_ROOT / "frozen-resnet18-multiview-action.json")
    model = build_visual_action_verifier(config, seed=0).eval()
    features = torch.randn(1, 3, 512)
    actions = torch.randn(1, 16, 8)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, :5] = True
    changed = actions.clone()
    changed[:, 5:] = 10_000.0

    with torch.inference_mode():
        first = model(features, actions, mask)
        second = model(features, changed, mask)

    assert torch.equal(first, second)


def test_distillation_loss_validates_privileged_probabilities() -> None:
    """Teacher targets are optional soft labels and must be valid probabilities."""
    logits = torch.tensor([0.2, -0.3], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    teacher = torch.tensor([0.8, 0.1])
    loss = visual_distillation_loss(
        logits,
        targets,
        teacher,
        hard_label_weight=0.7,
        teacher_weight=0.3,
        temperature=2.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None

    with pytest.raises(ValueError, match="probabilities"):
        visual_distillation_loss(
            logits.detach(),
            targets,
            torch.tensor([1.2, 0.1]),
            hard_label_weight=0.7,
            teacher_weight=0.3,
            temperature=2.0,
        )


def test_cached_features_preserve_final_probabilities_exactly() -> None:
    """The end-to-end cache gate compares final probabilities, not features only."""
    config = load_visual_model_config(_ROOT / "frozen-resnet18-multiview-action.json")
    model = build_visual_action_verifier(config, seed=0)
    features = torch.randn(2, 3, 512)
    actions = torch.randn(2, 16, 8)
    mask = torch.ones(2, 16, dtype=torch.bool)

    assert (
        verify_cached_probability_equivalence(
            model, features, features.clone(), actions, mask
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="feature tensors differ"):
        verify_cached_probability_equivalence(
            model, features, features + 1e-6, actions, mask
        )


@pytest.mark.parametrize(
    "config_name",
    (
        "random-resnet18-multiview-action.json",
        "frozen-resnet18-singleview-action.json",
        "frozen-resnet18-multiview-action.json",
        "frozen-resnet18-multiview-distilled.json",
    ),
)
def test_all_four_models_tiny_overfit(config_name: str) -> None:
    """Every reviewed architecture can reduce loss on one fixed tiny batch."""
    config = load_visual_model_config(_ROOT / config_name)
    factory = _TinyImageEncoder if not config.backbone_frozen else None
    model = build_visual_action_verifier(
        config, seed=11, image_encoder_factory=factory
    ).eval()
    if config.backbone_frozen:
        visual = torch.randn(4, config.view_count, 512)
    else:
        visual = torch.randn(4, 3, 3, 224, 224)
    actions = torch.randn(4, 16, 8)
    mask = torch.ones(4, 16, dtype=torch.bool)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    losses: list[float] = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        logits = model(visual, actions, mask)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]
