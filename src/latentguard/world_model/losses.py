"""Masked multi-task losses for WM-v0 and its outcome-only ablation."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
    from torch.nn import functional as functional
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("world-model losses require the training extra") from exc

from .model import WorldModelOutput


@dataclass(frozen=True, slots=True)
class WorldModelTargets:
    """Aligned training targets and explicit availability masks."""

    future_latents: Tensor
    future_progress: Tensor
    progress_mask: Tensor
    event_labels: Tensor
    event_mask: Tensor
    terminal_success: Tensor
    terminal_success_mask: Tensor


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Fixed scalar weights for each supervised objective."""

    future_latent: float = 1.0
    progress: float = 0.5
    events: float = 0.5
    success: float = 0.5
    uncertainty: float = 0.05


@dataclass(frozen=True, slots=True)
class WorldModelLoss:
    """Total loss and detached component losses."""

    total: Tensor
    future_latent: Tensor
    progress: Tensor
    events: Tensor
    success: Tensor
    uncertainty: Tensor


DEFAULT_LOSS_WEIGHTS = LossWeights()


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape or mask.dtype is not torch.bool:
        raise ValueError("loss mask shape or dtype differs")
    if not bool(mask.any()):
        return values.sum() * 0.0
    return values[mask].mean()


def world_model_loss(
    output: WorldModelOutput,
    targets: WorldModelTargets,
    *,
    weights: LossWeights = DEFAULT_LOSS_WEIGHTS,
    include_future_latent: bool = True,
    event_positive_weights: Tensor | None = None,
) -> WorldModelLoss:
    """Compute normalized latent, masked auxiliary, and terminal losses."""

    if output.future_latents.shape != targets.future_latents.shape:
        raise ValueError("future latent target shape differs")
    predicted = functional.normalize(output.future_latents, dim=-1)
    expected = functional.normalize(targets.future_latents, dim=-1)
    latent = (1.0 - (predicted * expected).sum(dim=-1)).mean()
    if not include_future_latent:
        latent = latent * 0.0
    progress = _masked_mean(
        functional.smooth_l1_loss(
            output.future_progress, targets.future_progress, reduction="none"
        ),
        targets.progress_mask,
    )
    event_values = functional.binary_cross_entropy_with_logits(
        output.event_logits,
        targets.event_labels,
        reduction="none",
        pos_weight=event_positive_weights,
    )
    events = _masked_mean(event_values, targets.event_mask)
    success = _masked_mean(
        functional.binary_cross_entropy_with_logits(
            output.success_logit, targets.terminal_success, reduction="none"
        ),
        targets.terminal_success_mask,
    )
    latent_error = (
        (output.future_latents.detach() - targets.future_latents)
        .pow(2)
        .mean(dim=(-1, -2))
    )
    uncertainty = functional.smooth_l1_loss(
        output.uncertainty, latent_error, reduction="mean"
    )
    total = (
        weights.future_latent * latent
        + weights.progress * progress
        + weights.events * events
        + weights.success * success
        + weights.uncertainty * uncertainty
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("world-model loss is non-finite")
    return WorldModelLoss(total, latent, progress, events, success, uncertainty)
