"""Binary failure losses with explicit train-only class weighting."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from latentguard.training.config import LossMode


def _validate_binary_targets(targets: Tensor, *, context: str) -> None:
    if not isinstance(targets, Tensor):
        raise TypeError(f"{context}: targets must be a torch tensor")
    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError(f"{context}: targets must have non-empty shape [B]")
    if not torch.is_floating_point(targets):
        raise TypeError(f"{context}: targets must use a floating-point dtype")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError(f"{context}: targets must be finite")
    binary = torch.logical_or(targets == 0.0, targets == 1.0)
    if not bool(binary.all()):
        raise ValueError(f"{context}: targets must be exactly binary 0/1 values")


def compute_positive_class_weight(training_targets: Tensor) -> float:
    """Compute ``negative_count / positive_count`` from training labels only."""

    _validate_binary_targets(training_targets, context="class weight")
    positive_count = int((training_targets == 1.0).sum().item())
    negative_count = int((training_targets == 0.0).sum().item())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("class-weighted BCE requires both training classes")
    weight = negative_count / positive_count
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("computed positive class weight must be finite and positive")
    return float(weight)


def failure_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    mode: LossMode,
    positive_class_weight: float | None = None,
) -> Tensor:
    """Return mean BCE loss under one explicit weighting mode."""

    if not isinstance(logits, Tensor):
        raise TypeError("failure BCE logits must be a torch tensor")
    _validate_binary_targets(targets, context="failure BCE")
    if logits.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("failure BCE logits and targets must have identical shape [B]")
    if not torch.is_floating_point(logits):
        raise TypeError("failure BCE logits must use a floating-point dtype")
    if logits.device != targets.device:
        raise ValueError("failure BCE logits and targets must be on one device")
    if logits.dtype != targets.dtype:
        raise TypeError("failure BCE logits and targets must use the same dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("failure BCE logits must be finite")
    if not isinstance(mode, LossMode):
        raise TypeError("failure BCE mode must be a LossMode")

    if mode is LossMode.UNWEIGHTED_BCE:
        if positive_class_weight is not None:
            raise ValueError("unweighted BCE must not receive a class weight")
        return F.binary_cross_entropy_with_logits(logits, targets)

    if positive_class_weight is None:
        raise ValueError("class-weighted BCE requires a positive class weight")
    if (
        type(positive_class_weight) not in (int, float)
        or not math.isfinite(float(positive_class_weight))
        or float(positive_class_weight) <= 0.0
    ):
        raise ValueError("positive class weight must be finite and positive")
    weight = torch.tensor(
        float(positive_class_weight),
        dtype=logits.dtype,
        device=logits.device,
    )
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


class FailureBCELoss(nn.Module):
    """Stateful BCE module with a manifest-ready fixed class weight."""

    positive_class_weight: Tensor | None

    def __init__(
        self, mode: LossMode, *, positive_class_weight: float | None = None
    ) -> None:
        super().__init__()
        if not isinstance(mode, LossMode):
            raise TypeError("FailureBCELoss.mode must be a LossMode")
        if mode is LossMode.UNWEIGHTED_BCE and positive_class_weight is not None:
            raise ValueError("unweighted BCE must not receive a class weight")
        if mode is LossMode.CLASS_WEIGHTED_BCE and positive_class_weight is None:
            raise ValueError("class-weighted BCE requires a positive class weight")
        if positive_class_weight is not None and (
            type(positive_class_weight) not in (int, float)
            or not math.isfinite(float(positive_class_weight))
            or float(positive_class_weight) <= 0.0
        ):
            raise ValueError("positive class weight must be finite and positive")
        self.mode = mode
        weight = (
            None
            if positive_class_weight is None
            else torch.tensor(float(positive_class_weight), dtype=torch.float64)
        )
        self.register_buffer("positive_class_weight", weight, persistent=True)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return the configured mean binary failure loss."""

        weight = (
            None
            if self.positive_class_weight is None
            else float(self.positive_class_weight.item())
        )
        return failure_bce_with_logits(
            logits,
            targets,
            mode=self.mode,
            positive_class_weight=weight,
        )


def build_failure_loss(mode: LossMode, training_targets: Tensor) -> FailureBCELoss:
    """Build a failure loss, deriving any class weight from training labels only."""

    _validate_binary_targets(training_targets, context="failure loss builder")
    weight = (
        None
        if mode is LossMode.UNWEIGHTED_BCE
        else compute_positive_class_weight(training_targets)
    )
    return FailureBCELoss(mode, positive_class_weight=weight)


__all__ = [
    "FailureBCELoss",
    "build_failure_loss",
    "compute_positive_class_weight",
    "failure_bce_with_logits",
]
