"""The four fixed compact M4B visual-action verifier architectures."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from latentguard.visual_training.config import VisualModelConfigV1

FROZEN_MODEL_MAX_TRAINABLE_PARAMETERS = 3_000_000
PROBABILITY_EQUIVALENCE_TOLERANCE = 0.0


class TemporalActionEncoder(nn.Module):
    """Shared masked two-layer temporal encoder for every M4B ablation."""

    def __init__(self, config: VisualModelConfigV1) -> None:
        super().__init__()
        width = config.action_embedding_dimension
        self.action_projection = nn.Linear(8, width)
        self.position_embedding = nn.Parameter(torch.empty(16, width))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.action_attention_heads,
            dim_feedforward=config.action_feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.action_transformer_layers,
            enable_nested_tensor=False,
        )
        self.normalization = nn.LayerNorm(width)

    def forward(self, actions: Tensor, action_mask: Tensor) -> Tensor:
        """Return one fixed-size summary for finite masked actions [B,16,8]."""
        if (
            not isinstance(actions, Tensor)
            or actions.ndim != 3
            or tuple(actions.shape[1:]) != (16, 8)
        ):
            raise ValueError("actions must have shape [B,16,8]")
        if (
            not isinstance(action_mask, Tensor)
            or action_mask.dtype is not torch.bool
            or tuple(action_mask.shape) != tuple(actions.shape[:2])
        ):
            raise ValueError("action_mask must be bool [B,16]")
        if actions.shape[0] == 0 or not bool(action_mask.any(dim=1).all()):
            raise ValueError("every action example must have one valid step")
        if not torch.is_floating_point(actions) or not bool(
            torch.isfinite(actions).all()
        ):
            raise ValueError("actions must be finite floating point")
        masked = actions.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        projected = self.action_projection(masked) + self.position_embedding.unsqueeze(
            0
        )
        encoded = self.encoder(projected, src_key_padding_mask=~action_mask)
        encoded = self.normalization(encoded)
        weights = action_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return cast(Tensor, (encoded * weights).sum(dim=1) / weights.sum(dim=1))


class OrderedViewAggregator(nn.Module):
    """Shared projection plus learned slot attention over one or three views."""

    def __init__(self, config: VisualModelConfigV1) -> None:
        super().__init__()
        width = config.view_projection_dimension
        self.view_count = config.view_count
        self.projection = nn.Sequential(
            nn.Linear(512, width), nn.LayerNorm(width), nn.GELU()
        )
        self.slot_embedding = nn.Parameter(torch.empty(3, width))
        nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)
        self.attention = nn.Linear(width, 1)

    def forward(self, features: Tensor) -> Tensor:
        """Aggregate fixed-order features [B,V,512] without camera metadata."""
        if (
            not isinstance(features, Tensor)
            or features.ndim != 3
            or tuple(features.shape[1:]) != (self.view_count, 512)
        ):
            raise ValueError(f"features must have shape [B,{self.view_count},512]")
        if not torch.is_floating_point(features) or not bool(
            torch.isfinite(features).all()
        ):
            raise ValueError("features must be finite floating point")
        projected = self.projection(features)
        projected = projected + self.slot_embedding[: self.view_count].unsqueeze(0)
        weights = torch.softmax(self.attention(projected).squeeze(-1), dim=1)
        return cast(Tensor, (projected * weights.unsqueeze(-1)).sum(dim=1))


class VisualActionVerifier(nn.Module):
    """Student verifier whose forward signature exposes only allowlisted inputs."""

    def __init__(
        self,
        config: VisualModelConfigV1,
        *,
        image_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config.backbone_frozen:
            if image_encoder is not None:
                raise ValueError("frozen-feature models do not own an image encoder")
            self.image_encoder: nn.Module | None = None
        else:
            if image_encoder is None:
                try:
                    import torchvision.models as tv_models  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise ImportError("random ResNet-18 requires torchvision") from exc
                image_encoder = tv_models.resnet18(weights=None)
                image_encoder.fc = nn.Identity()
            self.image_encoder = image_encoder
        self.action_encoder = TemporalActionEncoder(config)
        self.view_aggregator = OrderedViewAggregator(config)
        fused = config.action_embedding_dimension + config.view_projection_dimension
        self.fusion = nn.Sequential(
            nn.Linear(fused, config.fusion_hidden_dimension),
            nn.LayerNorm(config.fusion_hidden_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dimension, 1),
        )

    def _visual_features(self, visual_input: Tensor) -> Tensor:
        if self.config.backbone_frozen:
            return visual_input
        if visual_input.ndim != 5 or tuple(visual_input.shape[1:]) != (3, 3, 224, 224):
            raise ValueError("random model RGB input must have shape [B,3,3,224,224]")
        if not torch.is_floating_point(visual_input) or not bool(
            torch.isfinite(visual_input).all()
        ):
            raise ValueError("RGB tensors must be finite floating point")
        encoder = self.image_encoder
        if encoder is None:
            raise RuntimeError("random model image encoder is missing")
        flattened = visual_input.reshape(-1, 3, 224, 224)
        observed = encoder(flattened)
        if not isinstance(observed, Tensor) or tuple(observed.shape) != (
            flattened.shape[0],
            512,
        ):
            raise ValueError("image encoder must return [B*V,512]")
        return observed.reshape(visual_input.shape[0], 3, 512)

    def forward(
        self, visual_input: Tensor, actions: Tensor, action_mask: Tensor
    ) -> Tensor:
        """Return failure logits from RGB/features, actions, and masks only."""
        features = self._visual_features(visual_input)
        visual_summary = self.view_aggregator(features)
        action_summary = self.action_encoder(actions, action_mask)
        if (
            visual_summary.device != action_summary.device
            or visual_summary.dtype != action_summary.dtype
        ):
            raise ValueError("visual and action tensors must share device and dtype")
        return cast(
            Tensor,
            self.fusion(torch.cat((visual_summary, action_summary), dim=-1)).squeeze(
                -1
            ),
        )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable scalar parameter counts."""
    return (
        sum(item.numel() for item in model.parameters()),
        sum(item.numel() for item in model.parameters() if item.requires_grad),
    )


def build_visual_action_verifier(
    config: VisualModelConfigV1,
    *,
    seed: int,
    image_encoder_factory: Callable[[], nn.Module] | None = None,
) -> VisualActionVerifier:
    """Deterministically construct one of exactly four reviewed M4B models."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = image_encoder_factory() if image_encoder_factory is not None else None
        model = VisualActionVerifier(config, image_encoder=encoder)
    _, trainable = count_parameters(model)
    if config.backbone_frozen and trainable >= FROZEN_MODEL_MAX_TRAINABLE_PARAMETERS:
        raise ValueError("frozen M4B model exceeds the 3M trainable-parameter policy")
    return model


def visual_distillation_loss(
    student_logits: Tensor,
    hard_targets: Tensor,
    teacher_probabilities: Tensor | None,
    *,
    hard_label_weight: float,
    teacher_weight: float,
    temperature: float,
) -> Tensor:
    """Compute fixed hard BCE plus optional soft binary distillation."""
    if student_logits.ndim != 1 or hard_targets.shape != student_logits.shape:
        raise ValueError("student logits and hard targets must be aligned vectors")
    hard = nn.functional.binary_cross_entropy_with_logits(student_logits, hard_targets)
    if teacher_probabilities is None:
        return hard_label_weight * hard
    if (
        teacher_probabilities.shape != student_logits.shape
        or not bool(torch.isfinite(teacher_probabilities).all())
        or not bool(
            ((teacher_probabilities >= 0.0) & (teacher_probabilities <= 1.0)).all()
        )
    ):
        raise ValueError("teacher probabilities must be finite aligned values in [0,1]")
    softened_student = student_logits / temperature
    soft = nn.functional.binary_cross_entropy_with_logits(
        softened_student,
        teacher_probabilities,
    ) * (temperature * temperature)
    return hard_label_weight * hard + teacher_weight * soft


def verify_cached_probability_equivalence(
    model: VisualActionVerifier,
    live_features: Tensor,
    cached_features: Tensor,
    actions: Tensor,
    action_mask: Tensor,
) -> float:
    """Require complete live/cached final probabilities to match exactly."""
    if not model.config.backbone_frozen:
        raise ValueError("probability equivalence requires a frozen-feature model")
    if live_features.shape != cached_features.shape or not torch.equal(
        live_features, cached_features
    ):
        raise ValueError("live and cached feature tensors differ")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            live = torch.sigmoid(model(live_features, actions, action_mask))
            cached = torch.sigmoid(model(cached_features, actions, action_mask))
    finally:
        model.train(was_training)
    maximum = float(torch.max(torch.abs(live - cached)).cpu())
    if maximum > PROBABILITY_EQUIVALENCE_TOLERANCE:
        raise ValueError("live/cached final probability comparison failed")
    return maximum


__all__ = [
    "FROZEN_MODEL_MAX_TRAINABLE_PARAMETERS",
    "PROBABILITY_EQUIVALENCE_TOLERANCE",
    "OrderedViewAggregator",
    "TemporalActionEncoder",
    "VisualActionVerifier",
    "build_visual_action_verifier",
    "count_parameters",
    "visual_distillation_loss",
    "verify_cached_probability_equivalence",
]
