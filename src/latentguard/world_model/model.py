"""Compact masked action-conditioned latent dynamics model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("world-model training requires the training extra") from exc


@dataclass(frozen=True, slots=True)
class WorldModelConfig:
    """Serializable architecture dimensions for WM-v0."""

    latent_dimension: int = 512
    proprio_dimension: int = 38
    action_dimension: int = 8
    action_horizon: int = 16
    prediction_horizon: int = 4
    view_count: int = 3
    event_count: int = 8
    model_dimension: int = 192
    attention_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """Require positive compatible transformer dimensions."""

        integer_fields = (
            "latent_dimension",
            "proprio_dimension",
            "action_dimension",
            "action_horizon",
            "prediction_horizon",
            "view_count",
            "event_count",
            "model_dimension",
            "attention_heads",
            "transformer_layers",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 1
            for name in integer_fields
        ):
            raise ValueError("world-model dimensions must be positive integers")
        if self.model_dimension % self.attention_heads:
            raise ValueError("model_dimension must be divisible by attention_heads")
        if (
            type(self.dropout) is not float
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be a finite value in [0,1)")


@dataclass(frozen=True, slots=True)
class WorldModelOutput:
    """All WM-v0 predictions with explicit aligned shapes."""

    future_latents: Tensor
    future_progress: Tensor
    event_logits: Tensor
    success_logit: Tensor
    uncertainty: Tensor


class ActionConditionedLatentWorldModel(nn.Module):
    """Predict short-horizon latent dynamics from current views and actions."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.model_dimension
        self.visual_projection = nn.Linear(config.latent_dimension, dimension)
        self.proprio_projection = nn.Linear(config.proprio_dimension, dimension)
        self.action_projection = nn.Linear(config.action_dimension, dimension)
        self.action_positions = nn.Parameter(
            torch.zeros(1, config.action_horizon, dimension)
        )
        self.view_positions = nn.Parameter(torch.zeros(1, config.view_count, dimension))
        self.future_queries = nn.Parameter(
            torch.zeros(1, config.prediction_horizon, dimension)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=config.attention_heads,
            dim_feedforward=dimension * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.transformer_layers, enable_nested_tensor=False
        )
        self.future_latent_head = nn.Linear(
            dimension, config.view_count * config.latent_dimension
        )
        self.progress_head = nn.Linear(dimension, 1)
        self.event_head = nn.Linear(dimension, config.event_count)
        self.success_head = nn.Linear(dimension, 1)
        self.uncertainty_head = nn.Linear(dimension, 1)
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.normal_(self.view_positions, std=0.02)
        nn.init.normal_(self.future_queries, std=0.02)

    def forward(
        self,
        current_latents: Tensor,
        proprio: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        *,
        prediction_horizon: int | None = None,
    ) -> WorldModelOutput:
        """Return aligned future, event, outcome, and uncertainty predictions."""

        batch = current_latents.shape[0]
        config = self.config
        if tuple(current_latents.shape[1:]) != (
            config.view_count,
            config.latent_dimension,
        ):
            raise ValueError("current_latents shape differs from model config")
        if tuple(proprio.shape) != (batch, config.proprio_dimension):
            raise ValueError("proprio shape differs from model config")
        if tuple(actions.shape) != (
            batch,
            config.action_horizon,
            config.action_dimension,
        ):
            raise ValueError("actions shape differs from model config")
        if action_mask.dtype is not torch.bool or tuple(action_mask.shape) != (
            batch,
            config.action_horizon,
        ):
            raise ValueError("action_mask must be bool [B,A]")
        if not bool(action_mask.any(dim=1).all()):
            raise ValueError("every batch row needs a valid action")
        horizon = prediction_horizon or config.prediction_horizon
        if not 1 <= horizon <= config.prediction_horizon:
            raise ValueError("prediction_horizon exceeds configured queries")
        visual = self.visual_projection(current_latents) + self.view_positions
        state = self.proprio_projection(proprio).unsqueeze(1)
        action = self.action_projection(actions) + self.action_positions
        query = self.future_queries[:, :horizon].expand(batch, -1, -1)
        tokens = torch.cat((visual, state, action, query), dim=1)
        prefix = config.view_count + 1
        padding = torch.cat(
            (
                torch.zeros(
                    batch,
                    prefix,
                    dtype=torch.bool,
                    device=action_mask.device,
                ),
                ~action_mask,
                torch.zeros(
                    batch,
                    horizon,
                    dtype=torch.bool,
                    device=action_mask.device,
                ),
            ),
            dim=1,
        )
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        future = encoded[:, -horizon:]
        future_latents = self.future_latent_head(future).reshape(
            batch, horizon, config.view_count, config.latent_dimension
        )
        progress = self.progress_head(future).squeeze(-1)
        events = self.event_head(future)
        success = self.success_head(future[:, -1]).squeeze(-1)
        uncertainty = torch.nn.functional.softplus(
            self.uncertainty_head(future).squeeze(-1)
        )
        return WorldModelOutput(
            future_latents=future_latents,
            future_progress=progress,
            event_logits=events,
            success_logit=success,
            uncertainty=uncertainty,
        )


class OutcomeOnlyWorldModel(ActionConditionedLatentWorldModel):
    """Matched backbone ablation trained without future-latent supervision."""


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def config_from_mapping(value: dict[str, Any]) -> WorldModelConfig:
    """Construct a strict model config from a JSON-native mapping."""

    fields = {field: value[field] for field in WorldModelConfig.__dataclass_fields__}
    return WorldModelConfig(**fields)
