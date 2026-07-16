"""Compact PyTorch Direct Action Verifier baseline architectures."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from latentguard.training.config import (
    ActivationName,
    ModelConfig,
    ModelType,
    ResolvedModelConfig,
    TrainingConfigurationError,
)

MAX_TRAINABLE_PARAMETERS = 2_000_000


def _activation_factory(name: ActivationName) -> Callable[[], nn.Module]:
    factories: dict[ActivationName, Callable[[], nn.Module]] = {
        ActivationName.RELU: nn.ReLU,
        ActivationName.GELU: nn.GELU,
        ActivationName.SILU: nn.SiLU,
    }
    return factories[name]


class _Projection(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dimension, output_dimension)
        self.normalization = nn.LayerNorm(output_dimension)
        self.activation = _activation_factory(activation)()

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, self.activation(self.normalization(self.linear(values))))


class _HiddenBlock(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        *,
        activation: ActivationName,
        dropout: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dimension, output_dimension)
        self.normalization = nn.LayerNorm(output_dimension)
        self.activation = _activation_factory(activation)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.activation(self.normalization(self.linear(values)))
        return cast(Tensor, self.dropout(hidden))


class _HiddenStack(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: tuple[int, ...],
        *,
        activation: ActivationName,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_dimension
        for width in hidden_dimensions:
            blocks.append(
                _HiddenBlock(
                    current,
                    width,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current = width
        self.blocks = nn.Sequential(*blocks)
        self.output_dimension = current

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, self.blocks(values))


class ActionVerifierModel(nn.Module):
    """Base class enforcing the common allowlisted M3B input contract."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def _validated_inputs(
        self, state: Tensor, actions: Tensor, action_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(state, Tensor) or not isinstance(actions, Tensor):
            raise TypeError("model state and actions must be torch tensors")
        if not isinstance(action_mask, Tensor):
            raise TypeError("model action_mask must be a torch tensor")
        if state.ndim != 2 or state.shape[1] != self.config.state_dimension:
            raise ValueError(
                "state must have shape [B, "
                f"{self.config.state_dimension}], got {tuple(state.shape)}"
            )
        expected_actions = (
            state.shape[0],
            self.config.action_horizon,
            self.config.action_dimension,
        )
        if actions.ndim != 3 or tuple(actions.shape) != expected_actions:
            raise ValueError(
                "actions must have shape "
                f"{expected_actions}, got {tuple(actions.shape)}"
            )
        expected_mask = (state.shape[0], self.config.action_horizon)
        if action_mask.ndim != 2 or tuple(action_mask.shape) != expected_mask:
            raise ValueError(
                "action_mask must have shape "
                f"{expected_mask}, got {tuple(action_mask.shape)}"
            )
        if state.shape[0] == 0:
            raise ValueError("model batches must contain at least one example")
        if not torch.is_floating_point(state) or not torch.is_floating_point(actions):
            raise TypeError("model state and actions must use floating-point dtypes")
        if action_mask.dtype is not torch.bool:
            raise TypeError("model action_mask must use torch.bool")
        if state.dtype != actions.dtype:
            raise TypeError("model state and actions must use the same dtype")
        if state.device != actions.device or state.device != action_mask.device:
            raise ValueError("model inputs must be on one device")
        if not bool(torch.isfinite(state).all()) or not bool(
            torch.isfinite(actions).all()
        ):
            raise ValueError("model inputs must contain only finite values")
        if not bool(action_mask.any(dim=1).all()):
            raise ValueError(
                "every example must contain at least one valid action step"
            )
        masked_actions = actions.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        return state, masked_actions, action_mask


class StateOnlyMLP(ActionVerifierModel):
    """Predict failure from the normalized 38-dimensional state only."""

    def __init__(self, config: ModelConfig) -> None:
        if config.model_type is not ModelType.STATE_ONLY_MLP:
            raise TrainingConfigurationError(
                "StateOnlyMLP requires model_type='state_only_mlp'"
            )
        super().__init__(config)
        projection_dimension = cast(int, config.state_projection_dimension)
        self.state_projection = _Projection(
            config.state_dimension, projection_dimension, config.activation
        )
        self.hidden = _HiddenStack(
            projection_dimension,
            config.hidden_dimensions,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.failure_head = nn.Linear(self.hidden.output_dimension, 1)

    def forward(self, state: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
        """Return one scalar failure logit per example."""

        state, _, _ = self._validated_inputs(state, actions, action_mask)
        hidden = self.hidden(self.state_projection(state))
        return cast(Tensor, self.failure_head(hidden).squeeze(-1))


class ActionOnlyMLP(ActionVerifierModel):
    """Predict failure from the masked, flattened action chunk only."""

    def __init__(self, config: ModelConfig) -> None:
        if config.model_type is not ModelType.ACTION_ONLY_MLP:
            raise TrainingConfigurationError(
                "ActionOnlyMLP requires model_type='action_only_mlp'"
            )
        super().__init__(config)
        projection_dimension = cast(int, config.action_projection_dimension)
        flattened_dimension = config.action_horizon * config.action_dimension
        self.action_projection = _Projection(
            flattened_dimension, projection_dimension, config.activation
        )
        self.hidden = _HiddenStack(
            projection_dimension,
            config.hidden_dimensions,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.failure_head = nn.Linear(self.hidden.output_dimension, 1)

    def forward(self, state: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
        """Return one scalar failure logit per example."""

        _, masked_actions, _ = self._validated_inputs(state, actions, action_mask)
        flattened = masked_actions.reshape(masked_actions.shape[0], -1)
        hidden = self.hidden(self.action_projection(flattened))
        return cast(Tensor, self.failure_head(hidden).squeeze(-1))


class StateActionMLP(ActionVerifierModel):
    """Predict failure from separately projected state and action inputs."""

    def __init__(self, config: ModelConfig) -> None:
        if config.model_type is not ModelType.STATE_ACTION_MLP:
            raise TrainingConfigurationError(
                "StateActionMLP requires model_type='state_action_mlp'"
            )
        super().__init__(config)
        state_projection_dimension = cast(int, config.state_projection_dimension)
        action_projection_dimension = cast(int, config.action_projection_dimension)
        self.state_projection = _Projection(
            config.state_dimension, state_projection_dimension, config.activation
        )
        self.action_projection = _Projection(
            config.action_horizon * config.action_dimension,
            action_projection_dimension,
            config.activation,
        )
        self.fusion = _HiddenStack(
            state_projection_dimension + action_projection_dimension,
            config.hidden_dimensions,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.failure_head = nn.Linear(self.fusion.output_dimension, 1)

    def forward(self, state: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
        """Return one scalar failure logit per example."""

        state, masked_actions, _ = self._validated_inputs(state, actions, action_mask)
        state_hidden = self.state_projection(state)
        action_hidden = self.action_projection(
            masked_actions.reshape(masked_actions.shape[0], -1)
        )
        fused = torch.cat((state_hidden, action_hidden), dim=-1)
        return cast(Tensor, self.failure_head(self.fusion(fused)).squeeze(-1))


class TemporalStateActionVerifier(ActionVerifierModel):
    """Compact two-layer temporal action encoder fused with structured state."""

    def __init__(self, config: ModelConfig) -> None:
        if config.model_type is not ModelType.TEMPORAL_STATE_ACTION_VERIFIER:
            raise TrainingConfigurationError(
                "TemporalStateActionVerifier requires the temporal model type"
            )
        super().__init__(config)
        state_projection_dimension = cast(int, config.state_projection_dimension)
        embedding_dimension = cast(int, config.temporal_embedding_dimension)
        transformer_layers = cast(int, config.transformer_layers)
        attention_heads = cast(int, config.attention_heads)
        feedforward_dimension = cast(int, config.transformer_feedforward_dimension)
        self.action_projection = nn.Linear(config.action_dimension, embedding_dimension)
        self.position_embedding = nn.Parameter(
            torch.empty(config.action_horizon, embedding_dimension)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=config.dropout,
            activation=config.activation.value,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.temporal_normalization = nn.LayerNorm(embedding_dimension)
        self.state_projection = _Projection(
            config.state_dimension, state_projection_dimension, config.activation
        )
        self.fusion = _HiddenStack(
            state_projection_dimension + embedding_dimension,
            config.hidden_dimensions,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.failure_head = nn.Linear(self.fusion.output_dimension, 1)

    def forward(self, state: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
        """Return one scalar failure logit per example."""

        state, masked_actions, action_mask = self._validated_inputs(
            state, actions, action_mask
        )
        projected = self.action_projection(masked_actions)
        projected = projected + self.position_embedding.unsqueeze(0)
        encoded = self.temporal_encoder(
            projected,
            src_key_padding_mask=~action_mask,
        )
        encoded = self.temporal_normalization(encoded)
        weights = action_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        summary = (encoded * weights).sum(dim=1) / weights.sum(dim=1)
        state_hidden = self.state_projection(state)
        fused = torch.cat((state_hidden, summary), dim=-1)
        return cast(Tensor, self.failure_head(self.fusion(fused)).squeeze(-1))


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the exact number of scalar parameters that require gradients."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def resolve_model_config(model: ActionVerifierModel) -> ResolvedModelConfig:
    """Bind a constructed model architecture to its exact parameter count."""

    return ResolvedModelConfig(
        architecture=model.config,
        parameter_count=count_trainable_parameters(model),
    )


def build_action_verifier_model(
    config: ModelConfig, *, seed: int
) -> ActionVerifierModel:
    """Construct one fixed baseline with deterministic CPU initialization."""

    if type(seed) is not int or seed < 0:
        raise ValueError("model initialization seed must be a non-negative integer")
    constructors: dict[ModelType, type[ActionVerifierModel]] = {
        ModelType.STATE_ONLY_MLP: StateOnlyMLP,
        ModelType.ACTION_ONLY_MLP: ActionOnlyMLP,
        ModelType.STATE_ACTION_MLP: StateActionMLP,
        ModelType.TEMPORAL_STATE_ACTION_VERIFIER: TemporalStateActionVerifier,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = constructors[config.model_type](config)
    parameter_count = count_trainable_parameters(model)
    if parameter_count >= MAX_TRAINABLE_PARAMETERS:
        raise TrainingConfigurationError(
            "M3B learned baselines must remain below "
            f"{MAX_TRAINABLE_PARAMETERS} trainable parameters; got {parameter_count}"
        )
    return model


build_model = build_action_verifier_model


__all__ = [
    "MAX_TRAINABLE_PARAMETERS",
    "ActionOnlyMLP",
    "ActionVerifierModel",
    "StateActionMLP",
    "StateOnlyMLP",
    "TemporalStateActionVerifier",
    "build_action_verifier_model",
    "build_model",
    "count_trainable_parameters",
    "resolve_model_config",
]
