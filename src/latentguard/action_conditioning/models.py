"""Single pre-registered lightweight probe family for LG-R2a."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

OUTPUT_SLICES = {
    "progress": slice(0, 3),
    "stagnation": slice(3, 6),
    "regression": slice(6, 9),
    "events": slice(9, 11),
    "terminal": slice(11, 12),
}


class ProbeModel(nn.Module):
    """Matched lightweight multi-task probe with one concat-gated fusion."""

    def __init__(
        self,
        *,
        variant: str,
        state_dimension: int = 2048,
        action_horizon: int = 7,
        action_dimension: int = 7,
        proprio_dimension: int = 8,
    ) -> None:
        """Construct a registered A/B/C/D probe variant."""

        super().__init__()
        if variant not in {
            "state_only",
            "action_only",
            "state_action",
            "state_action_proprio",
        }:
            raise ValueError(f"unknown probe variant: {variant}")
        self.variant = variant
        action_input = action_horizon * action_dimension + action_horizon
        self.action_encoder: nn.Module | None = None
        self.state_encoder: nn.Module | None = None
        self.proprio_encoder: nn.Module | None = None
        self.gate: nn.Module | None = None
        if variant == "state_only":
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dimension, 138),
                nn.LayerNorm(138),
                nn.GELU(),
            )
            self.trunk = nn.Sequential(nn.Linear(138, 128), nn.GELU())
        elif variant == "action_only":
            self.action_encoder = _action_encoder(action_input)
            self.trunk = nn.Sequential(nn.Linear(32, 128), nn.GELU())
        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dimension, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            self.action_encoder = _action_encoder(action_input)
            fusion_dimension = 160
            if variant == "state_action_proprio":
                self.proprio_encoder = nn.Sequential(
                    nn.Linear(proprio_dimension, 16), nn.GELU()
                )
                fusion_dimension += 16
            self.trunk = nn.Sequential(
                nn.Linear(fusion_dimension, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
            )
            self.gate = nn.Linear(32, 128)
        self.output = nn.Linear(128, 12)

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        action_mask: Tensor,
        proprio: Tensor,
    ) -> Mapping[str, Tensor]:
        """Predict three progress horizons and all registered binary targets."""

        action_flat = torch.cat(
            [action.flatten(start_dim=1), action_mask.to(action.dtype)], dim=1
        )
        if self.variant == "state_only":
            if self.state_encoder is None:
                raise RuntimeError("state encoder is unavailable")
            hidden = self.trunk(self.state_encoder(state))
        elif self.variant == "action_only":
            if self.action_encoder is None:
                raise RuntimeError("action encoder is unavailable")
            hidden = self.trunk(self.action_encoder(action_flat))
        else:
            if self.state_encoder is None or self.action_encoder is None:
                raise RuntimeError("fusion encoders are unavailable")
            state_hidden = self.state_encoder(state)
            action_hidden = self.action_encoder(action_flat)
            pieces = [state_hidden, action_hidden]
            if self.variant == "state_action_proprio":
                if self.proprio_encoder is None:
                    raise RuntimeError("proprio encoder is unavailable")
                pieces.append(self.proprio_encoder(proprio))
            hidden = self.trunk(torch.cat(pieces, dim=1))
            if self.gate is None:
                raise RuntimeError("fusion gate is unavailable")
            hidden = hidden * torch.sigmoid(self.gate(action_hidden))
        output = self.output(hidden)
        return {name: output[:, section] for name, section in OUTPUT_SLICES.items()}

    def parameter_report(self) -> dict[str, Any]:
        """Report total trainable and action-encoder parameter counts."""

        total = sum(parameter.numel() for parameter in self.parameters())
        action = (
            sum(parameter.numel() for parameter in self.action_encoder.parameters())
            if self.action_encoder is not None
            else 0
        )
        return {
            "variant": self.variant,
            "trainable_parameters": total,
            "action_encoder_parameters": action,
        }


def _action_encoder(input_dimension: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dimension, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Linear(64, 32),
        nn.GELU(),
    )
