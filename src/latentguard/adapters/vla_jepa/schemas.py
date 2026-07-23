"""Output records for the read-only VLA-JEPA adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class VLAJepaPolicyOutput:
    """Unmodified native policy output and provenance metadata."""

    action_chunk: Tensor
    action_mask: Tensor | None
    policy_metadata: dict[str, Any]


@dataclass(frozen=True)
class VLAJepaWorldModelOutput:
    """Read-only tensors exposed from the official training-time world-model path."""

    current_latent: Tensor | None
    predicted_future_latents: Tensor | None
    target_future_latents: Tensor | None
    predictor_metadata: dict[str, Any]
