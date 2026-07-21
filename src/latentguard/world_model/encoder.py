"""Frozen observation-encoder interface for action-conditioned dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from latentguard.visual_training.backbone import FrozenBackboneRuntime


class ObservationEncoder(Protocol):
    """Encode ordered RGB views without exposing integration details."""

    @property
    def identity(self) -> str:
        """Return the content-bound encoder identity."""
        ...

    @property
    def latent_dimension(self) -> int:
        """Return the feature dimension per view."""
        ...

    def encode(self, observations: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Encode uint8 `[batch,views,224,224,3]` into `[batch,views,D]`."""
        ...


@dataclass(frozen=True, slots=True)
class FrozenResNet18ObservationEncoder:
    """Adapt the accepted frozen M4B ResNet-18 runtime to WM-v0."""

    runtime: FrozenBackboneRuntime

    @property
    def identity(self) -> str:
        """Return the verified backbone-manifest digest."""

        return self.runtime.manifest.content_digest

    @property
    def latent_dimension(self) -> int:
        """Return the fixed ResNet-18 feature dimension."""

        return self.runtime.manifest.output_feature_dimension

    def encode(self, observations: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Encode all ordered views through one fixed-size frozen batch."""

        if (
            not isinstance(observations, np.ndarray)
            or observations.dtype != np.dtype("uint8")
            or observations.ndim != 5
            or tuple(observations.shape[2:]) != (224, 224, 3)
        ):
            raise ValueError("observations must be uint8 [batch,views,224,224,3]")
        batch, views = observations.shape[:2]
        flattened = np.asarray(
            observations.reshape(batch * views, 224, 224, 3), dtype=np.uint8
        )
        features = self.runtime.extract(flattened)
        return np.asarray(
            features.reshape(batch, views, self.latent_dimension), dtype=np.float32
        )
