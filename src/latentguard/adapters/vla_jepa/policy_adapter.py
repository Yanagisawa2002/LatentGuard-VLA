"""No-training, no-clipping wrapper around a loaded VLA-JEPA policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from latentguard.adapters.vla_jepa.constants import (
    CHECKPOINT_REVISION,
    LEROBOT_COMMIT,
)
from latentguard.adapters.vla_jepa.contracts import validate_action_chunk
from latentguard.adapters.vla_jepa.schemas import VLAJepaPolicyOutput

if TYPE_CHECKING:
    pass


class PolicyProtocol(Protocol):
    """Minimum native inference surface consumed by the adapter."""

    def predict_action_chunk(self, batch: dict[str, Any]) -> Any:
        """Return a batched native action chunk."""


class VLAJepaAdapter:
    """Expose native VLA-JEPA outputs without modifying their tensor values."""

    def __init__(self, policy: PolicyProtocol) -> None:
        self._policy = policy
        self.optimizer_steps = 0
        self.backward_calls = 0

    def predict_action(self, batch: Mapping[str, Any]) -> VLAJepaPolicyOutput:
        """Run native chunk inference and return the exact tensor object."""

        native = self._policy.predict_action_chunk(dict(batch))
        if hasattr(native, "detach"):
            inspected = native.detach().cpu().numpy()
        else:
            inspected = np.asarray(native)
        shape = validate_action_chunk(inspected)
        return VLAJepaPolicyOutput(
            action_chunk=native,
            action_mask=None,
            policy_metadata={
                "action_shape": list(shape),
                "action_mask_semantic": "not_emitted_by_native_inference",
                "checkpoint_revision": CHECKPOINT_REVISION,
                "lerobot_commit": LEROBOT_COMMIT,
                "output_modified": False,
                "optimizer_steps": self.optimizer_steps,
                "backward_calls": self.backward_calls,
            },
        )

    def inspect_world_model(self, batch: Mapping[str, Any]) -> Any:
        """Dispatch to the separate offline world-model inspector."""

        from latentguard.adapters.vla_jepa.world_model_adapter import (
            inspect_training_world_model,
        )

        return inspect_training_world_model(self._policy, dict(batch))
