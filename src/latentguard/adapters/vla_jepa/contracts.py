"""LIBERO observation/action contracts used by LG-R0 wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from latentguard.adapters.vla_jepa.constants import (
    EXPECTED_ACTION_DIMENSION,
    EXPECTED_IMAGE_KEYS,
    EXPECTED_STATE_DIMENSION,
)


class ContractError(ValueError):
    """Raised when an observation or action violates the frozen LG-R0 contract."""


@dataclass(frozen=True)
class LiberoObservation:
    """Policy-ready LIBERO observation with explicit batch semantics."""

    image: npt.NDArray[np.generic]
    image2: npt.NDArray[np.generic]
    state: npt.NDArray[np.generic]
    task: str


def validate_policy_batch(batch: Mapping[str, Any]) -> None:
    """Validate dimensions and finite values without modifying the input."""

    for key in EXPECTED_IMAGE_KEYS:
        if key not in batch:
            raise ContractError(f"missing image key: {key}")
        image = np.asarray(batch[key])
        if image.ndim not in (4, 5) or image.shape[-3] != 3:
            raise ContractError(
                f"{key} must be BCHW or BTCHW with three channels; got {image.shape}"
            )
        if not np.isfinite(image).all():
            raise ContractError(f"{key} contains non-finite values")
    if "observation.state" not in batch:
        raise ContractError("missing observation.state")
    state = np.asarray(batch["observation.state"])
    if state.ndim not in (2, 3) or state.shape[-1] != EXPECTED_STATE_DIMENSION:
        raise ContractError(
            "observation.state must end in the frozen 8-dimensional state; "
            f"got {state.shape}"
        )
    if not np.isfinite(state).all():
        raise ContractError("observation.state contains non-finite values")
    task = batch.get("task")
    if not isinstance(task, (str, list, tuple)):
        raise ContractError("task must be a string or a sequence of strings")


def validate_action_chunk(action_chunk: Any) -> tuple[int, int, int]:
    """Return the B/T/A shape after enforcing finite seven-dimensional actions."""

    action = np.asarray(action_chunk)
    if action.ndim != 3 or action.shape[-1] != EXPECTED_ACTION_DIMENSION:
        raise ContractError(
            f"action chunk must have shape [batch, horizon, 7]; got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise ContractError("action chunk contains non-finite values")
    return int(action.shape[0]), int(action.shape[1]), int(action.shape[2])
