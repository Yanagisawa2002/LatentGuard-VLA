"""Numeric action and pre-execution feature contracts for LG-R2a."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

ACTION_DIMENSION = 7
ACTION_HORIZON = 7
ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
    "gripper",
)
DEPLOYABLE_FEATURES = frozenset(
    {
        "current_vla_representation",
        "current_proprioception",
        "instruction_conditioned_current_representation",
        "numeric_action_chunk",
        "action_mask",
        "effective_action_horizon",
    }
)
PROHIBITED_FEATURES = frozenset(
    {
        "future_observation",
        "robometer_future_video_score",
        "topreward_future_video_score",
        "observed_progress_delta",
        "terminal_result",
        "privileged_stage",
        "actual_remaining_episode_length",
    }
)


@dataclass(frozen=True)
class ActionContract:
    """Frozen VLA-JEPA numeric action interface used by LG-R2a."""

    action_dimension: int = ACTION_DIMENSION
    action_horizon: int = ACTION_HORIZON
    execution_horizon: int = ACTION_HORIZON
    dimension_names: tuple[str, ...] = ACTION_NAMES
    dimension_order: str = "relative_position_xyz,relative_axis_angle_xyz,gripper"
    units: tuple[str, ...] = (
        "checkpoint_normalized_relative_translation",
        "checkpoint_normalized_relative_translation",
        "checkpoint_normalized_relative_translation",
        "checkpoint_normalized_relative_rotation",
        "checkpoint_normalized_relative_rotation",
        "checkpoint_normalized_relative_rotation",
        "official_binary_gripper_command",
    )
    lower_bounds: tuple[float, ...] = (-1.0,) * ACTION_DIMENSION
    upper_bounds: tuple[float, ...] = (1.0,) * ACTION_DIMENSION
    normalization: str = (
        "official VLA-JEPA postprocessor output after clipping, gripper "
        "pre-snap, checkpoint-statistic unnormalization, and official "
        "gripper conversion"
    )
    pose_semantics: str = "relative OSC pose command"
    gripper_convention: str = "-1=open, +1=closed"
    padding_value: float = 0.0
    padding_semantic: str = (
        "zero rows are padding only where action_mask=false; accepted LG-R2a "
        "probe samples have seven real rows and an all-true mask"
    )
    native_action_mask: str = (
        "native inference emits null; LG-R2a materializes true for every "
        "real action and false only for explicit zero padding"
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible contract."""

        return asdict(self)

    def validate(
        self,
        actions: npt.ArrayLike,
        masks: npt.ArrayLike,
        *,
        require_full_horizon: bool,
    ) -> dict[str, Any]:
        """Validate shape, values, bounds, masks, padding, and gripper commands."""

        values = np.asarray(actions, dtype=np.float64)
        mask = np.asarray(masks)
        expected_tail = (self.action_horizon, self.action_dimension)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                f"actions must have shape [N,{self.action_horizon},"
                f"{self.action_dimension}]"
            )
        if mask.shape != values.shape[:2]:
            raise ValueError("action mask must have shape [N,H]")
        if mask.dtype != np.bool_:
            raise ValueError("action mask must be boolean")
        if not np.isfinite(values).all():
            raise ValueError("actions must be finite")
        lower = np.asarray(self.lower_bounds, dtype=np.float64)
        upper = np.asarray(self.upper_bounds, dtype=np.float64)
        real = np.broadcast_to(mask[..., None], values.shape)
        if np.any(values[real] < np.broadcast_to(lower, values.shape)[real]):
            raise ValueError("action value falls below the frozen native bound")
        if np.any(values[real] > np.broadcast_to(upper, values.shape)[real]):
            raise ValueError("action value exceeds the frozen native bound")
        if np.any(values[~real] != self.padding_value):
            raise ValueError("masked action rows must use exact zero padding")
        if require_full_horizon and not mask.all():
            raise ValueError("accepted probe samples require a full real horizon")
        gripper = values[..., -1][mask]
        if gripper.size and not np.isin(gripper, [-1.0, 1.0]).all():
            raise ValueError("real gripper commands must use exact -1/+1")
        return {
            "status": "pass",
            "samples": int(values.shape[0]),
            "finite": True,
            "shape": list(values.shape),
            "full_horizon": bool(mask.all()),
            "effective_horizon_min": int(mask.sum(axis=1).min()),
            "effective_horizon_max": int(mask.sum(axis=1).max()),
            "observed_min": values.min(axis=(0, 1)).tolist(),
            "observed_max": values.max(axis=(0, 1)).tolist(),
        }


def validate_deployable_features(features: set[str] | frozenset[str]) -> None:
    """Reject future, outcome-derived, privileged, or unknown model inputs."""

    prohibited = sorted(features & PROHIBITED_FEATURES)
    unknown = sorted(features - DEPLOYABLE_FEATURES)
    if prohibited:
        raise ValueError(f"prohibited pre-execution features: {prohibited}")
    if unknown:
        raise ValueError(f"features are not allowlisted: {unknown}")
