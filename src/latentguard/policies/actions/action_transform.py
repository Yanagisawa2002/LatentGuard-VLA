"""Mathematically bounded action parameterization for native Box controllers."""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import torch
from torch import nn
from torch.nn import functional as F

from latentguard.replay.identity import canonical_json_bytes

ACTION_TRANSFORM_VERSION = "affine_tanh_v1"
ACTION_TRANSFORM_SCHEMA_VERSION = "bounded-action-transform-v1"


class ActionTransformError(ValueError):
    """Raised when action bounds or a transform input violate the contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise ActionTransformError(f"{context}: {reason}")


def _tensor(value: torch.Tensor | Sequence[float], *, context: str) -> torch.Tensor:
    result = torch.as_tensor(value)
    if not result.is_floating_point():
        result = result.to(torch.float32)
    if result.ndim != 1 or result.numel() < 1 or not torch.isfinite(result).all():
        _fail(context, "expected a non-empty finite floating vector")
    return result.detach().clone()


@dataclass(frozen=True, slots=True)
class ActionBounds:
    """Names, units, and exact lower/upper limits for one Box action vector."""

    lower: torch.Tensor
    upper: torch.Tensor
    names: tuple[str, ...]
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        lower = _tensor(self.lower, context="ActionBounds.lower")
        upper = _tensor(self.upper, context="ActionBounds.upper")
        if lower.shape != upper.shape or not torch.all(lower < upper):
            _fail("ActionBounds", "every finite lower bound must be below upper")
        dimension = lower.numel()
        if len(self.names) != dimension or len(self.units) != dimension:
            _fail("ActionBounds", "name/unit inventory differs from dimension")
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in (*self.names, *self.units)
        ):
            _fail("ActionBounds", "names and units must be canonical text")
        if len(set(self.names)) != dimension:
            _fail("ActionBounds", "dimension names must be unique")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        """Return the last-axis action dimension."""
        return len(self.names)

    def to_mapping(self) -> dict[str, object]:
        """Return a device-independent JSON-native bounds declaration."""
        return {
            "lower_bounds": self.lower.to(torch.float64).tolist(),
            "upper_bounds": self.upper.to(torch.float64).tolist(),
            "dimension_names": list(self.names),
            "units": list(self.units),
        }


class BoundedActionTransform(nn.Module):
    """Map legal native targets and unconstrained logits through one contract.

    Finite logits are squashed to ``(-1 + eps, 1 - eps)`` and then mapped by
    the exact affine Box transform. The tiny open-interval margin prevents
    finite-precision ``tanh`` saturation from becoming a boundary ambiguity;
    it is part of the declared model output, not an execution-time repair.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor

    def __init__(self, bounds: ActionBounds, eps: float = 1e-6) -> None:
        super().__init__()
        if type(eps) not in (int, float) or not math.isfinite(float(eps)):
            _fail("BoundedActionTransform.eps", "expected a finite scalar")
        if not 0.0 < float(eps) < 0.5:
            _fail("BoundedActionTransform.eps", "expected value in (0, 0.5)")
        self.names = bounds.names
        self.units = bounds.units
        self.eps = float(eps)
        lower = bounds.lower.detach().clone()
        upper = bounds.upper.detach().clone().to(lower)
        self.register_buffer("lower", lower, persistent=False)
        self.register_buffer("upper", upper, persistent=False)
        self.register_buffer("center", (lower + upper) / 2, persistent=False)
        self.register_buffer("scale", (upper - lower) / 2, persistent=False)

    @property
    def dimension(self) -> int:
        """Return the required final tensor dimension."""
        return len(self.names)

    def _input(self, value: torch.Tensor, *, context: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            _fail(context, "expected a floating torch.Tensor")
        if value.ndim < 1 or value.shape[-1] != self.dimension:
            _fail(context, "last dimension differs from action contract")
        if not torch.isfinite(value).all():
            _fail(context, "NaN and Inf fail closed")
        return value

    def normalize_target(self, environment_action: torch.Tensor) -> torch.Tensor:
        """Map a legal environment action to canonical ``[-1, 1]`` targets."""
        value = self._input(environment_action, context="normalize_target")
        lower = self.lower.to(device=value.device, dtype=value.dtype)
        upper = self.upper.to(device=value.device, dtype=value.dtype)
        if torch.any(value < lower) or torch.any(value > upper):
            _fail("normalize_target", "native training target exceeds exact bounds")
        center = self.center.to(device=value.device, dtype=value.dtype)
        scale = self.scale.to(device=value.device, dtype=value.dtype)
        return (value - center) / scale

    def squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map unconstrained finite model output to the canonical open interval."""
        value = self._input(raw_action, context="squash")
        # BF16/FP16 cannot represent a 1e-6 margin next to one. Compute and
        # retain the bounded policy output in at least FP32 so autocast cannot
        # silently round the declared open interval back to an endpoint.
        work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        return torch.tanh(value.to(work_dtype)) * (1.0 - self.eps)

    def to_environment(self, bounded_normalized_action: torch.Tensor) -> torch.Tensor:
        """Map an already bounded canonical action to exact native coordinates."""
        value = self._input(
            bounded_normalized_action,
            context="to_environment",
        )
        if torch.any(value < -1.0) or torch.any(value > 1.0):
            _fail("to_environment", "canonical action is outside [-1, 1]")
        center = self.center.to(device=value.device, dtype=value.dtype)
        scale = self.scale.to(device=value.device, dtype=value.dtype)
        native = center + scale * value
        lower = self.lower.to(device=value.device, dtype=value.dtype)
        upper = self.upper.to(device=value.device, dtype=value.dtype)
        if (
            not torch.isfinite(native).all()
            or torch.any(native < lower)
            or torch.any(native > upper)
        ):
            _fail("to_environment", "affine output violated its own Box contract")
        return native

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map unconstrained model output directly to legal native actions."""
        return self.to_environment(self.squash(raw_action))

    def to_mapping(self) -> dict[str, object]:
        """Return the complete persistent transform contract and its digest."""
        body: dict[str, object] = {
            "continuous_dimensions": list(range(self.dimension)),
            "dimension_names": list(self.names),
            "discrete_dimensions": [],
            "eps": self.eps,
            "gripper_mapping": {
                "continuous_box_dimension": self.dimension - 1,
                "expert_targets_observed": [-1.0, 1.0],
                "semantic": "normalized_mimic_joint_position_target_v1",
            },
            "lower_bounds": self.lower.to(torch.float64).tolist(),
            "parameterization_type": ACTION_TRANSFORM_VERSION,
            "schema_version": ACTION_TRANSFORM_SCHEMA_VERSION,
            "temporal_aggregation_mode": "disabled_receding_horizon_v1",
            "transform_version": ACTION_TRANSFORM_VERSION,
            "units": list(self.units),
            "upper_bounds": self.upper.to(torch.float64).tolist(),
        }
        body["transform_digest"] = (
            "sha256:"
            + hashlib.sha256(
                canonical_json_bytes(body, context="BoundedActionTransform")
            ).hexdigest()
        )
        return body

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> BoundedActionTransform:
        """Strictly rebuild a transform and verify its content identity."""
        required = {
            "continuous_dimensions",
            "dimension_names",
            "discrete_dimensions",
            "eps",
            "gripper_mapping",
            "lower_bounds",
            "parameterization_type",
            "schema_version",
            "temporal_aggregation_mode",
            "transform_digest",
            "transform_version",
            "units",
            "upper_bounds",
        }
        if set(value) != required:
            _fail("action transform", "field inventory differs")
        if (
            value["schema_version"] != ACTION_TRANSFORM_SCHEMA_VERSION
            or value["parameterization_type"] != ACTION_TRANSFORM_VERSION
            or value["transform_version"] != ACTION_TRANSFORM_VERSION
            or value["temporal_aggregation_mode"] != "disabled_receding_horizon_v1"
            or value["discrete_dimensions"] != []
        ):
            _fail("action transform", "parameterization semantic differs")
        names = value["dimension_names"]
        units = value["units"]
        lower = value["lower_bounds"]
        upper = value["upper_bounds"]
        if not all(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes))
            for item in (names, units, lower, upper)
        ):
            _fail("action transform", "bounds metadata is malformed")
        transform = cls(
            ActionBounds(
                lower=torch.tensor(cast(Sequence[float], lower), dtype=torch.float64),
                upper=torch.tensor(cast(Sequence[float], upper), dtype=torch.float64),
                names=tuple(cast(Sequence[str], names)),
                units=tuple(cast(Sequence[str], units)),
            ),
            eps=cast(float, value["eps"]),
        )
        if transform.to_mapping() != dict(value):
            _fail("action transform", "content digest or semantic fields changed")
        return transform


class BoundedActionRegressionHead(nn.Linear):
    """State-dict-compatible ACT regression head with an intrinsic squash."""

    def bind_transform(self, transform: BoundedActionTransform) -> None:
        """Bind the sole project transform without adding checkpoint tensor keys."""
        if transform.dimension != self.out_features:
            _fail("bounded action head", "output dimension differs from transform")
        object.__setattr__(self, "_transform_reference", weakref.ref(transform))
        object.__setattr__(self, "_last_raw_action", None)
        object.__setattr__(self, "_last_bounded_action", None)

    @classmethod
    def from_linear(
        cls,
        source: nn.Linear,
        transform: BoundedActionTransform,
    ) -> BoundedActionRegressionHead:
        """Reuse exact Linear parameters while changing only output semantics."""
        head = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        head.weight = source.weight
        if source.bias is not None:
            head.bias = source.bias
        head.bind_transform(transform)
        return head

    def transform(self) -> BoundedActionTransform:
        """Return the live action transform or fail after an invalid detach."""
        reference = cast(
            weakref.ReferenceType[BoundedActionTransform], self._transform_reference
        )
        transform = reference()
        if transform is None:
            _fail("bounded action head", "action transform lifetime ended")
        return transform

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Regress raw logits and squash them before ACT computes its loss."""
        raw = F.linear(value, self.weight, self.bias)
        bounded = self.transform().squash(raw)
        object.__setattr__(self, "_last_raw_action", raw)
        object.__setattr__(self, "_last_bounded_action", bounded)
        return bounded

    def latest_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose current-step raw/bounded tensors for diagnostics only."""
        raw = getattr(self, "_last_raw_action", None)
        bounded = getattr(self, "_last_bounded_action", None)
        if not isinstance(raw, torch.Tensor) or not isinstance(bounded, torch.Tensor):
            _fail("bounded action head", "no forward diagnostics are available")
        return raw, bounded


def attach_bounded_action_head(
    policy: Any,
    transform: BoundedActionTransform,
) -> BoundedActionRegressionHead:
    """Replace LeRobot ACT's final Linear with the state-dict-compatible head."""
    model = getattr(policy, "model", None)
    source = getattr(model, "action_head", None)
    if isinstance(source, BoundedActionRegressionHead):
        source.bind_transform(transform)
        return source
    if not isinstance(source, nn.Linear):
        _fail("ACT policy", "expected a public Linear action_head")
    head = BoundedActionRegressionHead.from_linear(source, transform)
    cast(Any, model).action_head = head
    return head


def action_bounds_from_contract(action_spec: Mapping[str, object]) -> ActionBounds:
    """Build exact tensor bounds from the frozen PickCube action section."""
    bounds = action_spec.get("bounds")
    names = action_spec.get("order")
    units = action_spec.get("units")
    if (
        action_spec.get("dimension") != 8
        or not isinstance(bounds, Mapping)
        or not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or not isinstance(units, Sequence)
        or isinstance(units, (str, bytes))
    ):
        _fail("action contract", "PickCube action metadata is incomplete")
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    if (
        not isinstance(lower, Sequence)
        or isinstance(lower, (str, bytes))
        or not isinstance(upper, Sequence)
        or isinstance(upper, (str, bytes))
    ):
        _fail("action contract", "PickCube bounds are malformed")
    return ActionBounds(
        lower=torch.tensor(cast(Sequence[float], lower), dtype=torch.float64),
        upper=torch.tensor(cast(Sequence[float], upper), dtype=torch.float64),
        names=tuple(cast(Sequence[str], names)),
        units=tuple(cast(Sequence[str], units)),
    )


def convex_action_aggregate(
    actions: torch.Tensor,
    weights: torch.Tensor,
    *,
    action_axis: int = -2,
) -> torch.Tensor:
    """Combine bounded actions using explicit non-negative unit-sum weights."""
    if not isinstance(actions, torch.Tensor) or not actions.is_floating_point():
        _fail("convex aggregation", "actions must be floating tensors")
    if not isinstance(weights, torch.Tensor) or not weights.is_floating_point():
        _fail("convex aggregation", "weights must be floating tensors")
    if not torch.isfinite(actions).all() or not torch.isfinite(weights).all():
        _fail("convex aggregation", "NaN and Inf fail closed")
    axis = action_axis if action_axis >= 0 else actions.ndim + action_axis
    if axis < 0 or axis >= actions.ndim - 1:
        _fail("convex aggregation", "action axis is invalid")
    if weights.shape != actions.shape[:-1]:
        _fail("convex aggregation", "weight shape must omit only action dimension")
    if torch.any(weights < 0):
        _fail("convex aggregation", "negative weights are prohibited")
    sums = weights.sum(dim=axis)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0):
        _fail("convex aggregation", "weights must sum to one")
    return (actions * weights.unsqueeze(-1)).sum(dim=axis)


__all__ = [
    "ACTION_TRANSFORM_SCHEMA_VERSION",
    "ACTION_TRANSFORM_VERSION",
    "ActionBounds",
    "ActionTransformError",
    "BoundedActionRegressionHead",
    "BoundedActionTransform",
    "action_bounds_from_contract",
    "attach_bounded_action_head",
    "convex_action_aggregate",
]
