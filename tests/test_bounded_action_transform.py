"""Property tests for the intrinsic P0.1 bounded action parameterization."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from latentguard.policies.actions import (
    ActionBounds,
    ActionTransformError,
    BoundedActionRegressionHead,
    BoundedActionTransform,
    convex_action_aggregate,
)


def _transform() -> BoundedActionTransform:
    return BoundedActionTransform(
        ActionBounds(
            lower=torch.tensor([-3.0, -1.0, -0.25]),
            upper=torch.tensor([1.0, 5.0, 0.75]),
            names=("joint_a", "joint_b", "gripper"),
            units=("rad", "rad", "normalized"),
        )
    )


def test_affine_tanh_is_intrinsically_bounded_for_extreme_logits() -> None:
    transform = _transform()
    raw = torch.tensor(
        [[-1.0e20, 0.0, 1.0e20], [1.0e8, -1.0e8, 3.0]],
        requires_grad=True,
    )
    canonical = transform.squash(raw)
    native = transform.to_environment(canonical)
    assert torch.all(torch.abs(canonical) < 1.0)
    assert torch.all(native > transform.lower)
    assert torch.all(native < transform.upper)
    native.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_autocast_dtypes_cannot_round_margin_back_to_endpoint(
    dtype: torch.dtype,
) -> None:
    transform = _transform()
    raw = torch.full((2, 3), 100.0, dtype=dtype, requires_grad=True)
    bounded = transform.squash(raw)
    assert bounded.dtype == torch.float32
    assert torch.all(bounded < 1.0)
    assert torch.all(bounded > -1.0)
    native = transform.to_environment(bounded)
    assert torch.all(native < transform.upper)
    assert torch.all(native > transform.lower)


def test_target_transform_preserves_asymmetric_box_endpoints() -> None:
    transform = _transform()
    native = torch.stack((transform.lower, transform.upper, transform.center))
    canonical = transform.normalize_target(native)
    assert torch.equal(canonical[0], torch.full((3,), -1.0))
    assert torch.equal(canonical[1], torch.full((3,), 1.0))
    assert torch.equal(canonical[2], torch.zeros(3))
    assert torch.allclose(transform.to_environment(canonical), native)


def test_degenerate_bounds_are_rejected() -> None:
    with pytest.raises(ActionTransformError, match="below upper"):
        ActionBounds(
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([0.0, 2.0]),
            names=("a", "b"),
            units=("rad", "rad"),
        )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_transform_fails_closed_on_nonfinite_values(value: float) -> None:
    transform = _transform()
    with pytest.raises(ActionTransformError, match="fail closed"):
        transform.squash(torch.tensor([[0.0, value, 0.0]]))


def test_transform_identity_is_strict_and_content_bound() -> None:
    transform = _transform()
    mapping = transform.to_mapping()
    restored = BoundedActionTransform.from_mapping(mapping)
    assert restored.to_mapping() == mapping
    changed = dict(mapping)
    changed["eps"] = 1.0e-5
    with pytest.raises(ActionTransformError, match="digest"):
        BoundedActionTransform.from_mapping(changed)


def test_regression_head_keeps_linear_state_keys_and_squashes_before_loss() -> None:
    transform = _transform()
    linear = nn.Linear(5, 3)
    head = BoundedActionRegressionHead.from_linear(linear, transform)
    assert set(head.state_dict()) == {"weight", "bias"}
    value = head(torch.randn(2, 4, 5))
    raw, bounded = head.latest_outputs()
    assert value is bounded
    assert raw.shape == bounded.shape == (2, 4, 3)
    assert torch.all(torch.abs(bounded) < 1.0)


def test_only_convex_action_aggregation_is_accepted() -> None:
    actions = torch.tensor([[[0.2, -0.9], [0.8, 0.7]], [[-0.5, 0.1], [0.5, 0.3]]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    result = convex_action_aggregate(actions, weights)
    assert torch.all(result <= 1.0)
    assert torch.all(result >= -1.0)
    with pytest.raises(ActionTransformError, match="negative"):
        convex_action_aggregate(actions, torch.tensor([[-0.1, 1.1], [0.5, 0.5]]))
