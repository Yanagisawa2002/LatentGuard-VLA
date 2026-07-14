"""CPU-only behavioral tests for the six deterministic M1 corruptions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from latentguard.corruptions import (
    ActionCorruption,
    ActionField,
    ActionLayout,
    ActionSemantic,
    AdditiveGaussianNoise,
    ConstantBias,
    CorruptionApplicabilityError,
    CorruptionConfigurationError,
    CorruptionRegistry,
    DuplicateCorruptionError,
    LocalTemporalPermutation,
    SegmentHold,
    SegmentZeroing,
    TemporalFieldShift,
    TemporalFillPolicy,
    UnknownCorruptionError,
    create_corruption,
    default_corruption_registry,
)
from latentguard.models import ActionChunk

_FLOAT32_DTYPE = np.dtype(np.float32)


def _layout(action_dim: int = 6) -> ActionLayout:
    return ActionLayout(
        action_dim,
        (
            ActionField("translation", (1, 3), ActionSemantic.TRANSLATION),
            ActionField("rotation", (0, 4), ActionSemantic.ROTATION),
            ActionField("gripper", (5,), ActionSemantic.GRIPPER),
        ),
    )


def _action(
    *,
    horizon: int = 6,
    action_dim: int = 6,
    dtype: np.dtype[np.generic] = _FLOAT32_DTYPE,
) -> ActionChunk:
    values = np.arange(horizon * action_dim).reshape(horizon, action_dim)
    return ActionChunk(values.astype(dtype), "tool-frame", 0.1)


def _all_corruptions() -> tuple[ActionCorruption, ...]:
    return (
        AdditiveGaussianNoise(standard_deviation=0.2, target_fields=("translation",)),
        ConstantBias(bias=(0.5, -0.25), target_fields=("translation",)),
        TemporalFieldShift(
            target_field="translation",
            shift_steps=1,
            fill_policy=TemporalFillPolicy.EDGE,
        ),
        SegmentHold(start_step=2, end_step=5, target_fields=("translation",)),
        SegmentZeroing(start_step=1, end_step=4, target_fields=("translation",)),
        LocalTemporalPermutation(
            start_step=1, end_step=6, target_fields=("translation",)
        ),
    )


@pytest.mark.parametrize("corruption", _all_corruptions())
def test_every_corruption_is_deterministic_detached_and_shape_dtype_safe(
    corruption: ActionCorruption,
) -> None:
    source = _action(action_dim=6)
    before = source.actions.copy()

    first = corruption.apply(source, _layout(), seed=1234)
    second = corruption.apply(source, _layout(), seed=1234)

    np.testing.assert_array_equal(source.actions, before)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.actions.shape == source.actions.shape
    assert first.actions.dtype == source.actions.dtype
    assert first.actions is not source.actions
    assert not first.actions.flags.writeable
    np.testing.assert_array_equal(
        first.actions[:, (0, 2, 4, 5)], before[:, (0, 2, 4, 5)]
    )


def test_gaussian_noise_changes_only_selected_dimensions_and_varies_by_seed() -> None:
    corruption = AdditiveGaussianNoise(
        standard_deviation=0.5, mean=0.2, target_indices=(2, 5)
    )
    source = _action()

    first = corruption.apply(source, _layout(), seed=1)
    second = corruption.apply(source, _layout(), seed=2)

    assert not np.array_equal(first.actions[:, (2, 5)], second.actions[:, (2, 5)])
    np.testing.assert_array_equal(
        first.actions[:, (0, 1, 3, 4)], source.actions[:, (0, 1, 3, 4)]
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_gaussian_noise_rejects_invalid_standard_deviation(value: float) -> None:
    with pytest.raises(CorruptionConfigurationError, match="standard_deviation"):
        AdditiveGaussianNoise(standard_deviation=value, target_indices=(0,))


def test_gaussian_noise_rejects_integer_action_without_silent_cast() -> None:
    corruption = AdditiveGaussianNoise(standard_deviation=1.0, target_indices=(0,))
    with pytest.raises(CorruptionApplicabilityError, match="floating-point"):
        corruption.apply(_action(dtype=np.dtype(np.int16)), _layout(), seed=1)


def test_constant_bias_matches_scalar_and_per_dimension_values() -> None:
    source = _action()
    scalar = ConstantBias(bias=2.5, target_indices=(1, 3)).apply(
        source, _layout(), seed=0
    )
    vector = ConstantBias(bias=(2.5, -1.5), target_indices=(1, 3)).apply(
        source, _layout(), seed=999
    )

    np.testing.assert_allclose(scalar.actions[:, 1], source.actions[:, 1] + 2.5)
    np.testing.assert_allclose(scalar.actions[:, 3], source.actions[:, 3] + 2.5)
    np.testing.assert_allclose(vector.actions[:, 1], source.actions[:, 1] + 2.5)
    np.testing.assert_allclose(vector.actions[:, 3], source.actions[:, 3] - 1.5)


def test_constant_bias_preserves_integer_dtype_when_exactly_representable() -> None:
    source = _action(dtype=np.dtype(np.int16))
    result = ConstantBias(bias=(2.0, -3.0), target_indices=(1, 3)).apply(
        source, _layout(), seed=4
    )

    assert result.actions.dtype == np.int16
    np.testing.assert_array_equal(result.actions[:, 1], source.actions[:, 1] + 2)
    np.testing.assert_array_equal(result.actions[:, 3], source.actions[:, 3] - 3)


def test_constant_bias_preserves_large_integer_parameter_exactly() -> None:
    value = 9007199254740993
    source = ActionChunk(np.zeros((2, 1), dtype=np.int64), "integer-frame", 0.1)
    layout = ActionLayout(1, (ActionField("selected", (0,), ActionSemantic.AUXILIARY),))
    corruption = ConstantBias(bias=value, target_indices=(0,))

    resolved = corruption.resolved_parameters_for(source, layout)
    result = corruption.apply(source, layout, seed=0)

    assert resolved["bias"] == (value,)
    assert result.actions.dtype == np.int64
    assert result.actions[0, 0] == value


def test_constant_bias_rejects_fractional_or_overflowing_integer_results() -> None:
    source = ActionChunk(np.array([[127, 1]], dtype=np.int8), "integer-frame", 0.1)
    layout = ActionLayout(2, (ActionField("selected", (0,), ActionSemantic.AUXILIARY),))
    with pytest.raises(CorruptionApplicabilityError, match="integral"):
        ConstantBias(bias=0.5, target_indices=(0,)).apply(source, layout, seed=0)
    with pytest.raises(CorruptionApplicabilityError, match="outside dtype range"):
        ConstantBias(bias=1.0, target_indices=(0,)).apply(source, layout, seed=0)


def test_constant_bias_rejects_incompatible_vector_length_and_nonfinite_values() -> (
    None
):
    with pytest.raises(CorruptionApplicabilityError, match="bias length"):
        ConstantBias(bias=(1.0,), target_fields=("translation",)).apply(
            _action(), _layout(), seed=0
        )
    with pytest.raises(CorruptionConfigurationError, match="finite"):
        ConstantBias(bias=(1.0, float("inf")), target_indices=(0, 1))


@pytest.mark.parametrize(
    ("shift", "policy", "expected"),
    [
        (1, TemporalFillPolicy.EDGE, np.array([5, 5, 11, 17, 23, 29])),
        (2, TemporalFillPolicy.ZERO, np.array([0, 0, 5, 11, 17, 23])),
        (-1, TemporalFillPolicy.EDGE, np.array([11, 17, 23, 29, 35, 35])),
        (-2, TemporalFillPolicy.ZERO, np.array([17, 23, 29, 35, 0, 0])),
    ],
)
def test_temporal_shift_sign_and_fill_policy(
    shift: int, policy: TemporalFillPolicy, expected: np.ndarray
) -> None:
    source = _action()
    result = TemporalFieldShift(
        target_field="gripper", shift_steps=shift, fill_policy=policy
    ).apply(source, _layout(), seed=8)

    np.testing.assert_array_equal(result.actions[:, 5], expected)
    np.testing.assert_array_equal(result.actions[:, :5], source.actions[:, :5])


def test_temporal_shift_rejects_zero_large_or_missing_field() -> None:
    with pytest.raises(CorruptionConfigurationError, match="zero shift"):
        TemporalFieldShift(
            target_field="gripper", shift_steps=0, fill_policy=TemporalFillPolicy.EDGE
        )
    with pytest.raises(CorruptionApplicabilityError, match="smaller than horizon"):
        TemporalFieldShift(
            target_field="gripper", shift_steps=6, fill_policy=TemporalFillPolicy.EDGE
        ).apply(_action(), _layout(), seed=0)
    with pytest.raises(CorruptionApplicabilityError, match="unknown field"):
        TemporalFieldShift(
            target_field="not-declared",
            shift_steps=1,
            fill_policy=TemporalFillPolicy.EDGE,
        ).apply(_action(), _layout(), seed=0)


def test_segment_hold_uses_immediately_preceding_value() -> None:
    source = _action()
    result = SegmentHold(
        start_step=2, end_step=5, target_fields=("translation",)
    ).apply(source, _layout(), seed=1)

    expected = np.repeat(source.actions[1:2, (1, 3)], 3, axis=0)
    np.testing.assert_array_equal(result.actions[2:5, (1, 3)], expected)
    np.testing.assert_array_equal(result.actions[[0, 1, 5]], source.actions[[0, 1, 5]])


def test_segment_hold_without_end_extends_to_horizon() -> None:
    source = _action(action_dim=6)
    result = SegmentHold(start_step=4, target_indices=(2,)).apply(
        source, _layout(), seed=5
    )
    np.testing.assert_array_equal(result.actions[4:, 2], [source.actions[3, 2]] * 2)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SegmentHold(start_step=0),
        lambda: SegmentHold(start_step=2, end_step=2),
        lambda: SegmentZeroing(start_step=-1, end_step=2),
        lambda: SegmentZeroing(start_step=3, end_step=3),
        lambda: LocalTemporalPermutation(start_step=1, end_step=2),
    ],
)
def test_segment_configuration_rejects_invalid_boundaries(
    factory: Callable[[], ActionCorruption],
) -> None:
    with pytest.raises(CorruptionConfigurationError):
        factory()


@pytest.mark.parametrize(
    "corruption",
    [
        SegmentHold(start_step=5, end_step=7),
        SegmentZeroing(start_step=5, end_step=7),
        LocalTemporalPermutation(start_step=4, end_step=7),
    ],
)
def test_segment_applicability_rejects_bounds_outside_horizon(
    corruption: ActionCorruption,
) -> None:
    with pytest.raises(CorruptionApplicabilityError, match="outside horizon"):
        corruption.apply(_action(), _layout(), seed=0)


def test_segment_zeroing_preserves_unselected_dimensions() -> None:
    source = _action()
    result = SegmentZeroing(start_step=1, end_step=4, target_fields=("gripper",)).apply(
        source, _layout(), seed=2
    )

    np.testing.assert_array_equal(result.actions[1:4, 5], 0)
    np.testing.assert_array_equal(result.actions[:, :5], source.actions[:, :5])
    np.testing.assert_array_equal(
        result.actions[[0, 4, 5], 5], source.actions[[0, 4, 5], 5]
    )


def test_local_permutation_preserves_selected_multiset_and_unselected_values() -> None:
    source = _action(horizon=8)
    corruption = LocalTemporalPermutation(
        start_step=1, end_step=7, target_fields=("rotation",)
    )
    result = corruption.apply(source, _layout(), seed=29)

    for dimension in (0, 4):
        np.testing.assert_array_equal(
            np.sort(result.actions[1:7, dimension]),
            np.sort(source.actions[1:7, dimension]),
        )
    assert not np.array_equal(result.actions[1:7, (0, 4)], source.actions[1:7, (0, 4)])
    np.testing.assert_array_equal(
        result.actions[:, (1, 2, 3, 5)], source.actions[:, (1, 2, 3, 5)]
    )


def test_all_dimension_segment_selection_supports_non_seven_dimensional_actions() -> (
    None
):
    source = _action(horizon=5, action_dim=9)
    layout = ActionLayout(
        9, (ActionField("only_declared", (8,), ActionSemantic.UNSPECIFIED),)
    )
    result = SegmentZeroing(start_step=1, end_step=3).apply(source, layout, seed=0)
    np.testing.assert_array_equal(result.actions[1:3], 0)
    assert result.actions.shape == (5, 9)


def test_dimension_mismatch_and_invalid_seed_are_explicit() -> None:
    corruption = SegmentZeroing(start_step=1, end_step=2)
    with pytest.raises(CorruptionApplicabilityError, match="does not match"):
        corruption.apply(_action(action_dim=5), _layout(6), seed=0)
    with pytest.raises(CorruptionConfigurationError, match="seed"):
        corruption.apply(_action(), _layout(), seed=-1)


def test_target_selection_rejects_empty_ambiguous_duplicate_or_out_of_range() -> None:
    with pytest.raises(CorruptionConfigurationError, match="must not be empty"):
        ConstantBias(bias=1.0, target_indices=())
    with pytest.raises(CorruptionConfigurationError, match="ambiguous"):
        ConstantBias(bias=1.0, target_fields=("translation",), target_indices=(1,))
    with pytest.raises(CorruptionConfigurationError, match="duplicate"):
        ConstantBias(bias=1.0, target_indices=(1, 1))
    with pytest.raises(CorruptionApplicabilityError, match="outside"):
        ConstantBias(bias=1.0, target_indices=(6,)).apply(_action(), _layout(), seed=0)


def test_configs_are_frozen_and_parameters_are_immutable_and_resolved() -> None:
    corruption = AdditiveGaussianNoise(
        standard_deviation=0.3,
        target_fields=["translation"],  # type: ignore[arg-type]
    )
    assert corruption.resolved_parameters == {
        "target_fields": ("translation",),
        "target_indices": None,
        "standard_deviation": 0.3,
        "mean": 0.0,
    }
    with pytest.raises(TypeError):
        corruption.resolved_parameters["mean"] = 2.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        corruption.mean = 2.0  # type: ignore[misc]


def test_parameters_resolve_fields_bias_all_dimensions_and_horizon() -> None:
    source = _action(horizon=6)
    layout = _layout()

    assert AdditiveGaussianNoise(
        standard_deviation=0.2, target_fields=("translation",)
    ).resolved_parameters_for(source, layout) == {
        "target_indices": (1, 3),
        "standard_deviation": 0.2,
        "mean": 0.0,
    }
    assert ConstantBias(
        bias=0.5, target_fields=("translation",)
    ).resolved_parameters_for(source, layout) == {
        "target_indices": (1, 3),
        "bias": (0.5, 0.5),
    }
    assert SegmentHold(start_step=2).resolved_parameters_for(source, layout) == {
        "start_step": 2,
        "end_step": 6,
        "target_indices": (0, 1, 2, 3, 4, 5),
    }
    shift = TemporalFieldShift(
        target_field="gripper",
        shift_steps=1,
        fill_policy=TemporalFillPolicy.ZERO,
    ).resolved_parameters_for(source, layout)
    assert shift["target_indices"] == (5,)
    assert shift["target_field"] == "gripper"
    with pytest.raises(TypeError):
        shift["shift_steps"] = 2  # type: ignore[index]


def test_registry_has_six_names_and_strict_configuration_factories() -> None:
    registry = default_corruption_registry()
    assert registry.names == (
        "additive_gaussian_noise",
        "constant_bias",
        "temporal_field_shift",
        "segment_hold",
        "segment_zeroing",
        "local_temporal_permutation",
    )
    corruption = create_corruption(
        "constant_bias", {"target_indices": [1, 3], "bias": [0.2, -0.4]}
    )
    assert isinstance(corruption, ConstantBias)
    assert corruption.resolved_parameters["bias"] == (0.2, -0.4)
    with pytest.raises(UnknownCorruptionError, match="unknown corruption"):
        create_corruption("cross_episode_swap", {})
    with pytest.raises(CorruptionConfigurationError, match="unknown surprise"):
        create_corruption(
            "constant_bias",
            {"target_indices": [1], "bias": 1.0, "surprise": True},
        )


def test_registry_rejects_duplicate_names_and_mismatched_factory() -> None:
    def factory(_: Mapping[str, object]) -> ActionCorruption:
        return SegmentZeroing(start_step=1, end_step=2)

    registry = CorruptionRegistry()
    registry.register("segment_zeroing", factory)
    with pytest.raises(DuplicateCorruptionError, match="duplicate"):
        registry.register("segment_zeroing", factory)

    mismatch = CorruptionRegistry((("claimed", factory),))
    with pytest.raises(CorruptionConfigurationError, match="mismatched"):
        mismatch.create("claimed", {})


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        (
            "additive_gaussian_noise",
            {"target_indices": [0], "standard_deviation": float("nan")},
        ),
        ("constant_bias", {"target_indices": [0], "bias": float("inf")}),
        (
            "temporal_field_shift",
            {"target_field": "gripper", "shift_steps": 1, "fill_policy": "bad"},
        ),
        ("segment_hold", {"start_step": 2.0}),
        ("segment_zeroing", {"start_step": 1, "end_step": 1}),
        ("local_temporal_permutation", {"start_step": 0, "end_step": 1}),
    ],
)
def test_factory_rejects_invalid_parameters(
    name: str, parameters: dict[str, object]
) -> None:
    with pytest.raises(CorruptionConfigurationError):
        create_corruption(name, parameters)
