"""Deterministic single-source action transformations for M1."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np

from latentguard.corruptions.base import (
    ActionCorruption,
    BaseActionCorruption,
    CorruptionConfigurationError,
)
from latentguard.corruptions.layout import ActionLayout, ActionLayoutError
from latentguard.corruptions.models import ResolvedParameterValue
from latentguard.models import ActionChunk

TargetNames: TypeAlias = tuple[str, ...] | None
TargetIndices: TypeAlias = tuple[int, ...] | None
FiniteNumber: TypeAlias = int | float
BiasValue: TypeAlias = FiniteNumber | tuple[FiniteNumber, ...]
BuiltinFactory: TypeAlias = Callable[[Mapping[str, object]], ActionCorruption]


class TemporalFillPolicy(StrEnum):
    """Fill behavior for values exposed by a temporal shift."""

    EDGE = "edge"
    ZERO = "zero"


def _finite_number(value: object, context: str) -> FiniteNumber:
    if type(value) not in (int, float):
        raise CorruptionConfigurationError(f"{context}: must be a finite number")
    number = cast(FiniteNumber, value)
    if isinstance(number, float) and not math.isfinite(number):
        raise CorruptionConfigurationError(f"{context}: must be a finite number")
    return number


def _finite_float(value: object, context: str) -> float:
    number = _finite_number(value, context)
    try:
        converted = float(number)
    except OverflowError as exc:
        raise CorruptionConfigurationError(
            f"{context}: is outside the finite floating-point range"
        ) from exc
    if not math.isfinite(converted):
        raise CorruptionConfigurationError(f"{context}: must be a finite number")
    if isinstance(number, int) and int(converted) != number:
        raise CorruptionConfigurationError(
            f"{context}: integer cannot be represented exactly as a float"
        )
    return converted


def _integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise CorruptionConfigurationError(f"{context}: must be an integer")
    return value


def _optional_names(value: object, context: str) -> TargetNames:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise CorruptionConfigurationError(f"{context}: must be an array of names")
    names = tuple(value)
    if not names:
        raise CorruptionConfigurationError(f"{context}: must not be empty")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise CorruptionConfigurationError(
            f"{context}: all field names must be non-empty strings"
        )
    typed_names = cast(tuple[str, ...], names)
    if len(set(typed_names)) != len(typed_names):
        raise CorruptionConfigurationError(
            f"{context}: duplicate field names are unsupported"
        )
    return typed_names


def _optional_indices(value: object, context: str) -> TargetIndices:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise CorruptionConfigurationError(f"{context}: must be an array of indices")
    indices = tuple(value)
    if not indices:
        raise CorruptionConfigurationError(f"{context}: must not be empty")
    if not all(type(index) is int for index in indices):
        raise CorruptionConfigurationError(f"{context}: all indices must be integers")
    typed_indices = cast(tuple[int, ...], indices)
    if any(index < 0 for index in typed_indices):
        raise CorruptionConfigurationError(f"{context}: indices must be non-negative")
    if len(set(typed_indices)) != len(typed_indices):
        raise CorruptionConfigurationError(
            f"{context}: duplicate indices are unsupported"
        )
    return typed_indices


def _validate_targets(
    fields: TargetNames,
    indices: TargetIndices,
    context: str,
    *,
    allow_all: bool,
) -> None:
    if fields is not None and indices is not None:
        raise CorruptionConfigurationError(
            f"{context}: target_fields and target_indices are ambiguous"
        )
    if not allow_all and fields is None and indices is None:
        raise CorruptionConfigurationError(
            f"{context}: provide target_fields or target_indices"
        )


def _parameters(
    values: Mapping[str, ResolvedParameterValue],
) -> Mapping[str, ResolvedParameterValue]:
    return MappingProxyType(dict(values))


def _resolve(
    corruption: BaseActionCorruption,
    action: ActionChunk,
    layout: ActionLayout,
    fields: TargetNames,
    indices: TargetIndices,
    *,
    allow_all: bool,
) -> tuple[int, ...]:
    corruption._validate_common(action, layout)
    try:
        return layout.resolve_selection(
            target_fields=fields,
            target_indices=indices,
            allow_all=allow_all,
        )
    except ActionLayoutError as exc:
        corruption._not_applicable(str(exc), action, layout)


def _require_float_action(
    corruption: BaseActionCorruption, action: ActionChunk, layout: ActionLayout
) -> None:
    if not np.issubdtype(action.actions.dtype, np.floating):
        corruption._not_applicable(
            "additive arithmetic requires a floating-point action dtype; "
            "integer casting would silently alter values",
            action,
            layout,
        )


def _validate_bias_dtype(
    corruption: ConstantBias,
    action: ActionChunk,
    layout: ActionLayout,
    selected: tuple[int, ...],
) -> None:
    if np.issubdtype(action.actions.dtype, np.floating):
        bias_values = (
            corruption.bias
            if isinstance(corruption.bias, tuple)
            else (corruption.bias,) * len(selected)
        )
        float_limits = np.finfo(action.actions.dtype)
        for value in bias_values:
            try:
                numeric = float(value)
            except OverflowError as exc:
                corruption._not_applicable(
                    f"bias cannot be represented by action dtype: {exc}",
                    action,
                    layout,
                )
            if not math.isfinite(numeric) or abs(numeric) > float_limits.max:
                corruption._not_applicable(
                    "bias is outside the finite action-dtype range",
                    action,
                    layout,
                )
            try:
                encoded = np.asarray(value, dtype=action.actions.dtype).item()
            except (OverflowError, TypeError, ValueError) as exc:
                corruption._not_applicable(
                    f"bias cannot be represented by action dtype: {exc}",
                    action,
                    layout,
                )
            if isinstance(value, int) and int(encoded) != value:
                corruption._not_applicable(
                    "integer bias cannot be represented exactly by action dtype",
                    action,
                    layout,
                )
        return
    if not np.issubdtype(action.actions.dtype, np.integer):
        corruption._not_applicable(
            "constant bias requires an integer or floating-point action dtype",
            action,
            layout,
        )
    bias_values = (
        corruption.bias
        if isinstance(corruption.bias, tuple)
        else (corruption.bias,) * len(selected)
    )
    if any(
        isinstance(value, float) and not value.is_integer() for value in bias_values
    ):
        corruption._not_applicable(
            "integer action dtype requires exactly integral bias values",
            action,
            layout,
        )
    integer_limits = np.iinfo(action.actions.dtype)
    for dimension, bias in zip(selected, bias_values, strict=True):
        delta = int(bias)
        for value in action.actions[:, dimension]:
            result = int(value) + delta
            if result < integer_limits.min or result > integer_limits.max:
                corruption._not_applicable(
                    "integer bias result is outside dtype range; "
                    "clipping is prohibited",
                    action,
                    layout,
                )


def _apply_bias(
    output: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    selected: tuple[int, ...],
    bias: BiasValue,
) -> None:
    if np.issubdtype(output.dtype, np.integer):
        bias_values = bias if isinstance(bias, tuple) else (bias,) * len(selected)
        for dimension, value in zip(selected, bias_values, strict=True):
            output[:, dimension] = np.asarray(
                [int(item) + int(value) for item in output[:, dimension]],
                dtype=output.dtype,
            )
        return
    with np.errstate(over="ignore", invalid="ignore"):
        output[:, selected] = np.add(
            output[:, selected], np.asarray(bias, dtype=output.dtype)
        )


def _ensure_finite(
    corruption: BaseActionCorruption,
    actions: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    source: ActionChunk,
    layout: ActionLayout,
) -> None:
    if not np.all(np.isfinite(actions)):
        corruption._not_applicable(
            "transformation produced NaN or infinity; clipping is prohibited",
            source,
            layout,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AdditiveGaussianNoise(BaseActionCorruption):
    """Add seeded Gaussian noise to explicitly selected action dimensions."""

    standard_deviation: float
    target_fields: TargetNames = None
    target_indices: TargetIndices = None
    mean: float = 0.0

    def __post_init__(self) -> None:
        """Canonicalize and validate the immutable configuration."""
        object.__setattr__(
            self,
            "target_fields",
            _optional_names(self.target_fields, f"{self.name}.target_fields"),
        )
        object.__setattr__(
            self,
            "target_indices",
            _optional_indices(self.target_indices, f"{self.name}.target_indices"),
        )
        object.__setattr__(
            self,
            "standard_deviation",
            _finite_float(self.standard_deviation, f"{self.name}.standard_deviation"),
        )
        object.__setattr__(self, "mean", _finite_float(self.mean, f"{self.name}.mean"))
        if self.standard_deviation <= 0.0:
            raise CorruptionConfigurationError(
                f"{self.name}.standard_deviation: must be greater than zero"
            )
        _validate_targets(
            self.target_fields, self.target_indices, self.name, allow_all=False
        )

    @property
    def name(self) -> str:
        """Return the stable corruption name."""
        return "additive_gaussian_noise"

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return canonical immutable Gaussian parameters."""
        return _parameters(
            {
                "target_fields": self.target_fields,
                "target_indices": self.target_indices,
                "standard_deviation": self.standard_deviation,
                "mean": self.mean,
            }
        )

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve the configured selection to exact action indices."""
        self.validate_for(action, layout)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=False,
        )
        return _parameters(
            {
                "target_indices": selected,
                "standard_deviation": self.standard_deviation,
                "mean": self.mean,
            }
        )

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Validate selection, action/layout dimensions, and floating dtype."""
        _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=False,
        )
        _require_float_action(self, action, layout)

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Apply seeded Gaussian noise without clipping or touching other dimensions."""
        self.validate_for(action, layout)
        self._validate_seed(seed)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=False,
        )
        output = np.array(action.actions, copy=True, order="C")
        noise = np.random.default_rng(seed).normal(
            loc=self.mean,
            scale=self.standard_deviation,
            size=(output.shape[0], len(selected)),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            output[:, selected] = output[:, selected] + noise
        _ensure_finite(self, output, action, layout)
        return self._result(action, output)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstantBias(BaseActionCorruption):
    """Add a scalar or per-selected-dimension constant bias."""

    bias: BiasValue
    target_fields: TargetNames = None
    target_indices: TargetIndices = None

    def __post_init__(self) -> None:
        """Canonicalize and validate the immutable configuration."""
        object.__setattr__(
            self,
            "target_fields",
            _optional_names(self.target_fields, f"{self.name}.target_fields"),
        )
        object.__setattr__(
            self,
            "target_indices",
            _optional_indices(self.target_indices, f"{self.name}.target_indices"),
        )
        if isinstance(self.bias, (list, tuple)):
            if not self.bias:
                raise CorruptionConfigurationError(
                    f"{self.name}.bias: must not be empty"
                )
            bias: BiasValue = tuple(
                _finite_number(value, f"{self.name}.bias") for value in self.bias
            )
        else:
            bias = _finite_number(self.bias, f"{self.name}.bias")
        object.__setattr__(self, "bias", bias)
        _validate_targets(
            self.target_fields, self.target_indices, self.name, allow_all=False
        )

    @property
    def name(self) -> str:
        """Return the stable corruption name."""
        return "constant_bias"

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return canonical immutable bias parameters."""
        return _parameters(
            {
                "target_fields": self.target_fields,
                "target_indices": self.target_indices,
                "bias": self.bias,
            }
        )

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve selection and expand scalar bias to every selected dimension."""
        self.validate_for(action, layout)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=False,
        )
        bias = (
            self.bias if isinstance(self.bias, tuple) else (self.bias,) * len(selected)
        )
        return _parameters({"target_indices": selected, "bias": bias})

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Validate selection, floating dtype, and per-dimension bias length."""
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=False,
        )
        if isinstance(self.bias, tuple) and len(self.bias) != len(selected):
            self._not_applicable(
                f"per-dimension bias length {len(self.bias)} does not match "
                f"selected dimension count {len(selected)}",
                action,
                layout,
            )
        _validate_bias_dtype(self, action, layout, selected)

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Apply constant bias without clipping or touching other dimensions."""
        self.validate_for(action, layout)
        self._validate_seed(seed)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=False,
        )
        output = np.array(action.actions, copy=True, order="C")
        _apply_bias(output, selected, self.bias)
        _ensure_finite(self, output, action, layout)
        return self._result(action, output)


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalFieldShift(BaseActionCorruption):
    """Shift one declared field in time using an explicit fill policy."""

    target_field: str
    shift_steps: int
    fill_policy: TemporalFillPolicy

    def __post_init__(self) -> None:
        """Canonicalize and validate shift configuration."""
        if not isinstance(self.target_field, str) or not self.target_field.strip():
            raise CorruptionConfigurationError(
                f"{self.name}.target_field: must be a non-empty string"
            )
        object.__setattr__(
            self,
            "shift_steps",
            _integer(self.shift_steps, f"{self.name}.shift_steps"),
        )
        if self.shift_steps == 0:
            raise CorruptionConfigurationError(
                f"{self.name}.shift_steps: zero shift is a no-op and is unsupported"
            )
        try:
            policy = TemporalFillPolicy(self.fill_policy)
        except (TypeError, ValueError) as exc:
            raise CorruptionConfigurationError(
                f"{self.name}.fill_policy: expected edge or zero"
            ) from exc
        object.__setattr__(self, "fill_policy", policy)

    @property
    def name(self) -> str:
        """Return the stable corruption name."""
        return "temporal_field_shift"

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return canonical immutable temporal-shift parameters."""
        return _parameters(
            {
                "target_field": self.target_field,
                "shift_steps": self.shift_steps,
                "fill_policy": self.fill_policy.value,
            }
        )

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve the target field to its exact declared indices."""
        self.validate_for(action, layout)
        selected = _resolve(
            self, action, layout, (self.target_field,), None, allow_all=False
        )
        return _parameters(
            {
                "target_field": self.target_field,
                "target_indices": selected,
                "shift_steps": self.shift_steps,
                "fill_policy": self.fill_policy.value,
            }
        )

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Validate the selected field and horizon-relative shift magnitude."""
        _resolve(self, action, layout, (self.target_field,), None, allow_all=False)
        if abs(self.shift_steps) >= action.actions.shape[0]:
            self._not_applicable(
                f"absolute shift {abs(self.shift_steps)} must be smaller than "
                f"horizon {action.actions.shape[0]}",
                action,
                layout,
            )

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Shift the field while preserving the horizon and all other dimensions."""
        self.validate_for(action, layout)
        self._validate_seed(seed)
        selected = _resolve(
            self, action, layout, (self.target_field,), None, allow_all=False
        )
        output = np.array(action.actions, copy=True, order="C")
        shift = self.shift_steps
        if shift > 0:
            output[shift:, selected] = action.actions[:-shift, selected]
            if self.fill_policy is TemporalFillPolicy.EDGE:
                output[:shift, selected] = action.actions[0, selected]
            else:
                output[:shift, selected] = 0
        else:
            advance = -shift
            output[:-advance, selected] = action.actions[advance:, selected]
            if self.fill_policy is TemporalFillPolicy.EDGE:
                output[-advance:, selected] = action.actions[-1, selected]
            else:
                output[-advance:, selected] = 0
        return self._result(action, output)


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentHold(BaseActionCorruption):
    """Hold selected dimensions at their value immediately before a segment."""

    start_step: int
    end_step: int | None = None
    target_fields: TargetNames = None
    target_indices: TargetIndices = None

    def __post_init__(self) -> None:
        """Canonicalize and validate segment configuration."""
        object.__setattr__(
            self, "start_step", _integer(self.start_step, f"{self.name}.start_step")
        )
        if self.start_step < 1:
            raise CorruptionConfigurationError(
                f"{self.name}.start_step: must be at least 1 to have a preceding value"
            )
        if self.end_step is not None:
            object.__setattr__(
                self, "end_step", _integer(self.end_step, f"{self.name}.end_step")
            )
            if self.end_step <= self.start_step:
                raise CorruptionConfigurationError(
                    f"{self.name}.end_step: must be greater than start_step"
                )
        object.__setattr__(
            self,
            "target_fields",
            _optional_names(self.target_fields, f"{self.name}.target_fields"),
        )
        object.__setattr__(
            self,
            "target_indices",
            _optional_indices(self.target_indices, f"{self.name}.target_indices"),
        )
        _validate_targets(
            self.target_fields, self.target_indices, self.name, allow_all=True
        )

    @property
    def name(self) -> str:
        """Return the stable corruption name."""
        return "segment_hold"

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return canonical immutable segment-hold parameters."""
        return _parameters(
            {
                "start_step": self.start_step,
                "end_step": self.end_step,
                "target_fields": self.target_fields,
                "target_indices": self.target_indices,
            }
        )

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve selection and an omitted end step against the action horizon."""
        self.validate_for(action, layout)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        end = action.actions.shape[0] if self.end_step is None else self.end_step
        return _parameters(
            {
                "start_step": self.start_step,
                "end_step": end,
                "target_indices": selected,
            }
        )

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Validate selection and segment bounds for the source horizon."""
        _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        horizon = action.actions.shape[0]
        end = horizon if self.end_step is None else self.end_step
        if self.start_step >= horizon or end > horizon:
            self._not_applicable(
                f"segment [{self.start_step}, {end}) outside horizon {horizon}",
                action,
                layout,
            )

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Hold the selected segment without shortening the action horizon."""
        self.validate_for(action, layout)
        self._validate_seed(seed)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        end = action.actions.shape[0] if self.end_step is None else self.end_step
        output = np.array(action.actions, copy=True, order="C")
        output[self.start_step : end, selected] = action.actions[
            self.start_step - 1, selected
        ]
        return self._result(action, output)


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentZeroing(BaseActionCorruption):
    """Set selected dimensions to zero inside a validated non-empty segment."""

    start_step: int
    end_step: int
    target_fields: TargetNames = None
    target_indices: TargetIndices = None

    def __post_init__(self) -> None:
        """Canonicalize and validate segment configuration."""
        object.__setattr__(
            self, "start_step", _integer(self.start_step, f"{self.name}.start_step")
        )
        object.__setattr__(
            self, "end_step", _integer(self.end_step, f"{self.name}.end_step")
        )
        if self.start_step < 0:
            raise CorruptionConfigurationError(
                f"{self.name}.start_step: must be non-negative"
            )
        if self.end_step <= self.start_step:
            raise CorruptionConfigurationError(
                f"{self.name}.end_step: must be greater than start_step"
            )
        object.__setattr__(
            self,
            "target_fields",
            _optional_names(self.target_fields, f"{self.name}.target_fields"),
        )
        object.__setattr__(
            self,
            "target_indices",
            _optional_indices(self.target_indices, f"{self.name}.target_indices"),
        )
        _validate_targets(
            self.target_fields, self.target_indices, self.name, allow_all=True
        )

    @property
    def name(self) -> str:
        """Return the stable corruption name."""
        return "segment_zeroing"

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return canonical immutable segment-zeroing parameters."""
        return _parameters(
            {
                "start_step": self.start_step,
                "end_step": self.end_step,
                "target_fields": self.target_fields,
                "target_indices": self.target_indices,
            }
        )

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve the selected fields or all dimensions to exact indices."""
        self.validate_for(action, layout)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        return _parameters(
            {
                "start_step": self.start_step,
                "end_step": self.end_step,
                "target_indices": selected,
            }
        )

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Validate selection and segment bounds for the source horizon."""
        _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        horizon = action.actions.shape[0]
        if self.start_step >= horizon or self.end_step > horizon:
            self._not_applicable(
                f"segment [{self.start_step}, {self.end_step}) outside "
                f"horizon {horizon}",
                action,
                layout,
            )

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Zero the selected segment while preserving shape and other dimensions."""
        self.validate_for(action, layout)
        self._validate_seed(seed)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        output = np.array(action.actions, copy=True, order="C")
        output[self.start_step : self.end_step, selected] = 0
        return self._result(action, output)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTemporalPermutation(BaseActionCorruption):
    """Seed-permute selected dimensions within one local time segment."""

    start_step: int
    end_step: int
    target_fields: TargetNames = None
    target_indices: TargetIndices = None

    def __post_init__(self) -> None:
        """Canonicalize and validate segment configuration."""
        object.__setattr__(
            self, "start_step", _integer(self.start_step, f"{self.name}.start_step")
        )
        object.__setattr__(
            self, "end_step", _integer(self.end_step, f"{self.name}.end_step")
        )
        if self.start_step < 0:
            raise CorruptionConfigurationError(
                f"{self.name}.start_step: must be non-negative"
            )
        if self.end_step - self.start_step < 2:
            raise CorruptionConfigurationError(
                f"{self.name}: segment must contain at least two control steps"
            )
        object.__setattr__(
            self,
            "target_fields",
            _optional_names(self.target_fields, f"{self.name}.target_fields"),
        )
        object.__setattr__(
            self,
            "target_indices",
            _optional_indices(self.target_indices, f"{self.name}.target_indices"),
        )
        _validate_targets(
            self.target_fields, self.target_indices, self.name, allow_all=True
        )

    @property
    def name(self) -> str:
        """Return the stable corruption name."""
        return "local_temporal_permutation"

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return canonical immutable local-permutation parameters."""
        return _parameters(
            {
                "start_step": self.start_step,
                "end_step": self.end_step,
                "target_fields": self.target_fields,
                "target_indices": self.target_indices,
            }
        )

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve the selected fields or all dimensions to exact indices."""
        self.validate_for(action, layout)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        return _parameters(
            {
                "start_step": self.start_step,
                "end_step": self.end_step,
                "target_indices": selected,
            }
        )

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Validate selection and local segment bounds for the source horizon."""
        _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        horizon = action.actions.shape[0]
        if self.start_step >= horizon or self.end_step > horizon:
            self._not_applicable(
                f"segment [{self.start_step}, {self.end_step}) outside "
                f"horizon {horizon}",
                action,
                layout,
            )

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Permute selected rows locally while preserving their exact multiset."""
        self.validate_for(action, layout)
        self._validate_seed(seed)
        selected = _resolve(
            self,
            action,
            layout,
            self.target_fields,
            self.target_indices,
            allow_all=True,
        )
        size = self.end_step - self.start_step
        permutation = np.random.default_rng(seed).permutation(size)
        if np.array_equal(permutation, np.arange(size)):
            permutation = np.roll(permutation, 1)
        source_segment = action.actions[self.start_step : self.end_step]
        output = np.array(action.actions, copy=True, order="C")
        output[self.start_step : self.end_step, selected] = source_segment[permutation][
            :, selected
        ]
        return self._result(action, output)


_WINDOW_PARAMETER_NAMES = frozenset({"severity_id", "window_end", "window_start"})


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowScopedCorruption(BaseActionCorruption):
    """Apply one existing corruption only inside an explicit action-step window."""

    inner: ActionCorruption
    window_start: int
    window_end: int
    severity_id: str

    def __post_init__(self) -> None:
        """Validate the immutable wrapper without changing the inner semantic."""
        if not isinstance(self.inner, ActionCorruption):
            raise CorruptionConfigurationError(
                "window_scoped_corruption.inner: expected ActionCorruption"
            )
        object.__setattr__(
            self,
            "window_start",
            _integer(self.window_start, "window_scoped_corruption.window_start"),
        )
        object.__setattr__(
            self,
            "window_end",
            _integer(self.window_end, "window_scoped_corruption.window_end"),
        )
        if self.window_start < 0:
            raise CorruptionConfigurationError(
                "window_scoped_corruption.window_start: must be non-negative"
            )
        if self.window_end <= self.window_start:
            raise CorruptionConfigurationError(
                "window_scoped_corruption.window_end: must be greater than window_start"
            )
        if (
            not isinstance(self.severity_id, str)
            or not self.severity_id
            or self.severity_id != self.severity_id.strip()
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.severity_id
            )
        ):
            raise CorruptionConfigurationError(
                "window_scoped_corruption.severity_id: expected a non-empty "
                "canonical string"
            )
        conflicts = _WINDOW_PARAMETER_NAMES.intersection(self.inner.resolved_parameters)
        if conflicts:
            raise CorruptionConfigurationError(
                "window_scoped_corruption.inner: reserved resolved parameter names "
                + ", ".join(sorted(conflicts))
            )

    @property
    def name(self) -> str:
        """Preserve the wrapped transformation's stable registry name."""
        return self.inner.name

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Return unresolved inner parameters plus the explicit window contract."""
        return self._with_window(self.inner.resolved_parameters)

    def resolved_parameters_for(
        self, action: ActionChunk, layout: ActionLayout
    ) -> Mapping[str, ResolvedParameterValue]:
        """Resolve the wrapped transformation against only the declared window."""
        window = self._window_action(action, layout)
        return self._with_window(self.inner.resolved_parameters_for(window, layout))

    def validate_for(self, action: ActionChunk, layout: ActionLayout) -> None:
        """Require a complete in-bounds window and applicable inner transform."""
        window = self._window_action(action, layout)
        self.inner.validate_for(window, layout)

    def apply(
        self, action: ActionChunk, layout: ActionLayout, *, seed: int
    ) -> ActionChunk:
        """Splice a transformed window into an otherwise byte-identical action."""
        window = self._window_action(action, layout)
        self._validate_seed(seed)
        transformed = self.inner.apply(window, layout, seed=seed)
        if (
            transformed.actions.shape != window.actions.shape
            or transformed.actions.dtype != window.actions.dtype
            or transformed.coordinate_frame != window.coordinate_frame
            or transformed.control_period_s != window.control_period_s
            or transformed.schema_version != window.schema_version
        ):
            self._not_applicable(
                "wrapped corruption changed the window action contract",
                action,
                layout,
            )
        output = np.array(action.actions, copy=True, order="C")
        output[self.window_start : self.window_end] = transformed.actions
        return self._result(action, output)

    def _window_action(self, action: ActionChunk, layout: ActionLayout) -> ActionChunk:
        self._validate_common(action, layout)
        horizon = action.actions.shape[0]
        if self.window_end > horizon:
            self._not_applicable(
                f"window [{self.window_start}, {self.window_end}) outside horizon "
                f"{horizon}",
                action,
                layout,
            )
        return ActionChunk(
            actions=action.actions[self.window_start : self.window_end],
            coordinate_frame=action.coordinate_frame,
            control_period_s=action.control_period_s,
            schema_version=action.schema_version,
        )

    def _with_window(
        self, parameters: Mapping[str, ResolvedParameterValue]
    ) -> Mapping[str, ResolvedParameterValue]:
        conflicts = _WINDOW_PARAMETER_NAMES.intersection(parameters)
        if conflicts:
            raise CorruptionConfigurationError(
                "window_scoped_corruption.inner: reserved resolved parameter names "
                + ", ".join(sorted(conflicts))
            )
        return _parameters(
            {
                **parameters,
                "window_start": self.window_start,
                "window_end": self.window_end,
                "severity_id": self.severity_id,
            }
        )


def _expect_fields(
    parameters: Mapping[str, object],
    corruption_name: str,
    required: set[str],
    optional: set[str],
) -> None:
    if not isinstance(parameters, Mapping) or not all(
        isinstance(key, str) for key in parameters
    ):
        raise CorruptionConfigurationError(
            f"{corruption_name}.parameters: expected an object with string keys"
        )
    actual = set(parameters)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise CorruptionConfigurationError(
            f"{corruption_name}.parameters: invalid fields ({'; '.join(details)})"
        )


def _targets_from_parameters(
    parameters: Mapping[str, object], corruption_name: str
) -> tuple[TargetNames, TargetIndices]:
    return (
        _optional_names(
            parameters.get("target_fields"), f"{corruption_name}.target_fields"
        ),
        _optional_indices(
            parameters.get("target_indices"), f"{corruption_name}.target_indices"
        ),
    )


def _create_gaussian(parameters: Mapping[str, object]) -> ActionCorruption:
    name = "additive_gaussian_noise"
    _expect_fields(
        parameters,
        name,
        {"standard_deviation"},
        {"mean", "target_fields", "target_indices"},
    )
    fields, indices = _targets_from_parameters(parameters, name)
    return AdditiveGaussianNoise(
        standard_deviation=_finite_float(
            parameters["standard_deviation"], f"{name}.standard_deviation"
        ),
        mean=_finite_float(parameters.get("mean", 0.0), f"{name}.mean"),
        target_fields=fields,
        target_indices=indices,
    )


def _create_bias(parameters: Mapping[str, object]) -> ActionCorruption:
    name = "constant_bias"
    _expect_fields(parameters, name, {"bias"}, {"target_fields", "target_indices"})
    fields, indices = _targets_from_parameters(parameters, name)
    raw_bias = parameters["bias"]
    if isinstance(raw_bias, (list, tuple)):
        bias: BiasValue = tuple(
            _finite_number(value, f"{name}.bias") for value in raw_bias
        )
    else:
        bias = _finite_number(raw_bias, f"{name}.bias")
    return ConstantBias(bias=bias, target_fields=fields, target_indices=indices)


def _create_shift(parameters: Mapping[str, object]) -> ActionCorruption:
    name = "temporal_field_shift"
    _expect_fields(
        parameters,
        name,
        {"target_field", "shift_steps", "fill_policy"},
        set(),
    )
    return TemporalFieldShift(
        target_field=cast(str, parameters["target_field"]),
        shift_steps=_integer(parameters["shift_steps"], f"{name}.shift_steps"),
        fill_policy=cast(TemporalFillPolicy, parameters["fill_policy"]),
    )


def _create_hold(parameters: Mapping[str, object]) -> ActionCorruption:
    name = "segment_hold"
    _expect_fields(
        parameters,
        name,
        {"start_step"},
        {"end_step", "target_fields", "target_indices"},
    )
    fields, indices = _targets_from_parameters(parameters, name)
    raw_end = parameters.get("end_step")
    return SegmentHold(
        start_step=_integer(parameters["start_step"], f"{name}.start_step"),
        end_step=(None if raw_end is None else _integer(raw_end, f"{name}.end_step")),
        target_fields=fields,
        target_indices=indices,
    )


def _create_zeroing(parameters: Mapping[str, object]) -> ActionCorruption:
    name = "segment_zeroing"
    _expect_fields(
        parameters,
        name,
        {"start_step", "end_step"},
        {"target_fields", "target_indices"},
    )
    fields, indices = _targets_from_parameters(parameters, name)
    return SegmentZeroing(
        start_step=_integer(parameters["start_step"], f"{name}.start_step"),
        end_step=_integer(parameters["end_step"], f"{name}.end_step"),
        target_fields=fields,
        target_indices=indices,
    )


def _create_permutation(parameters: Mapping[str, object]) -> ActionCorruption:
    name = "local_temporal_permutation"
    _expect_fields(
        parameters,
        name,
        {"start_step", "end_step"},
        {"target_fields", "target_indices"},
    )
    fields, indices = _targets_from_parameters(parameters, name)
    return LocalTemporalPermutation(
        start_step=_integer(parameters["start_step"], f"{name}.start_step"),
        end_step=_integer(parameters["end_step"], f"{name}.end_step"),
        target_fields=fields,
        target_indices=indices,
    )


BUILTIN_CORRUPTION_FACTORIES: Mapping[str, BuiltinFactory] = MappingProxyType(
    {
        "additive_gaussian_noise": _create_gaussian,
        "constant_bias": _create_bias,
        "temporal_field_shift": _create_shift,
        "segment_hold": _create_hold,
        "segment_zeroing": _create_zeroing,
        "local_temporal_permutation": _create_permutation,
    }
)
"""Strict factories for all M1 single-source transformations."""
