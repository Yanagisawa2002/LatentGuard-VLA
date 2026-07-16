"""Training-only standardization for direct action-verifier inputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.action_verifier import DatasetSplit
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.dataset import (
    ACCEPTED_ACTION_DIMENSION,
    ACCEPTED_STATE_DIMENSION,
    AcceptedActionVerifierDatasetV1,
    ActionVerifierModelExampleV1,
)

PREPROCESSING_SCHEMA_VERSION = "1.0"
PREPROCESSING_SEMANTIC = "training_split_standardization_v1"
PREPROCESSING_STATISTICS_DTYPE = np.dtype("<f8")
_MAX_PREPROCESSING_BYTES = 1024 * 1024
_FIELDS = frozenset(
    {
        "schema_version",
        "semantic",
        "dataset_digest",
        "training_split_digest",
        "state_component_count",
        "action_component_count",
        "state_observation_count",
        "valid_action_step_count",
        "minimum_standard_deviation",
        "statistics_dtype",
        "state_mean",
        "state_standard_deviation",
        "action_mean",
        "action_standard_deviation",
        "content_digest",
    }
)


class PreprocessingError(ValueError):
    """Raised when training statistics are malformed, changed, or misbound."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PreprocessingError(f"{context}: {reason}")


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _freeze_float64(
    value: NDArray[Any], *, shape: tuple[int, ...], context: str
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        _fail(context, "expected numpy.ndarray")
    if value.dtype != PREPROCESSING_STATISTICS_DTYPE or value.shape != shape:
        _fail(
            context,
            f"expected dtype {PREPROCESSING_STATISTICS_DTYPE.str} and shape {shape}",
        )
    if not bool(np.all(np.isfinite(value))):
        _fail(context, "statistics must be finite")
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


@dataclass(frozen=True, slots=True, eq=False)
class PreprocessingStateV1:
    """Content-bound train-split means and population standard deviations."""

    dataset_digest: str
    training_split_digest: str
    state_observation_count: int
    valid_action_step_count: int
    minimum_standard_deviation: float
    state_mean: NDArray[Any]
    state_standard_deviation: NDArray[Any]
    action_mean: NDArray[Any]
    action_standard_deviation: NDArray[Any]
    state_component_count: int = ACCEPTED_STATE_DIMENSION
    action_component_count: int = ACCEPTED_ACTION_DIMENSION
    statistics_dtype: str = PREPROCESSING_STATISTICS_DTYPE.str
    semantic: str = PREPROCESSING_SEMANTIC
    schema_version: str = PREPROCESSING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate bindings, counts, shapes, and the standard-deviation floor."""

        if not _valid_digest(self.dataset_digest):
            _fail("PreprocessingStateV1.dataset_digest", "invalid digest")
        if not _valid_digest(self.training_split_digest):
            _fail("PreprocessingStateV1.training_split_digest", "invalid digest")
        if self.schema_version != PREPROCESSING_SCHEMA_VERSION:
            _fail("PreprocessingStateV1.schema_version", "unsupported version")
        if self.semantic != PREPROCESSING_SEMANTIC:
            _fail("PreprocessingStateV1.semantic", "unsupported semantic")
        if self.statistics_dtype != PREPROCESSING_STATISTICS_DTYPE.str:
            _fail("PreprocessingStateV1.statistics_dtype", "expected float64")
        if self.state_component_count != ACCEPTED_STATE_DIMENSION:
            _fail("PreprocessingStateV1.state_component_count", "unexpected dimension")
        if self.action_component_count != ACCEPTED_ACTION_DIMENSION:
            _fail("PreprocessingStateV1.action_component_count", "unexpected dimension")
        for name in ("state_observation_count", "valid_action_step_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                _fail(f"PreprocessingStateV1.{name}", "expected positive int")
        if (
            type(self.minimum_standard_deviation) not in (int, float)
            or not math.isfinite(float(self.minimum_standard_deviation))
            or float(self.minimum_standard_deviation) <= 0.0
        ):
            _fail(
                "PreprocessingStateV1.minimum_standard_deviation",
                "expected finite positive value",
            )
        state_mean = _freeze_float64(
            self.state_mean,
            shape=(self.state_component_count,),
            context="PreprocessingStateV1.state_mean",
        )
        state_std = _freeze_float64(
            self.state_standard_deviation,
            shape=(self.state_component_count,),
            context="PreprocessingStateV1.state_standard_deviation",
        )
        action_mean = _freeze_float64(
            self.action_mean,
            shape=(self.action_component_count,),
            context="PreprocessingStateV1.action_mean",
        )
        action_std = _freeze_float64(
            self.action_standard_deviation,
            shape=(self.action_component_count,),
            context="PreprocessingStateV1.action_standard_deviation",
        )
        floor = float(self.minimum_standard_deviation)
        if bool(np.any(state_std < floor)) or bool(np.any(action_std < floor)):
            _fail("PreprocessingStateV1", "standard deviation is below saved floor")
        object.__setattr__(self, "minimum_standard_deviation", floor)
        object.__setattr__(self, "state_mean", state_mean)
        object.__setattr__(self, "state_standard_deviation", state_std)
        object.__setattr__(self, "action_mean", action_mean)
        object.__setattr__(self, "action_standard_deviation", action_std)

    def _identity_payload(self) -> Mapping[str, object]:
        """Return canonical semantic content excluding its own digest."""

        return {
            "action_component_count": self.action_component_count,
            "action_mean": self.action_mean.tolist(),
            "action_standard_deviation": self.action_standard_deviation.tolist(),
            "dataset_digest": self.dataset_digest,
            "minimum_standard_deviation": self.minimum_standard_deviation,
            "schema_version": self.schema_version,
            "semantic": self.semantic,
            "state_component_count": self.state_component_count,
            "state_mean": self.state_mean.tolist(),
            "state_observation_count": self.state_observation_count,
            "state_standard_deviation": self.state_standard_deviation.tolist(),
            "statistics_dtype": self.statistics_dtype,
            "training_split_digest": self.training_split_digest,
            "valid_action_step_count": self.valid_action_step_count,
        }

    @property
    def content_digest(self) -> str:
        """Return the deterministic digest of bindings and exact statistics."""

        encoded = canonical_json_bytes(
            self._identity_payload(), context="PreprocessingStateV1"
        )
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def as_mapping(self) -> Mapping[str, object]:
        """Return strict JSON-ready content including its verified digest."""

        return {**self._identity_payload(), "content_digest": self.content_digest}


def _fit_preprocessing_statistics(
    examples: Sequence[ActionVerifierModelExampleV1],
    *,
    dataset_digest: str,
    training_split_digest: str,
    minimum_standard_deviation: float = 1e-6,
) -> PreprocessingStateV1:
    """Fit population statistics from an explicitly supplied training set only."""

    values = tuple(examples)
    if not values or any(
        not isinstance(item, ActionVerifierModelExampleV1) for item in values
    ):
        _fail("_fit_preprocessing_statistics.examples", "training examples required")
    if len({item.sample_index for item in values}) != len(values):
        _fail("_fit_preprocessing_statistics.examples", "duplicate sample index")
    if (
        type(minimum_standard_deviation) not in (int, float)
        or not math.isfinite(float(minimum_standard_deviation))
        or float(minimum_standard_deviation) <= 0.0
    ):
        _fail("_fit_preprocessing_statistics.floor", "expected finite positive value")
    state_values = np.stack([item.state_vector for item in values]).astype(
        PREPROCESSING_STATISTICS_DTYPE, copy=False
    )
    valid_actions = [
        item.action_chunk[item.action_mask].astype(
            PREPROCESSING_STATISTICS_DTYPE, copy=False
        )
        for item in values
        if bool(np.any(item.action_mask))
    ]
    if not valid_actions:
        _fail("_fit_preprocessing_statistics.action_mask", "no valid action steps")
    action_values = np.concatenate(valid_actions, axis=0)
    floor = float(minimum_standard_deviation)
    state_std = np.maximum(np.std(state_values, axis=0, ddof=0), floor)
    action_std = np.maximum(np.std(action_values, axis=0, ddof=0), floor)
    return PreprocessingStateV1(
        dataset_digest=dataset_digest,
        training_split_digest=training_split_digest,
        state_observation_count=len(values),
        valid_action_step_count=action_values.shape[0],
        minimum_standard_deviation=floor,
        state_mean=np.mean(state_values, axis=0, dtype=np.float64),
        state_standard_deviation=np.asarray(state_std, dtype=np.float64),
        action_mean=np.mean(action_values, axis=0, dtype=np.float64),
        action_standard_deviation=np.asarray(action_std, dtype=np.float64),
    )


def fit_preprocessing_state(
    dataset: AcceptedActionVerifierDatasetV1,
    *,
    minimum_standard_deviation: float = 1e-6,
) -> PreprocessingStateV1:
    """Fit once from the preserved M3A training split, never validation or test."""

    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        _fail("fit_preprocessing_state.dataset", "expected accepted dataset")
    indices = dataset.indices_for_split(DatasetSplit.TRAIN)
    return _fit_preprocessing_statistics(
        tuple(dataset[index] for index in indices),
        dataset_digest=dataset.dataset_digest,
        training_split_digest=dataset.training_split_digest,
        minimum_standard_deviation=minimum_standard_deviation,
    )


def validate_preprocessing_binding(
    state: PreprocessingStateV1, dataset: AcceptedActionVerifierDatasetV1
) -> None:
    """Reject statistics fitted for a different dataset or training assignment."""

    if not isinstance(state, PreprocessingStateV1):
        _fail("validate_preprocessing_binding.state", "invalid preprocessing state")
    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        _fail("validate_preprocessing_binding.dataset", "invalid dataset")
    if state.dataset_digest != dataset.dataset_digest:
        _fail("validate_preprocessing_binding", "dataset digest changed")
    if state.training_split_digest != dataset.training_split_digest:
        _fail("validate_preprocessing_binding", "training split digest changed")
    expected_train_count = len(dataset.indices_for_split(DatasetSplit.TRAIN))
    if state.state_observation_count != expected_train_count:
        _fail("validate_preprocessing_binding", "training sample count changed")


def normalize_example(
    example: ActionVerifierModelExampleV1, state: PreprocessingStateV1
) -> ActionVerifierModelExampleV1:
    """Apply frozen statistics and zero masked action positions after scaling."""

    if not isinstance(example, ActionVerifierModelExampleV1):
        _fail("normalize_example.example", "invalid example")
    if not isinstance(state, PreprocessingStateV1):
        _fail("normalize_example.state", "invalid preprocessing state")
    normalized_state = (
        (example.state_vector.astype(np.float64) - state.state_mean)
        / state.state_standard_deviation
    ).astype(np.float32)
    normalized_actions = (
        example.action_chunk.astype(np.float64) - state.action_mean
    ) / state.action_standard_deviation
    normalized_actions[~example.action_mask] = 0.0
    converted_actions = normalized_actions.astype(np.float32)
    if not bool(np.all(np.isfinite(normalized_state))) or not bool(
        np.all(np.isfinite(converted_actions))
    ):
        _fail("normalize_example", "normalization produced non-finite values")
    return ActionVerifierModelExampleV1(
        state_vector=normalized_state,
        action_chunk=converted_actions,
        action_mask=example.action_mask,
        failure_target=example.failure_target,
        sample_index=example.sample_index,
    )


def save_preprocessing_state(state: PreprocessingStateV1, path: Path) -> Path:
    """Write one strict JSON preprocessing state to an absent destination."""

    if not isinstance(state, PreprocessingStateV1):
        _fail("save_preprocessing_state.state", "invalid preprocessing state")
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        _fail("save_preprocessing_state.path", "destination must be absent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        state.as_mapping(),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
    except OSError as exc:
        raise PreprocessingError(f"save_preprocessing_state: {exc}") from exc
    return destination


def _reject_constant(value: str) -> NoReturn:
    _fail("PreprocessingStateV1", f"non-finite JSON constant {value!r}")


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("PreprocessingStateV1", f"duplicate field {key!r}")
        result[key] = value
    return result


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected object")
    return cast(dict[str, object], value)


def _field(value: Mapping[str, object], name: str) -> object:
    if name not in value:
        _fail("PreprocessingStateV1", f"missing field {name!r}")
    return value[name]


def _text(value: Mapping[str, object], name: str) -> str:
    item = _field(value, name)
    if not isinstance(item, str):
        _fail(f"PreprocessingStateV1.{name}", "expected text")
    return item


def _integer(value: Mapping[str, object], name: str) -> int:
    item = _field(value, name)
    if type(item) is not int:
        _fail(f"PreprocessingStateV1.{name}", "expected int")
    return item


def _number(value: Mapping[str, object], name: str) -> float:
    item = _field(value, name)
    if type(item) not in (int, float):
        _fail(f"PreprocessingStateV1.{name}", "expected number")
    number = float(cast(int | float, item))
    if not math.isfinite(number):
        _fail(f"PreprocessingStateV1.{name}", "expected finite number")
    return number


def _float_array(value: Mapping[str, object], name: str, size: int) -> NDArray[Any]:
    item = _field(value, name)
    if not isinstance(item, list) or len(item) != size:
        _fail(f"PreprocessingStateV1.{name}", f"expected {size} values")
    decoded = np.asarray(
        [
            _number({"value": component}, "value")
            for component in cast(list[object], item)
        ],
        dtype=np.float64,
    )
    return decoded


def load_preprocessing_state(path: Path) -> PreprocessingStateV1:
    """Load strict JSON, reconstruct statistics, and reject content tampering."""

    source = Path(path).absolute()
    try:
        if (
            source.is_symlink()
            or not source.is_file()
            or source.resolve() != source
            or source.stat().st_nlink != 1
        ):
            _fail("load_preprocessing_state.path", "expected unlinked regular file")
        if source.stat().st_size > _MAX_PREPROCESSING_BYTES:
            _fail("load_preprocessing_state.path", "file is unexpectedly large")
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except PreprocessingError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreprocessingError(f"load_preprocessing_state: {exc}") from exc
    value = _mapping(raw, "PreprocessingStateV1")
    if set(value) != set(_FIELDS):
        _fail("PreprocessingStateV1", "fields differ from schema")
    state_components = _integer(value, "state_component_count")
    action_components = _integer(value, "action_component_count")
    state = PreprocessingStateV1(
        dataset_digest=_text(value, "dataset_digest"),
        training_split_digest=_text(value, "training_split_digest"),
        state_observation_count=_integer(value, "state_observation_count"),
        valid_action_step_count=_integer(value, "valid_action_step_count"),
        minimum_standard_deviation=_number(value, "minimum_standard_deviation"),
        state_mean=_float_array(value, "state_mean", state_components),
        state_standard_deviation=_float_array(
            value, "state_standard_deviation", state_components
        ),
        action_mean=_float_array(value, "action_mean", action_components),
        action_standard_deviation=_float_array(
            value, "action_standard_deviation", action_components
        ),
        state_component_count=state_components,
        action_component_count=action_components,
        statistics_dtype=_text(value, "statistics_dtype"),
        semantic=_text(value, "semantic"),
        schema_version=_text(value, "schema_version"),
    )
    if state.content_digest != _text(value, "content_digest"):
        _fail("PreprocessingStateV1.content_digest", "content changed")
    return state


__all__ = [
    "PREPROCESSING_SCHEMA_VERSION",
    "PREPROCESSING_SEMANTIC",
    "PreprocessingError",
    "PreprocessingStateV1",
    "fit_preprocessing_state",
    "load_preprocessing_state",
    "normalize_example",
    "save_preprocessing_state",
    "validate_preprocessing_binding",
]
