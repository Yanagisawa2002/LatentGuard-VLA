"""Deterministic NumPy batching with an optional lazy PyTorch conversion."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.training.dataset import (
    ACCEPTED_ACTION_DIMENSION,
    ACCEPTED_ACTION_HORIZON,
    ACCEPTED_STATE_DIMENSION,
    AcceptedActionVerifierDatasetV1,
    ActionVerifierModelExampleV1,
)
from latentguard.training.preprocessing import PreprocessingStateV1, normalize_example


class BatchingError(ValueError):
    """Raised when a batch plan or projected example is malformed."""


class TorchTrainingDependencyError(RuntimeError):
    """Raised when an explicitly requested lazy PyTorch operation is unavailable."""


def _freeze(value: NDArray[Any]) -> NDArray[Any]:
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


@dataclass(frozen=True, slots=True, eq=False)
class NumpyVerifierBatchV1:
    """One immutable metadata-free model batch plus targets and join indices."""

    state_vectors: NDArray[Any]
    action_chunks: NDArray[Any]
    action_masks: NDArray[Any]
    failure_targets: NDArray[Any]
    sample_indices: NDArray[Any]

    def __post_init__(self) -> None:
        """Require the fixed model contract and detach all batch arrays."""

        arrays = (
            self.state_vectors,
            self.action_chunks,
            self.action_masks,
            self.failure_targets,
            self.sample_indices,
        )
        if any(not isinstance(value, np.ndarray) for value in arrays):
            raise BatchingError("NumpyVerifierBatchV1: all fields must be arrays")
        batch_size = self.state_vectors.shape[0] if self.state_vectors.ndim == 2 else 0
        if batch_size <= 0:
            raise BatchingError("NumpyVerifierBatchV1: empty batches are invalid")
        contracts = (
            (
                self.state_vectors,
                np.dtype("<f4"),
                (batch_size, ACCEPTED_STATE_DIMENSION),
            ),
            (
                self.action_chunks,
                np.dtype("<f4"),
                (batch_size, ACCEPTED_ACTION_HORIZON, ACCEPTED_ACTION_DIMENSION),
            ),
            (
                self.action_masks,
                np.dtype(np.bool_),
                (batch_size, ACCEPTED_ACTION_HORIZON),
            ),
            (self.failure_targets, np.dtype("<f4"), (batch_size,)),
            (self.sample_indices, np.dtype("<i8"), (batch_size,)),
        )
        for value, dtype, shape in contracts:
            if value.dtype != dtype or value.shape != shape:
                raise BatchingError(
                    "NumpyVerifierBatchV1: dtype or shape differs from contract"
                )
        if not bool(np.all(np.isfinite(self.state_vectors))) or not bool(
            np.all(np.isfinite(self.action_chunks))
        ):
            raise BatchingError("NumpyVerifierBatchV1: model inputs must be finite")
        if not bool(np.all(np.isin(self.failure_targets, (0.0, 1.0)))):
            raise BatchingError("NumpyVerifierBatchV1: targets must be binary")
        if bool(np.any(self.sample_indices < 0)):
            raise BatchingError(
                "NumpyVerifierBatchV1: sample indices must be non-negative"
            )
        for name in (
            "state_vectors",
            "action_chunks",
            "action_masks",
            "failure_targets",
            "sample_indices",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def model_inputs(self) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Return only state, action, and mask arrays accepted by learned models."""

        return self.state_vectors, self.action_chunks, self.action_masks


@dataclass(frozen=True, slots=True)
class TorchVerifierBatchV1:
    """Opaque lazy-PyTorch batch without importing torch at module import time."""

    state_vectors: object
    action_chunks: object
    action_masks: object
    failure_targets: object
    sample_indices: object

    def model_inputs(self) -> tuple[object, object, object]:
        """Return the exact three tensors accepted by learned model forwards."""

        return self.state_vectors, self.action_chunks, self.action_masks


def collate_numpy_examples(
    examples: Sequence[ActionVerifierModelExampleV1],
    *,
    preprocessing: PreprocessingStateV1 | None = None,
) -> NumpyVerifierBatchV1:
    """Collate a non-empty example sequence, optionally using frozen statistics."""

    values = tuple(examples)
    if not values or any(
        not isinstance(item, ActionVerifierModelExampleV1) for item in values
    ):
        raise BatchingError("collate_numpy_examples: valid examples are required")
    projected = (
        values
        if preprocessing is None
        else tuple(normalize_example(item, preprocessing) for item in values)
    )
    return NumpyVerifierBatchV1(
        state_vectors=np.stack([item.state_vector for item in projected]).astype(
            np.float32, copy=False
        ),
        action_chunks=np.stack([item.action_chunk for item in projected]).astype(
            np.float32, copy=False
        ),
        action_masks=np.stack([item.action_mask for item in projected]).astype(
            np.bool_, copy=False
        ),
        failure_targets=np.asarray(
            [item.failure_target for item in projected], dtype=np.float32
        ),
        sample_indices=np.asarray(
            [item.sample_index for item in projected], dtype=np.int64
        ),
    )


def iter_numpy_batches(
    dataset: AcceptedActionVerifierDatasetV1,
    indices: Iterable[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
    preprocessing: PreprocessingStateV1 | None = None,
) -> Iterator[NumpyVerifierBatchV1]:
    """Yield deterministic bounded batches from an explicit preserved split."""

    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        raise BatchingError("iter_numpy_batches: expected accepted dataset")
    if type(batch_size) is not int or batch_size <= 0:
        raise BatchingError("iter_numpy_batches: batch_size must be positive")
    if type(shuffle) is not bool or type(drop_last) is not bool:
        raise BatchingError("iter_numpy_batches: flags must be boolean")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise BatchingError("iter_numpy_batches: seed must be uint64")
    order = np.asarray(tuple(indices), dtype=np.int64)
    if order.ndim != 1 or order.size == 0:
        raise BatchingError("iter_numpy_batches: indices must be non-empty")
    if len(set(order.tolist())) != order.size:
        raise BatchingError("iter_numpy_batches: duplicate indices are invalid")
    if bool(np.any(order < 0)) or bool(np.any(order >= len(dataset))):
        raise BatchingError("iter_numpy_batches: index is out of range")
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    for start in range(0, order.size, batch_size):
        stop = min(start + batch_size, order.size)
        if drop_last and stop - start < batch_size:
            break
        yield collate_numpy_examples(
            tuple(dataset[int(index)] for index in order[start:stop]),
            preprocessing=preprocessing,
        )


def to_torch_batch(
    batch: NumpyVerifierBatchV1,
    *,
    floating_dtype: str = "float32",
    device: str | None = None,
) -> TorchVerifierBatchV1:
    """Copy an immutable NumPy batch into detached tensors via a lazy import."""

    if not isinstance(batch, NumpyVerifierBatchV1):
        raise BatchingError("to_torch_batch: expected NumpyVerifierBatchV1")
    if floating_dtype not in {"float32", "float64"}:
        raise BatchingError("to_torch_batch: unsupported floating dtype")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise TorchTrainingDependencyError(
            "PyTorch is required only for the training optional dependency"
        ) from exc
    module = cast(Any, torch)
    float_dtype = getattr(module, floating_dtype)

    def tensor(value: NDArray[Any], dtype: object) -> object:
        copied = np.array(value, copy=True, order="C")
        result: object = module.tensor(copied, dtype=dtype, device=device).detach()
        return result

    return TorchVerifierBatchV1(
        state_vectors=tensor(batch.state_vectors, float_dtype),
        action_chunks=tensor(batch.action_chunks, float_dtype),
        action_masks=tensor(batch.action_masks, module.bool),
        failure_targets=tensor(batch.failure_targets, float_dtype),
        sample_indices=tensor(batch.sample_indices, module.int64),
    )


def make_torch_collate_fn(
    *,
    preprocessing: PreprocessingStateV1 | None = None,
    floating_dtype: str = "float32",
    device: str | None = None,
) -> Callable[[Sequence[ActionVerifierModelExampleV1]], TorchVerifierBatchV1]:
    """Build a PyTorch DataLoader-compatible collator without importing torch."""

    def collate(
        examples: Sequence[ActionVerifierModelExampleV1],
    ) -> TorchVerifierBatchV1:
        return to_torch_batch(
            collate_numpy_examples(examples, preprocessing=preprocessing),
            floating_dtype=floating_dtype,
            device=device,
        )

    return collate


__all__ = [
    "BatchingError",
    "NumpyVerifierBatchV1",
    "TorchTrainingDependencyError",
    "TorchVerifierBatchV1",
    "collate_numpy_examples",
    "iter_numpy_batches",
    "make_torch_collate_fn",
    "to_torch_batch",
]
