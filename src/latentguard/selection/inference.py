"""Label-free verifier candidate input preparation and timed model inference."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
    PickCubeVerifierStateV1,
)
from latentguard.training.dataset import (
    ACCEPTED_ACTION_DIMENSION,
    ACCEPTED_ACTION_HORIZON,
    ACCEPTED_STATE_DIMENSION,
)
from latentguard.training.preprocessing import PreprocessingStateV1

INFERENCE_LATENCY_SCHEMA_VERSION = "1.0"
INFERENCE_LATENCY_SEMANTIC = (
    "eval_inference_mode_synchronized_wall_clock_linear_quantiles_v1"
)
NORMALIZED_CANDIDATE_BATCH_SCHEMA_VERSION = "1.0"
RAW_CANDIDATE_ACTION_DTYPE = np.dtype("<f8")


class VerifierInferenceError(ValueError):
    """Raised when candidate inputs or model inference violate the contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VerifierInferenceError(f"{context}: {reason}")


def _freeze(value: NDArray[Any]) -> NDArray[Any]:
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


def _candidate_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        _fail("candidate_ids", "expected an ordered sequence")
    result = tuple(value)
    if not result:
        _fail("candidate_ids", "must not be empty")
    for index, candidate_id in enumerate(result):
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id != candidate_id.strip()
            or "\x00" in candidate_id
        ):
            _fail(f"candidate_ids[{index}]", "expected canonical non-empty text")
    if len(set(result)) != len(result):
        _fail("candidate_ids", "duplicate candidate identity")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class NormalizedCandidateBatchV1:
    """Immutable metadata-free float32 model inputs plus stable candidate IDs."""

    candidate_ids: tuple[str, ...]
    state_vectors: NDArray[Any]
    action_chunks: NDArray[Any]
    action_masks: NDArray[Any]
    schema_version: str = NORMALIZED_CANDIDATE_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the exact fixed model shapes and detach every array."""

        candidate_ids = _candidate_ids(self.candidate_ids)
        count = len(candidate_ids)
        contracts = (
            (
                self.state_vectors,
                np.dtype("<f4"),
                (count, ACCEPTED_STATE_DIMENSION),
                "state_vectors",
            ),
            (
                self.action_chunks,
                np.dtype("<f4"),
                (count, ACCEPTED_ACTION_HORIZON, ACCEPTED_ACTION_DIMENSION),
                "action_chunks",
            ),
            (
                self.action_masks,
                np.dtype(np.bool_),
                (count, ACCEPTED_ACTION_HORIZON),
                "action_masks",
            ),
        )
        for value, dtype, shape, name in contracts:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != dtype
                or value.shape != shape
            ):
                _fail(
                    f"NormalizedCandidateBatchV1.{name}",
                    f"expected dtype {dtype.str} and shape {shape}",
                )
        if not bool(np.all(np.isfinite(self.state_vectors))) or not bool(
            np.all(np.isfinite(self.action_chunks))
        ):
            _fail("NormalizedCandidateBatchV1", "model inputs must be finite")
        if not bool(np.all(np.any(self.action_masks, axis=1))):
            _fail(
                "NormalizedCandidateBatchV1.action_masks",
                "every candidate requires at least one valid step",
            )
        object.__setattr__(self, "candidate_ids", candidate_ids)
        for name in ("state_vectors", "action_chunks", "action_masks"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if self.schema_version != NORMALIZED_CANDIDATE_BATCH_SCHEMA_VERSION:
            _fail("NormalizedCandidateBatchV1.schema_version", "unsupported version")

    @property
    def candidate_count(self) -> int:
        """Return the number of candidates represented by the batch."""

        return len(self.candidate_ids)

    def model_inputs(self) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Return only the three allowlisted arrays supplied to a verifier."""

        return self.state_vectors, self.action_chunks, self.action_masks


def normalize_candidate_inputs(
    state: PickCubeVerifierStateV1,
    candidate_actions: NDArray[Any],
    action_masks: NDArray[Any],
    candidate_ids: Sequence[str],
    preprocessing: PreprocessingStateV1,
) -> NormalizedCandidateBatchV1:
    """Validate and normalize blind inputs without labels or reporting metadata."""

    if not isinstance(state, PickCubeVerifierStateV1):
        _fail("state", "expected PickCubeVerifierStateV1")
    if (
        state.semantic != PICKCUBE_VERIFIER_STATE_SEMANTIC
        or state.values.dtype != np.dtype("<f4")
        or state.values.shape != (ACCEPTED_STATE_DIMENSION,)
    ):
        _fail("state", "semantic, dtype, or shape differs")
    if not isinstance(preprocessing, PreprocessingStateV1):
        _fail("preprocessing", "expected PreprocessingStateV1")
    if (
        preprocessing.state_component_count != ACCEPTED_STATE_DIMENSION
        or preprocessing.action_component_count != ACCEPTED_ACTION_DIMENSION
    ):
        _fail("preprocessing", "component dimensions differ")
    ids = _candidate_ids(candidate_ids)
    count = len(ids)
    if (
        not isinstance(candidate_actions, np.ndarray)
        or candidate_actions.dtype != RAW_CANDIDATE_ACTION_DTYPE
        or candidate_actions.shape
        != (count, ACCEPTED_ACTION_HORIZON, ACCEPTED_ACTION_DIMENSION)
        or not candidate_actions.flags.c_contiguous
    ):
        _fail(
            "candidate_actions",
            "expected C-contiguous little-endian float64 [K, 16, 8]",
        )
    if (
        not isinstance(action_masks, np.ndarray)
        or action_masks.dtype != np.dtype(np.bool_)
        or action_masks.shape != (count, ACCEPTED_ACTION_HORIZON)
        or not action_masks.flags.c_contiguous
    ):
        _fail("action_masks", "expected C-contiguous bool [K, 16]")
    if not bool(np.all(np.isfinite(candidate_actions))):
        _fail("candidate_actions", "all values must be finite")
    if not bool(np.all(np.any(action_masks, axis=1))):
        _fail("action_masks", "every candidate requires at least one valid step")

    normalized_state = (
        (state.values.astype(np.float64) - preprocessing.state_mean)
        / preprocessing.state_standard_deviation
    ).astype(np.float32)
    states = np.repeat(normalized_state[None, :], count, axis=0)
    normalized_actions = (
        candidate_actions.astype(np.float64, copy=True) - preprocessing.action_mean
    ) / preprocessing.action_standard_deviation
    normalized_actions[~action_masks] = 0.0
    actions = np.asarray(normalized_actions, dtype=np.float32, order="C")
    if not bool(np.all(np.isfinite(states))) or not bool(np.all(np.isfinite(actions))):
        _fail("normalize_candidate_inputs", "normalization produced non-finite values")
    return NormalizedCandidateBatchV1(
        candidate_ids=ids,
        state_vectors=np.asarray(states, dtype=np.float32, order="C"),
        action_chunks=actions,
        action_masks=np.asarray(action_masks, dtype=np.bool_, order="C"),
    )


@dataclass(frozen=True, slots=True)
class InferenceLatencyReportV1:
    """Fixed warm-up, quantile, throughput, and memory reporting schema."""

    device: str
    candidate_count: int
    warmup_iterations: int
    measurement_iterations: int
    per_candidate_seconds_p50: float
    per_candidate_seconds_p95: float
    per_candidate_seconds_p99: float
    per_group_seconds_p50: float
    per_group_seconds_p95: float
    per_group_seconds_p99: float
    candidates_per_second: float
    peak_allocated_device_memory_bytes: int
    quantile_method: str = "linear"
    semantic: str = INFERENCE_LATENCY_SEMANTIC
    schema_version: str = INFERENCE_LATENCY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate timing values and the fixed measurement semantics."""

        if not isinstance(self.device, str) or not self.device:
            _fail("InferenceLatencyReportV1.device", "expected text")
        for name in (
            "candidate_count",
            "measurement_iterations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                _fail(f"InferenceLatencyReportV1.{name}", "expected positive int")
        if type(self.warmup_iterations) is not int or self.warmup_iterations < 0:
            _fail(
                "InferenceLatencyReportV1.warmup_iterations",
                "expected non-negative int",
            )
        for name in (
            "per_candidate_seconds_p50",
            "per_candidate_seconds_p95",
            "per_candidate_seconds_p99",
            "per_group_seconds_p50",
            "per_group_seconds_p95",
            "per_group_seconds_p99",
            "candidates_per_second",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                _fail(
                    f"InferenceLatencyReportV1.{name}",
                    "expected finite positive value",
                )
        if (
            type(self.peak_allocated_device_memory_bytes) is not int
            or self.peak_allocated_device_memory_bytes < 0
        ):
            _fail(
                "InferenceLatencyReportV1.peak_allocated_device_memory_bytes",
                "expected non-negative int",
            )
        if self.quantile_method != "linear":
            _fail("InferenceLatencyReportV1.quantile_method", "method changed")
        if self.semantic != INFERENCE_LATENCY_SEMANTIC:
            _fail("InferenceLatencyReportV1.semantic", "semantic changed")
        if self.schema_version != INFERENCE_LATENCY_SCHEMA_VERSION:
            _fail("InferenceLatencyReportV1.schema_version", "unsupported version")

    @classmethod
    def from_group_measurements(
        cls,
        measurements: Sequence[float],
        *,
        device: str,
        candidate_count: int,
        warmup_iterations: int,
        peak_allocated_device_memory_bytes: int,
    ) -> InferenceLatencyReportV1:
        """Compute deterministic linear quantiles from per-group durations."""

        values = np.asarray(tuple(measurements), dtype=np.float64)
        if (
            values.ndim != 1
            or values.size == 0
            or not bool(np.all(np.isfinite(values)))
            or not bool(np.all(values > 0.0))
        ):
            _fail("measurements", "expected finite positive group durations")
        if type(candidate_count) is not int or candidate_count <= 0:
            _fail("candidate_count", "expected positive int")
        quantiles = np.quantile(values, (0.50, 0.95, 0.99), method="linear")
        per_candidate = quantiles / candidate_count
        throughput = candidate_count / float(np.mean(values))
        return cls(
            device=device,
            candidate_count=candidate_count,
            warmup_iterations=warmup_iterations,
            measurement_iterations=int(values.size),
            per_candidate_seconds_p50=float(per_candidate[0]),
            per_candidate_seconds_p95=float(per_candidate[1]),
            per_candidate_seconds_p99=float(per_candidate[2]),
            per_group_seconds_p50=float(quantiles[0]),
            per_group_seconds_p95=float(quantiles[1]),
            per_group_seconds_p99=float(quantiles[2]),
            candidates_per_second=throughput,
            peak_allocated_device_memory_bytes=peak_allocated_device_memory_bytes,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the stable latency report field inventory."""

        return {
            "candidate_count": self.candidate_count,
            "candidates_per_second": self.candidates_per_second,
            "device": self.device,
            "measurement_iterations": self.measurement_iterations,
            "peak_allocated_device_memory_bytes": (
                self.peak_allocated_device_memory_bytes
            ),
            "per_candidate_seconds_p50": self.per_candidate_seconds_p50,
            "per_candidate_seconds_p95": self.per_candidate_seconds_p95,
            "per_candidate_seconds_p99": self.per_candidate_seconds_p99,
            "per_group_seconds_p50": self.per_group_seconds_p50,
            "per_group_seconds_p95": self.per_group_seconds_p95,
            "per_group_seconds_p99": self.per_group_seconds_p99,
            "quantile_method": self.quantile_method,
            "schema_version": self.schema_version,
            "semantic": self.semantic,
            "warmup_iterations": self.warmup_iterations,
        }


@dataclass(frozen=True, slots=True, eq=False)
class SeedLogitInferenceV1:
    """One model's deterministic raw logits and operational timing report."""

    logits: NDArray[np.float64]
    timing: InferenceLatencyReportV1

    def __post_init__(self) -> None:
        """Freeze a finite logit for every measured candidate."""

        if (
            not isinstance(self.logits, np.ndarray)
            or self.logits.dtype != np.dtype("<f8")
            or self.logits.shape != (self.timing.candidate_count,)
            or not bool(np.all(np.isfinite(self.logits)))
        ):
            _fail("SeedLogitInferenceV1.logits", "invalid logit array")
        object.__setattr__(self, "logits", _freeze(self.logits))


def infer_prepared_seed_logits_once(
    model: object,
    batch: NormalizedCandidateBatchV1,
    *,
    device: str = "cpu",
) -> NDArray[np.float64]:
    """Run one already-loaded seed model once without timing or synchronization.

    This bounded primitive is used by the ensemble profiler so one outer timer can
    cover normalization, all five forwards, calibration, aggregation, and ranking.
    Device synchronization deliberately remains the outer caller's responsibility.
    """

    if not isinstance(batch, NormalizedCandidateBatchV1):
        _fail("infer_prepared_seed_logits_once.batch", "expected normalized batch")
    if not isinstance(device, str) or not device or device != device.strip():
        _fail("infer_prepared_seed_logits_once.device", "expected canonical text")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise VerifierInferenceError(
            "infer_prepared_seed_logits_once: PyTorch training dependency is "
            "unavailable"
        ) from exc
    if not isinstance(model, torch.nn.Module):
        _fail("infer_prepared_seed_logits_once.model", "expected torch.nn.Module")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        _fail("infer_prepared_seed_logits_once.device", "CUDA is unavailable")
    if model.training:
        _fail(
            "infer_prepared_seed_logits_once.model",
            "model must already be in eval mode",
        )
    states = torch.tensor(
        np.array(batch.state_vectors, copy=True), dtype=torch.float32, device=target
    )
    actions = torch.tensor(
        np.array(batch.action_chunks, copy=True), dtype=torch.float32, device=target
    )
    masks = torch.tensor(
        np.array(batch.action_masks, copy=True), dtype=torch.bool, device=target
    )
    with torch.inference_mode():
        output = model(states, actions, masks)
    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != 1
        or tuple(output.shape) != (batch.candidate_count,)
        or not bool(torch.isfinite(output).all())
    ):
        _fail(
            "infer_prepared_seed_logits_once.output",
            "expected finite logits [K]",
        )
    return np.asarray(output.detach().cpu().numpy(), dtype=np.float64)


def infer_seed_logits(
    model: object,
    batch: NormalizedCandidateBatchV1,
    *,
    device: str = "cpu",
    warmup_iterations: int = 2,
    measurement_iterations: int = 5,
    clock: Callable[[], float] = perf_counter,
) -> SeedLogitInferenceV1:
    """Run one model without labels, using lazy Torch import and fixed timing rules."""

    if not isinstance(batch, NormalizedCandidateBatchV1):
        _fail("infer_seed_logits.batch", "expected NormalizedCandidateBatchV1")
    if type(warmup_iterations) is not int or warmup_iterations < 0:
        _fail("infer_seed_logits.warmup_iterations", "expected non-negative int")
    if type(measurement_iterations) is not int or measurement_iterations <= 0:
        _fail("infer_seed_logits.measurement_iterations", "expected positive int")
    if not isinstance(device, str) or not device or device != device.strip():
        _fail("infer_seed_logits.device", "expected canonical text")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise VerifierInferenceError(
            "infer_seed_logits: PyTorch training dependency is unavailable"
        ) from exc
    if not isinstance(model, torch.nn.Module):
        _fail("infer_seed_logits.model", "expected torch.nn.Module")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        _fail("infer_seed_logits.device", "CUDA is unavailable")
    states = torch.tensor(
        np.array(batch.state_vectors, copy=True), dtype=torch.float32, device=target
    )
    actions = torch.tensor(
        np.array(batch.action_chunks, copy=True), dtype=torch.float32, device=target
    )
    masks = torch.tensor(
        np.array(batch.action_masks, copy=True), dtype=torch.bool, device=target
    )
    model.to(target)
    model.eval()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def run_once() -> NDArray[np.float64]:
        with torch.inference_mode():
            output = model(states, actions, masks)
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 1
            or tuple(output.shape) != (batch.candidate_count,)
            or not bool(torch.isfinite(output).all())
        ):
            _fail("infer_seed_logits.output", "expected finite logits [K]")
        return np.asarray(output.detach().cpu().numpy(), dtype=np.float64)

    for _ in range(warmup_iterations):
        run_once()
    synchronize()
    measurements: list[float] = []
    reference: NDArray[np.float64] | None = None
    for _ in range(measurement_iterations):
        synchronize()
        start = clock()
        observed = run_once()
        synchronize()
        finish = clock()
        if not math.isfinite(start) or not math.isfinite(finish) or finish <= start:
            _fail("infer_seed_logits.clock", "clock must be finite and increasing")
        measurements.append(finish - start)
        if reference is None:
            reference = observed
        elif not np.array_equal(reference, observed):
            _fail("infer_seed_logits.output", "repeated inference changed logits")
    if reference is None:  # pragma: no cover - positive iterations close this branch
        _fail("infer_seed_logits", "no measured inference result")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
    )
    return SeedLogitInferenceV1(
        logits=np.asarray(reference, dtype=np.float64),
        timing=InferenceLatencyReportV1.from_group_measurements(
            measurements,
            device=str(target),
            candidate_count=batch.candidate_count,
            warmup_iterations=warmup_iterations,
            peak_allocated_device_memory_bytes=peak_memory,
        ),
    )


__all__ = [
    "INFERENCE_LATENCY_SCHEMA_VERSION",
    "INFERENCE_LATENCY_SEMANTIC",
    "NORMALIZED_CANDIDATE_BATCH_SCHEMA_VERSION",
    "RAW_CANDIDATE_ACTION_DTYPE",
    "InferenceLatencyReportV1",
    "NormalizedCandidateBatchV1",
    "SeedLogitInferenceV1",
    "VerifierInferenceError",
    "infer_prepared_seed_logits_once",
    "infer_seed_logits",
    "normalize_candidate_inputs",
]
