"""Five-seed calibrated verifier inference and stable candidate ranking."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PickCubeVerifierStateV1,
)
from latentguard.selection.checkpoint_bundle import (
    EXPECTED_FIVE_SEEDS,
    VERIFIER_ENSEMBLE_SEMANTIC,
    BundleLoadingDiagnosticsV1,
    LoadedVerifierBundleV1,
)
from latentguard.selection.inference import (
    InferenceLatencyReportV1,
    infer_prepared_seed_logits_once,
    normalize_candidate_inputs,
)

ENSEMBLE_INFERENCE_SCHEMA_VERSION = "1.0"
ENSEMBLE_PROFILE_WARMUP_ITERATIONS = 2
ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS = 5
ENSEMBLE_END_TO_END_PROFILE_SEMANTIC = (
    "normalize_five_seed_forward_calibration_mean_stable_ranking_v1"
)


class VerifierEnsembleError(ValueError):
    """Raised when five-seed inference or deterministic ranking is invalid."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VerifierEnsembleError(f"{context}: {reason}")


def _freeze(value: NDArray[Any]) -> NDArray[Any]:
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype).reshape(
        detached.shape
    )


@dataclass(frozen=True, slots=True)
class EnsembleEndToEndProfileV1:
    """Bounded in-memory durations for complete five-seed selector calls."""

    device: str
    candidate_count: int
    group_durations_seconds: tuple[float, ...] = field(repr=False)
    peak_allocated_device_memory_bytes: int = 0
    warmup_iterations: int = ENSEMBLE_PROFILE_WARMUP_ITERATIONS
    measurement_iterations: int = ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
    semantic: str = ENSEMBLE_END_TO_END_PROFILE_SEMANTIC
    schema_version: str = ENSEMBLE_INFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require the frozen 2/5 protocol and finite positive raw durations."""

        durations = tuple(float(item) for item in self.group_durations_seconds)
        if not isinstance(self.device, str) or not self.device:
            _fail("EnsembleEndToEndProfileV1.device", "expected text")
        if type(self.candidate_count) is not int or self.candidate_count <= 0:
            _fail("EnsembleEndToEndProfileV1.candidate_count", "expected positive int")
        if (
            self.warmup_iterations != ENSEMBLE_PROFILE_WARMUP_ITERATIONS
            or self.measurement_iterations != ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
            or len(durations) != ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
        ):
            _fail(
                "EnsembleEndToEndProfileV1",
                "fixed 2-warmup/5-measurement contract changed",
            )
        if any(not math.isfinite(item) or item <= 0.0 for item in durations):
            _fail(
                "EnsembleEndToEndProfileV1.group_durations_seconds",
                "expected finite positive durations",
            )
        if (
            type(self.peak_allocated_device_memory_bytes) is not int
            or self.peak_allocated_device_memory_bytes < 0
        ):
            _fail(
                "EnsembleEndToEndProfileV1.peak_allocated_device_memory_bytes",
                "expected non-negative int",
            )
        if self.semantic != ENSEMBLE_END_TO_END_PROFILE_SEMANTIC:
            _fail("EnsembleEndToEndProfileV1.semantic", "semantic changed")
        if self.schema_version != ENSEMBLE_INFERENCE_SCHEMA_VERSION:
            _fail("EnsembleEndToEndProfileV1.schema_version", "unsupported version")
        object.__setattr__(self, "group_durations_seconds", durations)

    @property
    def latency_report(self) -> InferenceLatencyReportV1:
        """Aggregate this group's five raw repetitions without exposing them."""

        return InferenceLatencyReportV1.from_group_measurements(
            self.group_durations_seconds,
            device=self.device,
            candidate_count=self.candidate_count,
            warmup_iterations=self.warmup_iterations,
            peak_allocated_device_memory_bytes=(
                self.peak_allocated_device_memory_bytes
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return only summary statistics; raw duration samples remain in memory."""

        return {
            **self.latency_report.to_dict(),
            "measurement_scope": self.semantic,
            "raw_duration_values_persisted": False,
        }


@dataclass(frozen=True, slots=True)
class EnsembleInferenceDiagnosticsV1:
    """Bundle loading plus complete five-seed selector-call diagnostics."""

    bundle_loading: BundleLoadingDiagnosticsV1
    end_to_end_profile: EnsembleEndToEndProfileV1
    seed_order: tuple[int, ...] = EXPECTED_FIVE_SEEDS
    schema_version: str = ENSEMBLE_INFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require one profile whose runtime device matches the loaded bundle."""

        if self.seed_order != EXPECTED_FIVE_SEEDS:
            _fail("EnsembleInferenceDiagnosticsV1", "seed inventory changed")
        if not isinstance(self.end_to_end_profile, EnsembleEndToEndProfileV1):
            _fail("EnsembleInferenceDiagnosticsV1", "end-to-end profile is invalid")
        if self.end_to_end_profile.device != self.bundle_loading.device:
            _fail("EnsembleInferenceDiagnosticsV1", "device differs across stages")
        if self.schema_version != ENSEMBLE_INFERENCE_SCHEMA_VERSION:
            _fail(
                "EnsembleInferenceDiagnosticsV1.schema_version",
                "unsupported version",
            )

    def to_dict(self) -> dict[str, object]:
        """Return stable nested diagnostics without model inputs or runtime paths."""

        return {
            "bundle_loading": self.bundle_loading.to_dict(),
            "end_to_end_profile": self.end_to_end_profile.to_dict(),
            "schema_version": self.schema_version,
            "seed_order": list(self.seed_order),
        }


@dataclass(frozen=True, slots=True, eq=False)
class _EnsembleComputationV1:
    """One complete unpersisted selector computation."""

    candidate_ids: tuple[str, ...]
    per_seed_raw_logits: NDArray[np.float64]
    per_seed_calibrated_failure_probabilities: NDArray[np.float64]
    ensemble_failure_probabilities: NDArray[np.float64]
    ranking: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class EnsembleInferenceResultV1:
    """Raw and calibrated five-seed predictions with stable ID-based ranking."""

    candidate_ids: tuple[str, ...]
    seed_order: tuple[int, ...]
    per_seed_raw_logits: NDArray[Any]
    per_seed_calibrated_failure_probabilities: NDArray[Any]
    ensemble_failure_probabilities: NDArray[Any]
    ranking: tuple[str, ...]
    diagnostics: EnsembleInferenceDiagnosticsV1
    ensemble_semantic: str = VERIFIER_ENSEMBLE_SEMANTIC
    schema_version: str = ENSEMBLE_INFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate exact matrices, arithmetic mean, and stable tie-breaking."""

        candidate_ids = tuple(self.candidate_ids)
        ranking = tuple(self.ranking)
        seed_order = tuple(self.seed_order)
        count = len(candidate_ids)
        if (
            count <= 0
            or len(set(candidate_ids)) != count
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in candidate_ids
            )
        ):
            _fail("EnsembleInferenceResultV1.candidate_ids", "invalid identities")
        if seed_order != EXPECTED_FIVE_SEEDS:
            _fail("EnsembleInferenceResultV1.seed_order", "seed set changed")
        contracts = (
            (
                self.per_seed_raw_logits,
                (len(seed_order), count),
                "per_seed_raw_logits",
            ),
            (
                self.per_seed_calibrated_failure_probabilities,
                (len(seed_order), count),
                "per_seed_calibrated_failure_probabilities",
            ),
            (
                self.ensemble_failure_probabilities,
                (count,),
                "ensemble_failure_probabilities",
            ),
        )
        for value, shape, name in contracts:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f8")
                or value.shape != shape
                or not bool(np.all(np.isfinite(value)))
            ):
                _fail(
                    f"EnsembleInferenceResultV1.{name}",
                    f"expected finite float64 array {shape}",
                )
        calibrated = self.per_seed_calibrated_failure_probabilities
        if not bool(np.all((calibrated >= 0.0) & (calibrated <= 1.0))) or not bool(
            np.all(
                (self.ensemble_failure_probabilities >= 0.0)
                & (self.ensemble_failure_probabilities <= 1.0)
            )
        ):
            _fail("EnsembleInferenceResultV1", "probabilities must lie in [0, 1]")
        expected_mean = np.mean(calibrated, axis=0, dtype=np.float64)
        if not np.array_equal(expected_mean, self.ensemble_failure_probabilities):
            _fail(
                "EnsembleInferenceResultV1.ensemble_failure_probabilities",
                "not the exact arithmetic five-seed mean",
            )
        expected_ranking = tuple(
            candidate_id
            for _, candidate_id in sorted(
                zip(
                    self.ensemble_failure_probabilities.tolist(),
                    candidate_ids,
                    strict=True,
                ),
                key=lambda item: (item[0], item[1]),
            )
        )
        if ranking != expected_ranking or set(ranking) != set(candidate_ids):
            _fail(
                "EnsembleInferenceResultV1.ranking",
                "must sort by probability then stable candidate ID",
            )
        if (
            self.diagnostics.seed_order != seed_order
            or self.diagnostics.end_to_end_profile.candidate_count != count
        ):
            _fail("EnsembleInferenceResultV1.diagnostics", "inventory differs")
        if self.ensemble_semantic != VERIFIER_ENSEMBLE_SEMANTIC:
            _fail("EnsembleInferenceResultV1.ensemble_semantic", "semantic changed")
        if self.schema_version != ENSEMBLE_INFERENCE_SCHEMA_VERSION:
            _fail("EnsembleInferenceResultV1.schema_version", "unsupported version")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "seed_order", seed_order)
        object.__setattr__(self, "ranking", ranking)
        for name in (
            "per_seed_raw_logits",
            "per_seed_calibrated_failure_probabilities",
            "ensemble_failure_probabilities",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    @property
    def selected_candidate_id(self) -> str:
        """Return the lowest predicted-failure candidate."""

        return self.ranking[0]

    def probability_for(self, candidate_id: str) -> float:
        """Return one candidate's ensemble failure probability by stable ID."""

        try:
            index = self.candidate_ids.index(candidate_id)
        except ValueError as exc:
            raise VerifierEnsembleError(
                f"unknown candidate identity {candidate_id!r}"
            ) from exc
        return float(self.ensemble_failure_probabilities[index])


def score_verifier_ensemble(
    bundle: LoadedVerifierBundleV1,
    state: PickCubeVerifierStateV1,
    candidate_actions: NDArray[Any],
    action_masks: NDArray[Any],
    candidate_ids: Sequence[str],
    *,
    device: str | None = None,
    warmup_iterations: int = 2,
    measurement_iterations: int = 5,
    clock: Callable[[], float] = perf_counter,
) -> EnsembleInferenceResultV1:
    """Score and profile one complete five-seed selector call end to end."""

    if not isinstance(bundle, LoadedVerifierBundleV1):
        _fail("score_verifier_ensemble.bundle", "invalid loaded bundle")
    if (
        warmup_iterations != ENSEMBLE_PROFILE_WARMUP_ITERATIONS
        or measurement_iterations != ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
    ):
        _fail(
            "score_verifier_ensemble",
            "M3C requires exactly 2 warmups and 5 measured selector calls",
        )
    runtime_device = bundle.loading_diagnostics.device if device is None else device
    if runtime_device != bundle.loading_diagnostics.device:
        _fail(
            "score_verifier_ensemble.device",
            "must match the digest-validated bundle loading device",
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise VerifierEnsembleError(
            "score_verifier_ensemble: PyTorch training dependency is unavailable"
        ) from exc
    target = torch.device(runtime_device)
    if target.type == "cuda" and not torch.cuda.is_available():
        _fail("score_verifier_ensemble.device", "CUDA is unavailable")
    for loaded_seed in bundle.seeds:
        if not isinstance(loaded_seed.model, torch.nn.Module):
            _fail("score_verifier_ensemble.model", "expected torch.nn.Module")
        loaded_seed.model.to(target)
        loaded_seed.model.eval()

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def compute_once() -> _EnsembleComputationV1:
        batch = normalize_candidate_inputs(
            state,
            candidate_actions,
            action_masks,
            candidate_ids,
            bundle.preprocessing,
        )
        logits: list[NDArray[np.float64]] = []
        probabilities: list[NDArray[np.float64]] = []
        for loaded_seed in bundle.seeds:
            raw_logits = infer_prepared_seed_logits_once(
                loaded_seed.model,
                batch,
                device=runtime_device,
            )
            calibrated = loaded_seed.calibration.apply(raw_logits)
            if (
                calibrated.dtype != np.dtype("<f8")
                or calibrated.shape != (batch.candidate_count,)
                or not bool(np.all(np.isfinite(calibrated)))
                or not bool(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))
            ):
                _fail("score_verifier_ensemble.calibration", "invalid probabilities")
            logits.append(np.asarray(raw_logits, dtype=np.float64))
            probabilities.append(np.asarray(calibrated, dtype=np.float64))
        raw_matrix = np.stack(logits).astype(np.float64, copy=False)
        calibrated_matrix = np.stack(probabilities).astype(np.float64, copy=False)
        ensemble = np.mean(calibrated_matrix, axis=0, dtype=np.float64)
        ranking = tuple(
            candidate_id
            for _, candidate_id in sorted(
                zip(ensemble.tolist(), batch.candidate_ids, strict=True),
                key=lambda item: (item[0], item[1]),
            )
        )
        if not all(math.isfinite(float(value)) for value in ensemble):
            _fail("score_verifier_ensemble", "ensemble produced non-finite values")
        return _EnsembleComputationV1(
            candidate_ids=batch.candidate_ids,
            per_seed_raw_logits=np.asarray(raw_matrix, dtype=np.float64),
            per_seed_calibrated_failure_probabilities=np.asarray(
                calibrated_matrix, dtype=np.float64
            ),
            ensemble_failure_probabilities=np.asarray(ensemble, dtype=np.float64),
            ranking=ranking,
        )

    for _ in range(warmup_iterations):
        compute_once()
    synchronize()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    durations: list[float] = []
    reference: _EnsembleComputationV1 | None = None
    for _ in range(measurement_iterations):
        synchronize()
        start = clock()
        observed = compute_once()
        synchronize()
        finish = clock()
        if not math.isfinite(start) or not math.isfinite(finish) or finish <= start:
            _fail(
                "score_verifier_ensemble.clock",
                "clock must be finite and increasing",
            )
        durations.append(finish - start)
        if reference is None:
            reference = observed
        elif (
            observed.candidate_ids != reference.candidate_ids
            or observed.ranking != reference.ranking
            or not np.array_equal(
                observed.per_seed_raw_logits, reference.per_seed_raw_logits
            )
            or not np.array_equal(
                observed.per_seed_calibrated_failure_probabilities,
                reference.per_seed_calibrated_failure_probabilities,
            )
            or not np.array_equal(
                observed.ensemble_failure_probabilities,
                reference.ensemble_failure_probabilities,
            )
        ):
            _fail("score_verifier_ensemble", "repeated selector call changed outputs")
    if reference is None:  # pragma: no cover - fixed positive count closes this branch
        _fail("score_verifier_ensemble", "no measured selector result")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
    )
    return EnsembleInferenceResultV1(
        candidate_ids=reference.candidate_ids,
        seed_order=tuple(item.identity.seed for item in bundle.seeds),
        per_seed_raw_logits=reference.per_seed_raw_logits,
        per_seed_calibrated_failure_probabilities=(
            reference.per_seed_calibrated_failure_probabilities
        ),
        ensemble_failure_probabilities=reference.ensemble_failure_probabilities,
        ranking=reference.ranking,
        diagnostics=EnsembleInferenceDiagnosticsV1(
            bundle_loading=bundle.loading_diagnostics,
            end_to_end_profile=EnsembleEndToEndProfileV1(
                device=runtime_device,
                candidate_count=len(reference.candidate_ids),
                group_durations_seconds=tuple(durations),
                peak_allocated_device_memory_bytes=peak_memory,
            ),
        ),
    )


__all__ = [
    "ENSEMBLE_END_TO_END_PROFILE_SEMANTIC",
    "ENSEMBLE_INFERENCE_SCHEMA_VERSION",
    "ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS",
    "ENSEMBLE_PROFILE_WARMUP_ITERATIONS",
    "EnsembleEndToEndProfileV1",
    "EnsembleInferenceDiagnosticsV1",
    "EnsembleInferenceResultV1",
    "VerifierEnsembleError",
    "score_verifier_ensemble",
]
