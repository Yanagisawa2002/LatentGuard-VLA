"""Frozen calibration and one-shot internal M3A visual evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray

from latentguard.action_verifier import DatasetSplit
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.calibration import (
    FrozenThresholdStateV1,
    TemperatureCalibrationStateV1,
    fit_temperature_scaling,
    fit_validation_thresholds,
)
from latentguard.training.metrics import (
    GroupCandidate,
    evaluate_binary_metrics,
    evaluate_coverage_risk,
    evaluate_group_ranking,
)
from latentguard.visual_training.cache import LoadedFeatureCacheV1
from latentguard.visual_training.checkpoint import (
    checkpoint_content_digest,
    inspect_visual_checkpoint,
    load_visual_checkpoint,
)
from latentguard.visual_training.config import (
    VisualModelConfigV1,
    VisualTrainingConfigV1,
)
from latentguard.visual_training.data import (
    TEST_DOMAIN_IDS,
    TRAIN_DOMAIN_IDS,
    VisualActionTrainingDatasetV1,
)
from latentguard.visual_training.models import (
    VisualActionVerifier,
    build_visual_action_verifier,
)
from latentguard.visual_training.training import (
    ActionNormalizationV1,
    VisualInputProvider,
    fit_action_normalization,
)


class VisualEvaluationError(ValueError):
    """Raised when frozen internal evaluation identity or ordering differs."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualEvaluationError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    payload = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _summary(values: list[float]) -> dict[str, float]:
    """Return the fixed five-statistic summary for one three-seed metric."""
    observed = np.asarray(values, dtype=np.float64)
    if observed.shape != (3,) or not bool(np.all(np.isfinite(observed))):
        _fail("three-seed aggregate", "expected three finite values")
    return {
        "maximum": float(np.max(observed)),
        "mean": float(np.mean(observed)),
        "median": float(np.median(observed)),
        "minimum": float(np.min(observed)),
        "standard_deviation": float(np.std(observed, ddof=0)),
    }


def _trajectory_bootstrap(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    trajectory_ids: tuple[str, ...],
    *,
    seed: int = 271828,
    samples: int = 2000,
) -> dict[str, object]:
    """Compute a deterministic trajectory-level paired mean interval."""
    if first.shape != second.shape or first.shape != (len(trajectory_ids),):
        _fail("trajectory bootstrap", "aligned vectors are required")
    unique = tuple(sorted(set(trajectory_ids)))
    by_trajectory = {
        item: np.asarray(
            [index for index, value in enumerate(trajectory_ids) if value == item],
            dtype=np.int64,
        )
        for item in unique
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(unique), size=len(unique))
        positions = np.concatenate([by_trajectory[unique[item]] for item in draw])
        estimates[index] = float(np.mean(first[positions] - second[positions]))
    return {
        "bootstrap_samples": samples,
        "confidence_lower": float(np.quantile(estimates, 0.025, method="linear")),
        "confidence_upper": float(np.quantile(estimates, 0.975, method="linear")),
        "estimate": float(np.mean(first - second)),
        "resampling_unit": "source_trajectory",
        "seed": seed,
    }


def _group_candidates(
    dataset: VisualActionTrainingDatasetV1,
    indices: tuple[int, ...],
    probabilities: NDArray[np.float64],
) -> tuple[GroupCandidate, ...]:
    return tuple(
        GroupCandidate(
            sample_id=dataset.reporting[index].candidate_sample_id,
            group_id=dataset.reporting[index].candidate_group_id,
            source_trajectory_id=dataset.reporting[index].source_trajectory_id,
            candidate_type=dataset.reporting[index].candidate_type.value,
            failure_target=dataset.examples[index].failure_target,
            failure_probability=float(probabilities[position]),
        )
        for position, index in enumerate(indices)
    )


def _selection_vector(
    candidates: tuple[GroupCandidate, ...],
) -> tuple[NDArray[np.float64], tuple[str, ...], tuple[str, ...]]:
    grouped: dict[str, list[GroupCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.group_id, []).append(candidate)
    successes: list[float] = []
    groups: list[str] = []
    trajectories: list[str] = []
    for group_id in sorted(grouped):
        corrupted = [
            item for item in grouped[group_id] if item.candidate_type == "corrupted"
        ]
        if not corrupted:
            _fail("candidate selection", "group has no corrupted candidates")
        selected = min(
            corrupted, key=lambda item: (item.failure_probability, item.sample_id)
        )
        successes.append(float(selected.failure_target == 0))
        groups.append(group_id)
        trajectories.append(selected.source_trajectory_id)
    return (
        np.asarray(successes, dtype=np.float64),
        tuple(groups),
        tuple(trajectories),
    )


def _serialized_metric_value(row: Mapping[str, object], name: str) -> float:
    metrics = row.get("binary_metrics")
    if not isinstance(metrics, Mapping):
        _fail("three-seed aggregate", "binary metrics are absent")
    metric = metrics.get(name)
    if not isinstance(metric, Mapping):
        _fail("three-seed aggregate", f"metric {name!r} is absent")
    value = metric.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail("three-seed aggregate", f"metric {name!r} is undefined")
    return float(value)


@dataclass(frozen=True, slots=True)
class LoadedPromotedSeedV1:
    """One frozen M4B seed and validation-fitted post-training state."""

    seed: int
    model: VisualActionVerifier
    checkpoint_digest: str
    checkpoint_identity: str
    calibration: TemperatureCalibrationStateV1 | None = None
    thresholds: FrozenThresholdStateV1 | None = None


def _load_models(
    model_config: VisualModelConfigV1,
    training_config: VisualTrainingConfigV1,
    checkpoints: tuple[Path, Path, Path],
    *,
    device: torch.device,
) -> tuple[LoadedPromotedSeedV1, ...]:
    result: list[LoadedPromotedSeedV1] = []
    for expected_seed, path in enumerate(checkpoints):
        metadata = inspect_visual_checkpoint(path)
        if (
            metadata.progress.binding.seed != expected_seed
            or metadata.progress.binding.model_config_digest
            != model_config.content_digest
            or metadata.progress.binding.training_config_digest
            != training_config.content_digest
            or not metadata.progress.complete
            or dict(metadata.model_config) != model_config.as_mapping()
            or dict(metadata.training_config) != training_config.as_mapping()
        ):
            _fail("promoted checkpoint", "seed, model, or completion differs")
        model = build_visual_action_verifier(model_config, seed=expected_seed)
        load_visual_checkpoint(
            path,
            expected_binding=metadata.progress.binding,
            model=model,
            restore_rng=False,
            model_only=True,
        )
        model.to(device)
        model.eval()
        result.append(
            LoadedPromotedSeedV1(
                seed=expected_seed,
                model=model,
                checkpoint_digest=checkpoint_content_digest(path),
                checkpoint_identity=metadata.progress.binding.run_identity,
            )
        )
    return tuple(result)


def _infer_domain(
    model: VisualActionVerifier,
    dataset: VisualActionTrainingDatasetV1,
    provider: VisualInputProvider,
    normalization: ActionNormalizationV1,
    *,
    split: DatasetSplit,
    domain_id: str,
    device: torch.device,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    indices = dataset.indices_for_split(split)
    logits: list[NDArray[Any]] = []
    targets: list[int] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        packet_ids = tuple(
            dataset.reporting[index].packet_ids_by_domain[domain_id]
            for index in batch_indices
        )
        examples = tuple(dataset.examples[index] for index in batch_indices)
        visuals = provider.batch(packet_ids, view_count=model.config.view_count)
        masks = np.stack([item.action_mask for item in examples]).astype(np.bool_)
        actions = normalization.apply(
            np.stack([item.action_chunk for item in examples]), masks
        )
        with torch.inference_mode():
            output = model(
                torch.tensor(visuals, device=device),
                torch.tensor(actions, device=device),
                torch.tensor(masks, device=device),
            )
        if tuple(output.shape) != (len(batch_indices),) or not bool(
            torch.isfinite(output).all()
        ):
            _fail("internal inference", "model output contract differs")
        logits.append(output.detach().cpu().numpy())
        targets.extend(item.failure_target for item in examples)
    return (
        np.concatenate(logits).astype(np.float64),
        np.asarray(targets, dtype=np.int64),
    )


def evaluate_promoted_visual_ensemble(
    model_config: VisualModelConfigV1,
    training_config: VisualTrainingConfigV1,
    checkpoints: tuple[Path, Path, Path],
    dataset: VisualActionTrainingDatasetV1,
    visual_dataset: object,
    visual_root: Path,
    feature_cache: LoadedFeatureCacheV1 | None,
    output_dir: Path,
    *,
    device: str,
) -> Mapping[str, object]:
    """Fit validation-only postprocessing then open internal test exactly once."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        _fail("internal evaluation", "CUDA is unavailable")
    normalization = fit_action_normalization(dataset)
    provider = VisualInputProvider(
        visual_dataset=visual_dataset,
        visual_root=visual_root,
        feature_cache=feature_cache,
        frozen=model_config.backbone_frozen,
    )
    seeds = _load_models(model_config, training_config, checkpoints, device=target)
    checkpoint_bindings = tuple(
        inspect_visual_checkpoint(path).progress.binding for path in checkpoints
    )
    first_binding = checkpoint_bindings[0]
    common_binding_fields = (
        "git_sha",
        "visual_dataset_digest",
        "structured_dataset_digest",
        "split_digest",
        "backbone_digest",
        "feature_cache_digest",
        "teacher_cache_digest",
        "action_preprocessing_digest",
        "model_config_digest",
        "training_config_digest",
    )
    if any(
        any(
            getattr(item, field) != getattr(first_binding, field)
            for field in common_binding_fields
        )
        for item in checkpoint_bindings[1:]
    ):
        _fail("internal evaluation", "three-seed semantic binding differs")
    if (
        first_binding.visual_dataset_digest != dataset.visual_dataset_digest
        or first_binding.structured_dataset_digest != dataset.structured_dataset_digest
        or first_binding.split_digest != dataset.split_digest
        or first_binding.feature_cache_digest
        != (None if feature_cache is None else feature_cache.content_digest)
    ):
        _fail("internal evaluation", "dataset or feature binding differs")
    calibrated_seeds: list[LoadedPromotedSeedV1] = []
    validation_targets: NDArray[np.int64] | None = None
    for seed in seeds:
        domain_logits: list[NDArray[np.float64]] = []
        domain_targets: list[NDArray[np.int64]] = []
        for domain in TRAIN_DOMAIN_IDS:
            logits, labels = _infer_domain(
                seed.model,
                dataset,
                provider,
                normalization,
                split=DatasetSplit.VALIDATION,
                domain_id=domain,
                device=target,
                batch_size=training_config.batch_size,
            )
            domain_logits.append(logits)
            domain_targets.append(labels)
        logits = np.concatenate(domain_logits)
        labels = np.concatenate(domain_targets)
        if validation_targets is None:
            validation_targets = labels
        elif not np.array_equal(validation_targets, labels):
            _fail("internal evaluation", "validation ordering changed across seeds")
        metadata = inspect_visual_checkpoint(checkpoints[seed.seed])
        calibration = fit_temperature_scaling(
            logits,
            labels,
            split="validation",
            dataset_digest=dataset.structured_dataset_digest,
            split_digest=dataset.split_digest,
            checkpoint_identity=seed.checkpoint_identity,
            checkpoint_content_digest=seed.checkpoint_digest,
            model_config_digest=model_config.content_digest,
            preprocessing_digest=normalization.content_digest,
        )
        probabilities = calibration.apply(logits)
        thresholds = fit_validation_thresholds(
            probabilities,
            labels,
            split="validation",
            dataset_digest=dataset.structured_dataset_digest,
            split_digest=dataset.split_digest,
            calibration_digest=calibration.content_digest,
            validation_prediction_digest=calibration.validation_prediction_digest,
            target_failure_recall=training_config.target_failure_recall,
        )
        if (
            metadata.progress.binding.action_preprocessing_digest
            != normalization.content_digest
        ):
            _fail("internal evaluation", "checkpoint preprocessing binding differs")
        calibrated_seeds.append(
            LoadedPromotedSeedV1(
                seed=seed.seed,
                model=seed.model,
                checkpoint_digest=seed.checkpoint_digest,
                checkpoint_identity=seed.checkpoint_identity,
                calibration=calibration,
                thresholds=thresholds,
            )
        )

    domain_reports: dict[str, object] = {}
    selection_vectors: dict[str, NDArray[np.float64]] = {}
    selection_group_ids: tuple[str, ...] | None = None
    selection_trajectories: tuple[str, ...] | None = None
    test_indices = dataset.indices_for_split(DatasetSplit.TEST)
    for domain in TEST_DOMAIN_IDS:
        probabilities_by_seed: list[NDArray[np.float64]] = []
        reference_targets: NDArray[np.int64] | None = None
        seed_reports: list[dict[str, object]] = []
        for seed in calibrated_seeds:
            logits, labels = _infer_domain(
                seed.model,
                dataset,
                provider,
                normalization,
                split=DatasetSplit.TEST,
                domain_id=domain,
                device=target,
                batch_size=training_config.batch_size,
            )
            assert seed.calibration is not None
            assert seed.thresholds is not None
            probabilities = seed.calibration.apply(logits)
            probabilities_by_seed.append(probabilities)
            if reference_targets is None:
                reference_targets = labels
            elif not np.array_equal(reference_targets, labels):
                _fail("internal evaluation", "test ordering changed across seeds")
            binary = evaluate_binary_metrics(
                labels,
                probabilities,
                threshold=seed.thresholds.maximum_balanced_accuracy.threshold,
            )
            ranking = evaluate_group_ranking(
                _group_candidates(dataset, test_indices, probabilities)
            )
            seed_reports.append(
                {
                    "binary_metrics": binary.to_dict(),
                    "group_ranking": ranking.to_dict(),
                    "seed": seed.seed,
                }
            )
        assert reference_targets is not None
        ensemble = np.mean(np.stack(probabilities_by_seed), axis=0, dtype=np.float64)
        threshold = float(
            np.mean(
                [
                    seed.thresholds.maximum_balanced_accuracy.threshold
                    for seed in calibrated_seeds
                    if seed.thresholds is not None
                ]
            )
        )
        binary = evaluate_binary_metrics(
            reference_targets, ensemble, threshold=threshold
        )
        candidates = _group_candidates(dataset, test_indices, ensemble)
        ranking = evaluate_group_ranking(candidates)
        selected, group_ids, trajectories = _selection_vector(candidates)
        if selection_group_ids is None:
            selection_group_ids = group_ids
            selection_trajectories = trajectories
        elif group_ids != selection_group_ids or trajectories != selection_trajectories:
            _fail("internal evaluation", "domain group ordering changed")
        selection_vectors[domain] = selected
        metric_fields = {
            "brier_score": [
                _serialized_metric_value(row, "brier_score") for row in seed_reports
            ],
            "expected_calibration_error": [
                _serialized_metric_value(row, "expected_calibration_error")
                for row in seed_reports
            ],
            "failure_auprc": [
                _serialized_metric_value(row, "failure_auprc") for row in seed_reports
            ],
            "roc_auc": [
                _serialized_metric_value(row, "roc_auc") for row in seed_reports
            ],
        }
        domain_reports[domain] = {
            "binary_metrics": binary.to_dict(),
            "candidate_selection": {
                "coverage": 1.0,
                "group_count": len(selected),
                "selected_success_rate": float(np.mean(selected)),
                "task_failure_rate": 1.0 - float(np.mean(selected)),
            },
            "coverage_risk": evaluate_coverage_risk(
                reference_targets, ensemble
            ).to_dict(),
            "group_ranking": ranking.to_dict(),
            "prediction_digest": _digest(
                {
                    "candidate_ids": [item.sample_id for item in candidates],
                    "probabilities": ensemble.tolist(),
                },
                "M4BInternalDomainPredictionsV1",
            ),
            "raw_predictions_persisted": False,
            "seed_metrics": seed_reports,
            "three_seed_statistics": {
                name: _summary(values) for name, values in metric_fields.items()
            },
        }
    assert selection_group_ids is not None and selection_trajectories is not None
    canonical_row = cast(dict[str, object], domain_reports["canonical"])
    canonical_binary = cast(Mapping[str, object], canonical_row["binary_metrics"])
    canonical_auprc = cast(
        float, cast(Mapping[str, object], canonical_binary["failure_auprc"])["value"]
    )
    canonical_success = float(np.mean(selection_vectors["canonical"]))
    for domain, value in domain_reports.items():
        row = cast(dict[str, object], value)
        row_binary = cast(Mapping[str, object], row["binary_metrics"])
        auprc = cast(
            float,
            cast(Mapping[str, object], row_binary["failure_auprc"])["value"],
        )
        row["robustness"] = {
            "failure_auprc_drop_from_canonical": canonical_auprc - auprc,
            "selected_success_drop_from_canonical": canonical_success
            - float(np.mean(selection_vectors[domain])),
            "selected_success_paired_bootstrap_vs_canonical": _trajectory_bootstrap(
                selection_vectors[domain],
                selection_vectors["canonical"],
                selection_trajectories,
            ),
        }
    body: dict[str, object] = {
        "calibration": [
            seed.calibration.to_dict() for seed in calibrated_seeds if seed.calibration
        ],
        "checkpoint_digests": [seed.checkpoint_digest for seed in calibrated_seeds],
        "domains": domain_reports,
        "internal_test_opened_once": True,
        "ensemble_semantic": (
            "mean_of_per_seed_validation_temperature_calibrated_probabilities_v1"
        ),
        "external_dataset_opened": False,
        "feature_cache_digest": first_binding.feature_cache_digest,
        "git_sha": first_binding.git_sha,
        "model_config_digest": model_config.content_digest,
        "model_id": model_config.model_id,
        "schema_version": "1.0",
        "seed_order": [0, 1, 2],
        "split_digest": first_binding.split_digest,
        "structured_dataset_digest": first_binding.structured_dataset_digest,
        "test_domain_order": list(TEST_DOMAIN_IDS),
        "training_or_selection_inputs": "m3a_development_only",
        "training_config_digest": training_config.content_digest,
        "visual_dataset_digest": first_binding.visual_dataset_digest,
        "thresholds": [
            seed.thresholds.to_dict() for seed in calibrated_seeds if seed.thresholds
        ],
    }
    report = {**body, "content_digest": _digest(body, "M4BInternalEvaluationV1")}
    destination = Path(output_dir) / "internal-evaluation.json"
    if destination.exists():
        _fail("internal evaluation", "completed output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "LoadedPromotedSeedV1",
    "VisualEvaluationError",
    "evaluate_promoted_visual_ensemble",
]
