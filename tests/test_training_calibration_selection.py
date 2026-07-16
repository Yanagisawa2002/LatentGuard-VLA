"""CPU-only tests for calibration, selection, statistics, and strict reports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latentguard.training.calibration import (
    TemperatureCalibrationStateV1,
    fit_temperature_scaling,
    fit_validation_thresholds,
    frozen_thresholds_from_dict,
)
from latentguard.training.evaluation import (
    EvaluationMetadataV1,
    evaluate_action_verifier_logits,
)
from latentguard.training.metrics import GroupCandidate, evaluate_binary_metrics
from latentguard.training.reporting import (
    ReportValidationError,
    StrictReportV1,
    load_strict_report,
    save_strict_report,
)
from latentguard.training.selection import (
    DEFAULT_FIVE_SEEDS,
    DEFAULT_MODEL_TYPES,
    SeedValidationResult,
    select_model_architecture,
    selection_record_from_dict,
)
from latentguard.training.statistics import (
    group_bootstrap_top1_success_difference,
    paired_trajectory_bootstrap,
)

_DATASET_DIGEST = "sha256:" + "1" * 64
_SPLIT_DIGEST = "sha256:" + "2" * 64
_MODEL_DIGEST = "sha256:" + "3" * 64
_TRAINING_DIGEST = "sha256:" + "4" * 64
_PREPROCESSING_DIGEST = "sha256:" + "5" * 64
_SELECTION_DIGEST = "sha256:" + "6" * 64
_CHECKPOINT_DIGEST = "sha256:" + "7" * 64
_CHECKPOINT_IDENTITY = "checkpoint-validation-fixture"


def _calibration_binding() -> dict[str, str]:
    return {
        "checkpoint_identity": _CHECKPOINT_IDENTITY,
        "checkpoint_content_digest": _CHECKPOINT_DIGEST,
        "model_config_digest": _MODEL_DIGEST,
        "preprocessing_digest": _PREPROCESSING_DIGEST,
    }


def test_temperature_scaling_is_validation_only_positive_and_rank_preserving() -> None:
    logits = np.asarray([-5.0, -2.0, 2.0, 5.0], dtype=np.float64)
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    before = evaluate_binary_metrics(
        targets, 1.0 / (1.0 + np.exp(-logits))
    ).negative_log_likelihood.value

    state = fit_temperature_scaling(
        logits,
        targets,
        split="validation",
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
        **_calibration_binding(),
    )
    calibrated = state.apply(logits)
    after = evaluate_binary_metrics(targets, calibrated).negative_log_likelihood.value

    assert state.temperature > 0.0
    assert np.array_equal(np.argsort(logits), np.argsort(calibrated))
    assert before is not None and after is not None and after <= before
    restored = TemperatureCalibrationStateV1.from_dict(state.to_dict())
    assert restored == state
    changed = dict(state.to_dict())
    changed["temperature"] = state.temperature + 1.0
    with pytest.raises(ValueError, match="content changed"):
        TemperatureCalibrationStateV1.from_dict(changed)
    with pytest.raises(ValueError, match="validation"):
        fit_temperature_scaling(
            logits,
            targets,
            split="test",
            dataset_digest=_DATASET_DIGEST,
            split_digest=_SPLIT_DIGEST,
            **_calibration_binding(),
        )


def test_validation_threshold_policies_are_deterministic_and_never_fit_test() -> None:
    probabilities = np.asarray([0.9, 0.6, 0.7, 0.1], dtype=np.float64)
    targets = np.asarray([1, 1, 0, 0], dtype=np.int64)
    calibration_digest = "sha256:" + "7" * 64
    prediction_digest = "sha256:" + "8" * 64

    state = fit_validation_thresholds(
        probabilities,
        targets,
        split="validation",
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
        calibration_digest=calibration_digest,
        validation_prediction_digest=prediction_digest,
        target_failure_recall=1.0,
    )

    assert state.maximum_balanced_accuracy.threshold == pytest.approx(0.9)
    assert state.target_failure_recall.threshold == pytest.approx(0.6)
    assert state.target_failure_recall.target_met is True
    assert state.target_failure_recall.achieved_failure_recall.value == pytest.approx(
        1.0
    )
    assert frozen_thresholds_from_dict(state.to_dict()) == state
    changed = dict(state.to_dict())
    changed["comparison_semantic"] = "changed"
    with pytest.raises(ValueError):
        frozen_thresholds_from_dict(changed)
    with pytest.raises(ValueError, match="validation"):
        fit_validation_thresholds(
            probabilities,
            targets,
            split="test",
            dataset_digest=_DATASET_DIGEST,
            split_digest=_SPLIT_DIGEST,
            calibration_digest=calibration_digest,
            validation_prediction_digest=prediction_digest,
        )


def _selection_results() -> tuple[SeedValidationResult, ...]:
    results: list[SeedValidationResult] = []
    primary = {
        "state_only_mlp": 0.45,
        "action_only_mlp": 0.55,
        "state_action_mlp": 0.70,
        "temporal_state_action_verifier": 0.65,
    }
    parameters = {
        "state_only_mlp": 40_000,
        "action_only_mlp": 51_000,
        "state_action_mlp": 122_000,
        "temporal_state_action_verifier": 325_000,
    }
    for model in DEFAULT_MODEL_TYPES:
        for seed in DEFAULT_FIVE_SEEDS:
            results.append(
                SeedValidationResult(
                    model_type=model,
                    seed=seed,
                    failure_auprc=primary[model] + seed * 0.001,
                    brier_score=0.30 - primary[model] / 10.0,
                    pairwise_concordance=primary[model],
                    parameter_count=parameters[model],
                    model_config_digest="sha256:" + f"{seed + 10:064x}"[-64:],
                    training_config_digest=_TRAINING_DIGEST,
                    checkpoint_identity=f"checkpoint-{model}-{seed}",
                    checkpoint_content_digest=("sha256:" + f"{seed + 100:064x}"[-64:]),
                    checkpoint_kind="best",
                    checkpoint_epoch=seed,
                    validation_prediction_digest=(
                        "sha256:" + f"{seed + 200:064x}"[-64:]
                    ),
                )
            )
    # Configuration must be stable across seeds for each model.
    for model_index, model in enumerate(DEFAULT_MODEL_TYPES):
        digest = "sha256:" + f"{model_index + 10:064x}"[-64:]
        results = [
            SeedValidationResult(
                model_type=item.model_type,
                seed=item.seed,
                failure_auprc=item.failure_auprc,
                brier_score=item.brier_score,
                pairwise_concordance=item.pairwise_concordance,
                parameter_count=item.parameter_count,
                model_config_digest=(
                    digest if item.model_type == model else item.model_config_digest
                ),
                training_config_digest=item.training_config_digest,
                checkpoint_identity=item.checkpoint_identity,
                checkpoint_content_digest=item.checkpoint_content_digest,
                checkpoint_kind=item.checkpoint_kind,
                checkpoint_epoch=item.checkpoint_epoch,
                validation_prediction_digest=item.validation_prediction_digest,
            )
            for item in results
        ]
    return tuple(results)


def test_selection_uses_five_seed_validation_aggregates_not_best_seed() -> None:
    record = select_model_architecture(
        _selection_results(),
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
    )

    assert record.selected_model_type == "state_action_mlp"
    assert record.expected_seeds == DEFAULT_FIVE_SEEDS
    assert len(record.candidates) == 4
    assert all(candidate.failure_auprc.count == 5 for candidate in record.candidates)
    assert record.frozen is True
    assert record.content_digest.startswith("sha256:")
    assert selection_record_from_dict(record.to_dict()) == record
    changed = dict(record.to_dict())
    changed["selected_model_type"] = "state_only_mlp"
    with pytest.raises(ValueError):
        selection_record_from_dict(changed)


def test_selection_record_rejects_forged_noncanonical_winner() -> None:
    record = select_model_architecture(
        _selection_results(),
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
    )
    forged = next(
        candidate
        for candidate in record.candidates
        if candidate.model_type == "state_only_mlp"
    )

    with pytest.raises(ValueError, match="canonical validation aggregate winner"):
        replace(
            record,
            selected_model_type=forged.model_type,
            selected_model_config_digest=forged.model_config_digest,
        )


def test_selection_record_rejects_missing_canonical_candidate() -> None:
    record = select_model_architecture(
        _selection_results(),
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
    )

    with pytest.raises(ValueError, match="canonical four-architecture inventory"):
        replace(record, candidates=record.candidates[:-1])


def test_selection_rejects_test_metrics_and_incomplete_seed_matrix() -> None:
    with pytest.raises(ValueError, match="validation"):
        SeedValidationResult(
            model_type="state_only_mlp",
            seed=0,
            failure_auprc=0.5,
            brier_score=0.2,
            pairwise_concordance=0.6,
            parameter_count=10,
            model_config_digest=_MODEL_DIGEST,
            training_config_digest=_TRAINING_DIGEST,
            checkpoint_identity="checkpoint",
            checkpoint_content_digest=_CHECKPOINT_DIGEST,
            checkpoint_kind="best",
            checkpoint_epoch=0,
            validation_prediction_digest=_SELECTION_DIGEST,
            split="test",
        )
    with pytest.raises(ValueError, match="seed inventory"):
        select_model_architecture(
            _selection_results()[:-1],
            dataset_digest=_DATASET_DIGEST,
            split_digest=_SPLIT_DIGEST,
        )


def test_paired_bootstrap_resamples_whole_trajectories_deterministically() -> None:
    targets = np.asarray([0, 1] * 6, dtype=np.int64)
    first = np.asarray([0.1, 0.9] * 6, dtype=np.float64)
    second = np.asarray([0.8, 0.2] * 6, dtype=np.float64)
    trajectories = [value for index in range(6) for value in (f"t{index}",) * 2]

    auprc = paired_trajectory_bootstrap(
        targets,
        first,
        second,
        trajectories,
        metric="failure_auprc",
        resamples=100,
        seed=9,
    )
    repeated = paired_trajectory_bootstrap(
        targets,
        first,
        second,
        trajectories,
        metric="failure_auprc",
        resamples=100,
        seed=9,
    )
    brier = paired_trajectory_bootstrap(
        targets,
        first,
        second,
        trajectories,
        metric="brier_score",
        resamples=100,
        seed=9,
    )

    assert auprc == repeated
    assert auprc.observed_difference.value is not None
    assert auprc.observed_difference.value > 0.0
    assert auprc.valid_resamples == 100
    assert brier.observed_difference.value is not None
    assert brier.observed_difference.value < 0.0
    assert auprc.sampling_unit == "original_source_trajectory"


def _group_records(good: bool) -> tuple[GroupCandidate, ...]:
    records: list[GroupCandidate] = []
    for index in range(6):
        trajectory = f"t{index}"
        group = f"g{index}"
        records.extend(
            [
                GroupCandidate(f"s{index}", group, trajectory, "source", 0, 0.05),
                GroupCandidate(
                    f"ok{index}",
                    group,
                    trajectory,
                    "corrupted",
                    0,
                    0.1 if good else 0.9,
                ),
                GroupCandidate(
                    f"bad{index}",
                    group,
                    trajectory,
                    "corrupted",
                    1,
                    0.9 if good else 0.1,
                ),
            ]
        )
    return tuple(records)


def test_group_bootstrap_top1_uses_trajectory_clusters() -> None:
    report = group_bootstrap_top1_success_difference(
        _group_records(True),
        _group_records(False),
        resamples=50,
        seed=11,
    )

    assert report.observed_difference.value == pytest.approx(1.0)
    assert report.confidence_lower == pytest.approx(1.0)
    assert report.confidence_upper == pytest.approx(1.0)
    assert report.valid_resamples == 50


def test_strict_report_round_trip_is_idempotent_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    report = StrictReportV1(
        report_type="aggregate_metrics_v1",
        payload={"count": 5, "metric": 0.75},
    )
    destination = tmp_path / "aggregate.json"

    assert save_strict_report(report, destination) == destination
    assert save_strict_report(report, destination) == destination
    assert load_strict_report(destination) == report
    value = json.loads(destination.read_text(encoding="utf-8"))
    value["payload"]["metric"] = 0.1
    destination.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReportValidationError, match="content changed"):
        load_strict_report(destination)
    with pytest.raises(ReportValidationError, match="runtime paths"):
        StrictReportV1(
            report_type="aggregate_metrics_v1",
            payload={"dataset_path": "D:/private/dataset"},
        )


def _evaluation_metadata() -> tuple[EvaluationMetadataV1, ...]:
    return (
        EvaluationMetadataV1(
            0, "s0", "g0", "t0", "test", "source", None, None, "early"
        ),
        EvaluationMetadataV1(
            1, "c0", "g0", "t0", "test", "corrupted", "noise", "mild", "early"
        ),
        EvaluationMetadataV1(
            2, "c1", "g0", "t0", "test", "corrupted", "bias", "severe", "early"
        ),
        EvaluationMetadataV1(3, "s1", "g1", "t1", "test", "source", None, None, "late"),
        EvaluationMetadataV1(
            4, "c2", "g1", "t1", "test", "corrupted", "noise", "mild", "late"
        ),
        EvaluationMetadataV1(
            5, "c3", "g1", "t1", "test", "corrupted", "bias", "severe", "late"
        ),
    )


def test_test_evaluation_consumes_frozen_artifacts_and_has_no_fit_path() -> None:
    logits = np.asarray([-3.0, -2.0, 2.0, -3.0, -1.0, 3.0])
    targets = np.asarray([0, 0, 1, 0, 0, 1], dtype=np.int64)
    calibration = fit_temperature_scaling(
        logits,
        targets,
        split="validation",
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
        **_calibration_binding(),
    )
    calibrated = calibration.apply(logits)
    thresholds = fit_validation_thresholds(
        calibrated,
        targets,
        split="validation",
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
        calibration_digest=calibration.content_digest,
        validation_prediction_digest=calibration.validation_prediction_digest,
    )

    report, predictions = evaluate_action_verifier_logits(
        logits,
        targets,
        _evaluation_metadata(),
        split="test",
        dataset_digest=_DATASET_DIGEST,
        split_digest=_SPLIT_DIGEST,
        checkpoint_identity=_CHECKPOINT_IDENTITY,
        checkpoint_content_digest=_CHECKPOINT_DIGEST,
        model_config_digest=_MODEL_DIGEST,
        preprocessing_digest=_PREPROCESSING_DIGEST,
        calibration=calibration,
        thresholds=thresholds,
        selection_digest=_SELECTION_DIGEST,
    )

    assert report.calibrated is not None
    assert len(report.threshold_evaluations) == 2
    assert len(predictions) == 6
    assert report.uncalibrated.corrupted_only_metrics.sample_count == 4
    assert report.uncalibrated.group_ranking.group_count == 2
    assert "source_trajectory" in {
        item.dimension for item in report.uncalibrated.slices
    }
    assert all("state" not in item.to_dict() for item in predictions)
    mismatched_thresholds = replace(
        thresholds,
        validation_prediction_digest="sha256:" + "9" * 64,
    )
    with pytest.raises(ValueError, match="prediction binding"):
        evaluate_action_verifier_logits(
            logits,
            targets,
            _evaluation_metadata(),
            split="test",
            dataset_digest=_DATASET_DIGEST,
            split_digest=_SPLIT_DIGEST,
            checkpoint_identity=_CHECKPOINT_IDENTITY,
            checkpoint_content_digest=_CHECKPOINT_DIGEST,
            model_config_digest=_MODEL_DIGEST,
            preprocessing_digest=_PREPROCESSING_DIGEST,
            calibration=calibration,
            thresholds=mismatched_thresholds,
            selection_digest=_SELECTION_DIGEST,
        )
    with pytest.raises(ValueError, match="frozen"):
        evaluate_action_verifier_logits(
            logits,
            targets,
            _evaluation_metadata(),
            split="test",
            dataset_digest=_DATASET_DIGEST,
            split_digest=_SPLIT_DIGEST,
            checkpoint_identity="trusted-checkpoint-id",
            checkpoint_content_digest=_CHECKPOINT_DIGEST,
            model_config_digest=_MODEL_DIGEST,
            preprocessing_digest=_PREPROCESSING_DIGEST,
        )
