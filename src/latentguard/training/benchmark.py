"""Fixed multi-seed M3B benchmark orchestration and frozen test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray

from latentguard.action_verifier import DatasetSplit
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.baselines import (
    fit_action_magnitude_baseline,
    fit_constant_baselines,
)
from latentguard.training.calibration import (
    compute_validation_prediction_digest,
    fit_temperature_scaling,
    fit_validation_thresholds,
    frozen_thresholds_from_dict,
    temperature_calibration_from_dict,
)
from latentguard.training.checkpoint import (
    CheckpointBindingV1,
    compute_checkpoint_content_digest,
    inspect_training_checkpoint,
    load_training_checkpoint,
)
from latentguard.training.config import (
    BenchmarkConfig,
    ModelConfig,
    ModelType,
    ResolvedModelConfig,
    TrainingConfig,
    load_benchmark_config,
    load_model_config,
    load_training_config,
)
from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
from latentguard.training.evaluation import (
    ActionVerifierEvaluationReportV1,
    CompactPredictionV1,
    EvaluationMetadataV1,
    ProbabilityEvaluationReport,
    evaluate_action_verifier_logits,
)
from latentguard.training.inference import (
    SplitInferenceResultV1,
    infer_action_verifier_split,
)
from latentguard.training.losses import build_failure_loss
from latentguard.training.manifest import (
    TrainingRunEnvironment,
    TrainingRunIdentity,
    TrainingRunManifest,
    TrainingRunStatus,
)
from latentguard.training.metrics import (
    GroupCandidate,
    evaluate_binary_metrics,
    evaluate_coverage_risk,
    evaluate_group_ranking,
    evaluate_slices,
)
from latentguard.training.models import build_model, resolve_model_config
from latentguard.training.preprocessing import (
    PreprocessingStateV1,
    fit_preprocessing_state,
    load_preprocessing_state,
    save_preprocessing_state,
)
from latentguard.training.reporting import (
    StrictReportV1,
    load_strict_report,
    save_strict_report,
)
from latentguard.training.selection import (
    SeedValidationResult,
    SelectionRecordV1,
    aggregate_five_seed_values,
    select_model_architecture,
    selection_record_from_dict,
)
from latentguard.training.statistics import (
    group_bootstrap_top1_success_difference,
    paired_trajectory_bootstrap,
)
from latentguard.training.trainer import (
    TRAINING_HISTORY_FORMAT,
    TRAINING_HISTORY_VERSION,
    EpochRecordV1,
    run_forward_backward_gate,
    run_tiny_overfit_gate,
    set_training_seed,
    train_action_verifier,
)

BENCHMARK_REPORT_TYPE = "m3b_benchmark_summary_v1"
MODEL_CONFIG_FILENAMES: Mapping[ModelType, str] = {
    ModelType.STATE_ONLY_MLP: "state-only-mlp.json",
    ModelType.ACTION_ONLY_MLP: "action-only-mlp.json",
    ModelType.STATE_ACTION_MLP: "state-action-mlp.json",
    ModelType.TEMPORAL_STATE_ACTION_VERIFIER: "state-action-transformer.json",
}


class BenchmarkError(RuntimeError):
    """Raised when a benchmark gate, resume identity, or report is invalid."""


def _fail(context: str, reason: str) -> NoReturn:
    raise BenchmarkError(f"{context}: {reason}")


def _content_digest(payload: object, *, context: str) -> str:
    encoded = canonical_json_bytes(payload, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _file_digest(path: Path, *, context: str) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected a regular file")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise BenchmarkError(f"{context}: could not hash file: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def _load_json_object(path: Path, *, context: str) -> dict[str, object]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected a regular JSON file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{context}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail(context, "expected a JSON object")
    return value


def _atomic_json(path: Path, payload: object) -> Path:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    staging.write_text(serialized, encoding="utf-8", newline="\n")
    staging.replace(destination)
    return destination


def _atomic_text(path: Path, content: str) -> Path:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.rstrip() + "\n"
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_text(encoding="utf-8") != normalized
        ):
            _fail("completed text artifact", "existing content differs")
        return destination
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    staging.write_text(normalized, encoding="utf-8", newline="\n")
    staging.replace(destination)
    return destination


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return result.stdout.strip()


def _environment() -> TrainingRunEnvironment:
    gpu_model = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None
    )
    gpu_driver_version = None
    if torch.cuda.is_available():
        try:
            gpu_driver_version = (
                subprocess.run(
                    (
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                        "--id=0",
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=30,
                )
                .stdout.splitlines()[0]
                .strip()
            )
        except (OSError, subprocess.SubprocessError, IndexError):
            gpu_driver_version = None
    try:
        package_version = importlib.metadata.version("latentguard-vla")
    except importlib.metadata.PackageNotFoundError:
        package_version = "editable-uninstalled"
    return TrainingRunEnvironment(
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_model=gpu_model,
        gpu_driver_version=gpu_driver_version,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        package_versions=(
            f"latentguard-vla={package_version}",
            f"numpy={np.__version__}",
            f"torch={torch.__version__}",
        ),
        launch_command=tuple(sys.argv),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        nondeterministic_operations=(),
    )


def _resolved_training_config(
    base: TrainingConfig, args: argparse.Namespace
) -> TrainingConfig:
    if args.mode == "full" and any(
        value is not None
        for value in (args.max_steps, args.max_epochs, args.limit_samples)
    ):
        _fail(
            "full benchmark",
            "bounded max-step, max-epoch, and sample-limit overrides are smoke-only",
        )
    if args.mode == "smoke":
        max_steps = 50 if args.max_steps is None else args.max_steps
        max_epochs = (
            min(base.max_epochs, 10) if args.max_epochs is None else args.max_epochs
        )
    else:
        max_steps = base.max_steps
        max_epochs = base.max_epochs
    return replace(
        base,
        max_steps=max_steps,
        max_epochs=max_epochs,
        limit_samples=(
            base.limit_samples if args.limit_samples is None else args.limit_samples
        ),
        num_workers=(
            base.num_workers if args.num_workers is None else args.num_workers
        ),
    )


def _load_model_configs(
    directory: Path, benchmark: BenchmarkConfig
) -> dict[ModelType, ModelConfig]:
    result: dict[ModelType, ModelConfig] = {}
    for model_type in benchmark.model_types:
        path = Path(directory) / MODEL_CONFIG_FILENAMES[model_type]
        configuration = load_model_config(path)
        if configuration.model_type is not model_type:
            _fail("model config", f"{path.name} has a different model type")
        result[model_type] = configuration
    return result


def _run_identity(
    dataset: AcceptedActionVerifierDatasetV1,
    preprocessing: PreprocessingStateV1,
    resolved_model: ResolvedModelConfig,
    training_config: TrainingConfig,
    seed: int,
) -> TrainingRunIdentity:
    return TrainingRunIdentity(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved_model.content_digest,
        training_config_digest=training_config.content_digest,
        seed=seed,
        code_semantic_version="direct_action_verifier_training_v1",
    )


def _checkpoint_binding(
    identity: TrainingRunIdentity,
) -> CheckpointBindingV1:
    return CheckpointBindingV1(
        dataset_digest=identity.dataset_digest,
        split_digest=identity.split_digest,
        preprocessing_digest=identity.preprocessing_digest,
        model_config_digest=identity.model_config_digest,
        training_config_digest=identity.training_config_digest,
        run_manifest_identity=identity.run_id,
    )


def _positive_class_weight(
    dataset: AcceptedActionVerifierDatasetV1,
    training_config: TrainingConfig,
) -> float | None:
    targets = torch.tensor(
        [
            dataset[index].failure_target
            for index in dataset.indices_for_split(DatasetSplit.TRAIN)
        ],
        dtype=torch.float32,
    )
    loss = build_failure_loss(training_config.loss_mode, targets)
    return (
        None
        if loss.positive_class_weight is None
        else float(loss.positive_class_weight.item())
    )


def _seed_result_from_report(report: StrictReportV1) -> SeedValidationResult:
    value = report.payload
    expected = {
        "brier_score",
        "checkpoint_content_digest",
        "checkpoint_epoch",
        "checkpoint_identity",
        "checkpoint_kind",
        "failure_auprc",
        "model_config_digest",
        "model_type",
        "pairwise_concordance",
        "parameter_count",
        "seed",
        "split",
        "training_config_digest",
        "validation_prediction_digest",
    }
    if set(value) != expected:
        _fail("validation result", "unexpected or missing fields")
    try:
        return SeedValidationResult(
            model_type=cast(str, value["model_type"]),
            seed=cast(int, value["seed"]),
            failure_auprc=cast(float, value["failure_auprc"]),
            brier_score=cast(float, value["brier_score"]),
            pairwise_concordance=cast(float, value["pairwise_concordance"]),
            parameter_count=cast(int, value["parameter_count"]),
            model_config_digest=cast(str, value["model_config_digest"]),
            training_config_digest=cast(str, value["training_config_digest"]),
            checkpoint_identity=cast(str, value["checkpoint_identity"]),
            checkpoint_content_digest=cast(str, value["checkpoint_content_digest"]),
            checkpoint_kind=cast(str, value["checkpoint_kind"]),
            checkpoint_epoch=cast(int, value["checkpoint_epoch"]),
            validation_prediction_digest=cast(
                str, value["validation_prediction_digest"]
            ),
            split=cast(str, value["split"]),
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"validation result: {exc}") from exc


def _save_predictions(
    predictions: Sequence[CompactPredictionV1], path: Path, *, report_type: str
) -> StrictReportV1:
    report = StrictReportV1(
        report_type,
        {"predictions": [item.to_dict() for item in predictions]},
    )
    save_strict_report(report, path)
    return report


def _validation_result(
    *,
    model_type: ModelType,
    seed: int,
    resolved_model: ResolvedModelConfig,
    training_config: TrainingConfig,
    identity: TrainingRunIdentity,
    report: ActionVerifierEvaluationReportV1,
    checkpoint_content_digest: str,
    checkpoint_epoch: int,
    validation_prediction_digest: str,
) -> SeedValidationResult:
    corrupted = report.uncalibrated.corrupted_only_metrics
    ranking = report.uncalibrated.group_ranking
    if (
        corrupted.failure_auprc.value is None
        or corrupted.brier_score.value is None
        or ranking.pairwise_success_over_failure_concordance.value is None
    ):
        _fail("validation", "required selection metric is undefined")
    return SeedValidationResult(
        model_type=model_type.value,
        seed=seed,
        failure_auprc=corrupted.failure_auprc.value,
        brier_score=corrupted.brier_score.value,
        pairwise_concordance=(ranking.pairwise_success_over_failure_concordance.value),
        parameter_count=resolved_model.parameter_count,
        model_config_digest=resolved_model.content_digest,
        training_config_digest=training_config.content_digest,
        checkpoint_identity=identity.run_id,
        checkpoint_content_digest=checkpoint_content_digest,
        checkpoint_kind="best",
        checkpoint_epoch=checkpoint_epoch,
        validation_prediction_digest=validation_prediction_digest,
    )


def _run_directory(
    root: Path, model_type: ModelType, seed: int, *, force_rerun: bool
) -> Path:
    canonical = root / "runs" / model_type.value / f"seed-{seed}"
    if not force_rerun:
        return canonical
    rerun_root = root / "reruns" / model_type.value
    ordinal = 1
    while (rerun_root / f"seed-{seed}-rerun-{ordinal}").exists():
        ordinal += 1
    return rerun_root / f"seed-{seed}-rerun-{ordinal}"


def _validate_run_manifest(
    path: Path,
    *,
    identity: TrainingRunIdentity,
    dataset: AcceptedActionVerifierDatasetV1,
    resolved_model: ResolvedModelConfig,
    training_config: TrainingConfig,
    positive_class_weight: float | None,
    git_sha: str,
    branch: str,
    expected_status: TrainingRunStatus,
) -> dict[str, object]:
    payload = _load_json_object(path, context="run manifest")
    required = {
        "schema_version",
        "run_id",
        "identity",
        "acceptance_report_digest",
        "model_configuration",
        "training_configuration",
        "positive_class_weight",
        "git_sha",
        "branch",
        "environment",
        "started_at_utc",
        "finished_at_utc",
        "status",
    }
    if set(payload) != required:
        _fail("run manifest", "unexpected or missing fields")
    expected_finished = expected_status is TrainingRunStatus.COMPLETED
    if (
        payload["schema_version"] != "1.0"
        or payload["run_id"] != identity.run_id
        or payload["identity"] != identity.as_mapping()
        or payload["acceptance_report_digest"] != dataset.acceptance_report_digest
        or payload["model_configuration"] != resolved_model.as_mapping()
        or payload["training_configuration"] != training_config.as_mapping()
        or payload["positive_class_weight"] != positive_class_weight
        or payload["git_sha"] != git_sha
        or payload["branch"] != branch
        or payload["status"] != expected_status.value
        or (payload["finished_at_utc"] is not None) != expected_finished
        or not isinstance(payload["started_at_utc"], str)
        or not payload["started_at_utc"]
    ):
        _fail("run manifest", "semantic provenance differs")
    environment_payload = payload["environment"]
    environment_fields = {
        "cublas_workspace_config",
        "cuda_version",
        "deterministic_algorithms",
        "gpu_driver_version",
        "gpu_model",
        "hostname",
        "launch_command",
        "nondeterministic_operations",
        "numpy_version",
        "package_versions",
        "python_version",
        "pytorch_version",
    }
    if not isinstance(environment_payload, dict) or set(environment_payload) != (
        environment_fields
    ):
        _fail("run manifest", "environment inventory is malformed")
    packages = environment_payload["package_versions"]
    launch_command = environment_payload["launch_command"]
    nondeterministic = environment_payload["nondeterministic_operations"]
    if not all(
        isinstance(value, list)
        for value in (packages, launch_command, nondeterministic)
    ):
        _fail("run manifest", "environment sequences are malformed")
    try:
        environment = TrainingRunEnvironment(
            hostname=cast(str, environment_payload["hostname"]),
            python_version=cast(str, environment_payload["python_version"]),
            numpy_version=cast(str, environment_payload["numpy_version"]),
            pytorch_version=cast(str, environment_payload["pytorch_version"]),
            cuda_version=cast(str | None, environment_payload["cuda_version"]),
            gpu_model=cast(str | None, environment_payload["gpu_model"]),
            gpu_driver_version=cast(
                str | None, environment_payload["gpu_driver_version"]
            ),
            cublas_workspace_config=cast(
                str | None, environment_payload["cublas_workspace_config"]
            ),
            package_versions=tuple(cast(list[str], packages)),
            launch_command=tuple(cast(list[str], launch_command)),
            deterministic_algorithms=cast(
                bool, environment_payload["deterministic_algorithms"]
            ),
            nondeterministic_operations=tuple(cast(list[str], nondeterministic)),
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"run manifest: invalid environment: {exc}") from exc
    if environment.as_mapping() != environment_payload:
        _fail("run manifest", "environment values are malformed")
    return payload


def _compact_predictions_from_report(
    report: StrictReportV1, *, expected_split: str
) -> tuple[CompactPredictionV1, ...]:
    if set(report.payload) != {"predictions"} or not isinstance(
        report.payload["predictions"], list
    ):
        _fail("compact predictions", "unexpected or missing fields")
    expected_fields = {
        "calibrated_failure_probability",
        "candidate_type",
        "failure_target",
        "group_id",
        "raw_logit",
        "sample_id",
        "source_trajectory_id",
        "split",
        "uncalibrated_failure_probability",
    }
    predictions: list[CompactPredictionV1] = []
    for index, raw in enumerate(report.payload["predictions"]):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            _fail("compact predictions", f"row {index} has an invalid inventory")
        text_fields = {
            name: raw[name]
            for name in (
                "candidate_type",
                "group_id",
                "sample_id",
                "source_trajectory_id",
                "split",
            )
        }
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in text_fields.values()
        ):
            _fail("compact predictions", f"row {index} has invalid text")
        failure_target = raw["failure_target"]
        raw_logit = raw["raw_logit"]
        uncalibrated = raw["uncalibrated_failure_probability"]
        calibrated = raw["calibrated_failure_probability"]
        if (
            type(failure_target) is not int
            or isinstance(raw_logit, bool)
            or not isinstance(raw_logit, (int, float))
            or isinstance(uncalibrated, bool)
            or not isinstance(uncalibrated, (int, float))
            or (
                calibrated is not None
                and (
                    isinstance(calibrated, bool)
                    or not isinstance(calibrated, (int, float))
                )
            )
        ):
            _fail("compact predictions", f"row {index} has invalid numerics")
        prediction = CompactPredictionV1(
            sample_id=cast(str, raw["sample_id"]),
            group_id=cast(str, raw["group_id"]),
            source_trajectory_id=cast(str, raw["source_trajectory_id"]),
            split=cast(str, raw["split"]),
            candidate_type=cast(str, raw["candidate_type"]),
            failure_target=failure_target,
            raw_logit=float(raw_logit),
            uncalibrated_failure_probability=float(uncalibrated),
            calibrated_failure_probability=(
                None if calibrated is None else float(calibrated)
            ),
        )
        if prediction.split != expected_split or prediction.candidate_type not in {
            "source",
            "corrupted",
        }:
            _fail("compact predictions", f"row {index} has invalid semantics")
        predictions.append(prediction)
    if not predictions or len({item.sample_id for item in predictions}) != len(
        predictions
    ):
        _fail("compact predictions", "sample inventory is empty or duplicated")
    return tuple(predictions)


def _compact_prediction_digest(
    predictions: Sequence[CompactPredictionV1],
) -> str:
    return _content_digest(
        {
            "predictions": [
                item.to_dict()
                for item in sorted(predictions, key=lambda item: item.sample_id)
            ],
            "semantic": "compact_action_verifier_predictions_v1",
        },
        context="CompactPredictionSetV1",
    )


def _validate_completed_run(
    *,
    run_root: Path,
    model_type: ModelType,
    seed: int,
    identity: TrainingRunIdentity,
    dataset: AcceptedActionVerifierDatasetV1,
    resolved_model: ResolvedModelConfig,
    training_config: TrainingConfig,
    positive_class_weight: float | None,
    git_sha: str,
    branch: str,
) -> SeedValidationResult:
    expected_root_entries = {
        "checkpoints",
        "completed.json",
        "history.json",
        "run-manifest.json",
        "training-summary.json",
        "validation-evaluation.json",
        "validation-predictions.json",
        "validation-result.json",
    }
    if (
        not run_root.is_dir()
        or run_root.is_symlink()
        or {item.name for item in run_root.iterdir()} != expected_root_entries
    ):
        _fail("completed run", "run-directory inventory differs")
    checkpoint_root = run_root / "checkpoints"
    if (
        not checkpoint_root.is_dir()
        or checkpoint_root.is_symlink()
        or {item.name for item in checkpoint_root.iterdir()}
        != {"best.pt", "final.pt", "periodic.pt"}
    ):
        _fail("completed run", "checkpoint inventory differs")
    result_path = run_root / "validation-result.json"
    summary_path = run_root / "training-summary.json"
    completed_path = run_root / "completed.json"
    completion = _load_json_object(completed_path, context="completed run marker")
    if set(completion) != {
        "best_checkpoint_content_digest",
        "final_checkpoint_content_digest",
        "history_file_digest",
        "periodic_checkpoint_content_digest",
        "run_id",
        "run_manifest_file_digest",
        "training_summary_digest",
        "validation_evaluation_digest",
        "validation_predictions_digest",
        "validation_result_digest",
    }:
        _fail("completed run marker", "unexpected or missing fields")
    result_report = load_strict_report(
        result_path,
        expected_report_type="seed_validation_result_v1",
        expected_content_digest=cast(str, completion["validation_result_digest"]),
    )
    result = _seed_result_from_report(result_report)
    summary_report = load_strict_report(
        summary_path,
        expected_report_type="training_run_summary_v1",
        expected_content_digest=cast(str, completion["training_summary_digest"]),
    )
    summary = summary_report.payload
    expected_summary_fields = {
        "acceptance_report_digest",
        "best_checkpoint_content_digest",
        "best_epoch",
        "best_validation_failure_auprc",
        "completed",
        "cumulative_epoch_duration_seconds",
        "environment",
        "final_checkpoint_content_digest",
        "final_epoch",
        "final_validation_corrupted_brier_score",
        "final_validation_corrupted_failure_auprc",
        "final_validation_loss",
        "global_step",
        "model_type",
        "peak_gpu_memory_bytes",
        "positive_class_weight",
        "resolved_model_configuration",
        "resolved_training_configuration",
        "run_id",
        "seed",
        "training_duration_seconds_current_attempt",
    }
    if set(summary) != expected_summary_fields:
        _fail("completed run", "training-summary inventory differs")
    validation_evaluation = load_strict_report(
        run_root / "validation-evaluation.json",
        expected_report_type="validation_evaluation_v1",
        expected_content_digest=cast(str, completion["validation_evaluation_digest"]),
    )
    validation_predictions_report = load_strict_report(
        run_root / "validation-predictions.json",
        expected_report_type="validation_predictions_v1",
        expected_content_digest=cast(str, completion["validation_predictions_digest"]),
    )
    validation_predictions = _compact_predictions_from_report(
        validation_predictions_report, expected_split="validation"
    )
    evaluation_payload = validation_evaluation.payload
    if (
        evaluation_payload.get("split") != "validation"
        or evaluation_payload.get("dataset_digest") != dataset.dataset_digest
        or evaluation_payload.get("split_digest") != dataset.split_digest
        or evaluation_payload.get("checkpoint_identity") != identity.run_id
        or evaluation_payload.get("checkpoint_content_digest")
        != result.checkpoint_content_digest
        or evaluation_payload.get("model_config_digest")
        != resolved_model.content_digest
        or evaluation_payload.get("preprocessing_digest")
        != identity.preprocessing_digest
        or evaluation_payload.get("sample_count")
        != len(dataset.indices_for_split(DatasetSplit.VALIDATION))
        or evaluation_payload.get("selection_digest") is not None
        or evaluation_payload.get("calibration_digest") is not None
        or evaluation_payload.get("threshold_digest") is not None
        or evaluation_payload.get("prediction_digest")
        != _compact_prediction_digest(validation_predictions)
    ):
        _fail("completed run", "validation evaluation binding differs")
    labels = np.asarray(
        [item.failure_target for item in validation_predictions], dtype=np.int64
    )
    logits = np.asarray(
        [item.raw_logit for item in validation_predictions], dtype=np.float64
    )
    probabilities = np.asarray(
        [item.uncalibrated_failure_probability for item in validation_predictions],
        dtype=np.float64,
    )
    corrupted = np.asarray(
        [item.candidate_type == "corrupted" for item in validation_predictions],
        dtype=np.bool_,
    )
    recomputed_metrics = evaluate_binary_metrics(
        labels[corrupted], probabilities[corrupted]
    )
    recomputed_ranking = evaluate_group_ranking(
        tuple(
            GroupCandidate(
                sample_id=item.sample_id,
                group_id=item.group_id,
                source_trajectory_id=item.source_trajectory_id,
                candidate_type=item.candidate_type,
                failure_target=item.failure_target,
                failure_probability=item.uncalibrated_failure_probability,
            )
            for item in validation_predictions
        )
    )
    if (
        recomputed_metrics.failure_auprc.value != result.failure_auprc
        or recomputed_metrics.brier_score.value != result.brier_score
        or recomputed_ranking.pairwise_success_over_failure_concordance.value
        != result.pairwise_concordance
        or compute_validation_prediction_digest(logits, labels)
        != result.validation_prediction_digest
    ):
        _fail("completed run", "validation result differs from predictions")
    if (
        completion["run_id"] != identity.run_id
        or result.model_type != model_type.value
        or result.seed != seed
        or result.parameter_count != resolved_model.parameter_count
        or result.model_config_digest != resolved_model.content_digest
        or result.training_config_digest != training_config.content_digest
        or result.checkpoint_identity != identity.run_id
        or summary.get("run_id") != identity.run_id
        or summary.get("completed") is not True
        or summary.get("acceptance_report_digest") != dataset.acceptance_report_digest
        or summary.get("resolved_model_configuration") != resolved_model.as_mapping()
        or summary.get("resolved_training_configuration")
        != training_config.as_mapping()
        or summary.get("positive_class_weight") != positive_class_weight
        or summary.get("model_type") != model_type.value
        or summary.get("seed") != seed
    ):
        _fail("completed run", "result or summary identity differs")
    manifest_payload = _validate_run_manifest(
        run_root / "run-manifest.json",
        identity=identity,
        dataset=dataset,
        resolved_model=resolved_model,
        training_config=training_config,
        positive_class_weight=positive_class_weight,
        git_sha=git_sha,
        branch=branch,
        expected_status=TrainingRunStatus.COMPLETED,
    )
    manifest_environment = cast(Mapping[str, object], manifest_payload["environment"])
    sanitized_environment = {
        key: manifest_environment[key]
        for key in (
            "cublas_workspace_config",
            "cuda_version",
            "deterministic_algorithms",
            "gpu_driver_version",
            "gpu_model",
            "nondeterministic_operations",
            "numpy_version",
            "package_versions",
            "python_version",
            "pytorch_version",
        )
    }
    if summary.get("environment") != sanitized_environment:
        _fail("completed run", "manifest and summary environments differ")
    if (
        _file_digest(run_root / "run-manifest.json", context="run manifest")
        != completion["run_manifest_file_digest"]
        or _file_digest(run_root / "history.json", context="training history")
        != completion["history_file_digest"]
    ):
        _fail("completed run", "manifest or history bytes differ")
    history_payload = _load_json_object(
        run_root / "history.json", context="training history"
    )
    if (
        set(history_payload) != {"epochs", "format", "version"}
        or history_payload["format"] != TRAINING_HISTORY_FORMAT
        or history_payload["version"] != TRAINING_HISTORY_VERSION
        or not isinstance(history_payload["epochs"], list)
    ):
        _fail("completed run", "training history inventory differs")
    try:
        history = tuple(
            EpochRecordV1(**item)
            for item in history_payload["epochs"]
            if isinstance(item, dict)
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"completed run: invalid training history: {exc}") from exc
    if len(history) != len(history_payload["epochs"]) or not history:
        _fail("completed run", "training history entries are malformed")
    final_history = history[-1]
    if (
        tuple(item.epoch for item in history) != tuple(range(len(history)))
        or summary.get("final_epoch") != final_history.epoch
        or summary.get("global_step") != final_history.global_step
        or summary.get("final_validation_loss") != final_history.validation_loss
        or summary.get("final_validation_corrupted_failure_auprc")
        != final_history.validation_corrupted_failure_auprc
        or summary.get("final_validation_corrupted_brier_score")
        != final_history.validation_corrupted_brier_score
        or summary.get("cumulative_epoch_duration_seconds")
        != sum(item.duration_seconds for item in history)
    ):
        _fail("completed run", "training history cursor differs")
    best_checkpoint = checkpoint_root / "best.pt"
    final_checkpoint = checkpoint_root / "final.pt"
    periodic_checkpoint = checkpoint_root / "periodic.pt"
    best_digest = compute_checkpoint_content_digest(best_checkpoint)
    final_digest = compute_checkpoint_content_digest(final_checkpoint)
    periodic_digest = compute_checkpoint_content_digest(periodic_checkpoint)
    best_metadata = inspect_training_checkpoint(best_checkpoint)
    final_metadata = inspect_training_checkpoint(final_checkpoint)
    periodic_metadata = inspect_training_checkpoint(periodic_checkpoint)
    if (
        best_digest != result.checkpoint_content_digest
        or best_digest != completion["best_checkpoint_content_digest"]
        or best_digest != summary.get("best_checkpoint_content_digest")
        or final_digest != completion["final_checkpoint_content_digest"]
        or final_digest != summary.get("final_checkpoint_content_digest")
        or periodic_digest != completion["periodic_checkpoint_content_digest"]
        or best_metadata.progress.kind != "best"
        or best_metadata.progress.epoch != result.checkpoint_epoch
        or best_metadata.progress.epoch != summary.get("best_epoch")
        or best_metadata.progress.binding != _checkpoint_binding(identity)
        or final_metadata.progress.kind != "final"
        or periodic_metadata.progress.kind != "periodic"
        or final_metadata.progress.binding != _checkpoint_binding(identity)
        or periodic_metadata.progress.binding != _checkpoint_binding(identity)
        or final_metadata.progress.epoch != final_history.epoch
        or periodic_metadata.progress.epoch != final_history.epoch
        or final_metadata.progress.global_step != final_history.global_step
        or periodic_metadata.progress.global_step != final_history.global_step
        or final_metadata.progress.best_epoch != best_metadata.progress.epoch
        or periodic_metadata.progress.best_epoch != best_metadata.progress.epoch
        or final_metadata.progress.best_validation_metric
        != best_metadata.progress.best_validation_metric
        or periodic_metadata.progress.best_validation_metric
        != best_metadata.progress.best_validation_metric
    ):
        _fail("completed run", "checkpoint inventory or binding differs")
    return result


def _train_one(
    *,
    root: Path,
    model_type: ModelType,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset: AcceptedActionVerifierDatasetV1,
    preprocessing: PreprocessingStateV1,
    seed: int,
    args: argparse.Namespace,
    git_sha: str,
    branch: str,
) -> tuple[SeedValidationResult, Path]:
    initial_model = build_model(model_config, seed=seed)
    resolved_model = resolve_model_config(initial_model)
    identity = _run_identity(
        dataset, preprocessing, resolved_model, training_config, seed
    )
    run_root = _run_directory(root, model_type, seed, force_rerun=False)
    result_path = run_root / "validation-result.json"
    completed_path = run_root / "completed.json"
    positive_class_weight = _positive_class_weight(dataset, training_config)
    if completed_path.exists():
        return (
            _validate_completed_run(
                run_root=run_root,
                model_type=model_type,
                seed=seed,
                identity=identity,
                dataset=dataset,
                resolved_model=resolved_model,
                training_config=training_config,
                positive_class_weight=positive_class_weight,
                git_sha=git_sha,
                branch=branch,
            ),
            run_root,
        )
    populated = run_root.exists() and any(run_root.iterdir())
    if populated and not args.resume:
        _fail("incomplete run", f"{model_type.value}/{seed} requires --resume")
    run_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    if populated:
        prior_manifest = _validate_run_manifest(
            run_root / "run-manifest.json",
            identity=identity,
            dataset=dataset,
            resolved_model=resolved_model,
            training_config=training_config,
            positive_class_weight=positive_class_weight,
            git_sha=git_sha,
            branch=branch,
            expected_status=TrainingRunStatus.RUNNING,
        )
        started = cast(str, prior_manifest["started_at_utc"])
    set_training_seed(
        seed,
        deterministic_algorithms=training_config.deterministic_algorithms,
    )
    environment = _environment()
    running_manifest = TrainingRunManifest(
        identity=identity,
        acceptance_report_digest=dataset.acceptance_report_digest,
        model_configuration=resolved_model,
        training_configuration=training_config,
        positive_class_weight=positive_class_weight,
        git_sha=git_sha,
        branch=branch,
        environment=environment,
        started_at_utc=started,
        finished_at_utc=None,
        status=TrainingRunStatus.RUNNING,
    )
    _atomic_json(run_root / "run-manifest.json", running_manifest.as_mapping())
    binding = _checkpoint_binding(identity)
    run_forward_backward_gate(
        initial_model,
        dataset,
        preprocessing,
        training_config,
        device=args.device,
    )
    periodic = run_root / "checkpoints" / "periodic.pt"
    if populated and not periodic.is_file():
        _fail("resume", "matching periodic checkpoint is unavailable")
    resume_checkpoint = periodic if populated else None
    training_result = train_action_verifier(
        model=initial_model,
        resolved_model_config=resolved_model,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=run_root,
        seed=seed,
        device=args.device,
        resume_checkpoint=resume_checkpoint,
        profile_memory=(args.profile_memory or args.device.startswith("cuda")),
    )
    if training_result.positive_class_weight != positive_class_weight:
        _fail("training result", "class weight differs from the run manifest")
    best_metadata = inspect_training_checkpoint(training_result.best_checkpoint)
    best_checkpoint_digest = compute_checkpoint_content_digest(
        training_result.best_checkpoint
    )
    if (
        best_metadata.progress.kind != "best"
        or best_metadata.progress.epoch != training_result.best_epoch
        or best_metadata.progress.binding != binding
    ):
        _fail("training result", "best checkpoint identity differs")
    best_model = build_model(model_config, seed=0)
    load_training_checkpoint(
        training_result.best_checkpoint,
        expected_binding=binding,
        model=best_model,
        restore_rng=False,
    )
    validation_inference = infer_action_verifier_split(
        best_model,
        dataset,
        preprocessing,
        split=DatasetSplit.VALIDATION,
        batch_size=training_config.batch_size,
        device=args.device,
    )
    evaluation, predictions = evaluate_action_verifier_logits(
        validation_inference.logits,
        validation_inference.failure_targets,
        validation_inference.metadata,
        split="validation",
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        checkpoint_identity=identity.run_id,
        checkpoint_content_digest=best_checkpoint_digest,
        model_config_digest=resolved_model.content_digest,
        preprocessing_digest=preprocessing.content_digest,
    )
    validation_evaluation_report = StrictReportV1(
        "validation_evaluation_v1", evaluation.to_dict()
    )
    save_strict_report(
        validation_evaluation_report,
        run_root / "validation-evaluation.json",
    )
    validation_predictions_report = _save_predictions(
        predictions,
        run_root / "validation-predictions.json",
        report_type="validation_predictions_v1",
    )
    seed_result = _validation_result(
        model_type=model_type,
        seed=seed,
        resolved_model=resolved_model,
        training_config=training_config,
        identity=identity,
        report=evaluation,
        checkpoint_content_digest=best_checkpoint_digest,
        checkpoint_epoch=training_result.best_epoch,
        validation_prediction_digest=compute_validation_prediction_digest(
            validation_inference.logits,
            validation_inference.failure_targets,
        ),
    )
    seed_report = StrictReportV1("seed_validation_result_v1", seed_result.to_dict())
    save_strict_report(seed_report, result_path)
    final_history = training_result.history[-1]
    final_checkpoint_digest = compute_checkpoint_content_digest(
        training_result.final_checkpoint
    )
    training_summary = StrictReportV1(
        "training_run_summary_v1",
        {
            "acceptance_report_digest": dataset.acceptance_report_digest,
            "best_checkpoint_content_digest": best_checkpoint_digest,
            "best_epoch": training_result.best_epoch,
            "best_validation_failure_auprc": (
                training_result.best_validation_failure_auprc
            ),
            "completed": True,
            "cumulative_epoch_duration_seconds": sum(
                item.duration_seconds for item in training_result.history
            ),
            "environment": {
                "cublas_workspace_config": environment.cublas_workspace_config,
                "cuda_version": environment.cuda_version,
                "deterministic_algorithms": environment.deterministic_algorithms,
                "gpu_driver_version": environment.gpu_driver_version,
                "gpu_model": environment.gpu_model,
                "nondeterministic_operations": list(
                    environment.nondeterministic_operations
                ),
                "numpy_version": environment.numpy_version,
                "package_versions": list(environment.package_versions),
                "python_version": environment.python_version,
                "pytorch_version": environment.pytorch_version,
            },
            "final_checkpoint_content_digest": final_checkpoint_digest,
            "final_epoch": training_result.final_epoch,
            "final_validation_corrupted_brier_score": (
                final_history.validation_corrupted_brier_score
            ),
            "final_validation_corrupted_failure_auprc": (
                final_history.validation_corrupted_failure_auprc
            ),
            "final_validation_loss": final_history.validation_loss,
            "global_step": training_result.global_step,
            "model_type": model_type.value,
            "peak_gpu_memory_bytes": training_result.peak_gpu_memory_bytes,
            "positive_class_weight": training_result.positive_class_weight,
            "resolved_model_configuration": resolved_model.as_mapping(),
            "resolved_training_configuration": training_config.as_mapping(),
            "run_id": identity.run_id,
            "seed": seed,
            "training_duration_seconds_current_attempt": (
                training_result.duration_seconds
            ),
        },
    )
    save_strict_report(training_summary, run_root / "training-summary.json")
    completed_manifest = TrainingRunManifest(
        identity=identity,
        acceptance_report_digest=dataset.acceptance_report_digest,
        model_configuration=resolved_model,
        training_configuration=training_config,
        positive_class_weight=positive_class_weight,
        git_sha=git_sha,
        branch=branch,
        environment=environment,
        started_at_utc=started,
        finished_at_utc=datetime.now(UTC).isoformat(),
        status=TrainingRunStatus.COMPLETED,
    )
    _atomic_json(run_root / "run-manifest.json", completed_manifest.as_mapping())
    _atomic_json(
        completed_path,
        {
            "best_checkpoint_content_digest": best_checkpoint_digest,
            "final_checkpoint_content_digest": final_checkpoint_digest,
            "history_file_digest": _file_digest(
                run_root / "history.json", context="training history"
            ),
            "periodic_checkpoint_content_digest": (
                compute_checkpoint_content_digest(
                    run_root / "checkpoints" / "periodic.pt"
                )
            ),
            "run_id": identity.run_id,
            "run_manifest_file_digest": _file_digest(
                run_root / "run-manifest.json", context="run manifest"
            ),
            "training_summary_digest": training_summary.content_digest,
            "validation_evaluation_digest": (
                validation_evaluation_report.content_digest
            ),
            "validation_predictions_digest": (
                validation_predictions_report.content_digest
            ),
            "validation_result_digest": seed_report.content_digest,
        },
    )
    return (
        _validate_completed_run(
            run_root=run_root,
            model_type=model_type,
            seed=seed,
            identity=identity,
            dataset=dataset,
            resolved_model=resolved_model,
            training_config=training_config,
            positive_class_weight=positive_class_weight,
            git_sha=git_sha,
            branch=branch,
        ),
        run_root,
    )


def _fit_and_test_one(
    *,
    run_root: Path,
    evaluation_root: Path,
    seed_result: SeedValidationResult,
    dataset: AcceptedActionVerifierDatasetV1,
    selection: SelectionRecordV1,
    device: str,
    batch_size: int,
    target_failure_recall: float,
) -> tuple[
    ActionVerifierEvaluationReportV1,
    tuple[CompactPredictionV1, ...],
    SplitInferenceResultV1,
]:
    checkpoint_path = run_root / "checkpoints" / "best.pt"
    checkpoint_content_digest = compute_checkpoint_content_digest(checkpoint_path)
    metadata = inspect_training_checkpoint(checkpoint_path)
    if (
        metadata.progress.kind != "best"
        or metadata.progress.epoch != seed_result.checkpoint_epoch
        or metadata.progress.binding.run_manifest_identity
        != seed_result.checkpoint_identity
        or checkpoint_content_digest != seed_result.checkpoint_content_digest
        or metadata.model_configuration.content_digest
        != seed_result.model_config_digest
        or metadata.training_configuration.content_digest
        != seed_result.training_config_digest
    ):
        _fail("test evaluation", "best checkpoint differs from frozen selection")
    model = build_model(metadata.model_configuration.architecture, seed=0)
    load_training_checkpoint(
        checkpoint_path,
        expected_binding=metadata.progress.binding,
        model=model,
        restore_rng=False,
    )
    validation = infer_action_verifier_split(
        model,
        dataset,
        metadata.preprocessing_state,
        split=DatasetSplit.VALIDATION,
        batch_size=batch_size,
        device=device,
    )
    calibration = fit_temperature_scaling(
        validation.logits,
        validation.failure_targets,
        split="validation",
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        checkpoint_identity=seed_result.checkpoint_identity,
        checkpoint_content_digest=checkpoint_content_digest,
        model_config_digest=seed_result.model_config_digest,
        preprocessing_digest=metadata.preprocessing_state.content_digest,
    )
    if (
        calibration.validation_prediction_digest
        != seed_result.validation_prediction_digest
    ):
        _fail("test evaluation", "validation prediction binding differs")
    thresholds = fit_validation_thresholds(
        calibration.apply(validation.logits),
        validation.failure_targets,
        split="validation",
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        calibration_digest=calibration.content_digest,
        validation_prediction_digest=calibration.validation_prediction_digest,
        target_failure_recall=target_failure_recall,
    )
    test = infer_action_verifier_split(
        model,
        dataset,
        metadata.preprocessing_state,
        split=DatasetSplit.TEST,
        batch_size=batch_size,
        device=device,
    )
    evaluation, predictions = evaluate_action_verifier_logits(
        test.logits,
        test.failure_targets,
        test.metadata,
        split="test",
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        checkpoint_identity=metadata.progress.binding.run_manifest_identity,
        checkpoint_content_digest=checkpoint_content_digest,
        model_config_digest=metadata.model_configuration.content_digest,
        preprocessing_digest=metadata.preprocessing_state.content_digest,
        calibration=calibration,
        thresholds=thresholds,
        selection_digest=selection.content_digest,
    )
    save_strict_report(
        StrictReportV1("temperature_calibration_v1", calibration.to_dict()),
        evaluation_root / "calibration.json",
    )
    save_strict_report(
        StrictReportV1("frozen_thresholds_v1", thresholds.to_dict()),
        evaluation_root / "thresholds.json",
    )
    save_strict_report(
        StrictReportV1("test_evaluation_v1", evaluation.to_dict()),
        evaluation_root / "test-evaluation.json",
    )
    _save_predictions(
        predictions,
        evaluation_root / "test-predictions.json",
        report_type="test_predictions_v1",
    )
    return evaluation, predictions, test


def _validate_test_artifacts(
    *,
    evaluation_root: Path,
    seed_result: SeedValidationResult,
    dataset: AcceptedActionVerifierDatasetV1,
    selection: SelectionRecordV1,
) -> dict[str, str]:
    calibration_report = load_strict_report(
        evaluation_root / "calibration.json",
        expected_report_type="temperature_calibration_v1",
    )
    threshold_report = load_strict_report(
        evaluation_root / "thresholds.json",
        expected_report_type="frozen_thresholds_v1",
    )
    evaluation_report = load_strict_report(
        evaluation_root / "test-evaluation.json",
        expected_report_type="test_evaluation_v1",
    )
    prediction_report = load_strict_report(
        evaluation_root / "test-predictions.json",
        expected_report_type="test_predictions_v1",
    )
    calibration = temperature_calibration_from_dict(calibration_report.payload)
    thresholds = frozen_thresholds_from_dict(threshold_report.payload)
    authorized = [
        item
        for candidate in selection.candidates
        for item in candidate.seed_results
        if item == seed_result
    ]
    if (
        len(authorized) != 1
        or calibration.dataset_digest != dataset.dataset_digest
        or calibration.split_digest != dataset.split_digest
        or calibration.checkpoint_identity != seed_result.checkpoint_identity
        or calibration.checkpoint_content_digest
        != seed_result.checkpoint_content_digest
        or calibration.model_config_digest != seed_result.model_config_digest
        or calibration.validation_prediction_digest
        != seed_result.validation_prediction_digest
        or thresholds.dataset_digest != dataset.dataset_digest
        or thresholds.split_digest != dataset.split_digest
        or thresholds.calibration_digest != calibration.content_digest
        or thresholds.validation_prediction_digest
        != calibration.validation_prediction_digest
    ):
        _fail("test artifacts", "selection, calibration, or threshold binding differs")
    predictions = _compact_predictions_from_report(
        prediction_report, expected_split="test"
    )
    expected_by_id = {
        dataset.reporting_for(index).sample_id: (
            dataset[index],
            dataset.reporting_for(index),
        )
        for index in dataset.indices_for_split(DatasetSplit.TEST)
    }
    if {item.sample_id for item in predictions} != set(expected_by_id):
        _fail("test artifacts", "test sample inventory differs")
    metadata: list[EvaluationMetadataV1] = []
    for sample_index, prediction in enumerate(predictions):
        example, reporting = expected_by_id[prediction.sample_id]
        if (
            prediction.group_id != reporting.group_id
            or prediction.source_trajectory_id != reporting.source_trajectory
            or prediction.candidate_type != reporting.candidate_type.value
            or prediction.failure_target != example.failure_target
        ):
            _fail("test artifacts", "prediction reporting join differs")
        metadata.append(
            EvaluationMetadataV1(
                sample_index=sample_index,
                sample_id=reporting.sample_id,
                group_id=reporting.group_id,
                source_trajectory_id=reporting.source_trajectory,
                split="test",
                candidate_type=reporting.candidate_type.value,
                corruption_family=reporting.corruption_family,
                corruption_severity=reporting.severity,
                anchor_selection_reason=reporting.anchor_selection_reason,
            )
        )
    recomputed_evaluation, recomputed_predictions = evaluate_action_verifier_logits(
        np.asarray([item.raw_logit for item in predictions], dtype=np.float64),
        np.asarray([item.failure_target for item in predictions], dtype=np.int64),
        tuple(metadata),
        split="test",
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        checkpoint_identity=seed_result.checkpoint_identity,
        checkpoint_content_digest=seed_result.checkpoint_content_digest,
        model_config_digest=seed_result.model_config_digest,
        preprocessing_digest=calibration.preprocessing_digest,
        calibration=calibration,
        thresholds=thresholds,
        selection_digest=selection.content_digest,
    )
    if (
        recomputed_evaluation.to_dict() != evaluation_report.payload
        or recomputed_predictions != predictions
    ):
        _fail("test artifacts", "evaluation differs from compact predictions")
    return {
        "calibration.json": calibration_report.content_digest,
        "test-evaluation.json": evaluation_report.content_digest,
        "test-predictions.json": prediction_report.content_digest,
        "thresholds.json": threshold_report.content_digest,
    }


def _group_candidates(
    metadata: Sequence[EvaluationMetadataV1],
    targets: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> tuple[GroupCandidate, ...]:
    return tuple(
        GroupCandidate(
            sample_id=item.sample_id,
            group_id=item.group_id,
            source_trajectory_id=item.source_trajectory_id,
            candidate_type=item.candidate_type,
            failure_target=int(targets[index]),
            failure_probability=float(probabilities[index]),
        )
        for index, item in enumerate(metadata)
    )


def _baseline_report(
    name: str,
    inference: SplitInferenceResultV1,
    probabilities: NDArray[np.float64],
) -> dict[str, object]:
    corrupted = np.asarray(
        [item.candidate_type == "corrupted" for item in inference.metadata],
        dtype=np.bool_,
    )
    slice_values: dict[str, Sequence[str | None]] = {
        "anchor_selection_reason": [
            item.anchor_selection_reason for item in inference.metadata
        ],
        "candidate_type": [item.candidate_type for item in inference.metadata],
        "corruption_family": [item.corruption_family for item in inference.metadata],
        "corruption_severity": [
            item.corruption_severity for item in inference.metadata
        ],
        "source_trajectory": [item.source_trajectory_id for item in inference.metadata],
        "split": [item.split for item in inference.metadata],
    }
    return {
        "baseline": name,
        "candidate_metrics": evaluate_binary_metrics(
            inference.failure_targets, probabilities
        ).to_dict(),
        "corrupted_only_metrics": evaluate_binary_metrics(
            inference.failure_targets[corrupted], probabilities[corrupted]
        ).to_dict(),
        "group_ranking": evaluate_group_ranking(
            _group_candidates(
                inference.metadata, inference.failure_targets, probabilities
            )
        ).to_dict(),
        "coverage_risk": evaluate_coverage_risk(
            inference.failure_targets, probabilities
        ).to_dict(),
        "slices": [
            item.to_dict()
            for item in evaluate_slices(
                inference.failure_targets, probabilities, slice_values
            )
        ],
    }


def _nonlearned_reports(
    dataset: AcceptedActionVerifierDatasetV1,
    test_inference: SplitInferenceResultV1,
) -> dict[str, object]:
    majority, prevalence = fit_constant_baselines(dataset)
    magnitude = fit_action_magnitude_baseline(dataset)
    test_examples = tuple(
        dataset[index] for index in dataset.indices_for_split(DatasetSplit.TEST)
    )
    probabilities = {
        "majority": np.asarray(
            majority.predict_failure_probability(len(test_examples)), dtype=np.float64
        ),
        "failure_prevalence": np.asarray(
            prevalence.predict_failure_probability(len(test_examples)), dtype=np.float64
        ),
        "action_magnitude": np.asarray(
            magnitude.predict_failure_probability(test_examples), dtype=np.float64
        ),
    }
    return {
        "fitted_digests": {
            "majority": majority.content_digest,
            "failure_prevalence": prevalence.content_digest,
            "action_magnitude": magnitude.content_digest,
        },
        "test": {
            name: _baseline_report(name, test_inference, values)
            for name, values in probabilities.items()
        },
    }


def _aggregate_selected_test(
    selection: SelectionRecordV1,
    evaluations: Mapping[tuple[str, int], ActionVerifierEvaluationReportV1],
) -> dict[str, object]:
    selected = selection.selected_model_type
    selected_reports = [
        report
        for (model_type, _), report in sorted(evaluations.items())
        if model_type == selected
    ]
    if len(selected_reports) != 5:
        _fail("test aggregation", "selected architecture requires five seeds")
    metric_names = (
        "failure_auprc",
        "roc_auc",
        "brier_score",
        "ece",
        "pairwise_concordance",
    )
    aggregates: dict[str, object] = {}
    for name in metric_names:
        values: list[float] = []
        for report in selected_reports:
            if report.calibrated is None:
                _fail("test aggregation", "calibrated report is missing")
            calibrated = report.calibrated
            if name == "failure_auprc":
                value = calibrated.corrupted_only_metrics.failure_auprc.value
            elif name == "roc_auc":
                value = calibrated.corrupted_only_metrics.roc_auc.value
            elif name == "brier_score":
                value = calibrated.corrupted_only_metrics.brier_score.value
            elif name == "ece":
                value = (
                    calibrated.corrupted_only_metrics.expected_calibration_error.value
                )
            else:
                group_ranking = calibrated.group_ranking
                value = group_ranking.pairwise_success_over_failure_concordance.value
            if value is None:
                _fail("test aggregation", f"{name} is undefined")
            values.append(value)
        aggregates[name] = aggregate_five_seed_values(values).to_dict()
    return {
        "selected_model_type": selected,
        "seed_count": len(selected_reports),
        "metrics": aggregates,
    }


def _aggregate_five_payloads(values: Sequence[object], *, context: str) -> object:
    if len(values) != 5:
        _fail(context, "expected exactly five seed values")
    if all(isinstance(value, Mapping) for value in values):
        mappings = [cast(Mapping[str, object], value) for value in values]
        keys = tuple(mappings[0])
        if any(tuple(mapping) != keys for mapping in mappings[1:]):
            _fail(context, "mapping inventory differs across seeds")
        if set(keys) == {"reason", "status", "value"}:
            defined = [
                mapping["value"]
                for mapping in mappings
                if mapping["status"] == "defined"
                and isinstance(mapping["value"], (int, float))
                and not isinstance(mapping["value"], bool)
            ]
            if len(defined) == 5:
                return aggregate_five_seed_values(
                    [float(value) for value in defined]
                ).to_dict()
            return {
                "defined_count": len(defined),
                "per_seed": [dict(mapping) for mapping in mappings],
            }
        return {
            key: _aggregate_five_payloads(
                [mapping[key] for mapping in mappings],
                context=f"{context}.{key}",
            )
            for key in keys
        }
    if all(isinstance(value, list) for value in values):
        sequences = [cast(list[object], value) for value in values]
        if len({len(value) for value in sequences}) != 1:
            _fail(context, "sequence length differs across seeds")
        return [
            _aggregate_five_payloads(
                [sequence[index] for sequence in sequences],
                context=f"{context}[{index}]",
            )
            for index in range(len(sequences[0]))
        ]
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        numbers = [float(cast(int | float, value)) for value in values]
        if any(not math.isfinite(value) for value in numbers):
            _fail(context, "numeric value is non-finite")
        if all(value == values[0] for value in values[1:]):
            return values[0]
        return aggregate_five_seed_values(numbers).to_dict()
    if all(value == values[0] for value in values[1:]):
        return values[0]
    return {"per_seed": list(values)}


def _aggregate_learned_test_reports(
    evaluations: Mapping[tuple[str, int], ActionVerifierEvaluationReportV1],
) -> dict[str, object]:
    grouped: dict[str, list[tuple[int, ActionVerifierEvaluationReportV1]]] = {}
    for (model_type, seed), report in sorted(evaluations.items()):
        grouped.setdefault(model_type, []).append((seed, report))
    result: dict[str, object] = {}
    for model_type in sorted(grouped):
        seed_reports = grouped[model_type]
        if len(seed_reports) != 5 or any(
            report.calibrated is None for _, report in seed_reports
        ):
            _fail("test aggregation", "each architecture requires five reports")
        result[model_type] = {
            "seeds": [seed for seed, _ in seed_reports],
            "uncalibrated": _aggregate_five_payloads(
                [report.uncalibrated.to_dict() for _, report in seed_reports],
                context=f"{model_type}.uncalibrated",
            ),
            "calibrated": _aggregate_five_payloads(
                [
                    cast(object, report.calibrated.to_dict())
                    for _, report in seed_reports
                    if report.calibrated is not None
                ],
                context=f"{model_type}.calibrated",
            ),
            "threshold_evaluations": _aggregate_five_payloads(
                [
                    [item.to_dict() for item in report.threshold_evaluations]
                    for _, report in seed_reports
                ],
                context=f"{model_type}.threshold_evaluations",
            ),
        }
    return result


def _mean_metric_text(values: Sequence[float | None], *, context: str) -> str:
    if len(values) != 5:
        _fail(context, "expected five seed metrics")
    defined = [value for value in values if value is not None]
    if any(not math.isfinite(value) for value in defined):
        _fail(context, "metric is non-finite")
    if len(defined) != 5:
        return f"undefined ({len(defined)}/5 seeds defined)"
    return f"{float(np.mean(np.asarray(defined, dtype=np.float64))):.6f}"


def _baseline_metric_text(metrics: Mapping[str, object], metric_name: str) -> str:
    metric = cast(Mapping[str, object], metrics[metric_name])
    value = metric["value"]
    return "undefined" if value is None else f"{float(cast(float, value)):.6f}"


def _render_final_markdown(
    *,
    branch: str,
    git_sha: str,
    dataset: AcceptedActionVerifierDatasetV1,
    selection: SelectionRecordV1,
    evaluations: Mapping[tuple[str, int], ActionVerifierEvaluationReportV1],
    training_summaries: Sequence[Mapping[str, object]],
    parameter_counts: Mapping[str, int],
    nonlearned: Mapping[str, object],
    comparisons: Mapping[str, object],
    acceptance: Mapping[str, object],
    joint_advantage: Mapping[str, object],
) -> str:
    selected_reports = [
        report
        for (model_type, _), report in sorted(evaluations.items())
        if model_type == selection.selected_model_type
    ]
    if len(selected_reports) != 5 or any(
        report.calibrated is None for report in selected_reports
    ):
        _fail("Markdown summary", "selected five-seed reports are incomplete")
    environments = [summary.get("environment") for summary in training_summaries]
    if not environments or not isinstance(environments[0], Mapping):
        _fail("Markdown summary", "training environment is unavailable")
    environment = cast(Mapping[str, object], environments[0])
    durations = [
        float(cast(float, summary["cumulative_epoch_duration_seconds"]))
        for summary in training_summaries
    ]
    selected_epochs = [
        int(cast(int, summary["best_epoch"])) for summary in training_summaries
    ]
    lines = [
        "# M3B Direct Action Verifier Baseline Results",
        "",
        "## Identity and execution",
        "",
        f"- Branch: `{branch}`",
        f"- Git SHA: `{git_sha}`",
        f"- Dataset digest: `{dataset.dataset_digest}`",
        f"- Acceptance-report digest: `{dataset.acceptance_report_digest}`",
        f"- Split digest: `{dataset.split_digest}`",
        "- Matrix: four learned architectures by five fixed seeds (20 runs)",
        f"- GPU: `{environment.get('gpu_model')}`",
        f"- Driver/CUDA/PyTorch: `{environment.get('gpu_driver_version')}` / "
        f"`{environment.get('cuda_version')}` / `{environment.get('pytorch_version')}`",
        f"- Python/NumPy: `{environment.get('python_version')}` / "
        f"`{environment.get('numpy_version')}`",
        f"- Total epoch time: {sum(durations):.3f} s; mean per run: "
        f"{float(np.mean(durations)):.3f} s",
        f"- Selected epochs across all runs: {selected_epochs}",
        "- Checkpoint reload and exact resume: passed for every smoke gate; "
        "completed-run inventories were revalidated before reuse.",
        "",
        "## Validation-only architecture selection",
        "",
        "| Architecture | Parameters | Failure AUPRC mean | Brier mean | "
        "Pairwise concordance mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in selection.candidates:
        lines.append(
            f"| {candidate.model_type} | {parameter_counts[candidate.model_type]} | "
            f"{candidate.failure_auprc.mean:.6f} | {candidate.brier_score.mean:.6f} | "
            f"{candidate.pairwise_concordance.mean:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Selected architecture: `{selection.selected_model_type}` using the fixed "
            "validation corrupted-only failure-AUPRC ordering and documented tie "
            "breakers.",
            "",
            "## Untouched test metrics for the selected five-seed architecture",
            "",
            "| Metric | Uncalibrated mean | Calibrated mean |",
            "|---|---:|---:|",
        ]
    )
    metric_fields = (
        ("failure_prevalence", "Failure prevalence"),
        ("roc_auc", "Failure ROC AUC"),
        ("failure_auprc", "Failure AUPRC"),
        ("success_auprc", "Success AUPRC"),
        ("negative_log_likelihood", "Negative log likelihood"),
        ("brier_score", "Brier score"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("precision", "Failure precision"),
        ("recall", "Failure recall"),
        ("f1", "Failure F1"),
        ("specificity", "Specificity"),
        ("matthews_correlation_coefficient", "Matthews correlation"),
        ("expected_calibration_error", "Expected calibration error"),
    )
    for field, label in metric_fields:
        uncalibrated = _mean_metric_text(
            [
                cast(
                    float | None,
                    getattr(report.uncalibrated.candidate_metrics, field).value,
                )
                for report in selected_reports
            ],
            context=f"Markdown summary uncalibrated {field}",
        )
        calibrated = _mean_metric_text(
            [
                cast(
                    float | None,
                    getattr(
                        cast(
                            ProbabilityEvaluationReport, report.calibrated
                        ).candidate_metrics,
                        field,
                    ).value,
                )
                for report in selected_reports
            ],
            context=f"Markdown summary calibrated {field}",
        )
        lines.append(f"| {label} | {uncalibrated} | {calibrated} |")
    calibrated_reports = [
        report.calibrated
        for report in selected_reports
        if report.calibrated is not None
    ]
    group_fields = (
        ("corrupted_top1_success", "Corrupted-only top-1 success"),
        ("random_choice_success_expectation", "Random-choice expectation"),
        (
            "pairwise_success_over_failure_concordance",
            "Pairwise success-over-failure concordance",
        ),
        ("corrupted_top1_failure_rate", "Corrupted-only top-1 failure rate"),
        ("mean_reciprocal_rank_first_success", "Mean reciprocal rank"),
        (
            "mean_source_success_margin_over_failed_corruptions",
            "Source-margin diagnostic",
        ),
    )
    lines.extend(
        ["", "## Group ranking", "", "| Metric | Five-seed mean |", "|---|---:|"]
    )
    for field, label in group_fields:
        value = _mean_metric_text(
            [
                cast(float | None, getattr(report.group_ranking, field).value)
                for report in calibrated_reports
            ],
            context=f"Markdown summary group {field}",
        )
        lines.append(f"| {label} | {value} |")
    lines.extend(
        [
            "",
            "## Coverage-risk (calibrated candidate scores)",
            "",
            "| Requested coverage | Observed coverage mean | Retained failure rate "
            "mean | Rejected failure recall mean |",
            "|---:|---:|---:|---:|",
        ]
    )
    for point_index in range(len(calibrated_reports[0].candidate_coverage_risk.points)):
        points = [
            report.candidate_coverage_risk.points[point_index]
            for report in calibrated_reports
        ]
        retained_risk = _mean_metric_text(
            [point.retained_failure_rate.value for point in points],
            context="Markdown summary retained risk",
        )
        rejected_recall = _mean_metric_text(
            [point.rejected_failure_recall.value for point in points],
            context="Markdown summary rejected recall",
        )
        lines.append(
            f"| {points[0].requested_coverage:.2f} | "
            f"{float(np.mean([point.observed_coverage for point in points])):.6f} | "
            f"{retained_risk} | {rejected_recall} |"
        )
    baseline_test = cast(Mapping[str, object], nonlearned["test"])
    lines.extend(
        [
            "",
            "## Non-learned test baselines",
            "",
            "| Baseline | Failure AUPRC | ROC AUC | Brier score |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("majority", "failure_prevalence", "action_magnitude"):
        baseline = cast(Mapping[str, object], baseline_test[name])
        metrics = cast(Mapping[str, object], baseline["candidate_metrics"])

        lines.append(
            f"| {name} | {_baseline_metric_text(metrics, 'failure_auprc')} | "
            f"{_baseline_metric_text(metrics, 'roc_auc')} | "
            f"{_baseline_metric_text(metrics, 'brier_score')} |"
        )
    comparison_items = cast(Mapping[str, object], comparisons["comparisons"])
    lines.extend(
        [
            "",
            "## Trajectory-level paired comparisons",
            "",
            "All intervals use original source trajectories as the bootstrap unit. "
            "Machine-readable observed differences, intervals, valid resamples, and "
            "seeds are in `benchmark-summary.json`.",
        ]
    )
    for model_type in sorted(comparison_items):
        lines.append(
            f"- Selected model versus `{model_type}`: recorded for failure AUPRC, "
            "Brier score, and corrupted top-1 success."
        )
    lines.extend(
        [
            "",
            "## Acceptance and claim boundary",
            "",
            f"- At least one learned architecture passed every fixed test target: "
            f"`{acceptance['at_least_one_learned_model_passed']}`.",
            f"- Selected validation aggregate strictly exceeded state-only and "
            f"action-only: "
            f"`{joint_advantage['selected_outperformed_state_and_action_only']}`.",
            "- Per-corruption, severity, anchor-reason, candidate-type, trajectory, "
            "and split slices remain in each content-bound `test-evaluation.json`.",
            "- These results are limited to fixed PickCube privileged structured state "
            "and one continuation policy. They do not establish visual/language "
            "understanding, cross-task transfer, general safety, causal diagnosis, or "
            "real-robot validity.",
            "- No VLM, LLM, LangMani, online simulator rollout, pretrained model, or "
            "multi-GPU/distributed training was used.",
        ]
    )
    return "\n".join(lines)


def _statistical_comparisons(
    *,
    selection: SelectionRecordV1,
    predictions: Mapping[tuple[str, int], tuple[CompactPredictionV1, ...]],
    benchmark: BenchmarkConfig,
) -> dict[str, object]:
    by_model: dict[str, list[tuple[CompactPredictionV1, ...]]] = {}
    for (model_type, _), values in sorted(predictions.items()):
        by_model.setdefault(model_type, []).append(values)
    selected_runs = by_model[selection.selected_model_type]
    ordered_ids = tuple(item.sample_id for item in selected_runs[0])

    def mean_probabilities(
        runs: Sequence[tuple[CompactPredictionV1, ...]],
    ) -> NDArray[np.float64]:
        if len(runs) != 5 or any(
            tuple(item.sample_id for item in run) != ordered_ids for run in runs
        ):
            _fail("statistical comparisons", "paired sample inventory differs")
        stacked = np.stack(
            [
                np.asarray(
                    [cast(float, item.calibrated_failure_probability) for item in run],
                    dtype=np.float64,
                )
                for run in runs
            ]
        )
        return np.asarray(np.mean(stacked, axis=0), dtype=np.float64)

    selected_probabilities = mean_probabilities(selected_runs)
    targets = np.asarray(
        [item.failure_target for item in selected_runs[0]], dtype=np.int64
    )
    trajectories = [item.source_trajectory_id for item in selected_runs[0]]
    selected_candidates = _group_candidates(
        tuple(
            EvaluationMetadataV1(
                sample_index=index,
                sample_id=item.sample_id,
                group_id=item.group_id,
                source_trajectory_id=item.source_trajectory_id,
                split=item.split,
                candidate_type=item.candidate_type,
                corruption_family=(
                    "unknown" if item.candidate_type == "corrupted" else None
                ),
                corruption_severity=(
                    "unknown" if item.candidate_type == "corrupted" else None
                ),
                anchor_selection_reason="reported_separately",
            )
            for index, item in enumerate(selected_runs[0])
        ),
        targets,
        selected_probabilities,
    )
    comparisons: dict[str, object] = {}
    for ordinal, model_type in enumerate(sorted(by_model)):
        if model_type == selection.selected_model_type:
            continue
        other_probabilities = mean_probabilities(by_model[model_type])
        other_candidates = tuple(
            replace(
                candidate,
                failure_probability=float(other_probabilities[index]),
            )
            for index, candidate in enumerate(selected_candidates)
        )
        comparisons[model_type] = {
            "failure_auprc_difference": paired_trajectory_bootstrap(
                targets,
                selected_probabilities,
                other_probabilities,
                trajectories,
                metric="failure_auprc",
                resamples=benchmark.bootstrap_replicates,
                seed=benchmark.bootstrap_seed + ordinal,
                confidence_level=benchmark.confidence_level,
            ).to_dict(),
            "brier_score_difference": paired_trajectory_bootstrap(
                targets,
                selected_probabilities,
                other_probabilities,
                trajectories,
                metric="brier_score",
                resamples=benchmark.bootstrap_replicates,
                seed=benchmark.bootstrap_seed + 100 + ordinal,
                confidence_level=benchmark.confidence_level,
            ).to_dict(),
            "corrupted_top1_success_difference": (
                group_bootstrap_top1_success_difference(
                    selected_candidates,
                    other_candidates,
                    resamples=benchmark.bootstrap_replicates,
                    seed=benchmark.bootstrap_seed + 200 + ordinal,
                    confidence_level=benchmark.confidence_level,
                ).to_dict()
            ),
        }
    return {
        "estimand": "difference_between_five_seed_mean_calibrated_predictors_v1",
        "bootstrap_unit": "source_trajectory_v1",
        "comparisons": comparisons,
    }


def _acceptance_assessment(
    evaluations: Mapping[tuple[str, int], ActionVerifierEvaluationReportV1],
    nonlearned: Mapping[str, object],
) -> dict[str, object]:
    prevalence_payload = cast(
        Mapping[str, object],
        cast(Mapping[str, object], nonlearned["test"])["failure_prevalence"],
    )
    prevalence_metrics = cast(
        Mapping[str, object], prevalence_payload["candidate_metrics"]
    )
    prevalence_brier = cast(
        float,
        cast(Mapping[str, object], prevalence_metrics["brier_score"])["value"],
    )
    by_model: dict[str, list[ActionVerifierEvaluationReportV1]] = {}
    for (model_type, _), report in sorted(evaluations.items()):
        by_model.setdefault(model_type, []).append(report)
    passing: list[dict[str, object]] = []
    all_models: list[dict[str, object]] = []
    for model_type in sorted(by_model):
        reports = by_model[model_type]
        if len(reports) != 5 or any(report.calibrated is None for report in reports):
            _fail("acceptance", "each architecture requires five calibrated runs")
        rows: list[tuple[float | None, ...]] = []
        for report in reports:
            calibrated = report.calibrated
            if calibrated is None:
                _fail("acceptance", "calibrated test report is missing")
            rows.append(
                (
                    calibrated.candidate_metrics.failure_prevalence.value,
                    calibrated.candidate_metrics.failure_auprc.value,
                    calibrated.candidate_metrics.roc_auc.value,
                    calibrated.candidate_metrics.brier_score.value,
                    calibrated.candidate_metrics.expected_calibration_error.value,
                    report.uncalibrated.candidate_metrics.expected_calibration_error.value,
                    calibrated.group_ranking.pairwise_success_over_failure_concordance.value,
                )
            )
        finite = all(
            value is not None and math.isfinite(value) for row in rows for value in row
        )
        means: tuple[float | None, ...]
        if finite:
            means = tuple(
                float(np.mean([cast(float, row[index]) for row in rows]))
                for index in range(7)
            )
        else:
            means = (None,) * 7
        prevalence, auprc, roc_auc, brier, ece, raw_ece, concordance = means
        passed = bool(
            finite
            and cast(float, auprc) >= cast(float, prevalence) + 0.10
            and cast(float, roc_auc) >= 0.70
            and cast(float, concordance) >= 0.65
            and cast(float, brier) < prevalence_brier
            and cast(float, ece) <= cast(float, raw_ece) + 0.01
        )
        item = {
            "aggregation_semantic": "five_seed_arithmetic_mean_v1",
            "model_type": model_type,
            "seed_count": len(reports),
            "passed": passed,
            "failure_prevalence": prevalence,
            "failure_auprc": auprc,
            "roc_auc": roc_auc,
            "pairwise_concordance": concordance,
            "brier_score": brier,
            "failure_prevalence_baseline_brier": prevalence_brier,
            "calibrated_ece": ece,
            "uncalibrated_ece": raw_ece,
            "all_required_metrics_finite": finite,
        }
        all_models.append(item)
        if passed:
            passing.append(item)
    return {
        "at_least_one_learned_model_passed": bool(passing),
        "passing_model_count": len(passing),
        "models": all_models,
    }


def _validation_joint_advantage(selection: SelectionRecordV1) -> dict[str, object]:
    by_model = {item.model_type: item for item in selection.candidates}
    selected = by_model[selection.selected_model_type]
    state = by_model[ModelType.STATE_ONLY_MLP.value]
    action = by_model[ModelType.ACTION_ONLY_MLP.value]
    passed = selected.failure_auprc.mean > max(
        state.failure_auprc.mean, action.failure_auprc.mean
    )
    return {
        "selected_model_type": selection.selected_model_type,
        "selected_validation_failure_auprc_mean": selected.failure_auprc.mean,
        "state_only_validation_failure_auprc_mean": state.failure_auprc.mean,
        "action_only_validation_failure_auprc_mean": action.failure_auprc.mean,
        "selected_outperformed_state_and_action_only": passed,
        "claim_joint_state_action_success": passed,
    }


def _select_benchmark_root(base: Path, *, force_rerun: bool) -> Path:
    root = Path(base).absolute()
    if not force_rerun:
        return root
    rerun_parent = root / "reruns"
    ordinal = 1
    while (rerun_parent / f"benchmark-rerun-{ordinal}").exists():
        ordinal += 1
    return rerun_parent / f"benchmark-rerun-{ordinal}"


def run_benchmark_command(
    args: argparse.Namespace,
    *,
    load_dataset: Callable[[argparse.Namespace], AcceptedActionVerifierDatasetV1],
) -> int:
    """Execute a bounded four-model smoke or the immutable 20-run full benchmark."""

    dataset = load_dataset(args)
    benchmark = load_benchmark_config(args.benchmark_config)
    base_training = load_training_config(args.training_config)
    training_config = _resolved_training_config(base_training, args)
    model_configs = _load_model_configs(args.config_dir, benchmark)
    preprocessing = fit_preprocessing_state(
        dataset,
        minimum_standard_deviation=training_config.minimum_standard_deviation,
    )
    git_sha = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current")
    seed_inventory = (benchmark.seeds[0],) if args.mode == "smoke" else benchmark.seeds
    plans: list[dict[str, object]] = []
    for model_type in benchmark.model_types:
        for seed in seed_inventory:
            model = build_model(model_configs[model_type], seed=seed)
            resolved = resolve_model_config(model)
            identity = _run_identity(
                dataset, preprocessing, resolved, training_config, seed
            )
            plans.append(
                {
                    "model_type": model_type.value,
                    "seed": seed,
                    "parameter_count": resolved.parameter_count,
                    "model_config_digest": resolved.content_digest,
                    "run_id": identity.run_id,
                }
            )
    orchestration_identity = _content_digest(
        {
            "acceptance_report_digest": dataset.acceptance_report_digest,
            "benchmark_config_digest": benchmark.content_digest,
            "code_semantic": "m3b_benchmark_orchestration_v1",
            "dataset_digest": dataset.dataset_digest,
            "git_sha": git_sha,
            "mode": args.mode,
            "planned_runs": plans,
            "preprocessing_digest": preprocessing.content_digest,
            "split_digest": dataset.split_digest,
            "training_config_digest": training_config.content_digest,
        },
        context="M3BBenchmarkOrchestrationV1",
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "dataset_digest": dataset.dataset_digest,
                    "acceptance_report_digest": dataset.acceptance_report_digest,
                    "benchmark_config_digest": benchmark.content_digest,
                    "orchestration_identity": orchestration_identity,
                    "split_digest": dataset.split_digest,
                    "training_config_digest": training_config.content_digest,
                    "run_count": len(plans),
                    "runs": plans,
                    "test_evaluation": args.mode == "full",
                    "dry_run": True,
                },
                sort_keys=True,
            )
        )
        return 0
    root = _select_benchmark_root(args.output_dir, force_rerun=args.force_rerun)
    benchmark_summary = root / "benchmark-summary.json"
    if benchmark_summary.exists():
        existing = load_strict_report(
            benchmark_summary, expected_report_type=BENCHMARK_REPORT_TYPE
        )
        if (
            existing.payload.get("orchestration_identity") != orchestration_identity
            or existing.payload.get("completed") is not True
            or existing.payload.get("run_count") != len(plans)
        ):
            _fail("completed benchmark", "orchestration identity differs")
        reused_results: list[SeedValidationResult] = []
        reused_run_roots: dict[tuple[str, int], Path] = {}
        for model_type in benchmark.model_types:
            for seed in seed_inventory:
                resolved = resolve_model_config(
                    build_model(model_configs[model_type], seed=seed)
                )
                identity = _run_identity(
                    dataset, preprocessing, resolved, training_config, seed
                )
                reused_run_root = _run_directory(
                    root, model_type, seed, force_rerun=False
                )
                reused_results.append(
                    _validate_completed_run(
                        run_root=reused_run_root,
                        model_type=model_type,
                        seed=seed,
                        identity=identity,
                        dataset=dataset,
                        resolved_model=resolved,
                        training_config=training_config,
                        positive_class_weight=_positive_class_weight(
                            dataset, training_config
                        ),
                        git_sha=git_sha,
                        branch=branch,
                    )
                )
                reused_run_roots[(model_type.value, seed)] = reused_run_root
        if args.mode == "full":
            selection_report = load_strict_report(
                root / "selection-record.json",
                expected_report_type="selection_record_v1",
            )
            selection = selection_record_from_dict(selection_report.payload)
            recomputed_selection = select_model_architecture(
                reused_results,
                dataset_digest=dataset.dataset_digest,
                split_digest=dataset.split_digest,
                expected_model_types=[item.value for item in benchmark.model_types],
                expected_seeds=benchmark.seeds,
            )
            if (
                selection.content_digest != recomputed_selection.content_digest
                or existing.payload.get("selection_record_digest")
                != selection.content_digest
                or existing.payload.get("validation_selection") != selection.to_dict()
            ):
                _fail("completed benchmark", "selection binding differs")
            observed_artifacts: dict[str, object] = {}
            for key in sorted(reused_run_roots):
                relative = f"{key[0]}/seed-{key[1]}"
                observed_artifacts[relative] = _validate_test_artifacts(
                    evaluation_root=root / "evaluations" / key[0] / f"seed-{key[1]}",
                    seed_result=next(
                        item
                        for item in reused_results
                        if item.model_type == key[0] and item.seed == key[1]
                    ),
                    dataset=dataset,
                    selection=selection,
                )
            if existing.payload.get("evaluation_artifacts") != observed_artifacts:
                _fail("completed benchmark", "evaluation artifact inventory differs")
            if _file_digest(
                root / "summary.md", context="benchmark Markdown summary"
            ) != existing.payload.get("summary_markdown_digest"):
                _fail("completed benchmark", "Markdown summary digest differs")
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "completed": True,
                    "reused": True,
                    "output_root": str(root),
                    "report_digest": existing.content_digest,
                },
                sort_keys=True,
            )
        )
        return 0
    if root.exists() and any(root.iterdir()):
        allowed = {
            "evaluations",
            "gates",
            "preprocessing.json",
            "reruns",
            "runs",
            "selection-record.json",
            "smoke-gates.json",
            "summary.md",
        }
        unexpected = sorted(
            item.name for item in root.iterdir() if item.name not in allowed
        )
        if unexpected:
            _fail("benchmark root", f"unexpected entries: {unexpected}")
    root.mkdir(parents=True, exist_ok=True)
    preprocessing_path = root / "preprocessing.json"
    if preprocessing_path.exists():
        loaded = load_preprocessing_state(preprocessing_path)
        if loaded.content_digest != preprocessing.content_digest:
            _fail("benchmark preprocessing", "existing state differs")
    else:
        save_preprocessing_state(preprocessing, preprocessing_path)
    if args.mode == "smoke":
        gate_reports: dict[str, object] = {}
        gate_seed = benchmark.seeds[0]
        for model_type in benchmark.model_types:
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            gate_model = build_model(model_configs[model_type], seed=gate_seed)
            gate_resolved = resolve_model_config(gate_model)
            gate_identity = _run_identity(
                dataset,
                preprocessing,
                gate_resolved,
                training_config,
                gate_seed,
            )
            gate = run_tiny_overfit_gate(
                model=gate_model,
                resolved_model_config=gate_resolved,
                training_config=training_config,
                dataset=dataset,
                preprocessing=preprocessing,
                binding=_checkpoint_binding(gate_identity),
                output_directory=root / "gates" / model_type.value,
                device=args.device,
            )
            gate_reports[model_type.value] = {
                "sample_count": gate.sample_count,
                "optimization_steps": gate.optimization_steps,
                "initial_loss": gate.initial_loss,
                "final_loss": gate.final_loss,
                "final_accuracy": gate.final_accuracy,
                "checkpoint_reload_valid": gate.checkpoint_reload_valid,
                "checkpoint_resume_valid": gate.checkpoint_resume_valid,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                    if args.device.startswith("cuda")
                    else 0
                ),
            }
        save_strict_report(
            StrictReportV1(
                "smoke_gate_summary_v1",
                {
                    "acceptance_report_digest": dataset.acceptance_report_digest,
                    "dataset_digest": dataset.dataset_digest,
                    "orchestration_identity": orchestration_identity,
                    "split_digest": dataset.split_digest,
                    "models": gate_reports,
                },
            ),
            root / "smoke-gates.json",
        )
    validation_results: list[SeedValidationResult] = []
    run_roots: dict[tuple[str, int], Path] = {}
    seed_results_by_key: dict[tuple[str, int], SeedValidationResult] = {}
    for model_type in benchmark.model_types:
        for seed in seed_inventory:
            result, run_root = _train_one(
                root=root,
                model_type=model_type,
                model_config=model_configs[model_type],
                training_config=training_config,
                dataset=dataset,
                preprocessing=preprocessing,
                seed=seed,
                args=args,
                git_sha=git_sha,
                branch=branch,
            )
            validation_results.append(result)
            key = (model_type.value, seed)
            run_roots[key] = run_root
            seed_results_by_key[key] = result
    if args.mode == "smoke":
        smoke_report = StrictReportV1(
            BENCHMARK_REPORT_TYPE,
            {
                "mode": "smoke",
                "acceptance_report_digest": dataset.acceptance_report_digest,
                "benchmark_config_digest": benchmark.content_digest,
                "branch": branch,
                "git_sha": git_sha,
                "dataset_digest": dataset.dataset_digest,
                "orchestration_identity": orchestration_identity,
                "split_digest": dataset.split_digest,
                "preprocessing_digest": preprocessing.content_digest,
                "training_config_digest": training_config.content_digest,
                "run_count": len(validation_results),
                "test_evaluation_performed": False,
                "runs": [item.to_dict() for item in validation_results],
                "planned_runs": plans,
                "completed": True,
            },
        )
        save_strict_report(smoke_report, root / "benchmark-summary.json")
        load_strict_report(
            root / "benchmark-summary.json",
            expected_report_type=BENCHMARK_REPORT_TYPE,
            expected_content_digest=smoke_report.content_digest,
        )
        print(
            json.dumps(
                {
                    "mode": "smoke",
                    "completed": True,
                    "run_count": len(validation_results),
                    "output_root": str(root),
                    "report_digest": smoke_report.content_digest,
                },
                sort_keys=True,
            )
        )
        return 0
    selection = select_model_architecture(
        validation_results,
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        expected_model_types=[item.value for item in benchmark.model_types],
        expected_seeds=benchmark.seeds,
    )
    save_strict_report(
        StrictReportV1("selection_record_v1", selection.to_dict()),
        root / "selection-record.json",
    )
    evaluations: dict[tuple[str, int], ActionVerifierEvaluationReportV1] = {}
    prediction_sets: dict[tuple[str, int], tuple[CompactPredictionV1, ...]] = {}
    reference_test: SplitInferenceResultV1 | None = None
    for key in sorted(run_roots):
        evaluation, predictions, test_inference = _fit_and_test_one(
            run_root=run_roots[key],
            evaluation_root=(root / "evaluations" / key[0] / f"seed-{key[1]}"),
            seed_result=seed_results_by_key[key],
            dataset=dataset,
            selection=selection,
            device=args.device,
            batch_size=training_config.batch_size,
            target_failure_recall=training_config.target_failure_recall,
        )
        evaluations[key] = evaluation
        prediction_sets[key] = predictions
        if reference_test is None:
            reference_test = test_inference
        elif tuple(item.sample_id for item in reference_test.metadata) != tuple(
            item.sample_id for item in test_inference.metadata
        ):
            _fail("test evaluation", "sample ordering differs across runs")
    if reference_test is None:
        _fail("test evaluation", "no test predictions were produced")
    evaluation_artifacts: dict[str, object] = {}
    for key in sorted(run_roots):
        relative = f"{key[0]}/seed-{key[1]}"
        evaluation_artifacts[relative] = _validate_test_artifacts(
            evaluation_root=root / "evaluations" / key[0] / f"seed-{key[1]}",
            seed_result=seed_results_by_key[key],
            dataset=dataset,
            selection=selection,
        )
    nonlearned = _nonlearned_reports(dataset, reference_test)
    selected_test = _aggregate_selected_test(selection, evaluations)
    learned_test_aggregates = _aggregate_learned_test_reports(evaluations)
    comparisons = _statistical_comparisons(
        selection=selection,
        predictions=prediction_sets,
        benchmark=benchmark,
    )
    acceptance = _acceptance_assessment(evaluations, nonlearned)
    joint_advantage = _validation_joint_advantage(selection)
    training_summaries = [
        dict(
            load_strict_report(
                run_roots[key] / "training-summary.json",
                expected_report_type="training_run_summary_v1",
            ).payload
        )
        for key in sorted(run_roots)
    ]
    parameter_counts = {
        model_type.value: resolve_model_config(
            build_model(model_configs[model_type], seed=0)
        ).parameter_count
        for model_type in benchmark.model_types
    }
    summary_markdown = _render_final_markdown(
        branch=branch,
        git_sha=git_sha,
        dataset=dataset,
        selection=selection,
        evaluations=evaluations,
        training_summaries=training_summaries,
        parameter_counts=parameter_counts,
        nonlearned=nonlearned,
        comparisons=comparisons,
        acceptance=acceptance,
        joint_advantage=joint_advantage,
    )
    summary_markdown_path = _atomic_text(root / "summary.md", summary_markdown)
    summary_markdown_digest = _file_digest(
        summary_markdown_path, context="benchmark Markdown summary"
    )
    final_report = StrictReportV1(
        BENCHMARK_REPORT_TYPE,
        {
            "mode": "full",
            "completed": True,
            "acceptance_report_digest": dataset.acceptance_report_digest,
            "orchestration_identity": orchestration_identity,
            "dataset_digest": dataset.dataset_digest,
            "split_digest": dataset.split_digest,
            "preprocessing_digest": preprocessing.content_digest,
            "training_config_digest": training_config.content_digest,
            "benchmark_config_digest": benchmark.content_digest,
            "git_sha": git_sha,
            "branch": branch,
            "gpu_model": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "cpu"
            ),
            "run_count": len(validation_results),
            "planned_runs": plans,
            "training_runs": training_summaries,
            "seeds": list(benchmark.seeds),
            "parameter_counts": parameter_counts,
            "validation_selection": selection.to_dict(),
            "selection_record_digest": selection.content_digest,
            "evaluation_artifacts": evaluation_artifacts,
            "summary_markdown_digest": summary_markdown_digest,
            "selected_test_aggregate": selected_test,
            "learned_test_aggregates": learned_test_aggregates,
            "nonlearned_baselines": nonlearned,
            "statistical_comparisons": comparisons,
            "acceptance_targets": acceptance,
            "joint_state_action_validation_advantage": joint_advantage,
            "test_evaluation_performed_only_after_selection_freeze": True,
            "calibration_fitted_on_validation_only": True,
        },
    )
    save_strict_report(final_report, root / "benchmark-summary.json")
    load_strict_report(
        root / "benchmark-summary.json",
        expected_report_type=BENCHMARK_REPORT_TYPE,
        expected_content_digest=final_report.content_digest,
    )
    if (
        _file_digest(root / "summary.md", context="benchmark Markdown summary")
        != summary_markdown_digest
    ):
        _fail("benchmark reload", "Markdown summary digest differs")
    reloaded_selection_report = load_strict_report(
        root / "selection-record.json",
        expected_report_type="selection_record_v1",
    )
    if (
        selection_record_from_dict(reloaded_selection_report.payload).content_digest
        != selection.content_digest
    ):
        _fail("benchmark reload", "selection record differs")
    for key in sorted(run_roots):
        relative = f"{key[0]}/seed-{key[1]}"
        if evaluation_artifacts[relative] != _validate_test_artifacts(
            evaluation_root=root / "evaluations" / key[0] / f"seed-{key[1]}",
            seed_result=seed_results_by_key[key],
            dataset=dataset,
            selection=selection,
        ):
            _fail("benchmark reload", "evaluation artifact inventory differs")
    print(
        json.dumps(
            {
                "mode": "full",
                "completed": True,
                "run_count": len(validation_results),
                "selected_model_type": selection.selected_model_type,
                "acceptance_passed": acceptance["at_least_one_learned_model_passed"],
                "output_root": str(root),
                "report_digest": final_report.content_digest,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "BENCHMARK_REPORT_TYPE",
    "BenchmarkError",
    "MODEL_CONFIG_FILENAMES",
    "run_benchmark_command",
]
