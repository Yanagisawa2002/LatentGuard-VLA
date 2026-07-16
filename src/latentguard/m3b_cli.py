"""Leakage-resistant M3B training, frozen evaluation, and benchmark commands."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from latentguard.training.config import TrainingConfig
    from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
    from latentguard.training.manifest import TrainingRunEnvironment

M3B_COMMANDS = frozenset(
    {
        "train-action-verifier",
        "evaluate-action-verifier",
        "benchmark-action-verifier",
    }
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _add_dataset_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument(
        "--anchor-manifest-dir",
        type=Path,
        required=True,
        help="accepted M3A anchor manifest used only for reporting slices",
    )


def add_m3b_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the three optional-training M3B commands without importing Torch."""

    train = subparsers.add_parser(
        "train-action-verifier",
        help="train one content-bound structured-state verifier baseline",
    )
    _add_dataset_gate_arguments(train)
    train.add_argument("--output-run-dir", type=Path, required=True)
    train.add_argument("--model-config", type=Path, required=True)
    train.add_argument("--training-config", type=Path, required=True)
    train.add_argument("--seed", type=_nonnegative_int, required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--max-steps", type=_positive_int)
    train.add_argument("--max-epochs", type=_positive_int)
    train.add_argument("--limit-samples", type=_positive_int)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--num-workers", type=_nonnegative_int)
    train.add_argument("--profile-memory", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-action-verifier",
        help="evaluate a trusted checkpoint with frozen validation artifacts",
    )
    _add_dataset_gate_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=_positive_int, default=128)
    evaluate.add_argument("--calibration", type=Path)
    evaluate.add_argument("--thresholds", type=Path)
    evaluate.add_argument("--selection-record", type=Path)
    evaluate.add_argument("--target-failure-recall", type=float, default=0.90)
    evaluate.add_argument("--dry-run", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark-action-verifier",
        help="run the fixed four-model, five-seed M3B baseline matrix",
    )
    _add_dataset_gate_arguments(benchmark)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument(
        "--config-dir", type=Path, default=Path("configs/training/m3b")
    )
    benchmark.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("configs/training/m3b/benchmark-five-seed.json"),
    )
    benchmark.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training/m3b/train-default.json"),
    )
    benchmark.add_argument("--mode", choices=("smoke", "full"), required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--force-rerun", action="store_true")
    benchmark.add_argument("--max-steps", type=_positive_int)
    benchmark.add_argument("--max-epochs", type=_positive_int)
    benchmark.add_argument("--limit-samples", type=_positive_int)
    benchmark.add_argument("--num-workers", type=_nonnegative_int)
    benchmark.add_argument("--profile-memory", action="store_true")
    benchmark.add_argument("--dry-run", action="store_true")


def _fail(context: str, reason: str) -> NoReturn:
    raise ValueError(f"{context}: {reason}")


def _anchor_selection_reasons(path: Path) -> tuple[dict[str, str], str]:
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )

    manifest = load_anchor_manifest(path)
    result = {
        record.anchor.anchor_id: record.anchor.selection_reason
        for record in manifest.records
    }
    if len(result) != len(manifest.records):
        _fail("anchor manifest", "duplicate anchor identity")
    return result, manifest.content_digest


def _load_accepted_dataset(
    args: argparse.Namespace,
) -> AcceptedActionVerifierDatasetV1:
    from latentguard.training.dataset import load_accepted_action_verifier_dataset

    reasons, manifest_digest = _anchor_selection_reasons(args.anchor_manifest_dir)
    return load_accepted_action_verifier_dataset(
        args.dataset_root,
        full_target_report=args.acceptance_report,
        anchor_selection_reasons=reasons,
        anchor_manifest_content_digest=manifest_digest,
    )


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return completed.stdout.strip()


def _resolved_training_configuration(args: argparse.Namespace) -> TrainingConfig:
    from latentguard.training.config import load_training_config

    configuration = load_training_config(args.training_config)
    return replace(
        configuration,
        max_steps=(
            configuration.max_steps if args.max_steps is None else args.max_steps
        ),
        max_epochs=(
            configuration.max_epochs if args.max_epochs is None else args.max_epochs
        ),
        limit_samples=(
            configuration.limit_samples
            if args.limit_samples is None
            else args.limit_samples
        ),
        num_workers=(
            configuration.num_workers if args.num_workers is None else args.num_workers
        ),
    )


def _run_environment(
    launch_arguments: tuple[str, ...],
) -> TrainingRunEnvironment:
    import numpy as np
    import torch

    from latentguard.training.manifest import TrainingRunEnvironment

    gpu_model = None
    gpu_driver_version = None
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())
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
        launch_command=launch_arguments,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        nondeterministic_operations=(),
    )


def _configure_deterministic_cuda_environment() -> None:
    """Set the fixed cuBLAS workspace contract before importing GPU modules."""

    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in (None, ":4096:8"):
        _fail(
            "CUBLAS_WORKSPACE_CONFIG",
            "M3B deterministic runs require ':4096:8'",
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _atomic_json(path: Path, value: object) -> Path:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination


def _prepare_run_directory(path: Path, *, resume: bool) -> Path:
    root = Path(path).absolute()
    if resume:
        if not root.is_dir() or root.is_symlink():
            _fail("resume", "output run directory does not exist")
        if (root / "completed.json").exists():
            _fail("resume", "completed immutable runs cannot be resumed")
    elif root.exists() and any(root.iterdir()):
        _fail("output run directory", "must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_train(args: argparse.Namespace) -> int:
    _configure_deterministic_cuda_environment()
    import torch

    from latentguard.action_verifier import DatasetSplit
    from latentguard.training.checkpoint import (
        CheckpointBindingV1,
        compute_checkpoint_content_digest,
        inspect_training_checkpoint,
    )
    from latentguard.training.config import (
        ResolvedModelConfig,
        TrainingConfig,
        load_model_config,
    )
    from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
    from latentguard.training.losses import build_failure_loss
    from latentguard.training.manifest import (
        TrainingRunIdentity,
        TrainingRunManifest,
        TrainingRunStatus,
    )
    from latentguard.training.models import build_model, resolve_model_config
    from latentguard.training.preprocessing import (
        fit_preprocessing_state,
        load_preprocessing_state,
        save_preprocessing_state,
    )
    from latentguard.training.reporting import StrictReportV1, save_strict_report
    from latentguard.training.trainer import (
        run_forward_backward_gate,
        set_training_seed,
        train_action_verifier,
    )

    dataset = _load_accepted_dataset(args)
    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        _fail("dataset", "accepted loader returned an unexpected value")
    model_configuration = load_model_config(args.model_config)
    training_configuration = _resolved_training_configuration(args)
    if not isinstance(training_configuration, TrainingConfig):
        _fail("training configuration", "unexpected resolved type")
    model = build_model(model_configuration, seed=args.seed)
    resolved_model = resolve_model_config(model)
    if not isinstance(resolved_model, ResolvedModelConfig):
        _fail("model", "could not resolve parameter inventory")
    preprocessing = fit_preprocessing_state(
        dataset,
        minimum_standard_deviation=(training_configuration.minimum_standard_deviation),
    )
    identity = TrainingRunIdentity(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved_model.content_digest,
        training_config_digest=training_configuration.content_digest,
        seed=args.seed,
        code_semantic_version="direct_action_verifier_training_v1",
    )
    training_targets = torch.tensor(
        [
            dataset[index].failure_target
            for index in dataset.indices_for_split(DatasetSplit.TRAIN)
        ],
        dtype=torch.float32,
    )
    configured_loss = build_failure_loss(
        training_configuration.loss_mode, training_targets
    )
    positive_class_weight = (
        None
        if configured_loss.positive_class_weight is None
        else float(configured_loss.positive_class_weight.item())
    )
    plan = {
        "run_id": identity.run_id,
        "dataset_digest": dataset.dataset_digest,
        "split_digest": dataset.split_digest,
        "preprocessing_digest": preprocessing.content_digest,
        "model_config_digest": resolved_model.content_digest,
        "training_config_digest": training_configuration.content_digest,
        "acceptance_report_digest": dataset.acceptance_report_digest,
        "model_type": model_configuration.model_type.value,
        "parameter_count": resolved_model.parameter_count,
        "seed": args.seed,
        "device": args.device,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0
    root = _prepare_run_directory(args.output_run_dir, resume=args.resume)
    preprocessing_path = root / "preprocessing.json"
    if preprocessing_path.exists():
        reloaded_preprocessing = load_preprocessing_state(preprocessing_path)
        if reloaded_preprocessing.content_digest != preprocessing.content_digest:
            _fail("resume", "preprocessing state differs")
    else:
        save_preprocessing_state(preprocessing, preprocessing_path)
    git_sha = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current")
    set_training_seed(
        args.seed,
        deterministic_algorithms=training_configuration.deterministic_algorithms,
    )
    started = datetime.now(UTC).isoformat()
    environment = _run_environment(tuple(sys.argv))
    manifest_path = root / "run-manifest.json"
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"resume: run manifest is unreadable: {exc}") from exc
        if (
            not isinstance(prior, dict)
            or prior.get("run_id") != identity.run_id
            or prior.get("git_sha") != git_sha
            or prior.get("branch") != branch
            or prior.get("status") != TrainingRunStatus.RUNNING.value
            or prior.get("acceptance_report_digest") != dataset.acceptance_report_digest
        ):
            _fail("resume", "run manifest provenance differs")
        prior_started = prior.get("started_at_utc")
        if not isinstance(prior_started, str) or not prior_started:
            _fail("resume", "run manifest start time is invalid")
        started = prior_started
    running_manifest = TrainingRunManifest(
        identity=identity,
        acceptance_report_digest=dataset.acceptance_report_digest,
        model_configuration=resolved_model,
        training_configuration=training_configuration,
        positive_class_weight=positive_class_weight,
        git_sha=git_sha,
        branch=branch,
        environment=environment,
        started_at_utc=started,
        finished_at_utc=None,
        status=TrainingRunStatus.RUNNING,
    )
    _atomic_json(manifest_path, running_manifest.as_mapping())
    binding = CheckpointBindingV1(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved_model.content_digest,
        training_config_digest=training_configuration.content_digest,
        run_manifest_identity=identity.run_id,
    )
    run_forward_backward_gate(
        model,
        dataset,
        preprocessing,
        training_configuration,
        device=args.device,
    )
    resume_checkpoint = root / "checkpoints" / "periodic.pt"
    if args.resume and not resume_checkpoint.is_file():
        _fail("resume", "periodic checkpoint is unavailable")
    result = train_action_verifier(
        model=model,
        resolved_model_config=resolved_model,
        training_config=training_configuration,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=root,
        seed=args.seed,
        device=args.device,
        resume_checkpoint=resume_checkpoint if args.resume else None,
        profile_memory=args.profile_memory,
    )
    if result.positive_class_weight != positive_class_weight:
        _fail("training result", "class weight differs from the run manifest")
    best_checkpoint_content_digest = compute_checkpoint_content_digest(
        result.best_checkpoint
    )
    final_checkpoint_content_digest = compute_checkpoint_content_digest(
        result.final_checkpoint
    )
    best_metadata = inspect_training_checkpoint(result.best_checkpoint)
    if (
        best_metadata.progress.kind != "best"
        or best_metadata.progress.epoch != result.best_epoch
        or best_metadata.progress.binding != binding
    ):
        _fail("training result", "best checkpoint binding differs")
    finished = datetime.now(UTC).isoformat()
    complete_manifest = TrainingRunManifest(
        identity=identity,
        acceptance_report_digest=dataset.acceptance_report_digest,
        model_configuration=resolved_model,
        training_configuration=training_configuration,
        positive_class_weight=positive_class_weight,
        git_sha=git_sha,
        branch=branch,
        environment=environment,
        started_at_utc=started,
        finished_at_utc=finished,
        status=TrainingRunStatus.COMPLETED,
    )
    _atomic_json(manifest_path, complete_manifest.as_mapping())
    summary = StrictReportV1(
        report_type="training_run_summary_v1",
        payload={
            **plan,
            "dry_run": False,
            "resolved_model_configuration": resolved_model.as_mapping(),
            "resolved_training_configuration": training_configuration.as_mapping(),
            "best_epoch": result.best_epoch,
            "best_checkpoint_content_digest": best_checkpoint_content_digest,
            "final_epoch": result.final_epoch,
            "final_checkpoint_content_digest": final_checkpoint_content_digest,
            "global_step": result.global_step,
            "best_validation_failure_auprc": (result.best_validation_failure_auprc),
            "duration_seconds": result.duration_seconds,
            "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
            "positive_class_weight": result.positive_class_weight,
            "environment": {
                "python_version": environment.python_version,
                "numpy_version": environment.numpy_version,
                "pytorch_version": environment.pytorch_version,
                "cuda_version": environment.cuda_version,
                "gpu_model": environment.gpu_model,
                "gpu_driver_version": environment.gpu_driver_version,
                "cublas_workspace_config": environment.cublas_workspace_config,
                "package_versions": list(environment.package_versions),
                "deterministic_algorithms": environment.deterministic_algorithms,
                "nondeterministic_operations": list(
                    environment.nondeterministic_operations
                ),
            },
            "checkpoint_reload_required": True,
            "completed": True,
        },
    )
    save_strict_report(summary, root / "summary.json")
    _atomic_json(
        root / "completed.json",
        {
            "best_checkpoint_content_digest": best_checkpoint_content_digest,
            "run_id": identity.run_id,
            "summary_digest": summary.content_digest,
        },
    )
    print(json.dumps({**plan, "completed": True}, sort_keys=True))
    if args.device.startswith("cuda") and args.profile_memory:
        torch.cuda.empty_cache()
    return 0


def _load_artifact_payload(path: Path, expected_type: str) -> dict[str, object]:
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(path, expected_report_type=expected_type)
    return dict(report.payload)


def _run_evaluate(args: argparse.Namespace) -> int:
    _configure_deterministic_cuda_environment()
    import torch

    from latentguard.action_verifier import DatasetSplit
    from latentguard.training.calibration import (
        compute_validation_prediction_digest,
        fit_temperature_scaling,
        fit_validation_thresholds,
        frozen_thresholds_from_dict,
        logits_to_failure_probabilities,
        temperature_calibration_from_dict,
    )
    from latentguard.training.checkpoint import (
        compute_checkpoint_content_digest,
        inspect_training_checkpoint,
        load_training_checkpoint,
    )
    from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
    from latentguard.training.evaluation import evaluate_action_verifier_logits
    from latentguard.training.inference import infer_action_verifier_split
    from latentguard.training.models import build_model, resolve_model_config
    from latentguard.training.preprocessing import validate_preprocessing_binding
    from latentguard.training.reporting import StrictReportV1, save_strict_report
    from latentguard.training.selection import selection_record_from_dict

    dataset = _load_accepted_dataset(args)
    if not isinstance(dataset, AcceptedActionVerifierDatasetV1):
        _fail("dataset", "accepted loader returned an unexpected value")
    metadata = inspect_training_checkpoint(args.checkpoint)
    torch.use_deterministic_algorithms(
        metadata.training_configuration.deterministic_algorithms
    )
    if (
        metadata.progress.binding.dataset_digest != dataset.dataset_digest
        or metadata.progress.binding.split_digest != dataset.split_digest
    ):
        _fail("checkpoint", "dataset or split digest differs")
    validate_preprocessing_binding(metadata.preprocessing_state, dataset)
    model = build_model(metadata.model_configuration.architecture, seed=0)
    if (
        resolve_model_config(model).content_digest
        != metadata.model_configuration.content_digest
    ):
        _fail("checkpoint", "resolved architecture differs")
    load_training_checkpoint(
        args.checkpoint,
        expected_binding=metadata.progress.binding,
        model=model,
        restore_rng=False,
    )
    checkpoint_content_digest = compute_checkpoint_content_digest(args.checkpoint)
    split = DatasetSplit(args.split)
    calibration = None
    thresholds = None
    selection_digest = None
    if split is DatasetSplit.TEST:
        if (
            args.calibration is None
            or args.thresholds is None
            or args.selection_record is None
        ):
            _fail(
                "test evaluation",
                "--calibration, --thresholds, and --selection-record are required",
            )
        calibration = temperature_calibration_from_dict(
            _load_artifact_payload(args.calibration, "temperature_calibration_v1")
        )
        thresholds = frozen_thresholds_from_dict(
            _load_artifact_payload(args.thresholds, "frozen_thresholds_v1")
        )
        selection = selection_record_from_dict(
            _load_artifact_payload(args.selection_record, "selection_record_v1")
        )
        matching_candidates = [
            candidate
            for candidate in selection.candidates
            if candidate.model_config_digest
            == metadata.model_configuration.content_digest
            and candidate.training_config_digest
            == metadata.training_configuration.content_digest
        ]
        authorized_results = [
            result
            for candidate in matching_candidates
            for result in candidate.seed_results
            if result.checkpoint_identity
            == metadata.progress.binding.run_manifest_identity
            and result.checkpoint_content_digest == checkpoint_content_digest
            and result.checkpoint_kind == metadata.progress.kind
            and result.checkpoint_epoch == metadata.progress.epoch
        ]
        if (
            selection.dataset_digest != dataset.dataset_digest
            or selection.split_digest != dataset.split_digest
            or len(matching_candidates) != 1
            or len(authorized_results) != 1
            or metadata.progress.kind != "best"
        ):
            _fail(
                "test evaluation",
                "frozen benchmark selection does not authorize this checkpoint",
            )
        if (
            calibration.dataset_digest != dataset.dataset_digest
            or calibration.split_digest != dataset.split_digest
            or calibration.checkpoint_identity
            != metadata.progress.binding.run_manifest_identity
            or calibration.checkpoint_content_digest != checkpoint_content_digest
            or calibration.model_config_digest
            != metadata.model_configuration.content_digest
            or calibration.preprocessing_digest
            != metadata.preprocessing_state.content_digest
            or thresholds.dataset_digest != dataset.dataset_digest
            or thresholds.split_digest != dataset.split_digest
            or thresholds.calibration_digest != calibration.content_digest
            or thresholds.validation_prediction_digest
            != calibration.validation_prediction_digest
        ):
            _fail(
                "test evaluation",
                "calibration or threshold binding differs from the checkpoint",
            )
        validation_inference = infer_action_verifier_split(
            model,
            dataset,
            metadata.preprocessing_state,
            split=DatasetSplit.VALIDATION,
            batch_size=args.batch_size,
            device=args.device,
        )
        observed_validation_digest = compute_validation_prediction_digest(
            validation_inference.logits,
            validation_inference.failure_targets,
        )
        if (
            observed_validation_digest != calibration.validation_prediction_digest
            or observed_validation_digest
            != authorized_results[0].validation_prediction_digest
        ):
            _fail(
                "test evaluation",
                "validation predictions differ from the frozen selection",
            )
        selection_digest = selection.content_digest
    plan = {
        "split": split.value,
        "sample_count": len(dataset.indices_for_split(split)),
        "dataset_digest": dataset.dataset_digest,
        "split_digest": dataset.split_digest,
        "checkpoint_identity": metadata.progress.binding.run_manifest_identity,
        "checkpoint_content_digest": checkpoint_content_digest,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0
    inference = infer_action_verifier_split(
        model,
        dataset,
        metadata.preprocessing_state,
        split=split,
        batch_size=args.batch_size,
        device=args.device,
    )
    if split is DatasetSplit.VALIDATION:
        calibration = fit_temperature_scaling(
            inference.logits,
            inference.failure_targets,
            split="validation",
            dataset_digest=dataset.dataset_digest,
            split_digest=dataset.split_digest,
            checkpoint_identity=metadata.progress.binding.run_manifest_identity,
            checkpoint_content_digest=checkpoint_content_digest,
            model_config_digest=metadata.model_configuration.content_digest,
            preprocessing_digest=metadata.preprocessing_state.content_digest,
        )
        thresholds = fit_validation_thresholds(
            calibration.apply(inference.logits),
            inference.failure_targets,
            split="validation",
            dataset_digest=dataset.dataset_digest,
            split_digest=dataset.split_digest,
            calibration_digest=calibration.content_digest,
            validation_prediction_digest=calibration.validation_prediction_digest,
            target_failure_recall=args.target_failure_recall,
        )
    output = Path(args.output_dir).absolute()
    if output.exists() and any(output.iterdir()):
        _fail("evaluation output", "must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    report, predictions = evaluate_action_verifier_logits(
        inference.logits,
        inference.failure_targets,
        inference.metadata,
        split=split.value,
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        checkpoint_identity=metadata.progress.binding.run_manifest_identity,
        checkpoint_content_digest=checkpoint_content_digest,
        model_config_digest=metadata.model_configuration.content_digest,
        preprocessing_digest=metadata.preprocessing_state.content_digest,
        calibration=calibration,
        thresholds=thresholds,
        selection_digest=selection_digest,
    )
    save_strict_report(
        StrictReportV1("action_verifier_evaluation_v1", report.to_dict()),
        output / "evaluation.json",
    )
    save_strict_report(
        StrictReportV1(
            "compact_predictions_v1",
            {"predictions": [item.to_dict() for item in predictions]},
        ),
        output / "predictions.json",
    )
    if (
        split is DatasetSplit.VALIDATION
        and calibration is not None
        and thresholds is not None
    ):
        save_strict_report(
            StrictReportV1("temperature_calibration_v1", calibration.to_dict()),
            output / "calibration.json",
        )
        save_strict_report(
            StrictReportV1("frozen_thresholds_v1", thresholds.to_dict()),
            output / "thresholds.json",
        )
    # Explicitly compute the same probability semantic used for threshold fitting.
    _ = logits_to_failure_probabilities(inference.logits)
    print(json.dumps({**plan, "completed": True}, sort_keys=True))
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    _configure_deterministic_cuda_environment()
    from latentguard.training.benchmark import run_benchmark_command

    return run_benchmark_command(args, load_dataset=_load_accepted_dataset)


def run_m3b_command(args: argparse.Namespace) -> int | None:
    """Dispatch one M3B command while preserving optional dependency isolation."""

    if args.command not in M3B_COMMANDS:
        return None
    try:
        if args.command == "train-action-verifier":
            return _run_train(args)
        if args.command == "evaluate-action-verifier":
            return _run_evaluate(args)
        return _run_benchmark(args)
    except KeyboardInterrupt:
        print(
            f"{args.command} interrupted: matching incomplete runs may resume",
            file=sys.stderr,
        )
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"{args.command} failed: {error}", file=sys.stderr)
        return 1


__all__ = ["M3B_COMMANDS", "add_m3b_subparsers", "run_m3b_command"]
