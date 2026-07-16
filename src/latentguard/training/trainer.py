"""Deterministic M3B optimization, validation, early stopping, and smoke gates."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from latentguard.action_verifier import CandidateType, DatasetSplit
from latentguard.training.batching import (
    collate_numpy_examples,
    iter_numpy_batches,
    to_torch_batch,
)
from latentguard.training.checkpoint import (
    CheckpointBindingV1,
    inspect_training_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from latentguard.training.config import ResolvedModelConfig, TrainingConfig
from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
from latentguard.training.losses import FailureBCELoss, build_failure_loss
from latentguard.training.metrics import BinaryMetricReport, evaluate_binary_metrics
from latentguard.training.preprocessing import PreprocessingStateV1

TRAINING_HISTORY_FORMAT = "latentguard-m3b-training-history"
TRAINING_HISTORY_VERSION = 1


class TrainerError(RuntimeError):
    """Raised when a training gate, optimization step, or output contract fails."""


def _fail(context: str, reason: str) -> NoReturn:
    raise TrainerError(f"{context}: {reason}")


@dataclass(frozen=True, slots=True)
class EpochRecordV1:
    """One compact epoch-level scalar history entry."""

    epoch: int
    global_step: int
    training_loss: float
    validation_loss: float
    validation_corrupted_failure_auprc: float
    validation_corrupted_brier_score: float
    duration_seconds: float

    def __post_init__(self) -> None:
        """Reject non-finite or out-of-range epoch scalars."""

        if type(self.epoch) is not int or self.epoch < 0:
            _fail("EpochRecordV1.epoch", "expected non-negative integer")
        if type(self.global_step) is not int or self.global_step <= 0:
            _fail("EpochRecordV1.global_step", "expected positive integer")
        for name in (
            "training_loss",
            "validation_loss",
            "duration_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                _fail(f"EpochRecordV1.{name}", "expected finite non-negative value")
        for name in (
            "validation_corrupted_failure_auprc",
            "validation_corrupted_brier_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                _fail(f"EpochRecordV1.{name}", "expected finite [0, 1] value")

    def as_mapping(self) -> dict[str, object]:
        """Return a strict JSON-native history entry."""

        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "validation_corrupted_failure_auprc": (
                self.validation_corrupted_failure_auprc
            ),
            "validation_corrupted_brier_score": (self.validation_corrupted_brier_score),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class TrainingResultV1:
    """Completed bounded or full training result with validation-only selection."""

    best_epoch: int
    final_epoch: int
    global_step: int
    best_validation_failure_auprc: float
    best_checkpoint: Path
    final_checkpoint: Path
    history: tuple[EpochRecordV1, ...]
    duration_seconds: float
    peak_gpu_memory_bytes: int
    positive_class_weight: float | None


@dataclass(frozen=True, slots=True)
class TinyOverfitResultV1:
    """Deterministic infrastructure-gate result for one learned architecture."""

    sample_count: int
    optimization_steps: int
    initial_loss: float
    final_loss: float
    final_accuracy: float
    checkpoint_reload_valid: bool
    checkpoint_resume_valid: bool


def set_training_seed(seed: int, *, deterministic_algorithms: bool) -> None:
    """Seed Python, NumPy, PyTorch CPU, and every visible CUDA generator."""

    if type(seed) is not int or not 0 <= seed < 2**32:
        _fail("set_training_seed.seed", "expected uint32")
    if type(deterministic_algorithms) is not bool:
        _fail("set_training_seed.deterministic_algorithms", "expected boolean")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def _split_targets(
    dataset: AcceptedActionVerifierDatasetV1, split: DatasetSplit
) -> Tensor:
    values = [
        dataset[index].failure_target for index in dataset.indices_for_split(split)
    ]
    return torch.tensor(values, dtype=torch.float32)


def bounded_split_indices(
    dataset: AcceptedActionVerifierDatasetV1,
    split: DatasetSplit,
    *,
    limit_samples: int | None,
) -> tuple[int, ...]:
    """Apply a smoke limit while preserving complete source trajectories."""

    indices = dataset.indices_for_split(split)
    if limit_samples is None:
        return indices
    if type(limit_samples) is not int or limit_samples <= 0:
        _fail("bounded_split_indices.limit_samples", "expected positive integer")
    by_trajectory: dict[str, list[int]] = {}
    for index in indices:
        trajectory = dataset.reporting_for(index).source_trajectory
        by_trajectory.setdefault(trajectory, []).append(index)
    selected: list[int] = []
    for trajectory in sorted(by_trajectory):
        members = by_trajectory[trajectory]
        if selected and len(selected) + len(members) > limit_samples:
            break
        if not selected and len(members) > limit_samples:
            _fail(
                "bounded_split_indices.limit_samples",
                "limit is smaller than one complete source trajectory",
            )
        selected.extend(members)
    if not selected:
        _fail("bounded_split_indices", "bounded split is empty")
    return tuple(sorted(selected))


def _tensor_batch(batch: object, device: torch.device) -> tuple[Tensor, ...]:
    converted = to_torch_batch(batch, device=str(device))  # type: ignore[arg-type]
    return (
        cast(Tensor, converted.state_vectors),
        cast(Tensor, converted.action_chunks),
        cast(Tensor, converted.action_masks),
        cast(Tensor, converted.failure_targets),
        cast(Tensor, converted.sample_indices),
    )


def _finite_gradients(model: nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.isfinite(value).all()) for value in gradients
    )


def _evaluate_validation(
    model: nn.Module,
    dataset: AcceptedActionVerifierDatasetV1,
    preprocessing: PreprocessingStateV1,
    loss: FailureBCELoss,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, BinaryMetricReport]:
    model.eval()
    losses: list[float] = []
    logits_parts: list[NDArray[np.float64]] = []
    target_parts: list[NDArray[np.int64]] = []
    index_parts: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for numpy_batch in iter_numpy_batches(
            dataset,
            dataset.indices_for_split(DatasetSplit.VALIDATION),
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            preprocessing=preprocessing,
        ):
            state, actions, mask, targets, sample_indices = _tensor_batch(
                numpy_batch, device
            )
            logits = model(state, actions, mask)
            if not bool(torch.isfinite(logits).all()):
                _fail("validation", "model produced non-finite logits")
            batch_loss = loss(logits, targets)
            if not bool(torch.isfinite(batch_loss)):
                _fail("validation", "loss is non-finite")
            losses.append(float(batch_loss.item()) * targets.numel())
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float64))
            target_parts.append(targets.detach().cpu().numpy().astype(np.int64))
            index_parts.append(sample_indices.detach().cpu().numpy().astype(np.int64))
    logits_array = np.concatenate(logits_parts)
    targets_array = np.concatenate(target_parts)
    indices_array = np.concatenate(index_parts)
    corrupted = np.asarray(
        [
            dataset.reporting_for(int(index)).candidate_type is CandidateType.CORRUPTED
            for index in indices_array
        ],
        dtype=np.bool_,
    )
    if not bool(np.any(corrupted)):
        _fail("validation", "corrupted-only selection subset is empty")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -80.0, 80.0)))
    report = evaluate_binary_metrics(
        targets_array[corrupted], probabilities[corrupted], threshold=0.5
    )
    if report.failure_auprc.value is None or report.brier_score.value is None:
        _fail("validation", "selection metrics require both corrupted classes")
    return float(sum(losses) / len(targets_array)), report


def run_forward_backward_gate(
    model: nn.Module,
    dataset: AcceptedActionVerifierDatasetV1,
    preprocessing: PreprocessingStateV1,
    training_config: TrainingConfig,
    *,
    device: str = "cpu",
) -> float:
    """Run one finite forward/loss/backward gate without changing optimizer state."""

    target_device = torch.device(device)
    model.to(target_device)
    indices = dataset.indices_for_split(DatasetSplit.TRAIN)
    numpy_batch = next(
        iter_numpy_batches(
            dataset,
            indices,
            batch_size=min(training_config.batch_size, len(indices)),
            shuffle=False,
            seed=0,
            preprocessing=preprocessing,
        )
    )
    state, actions, mask, targets, _ = _tensor_batch(numpy_batch, target_device)
    loss_fn = build_failure_loss(
        training_config.loss_mode, _split_targets(dataset, DatasetSplit.TRAIN)
    )
    loss_fn.to(target_device)
    model.zero_grad(set_to_none=True)
    logits = model(state, actions, mask)
    loss = loss_fn(logits, targets)
    loss.backward()
    if not bool(torch.isfinite(loss)) or not _finite_gradients(model):
        _fail("forward_backward_gate", "non-finite loss or gradients")
    model.zero_grad(set_to_none=True)
    return float(loss.item())


def run_tiny_overfit_gate(
    *,
    model: nn.Module,
    resolved_model_config: ResolvedModelConfig,
    training_config: TrainingConfig,
    dataset: AcceptedActionVerifierDatasetV1,
    preprocessing: PreprocessingStateV1,
    binding: CheckpointBindingV1,
    output_directory: Path,
    device: str = "cpu",
    steps: int = 200,
    maximum_samples: int = 64,
) -> TinyOverfitResultV1:
    """Overfit a deterministic train-only subset and verify reload plus resume."""

    if type(steps) is not int or steps <= 0:
        _fail("tiny_overfit.steps", "expected positive integer")
    if type(maximum_samples) is not int or maximum_samples < 4:
        _fail("tiny_overfit.maximum_samples", "expected at least four")
    candidates = [
        index
        for index in dataset.indices_for_split(DatasetSplit.TRAIN)
        if dataset.reporting_for(index).candidate_type is CandidateType.CORRUPTED
    ]
    failures = [index for index in candidates if dataset[index].failure_target == 1]
    successes = [index for index in candidates if dataset[index].failure_target == 0]
    if not failures or not successes:
        _fail("tiny_overfit", "train-only subset requires both classes")
    half = maximum_samples // 2
    failure_by_group = {
        dataset.reporting_for(index).group_index: index for index in reversed(failures)
    }
    success_by_group = {
        dataset.reporting_for(index).group_index: index for index in reversed(successes)
    }
    failure_groups = set(failure_by_group)
    success_groups = set(success_by_group)
    shared_groups = sorted(failure_groups & success_groups)
    failure_only_groups = sorted(failure_groups - success_groups)
    success_only_groups = sorted(success_groups - failure_groups)
    samples_per_class = min(
        half,
        len(failure_groups),
        len(success_groups),
        len(failure_groups | success_groups) // 2,
    )
    failure_shared_count = max(0, samples_per_class - len(failure_only_groups))
    success_shared_count = max(0, samples_per_class - len(success_only_groups))
    if failure_shared_count + success_shared_count > len(shared_groups):
        _fail("tiny_overfit", "could not construct a disjoint two-class subset")
    selected_failure_groups = (
        failure_only_groups[: samples_per_class - failure_shared_count]
        + shared_groups[:failure_shared_count]
    )
    selected_success_groups = (
        success_only_groups[: samples_per_class - success_shared_count]
        + shared_groups[
            failure_shared_count : failure_shared_count + success_shared_count
        ]
    )
    selected_failures = [failure_by_group[group] for group in selected_failure_groups]
    selected_successes = [success_by_group[group] for group in selected_success_groups]
    selected = tuple(sorted((*selected_failures, *selected_successes)))
    if len(selected) < 4:
        _fail("tiny_overfit", "train-only subset is too small")
    numpy_batch = collate_numpy_examples(
        tuple(dataset[index] for index in selected), preprocessing=preprocessing
    )
    target_device = torch.device(device)
    model.to(target_device)
    state, actions, mask, targets, _ = _tensor_batch(numpy_batch, target_device)
    subset_weight = build_failure_loss(training_config.loss_mode, targets.cpu())
    subset_weight.to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    set_training_seed(
        0,
        deterministic_algorithms=training_config.deterministic_algorithms,
    )
    model.train()
    with torch.no_grad():
        initial_loss = float(subset_weight(model(state, actions, mask), targets).item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(state, actions, mask)
        loss_value = subset_weight(logits, targets)
        loss_value.backward()
        if not _finite_gradients(model):
            _fail("tiny_overfit", "non-finite gradients")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.gradient_clip_norm
        )
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_logits = model(state, actions, mask)
        final_loss = float(subset_weight(final_logits, targets).item())
        accuracy = float(
            torch.mean(((final_logits >= 0.0) == (targets == 1.0)).float()).item()
        )
    if not math.isfinite(final_loss) or not (
        final_loss < initial_loss * 0.5 and accuracy >= 0.90
    ):
        _fail(
            "tiny_overfit",
            "loss/accuracy gate failed "
            f"(initial={initial_loss}, final={final_loss}, accuracy={accuracy})",
        )
    gate_root = Path(output_directory).absolute()
    gate_root.mkdir(parents=True, exist_ok=True)
    checkpoint = save_training_checkpoint(
        gate_root / "tiny-overfit.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        binding=binding,
        model_configuration=resolved_model_config,
        training_configuration=training_config,
        preprocessing_state=preprocessing,
        epoch=0,
        global_step=steps,
        best_validation_metric=0.0,
        best_epoch=0,
        kind="smoke",
    )
    from latentguard.training.models import build_model

    reloaded_model = build_model(resolved_model_config.architecture, seed=999)
    reloaded_model.to(target_device)
    reloaded_optimizer = torch.optim.AdamW(
        reloaded_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loaded = load_training_checkpoint(
        checkpoint,
        expected_binding=binding,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        restore_rng=True,
    )
    reloaded_model.eval()
    with torch.no_grad():
        reloaded_logits = reloaded_model(state, actions, mask)
    reload_valid = bool(torch.equal(final_logits, reloaded_logits))
    if not reload_valid:
        _fail("tiny_overfit", "checkpoint reload changed logits")
    reloaded_model.train()
    reloaded_optimizer.zero_grad(set_to_none=True)
    resumed_loss = subset_weight(reloaded_model(state, actions, mask), targets)
    resumed_loss.backward()
    reloaded_optimizer.step()
    resume_valid = loaded.global_step == steps and bool(torch.isfinite(resumed_loss))
    if not resume_valid:
        _fail("tiny_overfit", "checkpoint resume gate failed")
    return TinyOverfitResultV1(
        sample_count=len(selected),
        optimization_steps=steps,
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_accuracy=accuracy,
        checkpoint_reload_valid=reload_valid,
        checkpoint_resume_valid=resume_valid,
    )


def train_action_verifier(
    *,
    model: nn.Module,
    resolved_model_config: ResolvedModelConfig,
    training_config: TrainingConfig,
    dataset: AcceptedActionVerifierDatasetV1,
    preprocessing: PreprocessingStateV1,
    binding: CheckpointBindingV1,
    output_directory: Path,
    seed: int,
    device: str,
    max_steps_override: int | None = None,
    max_epochs_override: int | None = None,
    limit_samples: int | None = None,
    resume_checkpoint: Path | None = None,
    profile_memory: bool = False,
) -> TrainingResultV1:
    """Train one baseline and select its checkpoint using validation only."""

    set_training_seed(
        seed, deterministic_algorithms=training_config.deterministic_algorithms
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        _fail("train_action_verifier.device", "CUDA is unavailable")
    if (
        max_steps_override is not None
        and max_steps_override != training_config.max_steps
    ):
        _fail(
            "train_action_verifier.max_steps",
            "override must already be resolved into the training config digest",
        )
    if (
        max_epochs_override is not None
        and max_epochs_override != training_config.max_epochs
    ):
        _fail(
            "train_action_verifier.max_epochs",
            "override must already be resolved into the training config digest",
        )
    max_steps = training_config.max_steps
    max_epochs = training_config.max_epochs
    run_root = Path(output_directory).absolute()
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    history_path = run_root / "history.json"
    resolved_limit_samples = (
        training_config.limit_samples if limit_samples is None else limit_samples
    )
    train_indices = bounded_split_indices(
        dataset, DatasetSplit.TRAIN, limit_samples=resolved_limit_samples
    )
    training_targets = _split_targets(dataset, DatasetSplit.TRAIN)
    loss_fn = build_failure_loss(training_config.loss_mode, training_targets)
    positive_class_weight = (
        None
        if loss_fn.positive_class_weight is None
        else float(loss_fn.positive_class_weight.item())
    )
    model.to(target_device)
    loss_fn.to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = None
    start_epoch = 0
    global_step = 0
    best_metric = 0.0
    best_epoch = 0
    history: list[EpochRecordV1] = []
    patience_count = 0
    if resume_checkpoint is not None:
        loaded = load_training_checkpoint(
            resume_checkpoint,
            expected_binding=binding,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
        )
        start_epoch = loaded.epoch + 1
        global_step = loaded.global_step
        best_metric = loaded.best_validation_metric
        best_epoch = loaded.best_epoch
        patience_count = loaded.early_stopping_patience_count
        if loaded.kind != "periodic":
            _fail("resume checkpoint", "exact trainer resume requires periodic kind")
        best_checkpoint_metadata = inspect_training_checkpoint(
            checkpoint_root / "best.pt"
        )
        if (
            best_checkpoint_metadata.progress.kind != "best"
            or best_checkpoint_metadata.progress.binding != binding
            or best_checkpoint_metadata.progress.epoch != loaded.best_epoch
            or best_checkpoint_metadata.progress.best_validation_metric
            != loaded.best_validation_metric
        ):
            _fail("resume checkpoint", "best checkpoint cursor differs")
        if history_path.is_file():
            try:
                raw_history = json.loads(history_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise TrainerError(f"resume history could not be read: {exc}") from exc
            if (
                not isinstance(raw_history, dict)
                or raw_history.get("format") != TRAINING_HISTORY_FORMAT
                or raw_history.get("version") != TRAINING_HISTORY_VERSION
                or not isinstance(raw_history.get("epochs"), list)
            ):
                _fail("resume history", "malformed history file")
            for item in raw_history["epochs"]:
                if not isinstance(item, dict):
                    _fail("resume history", "malformed epoch entry")
                history.append(EpochRecordV1(**item))
            if (
                tuple(item.epoch for item in history) != tuple(range(loaded.epoch + 1))
                or not history
                or history[-1].global_step != loaded.global_step
            ):
                _fail(
                    "resume history",
                    "history cursor does not match the periodic checkpoint",
                )
        else:
            _fail("resume history", "history file is unavailable")
    if target_device.type == "cuda" and profile_memory:
        torch.cuda.reset_peak_memory_stats(target_device)
    start_time = time.perf_counter()
    stop = (max_steps is not None and global_step >= max_steps) or (
        patience_count >= training_config.early_stopping_patience
    )
    final_epoch = history[-1].epoch if history else max(start_epoch - 1, 0)
    for epoch in range(start_epoch, max_epochs):
        if stop:
            break
        epoch_start = time.perf_counter()
        model.train()
        weighted_loss = 0.0
        seen = 0
        for numpy_batch in iter_numpy_batches(
            dataset,
            train_indices,
            batch_size=training_config.batch_size,
            shuffle=True,
            seed=seed + epoch,
            preprocessing=preprocessing,
        ):
            state, actions, mask, targets, _ = _tensor_batch(numpy_batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(state, actions, mask)
            loss_value = loss_fn(logits, targets)
            if not bool(torch.isfinite(loss_value)):
                _fail("training", "loss is non-finite")
            loss_value.backward()
            if not _finite_gradients(model):
                _fail("training", "gradient is missing or non-finite")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                _fail("training", "gradient norm is non-finite")
            optimizer.step()
            batch_count = int(targets.numel())
            weighted_loss += float(loss_value.item()) * batch_count
            seen += batch_count
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        if seen == 0:
            _fail("training", "epoch processed no examples")
        validation_loss, validation_report = _evaluate_validation(
            model,
            dataset,
            preprocessing,
            loss_fn,
            batch_size=training_config.batch_size,
            device=target_device,
        )
        failure_auprc = cast(float, validation_report.failure_auprc.value)
        brier = cast(float, validation_report.brier_score.value)
        record = EpochRecordV1(
            epoch=epoch,
            global_step=global_step,
            training_loss=weighted_loss / seen,
            validation_loss=validation_loss,
            validation_corrupted_failure_auprc=failure_auprc,
            validation_corrupted_brier_score=brier,
            duration_seconds=time.perf_counter() - epoch_start,
        )
        history.append(record)
        improved = (
            failure_auprc > best_metric + training_config.early_stopping_min_delta
        )
        if improved or not (checkpoint_root / "best.pt").exists():
            best_metric = failure_auprc
            best_epoch = epoch
            patience_count = 0
            save_training_checkpoint(
                checkpoint_root / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                binding=binding,
                model_configuration=resolved_model_config,
                training_configuration=training_config,
                preprocessing_state=preprocessing,
                epoch=epoch,
                global_step=global_step,
                best_validation_metric=best_metric,
                best_epoch=best_epoch,
                kind="best",
                early_stopping_patience_count=0,
            )
        else:
            patience_count += 1
        if (epoch + 1) % training_config.checkpoint_interval_epochs == 0:
            save_training_checkpoint(
                checkpoint_root / "periodic.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                binding=binding,
                model_configuration=resolved_model_config,
                training_configuration=training_config,
                preprocessing_state=preprocessing,
                epoch=epoch,
                global_step=global_step,
                best_validation_metric=best_metric,
                best_epoch=best_epoch,
                kind="periodic",
                early_stopping_patience_count=patience_count,
            )
        history_payload = {
            "format": TRAINING_HISTORY_FORMAT,
            "version": TRAINING_HISTORY_VERSION,
            "epochs": [item.as_mapping() for item in history],
        }
        history_path.write_text(
            json.dumps(history_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        final_epoch = epoch
        if stop or patience_count >= training_config.early_stopping_patience:
            break
    if not history:
        _fail("training", "no epoch completed")
    final_checkpoint = save_training_checkpoint(
        checkpoint_root / "final.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        binding=binding,
        model_configuration=resolved_model_config,
        training_configuration=training_config,
        preprocessing_state=preprocessing,
        epoch=final_epoch,
        global_step=global_step,
        best_validation_metric=best_metric,
        best_epoch=best_epoch,
        kind="final",
        early_stopping_patience_count=patience_count,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(target_device))
        if target_device.type == "cuda" and profile_memory
        else 0
    )
    return TrainingResultV1(
        best_epoch=best_epoch,
        final_epoch=final_epoch,
        global_step=global_step,
        best_validation_failure_auprc=best_metric,
        best_checkpoint=checkpoint_root / "best.pt",
        final_checkpoint=final_checkpoint,
        history=tuple(history),
        duration_seconds=time.perf_counter() - start_time,
        peak_gpu_memory_bytes=peak_memory,
        positive_class_weight=positive_class_weight,
    )


__all__ = [
    "TRAINING_HISTORY_FORMAT",
    "TRAINING_HISTORY_VERSION",
    "EpochRecordV1",
    "TrainerError",
    "TinyOverfitResultV1",
    "TrainingResultV1",
    "bounded_split_indices",
    "run_forward_backward_gate",
    "run_tiny_overfit_gate",
    "set_training_seed",
    "train_action_verifier",
]
