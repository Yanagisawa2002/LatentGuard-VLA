"""Deterministic M4B training, validation-only selection, and resume workflow."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray

from latentguard.action_verifier import CandidateType, DatasetSplit
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.metrics import (
    GroupCandidate,
    average_precision_score,
    evaluate_binary_metrics,
    evaluate_group_ranking,
)
from latentguard.vision_data.serialization import load_visual_image
from latentguard.visual_training.cache import LoadedFeatureCacheV1, LoadedTeacherCacheV1
from latentguard.visual_training.checkpoint import (
    VisualCheckpointBindingV1,
    checkpoint_content_digest,
    load_visual_checkpoint,
    save_visual_checkpoint,
)
from latentguard.visual_training.config import (
    VisualTrainingConfigV1,
)
from latentguard.visual_training.data import (
    TRAIN_DOMAIN_IDS,
    VisualActionTrainingDatasetV1,
    packet_id_for_epoch,
)
from latentguard.visual_training.models import VisualActionVerifier, count_parameters


class VisualTrainingError(ValueError):
    """Raised when an M4B run violates training, selection, or resume policy."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualTrainingError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    payload = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True, eq=False)
class ActionNormalizationV1:
    """Training-only action statistics shared by every principal M4B model."""

    dataset_digest: str
    training_split_digest: str
    mean: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]
    valid_step_count: int
    semantic: str = "m4b_training_split_action_standardization_v1"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for name in ("mean", "standard_deviation"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f8")
                or value.shape != (8,)
                or not bool(np.all(np.isfinite(value)))
            ):
                _fail(f"ActionNormalizationV1.{name}", "expected finite float64 [8]")
        if (
            not bool(np.all(self.standard_deviation > 0.0))
            or type(self.valid_step_count) is not int
            or self.valid_step_count <= 0
        ):
            _fail("ActionNormalizationV1", "invalid statistics")

    def as_mapping(self, *, include_digest: bool = True) -> dict[str, object]:
        """Return exact statistics and their path-independent binding."""
        result: dict[str, object] = {
            "dataset_digest": self.dataset_digest,
            "mean": self.mean.tolist(),
            "schema_version": self.schema_version,
            "semantic": self.semantic,
            "standard_deviation": self.standard_deviation.tolist(),
            "training_split_digest": self.training_split_digest,
            "valid_step_count": self.valid_step_count,
        }
        if include_digest:
            result["content_digest"] = self.content_digest
        return result

    @property
    def content_digest(self) -> str:
        """Return the complete action preprocessing digest."""
        return _digest(
            self.as_mapping(include_digest=False), "M4BActionNormalizationV1"
        )

    def apply(self, actions: NDArray[Any], masks: NDArray[Any]) -> NDArray[np.float32]:
        """Standardize only valid steps and force masked steps to exact zero."""
        converted = (
            np.asarray(actions, dtype=np.float64) - self.mean
        ) / self.standard_deviation
        converted = np.where(
            np.asarray(masks, dtype=np.bool_)[..., None], converted, 0.0
        )
        if not bool(np.all(np.isfinite(converted))):
            _fail("ActionNormalizationV1.apply", "normalization became non-finite")
        return np.asarray(converted, dtype=np.float32)


def fit_action_normalization(
    dataset: VisualActionTrainingDatasetV1,
) -> ActionNormalizationV1:
    """Fit shared action statistics from the M3A training split only."""
    rows: list[NDArray[Any]] = []
    for index in dataset.indices_for_split(DatasetSplit.TRAIN):
        example = dataset.examples[index]
        rows.append(example.action_chunk[example.action_mask])
    values = np.concatenate(rows, axis=0).astype(np.float64)
    mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64)
    std = np.maximum(std, np.float64(1e-6))
    return ActionNormalizationV1(
        dataset_digest=dataset.structured_dataset_digest,
        training_split_digest=dataset.split_digest,
        mean=mean,
        standard_deviation=std,
        valid_step_count=values.shape[0],
    )


def action_normalization_from_mapping(
    value: Mapping[str, object],
) -> ActionNormalizationV1:
    """Strictly reconstruct checkpoint-bound action preprocessing."""
    expected = {
        "content_digest",
        "dataset_digest",
        "mean",
        "schema_version",
        "semantic",
        "standard_deviation",
        "training_split_digest",
        "valid_step_count",
    }
    if set(value) != expected:
        _fail("ActionNormalizationV1", "unexpected or missing fields")
    mean = value["mean"]
    std = value["standard_deviation"]
    if not isinstance(mean, list) or not isinstance(std, list):
        _fail("ActionNormalizationV1", "statistics must be arrays")
    result = ActionNormalizationV1(
        dataset_digest=cast(str, value["dataset_digest"]),
        training_split_digest=cast(str, value["training_split_digest"]),
        mean=np.asarray(mean, dtype=np.float64),
        standard_deviation=np.asarray(std, dtype=np.float64),
        valid_step_count=cast(int, value["valid_step_count"]),
        semantic=cast(str, value["semantic"]),
        schema_version=cast(str, value["schema_version"]),
    )
    if value["content_digest"] != result.content_digest:
        _fail("ActionNormalizationV1", "content digest differs")
    return result


@dataclass(frozen=True, slots=True)
class DomainValidationResultV1:
    """Complete validation metrics for one fixed render domain."""

    domain_id: str
    failure_auprc: float
    corrupted_only_failure_auprc: float
    pairwise_concordance: float
    brier_score: float
    candidate_count: int

    def as_mapping(self) -> dict[str, object]:
        """Return compact scalar validation fields."""
        return {
            "brier_score": self.brier_score,
            "candidate_count": self.candidate_count,
            "corrupted_only_failure_auprc": self.corrupted_only_failure_auprc,
            "domain_id": self.domain_id,
            "failure_auprc": self.failure_auprc,
            "pairwise_concordance": self.pairwise_concordance,
        }


@dataclass(frozen=True, slots=True)
class VisualTrainingResultV1:
    """Compact result for one seed with validation-only checkpoint selection."""

    model_id: str
    seed: int
    selected_epoch: int
    completed_epochs: int
    global_steps: int
    mean_corrupted_only_failure_auprc: float
    worst_domain_corrupted_only_failure_auprc: float
    domain_results: tuple[DomainValidationResultV1, ...]
    total_parameters: int
    trainable_parameters: int
    peak_gpu_memory_bytes: int
    duration_seconds: float
    best_checkpoint_digest: str
    resumed: bool
    zero_work_resume: bool

    def as_mapping(self) -> dict[str, object]:
        """Return compact JSON-safe run results without predictions or paths."""
        body: dict[str, object] = {
            "best_checkpoint_digest": self.best_checkpoint_digest,
            "completed_epochs": self.completed_epochs,
            "domain_results": [item.as_mapping() for item in self.domain_results],
            "duration_seconds": self.duration_seconds,
            "global_steps": self.global_steps,
            "mean_corrupted_only_failure_auprc": self.mean_corrupted_only_failure_auprc,
            "model_id": self.model_id,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "resumed": self.resumed,
            "seed": self.seed,
            "selected_epoch": self.selected_epoch,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "worst_domain_corrupted_only_failure_auprc": (
                self.worst_domain_corrupted_only_failure_auprc
            ),
            "zero_work_resume": self.zero_work_resume,
        }
        return {**body, "content_digest": _digest(body, "M4BVisualTrainingResultV1")}


class VisualInputProvider:
    """Resolve cached features or authoritative raw RGB without metadata features."""

    def __init__(
        self,
        *,
        visual_dataset: object,
        visual_root: Path,
        feature_cache: LoadedFeatureCacheV1 | None,
        frozen: bool,
    ) -> None:
        self._packets = {
            item.packet_id: item for item in cast(Any, visual_dataset).packets
        }
        self._root = Path(visual_root)
        self._cache = feature_cache
        self._frozen = frozen
        if frozen and feature_cache is None:
            _fail("visual inputs", "frozen models require the feature cache")
        if not frozen and feature_cache is not None:
            _fail("visual inputs", "random model must use authoritative RGB")

    def batch(
        self, packet_ids: tuple[str, ...], *, view_count: int
    ) -> NDArray[np.float32]:
        """Return [B,V,512] features or normalized [B,3,3,224,224] RGB."""
        if self._frozen:
            assert self._cache is not None
            return np.stack(
                [
                    np.stack(
                        [
                            self._cache.feature(packet_id, slot)
                            for slot in range(view_count)
                        ]
                    )
                    for packet_id in packet_ids
                ]
            ).astype(np.float32)
        images: list[NDArray[Any]] = []
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)
        for packet_id in packet_ids:
            packet = self._packets[packet_id]
            views = [
                load_visual_image(self._root, view.image_reference)
                for view in packet.views
            ]
            rgb = (np.stack(views).astype(np.float32) / 255.0 - mean) / std
            images.append(np.transpose(rgb, (0, 3, 1, 2)))
        return cast(NDArray[np.float32], np.stack(images).astype(np.float32))


def _batch_arrays(
    dataset: VisualActionTrainingDatasetV1,
    indices: NDArray[np.int64],
    packet_ids: tuple[str, ...],
    provider: VisualInputProvider,
    normalization: ActionNormalizationV1,
    *,
    view_count: int,
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]
]:
    examples = [dataset.examples[int(index)] for index in indices]
    actions = np.stack([item.action_chunk for item in examples])
    masks = np.stack([item.action_mask for item in examples]).astype(np.bool_)
    targets = np.asarray([item.failure_target for item in examples], dtype=np.float32)
    return (
        provider.batch(packet_ids, view_count=view_count),
        normalization.apply(actions, masks),
        masks,
        targets,
    )


def _validate_domains(
    model: VisualActionVerifier,
    dataset: VisualActionTrainingDatasetV1,
    provider: VisualInputProvider,
    normalization: ActionNormalizationV1,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[DomainValidationResultV1, ...]:
    indices = np.asarray(
        dataset.indices_for_split(DatasetSplit.VALIDATION), dtype=np.int64
    )
    model.eval()
    results: list[DomainValidationResultV1] = []
    for domain_id in TRAIN_DOMAIN_IDS:
        logits: list[NDArray[Any]] = []
        targets: list[NDArray[Any]] = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            packet_ids = tuple(
                dataset.reporting[int(index)].packet_ids_by_domain[domain_id]
                for index in batch_indices
            )
            visual, actions, masks, batch_targets = _batch_arrays(
                dataset,
                batch_indices,
                packet_ids,
                provider,
                normalization,
                view_count=model.config.view_count,
            )
            with torch.inference_mode():
                output = model(
                    torch.tensor(visual, device=device),
                    torch.tensor(actions, device=device),
                    torch.tensor(masks, device=device),
                )
            logits.append(output.detach().cpu().numpy())
            targets.append(batch_targets)
        all_logits = np.concatenate(logits).astype(np.float64)
        all_targets = np.concatenate(targets).astype(np.int64)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(all_logits, -60.0, 60.0)))
        report = evaluate_binary_metrics(all_targets, probabilities, threshold=0.5)
        corrupted_mask = np.asarray(
            [
                dataset.reporting[int(index)].candidate_type is CandidateType.CORRUPTED
                for index in indices
            ]
        )
        corrupted = average_precision_score(
            all_targets[corrupted_mask], probabilities[corrupted_mask]
        )
        candidates = tuple(
            GroupCandidate(
                sample_id=dataset.reporting[int(index)].candidate_sample_id,
                group_id=dataset.reporting[int(index)].candidate_group_id,
                source_trajectory_id=dataset.reporting[int(index)].source_trajectory_id,
                candidate_type=dataset.reporting[int(index)].candidate_type.value,
                failure_target=dataset.examples[int(index)].failure_target,
                failure_probability=float(probabilities[position]),
            )
            for position, index in enumerate(indices)
        )
        ranking = evaluate_group_ranking(candidates)
        concordance = ranking.pairwise_success_over_failure_concordance.value
        overall = report.failure_auprc.value
        brier = report.brier_score.value
        if (
            overall is None
            or corrupted.value is None
            or concordance is None
            or brier is None
        ):
            _fail("visual validation", "required metric is undefined")
        results.append(
            DomainValidationResultV1(
                domain_id=domain_id,
                failure_auprc=overall,
                corrupted_only_failure_auprc=corrupted.value,
                pairwise_concordance=concordance,
                brier_score=brier,
                candidate_count=len(indices),
            )
        )
    return tuple(results)


def train_visual_action_verifier(
    model: VisualActionVerifier,
    dataset: VisualActionTrainingDatasetV1,
    visual_dataset: object,
    visual_root: Path,
    feature_cache: LoadedFeatureCacheV1 | None,
    teacher_cache: LoadedTeacherCacheV1 | None,
    config: VisualTrainingConfigV1,
    output_dir: Path,
    *,
    git_sha: str,
    backbone_digest: str,
    seed: int,
    device: str,
    resume: bool = False,
    max_steps: int | None = None,
    max_epochs: int | None = None,
    limit_samples: int | None = None,
    mixed_precision: bool = False,
) -> VisualTrainingResultV1:
    """Train one fixed seed with validation-only early stopping and exact resume."""
    if model.config.distillation_enabled != (teacher_cache is not None):
        _fail("visual training", "teacher cache use differs from model contract")
    if teacher_cache is not None and (
        teacher_cache.dataset_digest != dataset.structured_dataset_digest
        or teacher_cache.split_digest != dataset.split_digest
        or teacher_cache.candidate_ids
        != tuple(item.candidate_sample_id for item in dataset.reporting)
    ):
        _fail("visual training", "teacher cache dataset binding differs")
    if feature_cache is not None and (
        feature_cache.dataset_digest != dataset.visual_dataset_digest
        or feature_cache.git_sha != git_sha
        or feature_cache.source_dataset_digest != dataset.structured_dataset_digest
        or feature_cache.split_digest != dataset.split_digest
        or feature_cache.backbone_digest != backbone_digest
        or feature_cache.model_freeze_digest is not None
    ):
        _fail("visual training", "feature cache dataset binding differs")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        _fail("visual training", "CUDA is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.to(target)
    normalization = fit_action_normalization(dataset)
    binding = VisualCheckpointBindingV1(
        git_sha=git_sha,
        visual_dataset_digest=dataset.visual_dataset_digest,
        structured_dataset_digest=dataset.structured_dataset_digest,
        split_digest=dataset.split_digest,
        backbone_digest=backbone_digest,
        feature_cache_digest=None
        if feature_cache is None
        else feature_cache.content_digest,
        teacher_cache_digest=None
        if teacher_cache is None
        else teacher_cache.content_digest,
        action_preprocessing_digest=normalization.content_digest,
        model_config_digest=model.config.content_digest,
        training_config_digest=config.content_digest,
        seed=seed,
    )
    output = Path(output_dir).absolute()
    if output.exists() and not resume:
        _fail("visual training", "completed or partial output requires --resume")
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "checkpoint-last.pt"
    best_path = output / "checkpoint-best.pt"
    optimizer = torch.optim.AdamW(
        (item for item in model.parameters() if item.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.max_epochs)
    )
    scaler: Any | None = None
    if mixed_precision and target.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    global_step = 0
    best_metric = 0.0
    best_epoch = 0
    patience = 0
    resumed = False
    if resume and last_path.is_file():
        state = load_visual_checkpoint(
            last_path,
            expected_binding=binding,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        resumed = True
        start_epoch = state.epoch
        global_step = state.global_step
        best_metric = state.best_validation_auprc
        best_epoch = state.best_epoch
        patience = state.patience_count
        if state.complete:
            if not best_path.is_file():
                _fail("visual training resume", "best checkpoint is missing")
            load_visual_checkpoint(
                best_path,
                expected_binding=binding,
                model=model,
                restore_rng=False,
                model_only=True,
            )
            domains = _validate_domains(
                model,
                dataset,
                VisualInputProvider(
                    visual_dataset=visual_dataset,
                    visual_root=visual_root,
                    feature_cache=feature_cache,
                    frozen=model.config.backbone_frozen,
                ),
                normalization,
                device=target,
                batch_size=config.batch_size,
            )
            total, trainable = count_parameters(model)
            return VisualTrainingResultV1(
                model_id=model.config.model_id,
                seed=seed,
                selected_epoch=best_epoch,
                completed_epochs=start_epoch,
                global_steps=global_step,
                mean_corrupted_only_failure_auprc=float(
                    np.mean([x.corrupted_only_failure_auprc for x in domains])
                ),
                worst_domain_corrupted_only_failure_auprc=min(
                    x.corrupted_only_failure_auprc for x in domains
                ),
                domain_results=domains,
                total_parameters=total,
                trainable_parameters=trainable,
                peak_gpu_memory_bytes=0,
                duration_seconds=0.0,
                best_checkpoint_digest=checkpoint_content_digest(best_path),
                resumed=True,
                zero_work_resume=True,
            )
    elif resume:
        _fail("visual training resume", "checkpoint-last.pt is absent")
    provider = VisualInputProvider(
        visual_dataset=visual_dataset,
        visual_root=visual_root,
        feature_cache=feature_cache,
        frozen=model.config.backbone_frozen,
    )
    train_indices = np.asarray(
        dataset.indices_for_split(DatasetSplit.TRAIN), dtype=np.int64
    )
    if limit_samples is not None:
        if type(limit_samples) is not int or limit_samples <= 0:
            _fail("visual training", "limit_samples must be positive")
        train_indices = train_indices[:limit_samples]
    failures = sum(
        dataset.examples[int(index)].failure_target for index in train_indices
    )
    successes = len(train_indices) - failures
    if failures == 0 or successes == 0:
        _fail("visual training", "training subset must contain both classes")
    pos_weight = torch.tensor(
        [successes / failures], dtype=torch.float32, device=target
    )
    effective_epochs = (
        config.max_epochs if max_epochs is None else min(config.max_epochs, max_epochs)
    )
    if effective_epochs <= 0:
        _fail("visual training", "max_epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        _fail("visual training", "max_steps must be positive")
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    completed_epochs = start_epoch
    stop = False
    for epoch in range(start_epoch, effective_epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(train_indices)
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            packet_ids = tuple(
                packet_id_for_epoch(
                    dataset.reporting[int(index)], epoch=epoch, seed=seed
                )
                for index in batch_indices
            )
            visual, actions, masks, targets = _batch_arrays(
                dataset,
                batch_indices,
                packet_ids,
                provider,
                normalization,
                view_count=model.config.view_count,
            )
            visual_t = torch.tensor(visual, device=target)
            actions_t = torch.tensor(actions, device=target)
            masks_t = torch.tensor(masks, device=target)
            targets_t = torch.tensor(targets, device=target)
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = scaler is not None
            with torch.autocast(device_type=target.type, enabled=autocast_enabled):
                logits = model(visual_t, actions_t, masks_t)
                hard_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_t, pos_weight=pos_weight
                )
                loss = config.hard_label_weight * hard_loss
                if teacher_cache is not None:
                    teacher_values = torch.tensor(
                        teacher_cache.ensemble_probabilities[batch_indices],
                        dtype=logits.dtype,
                        device=target,
                    )
                    soft = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits / config.distillation_temperature,
                        teacher_values,
                    ) * (config.distillation_temperature**2)
                    loss = loss + config.teacher_weight * soft
            if not bool(torch.isfinite(loss)):
                _fail("visual training", "loss became non-finite")
            if scaler is None:
                cast(Any, loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                stop = True
                break
        scheduler.step()
        domains = _validate_domains(
            model,
            dataset,
            provider,
            normalization,
            device=target,
            batch_size=config.batch_size,
        )
        primary = float(
            np.mean([item.corrupted_only_failure_auprc for item in domains])
        )
        if primary > best_metric or not best_path.exists():
            best_metric, best_epoch, patience = primary, epoch + 1, 0
            save_visual_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                binding=binding,
                model_config=model.config.as_mapping(),
                training_config=config.as_mapping(),
                action_preprocessing=normalization.as_mapping(),
                epoch=epoch + 1,
                global_step=global_step,
                best_validation_auprc=best_metric,
                best_epoch=best_epoch,
                patience_count=patience,
                complete=False,
            )
        else:
            patience += 1
        completed_epochs = epoch + 1
        complete = (
            stop
            or patience >= config.early_stopping_patience
            or completed_epochs >= effective_epochs
        )
        save_visual_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            binding=binding,
            model_config=model.config.as_mapping(),
            training_config=config.as_mapping(),
            action_preprocessing=normalization.as_mapping(),
            epoch=completed_epochs,
            global_step=global_step,
            best_validation_auprc=best_metric,
            best_epoch=best_epoch,
            patience_count=patience,
            complete=complete,
        )
        if complete:
            break
    if not best_path.is_file():
        _fail("visual training", "no best checkpoint was produced")
    selected = load_visual_checkpoint(
        best_path,
        expected_binding=binding,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_rng=False,
    )
    save_visual_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        binding=binding,
        model_config=model.config.as_mapping(),
        training_config=config.as_mapping(),
        action_preprocessing=normalization.as_mapping(),
        epoch=selected.epoch,
        global_step=selected.global_step,
        best_validation_auprc=selected.best_validation_auprc,
        best_epoch=selected.best_epoch,
        patience_count=selected.patience_count,
        complete=True,
    )
    final_domains = _validate_domains(
        model,
        dataset,
        provider,
        normalization,
        device=target,
        batch_size=config.batch_size,
    )
    duration = time.perf_counter() - started
    if not math.isfinite(duration) or duration < 0.0:
        _fail("visual training", "invalid elapsed duration")
    peak = int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0
    total, trainable = count_parameters(model)
    return VisualTrainingResultV1(
        model_id=model.config.model_id,
        seed=seed,
        selected_epoch=best_epoch,
        completed_epochs=completed_epochs,
        global_steps=global_step,
        mean_corrupted_only_failure_auprc=float(
            np.mean([x.corrupted_only_failure_auprc for x in final_domains])
        ),
        worst_domain_corrupted_only_failure_auprc=min(
            x.corrupted_only_failure_auprc for x in final_domains
        ),
        domain_results=final_domains,
        total_parameters=total,
        trainable_parameters=trainable,
        peak_gpu_memory_bytes=peak,
        duration_seconds=duration,
        best_checkpoint_digest=checkpoint_content_digest(best_path),
        resumed=resumed,
        zero_work_resume=False,
    )


@dataclass(frozen=True, slots=True)
class PromotionRecordV1:
    """Immutable validation-only result of M4B seed-zero screening."""

    screening_results: tuple[VisualTrainingResultV1, ...]
    promoted_model_ids: tuple[str, ...]
    validation_winner: str
    escalation_condition: str | None
    schema_version: str = "1.0"

    def as_mapping(self) -> dict[str, object]:
        """Return compact promotion evidence with no test or external metrics."""
        body: dict[str, object] = {
            "escalation_condition": self.escalation_condition,
            "outcome_source": "m3a_validation_only",
            "promoted_model_ids": list(self.promoted_model_ids),
            "schema_version": self.schema_version,
            "screening_results": [item.as_mapping() for item in self.screening_results],
            "test_opened": False,
            "validation_winner": self.validation_winner,
        }
        return {**body, "content_digest": _digest(body, "M4BPromotionRecordV1")}


def create_seed_zero_promotion_record(
    results: tuple[VisualTrainingResultV1, ...],
) -> PromotionRecordV1:
    """Promote the winner, direct baseline, and one essential ablation."""
    expected_ids = {
        "random_resnet18_multiview_action",
        "frozen_resnet18_singleview_action",
        "frozen_resnet18_multiview_action",
        "frozen_resnet18_multiview_action_distilled",
    }
    if (
        len(results) != 4
        or {item.model_id for item in results} != expected_ids
        or {item.seed for item in results} != {0}
    ):
        _fail("seed-zero promotion", "requires exactly four complete seed-zero results")
    ordered = sorted(
        results,
        key=lambda item: (
            -item.mean_corrupted_only_failure_auprc,
            -item.worst_domain_corrupted_only_failure_auprc,
            -float(np.mean([x.pairwise_concordance for x in item.domain_results])),
            float(np.mean([x.brier_score for x in item.domain_results])),
            item.trainable_parameters,
            item.model_id,
        ),
    )
    winner = ordered[0].model_id
    promoted = [winner]
    direct = "frozen_resnet18_multiview_action"
    if direct not in promoted:
        promoted.append(direct)
    essential = (
        "frozen_resnet18_singleview_action"
        if "multiview" in winner
        else "random_resnet18_multiview_action"
    )
    if essential not in promoted and len(promoted) < 3:
        promoted.append(essential)
    gap = (
        ordered[0].mean_corrupted_only_failure_auprc
        - ordered[1].mean_corrupted_only_failure_auprc
    )
    return PromotionRecordV1(
        screening_results=tuple(sorted(results, key=lambda item: item.model_id)),
        promoted_model_ids=tuple(promoted),
        validation_winner=winner,
        escalation_condition="top_validation_gap_below_0.02" if gap < 0.02 else None,
    )


def write_training_result(
    path: Path, result: VisualTrainingResultV1 | PromotionRecordV1
) -> Path:
    """Atomically publish one compact immutable M4B scalar report."""
    destination = Path(path)
    if destination.exists():
        _fail("M4B report", "refuses to overwrite an existing report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapping = result.as_mapping()
    destination.write_text(
        json.dumps(mapping, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "ActionNormalizationV1",
    "DomainValidationResultV1",
    "PromotionRecordV1",
    "VisualInputProvider",
    "VisualTrainingError",
    "VisualTrainingResultV1",
    "create_seed_zero_promotion_record",
    "action_normalization_from_mapping",
    "fit_action_normalization",
    "train_visual_action_verifier",
    "write_training_result",
]
