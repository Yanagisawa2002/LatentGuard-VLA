"""Checkpointable single-device WM-v0 training and evaluation smoke runtime."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("world-model training requires the training extra") from exc

from .baselines import (
    DirectVerifierSuccessBaseline,
    accepted_direct_verifier_config,
    build_direct_verifier_success_baseline,
)
from .features import EncodedWorldModelSample, load_feature_sample
from .losses import LossWeights, WorldModelTargets, world_model_loss
from .model import (
    ActionConditionedLatentWorldModel,
    OutcomeOnlyWorldModel,
    WorldModelConfig,
)


class _FeatureDataset(Dataset[EncodedWorldModelSample]):
    def __init__(self, paths: tuple[Path, ...]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> EncodedWorldModelSample:
        return load_feature_sample(self.paths[index])


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    """Runtime knobs separate from the serializable model architecture."""

    seed: int
    model: str
    batch_size: int
    learning_rate: float
    maximum_steps: int
    evaluation_interval: int
    checkpoint_interval: int
    output_directory: str
    loss_weights: LossWeights


@dataclass(frozen=True, slots=True)
class _Batch:
    current_latents: Tensor
    future_latents: Tensor
    proprio: Tensor
    actions: Tensor
    action_mask: Tensor
    future_progress: Tensor
    progress_mask: Tensor
    event_labels: Tensor
    event_mask: Tensor
    terminal_success: Tensor
    terminal_success_mask: Tensor


def _collate(values: list[EncodedWorldModelSample]) -> _Batch:
    def array(name: str) -> np.ndarray[Any, Any]:
        return np.stack([getattr(value, name) for value in values])

    return _Batch(
        current_latents=torch.from_numpy(array("current_latents")),
        future_latents=torch.from_numpy(array("future_latents")),
        proprio=torch.from_numpy(array("proprio")),
        actions=torch.from_numpy(array("actions")),
        action_mask=torch.from_numpy(array("action_mask")),
        future_progress=torch.from_numpy(array("future_progress")),
        progress_mask=torch.from_numpy(array("progress_mask")),
        event_labels=torch.from_numpy(array("event_labels")),
        event_mask=torch.from_numpy(array("event_mask")),
        terminal_success=torch.tensor(
            [value.terminal_success for value in values], dtype=torch.float32
        ),
        terminal_success_mask=torch.tensor(
            [value.terminal_success_mask for value in values], dtype=torch.bool
        ),
    )


def _move(batch: _Batch, device: torch.device) -> _Batch:
    return _Batch(
        current_latents=batch.current_latents.to(device),
        future_latents=batch.future_latents.to(device),
        proprio=batch.proprio.to(device),
        actions=batch.actions.to(device),
        action_mask=batch.action_mask.to(device),
        future_progress=batch.future_progress.to(device),
        progress_mask=batch.progress_mask.to(device),
        event_labels=batch.event_labels.to(device),
        event_mask=batch.event_mask.to(device),
        terminal_success=batch.terminal_success.to(device),
        terminal_success_mask=batch.terminal_success_mask.to(device),
    )


def _masked_success_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    values = functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    if not bool(mask.any()):
        return values.sum() * 0.0
    return values[mask].mean()


def _cycle(loader: DataLoader[_Batch]) -> Iterator[_Batch]:
    while True:
        yield from loader


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_paths(
    directory: Path, *, split: str, limit_samples: int | None
) -> tuple[Path, ...]:
    index = json.loads((Path(directory) / "index.json").read_text("utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("records"), list):
        raise ValueError("feature cache index is malformed")
    paths = tuple(
        Path(directory) / f"{record['sample_id']}.npz"
        for record in index["records"]
        if isinstance(record, dict) and record.get("split") == split
    )
    if limit_samples is not None:
        paths = paths[:limit_samples]
    if not paths:
        raise ValueError(f"feature cache has no {split!r} samples")
    return paths


def _evaluate_loss(
    model: ActionConditionedLatentWorldModel,
    loader: DataLoader[_Batch],
    *,
    device: torch.device,
    weights: LossWeights,
    include_future_latent: bool,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            output = model(
                batch.current_latents,
                batch.proprio,
                batch.actions,
                batch.action_mask,
            )
            targets = WorldModelTargets(
                batch.future_latents,
                batch.future_progress,
                batch.progress_mask,
                batch.event_labels,
                batch.event_mask,
                batch.terminal_success,
                batch.terminal_success_mask,
            )
            values.append(
                float(
                    world_model_loss(
                        output,
                        targets,
                        weights=weights,
                        include_future_latent=include_future_latent,
                    )
                    .total.detach()
                    .cpu()
                    .item()
                )
            )
    model.train()
    return float(np.mean(values))


def _evaluate_direct_loss(
    model: DirectVerifierSuccessBaseline,
    loader: DataLoader[_Batch],
    *,
    device: torch.device,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move(batch, device)
            success_logit = model(batch.proprio, batch.actions, batch.action_mask)
            loss = _masked_success_loss(
                success_logit,
                batch.terminal_success,
                batch.terminal_success_mask,
            )
            values.append(float(loss.detach().cpu().item()))
    model.train()
    return float(np.mean(values))


def train_world_model(
    *,
    model_config: WorldModelConfig,
    run_config: TrainingRunConfig,
    feature_directory: Path,
    max_steps: int | None = None,
    limit_samples: int | None = None,
    resume: Path | None = None,
    dry_run: bool = False,
    device_name: str | None = None,
) -> dict[str, object]:
    """Train or dry-run one deterministic model with save/resume evidence."""

    _seed_everything(run_config.seed)
    output = Path(run_config.output_directory).absolute()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_paths = _feature_paths(
        feature_directory, split="train", limit_samples=limit_samples
    )
    validation_paths = _feature_paths(
        feature_directory, split="validation", limit_samples=limit_samples
    )
    resolved = {
        "device": str(device),
        "feature_directory": str(Path(feature_directory).absolute()),
        "model": asdict(model_config),
        "run": {**asdict(run_config), "loss_weights": asdict(run_config.loss_weights)},
        "schema_version": "1.0",
        "train_sample_count": len(train_paths),
        "validation_sample_count": len(validation_paths),
    }
    (output / "resolved-config.json").write_text(
        json.dumps(resolved, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        return {"dry_run": True, **resolved}
    model_type: type[ActionConditionedLatentWorldModel]
    include_future_latent = run_config.model == "wm_v0"
    direct_model: DirectVerifierSuccessBaseline | None = None
    world_model: ActionConditionedLatentWorldModel | None = None
    if include_future_latent:
        model_type = ActionConditionedLatentWorldModel
        world_model = model_type(model_config).to(device)
    elif run_config.model == "outcome_only":
        model_type = OutcomeOnlyWorldModel
        world_model = model_type(model_config).to(device)
    elif run_config.model == "direct_verifier":
        if (
            model_config.proprio_dimension != 38
            or model_config.action_dimension != 8
            or model_config.action_horizon != 16
        ):
            raise ValueError(
                "direct verifier baseline requires the accepted 38-D, 16-by-8 contract"
            )
        direct_model = build_direct_verifier_success_baseline(
            accepted_direct_verifier_config(), seed=run_config.seed
        ).to(device)
    else:
        raise ValueError("model must be 'wm_v0', 'outcome_only', or 'direct_verifier'")
    model: nn.Module = direct_model if direct_model is not None else world_model  # type: ignore[assignment]
    if model is None:  # pragma: no cover - exhaustive branches above
        raise RuntimeError("training model was not constructed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.learning_rate)
    step = 0
    best_validation = float("inf")
    if resume is not None:
        checkpoint: Any = torch.load(resume, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must be a mapping")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        best_validation = float(checkpoint["best_validation"])
    generator = torch.Generator().manual_seed(run_config.seed)
    train_loader = cast(
        DataLoader[_Batch],
        DataLoader(
            _FeatureDataset(train_paths),
            batch_size=run_config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=_collate,
        ),
    )
    validation_loader = cast(
        DataLoader[_Batch],
        DataLoader(
            _FeatureDataset(validation_paths),
            batch_size=run_config.batch_size,
            shuffle=False,
            collate_fn=_collate,
        ),
    )
    target_steps = min(max_steps or run_config.maximum_steps, run_config.maximum_steps)
    history: list[dict[str, object]] = []
    model.train()
    for batch in _cycle(train_loader):
        if step >= target_steps:
            break
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if direct_model is not None:
            success_logit = direct_model(
                batch.proprio, batch.actions, batch.action_mask
            )
            loss_total = _masked_success_loss(
                success_logit,
                batch.terminal_success,
                batch.terminal_success_mask,
            )
        else:
            if world_model is None:  # pragma: no cover - exhaustive construction
                raise RuntimeError("world model is unavailable")
            prediction = world_model(
                batch.current_latents,
                batch.proprio,
                batch.actions,
                batch.action_mask,
            )
            target = WorldModelTargets(
                batch.future_latents,
                batch.future_progress,
                batch.progress_mask,
                batch.event_labels,
                batch.event_mask,
                batch.terminal_success,
                batch.terminal_success_mask,
            )
            losses = world_model_loss(
                prediction,
                target,
                weights=run_config.loss_weights,
                include_future_latent=include_future_latent,
            )
            loss_total = losses.total
        loss_total.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        record: dict[str, object] = {
            "step": step,
            "train_loss": float(loss_total.detach().cpu().item()),
        }
        if step % run_config.evaluation_interval == 0 or step == target_steps:
            if direct_model is not None:
                validation = _evaluate_direct_loss(
                    direct_model, validation_loader, device=device
                )
            else:
                if world_model is None:  # pragma: no cover
                    raise RuntimeError("world model is unavailable")
                validation = _evaluate_loss(
                    world_model,
                    validation_loader,
                    device=device,
                    weights=run_config.loss_weights,
                    include_future_latent=include_future_latent,
                )
            record["validation_loss"] = validation
            best_validation = min(best_validation, validation)
        history.append(record)
        if step % run_config.checkpoint_interval == 0 or step == target_steps:
            temporary = output / f".checkpoint-{step}.tmp-{os.getpid()}.pt"
            checkpoint_path = output / f"checkpoint-{step}.pt"
            torch.save(
                {
                    "best_validation": best_validation,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                temporary,
            )
            temporary.replace(checkpoint_path)
    summary: dict[str, object] = {
        "best_validation_loss": best_validation,
        "completed_steps": step,
        "device": str(device),
        "history": history,
        "model": run_config.model,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "schema_version": "1.0",
    }
    (output / "training-summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def run_config_from_mapping(value: dict[str, Any]) -> TrainingRunConfig:
    """Resolve the runtime subset of the checked-in JSON-compatible YAML."""

    raw_weights = value["loss_weights"]
    if not isinstance(raw_weights, dict):
        raise ValueError("loss_weights must be a mapping")
    return TrainingRunConfig(
        seed=int(value["seed"]),
        model=str(value["model"]),
        batch_size=int(value["batch_size"]),
        learning_rate=float(value["learning_rate"]),
        maximum_steps=int(value["maximum_steps"]),
        evaluation_interval=int(value["evaluation_interval"]),
        checkpoint_interval=int(value["checkpoint_interval"]),
        output_directory=str(value["output_directory"]),
        loss_weights=LossWeights(
            future_latent=float(raw_weights["future_latent"]),
            progress=float(raw_weights["progress"]),
            events=float(raw_weights["events"]),
            success=float(raw_weights["success"]),
            uncertainty=float(raw_weights["uncertainty"]),
        ),
    )
