"""Trusted content-bound M4B checkpoint save and exact resume."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from latentguard.replay.identity import canonical_json_bytes

VISUAL_CHECKPOINT_FORMAT = "latentguard-m4b-visual-training-checkpoint-v1"


class VisualCheckpointError(ValueError):
    """Raised when M4B checkpoint identity or trusted state differs."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualCheckpointError(f"{context}: {reason}")


def _valid_digest(value: object, context: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        _fail(context, "expected sha256 digest")
    return value


@dataclass(frozen=True, slots=True)
class VisualCheckpointBindingV1:
    """All semantic identities that must match before applying M4B state."""

    git_sha: str
    visual_dataset_digest: str
    structured_dataset_digest: str
    split_digest: str
    backbone_digest: str
    feature_cache_digest: str | None
    teacher_cache_digest: str | None
    action_preprocessing_digest: str
    model_config_digest: str
    training_config_digest: str
    seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.git_sha, str)
            or len(self.git_sha) != 40
            or any(item not in "0123456789abcdef" for item in self.git_sha)
        ):
            _fail("VisualCheckpointBindingV1.git_sha", "expected full lowercase SHA")
        for name in (
            "visual_dataset_digest",
            "structured_dataset_digest",
            "split_digest",
            "backbone_digest",
            "action_preprocessing_digest",
            "model_config_digest",
            "training_config_digest",
        ):
            _valid_digest(getattr(self, name), f"VisualCheckpointBindingV1.{name}")
        _valid_digest(
            self.feature_cache_digest,
            "VisualCheckpointBindingV1.feature_cache_digest",
            optional=True,
        )
        _valid_digest(
            self.teacher_cache_digest,
            "VisualCheckpointBindingV1.teacher_cache_digest",
            optional=True,
        )
        if type(self.seed) is not int or self.seed < 0:
            _fail("VisualCheckpointBindingV1.seed", "expected non-negative integer")

    def as_mapping(self) -> dict[str, object]:
        """Return path-independent binding fields."""
        return {
            "action_preprocessing_digest": self.action_preprocessing_digest,
            "backbone_digest": self.backbone_digest,
            "feature_cache_digest": self.feature_cache_digest,
            "git_sha": self.git_sha,
            "model_config_digest": self.model_config_digest,
            "seed": self.seed,
            "split_digest": self.split_digest,
            "structured_dataset_digest": self.structured_dataset_digest,
            "teacher_cache_digest": self.teacher_cache_digest,
            "training_config_digest": self.training_config_digest,
            "visual_dataset_digest": self.visual_dataset_digest,
        }

    @property
    def run_identity(self) -> str:
        """Return the deterministic M4B training run identity."""
        payload = canonical_json_bytes(self.as_mapping(), context="M4BTrainingRunV1")
        return f"m4b-run-sha256-{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class LoadedVisualCheckpointV1:
    """Validated checkpoint progress returned after state application."""

    epoch: int
    global_step: int
    best_validation_auprc: float
    best_epoch: int
    patience_count: int
    complete: bool
    binding: VisualCheckpointBindingV1


@dataclass(frozen=True, slots=True)
class VisualCheckpointMetadataV1:
    """Semantic checkpoint metadata inspected before model construction."""

    progress: LoadedVisualCheckpointV1
    model_config: Mapping[str, object]
    training_config: Mapping[str, object]
    action_preprocessing: Mapping[str, object]


def _binding_from_mapping(value: object) -> VisualCheckpointBindingV1:
    if not isinstance(value, Mapping):
        _fail("visual checkpoint binding", "expected object")
    try:
        return VisualCheckpointBindingV1(**dict(value))
    except TypeError as exc:
        raise VisualCheckpointError("visual checkpoint binding: fields differ") from exc


def _payload(path: Path) -> Mapping[str, object]:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file():
        _fail("visual checkpoint", "expected trusted regular file")
    try:
        value = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise VisualCheckpointError(f"visual checkpoint: {exc}") from exc
    required = {
        "action_preprocessing",
        "best_epoch",
        "best_validation_auprc",
        "binding",
        "complete",
        "epoch",
        "format",
        "global_step",
        "model_config",
        "model_state",
        "numpy_rng_state",
        "optimizer_state",
        "patience_count",
        "python_rng_state",
        "scaler_state",
        "scheduler_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "training_config",
        "version",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value["format"] != VISUAL_CHECKPOINT_FORMAT
        or value["version"] != 1
    ):
        _fail("visual checkpoint", "format or field inventory differs")
    return value


def checkpoint_content_digest(path: Path) -> str:
    """Hash trusted checkpoint bytes before any deserialization."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def inspect_visual_checkpoint(path: Path) -> VisualCheckpointMetadataV1:
    """Inspect strict trusted metadata without applying model or optimizer state."""
    payload = _payload(path)
    binding = _binding_from_mapping(payload["binding"])
    mappings: list[Mapping[str, object]] = []
    for name in ("model_config", "training_config", "action_preprocessing"):
        value = payload[name]
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            _fail(f"visual checkpoint {name}", "expected object")
        mappings.append(value)
    metric = payload["best_validation_auprc"]
    if not isinstance(metric, (int, float)) or isinstance(metric, bool):
        _fail("visual checkpoint", "best metric is malformed")
    for name in ("epoch", "global_step", "best_epoch", "patience_count"):
        if type(payload[name]) is not int:
            _fail("visual checkpoint", f"{name} is malformed")
    if type(payload["complete"]) is not bool:
        _fail("visual checkpoint", "complete is malformed")
    return VisualCheckpointMetadataV1(
        progress=LoadedVisualCheckpointV1(
            epoch=cast(int, payload["epoch"]),
            global_step=cast(int, payload["global_step"]),
            best_validation_auprc=float(metric),
            best_epoch=cast(int, payload["best_epoch"]),
            patience_count=cast(int, payload["patience_count"]),
            complete=payload["complete"],
            binding=binding,
        ),
        model_config=mappings[0],
        training_config=mappings[1],
        action_preprocessing=mappings[2],
    )


def save_visual_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: object | None,
    scaler: object | None,
    binding: VisualCheckpointBindingV1,
    model_config: Mapping[str, object],
    training_config: Mapping[str, object],
    action_preprocessing: Mapping[str, object],
    epoch: int,
    global_step: int,
    best_validation_auprc: float,
    best_epoch: int,
    patience_count: int,
    complete: bool,
) -> Path:
    """Atomically save complete optimizer/scaler/RNG state for exact resume."""
    if (
        type(epoch) is not int
        or epoch < 0
        or type(global_step) is not int
        or global_step < 0
    ):
        _fail("save visual checkpoint", "progress is invalid")
    if not 0.0 <= best_validation_auprc <= 1.0 or not 0 <= best_epoch <= epoch:
        _fail("save visual checkpoint", "best validation progress is invalid")
    if (
        type(patience_count) is not int
        or patience_count < 0
        or type(complete) is not bool
    ):
        _fail("save visual checkpoint", "patience or completion is invalid")
    scheduler_state = None
    if scheduler is not None:
        state_method = getattr(scheduler, "state_dict", None)
        if not callable(state_method):
            _fail("save visual checkpoint", "scheduler lacks state_dict")
        scheduler_state = state_method()
    scaler_state = None
    if scaler is not None:
        state_method = getattr(scaler, "state_dict", None)
        if not callable(state_method):
            _fail("save visual checkpoint", "scaler lacks state_dict")
        scaler_state = state_method()
    payload: dict[str, object] = {
        "action_preprocessing": dict(action_preprocessing),
        "best_epoch": best_epoch,
        "best_validation_auprc": best_validation_auprc,
        "binding": binding.as_mapping(),
        "complete": complete,
        "epoch": epoch,
        "format": VISUAL_CHECKPOINT_FORMAT,
        "global_step": global_step,
        "model_config": dict(model_config),
        "model_state": model.state_dict(),
        "numpy_rng_state": np.random.get_state(),
        "optimizer_state": optimizer.state_dict(),
        "patience_count": patience_count,
        "python_rng_state": random.getstate(),
        "scaler_state": scaler_state,
        "scheduler_state": scheduler_state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
        "training_config": dict(training_config),
        "version": 1,
    }
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_visual_checkpoint(
    path: Path,
    *,
    expected_binding: VisualCheckpointBindingV1,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    restore_rng: bool = True,
) -> LoadedVisualCheckpointV1:
    """Apply state only after exact identity, shape, and finiteness validation."""
    payload = _payload(path)
    binding = _binding_from_mapping(payload["binding"])
    if binding != expected_binding:
        _fail("visual checkpoint binding", "semantic identity drifted")
    state = payload["model_state"]
    if not isinstance(state, Mapping) or set(state) != set(model.state_dict()):
        _fail("visual checkpoint model", "parameter inventory differs")
    for key, expected in model.state_dict().items():
        observed = state[key]
        if (
            not isinstance(observed, Tensor)
            or observed.shape != expected.shape
            or observed.dtype != expected.dtype
        ):
            _fail("visual checkpoint model", f"tensor {key!r} differs")
        if torch.is_floating_point(observed) and not bool(
            torch.isfinite(observed).all()
        ):
            _fail("visual checkpoint model", f"tensor {key!r} is non-finite")
    try:
        model.load_state_dict(state, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state"])  # type: ignore[arg-type]
        if scheduler is not None:
            cast(Any, scheduler).load_state_dict(payload["scheduler_state"])
        elif payload["scheduler_state"] is not None:
            _fail("visual checkpoint scheduler", "runtime scheduler is absent")
        if scaler is not None:
            cast(Any, scaler).load_state_dict(payload["scaler_state"])
        elif payload["scaler_state"] is not None:
            _fail("visual checkpoint scaler", "runtime scaler is absent")
        if restore_rng:
            random.setstate(payload["python_rng_state"])  # type: ignore[arg-type]
            np.random.set_state(payload["numpy_rng_state"])  # type: ignore[arg-type]
            torch.set_rng_state(payload["torch_cpu_rng_state"])  # type: ignore[arg-type]
            cuda_states = payload["torch_cuda_rng_states"]
            if cuda_states:
                if (
                    not torch.cuda.is_available()
                    or not isinstance(cuda_states, list)
                    or len(cuda_states) != torch.cuda.device_count()
                ):
                    _fail("visual checkpoint RNG", "CUDA inventory changed")
                torch.cuda.set_rng_state_all(cuda_states)
    except VisualCheckpointError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise VisualCheckpointError(f"visual checkpoint apply failed: {exc}") from exc
    values = ("epoch", "global_step", "best_epoch", "patience_count")
    if any(type(payload[name]) is not int for name in values):
        _fail("visual checkpoint", "progress fields are malformed")
    metric = payload["best_validation_auprc"]
    if (
        not isinstance(metric, (int, float))
        or isinstance(metric, bool)
        or not 0.0 <= float(metric) <= 1.0
    ):
        _fail("visual checkpoint", "best metric is malformed")
    if type(payload["complete"]) is not bool:
        _fail("visual checkpoint", "completion field is malformed")
    return LoadedVisualCheckpointV1(
        epoch=cast(int, payload["epoch"]),
        global_step=cast(int, payload["global_step"]),
        best_validation_auprc=float(metric),
        best_epoch=cast(int, payload["best_epoch"]),
        patience_count=cast(int, payload["patience_count"]),
        complete=payload["complete"],
        binding=binding,
    )


__all__ = [
    "VISUAL_CHECKPOINT_FORMAT",
    "LoadedVisualCheckpointV1",
    "VisualCheckpointBindingV1",
    "VisualCheckpointError",
    "VisualCheckpointMetadataV1",
    "checkpoint_content_digest",
    "inspect_visual_checkpoint",
    "load_visual_checkpoint",
    "save_visual_checkpoint",
]
