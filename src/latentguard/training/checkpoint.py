"""Content-bound trusted checkpoint save, reload, and exact resume support."""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from latentguard.training.config import (
    ResolvedModelConfig,
    TrainingConfig,
    model_config_from_mapping,
    training_config_from_mapping,
)
from latentguard.training.preprocessing import PreprocessingStateV1

CHECKPOINT_FORMAT = "latentguard-m3b-training-checkpoint"
CHECKPOINT_VERSION = 2
CHECKPOINT_KINDS = frozenset({"periodic", "best", "final", "smoke"})

_PAYLOAD_FIELDS = frozenset(
    {
        "format",
        "version",
        "kind",
        "binding",
        "epoch",
        "global_step",
        "best_validation_metric",
        "best_epoch",
        "early_stopping_patience_count",
        "model_configuration",
        "training_configuration",
        "preprocessing_state",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
)


class CheckpointError(ValueError):
    """Raised when a trusted checkpoint is malformed or incompatibly bound."""


class StatefulScheduler(Protocol):
    """Minimal scheduler state interface required by checkpoint persistence."""

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state."""


def _fail(context: str, reason: str) -> NoReturn:
    raise CheckpointError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected sha256 digest")
    return value


@dataclass(frozen=True, slots=True)
class CheckpointBindingV1:
    """Every semantic identity that must match before checkpoint state is applied."""

    dataset_digest: str
    split_digest: str
    preprocessing_digest: str
    model_config_digest: str
    training_config_digest: str
    run_manifest_identity: str

    def __post_init__(self) -> None:
        """Validate fixed content identities and the path-independent run ID."""

        for name in (
            "dataset_digest",
            "split_digest",
            "preprocessing_digest",
            "model_config_digest",
            "training_config_digest",
        ):
            _digest(getattr(self, name), f"CheckpointBindingV1.{name}")
        if (
            not isinstance(self.run_manifest_identity, str)
            or not self.run_manifest_identity.startswith("m3b-run-sha256-")
            or len(self.run_manifest_identity) != len("m3b-run-sha256-") + 64
            or any(
                character not in "0123456789abcdef"
                for character in self.run_manifest_identity[len("m3b-run-sha256-") :]
            )
        ):
            _fail(
                "CheckpointBindingV1.run_manifest_identity",
                "expected canonical M3B run identity",
            )

    def as_mapping(self) -> dict[str, str]:
        """Return a strict serialization mapping."""

        return {
            "dataset_digest": self.dataset_digest,
            "split_digest": self.split_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "model_config_digest": self.model_config_digest,
            "training_config_digest": self.training_config_digest,
            "run_manifest_identity": self.run_manifest_identity,
        }


@dataclass(frozen=True, slots=True)
class LoadedCheckpointV1:
    """Validated progress state returned after a compatible checkpoint reload."""

    kind: str
    epoch: int
    global_step: int
    best_validation_metric: float
    best_epoch: int
    early_stopping_patience_count: int
    binding: CheckpointBindingV1


@dataclass(frozen=True, slots=True)
class CheckpointMetadataV1:
    """Strict semantic metadata reconstructed before model state is applied."""

    progress: LoadedCheckpointV1
    model_configuration: ResolvedModelConfig
    training_configuration: TrainingConfig
    preprocessing_state: PreprocessingStateV1


def _capture_numpy_rng_state() -> dict[str, object]:
    state = cast(tuple[Any, ...], np.random.get_state())
    return {
        "bit_generator": state[0],
        "keys": np.array(state[1], copy=True),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_numpy_rng_state(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        _fail("checkpoint.numpy_rng_state", "malformed state")
    bit_generator = value["bit_generator"]
    keys = value["keys"]
    position = value["position"]
    has_gauss = value["has_gauss"]
    cached = value["cached_gaussian"]
    if not isinstance(bit_generator, str) or not isinstance(keys, np.ndarray):
        _fail("checkpoint.numpy_rng_state", "malformed generator inventory")
    if keys.dtype != np.uint32 or keys.ndim != 1:
        _fail("checkpoint.numpy_rng_state", "malformed key array")
    if type(position) is not int or type(has_gauss) is not int:
        _fail("checkpoint.numpy_rng_state", "malformed scalar state")
    if type(cached) not in (int, float) or not math.isfinite(float(cached)):
        _fail("checkpoint.numpy_rng_state", "malformed cached Gaussian")
    np.random.set_state(
        (bit_generator, np.array(keys, copy=True), position, has_gauss, float(cached))
    )


def _binding_from_mapping(value: object) -> CheckpointBindingV1:
    fields = {
        "dataset_digest",
        "split_digest",
        "preprocessing_digest",
        "model_config_digest",
        "training_config_digest",
        "run_manifest_identity",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail("checkpoint.binding", "unexpected or missing fields")
    return CheckpointBindingV1(
        dataset_digest=value["dataset_digest"],
        split_digest=value["split_digest"],
        preprocessing_digest=value["preprocessing_digest"],
        model_config_digest=value["model_config_digest"],
        training_config_digest=value["training_config_digest"],
        run_manifest_identity=value["run_manifest_identity"],
    )


def _validate_progress(
    payload: Mapping[str, object],
) -> tuple[int, int, float, int, int]:
    epoch = payload["epoch"]
    global_step = payload["global_step"]
    best_metric = payload["best_validation_metric"]
    best_epoch = payload["best_epoch"]
    patience_count = payload["early_stopping_patience_count"]
    if type(epoch) is not int or epoch < 0:
        _fail("checkpoint.epoch", "expected non-negative integer")
    if type(global_step) is not int or global_step < 0:
        _fail("checkpoint.global_step", "expected non-negative integer")
    if type(best_metric) not in (int, float):
        _fail("checkpoint.best_validation_metric", "expected finite [0, 1] value")
    resolved_best_metric = float(cast(float, best_metric))
    if (
        not math.isfinite(resolved_best_metric)
        or not 0.0 <= resolved_best_metric <= 1.0
    ):
        _fail("checkpoint.best_validation_metric", "expected finite [0, 1] value")
    if type(best_epoch) is not int or not 0 <= best_epoch <= epoch:
        _fail("checkpoint.best_epoch", "expected completed epoch index")
    if type(patience_count) is not int or patience_count < 0:
        _fail(
            "checkpoint.early_stopping_patience_count",
            "expected non-negative integer",
        )
    return epoch, global_step, resolved_best_metric, best_epoch, patience_count


def _validate_model_state(model: nn.Module, value: object) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, Tensor)
        for key, item in value.items()
    ):
        _fail("checkpoint.model_state", "expected a tensor state dictionary")
    expected = model.state_dict()
    if set(value) != set(expected):
        _fail("checkpoint.model_state", "parameter inventory changed")
    for key, expected_tensor in expected.items():
        observed = value[key]
        if (
            observed.shape != expected_tensor.shape
            or observed.dtype != expected_tensor.dtype
        ):
            _fail("checkpoint.model_state", f"incompatible tensor {key!r}")
        if torch.is_floating_point(observed) and not bool(
            torch.isfinite(observed).all()
        ):
            _fail("checkpoint.model_state", f"non-finite tensor {key!r}")
    return value


def _validate_optimizer_state(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        _fail("checkpoint.optimizer_state", "malformed optimizer state")
    if not isinstance(value["state"], Mapping) or not isinstance(
        value["param_groups"], list
    ):
        _fail("checkpoint.optimizer_state", "malformed optimizer inventory")
    return dict(value)


def _load_trusted_payload(path: Path) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail("checkpoint.path", "expected a regular trusted file")
    try:
        # Checkpoints are pickle-based and must only come from this trusted run root.
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint load failed: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS:
        _fail("checkpoint", "unexpected or missing fields")
    if (
        payload["format"] != CHECKPOINT_FORMAT
        or payload["version"] != CHECKPOINT_VERSION
    ):
        _fail("checkpoint", "unsupported format or version")
    return payload


def compute_checkpoint_content_digest(path: Path) -> str:
    """Hash one regular trusted checkpoint without deserializing it."""

    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail("checkpoint.path", "expected a regular trusted file")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointError(f"checkpoint digest failed: {exc}") from exc
    return f"sha256:{digest.hexdigest()}"


def _reconstruct_metadata(payload: Mapping[str, object]) -> CheckpointMetadataV1:
    kind = payload["kind"]
    if not isinstance(kind, str) or kind not in CHECKPOINT_KINDS:
        _fail("checkpoint.kind", "unsupported value")
    binding = _binding_from_mapping(payload["binding"])
    epoch, global_step, best_metric, best_epoch, patience_count = _validate_progress(
        payload
    )
    raw_model = payload["model_configuration"]
    if not isinstance(raw_model, Mapping) or "parameter_count" not in raw_model:
        _fail("checkpoint.model_configuration", "malformed resolved configuration")
    parameter_count = raw_model["parameter_count"]
    if type(parameter_count) is not int or parameter_count <= 0:
        _fail("checkpoint.model_configuration", "invalid parameter count")
    architecture_payload = {
        key: value for key, value in raw_model.items() if key != "parameter_count"
    }
    try:
        architecture = model_config_from_mapping(architecture_payload)
        model_configuration = ResolvedModelConfig(
            architecture=architecture, parameter_count=parameter_count
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint.model_configuration: {exc}") from exc
    if model_configuration.content_digest != binding.model_config_digest:
        _fail("checkpoint.model_configuration", "digest mismatch")
    raw_training = payload["training_configuration"]
    if not isinstance(raw_training, Mapping):
        _fail("checkpoint.training_configuration", "malformed configuration")
    try:
        training_configuration = training_config_from_mapping(raw_training)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint.training_configuration: {exc}") from exc
    if training_configuration.content_digest != binding.training_config_digest:
        _fail("checkpoint.training_configuration", "digest mismatch")
    raw_preprocessing = payload["preprocessing_state"]
    if not isinstance(raw_preprocessing, Mapping):
        _fail("checkpoint.preprocessing_state", "malformed state")
    required_preprocessing = {
        "schema_version",
        "semantic",
        "dataset_digest",
        "training_split_digest",
        "state_component_count",
        "action_component_count",
        "state_observation_count",
        "valid_action_step_count",
        "minimum_standard_deviation",
        "statistics_dtype",
        "state_mean",
        "state_standard_deviation",
        "action_mean",
        "action_standard_deviation",
        "content_digest",
    }
    if set(raw_preprocessing) != required_preprocessing:
        _fail("checkpoint.preprocessing_state", "unexpected or missing fields")
    try:
        preprocessing_state = PreprocessingStateV1(
            dataset_digest=raw_preprocessing["dataset_digest"],
            training_split_digest=raw_preprocessing["training_split_digest"],
            state_observation_count=raw_preprocessing["state_observation_count"],
            valid_action_step_count=raw_preprocessing["valid_action_step_count"],
            minimum_standard_deviation=raw_preprocessing["minimum_standard_deviation"],
            state_mean=np.asarray(raw_preprocessing["state_mean"], dtype=np.float64),
            state_standard_deviation=np.asarray(
                raw_preprocessing["state_standard_deviation"], dtype=np.float64
            ),
            action_mean=np.asarray(raw_preprocessing["action_mean"], dtype=np.float64),
            action_standard_deviation=np.asarray(
                raw_preprocessing["action_standard_deviation"], dtype=np.float64
            ),
            state_component_count=raw_preprocessing["state_component_count"],
            action_component_count=raw_preprocessing["action_component_count"],
            statistics_dtype=raw_preprocessing["statistics_dtype"],
            semantic=raw_preprocessing["semantic"],
            schema_version=raw_preprocessing["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint.preprocessing_state: {exc}") from exc
    if (
        raw_preprocessing["content_digest"] != preprocessing_state.content_digest
        or preprocessing_state.content_digest != binding.preprocessing_digest
    ):
        _fail("checkpoint.preprocessing_state", "digest mismatch")
    if preprocessing_state.dataset_digest != binding.dataset_digest:
        _fail("checkpoint.preprocessing_state", "dataset binding mismatch")
    return CheckpointMetadataV1(
        progress=LoadedCheckpointV1(
            kind=kind,
            epoch=epoch,
            global_step=global_step,
            best_validation_metric=best_metric,
            best_epoch=best_epoch,
            early_stopping_patience_count=patience_count,
            binding=binding,
        ),
        model_configuration=model_configuration,
        training_configuration=training_configuration,
        preprocessing_state=preprocessing_state,
    )


def inspect_training_checkpoint(path: Path) -> CheckpointMetadataV1:
    """Inspect semantic metadata in a trusted checkpoint without applying state."""

    return _reconstruct_metadata(_load_trusted_payload(path))


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: StatefulScheduler | None,
    binding: CheckpointBindingV1,
    model_configuration: ResolvedModelConfig,
    training_configuration: TrainingConfig,
    preprocessing_state: PreprocessingStateV1,
    epoch: int,
    global_step: int,
    best_validation_metric: float,
    best_epoch: int,
    kind: str,
    early_stopping_patience_count: int = 0,
) -> Path:
    """Atomically save one trusted local checkpoint with complete resume state."""

    if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer):
        _fail("save_training_checkpoint", "invalid model or optimizer")
    if not isinstance(binding, CheckpointBindingV1):
        _fail("save_training_checkpoint.binding", "invalid binding")
    if kind not in CHECKPOINT_KINDS:
        _fail("save_training_checkpoint.kind", "unsupported checkpoint kind")
    progress_payload: dict[str, object] = {
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_metric": best_validation_metric,
        "best_epoch": best_epoch,
        "early_stopping_patience_count": early_stopping_patience_count,
    }
    _validate_progress(progress_payload)
    if model_configuration.content_digest != binding.model_config_digest:
        _fail("save_training_checkpoint", "model configuration digest changed")
    if training_configuration.content_digest != binding.training_config_digest:
        _fail("save_training_checkpoint", "training configuration digest changed")
    if preprocessing_state.content_digest != binding.preprocessing_digest:
        _fail("save_training_checkpoint", "preprocessing digest changed")
    payload: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "kind": kind,
        "binding": binding.as_mapping(),
        **progress_payload,
        "model_configuration": model_configuration.as_mapping(),
        "training_configuration": training_configuration.as_mapping(),
        "preprocessing_state": preprocessing_state.as_mapping(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (None if scheduler is None else scheduler.state_dict()),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": _capture_numpy_rng_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if temporary.exists() and temporary.is_file():
            temporary.unlink()
        raise CheckpointError(f"save_training_checkpoint: {exc}") from exc
    return destination


def load_training_checkpoint(
    path: Path,
    *,
    expected_binding: CheckpointBindingV1,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: StatefulScheduler | None = None,
    restore_rng: bool = True,
) -> LoadedCheckpointV1:
    """Load a trusted local checkpoint after all semantic compatibility checks."""

    if not isinstance(expected_binding, CheckpointBindingV1):
        _fail("load_training_checkpoint.expected_binding", "invalid binding")
    if not isinstance(model, nn.Module):
        _fail("load_training_checkpoint.model", "invalid model")
    payload = _load_trusted_payload(path)
    metadata = _reconstruct_metadata(payload)
    if metadata.progress.binding != expected_binding:
        _fail("checkpoint.binding", "semantic run identity changed")
    model_state = _validate_model_state(model, payload["model_state"])
    optimizer_state = _validate_optimizer_state(payload["optimizer_state"])
    if optimizer is None and metadata.progress.global_step > 0:
        # Evaluation-only reloads intentionally omit optimizer application.
        pass
    elif optimizer is not None and not isinstance(optimizer, Optimizer):
        _fail("load_training_checkpoint.optimizer", "invalid optimizer")
    scheduler_state = payload["scheduler_state"]
    if scheduler is None and scheduler_state is not None:
        _fail("checkpoint.scheduler_state", "scheduler configuration changed")
    if scheduler is not None and not isinstance(scheduler_state, dict):
        _fail("checkpoint.scheduler_state", "malformed scheduler state")
    validated_scheduler_state = cast(dict[str, Any] | None, scheduler_state)
    cpu_rng = payload["torch_cpu_rng_state"]
    cuda_rngs = payload["torch_cuda_rng_states"]
    if (
        not isinstance(cpu_rng, Tensor)
        or cpu_rng.dtype != torch.uint8
        or cpu_rng.ndim != 1
    ):
        _fail("checkpoint.torch_cpu_rng_state", "malformed state")
    if not isinstance(cuda_rngs, list) or any(
        not isinstance(item, Tensor) or item.dtype != torch.uint8 or item.ndim != 1
        for item in cuda_rngs
    ):
        _fail("checkpoint.torch_cuda_rng_states", "malformed state")
    python_rng = payload["python_rng_state"]
    if not isinstance(python_rng, tuple):
        _fail("checkpoint.python_rng_state", "malformed state")

    try:
        model.load_state_dict(model_state, strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            if validated_scheduler_state is None:
                _fail("checkpoint.scheduler_state", "missing scheduler state")
            scheduler.load_state_dict(validated_scheduler_state)
        if restore_rng:
            random.setstate(python_rng)
            _restore_numpy_rng_state(payload["numpy_rng_state"])
            torch.set_rng_state(cpu_rng)
            if cuda_rngs:
                if not torch.cuda.is_available():
                    _fail("checkpoint.torch_cuda_rng_states", "CUDA is unavailable")
                if len(cuda_rngs) != torch.cuda.device_count():
                    _fail("checkpoint.torch_cuda_rng_states", "CUDA inventory changed")
                torch.cuda.set_rng_state_all(cuda_rngs)
    except CheckpointError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint state application failed: {exc}") from exc
    return LoadedCheckpointV1(
        kind=metadata.progress.kind,
        epoch=metadata.progress.epoch,
        global_step=metadata.progress.global_step,
        best_validation_metric=metadata.progress.best_validation_metric,
        best_epoch=metadata.progress.best_epoch,
        early_stopping_patience_count=(metadata.progress.early_stopping_patience_count),
        binding=metadata.progress.binding,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_KINDS",
    "CHECKPOINT_VERSION",
    "CheckpointBindingV1",
    "CheckpointError",
    "CheckpointMetadataV1",
    "LoadedCheckpointV1",
    "compute_checkpoint_content_digest",
    "inspect_training_checkpoint",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
