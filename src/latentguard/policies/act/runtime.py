"""Optional LeRobot 0.6 runtime for native PickCube ACT training and reload."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.data import (
    DemoEpisodeReference,
    PickCubeDemoSplit,
    action_chunk_at,
    collect_episode_references,
)
from latentguard.policies.act.types import (
    PICKCUBE_ACT_ACTION_FEATURE,
    PICKCUBE_ACT_IMAGE_FEATURE,
    PICKCUBE_ACT_LEROBOT_VERSION,
    PICKCUBE_ACT_STATE_FEATURE,
    PickCubeActExperimentConfig,
)


class PickCubeActRuntimeError(RuntimeError):
    """Raised when optional ACT data, model, or checkpoint runtime drifts."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActRuntimeError(f"{context}: {reason}")


def _read_mapping(path: Path, *, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeActRuntimeError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def installed_runtime_versions() -> Mapping[str, str | None]:
    """Return exact runtime versions required in run manifests."""
    try:
        lerobot_version = metadata.version("lerobot")
    except metadata.PackageNotFoundError as exc:
        raise PickCubeActRuntimeError("LeRobot is not installed") from exc
    if lerobot_version != PICKCUBE_ACT_LEROBOT_VERSION:
        _fail(
            "LeRobot",
            f"expected {PICKCUBE_ACT_LEROBOT_VERSION}, observed {lerobot_version}",
        )
    return {
        "cuda": torch.version.cuda,
        "lerobot": lerobot_version,
        "torch": torch.__version__,
    }


def seed_act_runtime(seed: int) -> Mapping[str, object]:
    """Seed Python, NumPy, Torch, and deterministic CUDA algorithms."""
    if type(seed) is not int or seed < 0:
        _fail("seed", "expected non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "python_hash_seed": str(seed),
        "seed": seed,
        "torch_deterministic_algorithms": True,
    }


class DeterministicResumeBatchSampler:
    """Infinite deterministic epoch permutations addressable by optimizer step."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        start_step: int = 0,
    ) -> None:
        if min(dataset_size, batch_size) < 1 or min(seed, start_step) < 0:
            _fail("batch sampler", "sizes and offsets are invalid")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.batches_per_epoch = (dataset_size + batch_size - 1) // batch_size

    def __iter__(self) -> Iterator[list[int]]:
        epoch, first_batch = divmod(self.start_step, self.batches_per_epoch)
        while True:
            generator = torch.Generator().manual_seed((self.seed + epoch) % (2**63 - 1))
            permutation = torch.randperm(
                self.dataset_size, generator=generator
            ).tolist()
            for batch_index in range(first_batch, self.batches_per_epoch):
                start = batch_index * self.batch_size
                yield permutation[start : start + self.batch_size]
            epoch += 1
            first_batch = 0


@dataclass(slots=True)
class _EpisodeArrays:
    rgb: NDArray[Any]
    state: NDArray[Any]
    action: NDArray[Any]


class PickCubeActDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Episode-bound frame dataset with future action chunks and no privileged input."""

    def __init__(
        self,
        root: Path,
        references: Sequence[DemoEpisodeReference],
        *,
        chunk_size: int,
        limit_samples: int | None = None,
        episode_cache_size: int = 16,
    ) -> None:
        self.root = Path(root).absolute()
        self.references = tuple(references)
        if not self.references:
            _fail("ACT dataset", "episode view is empty")
        if type(chunk_size) is not int or chunk_size < 1:
            _fail("ACT dataset", "chunk size is invalid")
        if type(episode_cache_size) is not int or episode_cache_size < 1:
            _fail("ACT dataset", "cache size is invalid")
        self.chunk_size = chunk_size
        self.episode_cache_size = episode_cache_size
        samples = [
            (episode_index, frame_index)
            for episode_index, reference in enumerate(self.references)
            for frame_index in range(reference.frame_count)
        ]
        if limit_samples is not None:
            if type(limit_samples) is not int or limit_samples < 1:
                _fail("ACT dataset", "sample limit is invalid")
            samples = samples[:limit_samples]
        if not samples:
            _fail("ACT dataset", "frame view is empty")
        self.samples = tuple(samples)
        self._cache: OrderedDict[str, _EpisodeArrays] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _arrays(self, reference: DemoEpisodeReference) -> _EpisodeArrays:
        cached = self._cache.pop(reference.episode_id, None)
        if cached is not None:
            self._cache[reference.episode_id] = cached
            return cached
        directory = self.root.joinpath(
            *PurePosixPath(reference.relative_directory).parts
        )
        rgb = np.load(directory / "rgb.npy", allow_pickle=False, mmap_mode="r")
        state = np.load(
            directory / "proprioception.npy", allow_pickle=False, mmap_mode="r"
        )
        action = np.load(directory / "actions.npy", allow_pickle=False, mmap_mode="r")
        if (
            rgb.dtype != np.dtype(np.uint8)
            or rgb.shape != (reference.frame_count, 224, 224, 3)
            or state.dtype != np.dtype(np.float32)
            or state.shape != (reference.frame_count, 18)
            or action.ndim != 2
            or action.shape != (reference.frame_count, 8)
            or not np.issubdtype(action.dtype, np.floating)
        ):
            _fail("ACT dataset", "persisted episode array contract changed")
        arrays = _EpisodeArrays(rgb=rgb, state=state, action=action)
        self._cache[reference.episode_id] = arrays
        while len(self._cache) > self.episode_cache_size:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, frame_index = self.samples[index]
        reference = self.references[episode_index]
        arrays = self._arrays(reference)
        chunk, padding = action_chunk_at(
            arrays.action,
            frame_index,
            self.chunk_size,
        )
        image = np.asarray(arrays.rgb[frame_index]).transpose(2, 0, 1).copy()
        state = np.asarray(arrays.state[frame_index]).copy()
        return {
            PICKCUBE_ACT_ACTION_FEATURE: torch.from_numpy(chunk),
            PICKCUBE_ACT_IMAGE_FEATURE: torch.from_numpy(image),
            PICKCUBE_ACT_STATE_FEATURE: torch.from_numpy(state),
            "action_is_pad": torch.from_numpy(padding),
        }


def validate_dataset_for_training(
    root: Path,
    *,
    experiment: PickCubeActExperimentConfig,
) -> tuple[
    tuple[DemoEpisodeReference, ...],
    Mapping[str, object],
    Mapping[str, object],
]:
    """Validate the immutable 500-episode inventory and train-only statistics."""
    dataset_root = Path(root).absolute()
    references = collect_episode_references(dataset_root)
    manifest = _read_mapping(
        dataset_root / "dataset_manifest.json", context="dataset manifest"
    )
    normalization = _read_mapping(
        dataset_root / "normalization_stats.json", context="normalization stats"
    )
    if manifest.get("schema_version") != "pickcube-act-dataset-manifest-v1":
        _fail("dataset manifest", "schema mismatch")
    if manifest.get("episode_count") != experiment.expected_episode_count:
        _fail("dataset manifest", "episode count differs from experiment")
    expected_counts = {
        PickCubeDemoSplit.TRAIN.value: experiment.expected_split_counts[0],
        PickCubeDemoSplit.VALIDATION.value: experiment.expected_split_counts[1],
        PickCubeDemoSplit.TEST.value: experiment.expected_split_counts[2],
    }
    if manifest.get("split_episode_counts") != expected_counts:
        _fail("dataset manifest", "split counts differ from 400/50/50")
    manifest_episodes = manifest.get("episodes")
    if not isinstance(manifest_episodes, Sequence) or isinstance(
        manifest_episodes, (str, bytes)
    ):
        _fail("dataset manifest", "episode inventory is missing")
    if [reference.to_mapping() for reference in references] != list(manifest_episodes):
        _fail("dataset manifest", "episode inventory or file digest changed")
    dataset_digest = manifest.get("dataset_digest")
    if normalization.get("source_dataset_digest") != dataset_digest:
        _fail("normalization stats", "source dataset digest differs")
    if normalization.get("source_split") != "train":
        _fail("normalization stats", "statistics are not train-only")
    if len(references) != experiment.expected_episode_count:
        _fail("dataset", "reloaded episode count changed")
    return references, manifest, normalization


def split_references(
    references: Sequence[DemoEpisodeReference], split: PickCubeDemoSplit
) -> tuple[DemoEpisodeReference, ...]:
    """Return an ordered episode-level split view."""
    result = tuple(item for item in references if item.split is split)
    if not result:
        _fail("dataset split", f"{split.value} is empty")
    return result


def _stat_tensor(
    value: Mapping[str, object],
    name: str,
    *,
    shape: tuple[int, ...],
) -> torch.Tensor:
    raw = value.get(name)
    tensor = torch.as_tensor(raw, dtype=torch.float32)
    if tensor.numel() != int(np.prod(shape)) or not torch.isfinite(tensor).all():
        _fail("normalization stats", f"{name} shape/value is invalid")
    return tensor.reshape(shape)


def processor_statistics(
    value: Mapping[str, object],
) -> Mapping[str, Mapping[str, torch.Tensor]]:
    """Translate strict train-only reports into LeRobot processor tensors."""
    feature_specs = (
        (PICKCUBE_ACT_IMAGE_FEATURE, "image", (3, 1, 1)),
        (PICKCUBE_ACT_STATE_FEATURE, "proprioception", (18,)),
        (PICKCUBE_ACT_ACTION_FEATURE, "action", (8,)),
    )
    result: dict[str, Mapping[str, torch.Tensor]] = {}
    for feature, field, shape in feature_specs:
        raw = value.get(field)
        if not isinstance(raw, Mapping):
            _fail("normalization stats", f"{field} is missing")
        if field == "image":
            converted: Mapping[str, object] = {
                "mean": raw.get("channel_mean"),
                "standard_deviation": raw.get("channel_standard_deviation"),
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [1.0, 1.0, 1.0],
            }
        else:
            converted = raw
        result[feature] = {
            "count": torch.tensor([1], dtype=torch.int64),
            "max": _stat_tensor(converted, "maximum", shape=shape),
            "mean": _stat_tensor(converted, "mean", shape=shape),
            "min": _stat_tensor(converted, "minimum", shape=shape),
            "std": _stat_tensor(converted, "standard_deviation", shape=shape),
        }
    return result


def build_policy_and_processors(
    experiment: PickCubeActExperimentConfig,
    statistics: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[Any, Any, Any]:
    """Build standard LeRobot ACT and its public processors without downloads."""
    installed_runtime_versions()
    try:
        from lerobot.configs import (  # type: ignore[import-not-found]
            FeatureType,
            NormalizationMode,
            PolicyFeature,
        )
        from lerobot.policies.act import (  # type: ignore[import-not-found]
            ACTConfig,
            ACTPolicy,
            make_act_pre_post_processors,
        )
    except (ImportError, RuntimeError, OSError) as exc:
        raise PickCubeActRuntimeError("LeRobot ACT runtime is unavailable") from exc
    model = experiment.model
    optimization = experiment.optimization
    config = ACTConfig(
        input_features={
            model.image_feature_key: PolicyFeature(
                FeatureType.VISUAL, model.image_shape_chw
            ),
            model.state_feature_key: PolicyFeature(
                FeatureType.STATE, (model.state_dimension,)
            ),
        },
        output_features={
            model.action_feature_key: PolicyFeature(
                FeatureType.ACTION, (model.action_dimension,)
            )
        },
        device=experiment.device,
        use_amp=False,
        use_peft=False,
        push_to_hub=False,
        chunk_size=model.chunk_size,
        n_action_steps=model.n_action_steps,
        n_obs_steps=model.n_obs_steps,
        normalization_mapping={
            "VISUAL": NormalizationMode(model.normalization_mode),
            "STATE": NormalizationMode(model.normalization_mode),
            "ACTION": NormalizationMode(model.normalization_mode),
        },
        vision_backbone=model.vision_backbone,
        pretrained_backbone_weights=model.pretrained_backbone_weights,
        replace_final_stride_with_dilation=False,
        pre_norm=False,
        dim_model=model.dim_model,
        n_heads=model.n_heads,
        dim_feedforward=model.dim_feedforward,
        feedforward_activation="relu",
        n_encoder_layers=model.n_encoder_layers,
        n_decoder_layers=model.n_decoder_layers,
        use_vae=model.use_vae,
        latent_dim=model.latent_dim,
        n_vae_encoder_layers=model.n_vae_encoder_layers,
        temporal_ensemble_coeff=None,
        dropout=model.dropout,
        kl_weight=model.kl_weight,
        optimizer_lr=optimization.learning_rate,
        optimizer_weight_decay=optimization.weight_decay,
        optimizer_lr_backbone=optimization.backbone_learning_rate,
    )
    if str(config.device) != experiment.device:
        _fail("ACT config", "requested CUDA device was silently changed")
    policy = ACTPolicy(config).to(torch.device(experiment.device))
    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        dataset_stats={
            feature: {name: tensor.clone() for name, tensor in values.items()}
            for feature, values in statistics.items()
        },
    )
    return policy, preprocessor, postprocessor


def build_optimizer(policy: Any, experiment: PickCubeActExperimentConfig) -> Any:
    """Build the pinned public ACT AdamW parameter groups."""
    groups = policy.get_optim_params()
    if not isinstance(groups, list) or len(groups) != 2:
        _fail("ACT optimizer", "expected backbone and non-backbone groups")
    groups[1]["lr"] = experiment.optimization.backbone_learning_rate
    return torch.optim.AdamW(
        groups,
        lr=experiment.optimization.learning_rate,
        weight_decay=experiment.optimization.weight_decay,
    )


def prepare_training_batch(batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Project only allowlisted tensors and convert RGB bytes to [0,1]."""
    expected = {
        PICKCUBE_ACT_IMAGE_FEATURE,
        PICKCUBE_ACT_STATE_FEATURE,
        PICKCUBE_ACT_ACTION_FEATURE,
        "action_is_pad",
    }
    if set(batch) != expected:
        _fail("ACT batch", "feature inventory changed")
    image = batch[PICKCUBE_ACT_IMAGE_FEATURE]
    state = batch[PICKCUBE_ACT_STATE_FEATURE]
    action = batch[PICKCUBE_ACT_ACTION_FEATURE]
    padding = batch["action_is_pad"]
    if not all(
        isinstance(item, torch.Tensor) for item in (image, state, action, padding)
    ):
        _fail("ACT batch", "features must be Torch tensors")
    image_tensor = cast(torch.Tensor, image)
    if image_tensor.dtype != torch.uint8 or image_tensor.shape[-3:] != (3, 224, 224):
        _fail("ACT batch", "RGB must be uint8[...,3,224,224]")
    result = {
        PICKCUBE_ACT_ACTION_FEATURE: cast(torch.Tensor, action).to(torch.float32),
        PICKCUBE_ACT_IMAGE_FEATURE: image_tensor.to(torch.float32).div(255.0),
        PICKCUBE_ACT_STATE_FEATURE: cast(torch.Tensor, state).to(torch.float32),
        "action_is_pad": cast(torch.Tensor, padding).to(torch.bool),
    }
    for name, tensor in result.items():
        if tensor.dtype != torch.bool and not torch.isfinite(tensor).all():
            _fail("ACT batch", f"{name} contains nonfinite values")
    return result


def save_checkpoint(
    *,
    run_root: Path,
    step: int,
    examples_processed: int,
    training_identity: Mapping[str, object],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    optimizer: Any,
    metric: Mapping[str, object],
) -> Path:
    """Atomically save model, processors, optimizer, RNG, and byte inventory."""
    if step < 1 or examples_processed < 1:
        _fail("checkpoint", "step and example count must be positive")
    root = Path(run_root).absolute()
    final = root / "checkpoints" / f"step-{step:08d}"
    if final.exists() or final.is_symlink():
        _fail("checkpoint", "completed step already exists")
    staging_parent = root / ".checkpoint-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"step-{step:08d}-", dir=staging_parent))
    try:
        pretrained = staging / "pretrained_model"
        pretrained.mkdir()
        policy.save_pretrained(pretrained, push_to_hub=False)
        preprocessor.save_pretrained(
            pretrained,
            push_to_hub=False,
            config_filename="policy_preprocessor.json",
        )
        postprocessor.save_pretrained(
            pretrained,
            push_to_hub=False,
            config_filename="policy_postprocessor.json",
        )
        state_root = staging / "training_state"
        state_root.mkdir()
        torch.save(optimizer.state_dict(), state_root / "optimizer.pt")
        torch.save(
            {
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            state_root / "rng_state.pt",
        )
        write_atomic_json(
            state_root / "training_state.json",
            {
                "examples_processed": examples_processed,
                "global_step": step,
                "metric": dict(metric),
                "training_identity": dict(training_identity),
            },
        )
        artifacts = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "checkpoint_manifest.json":
                digest = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
                artifacts.append(
                    {
                        "path": PurePosixPath(
                            *path.relative_to(staging).parts
                        ).as_posix(),
                        "sha256": digest,
                        "size_bytes": path.stat().st_size,
                    }
                )
        write_atomic_json(
            staging / "checkpoint_manifest.json",
            {
                "artifacts": artifacts,
                "examples_processed": examples_processed,
                "global_step": step,
                "schema_version": "pickcube-native-act-checkpoint-v1",
                "training_identity": dict(training_identity),
            },
        )
        write_atomic_json(staging / "complete.json", {"complete": True, "step": step})
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _validate_checkpoint(
    checkpoint: Path,
    expected_identity: Mapping[str, object],
) -> Mapping[str, object]:
    root = Path(checkpoint).absolute()
    manifest = _read_mapping(root / "checkpoint_manifest.json", context="checkpoint")
    complete = _read_mapping(root / "complete.json", context="checkpoint completion")
    if (
        manifest.get("schema_version") != "pickcube-native-act-checkpoint-v1"
        or complete.get("complete") is not True
        or manifest.get("training_identity") != dict(expected_identity)
    ):
        _fail("checkpoint", "completion or resume identity differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        _fail("checkpoint", "artifact inventory is missing")
    expected_paths: set[str] = set()
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            _fail("checkpoint", "artifact record is malformed")
        relative = raw.get("path")
        digest = raw.get("sha256")
        if not isinstance(relative, str) or relative in expected_paths:
            _fail("checkpoint", "artifact path is invalid or duplicate")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            _fail("checkpoint", f"artifact {relative} is missing")
        observed = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        if observed != digest or path.stat().st_size != raw.get("size_bytes"):
            _fail("checkpoint", f"artifact {relative} changed")
        expected_paths.add(relative)
    actual_paths = {
        PurePosixPath(*path.relative_to(root).parts).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"checkpoint_manifest.json", "complete.json"}
    }
    if actual_paths != expected_paths:
        _fail("checkpoint", "artifact inventory is incomplete")
    return manifest


def load_checkpoint(
    *,
    checkpoint: Path,
    expected_identity: Mapping[str, object],
    experiment: PickCubeActExperimentConfig,
) -> tuple[Any, Any, Any, Any, Mapping[str, object]]:
    """Integrity-check and locally reload a complete ACT training checkpoint."""
    _validate_checkpoint(checkpoint, expected_identity)
    pretrained = Path(checkpoint).absolute() / "pretrained_model"
    try:
        from lerobot.policies import (  # type: ignore[import-not-found]
            make_pre_post_processors,
        )
        from lerobot.policies.act import ACTPolicy

        policy = ACTPolicy.from_pretrained(
            pretrained,
            local_files_only=True,
            strict=True,
        ).to(torch.device(experiment.device))
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(pretrained),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PickCubeActRuntimeError("checkpoint runtime reload failed") from exc
    optimizer = build_optimizer(policy, experiment)
    optimizer_state = torch.load(
        Path(checkpoint) / "training_state" / "optimizer.pt",
        map_location=experiment.device,
        weights_only=False,
    )
    optimizer.load_state_dict(optimizer_state)
    training_state = _read_mapping(
        Path(checkpoint) / "training_state" / "training_state.json",
        context="training state",
    )
    return policy, preprocessor, postprocessor, optimizer, training_state


__all__ = [
    "DeterministicResumeBatchSampler",
    "PickCubeActDataset",
    "PickCubeActRuntimeError",
    "build_optimizer",
    "build_policy_and_processors",
    "installed_runtime_versions",
    "load_checkpoint",
    "prepare_training_batch",
    "processor_statistics",
    "save_checkpoint",
    "seed_act_runtime",
    "split_references",
    "validate_dataset_for_training",
]
