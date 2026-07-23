"""Strict local-only loader for the frozen VLA-JEPA stack."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latentguard.adapters.vla_jepa.constants import CHECKPOINT_REVISION
from latentguard.adapters.vla_jepa.identity import validate_snapshot_manifest

_TIED_EMBEDDING_KEY = "model.qwen.model.model.language_model.embed_tokens.weight"
_TIED_LM_HEAD_KEY = "model.qwen.model.lm_head.weight"


class ModelLoadError(RuntimeError):
    """Raised when exact local VLA-JEPA loading cannot be proven."""


@dataclass(frozen=True)
class LoadedVLAJepa:
    """Loaded policy plus the exact local execution bindings."""

    policy: Any
    config: Any
    checkpoint_path: Path
    qwen_path: Path
    vjepa_path: Path
    missing_keys: tuple[str, ...]
    unexpected_noncritical_keys: tuple[str, ...]
    checkpoint_revision: str = CHECKPOINT_REVISION


def _validate_tied_checkpoint_weights(model_file: Path) -> None:
    """Verify the sole allowed duplicate checkpoint key is byte-value equal."""

    safetensors_module = importlib.import_module("safetensors")
    torch = importlib.import_module("torch")
    with safetensors_module.safe_open(
        model_file,
        framework="pt",
        device="cpu",
    ) as handle:
        keys = set(handle.keys())
        required = {_TIED_EMBEDDING_KEY, _TIED_LM_HEAD_KEY}
        if not required.issubset(keys):
            raise ModelLoadError(
                "checkpoint does not contain both tied Qwen embedding tensors"
            )
        embedding = handle.get_slice(_TIED_EMBEDDING_KEY)
        lm_head = handle.get_slice(_TIED_LM_HEAD_KEY)
        if embedding.get_shape() != lm_head.get_shape():
            raise ModelLoadError("tied Qwen embedding tensor shapes differ")
        row_count = int(embedding.get_shape()[0])
        for start in range(0, row_count, 1024):
            stop = min(start + 1024, row_count)
            if not torch.equal(embedding[start:stop], lm_head[start:stop]):
                raise ModelLoadError("tied Qwen embedding checkpoint tensors differ")


def load_local_vla_jepa(
    checkpoint_path: Path,
    qwen_path: Path,
    vjepa_path: Path,
    *,
    asset_manifest: dict[str, Any],
    device: str = "cuda",
) -> LoadedVLAJepa:
    """Load exact local snapshots, rejecting partial or non-strict state loading."""

    for label, path in (
        ("checkpoint", checkpoint_path),
        ("qwen", qwen_path),
        ("vjepa", vjepa_path),
    ):
        if not path.is_dir():
            raise ModelLoadError(f"{label} snapshot is missing: {path}")
    validate_snapshot_manifest(
        asset_manifest,
        {
            "lerobot/VLA-JEPA-LIBERO": checkpoint_path,
            "Qwen/Qwen3-VL-2B-Instruct": qwen_path,
            "facebook/vjepa2-vitl-fpc64-256": vjepa_path,
        },
    )
    importlib.import_module("lerobot.policies.vla_jepa.configuration_vla_jepa")
    configs_module = importlib.import_module("lerobot.configs.policies")
    policy_module = importlib.import_module(
        "lerobot.policies.vla_jepa.modeling_vla_jepa"
    )
    config = configs_module.PreTrainedConfig.from_pretrained(
        checkpoint_path,
        local_files_only=True,
    )
    if config.type != "vla_jepa":
        raise ModelLoadError(
            f"expected vla_jepa checkpoint config, observed {config.type!r}"
        )
    config.device = device
    config.qwen_model_name = str(qwen_path)
    config.jepa_encoder_name = str(vjepa_path)
    model_file = checkpoint_path / "model.safetensors"
    _validate_tied_checkpoint_weights(model_file)
    try:
        policy = policy_module.VLAJEPAPolicy(config)
        safetensors_torch = importlib.import_module("safetensors.torch")
        missing_keys, unexpected_keys = safetensors_torch.load_model(
            policy,
            str(model_file),
            strict=False,
            device=config.device,
        )
    except Exception as exc:
        raise ModelLoadError(f"fail-closed VLA-JEPA load failed: {exc}") from exc
    missing = tuple(sorted(missing_keys))
    unexpected = tuple(sorted(unexpected_keys))
    if missing:
        raise ModelLoadError(f"checkpoint has missing keys: {missing}")
    if unexpected != (_TIED_EMBEDDING_KEY,):
        raise ModelLoadError(
            "checkpoint has unexpected keys outside the reviewed tied-weight "
            f"exception: {unexpected}"
        )
    embedding_weight = policy.model.qwen.model.model.language_model.embed_tokens.weight
    lm_head_weight = policy.model.qwen.model.lm_head.weight
    if embedding_weight.data_ptr() != lm_head_weight.data_ptr():
        raise ModelLoadError("runtime Qwen embedding and LM head are not tied")
    policy.to(config.device)
    policy.eval()
    return LoadedVLAJepa(
        policy=policy,
        config=config,
        checkpoint_path=checkpoint_path.resolve(),
        qwen_path=qwen_path.resolve(),
        vjepa_path=vjepa_path.resolve(),
        missing_keys=missing,
        unexpected_noncritical_keys=unexpected,
    )
