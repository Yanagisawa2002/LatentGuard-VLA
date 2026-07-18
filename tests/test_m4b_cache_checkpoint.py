"""Integrity and exact-resume tests for M4B caches and checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from latentguard.visual_training.cache import (
    VisualCacheError,
    extract_feature_cache,
    load_feature_cache,
    load_teacher_cache,
    save_teacher_cache,
)
from latentguard.visual_training.checkpoint import (
    VisualCheckpointBindingV1,
    VisualCheckpointError,
    inspect_visual_checkpoint,
    load_visual_checkpoint,
    save_visual_checkpoint,
)
from latentguard.visual_training.config import (
    load_visual_model_config,
    load_visual_training_config,
)
from latentguard.visual_training.models import build_visual_action_verifier

_ROOT = Path("configs/training/m4b")


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def test_teacher_cache_round_trip_resume_and_tamper(tmp_path: Path) -> None:
    """Teacher logits are computed once and every durable byte is verified."""
    output = tmp_path / "teacher"
    raw = np.arange(15, dtype=np.float64).reshape(5, 3)
    calibrated = np.linspace(0.1, 0.9, 15, dtype=np.float64).reshape(5, 3)
    kwargs = {
        "git_sha": "a" * 40,
        "dataset_digest": _digest("dataset"),
        "split_digest": _digest("split"),
        "teacher_bundle_digest": _digest("teacher"),
        "candidate_ids": ("a", "b", "c"),
        "raw_logits": raw,
        "calibrated_probabilities": calibrated,
    }
    loaded, computed = save_teacher_cache(output, **kwargs)
    resumed, recomputed = save_teacher_cache(output, resume=True, **kwargs)

    assert computed == 3
    assert recomputed == 0
    assert resumed.content_digest == loaded.content_digest
    assert loaded.training_probability(0, is_training=True) == pytest.approx(
        float(np.mean(calibrated[:, 0]))
    )
    with pytest.raises(VisualCacheError, match="training-only"):
        loaded.training_probability(0, is_training=False)

    path = output / "raw_logits.npy"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(VisualCacheError, match="bytes changed"):
        load_teacher_cache(output)


def test_feature_cache_deduplicated_round_trip_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One packet/view row is extracted once, reloaded exactly, and then reused."""
    packets = tuple(
        SimpleNamespace(
            packet_id=f"packet-{packet}",
            views=tuple(
                SimpleNamespace(
                    pixel_sha256=_digest(f"image-{packet}-{slot}"),
                    image_reference=f"image-{packet}-{slot}.npy",
                )
                for slot in range(3)
            ),
        )
        for packet in range(2)
    )
    dataset = SimpleNamespace(
        content_digest=_digest("visual"),
        source_dataset_digest=_digest("structured"),
        split_digest=_digest("split"),
        packets=packets,
    )
    manifest = SimpleNamespace(
        content_digest=_digest("backbone"),
        preprocessing_semantic="preprocess-v1",
        weight_content_digest=_digest("weight"),
    )

    class _Runtime:
        def __init__(self) -> None:
            self.manifest = manifest

        def extract(self, images: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            values = np.mean(images, axis=(1, 2, 3), dtype=np.float64).astype(
                np.float32
            )
            return np.repeat(values[:, None], 512, axis=1)

    monkeypatch.setattr(
        "latentguard.visual_training.cache.load_visual_image",
        lambda root, reference: np.full((224, 224, 3), len(str(reference)), np.uint8),
    )
    output = tmp_path / "features"
    loaded, extracted = extract_feature_cache(
        dataset,
        tmp_path,
        _Runtime(),  # type: ignore[arg-type]
        output,
        git_sha="a" * 40,
        batch_size=4,
    )
    resumed, duplicate = extract_feature_cache(
        dataset,
        tmp_path,
        _Runtime(),  # type: ignore[arg-type]
        output,
        git_sha="a" * 40,
        batch_size=4,
        resume=True,
    )

    assert extracted == 6
    assert duplicate == 0
    assert len(loaded.entries) == 6
    assert resumed.content_digest == loaded.content_digest
    assert np.array_equal(
        loaded.feature("packet-0", 0), load_feature_cache(output).features[0]
    )


def test_checkpoint_round_trip_and_binding_drift(tmp_path: Path) -> None:
    """Model/optimizer state applies only under the exact semantic binding."""
    model_config = load_visual_model_config(
        _ROOT / "frozen-resnet18-multiview-action.json"
    )
    training_config = load_visual_training_config(_ROOT / "train-default.json")
    model = build_visual_action_verifier(model_config, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    binding = VisualCheckpointBindingV1(
        git_sha="a" * 40,
        visual_dataset_digest=_digest("visual"),
        structured_dataset_digest=_digest("structured"),
        split_digest=_digest("split"),
        backbone_digest=_digest("backbone"),
        feature_cache_digest=_digest("features"),
        teacher_cache_digest=None,
        action_preprocessing_digest=_digest("actions"),
        model_config_digest=model_config.content_digest,
        training_config_digest=training_config.content_digest,
        seed=0,
    )
    path = tmp_path / "checkpoint.pt"
    save_visual_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        binding=binding,
        model_config=model_config.as_mapping(),
        training_config=training_config.as_mapping(),
        action_preprocessing={"content_digest": _digest("actions")},
        epoch=1,
        global_step=7,
        best_validation_auprc=0.6,
        best_epoch=1,
        patience_count=0,
        complete=True,
    )
    restored = build_visual_action_verifier(model_config, seed=1)
    progress = load_visual_checkpoint(
        path,
        expected_binding=binding,
        model=restored,
        optimizer=None,
        restore_rng=False,
        model_only=True,
    )

    assert progress.complete
    assert progress.global_step == 7
    assert inspect_visual_checkpoint(path).progress.binding == binding
    assert all(
        torch.equal(first, second)
        for first, second in zip(
            model.state_dict().values(), restored.state_dict().values(), strict=True
        )
    )

    drifted = VisualCheckpointBindingV1(**{**binding.as_mapping(), "git_sha": "b" * 40})
    with pytest.raises(VisualCheckpointError, match="identity drifted"):
        load_visual_checkpoint(
            path,
            expected_binding=drifted,
            model=restored,
            restore_rng=False,
            model_only=True,
        )


def test_model_only_checkpoint_load_ignores_bound_training_runtime_state(
    tmp_path: Path,
) -> None:
    """Inference restores exact model weights without requiring a scheduler."""
    model_config = load_visual_model_config(
        Path("configs/training/m4b/frozen-resnet18-multiview-action.json")
    )
    training_config = load_visual_training_config(
        Path("configs/training/m4b/train-default.json")
    )
    model = build_visual_action_verifier(model_config, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    binding = VisualCheckpointBindingV1(
        git_sha="a" * 40,
        visual_dataset_digest=_digest("visual"),
        structured_dataset_digest=_digest("structured"),
        split_digest=_digest("split"),
        backbone_digest=_digest("backbone"),
        feature_cache_digest=_digest("features"),
        teacher_cache_digest=None,
        action_preprocessing_digest=_digest("actions"),
        model_config_digest=model_config.content_digest,
        training_config_digest=training_config.content_digest,
        seed=0,
    )
    path = tmp_path / "checkpoint-with-scheduler.pt"
    save_visual_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        binding=binding,
        model_config=model_config.as_mapping(),
        training_config=training_config.as_mapping(),
        action_preprocessing={"content_digest": _digest("actions")},
        epoch=1,
        global_step=3,
        best_validation_auprc=0.7,
        best_epoch=1,
        patience_count=0,
        complete=True,
    )
    restored = build_visual_action_verifier(model_config, seed=1)

    progress = load_visual_checkpoint(
        path,
        expected_binding=binding,
        model=restored,
        restore_rng=False,
        model_only=True,
    )

    assert progress.complete
    assert all(
        torch.equal(first, second)
        for first, second in zip(
            model.state_dict().values(), restored.state_dict().values(), strict=True
        )
    )
