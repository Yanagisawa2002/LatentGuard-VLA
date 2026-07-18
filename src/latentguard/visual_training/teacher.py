"""One-time privileged M3B ensemble target extraction for M4B distillation."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

from latentguard.selection.checkpoint_bundle import (
    LoadedVerifierBundleV1,
    VerifierBundleV1,
    VerifierRuntimeArtifactsV1,
    VerifierSeedArtifactPathsV1,
    load_verifier_bundle,
)
from latentguard.training.dataset import AcceptedActionVerifierDatasetV1
from latentguard.training.reporting import load_strict_report
from latentguard.visual_training.cache import LoadedTeacherCacheV1, save_teacher_cache


class VisualTeacherError(ValueError):
    """Raised when the accepted five-seed teacher binding or output differs."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualTeacherError(f"{context}: {reason}")


def load_accepted_teacher_bundle(
    report_path: Path,
    *,
    preprocessing: Path,
    checkpoints: tuple[Path, Path, Path, Path, Path],
    calibrations: tuple[Path, Path, Path, Path, Path],
    thresholds: tuple[Path, Path, Path, Path, Path],
    device: str,
) -> LoadedVerifierBundleV1:
    """Strictly validate the committed temporal bundle and five runtime seeds."""
    report = load_strict_report(
        report_path, expected_report_type="m3c_verifier_bundle_v1"
    )
    bundle = VerifierBundleV1.from_dict(report.payload)
    if bundle.architecture != "temporal_state_action_verifier":
        _fail("privileged teacher", "expected accepted temporal architecture")
    artifacts = VerifierRuntimeArtifactsV1(
        preprocessing=preprocessing,
        seeds={
            seed: VerifierSeedArtifactPathsV1(
                checkpoint=checkpoints[seed],
                calibration=calibrations[seed],
                thresholds=thresholds[seed],
            )
            for seed in range(5)
        },
    )
    return load_verifier_bundle(bundle, artifacts, device=device)


def prepare_teacher_targets(
    bundle: LoadedVerifierBundleV1,
    dataset: AcceptedActionVerifierDatasetV1,
    output_dir: Path,
    *,
    git_sha: str,
    device: str,
    batch_size: int = 256,
    limit_samples: int | None = None,
    resume: bool = False,
) -> tuple[LoadedTeacherCacheV1, int]:
    """Infer five logits/calibrated probabilities once without persisting inputs."""
    if (
        bundle.identity.dataset_digest != dataset.dataset_digest
        or bundle.identity.split_digest != dataset.split_digest
    ):
        _fail("privileged teacher", "accepted dataset or split differs")
    count = len(dataset) if limit_samples is None else min(len(dataset), limit_samples)
    if count <= 0 or type(batch_size) is not int or batch_size <= 0:
        _fail("privileged teacher", "sample and batch counts must be positive")
    candidate_ids = tuple(item.sample_id for item in dataset.reporting[:count])
    if Path(output_dir).exists() and resume:
        from latentguard.visual_training.cache import load_teacher_cache

        loaded = load_teacher_cache(output_dir)
        if (
            loaded.git_sha != git_sha
            or loaded.dataset_digest != dataset.dataset_digest
            or loaded.split_digest != dataset.split_digest
            or loaded.teacher_bundle_digest != bundle.identity.content_digest
            or loaded.candidate_ids != candidate_ids
        ):
            _fail("privileged teacher resume", "completed cache identity differs")
        return loaded, 0
    raw_logits = np.empty((5, count), dtype=np.float64)
    target = torch.device(device)
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        examples = dataset.examples[start:end]
        states = np.stack([item.state_vector for item in examples]).astype(np.float64)
        actions = np.stack([item.action_chunk for item in examples]).astype(np.float64)
        masks = np.stack([item.action_mask for item in examples]).astype(np.bool_)
        preprocessing = bundle.preprocessing
        normalized_states = (
            (states - preprocessing.state_mean) / preprocessing.state_standard_deviation
        ).astype(np.float32)
        normalized_actions = (
            actions - preprocessing.action_mean
        ) / preprocessing.action_standard_deviation
        normalized_actions[~masks] = 0.0
        state_t = torch.tensor(normalized_states, device=target)
        action_t = torch.tensor(normalized_actions.astype(np.float32), device=target)
        mask_t = torch.tensor(masks, device=target)
        for seed_index, seed in enumerate(bundle.seeds):
            model = seed.model
            if not isinstance(model, torch.nn.Module):
                _fail("privileged teacher", "runtime teacher is not a torch model")
            model.eval()
            with torch.inference_mode():
                output = model(state_t, action_t, mask_t)
            if tuple(output.shape) != (end - start,) or not bool(
                torch.isfinite(output).all()
            ):
                _fail("privileged teacher", "model output contract differs")
            raw_logits[seed_index, start:end] = output.detach().cpu().numpy()
    calibrated = np.stack(
        [
            seed.calibration.apply(raw_logits[index])
            for index, seed in enumerate(bundle.seeds)
        ]
    ).astype(np.float64)
    return save_teacher_cache(
        output_dir,
        git_sha=git_sha,
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        teacher_bundle_digest=bundle.identity.content_digest,
        candidate_ids=candidate_ids,
        raw_logits=raw_logits,
        calibrated_probabilities=calibrated,
        resume=resume,
    )


__all__ = [
    "VisualTeacherError",
    "load_accepted_teacher_bundle",
    "prepare_teacher_targets",
]
