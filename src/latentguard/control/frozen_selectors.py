"""Frozen nonvisual and visual selectors used by the M4C control matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from latentguard.control.models import (
    ClosedLoopCandidatePoolV1,
    SelectorOutputV1,
    content_digest,
)
from latentguard.control.runner import RuntimeBoundaryV1
from latentguard.control.selectors import (
    build_fixed_visual_batch,
    consume_fixed_visual_features,
    deterministic_random_output,
    fixed_primary_output,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PickCubeVerifierStateSchemaV1,
    PickCubeVerifierStateV1,
)
from latentguard.selection.checkpoint_bundle import LoadedVerifierBundleV1
from latentguard.selection.inference import (
    infer_prepared_seed_logits_once,
    normalize_candidate_inputs,
)
from latentguard.training.calibration import TemperatureCalibrationStateV1
from latentguard.visual_training.backbone import FrozenBackboneRuntime
from latentguard.visual_training.checkpoint import (
    checkpoint_content_digest,
    inspect_visual_checkpoint,
    load_visual_checkpoint,
)
from latentguard.visual_training.config import VisualModelConfigV1
from latentguard.visual_training.external import load_model_freeze_manifest
from latentguard.visual_training.models import (
    VisualActionVerifier,
    build_visual_action_verifier,
)
from latentguard.visual_training.training import (
    ActionNormalizationV1,
    action_normalization_from_mapping,
)


@dataclass(frozen=True, slots=True)
class FixedPrimarySelector:
    """Frozen non-learned primary-candidate baseline."""

    primary_definition_ordinal: int = 0
    selector_id: str = "fixed_primary_v1"
    visual: bool = False

    def select(
        self, pool: ClosedLoopCandidatePoolV1, boundary: RuntimeBoundaryV1
    ) -> SelectorOutputV1:
        """Return the fixed primary ranking without reading observations."""

        del boundary
        return fixed_primary_output(
            pool, primary_definition_ordinal=self.primary_definition_ordinal
        )


@dataclass(frozen=True, slots=True)
class DeterministicRandomSelector:
    """Frozen content-derived random baseline."""

    selector_id: str = "deterministic_random_v1"
    visual: bool = False

    def select(
        self, pool: ClosedLoopCandidatePoolV1, boundary: RuntimeBoundaryV1
    ) -> SelectorOutputV1:
        """Return the stable pool-bound pseudo-random ranking."""

        del boundary
        return deterministic_random_output(pool)


@dataclass(slots=True)
class FrozenStructuredEnsembleSelector:
    """One accepted M3B five-seed calibrated ensemble."""

    selector_id: str
    bundle: LoadedVerifierBundleV1
    verifier_schema: PickCubeVerifierStateSchemaV1
    visual: bool = False

    def select(
        self, pool: ClosedLoopCandidatePoolV1, boundary: RuntimeBoundaryV1
    ) -> SelectorOutputV1:
        """Score all eight candidates once and rank minimum failure probability."""

        candidate_actions = np.ascontiguousarray(
            np.stack([item.action_chunk for item in pool.candidates]),
            dtype=np.float64,
        )
        masks = np.ascontiguousarray(
            np.stack([item.action_mask for item in pool.candidates]), dtype=np.bool_
        )
        state = PickCubeVerifierStateV1(
            schema=self.verifier_schema, values=boundary.verifier_state
        )
        batch = normalize_candidate_inputs(
            state,
            candidate_actions,
            masks,
            pool.ordered_candidate_ids,
            self.bundle.preprocessing,
        )
        device = self.bundle.loading_diagnostics.device
        per_seed = []
        for seed in self.bundle.seeds:
            model = cast(Any, seed.model)
            model.eval()
            logits = infer_prepared_seed_logits_once(model, batch, device=device)
            per_seed.append(seed.calibration.apply(logits))
        probabilities = np.mean(np.stack(per_seed), axis=0, dtype=np.float64)
        scores = dict(
            zip(pool.ordered_candidate_ids, probabilities.tolist(), strict=True)
        )
        ranking = tuple(sorted(scores, key=lambda item: (scores[item], item)))
        return SelectorOutputV1(
            selector_id=self.selector_id,
            scores=scores,
            ranking=ranking,
            checkpoint_ensemble_identity=self.bundle.identity.content_digest,
            probabilities=True,
        )


@dataclass(slots=True)
class FrozenVisualEnsembleSelector:
    """One accepted M4B three-seed visual verifier with fixed batch-128 backbone."""

    selector_id: str
    backbone: FrozenBackboneRuntime
    models: tuple[VisualActionVerifier, ...]
    calibrations: tuple[TemperatureCalibrationStateV1, ...]
    action_normalization: ActionNormalizationV1
    checkpoint_content_digests: tuple[str, ...]
    device: str = "cuda"
    visual: bool = True

    def __post_init__(self) -> None:
        """Require exactly three aligned frozen seeds."""

        if not (
            len(self.models)
            == len(self.calibrations)
            == len(self.checkpoint_content_digests)
            == 3
        ):
            raise ValueError("M4C visual selector requires exactly three seeds")

    @property
    def ensemble_identity(self) -> str:
        """Return the fixed checkpoint/calibration/backbone identity."""

        return content_digest(
            {
                "backbone_manifest_digest": self.backbone.manifest.content_digest,
                "calibration_digests": [
                    item.content_digest for item in self.calibrations
                ],
                "checkpoint_content_digests": list(self.checkpoint_content_digests),
                "schema_version": "1.0",
                "selector_id": self.selector_id,
            },
            context="M4CVisualEnsembleIdentityV1",
        )

    def select(
        self, pool: ClosedLoopCandidatePoolV1, boundary: RuntimeBoundaryV1
    ) -> SelectorOutputV1:
        """Extract one exact batch of 128, consume slots 0..2, and score once."""

        if boundary.images is None:
            raise ValueError("M4C visual selector requires three current views")
        fixed_batch = build_fixed_visual_batch(boundary.images)
        extracted = self.backbone.extract(fixed_batch)
        three_features = consume_fixed_visual_features(extracted)
        visual_features = np.repeat(three_features[None, :, :], 8, axis=0)
        actions = np.ascontiguousarray(
            np.stack([item.action_chunk for item in pool.candidates]),
            dtype=np.float64,
        )
        masks = np.ascontiguousarray(
            np.stack([item.action_mask for item in pool.candidates]), dtype=np.bool_
        )
        normalized_actions = self.action_normalization.apply(actions, masks)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional remote dependency
            raise RuntimeError("M4C visual inference requires PyTorch") from exc
        target = torch.device(self.device)
        feature_tensor = torch.as_tensor(
            visual_features, dtype=torch.float32, device=target
        )
        action_tensor = torch.as_tensor(
            normalized_actions, dtype=torch.float32, device=target
        )
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=target)
        per_seed = []
        with torch.inference_mode():
            for model, calibration in zip(self.models, self.calibrations, strict=True):
                model.to(target)
                model.eval()
                logits = model(feature_tensor, action_tensor, mask_tensor)
                raw = np.asarray(logits.detach().cpu().numpy(), dtype=np.float64)
                per_seed.append(calibration.apply(raw))
        probabilities = np.mean(np.stack(per_seed), axis=0, dtype=np.float64)
        scores = dict(
            zip(pool.ordered_candidate_ids, probabilities.tolist(), strict=True)
        )
        ranking = tuple(sorted(scores, key=lambda item: (scores[item], item)))
        return SelectorOutputV1(
            selector_id=self.selector_id,
            scores=scores,
            ranking=ranking,
            checkpoint_ensemble_identity=self.ensemble_identity,
            probabilities=True,
        )


def load_frozen_visual_selector(
    *,
    selector_id: str,
    backbone: FrozenBackboneRuntime,
    model_config: VisualModelConfigV1,
    model_freeze_manifest: Path,
    checkpoints: tuple[Path, Path, Path],
    device: str,
) -> FrozenVisualEnsembleSelector:
    """Digest-verify and load one promoted M4B three-seed visual ensemble."""

    freeze = load_model_freeze_manifest(model_freeze_manifest)
    if freeze.get("model_config_digest") != model_config.content_digest:
        raise ValueError("M4C visual freeze model configuration differs")
    if freeze.get("seed_order") != [0, 1, 2]:
        raise ValueError("M4C visual freeze seed inventory differs")
    raw_calibrations = freeze.get("calibration")
    raw_checkpoint_digests = freeze.get("checkpoint_digests")
    if not isinstance(raw_calibrations, list) or len(raw_calibrations) != 3:
        raise ValueError("M4C visual freeze requires three calibrations")
    if (
        not isinstance(raw_checkpoint_digests, list)
        or len(raw_checkpoint_digests) != 3
        or any(not isinstance(item, str) for item in raw_checkpoint_digests)
    ):
        raise ValueError("M4C visual freeze checkpoint inventory differs")
    calibrations = tuple(
        TemperatureCalibrationStateV1.from_dict(cast(dict[str, object], item))
        for item in raw_calibrations
    )
    models: list[VisualActionVerifier] = []
    normalization: ActionNormalizationV1 | None = None
    observed_digests: list[str] = []
    for seed, checkpoint in enumerate(checkpoints):
        observed_digest = checkpoint_content_digest(checkpoint)
        if observed_digest != raw_checkpoint_digests[seed]:
            raise ValueError("M4C visual checkpoint bytes differ from freeze")
        metadata = inspect_visual_checkpoint(checkpoint)
        if (
            metadata.progress.binding.seed != seed
            or metadata.progress.binding.model_config_digest
            != model_config.content_digest
            or metadata.progress.binding.backbone_digest
            != backbone.manifest.content_digest
        ):
            raise ValueError("M4C visual checkpoint semantic binding differs")
        seed_normalization = action_normalization_from_mapping(
            metadata.action_preprocessing
        )
        if normalization is None:
            normalization = seed_normalization
        elif seed_normalization.content_digest != normalization.content_digest:
            raise ValueError("M4C visual action preprocessing differs across seeds")
        model = build_visual_action_verifier(model_config, seed=seed)
        load_visual_checkpoint(
            checkpoint,
            expected_binding=metadata.progress.binding,
            model=model,
            restore_rng=False,
            model_only=True,
        )
        models.append(model)
        observed_digests.append(observed_digest)
    assert normalization is not None
    return FrozenVisualEnsembleSelector(
        selector_id=selector_id,
        backbone=backbone,
        models=tuple(models),
        calibrations=calibrations,
        action_normalization=normalization,
        checkpoint_content_digests=tuple(observed_digests),
        device=device,
    )


__all__ = [
    "DeterministicRandomSelector",
    "FixedPrimarySelector",
    "FrozenStructuredEnsembleSelector",
    "FrozenVisualEnsembleSelector",
    "load_frozen_visual_selector",
]
