"""Outcome-free M3C visual selection and existing-outcome-only evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.calibration import TemperatureCalibrationStateV1
from latentguard.vision_data.models import VisualVerifierExternalDatasetV1
from latentguard.visual_training.cache import LoadedFeatureCacheV1
from latentguard.visual_training.checkpoint import (
    checkpoint_content_digest,
    inspect_visual_checkpoint,
    load_visual_checkpoint,
)
from latentguard.visual_training.config import VisualModelConfigV1
from latentguard.visual_training.data import (
    EXTERNAL_VISUAL_DATASET_DIGEST,
    TEST_DOMAIN_IDS,
)
from latentguard.visual_training.models import build_visual_action_verifier
from latentguard.visual_training.training import action_normalization_from_mapping


class ExternalVisualEvaluationError(ValueError):
    """Raised when blind selection or the immutable outcome join differs."""


def _fail(context: str, reason: str) -> NoReturn:
    raise ExternalVisualEvaluationError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    payload = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _load_self_digesting_json(path: Path, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalVisualEvaluationError(f"{context}: {exc}") from exc
    if not isinstance(value, dict) or "content_digest" not in value:
        _fail(context, "expected self-digesting object")
    body = {key: item for key, item in value.items() if key != "content_digest"}
    expected = _digest(
        body,
        "M4BInternalEvaluationV1"
        if "internal_test_opened_once" in value
        else "M4BExternalSelectionManifestV1",
    )
    if value["content_digest"] != expected:
        _fail(context, "content digest differs")
    return cast(Mapping[str, object], value)


def load_model_freeze_manifest(path: Path) -> Mapping[str, object]:
    """Load the post-internal-test model freeze required before M3C images."""
    value = _load_self_digesting_json(path, "model freeze manifest")
    if (
        value.get("internal_test_opened_once") is not True
        or value.get("external_dataset_opened") is not False
        or value.get("seed_order") != [0, 1, 2]
    ):
        _fail("model freeze manifest", "freeze or evaluation ordering differs")
    return value


def _checkpoint_matches_external_freeze(
    metadata: object,
    *,
    seed: int,
    model_config_digest: str,
    observed_checkpoint_digest: str,
    expected_checkpoint_digest: object,
    freeze_git_sha: object,
    feature_backbone_digest: str,
) -> bool:
    """Bind a trained checkpoint to its freeze, not a later cache execution."""
    binding = cast(Any, metadata).progress.binding
    return bool(
        binding.seed == seed
        and binding.model_config_digest == model_config_digest
        and observed_checkpoint_digest == expected_checkpoint_digest
        and cast(Any, metadata).progress.complete
        and isinstance(freeze_git_sha, str)
        and binding.git_sha == freeze_git_sha
        and binding.backbone_digest == feature_backbone_digest
    )


def select_external_visual_candidates(
    *,
    visual_dataset: VisualVerifierExternalDatasetV1,
    candidate_pool: object,
    feature_cache: LoadedFeatureCacheV1,
    model_config: VisualModelConfigV1,
    checkpoints: tuple[Path, Path, Path],
    model_freeze_manifest: Path,
    external_config_digest: str,
    selection_semantic: str,
    domain_id: str,
    output_path: Path,
    device: str,
) -> Mapping[str, object]:
    """Score M3C candidates without accepting or loading any outcome path."""
    if (
        visual_dataset.content_digest != EXTERNAL_VISUAL_DATASET_DIGEST
        or visual_dataset.training_allowed
    ):
        _fail(
            "external visual selection",
            "external dataset identity or permission differs",
        )
    if domain_id not in TEST_DOMAIN_IDS:
        _fail("external visual selection", "unsupported external domain")
    if feature_cache.dataset_digest != visual_dataset.content_digest:
        _fail("external visual selection", "feature cache dataset binding differs")
    if (
        feature_cache.source_dataset_digest != visual_dataset.candidate_pool_identity
        or feature_cache.split_digest != visual_dataset.source_set_identity
    ):
        _fail("external visual selection", "feature cache source binding differs")
    if not model_config.backbone_frozen:
        _fail(
            "external visual selection",
            "promoted external model must use frozen features",
        )
    freeze = load_model_freeze_manifest(model_freeze_manifest)
    if feature_cache.model_freeze_digest != freeze["content_digest"]:
        _fail("external visual selection", "feature cache freeze binding differs")
    if freeze.get("model_config_digest") != model_config.content_digest or freeze.get(
        "seed_order"
    ) != [0, 1, 2]:
        _fail("external visual selection", "frozen model or seed identity differs")
    calibration_rows = freeze.get("calibration")
    if not isinstance(calibration_rows, list) or len(calibration_rows) != 3:
        _fail("external visual selection", "three calibrations are required")
    calibrations = tuple(
        TemperatureCalibrationStateV1.from_dict(cast(Mapping[str, object], item))
        for item in calibration_rows
    )
    target = torch.device(device)
    models = []
    normalization = None
    checkpoint_digests = freeze.get("checkpoint_digests")
    if not isinstance(checkpoint_digests, list) or len(checkpoint_digests) != 3:
        _fail("external visual selection", "checkpoint freeze inventory differs")
    for seed, path in enumerate(checkpoints):
        metadata = inspect_visual_checkpoint(path)
        if not _checkpoint_matches_external_freeze(
            metadata,
            seed=seed,
            model_config_digest=model_config.content_digest,
            observed_checkpoint_digest=checkpoint_content_digest(path),
            expected_checkpoint_digest=checkpoint_digests[seed],
            freeze_git_sha=freeze.get("git_sha"),
            feature_backbone_digest=feature_cache.backbone_digest,
        ):
            _fail("external visual selection", "checkpoint binding differs")
        if normalization is None:
            normalization = action_normalization_from_mapping(
                metadata.action_preprocessing
            )
        elif (
            action_normalization_from_mapping(
                metadata.action_preprocessing
            ).content_digest
            != normalization.content_digest
        ):
            _fail(
                "external visual selection", "action preprocessing differs across seeds"
            )
        model = build_visual_action_verifier(model_config, seed=seed)
        load_visual_checkpoint(
            path,
            expected_binding=metadata.progress.binding,
            model=model,
            restore_rng=False,
            model_only=True,
        )
        model.to(target)
        model.eval()
        models.append(model)
    assert normalization is not None
    bindings = {
        item.candidate_sample_id: item for item in visual_dataset.candidate_bindings
    }
    packets = {item.packet_id: item for item in visual_dataset.packets}
    groups = tuple(cast(Any, candidate_pool).groups)
    proposal_ids = tuple(
        candidate.proposal_id for group in groups for candidate in group.candidates
    )
    if set(bindings) != set(proposal_ids):
        _fail("external visual selection", "candidate binding coverage differs")
    decisions: list[dict[str, object]] = []
    for group in groups:
        ids = tuple(item.proposal_id for item in group.candidates)
        packet_ids: list[str] = []
        for proposal_id in ids:
            binding = bindings[proposal_id]
            matches = [
                packet_id
                for packet_id in binding.packet_ids
                if packets[packet_id].render_domain_id == domain_id
            ]
            if len(matches) != 1:
                _fail("external visual selection", "domain packet coverage differs")
            packet_ids.append(matches[0])
        features = np.stack(
            [
                np.stack([feature_cache.feature(packet_id, slot) for slot in range(3)])
                for packet_id in packet_ids
            ]
        ).astype(np.float32)
        masks = np.stack([item.action_mask for item in group.candidates]).astype(
            np.bool_
        )
        actions = normalization.apply(
            np.stack([item.action_chunk for item in group.candidates]), masks
        )
        probabilities: list[NDArray[Any]] = []
        for seed, model in enumerate(models):
            with torch.inference_mode():
                logits = model(
                    torch.tensor(features, device=target),
                    torch.tensor(actions, device=target),
                    torch.tensor(masks, device=target),
                )
            probabilities.append(
                calibrations[seed].apply(logits.detach().cpu().numpy())
            )
        ensemble = np.mean(np.stack(probabilities), axis=0, dtype=np.float64)
        ranking = tuple(
            candidate_id
            for _, candidate_id in sorted(
                zip(ensemble.tolist(), ids, strict=True),
                key=lambda item: (item[0], item[1]),
            )
        )
        decisions.append(
            {
                "group_id": group.group_id,
                "predicted_failure_probabilities": {
                    candidate_id: float(ensemble[ids.index(candidate_id)])
                    for candidate_id in ids
                },
                "ranking": list(ranking),
                "selected_proposal_id": ranking[0],
            }
        )
    body: dict[str, object] = {
        "candidate_pool_digest": cast(Any, candidate_pool).content_digest,
        "decisions": decisions,
        "domain_id": domain_id,
        "external_visual_dataset_digest": visual_dataset.content_digest,
        "external_evaluation_config_digest": external_config_digest,
        "feature_cache_digest": feature_cache.content_digest,
        "model_freeze_digest": freeze["content_digest"],
        "outcome_input_paths": [],
        "outcomes_available_during_selection": False,
        "schema_version": "1.0",
        "seed_order": [0, 1, 2],
        "selection_semantic": selection_semantic,
    }
    manifest = {
        **body,
        "content_digest": _digest(body, "M4BExternalSelectionManifestV1"),
    }
    destination = Path(output_path)
    if destination.exists():
        _fail("external visual selection", "completed output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _selector_successes(
    decisions: Sequence[object], outcome_by_id: Mapping[str, object]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    by_selector: dict[str, dict[str, float]] = {}
    for decision in decisions:
        selector_id = cast(Any, decision).selector_id
        group_id = cast(Any, decision).group_id
        selected = cast(Any, decision).selected_proposal_id
        if selected is None:
            continue
        successes = by_selector.setdefault(selector_id, {})
        if group_id in successes:
            _fail("external visual evaluation", "duplicate selector/group decision")
        successes[group_id] = float(cast(Any, outcome_by_id[selected]).success)
    return (
        {
            key: float(np.mean(tuple(value.values())))
            for key, value in by_selector.items()
        },
        by_selector,
    )


def _paired_bootstrap(
    first: NDArray[Any],
    second: NDArray[Any],
    trajectory_ids: tuple[str, ...],
    *,
    seed: int = 271828,
    samples: int = 2000,
) -> dict[str, object]:
    unique = tuple(sorted(set(trajectory_ids)))
    by_trajectory = {
        item: np.asarray(
            [index for index, value in enumerate(trajectory_ids) if value == item]
        )
        for item in unique
    }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([by_trajectory[unique[item]] for item in sampled])
        values[index] = float(np.mean(first[indices] - second[indices]))
    return {
        "bootstrap_samples": samples,
        "confidence_lower": float(np.quantile(values, 0.025, method="linear")),
        "confidence_upper": float(np.quantile(values, 0.975, method="linear")),
        "estimate": float(np.mean(first - second)),
        "resampling_unit": "source_trajectory",
        "seed": seed,
    }


def evaluate_external_visual_selections(
    *,
    visual_selection_paths: tuple[Path, Path, Path],
    candidate_pool: object,
    original_selection_manifest: object,
    selected_dataset: object,
    remainder_dataset: object,
    accepted_m3c_report: Mapping[str, object],
    external_config_digest: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    output_path: Path,
) -> Mapping[str, object]:
    """Join frozen visual choices to accepted outcomes, without replay."""
    from latentguard.selection.evaluation import join_complementary_replay_phases
    from latentguard.selection.outcomes import candidate_outcomes_from_replay_join

    joined = join_complementary_replay_phases(
        candidate_pool=cast(Any, candidate_pool),
        selection_manifest=cast(Any, original_selection_manifest),
        selected_dataset=cast(Any, selected_dataset),
        remainder_dataset=cast(Any, remainder_dataset),
    )
    outcomes = candidate_outcomes_from_replay_join(cast(Any, candidate_pool), joined)
    outcome_by_id = {item.proposal_id: item for item in outcomes}
    original_rates, original_vectors = _selector_successes(
        cast(Any, original_selection_manifest).selections, outcome_by_id
    )
    trajectory_by_group = {
        group.group_id: group.source_trajectory_id
        for group in cast(Any, candidate_pool).groups
    }
    domain_reports: dict[str, object] = {}
    for path in visual_selection_paths:
        selection = _load_self_digesting_json(path, "external visual selection")
        if (
            selection["candidate_pool_digest"]
            != cast(Any, candidate_pool).content_digest
            or selection["outcomes_available_during_selection"] is not False
        ):
            _fail(
                "external visual evaluation", "selection binding or blindness differs"
            )
        decisions = cast(list[Mapping[str, object]], selection["decisions"])
        visual_success = np.asarray(
            [
                float(
                    cast(
                        bool,
                        outcome_by_id[cast(str, item["selected_proposal_id"])].success,
                    )
                )
                for item in decisions
            ],
            dtype=np.float64,
        )
        group_ids = tuple(cast(str, item["group_id"]) for item in decisions)
        expected_group_ids = tuple(
            group.group_id for group in cast(Any, candidate_pool).groups
        )
        if set(group_ids) != set(expected_group_ids) or len(group_ids) != len(
            expected_group_ids
        ):
            _fail("external visual evaluation", "visual group coverage differs")
        trajectories = tuple(trajectory_by_group[item] for item in group_ids)
        groups_by_id = {
            group.group_id: group for group in cast(Any, candidate_pool).groups
        }
        random_expectation = np.asarray(
            [
                np.mean(
                    [
                        float(
                            cast(
                                bool,
                                outcome_by_id[candidate.proposal_id].success,
                            )
                        )
                        for candidate in groups_by_id[group_id].candidates
                    ]
                )
                for group_id in group_ids
            ],
            dtype=np.float64,
        )
        action_successes = original_vectors.get("action_only_ensemble_v1")
        if action_successes is None or set(action_successes) != set(group_ids):
            _fail("external visual evaluation", "action-only coverage differs")
        action_vector = np.asarray(
            [action_successes[group_id] for group_id in group_ids], dtype=np.float64
        )
        selected_success = float(np.mean(visual_success))
        random_success = float(np.mean(random_expectation))
        failure_rate = 1.0 - selected_success
        random_failure = 1.0 - random_success
        domain_reports[cast(str, selection["domain_id"])] = {
            "coverage": 1.0,
            "improvement_over_random": selected_success - random_success,
            "relative_failure_reduction_vs_random": (
                (random_failure - failure_rate) / random_failure
                if random_failure
                else 0.0
            ),
            "selected_success_rate": selected_success,
            "task_failure_rate": failure_rate,
            "regret_to_oracle": original_rates.get(
                "oracle_analysis_only_v1", selected_success
            )
            - selected_success,
            "visual_minus_action_only_bootstrap": _paired_bootstrap(
                visual_success,
                action_vector,
                trajectories,
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            ),
            "visual_minus_random_bootstrap": _paired_bootstrap(
                visual_success,
                random_expectation,
                trajectories,
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            ),
        }
    canonical = cast(Mapping[str, object], domain_reports.get("canonical"))
    if not canonical:
        _fail("external visual evaluation", "canonical domain is absent")
    canonical_success = cast(float, canonical["selected_success_rate"])
    for value in domain_reports.values():
        row = cast(dict[str, object], value)
        row["robustness_drop_from_canonical"] = canonical_success - cast(
            float, row["selected_success_rate"]
        )
    body: dict[str, object] = {
        "accepted_m3c_report_digest": accepted_m3c_report["content_digest"],
        "candidate_pool_digest": cast(Any, candidate_pool).content_digest,
        "domains": domain_reports,
        "existing_outcomes_joined_after_selection": True,
        "external_outcome_tuning_permitted": False,
        "external_evaluation_config_digest": external_config_digest,
        "original_baseline_selected_success_rates": {
            key: original_rates[key]
            for key in (
                "deterministic_random_v1",
                "action_only_ensemble_v1",
                "state_action_mlp_ensemble_v1",
                "temporal_ensemble_v1",
                "oracle_analysis_only_v1",
            )
            if key in original_rates
        },
        "replay_evidence_digest": joined.content_digest,
        "schema_version": "1.0",
        "simulator_replay_performed": False,
    }
    report = {**body, "content_digest": _digest(body, "M4BExternalEvaluationV1")}
    destination = Path(output_path)
    if destination.exists():
        _fail("external visual evaluation", "completed output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "ExternalVisualEvaluationError",
    "evaluate_external_visual_selections",
    "load_model_freeze_manifest",
    "select_external_visual_candidates",
]
