from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PickCubeVerifierStateSchemaV1,
    PickCubeVerifierStateV1,
)
from latentguard.selection.checkpoint_bundle import (
    ACCEPTED_M3B_SPLIT_DIGEST,
    EXPECTED_FIVE_SEEDS,
    BundleLoadingDiagnosticsV1,
    LoadedVerifierBundleV1,
    VerifierBundleError,
    VerifierBundleV1,
    VerifierCheckpointIdentityV1,
    VerifierRuntimeArtifactsV1,
    VerifierSeedArtifactPathsV1,
    load_verifier_bundle,
)
from latentguard.selection.ensemble import (
    EnsembleEndToEndProfileV1,
    VerifierEnsembleError,
    score_verifier_ensemble,
)
from latentguard.selection.inference import (
    INFERENCE_LATENCY_SEMANTIC,
    InferenceLatencyReportV1,
    VerifierInferenceError,
    normalize_candidate_inputs,
)
from latentguard.selection.preparation import (
    SELECTION_BUNDLE_REPORT_TYPE,
    load_prepared_bundle,
)
from latentguard.training.calibration import (
    TemperatureCalibrationStateV1,
    fit_validation_thresholds,
)
from latentguard.training.checkpoint import (
    CheckpointBindingV1,
    compute_checkpoint_content_digest,
    save_training_checkpoint,
)
from latentguard.training.config import load_model_config, load_training_config
from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST
from latentguard.training.models import build_model, resolve_model_config
from latentguard.training.preprocessing import (
    PreprocessingStateV1,
    save_preprocessing_state,
)
from latentguard.training.reporting import StrictReportV1, save_strict_report


class _IncreasingClock:
    def __init__(self, step: float = 0.001) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


@dataclass(frozen=True)
class _RuntimeFixture:
    bundle: VerifierBundleV1
    artifacts: VerifierRuntimeArtifactsV1


@pytest.fixture(scope="module")
def verifier_runtime(tmp_path_factory: pytest.TempPathFactory) -> _RuntimeFixture:
    import torch

    root = tmp_path_factory.mktemp("verifier-bundle")
    model_config = load_model_config(Path("configs/training/m3b/action-only-mlp.json"))
    training_config = load_training_config(
        Path("configs/training/m3b/train-default.json")
    )
    resolved_model = resolve_model_config(build_model(model_config, seed=0))
    preprocessing = PreprocessingStateV1(
        dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
        training_split_digest="sha256:" + "a" * 64,
        state_observation_count=10,
        valid_action_step_count=160,
        minimum_standard_deviation=1e-6,
        state_mean=np.zeros(38, dtype=np.float64),
        state_standard_deviation=np.ones(38, dtype=np.float64),
        action_mean=np.zeros(8, dtype=np.float64),
        action_standard_deviation=np.ones(8, dtype=np.float64),
    )
    preprocessing_path = save_preprocessing_state(
        preprocessing, root / "preprocessing.json"
    )
    identities: list[VerifierCheckpointIdentityV1] = []
    runtime_paths: dict[int, VerifierSeedArtifactPathsV1] = {}
    for seed in EXPECTED_FIVE_SEEDS:
        seed_root = root / f"seed-{seed}"
        seed_root.mkdir()
        run_identity = f"m3b-run-sha256-{seed:064x}"
        binding = CheckpointBindingV1(
            dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
            split_digest=ACCEPTED_M3B_SPLIT_DIGEST,
            preprocessing_digest=preprocessing.content_digest,
            model_config_digest=resolved_model.content_digest,
            training_config_digest=training_config.content_digest,
            run_manifest_identity=run_identity,
        )
        model = build_model(model_config, seed=seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        checkpoint = save_training_checkpoint(
            seed_root / "best.pt",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            binding=binding,
            model_configuration=resolved_model,
            training_configuration=training_config,
            preprocessing_state=preprocessing,
            epoch=seed + 1,
            global_step=0,
            best_validation_metric=0.5,
            best_epoch=seed + 1,
            kind="best",
        )
        checkpoint_digest = compute_checkpoint_content_digest(checkpoint)
        prediction_digest = f"sha256:{(seed + 100):064x}"
        calibration = TemperatureCalibrationStateV1(
            dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
            split_digest=ACCEPTED_M3B_SPLIT_DIGEST,
            checkpoint_identity=run_identity,
            checkpoint_content_digest=checkpoint_digest,
            model_config_digest=resolved_model.content_digest,
            preprocessing_digest=preprocessing.content_digest,
            validation_prediction_digest=prediction_digest,
            temperature=1.0 + seed / 10.0,
        )
        thresholds = fit_validation_thresholds(
            np.asarray((0.1, 0.8, 0.2, 0.9), dtype=np.float64),
            np.asarray((0, 1, 0, 1), dtype=np.int64),
            split="validation",
            dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
            split_digest=ACCEPTED_M3B_SPLIT_DIGEST,
            calibration_digest=calibration.content_digest,
            validation_prediction_digest=prediction_digest,
        )
        calibration_report = StrictReportV1(
            "temperature_calibration_v1", calibration.to_dict()
        )
        threshold_report = StrictReportV1("frozen_thresholds_v1", thresholds.to_dict())
        calibration_path = save_strict_report(
            calibration_report, seed_root / "calibration.json"
        )
        threshold_path = save_strict_report(
            threshold_report, seed_root / "thresholds.json"
        )
        identities.append(
            VerifierCheckpointIdentityV1(
                seed=seed,
                checkpoint_identity=run_identity,
                checkpoint_content_digest=checkpoint_digest,
                checkpoint_epoch=seed + 1,
                calibration_digest=calibration.content_digest,
                calibration_report_digest=calibration_report.content_digest,
                threshold_digest=thresholds.content_digest,
                threshold_report_digest=threshold_report.content_digest,
                validation_prediction_digest=prediction_digest,
            )
        )
        runtime_paths[seed] = VerifierSeedArtifactPathsV1(
            checkpoint=checkpoint,
            calibration=calibration_path,
            thresholds=threshold_path,
        )
    bundle = VerifierBundleV1(
        architecture=model_config.model_type.value,
        checkpoints=tuple(identities),
        model_configuration_digest=resolved_model.content_digest,
        training_configuration_digest=training_config.content_digest,
        preprocessing_digest=preprocessing.content_digest,
    )
    return _RuntimeFixture(
        bundle=bundle,
        artifacts=VerifierRuntimeArtifactsV1(
            preprocessing=preprocessing_path, seeds=runtime_paths
        ),
    )


def _state() -> PickCubeVerifierStateV1:
    schema = PickCubeVerifierStateSchemaV1(
        joint_names=tuple(f"joint_{index}" for index in range(9))
    )
    assert schema.dimension == 38
    return PickCubeVerifierStateV1(
        schema=schema, values=np.arange(38, dtype=np.float32)
    )


def _inputs(count: int = 3) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    actions = np.zeros((count, 16, 8), dtype=np.float64)
    for index in range(count):
        actions[index, :, :] = index / 10.0
    masks = np.ones((count, 16), dtype=np.bool_)
    return actions, masks


def test_bundle_import_is_torch_lazy_and_identity_excludes_runtime_paths(
    verifier_runtime: _RuntimeFixture,
) -> None:
    command = (
        "import sys; sys.path.insert(0, 'src'); "
        "import latentguard.selection.checkpoint_bundle; "
        "import latentguard.selection.inference; "
        "import latentguard.selection.ensemble; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
    encoded = json.dumps(verifier_runtime.bundle.to_dict(), sort_keys=True)
    assert "best.pt" not in encoded
    assert str(verifier_runtime.artifacts.preprocessing) not in encoded
    reloaded = VerifierBundleV1.from_dict(verifier_runtime.bundle.to_dict())
    assert reloaded.content_digest == verifier_runtime.bundle.content_digest

    moved_paths = VerifierRuntimeArtifactsV1(
        preprocessing=Path("Z:/runtime-only/preprocessing.json"),
        seeds={
            seed: VerifierSeedArtifactPathsV1(
                checkpoint=Path(f"Z:/runtime-only/{seed}/best.pt"),
                calibration=Path(f"Z:/runtime-only/{seed}/calibration.json"),
                thresholds=Path(f"Z:/runtime-only/{seed}/thresholds.json"),
            )
            for seed in EXPECTED_FIVE_SEEDS
        },
    )
    assert moved_paths != verifier_runtime.artifacts
    assert reloaded.content_digest == verifier_runtime.bundle.content_digest


def test_bundle_rejects_wrong_dataset_duplicate_seed_and_changed_content(
    verifier_runtime: _RuntimeFixture,
) -> None:
    with pytest.raises(VerifierBundleError, match="wrong accepted M3A dataset"):
        replace(verifier_runtime.bundle, dataset_digest="sha256:" + "f" * 64)
    duplicate = list(verifier_runtime.bundle.checkpoints)
    duplicate[1] = replace(duplicate[1], seed=0)
    with pytest.raises(VerifierBundleError, match="one ordered checkpoint"):
        replace(verifier_runtime.bundle, checkpoints=tuple(duplicate))
    forged = verifier_runtime.bundle.to_dict()
    forged["preprocessing_digest"] = "sha256:" + "e" * 64
    with pytest.raises(VerifierBundleError, match="bundle content changed"):
        VerifierBundleV1.from_dict(forged)


def test_runtime_loader_validates_all_five_artifacts_and_checkpoint_bytes_first(
    verifier_runtime: _RuntimeFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_verifier_bundle(
        verifier_runtime.bundle,
        verifier_runtime.artifacts,
        clock=_IncreasingClock(),
    )
    assert tuple(item.identity.seed for item in loaded.seeds) == EXPECTED_FIVE_SEEDS
    assert loaded.preprocessing.content_digest == (
        verifier_runtime.bundle.preprocessing_digest
    )
    assert loaded.loading_diagnostics.device == "cpu"
    assert len(loaded.loading_diagnostics.per_seed_model_seconds) == 5

    changed_checkpoint = tmp_path / "changed.pt"
    changed_checkpoint.write_bytes(
        verifier_runtime.artifacts.seeds[0].checkpoint.read_bytes() + b"changed"
    )
    changed_paths = dict(verifier_runtime.artifacts.seeds)
    changed_paths[0] = replace(changed_paths[0], checkpoint=changed_checkpoint)

    import torch

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "checkpoint was deserialized before digest"
        ),
    )
    with pytest.raises(VerifierBundleError, match="byte digest changed"):
        load_verifier_bundle(
            verifier_runtime.bundle,
            VerifierRuntimeArtifactsV1(
                preprocessing=verifier_runtime.artifacts.preprocessing,
                seeds=changed_paths,
            ),
            clock=_IncreasingClock(),
        )


def test_runtime_loader_rejects_architecture_and_calibration_binding_drift(
    verifier_runtime: _RuntimeFixture,
) -> None:
    wrong_architecture = replace(
        verifier_runtime.bundle, architecture="state_action_mlp"
    )
    with pytest.raises(VerifierBundleError, match="semantic binding differs"):
        load_verifier_bundle(
            wrong_architecture,
            verifier_runtime.artifacts,
            clock=_IncreasingClock(),
        )
    changed_identity = replace(
        verifier_runtime.bundle.checkpoints[0],
        calibration_digest="sha256:" + "d" * 64,
    )
    changed_bundle = replace(
        verifier_runtime.bundle,
        checkpoints=(changed_identity, *verifier_runtime.bundle.checkpoints[1:]),
    )
    with pytest.raises(VerifierBundleError, match="calibration.*binding differs"):
        load_verifier_bundle(
            changed_bundle,
            verifier_runtime.artifacts,
            clock=_IncreasingClock(),
        )


def test_label_free_normalization_has_exact_allowlist_and_detaches_inputs(
    verifier_runtime: _RuntimeFixture,
) -> None:
    loaded = load_verifier_bundle(
        verifier_runtime.bundle,
        verifier_runtime.artifacts,
        clock=_IncreasingClock(),
    )
    actions, masks = _inputs()
    masks[1, 8:] = False
    batch = normalize_candidate_inputs(
        _state(),
        actions,
        masks,
        ("candidate-z", "candidate-a", "candidate-m"),
        loaded.preprocessing,
    )
    assert tuple(inspect.signature(normalize_candidate_inputs).parameters) == (
        "state",
        "candidate_actions",
        "action_masks",
        "candidate_ids",
        "preprocessing",
    )
    assert batch.model_inputs() == (
        batch.state_vectors,
        batch.action_chunks,
        batch.action_masks,
    )
    assert batch.state_vectors.shape == (3, 38)
    assert batch.action_chunks.dtype == np.float32
    assert np.all(batch.action_chunks[1, 8:] == 0.0)
    actions[:] = 9.0
    masks[:] = False
    assert np.all(batch.action_chunks[0] == 0.0)
    assert np.all(batch.action_masks[0])

    float32_actions = np.zeros((3, 16, 8), dtype=np.float32)
    with pytest.raises(VerifierInferenceError, match="little-endian float64"):
        normalize_candidate_inputs(
            _state(),
            float32_actions,
            np.ones((3, 16), dtype=np.bool_),
            ("a", "b", "c"),
            loaded.preprocessing,
        )
    with pytest.raises(VerifierInferenceError, match="duplicate candidate"):
        normalize_candidate_inputs(
            _state(),
            np.zeros((3, 16, 8), dtype=np.float64),
            np.ones((3, 16), dtype=np.bool_),
            ("a", "a", "c"),
            loaded.preprocessing,
        )


def test_five_seed_calibrated_mean_and_candidate_id_tie_break_are_stable(
    verifier_runtime: _RuntimeFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    loaded = load_verifier_bundle(
        verifier_runtime.bundle,
        verifier_runtime.artifacts,
        clock=_IncreasingClock(),
    )

    class ConstantLogitModel(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value
            self.seen_shapes: tuple[tuple[int, ...], ...] | None = None
            self.call_count = 0

        def forward(
            self, state: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor
        ) -> torch.Tensor:
            self.call_count += 1
            self.seen_shapes = (
                tuple(state.shape),
                tuple(actions.shape),
                tuple(masks.shape),
            )
            return torch.full(
                (state.shape[0],), self.value, dtype=state.dtype, device=state.device
            )

    models = tuple(ConstantLogitModel(seed - 2.0) for seed in EXPECTED_FIVE_SEEDS)
    fake_loaded = LoadedVerifierBundleV1(
        identity=loaded.identity,
        preprocessing=loaded.preprocessing,
        seeds=tuple(
            replace(item, model=model)
            for item, model in zip(loaded.seeds, models, strict=True)
        ),
        loading_diagnostics=BundleLoadingDiagnosticsV1(
            device="cpu",
            total_seconds=0.1,
            per_seed_model_seconds=(0.01,) * 5,
        ),
    )
    actions, masks = _inputs()
    ids = ("candidate-z", "candidate-a", "candidate-m")
    normalization_calls = 0
    original_normalize = normalize_candidate_inputs

    def counted_normalize(*args: object, **kwargs: object) -> object:
        nonlocal normalization_calls
        normalization_calls += 1
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(
        "latentguard.selection.ensemble.normalize_candidate_inputs",
        counted_normalize,
    )
    result = score_verifier_ensemble(
        fake_loaded,
        _state(),
        actions,
        masks,
        ids,
        clock=_IncreasingClock(),
    )
    expected_seed_probabilities = np.stack(
        [
            item.calibration.apply(np.full(3, model.value, dtype=np.float64))
            for item, model in zip(fake_loaded.seeds, models, strict=True)
        ]
    )
    assert np.array_equal(
        result.per_seed_calibrated_failure_probabilities,
        expected_seed_probabilities,
    )
    assert np.array_equal(
        result.ensemble_failure_probabilities,
        np.mean(expected_seed_probabilities, axis=0, dtype=np.float64),
    )
    assert result.ranking == ("candidate-a", "candidate-m", "candidate-z")
    assert result.selected_candidate_id == "candidate-a"
    assert result.probability_for("candidate-z") == pytest.approx(
        result.ensemble_failure_probabilities[0]
    )
    assert all(model.seen_shapes == ((3, 38), (3, 16, 8), (3, 16)) for model in models)
    assert all(model.call_count == 7 for model in models)
    assert normalization_calls == 7
    profile = result.diagnostics.end_to_end_profile
    assert profile.group_durations_seconds == pytest.approx((0.001,) * 5)
    assert profile.latency_report.measurement_iterations == 5
    assert "group_durations_seconds" not in result.diagnostics.to_dict()
    assert (
        result.diagnostics.to_dict()["end_to_end_profile"][
            "raw_duration_values_persisted"
        ]
        is False
    )
    permutation = np.asarray((1, 2, 0), dtype=np.int64)
    permuted = score_verifier_ensemble(
        fake_loaded,
        _state(),
        np.ascontiguousarray(actions[permutation]),
        np.ascontiguousarray(masks[permutation]),
        tuple(ids[index] for index in permutation),
        clock=_IncreasingClock(),
    )
    assert permuted.ranking == result.ranking
    assert normalization_calls == 14
    with pytest.raises(VerifierEnsembleError, match="exactly 2 warmups and 5"):
        score_verifier_ensemble(
            fake_loaded,
            _state(),
            actions,
            masks,
            ids,
            warmup_iterations=0,
            measurement_iterations=5,
        )


def test_end_to_end_profile_never_serializes_raw_duration_values() -> None:
    profile = EnsembleEndToEndProfileV1(
        device="cpu",
        candidate_count=8,
        group_durations_seconds=(0.008, 0.016, 0.024, 0.032, 0.040),
    )
    payload = profile.to_dict()
    assert payload["per_group_seconds_p50"] == pytest.approx(0.024)
    assert payload["per_group_seconds_p95"] == pytest.approx(0.0384)
    assert payload["per_group_seconds_p99"] == pytest.approx(0.03968)
    assert payload["raw_duration_values_persisted"] is False
    assert "group_durations_seconds" not in payload


def test_latency_report_schema_is_fixed_and_uses_linear_quantiles() -> None:
    report = InferenceLatencyReportV1.from_group_measurements(
        (0.008, 0.016, 0.024),
        device="cpu",
        candidate_count=8,
        warmup_iterations=2,
        peak_allocated_device_memory_bytes=0,
    )
    assert report.semantic == INFERENCE_LATENCY_SEMANTIC
    assert report.per_group_seconds_p50 == pytest.approx(0.016)
    assert report.per_candidate_seconds_p50 == pytest.approx(0.002)
    assert report.candidates_per_second == pytest.approx(500.0)
    assert set(report.to_dict()) == {
        "candidate_count",
        "candidates_per_second",
        "device",
        "measurement_iterations",
        "peak_allocated_device_memory_bytes",
        "per_candidate_seconds_p50",
        "per_candidate_seconds_p95",
        "per_candidate_seconds_p99",
        "per_group_seconds_p50",
        "per_group_seconds_p95",
        "per_group_seconds_p99",
        "quantile_method",
        "schema_version",
        "semantic",
        "warmup_iterations",
    }


def test_prepared_bundle_report_round_trip_excludes_runtime_paths(
    verifier_runtime: _RuntimeFixture, tmp_path: Path
) -> None:
    report = StrictReportV1(
        SELECTION_BUNDLE_REPORT_TYPE,
        verifier_runtime.bundle.to_dict(),
    )
    path = save_strict_report(report, tmp_path / "action-only-bundle.json")
    loaded = load_prepared_bundle(path)
    assert loaded.content_digest == verifier_runtime.bundle.content_digest
    assert str(verifier_runtime.artifacts.preprocessing) not in path.read_text(
        encoding="utf-8"
    )
