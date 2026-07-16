from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from latentguard.action_verifier import DatasetSplit
from latentguard.cli import main
from latentguard.training.calibration import frozen_thresholds_from_dict
from latentguard.training.checkpoint import (
    CheckpointBindingV1,
    compute_checkpoint_content_digest,
    inspect_training_checkpoint,
)
from latentguard.training.config import (
    ModelType,
    load_model_config,
    load_training_config,
)
from latentguard.training.manifest import TrainingRunIdentity
from latentguard.training.models import build_model, resolve_model_config
from latentguard.training.preprocessing import fit_preprocessing_state
from latentguard.training.reporting import (
    StrictReportV1,
    load_strict_report,
    save_strict_report,
)
from latentguard.training.selection import (
    SeedValidationResult,
    select_model_architecture,
)
from latentguard.training.trainer import train_action_verifier
from test_training_checkpoint_trainer import _projected_dataset


def _patch_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    from latentguard import m3b_cli

    monkeypatch.setattr(
        m3b_cli, "_load_accepted_dataset", lambda _: _projected_dataset()
    )


def test_m3b_help_registration_keeps_torch_lazy() -> None:
    command = (
        "import sys; from latentguard.cli import _build_parser; "
        "_build_parser(); print('torch' in sys.modules)"
    )
    completed = subprocess.run(
        [str(Path(".venv/Scripts/python.exe")), "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"
    for command_name in (
        "train-action-verifier",
        "evaluate-action-verifier",
        "benchmark-action-verifier",
    ):
        with pytest.raises(SystemExit) as error:
            main([command_name, "--help"])
        assert error.value.code == 0


def test_train_and_benchmark_dry_runs_are_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_dataset(monkeypatch)
    train_output = tmp_path / "train-output"
    train_result = main(
        [
            "train-action-verifier",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--acceptance-report",
            str(tmp_path / "report.json"),
            "--anchor-manifest-dir",
            str(tmp_path / "anchors"),
            "--output-run-dir",
            str(train_output),
            "--model-config",
            "configs/training/m3b/state-action-mlp.json",
            "--training-config",
            "configs/training/m3b/train-default.json",
            "--seed",
            "3",
            "--max-steps",
            "4",
            "--max-epochs",
            "2",
            "--limit-samples",
            "12",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    assert train_result == 0
    assert not train_output.exists()

    benchmark_output = tmp_path / "benchmark-output"
    benchmark_result = main(
        [
            "benchmark-action-verifier",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--acceptance-report",
            str(tmp_path / "report.json"),
            "--anchor-manifest-dir",
            str(tmp_path / "anchors"),
            "--output-dir",
            str(benchmark_output),
            "--mode",
            "smoke",
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    assert benchmark_result == 0
    assert not benchmark_output.exists()


def test_validation_evaluation_cli_and_test_freeze_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_dataset(monkeypatch)
    dataset = _projected_dataset()
    model_config = load_model_config(Path("configs/training/m3b/state-action-mlp.json"))
    training_config = replace(
        load_training_config(Path("configs/training/m3b/train-default.json")),
        batch_size=4,
        max_epochs=2,
        max_steps=3,
    )
    preprocessing = fit_preprocessing_state(dataset)
    model = build_model(model_config, seed=0)
    resolved = resolve_model_config(model)
    identity = TrainingRunIdentity(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved.content_digest,
        training_config_digest=training_config.content_digest,
        seed=0,
        code_semantic_version="direct_action_verifier_training_v1",
    )
    binding = CheckpointBindingV1(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved.content_digest,
        training_config_digest=training_config.content_digest,
        run_manifest_identity=identity.run_id,
    )
    trained = train_action_verifier(
        model=model,
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=tmp_path / "run",
        seed=0,
        device="cpu",
    )
    common = [
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--acceptance-report",
        str(tmp_path / "report.json"),
        "--anchor-manifest-dir",
        str(tmp_path / "anchors"),
        "--checkpoint",
        str(trained.best_checkpoint),
        "--device",
        "cpu",
    ]
    validation_output = tmp_path / "validation"
    assert (
        main(
            [
                "evaluate-action-verifier",
                *common,
                "--split",
                "validation",
                "--output-dir",
                str(validation_output),
            ]
        )
        == 0
    )
    assert (validation_output / "evaluation.json").is_file()
    assert (validation_output / "calibration.json").is_file()
    assert (validation_output / "thresholds.json").is_file()

    import latentguard.training.inference as inference_module

    original_inference = inference_module.infer_action_verifier_split
    observed_splits: list[DatasetSplit] = []

    def spy_inference(*args: object, **kwargs: object) -> object:
        observed_splits.append(kwargs["split"])  # type: ignore[arg-type]
        return original_inference(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(inference_module, "infer_action_verifier_split", spy_inference)
    assert (
        main(
            [
                "evaluate-action-verifier",
                *common,
                "--split",
                "test",
                "--output-dir",
                str(tmp_path / "forbidden-test"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "forbidden-test").exists()
    assert DatasetSplit.TEST not in observed_splits
    monkeypatch.setattr(
        inference_module,
        "infer_action_verifier_split",
        original_inference,
    )

    calibration = load_strict_report(
        validation_output / "calibration.json",
        expected_report_type="temperature_calibration_v1",
    ).payload
    checkpoint_digest = compute_checkpoint_content_digest(trained.best_checkpoint)
    checkpoint_metadata = inspect_training_checkpoint(trained.best_checkpoint)
    model_files = {
        ModelType.STATE_ONLY_MLP: "state-only-mlp.json",
        ModelType.ACTION_ONLY_MLP: "action-only-mlp.json",
        ModelType.STATE_ACTION_MLP: "state-action-mlp.json",
        ModelType.TEMPORAL_STATE_ACTION_VERIFIER: "state-action-transformer.json",
    }
    selection_inputs: list[SeedValidationResult] = []
    for ordinal, (model_type, filename) in enumerate(model_files.items()):
        candidate = resolve_model_config(
            build_model(
                load_model_config(Path("configs/training/m3b") / filename),
                seed=0,
            )
        )
        primary = 0.80 if model_type is ModelType.STATE_ACTION_MLP else 0.40
        for seed in range(5):
            selected_run = model_type is ModelType.STATE_ACTION_MLP and seed == 0
            selection_inputs.append(
                SeedValidationResult(
                    model_type=model_type.value,
                    seed=seed,
                    failure_auprc=primary + seed * 0.001,
                    brier_score=0.20 if selected_run else 0.30,
                    pairwise_concordance=primary,
                    parameter_count=candidate.parameter_count,
                    model_config_digest=candidate.content_digest,
                    training_config_digest=training_config.content_digest,
                    checkpoint_identity=(
                        identity.run_id
                        if selected_run
                        else f"fixture-{model_type.value}-{seed}"
                    ),
                    checkpoint_content_digest=(
                        checkpoint_digest
                        if selected_run
                        else "sha256:" + f"{ordinal * 10 + seed + 1:064x}"[-64:]
                    ),
                    checkpoint_kind="best",
                    checkpoint_epoch=(
                        checkpoint_metadata.progress.epoch if selected_run else seed
                    ),
                    validation_prediction_digest=(
                        str(calibration["validation_prediction_digest"])
                        if selected_run
                        else "sha256:" + f"{ordinal * 10 + seed + 101:064x}"[-64:]
                    ),
                )
            )
    selection = select_model_architecture(
        selection_inputs,
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
    )
    selection_path = tmp_path / "selection.json"
    save_strict_report(
        StrictReportV1("selection_record_v1", selection.to_dict()),
        selection_path,
    )
    threshold_payload = load_strict_report(
        validation_output / "thresholds.json",
        expected_report_type="frozen_thresholds_v1",
    ).payload
    wrong_thresholds = replace(
        frozen_thresholds_from_dict(threshold_payload),
        dataset_digest="sha256:" + "f" * 64,
    )
    wrong_threshold_path = tmp_path / "wrong-thresholds.json"
    save_strict_report(
        StrictReportV1("frozen_thresholds_v1", wrong_thresholds.to_dict()),
        wrong_threshold_path,
    )
    observed_splits.clear()
    monkeypatch.setattr(inference_module, "infer_action_verifier_split", spy_inference)
    assert (
        main(
            [
                "evaluate-action-verifier",
                *common,
                "--split",
                "test",
                "--output-dir",
                str(tmp_path / "wrong-threshold-test"),
                "--calibration",
                str(validation_output / "calibration.json"),
                "--thresholds",
                str(wrong_threshold_path),
                "--selection-record",
                str(selection_path),
                "--dry-run",
            ]
        )
        == 1
    )
    assert DatasetSplit.TEST not in observed_splits
    monkeypatch.setattr(
        inference_module,
        "infer_action_verifier_split",
        original_inference,
    )
    test_output = tmp_path / "authorized-test"
    assert (
        main(
            [
                "evaluate-action-verifier",
                *common,
                "--split",
                "test",
                "--output-dir",
                str(test_output),
                "--calibration",
                str(validation_output / "calibration.json"),
                "--thresholds",
                str(validation_output / "thresholds.json"),
                "--selection-record",
                str(selection_path),
            ]
        )
        == 0
    )
    assert (test_output / "evaluation.json").is_file()


def test_invalid_training_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_dataset(monkeypatch)
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version":"1.0","unknown":true}', encoding="utf-8")
    result = main(
        [
            "train-action-verifier",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--acceptance-report",
            str(tmp_path / "report.json"),
            "--anchor-manifest-dir",
            str(tmp_path / "anchors"),
            "--output-run-dir",
            str(tmp_path / "output"),
            "--model-config",
            str(invalid),
            "--training-config",
            "configs/training/m3b/train-default.json",
            "--seed",
            "0",
            "--dry-run",
        ]
    )
    assert result == 1
    assert not (tmp_path / "output").exists()
