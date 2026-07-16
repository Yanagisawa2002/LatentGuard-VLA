from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

import latentguard.training.trainer as trainer_module
from latentguard.action_verifier import CandidateType, DatasetSplit
from latentguard.training.checkpoint import (
    CheckpointBindingV1,
    CheckpointError,
    inspect_training_checkpoint,
    load_training_checkpoint,
)
from latentguard.training.config import load_model_config, load_training_config
from latentguard.training.dataset import (
    AcceptedActionVerifierDatasetV1,
    ActionVerifierModelExampleV1,
    ActionVerifierReportingMetadataV1,
)
from latentguard.training.inference import infer_action_verifier_split
from latentguard.training.manifest import TrainingRunIdentity
from latentguard.training.models import build_model, resolve_model_config
from latentguard.training.preprocessing import fit_preprocessing_state
from latentguard.training.trainer import (
    run_forward_backward_gate,
    run_tiny_overfit_gate,
    train_action_verifier,
)


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _projected_dataset() -> AcceptedActionVerifierDatasetV1:
    examples: list[ActionVerifierModelExampleV1] = []
    reporting: list[ActionVerifierReportingMetadataV1] = []
    groups: dict[int, tuple[int, ...]] = {}
    splits: dict[DatasetSplit, list[int]] = {split: [] for split in DatasetSplit}
    for group_index in range(7):
        split = (
            DatasetSplit.TRAIN
            if group_index < 4
            else DatasetSplit.VALIDATION
            if group_index < 6
            else DatasetSplit.TEST
        )
        members: list[int] = []
        for member_index, target in enumerate((0, 0, 1)):
            sample_index = len(examples)
            members.append(sample_index)
            splits[split].append(sample_index)
            state = np.full(38, float(group_index - 2), dtype=np.float32)
            actions = np.full(
                (16, 8),
                float(group_index + target * 4 + member_index / 10),
                dtype=np.float64,
            )
            examples.append(
                ActionVerifierModelExampleV1(
                    state_vector=state,
                    action_chunk=actions,
                    action_mask=np.ones(16, dtype=np.bool_),
                    failure_target=target,
                    sample_index=sample_index,
                )
            )
            candidate_type = (
                CandidateType.SOURCE if member_index == 0 else CandidateType.CORRUPTED
            )
            reporting.append(
                ActionVerifierReportingMetadataV1(
                    sample_index=sample_index,
                    group_index=group_index,
                    sample_id=f"sample-{sample_index:03d}",
                    group_id=f"group-{group_index:03d}",
                    anchor_id=f"anchor-{group_index:03d}",
                    dataset_split=split,
                    candidate_type=candidate_type,
                    corruption_family=(
                        None if candidate_type is CandidateType.SOURCE else "fixture"
                    ),
                    severity=(
                        None if candidate_type is CandidateType.SOURCE else "bounded"
                    ),
                    source_trajectory=f"trajectory-{group_index:03d}",
                    anchor_selection_reason="fixture_anchor",
                )
            )
        groups[group_index] = tuple(members)
    return AcceptedActionVerifierDatasetV1(
        dataset_digest=_digest("dataset"),
        split_digest=_digest("split"),
        training_split_digest=_digest("train"),
        acceptance_report_digest=_digest("report"),
        examples=tuple(examples),
        reporting=tuple(reporting),
        split_indices=MappingProxyType(
            {split: tuple(values) for split, values in splits.items()}
        ),
        group_members=MappingProxyType(groups),
    )


def _training_context() -> tuple[
    AcceptedActionVerifierDatasetV1,
    object,
    object,
    object,
    CheckpointBindingV1,
]:
    dataset = _projected_dataset()
    model_config = load_model_config(Path("configs/training/m3b/state-only-mlp.json"))
    training_config = replace(
        load_training_config(Path("configs/training/m3b/train-default.json")),
        batch_size=4,
        max_epochs=3,
        max_steps=4,
        early_stopping_patience=3,
        checkpoint_interval_epochs=1,
    )
    preprocessing = fit_preprocessing_state(dataset)
    model = build_model(model_config, seed=7)
    resolved = resolve_model_config(model)
    identity = TrainingRunIdentity(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved.content_digest,
        training_config_digest=training_config.content_digest,
        seed=7,
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
    return dataset, model, resolved, training_config, binding


def test_forward_overfit_checkpoint_and_bounded_training(tmp_path: Path) -> None:
    dataset, model, resolved, training_config, binding = _training_context()
    preprocessing = fit_preprocessing_state(dataset)
    initial_loss = run_forward_backward_gate(
        model, dataset, preprocessing, training_config, device="cpu"
    )
    assert initial_loss > 0.0

    gate = run_tiny_overfit_gate(
        model=build_model(resolved.architecture, seed=2),
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=tmp_path / "gate",
        steps=80,
        maximum_samples=64,
    )
    assert gate.final_loss < gate.initial_loss * 0.5
    assert gate.final_accuracy >= 0.9
    assert gate.checkpoint_reload_valid
    assert gate.checkpoint_resume_valid

    result = train_action_verifier(
        model=build_model(resolved.architecture, seed=7),
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=tmp_path / "run",
        seed=7,
        device="cpu",
    )
    assert result.global_step == 4
    assert result.best_checkpoint.is_file()
    assert result.final_checkpoint.is_file()
    assert result.history

    metadata = inspect_training_checkpoint(result.best_checkpoint)
    reloaded = build_model(metadata.model_configuration.architecture, seed=999)
    loaded = load_training_checkpoint(
        result.best_checkpoint,
        expected_binding=binding,
        model=reloaded,
        restore_rng=False,
    )
    assert loaded.global_step > 0
    inference = infer_action_verifier_split(
        reloaded,
        dataset,
        preprocessing,
        split=DatasetSplit.VALIDATION,
        batch_size=4,
        device="cpu",
    )
    assert inference.logits.shape == (6,)
    assert tuple(item.sample_index for item in inference.metadata) == tuple(range(6))


def test_checkpoint_rejects_changed_binding_and_malformed_optimizer(
    tmp_path: Path,
) -> None:
    dataset, _, resolved, training_config, binding = _training_context()
    preprocessing = fit_preprocessing_state(dataset)
    result = train_action_verifier(
        model=build_model(resolved.architecture, seed=7),
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=tmp_path / "run",
        seed=7,
        device="cpu",
    )
    changed = replace(binding, dataset_digest=_digest("changed"))
    with pytest.raises(CheckpointError, match="binding"):
        load_training_checkpoint(
            result.final_checkpoint,
            expected_binding=changed,
            model=build_model(resolved.architecture, seed=0),
            restore_rng=False,
        )

    payload = torch.load(result.final_checkpoint, weights_only=False)
    payload["optimizer_state"] = {"state": []}
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(CheckpointError, match="optimizer"):
        load_training_checkpoint(
            tampered,
            expected_binding=binding,
            model=build_model(resolved.architecture, seed=0),
            restore_rng=False,
        )


def test_semantic_overrides_must_be_in_training_digest(tmp_path: Path) -> None:
    dataset, _, resolved, training_config, binding = _training_context()
    with pytest.raises(RuntimeError, match="resolved into the training config"):
        train_action_verifier(
            model=build_model(resolved.architecture, seed=7),
            resolved_model_config=resolved,
            training_config=training_config,
            dataset=dataset,
            preprocessing=fit_preprocessing_state(dataset),
            binding=binding,
            output_directory=tmp_path / "run",
            seed=7,
            device="cpu",
            max_steps_override=3,
        )


def test_interrupted_resume_matches_uninterrupted_training_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _projected_dataset()
    preprocessing = fit_preprocessing_state(dataset)
    model_config = load_model_config(Path("configs/training/m3b/state-only-mlp.json"))
    training_config = replace(
        load_training_config(Path("configs/training/m3b/train-default.json")),
        batch_size=4,
        max_epochs=8,
        max_steps=None,
        early_stopping_patience=3,
        early_stopping_min_delta=1.0,
        checkpoint_interval_epochs=1,
    )
    seed = 19
    resolved = resolve_model_config(build_model(model_config, seed=seed))
    identity = TrainingRunIdentity(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved.content_digest,
        training_config_digest=training_config.content_digest,
        seed=seed,
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

    class SimulatedInterruption(RuntimeError):
        pass

    original_iter_numpy_batches = trainer_module.iter_numpy_batches
    shuffled_iterator_calls = 0

    def interrupt_before_third_training_epoch(
        *args: object, **kwargs: object
    ) -> object:
        nonlocal shuffled_iterator_calls
        if kwargs.get("shuffle") is True:
            shuffled_iterator_calls += 1
            if shuffled_iterator_calls == 3:
                raise SimulatedInterruption("simulated process interruption")
        return original_iter_numpy_batches(*args, **kwargs)  # type: ignore[arg-type]

    interrupted_root = tmp_path / "interrupted"
    monkeypatch.setattr(
        trainer_module,
        "iter_numpy_batches",
        interrupt_before_third_training_epoch,
    )
    with pytest.raises(SimulatedInterruption, match="simulated process interruption"):
        train_action_verifier(
            model=build_model(resolved.architecture, seed=seed),
            resolved_model_config=resolved,
            training_config=training_config,
            dataset=dataset,
            preprocessing=preprocessing,
            binding=binding,
            output_directory=interrupted_root,
            seed=seed,
            device="cpu",
        )
    monkeypatch.setattr(
        trainer_module,
        "iter_numpy_batches",
        original_iter_numpy_batches,
    )

    periodic_checkpoint = interrupted_root / "checkpoints" / "periodic.pt"
    interrupted_progress = inspect_training_checkpoint(periodic_checkpoint).progress
    assert interrupted_progress.epoch == 1
    assert interrupted_progress.early_stopping_patience_count == 1
    raw_history = json.loads(
        (interrupted_root / "history.json").read_text(encoding="utf-8")
    )
    assert [item["epoch"] for item in raw_history["epochs"]] == [0, 1]
    assert raw_history["epochs"][-1]["global_step"] == (
        interrupted_progress.global_step
    )

    resumed = train_action_verifier(
        model=build_model(resolved.architecture, seed=999),
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=interrupted_root,
        seed=seed,
        device="cpu",
        resume_checkpoint=periodic_checkpoint,
    )
    uninterrupted = train_action_verifier(
        model=build_model(resolved.architecture, seed=seed),
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=tmp_path / "uninterrupted",
        seed=seed,
        device="cpu",
    )

    assert resumed.final_epoch == uninterrupted.final_epoch == 3
    assert resumed.global_step == uninterrupted.global_step
    assert resumed.best_epoch == uninterrupted.best_epoch == 0
    assert (
        resumed.best_validation_failure_auprc
        == uninterrupted.best_validation_failure_auprc
    )
    assert tuple(replace(item, duration_seconds=0.0) for item in resumed.history) == (
        tuple(replace(item, duration_seconds=0.0) for item in uninterrupted.history)
    )
    assert (
        inspect_training_checkpoint(
            resumed.final_checkpoint
        ).progress.early_stopping_patience_count
        == inspect_training_checkpoint(
            uninterrupted.final_checkpoint
        ).progress.early_stopping_patience_count
        == training_config.early_stopping_patience
    )

    resumed_model = build_model(resolved.architecture, seed=0)
    uninterrupted_model = build_model(resolved.architecture, seed=1)
    load_training_checkpoint(
        resumed.final_checkpoint,
        expected_binding=binding,
        model=resumed_model,
        restore_rng=False,
    )
    load_training_checkpoint(
        uninterrupted.final_checkpoint,
        expected_binding=binding,
        model=uninterrupted_model,
        restore_rng=False,
    )
    assert resumed_model.state_dict().keys() == uninterrupted_model.state_dict().keys()
    for name, resumed_tensor in resumed_model.state_dict().items():
        assert torch.equal(resumed_tensor, uninterrupted_model.state_dict()[name])

    resumed_payload = torch.load(resumed.final_checkpoint, weights_only=False)
    uninterrupted_payload = torch.load(
        uninterrupted.final_checkpoint, weights_only=False
    )
    assert (
        resumed_payload["python_rng_state"] == uninterrupted_payload["python_rng_state"]
    )
    assert torch.equal(
        resumed_payload["torch_cpu_rng_state"],
        uninterrupted_payload["torch_cpu_rng_state"],
    )
    resumed_numpy_rng = resumed_payload["numpy_rng_state"]
    uninterrupted_numpy_rng = uninterrupted_payload["numpy_rng_state"]
    assert set(resumed_numpy_rng) == set(uninterrupted_numpy_rng)
    for key in resumed_numpy_rng:
        if key == "keys":
            assert np.array_equal(resumed_numpy_rng[key], uninterrupted_numpy_rng[key])
        else:
            assert resumed_numpy_rng[key] == uninterrupted_numpy_rng[key]

    terminal_history = (tmp_path / "uninterrupted" / "history.json").read_bytes()
    terminal_resume = train_action_verifier(
        model=build_model(resolved.architecture, seed=999),
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=tmp_path / "uninterrupted",
        seed=seed,
        device="cpu",
        resume_checkpoint=(tmp_path / "uninterrupted" / "checkpoints" / "periodic.pt"),
    )
    assert terminal_resume.final_epoch == uninterrupted.final_epoch
    assert terminal_resume.global_step == uninterrupted.global_step
    assert terminal_resume.history == uninterrupted.history
    assert (
        tmp_path / "uninterrupted" / "history.json"
    ).read_bytes() == terminal_history


@pytest.mark.parametrize(
    "config_name",
    (
        "state-only-mlp.json",
        "action-only-mlp.json",
        "state-action-mlp.json",
        "state-action-transformer.json",
    ),
)
def test_every_model_passes_tiny_overfit_reload_and_resume(
    config_name: str, tmp_path: Path
) -> None:
    dataset = _projected_dataset()
    preprocessing = fit_preprocessing_state(dataset)
    model_config = load_model_config(Path("configs/training/m3b") / config_name)
    training_config = replace(
        load_training_config(Path("configs/training/m3b/train-default.json")),
        batch_size=4,
        max_epochs=2,
        max_steps=4,
    )
    model = build_model(model_config, seed=11)
    resolved = resolve_model_config(model)
    identity = TrainingRunIdentity(
        dataset_digest=dataset.dataset_digest,
        split_digest=dataset.split_digest,
        preprocessing_digest=preprocessing.content_digest,
        model_config_digest=resolved.content_digest,
        training_config_digest=training_config.content_digest,
        seed=11,
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
    result = run_tiny_overfit_gate(
        model=model,
        resolved_model_config=resolved,
        training_config=training_config,
        dataset=dataset,
        preprocessing=preprocessing,
        binding=binding,
        output_directory=tmp_path / config_name,
        steps=100,
        maximum_samples=4,
    )
    assert result.final_accuracy >= 0.9
    assert result.checkpoint_reload_valid
    assert result.checkpoint_resume_valid
