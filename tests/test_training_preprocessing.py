"""CPU-only tests for train-only verifier preprocessing and batching."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import latentguard.training.preprocessing as preprocessing_module
from latentguard.action_verifier import CandidateType, DatasetSplit
from latentguard.training.baselines import (
    fit_action_magnitude_baseline,
    fit_constant_baselines,
)
from latentguard.training.batching import collate_numpy_examples
from latentguard.training.dataset import (
    AcceptedActionVerifierDatasetV1,
    ActionVerifierModelExampleV1,
    ActionVerifierReportingMetadataV1,
)
from latentguard.training.preprocessing import (
    PreprocessingError,
    fit_preprocessing_state,
    load_preprocessing_state,
    normalize_example,
    save_preprocessing_state,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _example(
    index: int,
    value: float,
    *,
    target: int = 0,
    mask: np.ndarray | None = None,
) -> ActionVerifierModelExampleV1:
    resolved_mask = np.ones(16, dtype=np.bool_) if mask is None else mask
    return ActionVerifierModelExampleV1(
        state_vector=np.full(38, value, dtype=np.float32),
        action_chunk=np.full((16, 8), value, dtype=np.float32),
        action_mask=resolved_mask,
        failure_target=target,
        sample_index=index,
    )


def _accepted_dataset() -> AcceptedActionVerifierDatasetV1:
    examples = (
        _example(0, 0.0, target=0),
        _example(1, 2.0, target=1),
        _example(2, 100.0, target=1),
        _example(3, 200.0, target=1),
    )
    splits = (
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
        DatasetSplit.TEST,
    )
    reporting = tuple(
        ActionVerifierReportingMetadataV1(
            sample_index=index,
            group_index=index,
            sample_id=f"sample-{index}",
            group_id=f"group-{index}",
            anchor_id=f"anchor-{index}",
            dataset_split=split,
            candidate_type=CandidateType.SOURCE,
            corruption_family=None,
            severity=None,
            source_trajectory=f"trajectory-{index}",
            anchor_selection_reason="early_trajectory",
        )
        for index, split in enumerate(splits)
    )
    return AcceptedActionVerifierDatasetV1(
        dataset_digest=_digest("a"),
        split_digest=_digest("b"),
        training_split_digest=_digest("c"),
        acceptance_report_digest=_digest("d"),
        examples=examples,
        reporting=reporting,
        split_indices=MappingProxyType(
            {
                DatasetSplit.TRAIN: (0, 1),
                DatasetSplit.VALIDATION: (2,),
                DatasetSplit.TEST: (3,),
            }
        ),
        group_members=MappingProxyType({0: (0,), 1: (1,), 2: (2,), 3: (3,)}),
    )


def test_fit_uses_train_split_only_and_constant_baselines_ignore_holdouts() -> None:
    dataset = _accepted_dataset()
    state = fit_preprocessing_state(dataset, minimum_standard_deviation=0.25)
    np.testing.assert_allclose(state.state_mean, np.ones(38))
    np.testing.assert_allclose(state.action_mean, np.ones(8))
    assert state.state_observation_count == 2
    assert state.valid_action_step_count == 32

    majority, prevalence = fit_constant_baselines(dataset)
    assert majority.failure_probability == 0.0
    assert prevalence.failure_probability == 0.5
    action = fit_action_magnitude_baseline(dataset)
    np.testing.assert_allclose(action.action_mean, np.ones(8))


def test_masked_steps_do_not_affect_fit_and_normalize_to_zero() -> None:
    mask = np.ones(16, dtype=np.bool_)
    mask[-1] = False
    first = _example(0, 1.0, mask=mask)
    changed_actions = np.array(first.action_chunk, copy=True)
    changed_actions[-1] = 1_000_000.0
    second = ActionVerifierModelExampleV1(
        state_vector=first.state_vector,
        action_chunk=changed_actions,
        action_mask=mask,
        failure_target=0,
        sample_index=1,
    )
    state = preprocessing_module._fit_preprocessing_statistics(
        (first, second),
        dataset_digest=_digest("a"),
        training_split_digest=_digest("b"),
        minimum_standard_deviation=0.5,
    )
    np.testing.assert_allclose(state.action_mean, np.ones(8))
    np.testing.assert_allclose(state.action_standard_deviation, np.full(8, 0.5))
    assert state.valid_action_step_count == 30
    normalized = normalize_example(second, state)
    np.testing.assert_array_equal(normalized.action_chunk[-1], np.zeros(8))
    assert not normalized.action_chunk.flags.writeable


def test_preprocessing_json_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    state = fit_preprocessing_state(_accepted_dataset())
    path = save_preprocessing_state(state, tmp_path / "preprocessing.json")
    loaded = load_preprocessing_state(path)
    assert loaded.content_digest == state.content_digest
    np.testing.assert_array_equal(loaded.state_mean, state.state_mean)
    assert not loaded.state_mean.flags.writeable

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state_mean"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PreprocessingError, match="content changed"):
        load_preprocessing_state(path)


def test_numpy_batch_contains_no_reporting_metadata() -> None:
    dataset = _accepted_dataset()
    state = fit_preprocessing_state(dataset)
    batch = collate_numpy_examples(
        tuple(
            dataset[index] for index in dataset.indices_for_split(DatasetSplit.TRAIN)
        ),
        preprocessing=state,
    )
    assert batch.state_vectors.shape == (2, 38)
    assert batch.action_chunks.shape == (2, 16, 8)
    assert batch.model_inputs() == (
        batch.state_vectors,
        batch.action_chunks,
        batch.action_masks,
    )
    assert not hasattr(batch, "candidate_type")
    assert not hasattr(batch, "source_trajectory")
