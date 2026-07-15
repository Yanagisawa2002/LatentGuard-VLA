from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latentguard.action_verifier import (
    ActionVerifierCandidateGroupV1,
    ActionVerifierDatasetV1,
    ActionVerifierSampleV1,
    ActionVerifierSerializationError,
    ActionVerifierValidationError,
    CandidateType,
    DatasetSplit,
    SplitLeakageError,
    TrajectorySplitAssignmentV1,
    TrajectorySplitSourceV1,
    UnsupportedActionVerifierSerializationVersionError,
    assign_trajectory_splits,
    compute_split_policy_identifier,
    load_action_verifier_dataset,
    save_action_verifier_dataset,
    validate_action_verifier_evidence_references,
    validate_no_split_leakage,
)
from latentguard.models import FailureEvent, LabelSource, LabelStrength


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _digest(label: str) -> str:
    return f"sha256:{_hex(label)}"


def _evidence_id(label: str) -> str:
    return f"evd-sha256-{_hex(label)}"


def _baseline_evidence_id(label: str) -> str:
    return f"mspc-anchor-baseline-{_hex(label)}"


def _proposal_id(label: str) -> str:
    return f"cap-sha256-{_hex(label)}"


def _sample(
    *,
    candidate_type: CandidateType,
    state_vector: np.ndarray | None = None,
    action_chunk: np.ndarray | None = None,
) -> ActionVerifierSampleV1:
    source = candidate_type is CandidateType.SOURCE
    state = np.arange(6, dtype=np.float32) if state_vector is None else state_vector
    if action_chunk is None:
        chunk = np.arange(64, dtype=np.float32).reshape(16, 4) / 100.0
        if not source:
            chunk = chunk.copy()
            chunk[0, 0] += np.float32(0.5)
    else:
        chunk = action_chunk
    return ActionVerifierSampleV1(
        anchor_id="mspc-anchor-0",
        source_trajectory_id="trajectory-0",
        source_seed=7,
        split_group_id="mspc-split-0",
        dataset_split=DatasetSplit.TRAIN,
        task_id="maniskill/PickCube-v1",
        instruction="Pick up the cube and place it at the goal.",
        state_content_digest=_digest("state-0"),
        state_vector_semantic="PickCubeVerifierStateV1",
        state_vector_schema_digest=_digest("state-schema"),
        state_vector=state,
        candidate_action_chunk=chunk,
        action_mask=np.ones(16, dtype=np.bool_),
        continuation_identity=_digest("continuation-0"),
        candidate_type=candidate_type,
        proposal_id=None if source else _proposal_id("proposal-0"),
        corruption_type=None if source else "windowed_constant_bias",
        severity_id=None if source else "moderate-bias-v1",
        final_task_success=source,
        progress_semantic="pickcube_official_normalized_dense_reward_v0",
        progress_before=0.25,
        progress_after_candidate_chunk=0.8 if source else 0.2,
        progress_delta=0.55 if source else -0.05,
        final_unsafe=False,
        failure_events=(
            ()
            if source
            else (
                FailureEvent(
                    failure_type="task_failure",
                    timestamp_s=1.0,
                    probability=1.0,
                ),
            )
        ),
        strong_simulator_evidence_id=(
            _baseline_evidence_id("baseline")
            if source
            else _evidence_id("counterfactual")
        ),
        evidence_dataset_digest=_digest("evidence-dataset"),
        source_dataset_digest=_digest("source-dataset"),
        corruption_dataset_digest=_digest("corruption-dataset"),
        compatibility_identity=_digest("compatibility"),
        adapter_version="1.0.0",
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
    )


def _dataset(
    *,
    source: ActionVerifierSampleV1 | None = None,
    corrupted: ActionVerifierSampleV1 | None = None,
    baseline_evidence_id: str | None = None,
) -> ActionVerifierDatasetV1:
    source_sample = (
        _sample(candidate_type=CandidateType.SOURCE) if source is None else source
    )
    corrupted_sample = (
        _sample(candidate_type=CandidateType.CORRUPTED)
        if corrupted is None
        else corrupted
    )
    group = ActionVerifierCandidateGroupV1(
        anchor_id=source_sample.anchor_id,
        source_trajectory_id=source_sample.source_trajectory_id,
        source_seed=source_sample.source_seed,
        split_group_id=source_sample.split_group_id,
        dataset_split=source_sample.dataset_split,
        task_id=source_sample.task_id,
        state_content_digest=source_sample.state_content_digest,
        state_vector_semantic=source_sample.state_vector_semantic,
        continuation_identity=source_sample.continuation_identity,
        source_sample_id=source_sample.sample_id,
        corrupted_sample_ids=(corrupted_sample.sample_id,),
        baseline_evidence_id=(
            source_sample.strong_simulator_evidence_id
            if baseline_evidence_id is None
            else baseline_evidence_id
        ),
    )
    split_counts = {
        DatasetSplit.TRAIN: 1,
        DatasetSplit.VALIDATION: 0,
        DatasetSplit.TEST: 0,
    }
    policy_id = compute_split_policy_identifier(
        source_trajectory_ids=(source_sample.source_trajectory_id,),
        split_counts=split_counts,
        split_seed=0,
    )
    assignment = TrajectorySplitAssignmentV1(
        split_policy_id=policy_id,
        source_trajectory_id=source_sample.source_trajectory_id,
        source_seed=source_sample.source_seed,
        split_group_id=source_sample.split_group_id,
        dataset_split=source_sample.dataset_split,
        state_digests=(source_sample.state_content_digest,),
        anchor_ids=(source_sample.anchor_id,),
        proposal_ids=(corrupted_sample.proposal_id or "unreachable",),
    )
    return ActionVerifierDatasetV1(
        state_vector_semantic=source_sample.state_vector_semantic,
        state_vector_schema_digest=source_sample.state_vector_schema_digest,
        state_vector_dimension=source_sample.state_vector.shape[0],
        action_dimension=source_sample.candidate_action_chunk.shape[1],
        split_policy_id=policy_id,
        samples=tuple(
            sorted((source_sample, corrupted_sample), key=lambda item: item.sample_id)
        ),
        candidate_groups=(group,),
        split_assignments=(assignment,),
    )


def _split_source(index: int) -> TrajectorySplitSourceV1:
    return TrajectorySplitSourceV1(
        source_trajectory_id=f"trajectory-{index:02d}",
        source_seed=index,
        split_group_id=f"split-group-{index:02d}",
        state_digests=(_digest(f"state-{index}"),),
        anchor_ids=(f"anchor-{index:02d}",),
        proposal_ids=(_proposal_id(f"proposal-{index}"),),
    )


def _assignment(label: str, split: DatasetSplit) -> TrajectorySplitAssignmentV1:
    return TrajectorySplitAssignmentV1(
        split_policy_id=f"avp-sha256-{_hex('policy')}",
        source_trajectory_id=f"trajectory-{label}",
        source_seed=1 if label == "a" else 2,
        split_group_id=f"group-{label}",
        dataset_split=split,
        state_digests=(_digest(f"state-{label}"),),
        anchor_ids=(f"anchor-{label}",),
        proposal_ids=(_proposal_id(f"proposal-{label}"),),
    )


def _read_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_sample_detaches_arrays_and_identity_binds_content() -> None:
    original_state = np.arange(6, dtype=np.float32)
    sample = _sample(
        candidate_type=CandidateType.SOURCE,
        state_vector=original_state,
    )
    original_state[0] = 99.0

    assert sample.state_vector[0] == 0.0
    assert not sample.state_vector.flags.writeable
    assert not sample.candidate_action_chunk.flags.writeable
    assert not sample.action_mask.flags.writeable

    changed_actions = sample.candidate_action_chunk.copy()
    changed_actions[0, 0] += np.float32(0.01)
    changed = replace(sample, candidate_action_chunk=changed_actions)
    assert changed.sample_id != sample.sample_id


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"action_mask": np.zeros(16, dtype=np.bool_)}, "mask of all true"),
        (
            {"candidate_action_chunk": np.zeros((15, 4), dtype=np.float32)},
            "candidate_action_chunk",
        ),
        ({"label_strength": LabelStrength.WEAK}, "require strong evidence"),
        ({"simulator_replay_verified": False}, "require verified simulator replay"),
        ({"final_task_success": False}, "successful remaining-trajectory baseline"),
        ({"progress_before": None}, "declared progress semantic requires"),
        ({"progress_delta": 0.7}, "must equal"),
    ),
)
def test_sample_validation_fails_closed(
    changes: dict[str, object], message: str
) -> None:
    sample = _sample(candidate_type=CandidateType.SOURCE)
    with pytest.raises(ActionVerifierValidationError, match=message):
        replace(sample, **changes)


def test_corrupted_sample_requires_complete_corruption_identity() -> None:
    sample = _sample(candidate_type=CandidateType.CORRUPTED)
    with pytest.raises(ActionVerifierValidationError, match="proposal_id"):
        replace(sample, proposal_id=None)


def test_dataset_group_binds_one_exact_state_and_baseline_evidence() -> None:
    source = _sample(candidate_type=CandidateType.SOURCE)
    changed_state = source.state_vector.copy()
    changed_state[0] += np.float32(1.0)
    corrupted = _sample(
        candidate_type=CandidateType.CORRUPTED,
        state_vector=changed_state,
    )

    with pytest.raises(ActionVerifierValidationError, match="same exact state vector"):
        _dataset(source=source, corrupted=corrupted)

    with pytest.raises(ActionVerifierValidationError, match="baseline_evidence_id"):
        _dataset(baseline_evidence_id=_baseline_evidence_id("wrong-baseline"))


def test_dataset_digest_is_deterministic_and_label_sensitive() -> None:
    first = _dataset()
    second = _dataset()
    assert first.content_digest == second.content_digest

    corrupted = _sample(candidate_type=CandidateType.CORRUPTED)
    changed = replace(corrupted, final_task_success=True, failure_events=())
    assert _dataset(corrupted=changed).content_digest != first.content_digest


def test_corrupted_sample_rejects_baseline_evidence_identifier() -> None:
    source = _sample(candidate_type=CandidateType.SOURCE)
    corrupted = _sample(candidate_type=CandidateType.CORRUPTED)
    with pytest.raises(
        ActionVerifierValidationError, match="content-derived identifier"
    ):
        replace(
            corrupted,
            strong_simulator_evidence_id=source.strong_simulator_evidence_id,
        )


def test_evidence_foreign_keys_are_checked_against_bound_inventory() -> None:
    dataset = _dataset()
    evidence_ids = tuple(
        sample.strong_simulator_evidence_id for sample in dataset.samples
    )
    validate_action_verifier_evidence_references(
        dataset,
        available_evidence_ids=evidence_ids,
        evidence_dataset_digest=dataset.samples[0].evidence_dataset_digest,
    )

    with pytest.raises(ActionVerifierValidationError, match="missing referenced"):
        validate_action_verifier_evidence_references(
            dataset,
            available_evidence_ids=evidence_ids[:1],
            evidence_dataset_digest=dataset.samples[0].evidence_dataset_digest,
        )
    with pytest.raises(ActionVerifierValidationError, match="does not match"):
        validate_action_verifier_evidence_references(
            dataset,
            available_evidence_ids=evidence_ids,
            evidence_dataset_digest=_digest("wrong-evidence-dataset"),
        )


def test_split_assignment_is_input_order_independent_with_smoke_quotas() -> None:
    sources = tuple(_split_source(index) for index in range(6))
    counts = {
        DatasetSplit.TRAIN: 4,
        DatasetSplit.VALIDATION: 1,
        DatasetSplit.TEST: 1,
    }
    forward = assign_trajectory_splits(sources, split_counts=counts, split_seed=123)
    reverse = assign_trajectory_splits(
        tuple(reversed(sources)), split_counts=counts, split_seed=123
    )

    assert forward == reverse
    assert Counter(item.dataset_split for item in forward) == Counter(counts)
    assert len({item.split_policy_id for item in forward}) == 1


def test_default_split_assignment_enforces_full_48_6_6_quota() -> None:
    assignments = assign_trajectory_splits(
        tuple(_split_source(index) for index in range(60)), split_seed=5
    )
    assert Counter(item.dataset_split for item in assignments) == Counter(
        {
            DatasetSplit.TRAIN: 48,
            DatasetSplit.VALIDATION: 6,
            DatasetSplit.TEST: 6,
        }
    )


def test_split_quota_must_exactly_cover_population() -> None:
    with pytest.raises(ActionVerifierValidationError, match="exactly cover"):
        assign_trajectory_splits(
            (_split_source(0),),
            split_counts={DatasetSplit.TRAIN: 2},
        )


@pytest.mark.parametrize(
    "category",
    (
        "source_trajectory_id",
        "source_seed",
        "state_digests",
        "anchor_ids",
        "proposal_ids",
        "split_group_id",
    ),
)
def test_six_hard_leakage_classes_are_rejected(category: str) -> None:
    train = _assignment("a", DatasetSplit.TRAIN)
    test = _assignment("b", DatasetSplit.TEST)
    test = replace(test, **{category: getattr(train, category)})

    with pytest.raises(SplitLeakageError, match="appears in"):
        validate_no_split_leakage((train, test))


def test_identical_state_tree_digest_across_distinct_trajectories_is_leakage() -> None:
    train = _assignment("source-a", DatasetSplit.TRAIN)
    test = replace(_assignment("source-b", DatasetSplit.TEST), source_seed=3)
    assert train.source_trajectory_id != test.source_trajectory_id
    assert train.source_seed != test.source_seed
    test = replace(test, state_digests=train.state_digests)

    with pytest.raises(SplitLeakageError, match="state_digest"):
        validate_no_split_leakage((train, test))


def test_serialization_round_trip_uses_three_compact_stacked_arrays(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    output = tmp_path / "dataset"
    manifest_path = save_action_verifier_dataset(dataset, output)
    loaded = load_action_verifier_dataset(output)
    manifest = _read_manifest(manifest_path)

    assert loaded.content_digest == dataset.content_digest
    assert manifest["dataset_content_digest"] == dataset.content_digest
    assert manifest["array_count"] == 3
    assert set((output / "arrays").iterdir()) == {
        output / "arrays" / "state_vectors.npy",
        output / "arrays" / "candidate_action_chunks.npy",
        output / "arrays" / "action_masks.npy",
    }
    with (output / "arrays" / "candidate_action_chunks.npy").open("rb") as stream:
        actions = np.load(stream, allow_pickle=False)
    assert actions.shape == (2, 16, 4)
    assert actions.dtype == np.dtype(np.float32)


def test_serialization_rejects_array_content_tampering(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    save_action_verifier_dataset(_dataset(), output)
    array_path = output / "arrays" / "state_vectors.npy"
    content = bytearray(array_path.read_bytes())
    content[-1] ^= 1
    array_path.write_bytes(content)

    with pytest.raises(ActionVerifierSerializationError, match="file content mismatch"):
        load_action_verifier_dataset(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dtype", ">f4", "dtype"),
        ("shape", [2, 7], "shape"),
        ("path", "../state_vectors.npy", "path"),
    ),
)
def test_serialization_rejects_array_metadata_tampering(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    output = tmp_path / "dataset"
    manifest_path = save_action_verifier_dataset(_dataset(), output)
    manifest = _read_manifest(manifest_path)
    arrays = manifest["arrays"]
    assert isinstance(arrays, dict)
    state_reference = arrays["state_vectors"]
    assert isinstance(state_reference, dict)
    state_reference[field] = value
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ActionVerifierSerializationError, match=message):
        load_action_verifier_dataset(output)


def test_serialization_rejects_extra_inventory_and_nonempty_destination(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dataset"
    save_action_verifier_dataset(_dataset(), output)
    (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ActionVerifierSerializationError, match="inventory"):
        load_action_verifier_dataset(output)

    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ActionVerifierSerializationError, match="absent or empty"):
        save_action_verifier_dataset(_dataset(), destination)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_serialization_rejects_duplicate_manifest_fields(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    manifest_path = save_action_verifier_dataset(_dataset(), output)
    payload = manifest_path.read_text(encoding="utf-8")
    duplicate = payload.replace(
        '  "action_dimension": 4,',
        '  "action_dimension": 4,\n  "action_dimension": 4,',
        1,
    )
    assert duplicate != payload
    manifest_path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ActionVerifierSerializationError, match="duplicate field"):
        load_action_verifier_dataset(output)


def test_serialization_rejects_unsupported_version(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    manifest_path = save_action_verifier_dataset(_dataset(), output)
    manifest = _read_manifest(manifest_path)
    manifest["serialization_version"] = 2
    _write_manifest(manifest_path, manifest)

    with pytest.raises(
        UnsupportedActionVerifierSerializationVersionError,
        match="unsupported version",
    ):
        load_action_verifier_dataset(output)


def test_serialization_wraps_logical_model_validation(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    manifest_path = save_action_verifier_dataset(_dataset(), output)
    manifest = _read_manifest(manifest_path)
    samples = manifest["samples"]
    assert isinstance(samples, list)
    source = next(
        item
        for item in samples
        if isinstance(item, dict) and item.get("candidate_type") == "source"
    )
    assert isinstance(source, dict)
    source["final_task_success"] = False
    _write_manifest(manifest_path, manifest)

    with pytest.raises(
        ActionVerifierSerializationError,
        match="successful remaining-trajectory baseline",
    ):
        load_action_verifier_dataset(output)


def test_serialization_rejects_hard_linked_arrays(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    save_action_verifier_dataset(_dataset(), output)
    array_path = output / "arrays" / "action_masks.npy"
    outside = tmp_path / "outside.npy"
    outside.write_bytes(array_path.read_bytes())
    array_path.unlink()
    try:
        os.link(outside, array_path)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    with pytest.raises(ActionVerifierSerializationError, match="hard links"):
        load_action_verifier_dataset(output)
