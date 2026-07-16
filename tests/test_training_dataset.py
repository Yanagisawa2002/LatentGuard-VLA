"""CPU-only tests for the strict M3B accepted-dataset boundary."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import latentguard.training.dataset as dataset_module
from latentguard.action_verifier import (
    ActionVerifierCandidateGroupV1,
    ActionVerifierDatasetV1,
    ActionVerifierSampleV1,
    CandidateType,
    DatasetSplit,
    TrajectorySplitAssignmentV1,
)
from latentguard.models import LabelSource, LabelStrength
from latentguard.training.dataset import (
    ActionVerifierModelExampleV1,
    TrainingDatasetError,
    load_accepted_action_verifier_dataset,
    validate_full_target_report,
)


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _digest(label: str) -> str:
    return f"sha256:{_hex(label)}"


def _sample(*, corrupted: bool, success: bool) -> ActionVerifierSampleV1:
    suffix = "corrupted" if corrupted else "source"
    return ActionVerifierSampleV1(
        anchor_id="anchor-0",
        source_trajectory_id="trajectory-0",
        source_seed=7,
        split_group_id="split-group-0",
        dataset_split=DatasetSplit.TRAIN,
        task_id="PickCube-v1",
        instruction="pick cube",
        state_content_digest=_digest("state"),
        state_vector_semantic="PickCubeVerifierStateV1",
        state_vector_schema_digest=_digest("state-schema"),
        state_vector=np.arange(38, dtype=np.float32),
        candidate_action_chunk=np.full((16, 8), 0.25, dtype=np.float64),
        action_mask=np.ones(16, dtype=np.bool_),
        continuation_identity=_digest("continuation"),
        candidate_type=(CandidateType.CORRUPTED if corrupted else CandidateType.SOURCE),
        proposal_id="proposal-0" if corrupted else None,
        corruption_type="constant_bias" if corrupted else None,
        severity_id="severe" if corrupted else None,
        final_task_success=success,
        progress_semantic="pickcube_binary_completion_v0",
        progress_before=0.0,
        progress_after_candidate_chunk=1.0 if success else 0.0,
        progress_delta=1.0 if success else 0.0,
        final_unsafe=False,
        failure_events=(),
        strong_simulator_evidence_id=(
            f"evd-sha256-{_hex(suffix)}"
            if corrupted
            else f"mspc-anchor-baseline-{_hex(suffix)}"
        ),
        evidence_dataset_digest=_digest("evidence"),
        source_dataset_digest=_digest("source-dataset"),
        corruption_dataset_digest=_digest("corruption-dataset"),
        compatibility_identity=_digest("compatibility"),
        adapter_version="1.0.0",
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
    )


def _small_dataset() -> ActionVerifierDatasetV1:
    source = _sample(corrupted=False, success=True)
    corrupted = _sample(corrupted=True, success=False)
    group = ActionVerifierCandidateGroupV1(
        anchor_id="anchor-0",
        source_trajectory_id="trajectory-0",
        source_seed=7,
        split_group_id="split-group-0",
        dataset_split=DatasetSplit.TRAIN,
        task_id="PickCube-v1",
        state_content_digest=_digest("state"),
        state_vector_semantic="PickCubeVerifierStateV1",
        continuation_identity=_digest("continuation"),
        source_sample_id=source.sample_id,
        corrupted_sample_ids=(corrupted.sample_id,),
        baseline_evidence_id=source.strong_simulator_evidence_id,
    )
    policy = f"avp-sha256-{_hex('policy')}"
    assignment = TrajectorySplitAssignmentV1(
        split_policy_id=policy,
        source_trajectory_id="trajectory-0",
        source_seed=7,
        split_group_id="split-group-0",
        dataset_split=DatasetSplit.TRAIN,
        state_digests=(_digest("state"),),
        anchor_ids=("anchor-0",),
        proposal_ids=("proposal-0",),
    )
    return ActionVerifierDatasetV1(
        state_vector_semantic="PickCubeVerifierStateV1",
        state_vector_schema_digest=_digest("state-schema"),
        state_vector_dimension=38,
        action_dimension=8,
        split_policy_id=policy,
        samples=tuple(sorted((source, corrupted), key=lambda item: item.sample_id)),
        candidate_groups=(group,),
        split_assignments=(assignment,),
    )


def _full_report(dataset_digest: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "validation_scope": "serialized_reload",
        "valid": True,
        "full_target_required": True,
        "full_target_valid": True,
        "leakage_valid": True,
        "evidence_references_valid": True,
        "continuation_integrity_valid": True,
        "resume_idempotence_valid": True,
        "verifier_state_restoration_valid": True,
        "anchor_manifest_content_digest": _digest("anchor-manifest"),
        "dataset_content_digest": dataset_digest,
        "sample_count": 3240,
        "candidate_group_count": 360,
        "source_trajectory_count": 60,
        "source_sample_count": 360,
        "corrupted_sample_count": 2880,
        "corrupted_success_count": 2289,
        "corrupted_failure_count": 591,
        "baseline_success_count": 360,
        "evidence_reference_count": 3240,
        "continuation_proposal_count": 2880,
        "restoration_evidence_count": 2880,
        "evaluation_execution_error_count": 0,
        "evaluation_indeterminate_count": 0,
        "evaluation_invalid_count": 0,
        "state_vector_dimension": 38,
        "state_vector_semantic": "PickCubeVerifierStateV1",
        "chunk_horizon": 16,
        "action_dimension": 8,
        "progress_semantic": "pickcube_binary_completion_v0",
        "split_trajectory_counts": {"train": 48, "validation": 6, "test": 6},
        "split_sample_counts": {"train": 2592, "validation": 324, "test": 324},
        "restoration_maximum_absolute_error": 1.1920928955078125e-7,
        "verifier_state_restoration_maximum_absolute_error": 0.0,
    }


def test_full_target_report_is_content_bound_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    expected = _digest("accepted-dataset")
    report_path = tmp_path / "validation-report.json"
    report_path.write_text(
        json.dumps(_full_report(expected), sort_keys=True), encoding="utf-8"
    )
    report_digest = validate_full_target_report(
        report_path,
        expected_dataset_digest=expected,
        expected_anchor_manifest_digest=_digest("anchor-manifest"),
    )
    assert report_digest.startswith("sha256:")

    payload = _full_report(expected)
    payload["full_target_valid"] = False
    report_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(TrainingDatasetError, match="full_target_valid"):
        validate_full_target_report(report_path, expected_dataset_digest=expected)

    report_path.write_text(
        json.dumps(_full_report(expected), sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(TrainingDatasetError, match="anchor_manifest_content_digest"):
        validate_full_target_report(
            report_path,
            expected_dataset_digest=expected,
            expected_anchor_manifest_digest=_digest("different-anchor-manifest"),
        )


def test_loader_projects_exact_allowlist_and_keeps_reporting_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _small_dataset()
    monkeypatch.setattr(
        dataset_module, "load_action_verifier_dataset", lambda _: source
    )
    monkeypatch.setattr(
        dataset_module, "_validate_accepted_dataset", lambda *a, **k: None
    )
    monkeypatch.setattr(
        dataset_module,
        "validate_full_target_report",
        lambda *a, **k: _digest("report"),
    )
    loaded = load_accepted_action_verifier_dataset(
        tmp_path,
        full_target_report=tmp_path / "unused.json",
        expected_dataset_digest=source.content_digest,
        anchor_selection_reasons={"anchor-0": "early_trajectory"},
        anchor_manifest_content_digest=_digest("anchor-manifest"),
    )

    assert len(loaded) == 2
    assert {
        field.name for field in dataclasses.fields(ActionVerifierModelExampleV1)
    } == {
        "state_vector",
        "action_chunk",
        "action_mask",
        "failure_target",
        "sample_index",
    }
    assert not loaded[0].state_vector.flags.writeable
    assert not loaded[0].action_chunk.flags.writeable
    assert loaded.reporting_for(0).anchor_selection_reason == "early_trajectory"
    assert loaded.reporting_for(0).sample_id.startswith("avs-sha256-")
    assert loaded.reporting_for(0).group_id.startswith("avg-sha256-")
    assert loaded.reporting_for(0).anchor_id == "anchor-0"
    assert not hasattr(loaded[0], "candidate_type")
    assert not hasattr(loaded[0], "sample_id")
    assert not hasattr(loaded[0], "source_trajectory_id")
    assert loaded.split_digest.startswith("sha256:")
    assert loaded.training_split_digest.startswith("sha256:")


def test_loader_rejects_partial_anchor_reporting_join(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _small_dataset()
    monkeypatch.setattr(
        dataset_module, "load_action_verifier_dataset", lambda _: source
    )
    monkeypatch.setattr(
        dataset_module, "_validate_accepted_dataset", lambda *a, **k: None
    )
    monkeypatch.setattr(
        dataset_module,
        "validate_full_target_report",
        lambda *a, **k: _digest("report"),
    )
    with pytest.raises(TrainingDatasetError, match="coverage differs"):
        load_accepted_action_verifier_dataset(
            tmp_path,
            full_target_report=tmp_path / "unused.json",
            expected_dataset_digest=source.content_digest,
            anchor_selection_reasons={},
            anchor_manifest_content_digest=_digest("anchor-manifest"),
        )


def test_dataset_digest_gate_precedes_fixed_inventory_acceptance() -> None:
    """A content mismatch is rejected as identity drift, not count drift."""

    with pytest.raises(TrainingDatasetError, match="unexpected dataset content"):
        dataset_module._validate_accepted_dataset(
            _small_dataset(), expected_dataset_digest=_digest("different-dataset")
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "state_vector",
            np.zeros(38, dtype=np.float64),
            "expected dtype <f4",
        ),
        (
            "action_chunk",
            np.zeros((16, 8), dtype=np.float32),
            "expected dtype <f8",
        ),
        (
            "action_chunk",
            np.zeros((15, 8), dtype=np.float64),
            "expected shape",
        ),
        (
            "action_mask",
            np.ones(16, dtype=np.uint8),
            "expected dtype \\|b1",
        ),
        (
            "state_vector",
            np.full(38, np.nan, dtype=np.float32),
            "all values must be finite",
        ),
    ),
)
def test_model_boundary_rejects_dtype_shape_and_nonfinite_drift(
    field: str, value: np.ndarray, message: str
) -> None:
    """Deployable arrays cross the boundary only under the exact contract."""

    values = {
        "state_vector": np.zeros(38, dtype=np.float32),
        "action_chunk": np.zeros((16, 8), dtype=np.float64),
        "action_mask": np.ones(16, dtype=np.bool_),
        "failure_target": 0,
        "sample_index": 0,
    }
    values[field] = value
    with pytest.raises(TrainingDatasetError, match=message):
        ActionVerifierModelExampleV1(**values)


def test_model_boundary_detaches_caller_owned_arrays() -> None:
    """Later caller mutation cannot alter an accepted training example."""

    state = np.arange(38, dtype=np.float32)
    action = np.ones((16, 8), dtype=np.float64)
    mask = np.ones(16, dtype=np.bool_)
    example = ActionVerifierModelExampleV1(
        state_vector=state,
        action_chunk=action,
        action_mask=mask,
        failure_target=1,
        sample_index=0,
    )
    state[:] = -1.0
    action[:] = -1.0
    mask[:] = False

    assert np.array_equal(example.state_vector, np.arange(38, dtype=np.float32))
    assert bool(np.all(example.action_chunk == 1.0))
    assert bool(np.all(example.action_mask))
