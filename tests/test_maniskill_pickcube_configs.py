from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from latentguard.corruptions.config import load_corruption_plan
from latentguard.evaluation.models import EvaluationStatus
from latentguard.evaluation.serialization import (
    MANIFEST_NAME as EVALUATION_MANIFEST_NAME,
)
from latentguard.evaluation.serialization import (
    EvaluationDataset,
    load_evaluation_dataset,
)
from latentguard.integrations.maniskill_pickcube.compatibility import (
    load_compatibility_report,
    validate_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
    validate_maniskill_pickcube_action_layout_binding,
)
from latentguard.integrations.maniskill_pickcube.serialization import (
    action_contract_from_compatibility,
)
from latentguard.models import ActionChunk, LabelSource, LabelStrength

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIRECTORY = _ROOT / "configs" / "integrations" / "maniskill_pickcube"
_ACTION_LAYOUT = _CONFIG_DIRECTORY / "action-layout-v1.json"
_CORRUPTION_PLAN = _CONFIG_DIRECTORY / "corruptions-v1.json"
_M3A_CORRUPTION_PLAN = _CONFIG_DIRECTORY / "m3a-corruptions-v1.json"
_EXPECTED_CONTRACT = _CONFIG_DIRECTORY / "expected-contract-v1.json"
_REPORT_ROOT = _ROOT / "reports" / "m2c"
_COMPATIBILITY_REPORT = (
    _REPORT_ROOT
    / "20260715T080753Z_m2c-pickcube-trusted_aafe838_seed0"
    / "compatibility-trusted.json"
)
_GATE_EVALUATION_MANIFEST = (
    _REPORT_ROOT
    / "20260715T080829Z_m2c-pickcube-replay1-gate_aafe838_seed271828"
    / "evaluation-manifest.json"
)
_FINAL_EVALUATION_MANIFEST = (
    _REPORT_ROOT
    / "20260715T080910Z_m2c-pickcube-replay12_aafe838_seed271828"
    / "evaluation-manifest.json"
)
_FINAL_REPLAY_SUMMARY = (
    _REPORT_ROOT
    / "20260715T080910Z_m2c-pickcube-replay12_aafe838_seed271828"
    / "replay-summary.json"
)


def _all_mapping_keys(value: object) -> set[str]:
    """Collect every JSON object key from one already parsed document."""
    if isinstance(value, dict):
        item = cast(dict[str, object], value)
        return set(item).union(*(_all_mapping_keys(child) for child in item.values()))
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(child) for child in value))
    return set()


def _load_retrieved_evaluation_manifest(
    manifest_path: Path, tmp_path: Path
) -> EvaluationDataset:
    """Load a retrieved, renamed manifest through the strict bundle loader."""
    bundle_root = tmp_path / manifest_path.parent.name
    bundle_root.mkdir()
    (bundle_root / EVALUATION_MANIFEST_NAME).write_bytes(manifest_path.read_bytes())
    return load_evaluation_dataset(bundle_root)


def test_real_configs_bind_exactly_to_retrieved_compatibility_report() -> None:
    report = load_compatibility_report(_COMPATIBILITY_REPORT)
    expected = load_expected_contract(_EXPECTED_CONTRACT)
    assert expected.trusted_replay_ready
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    assert binding.trusted_replay_ready
    layout = load_maniskill_pickcube_action_layout(_ACTION_LAYOUT)
    plan = load_corruption_plan(_CORRUPTION_PLAN)

    layout_digest = validate_maniskill_pickcube_action_layout_binding(
        layout,
        binding,
        plan.action_layout,
    )

    assert report.schema_version == "1.1"
    assert report.state_round_trip.passed
    assert layout_digest == layout.action_layout_digest
    assert layout.action_dim == report.action_space.action_dimension == 8
    assert layout.action_contract_digest == report.action_contract_digest
    assert layout.control_mode == report.control_mode == "pd_joint_pos"
    assert layout.control_period_s == report.control_period_s == pytest.approx(0.05)
    assert layout.coordinate_frame == "unspecified"
    assert tuple((field.name, field.indices) for field in layout.fields) == (
        ("arm", tuple(range(7))),
        ("gripper", (7,)),
    )
    assert all(field.semantic.value == "unspecified" for field in layout.fields)
    assert all(field.units == "unspecified" for field in layout.fields)
    assert tuple(
        field.metadata["controller_component_identity"] for field in layout.fields
    ) == tuple(
        component.configuration_identity for component in report.controller.components
    )
    assert dict(layout.metadata) == {
        "action_contract_digest": report.action_contract_digest,
        "compatibility_identity": report.compatibility_identity,
        "compatibility_report_schema_version": report.schema_version,
        "controller_configuration_identity": (report.controller.configuration_identity),
        "source_solver_identity": report.source_solver.source_sha256,
        "state_tree_structure_digest": report.state_tree_structure_digest,
        "state_verification_semantic": (report.state_round_trip.comparison_semantic),
        "state_verification_tolerance": report.state_round_trip.tolerance,
        "task_implementation_identity": report.task_implementation.source_sha256,
    }


def test_m3a_segment_zeroing_is_maximal_and_action_contract_valid() -> None:
    report = load_compatibility_report(_COMPATIBILITY_REPORT)
    expected = load_expected_contract(_EXPECTED_CONTRACT)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(_ACTION_LAYOUT)
    plan = load_corruption_plan(_M3A_CORRUPTION_PLAN)
    zeroing = plan.corruptions[-1]
    dtype = np.dtype(report.action_space.dtype)
    lower = np.asarray(report.action_space.lower_bounds, dtype=dtype)
    upper = np.asarray(report.action_space.upper_bounds, dtype=dtype)
    source = ActionChunk(
        actions=np.broadcast_to((lower + upper) / 2.0, (16, lower.shape[0])).copy(),
        coordinate_frame=layout.coordinate_frame,
        control_period_s=report.control_period_s,
    )

    parameters = zeroing.resolved_parameters_for(source, plan.action_layout)
    targets = cast(tuple[int, ...], parameters["target_indices"])
    assert zeroing.name == "segment_zeroing"
    assert parameters["severity_id"] == "severe_contract_safe_zero_0_16"
    assert targets == (0, 1, 2, 4, 5, 6, 7)
    assert 3 not in targets
    assert upper[3] < 0.0
    assert all(lower[index] <= 0.0 <= upper[index] for index in targets)

    transformed = zeroing.apply(source, plan.action_layout, seed=0)
    contract = action_contract_from_compatibility(
        binding,
        coordinate_frame=layout.coordinate_frame,
    )
    contract.validate_action_chunk(transformed, role="M3A segment-zeroing regression")


def test_real_corruption_plan_has_three_unrepaired_m1_transformations() -> None:
    document = json.loads(_CORRUPTION_PLAN.read_text(encoding="utf-8"))
    plan = load_corruption_plan(_CORRUPTION_PLAN)
    source_values = np.arange(6 * 8, dtype=np.float32).reshape(6, 8) / 100.0
    source = ActionChunk(
        actions=source_values,
        coordinate_frame="unspecified",
        control_period_s=0.05,
    )
    source_snapshot = source.actions.tobytes(order="C")

    assert tuple(corruption.name for corruption in plan.corruptions) == (
        "additive_gaussian_noise",
        "temporal_field_shift",
        "segment_hold",
    )
    gaussian, shift, hold = plan.corruptions
    gaussian_parameters = gaussian.resolved_parameters_for(source, plan.action_layout)
    shift_parameters = shift.resolved_parameters_for(source, plan.action_layout)
    hold_parameters = hold.resolved_parameters_for(source, plan.action_layout)

    assert gaussian_parameters == {
        "target_indices": tuple(range(7)),
        "standard_deviation": 0.005,
        "mean": 0.0,
    }
    assert shift_parameters == {
        "target_field": "arm",
        "target_indices": tuple(range(7)),
        "shift_steps": 1,
        "fill_policy": "edge",
    }
    assert hold_parameters == {
        "start_step": 1,
        "end_step": source.actions.shape[0],
        "target_indices": tuple(range(8)),
    }
    raw_corruptions = cast(dict[str, object], document)["corruptions"]
    assert isinstance(raw_corruptions, list)
    hold_document = cast(dict[str, object], raw_corruptions[2])
    assert hold_document["parameters"] == {
        "start_step": 1,
        "target_indices": list(range(8)),
    }
    assert not _all_mapping_keys(document).intersection(
        {"clip", "clipping", "normalize", "normalization", "repair", "retarget"}
    )

    gaussian_result = gaussian.apply(source, plan.action_layout, seed=0)
    shift_result = shift.apply(source, plan.action_layout, seed=0)
    hold_result = hold.apply(source, plan.action_layout, seed=0)

    for result in (gaussian_result, shift_result, hold_result):
        assert result.actions.shape == source.actions.shape
        assert result.actions.dtype == source.actions.dtype
    assert np.array_equal(gaussian_result.actions[:, 7], source.actions[:, 7])
    assert not np.array_equal(gaussian_result.actions[:, :7], source.actions[:, :7])
    assert np.array_equal(shift_result.actions[0, :7], source.actions[0, :7])
    assert np.array_equal(shift_result.actions[1:, :7], source.actions[:-1, :7])
    assert np.array_equal(shift_result.actions[:, 7], source.actions[:, 7])
    assert np.array_equal(hold_result.actions[0], source.actions[0])
    assert np.array_equal(
        hold_result.actions[1:],
        np.repeat(source.actions[[0]], source.actions.shape[0] - 1, axis=0),
    )
    assert source.actions.tobytes(order="C") == source_snapshot


def test_retrieved_replay_gate_is_conclusive_without_execution_errors(
    tmp_path: Path,
) -> None:
    dataset = _load_retrieved_evaluation_manifest(
        _GATE_EVALUATION_MANIFEST,
        tmp_path,
    )

    assert dataset.summary.conclusive == 1
    assert dataset.summary.execution_error == 0
    assert dataset.summary.total_attempts == 1


def test_retrieved_final_replay_evidence_and_summary_are_consistent(
    tmp_path: Path,
) -> None:
    dataset = _load_retrieved_evaluation_manifest(
        _FINAL_EVALUATION_MANIFEST,
        tmp_path,
    )
    configuration = dataset.resolved_evaluator_configuration

    assert dataset.summary.conclusive == 12
    assert dataset.summary.execution_error == 0
    assert dataset.summary.total_attempts == 12
    assert sum(evidence.success is True for evidence in dataset.evidence) == 8
    assert sum(evidence.success is False for evidence in dataset.evidence) == 4
    assert configuration["adapter_version"] == "1.1.1"
    assert all(
        evidence.status is EvaluationStatus.CONCLUSIVE
        and evidence.label_source is LabelSource.SIMULATOR
        and evidence.label_strength is LabelStrength.STRONG
        and evidence.simulator_replay_verified
        and evidence.metrics["replay_baseline_valid"] is True
        and evidence.metrics["replay_baseline_restoration_complete_state_comparison"]
        is True
        and evidence.metrics["replay_corrupted_restoration_complete_state_comparison"]
        is True
        and evidence.metrics["replay_baseline_restoration_compared_component_count"]
        == 70
        and evidence.metrics["replay_corrupted_restoration_compared_component_count"]
        == 70
        and cast(
            float,
            evidence.metrics["replay_baseline_restoration_maximum_absolute_error"],
        )
        <= 1e-6
        and cast(
            float,
            evidence.metrics["replay_corrupted_restoration_maximum_absolute_error"],
        )
        <= 1e-6
        for evidence in dataset.evidence
    )

    summary = cast(
        dict[str, object],
        json.loads(_FINAL_REPLAY_SUMMARY.read_text(encoding="utf-8")),
    )
    assert summary["adapter_version"] == "1.1.1"
    assert summary["selected_proposal_count"] == dataset.summary.total_attempts
    assert summary["conclusive_success_count"] == 8
    assert summary["conclusive_task_failure_count"] == 4
    assert summary["execution_error_count"] == dataset.summary.execution_error
    assert summary["strong_simulator_verified_count"] == len(dataset.evidence)
    assert summary["valid_baseline_count"] == len(dataset.evidence)
    assert summary["per_corruption"] == {
        "additive_gaussian_noise": {
            "conclusive_success": 4,
            "conclusive_task_failure": 0,
            "execution_error": 0,
            "other": 0,
            "selected": 4,
        },
        "segment_hold": {
            "conclusive_success": 0,
            "conclusive_task_failure": 4,
            "execution_error": 0,
            "other": 0,
            "selected": 4,
        },
        "temporal_field_shift": {
            "conclusive_success": 4,
            "conclusive_task_failure": 0,
            "execution_error": 0,
            "other": 0,
            "selected": 4,
        },
    }
    manifest_sha256 = (
        "sha256:" + hashlib.sha256(_FINAL_EVALUATION_MANIFEST.read_bytes()).hexdigest()
    )
    assert summary["resume_validation"] == {
        "evaluation_manifest_sha256": manifest_sha256,
        "manifest_unchanged": True,
        "resumed_without_rerun_count": 12,
    }
