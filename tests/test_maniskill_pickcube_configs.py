from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from latentguard.corruptions.config import load_corruption_plan
from latentguard.integrations.maniskill_pickcube.compatibility import (
    load_compatibility_report,
    validate_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
    validate_maniskill_pickcube_action_layout_binding,
)
from latentguard.models import ActionChunk

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIRECTORY = _ROOT / "configs" / "integrations" / "maniskill_pickcube"
_ACTION_LAYOUT = _CONFIG_DIRECTORY / "action-layout-v1.json"
_CORRUPTION_PLAN = _CONFIG_DIRECTORY / "corruptions-v1.json"
_EXPECTED_CONTRACT = _CONFIG_DIRECTORY / "expected-contract-v1.json"
_COMPATIBILITY_REPORT = (
    _ROOT
    / "reports"
    / "m2c"
    / "20260715T073250Z_m2c-pickcube-compat_bb2c35c_seed0_numeric-v1"
    / "compatibility-discovery.json"
)


def _all_mapping_keys(value: object) -> set[str]:
    """Collect every JSON object key from one already parsed document."""
    if isinstance(value, dict):
        item = cast(dict[str, object], value)
        return set(item).union(*(_all_mapping_keys(child) for child in item.values()))
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(child) for child in value))
    return set()


def test_real_configs_bind_exactly_to_retrieved_compatibility_report() -> None:
    report = load_compatibility_report(_COMPATIBILITY_REPORT)
    expected = load_expected_contract(_EXPECTED_CONTRACT)
    assert expected.trusted_replay_ready
    binding = validate_compatibility_report(report, expected, require_trusted=True)
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
