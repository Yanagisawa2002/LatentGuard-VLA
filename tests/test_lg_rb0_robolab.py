"""CPU-only tests for the LG-RB0 RoboLab replay contract."""

from __future__ import annotations

import numpy as np
import pytest

from latentguard.adapters.robolab.branch_runner import (
    BranchGateInput,
    evaluate_lg_rb1_gate,
)
from latentguard.adapters.robolab.replay_adapter import ReplayContract
from latentguard.adapters.robolab.state_schema import (
    compare_state_trees,
    state_tree_sha256,
)


def test_state_tree_comparison_is_complete_and_strict() -> None:
    expected = {
        "articulation": {
            "robot": {"joint_position": np.asarray([[0.0, 1.0]], dtype=np.float32)}
        },
        "rigid_object": {
            "banana": {"root_pose": np.asarray([[1.0, 2.0]], dtype=np.float32)}
        },
    }
    observed = {
        "articulation": {
            "robot": {"joint_position": np.asarray([[0.0, 1.0]], dtype=np.float32)}
        },
        "rigid_object": {
            "banana": {"root_pose": np.asarray([[1.0, 2.0 + 5e-7]], dtype=np.float32)}
        },
    }
    comparison = compare_state_trees(expected, observed, tolerance=1e-6)
    assert comparison.matches
    assert comparison.maximum_absolute_error > 0
    assert state_tree_sha256(expected) != state_tree_sha256(observed)


def test_state_tree_comparison_rejects_missing_dtype_and_nonfinite() -> None:
    expected = {
        "a": np.asarray([1.0], dtype=np.float32),
        "b": np.asarray([2.0], dtype=np.float32),
        "c": np.asarray([3.0], dtype=np.float32),
    }
    observed = {
        "a": np.asarray([1.0], dtype=np.float64),
        "c": np.asarray([np.nan], dtype=np.float32),
        "d": np.asarray([4.0], dtype=np.float32),
    }
    comparison = compare_state_trees(expected, observed, tolerance=1e-6)
    assert not comparison.matches
    assert comparison.missing_paths == ("b",)
    assert comparison.unexpected_paths == ("d",)
    assert comparison.dtype_mismatches == ("a",)
    assert comparison.non_finite_paths == ("c",)


def test_empty_state_categories_remain_structurally_bound() -> None:
    expected = {"articulation": {}, "rigid_object": {"a": np.asarray([1.0])}}
    observed = {"rigid_object": {"a": np.asarray([1.0])}}
    comparison = compare_state_trees(expected, observed, tolerance=0.0)
    assert not comparison.matches
    assert comparison.missing_paths == ("articulation/<empty-mapping>",)


def test_replay_contract_is_frozen() -> None:
    contract = ReplayContract(
        official_state_tolerance=0.01,
        takeover_state_tolerance=1e-6,
        repeats=3,
        branch_checkpoints=(1, 5, 10),
        anchor_fractions=(0.25, 0.5, 0.75),
    )
    contract.validate()
    with pytest.raises(ValueError, match="official RoboLab"):
        ReplayContract(
            official_state_tolerance=0.02,
            takeover_state_tolerance=1e-6,
            repeats=3,
            branch_checkpoints=(1, 5, 10),
            anchor_fractions=(0.25, 0.5, 0.75),
        ).validate()


def _gate(**overrides: object) -> BranchGateInput:
    values: dict[str, object] = {
        "valid_recorded_episodes": 10,
        "faithful_initial_restore_failures": 0,
        "faithful_per_step_failures": 0,
        "faithful_terminal_mismatches": 0,
        "faithful_success_mismatches": 0,
        "prefix_mismatches": 0,
        "branch_mismatches": 0,
        "isolation_mismatches": 0,
        "semantic_coverage_complete": True,
        "official_stack_valid": True,
        "source_validation_passed": True,
        "remote_audit_passed": True,
    }
    values.update(overrides)
    return BranchGateInput(**values)  # type: ignore[arg-type]


def test_gate_distinguishes_result_a_b_and_c() -> None:
    accepted = evaluate_lg_rb1_gate(_gate())
    takeover_failed = evaluate_lg_rb1_gate(_gate(prefix_mismatches=1))
    replay_failed = evaluate_lg_rb1_gate(_gate(faithful_per_step_failures=1))
    assert (accepted["result"], accepted["LG_RB1_AUTHORIZED"]) == ("A", True)
    assert (takeover_failed["result"], takeover_failed["LG_RB1_AUTHORIZED"]) == (
        "B",
        False,
    )
    assert (replay_failed["result"], replay_failed["LG_RB1_AUTHORIZED"]) == (
        "C",
        False,
    )
