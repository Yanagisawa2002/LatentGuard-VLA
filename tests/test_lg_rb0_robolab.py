"""CPU-only tests for the LG-RB0 RoboLab replay contract."""

from __future__ import annotations

import numpy as np
import pytest

from latentguard.adapters.robolab.branch_runner import (
    BranchGateInput,
    evaluate_lg_rb1_gate,
)
from latentguard.adapters.robolab.orchestration import (
    merge_faithful_results,
    merge_recording_manifests,
    merge_takeover_results,
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
        "faithful_expected_replays": 30,
        "faithful_completed_replays": 30,
        "faithful_execution_errors": 0,
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
    replay_error = evaluate_lg_rb1_gate(
        _gate(
            faithful_completed_replays=0,
            faithful_execution_errors=30,
            prefix_mismatches=None,
            branch_mismatches=None,
            isolation_mismatches=None,
            semantic_coverage_complete=None,
        )
    )
    assert (accepted["result"], accepted["LG_RB1_AUTHORIZED"]) == ("A", True)
    assert (takeover_failed["result"], takeover_failed["LG_RB1_AUTHORIZED"]) == (
        "B",
        False,
    )
    assert (replay_failed["result"], replay_failed["LG_RB1_AUTHORIZED"]) == (
        "C",
        False,
    )
    assert (replay_error["result"], replay_error["LG_RB1_AUTHORIZED"]) == (
        "C",
        False,
    )


def test_process_isolated_recording_and_faithful_merges() -> None:
    recording_shards = []
    faithful_shards = []
    for index in range(2):
        recording_shards.append(
            {
                "status": "pass",
                "controller_kind": "deterministic_fixed_mechanics_probe",
                "policy_or_candidate_source": False,
                "episode_count": 1,
                "valid_episode_count": 1,
                "recordings": [{"recording_id": f"episode-{index}", "valid": True}],
                "raw_recordings_in_git": False,
                "training_performed": False,
                "final_seeds_accessed": False,
            }
        )
        faithful_shards.append(
            {
                "status": "pass",
                "episode_count": 1,
                "repeats_per_episode": 3,
                "official_state_tolerance": 0.01,
                "initial_restore_failures": 0,
                "per_step_state_failures": 0,
                "terminal_mismatches": 0,
                "success_mismatches": 0,
                "expected_replay_count": 3,
                "completed_replay_count": 3,
                "execution_error_count": 0,
                "details": [{"repeat": repeat} for repeat in range(3)],
            }
        )
    recording = merge_recording_manifests(
        recording_shards,
        expected_episode_count=2,
    )
    faithful = merge_faithful_results(
        faithful_shards,
        expected_episode_count=2,
    )
    assert recording["status"] == faithful["status"] == "pass"
    assert recording["episode_count"] == faithful["episode_count"] == 2
    assert len(faithful["details"]) == 6
    faithful_shards[0]["status"] = "fail"
    faithful_shards[0]["completed_replay_count"] = 0
    faithful_shards[0]["execution_error_count"] = 3
    failed = merge_faithful_results(
        faithful_shards,
        expected_episode_count=2,
    )
    assert failed["status"] == "fail"
    assert failed["completed_replay_count"] == 3
    assert failed["execution_error_count"] == 3


def test_process_isolated_takeover_merge_preserves_failure() -> None:
    prefix = [
        {
            "state_tolerance": 1e-6,
            "anchors_per_episode": 3,
            "repeats_per_anchor": 3,
            "mismatch_count": 1,
            "semantic_coverage_complete": True,
            "details": [{"anchor": index} for index in range(3)],
        }
    ]
    branch = [
        {
            "state_tolerance": 1e-6,
            "branches": ["A", "B"],
            "checkpoints": [1, 5, 10],
            "mismatch_count": 0,
            "semantic_coverage_complete": True,
            "details": [{"branch": index} for index in range(6)],
        }
    ]
    isolation = [
        {
            "orders": ["A-B-A", "B-A-B"],
            "mismatch_count": 0,
            "semantic_coverage_complete": True,
            "details": [{"order": index} for index in range(6)],
        }
    ]
    merged = merge_takeover_results(
        prefix,
        branch,
        isolation,
        expected_episode_count=1,
    )
    assert merged["prefix_replay_validation"]["status"] == "fail"
    assert merged["branch_determinism_validation"]["status"] == "pass"
    assert merged["branch_isolation_validation"]["status"] == "pass"
