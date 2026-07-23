from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

from latentguard.progress.lg_r1b import (
    FAILURE_TAXONOMY,
    FINAL_SEEDS,
    LGR2GateInput,
    PilotGateInput,
    build_phase_jobs,
    evaluate_lg_r2_gate,
    evaluate_pilot_gate,
    evaluate_standard_failure_yield,
    validate_registry,
)
from latentguard.progress.stages.libero_registry import LiberoStageRegistry

ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict[str, object]:
    payload = yaml.safe_load(
        (ROOT / "configs" / "lg_r1b" / "task_registry.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _entity(
    position: tuple[float, float, float],
    *,
    contact: bool = False,
    turn_on: bool = False,
    open_: bool = False,
) -> dict[str, object]:
    return {
        "position": list(position),
        "contact_with_gripper": contact,
        "turn_on": turn_on,
        "open": open_,
        "close": not open_,
    }


def test_lg_r1b_registry_is_frozen_before_any_outcome() -> None:
    registry = _registry()
    validation = validate_registry(registry)
    assert validation["task_count"] == 16
    assert validation["groups"] == ["primary", "secondary"]
    assert validation["phase_episode_counts"] == {
        "primary_expansion_1": 80,
        "primary_expansion_2": 80,
        "primary_expansion_3": 80,
        "primary_pilot": 80,
        "secondary_completion": 40,
        "secondary_pilot": 80,
    }
    assert not validation["historical_task_overlap"]
    assert not validation["historical_seed_overlap"]
    assert not validation["sealed_final_seed_overlap"]
    assert not validation["cross_experiment_seed_reuse"]


def test_required_primary_tasks_and_standard_horizons_are_exact() -> None:
    tasks = _registry()["tasks"]
    assert isinstance(tasks, list)
    primary = {
        (str(task["suite"]), int(task["task_id"])): task
        for task in tasks
        if task["group"] == "primary"
    }
    assert {("libero_10", task_id) for task_id in (2, 4, 8, 9)} <= set(primary)
    assert all(
        task["episode_horizon"] == (300 if task["suite"] == "libero_goal" else 520)
        for task in tasks
    )
    assert all(task["zero_shot_held_out"] is True for task in tasks)
    assert all(task["synthetic_failure_allowed"] is False for task in tasks)


def test_every_lg_r1b_seed_is_unique_and_final_range_is_sealed() -> None:
    registry = _registry()
    jobs = [
        job
        for phase in (
            "primary_pilot",
            "primary_expansion_1",
            "primary_expansion_2",
            "primary_expansion_3",
            "secondary_pilot",
            "secondary_completion",
        )
        for job in build_phase_jobs(registry, phase)
    ]
    seeds = [int(job["seed"]) for job in jobs]
    assert len(seeds) == 440
    assert len(set(seeds)) == len(seeds)
    assert not (set(seeds) & set(registry["historical_seeds"]))
    assert not (set(seeds) & FINAL_SEEDS)


@pytest.mark.parametrize(
    ("failures", "status", "action"),
    [
        (10, "PILOT_A", "uniform_primary_expansion"),
        (3, "PILOT_B", "uniform_primary_expansion_to_20_40_per_task"),
        (2, "PILOT_C", "activate_preregistered_secondary_group"),
    ],
)
def test_pilot_gate_uses_only_frozen_branches(
    failures: int,
    status: str,
    action: str,
) -> None:
    result = evaluate_pilot_gate(
        PilotGateInput(
            valid_episodes=80,
            expected_episodes=80,
            natural_failures=failures,
            infrastructure_errors=0,
        )
    )
    assert result["status"] == status
    assert result["next_action"] == action


def test_incomplete_pilot_never_selects_an_expansion_branch() -> None:
    result = evaluate_pilot_gate(
        PilotGateInput(
            valid_episodes=79,
            expected_episodes=80,
            natural_failures=20,
            infrastructure_errors=0,
        )
    )
    assert result["status"] == "PILOT_INCOMPLETE"
    assert result["next_action"] == "stop_and_repair_infrastructure"


def test_two_group_200_episode_low_yield_stops_more_libero() -> None:
    result = evaluate_standard_failure_yield(
        valid_episodes=200,
        task_groups=2,
        natural_failures=9,
    )
    assert result["status"] == "STANDARD_LIBERO_FAILURE_YIELD_TOO_LOW"
    assert result["stop_adding_libero_episodes"] is True
    keep_going = evaluate_standard_failure_yield(
        valid_episodes=200,
        task_groups=2,
        natural_failures=10,
    )
    assert keep_going["stop_adding_libero_episodes"] is False


def test_toggle_place_uses_control_and_object_predicates() -> None:
    registry = LiberoStageRegistry(_registry()["tasks"])
    adapter = registry.resolve("libero_10", 2)
    initial = {
        "eef_position": [0.5, 0.0, 0.0],
        "objects": {
            "flat_stove_1": _entity((0.0, 0.0, 0.0)),
            "moka_pot_1": _entity((0.3, 0.0, 0.0)),
            "flat_stove_1_cook_region": _entity((0.0, 0.0, 0.0)),
        },
        "goal_predicates": [
            {"predicate": ["turnon", "flat_stove_1"], "satisfied": False},
            {
                "predicate": [
                    "on",
                    "moka_pot_1",
                    "flat_stove_1_cook_region",
                ],
                "satisfied": False,
            },
        ],
        "terminal_success": False,
        "terminal_failure": False,
    }
    metadata = {"initial_privileged_state": initial}
    activated = {
        **initial,
        "goal_predicates": [
            {"predicate": ["turnon", "flat_stove_1"], "satisfied": True},
            initial["goal_predicates"][1],
        ],
    }
    label = adapter.label_step(activated, metadata)
    assert label.stage_name == "approach_object"
    assert label.evidence["control_active"] is True
    failure = adapter.label_step(
        {**activated, "terminal_failure": True},
        metadata,
    )
    assert failure.overall_progress > 0.0
    assert failure.terminal_failure is True


def test_multi_place_ignores_non_object_invariant_for_subgoal_count() -> None:
    registry = LiberoStageRegistry(_registry()["tasks"])
    adapter = registry.resolve("libero_10", 8)
    state = {
        "eef_position": [0.3, 0.0, 0.0],
        "objects": {
            "moka_pot_1": _entity((0.0, 0.0, 0.0)),
            "moka_pot_2": _entity((0.2, 0.0, 0.0)),
            "flat_stove_1": _entity((0.4, 0.0, 0.0), turn_on=True),
            "flat_stove_1_cook_region": _entity((0.4, 0.0, 0.0)),
        },
        "goal_predicates": [
            {
                "predicate": [
                    "on",
                    "moka_pot_1",
                    "flat_stove_1_cook_region",
                ],
                "satisfied": True,
            },
            {
                "predicate": [
                    "on",
                    "moka_pot_2",
                    "flat_stove_1_cook_region",
                ],
                "satisfied": False,
            },
            {"predicate": ["turnon", "flat_stove_1"], "satisfied": True},
        ],
        "terminal_success": False,
        "terminal_failure": False,
    }
    label = adapter.label_step(state, {"initial_privileged_state": state})
    assert label.stage_name == "approach_second_object"
    assert label.evidence["object_goal_flags"] == [True, False]


def test_open_place_locates_object_goal_when_open_goal_is_first() -> None:
    registry = LiberoStageRegistry(_registry()["tasks"])
    adapter = registry.resolve("libero_90", 8)
    state = {
        "eef_position": [0.3, 0.0, 0.0],
        "objects": {
            "wooden_cabinet_1": _entity((0.3, 0.0, 0.0), open_=True),
            "wooden_cabinet_1_top_region": _entity((0.3, 0.0, 0.0)),
            "akita_black_bowl_1": _entity((0.1, 0.0, 0.0)),
        },
        "goal_predicates": [
            {
                "predicate": ["open", "wooden_cabinet_1_top_region"],
                "satisfied": True,
            },
            {
                "predicate": [
                    "in",
                    "akita_black_bowl_1",
                    "wooden_cabinet_1_top_region",
                ],
                "satisfied": False,
            },
        ],
        "terminal_success": False,
        "terminal_failure": False,
    }
    label = adapter.label_step(state, {"initial_privileged_state": state})
    assert label.evidence["target"] == "wooden_cabinet_1_top_region"


def test_lg_r2_full_and_limited_gates_fail_closed() -> None:
    base = {
        "valid_total_rollouts": 250,
        "tasks": 8,
        "suites": 2,
        "natural_failed_episodes": 20,
        "failed_tasks": 3,
        "failure_windows": 250,
        "matched_success_windows": 250,
        "stage_annotation_passed": True,
        "held_out_zero_shot_complete": True,
        "episode_leakage": 0,
        "seed_leakage": 0,
        "processor_identity_completeness": 1.0,
        "checkpoint_identity_completeness": 1.0,
        "infrastructure_failures_excluded": True,
        "failure_categories": 2,
        "held_out_test_failures": 1,
    }
    passed = evaluate_lg_r2_gate(LGR2GateInput(**base))
    assert passed["LG_R2_AUTHORIZED"] is True
    limited_input = {
        **base,
        "valid_total_rollouts": 160,
        "natural_failed_episodes": 10,
        "failure_windows": 150,
        "matched_success_windows": 100,
    }
    limited = evaluate_lg_r2_gate(LGR2GateInput(**limited_input))
    assert limited["LG_R2_AUTHORIZED"] is False
    assert limited["LG_R2_LIMITED_AUTHORIZED"] is True
    failed = evaluate_lg_r2_gate(
        LGR2GateInput(**{**limited_input, "held_out_test_failures": 0})
    )
    assert failed["status"] == "FAILURE_DATA_GATE_NOT_MET"


def test_taxonomy_and_claim_boundaries_are_frozen() -> None:
    assert FAILURE_TAXONOMY[0] == "HORIZON_EXHAUSTION"
    assert "ENVIRONMENT_ERROR" in FAILURE_TAXONOMY
    paths = [
        *sorted((ROOT / "scripts").glob("lg_r1b_*.py")),
        ROOT / "src" / "latentguard" / "progress" / "lg_r1b.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert "import langmani" not in text
    assert "import robolab" not in text
    assert "class failurehead" not in text
    assert "select_intervention_threshold" not in text


def test_lg_r1b_registry_cli_is_importable() -> None:
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        module = importlib.import_module("lg_r1b_build_task_registry")
    finally:
        sys.path.remove(str(scripts))
    result = module.build_manifest(ROOT / "configs" / "lg_r1b" / "task_registry.yaml")
    assert result["status"] == "pass"
    assert result["phase_episode_counts"]["primary_pilot"] == 80
