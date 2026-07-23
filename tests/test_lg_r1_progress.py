from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from latentguard.progress.dataset import (
    EpisodeAssignment,
    assert_no_split_leakage,
    assign_episode_split,
    build_progress_windows,
)
from latentguard.progress.gates import LG_R2GateInput, evaluate_lg_r2_gate
from latentguard.progress.metrics import (
    evaluate_progress,
    evaluate_stage_predictions,
    pairwise_accuracy,
)
from latentguard.progress.models import StageLabel, compose_progress
from latentguard.progress.rollout import (
    RolloutJob,
    action_step_record,
    build_rollout_jobs,
    completed_episode_ids,
    validate_seed_isolation,
    write_completion_marker,
)
from latentguard.progress.serialization import (
    read_json,
    write_json_atomic,
)
from latentguard.progress.stages.libero_registry import LiberoStageRegistry

ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict[str, object]:
    payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _state(
    *,
    object_name: str = "object_1",
    target_name: str = "target_1",
    object_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_position: tuple[float, float, float] = (0.4, 0.0, 0.0),
    eef_position: tuple[float, float, float] = (0.3, 0.0, 0.0),
    contact: bool = False,
    goal: bool = False,
    terminal_success: bool = False,
    terminal_failure: bool = False,
) -> dict[str, object]:
    return {
        "eef_position": list(eef_position),
        "objects": {
            object_name: {
                "position": list(object_position),
                "contact_with_gripper": contact,
            },
            target_name: {
                "position": list(target_position),
                "open": False,
                "close": True,
            },
        },
        "goal_predicates": [
            {
                "predicate": ["in", object_name, target_name],
                "satisfied": goal,
            }
        ],
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }


def _pick_adapter() -> object:
    registry = LiberoStageRegistry(
        [
            {
                "suite": "suite",
                "task_id": 1,
                "adapter": "pick_place",
                "manipulated_objects": ["object_1"],
                "goal_predicates": [["in", "object_1", "target_1"]],
            }
        ]
    )
    return registry.resolve("suite", 1)


def test_exact_stack_and_task_registry_are_frozen() -> None:
    registry = _yaml("configs/lg_r1/task_registry.yaml")
    policy = registry["source_policy"]
    assert isinstance(policy, dict)
    assert policy["checkpoint_revision"] == ("735d9f692981e286ade093b5046627eda876e5d0")
    assert policy["lerobot_commit"] == ("30da8e687a6dfc617fcd94afc367ac7071c376ce")
    tasks = registry["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == 8
    assert len({task["suite"] for task in tasks}) == 4
    assert registry["frozen_before_rollout"] is True


def test_sarm_revision_and_claim_boundary_are_exact() -> None:
    config = _yaml("configs/lg_r1/sarm.yaml")
    assert config["lerobot_commit"] == ("30da8e687a6dfc617fcd94afc367ac7071c376ce")
    assert config["clip_revision"] == ("3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    assert config["official_checkpoint"] is None
    assert "SARM-style small baseline" in str(config["claim_label"])


def test_frozen_seed_groups_do_not_overlap() -> None:
    registry = _yaml("configs/lg_r1/task_registry.yaml")
    validate_seed_isolation(registry)
    schedule = registry["seed_schedule"]
    assert isinstance(schedule, dict)
    pilot = set(schedule["pilot"])
    extension = set(schedule["extension"])
    history = set(schedule["historical_development"])
    assert not (pilot & extension or pilot & history or extension & history)
    assert not ((pilot | extension | history) & set(range(900000, 900100)))


def test_rollout_schedule_records_all_80_or_160() -> None:
    registry = _yaml("configs/lg_r1/task_registry.yaml")
    pilot = build_rollout_jobs(registry, include_extension=False)
    extended = build_rollout_jobs(registry, include_extension=True)
    assert len(pilot) == 80
    assert len(extended) == 160
    assert len({job.episode_id for job in extended}) == 160
    assert {job.phase for job in extended} == {"pilot", "extension"}


def test_atomic_completion_and_resume_do_not_duplicate(tmp_path: Path) -> None:
    job = RolloutJob(
        episode_id="suite-task1-seed2000",
        suite="suite",
        task_id=1,
        task_name="task",
        instruction="do task",
        episode_horizon=10,
        seed=2000,
        phase="pilot",
    )
    write_completion_marker(
        tmp_path,
        job,
        frame_count=10,
        success=False,
        termination_reason="horizon_exhausted",
        dataset_locator="dataset_root/episode",
        sidecar_locator="sidecar_root/episode.jsonl",
    )
    assert completed_episode_ids(tmp_path) == {job.episode_id}
    marker = read_json(tmp_path / f"{job.episode_id}.json")
    assert marker["status"] == "complete"
    assert not (tmp_path / f"{job.episode_id}.json.partial").exists()


def test_generated_chunk_action_mask_and_padding_are_preserved() -> None:
    chunk = [[float(row * 7 + column) for column in range(7)] for row in range(7)]
    record = action_step_record(
        step_index=3,
        generated_chunk=chunk,
        chunk_offset=3,
        selected_policy_action=chunk[3],
        executed_action=[value / 10.0 for value in chunk[3]],
    )
    assert record["generated_action_chunk"] == chunk
    assert record["selected_policy_action"] == chunk[3]
    assert record["action_mask"] is None
    assert record["action_is_pad"] is False
    with pytest.raises(ValueError, match="differs"):
        action_step_record(
            step_index=3,
            generated_chunk=chunk,
            chunk_offset=3,
            selected_policy_action=chunk[2],
            executed_action=chunk[3],
        )


def test_stage_registry_and_pick_place_semantics() -> None:
    adapter = _pick_adapter()
    initial = _state()
    metadata = {"initial_privileged_state": initial}
    approach = adapter.label_step(initial, metadata)
    contact = adapter.label_step(
        _state(eef_position=(0.01, 0.0, 0.0), contact=True),
        metadata,
    )
    lifted = adapter.label_step(
        _state(
            object_position=(0.0, 0.0, 0.1),
            eef_position=(0.01, 0.0, 0.1),
            contact=True,
        ),
        metadata,
    )
    success = adapter.label_step(
        _state(goal=True, terminal_success=True),
        metadata,
    )
    assert approach.stage_id == 0
    assert contact.stage_name == "grasp_contact"
    assert lifted.stage_id >= 3
    assert success.overall_progress == 1.0
    assert success.terminal_success is True


def test_failure_is_not_forced_to_zero_and_timeout_is_not_failure() -> None:
    adapter = _pick_adapter()
    initial = _state()
    metadata = {"initial_privileged_state": initial}
    failure = adapter.label_step(
        _state(
            eef_position=(0.01, 0.0, 0.0),
            contact=True,
            terminal_failure=True,
        ),
        metadata,
    )
    timeout = adapter.label_step(
        _state(eef_position=(0.01, 0.0, 0.0), contact=True),
        metadata,
    )
    assert failure.overall_progress > 0.0
    assert failure.terminal_failure is True
    assert timeout.terminal_failure is False


def test_stage_and_progress_can_regress() -> None:
    adapter = _pick_adapter()
    initial = _state()
    metadata = {"initial_privileged_state": initial}
    high = adapter.label_step(
        _state(
            object_position=(0.0, 0.0, 0.1),
            eef_position=(0.01, 0.0, 0.1),
            contact=True,
        ),
        metadata,
    )
    low = adapter.label_step(initial, metadata)
    assert low.stage_id < high.stage_id
    windows = build_progress_windows(
        episode_id="episode",
        split="train",
        labels=[high, low],
        window_steps={"short": 1},
        stride=1,
        progress_epsilon=0.02,
    )
    assert windows[0].regression is True
    assert windows[0].progress_delta < 0.0


def test_stage_label_validation_rejects_invalid_values() -> None:
    assert compose_progress(1, 4, 0.5, terminal_success=False) == 0.375
    with pytest.raises(ValueError, match="overall_progress"):
        StageLabel(
            stage_id=1,
            stage_name="bad",
            stage_completion=0.5,
            overall_progress=float("nan"),
            terminal_success=False,
            terminal_failure=False,
            evidence={},
        )
    with pytest.raises(ValueError, match="terminal success"):
        StageLabel(
            stage_id=1,
            stage_name="bad",
            stage_completion=1.0,
            overall_progress=0.5,
            terminal_success=True,
            terminal_failure=False,
            evidence={},
        )


def test_episode_and_seed_splits_are_exclusive() -> None:
    first = assign_episode_split(
        episode_id="a",
        suite="suite",
        task_id=1,
        seed=1,
        train_seeds={1},
        validation_seeds={2},
        test_seeds={3},
    )
    second = assign_episode_split(
        episode_id="b",
        suite="suite",
        task_id=2,
        seed=1,
        train_seeds={1},
        validation_seeds={2},
        test_seeds={3},
    )
    assert_no_split_leakage([first, second])
    with pytest.raises(ValueError, match="seed 1 crosses"):
        assert_no_split_leakage(
            [
                first,
                EpisodeAssignment(
                    episode_id="b",
                    suite="suite",
                    task_id=2,
                    seed=1,
                    split="test",
                    held_out_task=False,
                ),
            ]
        )


def test_metrics_and_lg_r2_gate_fail_closed() -> None:
    stage = evaluate_stage_predictions([0, 1], [0, 0], stage_count=2)
    progress = evaluate_progress([0.0, 1.0], [0.1, 0.8])
    assert stage.accuracy == 0.5
    assert progress.mae == pytest.approx(0.15)
    assert pairwise_accuracy([-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5], epsilon=0.02) == 1.0
    evidence = LG_R2GateInput(
        valid_rollout_episodes=100,
        tasks=4,
        suites=2,
        natural_failed_episodes=9,
        failure_windows=100,
        successful_windows=100,
        stage_annotation_passed=True,
        sarm_evaluation_complete=True,
        episode_split_leakage=0,
        processor_identity_completeness=1.0,
        checkpoint_identity_completeness=1.0,
    )
    assert evaluate_lg_r2_gate(evidence)["LG_R2_AUTHORIZED"] is False


def test_privileged_fields_are_supervision_only() -> None:
    dataset = _yaml("configs/lg_r1/dataset.yaml")
    rollout = _yaml("configs/lg_r1/rollout_collection.yaml")
    assert dataset["allow_privileged_model_inputs"] is False
    sidecar = rollout["sidecar"]
    assert isinstance(sidecar, dict)
    assert sidecar["model_input_allowed"] is False


def test_lg_r1_validate_only_clis(tmp_path: Path) -> None:
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        task_module = importlib.import_module("lg_r1_build_task_registry")
        manifest = task_module.build_manifest(
            ROOT / "configs" / "lg_r1" / "task_registry.yaml"
        )
    finally:
        sys.path.remove(str(scripts))
    assert manifest["status"] == "pass"
    output = tmp_path / "manifest.json"
    write_json_atomic(output, manifest)
    assert json.loads(output.read_text(encoding="utf-8"))["task_count"] == 8


def test_lg_r1_runtime_has_no_forbidden_integrations_or_corruption() -> None:
    paths = [
        *sorted((ROOT / "src" / "latentguard" / "progress").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("lg_r1_*.py")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert "import langmani" not in text
    assert "import robolab" not in text
    assert "import rclpy" not in text
    assert "synthetic corruption" not in text
    assert "train_failure_head" not in text
    assert "class failurehead" not in text
    assert "select_intervention_threshold" not in text
