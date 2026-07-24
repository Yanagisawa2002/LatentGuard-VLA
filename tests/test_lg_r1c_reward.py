from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from latentguard.rewards.metrics import (
    binary_metrics,
    recall_operating_point,
    task_macro_binary_metrics,
)
from latentguard.rewards.schemas import (
    CandidateTimeAvailability,
    evaluate_reward_gate,
    validate_feature_contract,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _yaml(relative: str) -> dict[str, object]:
    payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _script(name: str) -> object:
    sys.path.insert(0, str(SCRIPTS))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SCRIPTS))


def test_exact_external_commits_revisions_and_hashes_are_frozen() -> None:
    stack = _yaml("configs/lg_r1c/reward_stack.yaml")
    lerobot = stack["lerobot"]
    robometer = stack["robometer"]
    topreward = stack["topreward"]
    assert isinstance(lerobot, dict)
    assert isinstance(robometer, dict)
    assert isinstance(topreward, dict)
    assert lerobot["commit"] == "30da8e687a6dfc617fcd94afc367ac7071c376ce"
    assert robometer["upstream_commit"] == "5b815254bf31ee1bea3753c3a2da9f9033736d9a"
    assert topreward["upstream_commit"] == "4877a0ee5098cbec18485125466631b1dcc4a573"
    for revision in (
        robometer["checkpoint_revision"],
        robometer["base_model_revision"],
        robometer["processor_revision"],
        topreward["model_revision"],
        topreward["processor_revision"],
    ):
        assert isinstance(revision, str)
        assert len(revision) == 40
        assert set(revision) <= set("0123456789abcdef")
    checkpoint_files = robometer["checkpoint_files"]
    assert isinstance(checkpoint_files, dict)
    checkpoint = checkpoint_files["model.safetensors"]
    assert checkpoint["bytes"] == 8894103800
    assert len(checkpoint["sha256"]) == 64
    weight_files = topreward["weight_files"]
    assert isinstance(weight_files, dict)
    assert len(weight_files) == 4
    assert all(len(value["sha256"]) == 64 for value in weight_files.values())


def test_reward_stack_audit_is_pre_inference_and_inference_only() -> None:
    module = _script("lg_r1c_audit_reward_stack")
    payload = module.build_manifest(ROOT / "configs" / "lg_r1c" / "reward_stack.yaml")
    assert payload["status"] == "frozen_before_inference"
    findings = payload["audit_findings"]
    assert findings["lerobot_reward_ports_inference_only"] is True
    assert findings["robometer_compute_reward_returns_last_frame_only"] is True
    assert (
        findings["robometer_read_only_private_logits_expose_per_frame_outputs"] is True
    )
    assert findings["topreward_per_frame_output_available"] is False
    assert findings["task_specific_training_or_prompt_tuning_allowed"] is False


def test_frozen_input_identity_matches_lg_r1b_without_runtime_rewrite() -> None:
    module = _script("lg_r1c_validate_input_dataset")
    payload = module.build_manifest(
        ROOT / "configs" / "lg_r1c" / "dataset.yaml",
        runtime=None,
    )
    assert payload["counts"] == {
        "episodes": 320,
        "tasks": 8,
        "suites": 2,
        "successes": 293,
        "failures": 27,
        "failure_producing_tasks": 4,
        "frames": 79326,
        "failure_windows": 7025,
        "matched_success_windows": 2315,
        "episode_leakage": 0,
        "seed_leakage": 0,
    }
    assert payload["original_split_reused"] is True
    assert payload["resplit_performed"] is False
    assert payload["new_rollouts"] == 0
    assert payload["synthetic_failures"] == 0
    assert payload["sealed_final_seed_overlap"] == []


def test_window_sampling_is_deterministic_and_models_share_endpoints() -> None:
    module = _script("lg_r1c_build_reward_windows")
    expected = [3, 6, 9, 12, 14, 17, 20, 23]
    assert module._uniform_indices(3, 23, 8) == expected
    assert module._uniform_indices(3, 23, 8) == expected
    windows = _yaml("configs/lg_r1c/windows.yaml")
    assert windows["frozen_before_model_inference"] is True
    assert windows["same_endpoint_across_models"] is True
    assert windows["same_instruction_across_models"] is True
    assert windows["same_camera_across_models"] is True
    assert windows["model_result_dependent_sampling"] is False
    assert windows["anchor_fractions"] == [0.25, 0.5, 0.75, 1.0]
    contexts = windows["contexts"]
    assert isinstance(contexts, dict)
    assert [contexts[name]["span_steps"] for name in ("short", "medium", "long")] == [
        8,
        32,
        64,
    ]


def test_task_macro_equal_weights_tasks_not_windows() -> None:
    labels = [1, 0, 1, 1, 1, 1]
    scores = [0.9, 0.1, 0.1, 0.2, 0.3, 0.4]
    tasks = ["small", "small", "large", "large", "large", "large"]
    result = task_macro_binary_metrics(labels, scores, tasks)
    assert result["eligible_auroc_tasks"] == 1
    assert result["per_task"]["small"]["auroc"] == pytest.approx(1.0)
    assert result["per_task"]["large"]["samples"] == 4


def test_failure_metrics_report_prevalence_and_recall_operating_points() -> None:
    labels = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    summary = binary_metrics(labels, scores)
    assert summary.prevalence == pytest.approx(0.4)
    assert summary.auroc == pytest.approx(5 / 6)
    point = recall_operating_point(labels, scores, requested_recall=0.5)
    assert point.recall >= 0.5
    assert point.precision == pytest.approx(1.0)


def test_calibration_is_validation_only_and_zero_shot_freeze_precedes_it() -> None:
    config = _yaml("configs/lg_r1c/calibration.yaml")
    assert config["selection_split"] == "validation"
    assert config["test_labels_available_during_fit"] is False
    assert config["zero_shot_freeze_required"] is True
    assert config["task_id_input"] is False
    assert config["stage_id_input"] is False
    assert config["privileged_input"] is False
    text = (SCRIPTS / "lg_r1c_calibrate_rewards.py").read_text(encoding="utf-8")
    assert "_require_zero_shot_freeze(destination)" in text
    assert "_load_validation_windows(destination)" in text
    assert "from _lg_r1c_reward_runtime import load_windows" not in text
    assert '"fit_splits": ["validation"]' in text
    assert '"test_labels_used_for_fit": False' in text
    builder = (SCRIPTS / "lg_r1c_build_reward_windows.py").read_text(encoding="utf-8")
    assert "calibration_validation_windows.jsonl" in builder
    assert '"test_labels_included": False' in builder


def test_zero_shot_freeze_does_not_compute_or_load_test_metrics(
    tmp_path: Path,
) -> None:
    module = _script("lg_r1c_evaluate_rewards")
    for name in (
        "time_predictions.jsonl",
        "sarm_predictions.jsonl",
        "robometer_predictions.jsonl",
        "topreward_predictions.jsonl",
    ):
        (tmp_path / name).write_text(
            json.dumps({"window_id": "window-1"}) + "\n",
            encoding="utf-8",
        )
    (tmp_path / "window_manifest.json").write_text(
        json.dumps({"window_count": 1}) + "\n",
        encoding="utf-8",
    )
    result = module._freeze_zero_shot_predictions(tmp_path)
    assert result["test_labels_read"] is False
    assert result["test_metrics_computed"] is False
    assert result["prediction_windows"] == 1


def test_fitted_gate_uses_test_task_macro_and_test_failure_scope() -> None:
    module = _script("lg_r1c_evaluate_rewards")
    passing_failure = {
        "prevalence": 0.1,
        "auprc": 0.3,
        "operating_points": {
            "0.6": {
                "precision": 0.3,
                "recall": 0.6,
                "false_positive_rate": 0.2,
            }
        },
    }
    summary = {
        "progress": {
            "task_macro": {"spearman": 0.0, "pairwise_accuracy": 0.0},
            "task_macro_by_split": {
                "test": {"spearman": 0.6, "pairwise_accuracy": 0.7}
            },
        },
        "success": {
            "task_macro": {"auroc": 0.0},
            "task_macro_by_split": {"test": {"auroc": 0.8}},
        },
        "episode_failure": {
            "all": {"prevalence": 0.1, "auprc": 0.0, "operating_points": {}},
            "by_split": {"test": passing_failure},
            "leave_task6_out": {"auprc": 0.0},
            "leave_task6_out_by_split": {"test": {"auprc": 0.25}},
        },
    }
    gate = module._candidate_gate(summary, fitted=True)
    assert gate["evaluation_scope"] == "test"
    assert gate["LG_R2_REWARD_BASELINE_AUTHORIZED"] is True


def test_robometer_per_frame_and_topreward_prompt_contracts_are_frozen() -> None:
    robometer = _yaml("configs/lg_r1c/robometer.yaml")
    topreward = _yaml("configs/lg_r1c/topreward.yaml")
    assert robometer["maximum_frames"] == 8
    assert robometer["per_frame_private_read_only_adapter"] is True
    assert robometer["strict_checkpoint_load"] is True
    assert (
        topreward["prompt_suffix_template"]
        == "{instruction} Decide whether the above statement is True or not. "
        "The answer is: True"
    )
    assert topreward["scored_token"] == "True"
    assert topreward["normalization"] == "exp_raw_log_probability"
    assert topreward["common_frames"] == 8
    assert topreward["native_diagnostic_frames"] == 16


def test_exact_local_snapshot_requires_every_file_bound_to_revision(
    tmp_path: Path,
) -> None:
    module = _script("_lg_r1c_reward_runtime")
    revision = "a" * 40
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
        (metadata / f"{name}.metadata").write_text(
            f"{revision}\n{'b' * 64}\n0.0\n",
            encoding="utf-8",
        )
    assert module._is_exact_local_snapshot(tmp_path, revision=revision)
    (metadata / "model.safetensors.metadata").write_text(
        f"{'c' * 40}\n{'b' * 64}\n0.0\n",
        encoding="utf-8",
    )
    assert not module._is_exact_local_snapshot(tmp_path, revision=revision)


def test_failure_taxonomy_leave_task6_and_prevalence_are_explicit() -> None:
    evaluation = _yaml("configs/lg_r1c/evaluation.yaml")
    assert evaluation["failure_taxonomy"] == ["FAILED_PLACEMENT", "OBJECT_DROP"]
    leave = evaluation["leave_task6_out"]
    assert isinstance(leave, dict)
    assert leave == {"suite": "libero_10", "task_id": 6, "diagnostic_only": True}
    assert evaluation["task_macro_equal_weight"] is True
    assert evaluation["failure_recall_operating_points"] == [0.5, 0.6, 0.8]


def test_gate_fails_closed_and_uses_pre_registered_thresholds() -> None:
    passing = {
        "progress_spearman": 0.60,
        "pairwise_accuracy": 0.70,
        "task_macro_success_auroc": 0.75,
        "failure_auprc": 0.30,
        "failure_precision_at_recall": 0.25,
        "failure_recall": 0.60,
        "failure_fpr_at_recall": 0.35,
        "leave_task6_failure_auprc": 0.15,
    }
    result = evaluate_reward_gate(passing, prevalence=0.10)
    assert result["LG_R2_REWARD_BASELINE_AUTHORIZED"] is True
    missing = evaluate_reward_gate(
        {**passing, "progress_spearman": None},
        prevalence=0.10,
    )
    assert missing["LG_R2_REWARD_BASELINE_AUTHORIZED"] is False
    assert missing["missing_metrics"] == ["progress_spearman"]


def test_feature_contract_shapes_and_candidate_time_availability() -> None:
    module = _script("lg_r1c_build_lg_r2_contract")
    specifications = module._specifications(
        robometer_latency=10.0,
        robometer_memory=100,
        topreward_latency=20.0,
        topreward_memory=200,
    )
    assert specifications["current_vla_representation"].candidate_time_usable
    assert (
        specifications["numeric_action_chunk"].candidate_time_availability
        == CandidateTimeAvailability.CANDIDATE_ACTION_CONDITIONED
    )
    for name in (
        "robometer_progress",
        "robometer_success_probability",
        "topreward_score",
        "task_agnostic_reward_outputs",
    ):
        feature = specifications[name]
        assert feature.depends_on_future_observation is True
        assert feature.candidate_time_usable is False
        assert (
            feature.candidate_time_availability
            == CandidateTimeAvailability.EXECUTED_TRAJECTORY_REQUIRED
        )
    payload = {
        "schema_version": "latentguard.lg_r1c.lg_r2_features.v1",
        "feature_sets": [
            {
                "name": name,
                "features": [specifications["current_vla_representation"].to_dict()],
                "lg_r2_training_authorized": False,
            }
            for name in ("A", "B", "C", "D")
        ],
    }
    validate_feature_contract(payload)


def test_scope_scan_has_no_training_intervention_or_external_repo_imports() -> None:
    paths = [
        *sorted(SCRIPTS.glob("lg_r1c_*.py")),
        *sorted((ROOT / "src" / "latentguard" / "rewards").glob("*.py")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert "import langmani" not in text
    assert "import robolab" not in text
    assert "import pointworld" not in text
    assert "class failurehead" not in text
    assert "select_intervention_threshold" not in text
    assert not (SCRIPTS / "lg_r1c_collect_rollouts.py").exists()
    stack = _yaml("configs/lg_r1c/reward_stack.yaml")
    assert stack["robometer"]["foundation_model_training_allowed"] is False
    assert stack["topreward"]["foundation_model_training_allowed"] is False
    dataset = _yaml("configs/lg_r1c/dataset.yaml")
    assert dataset["new_rollout_allowed"] is False
    assert dataset["synthetic_failure_allowed"] is False
    assert dataset["sealed_final_seed_start"] == 900000
    assert dataset["sealed_final_seed_end"] == 900099
