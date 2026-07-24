"""Build the evidence-bound LG-F0 final summary from committed artifacts."""

# Human-readable artifact values intentionally remain complete strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "final" / "final_summary.json"


def _load(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def _source(relative_path: str, pointers: list[str]) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": _sha256(relative_path),
        "json_pointers": pointers,
    }


def _failure_metrics(block: dict[str, Any]) -> dict[str, float | int]:
    operating_point = block["operating_points"]["0.6"]
    return {
        "auprc": block["auprc"],
        "precision_at_recall": operating_point["precision"],
        "recall": operating_point["recall"],
        "false_positive_rate": operating_point["false_positive_rate"],
        "samples": block["samples"],
        "positives": block["positives"],
    }


def build_summary() -> dict[str, Any]:
    """Return the deterministic final summary derived from frozen artifacts."""
    r0_eval = _load("artifacts/lg_r0/libero_40ep_evaluation.json")
    r0_interface = _load("artifacts/lg_r0/world_model_interface_audit.json")
    r1_data = _load("artifacts/lg_r1/dataset_manifest.json")
    r1_sarm = _load("artifacts/lg_r1/sarm_results.json")
    r1_gate = _load("artifacts/lg_r1/lg_r2_gate.json")
    r1b_data = _load("artifacts/lg_r1b/dataset_manifest.json")
    r1b_sarm = _load("artifacts/lg_r1b/sarm_zero_shot_results.json")
    r1b_gate = _load("artifacts/lg_r1b/lg_r2_gate.json")
    r1c_summary = _load("artifacts/lg_r1c/evaluation_summary.json")
    r1c_gate = _load("artifacts/lg_r1c/lg_r2_gate.json")
    r2a_state = _load("artifacts/lg_r2a/state_only_results.json")
    r2a_action = _load("artifacts/lg_r2a/state_action_results.json")
    r2a_lto = _load("artifacts/lg_r2a/leave_task6_out_results.json")
    r2a_gate = _load("artifacts/lg_r2a/lg_r2b_gate.json")
    r2b_candidates = _load("artifacts/lg_r2b0/candidate_source_manifest.json")
    r2b_restore = _load("artifacts/lg_r2b0/state_restore_validation.json")
    r2b_gate = _load("artifacts/lg_r2b0/lg_r2b1_gate.json")
    rb_replay = _load("artifacts/lg_rb01/faithful_replay_validation.json")
    rb_gate = _load("artifacts/lg_rb01/lg_rb1_gate.json")

    r1_test = r1_sarm["test"]
    r1b_held_out = r1b_sarm["held_out_task"]
    r1b_in_distribution = r1b_sarm["lg_r1_in_distribution_test"]
    state_short = r2a_state["metrics"]["progress_delta"]["short"]
    action_short = r2a_action["metrics"]["progress_delta"]["short"]
    state_terminal = r2a_state["metrics"]["classification"]["terminal_failure"]
    action_terminal = r2a_action["metrics"]["classification"]["terminal_failure"]
    lto_state = r2a_lto["models"]["state_only"]
    lto_action = r2a_lto["models"]["state_action"]

    mae_relative_reduction = (state_short["mae"] - action_short["mae"]) / state_short[
        "mae"
    ]
    lto_mae_relative_reduction = (
        lto_state["progress_delta"]["short"]["mae"]
        - lto_action["progress_delta"]["short"]["mae"]
    ) / lto_state["progress_delta"]["short"]["mae"]

    reward_models: dict[str, dict[str, Any]] = {}
    for model_id, label in (
        ("frozen_sarm", "Frozen SARM"),
        ("robometer_zero_shot", "ROBOMETER zero-shot"),
        ("topreward_zero_shot", "TOPReward zero-shot"),
        ("task_agnostic_ensemble", "Task-agnostic ensemble"),
    ):
        block = r1c_summary["models"][model_id]["episode_failure"]["all"]
        reward_models[model_id] = {
            "label": label,
            "all_episode_metrics": _failure_metrics(block),
            "formal_gate_scope": r1c_gate["candidate_gates"][model_id][
                "evaluation_scope"
            ],
            "formal_gate_failure_auprc": r1c_gate["candidate_gates"][model_id][
                "candidate_metrics"
            ]["failure_auprc"],
            "promoted": r1c_gate["candidate_gates"][model_id]["status"] == "promoted",
        }

    repeat_rows: list[dict[str, Any]] = []
    for repeat in range(rb_replay["repeats_per_episode"]):
        rows = [row for row in rb_replay["details"] if row["repeat"] == repeat]
        repeat_rows.append(
            {
                "repeat": repeat,
                "attempted": len(rows),
                "completed": sum(bool(row["replay_completed"]) for row in rows),
                "strict_passes": sum(
                    row["strict_per_step_failure_count"] == 0 for row in rows
                ),
                "strict_failures": sum(
                    row["strict_per_step_failure_count"] > 0 for row in rows
                ),
                "strict_failed_steps": sum(
                    row["strict_per_step_failure_count"] for row in rows
                ),
                "official_failed_replays": sum(
                    not row["official_state_validator_pass"] for row in rows
                ),
                "maximum_official_state_error": max(
                    row["official_maximum_absolute_error"] for row in rows
                ),
            }
        )

    episodes = r1b_data["valid_rollout_episodes"]
    in_domain_spearman = r1b_in_distribution["progress"]["spearman"]
    held_out_spearman = r1b_held_out["progress"]["spearman"]
    standalone_reward_auprc = reward_models["robometer_zero_shot"][
        "all_episode_metrics"
    ]["auprc"]
    official_failures = rb_replay["official_state_validator_failures"]
    official_attempts = rb_replay["expected_replay_count"]

    conclusion_en = (
        f"Across {episodes} multi-task LIBERO episodes, task-specific progress "
        f"modeling collapsed from {in_domain_spearman:.3f} to "
        f"{held_out_spearman:.3f} Spearman correlation on held-out tasks; the "
        "best standalone zero-shot task-agnostic reward model reached only "
        f"{standalone_reward_auprc:.3f} failure AUPRC; numeric action "
        "conditioning improved short-horizon prediction MAE by just "
        f"{mae_relative_reduction * 100:.2f}%; and {official_failures} of "
        f"{official_attempts} repeated RoboLab replays violated the official "
        "0.01 state tolerance despite a successful compatibility patch."
    )
    conclusion_zh = (
        f"在{episodes}条多任务LIBERO轨迹上，任务特定进度模型的Spearman相关性从"
        f"域内{in_domain_spearman:.3f}降至未见任务{held_out_spearman:.3f}；最佳"
        f"独立零样本任务无关奖励模型的失败AUPRC仅为{standalone_reward_auprc:.3f}；"
        f"加入数值动作后短期预测MAE只改善{mae_relative_reduction * 100:.2f}%；"
        f"修复RoboLab配置重放问题后，{official_attempts}次重复回放中仍有"
        f"{official_failures}次违反官方0.01状态容差。"
    )

    phases = [
        {
            "phase": "LG-R0",
            "research_question": "Can a pinned VLA-JEPA policy and temporal predictor execute on LIBERO data?",
            "dataset_task_scale": {
                "episodes": r0_eval["overall"]["completed_episodes"],
                "scope": r0_eval["result_scope"],
                "official_400_episode_benchmark": r0_eval[
                    "official_400_episode_equivalent"
                ],
            },
            "main_baseline": "Pinned LeRobot VLA-JEPA LIBERO checkpoint",
            "primary_metric": {
                "name": "development smoke success rate",
                "value": r0_eval["overall"]["success_rate"],
                "numerator": r0_eval["overall"]["successes"],
                "denominator": r0_eval["overall"]["expected_episodes"],
            },
            "result": {
                "policy_execution_and_temporal_prediction_ran": True,
                "external_numeric_candidate_actions_supported": r0_interface[
                    "external_numeric_candidate_actions_supported"
                ],
                "numeric_action_chunk_consumed_by_predictor": r0_interface[
                    "numeric_action_chunk_consumed_by_predictor"
                ],
                "native_scalar_scores": r0_interface["native_scalar_scores"],
            },
            "gate_decision": "Result B: interface mismatch for pre-execution candidate verification",
            "status_label": "partial",
            "status_score": 1,
            "what_was_learned": "A strong task-0 development smoke does not establish a candidate-verification interface.",
            "next_action": "Study progress and failure signals on collected multi-task trajectories.",
        },
        {
            "phase": "LG-R1",
            "research_question": "Can task-specific stage supervision provide a strong in-domain progress signal?",
            "dataset_task_scale": {
                "episodes": r1_data["valid_rollout_episodes"],
                "successful_episodes": r1_data["valid_rollout_episodes"]
                - r1_data["natural_failed_episodes"],
                "natural_failed_episodes": r1_data["natural_failed_episodes"],
                "tasks": r1_data["tasks"],
                "suites": r1_data["suites"],
            },
            "main_baseline": "SARM-style task-specific progress model",
            "primary_metric": {
                "name": "in-domain progress Spearman",
                "value": r1_test["progress"]["spearman"],
            },
            "result": {
                "progress_mae": r1_test["progress"]["mae"],
                "pairwise_accuracy": r1_test["pairwise"]["accuracy"],
                "stagnation_recall": r1_test["temporal"]["stagnation_detection_recall"],
            },
            "gate_decision": r1_gate["status"],
            "status_label": "partial",
            "status_score": 1,
            "what_was_learned": "The in-domain signal was strong, but one natural failure was insufficient for a failure-verification claim.",
            "next_action": "Collect natural failures and run held-out-task evaluation.",
        },
        {
            "phase": "LG-R1b",
            "research_question": "Does the task-specific progress ontology generalize to held-out tasks?",
            "dataset_task_scale": {
                "episodes": r1b_data["valid_rollout_episodes"],
                "successful_episodes": r1b_data["valid_rollout_episodes"]
                - r1b_data["natural_failed_episodes"],
                "natural_failed_episodes": r1b_data["natural_failed_episodes"],
                "tasks": r1b_data["tasks"],
                "suites": r1b_data["suites"],
                "frames": r1b_data["current_samples"],
            },
            "main_baseline": "Frozen LG-R1 SARM model",
            "primary_metric": {
                "name": "held-out-task progress Spearman",
                "value": r1b_held_out["progress"]["spearman"],
            },
            "result": {
                "progress_mae": r1b_held_out["progress"]["mae"],
                "pairwise_accuracy": r1b_held_out["pairwise"]["accuracy"],
                "stage_accuracy": r1b_held_out["stage"]["accuracy"],
                "stage_regression_recall": r1b_held_out["temporal"][
                    "stage_regression_detection_recall"
                ],
            },
            "gate_decision": f"failure-data gate {r1b_gate['status']}",
            "status_label": "failure-data gate passed",
            "status_score": 2,
            "what_was_learned": "The expanded dataset enabled failure study, while task-structured progress collapsed on unseen tasks.",
            "next_action": "Compare task-agnostic post-execution reward signals.",
        },
        {
            "phase": "LG-R1c",
            "research_question": "Can task-agnostic video reward models provide a transferable failure signal?",
            "dataset_task_scale": {
                "episodes": episodes,
                "natural_failed_episodes": r1b_data["natural_failed_episodes"],
                "frames": r1b_data["current_samples"],
            },
            "main_baseline": "ROBOMETER zero-shot",
            "primary_metric": {
                "name": "episode failure AUPRC",
                "value": standalone_reward_auprc,
            },
            "result": reward_models["robometer_zero_shot"]["all_episode_metrics"],
            "gate_decision": r1c_gate["result_class"],
            "status_label": "blocked",
            "status_score": 0,
            "what_was_learned": "Post-execution video reward signals remained weak and do not score unexecuted numeric action candidates.",
            "next_action": "Test numeric action conditioning directly without promoting a reward baseline.",
        },
        {
            "phase": "LG-R2a",
            "research_question": "Does numeric action conditioning add stable predictive value beyond state?",
            "dataset_task_scale": {
                "episodes": episodes,
                "frames": r1b_data["current_samples"],
                "grouped_cross_validation_folds": len(r2a_state["folds"]),
            },
            "main_baseline": "State-only versus state-plus-action predictors",
            "primary_metric": {
                "name": "short-horizon progress MAE relative reduction",
                "value": mae_relative_reduction,
            },
            "result": {
                "state_only_short_mae": state_short["mae"],
                "state_action_short_mae": action_short["mae"],
                "state_only_short_spearman": state_short["spearman"],
                "state_action_short_spearman": action_short["spearman"],
                "state_only_terminal_failure_auprc": state_terminal["auprc"],
                "state_action_terminal_failure_auprc": action_terminal["auprc"],
                "leave_task6_short_mae_relative_reduction": lto_mae_relative_reduction,
            },
            "gate_decision": r2a_gate["result_class"],
            "status_label": "blocked",
            "status_score": 0,
            "what_was_learned": "Actions carried local classification signal, but progress gain was small and did not remain stable under held-out-task evaluation.",
            "next_action": "Audit same-state replay before any candidate outcome experiment.",
        },
        {
            "phase": "LG-R2b0",
            "research_question": "Can LIBERO support strict same-state counterfactual branching?",
            "dataset_task_scale": {
                "probe_states": r2b_restore["probe_states"],
                "repeats_per_state": r2b_restore["repeats_per_state"],
                "generated_candidates": r2b_candidates["different_seed_generation"][
                    "requested"
                ],
                "unique_candidates": r2b_candidates["different_seed_generation"][
                    "unique"
                ],
            },
            "main_baseline": "Frozen VLA-JEPA candidates with LIBERO state restoration",
            "primary_metric": {
                "name": "stepped-trace restore failures",
                "value": r2b_restore["restore_failures"],
            },
            "result": {
                "terminal_mismatches": r2b_restore["terminal_result_mismatches"],
                "maximum_numeric_error": r2b_restore["maximum_observed_numeric_error"],
            },
            "gate_decision": f"Result {r2b_gate['result']}",
            "status_label": "blocked",
            "status_score": 0,
            "what_was_learned": "Real action diversity existed, but the frozen restoration contract could not support branching.",
            "next_action": "Evaluate one alternative replay platform without changing the verification claim.",
        },
        {
            "phase": "LG-RB0/RB0.1",
            "research_question": "Can RoboLab provide repeated faithful replay after one bounded compatibility patch?",
            "dataset_task_scale": {
                "tasks": 2,
                "recordings": rb_replay["episode_count"],
                "repeats_per_recording": rb_replay["repeats_per_episode"],
                "attempts": rb_replay["expected_replay_count"],
            },
            "main_baseline": "Recorded RoboLab actions with schema-aware config overlay",
            "primary_metric": {
                "name": "official state-validator failures",
                "value": rb_replay["official_state_validator_failures"],
                "denominator": rb_replay["expected_replay_count"],
                "tolerance": rb_replay["official_state_tolerance"],
            },
            "result": {
                "completed_replays": rb_replay["completed_replay_count"],
                "strict_per_step_failures": rb_replay["per_step_state_failures"],
                "terminal_mismatches": rb_replay["terminal_mismatches"],
                "success_mismatches": rb_replay["success_mismatches"],
                "maximum_official_state_error": max(
                    row["maximum_official_state_error"] for row in repeat_rows
                ),
            },
            "gate_decision": rb_gate["decision"],
            "status_label": "blocked",
            "status_score": 0,
            "what_was_learned": "Mechanical completion and terminal agreement do not establish stepwise replay fidelity.",
            "next_action": "Close the exact same-state counterfactual route.",
        },
    ]

    return {
        "schema_version": "latentguard.lg_f0.final_summary.v1",
        "status": "research_closeout",
        "positioning": {
            "en": "LatentGuard-VLA is a preregistered empirical study of whether VLA actions can be verified before execution using progress models, task-agnostic reward models, numeric action conditioning, and simulator counterfactual replay.",
            "zh": "LatentGuard-VLA 系统研究了任务进度模型、通用视频奖励模型、数值动作条件化和模拟器反事实重放，能否支持VLA动作执行前验证。",
        },
        "closeout_labels": [
            "Research closeout",
            "Counterfactual VLA verifier not validated",
            "No online intervention claim",
        ],
        "one_sentence_conclusion": {
            "en": conclusion_en,
            "zh": conclusion_zh,
        },
        "core_metrics": {
            "libero_multi_task_episodes": episodes,
            "progress_in_domain_spearman": in_domain_spearman,
            "progress_held_out_task_spearman": held_out_spearman,
            "best_standalone_zero_shot_reward_failure_auprc": standalone_reward_auprc,
            "short_horizon_mae_relative_reduction": mae_relative_reduction,
            "robolab_official_tolerance_failures": official_failures,
            "robolab_replay_attempts": official_attempts,
        },
        "progress_generalization": {
            "in_domain": {
                "mae": r1b_in_distribution["progress"]["mae"],
                "spearman": in_domain_spearman,
                "pairwise_accuracy": r1b_in_distribution["pairwise"]["accuracy"],
            },
            "held_out_task": {
                "mae": r1b_held_out["progress"]["mae"],
                "spearman": held_out_spearman,
                "pairwise_accuracy": r1b_held_out["pairwise"]["accuracy"],
            },
        },
        "reward_model_failure_metrics": reward_models,
        "action_conditioning": {
            "all_tasks": {
                "state_only": {
                    "short_mae": state_short["mae"],
                    "short_spearman": state_short["spearman"],
                    "terminal_failure_auprc": state_terminal["auprc"],
                },
                "state_action": {
                    "short_mae": action_short["mae"],
                    "short_spearman": action_short["spearman"],
                    "terminal_failure_auprc": action_terminal["auprc"],
                },
                "short_mae_relative_reduction": mae_relative_reduction,
            },
            "leave_task6_out": {
                "state_only": {
                    "short_mae": lto_state["progress_delta"]["short"]["mae"],
                    "short_spearman": lto_state["progress_delta"]["short"]["spearman"],
                    "terminal_failure_auprc": lto_state["classification"][
                        "terminal_failure"
                    ]["auprc"],
                },
                "state_action": {
                    "short_mae": lto_action["progress_delta"]["short"]["mae"],
                    "short_spearman": lto_action["progress_delta"]["short"]["spearman"],
                    "terminal_failure_auprc": lto_action["classification"][
                        "terminal_failure"
                    ]["auprc"],
                },
                "short_mae_relative_reduction": lto_mae_relative_reduction,
            },
        },
        "robolab_replay_reproducibility": {
            "official_state_tolerance": rb_replay["official_state_tolerance"],
            "strict_state_tolerance": rb_replay["strict_state_tolerance"],
            "repeat_rows": repeat_rows,
        },
        "phases": phases,
        "contributions": {
            "completed_research_engineering": [
                "LeRobot, VLA-JEPA, and LIBERO integration",
                "Multi-task on-policy rollout and natural-failure datasets",
                "Stage/progress labeling and held-out-task evaluation",
                "Unified reward-model comparison",
                "Action-conditioning negative controls and episode-grouped cross-validation",
                "Simulator replay determinism gates and artifact identity",
                "A bounded upstream-quality RoboLab compatibility patch",
            ],
            "empirical_findings": [
                "Strong in-domain progress does not imply held-out-task generalization.",
                "Post-execution reward does not automatically become a pre-execution candidate verifier.",
                "Single-policy on-policy data provides weak identification of incremental action value.",
                "Common replay APIs did not satisfy strict repeated same-state branching requirements.",
                "Terminal agreement does not imply stepwise dynamics agreement.",
            ],
            "unvalidated_goals": [
                "Candidate outcome model",
                "Candidate ranking",
                "Incremental value of a learned world representation",
                "Closed-loop intervention improvement",
                "Real-robot deployment",
            ],
        },
        "stop_statement": {
            "en": "The exact same-state counterfactual route is closed. The project will not move to a third simulator, add another reward model, or continue replay-infrastructure repair.",
            "zh": "精确同状态反事实路线已经关闭。本项目不会迁移到第三个模拟器，不会继续增加奖励模型，也不会继续修复重放基础设施。",
        },
        "sources": [
            _source(
                "artifacts/lg_r0/libero_40ep_evaluation.json",
                [
                    "/overall/completed_episodes",
                    "/overall/success_rate",
                    "/overall/successes",
                    "/official_400_episode_equivalent",
                    "/result_scope",
                ],
            ),
            _source(
                "artifacts/lg_r0/world_model_interface_audit.json",
                [
                    "/external_numeric_candidate_actions_supported",
                    "/native_scalar_scores",
                    "/numeric_action_chunk_consumed_by_predictor",
                ],
            ),
            _source(
                "artifacts/lg_r1/dataset_manifest.json",
                [
                    "/valid_rollout_episodes",
                    "/natural_failed_episodes",
                    "/tasks",
                    "/suites",
                ],
            ),
            _source(
                "artifacts/lg_r1/sarm_results.json",
                ["/test/progress", "/test/pairwise", "/test/temporal"],
            ),
            _source(
                "artifacts/lg_r1/lg_r2_gate.json",
                ["/LG_R2_AUTHORIZED", "/status", "/checks/natural_failed_episodes"],
            ),
            _source(
                "artifacts/lg_r1b/dataset_manifest.json",
                [
                    "/valid_rollout_episodes",
                    "/natural_failed_episodes",
                    "/current_samples",
                    "/tasks",
                    "/suites",
                ],
            ),
            _source(
                "artifacts/lg_r1b/sarm_zero_shot_results.json",
                ["/held_out_task", "/lg_r1_in_distribution_test"],
            ),
            _source(
                "artifacts/lg_r1b/lg_r2_gate.json",
                ["/LG_R2_AUTHORIZED", "/status", "/full_checks"],
            ),
            _source(
                "artifacts/lg_r1c/evaluation_summary.json",
                ["/models", "/gate", "/result_class"],
            ),
            _source(
                "artifacts/lg_r1c/lg_r2_gate.json",
                ["/candidate_gates", "/result_class", "/status"],
            ),
            _source(
                "artifacts/lg_r2a/state_only_results.json",
                [
                    "/metrics/progress_delta/short",
                    "/metrics/classification/terminal_failure",
                ],
            ),
            _source(
                "artifacts/lg_r2a/state_action_results.json",
                [
                    "/metrics/progress_delta/short",
                    "/metrics/classification/terminal_failure",
                ],
            ),
            _source(
                "artifacts/lg_r2a/leave_task6_out_results.json",
                ["/models/state_only", "/models/state_action"],
            ),
            _source(
                "artifacts/lg_r2a/lg_r2b_gate.json",
                ["/inputs/incremental_value", "/inputs/robustness", "/result_class"],
            ),
            _source(
                "artifacts/lg_r2b0/candidate_source_manifest.json",
                ["/different_seed_generation", "/policy_generated_candidate_ratio"],
            ),
            _source(
                "artifacts/lg_r2b0/state_restore_validation.json",
                [
                    "/restore_failures",
                    "/terminal_result_mismatches",
                    "/maximum_observed_numeric_error",
                    "/status",
                ],
            ),
            _source(
                "artifacts/lg_r2b0/lg_r2b1_gate.json",
                ["/result", "/status", "/stop_reason"],
            ),
            _source(
                "artifacts/lg_rb01/faithful_replay_validation.json",
                [
                    "/completed_replay_count",
                    "/official_state_validator_failures",
                    "/per_step_state_failures",
                    "/details",
                ],
            ),
            _source(
                "artifacts/lg_rb01/lg_rb1_gate.json",
                ["/decision", "/authorization", "/route_status"],
            ),
        ],
    }


def _write_summary(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


def main() -> None:
    """Build or validate the committed final summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = build_summary()
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            raise SystemExit(f"missing final summary: {output}")
        observed = json.loads(output.read_text(encoding="utf-8"))
        if observed != summary:
            raise SystemExit("final summary differs from frozen source artifacts")
        print("final-summary consistency: pass")
        return
    _write_summary(summary, output)
    print(output)


if __name__ == "__main__":
    main()
