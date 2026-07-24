"""Collect real same-state counterfactual branches with fixed continuation."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2b0_common import (
    batch_observation,
    current_observation,
    load_runtime_stack,
    make_libero_environment,
    observation_sha256,
    output_root,
    privileged_state,
    proprioception_sha256,
    read_json,
    read_yaml,
    rendered_observation_sha256,
    resolve_repo_path,
    runtime_identity,
    select_primary_action,
    sha256_path,
    step_native,
    validate_execution_checkout,
    validate_no_final_seed,
    write_json,
)

from latentguard.counterfactual.libero_state import (
    capture_libero_state,
    load_libero_state,
    restore_libero_state,
)
from latentguard.counterfactual.models import Recoverability, TerminalOutcome
from latentguard.progress.stages.libero_registry import LiberoStageRegistry


def _object_pose_delta(
    anchor: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in sorted(anchor["objects"]):
        if name not in current["objects"]:
            raise ValueError(f"branch state lost object {name}")
        left = np.asarray(anchor["objects"][name]["position"], dtype=np.float64)
        right = np.asarray(current["objects"][name]["position"], dtype=np.float64)
        output[name] = float(np.linalg.norm(right - left))
    return output


def _step_record(
    *,
    local_step: int,
    action_source: str,
    action: np.ndarray,
    state: dict[str, Any],
    label: Any,
    raw: dict[str, Any],
    anchor_state: dict[str, Any],
    single_env: Any,
    success: bool,
) -> dict[str, Any]:
    contact = bool(label.evidence.get("contact", False))
    task_env = single_env._env.env
    qpos = np.asarray(task_env.sim.data.qpos, dtype=np.float64)
    qvel = np.asarray(task_env.sim.data.qvel, dtype=np.float64)
    return {
        "local_step": local_step,
        "action_source": action_source,
        "executed_action": np.asarray(action, dtype=np.float32).tolist(),
        "progress": float(label.overall_progress),
        "stage_id": int(label.stage_id),
        "stage_name": label.stage_name,
        "contact": contact,
        "goal_predicates": state["goal_predicates"],
        "object_state": state["objects"],
        "object_pose_delta": _object_pose_delta(anchor_state, state),
        "eef_position": state["eef_position"],
        "gripper_qpos": state["gripper_qpos"],
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "proprioception_sha256": proprioception_sha256(raw),
        "terminal_success": success,
        "sim_state_sha256": state["sim_state_sha256"],
        "render_sha256": rendered_observation_sha256(raw),
    }


def _events(
    records: list[dict[str, Any]],
    *,
    anchor_stage: str,
    progress_epsilon: float,
    stagnation_threshold: float,
) -> dict[str, bool]:
    if not records:
        return {
            "OBJECT_DROP": False,
            "FAILED_PLACEMENT": False,
            "MISSED_GRASP": False,
            "LOSS_OF_CONTACT": False,
        }
    short = records[:49]
    progresses = [float(item["progress"]) for item in short]
    stages = [str(item["stage_name"]).lower() for item in short]
    contacts = [bool(item["contact"]) for item in short]
    maximum_index = int(np.argmax(progresses))
    drop = (
        any(
            token in stage
            for stage in stages[: maximum_index + 1]
            for token in ("lift", "transport", "place", "manipulate")
        )
        and progresses[maximum_index] - progresses[-1] > progress_epsilon
    )
    placement_reached = any(
        "place" in stage or "arrangement" in stage for stage in stages
    )
    failed_placement = (
        placement_reached
        and not bool(short[-1]["terminal_success"])
        and abs(progresses[-1] - max(progresses)) <= stagnation_threshold
    )
    anchor_lower = anchor_stage.lower()
    grasp_relevant = any(
        token in anchor_lower for token in ("approach", "pre_grasp", "pre_contact")
    )
    missed_grasp = grasp_relevant and not any(contacts)
    loss_contact = any(
        contacts[index] and not any(contacts[index + 1 :])
        for index in range(len(contacts) - 1)
    )
    return {
        "OBJECT_DROP": drop,
        "FAILED_PLACEMENT": failed_placement,
        "MISSED_GRASP": missed_grasp,
        "LOSS_OF_CONTACT": loss_contact,
    }


def _boundary_summary(
    records: list[dict[str, Any]],
    *,
    horizon: int,
    anchor_progress: float,
) -> dict[str, Any]:
    index = min(horizon, len(records)) - 1
    if index < 0:
        raise ValueError("branch produced no short-horizon records")
    record = records[index]
    return {
        "requested_horizon": horizon,
        "observed_step": index + 1,
        "terminal_carry_forward": len(records) < horizon,
        "progress": float(record["progress"]),
        "progress_delta": float(record["progress"]) - anchor_progress,
        "stage_id": int(record["stage_id"]),
        "stage_name": record["stage_name"],
        "contact": bool(record["contact"]),
        "object_pose_delta": record["object_pose_delta"],
        "sim_state_sha256": record["sim_state_sha256"],
    }


def _branch_id(anchor_id: str, candidate_id: str) -> str:
    candidate_suffix = candidate_id.rsplit(":", 1)[-1]
    return f"{anchor_id}__{candidate_suffix}"


def main() -> None:
    """Execute 3-4 real candidates per accepted anchor with common continuation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r2b0/collection.yaml"),
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-branches", type=int)
    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    validate_execution_checkout(args.expected_commit)
    runtime = output_root()
    anchors = read_json(runtime / "anchor_registry.json")
    candidates = read_json(runtime / "candidate_manifest.json")
    diversity = read_json(runtime / "candidate_diversity_report.json")
    source = read_json(runtime / "candidate_source_manifest.json")
    restore = read_json(runtime / "state_restore_validation.json")
    if any(
        payload.get("status") != "pass"
        for payload in (anchors, candidates, diversity, source, restore)
    ):
        raise RuntimeError("one or more prerequisite manifests is not accepted")
    if diversity.get("candidate_diversity_gate", {}).get("status") != "pass":
        raise RuntimeError("candidate diversity gate has not passed")
    task_registry_path = resolve_repo_path(Path(str(config["task_registry"])))
    task_registry_payload = read_yaml(task_registry_path)
    stage_registry = LiberoStageRegistry(task_registry_payload["tasks"])
    stage_config_path = resolve_repo_path(Path(str(config["stage_config"])))
    stage_config = read_yaml(stage_config_path)
    anchor_by_id = {str(item["anchor_id"]): item for item in anchors["anchors"]}
    jobs = [
        (anchor_by_id[str(group["anchor_id"])], candidate)
        for group in candidates["anchors"]
        if group["status"] == "pass"
        for candidate in group["candidates"]
    ]
    if args.max_branches is not None:
        if args.max_branches < 1:
            raise ValueError("--max-branches must be positive")
        jobs = jobs[: args.max_branches]
    continuation_seeds = [
        int(config["continuation_seed_base"]) + index
        for index in range(len(anchor_by_id))
    ]
    validate_no_final_seed(continuation_seeds, "branch continuation")
    branch_root = runtime / "branches"
    completion_root = runtime / "branch_completed"
    attempt_root = runtime / "branch_attempts"
    for path in (branch_root, completion_root, attempt_root):
        path.mkdir(parents=True, exist_ok=True)
    completed = {path.stem for path in completion_root.glob("*.json")}
    if completed and not args.resume:
        raise ValueError("completed branches exist; pass --resume")
    stack, preprocessor, postprocessor = load_runtime_stack()
    errors: list[dict[str, Any]] = []
    branch_summaries: list[dict[str, Any]] = []
    current_anchor_id: str | None = None
    current_resources: tuple[Any, Any, Any, Any] | None = None
    try:
        for anchor, candidate in jobs:
            branch_id = _branch_id(
                str(anchor["anchor_id"]),
                str(candidate["candidate_id"]),
            )
            final_path = branch_root / f"{branch_id}.json"
            if branch_id in completed:
                branch_summaries.append(read_json(final_path)["summary"])
                continue
            if current_anchor_id != anchor["anchor_id"]:
                if current_resources is not None:
                    current_resources[0].close()
                vector_env, single_env, env_pre, env_post, _ = make_libero_environment(
                    suite=str(anchor["suite"]),
                    task_id=int(anchor["task_id"]),
                    episode_horizon=int(
                        config["episode_horizons"][str(anchor["suite"])]
                    ),
                    seed=int(anchor["seed"]),
                    stack=stack,
                )
                current_resources = (
                    vector_env,
                    single_env,
                    env_pre,
                    env_post,
                )
                current_anchor_id = str(anchor["anchor_id"])
            if current_resources is None:
                raise RuntimeError("branch environment was not initialized")
            _, single_env, env_pre, env_post = current_resources
            snapshot = load_libero_state(
                runtime / "anchors" / str(anchor["anchor_id"]) / "state"
            )
            restoration = restore_libero_state(
                single_env,
                snapshot,
                atol=float(config["state_restore_tolerance"]),
            )
            if not restoration.within_tolerance:
                raise RuntimeError("branch anchor restoration mismatch")
            raw = current_observation(single_env)
            instruction = str(anchor["source_rollout_identity"]["instruction"])
            if observation_sha256(raw, instruction) != anchor["observation_hash"]:
                raise RuntimeError("branch anchor observation contamination")
            raw_batched = batch_observation(raw)
            initial_state = anchor["source_rollout_identity"][
                "initial_privileged_state"
            ]
            anchor_state = privileged_state(
                single_env,
                raw,
                terminal_success=False,
            )
            adapter = stage_registry.resolve(
                str(anchor["suite"]),
                int(anchor["task_id"]),
            )
            anchor_label = adapter.label_step(
                anchor_state,
                {
                    "initial_privileged_state": initial_state,
                    "distance_thresholds": stage_config["distance_thresholds"],
                },
            )
            if (
                anchor_label.stage_name != anchor["stage"]
                or int(anchor_label.stage_id) != int(anchor["stage_id"])
                or abs(float(anchor_label.overall_progress) - float(anchor["progress"]))
                > float(config["progress_epsilon"])
            ):
                raise RuntimeError("branch anchor stage identity mismatch")
            action_chunk = np.asarray(
                candidate["native_action_chunk"],
                dtype=np.float32,
            )
            records: list[dict[str, Any]] = []
            success = False
            environment_done = False
            continuation_started = False
            maximum_steps = int(config["episode_horizons"][str(anchor["suite"])]) - int(
                anchor["step_index"]
            )
            started = time.perf_counter()
            for local_step in range(maximum_steps):
                if local_step < action_chunk.shape[0]:
                    action = action_chunk[local_step]
                    action_source = "real_policy_candidate"
                else:
                    if not continuation_started:
                        import torch

                        stack.policy.reset()
                        anchor_ordinal = list(anchor_by_id).index(
                            str(anchor["anchor_id"])
                        )
                        continuation_seed = (
                            int(config["continuation_seed_base"]) + anchor_ordinal
                        )
                        torch.manual_seed(continuation_seed)
                        torch.cuda.manual_seed_all(continuation_seed)
                        continuation_started = True
                    _, action = select_primary_action(
                        stack=stack,
                        raw_observation=raw_batched,
                        instruction=instruction,
                        env_preprocessor=env_pre,
                        env_postprocessor=env_post,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    action_source = "fixed_primary_continuation"
                next_raw, _, environment_done, info = step_native(
                    single_env,
                    action,
                )
                success = bool(info["is_success"])
                state = privileged_state(
                    single_env,
                    next_raw,
                    terminal_success=success,
                )
                label = adapter.label_step(
                    state,
                    {
                        "initial_privileged_state": initial_state,
                        "distance_thresholds": stage_config["distance_thresholds"],
                    },
                )
                records.append(
                    _step_record(
                        local_step=local_step,
                        action_source=action_source,
                        action=action,
                        state=state,
                        label=label,
                        raw=next_raw,
                        anchor_state=anchor_state,
                        single_env=single_env,
                        success=success,
                    )
                )
                raw_batched = batch_observation(next_raw)
                if environment_done:
                    break
            boundaries = {
                str(horizon): _boundary_summary(
                    records,
                    horizon=int(horizon),
                    anchor_progress=float(anchor["progress"]),
                )
                for horizon in config["short_horizons"]
            }
            terminal_outcome = (
                TerminalOutcome.SUCCESS
                if success
                else (
                    TerminalOutcome.FAILURE
                    if len(records) == maximum_steps
                    else TerminalOutcome.UNRESOLVED_HORIZON
                )
            )
            recoverability = (
                Recoverability.RECOVERED
                if success and len(records) > action_chunk.shape[0]
                else (
                    Recoverability.FAILED
                    if terminal_outcome == TerminalOutcome.FAILURE
                    else Recoverability.NOT_DETERMINED
                )
            )
            events = _events(
                records,
                anchor_stage=str(anchor["stage"]),
                progress_epsilon=float(config["event_progress_epsilon"]),
                stagnation_threshold=float(config["stagnation_threshold"]),
            )
            progress_deltas = {
                f"progress_delta_{horizon}": float(
                    boundaries[str(horizon)]["progress_delta"]
                )
                for horizon in config["short_horizons"]
            }
            after_snapshot = capture_libero_state(single_env)
            summary = {
                "branch_id": branch_id,
                "anchor_id": anchor["anchor_id"],
                "candidate_id": candidate["candidate_id"],
                "suite": anchor["suite"],
                "task_id": anchor["task_id"],
                "seed": anchor["seed"],
                "anchor_stage": anchor["stage"],
                "anchor_stage_id": anchor["stage_id"],
                "anchor_progress": anchor["progress"],
                "candidate_content_sha256": candidate["content_sha256"],
                "candidate_inference_seed": candidate["inference_seed"],
                "executed_candidate_steps": min(
                    len(records),
                    int(action_chunk.shape[0]),
                ),
                "total_executed_steps": len(records),
                "boundaries": boundaries,
                "progress_deltas": progress_deltas,
                "stage_transition": any(
                    int(item["stage_id"]) > int(anchor["stage_id"])
                    for item in records[:49]
                ),
                "stage_regression": any(
                    int(item["stage_id"]) < int(anchor["stage_id"])
                    for item in records[:49]
                ),
                "stagnation": abs(progress_deltas["progress_delta_49"])
                <= float(config["stagnation_threshold"]),
                "event_labels": events,
                "terminal_outcome": terminal_outcome.value,
                "recoverability": recoverability.value,
                "simulator_state_hash_before": snapshot.content_sha256,
                "simulator_state_hash_after": after_snapshot.content_sha256,
                "continuation_seed": (
                    int(config["continuation_seed_base"])
                    + list(anchor_by_id).index(str(anchor["anchor_id"]))
                ),
                "continuation_checkpoint_revision": candidate["checkpoint_revision"],
                "policy_queue_cleared_before_continuation": True,
                "continuation_reencoded_current_observation": True,
                "elapsed_seconds": time.perf_counter() - started,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "intervention": False,
            }
            attempt_path = attempt_root / f"{branch_id}-{time.time_ns()}.json"
            write_json(
                attempt_path,
                {
                    "schema_version": ("latentguard.lg_r2b0.counterfactual_branch.v1"),
                    "summary": summary,
                    "records": records,
                },
            )
            os.replace(attempt_path, final_path)
            write_json(
                completion_root / f"{branch_id}.json",
                {
                    "branch_id": branch_id,
                    "branch_sha256": sha256_path(final_path),
                    "anchor_id": anchor["anchor_id"],
                    "candidate_id": candidate["candidate_id"],
                },
            )
            branch_summaries.append(summary)
    except Exception as exc:
        errors.append(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if current_resources is not None:
            current_resources[0].close()
    expected_branches = len(jobs)
    completed_branches = len(branch_summaries)
    passed = not errors and completed_branches == expected_branches
    dataset_manifest = {
        "schema_version": "latentguard.lg_r2b0.dataset_manifest.v1",
        "status": "pass" if passed else "incomplete",
        "runtime_identity": runtime_identity(),
        "config_sha256": sha256_path(config_path),
        "anchor_registry_sha256": sha256_path(runtime / "anchor_registry.json"),
        "candidate_manifest_sha256": sha256_path(runtime / "candidate_manifest.json"),
        "candidate_diversity_report_sha256": sha256_path(
            runtime / "candidate_diversity_report.json"
        ),
        "expected_branches": expected_branches,
        "valid_branches": completed_branches,
        "valid_anchors": len({str(item["anchor_id"]) for item in branch_summaries}),
        "tasks": len(
            {(str(item["suite"]), int(item["task_id"])) for item in branch_summaries}
        ),
        "suites": len({str(item["suite"]) for item in branch_summaries}),
        "task6_branch_ratio": (
            sum(int(item["task_id"]) == 6 for item in branch_summaries)
            / completed_branches
            if completed_branches
            else 0.0
        ),
        "policy_generated_candidate_ratio": 1.0,
        "synthetic_candidate_ratio": 0.0,
        "branch_contamination": 0 if passed else len(errors),
        "candidate_metadata_completeness": 1.0,
        "policy_checkpoint_identity_completeness": 1.0,
        "short_horizons": list(config["short_horizons"]),
        "supported_local_events": [
            "OBJECT_DROP",
            "FAILED_PLACEMENT",
            "MISSED_GRASP",
            "LOSS_OF_CONTACT",
        ],
        "unsupported_local_events": [
            "WRONG_OBJECT",
            "WRONG_TARGET",
            "COLLISION",
        ],
        "common_continuation_per_anchor": True,
        "branches": branch_summaries,
        "branch_runtime_locator": "LG_R2B0_OUTPUT_ROOT/branches",
        "errors": errors,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "candidate_ranking_model_trained": False,
        "online_selection": False,
        "interventions": 0,
        "final_seeds_accessed": False,
    }
    path = runtime / "dataset_manifest.json"
    write_json(path, dataset_manifest)
    print(
        json.dumps(
            {
                "status": dataset_manifest["status"],
                "valid_branches": completed_branches,
                "expected_branches": expected_branches,
                "errors": len(errors),
                "output": str(path),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(8)


if __name__ == "__main__":
    main()
