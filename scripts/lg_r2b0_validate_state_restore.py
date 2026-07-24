"""Run the pre-registered repeated LIBERO restore determinism gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2b0_common import (
    batch_observation,
    compare_rendered_observations,
    current_observation,
    generate_policy_candidate,
    load_runtime_stack,
    make_libero_environment,
    output_root,
    privileged_state,
    read_yaml,
    rendered_observation_sha256,
    resolve_repo_path,
    runtime_identity,
    select_primary_action,
    sha256_path,
    step_native,
    unbatch_observation,
    validate_execution_checkout,
    validate_no_final_seed,
    write_json,
)

from latentguard.counterfactual.libero_state import (
    LIBERO_STATE_COMPARISON,
    capture_libero_state,
    compare_libero_states,
    restore_libero_state,
)
from latentguard.progress.stages.libero_registry import LiberoStageRegistry


def _trace_state(
    single_env: Any,
    raw: dict[str, Any],
    *,
    initial_state: dict[str, Any],
    adapter: Any,
    distance_thresholds: dict[str, Any],
    terminal_success: bool,
    executed_action: np.ndarray,
) -> dict[str, Any]:
    state = privileged_state(
        single_env,
        raw,
        terminal_success=terminal_success,
    )
    label = adapter.label_step(
        state,
        {
            "initial_privileged_state": initial_state,
            "distance_thresholds": distance_thresholds,
        },
    )
    control_env = single_env._env
    task_env = control_env.env
    numeric = [
        np.asarray(task_env.sim.data.qpos, dtype=np.float64).reshape(-1),
        np.asarray(task_env.sim.data.qvel, dtype=np.float64).reshape(-1),
        np.asarray(state["eef_position"], dtype=np.float64).reshape(-1),
    ]
    for name in sorted(state["objects"]):
        item = state["objects"][name]
        numeric.extend(
            (
                np.asarray(item["position"], dtype=np.float64).reshape(-1),
                np.asarray(item["quaternion"], dtype=np.float64).reshape(-1),
            )
        )
    predicates = tuple(
        (
            tuple(str(value) for value in item["predicate"]),
            bool(item["satisfied"]),
        )
        for item in state["goal_predicates"]
    )
    return {
        "numeric": np.concatenate(numeric).tolist(),
        "executed_action": np.asarray(executed_action, dtype=np.float64).tolist(),
        "task_predicates": predicates,
        "progress": float(label.overall_progress),
        "stage_id": int(label.stage_id),
        "stage_name": label.stage_name,
        "terminal_success": terminal_success,
        "render_sha256": rendered_observation_sha256(raw),
    }


def _compare_trace(
    expected: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    *,
    atol: float,
) -> dict[str, Any]:
    if len(expected) != len(observed):
        return {
            "passed": False,
            "maximum_absolute_error": None,
            "mismatch": "trace length",
        }
    maximum = 0.0
    for step, (left, right) in enumerate(zip(expected, observed, strict=True)):
        left_numeric = np.asarray(
            left["numeric"] + left["executed_action"],
            dtype=np.float64,
        )
        right_numeric = np.asarray(
            right["numeric"] + right["executed_action"],
            dtype=np.float64,
        )
        if left_numeric.shape != right_numeric.shape:
            return {
                "passed": False,
                "maximum_absolute_error": None,
                "mismatch": f"numeric shape at step {step}",
            }
        if left_numeric.size:
            maximum = max(
                maximum,
                float(np.max(np.abs(left_numeric - right_numeric))),
            )
        for key in (
            "task_predicates",
            "stage_id",
            "stage_name",
            "terminal_success",
            "render_sha256",
        ):
            if left[key] != right[key]:
                return {
                    "passed": False,
                    "maximum_absolute_error": maximum,
                    "mismatch": f"{key} at step {step}",
                }
        maximum = max(maximum, abs(float(left["progress"]) - float(right["progress"])))
    return {
        "passed": maximum <= atol,
        "maximum_absolute_error": maximum,
        "mismatch": None if maximum <= atol else "numeric tolerance",
    }


def main() -> None:
    """Validate 10 states with five identical restored executions each."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r2b0/state_restore.yaml"),
    )
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    validate_execution_checkout(args.expected_commit)
    probes = list(config["probes"])
    repeats = int(config["repeats"])
    if len(probes) < 10 or repeats < 5:
        raise ValueError("restore gate requires at least 10 states and five repeats")
    validate_no_final_seed(
        [int(item["seed"]) for item in probes],
        "restore environment",
    )
    validate_no_final_seed(
        [int(item["candidate_inference_seed"]) for item in probes],
        "restore candidate inference",
    )
    task_registry = read_yaml(resolve_repo_path(Path(str(config["task_registry"]))))
    stage_registry = LiberoStageRegistry(task_registry["tasks"])
    stage_config = read_yaml(resolve_repo_path(Path(str(config["stage_config"]))))
    stack, preprocessor, postprocessor = load_runtime_stack()
    tolerance = float(config["numeric_tolerance"])
    pixel_maximum_tolerance = float(config["pixel_maximum_absolute_tolerance"])
    pixel_mean_tolerance = float(config["pixel_mean_absolute_tolerance"])
    pixel_ratio_tolerance = float(config["pixel_different_value_ratio_tolerance"])
    probe_results: list[dict[str, Any]] = []
    restore_failures = 0
    render_mismatches = 0
    post_render_state_mismatches = 0
    terminal_mismatches = 0
    predicate_mismatches = 0
    maximum_error = 0.0
    maximum_pixel_error = 0.0
    maximum_pixel_mean_error = 0.0
    maximum_pixel_ratio = 0.0
    for probe_index, probe in enumerate(probes):
        suite = str(probe["suite"])
        task_id = int(probe["task_id"])
        seed = int(probe["seed"])
        step_index = int(probe["step_index"])
        vector_env, single_env, env_pre, env_post, raw_batched = (
            make_libero_environment(
                suite=suite,
                task_id=task_id,
                episode_horizon=int(probe["episode_horizon"]),
                seed=seed,
                stack=stack,
            )
        )
        try:
            import torch

            primary_seed = int(config["source_policy_seed_base"]) + seed
            torch.manual_seed(primary_seed)
            torch.cuda.manual_seed_all(primary_seed)
            stack.policy.reset()
            initial_state = privileged_state(
                single_env,
                unbatch_observation(raw_batched),
                terminal_success=False,
            )
            for _ in range(step_index):
                _, action = select_primary_action(
                    stack=stack,
                    raw_observation=raw_batched,
                    instruction=str(probe["instruction"]),
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                raw, _, terminated, _ = step_native(single_env, action)
                if terminated:
                    raise RuntimeError("restore probe source terminated before anchor")
                raw_batched = batch_observation(raw)
            anchor_raw = unbatch_observation(raw_batched)
            snapshot = capture_libero_state(single_env)
            candidate = generate_policy_candidate(
                stack=stack,
                raw_observation=raw_batched,
                instruction=str(probe["instruction"]),
                env_preprocessor=env_pre,
                env_postprocessor=env_post,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                anchor_id=f"restore-probe-{probe_index}",
                candidate_id=f"restore-probe-{probe_index}-candidate",
                inference_seed=int(probe["candidate_inference_seed"]),
            )
            adapter = stage_registry.resolve(suite, task_id)
            traces: list[list[dict[str, Any]]] = []
            terminals: list[dict[str, Any]] = []
            restoration_records: list[dict[str, Any]] = []
            render_records: list[dict[str, float | int | bool]] = []
            post_render_state_records: list[dict[str, Any]] = []
            for _repeat in range(repeats):
                restoration = restore_libero_state(
                    single_env,
                    snapshot,
                    atol=tolerance,
                )
                restoration_records.append(asdict(restoration))
                if not restoration.within_tolerance:
                    restore_failures += 1
                    break
                current = current_observation(single_env)
                render = compare_rendered_observations(anchor_raw, current)
                render_records.append(render)
                maximum_pixel_error = max(
                    maximum_pixel_error,
                    float(render["maximum_absolute_error"]),
                )
                maximum_pixel_mean_error = max(
                    maximum_pixel_mean_error,
                    float(render["mean_absolute_error"]),
                )
                maximum_pixel_ratio = max(
                    maximum_pixel_ratio,
                    float(render["different_value_ratio"]),
                )
                render_passed = (
                    float(render["maximum_absolute_error"]) <= pixel_maximum_tolerance
                    and float(render["mean_absolute_error"]) <= pixel_mean_tolerance
                    and float(render["different_value_ratio"]) <= pixel_ratio_tolerance
                )
                if not render_passed:
                    render_mismatches += 1
                    break
                post_render_state = compare_libero_states(
                    snapshot,
                    capture_libero_state(single_env),
                    atol=tolerance,
                )
                post_render_state_records.append(asdict(post_render_state))
                if not post_render_state.within_tolerance:
                    post_render_state_mismatches += 1
                    break
                raw_batched = batch_observation(current)
                trace: list[dict[str, Any]] = []
                success = False
                terminated = False
                continuation_initialized = False
                for local_step in range(int(config["probe_horizon"])):
                    if local_step < candidate.native_action_chunk.shape[0]:
                        action = candidate.native_action_chunk[local_step]
                    else:
                        if not continuation_initialized:
                            stack.policy.reset()
                            continuation_seed = (
                                int(config["continuation_seed_base"]) + probe_index
                            )
                            validate_no_final_seed(
                                [continuation_seed],
                                "restore continuation",
                            )
                            torch.manual_seed(continuation_seed)
                            torch.cuda.manual_seed_all(continuation_seed)
                            continuation_initialized = True
                        _, action = select_primary_action(
                            stack=stack,
                            raw_observation=raw_batched,
                            instruction=str(probe["instruction"]),
                            env_preprocessor=env_pre,
                            env_postprocessor=env_post,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )
                    raw, _, terminated, info = step_native(single_env, action)
                    success = bool(info["is_success"])
                    trace.append(
                        _trace_state(
                            single_env,
                            raw,
                            initial_state=initial_state,
                            adapter=adapter,
                            distance_thresholds=stage_config["distance_thresholds"],
                            terminal_success=success,
                            executed_action=action,
                        )
                    )
                    raw_batched = batch_observation(raw)
                    if terminated:
                        break
                traces.append(trace)
                terminals.append(
                    {
                        "success": success,
                        "terminated": terminated,
                        "steps": len(trace),
                        "final_predicates": (
                            trace[-1]["task_predicates"] if trace else ()
                        ),
                    }
                )
            comparisons = []
            for repeat in range(1, len(traces)):
                comparison = _compare_trace(
                    traces[0],
                    traces[repeat],
                    atol=tolerance,
                )
                comparisons.append(comparison)
                if comparison["maximum_absolute_error"] is not None:
                    maximum_error = max(
                        maximum_error,
                        float(comparison["maximum_absolute_error"]),
                    )
                if not comparison["passed"]:
                    if (
                        comparison["mismatch"]
                        and "task_predicates" in comparison["mismatch"]
                    ):
                        predicate_mismatches += 1
                    else:
                        restore_failures += 1
            if terminals and any(item != terminals[0] for item in terminals[1:]):
                terminal_mismatches += 1
            probe_results.append(
                {
                    "probe_id": f"restore-probe-{probe_index}",
                    "suite": suite,
                    "task_id": task_id,
                    "seed": seed,
                    "step_index": step_index,
                    "snapshot_content_sha256": snapshot.content_sha256,
                    "snapshot_structure_sha256": snapshot.structure_sha256,
                    "candidate_content_sha256": candidate.content_sha256,
                    "restorations": restoration_records,
                    "render_comparisons": render_records,
                    "post_render_state_comparisons": post_render_state_records,
                    "trace_comparisons": comparisons,
                    "terminal_results": terminals,
                }
            )
        finally:
            vector_env.close()
    passed = (
        restore_failures == 0
        and render_mismatches == 0
        and post_render_state_mismatches == 0
        and terminal_mismatches == 0
        and predicate_mismatches == 0
        and len(probe_results) == len(probes)
    )
    payload = {
        "schema_version": "latentguard.lg_r2b0.state_restore_validation.v1",
        "status": "pass" if passed else "fail",
        "stop_reason": None if passed else "STATE_RESTORE_DETERMINISM_GATE_FAILED",
        "runtime_identity": runtime_identity(),
        "config_sha256": sha256_path(config_path),
        "comparison_semantic": LIBERO_STATE_COMPARISON,
        "numeric_tolerance": tolerance,
        "pixel_comparison": "complete_array_error_statistics_v1",
        "pixel_tolerances": {
            "maximum_absolute_error": pixel_maximum_tolerance,
            "mean_absolute_error": pixel_mean_tolerance,
            "different_value_ratio": pixel_ratio_tolerance,
        },
        "probe_states": len(probes),
        "repeats_per_state": repeats,
        "probe_horizon": int(config["probe_horizon"]),
        "restore_failures": restore_failures,
        "render_mismatches": render_mismatches,
        "post_render_state_mismatches": post_render_state_mismatches,
        "terminal_result_mismatches": terminal_mismatches,
        "task_predicate_mismatches": predicate_mismatches,
        "maximum_observed_numeric_error": maximum_error,
        "maximum_observed_pixel_absolute_error": maximum_pixel_error,
        "maximum_observed_pixel_mean_absolute_error": maximum_pixel_mean_error,
        "maximum_observed_pixel_different_value_ratio": maximum_pixel_ratio,
        "probe_results": probe_results,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "final_seeds_accessed": False,
    }
    path = output_root() / "state_restore_validation.json"
    write_json(path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "restore_failures": restore_failures,
                "render_mismatches": render_mismatches,
                "post_render_state_mismatches": post_render_state_mismatches,
                "terminal_mismatches": terminal_mismatches,
                "predicate_mismatches": predicate_mismatches,
                "output": str(path),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
