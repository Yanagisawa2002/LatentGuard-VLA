"""Run the frozen single-episode or 40-episode LIBERO smoke schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from _lg_r0_runtime import (
    configure_libero_assets,
    cuda_memory,
    instrument_policy,
    load_processors,
    load_stack,
    output_root,
    read_json,
    sha256_path,
    write_json,
)


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _error_category(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "checkpoint" in text or "state_dict" in text or "safetensor" in text:
        return "CHECKPOINT_MISMATCH"
    if "processor" in text or "normaliz" in text:
        return "PROCESSOR_FAILURE"
    if "action" in text or "shape" in text or "bound" in text:
        return "ACTION_CONTRACT_FAILURE"
    if "libero" in text or "mujoco" in text or "egl" in text or "environment" in text:
        return "ENVIRONMENT_FAILURE"
    return "UNKNOWN"


def main() -> None:
    """Execute a fixed evaluation schedule without training or intervention."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite-name")
    parser.add_argument(
        "--execution-site",
        choices=("local", "remote"),
        default="local",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout

    config = read_json(args.config)
    libero_assets = configure_libero_assets()
    torch.cuda.reset_peak_memory_stats()
    stack = load_stack()
    preprocessor, postprocessor = load_processors(stack)
    instrumentation = instrument_policy(stack.policy)
    run_id = f"{args.config.stem}-{int(time.time())}"
    run_dir = output_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    episodes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    suite_results: dict[str, dict[str, Any]] = {}

    try:
        all_suites = list(config["suites"])
        selected_suites = [
            suite
            for suite in all_suites
            if args.suite_name is None or suite["name"] == args.suite_name
        ]
        if not selected_suites:
            raise ValueError(f"suite not found in config: {args.suite_name}")
        for suite in selected_suites:
            suite_index = all_suites.index(suite)
            env_config = LiberoEnv(
                task=suite["name"],
                task_ids=suite["task_ids"],
                episode_length=suite["episode_horizon"],
                obs_type="pixels_agent_pos",
                camera_name=",".join(config["camera_names"]),
                observation_height=config["observation_height"],
                observation_width=config["observation_width"],
                init_states=config["init_states"],
                control_mode=config["control_mode"],
            )
            env_map = make_env(env_config, n_envs=1, use_async_envs=False)
            env = env_map[suite["name"]][suite["task_ids"][0]]
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=env_config,
                policy_cfg=stack.config,
            )
            suite_episodes: list[dict[str, Any]] = []
            try:
                instruction = str(list(env.call("task_description"))[0])
                task_name = str(list(env.call("task"))[0])
                for episode_index, seed in enumerate(suite["seeds"]):
                    sampling_seed = (
                        int(config["policy_sampling_seed"])
                        + suite_index * 100_000
                        + int(seed)
                    )
                    torch.manual_seed(sampling_seed)
                    torch.cuda.manual_seed_all(sampling_seed)
                    query_start = instrumentation.policy_queries
                    chunk_start = len(instrumentation.chunks)
                    started = time.perf_counter()
                    try:
                        data = rollout(
                            env=env,
                            policy=stack.policy,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            seeds=[seed],
                            return_observations=False,
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "suite": suite["name"],
                                "task": task_name,
                                "seed": str(seed),
                                "category": _error_category(exc),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        raise
                    elapsed = time.perf_counter() - started
                    done = data["done"][0]
                    first_done = int(torch.argmax(done.to(torch.int64)).item())
                    episode_length = first_done + 1
                    actions = data["action"][0, :episode_length].to(torch.float32)
                    success = bool(data["success"][0, :episode_length].any().item())
                    finite = bool(torch.isfinite(actions).all().item())
                    bounds_ok = bool(
                        ((actions >= -1.0) & (actions <= 1.0)).all().item()
                    )
                    action_array = actions.cpu().numpy()
                    episode_id = (
                        f"{suite['name']}-task{suite['task_ids'][0]}-"
                        f"seed{seed}-episode{episode_index}"
                    )
                    trace = {
                        "schema_version": "latentguard.lg_r0.episode_trace.v1",
                        "episode_id": episode_id,
                        "suite": suite["name"],
                        "task": task_name,
                        "task_id": suite["task_ids"][0],
                        "seed": seed,
                        "policy_sampling_seed": sampling_seed,
                        "instruction": instruction,
                        "observation_keys": [
                            "observation.images.image",
                            "observation.images.image2",
                            "observation.state",
                        ],
                        "generated_action_chunks": instrumentation.chunks[chunk_start:],
                        "executed_actions": action_array.tolist(),
                        "action_mask": None,
                        "action_mask_semantic": "not_emitted_by_native_inference",
                        "checkpoint_revision": config["checkpoint_revision"],
                        "processor_revision": config["processor_revision"],
                        "success": success,
                        "termination": "success" if success else "horizon_exhausted",
                        "episode_length": episode_length,
                        "optimizer_steps": 0,
                        "backward_calls": 0,
                    }
                    trace_path = run_dir / "traces" / f"{episode_id}.json"
                    write_json(trace_path, trace)
                    action_sha = hashlib.sha256(
                        np.ascontiguousarray(action_array).tobytes()
                    ).hexdigest()
                    episode = {
                        "episode_id": episode_id,
                        "suite": suite["name"],
                        "task": task_name,
                        "task_id": suite["task_ids"][0],
                        "seed": seed,
                        "policy_sampling_seed": sampling_seed,
                        "success": success,
                        "termination": "success" if success else "horizon_exhausted",
                        "episode_length": episode_length,
                        "elapsed_seconds": elapsed,
                        "policy_queries": instrumentation.policy_queries - query_start,
                        "action_finite": finite,
                        "executed_action_bounds_ok": bounds_ok,
                        "executed_action_sha256_float32": action_sha,
                        "trace_relative_path": str(trace_path.relative_to(run_dir)),
                        "trace_bytes": trace_path.stat().st_size,
                        "trace_sha256": sha256_path(trace_path),
                    }
                    suite_episodes.append(episode)
                    episodes.append(episode)
            finally:
                env.close()

            successes = sum(int(item["success"]) for item in suite_episodes)
            suite_results[suite["name"]] = {
                "episodes": len(suite_episodes),
                "successes": successes,
                "success_rate": successes / len(suite_episodes),
                "wilson_95": _wilson(successes, len(suite_episodes)),
                "mean_episode_length": float(
                    np.mean([item["episode_length"] for item in suite_episodes])
                ),
            }
    except Exception:
        pass

    successes = sum(int(item["success"]) for item in episodes)
    nonfinite = sum(int(not item["action_finite"]) for item in episodes)
    action_errors = sum(int(not item["executed_action_bounds_ok"]) for item in episodes)
    evaluated_suites = [
        suite
        for suite in config["suites"]
        if args.suite_name is None or suite["name"] == args.suite_name
    ]
    expected_episodes = sum(len(suite["seeds"]) for suite in evaluated_suites)
    status = (
        "pass"
        if len(episodes) == expected_episodes
        and not errors
        and nonfinite == 0
        and action_errors == 0
        else "fail"
    )
    payload = {
        "schema_version": "latentguard.lg_r0.libero_evaluation.v1",
        "status": status,
        "result_scope": (
            "one_episode_smoke"
            if expected_episodes == 1
            else (
                "suite_shard"
                if args.suite_name is not None
                else "local_40_episode_smoke"
            )
        ),
        "official_400_episode_equivalent": False,
        "execution_site": args.execution_site,
        "libero_asset_snapshot_name": libero_assets.name,
        "schedule": config,
        "episodes": episodes,
        "suite_results": suite_results,
        "overall": {
            "expected_episodes": expected_episodes,
            "completed_episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
            "wilson_95": _wilson(successes, len(episodes)),
            "mean_episode_length": (
                float(np.mean([item["episode_length"] for item in episodes]))
                if episodes
                else None
            ),
            "timeouts": sum(
                int(item["termination"] == "horizon_exhausted") for item in episodes
            ),
            "environment_errors": sum(
                int(item["category"] == "ENVIRONMENT_FAILURE") for item in errors
            ),
            "action_contract_errors": action_errors,
            "nonfinite_action_episodes": nonfinite,
        },
        "instrumentation": instrumentation.summary(),
        "cuda_memory": cuda_memory(),
        "errors": errors,
        "run_id": run_id,
        "run_locator": f"LG_R0_OUTPUT_ROOT/{run_id}",
        "optimizer_steps": instrumentation.optimizer_steps,
        "backward_calls": instrumentation.backward_calls,
        "interventions": 0,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "completed": len(episodes),
                "expected": expected_episodes,
                "successes": successes,
                "output": str(args.output),
            }
        )
    )
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
