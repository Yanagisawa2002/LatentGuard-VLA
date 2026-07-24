"""Freeze outcome-independent LG-R2b0 anchors before candidate generation."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2b0_common import (
    batch_observation,
    canonical_sha256,
    load_runtime_stack,
    make_libero_environment,
    observation_sha256,
    output_root,
    privileged_state,
    proprioception_sha256,
    read_yaml,
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

from latentguard.counterfactual.gates import validate_anchor_registry
from latentguard.counterfactual.libero_state import (
    LiberoStateSnapshot,
    capture_libero_state,
    save_libero_state,
)
from latentguard.counterfactual.models import AnchorRecord
from latentguard.progress.stages.libero_registry import LiberoStageRegistry


def _flatten_observation(
    value: Any,
    *,
    prefix: str = "",
    output: dict[str, np.ndarray],
) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child = f"{prefix}/{key}" if prefix else key
            _flatten_observation(value[key], prefix=child, output=output)
        return
    output[prefix] = np.ascontiguousarray(np.asarray(value))


def _save_observation(path: Path, raw: dict[str, Any]) -> str:
    arrays: dict[str, np.ndarray] = {}
    _flatten_observation(raw, output=arrays)
    np.savez_compressed(path, **arrays)
    return sha256_path(path)


def _public_anchor(record: AnchorRecord) -> dict[str, Any]:
    return {
        "anchor_id": record.anchor_id,
        "episode_id": record.episode_id,
        "task": record.task,
        "suite": record.suite,
        "task_id": record.task_id,
        "seed": record.seed,
        "step_index": record.step_index,
        "stage": record.stage,
        "stage_id": record.stage_id,
        "progress": record.progress,
        "simulator_state_hash": record.simulator_state_hash,
        "observation_hash": record.observation_hash,
        "proprioception_hash": record.proprioception_hash,
        "policy_observation_history_hash": (record.policy_observation_history_hash),
        "source_rollout_identity": dict(record.source_rollout_identity),
        "snapshot_locator": record.snapshot_locator,
    }


def _expand_source_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the compact, pre-registered task schedule deterministically."""
    jobs: list[dict[str, Any]] = []
    for schedule in config["task_schedules"]:
        suite = str(schedule["suite"])
        task_id = int(schedule["task_id"])
        for seed in schedule["seeds"]:
            seed_value = int(seed)
            identity = f"{suite}-task{task_id}-seed{seed_value}"
            jobs.append(
                {
                    **schedule,
                    "anchor_id": f"lg-r2b0-{identity}",
                    "episode_id": f"lg-r1b-{identity}",
                    "seed": seed_value,
                }
            )
    return jobs


def main() -> None:
    """Reproduce registered source episodes and freeze one anchor per episode."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r2b0/anchors.yaml"),
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    validate_execution_checkout(args.expected_commit)
    jobs = _expand_source_jobs(config)
    validate_no_final_seed(
        [int(job["seed"]) for job in jobs],
        "anchor source",
    )
    if len(jobs) != int(config["expected_anchors"]):
        raise ValueError("anchor source schedule does not match expected count")
    task_config_path = resolve_repo_path(Path(str(config["task_registry"])))
    task_config = read_yaml(task_config_path)
    registry = LiberoStageRegistry(task_config["tasks"])
    stage_config_path = resolve_repo_path(Path(str(config["stage_config"])))
    stage_config = read_yaml(stage_config_path)
    stack, preprocessor, postprocessor = load_runtime_stack()
    runtime = output_root()
    anchor_root = runtime / "anchors"
    anchor_root.mkdir(parents=True, exist_ok=True)
    registry_path = runtime / "anchor_registry.json"
    if registry_path.exists() and not args.resume:
        raise FileExistsError("anchor registry exists; pass --resume")
    existing: dict[str, dict[str, Any]] = {}
    if registry_path.is_file():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        existing = {str(item["anchor_id"]): item for item in payload.get("anchors", [])}
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for job in jobs:
        anchor_id = str(job["anchor_id"])
        if anchor_id in existing:
            records.append(existing[anchor_id])
            continue
        suite = str(job["suite"])
        task_id = int(job["task_id"])
        seed = int(job["seed"])
        episode_horizon = int(job["episode_horizon"])
        vector_env, single_env, env_pre, env_post, raw_batched = (
            make_libero_environment(
                suite=suite,
                task_id=task_id,
                episode_horizon=episode_horizon,
                seed=seed,
                stack=stack,
            )
        )
        try:
            import torch

            source_sampling_seed = int(config["source_policy_seed_base"]) + seed
            validate_no_final_seed(
                [source_sampling_seed],
                "anchor source policy",
            )
            torch.manual_seed(source_sampling_seed)
            torch.cuda.manual_seed_all(source_sampling_seed)
            stack.policy.reset()
            initial_state = privileged_state(
                single_env,
                unbatch_observation(raw_batched),
                terminal_success=False,
            )
            adapter = registry.resolve(suite, task_id)
            target_stage_ids = [int(value) for value in job["target_stage_priority"]]
            captured: dict[
                int,
                tuple[
                    int,
                    Any,
                    LiberoStateSnapshot,
                    dict[str, Any],
                ],
            ] = {}
            success = False
            for step_index in range(episode_horizon):
                raw = unbatch_observation(raw_batched)
                state = privileged_state(
                    single_env,
                    raw,
                    terminal_success=success,
                )
                label = adapter.label_step(
                    state,
                    {
                        "initial_privileged_state": initial_state,
                        "distance_thresholds": stage_config["distance_thresholds"],
                    },
                )
                enough_history = step_index >= int(config["minimum_anchor_step"])
                enough_future = episode_horizon - step_index >= int(
                    config["minimum_remaining_horizon"]
                )
                if (
                    label.stage_id in target_stage_ids
                    and label.stage_id not in captured
                    and enough_history
                    and enough_future
                ):
                    captured[label.stage_id] = (
                        step_index,
                        label,
                        capture_libero_state(single_env),
                        deepcopy(raw),
                    )
                _, action = select_primary_action(
                    stack=stack,
                    raw_observation=raw_batched,
                    instruction=str(job["instruction"]),
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                next_raw, _, terminated, info = step_native(single_env, action)
                success = bool(info["is_success"])
                raw_batched = batch_observation(next_raw)
                if terminated:
                    break
            selected_stage = next(
                (stage_id for stage_id in target_stage_ids if stage_id in captured),
                None,
            )
            if selected_stage is None:
                raise RuntimeError(
                    f"no registered target stage observed for {anchor_id}: "
                    f"{target_stage_ids}"
                )
            step_index, label, snapshot, raw = captured[selected_stage]
            anchor_dir = anchor_root / anchor_id
            if anchor_dir.exists():
                raise FileExistsError(f"partial anchor exists: {anchor_dir}")
            anchor_dir.mkdir()
            snapshot_path = anchor_dir / "state"
            save_libero_state(snapshot_path, snapshot)
            observation_path = anchor_dir / "observation.npz"
            observation_archive_sha256 = _save_observation(observation_path, raw)
            obs_hash = observation_sha256(raw, str(job["instruction"]))
            history_hash = canonical_sha256(
                {
                    "semantic": "current_observation_only_n_obs_steps_1",
                    "observation_sha256": obs_hash,
                    "policy_queue_reset_before_candidate": True,
                }
            )
            record = AnchorRecord(
                anchor_id=anchor_id,
                episode_id=str(job["episode_id"]),
                task=str(job["task_name"]),
                suite=suite,
                task_id=task_id,
                seed=seed,
                step_index=step_index,
                stage=label.stage_name,
                stage_id=int(label.stage_id),
                progress=float(label.overall_progress),
                simulator_state_hash=snapshot.content_sha256,
                observation_hash=obs_hash,
                proprioception_hash=proprioception_sha256(raw),
                policy_observation_history_hash=history_hash,
                source_rollout_identity={
                    "episode_id": str(job["episode_id"]),
                    "instruction": str(job["instruction"]),
                    "source_phase": str(job["source_phase"]),
                    "source_sampling_seed": source_sampling_seed,
                    "checkpoint_revision": task_config["source_policy"][
                        "checkpoint_revision"
                    ],
                    "processor_revision": task_config["source_policy"][
                        "processor_revision"
                    ],
                    "task_registry_sha256": sha256_path(task_config_path),
                    "stage_config_sha256": sha256_path(stage_config_path),
                    "selection_used_candidate_results": False,
                    "selection_used_branch_outcomes": False,
                    "observation_archive_sha256": observation_archive_sha256,
                    "initial_privileged_state": initial_state,
                },
                snapshot_locator=(f"LG_R2B0_OUTPUT_ROOT/anchors/{anchor_id}/state"),
            )
            records.append(_public_anchor(record))
        except Exception as exc:
            errors.append(
                {
                    "anchor_id": anchor_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        finally:
            vector_env.close()
        partial_payload = {
            "schema_version": "latentguard.lg_r2b0.anchor_registry.v1",
            "status": "building",
            "frozen_before_candidate_results": True,
            "runtime_identity": runtime_identity(),
            "config_sha256": sha256_path(config_path),
            "anchors": records,
            "errors": errors,
        }
        write_json(registry_path, partial_payload)
    validation = None
    if not errors and len(records) == len(jobs):
        validation = validate_anchor_registry(
            records,
            expected_anchors=int(config["expected_anchors"]),
            minimum_tasks=int(config["minimum_tasks"]),
            minimum_suites=int(config["minimum_suites"]),
            minimum_per_task=int(config["minimum_per_task"]),
            maximum_task6_ratio=float(config["maximum_task6_ratio"]),
        )
    passed = validation is not None
    payload = {
        "schema_version": "latentguard.lg_r2b0.anchor_registry.v1",
        "status": "pass" if passed else "fail",
        "frozen_before_candidate_results": True,
        "candidate_results_available_during_selection": False,
        "branch_outcomes_available_during_selection": False,
        "runtime_identity": runtime_identity(),
        "config_sha256": sha256_path(config_path),
        "source_task_registry_sha256": sha256_path(task_config_path),
        "source_stage_config_sha256": sha256_path(stage_config_path),
        "selection_semantic": "pre_registered_stage_priority_first_occurrence_v1",
        "validation": validation,
        "anchors": records,
        "errors": errors,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "final_seeds_accessed": False,
    }
    write_json(registry_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "anchors": len(records),
                "errors": len(errors),
                "output": str(registry_path),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
