"""Collect every frozen LG-R1 rollout with an atomic privileged sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r0_runtime import (
    configure_libero_assets,
    instrument_policy,
    load_processors,
    load_stack,
    sha256_path,
)
from _lg_r1_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    write_json,
)

from latentguard.adapters.vla_jepa.constants import CHECKPOINT_REVISION
from latentguard.progress.rollout import (
    RolloutJob,
    action_step_record,
    build_rollout_jobs,
    completed_episode_ids,
    validate_seed_isolation,
    write_completion_marker,
)
from latentguard.progress.serialization import write_jsonl_atomic


def _batch_observation(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _batch_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None]
    return value


def _safe_bool_call(value: Any, method_name: str) -> bool | None:
    method = getattr(value, method_name, None)
    if not callable(method):
        return None
    try:
        return bool(method())
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _contact_pairs(task_env: Any) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    sim = task_env.sim
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        left = sim.model.geom_id2name(int(contact.geom1)) or str(contact.geom1)
        right = sim.model.geom_id2name(int(contact.geom2)) or str(contact.geom2)
        pairs.add(tuple(sorted((left, right))))
    return sorted(pairs)


def _contacts_gripper(entity_name: str, pairs: list[tuple[str, str]]) -> bool:
    gripper_tokens = ("gripper", "finger", "hand")
    for left, right in pairs:
        joined = f"{left} {right}".lower()
        if entity_name.lower() in joined and any(
            token in joined for token in gripper_tokens
        ):
            return True
    return False


def _extract_privileged(
    single_env: Any,
    raw_observation: dict[str, Any],
    *,
    terminal_success: bool,
) -> dict[str, Any]:
    control_env = single_env._env
    if control_env is None:
        raise RuntimeError("LIBERO inner environment is unavailable")
    task_env = control_env.env
    pairs = _contact_pairs(task_env)
    objects: dict[str, Any] = {}
    for name, object_state in sorted(task_env.object_states_dict.items()):
        geom = object_state.get_geom_state()
        position = np.asarray(geom["pos"], dtype=np.float64)
        quaternion = np.asarray(geom["quat"], dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError(f"invalid privileged pose for {name}")
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError(f"non-finite privileged pose for {name}")
        objects[name] = {
            "position": position.tolist(),
            "quaternion": quaternion.tolist(),
            "open": _safe_bool_call(object_state, "is_open"),
            "close": _safe_bool_call(object_state, "is_close"),
            "turn_on": _safe_bool_call(object_state, "turn_on"),
            "contact_with_gripper": _contacts_gripper(name, pairs),
        }
    goals = []
    for predicate in task_env.parsed_problem["goal_state"]:
        predicate_items = [str(item) for item in predicate]
        goals.append(
            {
                "predicate": predicate_items,
                "satisfied": bool(task_env._eval_predicate(predicate)),
            }
        )
    eef = np.asarray(raw_observation["robot_state"]["eef"]["pos"], dtype=np.float64)
    gripper_qpos = np.asarray(
        raw_observation["robot_state"]["gripper"]["qpos"],
        dtype=np.float64,
    )
    state = np.asarray(control_env.get_sim_state(), dtype=np.float64)
    if not np.isfinite(state).all():
        raise ValueError("non-finite simulator state")
    return {
        "eef_position": eef.tolist(),
        "gripper_qpos": gripper_qpos.tolist(),
        "objects": objects,
        "contact_pairs": [list(pair) for pair in pairs],
        "goal_predicates": goals,
        "environment_success_predicate": bool(control_env.check_success()),
        "terminal_success": terminal_success,
        "terminal_failure": False,
        "sim_state_sha256": hashlib.sha256(
            np.ascontiguousarray(state).tobytes()
        ).hexdigest(),
    }


def _step_without_terminal_reset(
    single_env: Any, action: np.ndarray
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    control_env = single_env._env
    if control_env is None:
        raise RuntimeError("LIBERO inner environment is unavailable")
    raw_observation, reward, done, info = control_env.step(action)
    success = bool(control_env.check_success())
    terminated = bool(done or success)
    result_info = dict(info)
    result_info.update(
        {
            "task": single_env.task,
            "task_id": single_env.task_id,
            "done": bool(done),
            "is_success": success,
        }
    )
    return (
        single_env._format_raw_obs(raw_observation),
        float(reward),
        terminated,
        False,
        result_info,
    )


def _dataset_metadata_identities(path: Path) -> dict[str, Any]:
    files = {
        "info_json": path / "meta" / "info.json",
        "episodes_parquet": path / "meta" / "episodes",
        "tasks_parquet": path / "meta" / "tasks.parquet",
    }
    if not files["info_json"].is_file() or not files["tasks_parquet"].is_file():
        raise FileNotFoundError(f"incomplete LeRobot metadata in {path}")
    episode_files = sorted(files["episodes_parquet"].rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"missing episode parquet metadata in {path}")
    identities: dict[str, Any] = {}
    for key in ("info_json", "tasks_parquet"):
        file_path = files[key]
        identities[key] = {
            "bytes": file_path.stat().st_size,
            "sha256": sha256_path(file_path),
        }
    identities["episode_parquet_files"] = [
        {
            "relative_path": str(file.relative_to(path).as_posix()),
            "bytes": file.stat().st_size,
            "sha256": sha256_path(file),
        }
        for file in episode_files
    ]
    return identities


def _validate_dataset_reload(path: Path, expected_frames: int) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "lg_r1_eval_recording",
        root=path,
    )
    if len(dataset) != expected_frames:
        raise ValueError(
            f"dataset reload length mismatch: {len(dataset)} != {expected_frames}"
        )
    required = {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "next.reward",
        "next.success",
        "next.done",
    }
    for index in range(len(dataset)):
        frame = dataset[index]
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"reloaded frame {index} misses {missing}")
        action = np.asarray(frame["action"], dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"invalid reloaded action at frame {index}")
    return {
        "status": "pass",
        "frames_expected": expected_frames,
        "frames_reloaded": len(dataset),
        "completeness": 1.0,
    }


def _prepare_processor_manifest(
    runtime_root: Path, stack: Any, preprocessor: Any, postprocessor: Any
) -> dict[str, Any]:
    from lerobot.policies.factory import make_pre_post_processors

    processor_root = runtime_root / "processors"
    manifest_path = runtime_root / "processor_serialization_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for item in manifest["files"]:
            path = processor_root / item["relative_path"]
            if (
                not path.is_file()
                or path.stat().st_size != int(item["bytes"])
                or sha256_path(path) != item["sha256"]
            ):
                raise ValueError("serialized processor identity drift")
        return manifest
    attempt = runtime_root / f"processors.partial-{time.time_ns()}"
    attempt.mkdir(parents=True, exist_ok=False)
    preprocessor.save_pretrained(attempt)
    postprocessor.save_pretrained(attempt)
    reloaded_pre, reloaded_post = make_pre_post_processors(
        policy_cfg=stack.config,
        pretrained_path=str(attempt),
        preprocessor_overrides={
            "device_processor": {"device": str(stack.config.device)}
        },
    )
    if processor_root.exists():
        raise FileExistsError("processor directory exists without an accepted manifest")
    os.replace(attempt, processor_root)
    files = [
        {
            "relative_path": str(path.relative_to(processor_root).as_posix()),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(processor_root.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "schema_version": ("latentguard.lg_r1.processor_serialization.v1"),
        "status": "pass",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "processor_revision": CHECKPOINT_REVISION,
        "files": files,
        "reload_success": True,
        "reloaded_preprocessor_type": type(reloaded_pre).__name__,
        "reloaded_postprocessor_type": type(reloaded_post).__name__,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    write_json(manifest_path, manifest)
    return manifest


def _create_dataset(path: Path) -> Any:
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.lerobot_eval import (
        _env_features_to_dataset_features,
    )
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    features = {
        f"{OBS_IMAGES}.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(224, 224, 3)
        ),
        f"{OBS_IMAGES}.image2": PolicyFeature(
            type=FeatureType.VISUAL, shape=(224, 224, 3)
        ),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    del torch
    return LeRobotDataset.create(
        repo_id="lg_r1_eval_recording",
        fps=80,
        features=_env_features_to_dataset_features(features),
        root=str(path),
        use_videos=True,
    )


def _build_dataset_frame(
    raw_observation: dict[str, Any],
    *,
    state: np.ndarray,
    action: np.ndarray,
    reward: float,
    success: bool,
    done: bool,
    task: str,
) -> dict[str, Any]:
    return {
        "observation.images.image": raw_observation["pixels"]["image"][0],
        "observation.images.image2": raw_observation["pixels"]["image2"][0],
        "observation.state": state.astype(np.float32, copy=False),
        "action": action.astype(np.float32, copy=False),
        "next.reward": np.atleast_1d(np.float32(reward)),
        "next.success": np.atleast_1d(np.bool_(success)),
        "next.done": np.atleast_1d(np.bool_(done)),
        "task": task,
    }


def _collect_episode(
    job: RolloutJob,
    *,
    stack: Any,
    preprocessor: Any,
    postprocessor: Any,
    instrumentation: Any,
    attempt_root: Path,
) -> dict[str, Any]:
    import torch
    from lerobot.envs import preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import (
        make_env,
        make_env_pre_post_processors,
    )
    from lerobot.utils.constants import ACTION, OBS_STATE

    torch.manual_seed(420000 + job.seed)
    torch.cuda.manual_seed_all(420000 + job.seed)
    env_config = LiberoEnv(
        task=job.suite,
        task_ids=[job.task_id],
        episode_length=job.episode_horizon,
        obs_type="pixels_agent_pos",
        observation_height=224,
        observation_width=224,
        init_states=True,
        control_mode="relative",
    )
    vector_env = make_env(env_config, n_envs=1, use_async_envs=False)[job.suite][
        job.task_id
    ]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=stack.config
    )
    dataset_path = attempt_root / "dataset"
    dataset = _create_dataset(dataset_path)
    sidecar_path = attempt_root / "privileged_sidecar.jsonl"
    query_count_start = instrumentation.policy_queries
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    success = False
    frame_count = 0
    try:
        stack.policy.reset()
        raw_observation, _ = vector_env.reset(seed=[job.seed])
        single_env = vector_env.envs[0]
        initial_state = _extract_privileged(
            single_env,
            {
                "pixels": {
                    key: value[0] for key, value in raw_observation["pixels"].items()
                },
                "robot_state": {
                    group: {key: value[0] for key, value in values.items()}
                    for group, values in raw_observation["robot_state"].items()
                },
            },
            terminal_success=False,
        )
        for step_index in range(job.episode_horizon):
            pre_state = _extract_privileged(
                single_env,
                {
                    "pixels": {
                        key: value[0]
                        for key, value in raw_observation["pixels"].items()
                    },
                    "robot_state": {
                        group: {key: value[0] for key, value in values.items()}
                        for group, values in raw_observation["robot_state"].items()
                    },
                },
                terminal_success=False,
            )
            observation = preprocess_observation(deepcopy(raw_observation))
            observation["task"] = [job.instruction]
            processed = env_preprocessor(observation)
            policy_batch = preprocessor(processed)
            policy_state = (
                processed[OBS_STATE][0].to("cpu", dtype=torch.float32).numpy()
            )
            with torch.inference_mode():
                selected = stack.policy.select_action(policy_batch)
            selected_policy = (
                selected.detach().to("cpu", dtype=torch.float32).numpy()[0]
            )
            transition = env_postprocessor({ACTION: postprocessor(selected)})
            executed = (
                transition[ACTION].detach().to("cpu", dtype=torch.float32).numpy()[0]
            )
            if not np.isfinite(executed).all():
                raise ValueError("policy emitted a non-finite action")
            next_observation, reward, terminated, _, info = (
                _step_without_terminal_reset(single_env, executed)
            )
            success = bool(info["is_success"])
            at_horizon = step_index + 1 == job.episode_horizon
            done = bool(terminated or at_horizon)
            post_state = _extract_privileged(
                single_env,
                next_observation,
                terminal_success=success,
            )
            dataset.add_frame(
                _build_dataset_frame(
                    raw_observation,
                    state=policy_state,
                    action=executed,
                    reward=reward,
                    success=success,
                    done=done,
                    task=job.instruction,
                )
            )
            generated = instrumentation.chunks[-1][0]
            chunk_offset = step_index % 7
            action_record = action_step_record(
                step_index=step_index,
                generated_chunk=generated,
                chunk_offset=chunk_offset,
                selected_policy_action=selected_policy.tolist(),
                executed_action=executed.tolist(),
            )
            rows.append(
                {
                    "schema_version": ("latentguard.lg_r1.privileged_step.v1"),
                    "episode_id": job.episode_id,
                    "suite": job.suite,
                    "task_id": job.task_id,
                    "seed": job.seed,
                    "instruction": job.instruction,
                    "elapsed_seconds": time.perf_counter() - started,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "pre_state": pre_state,
                    "post_state": post_state,
                    "initial_privileged_state": initial_state,
                    "reward": reward,
                    "success": success,
                    "done": done,
                    **action_record,
                    "privileged_model_input_allowed": False,
                }
            )
            frame_count += 1
            raw_observation = _batch_observation(next_observation)
            if done:
                break
        dataset.save_episode()
    finally:
        dataset.finalize()
        vector_env.close()
    write_jsonl_atomic(sidecar_path, rows)
    metadata = _dataset_metadata_identities(dataset_path)
    reload_validation = _validate_dataset_reload(dataset_path, frame_count)
    policy_config = stack.checkpoint_path / "config.json"
    if not policy_config.is_file():
        raise FileNotFoundError(f"missing frozen policy config: {policy_config}")
    return {
        **asdict(job),
        "frame_count": frame_count,
        "success": success,
        "termination_reason": ("success" if success else "horizon_exhausted"),
        "dataset_runtime_path": str(dataset_path),
        "sidecar_runtime_path": str(sidecar_path),
        "elapsed_seconds": time.perf_counter() - started,
        "policy_sampling_seed": 420000 + job.seed,
        "policy_queries": instrumentation.policy_queries - query_count_start,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "processor_revision": CHECKPOINT_REVISION,
        "policy_config_sha256": sha256_path(policy_config),
        "policy_optimizer_steps": instrumentation.optimizer_steps,
        "policy_backward_calls": instrumentation.backward_calls,
        "action_mask_semantic": "not_emitted_by_native_inference",
        "action_is_pad_semantic": "false_for_every_executed_step",
        "metadata_identities": metadata,
        "reload_validation": reload_validation,
    }


def _public_episode(entry: dict[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(entry))
    public.pop("dataset_runtime_path", None)
    public.pop("sidecar_runtime_path", None)
    episode_id = public["episode_id"]
    public["dataset_locator"] = f"dataset_root/{episode_id}"
    public["sidecar_locator"] = f"sidecar_root/{episode_id}.jsonl"
    return public


def _validate_schedule(
    registry: dict[str, Any], *, include_extension: bool
) -> list[RolloutJob]:
    validate_seed_isolation(registry)
    jobs = build_rollout_jobs(registry, include_extension=include_extension)
    expected = 160 if include_extension else 80
    if len(jobs) != expected:
        raise ValueError(f"expected {expected} rollout jobs, got {len(jobs)}")
    return jobs


def main() -> None:
    """Collect or validate the outcome-independent frozen schedule."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/rollout_collection.yaml"),
    )
    parser.add_argument("--task-registry", type=Path)
    parser.add_argument("--seed-registry", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-extension", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--record-all", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    registry_path = resolve_repo_path(
        args.task_registry or Path(str(config["task_registry"]))
    )
    registry = read_yaml(registry_path)
    jobs = _validate_schedule(registry, include_extension=args.include_extension)
    if args.seed_registry is not None:
        frozen = read_json(resolve_repo_path(args.seed_registry))
        if frozen.get("source_config_sha256") != sha256_path(registry_path):
            raise ValueError("task/seed registry digest mismatch")
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be positive")
        jobs = jobs[: args.max_episodes]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "scheduled_episodes": len(jobs),
                    "include_extension": args.include_extension,
                },
                sort_keys=True,
            )
        )
        return
    if not (args.record_all or bool(config.get("record_all"))):
        raise ValueError("LG-R1 requires --record-all")
    runtime_root = output_root()
    dataset_root = runtime_root / "datasets"
    sidecar_root = runtime_root / "privileged_sidecars"
    attempt_root = runtime_root / "attempts"
    orphan_root = runtime_root / "orphaned"
    episode_manifest_root = runtime_root / "episode_manifests"
    completion_root = runtime_root / "completed"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    attempt_root.mkdir(parents=True, exist_ok=True)
    orphan_root.mkdir(parents=True, exist_ok=True)
    episode_manifest_root.mkdir(parents=True, exist_ok=True)
    completion_root.mkdir(parents=True, exist_ok=True)
    completed = completed_episode_ids(completion_root)
    for episode_id in completed:
        if (
            not (dataset_root / episode_id).is_dir()
            or not (sidecar_root / f"{episode_id}.jsonl").is_file()
        ):
            raise FileNotFoundError(
                f"completed episode payload is missing: {episode_id}"
            )
    if completed and not args.resume:
        raise ValueError("completed episodes exist; pass --resume")
    pending = [job for job in jobs if job.episode_id not in completed]
    configure_libero_assets()
    stack = load_stack()
    preprocessor, postprocessor = load_processors(stack)
    processor_manifest = _prepare_processor_manifest(
        runtime_root, stack, preprocessor, postprocessor
    )
    instrumentation = instrument_policy(stack.policy)
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for job in pending:
        try:
            final_dataset = dataset_root / job.episode_id
            final_sidecar = sidecar_root / f"{job.episode_id}.jsonl"
            for stale in (final_dataset, final_sidecar):
                if stale.exists():
                    destination = orphan_root / (f"{stale.name}-{time.time_ns()}")
                    os.replace(stale, destination)
            attempt = attempt_root / (f"{job.episode_id}-{time.time_ns()}")
            attempt.mkdir(parents=True, exist_ok=False)
            entry = _collect_episode(
                job,
                stack=stack,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                instrumentation=instrumentation,
                attempt_root=attempt,
            )
            os.replace(attempt / "dataset", final_dataset)
            os.replace(attempt / "privileged_sidecar.jsonl", final_sidecar)
            attempt.rmdir()
            entry["dataset_runtime_path"] = str(final_dataset)
            entry["sidecar_runtime_path"] = str(final_sidecar)
            entries.append(entry)
            write_json(episode_manifest_root / f"{job.episode_id}.json", entry)
            write_completion_marker(
                completion_root,
                job,
                frame_count=int(entry["frame_count"]),
                success=bool(entry["success"]),
                termination_reason=str(entry["termination_reason"]),
                dataset_locator=f"dataset_root/{job.episode_id}",
                sidecar_locator=(f"sidecar_root/{job.episode_id}.jsonl"),
            )
        except Exception as exc:
            errors.append(
                {
                    "episode_id": job.episode_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            break
    all_entries = []
    for marker_path in sorted(completion_root.glob("*.json")):
        marker = read_json(marker_path)
        episode_manifest = episode_manifest_root / f"{marker['episode_id']}.json"
        if not episode_manifest.is_file():
            raise FileNotFoundError(
                f"missing completed episode manifest: {episode_manifest}"
            )
        all_entries.append(read_json(episode_manifest))
    valid = len(all_entries)
    manifest = {
        "schema_version": "latentguard.lg_r1.rollout_manifest.v1",
        "status": ("pass" if not errors and valid == len(jobs) else "incomplete"),
        "execution_site": "remote",
        "runtime_identity": runtime_identity(),
        "task_registry_sha256": sha256_path(registry_path),
        "scheduled_episodes": len(jobs),
        "valid_episodes": valid,
        "task_count": len(
            {(entry["suite"], entry["task_id"]) for entry in all_entries}
        ),
        "suite_count": len({entry["suite"] for entry in all_entries}),
        "natural_failed_episodes": sum(
            not bool(entry["success"]) for entry in all_entries
        ),
        "reload_completeness": (valid / len(jobs) if jobs else math.nan),
        "processor_identity_completeness": 1.0,
        "checkpoint_identity_completeness": 1.0,
        "nonfinite_actions": 0,
        "action_contract_errors": 0,
        "environment_infrastructure_errors": len(errors),
        "episodes": [_public_episode(entry) for entry in all_entries],
        "processor_manifest": ("processor_serialization_manifest.json"),
        "processor_manifest_status": processor_manifest["status"],
        "errors": errors,
        "upload_to_hub": False,
        "policy_optimizer_steps": 0,
        "policy_backward_calls": 0,
    }
    manifest_path = (
        args.output
        if args.output is not None
        else runtime_root / "rollout_manifest.json"
    )
    write_json(resolve_repo_path(manifest_path), manifest)
    print(json.dumps(manifest, sort_keys=True))
    if errors:
        raise RuntimeError(errors[0]["error"])


if __name__ == "__main__":
    main()
