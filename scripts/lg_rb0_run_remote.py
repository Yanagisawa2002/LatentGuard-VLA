"""Remote-only recording, faithful replay, and takeover validation for LG-RB0."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2  # noqa: F401  # RoboLab requires OpenCV before Isaac Lab.
import numpy as np
import yaml
from isaaclab.app import AppLauncher

from latentguard.adapters.robolab.overlay_report import classify_overlay_skips
from latentguard.adapters.robolab.state_schema import (
    compare_state_trees,
    flatten_state_tree,
    state_tree_sha256,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    if isinstance(candidate, np.ndarray):
        return candidate.tolist()
    if isinstance(candidate, np.generic):
        return candidate.item()
    if isinstance(candidate, (str, int, float, bool)) or candidate is None:
        return candidate
    return str(candidate)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
    ).strip()


def _validate_checkout(expected_commit: str) -> None:
    root = Path(__file__).resolve().parents[1]
    if _git(root, "rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("remote checkout does not match expected commit")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("remote checkout is not clean")


def _validate_patched_robolab(
    *,
    expected_base_commit: str,
    expected_tree_digest: str,
) -> None:
    import robolab

    root = Path(robolab.__file__).resolve().parents[1]
    if _git(root, "remote", "get-url", "origin") != (
        "https://github.com/NVLabs/RoboLab.git"
    ):
        raise RuntimeError("RoboLab checkout does not use the official origin")
    if _git(root, "rev-parse", "HEAD") != expected_base_commit:
        raise RuntimeError("RoboLab checkout does not match the exact base")
    if _git(root, "write-tree") != expected_tree_digest:
        raise RuntimeError("RoboLab staged tree does not match the exact patch")
    if _git(root, "diff", "--name-only"):
        raise RuntimeError("RoboLab checkout has unstaged tracked changes")
    if _git(root, "ls-files", "--others", "--exclude-standard"):
        raise RuntimeError("RoboLab checkout has unexpected untracked files")


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def _row_tree(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _row_tree(item, index) for key, item in value.items()}
    array = np.asarray(value)
    return np.ascontiguousarray(array[index : index + 1])


def _without_cameras(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "cameras"}


def _anchor_indices(action_count: int, fractions: Sequence[float]) -> list[int]:
    if action_count < 3:
        raise ValueError("recording is too short for early/middle/late anchors")
    return [
        max(1, min(action_count - 1, int(np.floor(action_count * fraction))))
        for fraction in fractions
    ]


def _manager_terms(manager: Any) -> Any:
    getter = getattr(manager, "get_active_iterable_terms", None)
    if not callable(getter):
        return None
    return _to_jsonable(getter(0))


def _hash_runtime_value(value: Any) -> str:
    return _canonical_sha256(_to_jsonable(value))


def _rng_state() -> dict[str, str]:
    import torch

    numpy_state = np.random.get_state()
    torch_cpu = torch.get_rng_state().detach().cpu().numpy()
    result = {
        "python_sha256": hashlib.sha256(
            repr(random.getstate()).encode("utf-8")
        ).hexdigest(),
        "numpy_sha256": hashlib.sha256(
            repr(
                (
                    numpy_state[0],
                    numpy_state[1].tolist(),
                    numpy_state[2],
                    numpy_state[3],
                    numpy_state[4],
                )
            ).encode("utf-8")
        ).hexdigest(),
        "torch_cpu_sha256": hashlib.sha256(torch_cpu.tobytes()).hexdigest(),
    }
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        result["torch_cuda_sha256"] = _canonical_sha256(
            [
                hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()
                for state in cuda_states
            ]
        )
    else:
        result["torch_cuda_sha256"] = "cuda_unavailable"
    return result


def _selected_recorder_state(env: Any) -> dict[str, Any]:
    manager = getattr(env, "recorder_manager", None)
    terms = getattr(manager, "_terms", None)
    if isinstance(terms, Mapping):
        named_terms = sorted((str(name), term) for name, term in terms.items())
    elif isinstance(terms, Sequence) and not isinstance(terms, (str, bytes)):
        named_terms = [(str(index), term) for index, term in enumerate(terms)]
    else:
        named_terms = []
    term_states: dict[str, Any] = {}
    for name, term in named_terms:
        selected: dict[str, Any] = {"type": type(term).__qualname__}
        for attribute in (
            "initialized",
            "infos",
            "_events",
            "_prev_sm_state",
        ):
            if hasattr(term, attribute):
                selected[attribute] = _to_jsonable(getattr(term, attribute))
        state_machines = getattr(term, "subtask_state_machines", None)
        if isinstance(state_machines, Sequence):
            selected["condition_caches"] = [
                _to_jsonable(getattr(machine, "_condition_cache", None))
                for machine in state_machines
            ]
        term_states[name] = selected
    return {
        "manager_present": manager is not None,
        "term_count": len(named_terms),
        "terms": term_states,
    }


def _runtime_isolation_state(env: Any) -> tuple[dict[str, Any], dict[str, bool]]:
    action_manager = getattr(env, "action_manager", None)
    action_buffer = {
        attribute: _to_jsonable(getattr(action_manager, attribute))
        for attribute in ("action", "prev_action")
        if action_manager is not None and hasattr(action_manager, attribute)
    }
    recorder = _selected_recorder_state(env)
    config = _to_jsonable(env.cfg.to_dict())
    rng = _rng_state()
    runtime = {
        "action_buffer_sha256": _hash_runtime_value(action_buffer),
        "recorder_sha256": _hash_runtime_value(recorder),
        "rng": rng,
        "config_sha256": _hash_runtime_value(config),
        "shared_caches_sha256": _hash_runtime_value(
            {
                name: term.get("condition_caches")
                for name, term in recorder["terms"].items()
            }
        ),
    }
    coverage = {
        "action_buffer": bool(action_buffer),
        "recorder": recorder["manager_present"],
        "rng": bool(rng),
        "config": bool(config),
        "shared_caches": any(
            "condition_caches" in term for term in recorder["terms"].values()
        ),
    }
    return runtime, coverage


def _capture(env: Any, observation: Mapping[str, Any], step: int) -> dict[str, Any]:
    from robolab.core.logging.results import (
        get_all_env_events,
        get_current_subtask_info,
    )
    from robolab.core.world.world_state import get_world

    state = env.scene.get_state(is_relative=True)
    events = get_all_env_events(env)
    subtask = get_current_subtask_info(env, env_id=0)
    termination_terms = _manager_terms(env.termination_manager)
    predicate_state = _to_jsonable(getattr(get_world(env), "_predicate_state", None))
    terminated = bool(env.termination_manager.terminated[0].item())
    truncated = bool(env.termination_manager.time_outs[0].item())
    runtime_isolation, runtime_coverage = _runtime_isolation_state(env)
    replay_semantics = {
        "observation_sha256": state_tree_sha256(observation),
        "termination_terms": termination_terms,
        "predicate_state": predicate_state,
        "subtask": _to_jsonable(subtask),
        "events": _to_jsonable(None if events is None else events[0]),
        "terminated": terminated,
        "truncated": truncated,
        "success": terminated and not truncated,
    }
    replay_coverage = {
        "observation": True,
        "termination_terms": termination_terms is not None,
        "predicate_state": predicate_state is not None,
        "subtask": subtask is not None,
        "events": events is not None,
    }
    return {
        "step": step,
        "state": state,
        "state_sha256": state_tree_sha256(state),
        "semantics": replay_semantics,
        "semantic_sha256": _canonical_sha256(replay_semantics),
        "semantic_coverage": replay_coverage,
        "runtime_isolation": runtime_isolation,
        "runtime_isolation_sha256": _canonical_sha256(runtime_isolation),
        "runtime_isolation_coverage": runtime_coverage,
    }


def _compare_snapshots(
    expected: dict[str, Any],
    observed: dict[str, Any],
    *,
    tolerance: float,
    allowed_optional_empty_namespaces: Sequence[str],
    require_runtime_isolation: bool = False,
) -> dict[str, Any]:
    state = compare_state_trees(
        expected["state"],
        observed["state"],
        tolerance=tolerance,
        allowed_optional_empty_namespaces=allowed_optional_empty_namespaces,
    )
    semantic_match = expected["semantic_sha256"] == observed["semantic_sha256"]
    coverage = all(expected["semantic_coverage"].values()) and all(
        observed["semantic_coverage"].values()
    )
    isolation_match = (
        expected["runtime_isolation_sha256"] == observed["runtime_isolation_sha256"]
    )
    isolation_coverage = all(expected["runtime_isolation_coverage"].values()) and all(
        observed["runtime_isolation_coverage"].values()
    )
    isolation_ok = (
        isolation_match and isolation_coverage if require_runtime_isolation else True
    )
    return {
        "status": (
            "pass"
            if state.matches and semantic_match and coverage and isolation_ok
            else "fail"
        ),
        "state": state.to_dict(),
        "semantic_match": semantic_match,
        "semantic_coverage_complete": coverage,
        "runtime_isolation_required": require_runtime_isolation,
        "runtime_isolation_match": isolation_match,
        "runtime_isolation_coverage_complete": isolation_coverage,
        "expected_runtime_isolation": expected["runtime_isolation"],
        "observed_runtime_isolation": observed["runtime_isolation"],
        "expected_semantic_sha256": expected["semantic_sha256"],
        "observed_semantic_sha256": observed["semantic_sha256"],
    }


def _reset_for_replay(env: Any, hdf5_path: Path, prefix: Sequence[Any]) -> Any:
    import torch
    from robolab.core.replay import restore_recorded_initial_state

    manager = env.recorder_manager
    if manager is not None and hasattr(manager, "clear"):
        manager.clear()
    if hasattr(env, "reset_eval_state"):
        env.reset_eval_state()
    observation, _ = env.reset()
    restore_recorded_initial_state(env, str(hdf5_path), 0)
    for action in prefix:
        tensor = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        observation, _, _, _, _ = env.step(tensor)
    return observation


def _resolved_tasks(protocol: dict[str, Any]) -> dict[str, str]:
    from robolab.core.environments.factory import get_envs

    resolved: dict[str, str] = {}
    for query in protocol["recording"]["task_queries"]:
        matches = sorted(get_envs(task=query))
        if not matches:
            raise RuntimeError(f"task query resolved no environment: {query}")
        resolved[str(query)] = matches[0]
    return resolved


def _record(
    protocol: dict[str, Any],
    run_root: Path,
    artifact_dir: Path,
) -> None:
    import h5py
    import robolab.constants
    import torch
    from robolab.constants import set_output_dir
    from robolab.core.environments.runtime import create_env
    from robolab.core.utils.file_utils import (
        load_hdf5_episode_data,
        load_hdf5_initial_state,
        load_hdf5_states,
    )

    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    robolab.constants.RECORD_IMAGE_DATA = False
    tasks = _resolved_tasks(protocol)
    recordings: list[dict[str, Any]] = []
    for query, selected in tasks.items():
        for seed_value in protocol["recording"]["seeds"]:
            seed = int(seed_value)
            recording_id = f"{_safe_id(query)}-seed-{seed}"
            episode_dir = run_root / "recordings" / recording_id
            episode_dir.mkdir(parents=True, exist_ok=False)
            set_output_dir(str(episode_dir))
            env, env_cfg = create_env(
                selected,
                device=str(protocol["runtime"]["device"]),
                seed=seed,
                num_envs=1,
                use_fabric=True,
            )
            terminal = False
            success = False
            action_count = 0
            try:
                manager = env.recorder_manager
                if manager is None or not hasattr(manager, "set_hdf5_file"):
                    raise RuntimeError("official RoboLab recorder manager unavailable")
                manager.set_hdf5_file("data.hdf5")
                manager.set_episode_index(0, env_ids=[0])
                _, _ = env.reset()
                robot = env.scene["robot"]
                steps = int(protocol["recording"]["steps"])
                interval = int(protocol["recording"]["gripper_toggle_interval"])
                for index in range(steps):
                    arm = robot.data.joint_pos[0, :7].clone()
                    gripper_value = 0.0 if (index // interval) % 2 == 0 else 0.785398163
                    gripper = torch.tensor([gripper_value], device=env.device)
                    action = torch.cat([arm, gripper]).unsqueeze(0)
                    _, _, _, _, _ = env.step(action)
                    action_count += 1
                    if env.all_terminated:
                        terminal = True
                        result = env.get_env_results()[0]["success"]
                        success = bool(result)
                        break
                if not terminal:
                    manager.export_episodes(env_ids=[0])
                manager.clear()
            finally:
                env.close()
                gc.collect()
                torch.cuda.empty_cache()
            hdf5_path = episode_dir / "data.hdf5"
            cfg_path = episode_dir / "env_cfg.json"
            if not hdf5_path.is_file() or not cfg_path.is_file():
                raise RuntimeError(f"recording export missing for {recording_id}")
            actions = load_hdf5_episode_data(str(hdf5_path), 0, "actions")
            initial = load_hdf5_initial_state(str(hdf5_path), 0)
            states = load_hdf5_states(str(hdf5_path), 0)
            state_lengths = [
                int(leaf.shape[0]) for leaf in flatten_state_tree(states).values()
            ]
            with h5py.File(hdf5_path, "r") as handle:
                demo = handle["data/demo_0"]
                recorded_success = bool(demo.attrs.get("success", False))
                provenance = {
                    key: _to_jsonable(value)
                    for key, value in handle["data"].attrs.items()
                }
            valid = (
                len(actions) == action_count
                and bool(state_lengths)
                and min(state_lengths) == action_count
                and max(state_lengths) == action_count
                and bool(flatten_state_tree(_without_cameras(initial)))
            )
            outcome = {
                "recording_id": recording_id,
                "action_count": action_count,
                "terminal_at_recorded_horizon": terminal,
                "success_at_recorded_horizon": success,
                "hdf5_success_attribute": recorded_success,
            }
            _write(episode_dir / "outcome.json", outcome)
            recordings.append(
                {
                    "recording_id": recording_id,
                    "task_query": query,
                    "selected_env": selected,
                    "instruction": str(env_cfg.instruction),
                    "seed": seed,
                    "controller": protocol["recording"]["controller"],
                    "action_count": action_count,
                    "terminal_at_recorded_horizon": terminal,
                    "success_at_recorded_horizon": success,
                    "hdf5_success_attribute": recorded_success,
                    "hdf5_sha256": _sha256(hdf5_path),
                    "env_cfg_sha256": _sha256(cfg_path),
                    "outcome_sha256": _sha256(episode_dir / "outcome.json"),
                    "provenance": provenance,
                    "valid": valid,
                }
            )
    manifest = {
        "schema_version": "lg_rb0_recording_manifest_v1",
        "status": (
            "pass"
            if len(recordings) == int(protocol["recording"]["expected_episode_count"])
            and all(item["valid"] for item in recordings)
            else "fail"
        ),
        "controller_kind": "deterministic_fixed_mechanics_probe",
        "policy_or_candidate_source": False,
        "episode_count": len(recordings),
        "valid_episode_count": sum(bool(item["valid"]) for item in recordings),
        "recordings": recordings,
        "raw_recordings_in_git": False,
        "training_performed": False,
        "final_seeds_accessed": False,
    }
    _write(artifact_dir / "recording_manifest.json", manifest)
    _write(
        run_root / "recording_index.json",
        {
            "schema_version": "lg_rb0_recording_index_v1",
            "recording_ids": [item["recording_id"] for item in recordings],
        },
    )
    if manifest["status"] != "pass":
        raise RuntimeError("recording manifest gate failed")


def _load_recording(
    recording_root: Path,
    item: dict[str, Any],
) -> tuple[Path, list[Any], dict[str, Any]]:
    from robolab.core.utils.file_utils import load_hdf5_episode_data

    episode_dir = recording_root / "recordings" / str(item["recording_id"])
    hdf5_path = episode_dir / "data.hdf5"
    if _sha256(hdf5_path) != item["hdf5_sha256"]:
        raise RuntimeError("recording content digest changed")
    actions = list(load_hdf5_episode_data(str(hdf5_path), 0, "actions"))
    outcome = _read(episode_dir / "outcome.json")
    return hdf5_path, actions, outcome


def _replay_env(
    protocol: dict[str, Any],
    hdf5_path: Path,
    selected_env: str,
    replay_output: Path,
    stages: dict[str, Any] | None = None,
) -> tuple[Any, list[str]]:
    from robolab.constants import set_output_dir
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.core.replay import apply_recorded_env_cfg, load_recorded_env_cfg

    replay_output.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(replay_output))
    loaded = load_recorded_env_cfg(str(hdf5_path))
    if loaded is None:
        raise RuntimeError("recorded environment sidecar is missing")
    recorded_cfg, _ = loaded
    env_cfg = parse_env_cfg(
        selected_env,
        device=str(protocol["runtime"]["device"]),
        seed=0,
        num_envs=1,
        env_spacing=None,
        eye=None,
        lookat=None,
        use_fabric=True,
    )
    skipped = apply_recorded_env_cfg(env_cfg, recorded_cfg)
    if stages is not None:
        stages["env_config_overlay_completed"] = True
        stages["recorded_config_skipped_fields"] = list(skipped)
    if env_cfg.recorders is not None:
        env_cfg.recorders.dataset_export_dir_path = str(replay_output)
        env_cfg.recorders.dataset_filename = "replay.hdf5"
    env, _ = create_env(
        env_cfg,
        device=str(protocol["runtime"]["device"]),
        num_envs=1,
        use_fabric=True,
    )
    if stages is not None:
        stages["environment_created"] = True
    return env, list(skipped)


def _stage_detail(recording_id: str, repeat: int) -> dict[str, Any]:
    return {
        "recording_id": recording_id,
        "repeat": repeat,
        "status": "execution_error",
        "attempted": True,
        "env_config_overlay_completed": False,
        "environment_created": False,
        "replay_started": False,
        "replay_completed": False,
        "per_step_validation_completed": False,
        "terminal_validation_completed": False,
        "success_validation_completed": False,
        "initial_restore_pass": None,
        "initial_restore_comparison": None,
        "initial_restore_maximum_absolute_error": None,
        "recorded_config_skipped_fields": [],
        "recorded_config_overlay_report": None,
        "recorded_config_overlay_pass": None,
        "official_state_validator_pass": None,
        "official_maximum_absolute_error": None,
        "strict_per_step_failure_count": None,
        "terminal_match": None,
        "success_match": None,
        "observation_available": None,
        "execution_error": None,
    }


def _record_canonicalization(
    observations: dict[str, dict[str, Any]],
    comparison: Any,
    *,
    context: str,
) -> None:
    value = comparison.to_dict()["canonicalization"]
    structural = {
        side: {
            "allowed_optional_empty_namespaces": value[side][
                "allowed_optional_empty_namespaces"
            ],
            "removed_empty_namespaces": value[side]["removed_empty_namespaces"],
            "before_paths": value[side]["before_paths"],
            "after_paths": value[side]["after_paths"],
        }
        for side in ("expected", "observed")
    }
    key = json.dumps(structural, separators=(",", ":"), sort_keys=True)
    if key not in observations:
        observations[key] = {
            "count": 0,
            "first_context": context,
            "first_comparison": value,
        }
    observations[key]["count"] += 1


def _faithful(
    protocol: dict[str, Any],
    run_root: Path,
    recording_root: Path,
    artifact_dir: Path,
) -> None:
    import torch
    from robolab.core.replay import StateValidator
    from robolab.core.utils.file_utils import (
        load_hdf5_initial_state,
        load_hdf5_states,
    )

    manifest = _read(artifact_dir / "recording_manifest.json")
    details: list[dict[str, Any]] = []
    initial_failures = 0
    config_overlay_failures = 0
    state_failures = 0
    terminal_mismatches = 0
    success_mismatches = 0
    completed_replays = 0
    execution_errors = 0
    official_validator_failures = 0
    canonicalization_observations: dict[str, dict[str, Any]] = {}
    official_tolerance = float(protocol["faithful_replay"]["official_state_tolerance"])
    strict_tolerance = float(protocol["faithful_replay"]["strict_state_tolerance"])
    optional_empty = tuple(
        str(value)
        for value in protocol["state_schema"]["allowed_optional_empty_namespaces"]
    )
    repeats = int(protocol["faithful_replay"]["repeats"])
    for item in manifest["recordings"]:
        hdf5_path, actions, outcome = _load_recording(recording_root, item)
        recorded_initial = _without_cameras(load_hdf5_initial_state(str(hdf5_path), 0))
        recorded_states = load_hdf5_states(str(hdf5_path), 0)
        setup = _stage_detail(str(item["recording_id"]), 0)
        env = None
        try:
            env, skipped = _replay_env(
                protocol,
                hdf5_path,
                str(item["selected_env"]),
                run_root / "replay_runtime" / str(item["recording_id"]),
                stages=setup,
            )
            skip_report = classify_overlay_skips(skipped)
            for repeat in range(repeats):
                detail = _stage_detail(str(item["recording_id"]), repeat)
                detail.update(
                    {
                        "env_config_overlay_completed": setup[
                            "env_config_overlay_completed"
                        ],
                        "environment_created": setup["environment_created"],
                        "recorded_config_skipped_fields": skipped,
                        "recorded_config_overlay_report": skip_report.to_dict(),
                        "recorded_config_overlay_pass": not skip_report.fatal,
                    }
                )
                if skip_report.fatal:
                    detail["status"] = "fail"
                    config_overlay_failures += 1
                    details.append(detail)
                    continue
                try:
                    observation = _reset_for_replay(env, hdf5_path, [])
                    detail["replay_started"] = True
                    initial = compare_state_trees(
                        recorded_initial,
                        env.scene.get_state(is_relative=True),
                        tolerance=strict_tolerance,
                        allowed_optional_empty_namespaces=optional_empty,
                    )
                    _record_canonicalization(
                        canonicalization_observations,
                        initial,
                        context=f"{item['recording_id']}:repeat-{repeat}:initial",
                    )
                    initial_ok = initial.matches
                    initial_failures += int(not initial_ok)
                    config_overlay_ok = not skip_report.fatal
                    config_overlay_failures += int(not config_overlay_ok)
                    detail["initial_restore_pass"] = initial_ok
                    detail["initial_restore_comparison"] = initial.to_dict()
                    detail["initial_restore_maximum_absolute_error"] = (
                        initial.maximum_absolute_error
                    )
                    validator = StateValidator(
                        str(hdf5_path),
                        0,
                        tolerance=official_tolerance,
                    )
                    repeat_state_failures = 0
                    for step, action in enumerate(actions):
                        tensor = torch.as_tensor(
                            action,
                            dtype=torch.float32,
                            device=env.device,
                        ).unsqueeze(0)
                        observation, _, _, _, _ = env.step(tensor)
                        validator.check_step(env, step)
                        strict = compare_state_trees(
                            _row_tree(recorded_states, step),
                            env.scene.get_state(is_relative=True),
                            tolerance=strict_tolerance,
                            allowed_optional_empty_namespaces=optional_empty,
                        )
                        _record_canonicalization(
                            canonicalization_observations,
                            strict,
                            context=(
                                f"{item['recording_id']}:repeat-{repeat}:step-{step}"
                            ),
                        )
                        if not strict.matches:
                            repeat_state_failures += 1
                    detail["replay_completed"] = True
                    detail["per_step_validation_completed"] = True
                    terminal = bool(env.all_terminated)
                    result = env.get_env_results()[0]["success"]
                    success = bool(result) if result is not None else False
                    terminal_match = terminal == bool(
                        outcome["terminal_at_recorded_horizon"]
                    )
                    detail["terminal_validation_completed"] = True
                    success_match = success == bool(
                        outcome["success_at_recorded_horizon"]
                    )
                    detail["success_validation_completed"] = True
                    official_pass = validator.first_exceed_step is None
                    state_failures += repeat_state_failures
                    official_validator_failures += int(not official_pass)
                    terminal_mismatches += int(not terminal_match)
                    success_mismatches += int(not success_match)
                    completed_replays += 1
                    detail.update(
                        {
                            "status": (
                                "pass"
                                if initial_ok
                                and config_overlay_ok
                                and official_pass
                                and repeat_state_failures == 0
                                and terminal_match
                                and success_match
                                else "fail"
                            ),
                            "official_state_validator_pass": official_pass,
                            "official_maximum_absolute_error": validator.max_drift,
                            "strict_per_step_failure_count": repeat_state_failures,
                            "terminal_match": terminal_match,
                            "success_match": success_match,
                            "observation_available": observation is not None,
                        }
                    )
                except Exception as error:
                    execution_errors += 1
                    detail["execution_error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    traceback.print_exc()
                details.append(detail)
        except Exception as error:
            traceback.print_exc()
            for repeat in range(repeats):
                detail = _stage_detail(str(item["recording_id"]), repeat)
                detail.update(
                    {
                        "env_config_overlay_completed": setup[
                            "env_config_overlay_completed"
                        ],
                        "environment_created": setup["environment_created"],
                        "recorded_config_skipped_fields": setup[
                            "recorded_config_skipped_fields"
                        ],
                        "recorded_config_overlay_report": (
                            classify_overlay_skips(
                                setup["recorded_config_skipped_fields"]
                            ).to_dict()
                            if setup["env_config_overlay_completed"]
                            else None
                        ),
                        "recorded_config_overlay_pass": (
                            not classify_overlay_skips(
                                setup["recorded_config_skipped_fields"]
                            ).fatal
                            if setup["env_config_overlay_completed"]
                            else False
                        ),
                        "execution_error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                execution_errors += 1
                config_overlay_failures += int(
                    not bool(detail["recorded_config_overlay_pass"])
                )
                details.append(detail)
        finally:
            if env is not None:
                env.close()
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "schema_version": "lg_rb0_faithful_replay_validation_v1",
        "status": (
            "pass"
            if initial_failures
            == config_overlay_failures
            == state_failures
            == terminal_mismatches
            == success_mismatches
            == 0
            and official_validator_failures == 0
            and execution_errors == 0
            and completed_replays == len(manifest["recordings"]) * repeats
            else "fail"
        ),
        "episode_count": len(manifest["recordings"]),
        "repeats_per_episode": repeats,
        "official_state_tolerance": official_tolerance,
        "strict_state_tolerance": strict_tolerance,
        "pixel_tolerance": int(protocol["faithful_replay"]["pixel_tolerance"]),
        "initial_restore_failures": initial_failures,
        "recorded_config_overlay_failures": config_overlay_failures,
        "per_step_state_failures": state_failures,
        "official_state_validator_failures": official_validator_failures,
        "terminal_mismatches": terminal_mismatches,
        "success_mismatches": success_mismatches,
        "expected_replay_count": (len(manifest["recordings"]) * repeats),
        "completed_replay_count": completed_replays,
        "execution_error_count": execution_errors,
        "details": details,
    }
    _write(artifact_dir / "faithful_replay_validation.json", result)
    _write(
        artifact_dir / "state_schema_canonicalization.json",
        {
            "schema_version": "lg_rb01_state_schema_canonicalization_v1",
            "status": (
                "pass"
                if canonicalization_observations
                and initial_failures == state_failures == 0
                else "fail"
            ),
            "symmetric": True,
            "numeric_state_changed": False,
            "allowed_optional_empty_namespaces": list(optional_empty),
            "comparison_count": sum(
                int(value["count"]) for value in canonicalization_observations.values()
            ),
            "observed_patterns": list(canonicalization_observations.values()),
        },
    )


def _run_suffix(
    env: Any,
    hdf5_path: Path,
    prefix: Sequence[Any],
    suffix_action: Any,
    checkpoints: set[int],
    *,
    require_runtime_isolation: bool,
) -> tuple[dict[int, dict[str, Any]], bool]:
    import torch

    observation = _reset_for_replay(env, hdf5_path, prefix)
    snapshots: dict[int, dict[str, Any]] = {}
    for step in range(1, max(checkpoints) + 1):
        tensor = torch.as_tensor(
            suffix_action,
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0)
        observation, _, _, _, _ = env.step(tensor)
        if step in checkpoints:
            snapshots[step] = _capture(env, observation, step)
    coverage = all(
        all(snapshot["semantic_coverage"].values()) for snapshot in snapshots.values()
    )
    if require_runtime_isolation:
        coverage = coverage and all(
            all(snapshot["runtime_isolation_coverage"].values())
            for snapshot in snapshots.values()
        )
    return snapshots, coverage


def _suffix_actions(
    env: Any,
    hdf5_path: Path,
    prefix: Sequence[Any],
) -> dict[str, Any]:
    _reset_for_replay(env, hdf5_path, prefix)
    current = env.scene["robot"].data.joint_pos[0, :7]
    arm = current.detach().cpu().numpy().astype(np.float32, copy=True)
    return {
        "A": np.concatenate([arm, np.asarray([0.0], dtype=np.float32)]),
        "B": np.concatenate([arm, np.asarray([0.785398163], dtype=np.float32)]),
    }


def _takeover(
    protocol: dict[str, Any],
    run_root: Path,
    recording_root: Path,
    artifact_dir: Path,
    *,
    mode: str,
) -> None:
    if mode not in {"prefix", "branch", "isolation"}:
        raise ValueError(f"unknown takeover mode: {mode}")
    manifest = _read(artifact_dir / "recording_manifest.json")
    tolerance = float(protocol["takeover"]["state_tolerance"])
    optional_empty = tuple(
        str(value)
        for value in protocol["state_schema"]["allowed_optional_empty_namespaces"]
    )
    fractions = [float(value) for value in protocol["takeover"]["anchor_fractions"]]
    labels = [str(value) for value in protocol["takeover"]["anchor_labels"]]
    checkpoints = {int(value) for value in protocol["takeover"]["checkpoints"]}
    repeats = int(protocol["takeover"]["repeats"])
    prefix_details: list[dict[str, Any]] = []
    branch_details: list[dict[str, Any]] = []
    isolation_details: list[dict[str, Any]] = []
    prefix_mismatches = 0
    branch_mismatches = 0
    isolation_mismatches = 0
    prefix_coverage = True
    branch_coverage = True
    for item in manifest["recordings"]:
        hdf5_path, actions, _ = _load_recording(recording_root, item)
        env, skipped = _replay_env(
            protocol,
            hdf5_path,
            str(item["selected_env"]),
            run_root / "takeover_runtime" / str(item["recording_id"]),
        )
        overlay_report = classify_overlay_skips(skipped)
        if overlay_report.fatal:
            raise RuntimeError(
                "recorded environment config has fatal overlay skips: "
                + json.dumps(overlay_report.to_dict(), sort_keys=True)
            )
        try:
            anchors = _anchor_indices(len(actions), fractions)
            for label, anchor in zip(labels, anchors, strict=True):
                prefix = actions[:anchor]
                if mode == "prefix":
                    prefix_runs: list[dict[str, Any]] = []
                    for _ in range(repeats):
                        observation = _reset_for_replay(env, hdf5_path, prefix)
                        prefix_runs.append(_capture(env, observation, anchor))
                    prefix_comparisons = [
                        _compare_snapshots(
                            prefix_runs[0],
                            prefix_runs[index],
                            tolerance=tolerance,
                            allowed_optional_empty_namespaces=optional_empty,
                        )
                        for index in range(1, repeats)
                    ]
                    prefix_count = sum(
                        comparison["status"] != "pass"
                        for comparison in prefix_comparisons
                    )
                    prefix_mismatches += prefix_count
                    prefix_coverage = prefix_coverage and all(
                        all(run["semantic_coverage"].values()) for run in prefix_runs
                    )
                    prefix_details.append(
                        {
                            "recording_id": item["recording_id"],
                            "anchor_label": label,
                            "anchor_step": anchor,
                            "repeat_count": repeats,
                            "recorded_config_overlay_report": (
                                overlay_report.to_dict()
                            ),
                            "mismatch_count": prefix_count,
                            "comparisons": prefix_comparisons,
                        }
                    )
                    continue

                suffix_actions = _suffix_actions(env, hdf5_path, prefix)
                if mode == "branch":
                    for branch_id in ("A", "B"):
                        runs: list[dict[int, dict[str, Any]]] = []
                        for _ in range(repeats):
                            snapshots, coverage = _run_suffix(
                                env,
                                hdf5_path,
                                prefix,
                                suffix_actions[branch_id],
                                checkpoints,
                                require_runtime_isolation=False,
                            )
                            runs.append(snapshots)
                            branch_coverage = branch_coverage and coverage
                        comparisons = []
                        for repeat in range(1, repeats):
                            for checkpoint in sorted(checkpoints):
                                comparisons.append(
                                    {
                                        "repeat": repeat,
                                        "checkpoint": checkpoint,
                                        **_compare_snapshots(
                                            runs[0][checkpoint],
                                            runs[repeat][checkpoint],
                                            tolerance=tolerance,
                                            allowed_optional_empty_namespaces=(
                                                optional_empty
                                            ),
                                        ),
                                    }
                                )
                        mismatch_count = sum(
                            comparison["status"] != "pass" for comparison in comparisons
                        )
                        branch_mismatches += mismatch_count
                        branch_details.append(
                            {
                                "recording_id": item["recording_id"],
                                "anchor_label": label,
                                "anchor_step": anchor,
                                "branch_id": branch_id,
                                "repeat_count": repeats,
                                "checkpoints": sorted(checkpoints),
                                "recorded_config_overlay_report": (
                                    overlay_report.to_dict()
                                ),
                                "mismatch_count": mismatch_count,
                                "comparisons": comparisons,
                            }
                        )
                    continue

                branch_references: dict[str, dict[int, dict[str, Any]]] = {}
                for branch_id in ("A", "B"):
                    snapshots, coverage = _run_suffix(
                        env,
                        hdf5_path,
                        prefix,
                        suffix_actions[branch_id],
                        checkpoints,
                        require_runtime_isolation=True,
                    )
                    branch_references[branch_id] = snapshots
                    branch_coverage = branch_coverage and coverage
                for order in (("A", "B", "A"), ("B", "A", "B")):
                    sequence_runs: list[tuple[str, dict[int, dict[str, Any]]]] = []
                    for branch_id in order:
                        snapshots, coverage = _run_suffix(
                            env,
                            hdf5_path,
                            prefix,
                            suffix_actions[branch_id],
                            checkpoints,
                            require_runtime_isolation=True,
                        )
                        sequence_runs.append((branch_id, snapshots))
                        branch_coverage = branch_coverage and coverage
                    comparisons = []
                    for sequence_index, (branch_id, snapshots) in enumerate(
                        sequence_runs
                    ):
                        for checkpoint in sorted(checkpoints):
                            comparisons.append(
                                {
                                    "sequence_index": sequence_index,
                                    "branch_id": branch_id,
                                    "checkpoint": checkpoint,
                                    **_compare_snapshots(
                                        branch_references[branch_id][checkpoint],
                                        snapshots[checkpoint],
                                        tolerance=tolerance,
                                        allowed_optional_empty_namespaces=(
                                            optional_empty
                                        ),
                                        require_runtime_isolation=True,
                                    ),
                                }
                            )
                    mismatch_count = sum(
                        comparison["status"] != "pass" for comparison in comparisons
                    )
                    isolation_mismatches += mismatch_count
                    isolation_details.append(
                        {
                            "recording_id": item["recording_id"],
                            "anchor_label": label,
                            "anchor_step": anchor,
                            "order": "-".join(order),
                            "recorded_config_overlay_report": (
                                overlay_report.to_dict()
                            ),
                            "mismatch_count": mismatch_count,
                            "comparisons": comparisons,
                        }
                    )
        finally:
            env.close()
            gc.collect()
            import torch

            torch.cuda.empty_cache()
    if mode == "prefix":
        _write(
            artifact_dir / "prefix_replay_validation.json",
            {
                "schema_version": "lg_rb01_prefix_replay_validation_v1",
                "status": (
                    "pass" if prefix_mismatches == 0 and prefix_coverage else "fail"
                ),
                "state_tolerance": tolerance,
                "pixel_tolerance": int(protocol["takeover"]["pixel_tolerance"]),
                "anchors_per_episode": len(fractions),
                "repeats_per_anchor": repeats,
                "mismatch_count": prefix_mismatches,
                "semantic_coverage_complete": prefix_coverage,
                "details": prefix_details,
            },
        )
    elif mode == "branch":
        _write(
            artifact_dir / "branch_determinism_validation.json",
            {
                "schema_version": "lg_rb01_branch_determinism_validation_v1",
                "status": (
                    "pass" if branch_mismatches == 0 and branch_coverage else "fail"
                ),
                "state_tolerance": tolerance,
                "pixel_tolerance": int(protocol["takeover"]["pixel_tolerance"]),
                "branches": ["A", "B"],
                "checkpoints": sorted(checkpoints),
                "mismatch_count": branch_mismatches,
                "semantic_coverage_complete": branch_coverage,
                "details": branch_details,
            },
        )
    else:
        _write(
            artifact_dir / "branch_isolation_validation.json",
            {
                "schema_version": "lg_rb01_branch_isolation_validation_v1",
                "status": (
                    "pass" if isolation_mismatches == 0 and branch_coverage else "fail"
                ),
                "orders": ["A-B-A", "B-A-B"],
                "mismatch_count": isolation_mismatches,
                "semantic_coverage_complete": branch_coverage,
                "runtime_isolation_fields": [
                    "action_buffer",
                    "recorder",
                    "task_latch",
                    "rng",
                    "config",
                    "simulation_state",
                    "shared_caches",
                ],
                "details": isolation_details,
            },
        )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "record",
            "faithful",
            "takeover-prefix",
            "takeover-branch",
            "takeover-isolation",
        ],
        required=True,
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-robolab-base", required=True)
    parser.add_argument("--expected-robolab-tree", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    vulkan_icd = os.environ.get("VK_ICD_FILENAMES")
    if not vulkan_icd or Path(vulkan_icd).name != "nvidia_icd.json":
        raise RuntimeError("LG-RB0 requires one standard NVIDIA Vulkan ICD")
    _validate_checkout(args.expected_commit)
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import robolab.constants
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs,
        )

        protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
        if not isinstance(protocol, dict):
            raise ValueError("protocol must be a mapping")
        _validate_patched_robolab(
            expected_base_commit=args.expected_robolab_base,
            expected_tree_digest=args.expected_robolab_tree,
        )
        args.run_root.mkdir(parents=True, exist_ok=True)
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
        robolab.constants.RECORD_IMAGE_DATA = False
        auto_register_droid_envs()
        if args.phase == "record":
            _record(protocol, args.run_root, args.artifact_dir)
        elif args.phase == "faithful":
            _faithful(
                protocol,
                args.run_root,
                args.recording_root or args.run_root,
                args.artifact_dir,
            )
        else:
            _takeover(
                protocol,
                args.run_root,
                args.recording_root or args.run_root,
                args.artifact_dir,
                mode=args.phase.removeprefix("takeover-"),
            )
        print(json.dumps({"status": "complete", "phase": args.phase}, sort_keys=True))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    _main()
