"""Remote-only RoboLab stack and one-step environment validation for LG-RB0."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2  # noqa: F401  # RoboLab requires OpenCV before Isaac Lab.
import yaml
from isaaclab.app import AppLauncher
from packaging.version import Version

from latentguard.adapters.robolab.state_schema import flatten_state_tree


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
    ).strip()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _load_task_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    tasks = value.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("existing environment validation has invalid tasks")
    return [task for task in tasks if isinstance(task, dict)]


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Collect identities before Kit starts; never fork after AppLauncher."""
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise RuntimeError("remote LatentGuard checkout is not at expected commit")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("remote LatentGuard checkout is not clean")
    external_root_value = os.environ.get("LG_RB0_ROBOLAB_ROOT")
    if not external_root_value:
        raise RuntimeError("LG_RB0_ROBOLAB_ROOT must identify the frozen source")
    external_root = Path(external_root_value).resolve()
    actual_robolab_commit = _git(external_root, "rev-parse", "HEAD")
    if actual_robolab_commit != str(protocol["external_stack"]["commit"]):
        raise RuntimeError("RoboLab checkout does not match frozen commit")
    if _git(external_root, "status", "--porcelain"):
        raise RuntimeError("RoboLab source checkout is not clean")
    versions = {
        "python": platform.python_version(),
        "robolab_distribution": _package_version("robolab"),
        "isaacsim": _package_version("isaacsim"),
        "isaaclab": _package_version("isaaclab"),
        "torch": _package_version("torch"),
    }
    expected_versions = {
        "robolab_distribution": "0.2.1",
        "isaacsim": "5.1.0",
        "isaaclab": "2.3.2.post1",
    }
    if any(
        Version(versions[key]) != Version(value)
        for key, value in expected_versions.items()
    ):
        raise RuntimeError(f"frozen package identity mismatch: {versions}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("LG-RB0 requires Python 3.11")
    vulkan_icd_value = os.environ.get("VK_ICD_FILENAMES")
    if not vulkan_icd_value:
        raise RuntimeError("VK_ICD_FILENAMES must select one NVIDIA ICD")
    vulkan_icd_name = Path(vulkan_icd_value).name
    if vulkan_icd_name != "nvidia_icd.json":
        raise RuntimeError("LG-RB0 requires the standard NVIDIA Vulkan ICD")
    gpu_line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()[0]
    gpu_name, memory_mib, driver = [
        value.strip() for value in gpu_line.split(",", maxsplit=2)
    ]
    return {
        "repo": repo,
        "output": output,
        "protocol": protocol,
        "actual_robolab_commit": actual_robolab_commit,
        "versions": versions,
        "vulkan_icd_name": vulkan_icd_name,
        "gpu_name": gpu_name,
        "memory_mib": int(memory_mib),
        "driver": driver,
        "branch": _git(repo, "branch", "--show-current"),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--task-query")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    preflight = _preflight(args)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        import robolab
        import robolab.constants
        import torch
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs,
        )

        output = preflight["output"]
        protocol = preflight["protocol"]
        actual_robolab_commit = preflight["actual_robolab_commit"]
        versions = dict(preflight["versions"])
        versions["cuda_runtime"] = torch.version.cuda
        vulkan_icd_name = preflight["vulkan_icd_name"]

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = True
        auto_register_droid_envs()
        stack_manifest = {
            "schema_version": "lg_rb0_robolab_stack_manifest_v1",
            "status": "pass",
            "robolab": {
                "repository": protocol["external_stack"]["repository"],
                "release": protocol["external_stack"]["release"],
                "commit": actual_robolab_commit,
                "import_module": str(Path(robolab.__file__).name),
            },
            "versions": versions,
            "gpu": {
                "name": preflight["gpu_name"],
                "total_memory_bytes": preflight["memory_mib"] * 1024 * 1024,
                "driver": preflight["driver"],
            },
            "runtime": {
                "num_envs": 1,
                "headless": bool(args.headless),
                "device": str(protocol["runtime"]["device"]),
                "vulkan_icd_manifest": vulkan_icd_name,
            },
        }
        remote_audit = {
            "schema_version": "lg_rb0_remote_execution_audit_v1",
            "status": "pass",
            "branch": preflight["branch"],
            "commit": args.expected_commit,
            "tracked_checkout_clean": True,
            "external_source_clean": True,
            "network_turbo_sourced": os.environ.get("LG_RB0_NETWORK_TURBO") == "1",
            "num_envs": 1,
            "vulkan_icd_manifest": vulkan_icd_name,
            "training_performed": False,
            "checkpoint_generated": False,
            "policy_or_ranker_loaded": False,
            "final_seeds_accessed": False,
        }
        _write(output / "robolab_stack_manifest.json", stack_manifest)
        _write(output / "remote_execution_audit.json", remote_audit)
        validation_path = output / "environment_validation.json"
        task_records = _load_task_records(validation_path)
        expected_queries = [
            str(query) for query in protocol["recording"]["task_queries"]
        ]
        if args.task_query is not None:
            if args.task_query not in expected_queries:
                raise ValueError("task query is outside the frozen protocol")
            task_queries = [args.task_query]
        else:
            task_queries = expected_queries
        for query in task_queries:
            matches = sorted(get_envs(task=query))
            if not matches:
                raise RuntimeError(
                    f"RoboLab task query resolved no environments: {query}"
                )
            selected = matches[0]
            task_output = output / "probe_runtime" / str(query)
            task_output.mkdir(parents=True, exist_ok=True)
            set_output_dir(str(task_output))
            env, env_cfg = create_env(
                selected,
                device=str(protocol["runtime"]["device"]),
                seed=0,
                num_envs=1,
                use_fabric=True,
            )
            try:
                observation, _ = env.reset()
                robot = env.scene["robot"]
                arm = robot.data.joint_pos[0, :7]
                gripper = torch.tensor([0.0], device=env.device)
                action = torch.cat([arm, gripper]).unsqueeze(0)
                _, _, terminated, truncated, _ = env.step(action)
                state = flatten_state_tree(env.scene.get_state(is_relative=True))
                record = {
                    "task_query": query,
                    "matching_envs": matches,
                    "selected_env": selected,
                    "instruction": str(env_cfg.instruction),
                    "smoke_seed": 0,
                    "observation_groups": sorted(observation),
                    "state_leaf_count": len(state),
                    "state_paths": sorted(state),
                    "terminated_after_one_step": bool(terminated[0].item()),
                    "truncated_after_one_step": bool(truncated[0].item()),
                }
                task_records = [
                    task for task in task_records if task.get("task_query") != query
                ]
                task_records.append(record)
                task_records.sort(key=lambda task: str(task["task_query"]))
                completed_queries = {str(task["task_query"]) for task in task_records}
                complete = set(expected_queries) <= completed_queries
                environment_validation = {
                    "schema_version": "lg_rb0_environment_validation_v1",
                    "status": "pass" if complete else "partial",
                    "task_smoke_count": len(task_records),
                    "expected_task_smoke_count": len(expected_queries),
                    "tasks": task_records,
                    "one_step_gpu_smoke": complete,
                    "training_performed": False,
                    "simulator_rollout_kind": ("one_step_environment_mechanics_smoke"),
                }
                _write(validation_path, environment_validation)
                print(
                    json.dumps(
                        {
                            "status": environment_validation["status"],
                            "task": selected,
                            "output": str(output),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                env.close()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    _main()
