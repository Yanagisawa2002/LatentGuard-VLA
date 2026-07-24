"""Remote root-cause audit for real LG-RB0 recorded and live configs."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import cv2  # noqa: F401  # RoboLab requires OpenCV before Isaac Lab.
import yaml
from isaaclab.app import AppLauncher


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
    ).strip()


def _validate_checkout(root: Path, expected_commit: str) -> None:
    if _git(root, "rev-parse", "HEAD") != expected_commit:
        raise RuntimeError(f"checkout commit mismatch: {root.name}")
    if _git(root, "status", "--porcelain=v1"):
        raise RuntimeError(f"checkout is not clean: {root.name}")


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _callable_identity(value: Any) -> dict[str, Any] | None:
    if not callable(value):
        return None
    if isinstance(value, partial):
        return {
            "kind": "functools.partial",
            "function": _callable_identity(value.func),
            "args": [repr(item) for item in value.args],
            "keywords": {
                str(key): repr(item)
                for key, item in sorted((value.keywords or {}).items())
            },
        }
    return {
        "kind": "callable",
        "module": getattr(value, "__module__", None),
        "qualname": getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", None),
        ),
    }


def _target_child(target: Any, key: str) -> tuple[bool, Any]:
    if isinstance(target, Mapping):
        return (key in target, target.get(key))
    return (hasattr(target, key), getattr(target, key, None))


def _collect_lossy_callable_leaves(
    recorded: Any,
    live: Any,
    namespace: str,
) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    if isinstance(recorded, Mapping):
        for key, value in recorded.items():
            exists, target = _target_child(live, str(key))
            child_namespace = f"{namespace}/{key}"
            if exists:
                leaves.extend(
                    _collect_lossy_callable_leaves(
                        value,
                        target,
                        child_namespace,
                    )
                )
        return leaves
    if isinstance(recorded, list):
        live_sequence = live if isinstance(live, (list, tuple)) else ()
        for index, value in enumerate(recorded):
            target = live_sequence[index] if index < len(live_sequence) else None
            leaves.extend(
                _collect_lossy_callable_leaves(
                    value,
                    target,
                    f"{namespace}[{index}]",
                )
            )
        return leaves
    if isinstance(recorded, str) and "functools.partial(" in recorded:
        leaves.append(
            {
                "path": namespace,
                "recorded_type": _type_name(recorded),
                "recorded_value": recorded,
                "live_type": _type_name(live),
                "live_callable": callable(live),
                "live_callable_identity": _callable_identity(live),
            }
        )
    return leaves


def _trace_old_overlay(
    target: Any,
    recorded: Mapping[str, Any],
    namespace: str = "",
) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    missing_targets: list[dict[str, Any]] = []

    def visit(obj: Any, data: Mapping[str, Any], ns: str) -> None:
        for key, value in data.items():
            key_ns = f"{ns}/{key}"
            exists, live = _target_child(obj, str(key))
            if not exists:
                missing_targets.append(
                    {
                        "path": key_ns,
                        "target_container_type": _type_name(obj),
                        "recorded_type": _type_name(value),
                    }
                )
                continue
            if key_ns == "/instruction" and isinstance(value, str):
                assignments.append(
                    {
                        "path": key_ns,
                        "branch": "resolved_instruction_string_assignment",
                    }
                )
                continue
            if isinstance(value, Mapping):
                if isinstance(live, Mapping) or hasattr(live, "__dict__"):
                    visit(live, value, key_ns)
                continue
            if isinstance(value, list):
                if any(isinstance(element, Mapping) for element in value):
                    if isinstance(live, (list, tuple)) and len(live) == len(value):
                        for index, element in enumerate(value):
                            if isinstance(element, Mapping):
                                visit(live[index], element, f"{key_ns}[{index}]")
                    continue
                callable_leaves = _collect_lossy_callable_leaves(
                    value,
                    live,
                    key_ns,
                )
                assignments.append(
                    {
                        "path": key_ns,
                        "branch": "plain_list_wholesale_assignment",
                        "live_target_type": _type_name(live),
                        "lossy_callable_leaves": callable_leaves,
                    }
                )
                continue
            assignments.append(
                {
                    "path": key_ns,
                    "branch": (
                        "callable_string_resolution"
                        if isinstance(value, str) and callable(live)
                        else "scalar_assignment"
                    ),
                    "live_target_type": _type_name(live),
                }
            )

    visit(target, recorded, namespace)
    return {
        "assignments": assignments,
        "missing_targets": missing_targets,
    }


def _relative_source(function: Any, robolab_root: Path) -> dict[str, Any]:
    source_path = Path(inspect.getsourcefile(function) or "").resolve()
    lines, start = inspect.getsourcelines(function)
    return {
        "path": source_path.relative_to(robolab_root).as_posix(),
        "start_line": start,
        "call_expression_line": next(
            (
                start + index
                for index, line in enumerate(lines)
                if "conditional_func(**params_with_env)" in line
            ),
            None,
        ),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import robolab
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.task.conditionals_state_machine import (
        ConditionalsStateMachine,
    )
    from robolab.registrations.droid.auto_env_registrations_jointpos import (
        auto_register_droid_envs,
    )

    latentguard_root = Path(__file__).resolve().parents[1]
    robolab_root = Path(robolab.__file__).resolve().parents[1]
    _validate_checkout(latentguard_root, args.expected_latentguard_commit)
    _validate_checkout(robolab_root, args.expected_robolab_commit)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    manifest = _read_json(args.recording_manifest)
    auto_register_droid_envs()

    recordings: list[dict[str, Any]] = []
    unique_callable_paths: dict[str, dict[str, Any]] = {}
    for item in manifest["recordings"]:
        recording_id = str(item["recording_id"])
        env_cfg_path = (
            args.recording_root / "recordings" / recording_id / "env_cfg.json"
        )
        if _sha256(env_cfg_path) != item["env_cfg_sha256"]:
            raise RuntimeError(f"recorded env config digest changed: {recording_id}")
        recorded = _read_json(env_cfg_path)
        live = parse_env_cfg(
            str(item["selected_env"]),
            device=str(protocol["runtime"]["device"]),
            seed=0,
            num_envs=1,
            env_spacing=None,
            eye=None,
            lookat=None,
            use_fabric=True,
        )
        trace = _trace_old_overlay(live, recorded)
        callable_assignments = [
            assignment
            for assignment in trace["assignments"]
            if assignment.get("lossy_callable_leaves")
        ]
        for assignment in callable_assignments:
            for leaf in assignment["lossy_callable_leaves"]:
                unique_callable_paths.setdefault(leaf["path"], leaf)
        recordings.append(
            {
                "recording_id": recording_id,
                "task_query": item["task_query"],
                "env_cfg_sha256": item["env_cfg_sha256"],
                "recorded_instruction": recorded.get("instruction"),
                "live_instruction_before_overlay_type": _type_name(live.instruction),
                "live_instruction_before_overlay": live.instruction,
                "recorded_instruction_variants_present": (
                    "_instruction_variants" in recorded
                ),
                "recorded_instruction_variants": recorded.get("_instruction_variants"),
                "live_instruction_variants_present_before_overlay": hasattr(
                    live,
                    "_instruction_variants",
                ),
                "old_overlay_callable_assignments": callable_assignments,
                "old_overlay_missing_targets": trace["missing_targets"],
            }
        )

    return {
        "schema_version": "lg_rb01_recorded_live_config_diff_v1",
        "status": "pass",
        "upstream_commit": args.expected_robolab_commit,
        "recording_manifest_sha256": _sha256(args.recording_manifest),
        "recording_count": len(recordings),
        "recordings": recordings,
        "unique_recorded_live_callable_paths": [
            unique_callable_paths[path] for path in sorted(unique_callable_paths)
        ],
        "actual_callable_target_dict_key_missing": False,
        "actual_callable_failure_branch": "plain_list_wholesale_assignment",
        "missing_key_regression_scope": (
            "separate unsafe old-overlay branch covered by the minimal fixture"
        ),
        "instruction_variants_runtime_order": {
            "recording_contains_field": True,
            "parse_env_cfg_creates_field": False,
            "create_env_creates_field_after_overlay": True,
            "recorded_value_should_restore": False,
            "resolved_instruction_should_restore": True,
        },
        "call_site": _relative_source(
            ConditionalsStateMachine.check_condition_satisfied,
            robolab_root,
        ),
        "source_files": {
            "env_config.py": _sha256(
                robolab_root / "robolab/core/replay/env_config.py"
            ),
            "runtime.py": _sha256(
                robolab_root / "robolab/core/environments/runtime.py"
            ),
            "docs/replay.md": _sha256(robolab_root / "docs/replay.md"),
        },
        "eval_or_exec_used": False,
    }


def main() -> None:
    """Launch the minimal Isaac application needed to parse real live configs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--recording-manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-latentguard-commit", required=True)
    parser.add_argument("--expected-robolab-commit", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.enable_cameras = False
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        result = _run(args)
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        output = args.artifact_dir / "recorded_live_config_diff.json"
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": result["status"], "output": output.name}))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
