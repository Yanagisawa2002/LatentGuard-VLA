"""Stable, protected JSON reporting for the Robot Episode Toolkit."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from latentguard.toolkit.replay import ReplayStep


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        raise TypeError("ndarray is not allowed in audit or metrics JSON")
    return value


def report_json(value: object) -> str:
    """Return stable UTF-8-compatible JSON text with a terminal newline."""
    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )


def _atomic_new_text(path: Path, text: str) -> None:
    destination = path.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output already exists: {destination}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(value: object, path: Path) -> None:
    """Transactionally create a stable JSON report without overwriting."""
    _atomic_new_text(path, report_json(value))


def write_replay_jsonl(steps: Sequence[ReplayStep], path: Path) -> None:
    """Write deterministic debug rows, excluding camera pixel arrays."""
    lines: list[str] = []
    for step in steps:
        payload: dict[str, Any] = {
            "step_index": step.step_index,
            "observation_id": step.observation.observation_id,
            "observation_timestamp_s": step.observation_timestamp_s,
            "nominal_action_timestamp_s": step.nominal_action_timestamp_s,
            "camera_ids": sorted(
                camera.camera_id for camera in step.observation.cameras
            ),
            "robot_state": step.observation.robot_state.tolist(),
            "action": step.action.tolist(),
        }
        lines.append(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    _atomic_new_text(path, "\n".join(lines) + "\n")
