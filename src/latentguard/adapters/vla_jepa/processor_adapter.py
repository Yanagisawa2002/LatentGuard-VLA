"""Processor serialization helpers that preserve official LeRobot behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProcessorSerializationError(RuntimeError):
    """Raised when processor persistence cannot be verified."""


def serialize_processors(
    preprocessor: Any,
    postprocessor: Any,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Serialize official processors into separate immutable directories."""

    pre_dir = output_dir / "preprocessor"
    post_dir = output_dir / "postprocessor"
    preprocessor.save_pretrained(pre_dir)
    postprocessor.save_pretrained(post_dir)
    if not any(pre_dir.iterdir()) or not any(post_dir.iterdir()):
        raise ProcessorSerializationError("processor serialization produced no files")
    return pre_dir, post_dir


def write_processor_binding(
    path: Path,
    *,
    checkpoint_revision: str,
    processor_revision: str,
    files: list[dict[str, Any]],
) -> None:
    """Write a stable JSON identity binding for serialized processors."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "latentguard.lg_r0.processor_binding.v1",
        "checkpoint_revision": checkpoint_revision,
        "processor_revision": processor_revision,
        "files": files,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
