"""Audit the exact official SARM source and checkpoint boundary."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

from _lg_r1_common import (
    read_yaml,
    resolve_repo_path,
    sha256_path,
    write_json,
)


def _installed_source() -> tuple[Path | None, list[dict[str, Any]]]:
    try:
        specification = importlib.util.find_spec("lerobot.rewards.sarm")
    except ModuleNotFoundError:
        specification = None
    if specification is None or specification.origin is None:
        return None, []
    source_root = Path(specification.origin).parent
    files = []
    for path in sorted(source_root.glob("*.py")):
        files.append(
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return source_root, files


def build_manifest(config_path: Path) -> dict[str, Any]:
    """Build the source-backed SARM stack manifest."""

    config = read_yaml(config_path)
    source_root, files = _installed_source()
    installed_version: str | None = None
    try:
        installed_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        pass
    stack_available = source_root is not None
    return {
        "schema_version": "latentguard.lg_r1.sarm_stack_manifest.v1",
        "status": "pass" if stack_available else "source_review_only",
        "implementation": config["implementation"],
        "repository": "huggingface/lerobot",
        "release": "v0.6.0",
        "exact_commit": config["lerobot_commit"],
        "installed_version": installed_version,
        "official_release_component": True,
        "post_release_main_used": False,
        "module_available_in_runtime": stack_available,
        "source_files": files,
        "installation": "LeRobot reward-model extra in the frozen LG-R0 environment",
        "architecture": {
            "visual_language_backbone": config["clip_model_id"],
            "clip_revision": config["clip_revision"],
            "stage_module": "lerobot.rewards.sarm.StageTransformer",
            "progress_module": "lerobot.rewards.sarm.SubtaskTransformer",
            "frame_sampling": {
                "n_obs_steps": config["n_obs_steps"],
                "frame_gap": config["frame_gap"],
                "max_rewind_steps": config["max_rewind_steps"],
            },
            "inputs": [
                "CLIP image embeddings",
                "CLIP task-text embedding",
                "allowlisted robot state",
            ],
            "outputs": [
                "stage logits",
                "stage completion",
                "overall progress",
            ],
            "unsupported_outputs": [
                "action-conditioned pairwise progress",
                "reward delta",
            ],
        },
        "checkpoint": {
            "official_compatible_libero_checkpoint_found": False,
            "checkpoint_revision": None,
            "training_data": None,
            "zero_shot_supported_for_lg_r1_schema": False,
            "community_checkpoints_accepted_as_official": False,
        },
        "supervision": {
            "official_requires_subtask_annotations": True,
            "lg_r1_uses_simulator_grounded_stage_adapters": True,
            "label_protocol_is_official_sarm_reproduction": False,
        },
        "processor": {
            "serialization_required": True,
            "inference_memory_measured_in_remote_run": False,
            "training_memory_measured_in_remote_run": False,
        },
        "license": "Apache-2.0 source headers; upstream dependency terms apply",
        "decision": config["claim_label"],
    }


def main() -> None:
    """Write the exact source/checkpoint audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/sarm.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1/sarm_stack_manifest.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    payload = build_manifest(resolve_repo_path(args.config))
    if not args.validate_only:
        write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
