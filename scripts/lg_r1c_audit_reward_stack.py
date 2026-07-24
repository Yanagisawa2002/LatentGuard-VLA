"""Freeze exact external reward-model source and checkpoint identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1c_common import (
    read_yaml,
    resolve_repo_path,
    validate_repository_lineage,
    write_json,
)

LEROBOT_INTEGRATION_HASHES = {
    "robometer/configuration_robometer.py": (
        "5e27a39cd85cac42c8513636e5fe0e1c50619640bae9cd3f7920ea1cabd2938f"
    ),
    "robometer/modeling_robometer.py": (
        "8b762141a0bebe7fc26c9bb4b392510702c33e3c7618fdc907c18c6d766a1185"
    ),
    "robometer/processor_robometer.py": (
        "f3de52df2110346578098129cad8d3ce8c260807a2eabe4e4aca69757e8e085e"
    ),
    "topreward/configuration_topreward.py": (
        "670dbeab412006eef296b63e1976a8fd67100764ab3f248c0423f54847b931a4"
    ),
    "topreward/modeling_topreward.py": (
        "9eb80053dd0e537a5f3f07b68934865232772ab2a66dfb7d618b0c1c1440af1c"
    ),
    "topreward/processor_topreward.py": (
        "d193598da329102e1b8a9e88d7bad766bd07c8ca3f8e1c0a6802d4fa42ead471"
    ),
}


def build_manifest(config_path: Path) -> dict[str, Any]:
    """Build the pre-inference external stack manifest."""

    config = read_yaml(resolve_repo_path(config_path))
    if config.get("schema_version") != "latentguard.lg_r1c.reward_stack.v1":
        raise ValueError("unexpected reward-stack schema")
    robometer = config.get("robometer")
    topreward = config.get("topreward")
    lerobot = config.get("lerobot")
    if not all(isinstance(item, dict) for item in (robometer, topreward, lerobot)):
        raise ValueError("reward stack requires lerobot, robometer, and topreward")
    assert isinstance(robometer, dict)
    assert isinstance(topreward, dict)
    assert isinstance(lerobot, dict)
    if lerobot.get("commit") != "30da8e687a6dfc617fcd94afc367ac7071c376ce":
        raise ValueError("LeRobot integration commit drift")
    if robometer.get("checkpoint_revision") in {None, "main"}:
        raise ValueError("ROBOMETER checkpoint must use an exact revision")
    if topreward.get("model_revision") in {None, "main"}:
        raise ValueError("TOPReward backbone must use an exact revision")
    if robometer.get("foundation_model_training_allowed") is not False:
        raise ValueError("ROBOMETER training must be prohibited")
    if topreward.get("foundation_model_training_allowed") is not False:
        raise ValueError("TOPReward training must be prohibited")
    return {
        "schema_version": "latentguard.lg_r1c.reward_stack_manifest.v1",
        "status": "frozen_before_inference",
        "repository": validate_repository_lineage(),
        "sources": {
            "lerobot": lerobot,
            "robometer_upstream": {
                "repository": robometer["upstream_repository"],
                "commit": robometer["upstream_commit"],
                "license": robometer["upstream_license"],
            },
            "topreward_upstream": {
                "repository": topreward["upstream_repository"],
                "commit": topreward["upstream_commit"],
                "license": topreward["upstream_license"],
            },
        },
        "lerobot_integration_file_sha256": LEROBOT_INTEGRATION_HASHES,
        "lerobot_integration_hash_semantic": "canonical_lf_source_bytes",
        "robometer": robometer,
        "topreward": topreward,
        "audit_findings": {
            "lerobot_reward_ports_inference_only": True,
            "robometer_compute_reward_returns_last_frame_only": True,
            "robometer_read_only_private_logits_expose_per_frame_outputs": True,
            "robometer_preference_head_officially_loaded": True,
            "robometer_preference_head_queried": False,
            "topreward_output_is_window_scalar": True,
            "topreward_per_frame_output_available": False,
            "topreward_curve_requires_rolling_windows": True,
            "task_specific_training_or_prompt_tuning_allowed": False,
        },
    }


def main() -> None:
    """Write the frozen reward-stack manifest."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/reward_stack.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1c/reward_stack_manifest.json"),
    )
    args = parser.parse_args()
    payload = build_manifest(args.config)
    write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
