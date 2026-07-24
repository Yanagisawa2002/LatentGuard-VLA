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
        "0a3616fdb69da2e2dddd0c5d799509254a68e289cbb8c4b9744f14356bea3de8"
    ),
    "robometer/modeling_robometer.py": (
        "a2787de1529fa768cf503bbfd9f43ae0c6d98c5bd2e3adfc926ab5477e126845"
    ),
    "robometer/processor_robometer.py": (
        "671dc4179eadd894c43aa8dd0c1503f8b7a7bfd325c79143610e20e79222b497"
    ),
    "topreward/configuration_topreward.py": (
        "6b709fafdd3a335fd458c12da415ff944696c84039489d57d40e1aded0201539"
    ),
    "topreward/modeling_topreward.py": (
        "a4b9e09bf3c1e9fc769c9653824c634c7f74d3e8c761c14b1646e38d01f8b569"
    ),
    "topreward/processor_topreward.py": (
        "cc5ff518c8030e7629a667d3e7a15cf095c09ba79aaf80cd1a566a788b433f7d"
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
