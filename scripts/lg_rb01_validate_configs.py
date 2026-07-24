"""Validate the LG-RB0.1 protocol without network, GPU, or RoboLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from latentguard.adapters.robolab.replay_adapter import ReplayContract

FINAL_SEEDS = frozenset(range(900_000, 900_100))
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
PATCH_SHA256 = "5b72f598c8a7fb8c07f1c3c2a35e136a805cb5b87b77e1f3cfbc4398c4939310"
PATCHED_TREE_DIGEST = "050397d38fb9e8ea4b9acb557b5b01f16ac64ca6"
OPTIONAL_EMPTY_NAMESPACES = ("/deformable_object", "/gripper")


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("LG-RB0.1 protocol must be a mapping")
    return value


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Reject drift in the one-patch replay compatibility contract."""
    if (
        config.get("schema_version") != "lg_rb01_protocol_v1"
        or config.get("milestone") != "LG-RB0.1"
    ):
        raise ValueError("LG-RB0.1 protocol identity changed")
    stack = config["external_stack"]
    if (
        stack["repository"] != "https://github.com/NVLabs/RoboLab.git"
        or stack["release"] != "v0.2.1"
        or stack["commit"] != ROBOLAB_COMMIT
    ):
        raise ValueError("RoboLab release identity changed")
    if (
        str(stack["python"]) != "3.11"
        or str(stack["isaac_sim"]) != "5.1.0"
        or str(stack["isaac_lab"]) != "2.3.2.post1"
    ):
        raise ValueError("supported Isaac/Python stack changed")
    patch = config["upstream_patch"]
    if (
        patch["sha256"] != PATCH_SHA256
        or patch["patched_tree_digest"] != PATCHED_TREE_DIGEST
    ):
        raise ValueError("upstream patch identity changed")

    runtime = config["runtime"]
    faithful = config["faithful_replay"]
    takeover = config["takeover"]
    contract = ReplayContract(
        official_state_tolerance=float(faithful["official_state_tolerance"]),
        takeover_state_tolerance=float(takeover["state_tolerance"]),
        repeats=int(faithful["repeats"]),
        branch_checkpoints=tuple(int(value) for value in takeover["checkpoints"]),
        anchor_fractions=tuple(float(value) for value in takeover["anchor_fractions"]),
        num_envs=int(runtime["num_envs"]),
    )
    contract.validate()
    if float(faithful["strict_state_tolerance"]) != 1e-6:
        raise ValueError("strict faithful state tolerance must remain 1e-6")
    if int(faithful["pixel_tolerance"]) != 0 or int(takeover["pixel_tolerance"]) != 0:
        raise ValueError("pixel comparison must remain exact")
    if faithful["allowed_skip_categories"] != ["EXPECTED_RUNTIME_PRESERVATION"]:
        raise ValueError("only expected runtime preservation may be non-fatal")
    if int(takeover["repeats"]) != contract.repeats:
        raise ValueError("faithful and takeover repeat counts must match")

    recording = config["recording"]
    if recording.get("reuse_milestone") != "LG-RB0" or bool(
        recording.get("rerecording_allowed")
    ):
        raise ValueError("LG-RB0.1 must reuse the frozen LG-RB0 recordings")
    seeds = [int(value) for value in recording["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("recording seeds must be unique")
    overlap = sorted(set(seeds) & FINAL_SEEDS)
    if overlap:
        raise ValueError(f"recording seeds access sealed final range: {overlap}")
    expected = len(recording["task_queries"]) * len(seeds)
    if int(recording["expected_episode_count"]) != expected or expected != 10:
        raise ValueError("LG-RB0.1 must reuse exactly ten recordings")
    if tuple(config["state_schema"]["allowed_optional_empty_namespaces"]) != (
        OPTIONAL_EMPTY_NAMESPACES
    ):
        raise ValueError("optional empty namespace allowlist changed")
    if set(takeover["branches"]) != {"A", "B"}:
        raise ValueError("exactly the preregistered A and B branches are required")
    if takeover["isolation_orders"] != ["A-B-A", "B-A-B"]:
        raise ValueError("isolation order changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ValueError("every LG-RB0.1 prohibition must remain active")
    return {
        "schema_version": "lg_rb01_source_validation_v1",
        "status": "pass",
        "robolab_commit": ROBOLAB_COMMIT,
        "patch_sha256": PATCH_SHA256,
        "patched_tree_digest": PATCHED_TREE_DIGEST,
        "recorded_episode_count": expected,
        "recording_reused": True,
        "recording_seeds": seeds,
        "final_seeds_accessed": False,
        "allowed_optional_empty_namespaces": list(OPTIONAL_EMPTY_NAMESPACES),
        "contract": {
            "official_state_tolerance": contract.official_state_tolerance,
            "strict_state_tolerance": float(faithful["strict_state_tolerance"]),
            "takeover_state_tolerance": contract.takeover_state_tolerance,
            "pixel_tolerance": 0,
            "repeats": contract.repeats,
            "branch_checkpoints": list(contract.branch_checkpoints),
            "anchor_fractions": list(contract.anchor_fractions),
            "num_envs": contract.num_envs,
        },
    }


def main() -> None:
    """Validate the protocol and optionally write compact source evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_rb01/protocol.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_protocol(_load(args.config))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
