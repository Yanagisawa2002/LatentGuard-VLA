"""Validate the frozen LG-RB0 protocol without network, GPU, or RoboLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from latentguard.adapters.robolab.replay_adapter import ReplayContract

FINAL_SEEDS = frozenset(range(900_000, 900_100))
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("LG-RB0 protocol must be a mapping")
    return value


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Reject changes to identity, seed isolation, and registered gate values."""
    stack = config["external_stack"]
    if stack["repository"] != "https://github.com/NVLabs/RoboLab.git":
        raise ValueError("RoboLab must use the official NVLabs repository")
    if stack["release"] != "v0.2.1" or stack["commit"] != ROBOLAB_COMMIT:
        raise ValueError("RoboLab release identity changed")
    if (
        str(stack["python"]) != "3.11"
        or str(stack["isaac_sim"]) != "5.1.0"
        or str(stack["isaac_lab"]) != "2.3.2.post1"
    ):
        raise ValueError("supported Isaac/Python stack changed")
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
    if int(takeover["repeats"]) != contract.repeats:
        raise ValueError("faithful and takeover repeat counts must match")
    recording = config["recording"]
    seeds = [int(value) for value in recording["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("recording seeds must be unique")
    overlap = sorted(set(seeds) & FINAL_SEEDS)
    if overlap:
        raise ValueError(f"recording seeds access sealed final range: {overlap}")
    expected = len(recording["task_queries"]) * len(seeds)
    if int(recording["expected_episode_count"]) != expected or expected < 10:
        raise ValueError("LG-RB0 must preregister at least ten recordings")
    if set(takeover["branches"]) != {"A", "B"}:
        raise ValueError("exactly the preregistered A and B branches are required")
    if takeover["isolation_orders"] != ["A-B-A", "B-A-B"]:
        raise ValueError("isolation order changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise ValueError("every LG-RB0 prohibition must remain active")
    return {
        "schema_version": "lg_rb0_source_validation_v1",
        "status": "pass",
        "robolab_commit": ROBOLAB_COMMIT,
        "recorded_episode_count": expected,
        "recording_seeds": seeds,
        "final_seeds_accessed": False,
        "contract": {
            "official_state_tolerance": contract.official_state_tolerance,
            "takeover_state_tolerance": contract.takeover_state_tolerance,
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
        default=Path("configs/lg_rb0/protocol.yaml"),
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
