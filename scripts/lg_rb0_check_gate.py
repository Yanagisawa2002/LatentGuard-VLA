"""Aggregate immutable LG-RB0 runtime evidence into the LG-RB1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latentguard.adapters.robolab.branch_runner import (
    BranchGateInput,
    evaluate_lg_rb1_gate,
)


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    """Read all required artifacts and write the conservative result gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    source = _read(root / "source_validation.json")
    stack = _read(root / "robolab_stack_manifest.json")
    remote = _read(root / "remote_execution_audit.json")
    recordings = _read(root / "recording_manifest.json")
    faithful = _read(root / "faithful_replay_validation.json")
    prefix = _read(root / "prefix_replay_validation.json")
    branch = _read(root / "branch_determinism_validation.json")
    isolation = _read(root / "branch_isolation_validation.json")
    input_evidence = BranchGateInput(
        valid_recorded_episodes=int(recordings["valid_episode_count"]),
        faithful_initial_restore_failures=int(faithful["initial_restore_failures"]),
        faithful_per_step_failures=int(faithful["per_step_state_failures"]),
        faithful_terminal_mismatches=int(faithful["terminal_mismatches"]),
        faithful_success_mismatches=int(faithful["success_mismatches"]),
        prefix_mismatches=int(prefix["mismatch_count"]),
        branch_mismatches=int(branch["mismatch_count"]),
        isolation_mismatches=int(isolation["mismatch_count"]),
        semantic_coverage_complete=bool(
            prefix["semantic_coverage_complete"]
            and branch["semantic_coverage_complete"]
        ),
        official_stack_valid=stack["status"] == "pass",
        source_validation_passed=source["status"] == "pass",
        remote_audit_passed=remote["status"] == "pass",
    )
    gate = evaluate_lg_rb1_gate(input_evidence)
    gate["inputs"] = {
        "valid_recorded_episodes": input_evidence.valid_recorded_episodes,
        "faithful_initial_restore_failures": (
            input_evidence.faithful_initial_restore_failures
        ),
        "faithful_per_step_failures": input_evidence.faithful_per_step_failures,
        "faithful_terminal_mismatches": input_evidence.faithful_terminal_mismatches,
        "faithful_success_mismatches": input_evidence.faithful_success_mismatches,
        "prefix_mismatches": input_evidence.prefix_mismatches,
        "branch_mismatches": input_evidence.branch_mismatches,
        "isolation_mismatches": input_evidence.isolation_mismatches,
        "semantic_coverage_complete": input_evidence.semantic_coverage_complete,
    }
    path = root / "lg_rb1_gate.json"
    path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
