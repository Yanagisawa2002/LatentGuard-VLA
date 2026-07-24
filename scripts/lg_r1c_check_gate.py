"""Recompute the frozen LG-R1c promotion gate from an evaluation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r1c_common import read_json, resolve_repo_path, write_json


def main() -> None:
    """Validate that the persisted gate matches the final evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/lg_r1c/evaluation_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1c/lg_r2_gate.json"),
    )
    args = parser.parse_args()
    summary = read_json(resolve_repo_path(args.summary))
    gate = summary.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("evaluation summary does not contain a gate")
    if gate.get("thresholds_frozen_before_test") is not True:
        raise ValueError("LG-R1c thresholds were not frozen before test")
    write_json(resolve_repo_path(args.output), gate)
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
