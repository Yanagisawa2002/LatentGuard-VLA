"""Re-evaluate the frozen LG-R2b promotion gate from a summary artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from _lg_r2a_common import read_json, write_json

from latentguard.action_conditioning.gates import evaluate_lg_r2b_gate


def main() -> None:
    """Evaluate and optionally write the gate."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = read_json(args.summary)
    gate = evaluate_lg_r2b_gate(summary["gate"]["inputs"])
    if args.output is not None:
        write_json(args.output, gate)
    print(gate)


if __name__ == "__main__":
    main()
