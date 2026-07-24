"""Check the frozen real-candidate diversity gate before branch execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r2b0_common import (
    read_json,
    runtime_identity,
    validate_execution_checkout,
    write_json,
)


def main() -> None:
    """Fail closed when the candidate pool cannot support the registered pilot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    validate_execution_checkout(args.expected_commit)
    manifest = read_json(args.manifest.resolve())
    if manifest.get("status") != "pass":
        raise ValueError("candidate diversity report is incomplete")
    anchors = int(manifest["anchors_with_at_least_three_distinct"])
    ratio = float(manifest["anchors_with_at_least_three_distinct_ratio"])
    exact_rate = float(manifest["exact_duplicate_rate"])
    gates = {
        "valid_anchor_count": anchors >= 50,
        "anchors_with_three_distinct_ratio": ratio >= 0.70,
        "exact_duplicate_rate": exact_rate <= 0.20,
    }
    passed = all(gates.values())
    manifest["candidate_diversity_gate"] = {
        "status": "pass" if passed else "fail",
        "gates": gates,
        "checked_runtime_identity": runtime_identity(),
    }
    write_json(args.manifest.resolve(), manifest)
    print(
        json.dumps(
            {
                "status": "pass" if passed else "fail",
                "valid_anchors": anchors,
                "ratio": ratio,
                "exact_duplicate_rate": exact_rate,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
