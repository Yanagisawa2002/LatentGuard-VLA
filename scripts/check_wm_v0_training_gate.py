"""Fail closed unless a strict WM-v0 D1 formal-training gate passes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="report a failed gate with exit code zero for diagnostic pipelines",
    )
    return parser


def main() -> int:
    """Return nonzero for an unauthorized formal-training manifest."""

    args = _parser().parse_args()
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("D1 manifest must be an object")
    if raw.get("schema_version") != "wm-v0-d1-dataset-manifest-v1":
        raise ValueError("unexpected D1 manifest schema")
    gate = raw.get("formal_training_gate")
    if not isinstance(gate, Mapping) or type(gate.get("authorized")) is not bool:
        raise ValueError("D1 manifest formal-training gate is incomplete")
    result = {
        "formal_training_gate_passed": cast(bool, gate["authorized"]),
        "failed_checks": gate.get("failed_checks"),
        "manifest": str(args.manifest),
        "sample_count": raw.get("sample_count"),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if gate["authorized"] or args.allow_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
