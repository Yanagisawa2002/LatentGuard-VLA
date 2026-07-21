"""Evaluate the fixed WM-v0 D2 real-candidate pilot gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from latentguard.policies.d2_gate import evaluate_d2_gate


def _read_json(path: Path) -> Mapping[str, object]:
    source = path.absolute()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"expected regular unlinked manifest: {source}")
    value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(value, Mapping):
        raise ValueError("D2 manifest must be a JSON object")
    return cast(Mapping[str, object], value)


def main() -> int:
    """Print the complete D2 gate and fail unless explicitly diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args()
    result = evaluate_d2_gate(_read_json(args.manifest))
    print(json.dumps(result.to_mapping(), indent=2, sort_keys=True))
    if not result.authorized and not args.allow_failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
