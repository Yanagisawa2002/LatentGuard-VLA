"""Fail-closed validation for a persisted WM-v0 D2 policy registry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from latentguard.policies import PolicyRegistry


def _read_json(path: Path) -> Mapping[str, object]:
    source = path.absolute()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"expected regular unlinked registry: {source}")
    value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(value, Mapping):
        raise ValueError("policy registry must be a JSON object")
    return cast(Mapping[str, object], value)


def main() -> int:
    """Validate structure and require an accepted policy unless diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    registry = PolicyRegistry.from_mapping(_read_json(args.registry))
    payload = {
        "accepted_compatible_policy_count": registry.accepted_count,
        "registry_digest": registry.registry_digest,
        "result": registry.result,
        "valid_structure": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if registry.accepted_count == 0 and not args.allow_blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
