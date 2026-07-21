"""Summarize the WM-v0 D2 policy-asset audit configuration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from latentguard.policies import PolicyRegistry


def _read_json(path: Path) -> Mapping[str, object]:
    source = path.absolute()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"expected regular unlinked config: {source}")
    value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(value, Mapping):
        raise ValueError("policy registry config must be a JSON object")
    return cast(Mapping[str, object], value)


def main() -> int:
    """Validate the audit input and print its bounded status inventory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    registry = PolicyRegistry.from_mapping(_read_json(args.config))
    statuses = Counter(entry.compatibility_status.value for entry in registry.entries)
    payload = {
        "accepted_compatible_policy_count": registry.accepted_count,
        "audited_policy_count": len(registry.entries),
        "registry_digest": registry.registry_digest,
        "result": registry.result,
        "status_counts": dict(sorted(statuses.items())),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
