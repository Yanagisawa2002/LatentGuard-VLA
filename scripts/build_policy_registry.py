"""Build a canonical WM-v0 D2 policy registry from an audited config."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from latentguard.control.serialization import write_atomic_json
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
    """Write the validated, path-independent policy registry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/wm_v0_d2/policy_registry.json"),
    )
    args = parser.parse_args()
    registry = PolicyRegistry.from_mapping(_read_json(args.config))
    write_atomic_json(args.output, registry.to_mapping())
    print(
        json.dumps(
            {
                "accepted_compatible_policy_count": registry.accepted_count,
                "output": args.output.as_posix(),
                "registry_digest": registry.registry_digest,
                "result": registry.result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
