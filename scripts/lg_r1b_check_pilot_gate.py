"""Evaluate the frozen LG-R1b pilot branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r1b_common import read_json, resolve_repo_path, write_json

from latentguard.progress.lg_r1b import (
    PilotGateInput,
    evaluate_pilot_gate,
    evaluate_standard_failure_yield,
)


def main() -> None:
    """Write an auditable next-action decision from a complete pilot."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/lg_r1b/pilot_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1b/pilot_gate.json"),
    )
    args = parser.parse_args()
    manifest = read_json(resolve_repo_path(args.manifest))
    if int(manifest["valid_episodes"]) >= 200 and int(manifest["task_count"]) >= 16:
        result = evaluate_standard_failure_yield(
            valid_episodes=int(manifest["valid_episodes"]),
            task_groups=2,
            natural_failures=int(manifest["natural_failed_episodes"]),
        )
    else:
        result = evaluate_pilot_gate(
            PilotGateInput(
                valid_episodes=int(manifest["valid_episodes"]),
                expected_episodes=int(manifest["scheduled_episodes"]),
                natural_failures=int(manifest["natural_failed_episodes"]),
                infrastructure_errors=int(
                    manifest["environment_infrastructure_errors"]
                ),
            )
        )
    result["source_manifest"] = str(args.manifest.as_posix())
    write_json(resolve_repo_path(args.output), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
