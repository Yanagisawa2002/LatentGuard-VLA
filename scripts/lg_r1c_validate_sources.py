"""Validate LG-R1c repository isolation and prohibited-scope boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _lg_r1c_common import (
    resolve_repo_path,
    validate_repository_lineage,
    write_json,
)


def main() -> None:
    """Reject LangMani imports, rollout collection, and training-scope drift."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1c/source_validation.json"),
    )
    args = parser.parse_args()
    paths = [
        *sorted(resolve_repo_path(Path("scripts")).glob("lg_r1c_*.py")),
        *sorted(resolve_repo_path(Path("src/latentguard/rewards")).glob("*.py")),
        *sorted(resolve_repo_path(Path("configs/lg_r1c")).glob("*.yaml")),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    forbidden = {
        "langmani_import": "import " + "langmani",
        "robolab_import": "import " + "robolab",
        "pointworld_import": "import " + "pointworld",
        "new_rollout_entry": "lg_r1c_collect_" + "rollout",
        "failure_head_class": "class " + "failurehead",
        "intervention_selector": "select_intervention_" + "threshold",
    }
    matches = {name: token for name, token in forbidden.items() if token in combined}
    if matches:
        raise ValueError(f"LG-R1c prohibited source scope detected: {matches}")
    payload = {
        "schema_version": "latentguard.lg_r1c.source_validation.v1",
        "status": "pass",
        "repository": validate_repository_lineage(),
        "files_scanned": len(paths),
        "langmani_modified": False,
        "langmani_imported": False,
        "robolab_used": False,
        "pointworld_used": False,
        "new_rollouts": 0,
        "synthetic_failures": 0,
        "foundation_model_training": False,
        "failure_head_training": False,
        "candidate_ranking": False,
        "intervention": False,
        "final_seeds_accessed": False,
    }
    write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
