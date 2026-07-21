"""Audit the six-sample WM-v0 baseline without starting collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--provenance-summary", type=Path)
    parser.add_argument("--execution-audit", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    """Report candidate origins and outcome coverage from compact evidence."""

    args = _parser().parse_args()
    provenance_path = args.provenance_summary or _discover("provenance-summary.json")
    execution_path = args.execution_audit or _discover("execution-audit.json")
    provenance = _load(provenance_path)
    execution = _load(execution_path)
    anchors = provenance.get("anchors")
    if not isinstance(anchors, list):
        anchors = provenance.get("anchor_records")
    if not isinstance(anchors, list):
        anchors = provenance.get("inventory")
    if not isinstance(anchors, list):
        raise ValueError("provenance summary does not expose an anchor inventory")
    candidates: list[dict[str, Any]] = []
    for raw_anchor in anchors:
        if not isinstance(raw_anchor, dict):
            raise ValueError("provenance anchor must be an object")
        raw_candidates = raw_anchor.get("candidates")
        if isinstance(raw_candidates, list):
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    raise ValueError("provenance candidate must be an object")
                candidates.append(raw_candidate)
            continue
        candidate_ids = raw_anchor.get("candidate_ids")
        candidate_types = raw_anchor.get("candidate_types")
        if not isinstance(candidate_ids, list) or not isinstance(candidate_types, list):
            raise ValueError("provenance anchor candidates must be a list")
        if len(candidate_ids) != len(candidate_types):
            raise ValueError("candidate ID/type inventories differ")
        candidates.extend(
            {
                "candidate_id": candidate_id,
                "candidate_origin": "synthetic_corruption",
                "corruption_type": candidate_type,
            }
            for candidate_id, candidate_type in zip(
                candidate_ids, candidate_types, strict=True
            )
        )
    origins = Counter(
        str(item.get("candidate_origin", item.get("policy_source", "unknown")))
        for item in candidates
    )
    corruptions = Counter(
        str(item.get("corruption_type", item.get("candidate_type", "unknown")))
        for item in candidates
    )
    result: dict[str, object] = {
        "anchor_count": len(anchors),
        "baseline_commit": args.baseline_commit,
        "candidate_count": len(candidates),
        "candidate_origin_counts": dict(sorted(origins.items())),
        "checkpoint_count": len(
            {
                str(item["checkpoint_hash"])
                for item in candidates
                if item.get("checkpoint_hash") is not None
            }
        ),
        "corruption_counts": dict(sorted(corruptions.items())),
        "execution_evidence_present": bool(execution),
        "policy_generated_count": origins["policy_generated"],
        "schema_version": "wm-v0-d1-baseline-audit-v1",
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


def _discover(name: str) -> Path:
    matches = sorted(Path("outputs/wm_v0/retrieved").glob(f"**/{name}"))
    if not matches:
        raise FileNotFoundError(f"could not discover {name}; pass an explicit path")
    return matches[-1]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return cast(dict[str, Any], value)


if __name__ == "__main__":
    raise SystemExit(main())
