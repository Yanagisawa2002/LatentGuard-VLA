"""Strictly reload one WM-v0 D1 dataset and publish compact reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from latentguard.control.serialization import write_atomic_json
from latentguard.world_model.d1_collection import build_d1_reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    """Build quality reports from completed anchor directories only."""

    args = _parser().parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("validation config must be an object")
    config = cast(dict[str, Any], raw)
    root = Path(str(config["collection_root"]))
    manifest, quality = build_d1_reports(root)
    manifest_output = Path(
        str(config.get("manifest_output", root / "dataset-manifest.json"))
    )
    quality_output = Path(
        str(config.get("quality_output", root / "data-quality-report.json"))
    )
    write_atomic_json(manifest_output, manifest)
    write_atomic_json(quality_output, quality)
    print(
        json.dumps(
            {
                "formal_training_gate": quality["formal_training_gate"],
                "manifest": str(manifest_output),
                "pilot_gate": quality["pilot_gate"],
                "quality": str(quality_output),
                "valid_sample_count": quality["valid_sample_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
