"""Build a strict compact D1 dataset manifest from completed anchors."""

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
    """Reload all complete samples and write a content-bound manifest."""

    args = _parser().parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dataset config must be an object")
    config = cast(dict[str, Any], raw)
    root = Path(str(config["collection_root"]))
    manifest, quality = build_d1_reports(root)
    output = Path(str(config["manifest_output"]))
    write_atomic_json(output, manifest)
    print(
        json.dumps(
            {
                "content_digest": manifest["content_digest"],
                "formal_training_gate": quality["formal_training_gate"],
                "manifest": str(output),
                "sample_count": manifest["sample_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
