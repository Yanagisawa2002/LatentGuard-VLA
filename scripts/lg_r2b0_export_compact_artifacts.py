"""Copy only reviewed compact LG-R2b0 evidence and bind every byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REQUIRED = (
    "candidate_source_manifest.json",
    "state_restore_validation.json",
    "anchor_registry.json",
    "candidate_manifest.json",
    "candidate_diversity_report.json",
    "dataset_manifest.json",
    "dataset_validation.json",
    "outcome_diversity_report.json",
    "rank_stability_report.json",
    "lg_r2b1_gate.json",
    "source_validation.json",
    "remote_execution_audit.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_compact(source: Path, destination: Path) -> dict[str, Any]:
    """Copy the required JSON set, rejecting extra binary or oversized content."""
    source = source.resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    collisions = [
        name
        for name in (*REQUIRED, "artifact_hashes.json")
        if (destination / name).exists()
    ]
    if collisions:
        raise FileExistsError(
            f"compact artifact destination contains target files: {collisions}"
        )
    files: dict[str, Any] = {}
    for name in REQUIRED:
        origin = source / name
        if not origin.is_file():
            raise FileNotFoundError(f"required compact artifact is missing: {name}")
        if origin.stat().st_size > 25_000_000:
            raise ValueError(f"compact artifact exceeds size limit: {name}")
        payload = json.loads(origin.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"compact artifact is not a JSON object: {name}")
        target = destination / name
        shutil.copyfile(origin, target)
        files[name] = {
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    registry = {
        "schema_version": "latentguard.lg_r2b0.artifact_hashes.v1",
        "files": files,
    }
    registry["file_set_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (destination / "artifact_hashes.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry


def main() -> None:
    """Export one reviewed remote retrieval directory into tracked artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = export_compact(args.source, args.destination)
    print({"status": "pass", "files": len(report["files"])})


if __name__ == "__main__":
    main()
