"""Download exact LG-R0 model and LIBERO snapshots outside the Git checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _lg_r0_runtime import read_json, write_json

from latentguard.adapters.vla_jepa.constants import (
    LIBERO_ASSET_REPOSITORY,
    LIBERO_ASSET_REVISION,
)
from latentguard.adapters.vla_jepa.identity import validate_snapshot_manifest


def _tree_identity(root: Path) -> dict[str, Any]:
    """Return a deterministic content digest without retaining a large inventory."""

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".cache/"):
            continue
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                file_digest.update(block)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += size
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "content_tree_sha256": digest.hexdigest(),
    }


def main() -> None:
    """Materialize and verify all exact external assets."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--libero-asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    manifest = read_json(args.manifest)
    destinations = {
        "lerobot/VLA-JEPA-LIBERO": args.model_root / "vla-jepa-libero-735d9f6",
        "Qwen/Qwen3-VL-2B-Instruct": args.model_root / "qwen3-vl-2b-8964489",
        "facebook/vjepa2-vitl-fpc64-256": args.model_root / "vjepa2-vitl-b3c1679",
    }
    revisions: dict[str, str] = {}
    patterns: dict[str, list[str]] = {}
    for item in manifest["assets"]:
        repository = str(item["repository_or_model_id"])
        revision = str(item["revision"])
        existing = revisions.setdefault(repository, revision)
        if existing != revision:
            raise ValueError(f"manifest has multiple revisions for {repository}")
        patterns.setdefault(repository, []).append(str(item["relative_path"]))

    for repository, destination in destinations.items():
        snapshot_download(
            repo_id=repository,
            revision=revisions[repository],
            local_dir=destination,
            allow_patterns=patterns[repository],
        )
    validate_snapshot_manifest(manifest, destinations)

    snapshot_download(
        repo_id=LIBERO_ASSET_REPOSITORY,
        repo_type="dataset",
        revision=LIBERO_ASSET_REVISION,
        local_dir=args.libero_asset_root,
    )
    required_asset_directories = (
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "stable_hope_objects",
    )
    missing = [
        name
        for name in required_asset_directories
        if not (args.libero_asset_root / name).is_dir()
    ]
    if missing:
        raise RuntimeError(f"downloaded LIBERO asset snapshot is incomplete: {missing}")

    payload = {
        "schema_version": "latentguard.lg_r0.external_assets.v1",
        "status": "pass",
        "model_revisions": revisions,
        "model_roots_runtime_only": {
            key: str(value.resolve()) for key, value in destinations.items()
        },
        "model_asset_count": len(manifest["assets"]),
        "model_manifest_validation": "pass",
        "libero_assets": {
            "repository": LIBERO_ASSET_REPOSITORY,
            "revision": LIBERO_ASSET_REVISION,
            "runtime_path": str(args.libero_asset_root.resolve()),
            **_tree_identity(args.libero_asset_root),
        },
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "pass", "output": str(args.output)}))


if __name__ == "__main__":
    main()
