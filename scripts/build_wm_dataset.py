"""Strictly reload raw WM-v0 samples and build one frozen feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latentguard.visual_training.backbone import prepare_backbone
from latentguard.visual_training.config import load_backbone_config
from latentguard.world_model.encoder import FrozenResNet18ObservationEncoder
from latentguard.world_model.features import write_feature_cache
from latentguard.world_model.manifest import build_dataset_manifest
from latentguard.world_model.serialization import read_world_model_sample


def main() -> int:
    """Validate raw samples, apply the accepted frozen encoder, and stop on drift."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-samples", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text("utf-8"))
    identifiers = sorted(path.stem for path in args.samples.glob("*.json"))
    if args.limit_samples:
        identifiers = identifiers[: args.limit_samples]
    samples = tuple(
        (identifier, read_world_model_sample(args.samples, identifier))
        for identifier in identifiers
    )
    manifest = build_dataset_manifest(
        (sample for _, sample in samples),
        minimum_valid_samples=int(config["minimum_valid_samples"]),
    )
    manifest.write(args.output / "dataset-manifest.json")
    encoder = FrozenResNet18ObservationEncoder(
        prepare_backbone(
            load_backbone_config(args.backbone_config),
            args.backbone_manifest,
            device=args.device,
        )
    )
    index = write_feature_cache(args.output / "features", samples, encoder)
    print(
        json.dumps(
            {
                "feature_index": str(index),
                "formal_training_authorized": manifest.gate.authorized,
                "sample_count": manifest.sample_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
