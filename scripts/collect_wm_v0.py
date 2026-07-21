"""Collect real WM-v0 counterfactual futures through an explicit adapter factory."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from latentguard.world_model.collector import (
    ActionCandidate,
    CounterfactualCollector,
    WorldModelCollectionAdapter,
)
from latentguard.world_model.data_schema import CandidateSource, DatasetSplit
from latentguard.world_model.manifest import build_dataset_manifest
from latentguard.world_model.serialization import write_world_model_sample


def _factory(
    specification: str,
) -> Callable[[dict[str, Any]], WorldModelCollectionAdapter]:
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("adapter factory must use module:function syntax")
    value = getattr(importlib.import_module(module_name), function_name)
    if not callable(value):
        raise ValueError("adapter factory is not callable")
    return cast(Callable[[dict[str, Any]], WorldModelCollectionAdapter], value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-anchors", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Run a bounded exact-restore collection with an external integration."""

    args = _parser().parse_args()
    config: dict[str, Any] = json.loads(args.config.read_text("utf-8"))
    jobs: Any = json.loads(args.jobs.read_text("utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty JSON list")
    selected = jobs[: args.limit_anchors] if args.limit_anchors else jobs
    if args.dry_run:
        print(
            json.dumps({"anchor_count": len(selected), "dry_run": True}, sort_keys=True)
        )
        return 0
    adapter = _factory(args.adapter_factory)(config)
    collector = CounterfactualCollector(
        adapter,
        prediction_horizon=int(config["prediction_horizon"]),
        observation_stride=int(config["observation_stride"]),
    )
    samples = []
    for raw in selected:
        if not isinstance(raw, dict):
            raise ValueError("each collection job must be an object")
        candidates = tuple(
            ActionCandidate(
                candidate_id=str(candidate["candidate_id"]),
                actions=np.asarray(candidate["actions"], dtype=np.float32),
                action_mask=np.asarray(candidate["action_mask"], dtype=np.bool_),
                policy_source=CandidateSource(str(candidate["policy_source"])),
                corruption_type=(
                    None
                    if candidate.get("corruption_type") is None
                    else str(candidate["corruption_type"])
                ),
            )
            for candidate in raw["candidates"]
        )
        samples.extend(
            collector.collect(
                episode_id=str(raw["episode_id"]),
                source_episode_id=str(raw["source_episode_id"]),
                split_group_id=str(raw["split_group_id"]),
                anchor_id=str(raw["anchor_id"]),
                task_id=str(raw["task_id"]),
                instruction=(
                    None if raw.get("instruction") is None else str(raw["instruction"])
                ),
                split=DatasetSplit(str(raw["split"])),
                candidates=candidates,
            )
        )
    for sample in samples:
        write_world_model_sample(args.output / "samples", sample)
    manifest = build_dataset_manifest(samples)
    manifest.write(args.output / "dataset-manifest.json")
    print(json.dumps(manifest.as_mapping(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
