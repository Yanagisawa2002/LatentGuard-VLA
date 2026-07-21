"""Collect real WM-v0 futures, including transactional D1 collection."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from latentguard.world_model.candidates import (
    CandidateContext,
    CandidateOrigin,
    CandidateProvider,
    PolicyCandidate,
)
from latentguard.world_model.collector import (
    ActionCandidate,
    CounterfactualCollector,
    WorldModelCollectionAdapter,
)
from latentguard.world_model.d1_collection import (
    D1AnchorJob,
    D1CollectionRunner,
    EpisodePhase,
    load_or_build_d1_reports,
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
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--adapter-factory")
    parser.add_argument("--candidate-provider-factory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-anchors", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-anchors", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--policy-source", choices=[item.value for item in CandidateOrigin]
    )
    parser.add_argument("--scene-group")
    parser.add_argument("--episode-range", help="zero-based START:STOP job slice")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


class _ManifestCandidateProvider:
    """Expose a frozen outcome-free job manifest through CandidateProvider."""

    def __init__(self, candidates: list[Mapping[str, object]]) -> None:
        self._candidates = candidates

    def generate(
        self,
        observation: dict[str, np.ndarray],  # type: ignore[type-arg]
        proprioception: np.ndarray,  # type: ignore[type-arg]
        instruction: str | None,
        context: CandidateContext,
    ) -> list[PolicyCandidate]:
        """Load predeclared candidates without claiming policy inference."""

        del observation, proprioception, instruction, context
        return [_policy_candidate(item) for item in self._candidates]


class _FilteredCandidateProvider:
    """Apply an explicit origin filter without changing provider metadata."""

    def __init__(self, provider: CandidateProvider, origin: CandidateOrigin) -> None:
        self._provider = provider
        self._origin = origin

    def generate(
        self,
        observation: dict[str, np.ndarray],  # type: ignore[type-arg]
        proprioception: np.ndarray,  # type: ignore[type-arg]
        instruction: str | None,
        context: CandidateContext,
    ) -> list[PolicyCandidate]:
        """Return only candidates whose declared origin matches the filter."""

        return [
            candidate
            for candidate in self._provider.generate(
                observation, proprioception, instruction, context
            )
            if candidate.candidate_origin is self._origin
        ]


def main() -> int:
    """Run a bounded exact-restore collection with an external integration."""

    args = _parser().parse_args()
    config: dict[str, Any] = json.loads(args.config.read_text("utf-8"))
    output_value = args.output or _optional_path(config.get("output_directory"))
    if output_value is None:
        raise ValueError("collection output must be provided by CLI or config")
    if args.validate_only:
        manifest, quality = load_or_build_d1_reports(output_value)
        print(json.dumps({"manifest": manifest, "quality": quality}, sort_keys=True))
        return 0
    jobs_path = args.jobs or _optional_path(config.get("jobs"))
    if jobs_path is None:
        raise ValueError("collection jobs must be provided by CLI or config")
    jobs: Any = json.loads(jobs_path.read_text("utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty JSON list")
    selected = _select_jobs(jobs, args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "anchor_count": len(selected),
                    "candidate_count": sum(
                        len(item.get("candidates", []))
                        for item in selected
                        if isinstance(item, dict)
                    ),
                    "dry_run": True,
                    "policy_source_filter": args.policy_source,
                    "scene_group_filter": args.scene_group,
                },
                sort_keys=True,
            )
        )
        return 0
    adapter_specification = args.adapter_factory or config.get("adapter_factory")
    if not isinstance(adapter_specification, str):
        raise ValueError("adapter factory must be provided by CLI or config")
    adapter = _factory(adapter_specification)(config)
    if str(config.get("schema_version", "")).startswith("wm-v0-d1"):
        return _run_d1(args, config, selected, adapter, output_value)
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
        write_world_model_sample(output_value / "samples", sample)
    manifest = build_dataset_manifest(samples)
    manifest.write(output_value / "dataset-manifest.json")
    print(json.dumps(manifest.as_mapping(), sort_keys=True))
    return 0


def _run_d1(
    args: argparse.Namespace,
    config: dict[str, Any],
    selected: list[dict[str, Any]],
    adapter: WorldModelCollectionAdapter,
    output: Path,
) -> int:
    """Run the transactional D1 path with either real or frozen providers."""

    jobs = tuple(_d1_job(raw) for raw in selected)
    provider_specification = args.candidate_provider_factory or config.get(
        "candidate_provider_factory"
    )
    if provider_specification is None:
        raw_by_anchor = {str(raw["anchor_id"]): raw for raw in selected}

        def provider_for_job(job: D1AnchorJob) -> CandidateProvider:
            raw_candidates = raw_by_anchor[job.anchor_id].get("candidates")
            if not isinstance(raw_candidates, list):
                raise ValueError("D1 manifest provider requires candidates list")
            values = [
                cast(Mapping[str, object], item)
                for item in raw_candidates
                if isinstance(item, Mapping)
            ]
            if len(values) != len(raw_candidates):
                raise ValueError("D1 candidate entries must be objects")
            provider: CandidateProvider = _ManifestCandidateProvider(values)
            return _origin_filter(provider, args.policy_source)

    else:
        if not isinstance(provider_specification, str):
            raise ValueError("candidate provider factory must use module:function")
        provider_factory = _candidate_provider_factory(provider_specification)
        shared_provider = provider_factory(config)

        def provider_for_job(job: D1AnchorJob) -> CandidateProvider:
            del job
            return _origin_filter(shared_provider, args.policy_source)

    result = D1CollectionRunner(
        adapter=adapter,
        output_root=output,
        prediction_horizon=int(config["prediction_horizon"]),
        observation_stride=int(config["observation_stride"]),
        expected_action_dimension=int(config["action_dimension"]),
        seed=int(config["seed"] if args.seed is None else args.seed),
    ).run(
        jobs,
        provider_for_job,
        resume=args.resume,
        max_samples=args.max_samples,
    )
    print(
        json.dumps(
            {
                "completed_anchor_count": result.completed_anchor_count,
                "formal_training_gate": result.quality_report["formal_training_gate"],
                "newly_completed_anchor_count": result.newly_completed_anchor_count,
                "pilot_gate": result.quality_report["pilot_gate"],
                "rejected_sample_count": result.rejected_sample_count,
                "valid_sample_count": result.valid_sample_count,
                "zero_work_resume": result.zero_work_resume,
            },
            sort_keys=True,
        )
    )
    return 0


def _candidate_provider_factory(
    specification: str,
) -> Callable[[dict[str, Any]], CandidateProvider]:
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("candidate provider factory must use module:function syntax")
    value = getattr(importlib.import_module(module_name), function_name)
    if not callable(value):
        raise ValueError("candidate provider factory is not callable")
    return cast(Callable[[dict[str, Any]], CandidateProvider], value)


def _origin_filter(
    provider: CandidateProvider, origin: str | None
) -> CandidateProvider:
    if origin is None:
        return provider
    return _FilteredCandidateProvider(provider, CandidateOrigin(origin))


def _optional_path(value: object) -> Path | None:
    return None if not isinstance(value, str) else Path(value)


def _select_jobs(jobs: list[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for raw in jobs:
        if not isinstance(raw, dict):
            raise ValueError("each collection job must be an object")
        if args.scene_group is not None and raw.get("scene_group") != args.scene_group:
            continue
        values.append(raw)
    if args.episode_range is not None:
        start_text, separator, stop_text = args.episode_range.partition(":")
        if not separator:
            raise ValueError("--episode-range must use START:STOP")
        start, stop = int(start_text), int(stop_text)
        if start < 0 or stop <= start:
            raise ValueError("--episode-range requires 0 <= START < STOP")
        values = values[start:stop]
    limit = args.max_anchors or args.limit_anchors
    if limit is not None:
        if limit < 1:
            raise ValueError("--max-anchors must be positive")
        values = values[:limit]
    if not values:
        raise ValueError("collection filters selected no jobs")
    return values


def _d1_job(raw: Mapping[str, object]) -> D1AnchorJob:
    return D1AnchorJob(
        episode_id=str(raw["episode_id"]),
        source_episode_id=str(raw["source_episode_id"]),
        split_group_id=str(raw["split_group_id"]),
        anchor_id=str(raw["anchor_id"]),
        task_id=str(raw["task_id"]),
        instruction=(
            None if raw.get("instruction") is None else str(raw["instruction"])
        ),
        split=DatasetSplit(str(raw["split"])),
        scene_group=str(raw["scene_group"]),
        episode_phase=EpisodePhase(str(raw.get("episode_phase", "unknown"))),
        task_progress_t=(
            None
            if raw.get("task_progress_t") is None
            else float(cast(float, raw["task_progress_t"]))
        ),
        time_index=int(cast(int, raw["time_index"])),
        distance_to_target=(
            None
            if raw.get("distance_to_target") is None
            else float(cast(float, raw["distance_to_target"]))
        ),
        grasp_state=(
            None if raw.get("grasp_state") is None else bool(raw["grasp_state"])
        ),
        object_state=str(raw.get("object_state", "unknown")),
    )


def _policy_candidate(raw: Mapping[str, object]) -> PolicyCandidate:
    action_mask_value = raw.get("action_mask")
    sampling = raw.get("sampling_config")
    if not isinstance(sampling, Mapping):
        raise ValueError("D1 sampling_config must be an object")
    return PolicyCandidate(
        candidate_id=str(raw["candidate_id"]),
        action_chunk=np.asarray(raw["actions"], dtype=np.float32),
        action_mask=(
            None
            if action_mask_value is None
            else np.asarray(action_mask_value, dtype=np.bool_)
        ),
        candidate_origin=CandidateOrigin(str(raw["candidate_origin"])),
        policy_family=str(raw["policy_family"]),
        policy_name=str(raw["policy_name"]),
        checkpoint_id=str(raw["checkpoint_id"]),
        checkpoint_hash=(
            None if raw.get("checkpoint_hash") is None else str(raw["checkpoint_hash"])
        ),
        inference_seed=(
            None if raw.get("inference_seed") is None else int(raw["inference_seed"])
        ),
        sampling_config=cast(Mapping[str, object], sampling),
        action_horizon=int(raw["action_horizon"]),
        generation_latency_ms=float(raw["generation_latency_ms"]),
        corruption_type=(
            None if raw.get("corruption_type") is None else str(raw["corruption_type"])
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
