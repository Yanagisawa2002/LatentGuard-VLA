"""Collect one pre-registered LG-R1b phase with the frozen LG-R1 stack."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from _lg_r0_runtime import (
    configure_libero_assets,
    instrument_policy,
    load_processors,
    load_stack,
)
from _lg_r1b_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    write_json,
)
from lg_r1_collect_rollouts import (
    _collect_episode,
    _prepare_processor_manifest,
    _public_episode,
)

from latentguard.progress.lg_r1b import build_phase_jobs, validate_registry
from latentguard.progress.rollout import (
    RolloutJob,
    completed_episode_ids,
    write_completion_marker,
)


def _rollout_jobs(registry: dict[str, Any], phase: str) -> list[RolloutJob]:
    return [
        RolloutJob(
            episode_id=str(job["episode_id"]),
            suite=str(job["suite"]),
            task_id=int(job["task_id"]),
            task_name=str(job["task_name"]),
            instruction=str(job["instruction"]),
            episode_horizon=int(job["episode_horizon"]),
            seed=int(job["seed"]),
            phase=str(job["phase"]),
        )
        for job in build_phase_jobs(registry, phase)
    ]


def _load_completed_entries(
    completion_root: Path,
    episode_manifest_root: Path,
) -> list[dict[str, Any]]:
    entries = []
    for marker_path in sorted(completion_root.glob("*.json")):
        marker = read_json(marker_path)
        manifest_path = episode_manifest_root / f"{marker['episode_id']}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing episode manifest: {manifest_path}")
        entries.append(read_json(manifest_path))
    return entries


def _manifest(
    *,
    entries: list[dict[str, Any]],
    registry_path: Path,
    expected_episode_ids: set[str] | None,
    errors: list[dict[str, str]],
    processor_status: str,
    phase: str,
) -> dict[str, Any]:
    selected = (
        [entry for entry in entries if entry["episode_id"] in expected_episode_ids]
        if expected_episode_ids is not None
        else entries
    )
    scheduled = (
        len(expected_episode_ids) if expected_episode_ids is not None else len(selected)
    )
    valid = len(selected)
    return {
        "schema_version": "latentguard.lg_r1b.rollout_manifest.v1",
        "status": "pass" if not errors and valid == scheduled else "incomplete",
        "phase": phase,
        "execution_site": "remote",
        "runtime_identity": runtime_identity(),
        "task_registry_sha256": sha256_path(registry_path),
        "scheduled_episodes": scheduled,
        "valid_episodes": valid,
        "task_count": len({(entry["suite"], entry["task_id"]) for entry in selected}),
        "suite_count": len({entry["suite"] for entry in selected}),
        "natural_failed_episodes": sum(
            not bool(entry["success"]) for entry in selected
        ),
        "failed_tasks": len(
            {
                (entry["suite"], entry["task_id"])
                for entry in selected
                if not bool(entry["success"])
            }
        ),
        "reload_completeness": valid / scheduled if scheduled else math.nan,
        "processor_identity_completeness": 1.0,
        "checkpoint_identity_completeness": 1.0,
        "nonfinite_actions": 0,
        "action_contract_errors": 0,
        "environment_infrastructure_errors": len(errors),
        "episodes": [_public_episode(entry) for entry in selected],
        "processor_manifest": "processor_serialization_manifest.json",
        "processor_manifest_status": processor_status,
        "errors": errors,
        "upload_to_hub": False,
        "policy_optimizer_steps": 0,
        "policy_backward_calls": 0,
        "synthetic_failure_samples": 0,
        "action_corruption_used": False,
    }


def main() -> None:
    """Collect or validate exactly one outcome-independent phase."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1b/pilot.yaml"),
    )
    parser.add_argument("--phase")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    registry_path = resolve_repo_path(Path(str(config["task_registry"])))
    registry = read_yaml(registry_path)
    validate_registry(registry)
    phase = str(args.phase or config.get("phase", ""))
    if not phase:
        raise ValueError("rollout phase must be explicit")
    jobs = _rollout_jobs(registry, phase)
    expected = config.get("expected_episodes")
    if expected is not None and len(jobs) != int(expected):
        raise ValueError(f"phase {phase} has {len(jobs)} jobs, expected {expected}")
    if args.max_episodes is not None:
        if args.max_episodes < 1:
            raise ValueError("--max-episodes must be positive")
        jobs = jobs[: args.max_episodes]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "phase": phase,
                    "scheduled_episodes": len(jobs),
                    "seeds": [job.seed for job in jobs],
                },
                sort_keys=True,
            )
        )
        return
    if config.get("record_all") is not True:
        raise ValueError("LG-R1b requires complete episode recording")
    runtime = output_root()
    dataset_root = runtime / "datasets"
    sidecar_root = runtime / "privileged_sidecars"
    attempt_root = runtime / "attempts"
    orphan_root = runtime / "orphaned"
    episode_manifest_root = runtime / "episode_manifests"
    completion_root = runtime / "completed"
    for path in (
        dataset_root,
        sidecar_root,
        attempt_root,
        orphan_root,
        episode_manifest_root,
        completion_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
    completed = completed_episode_ids(completion_root)
    if completed and not args.resume:
        raise ValueError("completed episodes exist; pass --resume")
    pending = [job for job in jobs if job.episode_id not in completed]
    configure_libero_assets()
    stack = load_stack()
    preprocessor, postprocessor = load_processors(stack)
    processor = _prepare_processor_manifest(runtime, stack, preprocessor, postprocessor)
    instrumentation = instrument_policy(stack.policy)
    errors: list[dict[str, str]] = []
    for job in pending:
        try:
            final_dataset = dataset_root / job.episode_id
            final_sidecar = sidecar_root / f"{job.episode_id}.jsonl"
            for stale in (final_dataset, final_sidecar):
                if stale.exists():
                    os.replace(
                        stale,
                        orphan_root / f"{stale.name}-{time.time_ns()}",
                    )
            attempt = attempt_root / f"{job.episode_id}-{time.time_ns()}"
            attempt.mkdir(parents=True, exist_ok=False)
            entry = _collect_episode(
                job,
                stack=stack,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                instrumentation=instrumentation,
                attempt_root=attempt,
            )
            os.replace(attempt / "dataset", final_dataset)
            os.replace(attempt / "privileged_sidecar.jsonl", final_sidecar)
            attempt.rmdir()
            entry["dataset_runtime_path"] = str(final_dataset)
            entry["sidecar_runtime_path"] = str(final_sidecar)
            write_json(episode_manifest_root / f"{job.episode_id}.json", entry)
            write_completion_marker(
                completion_root,
                job,
                frame_count=int(entry["frame_count"]),
                success=bool(entry["success"]),
                termination_reason=str(entry["termination_reason"]),
                dataset_locator=f"dataset_root/{job.episode_id}",
                sidecar_locator=f"sidecar_root/{job.episode_id}.jsonl",
            )
        except Exception as exc:
            errors.append(
                {
                    "episode_id": job.episode_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            break
    entries = _load_completed_entries(completion_root, episode_manifest_root)
    phase_ids = {job.episode_id for job in jobs}
    phase_manifest = _manifest(
        entries=entries,
        registry_path=registry_path,
        expected_episode_ids=phase_ids,
        errors=errors,
        processor_status=str(processor["status"]),
        phase=phase,
    )
    write_json(runtime / f"{phase}_manifest.json", phase_manifest)
    if phase == "primary_pilot":
        write_json(runtime / "pilot_manifest.json", phase_manifest)
    aggregate = _manifest(
        entries=entries,
        registry_path=registry_path,
        expected_episode_ids=None,
        errors=errors,
        processor_status=str(processor["status"]),
        phase="aggregate",
    )
    write_json(runtime / "rollout_manifest.json", aggregate)
    print(json.dumps(phase_manifest, sort_keys=True))
    if errors:
        raise RuntimeError(errors[0]["error"])


if __name__ == "__main__":
    main()
