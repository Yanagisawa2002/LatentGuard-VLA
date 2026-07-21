"""Prepare an outcome-blind, phase-aware PickCube WM-v0 D1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from latentguard.action_verifier.models import DatasetSplit as VerifierSplit
from latentguard.action_verifier.splits import (
    FULL_SPLIT_COUNTS,
    TrajectorySplitSourceV1,
    assign_trajectory_splits,
)
from latentguard.control.serialization import write_atomic_json
from latentguard.integrations.maniskill_pickcube.world_model_d1 import (
    pickcube_d1_anchor_metadata,
)

_REASONS = (
    "early_trajectory",
    "approach_phase",
    "first_grasp_transition",
    "late_transport",
    "near_placement",
    "evenly_spaced_fill",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--config-template", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-store", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-jobs", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=30)
    parser.add_argument("--candidates-per-anchor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=271828)
    return parser


def main() -> int:
    """Select pilot anchors and serialize only frozen synthetic candidates."""

    args = _parser().parse_args()
    if args.anchor_count != 30:
        raise ValueError("reviewed D1 pilot preparation requires exactly 30 anchors")
    if args.candidates_per_anchor != 4:
        raise ValueError(
            "reviewed D1 pilot requires exactly four candidates per anchor"
        )
    pool = _mapping(args.candidate_pool)
    anchors = _mapping(args.anchor_manifest)
    archive = _mapping(args.runtime_archive / "manifest.json")
    source_plans = _mapping(args.source_plans)
    template = _mapping(args.config_template)
    pool_groups = _list_of_mappings(pool, "groups")
    anchor_records = _list_of_mappings(anchors, "records")
    archive_episodes = _list_of_mappings(archive, "episodes")
    sources = _split_sources(pool_groups)
    if int(cast(int, source_plans.get("source_plan_count", -1))) != len(sources):
        raise ValueError("source-plan inventory differs from the candidate pool")
    assignments = assign_trajectory_splits(
        sources, split_counts=FULL_SPLIT_COUNTS, split_seed=args.seed
    )
    assignment_by_trajectory = {item.source_trajectory_id: item for item in assignments}
    records_by_trajectory: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in anchor_records:
        anchor = _nested_mapping(record, "anchor")
        records_by_trajectory[str(anchor["source_trajectory_id"])].append(record)
    selected = _select_records(records_by_trajectory, assignments)
    group_by_anchor = {str(group["anchor_id"]): group for group in pool_groups}
    episode_by_trajectory = {
        str(episode["source_trajectory_id"]): episode for episode in archive_episodes
    }
    jobs: list[dict[str, object]] = []
    runtime_anchors: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for ordinal, record in enumerate(selected):
        anchor = _nested_mapping(record, "anchor")
        anchor_id = str(anchor["anchor_id"])
        trajectory_id = str(anchor["source_trajectory_id"])
        state_index = int(cast(int, anchor["state_index"]))
        assignment = assignment_by_trajectory[trajectory_id]
        episode = episode_by_trajectory[trajectory_id]
        states = episode.get("states")
        if not isinstance(states, list) or not 0 <= state_index < len(states):
            raise ValueError("runtime archive state inventory differs")
        state = states[state_index]
        if not isinstance(state, Mapping):
            raise ValueError("runtime archive state must be an object")
        snapshot = _nested_mapping(state, "restored_task_snapshot")
        phase = pickcube_d1_anchor_metadata(
            selection_reason=str(anchor["selection_reason"]),
            state_index=state_index,
            task_snapshot=snapshot,
        )
        group = group_by_anchor[anchor_id]
        raw_candidates = _list_of_mappings(group, "candidates")
        if len(raw_candidates) < args.candidates_per_anchor:
            raise ValueError("candidate group is smaller than the reviewed pilot")
        candidates = [
            _candidate(
                candidate,
                pool_root=args.candidate_pool.parent,
                pool_digest=str(pool["candidate_pool_content_digest"]),
            )
            for candidate in raw_candidates[: args.candidates_per_anchor]
        ]
        split = assignment.dataset_split.value
        job = {
            "anchor_id": anchor_id,
            "candidates": candidates,
            "distance_to_target": phase.distance_to_target,
            "episode_id": f"wm-v0-d1-pilot-{ordinal:03d}",
            "episode_phase": phase.episode_phase.value,
            "grasp_state": phase.grasp_state,
            "instruction": "Pick up the cube and place it at the goal.",
            "object_state": phase.object_state,
            "scene_group": f"pickcube-seed-{assignment.source_seed}",
            "source_episode_id": assignment.split_group_id,
            "split": split,
            "split_group_id": assignment.split_group_id,
            "task_id": "PickCube-v1",
            "task_progress_t": phase.task_progress_t,
            "time_index": state_index,
        }
        jobs.append(job)
        runtime_anchors.append(
            {
                "anchor_id": anchor_id,
                "source_trajectory_id": trajectory_id,
                "state_index": state_index,
            }
        )
        inventory.append(
            {
                "anchor_id": anchor_id,
                "candidate_ids": [item["candidate_id"] for item in candidates],
                "episode_phase": phase.episode_phase.value,
                "scene_group": job["scene_group"],
                "selection_reason": anchor["selection_reason"],
                "source_trajectory_id": trajectory_id,
                "split": split,
                "state_index": state_index,
            }
        )
    config = dict(template)
    config.update(
        {
            "jobs": str(args.output_jobs.absolute()),
            "output_directory": str(args.output_root.absolute()),
            "seed": args.seed,
        }
    )
    repo = args.repo_root.absolute()
    config["pickcube_runtime"] = {
        "action_layout": str(
            repo / "configs/integrations/maniskill_pickcube/action-layout-v1.json"
        ),
        "anchors": runtime_anchors,
        "camera_rig_config": str(repo / "configs/vision/m4a/camera-rig-v1.json"),
        "compatibility_report": str(
            repo
            / "reports/m2c"
            / "20260715T080753Z_m2c-pickcube-trusted_aafe838_seed0"
            / "compatibility-trusted.json"
        ),
        "expected_contract": str(
            repo / "configs/integrations/maniskill_pickcube/expected-contract-v1.json"
        ),
        "render_domain_config": str(repo / "configs/vision/m4a/render-domains-v1.json"),
        "render_seed": args.seed,
        "source_archive_dir": str(args.runtime_archive.absolute()),
        "source_plan_manifest": str(args.source_plans.absolute()),
        "state_store": str(args.state_store.absolute()),
        "visual_domain": "canonical",
    }
    provenance = {
        "anchor_count": len(jobs),
        "candidate_count": sum(
            len(cast(list[object], job["candidates"])) for job in jobs
        ),
        "candidate_origin": "synthetic_corruption",
        "candidate_pool_content_digest": pool["candidate_pool_content_digest"],
        "candidates_per_anchor": args.candidates_per_anchor,
        "historical_outcomes_consulted": False,
        "inventory": inventory,
        "policy_generated_count": 0,
        "schema_version": "wm-v0-d1-pilot-provenance-v1",
        "selection_rule": (
            "outcome-blind deterministic 24/3/3 split selection over 60 frozen "
            "source trajectories with phase-reason cycling"
        ),
        "source_archive_content_digest": archive["archive_content_digest"],
        "split_policy_id": assignments[0].split_policy_id,
    }
    _write_json_value(args.output_jobs, jobs)
    write_atomic_json(args.output_config, config)
    write_atomic_json(args.output_provenance, provenance)
    print(json.dumps(provenance, sort_keys=True))
    return 0


def _select_records(
    records_by_trajectory: Mapping[str, Sequence[Mapping[str, object]]],
    assignments: Sequence[Any],
) -> list[Mapping[str, object]]:
    quotas = {
        VerifierSplit.TRAIN: 24,
        VerifierSplit.VALIDATION: 3,
        VerifierSplit.TEST: 3,
    }
    selected: list[Mapping[str, object]] = []
    for split in VerifierSplit:
        available = [
            item
            for item in assignments
            if item.dataset_split is split
            and item.source_trajectory_id in records_by_trajectory
        ]
        used: set[str] = set()
        for ordinal in range(quotas[split]):
            reason = _REASONS[ordinal % len(_REASONS)]
            choice: tuple[Any, Mapping[str, object]] | None = None
            for assignment in available:
                if assignment.source_trajectory_id in used:
                    continue
                for record in records_by_trajectory[assignment.source_trajectory_id]:
                    if (
                        _nested_mapping(record, "anchor").get("selection_reason")
                        == reason
                    ):
                        choice = (assignment, record)
                        break
                if choice is not None:
                    break
            if choice is None:
                assignment = next(
                    item for item in available if item.source_trajectory_id not in used
                )
                choice = (
                    assignment,
                    records_by_trajectory[assignment.source_trajectory_id][0],
                )
            used.add(choice[0].source_trajectory_id)
            selected.append(choice[1])
    return selected


def _split_sources(
    pool_groups: Sequence[Mapping[str, object]],
) -> tuple[TrajectorySplitSourceV1, ...]:
    by_trajectory: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for group in pool_groups:
        trajectory = _nested_mapping(group, "trajectory")
        by_trajectory[str(trajectory["source_trajectory_id"])].append(group)
    sources: list[TrajectorySplitSourceV1] = []
    for trajectory_id, groups in sorted(by_trajectory.items()):
        trajectory = _nested_mapping(groups[0], "trajectory")
        sources.append(
            TrajectorySplitSourceV1(
                source_trajectory_id=trajectory_id,
                source_seed=int(cast(int, trajectory["source_seed"])),
                split_group_id=str(trajectory["split_group_id"]),
                state_digests=tuple(
                    sorted(str(item) for item in trajectory["complete_state_digests"])
                ),
                anchor_ids=tuple(sorted(str(group["anchor_id"]) for group in groups)),
                proposal_ids=tuple(
                    sorted(
                        str(candidate["proposal_id"])
                        for group in groups
                        for candidate in _list_of_mappings(group, "candidates")
                    )
                ),
            )
        )
    return tuple(sources)


def _candidate(
    raw: Mapping[str, object], *, pool_root: Path, pool_digest: str
) -> dict[str, object]:
    action_record = _nested_mapping(raw, "action_chunk")
    mask_record = _nested_mapping(raw, "action_mask")
    actions = _load_array(pool_root, action_record)
    mask = _load_array(pool_root, mask_record)
    return {
        "action_horizon": int(actions.shape[0]),
        "action_mask": np.asarray(mask, dtype=np.bool_).tolist(),
        "actions": np.asarray(actions, dtype=np.float32).tolist(),
        "candidate_id": str(raw["proposal_id"]),
        "candidate_origin": "synthetic_corruption",
        "checkpoint_hash": None,
        "checkpoint_id": "not_applicable",
        "corruption_type": str(raw["corruption_type"]),
        "generation_latency_ms": 0.0,
        "inference_seed": int(cast(int, raw["seed"])),
        "policy_family": "m3c_frozen_corruption",
        "policy_name": "frozen_m3c_candidate_pool_v1",
        "sampling_config": {
            "candidate_pool_content_digest": pool_digest,
            "configuration_ordinal": raw["configuration_ordinal"],
            "distribution": raw["distribution"],
            "severity_id": raw["severity_id"],
        },
    }


def _load_array(root: Path, record: Mapping[str, object]) -> np.ndarray:  # type: ignore[type-arg]
    path = root / str(record["path"])
    observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != record["file_sha256"]:
        raise ValueError("candidate array file digest differs")
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]:
        raise ValueError("candidate array contract differs")
    if not bool(np.isfinite(value).all()):
        raise ValueError("candidate array contains non-finite values")
    return value


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def _write_json_value(path: Path, value: object) -> Path:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _nested_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], result)


def _list_of_mappings(
    value: Mapping[str, object], name: str
) -> list[Mapping[str, object]]:
    result = value.get(name)
    if not isinstance(result, list) or not all(
        isinstance(item, Mapping) for item in result
    ):
        raise ValueError(f"{name} must be a list of objects")
    return cast(list[Mapping[str, object]], result)


if __name__ == "__main__":
    raise SystemExit(main())
