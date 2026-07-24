"""Validate the complete LG-R2b0 pre-registration without external runtimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from latentguard.counterfactual.candidates import CandidateDiversityThresholds

FINAL_SEEDS = frozenset(range(900_000, 900_100))


def _read(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config is not a mapping: {path}")
    if payload.get("milestone") != "LG-R2b0":
        raise ValueError(f"config milestone drift: {path}")
    return payload


def validate_configs(root: Path) -> dict[str, Any]:
    """Return the frozen schedule, seed, threshold, and gate validation."""
    root = root.resolve()
    candidates = _read(root / "candidates.yaml")
    restore = _read(root / "state_restore.yaml")
    anchors = _read(root / "anchors.yaml")
    collection = _read(root / "collection.yaml")
    analysis = _read(root / "analysis.yaml")
    thresholds = candidates["diversity_thresholds"]
    CandidateDiversityThresholds(
        near_full_chunk_l2=float(thresholds["near_full_chunk_l2"]),
        meaningful_full_chunk_l2=float(thresholds["meaningful_full_chunk_l2"]),
        meaningful_endpoint_translation=float(
            thresholds["meaningful_endpoint_translation"]
        ),
        meaningful_cumulative_rotation=float(
            thresholds["meaningful_cumulative_rotation"]
        ),
        gripper_disagreement_epsilon=float(thresholds["gripper_disagreement_epsilon"]),
    )
    schedules = anchors["task_schedules"]
    operational_seeds = [
        *[int(value) for value in candidates["source_audit"]["sampling_seeds"]],
        int(candidates["candidate_seed_base"]),
        int(anchors["source_policy_seed_base"]),
        int(restore["source_policy_seed_base"]),
        int(restore["continuation_seed_base"]),
        int(collection["continuation_seed_base"]),
    ]
    anchor_seeds = [int(seed) for schedule in schedules for seed in schedule["seeds"]]
    probe_seeds = [
        int(probe[key])
        for probe in restore["probes"]
        for key in ("seed", "candidate_inference_seed")
    ]
    overlap = sorted(set(operational_seeds + anchor_seeds + probe_seeds) & FINAL_SEEDS)
    if overlap:
        raise ValueError(f"sealed final seeds configured: {overlap}")
    task_keys = {
        (str(schedule["suite"]), int(schedule["task_id"])) for schedule in schedules
    }
    suites = {suite for suite, _ in task_keys}
    anchor_count = len(anchor_seeds)
    task6 = sum(
        len(schedule["seeds"])
        for schedule in schedules
        if int(schedule["task_id"]) == 6
    )
    if anchor_count != int(anchors["expected_anchors"]) or anchor_count != 60:
        raise ValueError("anchor schedule must contain exactly 60 anchors")
    if len(set(anchor_seeds)) != anchor_count:
        raise ValueError("anchor source seeds must be unique")
    if len(task_keys) < 6 or len(suites) < 2:
        raise ValueError("anchor task/suite coverage is insufficient")
    if task6 / anchor_count > float(anchors["maximum_task6_ratio"]):
        raise ValueError("task 6 anchor ratio exceeds the registered gate")
    if len(restore["probes"]) < 10 or int(restore["repeats"]) < 5:
        raise ValueError("restore audit is below the registered minimum")
    if candidates["candidates_per_anchor"] != 4:
        raise ValueError("candidate target must remain four")
    if candidates["minimum_valid_candidates"] != 3:
        raise ValueError("minimum distinct candidates must remain three")
    if collection["short_horizons"] != [7, 21, 49]:
        raise ValueError("short-horizon contract drift")
    gate = analysis["promotion_gate"]
    expected_gate = {
        "minimum_valid_anchors": 50,
        "minimum_valid_branches": 150,
        "minimum_three_distinct_anchor_ratio": 0.70,
        "maximum_exact_duplicate_rate": 0.20,
        "minimum_progress_spread_anchor_ratio": 0.30,
        "minimum_local_event_disagreement_anchors": 15,
        "minimum_mixed_terminal_anchors": 10,
        "require_non_task6_divergence": True,
        "maximum_rank_tie_rate": 0.70,
    }
    if gate != expected_gate:
        raise ValueError("promotion gate drift")
    return {
        "schema_version": "latentguard.lg_r2b0.config_validation.v1",
        "status": "pass",
        "configs": 5,
        "anchors": anchor_count,
        "tasks": len(task_keys),
        "suites": len(suites),
        "task6_anchor_ratio": task6 / anchor_count,
        "restore_probes": len(restore["probes"]),
        "restore_repeats": int(restore["repeats"]),
        "candidate_target": int(candidates["candidates_per_anchor"]),
        "minimum_distinct_candidates": int(candidates["minimum_valid_candidates"]),
        "final_seed_overlap": overlap,
    }


def main() -> None:
    """Validate configs and optionally write the compact result."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/lg_r2b0"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_configs(args.config_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
