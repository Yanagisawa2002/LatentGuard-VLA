"""Generate and freeze real policy candidates for every accepted anchor."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from _lg_r2b0_common import (
    batch_observation,
    current_observation,
    frozen_policy_identity,
    generate_policy_candidate,
    load_runtime_stack,
    make_libero_environment,
    observation_sha256,
    output_root,
    proprioception_sha256,
    public_candidate,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    validate_execution_checkout,
    validate_no_final_seed,
    write_json,
)

from latentguard.counterfactual.candidates import (
    CandidateDiversityThresholds,
    candidate_group_metrics,
    classify_candidate_group,
)
from latentguard.counterfactual.libero_state import (
    load_libero_state,
    restore_libero_state,
)
from latentguard.counterfactual.models import CandidateDisposition


def _thresholds(config: dict[str, Any]) -> CandidateDiversityThresholds:
    values = config["diversity_thresholds"]
    return CandidateDiversityThresholds(
        near_full_chunk_l2=float(values["near_full_chunk_l2"]),
        meaningful_full_chunk_l2=float(values["meaningful_full_chunk_l2"]),
        meaningful_endpoint_translation=float(
            values["meaningful_endpoint_translation"]
        ),
        meaningful_cumulative_rotation=float(values["meaningful_cumulative_rotation"]),
        gripper_disagreement_epsilon=float(values["gripper_disagreement_epsilon"]),
    )


def main() -> None:
    """Generate isolated native samples after source and restoration gates pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r2b0/candidates.yaml"),
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    validate_execution_checkout(args.expected_commit)
    runtime = output_root()
    source = read_json(runtime / "candidate_source_manifest.json")
    restore = read_json(runtime / "state_restore_validation.json")
    anchors = read_json(runtime / "anchor_registry.json")
    if source.get("status") != "pass":
        raise RuntimeError("real candidate source gate has not passed")
    if restore.get("status") != "pass":
        raise RuntimeError("state restore gate has not passed")
    if anchors.get("status") != "pass":
        raise RuntimeError("anchor registry has not been frozen")
    if anchors.get("frozen_before_candidate_results") is not True:
        raise RuntimeError("anchor registry freeze evidence is missing")
    manifest_path = runtime / "candidate_manifest.json"
    diversity_path = runtime / "candidate_diversity_report.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError("candidate manifest exists; pass --resume")
    threshold = _thresholds(config)
    target = int(config["candidates_per_anchor"])
    minimum = int(config["minimum_valid_candidates"])
    attempts = int(config["maximum_sampling_attempts"])
    if target != 4 or minimum != 3 or attempts < target:
        raise ValueError("candidate count contract drift")
    seed_base = int(config["candidate_seed_base"])
    all_candidate_seeds = [
        seed_base + anchor_index * attempts + attempt
        for anchor_index in range(len(anchors["anchors"]))
        for attempt in range(attempts)
    ]
    validate_no_final_seed(all_candidate_seeds, "candidate generation")
    stack, preprocessor, postprocessor = load_runtime_stack()
    manifest_anchors: list[dict[str, Any]] = []
    diversity_anchors: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(anchors["anchors"]):
        suite = str(anchor["suite"])
        task_id = int(anchor["task_id"])
        seed = int(anchor["seed"])
        horizon = int(config["episode_horizons"][suite])
        vector_env, single_env, env_pre, env_post, _ = make_libero_environment(
            suite=suite,
            task_id=task_id,
            episode_horizon=horizon,
            seed=seed,
            stack=stack,
        )
        try:
            snapshot_path = runtime / "anchors" / str(anchor["anchor_id"]) / "state"
            snapshot = load_libero_state(snapshot_path)
            if snapshot.content_sha256 != anchor["simulator_state_hash"]:
                raise ValueError("anchor snapshot identity drift")
            restoration = restore_libero_state(
                single_env,
                snapshot,
                atol=float(config["state_restore_tolerance"]),
            )
            if not restoration.within_tolerance:
                raise ValueError("fresh-session anchor restoration failed")
            raw = current_observation(single_env)
            instruction = str(anchor["source_rollout_identity"].get("instruction", ""))
            if not instruction:
                raise ValueError("anchor instruction identity is missing")
            if observation_sha256(raw, instruction) != anchor["observation_hash"]:
                raise ValueError("fresh-session observation identity mismatch")
            if proprioception_sha256(raw) != anchor["proprioception_hash"]:
                raise ValueError("fresh-session proprioception identity mismatch")
            raw_batched = batch_observation(raw)
            generated = []
            selected = []
            for attempt in range(attempts):
                inference_seed = seed_base + anchor_index * attempts + attempt
                candidate = generate_policy_candidate(
                    stack=stack,
                    raw_observation=raw_batched,
                    instruction=instruction,
                    env_preprocessor=env_pre,
                    env_postprocessor=env_post,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    anchor_id=str(anchor["anchor_id"]),
                    candidate_id=f"{anchor['anchor_id']}:sample{attempt}",
                    inference_seed=inference_seed,
                )
                provisional = generated + [candidate]
                dispositions = classify_candidate_group(
                    [item.native_action_chunk for item in provisional],
                    threshold,
                )
                candidate = replace(candidate, disposition=dispositions[-1])
                generated.append(candidate)
                if (
                    candidate.disposition == CandidateDisposition.MEANINGFULLY_DISTINCT
                    and len(selected) < target
                ):
                    selected.append(candidate)
                if len(selected) == target:
                    break
            valid = len(selected) >= minimum
            if not valid:
                status = "INSUFFICIENT_REAL_CANDIDATE_DIVERSITY"
                selected = []
            else:
                status = "pass"
            metrics = candidate_group_metrics(
                [candidate.candidate_id for candidate in selected],
                [candidate.native_action_chunk for candidate in selected],
                threshold,
            )
            manifest_anchors.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "status": status,
                    "instruction": instruction,
                    "generated_candidates": len(generated),
                    "valid_candidates": len(selected),
                    "candidate_ids": [candidate.candidate_id for candidate in selected],
                    "candidates": [
                        public_candidate(candidate) for candidate in selected
                    ],
                    "all_generated_audit": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "content_sha256": candidate.content_sha256,
                            "inference_seed": candidate.inference_seed,
                            "disposition": candidate.disposition.value,
                        }
                        for candidate in generated
                    ],
                    "restoration": {
                        "expected_sha256": restoration.expected_sha256,
                        "observed_sha256": restoration.observed_sha256,
                        "within_tolerance": restoration.within_tolerance,
                        "maximum_absolute_error": (restoration.maximum_absolute_error),
                    },
                }
            )
            diversity_anchors.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "status": status,
                    **metrics,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        finally:
            vector_env.close()
    valid_anchor_count = sum(
        int(item["valid_candidates"]) >= minimum for item in manifest_anchors
    )
    selected_candidates = sum(
        int(item["valid_candidates"]) for item in manifest_anchors
    )
    completed = not errors and len(manifest_anchors) == len(anchors["anchors"])
    manifest = {
        "schema_version": "latentguard.lg_r2b0.candidate_manifest.v1",
        "status": "pass" if completed else "fail",
        "frozen_before_branch_outcomes": completed,
        "branch_outcomes_available_during_generation": False,
        "runtime_identity": runtime_identity(),
        "config_sha256": sha256_path(config_path),
        "anchor_registry_sha256": sha256_path(runtime / "anchor_registry.json"),
        "candidate_source_manifest_sha256": sha256_path(
            runtime / "candidate_source_manifest.json"
        ),
        "state_restore_validation_sha256": sha256_path(
            runtime / "state_restore_validation.json"
        ),
        "policy_identity": frozen_policy_identity(),
        "valid_anchors": valid_anchor_count,
        "selected_candidates": selected_candidates,
        "policy_generated_candidate_ratio": 1.0,
        "synthetic_candidate_ratio": 0.0,
        "anchors": manifest_anchors,
        "errors": errors,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "final_seeds_accessed": False,
    }
    write_json(manifest_path, manifest)
    total_generated = sum(
        int(item["generated_candidates"]) for item in manifest_anchors
    )
    exact_duplicates = sum(
        item["disposition"] == CandidateDisposition.EXACT_DUPLICATE.value
        for anchor in manifest_anchors
        for item in anchor["all_generated_audit"]
    )
    near_duplicates = sum(
        item["disposition"] == CandidateDisposition.NEAR_DUPLICATE.value
        for anchor in manifest_anchors
        for item in anchor["all_generated_audit"]
    )
    distinct_anchor_ratio = (
        valid_anchor_count / len(anchors["anchors"]) if anchors["anchors"] else 0.0
    )
    diversity = {
        "schema_version": "latentguard.lg_r2b0.candidate_diversity_report.v1",
        "status": "pass" if completed else "fail",
        "runtime_identity": runtime_identity(),
        "candidate_manifest_sha256": sha256_path(manifest_path),
        "thresholds": {
            "near_full_chunk_l2": threshold.near_full_chunk_l2,
            "meaningful_full_chunk_l2": threshold.meaningful_full_chunk_l2,
            "meaningful_endpoint_translation": (
                threshold.meaningful_endpoint_translation
            ),
            "meaningful_cumulative_rotation": (
                threshold.meaningful_cumulative_rotation
            ),
            "gripper_disagreement_epsilon": (threshold.gripper_disagreement_epsilon),
            "registered_before_candidate_results": True,
        },
        "anchors": diversity_anchors,
        "total_generated_candidates": total_generated,
        "selected_valid_candidates": selected_candidates,
        "anchors_with_at_least_three_distinct": valid_anchor_count,
        "anchors_with_at_least_three_distinct_ratio": distinct_anchor_ratio,
        "exact_duplicates": exact_duplicates,
        "exact_duplicate_rate": (
            exact_duplicates / total_generated if total_generated else 0.0
        ),
        "near_duplicates": near_duplicates,
        "near_duplicate_rate": (
            near_duplicates / total_generated if total_generated else 0.0
        ),
    }
    write_json(diversity_path, diversity)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "valid_anchors": valid_anchor_count,
                "selected_candidates": selected_candidates,
                "exact_duplicate_rate": diversity["exact_duplicate_rate"],
                "output": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    if not completed:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
