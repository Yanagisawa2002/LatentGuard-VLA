"""Audit the native frozen VLA-JEPA same-observation sampling path."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2b0_common import (
    array_sha256,
    frozen_policy_identity,
    generate_policy_candidate,
    load_runtime_stack,
    make_libero_environment,
    observation_sha256,
    output_root,
    policy_batch,
    public_candidate,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    sha256_path,
    unbatch_observation,
    validate_execution_checkout,
    validate_no_final_seed,
    write_json,
)


def _batch_digest(batch: dict[str, Any]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        digest.update(key.encode("utf-8"))
        if hasattr(value, "detach"):
            array = value.detach().to("cpu").numpy()
            digest.update(array_sha256(array).encode("ascii"))
        else:
            digest.update(
                json.dumps(
                    value,
                    allow_nan=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            )
    return digest.hexdigest()


def main() -> None:
    """Prove or reject a real multi-candidate source before branch rollout."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r2b0/candidates.yaml"),
    )
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = resolve_repo_path(args.config)
    config = read_yaml(config_path)
    validate_execution_checkout(args.expected_commit)
    source = config["source_audit"]
    sampling_seeds = [int(seed) for seed in source["sampling_seeds"]]
    validate_no_final_seed(sampling_seeds, "candidate-source audit")
    if len(sampling_seeds) < 4 or len(set(sampling_seeds)) != len(sampling_seeds):
        raise ValueError("source audit requires at least four unique sampling seeds")
    stack, preprocessor, postprocessor = load_runtime_stack()
    vector_env, _, env_preprocessor, env_postprocessor, raw_batched = (
        make_libero_environment(
            suite=str(source["suite"]),
            task_id=int(source["task_id"]),
            episode_horizon=int(source["episode_horizon"]),
            seed=int(source["seed"]),
            stack=stack,
        )
    )
    try:
        raw = unbatch_observation(raw_batched)
        observation_hash = observation_sha256(raw, str(source["instruction"]))
        first_batch = policy_batch(
            raw_batched,
            instruction=str(source["instruction"]),
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
        )
        second_batch = policy_batch(
            raw_batched,
            instruction=str(source["instruction"]),
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
        )
        batch_hashes = [_batch_digest(first_batch), _batch_digest(second_batch)]
        repeated = [
            generate_policy_candidate(
                stack=stack,
                raw_observation=raw_batched,
                instruction=str(source["instruction"]),
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                anchor_id="candidate-source-audit",
                candidate_id=f"fixed-seed-repeat-{index}",
                inference_seed=sampling_seeds[0],
            )
            for index in range(2)
        ]
        candidates = [
            generate_policy_candidate(
                stack=stack,
                raw_observation=raw_batched,
                instruction=str(source["instruction"]),
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                anchor_id="candidate-source-audit",
                candidate_id=f"seed-{seed}",
                inference_seed=seed,
            )
            for seed in sampling_seeds
        ]
        fixed_reproducible = (
            repeated[0].content_sha256 == repeated[1].content_sha256
            and np.array_equal(
                repeated[0].normalized_action_chunk,
                repeated[1].normalized_action_chunk,
            )
            and np.array_equal(
                repeated[0].native_action_chunk,
                repeated[1].native_action_chunk,
            )
        )
        unique_hashes = {candidate.content_sha256 for candidate in candidates}
        processor_deterministic = len(set(batch_hashes)) == 1
        real_source_available = (
            processor_deterministic and fixed_reproducible and len(unique_hashes) >= 2
        )
        policy_signature = str(inspect.signature(stack.policy.predict_action_chunk))
        action_head_signature = str(
            inspect.signature(stack.policy.model.action_model.predict_action)
        )
        payload = {
            "schema_version": "latentguard.lg_r2b0.candidate_source_manifest.v1",
            "status": "pass" if real_source_available else "fail",
            "stop_reason": (
                None
                if real_source_available
                else "REAL_MULTI_CANDIDATE_SOURCE_NOT_AVAILABLE"
            ),
            "runtime_identity": runtime_identity(),
            "config_sha256": sha256_path(config_path),
            "policy_identity": frozen_policy_identity(),
            "source_priority": "A",
            "source_description": (
                "same frozen checkpoint and observation; policy reset before each "
                "native flow sample; only global PyTorch inference seed changes"
            ),
            "sampling_api": {
                "policy_signature": policy_signature,
                "action_head_signature": action_head_signature,
                "native_flow_matching": True,
                "initial_noise_source": (
                    "torch.randn in VLAJEPAActionHead.predict_action"
                ),
                "explicit_generator_argument": False,
                "explicit_inference_seed_argument": False,
                "explicit_noise_argument_exposed_but_effective": False,
                "sampling_steps": int(stack.config.num_inference_timesteps),
                "solver": "explicit_euler",
                "official_sampling_configuration_changes_used": False,
                "shared_observation_encoding_across_candidates": False,
                "batched_independent_seed_generation": False,
            },
            "observation_hash": observation_hash,
            "processor_batch_hashes": batch_hashes,
            "processor_deterministic": processor_deterministic,
            "fixed_seed_reproduction": {
                "passed": fixed_reproducible,
                "content_hashes": [candidate.content_sha256 for candidate in repeated],
            },
            "different_seed_generation": {
                "passed": len(unique_hashes) >= 2,
                "requested": len(candidates),
                "unique": len(unique_hashes),
                "content_hashes": [
                    candidate.content_sha256 for candidate in candidates
                ],
            },
            "candidates": [public_candidate(candidate) for candidate in candidates],
            "policy_generated_candidate_ratio": 1.0,
            "synthetic_candidate_ratio": 0.0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "final_seeds_accessed": False,
        }
    finally:
        vector_env.close()
    path = output_root() / "candidate_source_manifest.json"
    write_json(path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "unique_candidates": len(unique_hashes),
                "output": str(path),
                "stop_reason": payload["stop_reason"],
            },
            sort_keys=True,
        )
    )
    if not real_source_available:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
