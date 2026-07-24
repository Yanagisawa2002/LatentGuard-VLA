"""Build the typed LG-R2 feature and candidate-time availability contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1c_common import (
    read_json,
    read_yaml,
    resolve_repo_path,
    write_json,
)

from latentguard.rewards.schemas import (
    CandidateTimeAvailability,
    FeatureSetContract,
    FeatureSpec,
    validate_feature_contract,
)


def _runtime_measurement(path: Path) -> tuple[float | None, int | None]:
    if not path.is_file():
        return None, None
    payload = read_json(path)
    latency = payload.get("mean_latency_ms")
    cuda = payload.get("cuda_memory")
    peak = cuda.get("peak_allocated_bytes") if isinstance(cuda, dict) else None
    return (
        float(latency) if latency is not None else None,
        int(peak) if peak is not None else None,
    )


def _specifications(
    *,
    robometer_latency: float | None,
    robometer_memory: int | None,
    topreward_latency: float | None,
    topreward_memory: int | None,
) -> dict[str, FeatureSpec]:
    return {
        "current_vla_representation": FeatureSpec(
            tensor_name="current_vla_representation",
            shape=("B", 2048),
            dtype="float32",
            pooling="mean_over_current_qwen_tokens",
            time_horizon="current_observation",
            online_computable=True,
            measured_latency_ms=None,
            measured_peak_gpu_bytes=None,
            depends_on_future_observation=False,
            candidate_time_availability=(CandidateTimeAvailability.CURRENT_STATE_ONLY),
            candidate_time_usable=True,
        ),
        "numeric_action_chunk": FeatureSpec(
            tensor_name="numeric_action_chunk",
            shape=("B", "H_action", "D_action"),
            dtype="float32",
            pooling="none",
            time_horizon="candidate_action_horizon",
            online_computable=True,
            measured_latency_ms=None,
            measured_peak_gpu_bytes=None,
            depends_on_future_observation=False,
            candidate_time_availability=(
                CandidateTimeAvailability.CANDIDATE_ACTION_CONDITIONED
            ),
            candidate_time_usable=True,
        ),
        "robometer_progress": FeatureSpec(
            tensor_name="robometer_progress",
            shape=("B",),
            dtype="float32",
            pooling="last_frame_from_eight_frame_window",
            time_horizon="executed_video_window",
            online_computable=False,
            measured_latency_ms=robometer_latency,
            measured_peak_gpu_bytes=robometer_memory,
            depends_on_future_observation=True,
            candidate_time_availability=(
                CandidateTimeAvailability.EXECUTED_TRAJECTORY_REQUIRED
            ),
            candidate_time_usable=False,
        ),
        "robometer_success_probability": FeatureSpec(
            tensor_name="robometer_success_probability",
            shape=("B",),
            dtype="float32",
            pooling="last_frame_from_eight_frame_window",
            time_horizon="executed_video_window",
            online_computable=False,
            measured_latency_ms=robometer_latency,
            measured_peak_gpu_bytes=robometer_memory,
            depends_on_future_observation=True,
            candidate_time_availability=(
                CandidateTimeAvailability.EXECUTED_TRAJECTORY_REQUIRED
            ),
            candidate_time_usable=False,
        ),
        "topreward_score": FeatureSpec(
            tensor_name="topreward_score",
            shape=("B",),
            dtype="float32",
            pooling="terminal_true_token_log_probability",
            time_horizon="executed_video_window",
            online_computable=False,
            measured_latency_ms=topreward_latency,
            measured_peak_gpu_bytes=topreward_memory,
            depends_on_future_observation=True,
            candidate_time_availability=(
                CandidateTimeAvailability.EXECUTED_TRAJECTORY_REQUIRED
            ),
            candidate_time_usable=False,
        ),
        "predicted_temporal_tokens": FeatureSpec(
            tensor_name="predicted_temporal_tokens",
            shape=("B", "T_pred", "N_patch", "D_jepa"),
            dtype="float32",
            pooling="none",
            time_horizon="predicted_latent_horizon",
            online_computable=True,
            measured_latency_ms=None,
            measured_peak_gpu_bytes=None,
            depends_on_future_observation=False,
            candidate_time_availability=(
                CandidateTimeAvailability.NOT_USABLE_FOR_PRE_EXECUTION_SELECTION
            ),
            candidate_time_usable=False,
        ),
        "qwen_action_token_hidden_states": FeatureSpec(
            tensor_name="qwen_action_token_hidden_states",
            shape=("B", "T_action_token", 2048),
            dtype="float32",
            pooling="none",
            time_horizon="policy_action_token_context",
            online_computable=True,
            measured_latency_ms=None,
            measured_peak_gpu_bytes=None,
            depends_on_future_observation=False,
            candidate_time_availability=(
                CandidateTimeAvailability.NOT_USABLE_FOR_PRE_EXECUTION_SELECTION
            ),
            candidate_time_usable=False,
        ),
        "task_agnostic_reward_outputs": FeatureSpec(
            tensor_name="task_agnostic_reward_outputs",
            shape=("B", 3),
            dtype="float32",
            pooling="robometer_progress_success_and_topreward_score",
            time_horizon="executed_video_window",
            online_computable=False,
            measured_latency_ms=(
                robometer_latency + topreward_latency
                if robometer_latency is not None and topreward_latency is not None
                else None
            ),
            measured_peak_gpu_bytes=max(
                value
                for value in (robometer_memory, topreward_memory)
                if value is not None
            )
            if any(value is not None for value in (robometer_memory, topreward_memory))
            else None,
            depends_on_future_observation=True,
            candidate_time_availability=(
                CandidateTimeAvailability.EXECUTED_TRAJECTORY_REQUIRED
            ),
            candidate_time_usable=False,
        ),
    }


def main() -> None:
    """Write feature sets A-D with fail-closed candidate-time semantics."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/feature_contract.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/lg_r1c/lg_r2_feature_contract.json"),
    )
    args = parser.parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    runtime = (
        args.runtime_root
        if args.runtime_root is not None
        else resolve_repo_path(Path("outputs/lg_r1c/local"))
    )
    rbm_latency, rbm_memory = _runtime_measurement(runtime / "robometer_inference.json")
    top_latency, top_memory = _runtime_measurement(runtime / "topreward_inference.json")
    specifications = _specifications(
        robometer_latency=rbm_latency,
        robometer_memory=rbm_memory,
        topreward_latency=top_latency,
        topreward_memory=top_memory,
    )
    sets = []
    feature_sets = config.get("feature_sets")
    if not isinstance(feature_sets, dict):
        raise ValueError("feature_sets config must be a mapping")
    for name in ("A", "B", "C", "D"):
        entry = feature_sets[name]
        if not isinstance(entry, dict):
            raise ValueError(f"feature set {name} must be a mapping")
        features = tuple(specifications[str(feature)] for feature in entry["features"])
        sets.append(
            FeatureSetContract(
                name=name,
                role=str(entry["role"]),
                features=features,
                lg_r2_training_authorized=False,
            ).to_dict()
        )
    payload: dict[str, Any] = {
        "schema_version": "latentguard.lg_r1c.lg_r2_features.v1",
        "status": "pass",
        "feature_sets": sets,
        "candidate_time_conclusion": (
            "ROBOMETER and TOPReward score executed observation windows and "
            "cannot directly score an unexecuted numeric action candidate."
        ),
        "reward_roles": [
            "post_hoc_evaluator",
            "auxiliary_target",
            "calibration_baseline",
        ],
        "reward_role_online_selector": False,
        "lg_r2_training_authorized": False,
    }
    validate_feature_contract(payload)
    write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
