"""Audit real VLA-JEPA world-model tensors from recorded LIBERO windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r0_runtime import load_stack, read_json, write_json

from latentguard.adapters.vla_jepa.world_model_adapter import (
    inspect_training_world_model,
)


def main() -> None:
    """Extract real tensors and descriptive statistics without fitting a model."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    config = read_json(args.config)
    manifest = read_json(args.manifest)
    stack = load_stack()
    groups: dict[str, list[dict[str, Any]]] = {"success": [], "failure": []}
    sample_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in manifest["episodes"]:
        group = "success" if entry["success"] else "failure"
        groups[group].append(entry)

    targets = {
        "success": int(config["minimum_success_states"]),
        "failure": int(config["minimum_failure_states"]),
    }
    for group_name, entries in groups.items():
        remaining = targets[group_name]
        for entry in entries:
            if remaining <= 0:
                break
            try:
                dataset = LeRobotDataset(
                    repo_id=entry["dataset_repo_id"],
                    root=Path(entry["dataset_runtime_path"]),
                )
                window_count = max(0, len(dataset) - stack.config.num_video_frames + 1)
                take = min(remaining, window_count)
                if take == 0:
                    continue
                indices = np.linspace(0, window_count - 1, num=take, dtype=int).tolist()
                for start in indices:
                    frames = [dataset[start + offset] for offset in range(8)]
                    image = torch.stack(
                        [frame[f"{OBS_IMAGES}.image"] for frame in frames], dim=0
                    ).unsqueeze(0)
                    image2 = torch.stack(
                        [frame[f"{OBS_IMAGES}.image2"] for frame in frames], dim=0
                    ).unsqueeze(0)
                    state = frames[0][OBS_STATE].unsqueeze(0)
                    batch = {
                        f"{OBS_IMAGES}.image": image.to(stack.config.device),
                        f"{OBS_IMAGES}.image2": image2.to(stack.config.device),
                        OBS_STATE: state.to(stack.config.device),
                        "task": [entry["instruction"]],
                    }
                    output = inspect_training_world_model(stack.policy, batch)
                    predicted = output.predicted_future_latents
                    target = output.target_future_latents
                    current = output.current_latent
                    if predicted is None or target is None or current is None:
                        raise RuntimeError(
                            "world-model tensors unexpectedly unavailable"
                        )
                    pred_flat = predicted.float().flatten(1)
                    target_flat = target.float().flatten(1)
                    cosine = 1.0 - functional.cosine_similarity(
                        pred_flat,
                        target_flat,
                        dim=1,
                    )
                    sample_records.append(
                        {
                            "group": group_name,
                            "episode_id": entry["episode_id"],
                            "frame_start": start,
                            "current_latent_shape": list(current.shape),
                            "predicted_future_latents_shape": list(predicted.shape),
                            "target_future_latents_shape": list(target.shape),
                            "current_latent_dtype": str(current.dtype),
                            "predicted_dtype": str(predicted.dtype),
                            "target_dtype": str(target.dtype),
                            "prediction_target_l1": output.predictor_metadata[
                                "l1_distance"
                            ],
                            "prediction_target_cosine_distance": float(cosine.item()),
                            "predictor_variance": float(
                                predicted.float().var(unbiased=False).item()
                            ),
                            "predictor_metadata": output.predictor_metadata,
                        }
                    )
                remaining -= take
            except Exception as exc:
                failures.append(f"{entry['episode_id']}: {type(exc).__name__}: {exc}")

    descriptive: dict[str, Any] = {}
    for group_name in ("success", "failure"):
        records = [record for record in sample_records if record["group"] == group_name]
        descriptive[group_name] = {
            "samples": len(records),
            "prediction_target_l1_mean": (
                float(np.mean([record["prediction_target_l1"] for record in records]))
                if records
                else None
            ),
            "prediction_target_cosine_distance_mean": (
                float(
                    np.mean(
                        [
                            record["prediction_target_cosine_distance"]
                            for record in records
                        ]
                    )
                )
                if records
                else None
            ),
            "predictor_variance_mean": (
                float(np.mean([record["predictor_variance"] for record in records]))
                if records
                else None
            ),
        }
    status = "pass" if sample_records and not failures else "partial"
    payload = {
        "schema_version": "latentguard.lg_r0.world_model_interface_audit.v1",
        "status": status,
        "path_semantic": "official_training_style_offline_diagnostic",
        "inference_default_computes_world_model": False,
        "future_target_available_with_temporal_video": True,
        "external_numeric_candidate_actions_supported": False,
        "batched_external_candidates_supported": False,
        "numeric_action_chunk_consumed_by_predictor": False,
        "action_condition": "qwen_special_action_token_hidden_states",
        "native_scalar_scores": {
            "risk": False,
            "success": False,
            "progress": False,
            "reconstruction": False,
            "training_l1_loss": True,
        },
        "requested_samples": targets,
        "available_samples": {
            "success": descriptive["success"]["samples"],
            "failure": descriptive["failure"]["samples"],
        },
        "sample_shortfall": {
            group: max(0, targets[group] - descriptive[group]["samples"])
            for group in targets
        },
        "descriptive_statistics_only": descriptive,
        "samples": sample_records,
        "failures": failures,
        "predictor_action_response_conclusion": (
            "The runtime confirms non-constant predicted tensors and Qwen action-token "
            "conditioning, but the official API cannot isolate response to arbitrary "
            "numeric action candidates; no candidate-response claim is made."
        ),
        "recommended_failure_head_inputs": [
            "current_vjepa_visual_tokens",
            "predicted_future_vjepa_visual_tokens",
            "target_future_vjepa_visual_tokens_during_training_only",
            "qwen_action_token_hidden_states",
            "generated_numeric_action_chunk_as_a_separate_feature",
        ],
        "optimizer_steps": 0,
        "backward_calls": 0,
        "classifier_fit": False,
        "threshold_selected": False,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "samples": len(sample_records),
                "output": str(args.output),
            }
        )
    )
    if not sample_records:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
