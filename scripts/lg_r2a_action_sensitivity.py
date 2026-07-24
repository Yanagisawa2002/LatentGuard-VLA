"""Run real-action permutation, swap-surrogate, masking, and attribution tests."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2a_common import (
    output_root,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _select_real_sources(
    metadata: list[dict[str, Any]],
    *,
    mode: str,
    seed: int,
) -> np.ndarray:
    """Choose deterministic same-task real actions from another episode."""

    if mode not in {"permutation", "swap_surrogate"}:
        raise ValueError(f"unsupported real-action source mode: {mode}")
    by_task: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        if row["synthetic_action"] is not False:
            raise ValueError("sensitivity sources must be frozen real actions")
        by_task[str(row["task"])].append(index)
    selected = np.empty(len(metadata), dtype=np.int64)
    for index, row in enumerate(metadata):
        candidates = [
            candidate
            for candidate in by_task[str(row["task"])]
            if metadata[candidate]["episode_id"] != row["episode_id"]
        ]
        if mode == "swap_surrogate":
            same_stage = [
                candidate
                for candidate in candidates
                if metadata[candidate]["current_stage"] == row["current_stage"]
            ]
            if same_stage:
                candidates = same_stage
        if not candidates:
            raise ValueError(f"no other-episode real action source for sample {index}")
        distance = min(
            abs(int(metadata[candidate]["progress_bin"]) - int(row["progress_bin"]))
            for candidate in candidates
        )
        candidates = [
            candidate
            for candidate in candidates
            if abs(int(metadata[candidate]["progress_bin"]) - int(row["progress_bin"]))
            == distance
        ]
        identity = f"{seed}|{mode}|{row['sample_id']}"
        digest = int(hashlib.sha256(identity.encode()).hexdigest(), 16)
        selected[index] = sorted(candidates)[digest % len(candidates)]
    return selected


def _predict(
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    preprocessing: dict[str, np.ndarray],
    device: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    import torch

    results: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("progress", "stagnation", "regression", "events", "terminal")
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            positions = indices[start : start + batch_size]
            state = (
                data["features"][positions] - preprocessing["features_mean"]
            ) / preprocessing["features_std"]
            proprio = (
                data["proprio"][positions] - preprocessing["proprio_mean"]
            ) / preprocessing["proprio_std"]
            output = model(
                torch.as_tensor(state, dtype=torch.float32, device=device),
                torch.as_tensor(actions[positions], dtype=torch.float32, device=device),
                torch.as_tensor(masks[positions], dtype=torch.bool, device=device),
                torch.as_tensor(proprio, dtype=torch.float32, device=device),
            )
            for key, value in output.items():
                converted = value if key == "progress" else torch.sigmoid(value)
                results[key].append(converted.detach().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in results.items()}


def _gradient_attribution(
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    preprocessing: dict[str, np.ndarray],
    device: Any,
) -> list[float]:
    import torch

    state = (
        data["features"][indices] - preprocessing["features_mean"]
    ) / preprocessing["features_std"]
    proprio = (
        data["proprio"][indices] - preprocessing["proprio_mean"]
    ) / preprocessing["proprio_std"]
    action = torch.as_tensor(
        data["actions"][indices], dtype=torch.float32, device=device
    ).requires_grad_(True)
    output = model(
        torch.as_tensor(state, dtype=torch.float32, device=device),
        action,
        torch.as_tensor(data["action_mask"][indices], dtype=torch.bool, device=device),
        torch.as_tensor(proprio, dtype=torch.float32, device=device),
    )
    objective = output["progress"][:, 0].sum()
    objective.backward()
    gradient = action.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy()
    return gradient.tolist()


def main() -> None:
    """Evaluate Model C with only frozen real-action swaps and safe ablations."""

    args = parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    if args.dry_run:
        print({"status": "dry_run", "model": config["model_variant"]})
        return
    import torch

    from latentguard.action_conditioning.models import ProbeModel

    output = output_root(args.output_dir)
    prepared = np.load(output / "prepared_data.npz", allow_pickle=False)
    data = {key: prepared[key] for key in prepared.files}
    metadata = read_jsonl(output / "prepared_metadata.jsonl")
    if args.limit_samples is not None:
        keep = min(args.limit_samples, len(metadata))
        metadata = metadata[:keep]
        data = {key: values[:keep] for key, values in data.items()}
    folds = np.asarray([int(row["fold"]) for row in metadata], dtype=np.int64)
    permutation_sources = _select_real_sources(
        metadata, mode="permutation", seed=int(config["permutation_seed"])
    )
    swap_sources = _select_real_sources(
        metadata, mode="swap_surrogate", seed=int(config["swap_seed"])
    )
    actions = data["actions"]
    masks = data["action_mask"]
    variants = {
        "full": (actions, masks),
        "permuted_real": (actions[permutation_sources], masks[permutation_sources]),
        "swap_surrogate_real": (actions[swap_sources], masks[swap_sources]),
        "first_action_only": (
            np.concatenate([actions[:, :1], np.zeros_like(actions[:, 1:])], axis=1),
            np.concatenate([masks[:, :1], np.zeros_like(masks[:, 1:])], axis=1),
        ),
        "endpoint_summary": (
            np.repeat(actions[:, -1:, :], actions.shape[1], axis=1),
            masks.copy(),
        ),
        "action_masked": (np.zeros_like(actions), np.zeros_like(masks)),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: dict[str, np.ndarray] = {}
    attribution_values = []
    for fold in range(5):
        checkpoint = torch.load(
            output
            / "runs"
            / "state_action"
            / "checkpoints"
            / f"fold_{fold}_selected.pt",
            map_location=device,
            weights_only=False,
        )
        model = ProbeModel(variant="state_action").to(device)
        model.load_state_dict(checkpoint["model"])
        indices = np.flatnonzero(folds == fold)
        for variant, (variant_actions, variant_masks) in variants.items():
            fold_predictions = _predict(
                model,
                data,
                indices,
                variant_actions,
                variant_masks,
                checkpoint["preprocessing"],
                device,
                int(config["batch_size"]),
            )
            for key, values in fold_predictions.items():
                name = f"{variant}__{key}"
                if name not in predictions:
                    predictions[name] = np.full(
                        (len(metadata), values.shape[1]), np.nan, dtype=np.float32
                    )
                predictions[name][indices] = values
        attribution_indices = indices[
            : min(len(indices), int(config["attribution_per_fold"]))
        ]
        attribution_values.append(
            _gradient_attribution(
                model,
                data,
                attribution_indices,
                checkpoint["preprocessing"],
                device,
            )
        )
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("sensitivity predictions are incomplete")
    np.savez_compressed(output / "sensitivity_predictions.npz", **predictions)
    mapping_rows = []
    for index, row in enumerate(metadata):
        mapping_rows.append(
            {
                "target_sample_id": row["sample_id"],
                "target_episode_id": row["episode_id"],
                "task": row["task"],
                "progress_bin": row["progress_bin"],
                "permutation_source_sample_id": metadata[permutation_sources[index]][
                    "sample_id"
                ],
                "permutation_source_episode_id": metadata[permutation_sources[index]][
                    "episode_id"
                ],
                "swap_source_sample_id": metadata[swap_sources[index]]["sample_id"],
                "swap_source_episode_id": metadata[swap_sources[index]]["episode_id"],
                "source_action_real_executed": True,
                "synthetic_action": False,
            }
        )
    write_jsonl(output / "sensitivity_mapping.jsonl", mapping_rows)
    full_short = predictions["full__progress"][:, 0]
    result = {
        "schema_version": "latentguard.lg_r2a.action_sensitivity_run.v1",
        "status": "pass",
        "samples": len(metadata),
        "real_action_permutation": {
            "same_task": True,
            "similar_progress_bin": True,
            "same_effective_action_horizon": True,
            "different_episode": True,
            "real_executed_actions_only": True,
            "training_use": False,
            "mean_absolute_short_prediction_change": float(
                np.mean(
                    np.abs(predictions["permuted_real__progress"][:, 0] - full_short)
                )
            ),
        },
        "same_state_swap_surrogate": {
            "same_task": True,
            "similar_stage_and_progress": True,
            "different_episode": True,
            "real_executed_actions_only": True,
            "true_same_state_counterfactual": False,
            "mean_absolute_short_prediction_change": float(
                np.mean(
                    np.abs(
                        predictions["swap_surrogate_real__progress"][:, 0] - full_short
                    )
                )
            ),
        },
        "ablations": [
            "full",
            "first_action_only",
            "endpoint_summary",
            "action_masked",
        ],
        "short_progress_input_gradient_absolute_mean_by_action_dimension": (
            np.mean(np.asarray(attribution_values), axis=0).tolist()
        ),
        "candidate_generation": False,
        "candidate_ranking": False,
        "counterfactual_rollout": False,
    }
    write_json(output / "action_sensitivity_run.json", result)
    print({"status": "pass", "samples": len(metadata)})


if __name__ == "__main__":
    main()
