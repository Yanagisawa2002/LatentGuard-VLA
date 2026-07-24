"""Train the allowed linear probe on frozen VLA-JEPA representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1_training import (
    bounded_samples,
    load_current_samples,
    peak_cuda_memory,
    save_torch_checkpoint,
    set_training_seed,
)
from _lg_r1c_common import (
    file_identity,
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    runtime_root,
    write_json,
    write_jsonl,
)
from lg_r1_train_representation_probe import (
    _build_feature_cache,
    _load_arrays,
    _sample_identity,
)

from latentguard.rewards.metrics import binary_metrics, progress_metrics


def _evaluate(
    head: Any,
    arrays: dict[str, np.ndarray],
    success: np.ndarray,
    *,
    split: str,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    mask = arrays["split"] == split
    features = torch.from_numpy(arrays["features"][mask]).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        progress, success_probability = head(features)
    target_progress = arrays["progress"][mask].astype(np.float64)
    predicted_progress = progress.detach().to("cpu").numpy().astype(np.float64)
    target_success = success[mask].astype(np.int64)
    predicted_success = (
        success_probability.detach().to("cpu").numpy().astype(np.float64)
    )
    summary = {
        "samples": int(mask.sum()),
        "progress": progress_metrics(
            target_progress.tolist(),
            predicted_progress.tolist(),
        ),
        "success": binary_metrics(
            target_success.tolist(),
            predicted_success.tolist(),
        ).to_dict(),
    }
    sample_ids = arrays["sample_id"][mask].tolist()
    predictions = [
        {
            "schema_version": "latentguard.lg_r1c.vlajepa_probe_prediction.v1",
            "sample_id": str(sample_id),
            "split": split,
            "predicted_progress": float(progress_value),
            "success_probability": float(success_value),
        }
        for sample_id, progress_value, success_value in zip(
            sample_ids,
            predicted_progress,
            predicted_success,
            strict=True,
        )
    ]
    return summary, predictions


def main() -> None:
    """Fit only the declared linear probe and evaluate test once."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/vlajepa_probe.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = runtime_root(args.runtime_root)
    destination = output_root(args.output_root)
    config = read_yaml(resolve_repo_path(args.config))
    if config.get("failure_head") is not False:
        raise ValueError("LG-R1c VLA-JEPA probe must not be a failure head")
    if config.get("backbone_frozen") is not True:
        raise ValueError("VLA-JEPA backbone must remain frozen")
    seed = int(args.seed if args.seed is not None else config["seed"])
    maximum_steps = int(
        args.max_steps if args.max_steps is not None else config["max_steps"]
    )
    limit = int(
        args.limit_samples
        if args.limit_samples is not None
        else config["feature_sample_limit"]
    )
    rows = bounded_samples(load_current_samples(source), limit=limit, seed=seed)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "samples": len(rows),
                    "max_steps": maximum_steps,
                    "backbone_frozen": True,
                    "failure_head": False,
                },
                sort_keys=True,
            )
        )
        return
    set_training_seed(seed)
    run_root = destination / "vlajepa-probe"
    cache_path = run_root / "frozen_qwen_features.npz"
    feature_manifest_path = run_root / "feature_manifest.json"
    identity = _sample_identity(rows)
    if not cache_path.is_file():
        feature_manifest = {
            "sample_identity": identity,
            **_build_feature_cache(
                rows,
                runtime=source,
                cache_path=cache_path,
                batch_size=min(int(config["batch_size"]), 16),
            ),
        }
        write_json(feature_manifest_path, feature_manifest)
    else:
        feature_manifest = read_json(feature_manifest_path)
        if feature_manifest["sample_identity"] != identity:
            raise ValueError("VLA-JEPA frozen feature sample identity drift")
    arrays = _load_arrays(cache_path)
    success_by_sample = {
        f"{row['episode_id']}:{row['frame_index']}": bool(row["episode_success"])
        for row in rows
    }
    success = np.asarray(
        [success_by_sample[str(sample)] for sample in arrays["sample_id"]],
        dtype=np.float32,
    )
    import torch
    from torch import nn
    from torch.nn import functional

    class LinearProbe(nn.Module):
        def __init__(self, feature_dim: int) -> None:
            super().__init__()
            self.progress = nn.Linear(feature_dim, 1)
            self.success = nn.Linear(feature_dim, 1)

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.sigmoid(self.progress(value)).squeeze(-1),
                torch.sigmoid(self.success(value)).squeeze(-1),
            )

    device = torch.device("cuda")
    head = LinearProbe(int(arrays["features"].shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = run_root / "last.pt"
    best_checkpoint = run_root / "best.pt"
    step = 0
    best_selection = float("inf")
    patience = 0
    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        head.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        best_selection = float(state["best_selection"])
        patience = int(state["patience"])
    train_indices = np.flatnonzero(arrays["split"] == "train")
    generator = np.random.default_rng(seed + step)
    while step < maximum_steps:
        selected = generator.choice(
            train_indices,
            size=min(int(config["batch_size"]), len(train_indices)),
            replace=len(train_indices) < int(config["batch_size"]),
        )
        features = torch.from_numpy(arrays["features"][selected]).to(
            device=device,
            dtype=torch.float32,
        )
        target_progress = torch.from_numpy(arrays["progress"][selected]).to(
            device=device,
            dtype=torch.float32,
        )
        target_success = torch.from_numpy(success[selected]).to(
            device=device,
            dtype=torch.float32,
        )
        predicted_progress, predicted_success = head(features)
        loss = float(config["progress_loss_weight"]) * functional.mse_loss(
            predicted_progress,
            target_progress,
        ) + float(config["success_loss_weight"]) * functional.binary_cross_entropy(
            predicted_success,
            target_success,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite VLA-JEPA probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        if step % int(config["eval_every_steps"]) == 0 or step == maximum_steps:
            validation, _ = _evaluate(
                head,
                arrays,
                success,
                split="validation",
                device=device,
            )
            selection = (
                float(validation["progress"]["mae"])
                + float(validation["success"]["brier"])
            ) / 2.0
            if selection < best_selection:
                best_selection = selection
                patience = 0
                save_torch_checkpoint(
                    best_checkpoint,
                    {
                        "model": head.state_dict(),
                        "step": step,
                        "validation": validation,
                    },
                )
            else:
                patience += 1
            save_torch_checkpoint(
                checkpoint,
                {
                    "model": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_selection": best_selection,
                    "patience": patience,
                },
            )
            if patience >= int(config["early_stopping_patience"]):
                break
    if not best_checkpoint.is_file():
        raise RuntimeError("no validation-selected VLA-JEPA probe checkpoint")
    selected_state = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    head.load_state_dict(selected_state["model"])
    validation, validation_predictions = _evaluate(
        head,
        arrays,
        success,
        split="validation",
        device=device,
    )
    test, test_predictions = _evaluate(
        head,
        arrays,
        success,
        split="test",
        device=device,
    )
    prediction_path = run_root / "predictions.jsonl"
    write_jsonl(
        prediction_path,
        [*validation_predictions, *test_predictions],
    )
    result = {
        "schema_version": "latentguard.lg_r1c.vlajepa_probe_results.v1",
        "status": "pass",
        "claim": "frozen-representation linear probe, not a failure head",
        "input_tensor": "VLAJEPAModel._encode_qwen embodied_action_tokens",
        "pooling": "token mean",
        "input_shape": list(arrays["features"].shape),
        "head": "two scalar linear projections",
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "optimizer_parameters": "linear probe only",
        "optimizer_steps": step,
        "best_validation_step": int(selected_state["step"]),
        "selection_split": "validation",
        "test_evaluations": 1,
        "vlajepa_frozen": True,
        "foundation_model_backward_calls": 0,
        "failure_head_trained": False,
        "validation": validation,
        "test": test,
        "feature_manifest": feature_manifest,
        "prediction_file": file_identity(
            prediction_path,
            locator="vlajepa-probe/predictions.jsonl",
        ),
        "cuda_memory": peak_cuda_memory(),
    }
    write_json(destination / "vlajepa_probe_results.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
