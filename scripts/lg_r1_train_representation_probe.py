"""Train a linear probe on frozen current VLA-JEPA/Qwen representations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r0_runtime import load_processors, load_stack
from _lg_r1_common import (
    output_root,
    read_json,
    resolve_repo_path,
    write_json,
)
from _lg_r1_training import (
    EpisodeDatasetCache,
    add_training_arguments,
    bounded_samples,
    load_current_samples,
    peak_cuda_memory,
    resolved_training_config,
    save_torch_checkpoint,
    set_training_seed,
    tensor_digest,
    write_run_manifest,
)

from latentguard.progress.metrics import (
    evaluate_progress,
    evaluate_stage_predictions,
)


def _sample_identity(rows: list[dict[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['episode_id']}:{row['frame_index']}\n".encode())
    return digest.hexdigest()


def _build_feature_cache(
    rows: list[dict[str, Any]],
    *,
    runtime: Path,
    cache_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    stack = load_stack()
    preprocessor, _ = load_processors(stack)
    stack.policy.requires_grad_(False)
    stack.policy.eval()
    before = tensor_digest(stack.policy)
    datasets = EpisodeDatasetCache(runtime)
    features: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        frames = [
            datasets.frame(str(row["episode_id"]), int(row["frame_index"]))
            for row in batch_rows
        ]
        policy_batch = {
            "observation.images.image": torch.stack(
                [frame["observation.images.image"] for frame in frames]
            ),
            "observation.images.image2": torch.stack(
                [frame["observation.images.image2"] for frame in frames]
            ),
            "observation.state": torch.stack(
                [frame["observation.state"] for frame in frames]
            ),
            "task": [str(row["instruction"]) for row in batch_rows],
        }
        processed = preprocessor(policy_batch)
        with torch.inference_mode():
            model_inputs = stack.policy._prepare_model_inputs(processed, training=False)
            tokens, _ = stack.policy.model._encode_qwen(
                model_inputs["images"],
                model_inputs["instructions"],
                need_action_tokens=False,
            )
            pooled = tokens.float().mean(dim=1)
        features.append(pooled.to("cpu").numpy())
    after = tensor_digest(stack.policy)
    if before != after:
        raise RuntimeError("VLA-JEPA backbone changed during feature extraction")
    arrays = {
        "features": np.concatenate(features, axis=0),
        "stage": np.asarray([int(row["stage_id"]) for row in rows], dtype=np.int64),
        "progress": np.asarray(
            [float(row["overall_progress"]) for row in rows],
            dtype=np.float32,
        ),
        "split": np.asarray([str(row["split"]) for row in rows]),
        "sample_id": np.asarray(
            [f"{row['episode_id']}:{row['frame_index']}" for row in rows]
        ),
    }
    temporary = cache_path.with_name(cache_path.name + ".partial.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, cache_path)
    return {
        "feature_shape": list(arrays["features"].shape),
        "pooling": "mean over embodied-action Qwen tokens",
        "input_tensor": "VLAJEPAModel._encode_qwen embodied_action_tokens",
        "backbone_digest_before": before,
        "backbone_digest_after": after,
        "backbone_frozen": True,
        "optimizer_contains_backbone": False,
    }


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _evaluate(
    head: Any,
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    mask = arrays["split"] == split
    features = torch.from_numpy(arrays["features"][mask]).to(
        device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        logits, progress = head(features)
    targets_stage = arrays["stage"][mask].tolist()
    predictions_stage = logits.argmax(dim=-1).to("cpu").tolist()
    targets_progress = arrays["progress"][mask].tolist()
    predictions_progress = progress.to("cpu").tolist()
    stage = evaluate_stage_predictions(targets_stage, predictions_stage, stage_count=8)
    progress_metrics = evaluate_progress(targets_progress, predictions_progress)
    return {
        "samples": int(mask.sum()),
        "stage": {
            "accuracy": stage.accuracy,
            "macro_f1": stage.macro_f1,
            "per_stage_f1": stage.per_stage_f1,
            "confusion_matrix": stage.confusion_matrix,
        },
        "progress": {
            "mae": progress_metrics.mae,
            "rmse": progress_metrics.rmse,
            "spearman": progress_metrics.spearman,
        },
    }


def main() -> None:
    """Train only the declared lightweight representation probe."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/representation_probe.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--result", type=Path)
    add_training_arguments(parser)
    args = parser.parse_args()
    config, run_dir = resolved_training_config(
        args.config, args, run_name="representation-probe"
    )
    if args.dry_run:
        write_run_manifest(
            run_dir,
            run_type="representation_probe",
            config=config,
            status="dry_run",
            optimizer_steps=0,
            checkpoint_path=None,
            extra={"training_started": False},
        )
        print(json.dumps({"status": "dry_run", "config": config}))
        return
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    stage_report = json.loads(
        (runtime / "stage_annotation_report.json").read_text(encoding="utf-8")
    )
    if stage_report.get("status") != "pass":
        raise ValueError("stage QA gate must pass before probe training")
    seed = int(config["seed"])
    set_training_seed(seed)
    rows = bounded_samples(
        load_current_samples(runtime),
        limit=(
            args.limit_samples
            if args.limit_samples is not None
            else int(config["feature_sample_limit"])
        ),
        seed=seed,
    )
    cache_path = run_dir / "frozen_qwen_features.npz"
    feature_manifest = {
        "sample_identity": _sample_identity(rows),
    }
    feature_manifest_path = run_dir / "feature_manifest.json"
    if not cache_path.is_file():
        feature_manifest.update(
            _build_feature_cache(
                rows,
                runtime=runtime,
                cache_path=cache_path,
                batch_size=min(int(config["batch_size"]), 16),
            )
        )
        write_json(feature_manifest_path, feature_manifest)
    else:
        feature_manifest = read_json(feature_manifest_path)
        if feature_manifest["sample_identity"] != _sample_identity(rows):
            raise ValueError("frozen Qwen feature sample identity drift")
    arrays = _load_arrays(cache_path)
    if len(arrays["features"]) != len(rows):
        raise ValueError("frozen feature cache length mismatch")
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class LinearProbe(nn.Module):
        def __init__(self, feature_dim: int) -> None:
            super().__init__()
            self.stage = nn.Linear(feature_dim, 8)
            self.progress = nn.Linear(feature_dim, 1)

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.stage(value), torch.sigmoid(self.progress(value)).squeeze(-1)

    device = torch.device("cuda")
    head = LinearProbe(int(arrays["features"].shape[1])).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = (
        resolve_repo_path(args.checkpoint)
        if args.checkpoint is not None
        else run_dir / "last.pt"
    )
    best_checkpoint = run_dir / "best.pt"
    step = 0
    best_mae = float("inf")
    patience = 0
    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint missing: {checkpoint}")
        resumed = torch.load(checkpoint, map_location=device, weights_only=False)
        head.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        step = int(resumed["step"])
        best_mae = float(resumed["best_mae"])
        patience = int(resumed["patience"])
    train_indices = np.flatnonzero(arrays["split"] == "train")
    generator = np.random.default_rng(seed + step)
    interrupted = False
    try:
        while step < int(config["max_steps"]):
            selected = generator.choice(
                train_indices,
                size=min(int(config["batch_size"]), len(train_indices)),
                replace=len(train_indices) < int(config["batch_size"]),
            )
            features = torch.from_numpy(arrays["features"][selected]).to(
                device=device, dtype=torch.float32
            )
            target_stage = torch.from_numpy(arrays["stage"][selected]).to(
                device=device, dtype=torch.long
            )
            target_progress = torch.from_numpy(arrays["progress"][selected]).to(
                device=device, dtype=torch.float32
            )
            logits, progress = head(features)
            loss = float(config["stage_loss_weight"]) * functional.cross_entropy(
                logits, target_stage
            ) + float(config["progress_loss_weight"]) * functional.mse_loss(
                progress, target_progress
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite representation probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % int(config["eval_every_steps"]) == 0 or step == int(
                config["max_steps"]
            ):
                validation = _evaluate(head, arrays, split="validation", device=device)
                validation_mae = float(validation["progress"]["mae"])
                improved = validation_mae < best_mae
                if improved:
                    best_mae = validation_mae
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
                        "best_mae": best_mae,
                        "patience": patience,
                    },
                )
                if patience >= int(config["early_stopping_patience"]):
                    break
    except KeyboardInterrupt:
        interrupted = True
        save_torch_checkpoint(
            checkpoint,
            {
                "model": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_mae": best_mae,
                "patience": patience,
            },
        )
    if interrupted:
        write_run_manifest(
            run_dir,
            run_type="representation_probe",
            config=config,
            status="interrupted",
            optimizer_steps=step,
            checkpoint_path=checkpoint,
            extra=feature_manifest,
        )
        return
    if not best_checkpoint.is_file():
        raise RuntimeError("no validation-selected probe checkpoint")
    selected_state = torch.load(
        best_checkpoint, map_location=device, weights_only=False
    )
    head.load_state_dict(selected_state["model"])
    reloaded = LinearProbe(int(arrays["features"].shape[1])).to(device)
    reloaded.load_state_dict(selected_state["model"])
    selected_validation = _evaluate(head, arrays, split="validation", device=device)
    reloaded_validation = _evaluate(reloaded, arrays, split="validation", device=device)
    if reloaded_validation != selected_validation:
        raise RuntimeError("probe checkpoint reload changed validation output")
    results = {
        "schema_version": ("latentguard.lg_r1.representation_probe_results.v1"),
        "status": "pass",
        "claim": "frozen-representation linear probe, not a failure head",
        "input_tensor": ("VLAJEPAModel._encode_qwen embodied_action_tokens"),
        "input_shape": list(arrays["features"].shape),
        "pooling": "token mean",
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "optimizer_parameters": "linear probe only",
        "vlajepa_frozen": True,
        "optimizer_steps": step,
        "best_validation_step": int(selected_state["step"]),
        "validation": selected_validation,
        "test": _evaluate(head, arrays, split="test", device=device),
        "checkpoint_resume_validation": "pass",
        "feature_manifest": feature_manifest,
        "cuda_memory": peak_cuda_memory(),
    }
    result_path = (
        resolve_repo_path(args.result)
        if args.result is not None
        else runtime / "representation_probe_results.json"
    )
    write_json(result_path, results)
    write_json(run_dir / "results.json", results)
    write_run_manifest(
        run_dir,
        run_type="representation_probe",
        config=config,
        status="pass",
        optimizer_steps=step,
        checkpoint_path=best_checkpoint,
        extra=feature_manifest,
    )
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
