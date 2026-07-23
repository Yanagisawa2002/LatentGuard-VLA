"""Train the bounded SARM-style baseline on frozen CLIP features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    sha256_path,
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
    pairwise_accuracy,
)
from latentguard.progress.serialization import write_jsonl_atomic


def _stage_counts(config: dict[str, Any]) -> dict[tuple[str, int], int]:
    registry = read_yaml(resolve_repo_path(config["task_registry"]))
    counts: dict[tuple[str, int], int] = {}
    for task in registry["tasks"]:
        counts[(str(task["suite"]), int(task["task_id"]))] = (
            5 if task["adapter"] == "toggle" else 8
        )
    return counts


def _sample_identity(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['episode_id']}:{row['frame_index']}\n".encode())
    return digest.hexdigest()


def _clip_output_tensor(value: Any) -> Any:
    if hasattr(value, "pooler_output"):
        output = value.pooler_output
        if output is None:
            raise ValueError("CLIP pooler_output is unavailable")
        return output
    return value


def _build_clip_cache(
    rows: list[dict[str, Any]],
    *,
    runtime: Path,
    cache_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda")
    model = CLIPModel.from_pretrained(
        config["clip_model_id"], revision=config["clip_revision"]
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        config["clip_model_id"],
        revision=config["clip_revision"],
        use_fast=True,
    )
    processor_root = cache_path.parent / "clip_processor"
    processor.save_pretrained(processor_root)
    reloaded_processor = CLIPProcessor.from_pretrained(
        processor_root,
        local_files_only=True,
        use_fast=True,
    )
    processor_files = [
        {
            "relative_path": str(path.relative_to(processor_root).as_posix()),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(processor_root.rglob("*"))
        if path.is_file()
    ]
    model.requires_grad_(False)
    model.eval()
    before = tensor_digest(model)
    cache = EpisodeDatasetCache(runtime)
    half = int(config["n_obs_steps"]) // 2
    deltas = [int(config["frame_gap"]) * offset for offset in range(-half, half + 1)]
    video_features: list[np.ndarray] = []
    state_features: list[np.ndarray] = []
    text_features: list[np.ndarray] = []
    clip_batch_size = 64
    for row in rows:
        episode_id = str(row["episode_id"])
        current = int(row["frame_index"])
        frames = [cache.frame(episode_id, current + delta) for delta in deltas]
        images = []
        states = []
        for frame in frames:
            image = frame["observation.images.image"]
            if isinstance(image, torch.Tensor):
                image_array = image.detach().to("cpu").permute(1, 2, 0).numpy()
            else:
                image_array = np.asarray(image)
            images.append(image_array)
            state = np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1)
            padded = np.zeros(32, dtype=np.float32)
            padded[: min(len(state), 32)] = state[:32]
            states.append(padded)
        encoded_images = []
        for start in range(0, len(images), clip_batch_size):
            inputs = processor(
                images=images[start : start + clip_batch_size],
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                encoded = _clip_output_tensor(model.get_image_features(**inputs))
            encoded_images.append(encoded.float().to("cpu").numpy())
        video_features.append(np.concatenate(encoded_images, axis=0))
        text_inputs = processor(
            text=[str(row["instruction"])],
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.inference_mode():
            text = _clip_output_tensor(model.get_text_features(**text_inputs))
        text_features.append(text.float().to("cpu").numpy()[0])
        state_features.append(np.stack(states))
    after = tensor_digest(model)
    if before != after:
        raise RuntimeError("CLIP backbone changed during feature extraction")
    counts = _stage_counts(config)
    arrays = {
        "video": np.stack(video_features).astype(np.float16),
        "text": np.stack(text_features).astype(np.float16),
        "state": np.stack(state_features).astype(np.float32),
        "stage": np.asarray([int(row["stage_id"]) for row in rows], dtype=np.int64),
        "completion": np.asarray(
            [float(row["stage_completion"]) for row in rows],
            dtype=np.float32,
        ),
        "progress": np.asarray(
            [float(row["overall_progress"]) for row in rows],
            dtype=np.float32,
        ),
        "stage_count": np.asarray(
            [counts[(str(row["suite"]), int(row["task_id"]))] for row in rows],
            dtype=np.float32,
        ),
        "split": np.asarray([str(row["split"]) for row in rows]),
        "episode_id": np.asarray([str(row["episode_id"]) for row in rows]),
        "frame_index": np.asarray(
            [int(row["frame_index"]) for row in rows], dtype=np.int64
        ),
        "episode_success": np.asarray(
            [bool(row["episode_success"]) for row in rows],
            dtype=np.bool_,
        ),
    }
    temporary = cache_path.with_name(cache_path.name + ".partial.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, cache_path)
    return {
        "clip_model_id": config["clip_model_id"],
        "clip_revision": config["clip_revision"],
        "clip_digest_before": before,
        "clip_digest_after": after,
        "clip_frozen": True,
        "optimizer_contains_clip": False,
        "processor_serialization": {
            "status": "pass",
            "files": processor_files,
            "reloaded_type": type(reloaded_processor).__name__,
        },
        "frame_deltas": deltas,
        "feature_shapes": {
            key: list(value.shape)
            for key, value in arrays.items()
            if key in {"video", "text", "state"}
        },
    }


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _models(config: dict[str, Any], device: Any) -> tuple[Any, Any]:
    from lerobot.rewards.sarm.modeling_sarm import (
        StageTransformer,
        SubtaskTransformer,
    )

    kwargs = {
        "d_model": int(config["hidden_dim"]),
        "vis_emb_dim": 512,
        "text_emb_dim": 512,
        "state_dim": 32,
        "n_layers": int(config["num_layers"]),
        "n_heads": int(config["num_heads"]),
        "dropout": float(config["dropout"]),
        "num_cameras": 1,
    }
    stage = StageTransformer(
        **kwargs,
        num_classes_sparse=8,
        num_classes_dense=8,
    ).to(device)
    subtask = SubtaskTransformer(**kwargs).to(device)
    return stage, subtask


def _forward(
    stage_model: Any,
    subtask_model: Any,
    *,
    video: Any,
    text: Any,
    state: Any,
    stage_count: Any,
    target_stage: Any | None,
    teacher_force: bool,
) -> tuple[Any, Any, Any]:
    import torch
    from torch.nn import functional as functional

    lengths = torch.full(
        (video.shape[0],),
        video.shape[1],
        device=video.device,
        dtype=torch.long,
    )
    image_sequence = video.unsqueeze(1)
    stage_logits_sequence = stage_model(
        image_sequence,
        text,
        state,
        lengths,
        scheme="dense",
    )
    stage_logits = stage_logits_sequence[:, -1]
    predicted_stage = stage_logits.argmax(dim=-1)
    prior_indices = (
        target_stage if teacher_force and target_stage is not None else predicted_stage
    )
    prior = functional.one_hot(prior_indices, num_classes=8).to(dtype=video.dtype)
    prior_sequence = prior[:, None, None, :].expand(-1, 1, video.shape[1], -1)
    completion_sequence = subtask_model(
        image_sequence,
        text,
        state,
        lengths,
        prior_sequence,
        scheme="dense",
    )
    completion = completion_sequence[:, -1]
    probabilities = torch.softmax(stage_logits, dim=-1)
    indices = torch.arange(8, device=video.device, dtype=probabilities.dtype)
    expected_stage = (probabilities * indices).sum(dim=-1)
    overall = torch.minimum(
        (expected_stage + completion) / stage_count,
        torch.ones_like(completion),
    )
    return stage_logits, completion, overall


def _split_metrics(
    stage_model: Any,
    subtask_model: Any,
    arrays: dict[str, np.ndarray],
    *,
    split: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    mask = arrays["split"] == split
    indices = np.flatnonzero(mask)
    stage_predictions: list[int] = []
    progress_predictions: list[float] = []
    completion_predictions: list[float] = []
    batch_size = 128
    stage_model.eval()
    subtask_model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            logits, completion, progress = _forward(
                stage_model,
                subtask_model,
                video=torch.from_numpy(arrays["video"][selected].astype(np.float32)).to(
                    device
                ),
                text=torch.from_numpy(arrays["text"][selected].astype(np.float32)).to(
                    device
                ),
                state=torch.from_numpy(arrays["state"][selected]).to(device),
                stage_count=torch.from_numpy(arrays["stage_count"][selected]).to(
                    device
                ),
                target_stage=None,
                teacher_force=False,
            )
            stage_predictions.extend(logits.argmax(dim=-1).to("cpu").tolist())
            completion_predictions.extend(completion.to("cpu").tolist())
            progress_predictions.extend(progress.to("cpu").tolist())
    stage_targets = arrays["stage"][indices].tolist()
    progress_targets = arrays["progress"][indices].tolist()
    completion_targets = arrays["completion"][indices].tolist()
    stage = evaluate_stage_predictions(stage_targets, stage_predictions, stage_count=8)
    progress = evaluate_progress(progress_targets, progress_predictions)
    completion = evaluate_progress(completion_targets, completion_predictions)
    prediction_by_index = {
        int(index): prediction
        for index, prediction in zip(indices, progress_predictions, strict=True)
    }
    target_deltas = []
    predicted_deltas = []
    same_stage = []
    for episode_id in sorted(set(arrays["episode_id"][indices])):
        episode_indices = [
            int(index) for index in indices if arrays["episode_id"][index] == episode_id
        ]
        episode_indices.sort(key=lambda index: arrays["frame_index"][index])
        for left, right in zip(episode_indices, episode_indices[1:], strict=False):
            target_deltas.append(
                float(arrays["progress"][right] - arrays["progress"][left])
            )
            predicted_deltas.append(
                prediction_by_index[right] - prediction_by_index[left]
            )
            same_stage.append(bool(arrays["stage"][left] == arrays["stage"][right]))
    same_mask = np.asarray(same_stage, dtype=bool)
    target_array = np.asarray(target_deltas, dtype=np.float64)
    predicted_array = np.asarray(predicted_deltas, dtype=np.float64)
    result = {
        "samples": len(indices),
        "stage": {
            "accuracy": stage.accuracy,
            "macro_f1": stage.macro_f1,
            "per_stage_f1": stage.per_stage_f1,
            "confusion_matrix": stage.confusion_matrix,
        },
        "stage_completion": {
            "mae": completion.mae,
            "rmse": completion.rmse,
            "spearman": completion.spearman,
        },
        "progress": {
            "mae": progress.mae,
            "rmse": progress.rmse,
            "spearman": progress.spearman,
        },
        "pairwise": {
            "accuracy": pairwise_accuracy(target_array, predicted_array, epsilon=0.02),
            "same_stage_accuracy": (
                pairwise_accuracy(
                    target_array[same_mask],
                    predicted_array[same_mask],
                    epsilon=0.02,
                )
                if same_mask.any()
                else None
            ),
            "cross_stage_accuracy": (
                pairwise_accuracy(
                    target_array[~same_mask],
                    predicted_array[~same_mask],
                    epsilon=0.02,
                )
                if (~same_mask).any()
                else None
            ),
        },
        "temporal": {
            "monotonicity_violation_rate": (
                float((predicted_array < -0.02).mean()) if len(predicted_array) else 0.0
            ),
            "stage_regression_detection_recall": (
                float((predicted_array[target_array < -0.02] < -0.02).mean())
                if (target_array < -0.02).any()
                else None
            ),
            "stagnation_detection_recall": (
                float(
                    (
                        np.abs(predicted_array[np.abs(target_array) <= 0.02]) <= 0.02
                    ).mean()
                )
                if (np.abs(target_array) <= 0.02).any()
                else None
            ),
            "negative_progress_detection_recall": (
                float((predicted_array[target_array < -0.02] < -0.02).mean())
                if (target_array < -0.02).any()
                else None
            ),
        },
    }
    return result


def _ranking_loss(
    predictions: Any,
    targets: Any,
    episode_ids: np.ndarray,
    frame_indices: np.ndarray,
) -> Any:
    import torch
    from torch.nn import functional as functional

    losses = []
    for episode_id in sorted(set(episode_ids.tolist())):
        positions = np.flatnonzero(episode_ids == episode_id)
        positions = positions[np.argsort(frame_indices[positions])]
        for left, right in zip(positions, positions[1:], strict=False):
            delta = targets[int(right)] - targets[int(left)]
            if torch.abs(delta) <= 0.02:
                continue
            sign = torch.sign(delta)
            predicted_delta = predictions[int(right)] - predictions[int(left)]
            losses.append(functional.relu(0.01 - sign * predicted_delta))
    return (
        torch.stack(losses).mean()
        if losses
        else torch.zeros((), device=predictions.device)
    )


def _write_predictions(
    path: Path,
    stage_model: Any,
    subtask_model: Any,
    arrays: dict[str, np.ndarray],
    *,
    device: Any,
) -> None:
    import torch

    rows: list[dict[str, Any]] = []
    stage_model.eval()
    subtask_model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays["stage"]), 128):
            selected = np.arange(start, min(start + 128, len(arrays["stage"])))
            logits, completion, progress = _forward(
                stage_model,
                subtask_model,
                video=torch.from_numpy(arrays["video"][selected].astype(np.float32)).to(
                    device
                ),
                text=torch.from_numpy(arrays["text"][selected].astype(np.float32)).to(
                    device
                ),
                state=torch.from_numpy(arrays["state"][selected]).to(device),
                stage_count=torch.from_numpy(arrays["stage_count"][selected]).to(
                    device
                ),
                target_stage=None,
                teacher_force=False,
            )
            predicted_stages = logits.argmax(dim=-1).to("cpu").tolist()
            predicted_completion = completion.to("cpu").tolist()
            predicted_progress = progress.to("cpu").tolist()
            for offset, index in enumerate(selected):
                rows.append(
                    {
                        "schema_version": ("latentguard.lg_r1.sarm_prediction.v1"),
                        "episode_id": str(arrays["episode_id"][index]),
                        "frame_index": int(arrays["frame_index"][index]),
                        "split": str(arrays["split"][index]),
                        "episode_success": bool(arrays["episode_success"][index]),
                        "target_stage": int(arrays["stage"][index]),
                        "predicted_stage": int(predicted_stages[offset]),
                        "target_stage_completion": float(arrays["completion"][index]),
                        "predicted_stage_completion": float(
                            predicted_completion[offset]
                        ),
                        "target_progress": float(arrays["progress"][index]),
                        "predicted_progress": float(predicted_progress[offset]),
                    }
                )
    write_jsonl_atomic(path, rows)


def main() -> None:
    """Train only the official small stage/progress modules."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1/sarm.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--result", type=Path)
    add_training_arguments(parser)
    args = parser.parse_args()
    config, run_dir = resolved_training_config(
        args.config, args, run_name="sarm-style-small"
    )
    if args.dry_run:
        write_run_manifest(
            run_dir,
            run_type="sarm_style_small",
            config=config,
            status="dry_run",
            optimizer_steps=0,
            checkpoint_path=None,
            extra={
                "training_started": False,
                "official_checkpoint": config["official_checkpoint"],
            },
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
        raise ValueError("stage QA gate must pass before SARM training")
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
    cache_path = run_dir / "frozen_clip_features.npz"
    feature_manifest_path = run_dir / "feature_manifest.json"
    feature_manifest: dict[str, Any]
    if not cache_path.is_file():
        feature_manifest = _build_clip_cache(
            rows,
            runtime=runtime,
            cache_path=cache_path,
            config=config,
        )
        feature_manifest["sample_identity"] = _sample_identity(rows)
        write_json(feature_manifest_path, feature_manifest)
    else:
        feature_manifest = read_json(feature_manifest_path)
        if feature_manifest["sample_identity"] != _sample_identity(rows):
            raise ValueError("frozen CLIP feature sample identity drift")
    arrays = _load_arrays(cache_path)
    if len(arrays["video"]) != len(rows):
        raise ValueError("frozen CLIP feature cache length mismatch")
    import torch
    from torch.nn import functional as functional

    device = torch.device("cuda")
    stage_model, subtask_model = _models(config, device)
    parameters = [
        *stage_model.parameters(),
        *subtask_model.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        parameters,
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
        stage_model.load_state_dict(resumed["stage_model"])
        subtask_model.load_state_dict(resumed["subtask_model"])
        optimizer.load_state_dict(resumed["optimizer"])
        step = int(resumed["step"])
        best_mae = float(resumed["best_mae"])
        patience = int(resumed["patience"])
    train_indices = np.flatnonzero(arrays["split"] == "train")
    generator = np.random.default_rng(seed + step)
    interrupted = False
    try:
        while step < int(config["max_steps"]):
            stage_model.train()
            subtask_model.train()
            selected = generator.choice(
                train_indices,
                size=min(int(config["batch_size"]), len(train_indices)),
                replace=len(train_indices) < int(config["batch_size"]),
            )
            video = torch.from_numpy(arrays["video"][selected].astype(np.float32)).to(
                device
            )
            text = torch.from_numpy(arrays["text"][selected].astype(np.float32)).to(
                device
            )
            state = torch.from_numpy(arrays["state"][selected]).to(device)
            target_stage = torch.from_numpy(arrays["stage"][selected]).to(device)
            target_completion = torch.from_numpy(arrays["completion"][selected]).to(
                device
            )
            target_progress = torch.from_numpy(arrays["progress"][selected]).to(device)
            stage_count = torch.from_numpy(arrays["stage_count"][selected]).to(device)
            teacher_force = bool(generator.random() < 0.75)
            logits, completion, progress = _forward(
                stage_model,
                subtask_model,
                video=video,
                text=text,
                state=state,
                stage_count=stage_count,
                target_stage=target_stage,
                teacher_force=teacher_force,
            )
            rank_loss = _ranking_loss(
                progress,
                target_progress,
                arrays["episode_id"][selected],
                arrays["frame_index"][selected],
            )
            loss = (
                float(config["stage_loss_weight"])
                * functional.cross_entropy(logits, target_stage)
                + float(config["completion_loss_weight"])
                * functional.mse_loss(completion, target_completion)
                + float(config["progress_loss_weight"])
                * functional.mse_loss(progress, target_progress)
                + float(config["ranking_loss_weight"]) * rank_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite SARM loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % int(config["eval_every_steps"]) == 0 or step == int(
                config["max_steps"]
            ):
                validation = _split_metrics(
                    stage_model,
                    subtask_model,
                    arrays,
                    split="validation",
                    device=device,
                )
                validation_mae = float(validation["progress"]["mae"])
                if validation_mae < best_mae:
                    best_mae = validation_mae
                    patience = 0
                    save_torch_checkpoint(
                        best_checkpoint,
                        {
                            "stage_model": stage_model.state_dict(),
                            "subtask_model": subtask_model.state_dict(),
                            "step": step,
                            "validation": validation,
                        },
                    )
                else:
                    patience += 1
                save_torch_checkpoint(
                    checkpoint,
                    {
                        "stage_model": stage_model.state_dict(),
                        "subtask_model": subtask_model.state_dict(),
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
                "stage_model": stage_model.state_dict(),
                "subtask_model": subtask_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_mae": best_mae,
                "patience": patience,
            },
        )
    if interrupted:
        write_run_manifest(
            run_dir,
            run_type="sarm_style_small",
            config=config,
            status="interrupted",
            optimizer_steps=step,
            checkpoint_path=checkpoint,
            extra=feature_manifest,
        )
        return
    if not best_checkpoint.is_file():
        raise RuntimeError("no validation-selected SARM checkpoint")
    selected_state = torch.load(
        best_checkpoint, map_location=device, weights_only=False
    )
    stage_model.load_state_dict(selected_state["stage_model"])
    subtask_model.load_state_dict(selected_state["subtask_model"])
    reload_stage, reload_subtask = _models(config, device)
    reload_stage.load_state_dict(selected_state["stage_model"])
    reload_subtask.load_state_dict(selected_state["subtask_model"])
    selected_validation = _split_metrics(
        stage_model,
        subtask_model,
        arrays,
        split="validation",
        device=device,
    )
    reloaded_validation = _split_metrics(
        reload_stage,
        reload_subtask,
        arrays,
        split="validation",
        device=device,
    )
    if reloaded_validation != selected_validation:
        raise RuntimeError("SARM checkpoint reload changed validation output")
    results = {
        "schema_version": "latentguard.lg_r1.sarm_results.v1",
        "status": "pass",
        "claim_label": config["claim_label"],
        "official_checkpoint_zero_shot": False,
        "official_checkpoint_compatible": False,
        "implementation": config["implementation"],
        "lerobot_commit": config["lerobot_commit"],
        "clip_model_id": config["clip_model_id"],
        "clip_revision": config["clip_revision"],
        "trained_parameters": ("StageTransformer and SubtaskTransformer only"),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "clip_frozen": True,
        "vlajepa_used": False,
        "optimizer_steps": step,
        "best_validation_step": int(selected_state["step"]),
        "validation": selected_validation,
        "test": _split_metrics(
            stage_model,
            subtask_model,
            arrays,
            split="test",
            device=device,
        ),
        "checkpoint_resume_validation": "pass",
        "feature_manifest": feature_manifest,
        "cuda_memory": peak_cuda_memory(),
    }
    result_path = (
        resolve_repo_path(args.result)
        if args.result is not None
        else runtime / "sarm_results.json"
    )
    prediction_path = runtime / "sarm_predictions.jsonl"
    _write_predictions(
        prediction_path,
        stage_model,
        subtask_model,
        arrays,
        device=device,
    )
    results["prediction_locator"] = "sarm_predictions.jsonl"
    write_json(result_path, results)
    write_json(run_dir / "results.json", results)
    write_run_manifest(
        run_dir,
        run_type="sarm_style_small",
        config=config,
        status="pass",
        optimizer_steps=step,
        checkpoint_path=best_checkpoint,
        extra=feature_manifest,
    )
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
