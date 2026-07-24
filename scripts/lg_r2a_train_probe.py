"""Train one pre-registered LG-R2a probe variant over five grouped folds."""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r2a_common import (
    output_root,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse required safety and resumability controls."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batch_indices(
    indices: np.ndarray, batch_size: int, generator: np.random.Generator
) -> list[np.ndarray]:
    order = generator.permutation(indices)
    return [
        order[start : start + batch_size] for start in range(0, len(order), batch_size)
    ]


def _preprocessing(
    data: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in ("features", "proprio"):
        values = data[key][indices].astype(np.float64)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        result[f"{key}_mean"] = mean.astype(np.float32)
        result[f"{key}_std"] = std.astype(np.float32)
    return result


def _class_weights(
    data: dict[str, np.ndarray], indices: np.ndarray, cap: float
) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for key in ("stagnation", "regression", "events", "terminal"):
        labels = data[key][indices]
        if key in {"stagnation", "regression"}:
            valid = data["progress_valid"][indices]
            positives = (labels * valid).sum(axis=0)
            negatives = valid.sum(axis=0) - positives
        else:
            positives = labels.sum(axis=0)
            negatives = labels.shape[0] - positives
        values = negatives / np.maximum(positives, 1.0)
        weights[key] = np.clip(values, 1.0, cap).astype(np.float32)
    return weights


def _inputs(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    preprocessing: dict[str, np.ndarray],
    device: Any,
) -> tuple[Any, Any, Any, Any]:
    import torch

    state = (
        data["features"][indices] - preprocessing["features_mean"]
    ) / preprocessing["features_std"]
    proprio = (
        data["proprio"][indices] - preprocessing["proprio_mean"]
    ) / preprocessing["proprio_std"]
    return (
        torch.as_tensor(state, dtype=torch.float32, device=device),
        torch.as_tensor(data["actions"][indices], dtype=torch.float32, device=device),
        torch.as_tensor(data["action_mask"][indices], dtype=torch.bool, device=device),
        torch.as_tensor(proprio, dtype=torch.float32, device=device),
    )


def _loss(
    outputs: dict[str, Any],
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    class_weights: dict[str, np.ndarray],
    weights: dict[str, float],
    device: Any,
) -> Any:
    import torch
    import torch.nn.functional as functional

    progress_target = torch.as_tensor(
        data["progress"][indices], dtype=torch.float32, device=device
    )
    progress_valid = torch.as_tensor(
        data["progress_valid"][indices], dtype=torch.bool, device=device
    )
    progress_raw = functional.huber_loss(
        outputs["progress"], progress_target, reduction="none", delta=0.1
    )
    progress_loss = progress_raw[progress_valid].mean()
    total = float(weights["progress"]) * progress_loss
    for key in ("stagnation", "regression", "events", "terminal"):
        target = torch.as_tensor(data[key][indices], dtype=torch.float32, device=device)
        pos_weight = torch.as_tensor(
            class_weights[key], dtype=torch.float32, device=device
        )
        binary_raw = functional.binary_cross_entropy_with_logits(
            outputs[key], target, pos_weight=pos_weight, reduction="none"
        )
        if key in {"stagnation", "regression"}:
            valid = torch.as_tensor(
                data["progress_valid"][indices], dtype=torch.bool, device=device
            )
            binary = binary_raw[valid].mean()
        else:
            binary = binary_raw.mean()
        total = total + float(weights[key]) * binary
    return total


def _predict(
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
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
            tensors = _inputs(data, positions, preprocessing, device)
            output = model(*tensors)
            for key, value in output.items():
                converted = value if key == "progress" else torch.sigmoid(value)
                results[key].append(converted.detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in results.items()}


def _validation_score(
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> float:
    valid = data["progress_valid"][indices, 0]
    short_mae = np.abs(
        predictions["progress"][valid, 0] - data["progress"][indices][valid, 0]
    ).mean()
    binary_brier = np.mean(
        [
            np.mean((predictions[key] - data[key][indices]) ** 2)
            for key in ("stagnation", "regression", "events", "terminal")
        ]
    )
    return float(short_mae + 0.25 * binary_brier)


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    if candidates.size > 200:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 201))
    best = (-1.0, 0.5)
    for threshold in candidates:
        prediction = scores >= threshold
        true_positive = float(np.sum(prediction & (labels == 1)))
        false_positive = float(np.sum(prediction & (labels == 0)))
        false_negative = float(np.sum(~prediction & (labels == 1)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        best = max(best, (f1, float(threshold)))
    return best[1]


def _thresholds(
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, list[float]]:
    return {
        key: [
            _best_f1_threshold(data[key][indices, column], values[:, column])
            for column in range(values.shape[1])
        ]
        for key, values in predictions.items()
        if key != "progress"
    }


def _json_arrays(payload: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {key: value.tolist() for key, value in payload.items()}


def _train_fold(
    *,
    fold: int,
    variant: str,
    data: dict[str, np.ndarray],
    folds: np.ndarray,
    config: dict[str, Any],
    run_dir: Path,
    seed: int,
    max_steps: int,
    checkpoint_every: int,
    eval_every: int,
    resume: bool,
    device: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from latentguard.action_conditioning.models import ProbeModel

    validation_fold = (fold + 1) % 5
    train_indices = np.flatnonzero((folds != fold) & (folds != validation_fold))
    validation_indices = np.flatnonzero(folds == validation_fold)
    test_indices = np.flatnonzero(folds == fold)
    preprocessing = _preprocessing(data, train_indices)
    class_weights = _class_weights(
        data, train_indices, float(config["class_weight_cap"])
    )
    _seed_everything(seed + fold)
    model = ProbeModel(variant=variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = run_dir / "checkpoints" / f"fold_{fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    best_score = math.inf
    best_state: dict[str, Any] | None = None
    best_step = 0
    patience = 0
    if resume and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        best_score = float(payload["best_score"])
        best_state = payload["best_state"]
        best_step = int(payload["best_step"])
        patience = int(payload["patience"])
    generator = np.random.default_rng(seed + fold)
    started = time.perf_counter()
    interrupted = False
    try:
        while step < max_steps and patience < int(config["early_stopping_patience"]):
            for batch in _batch_indices(
                train_indices, int(config["batch_size"]), generator
            ):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                tensors = _inputs(data, batch, preprocessing, device)
                output = dict(model(*tensors))
                loss = _loss(
                    output,
                    data,
                    batch,
                    class_weights,
                    config["loss_weights"],
                    device,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip_norm"])
                )
                optimizer.step()
                step += 1
                if step % eval_every == 0 or step == max_steps:
                    validation_predictions = _predict(
                        model,
                        data,
                        validation_indices,
                        preprocessing,
                        device,
                        int(config["batch_size"]),
                    )
                    score = _validation_score(
                        validation_predictions, data, validation_indices
                    )
                    if score < best_score - float(config["minimum_improvement"]):
                        best_score = score
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                        best_step = step
                        patience = 0
                    else:
                        patience += 1
                if step % checkpoint_every == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "best_score": best_score,
                            "best_state": best_state,
                            "best_step": best_step,
                            "patience": patience,
                            "preprocessing": preprocessing,
                        },
                        checkpoint,
                    )
                if step >= max_steps or patience >= int(
                    config["early_stopping_patience"]
                ):
                    break
    except KeyboardInterrupt:
        interrupted = True
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_score": best_score,
            "best_state": best_state,
            "best_step": best_step,
            "patience": patience,
            "preprocessing": preprocessing,
        },
        checkpoint,
    )
    if interrupted:
        raise KeyboardInterrupt
    if best_state is None:
        validation_predictions = _predict(
            model,
            data,
            validation_indices,
            preprocessing,
            device,
            int(config["batch_size"]),
        )
        best_score = _validation_score(validation_predictions, data, validation_indices)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_step = step
    model.load_state_dict(best_state)
    validation_predictions = _predict(
        model,
        data,
        validation_indices,
        preprocessing,
        device,
        int(config["batch_size"]),
    )
    selected_thresholds = _thresholds(validation_predictions, data, validation_indices)
    test_predictions = _predict(
        model,
        data,
        test_indices,
        preprocessing,
        device,
        int(config["batch_size"]),
    )
    torch.save(
        {
            "model": model.state_dict(),
            "preprocessing": preprocessing,
            "variant": variant,
            "fold": fold,
            "selected_thresholds": selected_thresholds,
        },
        run_dir / "checkpoints" / f"fold_{fold}_selected.pt",
    )
    latency_tensors = _inputs(
        data,
        validation_indices[: min(128, len(validation_indices))],
        preprocessing,
        device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            model(*latency_tensors)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (
        (time.perf_counter() - latency_start) * 1_000 / (20 * len(latency_tensors[0]))
    )
    report = {
        "fold": fold,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_indices),
        "optimizer_steps": step,
        "best_validation_step": best_step,
        "best_validation_score": best_score,
        "threshold_selection_split": "validation",
        "test_evaluations": 1,
        "selected_thresholds": selected_thresholds,
        "class_weights_train_only": _json_arrays(class_weights),
        "parameter_report": model.parameter_report(),
        "inference_latency_ms_per_sample": latency_ms,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {
        **test_predictions,
        "indices": test_indices,
    }, report


def main() -> None:
    """Train and validation-select one model family, testing each fold once."""

    args = parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    variant = str(config["variant"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    max_steps = int(
        args.max_steps if args.max_steps is not None else config["max_steps_per_fold"]
    )
    if args.dry_run:
        print(
            {
                "status": "dry_run",
                "variant": variant,
                "seed": seed,
                "max_steps": max_steps,
            }
        )
        return
    import torch

    output = output_root(args.output_dir)
    prepared = np.load(output / "prepared_data.npz", allow_pickle=False)
    data = {key: prepared[key] for key in prepared.files}
    metadata = read_jsonl(output / "prepared_metadata.jsonl")
    if args.limit_samples is not None:
        keep = min(args.limit_samples, len(metadata))
        metadata = metadata[:keep]
        data = {key: value[:keep] for key, value in data.items()}
    folds = np.asarray([int(row["fold"]) for row in metadata], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("training data must retain all five registered folds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_dir = output / "runs" / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_arrays = {
        "progress": np.full((len(metadata), 3), np.nan, dtype=np.float32),
        "stagnation": np.full((len(metadata), 3), np.nan, dtype=np.float32),
        "regression": np.full((len(metadata), 3), np.nan, dtype=np.float32),
        "events": np.full((len(metadata), 2), np.nan, dtype=np.float32),
        "terminal": np.full((len(metadata), 1), np.nan, dtype=np.float32),
    }
    test_counts = np.zeros(len(metadata), dtype=np.int8)
    fold_reports = []
    started = time.perf_counter()
    for fold in range(5):
        fold_predictions, report = _train_fold(
            fold=fold,
            variant=variant,
            data=data,
            folds=folds,
            config=config,
            run_dir=run_dir,
            seed=seed,
            max_steps=max_steps,
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            resume=args.resume,
            device=device,
        )
        indices = fold_predictions.pop("indices")
        for key, values in fold_predictions.items():
            prediction_arrays[key][indices] = values
        test_counts[indices] += 1
        fold_reports.append(report)
    if not np.all(test_counts == 1):
        raise RuntimeError("each grouped-CV sample must be tested exactly once")
    if any(not np.isfinite(value).all() for value in prediction_arrays.values()):
        raise RuntimeError("OOF predictions contain non-finite or missing values")
    prediction_path = run_dir / "oof_predictions.npz"
    np.savez_compressed(prediction_path, **prediction_arrays, test_count=test_counts)
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    result = {
        "schema_version": "latentguard.lg_r2a.probe_run.v1",
        "status": "pass",
        "variant": variant,
        "seed": seed,
        "architecture_search": False,
        "fusion": "single pre-registered concatenation plus action gate",
        "foundation_models_frozen": True,
        "feature_cache_frozen": True,
        "trainable_components": ["diagnostic_head"]
        + ([] if variant == "state_only" else ["action_encoder"]),
        "folds": fold_reports,
        "test_consumption": "one evaluation per sample after fold validation selection",
        "prediction_file": "oof_predictions.npz",
        "peak_gpu_memory_bytes": peak_memory,
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_identity": runtime_identity(),
        "resolved_config": config,
        "checkpoint_resume_supported": True,
        "graceful_interruption_supported": True,
    }
    write_json(run_dir / "run_manifest.json", result)
    write_json(output / f"{variant}_run.json", result)
    print(
        {
            "status": "pass",
            "variant": variant,
            "samples": len(metadata),
            "peak_gpu_memory_bytes": peak_memory,
        }
    )


if __name__ == "__main__":
    main()
