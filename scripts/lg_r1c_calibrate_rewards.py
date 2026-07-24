"""Fit validation-only scalar calibrators and the simple task-agnostic ensemble."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1c_common import (
    file_identity,
    output_root,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    sha256_path,
    write_json,
    write_jsonl,
)


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["window_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate prediction window in {path}")
    return result


def _fit_affine(features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    design = np.concatenate(
        [features, np.ones((len(features), 1), dtype=np.float64)],
        axis=1,
    )
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    return {
        "coefficients": coefficients[:-1].tolist(),
        "intercept": float(coefficients[-1]),
    }


def _apply_affine(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    return np.clip(features @ coefficients + float(model["intercept"]), 0.0, 1.0)


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (features - mean) / scale
    weights = np.zeros(features.shape[1], dtype=np.float64)
    prevalence = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    intercept = math.log(prevalence / (1.0 - prevalence))
    for _ in range(steps):
        logits = np.clip(normalized @ weights + intercept, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = probabilities - labels
        weights -= learning_rate * (normalized.T @ residual / len(labels))
        intercept -= learning_rate * float(residual.mean())
    return {
        "weights": weights.tolist(),
        "intercept": intercept,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "optimizer_steps": steps,
        "learning_rate": learning_rate,
    }


def _apply_logistic(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    normalized = (features - mean) / scale
    logits = np.clip(
        normalized @ weights + float(model["intercept"]),
        -30.0,
        30.0,
    )
    return 1.0 / (1.0 + np.exp(-logits))


def _require_zero_shot_freeze(destination: Path) -> dict[str, Any]:
    freeze_path = destination / "zero_shot_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("zero-shot results must be frozen before calibration")
    freeze = read_json(freeze_path)
    for name, expected in freeze["prediction_sha256"].items():
        path = destination / str(name)
        if sha256_path(path) != expected:
            raise ValueError(f"zero-shot prediction changed after freeze: {name}")
    return freeze


def _load_validation_windows(destination: Path) -> list[dict[str, Any]]:
    manifest = read_json(destination / "window_manifest.json")
    projection = manifest.get("calibration_validation_windows")
    if not isinstance(projection, dict):
        raise ValueError("validation-only calibration projection is missing")
    if projection.get("selection_split") != "validation":
        raise ValueError("calibration projection split drift")
    if projection.get("test_labels_included") is not False:
        raise ValueError("calibration projection exposes test labels")
    identity = projection.get("file")
    if not isinstance(identity, dict):
        raise ValueError("calibration projection identity is missing")
    path = destination / str(identity["locator"])
    if sha256_path(path) != identity["sha256"]:
        raise ValueError("calibration validation projection hash drift")
    rows = read_jsonl(path)
    if len(rows) != int(projection["rows"]):
        raise ValueError("calibration validation projection count drift")
    if any(row.get("split") != "validation" for row in rows):
        raise ValueError("calibration projection contains a non-validation row")
    return rows


def main() -> None:
    """Fit pre-declared scalar models on validation labels only."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/calibration.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    destination = output_root(args.output_root)
    config = read_yaml(resolve_repo_path(args.config))
    manifest = read_json(destination / "window_manifest.json")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "selection_split": config["selection_split"],
                    "windows": int(manifest["window_count"]),
                    "foundation_model_training": False,
                },
                sort_keys=True,
            )
        )
        return
    freeze = _require_zero_shot_freeze(destination)
    validation_windows = _load_validation_windows(destination)
    robometer = _prediction_map(destination / "robometer_predictions.jsonl")
    topreward = _prediction_map(destination / "topreward_predictions.jsonl")
    if set(robometer) != set(topreward):
        raise ValueError("zero-shot model endpoints differ")
    primary = [
        row
        for row in validation_windows
        if row["purpose"] == "anchor_grid" and row["context"] == "medium"
    ]
    if not primary:
        raise ValueError("validation anchor windows are empty")
    progress_features = np.asarray(
        [
            [
                float(robometer[str(row["window_id"])]["last_frame_progress"]),
                float(topreward[str(row["window_id"])]["normalized_window_reward"]),
            ]
            for row in primary
        ],
        dtype=np.float64,
    )
    progress_targets = np.asarray(
        [float(row["progress_target"]) for row in primary],
        dtype=np.float64,
    )
    success_features = np.asarray(
        [
            [
                float(robometer[str(row["window_id"])]["last_frame_progress"]),
                float(
                    robometer[str(row["window_id"])]["last_frame_success_probability"]
                ),
                float(topreward[str(row["window_id"])]["normalized_window_reward"]),
            ]
            for row in primary
        ],
        dtype=np.float64,
    )
    success_targets = np.asarray(
        [bool(row["terminal_success"]) for row in primary],
        dtype=np.float64,
    )
    matched_validation = [
        row for row in validation_windows if row["purpose"] == "failure_matched"
    ]
    failure_features = np.asarray(
        [
            [
                float(robometer[str(row["window_id"])]["last_frame_progress"]),
                float(
                    robometer[str(row["window_id"])]["last_frame_success_probability"]
                ),
                float(topreward[str(row["window_id"])]["normalized_window_reward"]),
            ]
            for row in matched_validation
        ],
        dtype=np.float64,
    )
    failure_targets = np.asarray(
        [bool(row["failure_label"]) for row in matched_validation],
        dtype=np.float64,
    )
    steps = int(config["maximum_optimizer_steps"])
    learning_rate = float(config["learning_rate"])
    models = {
        "robometer_progress_affine": _fit_affine(
            progress_features[:, :1],
            progress_targets,
        ),
        "topreward_progress_affine": _fit_affine(
            progress_features[:, 1:2],
            progress_targets,
        ),
        "ensemble_progress_affine": _fit_affine(
            progress_features,
            progress_targets,
        ),
        "robometer_success_logistic": _fit_logistic(
            success_features[:, 1:2],
            success_targets,
            steps=steps,
            learning_rate=learning_rate,
        ),
        "topreward_success_logistic": _fit_logistic(
            success_features[:, 2:3],
            success_targets,
            steps=steps,
            learning_rate=learning_rate,
        ),
        "ensemble_success_logistic": _fit_logistic(
            success_features,
            success_targets,
            steps=steps,
            learning_rate=learning_rate,
        ),
        "ensemble_failure_logistic": _fit_logistic(
            failure_features,
            failure_targets,
            steps=steps,
            learning_rate=learning_rate,
        ),
    }
    output_rows = []
    for window_id in robometer:
        rbm_progress = float(robometer[window_id]["last_frame_progress"])
        rbm_success = float(robometer[window_id]["last_frame_success_probability"])
        top_score = float(topreward[window_id]["normalized_window_reward"])
        progress_pair = np.asarray([[rbm_progress, top_score]])
        all_features = np.asarray([[rbm_progress, rbm_success, top_score]])
        output_rows.append(
            {
                "schema_version": "latentguard.lg_r1c.calibrated_prediction.v1",
                "window_id": window_id,
                "robometer_calibrated_progress": float(
                    _apply_affine(
                        models["robometer_progress_affine"],
                        progress_pair[:, :1],
                    )[0]
                ),
                "robometer_calibrated_success": float(
                    _apply_logistic(
                        models["robometer_success_logistic"],
                        all_features[:, 1:2],
                    )[0]
                ),
                "topreward_calibrated_progress": float(
                    _apply_affine(
                        models["topreward_progress_affine"],
                        progress_pair[:, 1:2],
                    )[0]
                ),
                "topreward_calibrated_success": float(
                    _apply_logistic(
                        models["topreward_success_logistic"],
                        all_features[:, 2:3],
                    )[0]
                ),
                "ensemble_progress": float(
                    _apply_affine(
                        models["ensemble_progress_affine"],
                        progress_pair,
                    )[0]
                ),
                "ensemble_success": float(
                    _apply_logistic(
                        models["ensemble_success_logistic"],
                        all_features,
                    )[0]
                ),
                "ensemble_failure": float(
                    _apply_logistic(
                        models["ensemble_failure_logistic"],
                        all_features,
                    )[0]
                ),
            }
        )
    prediction_path = destination / "calibrated_predictions.jsonl"
    write_jsonl(prediction_path, output_rows)
    result = {
        "schema_version": "latentguard.lg_r1c.calibrated_reward_results.v1",
        "status": "pass",
        "selection_split": "validation",
        "fit_splits": ["validation"],
        "test_labels_used_for_fit": False,
        "test_labels_used_for_method_selection": False,
        "foundation_models_frozen": True,
        "zero_shot_freeze": freeze,
        "calibrators": models,
        "fit_samples": {
            "progress_and_success": len(primary),
            "failure": len(matched_validation),
        },
        "label_projection": {
            "selection_split": "validation",
            "test_labels_loaded": False,
            "file": manifest["calibration_validation_windows"]["file"],
        },
        "task_id_input": False,
        "stage_id_input": False,
        "privileged_input": False,
        "future_terminal_input": False,
        "prediction_file": file_identity(
            prediction_path,
            locator="calibrated_predictions.jsonl",
        ),
    }
    write_json(destination / "calibrated_reward_results.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
