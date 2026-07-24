"""Run the online-time and frozen-SARM baselines on common LG-R1c endpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from _lg_r1_training import peak_cuda_memory
from _lg_r1c_common import (
    file_identity,
    output_root,
    read_json,
    read_jsonl,
    read_yaml,
    resolve_repo_path,
    runtime_root,
    sha256_path,
    write_json,
    write_jsonl,
)
from _lg_r1c_reward_runtime import load_windows
from lg_r1_train_or_eval_sarm import (
    _build_clip_cache,
    _load_arrays,
    _models,
    _write_predictions,
)


def _time_predictions(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for window in windows:
        denominator = max(int(window["episode_horizon"]) - 1, 1)
        rows.append(
            {
                "schema_version": "latentguard.lg_r1c.scalar_prediction.v1",
                "model": "online_time",
                "window_id": window["window_id"],
                "episode_id": window["episode_id"],
                "frame_index": window["end_frame"],
                "score": min(int(window["end_frame"]) / denominator, 1.0),
                "observed_episode_length_used": False,
            }
        )
    return rows


def _unique_endpoint_rows(
    windows: list[dict[str, Any]],
    source: Path,
) -> list[dict[str, Any]]:
    current = read_jsonl(source / "progress_dataset" / "current_samples.jsonl")
    by_identity = {
        (str(row["episode_id"]), int(row["frame_index"])): row for row in current
    }
    identities = sorted(
        {(str(window["episode_id"]), int(window["end_frame"])) for window in windows}
    )
    rows = []
    for identity in identities:
        row = by_identity.get(identity)
        if row is None:
            raise ValueError(f"SARM endpoint missing from current samples: {identity}")
        rows.append(dict(row))
    return rows


def _run_sarm(
    *,
    windows: list[dict[str, Any]],
    source: Path,
    destination: Path,
    config: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    rows = _unique_endpoint_rows(windows, source)
    cache_root = destination / "sarm-common"
    cache_path = cache_root / "frozen_clip_features.npz"
    feature_manifest_path = cache_root / "feature_manifest.json"
    if not cache_path.is_file():
        manifest = _build_clip_cache(
            rows,
            runtime=source,
            cache_path=cache_path,
            config=config,
        )
        manifest["endpoint_count"] = len(rows)
        write_json(feature_manifest_path, manifest)
    else:
        manifest = read_json(feature_manifest_path)
        if int(manifest["endpoint_count"]) != len(rows):
            raise ValueError("SARM common endpoint cache drift")
    arrays = _load_arrays(cache_path)
    if len(arrays["video"]) != len(rows):
        raise ValueError("SARM cache length mismatch")
    import torch

    device = torch.device("cuda")
    stage_model, subtask_model = _models(config, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    stage_model.load_state_dict(state["stage_model"])
    subtask_model.load_state_dict(state["subtask_model"])
    endpoint_path = cache_root / "endpoint_predictions.jsonl"
    _write_predictions(
        endpoint_path,
        stage_model,
        subtask_model,
        arrays,
        device=device,
    )
    endpoint_predictions = read_jsonl(endpoint_path)
    by_identity = {
        (str(row["episode_id"]), int(row["frame_index"])): row
        for row in endpoint_predictions
    }
    predictions = []
    for window in windows:
        identity = (str(window["episode_id"]), int(window["end_frame"]))
        prediction = by_identity[identity]
        predictions.append(
            {
                "schema_version": "latentguard.lg_r1c.scalar_prediction.v1",
                "model": "frozen_sarm",
                "window_id": window["window_id"],
                "episode_id": window["episode_id"],
                "frame_index": window["end_frame"],
                "predicted_progress": float(prediction["predicted_progress"]),
                "predicted_stage": int(prediction["predicted_stage"]),
                "predicted_stage_completion": float(
                    prediction["predicted_stage_completion"]
                ),
            }
        )
    prediction_path = destination / "sarm_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    return {
        "schema_version": "latentguard.lg_r1c.sarm_results.v1",
        "status": "pass",
        "evaluation": "frozen SARM on exact common reward-window endpoints",
        "checkpoint_sha256": sha256_path(checkpoint),
        "checkpoint_mutated": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "unique_endpoints": len(rows),
        "windows": len(windows),
        "prediction_file": file_identity(
            prediction_path,
            locator="sarm_predictions.jsonl",
        ),
        "feature_manifest": manifest,
        "historical_lg_r1b": read_json(
            resolve_repo_path(Path("artifacts/lg_r1b/sarm_zero_shot_results.json"))
        ),
        "cuda_memory": peak_cuda_memory(),
    }


def main() -> None:
    """Execute baseline inference without adaptation or test-driven selection."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/baselines.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--skip-sarm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = runtime_root(args.runtime_root)
    destination = output_root(args.output_root)
    config = read_yaml(resolve_repo_path(args.config))
    windows = load_windows(destination)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "windows": len(windows),
                    "optimizer_steps": 0,
                    "new_rollouts": 0,
                },
                sort_keys=True,
            )
        )
        return
    time_path = destination / "time_predictions.jsonl"
    write_jsonl(time_path, _time_predictions(windows))
    time_result = {
        "schema_version": "latentguard.lg_r1c.time_baseline_results.v1",
        "status": "pass",
        "definition": config["time"]["definition"],
        "observed_terminal_length_used": False,
        "windows": len(windows),
        "prediction_file": file_identity(
            time_path,
            locator="time_predictions.jsonl",
        ),
    }
    write_json(destination / "time_baseline_results.json", time_result)
    payload: dict[str, Any] = {"time": time_result, "sarm": {"status": "skipped"}}
    if not args.skip_sarm:
        checkpoint = args.checkpoint
        if checkpoint is None:
            source_root = os.environ.get("LG_R1_SOURCE_ROOT")
            if not source_root:
                raise ValueError("--checkpoint or LG_R1_SOURCE_ROOT is required")
            checkpoint = Path(source_root) / str(config["sarm"]["checkpoint_locator"])
        base = read_yaml(resolve_repo_path(Path(str(config["sarm"]["base_config"]))))
        evaluation = read_yaml(
            resolve_repo_path(Path(str(config["sarm"]["frozen_eval_config"])))
        )
        merged = {**base, **evaluation}
        sarm = _run_sarm(
            windows=windows,
            source=source,
            destination=destination,
            config=merged,
            checkpoint=checkpoint,
        )
        write_json(destination / "sarm_results.json", sarm)
        payload["sarm"] = sarm
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
