"""Run exact-revision TOPReward zero-shot inference on frozen windows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _lg_r1c_common import (
    output_root,
    read_yaml,
    resolve_repo_path,
    runtime_identity,
    runtime_root,
    write_json,
)
from _lg_r1c_reward_runtime import (
    EpisodeDatasetCache,
    append_predictions,
    benchmark_summary,
    load_completed_ids,
    load_videos,
    load_windows,
    monotonic_time,
    processor_identity,
    snapshot_exact,
    tensor_digest,
    validate_snapshot_files,
)


def _run_view(
    *,
    model: object,
    processor: object,
    cache: EpisodeDatasetCache,
    windows: list[dict[str, object]],
    prediction_path: Path,
    model_revision: str,
    frame_key: str,
    index_key: str,
    batch_size: int,
    resume: bool,
) -> float:
    import torch

    if prediction_path.exists() and not resume:
        raise FileExistsError(f"TOPReward predictions exist: {prediction_path}")
    completed = load_completed_ids(
        prediction_path,
        model_revision=model_revision,
    )
    pending = [row for row in windows if str(row["window_id"]) not in completed]
    started = monotonic_time()
    for start in range(0, len(pending), batch_size):
        batch_windows = pending[start : start + batch_size]
        videos = load_videos(
            cache,
            batch_windows,
            frame_key=frame_key,
            index_key=index_key,
        )
        tasks = [str(row["instruction"]) for row in batch_windows]
        encoded = processor._encode_batch(videos, tasks, len(batch_windows))
        batch = {
            f"observation.topreward.{key}": value for key, value in encoded.items()
        }
        with torch.inference_mode():
            rewards = model.compute_reward(batch).detach().to("cpu").tolist()
        rows = []
        for window, raw in zip(batch_windows, rewards, strict=True):
            value = float(raw)
            rows.append(
                {
                    "schema_version": "latentguard.lg_r1c.topreward_prediction.v1",
                    "model_revision": model_revision,
                    "window_id": window["window_id"],
                    "raw_true_token_log_probability": value,
                    "normalized_window_reward": math.exp(value),
                    "normalization": "exp_raw_log_probability",
                    "per_frame_reward_available": False,
                }
            )
        append_predictions(prediction_path, rows)
    return monotonic_time() - started


def main() -> None:
    """Evaluate TOPReward without prompt tuning, an optimizer, or backward."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/topreward.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-native16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = runtime_root(args.runtime_root)
    destination = output_root(args.output_root)
    config = read_yaml(resolve_repo_path(args.config))
    stack = read_yaml(resolve_repo_path(Path(str(config["stack_config"]))))
    model_revision = str(config["model_revision"])
    windows = load_windows(destination)
    if args.limit_windows is not None:
        windows = windows[: int(args.limit_windows)]
    native_windows = [row for row in windows if "topreward_native_frame_indices" in row]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "common_windows": len(windows),
                    "native16_windows": len(native_windows),
                    "model_revision": model_revision,
                    "optimizer_steps": 0,
                },
                sort_keys=True,
            )
        )
        return
    snapshot = snapshot_exact(
        role="topreward-qwen3-vl-8b",
        model_id=str(config["model_id"]),
        revision=model_revision,
    )
    weight_files = validate_snapshot_files(
        snapshot,
        stack["topreward"]["weight_files"],
    )
    processor_files = processor_identity(snapshot)
    import torch
    from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel
    from lerobot.rewards.topreward.processor_topreward import (
        TOPRewardEncoderProcessorStep,
    )

    reward_config = TOPRewardConfig(
        vlm_name=str(snapshot),
        torch_dtype=str(config["torch_dtype"]),
        device=str(config["device"]),
        image_key=str(config["image_key"]),
        max_frames=int(config["native_diagnostic_frames"]),
        fps=float(config["fps"]),
        prompt_prefix=str(config["prompt_prefix"]),
        prompt_suffix_template=str(config["prompt_suffix_template"]),
        add_chat_template=bool(config["add_chat_template"]),
    )
    model = TOPRewardModel(reward_config).to(str(config["device"]))
    model.requires_grad_(False)
    model.eval()
    common_processor = TOPRewardEncoderProcessorStep(
        vlm_name=str(snapshot),
        image_key=str(config["image_key"]),
        max_frames=int(config["common_frames"]),
        fps=float(config["fps"]),
        prompt_prefix=str(config["prompt_prefix"]),
        prompt_suffix_template=str(config["prompt_suffix_template"]),
        add_chat_template=bool(config["add_chat_template"]),
    )
    token_ids = common_processor._processor.tokenizer.encode(
        str(config["scored_token"]),
        add_special_tokens=False,
    )
    if len(token_ids) != 1:
        raise ValueError("TOPReward frozen scored token is not one token")
    digest_before = tensor_digest(model)
    cache = EpisodeDatasetCache(source)
    torch.cuda.reset_peak_memory_stats()
    common_path = destination / "topreward_predictions.jsonl"
    common_elapsed = _run_view(
        model=model,
        processor=common_processor,
        cache=cache,
        windows=windows,
        prediction_path=common_path,
        model_revision=model_revision,
        frame_key=str(config["image_key"]),
        index_key="common_frame_indices",
        batch_size=int(config["batch_size"]),
        resume=args.resume,
    )
    native_path = destination / "topreward_native16_predictions.jsonl"
    native_elapsed = 0.0
    if native_windows and not args.skip_native16:
        native_processor = TOPRewardEncoderProcessorStep(
            vlm_name=str(snapshot),
            image_key=str(config["image_key"]),
            max_frames=int(config["native_diagnostic_frames"]),
            fps=float(config["fps"]),
            prompt_prefix=str(config["prompt_prefix"]),
            prompt_suffix_template=str(config["prompt_suffix_template"]),
            add_chat_template=bool(config["add_chat_template"]),
        )
        native_elapsed = _run_view(
            model=model,
            processor=native_processor,
            cache=cache,
            windows=native_windows,
            prediction_path=native_path,
            model_revision=model_revision,
            frame_key=str(config["image_key"]),
            index_key="topreward_native_frame_indices",
            batch_size=int(config["batch_size"]),
            resume=args.resume,
        )
    summary = benchmark_summary(
        prediction_path=common_path,
        model=model,
        digest_before=digest_before,
        elapsed_seconds=common_elapsed,
        windows=len(windows),
        model_identity={
            "model_id": config["model_id"],
            "revision": model_revision,
            "files": weight_files,
        },
        processor=processor_files,
    )
    summary.update(
        {
            "schema_version": "latentguard.lg_r1c.topreward_inference.v1",
            "runtime_identity": runtime_identity(),
            "strict_zero_shot": True,
            "prompt_frozen": True,
            "prompt_prefix": config["prompt_prefix"],
            "prompt_suffix_template": config["prompt_suffix_template"],
            "scored_token": config["scored_token"],
            "scored_token_id": token_ids[0],
            "per_frame_reward_available": False,
            "native16_windows": len(native_windows),
            "native16_elapsed_seconds": native_elapsed,
            "native16_prediction_file": (
                {
                    "locator": native_path.name,
                    "bytes": native_path.stat().st_size,
                    "sha256": __import__("hashlib")
                    .sha256(native_path.read_bytes())
                    .hexdigest(),
                }
                if native_path.is_file()
                else None
            ),
        }
    )
    write_json(destination / "topreward_inference.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
