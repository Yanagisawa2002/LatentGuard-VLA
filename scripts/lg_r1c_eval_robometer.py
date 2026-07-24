"""Run exact-revision ROBOMETER zero-shot inference on frozen windows."""

from __future__ import annotations

import argparse
import json
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
    videos_to_uint8_samples,
)


def main() -> None:
    """Evaluate ROBOMETER without constructing an optimizer or backward pass."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1c/robometer.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--limit-windows", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = runtime_root(args.runtime_root)
    destination = output_root(args.output_root)
    config = read_yaml(resolve_repo_path(args.config))
    stack = read_yaml(resolve_repo_path(Path(str(config["stack_config"]))))
    model_revision = str(config["checkpoint_revision"])
    windows = load_windows(destination)
    if args.limit_windows is not None:
        windows = windows[: int(args.limit_windows)]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "windows": len(windows),
                    "model_revision": model_revision,
                    "optimizer_steps": 0,
                },
                sort_keys=True,
            )
        )
        return
    checkpoint = snapshot_exact(
        role="robometer",
        model_id=str(config["checkpoint_id"]),
        revision=model_revision,
    )
    base_processor = snapshot_exact(
        role="qwen3-vl-4b-processor",
        model_id=str(config["base_model_id"]),
        revision=str(config["base_model_revision"]),
        allow_patterns=(
            "*.json",
            "*.txt",
            "*.model",
        ),
    )
    checkpoint_files = validate_snapshot_files(
        checkpoint,
        stack["robometer"]["checkpoint_files"],
    )
    processor_files = processor_identity(base_processor)
    import torch
    from lerobot.configs.rewards import RewardModelConfig
    from lerobot.rewards.robometer.modeling_robometer import (
        ROBOMETER_INPUT_KEYS,
        RobometerRewardModel,
        decode_progress_outputs,
    )
    from lerobot.rewards.robometer.processor_robometer import (
        RobometerEncoderProcessorStep,
    )

    reward_config = RewardModelConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    reward_config.device = str(config["device"])
    reward_config.pretrained_path = str(checkpoint)
    reward_config.base_model_id = str(base_processor)
    model = RobometerRewardModel.from_pretrained(
        checkpoint,
        config=reward_config,
        local_files_only=True,
        strict=bool(config["strict_checkpoint_load"]),
    )
    model.requires_grad_(False)
    model.eval()
    processor = RobometerEncoderProcessorStep(
        base_model_id=str(base_processor),
        image_key=str(config["image_key"]),
        max_frames=int(config["maximum_frames"]),
        use_multi_image=True,
        use_per_frame_progress_token=True,
    )
    digest_before = tensor_digest(model)
    prediction_path = destination / "robometer_predictions.jsonl"
    if prediction_path.exists() and not args.resume:
        raise FileExistsError("ROBOMETER predictions exist; pass --resume")
    completed = load_completed_ids(
        prediction_path,
        model_revision=model_revision,
    )
    pending = [row for row in windows if str(row["window_id"]) not in completed]
    cache = EpisodeDatasetCache(source)
    batch_size = int(config["batch_size"])
    started = monotonic_time()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(pending), batch_size):
        batch_windows = pending[start : start + batch_size]
        videos = load_videos(
            cache,
            batch_windows,
            frame_key=str(config["image_key"]),
            index_key="common_frame_indices",
        )
        instructions = [str(row["instruction"]) for row in batch_windows]
        encoded = processor.encode_samples(
            videos_to_uint8_samples(videos, instructions)
        )
        device = next(model.parameters()).device
        inputs = {
            key: (
                encoded[key].to(device) if hasattr(encoded[key], "to") else encoded[key]
            )
            for key in ROBOMETER_INPUT_KEYS
            if key in encoded
        }
        with torch.inference_mode():
            progress_logits, success_logits = model._compute_rbm_logits(dict(inputs))
        decoded = decode_progress_outputs(
            progress_logits,
            success_logits,
            is_discrete_mode=reward_config.use_discrete_progress,
        )
        output_rows = []
        for window, progress, success in zip(
            batch_windows,
            decoded["progress_pred"],
            decoded["success_probs"],
            strict=True,
        ):
            if len(progress) != len(window["common_frame_indices"]):
                raise ValueError("ROBOMETER per-frame progress shape mismatch")
            if len(success) != len(window["common_frame_indices"]):
                raise ValueError("ROBOMETER per-frame success shape mismatch")
            output_rows.append(
                {
                    "schema_version": "latentguard.lg_r1c.robometer_prediction.v1",
                    "model_revision": model_revision,
                    "window_id": window["window_id"],
                    "per_frame_progress": [float(value) for value in progress],
                    "last_frame_progress": float(progress[-1]),
                    "per_frame_success_probability": [
                        float(value) for value in success
                    ],
                    "last_frame_success_probability": float(success[-1]),
                }
            )
        append_predictions(prediction_path, output_rows)
    elapsed = monotonic_time() - started
    summary = benchmark_summary(
        prediction_path=prediction_path,
        model=model,
        digest_before=digest_before,
        elapsed_seconds=elapsed,
        windows=len(windows),
        model_identity={
            "model_id": config["checkpoint_id"],
            "revision": model_revision,
            "files": checkpoint_files,
        },
        processor=processor_files,
    )
    summary.update(
        {
            "schema_version": "latentguard.lg_r1c.robometer_inference.v1",
            "runtime_identity": runtime_identity(),
            "strict_zero_shot": True,
            "per_frame_read_only_adapter": True,
            "public_compute_reward_last_frame_only": True,
            "preference_head_queried": False,
        }
    )
    write_json(destination / "robometer_inference.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
