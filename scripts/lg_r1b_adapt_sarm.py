"""Optionally adapt only the SARM heads after freezing zero-shot evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r1_training import (
    load_current_samples,
    peak_cuda_memory,
    save_torch_checkpoint,
    set_training_seed,
)
from _lg_r1b_common import (
    output_root,
    read_json,
    read_yaml,
    resolve_repo_path,
    sha256_path,
    write_json,
)
from lg_r1_train_or_eval_sarm import (
    _build_clip_cache,
    _forward,
    _load_arrays,
    _models,
    _ranking_loss,
    _sample_identity,
    _split_metrics,
)


def _balanced_primary_rows(
    rows: list[dict[str, Any]], maximum_per_task: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row["split"]) == "zero_shot_only":
            continue
        grouped.setdefault((str(row["suite"]), int(row["task_id"])), []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda item: (str(item["episode_id"]), int(item["frame_index"])),
        )
        if len(items) > maximum_per_task:
            indices = np.linspace(
                0, len(items) - 1, num=maximum_per_task, dtype=np.int64
            )
            items = [items[int(index)] for index in indices]
        selected.extend(items)
    return selected


def _source_checkpoint(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.initial_checkpoint is not None:
        return resolve_repo_path(args.initial_checkpoint)
    source_root = os.environ.get("LG_R1_SOURCE_ROOT")
    if not source_root:
        raise ValueError("--initial-checkpoint or LG_R1_SOURCE_ROOT is required")
    return Path(source_root) / str(config["frozen_lg_r1_checkpoint_locator"])


def main() -> None:
    """Run only when the pre-registered zero-shot trigger fired."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lg_r1b/adaptation.yaml"),
    )
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    adaptation = read_yaml(resolve_repo_path(args.config))
    base = read_yaml(resolve_repo_path(Path(str(adaptation["base_sarm_config"]))))
    config = {**base, **adaptation}
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.seed is not None:
        config["seed"] = args.seed
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    zero_path = resolve_repo_path(Path(str(config["zero_shot_result"])))
    if not zero_path.is_file():
        zero_path = runtime / "sarm_zero_shot_results.json"
    zero = read_json(zero_path)
    freeze = read_json(runtime / "sarm_zero_shot_freeze.json")
    if freeze["result_sha256"] != sha256_path(zero_path):
        raise ValueError("zero-shot result changed after its freeze record")
    result_path = (
        resolve_repo_path(args.result)
        if args.result is not None
        else runtime / "sarm_adaptation_results.json"
    )
    if not bool(zero["adaptation_triggered"]):
        result = {
            "schema_version": "latentguard.lg_r1b.sarm_adaptation_results.v1",
            "status": "not_run_trigger_not_met",
            "zero_shot_result_sha256": sha256_path(zero_path),
            "optimizer_steps": 0,
            "test_accessed": False,
            "frozen_modules": config["frozen_modules"],
        }
        write_json(result_path, result)
        print(json.dumps(result, sort_keys=True))
        return
    initial = _source_checkpoint(args, config)
    if not initial.is_file():
        raise FileNotFoundError(f"initial SARM checkpoint missing: {initial}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "triggered": True,
                    "initial_checkpoint_sha256": sha256_path(initial),
                },
                sort_keys=True,
            )
        )
        return
    seed = int(config["seed"])
    set_training_seed(seed)
    maximum = int(config["maximum_samples_per_task"])
    if args.limit_samples is not None:
        maximum = min(maximum, int(args.limit_samples))
    rows = _balanced_primary_rows(load_current_samples(runtime), maximum)
    run_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir is not None
        else runtime / "sarm-adaptation"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = run_dir / "frozen_clip_features.npz"
    feature_manifest_path = run_dir / "feature_manifest.json"
    if not cache.is_file():
        feature_manifest = _build_clip_cache(
            rows,
            runtime=runtime,
            cache_path=cache,
            config=config,
        )
        feature_manifest["sample_identity"] = _sample_identity(rows)
        write_json(feature_manifest_path, feature_manifest)
    else:
        feature_manifest = read_json(feature_manifest_path)
        if feature_manifest["sample_identity"] != _sample_identity(rows):
            raise ValueError("adaptation feature cache sample identity drift")
    arrays = _load_arrays(cache)
    import torch
    from torch.nn import functional

    device = torch.device("cuda")
    stage_model, subtask_model = _models(config, device)
    initial_state = torch.load(initial, map_location=device, weights_only=False)
    stage_model.load_state_dict(initial_state["stage_model"])
    subtask_model.load_state_dict(initial_state["subtask_model"])
    parameters = [*stage_model.parameters(), *subtask_model.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    last = (
        resolve_repo_path(args.checkpoint)
        if args.checkpoint is not None
        else run_dir / "last.pt"
    )
    best = run_dir / "best.pt"
    step = 0
    best_mae = float("inf")
    patience = 0
    if args.resume:
        resumed = torch.load(last, map_location=device, weights_only=False)
        stage_model.load_state_dict(resumed["stage_model"])
        subtask_model.load_state_dict(resumed["subtask_model"])
        optimizer.load_state_dict(resumed["optimizer"])
        step = int(resumed["step"])
        best_mae = float(resumed["best_mae"])
        patience = int(resumed["patience"])
    train_indices = np.flatnonzero(arrays["split"] == "train")
    if not len(train_indices):
        raise ValueError("adaptation training task split has no samples")
    generator = np.random.default_rng(seed + step)
    while step < int(config["max_steps"]):
        selected = generator.choice(
            train_indices,
            size=min(int(config["batch_size"]), len(train_indices)),
            replace=len(train_indices) < int(config["batch_size"]),
        )
        video = torch.from_numpy(arrays["video"][selected].astype(np.float32)).to(
            device
        )
        text = torch.from_numpy(arrays["text"][selected].astype(np.float32)).to(device)
        state = torch.from_numpy(arrays["state"][selected]).to(device)
        target_stage = torch.from_numpy(arrays["stage"][selected]).to(device)
        target_completion = torch.from_numpy(arrays["completion"][selected]).to(device)
        target_progress = torch.from_numpy(arrays["progress"][selected]).to(device)
        stage_count = torch.from_numpy(arrays["stage_count"][selected]).to(device)
        logits, completion, progress = _forward(
            stage_model,
            subtask_model,
            video=video,
            text=text,
            state=state,
            stage_count=stage_count,
            target_stage=target_stage,
            teacher_force=bool(generator.random() < 0.75),
        )
        loss = (
            float(config["stage_loss_weight"])
            * functional.cross_entropy(logits, target_stage)
            + float(config["completion_loss_weight"])
            * functional.mse_loss(completion, target_completion)
            + float(config["progress_loss_weight"])
            * functional.mse_loss(progress, target_progress)
            + float(config["ranking_loss_weight"])
            * _ranking_loss(
                progress,
                target_progress,
                arrays["episode_id"][selected],
                arrays["frame_index"][selected],
            )
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite SARM adaptation loss")
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
                    best,
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
                last,
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
    if not best.is_file():
        raise RuntimeError("adaptation produced no validation-selected checkpoint")
    selected_state = torch.load(best, map_location=device, weights_only=False)
    stage_model.load_state_dict(selected_state["stage_model"])
    subtask_model.load_state_dict(selected_state["subtask_model"])
    adapted_held_out_test = _split_metrics(
        stage_model,
        subtask_model,
        arrays,
        split="test",
        device=device,
    )
    frozen_stage_model, frozen_subtask_model = _models(config, device)
    frozen_stage_model.load_state_dict(initial_state["stage_model"])
    frozen_subtask_model.load_state_dict(initial_state["subtask_model"])
    frozen_initial_held_out_test = _split_metrics(
        frozen_stage_model,
        frozen_subtask_model,
        arrays,
        split="test",
        device=device,
    )
    result = {
        "schema_version": "latentguard.lg_r1b.sarm_adaptation_results.v1",
        "status": "pass",
        "zero_shot_result_sha256": sha256_path(zero_path),
        "initial_checkpoint_sha256": sha256_path(initial),
        "task_level_split": True,
        "optimizer_steps": step,
        "best_validation_step": int(selected_state["step"]),
        "validation": selected_state["validation"],
        "held_out_test": adapted_held_out_test,
        "frozen_initial_held_out_test": frozen_initial_held_out_test,
        "test_accessed_before_selection": False,
        "test_accessed_after_selection": True,
        "frozen_zero_shot_held_out_task": zero["held_out_task"],
        "frozen_modules": config["frozen_modules"],
        "trainable_modules": config["trainable_modules"],
        "feature_manifest": feature_manifest,
        "cuda_memory": peak_cuda_memory(),
        "claim_boundary": "Task-level SARM adaptation; not a failure head.",
    }
    write_json(result_path, result)
    write_json(run_dir / "results.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
