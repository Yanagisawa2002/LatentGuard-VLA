"""Train, validate, checkpoint, and resume the native PickCube ACT policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.data import PickCubeDemoSplit
from latentguard.policies.act.runtime import (
    DeterministicResumeBatchSampler,
    PickCubeActDataset,
    PickCubeActRuntimeError,
    build_optimizer,
    build_policy_and_processors,
    installed_runtime_versions,
    load_checkpoint,
    prepare_training_batch,
    processor_statistics,
    save_checkpoint,
    seed_act_runtime,
    split_references,
    validate_dataset_for_training,
)
from latentguard.policies.act.types import (
    PickCubeActExperimentConfig,
    build_training_identity,
)


def _read_mapping(path: Path, *, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        raise PickCubeActRuntimeError(f"{context}: expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeActRuntimeError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise PickCubeActRuntimeError(f"{context}: expected JSON object")
    return cast(Mapping[str, object], value)


def _git(arguments: list[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _worker_seed(worker_id: int) -> None:
    del worker_id
    value = torch.initial_seed() % (2**32)
    import random

    import numpy as np

    random.seed(value)
    np.random.seed(value)


def _loader(
    dataset: PickCubeActDataset,
    *,
    experiment: PickCubeActExperimentConfig,
    start_step: int,
) -> Iterable[Mapping[str, object]]:
    sampler = DeterministicResumeBatchSampler(
        dataset_size=len(dataset),
        batch_size=experiment.optimization.batch_size,
        seed=experiment.seed,
        start_step=start_step,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=experiment.optimization.dataloader_workers,
        pin_memory=True,
        persistent_workers=experiment.optimization.dataloader_workers > 0,
        worker_init_fn=_worker_seed,
    )


def _autocast(experiment: PickCubeActExperimentConfig) -> Any:
    if experiment.optimization.mixed_precision == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise PickCubeActRuntimeError("CUDA device lacks bfloat16 support")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _processed(
    raw: Mapping[str, object],
    preprocessor: Any,
    experiment: PickCubeActExperimentConfig,
) -> dict[str, torch.Tensor]:
    value = preprocessor(prepare_training_batch(raw))
    if not isinstance(value, Mapping):
        raise PickCubeActRuntimeError("ACT preprocessor returned a non-mapping")
    result = cast(dict[str, torch.Tensor], dict(value))
    expected = {
        experiment.model.image_feature_key: (3, 224, 224),
        experiment.model.state_feature_key: (18,),
        experiment.model.action_feature_key: (experiment.model.chunk_size, 8),
        "action_is_pad": (experiment.model.chunk_size,),
    }
    for key, tail in expected.items():
        tensor = result.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.shape[1:] != tail:
            raise PickCubeActRuntimeError(
                f"processed ACT feature {key} has wrong shape"
            )
        if tensor.dtype != torch.bool and not torch.isfinite(tensor).all():
            raise PickCubeActRuntimeError(f"processed ACT feature {key} is nonfinite")
    return result


@torch.no_grad()
def _validation_loss(
    *,
    policy: Any,
    preprocessor: Any,
    batches: Iterable[Mapping[str, object]],
    experiment: PickCubeActExperimentConfig,
) -> float:
    policy.eval()
    losses: list[float] = []
    for raw in batches:
        batch = _processed(raw, preprocessor, experiment)
        with _autocast(experiment):
            loss, components = policy.forward(batch)
        if not torch.isfinite(loss) or any(
            not math.isfinite(float(value)) for value in components.values()
        ):
            raise PickCubeActRuntimeError("validation produced nonfinite loss")
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise PickCubeActRuntimeError("validation loader is empty")
    return sum(losses) / len(losses)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run_manifest(
    *,
    experiment: PickCubeActExperimentConfig,
    training_identity: Mapping[str, object],
    dataset_root: Path,
) -> Mapping[str, object]:
    if _git(["status", "--porcelain", "--untracked-files=no"]):
        raise PickCubeActRuntimeError("tracked source is dirty")
    return {
        "branch": _git(["branch", "--show-current"]),
        "dataset_root": str(Path(dataset_root).absolute()),
        "experiment": experiment.to_mapping(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "hostname": socket.gethostname(),
        "launch_command": list(sys.argv),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime_versions": dict(installed_runtime_versions()),
        "schema_version": "pickcube-native-act-run-manifest-v1",
        "training_identity": dict(training_identity),
    }


def train(
    *,
    config_path: Path,
    dataset_root: Path,
    contract_path: Path,
    output_dir: Path,
    dry_run: bool,
    max_steps: int | None,
    limit_samples: int | None,
    resume: Path | None,
) -> Mapping[str, object]:
    """Run the deterministic native ACT optimization lifecycle."""
    experiment = PickCubeActExperimentConfig.from_mapping(
        _read_mapping(config_path, context="training config")
    )
    if not torch.cuda.is_available():
        raise PickCubeActRuntimeError("CUDA is required for native ACT training")
    references, manifest, normalization = validate_dataset_for_training(
        dataset_root,
        experiment=experiment,
    )
    contract = _read_mapping(contract_path, context="ACT contract")
    contract_digest = contract.get("contract_digest")
    dataset_digest = manifest.get("dataset_digest")
    normalization_digest = normalization.get("normalization_digest")
    source_commit = _git(["rev-parse", "HEAD"])
    if not all(
        isinstance(value, str)
        for value in (contract_digest, dataset_digest, normalization_digest)
    ):
        raise PickCubeActRuntimeError("data or contract identity is incomplete")
    training_identity = build_training_identity(
        experiment=experiment,
        dataset_digest=cast(str, dataset_digest),
        normalization_digest=cast(str, normalization_digest),
        contract_digest=cast(str, contract_digest),
        source_commit=source_commit,
    )
    root = Path(output_dir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    resolved_path = root / "resolved_config.json"
    if not resolved_path.exists():
        write_atomic_json(resolved_path, experiment.to_mapping())
    run_manifest = _run_manifest(
        experiment=experiment,
        training_identity=training_identity,
        dataset_root=dataset_root,
    )
    if not (root / "run_manifest.json").exists():
        write_atomic_json(root / "run_manifest.json", run_manifest)
    if dry_run:
        summary = {
            "dataset_episode_count": len(references),
            "dry_run": True,
            "schema_version": "pickcube-native-act-training-summary-v1",
            "training_identity": dict(training_identity),
        }
        write_atomic_json(root / "dry_run_summary.json", summary)
        return summary

    versions = installed_runtime_versions()
    seed_settings = seed_act_runtime(experiment.seed)
    stats = processor_statistics(normalization)
    start_step = 0
    examples_processed = 0
    if resume is None:
        policy, preprocessor, postprocessor = build_policy_and_processors(
            experiment, stats
        )
        optimizer = build_optimizer(policy, experiment)
    else:
        policy, preprocessor, postprocessor, optimizer, state = load_checkpoint(
            checkpoint=resume,
            expected_identity=training_identity,
            experiment=experiment,
        )
        start_step = cast(int, state.get("global_step"))
        examples_processed = cast(int, state.get("examples_processed"))
        if type(start_step) is not int or type(examples_processed) is not int:
            raise PickCubeActRuntimeError("resume training state is malformed")
        control = state.get("training_control")
        if not isinstance(control, Mapping):
            raise PickCubeActRuntimeError("resume training control is missing")
        training_control = cast(Mapping[str, object], control)

    target_steps = (
        experiment.optimization.training_steps if max_steps is None else max_steps
    )
    if type(target_steps) is not int or target_steps < 1:
        raise PickCubeActRuntimeError("target steps must be positive")
    if start_step >= target_steps:
        return {
            "final_step": start_step,
            "resume_zero_work": True,
            "schema_version": "pickcube-native-act-training-summary-v1",
            "training_identity": dict(training_identity),
        }
    train_dataset = PickCubeActDataset(
        dataset_root,
        split_references(references, PickCubeDemoSplit.TRAIN),
        chunk_size=experiment.model.chunk_size,
        limit_samples=limit_samples,
    )
    validation_dataset = PickCubeActDataset(
        dataset_root,
        split_references(references, PickCubeDemoSplit.VALIDATION),
        chunk_size=experiment.model.chunk_size,
        limit_samples=limit_samples,
    )
    train_batches = _loader(
        train_dataset,
        experiment=experiment,
        start_step=start_step,
    )
    validation_batches = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=experiment.optimization.batch_size,
        shuffle=False,
        num_workers=experiment.optimization.dataloader_workers,
        pin_memory=True,
        persistent_workers=experiment.optimization.dataloader_workers > 0,
        worker_init_fn=_worker_seed,
    )
    iterator = iter(train_batches)
    interrupted = False

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        nonlocal interrupted
        interrupted = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    metrics_path = root / "metrics.jsonl"
    if resume is None:
        best_loss = math.inf
        best_step: int | None = None
        evaluations_without_improvement = 0
    else:
        saved_best_loss = training_control.get("best_validation_loss")
        saved_best_step = training_control.get("best_validation_step")
        saved_without_improvement = training_control.get(
            "evaluations_without_improvement"
        )
        if saved_best_loss is None and saved_best_step is None:
            best_loss = math.inf
            best_step = None
        elif (
            isinstance(saved_best_loss, (int, float))
            and math.isfinite(float(saved_best_loss))
            and type(saved_best_step) is int
            and 1 <= saved_best_step <= start_step
        ):
            best_loss = float(saved_best_loss)
            best_step = saved_best_step
        else:
            raise PickCubeActRuntimeError("resume best-validation state is malformed")
        if type(saved_without_improvement) is not int or saved_without_improvement < 0:
            raise PickCubeActRuntimeError("resume early-stop state is malformed")
        evaluations_without_improvement = saved_without_improvement
    checkpoints: list[Path] = []
    early_stopped = False
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(start_step + 1, target_steps + 1):
            step_started = time.perf_counter()
            raw = next(iterator)
            batch = _processed(raw, preprocessor, experiment)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(experiment):
                loss, components = policy.forward(batch)
            if not torch.isfinite(loss) or any(
                not math.isfinite(float(value)) for value in components.values()
            ):
                raise PickCubeActRuntimeError("training produced nonfinite loss")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in policy.parameters()
                if parameter.grad is not None
            ]
            if not gradients or any(
                not torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise PickCubeActRuntimeError("training gradients are invalid")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), experiment.optimization.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise PickCubeActRuntimeError("gradient norm is nonfinite")
            optimizer.step()
            torch.cuda.synchronize()
            examples_processed += int(
                batch[experiment.model.state_feature_key].shape[0]
            )
            validation_loss: float | None = None
            improved = False
            if (
                step % experiment.optimization.validation_interval == 0
                or step == target_steps
            ):
                validation_loss = _validation_loss(
                    policy=policy,
                    preprocessor=preprocessor,
                    batches=validation_batches,
                    experiment=experiment,
                )
                if (
                    best_loss - validation_loss
                    > experiment.optimization.early_stopping_min_delta
                ):
                    best_loss = validation_loss
                    best_step = step
                    evaluations_without_improvement = 0
                    improved = True
                else:
                    evaluations_without_improvement += 1
                policy.train()
            metric: dict[str, object] = {
                "action_loss": components.get("l1_loss"),
                "examples_processed": examples_processed,
                "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "kl_loss": components.get("kld_loss"),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step": step,
                "step_time_seconds": time.perf_counter() - step_started,
                "total_loss": float(loss.detach().cpu()),
                "validation_loss": validation_loss,
            }
            metric = {
                key: float(value.detach().cpu())
                if isinstance(value, torch.Tensor)
                else value
                for key, value in metric.items()
            }
            will_early_stop = (
                validation_loss is not None
                and evaluations_without_improvement
                >= experiment.optimization.early_stopping_patience_evaluations
            )
            should_checkpoint = (
                improved
                or step % experiment.optimization.checkpoint_interval == 0
                or step == target_steps
                or interrupted
                or will_early_stop
            )
            if should_checkpoint:
                checkpoint = save_checkpoint(
                    run_root=root,
                    step=step,
                    examples_processed=examples_processed,
                    training_identity=training_identity,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    optimizer=optimizer,
                    metric=metric,
                    training_control={
                        "best_validation_loss": (
                            best_loss if math.isfinite(best_loss) else None
                        ),
                        "best_validation_step": best_step,
                        "evaluations_without_improvement": (
                            evaluations_without_improvement
                        ),
                    },
                )
                checkpoints.append(checkpoint)
                metric["checkpoint"] = checkpoint.relative_to(root).as_posix()
            _append_jsonl(metrics_path, metric)
            print(json.dumps(metric, sort_keys=True), flush=True)
            if interrupted:
                break
            if will_early_stop:
                early_stopped = True
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    final_step = step
    all_checkpoints = sorted((root / "checkpoints").glob("step-*"))
    if not all_checkpoints:
        raise PickCubeActRuntimeError("training produced no checkpoint")
    if best_step is None:
        best_step = final_step
        best_loss = float(cast(float, metric["validation_loss"]))
    best = root / "checkpoints" / f"step-{best_step:08d}"
    if not best.is_dir():
        raise PickCubeActRuntimeError("best-validation checkpoint was not saved")
    midpoint = min(
        all_checkpoints,
        key=lambda path: abs(int(path.name.removeprefix("step-")) - final_step / 2),
    )
    roles = {
        "best_validation": best.relative_to(root).as_posix(),
        "early": all_checkpoints[0].relative_to(root).as_posix(),
        "final": all_checkpoints[-1].relative_to(root).as_posix(),
        "mid": midpoint.relative_to(root).as_posix(),
    }
    write_atomic_json(root / "checkpoint_roles.json", roles)
    summary = {
        "best_validation_loss": best_loss,
        "best_validation_step": best_step,
        "checkpoint_count": len(all_checkpoints),
        "checkpoint_roles": roles,
        "early_stopped": early_stopped,
        "examples_processed": examples_processed,
        "final_step": final_step,
        "interrupted": interrupted,
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resume_zero_work": False,
        "runtime_versions": dict(versions),
        "schema_version": "pickcube-native-act-training-summary-v1",
        "seed_settings": dict(seed_settings),
        "training_identity": dict(training_identity),
    }
    write_atomic_json(root / "training_summary.json", summary)
    return summary


def main() -> int:
    """Expose dry-run, bounded steps, sample limits, and checkpoint resume."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = train(
        config_path=args.config,
        dataset_root=args.dataset_root,
        contract_path=args.contract,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        limit_samples=args.limit_samples,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
