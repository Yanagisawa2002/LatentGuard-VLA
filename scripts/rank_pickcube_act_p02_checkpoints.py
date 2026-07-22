"""Rank bounded ACT checkpoints with phase- and gripper-conditioned metrics."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import numpy as np
import torch

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.data import PickCubeDemoSplit
from latentguard.policies.act.grasp_supervision import (
    close_transition_index_in_window,
    phase_from_expert_label,
    predicted_close_index,
)
from latentguard.policies.act.runtime import (
    PickCubeActDataset,
    load_inference_runtime,
    split_references,
    validate_dataset_for_training,
)
from latentguard.policies.act.types import PickCubeActExperimentConfig


class PickCubeP02CheckpointMetricError(RuntimeError):
    """Raised when checkpoint metrics cannot be recomputed exactly."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeP02CheckpointMetricError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeP02CheckpointMetricError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _validation_losses(run: Path) -> Mapping[int, float]:
    result: dict[int, float] = {}
    path = run / "metrics.jsonl"
    if not path.is_file() or path.is_symlink():
        _fail("training metrics", "metrics.jsonl is missing")
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(raw)
        step = value.get("step") if isinstance(value, Mapping) else None
        loss = value.get("validation_loss") if isinstance(value, Mapping) else None
        if type(step) is int and isinstance(loss, (int, float)) and math.isfinite(loss):
            result[step] = float(loss)
    return result


def _phase_inventory(
    dataset_root: Path, dataset: PickCubeActDataset
) -> Mapping[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    for episode_index, reference in enumerate(dataset.references):
        directory = dataset_root.joinpath(
            *PurePosixPath(reference.relative_directory).parts
        )
        metadata = _mapping(directory / "episode.json", "episode metadata")
        raw = metadata.get("phases")
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != reference.frame_count
        ):
            _fail("episode metadata", "phase inventory differs")
        result[episode_index] = tuple(
            phase_from_expert_label(cast(str, item)).value for item in raw
        )
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


@torch.no_grad()
def rank(
    *,
    dataset_root: Path,
    training_run: Path,
    contract_path: Path,
    output: Path,
) -> Mapping[str, object]:
    """Evaluate every complete checkpoint on the full immutable validation split."""
    root = Path(training_run).absolute()
    experiment = PickCubeActExperimentConfig.from_mapping(
        _mapping(root / "resolved_config.json", "resolved config")
    )
    if not experiment.bounded:
        _fail("training run", "only intrinsically bounded checkpoints are eligible")
    run_manifest = _mapping(root / "run_manifest.json", "run manifest")
    identity = run_manifest.get("training_identity")
    if not isinstance(identity, Mapping):
        _fail("training run", "training identity is missing")
    contract = _mapping(contract_path, "ACT contract")
    action = contract.get("action")
    bounds = action.get("bounds") if isinstance(action, Mapping) else None
    lower = bounds.get("lower") if isinstance(bounds, Mapping) else None
    upper = bounds.get("upper") if isinstance(bounds, Mapping) else None
    if not isinstance(lower, Sequence) or not isinstance(upper, Sequence):
        _fail("ACT contract", "action bounds are missing")
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    references, manifest, _ = validate_dataset_for_training(
        dataset_root, experiment=experiment
    )
    validation = split_references(references, PickCubeDemoSplit.VALIDATION)
    dataset = PickCubeActDataset(
        dataset_root,
        validation,
        chunk_size=experiment.model.chunk_size,
    )
    phases = _phase_inventory(Path(dataset_root).absolute(), dataset)
    validation_losses = _validation_losses(root)
    checkpoints = tuple(sorted((root / "checkpoints").glob("step-*")))
    if not checkpoints:
        _fail("training run", "no checkpoints exist")

    results: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        runtime = load_inference_runtime(
            checkpoint=checkpoint,
            expected_identity=cast(Mapping[str, object], identity),
            experiment=experiment,
            action_lower=cast(Sequence[float], lower),
            action_upper=cast(Sequence[float], upper),
        )
        arm_errors: list[float] = []
        gripper_errors: list[float] = []
        first_action_errors: list[float] = []
        position_errors: dict[int, list[float]] = defaultdict(list)
        phase_arm_errors: dict[str, list[float]] = defaultdict(list)
        phase_gripper_errors: dict[str, list[float]] = defaultdict(list)
        timing_errors: list[float] = []
        predicted_rows: list[np.ndarray[Any, Any]] = []
        target_close_count = 0
        predicted_close_count = 0
        true_close_count = 0
        boundary_violations = 0
        nonfinite = 0
        latencies: list[float] = []
        runtime.reset()
        for sample_index, (episode_index, frame_index) in enumerate(dataset.samples):
            sample = dataset[sample_index]
            started = time.perf_counter()
            predicted = runtime.predict_action_chunk_for_audit(
                sample[experiment.model.image_feature_key].permute(1, 2, 0).numpy(),
                sample[experiment.model.state_feature_key].numpy(),
            )
            latencies.append(time.perf_counter() - started)
            target = sample[experiment.model.action_feature_key].numpy()
            valid = np.logical_not(sample["action_is_pad"].numpy())
            valid_predicted = predicted[valid]
            valid_target = target[valid]
            predicted_rows.append(valid_predicted)
            nonfinite += int(valid_predicted.size - np.isfinite(valid_predicted).sum())
            boundary_violations += int(
                np.logical_or(
                    valid_predicted < lower_array,
                    valid_predicted > upper_array,
                ).sum()
            )
            absolute = np.abs(valid_predicted - valid_target)
            arm_errors.extend(np.mean(absolute[:, :7], axis=1).tolist())
            gripper_errors.extend(absolute[:, 7].tolist())
            first_action_errors.append(float(np.mean(absolute[0])))
            episode_phases = phases[episode_index]
            for chunk_position in range(len(valid_predicted)):
                position_errors[chunk_position].append(
                    float(np.mean(absolute[chunk_position]))
                )
                phase = episode_phases[frame_index + chunk_position]
                phase_arm_errors[phase].append(
                    float(np.mean(absolute[chunk_position, :7]))
                )
                phase_gripper_errors[phase].append(float(absolute[chunk_position, 7]))
            target_index = close_transition_index_in_window(
                episode_phases,
                window_start=frame_index,
                window_length=len(valid_predicted),
            )
            predicted_index = predicted_close_index(
                valid_predicted, close_threshold=-0.5
            )
            target_close_count += int(target_index is not None)
            predicted_close_count += int(predicted_index is not None)
            true_close_count += int(
                target_index is not None and predicted_index is not None
            )
            if target_index is not None and predicted_index is not None:
                timing_errors.append(float(predicted_index - target_index))
        values = np.concatenate(predicted_rows, axis=0)
        precision = (
            true_close_count / predicted_close_count if predicted_close_count else 0.0
        )
        recall = true_close_count / target_close_count if target_close_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        step = int(checkpoint.name.removeprefix("step-"))
        result: dict[str, object] = {
            "action_integrity_passed": boundary_violations == 0 and nonfinite == 0,
            "arm_action_mae": float(np.mean(arm_errors)),
            "checkpoint": checkpoint.name,
            "chunk_position_mae": {
                str(index): float(np.mean(position_errors[index]))
                for index in sorted(position_errors)
            },
            "close_event_f1": f1,
            "close_event_precision": precision,
            "close_event_recall": recall,
            "close_transition_timing_error_mean": _mean(timing_errors),
            "first_action_mae": float(np.mean(first_action_errors)),
            "gripper_action_mae": float(np.mean(gripper_errors)),
            "native_boundary_violation_count": boundary_violations,
            "nonfinite_action_count": nonfinite,
            "phase_arm_mae": {
                name: float(np.mean(items))
                for name, items in sorted(phase_arm_errors.items())
            },
            "phase_gripper_mae": {
                name: float(np.mean(items))
                for name, items in sorted(phase_gripper_errors.items())
            },
            "predicted_action_standard_deviation": np.std(values, axis=0).tolist(),
            "predicted_closed_endpoint_rate": float(np.mean(values[:, 7] < -0.95)),
            "predicted_open_endpoint_rate": float(np.mean(values[:, 7] > 0.95)),
            "query_latency_p95_seconds": float(np.quantile(latencies, 0.95)),
            "total_validation_loss": validation_losses.get(step),
            "validation_frame_count": len(dataset),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del runtime
        torch.cuda.empty_cache()

    def ranking_key(item: Mapping[str, object]) -> tuple[object, ...]:
        timing = item["close_transition_timing_error_mean"]
        return (
            item["action_integrity_passed"] is not True,
            -cast(float, item["close_event_f1"]),
            abs(cast(float, timing)) if isinstance(timing, (int, float)) else math.inf,
            cast(Mapping[str, float], item["phase_gripper_mae"]).get(
                "GRIPPER_CLOSING", math.inf
            ),
            cast(float, item["arm_action_mae"]),
            cast(float | None, item["total_validation_loss"])
            if item["total_validation_loss"] is not None
            else math.inf,
        )

    ranked = sorted(results, key=ranking_key)
    report = {
        "checkpoint_count": len(results),
        "checkpoints": results,
        "contract_digest": contract.get("contract_digest"),
        "dataset_digest": manifest.get("dataset_digest"),
        "passed": all(item["action_integrity_passed"] is True for item in results),
        "ranking": [item["checkpoint"] for item in ranked],
        "ranking_rule": [
            "action_integrity",
            "close_event_f1_descending",
            "absolute_close_transition_timing_error",
            "gripper_closing_phase_mae",
            "arm_action_mae",
            "total_validation_loss",
        ],
        "schema_version": "pickcube-act-p02-checkpoint-phase-metrics-v1",
        "training_identity_digest": identity.get("training_identity_digest"),
    }
    write_atomic_json(Path(output).absolute(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rank(
        dataset_root=args.dataset_root,
        training_run=args.training_run,
        contract_path=args.contract,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
