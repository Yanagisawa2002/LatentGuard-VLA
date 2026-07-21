"""Static, environment-free screening for every complete P0.1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import numpy as np
import torch

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.data import PickCubeDemoSplit
from latentguard.policies.act.runtime import (
    PickCubeActDataset,
    load_inference_runtime,
    split_references,
    validate_dataset_for_training,
)
from latentguard.policies.act.types import PickCubeActExperimentConfig
from latentguard.policies.actions import BoundedActionRegressionHead


class BoundedCheckpointScreenError(RuntimeError):
    """Raised when screening inputs or intrinsic action safety differ."""


def _fail(context: str, reason: str) -> NoReturn:
    raise BoundedCheckpointScreenError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundedCheckpointScreenError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _git_identity() -> Mapping[str, str]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    if git("status", "--short", "--untracked-files=no"):
        _fail("source", "tracked worktree is not clean")
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
    }


def _contract_bounds(
    contract: Mapping[str, object],
) -> tuple[Sequence[float], Sequence[float]]:
    action = contract.get("action")
    bounds = action.get("bounds") if isinstance(action, Mapping) else None
    lower = bounds.get("lower") if isinstance(bounds, Mapping) else None
    upper = bounds.get("upper") if isinstance(bounds, Mapping) else None
    if not isinstance(lower, Sequence) or not isinstance(upper, Sequence):
        _fail("contract", "action bounds are missing")
    return cast(Sequence[float], lower), cast(Sequence[float], upper)


def _anchor_phases(dataset_root: Path, dataset: PickCubeActDataset) -> Counter[str]:
    cache: dict[int, Sequence[str]] = {}
    counts: Counter[str] = Counter()
    for episode_index, frame_index in dataset.samples:
        phases = cache.get(episode_index)
        if phases is None:
            reference = dataset.references[episode_index]
            directory = dataset_root.joinpath(
                *PurePosixPath(reference.relative_directory).parts
            )
            metadata = _mapping(directory / "episode.json", "episode metadata")
            raw = metadata.get("phases")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                _fail("episode metadata", "phases are missing")
            phases = cast(Sequence[str], raw)
            cache[episode_index] = phases
        counts[phases[frame_index]] += 1
    return counts


@torch.no_grad()
def screen(
    *,
    dataset_root: Path,
    training_run: Path,
    contract_path: Path,
    config_path: Path,
    output: Path,
    checkpoint: Path | None = None,
) -> Mapping[str, object]:
    """Screen every checkpoint on one immutable validation observation view."""
    config = _mapping(config_path, "screen config")
    if config.get("schema_version") != "pickcube-act-bounded-checkpoint-screen-v1":
        _fail("screen config", "schema mismatch")
    minimum = config.get("minimum_anchor_count")
    if type(minimum) is not int or minimum < 2000:
        _fail("screen config", "anchor coverage was weakened")
    run = Path(training_run).absolute()
    experiment = PickCubeActExperimentConfig.from_mapping(
        _mapping(run / "resolved_config.json", "resolved config")
    )
    if not experiment.bounded:
        _fail("training run", "unbounded checkpoint is ineligible")
    manifest = _mapping(run / "run_manifest.json", "run manifest")
    identity = manifest.get("training_identity")
    if not isinstance(identity, Mapping):
        _fail("training run", "identity is missing")
    contract = _mapping(contract_path, "ACT contract")
    lower, upper = _contract_bounds(contract)
    references, _, _ = validate_dataset_for_training(
        dataset_root,
        experiment=experiment,
    )
    validation = split_references(references, PickCubeDemoSplit.VALIDATION)
    all_dataset = PickCubeActDataset(
        dataset_root,
        validation,
        chunk_size=experiment.model.chunk_size,
    )
    if len(all_dataset) < 1:
        _fail("validation anchors", "immutable observation view is empty")
    anchor_indices = tuple(
        (
            np.linspace(0, len(all_dataset) - 1, minimum, dtype=np.int64)
            if minimum <= len(all_dataset)
            else np.arange(minimum, dtype=np.int64) % len(all_dataset)
        ).tolist()
    )
    dataset = PickCubeActDataset(
        dataset_root,
        validation,
        chunk_size=experiment.model.chunk_size,
        limit_samples=None,
    )
    phase_counts = _anchor_phases(
        Path(dataset_root).absolute(),
        _IndexedDatasetView(dataset, anchor_indices),
    )
    required_phases = config.get("phase_order")
    if not isinstance(required_phases, Sequence) or any(
        phase_counts[str(phase)] == 0 for phase in required_phases
    ):
        _fail("validation anchors", "required phase coverage is missing")

    checkpoint_results: list[Mapping[str, object]] = []
    checkpoints = (
        (Path(checkpoint).absolute(),)
        if checkpoint is not None
        else tuple(sorted((run / "checkpoints").glob("step-*")))
    )
    if not checkpoints:
        _fail("training run", "no checkpoints exist")
    for checkpoint in checkpoints:
        try:
            checkpoint.relative_to(run / "checkpoints")
        except ValueError as exc:
            raise BoundedCheckpointScreenError(
                "checkpoint is outside the bound training run"
            ) from exc
        runtime = load_inference_runtime(
            checkpoint=checkpoint,
            expected_identity=cast(Mapping[str, object], identity),
            experiment=experiment,
            action_lower=lower,
            action_upper=upper,
        )
        head = getattr(getattr(runtime.policy, "model", None), "action_head", None)
        if not isinstance(head, BoundedActionRegressionHead):
            _fail("checkpoint", "bounded action head is missing")
        native_parts: list[np.ndarray[Any, Any]] = []
        canonical_parts: list[np.ndarray[Any, Any]] = []
        raw_parts: list[np.ndarray[Any, Any]] = []
        latencies: list[float] = []
        runtime.reset()
        for anchor_index in anchor_indices:
            sample = dataset[int(anchor_index)]
            started = time.perf_counter()
            native = runtime.predict_action_chunk_for_audit(
                sample[experiment.model.image_feature_key].permute(1, 2, 0).numpy(),
                sample[experiment.model.state_feature_key].numpy(),
            )
            latencies.append(time.perf_counter() - started)
            raw, canonical = head.latest_outputs()
            native_parts.append(native)
            raw_parts.append(
                raw.detach().to(torch.float32).cpu().numpy().reshape(-1, 8)
            )
            canonical_parts.append(
                canonical.detach().to(torch.float32).cpu().numpy().reshape(-1, 8)
            )
        native_values = np.concatenate(native_parts, axis=0)
        raw_values = np.concatenate(raw_parts, axis=0)
        canonical_values = np.concatenate(canonical_parts, axis=0)
        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        violations = np.logical_or(
            native_values < lower_array, native_values > upper_array
        )
        nonfinite = int(np.size(native_values) - np.isfinite(native_values).sum())
        digest = (
            "sha256:"
            + hashlib.sha256(
                np.asarray(native_values, dtype=np.float64).tobytes(order="C")
            ).hexdigest()
        )
        std = np.std(native_values, axis=0, dtype=np.float64)
        result = {
            "anchor_count": minimum,
            "canonical_maximum": np.max(canonical_values, axis=0).tolist(),
            "canonical_minimum": np.min(canonical_values, axis=0).tolist(),
            "checkpoint": checkpoint.name,
            "constant_output": bool(
                np.all(std <= float(config["constant_output_tolerance"]))
            ),
            "gripper_collapsed": bool(
                std[7] <= float(config["gripper_collapse_tolerance"])
            ),
            "native_boundary_violation_count": int(violations.sum()),
            "native_maximum": np.max(native_values, axis=0).tolist(),
            "native_minimum": np.min(native_values, axis=0).tolist(),
            "native_output_digest": digest,
            "native_standard_deviation": std.tolist(),
            "nonfinite_action_count": nonfinite,
            "query_latency_p95_seconds": float(np.quantile(latencies, 0.95)),
            "raw_maximum": np.max(raw_values, axis=0).tolist(),
            "raw_minimum": np.min(raw_values, axis=0).tolist(),
            "saturation_over_0_95_ratio": float(
                np.mean(np.abs(canonical_values) > 0.95)
            ),
            "valid_for_development": bool(
                not violations.any()
                and nonfinite == 0
                and not np.all(std <= float(config["constant_output_tolerance"]))
                and std[7] > float(config["gripper_collapse_tolerance"])
            ),
        }
        checkpoint_results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del runtime
        torch.cuda.empty_cache()
    report = {
        "anchor_count": minimum,
        "anchor_phase_counts": dict(sorted(phase_counts.items())),
        "checkpoint_count": len(checkpoint_results),
        "checkpoints": checkpoint_results,
        "contract_digest": contract.get("contract_digest"),
        "dataset_digest": identity.get("dataset_digest"),
        "all_checkpoints_valid": all(
            item["valid_for_development"] is True for item in checkpoint_results
        ),
        "passed": True,
        "schema_version": "pickcube-act-bounded-checkpoint-screen-report-v1",
        "source": dict(_git_identity()),
        "training_identity_digest": identity.get("training_identity_digest"),
        "valid_checkpoint_count": sum(
            item["valid_for_development"] is True for item in checkpoint_results
        ),
    }
    write_atomic_json(Path(output).absolute(), report)
    return report


class _IndexedDatasetView:
    """Expose selected sample identities solely for phase inventorying."""

    def __init__(self, dataset: PickCubeActDataset, indices: Sequence[int]) -> None:
        self.references = dataset.references
        self.samples = tuple(dataset.samples[index] for index in indices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pickcube_act_bounded/screen.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = screen(
        dataset_root=args.dataset_root,
        training_run=args.training_run,
        contract_path=args.contract,
        config_path=args.config,
        output=args.output,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
