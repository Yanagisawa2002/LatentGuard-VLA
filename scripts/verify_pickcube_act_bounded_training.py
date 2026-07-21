"""Verify bounded ACT tiny, smoke, or formal training evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import NoReturn, cast

from latentguard.control.serialization import write_atomic_json


class BoundedTrainingVerificationError(RuntimeError):
    """Raised when a P0.1 training gate is incomplete or unsafe."""


def _fail(context: str, reason: str) -> NoReturn:
    raise BoundedTrainingVerificationError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundedTrainingVerificationError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _metrics(path: Path) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = cast(object, json.loads(line))
        if not isinstance(value, Mapping):
            _fail("metrics", "record is not an object")
        records.append(cast(Mapping[str, object], value))
    if not records:
        _fail("metrics", "ledger is empty")
    return tuple(records)


def _strict_interior_bounds(record: Mapping[str, object]) -> bool:
    bounded = record.get("bounded_action")
    if not isinstance(bounded, Mapping):
        return False
    minimum = bounded.get("minimum")
    maximum = bounded.get("maximum")
    if (
        not isinstance(minimum, Sequence)
        or isinstance(minimum, (str, bytes))
        or not isinstance(maximum, Sequence)
        or isinstance(maximum, (str, bytes))
        or len(minimum) != 8
        or len(maximum) != 8
    ):
        return False
    try:
        lower_values = tuple(float(cast(int | float, value)) for value in minimum)
        upper_values = tuple(float(cast(int | float, value)) for value in maximum)
    except (TypeError, ValueError):
        return False
    return all(
        math.isfinite(lower) and math.isfinite(upper) and -1.0 < lower <= upper < 1.0
        for lower, upper in zip(lower_values, upper_values, strict=True)
    )


def verify(*, run_root: Path, stage: str, output: Path) -> Mapping[str, object]:
    """Recompute one training gate from immutable metrics and checkpoints."""
    root = Path(run_root).absolute()
    config = _mapping(root / "resolved_config.json", "resolved config")
    summary = _mapping(root / "training_summary.json", "training summary")
    transform = _mapping(root / "action_transform.json", "action transform")
    records = _metrics(root / "metrics.jsonl")
    expected_steps = {"smoke": 2, "tiny": 500, "formal": 20000}
    if stage not in expected_steps:
        _fail("stage", "expected smoke, tiny, or formal")
    if (
        config.get("schema_version") != "pickcube-native-act-bounded-experiment-v1"
        or transform.get("parameterization_type") != "affine_tanh_v1"
        or summary.get("final_step") != expected_steps[stage]
        or records[-1].get("step") != expected_steps[stage]
    ):
        _fail(stage, "identity or exact step count differs")
    if any(
        record.get("post_transform_boundary_violation_count") != 0
        or not _strict_interior_bounds(record)
        or not all(
            math.isfinite(float(cast(int | float, record[field])))
            for field in ("total_loss", "action_loss", "kl_loss", "gradient_norm")
        )
        for record in records
    ):
        _fail(
            stage,
            "nonfinite metric, non-interior bounded action, "
            "or post-transform violation",
        )
    checkpoints = tuple(sorted((root / "checkpoints").glob("step-*")))
    for checkpoint in checkpoints:
        manifest = _mapping(checkpoint / "checkpoint_manifest.json", "checkpoint")
        if (
            manifest.get("schema_version") != "pickcube_act_bounded_v1"
            or not (checkpoint / "pretrained_model" / "action_transform.json").is_file()
        ):
            _fail(stage, "checkpoint schema or transform asset differs")

    first_window = records[: min(20, len(records))]
    last_window = records[-min(20, len(records)) :]
    initial_action_loss = mean(
        float(cast(float, item["action_loss"])) for item in first_window
    )
    final_action_loss = mean(
        float(cast(float, item["action_loss"])) for item in last_window
    )
    final_per_dimension = records[-1].get("per_dimension_action_loss")
    if not isinstance(final_per_dimension, Sequence) or len(final_per_dimension) != 8:
        _fail(stage, "per-dimension loss is missing")
    tiny_loss_reduced = final_action_loss < initial_action_loss * 0.75
    final_saturation = records[-1].get("saturation")
    if not isinstance(final_saturation, Mapping):
        _fail(stage, "saturation report is missing")
    saturation_99 = float(
        cast(float, final_saturation["absolute_tanh_over_0_99_ratio"])
    )
    if stage == "tiny" and (not tiny_loss_reduced or saturation_99 >= 0.99):
        _fail("tiny", "loss did not fall materially or outputs collapsed to bounds")
    report = {
        "checkpoint_count": len(checkpoints),
        "final_action_loss_window_mean": final_action_loss,
        "final_per_dimension_action_loss": list(final_per_dimension),
        "final_saturation_over_0_99_ratio": saturation_99,
        "initial_action_loss_window_mean": initial_action_loss,
        "passed": True,
        "post_transform_boundary_violation_count": 0,
        "schema_version": "pickcube-act-bounded-training-verification-v1",
        "stage": stage,
        "step_count": len(records),
        "tiny_loss_materially_reduced": tiny_loss_reduced if stage == "tiny" else None,
        "training_identity": summary.get("training_identity"),
    }
    write_atomic_json(Path(output).absolute(), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "tiny", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(run_root=args.run_root, stage=args.stage, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
