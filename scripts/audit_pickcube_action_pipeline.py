"""Audit the frozen PickCube demonstration action distribution for P0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.data import collect_episode_references
from latentguard.policies.actions import (
    BoundedActionTransform,
    action_bounds_from_contract,
)
from latentguard.replay.identity import canonical_json_bytes


class PickCubeActionAuditError(RuntimeError):
    """Raised when immutable action data or its declared bounds differ."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActionAuditError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeActionAuditError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _digest(value: Mapping[str, object], context: str) -> str:
    body = dict(value)
    body.pop("dataset_digest", None)
    return (
        "sha256:"
        + hashlib.sha256(canonical_json_bytes(body, context=context)).hexdigest()
    )


def _stats(
    values: NDArray[np.float64], quantiles: Sequence[float]
) -> Mapping[str, object]:
    return {
        "exact_zero_ratio": np.mean(values == 0.0, axis=0).tolist(),
        "maximum": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "minimum": np.min(values, axis=0).tolist(),
        "quantiles": {
            format(level, ".6g"): np.quantile(values, level, axis=0).tolist()
            for level in quantiles
        },
        "standard_deviation": np.std(values, axis=0, dtype=np.float64).tolist(),
    }


def audit(
    *, dataset_root: Path, contract_path: Path, config_path: Path, output: Path
) -> Mapping[str, object]:
    """Recompute full-corpus native/canonical targets without loading images."""
    config = _mapping(config_path, "audit config")
    if config.get("schema_version") != "pickcube-act-bounded-audit-config-v1":
        _fail("audit config", "schema mismatch")
    raw_quantiles = config.get("quantiles")
    if not isinstance(raw_quantiles, Sequence) or isinstance(
        raw_quantiles, (str, bytes)
    ):
        _fail("audit config", "quantiles are malformed")
    quantiles = tuple(float(item) for item in raw_quantiles)
    if tuple(sorted(quantiles)) != quantiles or not all(
        0.0 <= item <= 1.0 for item in quantiles
    ):
        _fail("audit config", "quantiles must be sorted probabilities")

    root = Path(dataset_root).absolute()
    manifest = _mapping(root / "dataset_manifest.json", "dataset manifest")
    expected_digest = config.get("expected_dataset_digest")
    observed_digest = manifest.get("dataset_digest")
    if (
        observed_digest != expected_digest
        or _digest(manifest, "PickCubeActDatasetManifest") != observed_digest
    ):
        _fail("dataset manifest", "dataset digest differs")
    references = collect_episode_references(root)
    expected_episodes = config.get("expected_episode_count")
    expected_frames = config.get("expected_frame_count")
    if (
        len(references) != expected_episodes
        or sum(item.frame_count for item in references) != expected_frames
    ):
        _fail("dataset", "episode or frame count differs")

    contract = _mapping(contract_path, "ACT contract")
    action = contract.get("action")
    if not isinstance(action, Mapping):
        _fail("ACT contract", "action section is missing")
    transform = BoundedActionTransform(action_bounds_from_contract(action))
    native_parts: list[NDArray[np.float64]] = []
    phase_parts: dict[str, list[NDArray[np.float64]]] = defaultdict(list)
    split_parts: dict[str, list[NDArray[np.float64]]] = defaultdict(list)
    phase_counts: Counter[str] = Counter()
    for reference in references:
        directory = root.joinpath(*PurePosixPath(reference.relative_directory).parts)
        actions = np.asarray(
            np.load(directory / "actions.npy", allow_pickle=False, mmap_mode="r"),
            dtype=np.float64,
        )
        metadata = _mapping(directory / "episode.json", "episode metadata")
        phases = metadata.get("phases")
        if (
            not isinstance(phases, Sequence)
            or isinstance(phases, (str, bytes))
            or len(phases) != len(actions)
        ):
            _fail("episode metadata", "action/phase alignment differs")
        native_parts.append(actions)
        split_parts[reference.split.value].append(actions)
        for phase in sorted(set(cast(Sequence[str], phases))):
            indices = np.fromiter((value == phase for value in phases), dtype=np.bool_)
            phase_parts[phase].append(actions[indices])
            phase_counts[phase] += int(indices.sum())
    native = np.concatenate(native_parts, axis=0)
    canonical = transform.normalize_target(torch.from_numpy(native)).numpy()
    lower = transform.lower.numpy()
    upper = transform.upper.numpy()
    lower_hits = np.sum(native == lower, axis=0)
    upper_hits = np.sum(native == upper, axis=0)
    violations = np.logical_or(native < lower, native > upper)
    result: dict[str, object] = {
        "action_contract_digest": contract.get("contract_digest"),
        "canonical_target_distribution": _stats(canonical, quantiles),
        "dataset_digest": observed_digest,
        "dimension_names": list(transform.names),
        "episode_count": len(references),
        "exact_boundary_counts": {
            "lower": lower_hits.tolist(),
            "upper": upper_hits.tolist(),
        },
        "exact_boundary_ratios": {
            "lower": (lower_hits / len(native)).tolist(),
            "upper": (upper_hits / len(native)).tolist(),
        },
        "frame_count": len(native),
        "native_boundary_violation_count": int(violations.sum()),
        "native_target_distribution": _stats(native, quantiles),
        "native_target_distance_to_lower": _stats(native - lower, quantiles),
        "native_target_distance_to_upper": _stats(upper - native, quantiles),
        "phase_frame_counts": dict(sorted(phase_counts.items())),
        "phase_native_distributions": {
            phase: _stats(np.concatenate(parts, axis=0), quantiles)
            for phase, parts in sorted(phase_parts.items())
        },
        "schema_version": "pickcube-act-target-distribution-v1",
        "split_native_distributions": {
            split: _stats(np.concatenate(parts, axis=0), quantiles)
            for split, parts in sorted(split_parts.items())
        },
        "transform": transform.to_mapping(),
    }
    result["audit_digest"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(result, context="PickCubeActTargetDistributionV1")
        ).hexdigest()
    )
    write_atomic_json(Path(output).absolute(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pickcube_act_bounded/audit.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                dataset_root=args.dataset_root,
                contract_path=args.contract,
                config_path=args.config,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
