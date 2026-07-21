"""Measure native ACT checkpoint diversity on real content-bound anchors."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

import numpy as np
import torch

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.data import (
    PickCubeDemoSplit,
    collect_episode_references,
    read_demo_episode,
)
from latentguard.policies.act.diversity import summarize_checkpoint_diversity
from latentguard.policies.act.runtime import load_inference_runtime
from latentguard.policies.act.types import PickCubeActExperimentConfig


class PickCubeActDiversityCliError(RuntimeError):
    """Raised when diversity inputs are incomplete or path-unsafe."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActDiversityCliError(f"{context}: {reason}")


def _mapping(path: Path, *, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeActDiversityCliError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _integer(value: Mapping[str, object], field: str) -> int:
    raw = value.get(field)
    if type(raw) is not int or raw < 1:
        _fail("config", f"{field} must be a positive integer")
    return raw


def _number(value: Mapping[str, object], field: str) -> float:
    raw = value.get(field)
    if type(raw) not in (int, float):
        _fail("config", f"{field} must be numeric")
    return float(cast(int | float, raw))


def _role_arguments(values: Sequence[str]) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if value.count("=") != 1:
            _fail("role", "expected NAME=checkpoint-relative-path")
        name, path = value.split("=", 1)
        if (
            not name
            or name != name.strip()
            or name in result
            or not path
            or "\\" in path
        ):
            _fail("role", "name/path is invalid or duplicate")
        relative = PurePosixPath(path)
        if relative.is_absolute() or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            _fail("role", "checkpoint path must be normalized and relative")
        result[name] = relative.as_posix()
    return result


def evaluate_diversity(
    *,
    config_path: Path,
    dataset_root: Path,
    contract_path: Path,
    training_run: Path,
    output: Path,
    additional_roles: Sequence[str] = (),
) -> Mapping[str, object]:
    """Load each exact checkpoint once and compare its chunks over 20 anchors."""
    config = _mapping(config_path, context="diversity config")
    if config.get("schema_version") != "pickcube-native-act-diversity-config-v1":
        _fail("diversity config", "schema mismatch")
    if (
        config.get("anchor_split") != "validation"
        or config.get("anchor_frame_semantic") != "episode_midpoint_v1"
    ):
        _fail("diversity config", "anchor selection contract changed")
    anchor_count = _integer(config, "anchor_count")
    if anchor_count != 20:
        _fail("diversity config", "requires exactly 20 anchors")

    run_root = Path(training_run).absolute()
    experiment = PickCubeActExperimentConfig.from_mapping(
        _mapping(run_root / "resolved_config.json", context="resolved config")
    )
    run_manifest = _mapping(run_root / "run_manifest.json", context="run manifest")
    identity = run_manifest.get("training_identity")
    if not isinstance(identity, Mapping):
        _fail("run manifest", "training identity is missing")
    roles_value = _mapping(
        run_root / "checkpoint_roles.json", context="checkpoint roles"
    )
    required_roles = {"early", "mid", "best_validation", "final"}
    if set(roles_value) != required_roles:
        _fail("checkpoint roles", "early/mid/best/final inventory changed")
    roles: dict[str, str] = {}
    for name, value in roles_value.items():
        if not isinstance(value, str):
            _fail("checkpoint roles", "path must be text")
        roles[name] = value
    roles.update(_role_arguments(additional_roles))

    contract = _mapping(contract_path, context="ACT contract")
    action = contract.get("action")
    if not isinstance(action, Mapping):
        _fail("ACT contract", "action section is missing")
    bounds = action.get("bounds")
    if not isinstance(bounds, Mapping):
        _fail("ACT contract", "action bounds are missing")
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    if not isinstance(lower, Sequence) or not isinstance(upper, Sequence):
        _fail("ACT contract", "action bounds are malformed")

    references = tuple(
        reference
        for reference in collect_episode_references(dataset_root)
        if reference.split is PickCubeDemoSplit.VALIDATION
    )
    if len(references) < anchor_count:
        _fail("anchors", "validation split lacks required episodes")
    anchors = []
    for reference in references[:anchor_count]:
        episode = read_demo_episode(dataset_root, reference)
        frame_index = episode.frame_count // 2
        anchors.append(
            (
                f"{reference.episode_id}:frame-{frame_index}",
                episode.rgb[frame_index],
                episode.proprioception[frame_index],
            )
        )

    chunks_by_role = {}
    checkpoint_paths: dict[str, str] = {}
    contract_violations: dict[str, int] = {}
    maximum_bound_exceedance: dict[str, float] = {}
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    for role in sorted(roles):
        relative = PurePosixPath(roles[role])
        checkpoint = run_root.joinpath(*relative.parts).absolute()
        try:
            checkpoint.relative_to(run_root / "checkpoints")
        except ValueError as exc:
            raise PickCubeActDiversityCliError(
                f"role {role}: checkpoint escapes training run"
            ) from exc
        runtime = load_inference_runtime(
            checkpoint=checkpoint,
            expected_identity=identity,
            experiment=experiment,
            action_lower=cast(Sequence[float], lower),
            action_upper=cast(Sequence[float], upper),
        )
        role_chunks = []
        for _, rgb, state in anchors:
            runtime.reset()
            role_chunks.append(runtime.predict_action_chunk_for_audit(rgb, state))
        chunks_by_role[role] = role_chunks
        contract_violations[role] = sum(
            bool(np.any(chunk < lower_array) or np.any(chunk > upper_array))
            for chunk in role_chunks
        )
        maximum_bound_exceedance[role] = max(
            float(
                max(
                    np.max(np.maximum(lower_array - chunk, 0.0)),
                    np.max(np.maximum(chunk - upper_array, 0.0)),
                )
            )
            for chunk in role_chunks
        )
        checkpoint_paths[role] = checkpoint.relative_to(run_root).as_posix()
        del runtime
        torch.cuda.empty_cache()

    summary = dict(
        summarize_checkpoint_diversity(
            chunks_by_role=chunks_by_role,
            anchor_ids=[anchor[0] for anchor in anchors],
            chunk_l2_threshold=_number(config, "chunk_l2_distinct_threshold"),
            first_action_l2_threshold=_number(
                config, "first_action_l2_distinct_threshold"
            ),
            minimum_distinct_ratio=_number(
                config, "minimum_meaningfully_distinct_anchor_ratio"
            ),
        )
    )
    summary.update(
        {
            "action_contract_violation_anchor_count_by_role": contract_violations,
            "anchor_frame_semantic": config["anchor_frame_semantic"],
            "anchor_split": config["anchor_split"],
            "checkpoint_paths": checkpoint_paths,
            "contract_valid_roles": sorted(
                role for role, count in contract_violations.items() if count == 0
            ),
            "contract_digest": contract.get("contract_digest"),
            "maximum_action_bound_exceedance_by_role": maximum_bound_exceedance,
            "training_identity": dict(identity),
        }
    )
    valid_roles = set(cast(Sequence[str], summary["contract_valid_roles"]))
    summary["at_least_two_contract_valid_roles"] = len(valid_roles) >= 2
    summary["meaningfully_distinct_contract_valid_pair_exists"] = any(
        pair["left_role"] in valid_roles
        and pair["right_role"] in valid_roles
        and pair["passes_minimum_distinct_ratio"] is True
        for pair in cast(Sequence[Mapping[str, object]], summary["pair_comparisons"])
    )
    write_atomic_json(output, summary)
    return summary


def main() -> int:
    """Expose exact offline diversity analysis for a completed training run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", action="append", default=[])
    args = parser.parse_args()
    result = evaluate_diversity(
        config_path=args.config,
        dataset_root=args.dataset_root,
        contract_path=args.contract,
        training_run=args.training_run,
        output=args.output,
        additional_roles=args.role,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
