"""Collect resumable, exact, successful native PickCube ACT demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn, cast

from latentguard.control.serialization import write_atomic_json
from latentguard.integrations.maniskill_pickcube.compatibility import (
    load_compatibility_report,
    validate_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
)
from latentguard.integrations.maniskill_pickcube.serialization import (
    action_contract_from_compatibility,
    environment_settings_from_compatibility,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
)
from latentguard.integrations.maniskill_pickcube.visual_rendering import (
    LazyManiSkillPickCubeVisualRenderer,
    build_pickcube_visual_render_plan,
)
from latentguard.policies.act.collection import (
    collect_native_demo_episode,
    expert_gate_summary,
)
from latentguard.policies.act.data import (
    PickCubeDemoSplit,
    build_demo_dataset_reports,
    collect_episode_references,
    read_demo_episode,
    save_demo_episode,
    split_for_scene_seed,
)
from latentguard.policies.experts import (
    PickCubeMotionPlanningExpert,
    assert_pickcube_expert_has_no_teleport_calls,
)
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.configuration import (
    load_camera_rig_configuration,
    load_render_domain_configuration,
)
from latentguard.vision_data.domains import RenderDomainConfigurationV1


class PickCubeDemoCollectionCliError(RuntimeError):
    """Raised when collection configuration or progress is invalid."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeDemoCollectionCliError(f"{context}: {reason}")


def _mapping(path: Path, *, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeDemoCollectionCliError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _path(item: Mapping[str, object], field: str) -> Path:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        _fail("collection config", f"{field} must be a non-empty path")
    return Path(value)


def _integer(item: Mapping[str, object], field: str, *, minimum: int = 0) -> int:
    value = item.get(field)
    if type(value) is not int or value < minimum:
        _fail("collection config", f"{field} must be an integer >= {minimum}")
    return value


def _number(item: Mapping[str, object], field: str) -> float:
    value = item.get(field)
    if type(value) not in (int, float):
        _fail("collection config", f"{field} must be numeric")
    result = float(cast(int | float, value))
    if result <= 0.0:
        _fail("collection config", f"{field} must be positive")
    return result


def _split_targets(config: Mapping[str, object]) -> Mapping[PickCubeDemoSplit, int]:
    raw = config.get("split_success_targets")
    if not isinstance(raw, Mapping) or set(raw) != {
        "train",
        "validation",
        "test",
    }:
        _fail("collection config", "split targets must define train/validation/test")
    targets: dict[PickCubeDemoSplit, int] = {}
    for split in PickCubeDemoSplit:
        value = raw.get(split.value)
        if type(value) is not int or value < 1:
            _fail("collection config", f"{split.value} target must be positive")
        targets[split] = value
    if targets != {
        PickCubeDemoSplit.TRAIN: 400,
        PickCubeDemoSplit.VALIDATION: 50,
        PickCubeDemoSplit.TEST: 50,
    }:
        _fail("collection config", "P0 requires an exact 400/50/50 split")
    return targets


def _attempt_path(root: Path, seed: int) -> Path:
    return root / "attempts" / f"seed-{seed:010d}.json"


def _existing_attempts(root: Path) -> Mapping[int, Mapping[str, object]]:
    directory = root / "attempts"
    if not directory.exists():
        return {}
    if not directory.is_dir() or directory.is_symlink():
        _fail("attempt ledger", "expected unlinked directory")
    attempts: dict[int, Mapping[str, object]] = {}
    for path in sorted(directory.glob("seed-*.json")):
        value = _mapping(path, context="attempt record")
        seed = value.get("scene_seed")
        if type(seed) is not int or _attempt_path(root, seed) != path.absolute():
            _fail("attempt ledger", "seed path binding differs")
        if seed in attempts:
            _fail("attempt ledger", "duplicate seed")
        attempts[seed] = value
    return attempts


def _write_attempt(
    root: Path,
    *,
    seed: int,
    split: PickCubeDemoSplit,
    success: bool,
    episode_id: str | None,
    frame_count: int,
    failure_category: str | None,
) -> None:
    path = _attempt_path(root, seed)
    if path.exists() or path.is_symlink():
        _fail("attempt ledger", "attempt publication is not idempotent")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic_json(
        path,
        {
            "episode_id": episode_id,
            "failure_category": failure_category,
            "frame_count": frame_count,
            "scene_seed": seed,
            "schema_version": "pickcube-act-demo-attempt-v1",
            "split": split.value,
            "success": success,
        },
    )


def _progress(
    *,
    started: float,
    attempts: int,
    counts: Counter[PickCubeDemoSplit],
    targets: Mapping[PickCubeDemoSplit, int],
    seed: int,
) -> None:
    elapsed = max(time.monotonic() - started, 0.0)
    successes = sum(counts.values())
    target = sum(targets.values())
    remaining = target - successes
    rate = successes / elapsed if elapsed > 0.0 else 0.0
    eta = remaining / rate if rate > 0.0 else None
    print(
        json.dumps(
            {
                "attempt_count": attempts,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "latest_seed": seed,
                "split_success_counts": {
                    split.value: counts[split] for split in PickCubeDemoSplit
                },
                "successful_episode_count": successes,
                "target_episode_count": target,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def collect_dataset(
    *,
    config_path: Path,
    compatibility_path: Path,
    expert_gate_path: Path,
    contract_path: Path,
    output_root: Path,
    smoke_seed: int | None = None,
) -> Mapping[str, object]:
    """Resume the fixed 500-success collection and write strict reports."""
    config = _mapping(config_path, context="collection config")
    if config.get("schema_version") != "pickcube-act-demo-collection-config-v1":
        _fail("collection config", "schema mismatch")
    targets = _split_targets(config)
    seed_start = _integer(config, "seed_start")
    maximum_attempts = _integer(config, "maximum_attempts", minimum=500)
    environment_config = _mapping(
        _path(config, "environment_config"), context="environment config"
    )
    expert_gate = _mapping(expert_gate_path, context="expert gate")
    expert_summary = expert_gate_summary(expert_gate)
    contract = _mapping(contract_path, context="ACT contract")
    contract_digest = contract.get("contract_digest")
    if not isinstance(contract_digest, str):
        _fail("ACT contract", "contract digest is missing")
    contract_environment = contract.get("environment")
    if not isinstance(contract_environment, Mapping):
        _fail("ACT contract", "environment binding is missing")

    expected = load_expected_contract(_path(environment_config, "expected_contract"))
    report = load_compatibility_report(compatibility_path)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    if contract_environment.get("compatibility_identity") != (
        report.compatibility_identity
    ):
        _fail("ACT contract", "compatibility identity differs")
    layout = load_maniskill_pickcube_action_layout(
        _path(environment_config, "action_layout")
    )
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding, coordinate_frame=layout.coordinate_frame
    )
    key_contract = PickCubeTaskKeyContract.from_compatibility_report(report)
    rig = load_camera_rig_configuration(_path(environment_config, "camera_rig"))
    domains = load_render_domain_configuration(_path(config, "render_domain_config"))
    if not isinstance(rig, PickCubeMultiViewRigV1) or not isinstance(
        domains, RenderDomainConfigurationV1
    ):
        _fail("visual config", "loader returned an unexpected contract")
    domain_id = config.get("render_domain_id")
    if domain_id != "canonical":
        _fail("visual config", "P0 collection requires the canonical domain")
    render_seed = _integer(config, "render_seed")
    render_plan = build_pickcube_visual_render_plan(
        rig,
        domains.domain(cast(str, domain_id)),
        domains,
        render_seed,
    )
    assert_pickcube_expert_has_no_teleport_calls()
    expert = PickCubeMotionPlanningExpert(
        joint_velocity_scale=_number(config, "joint_velocity_scale"),
        joint_acceleration_scale=_number(config, "joint_acceleration_scale"),
        transport_refine_steps=_integer(
            config,
            "transport_refine_steps",
            minimum=1,
        ),
    )

    root = Path(output_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    if smoke_seed is not None:
        if type(smoke_seed) is not int or not 0 <= smoke_seed < 2**32:
            _fail("smoke", "seed must be uint32")
        if split_for_scene_seed(smoke_seed) is not PickCubeDemoSplit.TRAIN:
            _fail("smoke", "seed must belong to the training split")
        if (root / "episodes").exists():
            _fail("smoke", "output root must not contain existing episodes")
        episode = collect_native_demo_episode(
            seed=smoke_seed,
            settings=settings,
            action_contract=action_contract,
            key_contract=key_contract,
            render_plan=render_plan,
            compatibility_identity=report.compatibility_identity,
            contract_digest=contract_digest,
            expert=expert,
        )
        reference = save_demo_episode(root, episode)
        reloaded = read_demo_episode(root, reference)
        manifest, quality, normalization = build_demo_dataset_reports(
            root,
            contract_digest=contract_digest,
            expert_success_rate=float(cast(float, expert_summary["success_rate"])),
        )
        write_atomic_json(root / "dataset_manifest.json", dict(manifest))
        write_atomic_json(root / "data_quality_report.json", dict(quality))
        write_atomic_json(root / "normalization_stats.json", dict(normalization))
        summary = {
            "contract_digest": contract_digest,
            "dataset_digest": manifest["dataset_digest"],
            "episode_id": reloaded.episode_id,
            "frame_count": reloaded.frame_count,
            "scene_seed": smoke_seed,
            "schema_version": "pickcube-act-demo-smoke-v1",
            "state_preserving_capture": True,
        }
        write_atomic_json(root / "smoke_summary.json", summary)
        return summary
    references = (
        collect_episode_references(root) if (root / "episodes").exists() else ()
    )
    counts: Counter[PickCubeDemoSplit] = Counter(
        reference.split for reference in references
    )
    for split, count in counts.items():
        if count > targets[split]:
            _fail("dataset resume", f"{split.value} exceeds its frozen target")
    attempts = dict(_existing_attempts(root))
    reference_by_seed = {reference.scene_seed: reference for reference in references}
    if set(reference_by_seed) - set(attempts):
        _fail("dataset resume", "accepted episode is absent from attempt ledger")
    if any(
        value.get("success") is True and seed not in reference_by_seed
        for seed, value in attempts.items()
    ):
        _fail("dataset resume", "successful attempt lacks an episode directory")
    if len(attempts) > maximum_attempts:
        _fail("dataset resume", "attempt budget was exceeded")

    factory = LazyManiSkillSourceEnvironmentFactory()
    renderer = LazyManiSkillPickCubeVisualRenderer()
    started = time.monotonic()
    seed = seed_start
    while counts != Counter(targets):
        if len(attempts) >= maximum_attempts:
            _fail("collection", "maximum attempt budget exhausted")
        split = split_for_scene_seed(seed)
        if seed in attempts or counts[split] >= targets[split]:
            seed += 1
            continue
        try:
            episode = collect_native_demo_episode(
                seed=seed,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
                render_plan=render_plan,
                compatibility_identity=report.compatibility_identity,
                contract_digest=contract_digest,
                expert=expert,
                factory=factory,
                renderer=renderer,
            )
            reference = save_demo_episode(root, episode)
            _write_attempt(
                root,
                seed=seed,
                split=split,
                success=True,
                episode_id=reference.episode_id,
                frame_count=reference.frame_count,
                failure_category=None,
            )
            references = (*references, reference)
            counts[split] += 1
        except Exception as exc:
            category = f"{type(exc).__module__}.{type(exc).__name__}"
            _write_attempt(
                root,
                seed=seed,
                split=split,
                success=False,
                episode_id=None,
                frame_count=0,
                failure_category=category,
            )
        attempts = dict(_existing_attempts(root))
        _progress(
            started=started,
            attempts=len(attempts),
            counts=counts,
            targets=targets,
            seed=seed,
        )
        seed += 1

    manifest, quality, normalization = build_demo_dataset_reports(
        root,
        contract_digest=contract_digest,
        expert_success_rate=float(cast(float, expert_summary["success_rate"])),
    )
    if manifest.get("episode_count") != 500:
        _fail("collection", "reloaded dataset does not contain exactly 500 episodes")
    write_atomic_json(root / "dataset_manifest.json", dict(manifest))
    write_atomic_json(root / "data_quality_report.json", dict(quality))
    write_atomic_json(root / "normalization_stats.json", dict(normalization))
    failure_counts = Counter(
        cast(str, item.get("failure_category"))
        for item in attempts.values()
        if item.get("success") is False
    )
    summary: dict[str, object] = {
        "attempt_count": len(attempts),
        "contract_digest": contract_digest,
        "dataset_digest": manifest["dataset_digest"],
        "episode_count": manifest["episode_count"],
        "expert_gate": dict(expert_summary),
        "failure_taxonomy": dict(sorted(failure_counts.items())),
        "frame_count": manifest["frame_count"],
        "schema_version": "pickcube-act-demo-collection-summary-v1",
        "seed_start": seed_start,
        "split_counts": manifest["split_episode_counts"],
    }
    write_atomic_json(root / "collection_summary.json", summary)
    return summary


def main() -> int:
    """Validate or execute the fixed P0 native demonstration collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path)
    parser.add_argument("--expert-gate", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-seed", type=int)
    args = parser.parse_args()
    compatibility = args.compatibility_report
    if compatibility is None:
        value = os.environ.get("LATENTGUARD_PICKCUBE_COMPATIBILITY_REPORT")
        if not value:
            parser.error("pass --compatibility-report or set the environment variable")
        compatibility = Path(value)
    result = collect_dataset(
        config_path=args.config,
        compatibility_path=compatibility,
        expert_gate_path=args.expert_gate,
        contract_path=args.contract,
        output_root=args.output_root,
        smoke_seed=args.smoke_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
