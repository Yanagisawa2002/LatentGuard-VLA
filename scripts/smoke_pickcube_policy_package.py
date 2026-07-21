"""Controlled simulator smoke for a provisional native ACT PolicyPackage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import torch
from numpy.typing import NDArray

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
from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
    PickCubeEpisodeEndedError,
    RecordingEnvironmentProxy,
)
from latentguard.integrations.maniskill_pickcube.visual_rendering import (
    LazyManiSkillPickCubeVisualRenderer,
    PickCubeVisualRenderPlan,
    build_pickcube_visual_render_plan,
)
from latentguard.policies import PolicyCompatibilityStatus, PolicyRegistry
from latentguard.policies.act.contracts import PICKCUBE_ACTIVE_JOINT_NAMES
from latentguard.policies.act.runtime import (
    PickCubeActInferenceRuntime,
    load_packaged_inference_runtime,
)
from latentguard.policies.act.types import PickCubeActExperimentConfig
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.configuration import (
    load_camera_rig_configuration,
    load_render_domain_configuration,
)
from latentguard.vision_data.domains import RenderDomainConfigurationV1

_SAPIEN_VULKAN_FALLBACK_WARNING = (
    r"^Failed to find Vulkan ICD file\. This is probably due to an incorrect or "
    r"partial installation of the NVIDIA driver\. SAPIEN will attempt to provide "
    r"an ICD file anyway but it may not work\.$"
)


class PickCubeActPackageSmokeError(RuntimeError):
    """Raised when a provisional PolicyPackage fails controlled execution."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActPackageSmokeError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeActPackageSmokeError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _path(value: Mapping[str, object], field: str) -> Path:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw:
        _fail("environment config", f"{field} is missing")
    return Path(raw)


def _source_identity() -> Mapping[str, str]:
    repository = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PickCubeActPackageSmokeError(
                "unable to inspect smoke source"
            ) from exc
        return result.stdout.strip()

    if git("status", "--short", "--untracked-files=no"):
        _fail("smoke source", "tracked worktree is not clean")
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
    }


def _chunk_digest(chunk: NDArray[Any]) -> str:
    value = np.asarray(chunk)
    digest = hashlib.sha256(b"pickcube-native-act-package-smoke-v1\0")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _execute_once(
    *,
    runtime: PickCubeActInferenceRuntime,
    seed: int,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    render_plan: PickCubeVisualRenderPlan,
) -> Mapping[str, object]:
    factory = LazyManiSkillSourceEnvironmentFactory()
    renderer = LazyManiSkillPickCubeVisualRenderer()
    environment: object | None = None
    recorder: RecordingEnvironmentProxy | None = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_SAPIEN_VULKAN_FALLBACK_WARNING,
                category=UserWarning,
                module=r"^sapien\._vulkan_tricks$",
            )
            environment = factory.create_environment(
                settings,
                action_contract,
                purpose="official_source_generation",
            )
        recorder = RecordingEnvironmentProxy(
            environment,
            action_contract,
            trajectory_action_limit=50,
            stop_on_episode_end=True,
        )
        recorder.reset(seed=seed)
        handle = renderer.prepare(environment, render_plan)
        views = handle.render_views()
        by_id = {view.camera_id: view for view in views}
        if set(by_id) != {"front_oblique", "overhead", "side_oblique"}:
            _fail("package smoke", "camera inventory changed")
        names, state = factory.extract_named_robot_state(environment)
        if names != PICKCUBE_ACTIVE_JOINT_NAMES:
            _fail("package smoke", "proprioception order changed")
        runtime.reset()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        chunk = runtime.predict_action_chunk(
            np.asarray(by_id["front_oblique"].rgb), np.asarray(state)
        )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        executed = 0
        for action in chunk[:4]:
            try:
                recorder.step(action)
                executed += 1
            except PickCubeEpisodeEndedError:
                executed += 1
                break
        if executed < 1:
            _fail("package smoke", "no simulator action executed")
        return {
            "action_chunk_digest": _chunk_digest(chunk),
            "action_chunk_shape": list(chunk.shape),
            "executed_action_count": executed,
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "policy_query_latency_seconds": latency,
            "seed": seed,
        }
    finally:
        if recorder is not None:
            recorder.close()
        elif environment is not None:
            close = getattr(environment, "close", None)
            if callable(close):
                close()


def main() -> int:
    """Verify a provisional package and execute two exact fixed-seed smokes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--render-domain-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=910000)
    args = parser.parse_args()

    registry = PolicyRegistry.from_mapping(_mapping(args.registry, "registry"))
    matches = [entry for entry in registry.entries if entry.policy_id == args.policy_id]
    if len(matches) != 1:
        _fail("registry", "provisional package is missing or ambiguous")
    entry = matches[0]
    if (
        entry.compatibility_status is not PolicyCompatibilityStatus.UNVERIFIED
        or entry.missing_assets != ("controlled_package_smoke",)
    ):
        _fail("registry", "expected controlled-smoke-only provisional status")
    package = entry.package
    if package is None:
        _fail("registry", "provisional package is missing")
    package.verify_artifacts(args.package_root)

    environment_config = _mapping(args.environment_config, "environment config")
    expected = load_expected_contract(_path(environment_config, "expected_contract"))
    report = load_compatibility_report(args.compatibility_report)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(
        _path(environment_config, "action_layout")
    )
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding, coordinate_frame=layout.coordinate_frame
    )
    rig = load_camera_rig_configuration(_path(environment_config, "camera_rig"))
    domains = load_render_domain_configuration(args.render_domain_config)
    if not isinstance(rig, PickCubeMultiViewRigV1) or not isinstance(
        domains, RenderDomainConfigurationV1
    ):
        _fail("visual contract", "unexpected camera or domain configuration")
    render_plan = build_pickcube_visual_render_plan(
        rig, domains.domain("canonical"), domains, 0
    )

    experiment = PickCubeActExperimentConfig.from_mapping(
        _mapping(
            Path(args.package_root) / "resolved_training_config.json",
            "resolved training config",
        )
    )
    bounds = package.action_spec.get("bounds")
    if not isinstance(bounds, Mapping):
        _fail("action contract", "bounds are missing")
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    if not isinstance(lower, Sequence) or not isinstance(upper, Sequence):
        _fail("action contract", "bounds are malformed")
    runtime = load_packaged_inference_runtime(
        package=package,
        package_root=args.package_root,
        experiment=experiment,
        action_lower=cast(Sequence[float], lower),
        action_upper=cast(Sequence[float], upper),
        observation_spec=registry.required_observation_spec,
        action_spec=registry.required_action_spec,
        environment_spec=registry.required_environment_spec,
    )
    first = _execute_once(
        runtime=runtime,
        seed=args.seed,
        settings=settings,
        action_contract=action_contract,
        render_plan=render_plan,
    )
    second = _execute_once(
        runtime=runtime,
        seed=args.seed,
        settings=settings,
        action_contract=action_contract,
        render_plan=render_plan,
    )
    reproducible = (
        first["action_chunk_digest"] == second["action_chunk_digest"]
        and first["executed_action_count"] == second["executed_action_count"]
    )
    audit = {
        "executions": [dict(first), dict(second)],
        "fixed_seed_reproducible": reproducible,
        "package_digest": package.package_digest,
        "passed": reproducible,
        "policy_id": package.policy_id,
        "runtime_asset_digest": package.acceptance_evidence["runtime_asset_digest"],
        "schema_version": "pickcube-native-act-package-smoke-v1",
        "smoke_source": dict(_source_identity()),
    }
    write_atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if reproducible else 2


if __name__ == "__main__":
    raise SystemExit(main())
