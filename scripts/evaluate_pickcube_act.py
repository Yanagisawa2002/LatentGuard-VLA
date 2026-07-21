"""Evaluate one frozen native ACT checkpoint in independent PickCube episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
    build_pickcube_task_evidence,
)
from latentguard.integrations.maniskill_pickcube.visual_rendering import (
    LazyManiSkillPickCubeVisualRenderer,
    PickCubeVisualRenderPlan,
    build_pickcube_visual_render_plan,
)
from latentguard.policies.act.contracts import PICKCUBE_ACTIVE_JOINT_NAMES
from latentguard.policies.act.evaluation import (
    PickCubeActEpisodeEvaluation,
    classify_final_checkpoint,
    summarize_checkpoint_evaluation,
)
from latentguard.policies.act.runtime import (
    PickCubeActInferenceRuntime,
    PickCubeActRuntimeError,
    load_inference_runtime,
)
from latentguard.policies.act.types import PickCubeActExperimentConfig
from latentguard.replay.models import TerminalTaskEvidence, TerminalTaskStatus
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


class PickCubeActEvaluationCliError(RuntimeError):
    """Raised when evaluation input or runtime evidence fails closed."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActEvaluationCliError(f"{context}: {reason}")


def _mapping(path: Path, *, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeActEvaluationCliError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _path(value: Mapping[str, object], field: str) -> Path:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        _fail("config", f"{field} must be a path")
    return Path(raw)


def _integer(value: Mapping[str, object], field: str, *, minimum: int = 0) -> int:
    raw = value.get(field)
    if type(raw) is not int or raw < minimum:
        _fail("config", f"{field} must be an integer >= {minimum}")
    return raw


def _number(value: Mapping[str, object], field: str) -> float:
    raw = value.get(field)
    if type(raw) not in (int, float) or not math.isfinite(
        float(cast(int | float, raw))
    ):
        _fail("config", f"{field} must be finite")
    return float(cast(int | float, raw))


def _evaluator_source_identity() -> Mapping[str, str]:
    """Bind evaluation evidence to the exact clean tracked source revision."""
    repository = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PickCubeActEvaluationCliError(
                "evaluator source: unable to inspect Git identity"
            ) from exc
        return completed.stdout.strip()

    if git("status", "--short", "--untracked-files=no"):
        _fail("evaluator source", "tracked worktree is not clean")
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if len(commit) != 40 or not branch:
        _fail("evaluator source", "Git branch or full commit is unavailable")
    return {"branch": branch, "commit": commit}


def _checkpoint_digest(checkpoint: Path) -> str:
    manifest = checkpoint / "checkpoint_manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        _fail("checkpoint", "manifest is missing")
    return f"sha256:{hashlib.sha256(manifest.read_bytes()).hexdigest()}"


def _action_digest(actions: Sequence[NDArray[Any]]) -> str:
    if actions:
        array = np.stack(actions, axis=0).astype(np.float64, copy=False)
    else:
        array = np.empty((0, 8), dtype=np.float64)
    payload = b"pickcube-act-actions-f64-v1\0" + str(array.shape).encode("ascii")
    payload += b"\0" + array.tobytes(order="C")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _episode_from_mapping(value: Mapping[str, object]) -> PickCubeActEpisodeEvaluation:
    expected = {
        "action_content_digest",
        "action_contract_violation",
        "action_count",
        "action_smoothness",
        "grasp_success",
        "peak_gpu_memory_allocated_bytes",
        "post_grasp_drop",
        "query_latencies_seconds",
        "release_failure",
        "seed",
        "simulator_error",
        "success",
        "termination_category",
        "workspace_violation",
    }
    if set(value) != expected:
        _fail("episode result", "field inventory changed")
    latency = value["query_latencies_seconds"]
    if not isinstance(latency, Sequence) or isinstance(latency, (str, bytes)):
        _fail("episode result", "latency inventory is invalid")
    try:
        return PickCubeActEpisodeEvaluation(
            seed=cast(int, value["seed"]),
            success=cast(bool, value["success"]),
            termination_category=cast(str, value["termination_category"]),
            action_count=cast(int, value["action_count"]),
            grasp_success=cast(bool, value["grasp_success"]),
            post_grasp_drop=cast(bool, value["post_grasp_drop"]),
            release_failure=cast(bool, value["release_failure"]),
            workspace_violation=cast(bool, value["workspace_violation"]),
            simulator_error=cast(bool, value["simulator_error"]),
            action_contract_violation=cast(bool, value["action_contract_violation"]),
            action_content_digest=cast(str, value["action_content_digest"]),
            action_smoothness=cast(float | None, value["action_smoothness"]),
            query_latencies_seconds=tuple(cast(float, item) for item in latency),
            peak_gpu_memory_allocated_bytes=cast(
                int, value["peak_gpu_memory_allocated_bytes"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PickCubeActEvaluationCliError("episode result is malformed") from exc


def _task_flags(evidence: TerminalTaskEvidence) -> tuple[bool, bool, bool]:
    diagnostics = evidence.diagnostics
    grasped = diagnostics.get("pickcube_grasped")
    placed = diagnostics.get("pickcube_object_placed")
    static = diagnostics.get("pickcube_robot_static")
    if not all(type(value) is bool for value in (grasped, placed, static)):
        _fail("task evidence", "official boolean diagnostics are missing")
    return cast(bool, grasped), cast(bool, placed), cast(bool, static)


def _observe(
    *,
    environment: object,
    handle: object,
    factory: LazyManiSkillSourceEnvironmentFactory,
) -> tuple[NDArray[Any], NDArray[Any]]:
    render = getattr(handle, "render_views", None)
    if not callable(render):
        _fail("renderer", "prepared handle lacks render_views")
    views = render()
    by_id = {view.camera_id: view for view in views}
    if set(by_id) != {"front_oblique", "overhead", "side_oblique"}:
        _fail("renderer", "camera inventory changed")
    names, state = factory.extract_named_robot_state(environment)
    if names != PICKCUBE_ACTIVE_JOINT_NAMES:
        _fail("proprioception", "active joint order changed")
    return np.asarray(by_id["front_oblique"].rgb), np.asarray(state)


def _termination_category(
    *,
    evidence: TerminalTaskEvidence,
    ended: PickCubeEpisodeEndedError | None,
    grasp_ever: bool,
) -> tuple[str, bool, bool, bool, bool]:
    if evidence.status is not TerminalTaskStatus.COMPLETE:
        return "task_evidence_indeterminate", False, grasp_ever, False, False
    grasped, placed, static = _task_flags(evidence)
    success = evidence.success is True and evidence.unsafe is False
    if success:
        return "success", True, grasp_ever or grasped, False, False
    if evidence.unsafe is True:
        return "workspace_violation", False, grasp_ever or grasped, False, True
    post_grasp_drop = grasp_ever and not grasped and not placed
    release_failure = placed and not static
    if ended is not None and ended.truncated:
        category = "timeout"
    elif post_grasp_drop:
        category = "post_grasp_drop"
    elif not grasp_ever and not grasped:
        category = "grasp_failure"
    elif release_failure:
        category = "release_failure"
    else:
        category = "terminal_failure"
    return category, False, grasp_ever or grasped, post_grasp_drop, False


def _run_episode(
    *,
    seed: int,
    runtime: PickCubeActInferenceRuntime,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
    render_plan: PickCubeVisualRenderPlan,
    execution_horizon: int,
) -> PickCubeActEpisodeEvaluation:
    factory = LazyManiSkillSourceEnvironmentFactory()
    renderer = LazyManiSkillPickCubeVisualRenderer()
    environment: object | None = None
    recorder: RecordingEnvironmentProxy | None = None
    actions: list[NDArray[Any]] = []
    latencies: list[float] = []
    grasp_ever = False
    ended: PickCubeEpisodeEndedError | None = None
    action_contract_violation = False
    simulator_error = False
    category = "simulator_error"
    success = False
    post_grasp_drop = False
    release_failure = False
    workspace_violation = False
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
        runtime.reset()
        torch.cuda.reset_peak_memory_stats()
        while len(actions) < 50:
            rgb, state = _observe(
                environment=environment,
                handle=handle,
                factory=factory,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                chunk = runtime.predict_action_chunk(rgb, state)
            except PickCubeActRuntimeError:
                action_contract_violation = True
                category = "action_contract_violation"
                break
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            for action in chunk[:execution_horizon]:
                try:
                    recorder.step(action)
                    actions.append(np.array(action, copy=True))
                except PickCubeEpisodeEndedError as exc:
                    actions.append(np.array(action, copy=True))
                    ended = exc
                    break
                snapshot = factory.capture_task_snapshot(environment, key_contract)
                evidence = build_pickcube_task_evidence(snapshot, key_contract)
                if evidence.status is TerminalTaskStatus.COMPLETE:
                    grasped, _, _ = _task_flags(evidence)
                    grasp_ever = grasp_ever or grasped
            if ended is not None:
                break
        if not action_contract_violation:
            snapshot = factory.capture_task_snapshot(environment, key_contract)
            evidence = build_pickcube_task_evidence(snapshot, key_contract)
            category, success, grasp_ever, post_grasp_drop, workspace_violation = (
                _termination_category(
                    evidence=evidence,
                    ended=ended,
                    grasp_ever=grasp_ever,
                )
            )
            if evidence.status is TerminalTaskStatus.COMPLETE:
                _, placed, static = _task_flags(evidence)
                release_failure = not success and placed and not static
    except Exception:
        simulator_error = True
        category = "simulator_error"
        success = False
    finally:
        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        if recorder is not None:
            recorder.close()
        elif environment is not None:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
    smoothness = None
    if len(actions) > 1:
        differences = np.diff(np.stack(actions, axis=0), axis=0)
        smoothness = float(np.linalg.norm(differences, axis=1).mean())
    return PickCubeActEpisodeEvaluation(
        seed=seed,
        success=success,
        termination_category=category,
        action_count=len(actions),
        grasp_success=grasp_ever,
        post_grasp_drop=post_grasp_drop,
        release_failure=release_failure,
        workspace_violation=workspace_violation,
        simulator_error=simulator_error,
        action_contract_violation=action_contract_violation,
        action_content_digest=_action_digest(actions),
        action_smoothness=smoothness,
        query_latencies_seconds=tuple(latencies),
        peak_gpu_memory_allocated_bytes=peak_memory,
    )


def evaluate(
    *,
    config_path: Path,
    compatibility_path: Path,
    contract_path: Path,
    training_run: Path,
    checkpoint: Path,
    output_dir: Path,
    evaluation_kind: str,
) -> Mapping[str, object]:
    """Resume and summarize one exact checkpoint/seed closed-loop evaluation."""
    config = _mapping(config_path, context="evaluation config")
    evaluator_source = _evaluator_source_identity()
    if config.get("schema_version") != "pickcube-native-act-evaluation-config-v1":
        _fail("evaluation config", "schema mismatch")
    if _integer(config, "native_episode_horizon", minimum=1) != 50:
        _fail("evaluation config", "native horizon changed")
    execution_horizon = _integer(config, "execution_horizon", minimum=1)
    if execution_horizon != 4:
        _fail("evaluation config", "execution horizon changed")
    phase = config.get(evaluation_kind)
    if not isinstance(phase, Mapping) or evaluation_kind not in {
        "smoke",
        "development",
        "final",
    }:
        _fail("evaluation config", "evaluation kind is unsupported")
    count_field = {
        "smoke": "episode_count",
        "development": "episode_count_per_checkpoint",
        "final": "episode_count_per_promoted_checkpoint",
    }[evaluation_kind]
    episode_count = _integer(phase, count_field, minimum=1)
    if (evaluation_kind, episode_count) not in {
        ("smoke", 1),
        ("development", 30),
        ("final", 100),
    }:
        _fail("evaluation config", "episode count changed")
    seed_start = _integer(phase, "seed_start")

    contract = _mapping(contract_path, context="ACT contract")
    contract_environment = contract.get("environment")
    contract_action = contract.get("action")
    if not isinstance(contract_environment, Mapping) or not isinstance(
        contract_action, Mapping
    ):
        _fail("ACT contract", "environment/action binding is missing")
    bounds = contract_action.get("bounds")
    if not isinstance(bounds, Mapping):
        _fail("ACT contract", "action bounds are missing")
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    if not isinstance(lower, Sequence) or not isinstance(upper, Sequence):
        _fail("ACT contract", "action bounds are malformed")

    environment_config = _mapping(
        _path(config, "environment_config"), context="environment config"
    )
    expected = load_expected_contract(_path(environment_config, "expected_contract"))
    report = load_compatibility_report(compatibility_path)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    if (
        contract_environment.get("compatibility_identity")
        != report.compatibility_identity
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
        _fail("visual config", "evaluation requires canonical domain")
    render_plan = build_pickcube_visual_render_plan(
        rig,
        domains.domain(cast(str, domain_id)),
        domains,
        _integer(config, "render_seed"),
    )

    run_root = Path(training_run).absolute()
    resolved = PickCubeActExperimentConfig.from_mapping(
        _mapping(run_root / "resolved_config.json", context="resolved training config")
    )
    run_manifest = _mapping(run_root / "run_manifest.json", context="run manifest")
    training_identity = run_manifest.get("training_identity")
    if not isinstance(training_identity, Mapping):
        _fail("run manifest", "training identity is missing")
    checkpoint_root = Path(checkpoint).absolute()
    try:
        checkpoint_root.relative_to(run_root / "checkpoints")
    except ValueError as exc:
        raise PickCubeActEvaluationCliError(
            "checkpoint is outside the bound training run"
        ) from exc
    runtime = load_inference_runtime(
        checkpoint=checkpoint_root,
        expected_identity=training_identity,
        experiment=resolved,
        action_lower=cast(Sequence[float], lower),
        action_upper=cast(Sequence[float], upper),
    )

    root = Path(output_dir).absolute()
    episode_root = root / "episodes"
    episode_root.mkdir(parents=True, exist_ok=True)
    episodes: list[PickCubeActEpisodeEvaluation] = []
    for seed in range(seed_start, seed_start + episode_count):
        path = episode_root / f"seed-{seed:010d}.json"
        if path.exists():
            episode = _episode_from_mapping(_mapping(path, context="episode result"))
            if episode.seed != seed:
                _fail("episode result", "path/seed binding changed")
        else:
            episode = _run_episode(
                seed=seed,
                runtime=runtime,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
                render_plan=render_plan,
                execution_horizon=execution_horizon,
            )
            write_atomic_json(path, episode.to_mapping())
        episodes.append(episode)
        print(
            json.dumps(
                {
                    "completed": len(episodes),
                    "seed": seed,
                    "success": episode.success,
                    "target": episode_count,
                    "termination_category": episode.termination_category,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    repeat = _run_episode(
        seed=seed_start,
        runtime=runtime,
        settings=settings,
        action_contract=action_contract,
        key_contract=key_contract,
        render_plan=render_plan,
        execution_horizon=execution_horizon,
    )
    first = episodes[0]
    reproducible = (
        repeat.action_content_digest == first.action_content_digest
        and repeat.action_count == first.action_count
        and repeat.success == first.success
        and repeat.termination_category == first.termination_category
    )
    reproduction = {
        "action_digest_matches": (
            repeat.action_content_digest == first.action_content_digest
        ),
        "episode_outcome_matches": (
            repeat.success == first.success
            and repeat.termination_category == first.termination_category
        ),
        "reproducible": reproducible,
        "schema_version": "pickcube-native-act-fixed-seed-reproduction-v1",
        "seed": seed_start,
    }
    write_atomic_json(root / "fixed_seed_reproduction.json", reproduction)
    summary = dict(
        summarize_checkpoint_evaluation(
            checkpoint_id=checkpoint_root.name,
            checkpoint_digest=_checkpoint_digest(checkpoint_root),
            episodes=episodes,
            seed_start=seed_start,
        )
    )
    summary.update(
        {
            "compatibility_identity": report.compatibility_identity,
            "contract_digest": contract.get("contract_digest"),
            "evaluation_kind": evaluation_kind,
            "evaluator_source": dict(evaluator_source),
            "fixed_seed_reproduction": reproduction,
            "source_commit": run_manifest.get("git_commit"),
            "training_identity": dict(training_identity),
        }
    )
    if evaluation_kind == "final":
        promotion = config.get("promotion")
        if not isinstance(promotion, Mapping):
            _fail("evaluation config", "promotion thresholds are missing")
        summary["classification"] = classify_final_checkpoint(
            summary,
            assets_complete=True,
            action_contract_complete=(summary["action_contract_violation_count"] == 0),
            reproducible=reproducible,
            primary_success_rate=_number(promotion, "primary_success_rate"),
            secondary_success_rate=_number(promotion, "secondary_success_rate"),
        ).value
    write_atomic_json(root / "evaluation_summary.json", summary)
    return summary


def main() -> int:
    """Expose resumable development/final evaluation for one checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-kind",
        choices=("smoke", "development", "final"),
        required=True,
    )
    args = parser.parse_args()
    summary = evaluate(
        config_path=args.config,
        compatibility_path=args.compatibility_report,
        contract_path=args.contract,
        training_run=args.training_run,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        evaluation_kind=args.evaluation_kind,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
