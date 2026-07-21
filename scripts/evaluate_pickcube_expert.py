"""Run the fixed 100-seed PickCube native-expert quality gate."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

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
    PickCubeSourceGenerationError,
    PickCubeTrajectoryActionLimitError,
    RecordingEnvironmentProxy,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
    build_pickcube_task_evidence,
)
from latentguard.policies.experts import (
    ExpertEpisodeAudit,
    PickCubeExpertError,
    PickCubeExpertPhaseTracker,
    PickCubeMotionPlanningExpert,
    assert_pickcube_expert_has_no_teleport_calls,
    summarize_expert_evaluation,
)
from latentguard.replay.models import TerminalTaskStatus


def _mapping(path: Path, *, context: str) -> Mapping[str, object]:
    source = path.absolute()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{context}: expected regular unlinked configuration")
    value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a JSON object")
    return cast(Mapping[str, object], value)


def _path(item: Mapping[str, object], field: str) -> Path:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration {field}: expected non-empty path")
    return Path(value)


def _integer(item: Mapping[str, object], field: str, *, minimum: int) -> int:
    value = item.get(field)
    if type(value) is not int or value < minimum:
        raise ValueError(f"configuration {field}: expected integer >= {minimum}")
    return value


def _number(item: Mapping[str, object], field: str) -> float:
    value = item.get(field)
    if type(value) not in (int, float) or not np.isfinite(float(value)):
        raise ValueError(f"configuration {field}: expected finite number")
    return float(value)


def _compatibility_path(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    value = os.environ.get("LATENTGUARD_PICKCUBE_COMPATIBILITY_REPORT")
    if not value:
        raise ValueError(
            "pass --compatibility-report or set "
            "LATENTGUARD_PICKCUBE_COMPATIBILITY_REPORT"
        )
    return Path(value)


def _failure_from_result(result: object) -> str:
    if isinstance(result, tuple) and len(result) == 5:
        if bool(np.asarray(result[3]).reshape(()).item()):
            return "timeout"
        if bool(np.asarray(result[2]).reshape(()).item()):
            return "terminal_failure"
    return "task_not_completed"


def _failed_episode(
    *,
    seed: int,
    tracker: PickCubeExpertPhaseTracker,
    action_count: int,
    category: str,
    simulator_error: bool,
) -> ExpertEpisodeAudit:
    message = category.lower()
    return ExpertEpisodeAudit(
        seed=seed,
        success=False,
        action_count=action_count,
        phase_action_counts=tracker.counts_by_name(),
        simulator_error=simulator_error,
        nonfinite_action_count=int("non_finite" in message),
        action_out_of_bounds_count=int("out_of_bounds" in message),
        workspace_violation_count=int(category == "workspace_violation"),
        failure_category=category,
    )


def run_expert_gate(
    config_path: Path,
    compatibility_path: Path,
) -> Mapping[str, object]:
    """Execute every configured seed independently and retain all outcomes."""
    config = _mapping(config_path, context="expert evaluation config")
    if config.get("schema_version") != "pickcube-act-expert-eval-config-v1":
        raise ValueError("expert evaluation config schema mismatch")
    episode_count = _integer(config, "episode_count", minimum=100)
    seed_start = _integer(config, "seed_start", minimum=0)
    action_limit = _integer(config, "trajectory_action_limit", minimum=1)
    minimum_success_rate = _number(config, "minimum_success_rate")
    if minimum_success_rate != 0.95:
        raise ValueError("expert evaluation minimum success rate must equal 0.95")
    if seed_start + episode_count > 2**32:
        raise ValueError("expert evaluation seed schedule exceeds uint32")

    environment_config = _mapping(
        _path(config, "environment_config"),
        context="environment config",
    )
    expected = load_expected_contract(_path(environment_config, "expected_contract"))
    report = load_compatibility_report(compatibility_path)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(
        _path(environment_config, "action_layout")
    )
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding,
        coordinate_frame=layout.coordinate_frame,
    )
    key_contract = PickCubeTaskKeyContract.from_compatibility_report(report)
    source_audit_digest = assert_pickcube_expert_has_no_teleport_calls()
    factory = LazyManiSkillSourceEnvironmentFactory()
    expert = PickCubeMotionPlanningExpert()
    episodes: list[ExpertEpisodeAudit] = []

    for seed in range(seed_start, seed_start + episode_count):
        tracker = PickCubeExpertPhaseTracker()
        environment: object | None = None
        recorder: RecordingEnvironmentProxy | None = None
        try:
            environment = factory.create_environment(
                settings,
                action_contract,
                purpose="official_source_generation",
            )
            recorder = RecordingEnvironmentProxy(
                environment,
                action_contract,
                trajectory_action_limit=action_limit,
                boundary_capture=tracker.capture_boundary,
            )
            solver_result = expert.solve(
                recorder,
                seed=seed,
                phase_tracker=tracker,
            )
            recorder.verify_interception_complete()
            snapshot = factory.capture_task_snapshot(environment, key_contract)
            evidence = build_pickcube_task_evidence(snapshot, key_contract)
            category: str | None = None
            success = (
                evidence.status is TerminalTaskStatus.COMPLETE
                and evidence.success is True
                and evidence.unsafe is False
            )
            if evidence.unsafe is True:
                category = "workspace_violation"
                success = False
            elif evidence.status is not TerminalTaskStatus.COMPLETE:
                category = "task_evidence_indeterminate"
                success = False
            elif not success:
                category = _failure_from_result(recorder.last_step_result)
                if solver_result == -1:
                    category = "motion_planning_failure"
            if success:
                try:
                    tracker.require_complete()
                except PickCubeExpertError:
                    category = "expert_phase_incomplete"
                    success = False
            if success:
                episodes.append(
                    ExpertEpisodeAudit(
                        seed=seed,
                        success=True,
                        action_count=len(recorder.actions),
                        phase_action_counts=tracker.counts_by_name(),
                    )
                )
            else:
                assert category is not None
                episodes.append(
                    _failed_episode(
                        seed=seed,
                        tracker=tracker,
                        action_count=len(recorder.actions),
                        category=category,
                        simulator_error=False,
                    )
                )
        except PickCubeTrajectoryActionLimitError:
            episodes.append(
                _failed_episode(
                    seed=seed,
                    tracker=tracker,
                    action_count=0 if recorder is None else len(recorder.actions),
                    category="trajectory_action_limit",
                    simulator_error=False,
                )
            )
        except (PickCubeExpertError, PickCubeSourceGenerationError) as exc:
            message = str(exc).lower()
            category = "expert_execution_error"
            if "non_finite" in message:
                category = "action_non_finite"
            elif "out_of_bounds" in message:
                category = "action_out_of_bounds"
            episodes.append(
                _failed_episode(
                    seed=seed,
                    tracker=tracker,
                    action_count=0 if recorder is None else len(recorder.actions),
                    category=category,
                    simulator_error=False,
                )
            )
        except Exception:
            episodes.append(
                _failed_episode(
                    seed=seed,
                    tracker=tracker,
                    action_count=0 if recorder is None else len(recorder.actions),
                    category="simulator_error",
                    simulator_error=True,
                )
            )
        finally:
            if recorder is not None:
                recorder.close()
            elif environment is not None:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()

    summary = summarize_expert_evaluation(
        episodes,
        source_audit_digest=source_audit_digest,
    )
    payload = summary.to_mapping()
    payload.update(
        {
            "compatibility_identity": report.compatibility_identity,
            "environment_id": report.environment_id,
            "seed_end_exclusive": seed_start + episode_count,
            "seed_start": seed_start,
        }
    )
    return payload


def main() -> int:
    """Run the expert gate, persist evidence, and fail if collection is blocked."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = _mapping(args.config, context="expert evaluation config")
    output = args.output or _path(config, "output")
    payload = run_expert_gate(
        args.config,
        _compatibility_path(args.compatibility_report),
    )
    write_atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["authorized_for_demonstration_collection"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
