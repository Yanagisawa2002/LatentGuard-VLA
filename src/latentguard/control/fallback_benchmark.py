"""Cost-aware M4D development/final scheduling on the accepted M4C runner."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from latentguard.control.benchmark import (
    BenchmarkRunSummaryV1,
    load_episode_execution_seconds,
    record_episode_execution_seconds,
)
from latentguard.control.config import ClosedLoopConfigurationV1
from latentguard.control.fallback import (
    CandidateRiskScorer,
    FallbackShieldSelectorV1,
    FaultScheduleConfigurationV1,
    GateProfilesConfigurationV1,
    NominalScenario,
)
from latentguard.control.fallback_config import (
    FallbackPolicyConfigurationV1,
    FallbackSelectorMatrixV1,
)
from latentguard.control.models import ClosedLoopEpisodeRecordV1, EpisodeState
from latentguard.control.runner import (
    BoundSourcePlanV1,
    CandidatePoolFactory,
    ClosedLoopRuntime,
    run_closed_loop_episode,
)
from latentguard.control.serialization import ClosedLoopEpisodeStore


@dataclass(frozen=True, slots=True)
class FallbackBenchmarkExecutionSpecV1:
    """One source/policy/profile/scenario/domain execution."""

    source_trajectory_id: str
    policy_id: str
    mode: str
    scorer_id: str | None
    profile_id: str | None
    scenario: NominalScenario
    visual_domain: str

    @property
    def relative_directory(self) -> Path:
        """Return a deterministic runtime path with no machine identity."""

        profile = "none" if self.profile_id is None else self.profile_id
        return (
            Path(self.policy_id)
            / profile
            / self.scenario.value
            / self.visual_domain
            / self.source_trajectory_id
        )


def build_fallback_benchmark_schedule(
    sources: Sequence[BoundSourcePlanV1],
    matrix: FallbackSelectorMatrixV1,
    *,
    phase: str,
    selected_profiles: Mapping[str, str] | None = None,
) -> tuple[FallbackBenchmarkExecutionSpecV1, ...]:
    """Build exactly 32 development or 26 final episodes per source."""

    values = tuple(sources)
    if not values or phase not in {"development", "final"}:
        raise ValueError("M4D schedule requires sources and a valid phase")
    policies = (
        matrix.development_policies if phase == "development" else matrix.final_policies
    )
    result: list[FallbackBenchmarkExecutionSpecV1] = []
    for source in values:
        for policy in policies:
            if policy.mode == "gated":
                if phase == "development":
                    profiles: tuple[str | None, ...] = tuple(policy.gate_profiles)
                else:
                    if (
                        selected_profiles is None
                        or policy.policy_id not in selected_profiles
                    ):
                        raise ValueError(
                            "final M4D schedule lacks a frozen gate profile"
                        )
                    profiles = (selected_profiles[policy.policy_id],)
            else:
                profiles = (None,)
            domains = (
                matrix.development_visual_domains
                if phase == "development" and policy.visual
                else matrix.visual_domains
                if policy.visual
                else ("not_applicable",)
            )
            for profile in profiles:
                for scenario in matrix.scenarios:
                    for domain in domains:
                        result.append(
                            FallbackBenchmarkExecutionSpecV1(
                                source_trajectory_id=(
                                    source.identity.source_trajectory_id
                                ),
                                policy_id=policy.policy_id,
                                mode=policy.mode,
                                scorer_id=policy.scorer_id,
                                profile_id=profile,
                                scenario=NominalScenario(scenario),
                                visual_domain=domain,
                            )
                        )
    expected_per_source = 32 if phase == "development" else 26
    expected = len(values) * expected_per_source
    if len(result) != expected or len(set(result)) != expected:
        raise RuntimeError(
            f"M4D {phase} schedule differs from {expected_per_source}/source"
        )
    return tuple(result)


def _policy_by_id(
    matrix: FallbackSelectorMatrixV1, *, phase: str
) -> Mapping[str, FallbackPolicyConfigurationV1]:
    values = (
        matrix.development_policies if phase == "development" else matrix.final_policies
    )
    return {item.policy_id: item for item in values}


def run_fallback_benchmark_schedule(
    sources: Sequence[BoundSourcePlanV1],
    *,
    phase: str,
    matrix: FallbackSelectorMatrixV1,
    selected_profiles: Mapping[str, str] | None,
    gate_profiles: GateProfilesConfigurationV1,
    fault_schedule: FaultScheduleConfigurationV1,
    config: ClosedLoopConfigurationV1,
    scorers: Mapping[str, CandidateRiskScorer],
    candidate_factory: CandidatePoolFactory,
    runtime_factory: Callable[[], ClosedLoopRuntime],
    output_root: Path,
) -> BenchmarkRunSummaryV1:
    """Execute the frozen schedule once with exact M4C recovery and resume."""

    source_by_id = {item.identity.source_trajectory_id: item for item in sources}
    schedule = build_fallback_benchmark_schedule(
        sources,
        matrix,
        phase=phase,
        selected_profiles=selected_profiles,
    )
    policies = _policy_by_id(matrix, phase=phase)
    required_scorers = {
        item.scorer_id for item in policies.values() if item.scorer_id is not None
    }
    if not required_scorers <= set(scorers):
        raise ValueError("M4D runtime scorer inventory is incomplete")
    executed = 0
    zero_work = 0
    episodes: list[ClosedLoopEpisodeRecordV1] = []
    for spec in schedule:
        policy = policies[spec.policy_id]
        scorer = None if spec.scorer_id is None else scorers[spec.scorer_id]
        if scorer is not None and scorer.visual != policy.visual:
            raise ValueError("M4D scorer visual applicability differs")
        wrapper_policy_id = (
            spec.policy_id
            if phase == "final" or spec.profile_id is None
            else f"{spec.policy_id}__{spec.profile_id}"
        )
        selector = FallbackShieldSelectorV1(
            policy_id=wrapper_policy_id,
            scenario=spec.scenario,
            schedule=fault_schedule,
            mode=spec.mode,
            scorer=scorer,
            gate_profile=(
                None
                if spec.profile_id is None
                else gate_profiles.profile(spec.profile_id)
            ),
        )
        episode_directory = Path(output_root) / spec.relative_directory
        started = perf_counter()
        result = run_closed_loop_episode(
            source_by_id[spec.source_trajectory_id],
            selector=selector,
            visual_domain=spec.visual_domain,
            config=config,
            candidate_factory=candidate_factory,
            runtime=runtime_factory(),
            store=ClosedLoopEpisodeStore(episode_directory),
        )
        elapsed = perf_counter() - started
        if result.zero_work_resume:
            load_episode_execution_seconds(
                episode_directory,
                expected_episode_execution_id=result.episode.episode_execution_id,
            )
        else:
            record_episode_execution_seconds(
                episode_directory,
                episode_execution_id=result.episode.episode_execution_id,
                execution_seconds=elapsed,
            )
        episodes.append(result.episode)
        zero_work += int(result.zero_work_resume)
        executed += int(not result.zero_work_resume)
    counts = {
        state.value: sum(item.state is state for item in episodes)
        for state in (
            EpisodeState.SUCCESS,
            EpisodeState.TASK_FAILURE,
            EpisodeState.UNSAFE,
            EpisodeState.HORIZON_EXHAUSTED,
            EpisodeState.EXECUTION_ERROR,
        )
    }
    return BenchmarkRunSummaryV1(
        planned_episode_count=len(schedule),
        completed_episode_count=sum(item.state.terminal for item in episodes),
        executed_episode_count=executed,
        zero_work_episode_count=zero_work,
        outcome_counts=counts,
    )


def load_gate_selection(path: Path) -> tuple[Mapping[str, str], str, str]:
    """Load an immutable development-only gate selection for final execution."""

    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        raise ValueError("M4D gate selection is missing or unsafe")
    try:
        raw = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid M4D gate selection: {exc}") from exc
    expected = {
        "content_digest",
        "development_manifest_digest",
        "gate_profiles_digest",
        "outcomes_opened",
        "schema_version",
        "selected_profiles",
        "selection_semantic",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("M4D gate selection schema differs")
    value = cast(Mapping[str, Any], raw)
    if (
        value["outcomes_opened"] != "m4d_development_only"
        or value["schema_version"] != "1.0"
        or value["selection_semantic"]
        != "clean_constraint_then_fault_success_cost_tiebreak_v1"
    ):
        raise ValueError("M4D gate selection opened unauthorized outcomes")
    selected = value["selected_profiles"]
    if not isinstance(selected, Mapping):
        raise ValueError("M4D selected profile inventory is invalid")
    result: dict[str, str] = {}
    entry_fields = {
        "candidates",
        "clean_constraint_passed",
        "selected_profile_id",
    }
    candidate_fields = {
        "clean_constraint_passed",
        "clean_success_rate",
        "conservatism_ordinal",
        "fallback_invocation_rate",
        "fault_success_rate",
        "fault_unsuccessful_rate",
        "intervention_rate",
        "profile_id",
    }
    for policy, entry in selected.items():
        if (
            not isinstance(policy, str)
            or not isinstance(entry, Mapping)
            or set(entry) != entry_fields
        ):
            raise ValueError("M4D selected profile entry is invalid")
        candidates = entry.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 3
            or any(
                not isinstance(candidate, Mapping) or set(candidate) != candidate_fields
                for candidate in candidates
            )
        ):
            raise ValueError("M4D gate profile candidates are invalid")
        profile = entry.get("selected_profile_id")
        if profile not in {"conservative", "balanced", "responsive"}:
            raise ValueError("M4D selected profile is invalid")
        result[policy] = cast(str, profile)
    required = {
        "gated_action_only",
        "gated_direct_visual",
        "gated_distilled_visual",
        "gated_privileged_structured",
    }
    if set(result) != required:
        raise ValueError("M4D selected profile policy inventory differs")
    from latentguard.control.models import content_digest

    payload = {key: value[key] for key in expected if key != "content_digest"}
    expected_digest = content_digest(payload, context="M4DGateSelectionV1")
    if value["content_digest"] != expected_digest:
        raise ValueError("M4D gate selection content digest changed")
    return (
        result,
        cast(str, value["content_digest"]),
        cast(str, value["gate_profiles_digest"]),
    )


__all__ = [
    "FallbackBenchmarkExecutionSpecV1",
    "build_fallback_benchmark_schedule",
    "load_gate_selection",
    "run_fallback_benchmark_schedule",
]
