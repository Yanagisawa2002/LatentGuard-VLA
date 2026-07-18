"""Fixed M4C selector/domain schedule and idempotent benchmark orchestration."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from latentguard.control.config import (
    ClosedLoopConfigurationV1,
    SelectorMatrixConfigurationV1,
)
from latentguard.control.models import ClosedLoopEpisodeRecordV1, EpisodeState
from latentguard.control.runner import (
    BoundSourcePlanV1,
    CandidatePoolFactory,
    ClosedLoopRuntime,
    ClosedLoopSelector,
    run_closed_loop_episode,
)
from latentguard.control.serialization import (
    ClosedLoopEpisodeStore,
    write_exclusive_json,
)

EXECUTION_METRICS_FILENAME = "runtime-metrics.json"


def record_episode_execution_seconds(
    episode_directory: Path,
    *,
    episode_execution_id: str,
    execution_seconds: float,
) -> None:
    """Persist non-semantic wall time beside one completed episode."""

    if not math.isfinite(execution_seconds) or execution_seconds < 0.0:
        raise ValueError("episode execution time must be finite and non-negative")
    write_exclusive_json(
        Path(episode_directory) / EXECUTION_METRICS_FILENAME,
        {
            "episode_execution_id": episode_execution_id,
            "execution_seconds": execution_seconds,
            "schema_version": "1.0",
        },
    )


def load_episode_execution_seconds(
    episode_directory: Path,
    *,
    expected_episode_execution_id: str,
) -> float:
    """Strictly load the runtime-only wall-time evidence for one episode."""

    path = (Path(episode_directory) / EXECUTION_METRICS_FILENAME).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("episode runtime metrics are missing or unsafe")
    try:
        raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid episode runtime metrics: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "episode_execution_id",
        "execution_seconds",
        "schema_version",
    }:
        raise ValueError("episode runtime metrics schema differs")
    if (
        raw.get("schema_version") != "1.0"
        or raw.get("episode_execution_id") != expected_episode_execution_id
    ):
        raise ValueError("episode runtime metrics identity differs")
    value = raw.get("execution_seconds")
    if type(value) not in (int, float):
        raise ValueError("episode runtime metrics value is invalid")
    seconds = float(cast(int | float, value))
    if not math.isfinite(seconds):
        raise ValueError("episode runtime metrics value is invalid")
    if seconds < 0.0:
        raise ValueError("episode runtime metrics value is negative")
    return seconds


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionSpecV1:
    """One source/selector/domain episode in the fixed 600-episode matrix."""

    source_trajectory_id: str
    selector_id: str
    visual_domain: str

    @property
    def relative_directory(self) -> Path:
        """Return a safe deterministic runtime subdirectory."""

        return Path(self.selector_id) / self.visual_domain / self.source_trajectory_id


def build_benchmark_schedule(
    sources: Sequence[BoundSourcePlanV1],
    matrix: SelectorMatrixConfigurationV1,
) -> tuple[BenchmarkExecutionSpecV1, ...]:
    """Build 4 nonvisual plus 2x3 visual executions per source."""

    values = tuple(sources)
    if not values:
        raise ValueError("benchmark schedule requires sources")
    result: list[BenchmarkExecutionSpecV1] = []
    for source in values:
        for selector in matrix.selectors:
            domains = matrix.visual_domains if selector.visual else ("not_applicable",)
            for domain in domains:
                result.append(
                    BenchmarkExecutionSpecV1(
                        source_trajectory_id=source.identity.source_trajectory_id,
                        selector_id=selector.selector_id,
                        visual_domain=domain,
                    )
                )
    expected = len(values) * 10
    if len(result) != expected or len(set(result)) != expected:
        raise RuntimeError("benchmark schedule is not exactly ten episodes per source")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummaryV1:
    """Compact orchestration result with strict zero-work resume accounting."""

    planned_episode_count: int
    completed_episode_count: int
    executed_episode_count: int
    zero_work_episode_count: int
    outcome_counts: Mapping[str, int]

    @property
    def zero_work_resume(self) -> bool:
        """Return whether every planned episode was already complete."""

        return self.executed_episode_count == 0 and (
            self.zero_work_episode_count == self.planned_episode_count
        )


def run_benchmark_schedule(
    sources: Sequence[BoundSourcePlanV1],
    *,
    matrix: SelectorMatrixConfigurationV1,
    config: ClosedLoopConfigurationV1,
    selectors: Mapping[str, ClosedLoopSelector],
    candidate_factory: CandidatePoolFactory,
    runtime_factory: Callable[[], ClosedLoopRuntime],
    output_root: Path,
) -> BenchmarkRunSummaryV1:
    """Run the fixed matrix once and skip strictly validated terminal identities."""

    source_by_id = {item.identity.source_trajectory_id: item for item in sources}
    schedule = build_benchmark_schedule(sources, matrix)
    if set(selectors) != {item.selector_id for item in matrix.selectors}:
        raise ValueError("runtime selectors differ from the frozen matrix")
    executed = 0
    zero_work = 0
    episodes: list[ClosedLoopEpisodeRecordV1] = []
    for spec in schedule:
        episode_directory = Path(output_root) / spec.relative_directory
        started = perf_counter()
        result = run_closed_loop_episode(
            source_by_id[spec.source_trajectory_id],
            selector=selectors[spec.selector_id],
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


__all__ = [
    "BenchmarkExecutionSpecV1",
    "BenchmarkRunSummaryV1",
    "EXECUTION_METRICS_FILENAME",
    "build_benchmark_schedule",
    "load_episode_execution_seconds",
    "record_episode_execution_seconds",
    "run_benchmark_schedule",
]
