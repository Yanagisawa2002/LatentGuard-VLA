from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from latentguard.control.config import load_closed_loop_configuration
from latentguard.control.models import (
    EpisodeState,
    SelectorOutputV1,
    SourcePlanIdentityV1,
    content_digest,
)
from latentguard.control.runner import (
    BoundSourcePlanV1,
    RuntimeBoundaryV1,
    RuntimeStepResultV1,
    run_closed_loop_episode,
    simple_pool_from_actions,
)
from latentguard.control.serialization import ClosedLoopEpisodeStore


def digest(label: str) -> str:
    return content_digest({"label": label})


def source_plan(length: int = 36) -> BoundSourcePlanV1:
    actions = np.zeros((length, 8), dtype=np.float64)
    identity = SourcePlanIdentityV1(
        source_trajectory_id="fake-trajectory",
        split_group_id="fake-split",
        reset_seed=420000,
        source_action_digest=digest("source-actions"),
        initial_state_digest=digest("state-0"),
        complete_state_tree_digests=(digest("state-0"),),
        independent_replay_success=True,
        planner_identity="fake-official-planner",
        compatibility_identity=digest("compatibility"),
    )
    return BoundSourcePlanV1(identity, actions, "state-0")


@dataclass
class SharedWorld:
    step: int = 0
    success_at: int | None = None
    restore_count: int = 0

    def boundary(self) -> RuntimeBoundaryV1:
        return RuntimeBoundaryV1(
            state_reference=f"state-{self.step}",
            state_digest=digest(f"state-{self.step}"),
            verifier_state=np.full(38, self.step, dtype=np.float32),
        )


class FakeRuntime:
    def __init__(self, world: SharedWorld) -> None:
        self.world = world
        self.closed = False

    def start(
        self, source: BoundSourcePlanV1, *, visual_domain: str
    ) -> RuntimeBoundaryV1:
        assert visual_domain == "not_applicable"
        self.world.step = 0
        return self.world.boundary()

    def restore_boundary(
        self,
        source: BoundSourcePlanV1,
        *,
        state_reference: str,
        expected_state_digest: str,
        visual_domain: str,
    ) -> RuntimeBoundaryV1:
        self.world.restore_count += 1
        self.world.step = int(state_reference.rsplit("-", 1)[1])
        boundary = self.world.boundary()
        assert boundary.state_digest == expected_state_digest
        return boundary

    def execute(
        self,
        actions: np.ndarray,
        *,
        maximum_control_steps_remaining: int,
        visual_domain: str,
    ) -> RuntimeStepResultV1:
        executed = 0
        outcome = None
        for _ in actions[:maximum_control_steps_remaining]:
            self.world.step += 1
            executed += 1
            if (
                self.world.success_at is not None
                and self.world.step >= self.world.success_at
            ):
                outcome = EpisodeState.SUCCESS
                break
        return RuntimeStepResultV1(
            next_boundary=self.world.boundary(),
            executed_step_count=executed,
            task_evidence_digest=digest(f"evidence-{self.world.step}"),
            outcome=outcome,
        )

    def close(self) -> None:
        self.closed = True


class TerminalRuntime(FakeRuntime):
    def __init__(self, world: SharedWorld, outcome: EpisodeState) -> None:
        super().__init__(world)
        self.outcome = outcome

    def execute(
        self,
        actions: np.ndarray,
        *,
        maximum_control_steps_remaining: int,
        visual_domain: str,
    ) -> RuntimeStepResultV1:
        result = super().execute(
            actions,
            maximum_control_steps_remaining=maximum_control_steps_remaining,
            visual_domain=visual_domain,
        )
        return RuntimeStepResultV1(
            next_boundary=result.next_boundary,
            executed_step_count=result.executed_step_count,
            task_evidence_digest=result.task_evidence_digest,
            outcome=self.outcome,
        )


class FailingRuntime(FakeRuntime):
    def execute(
        self,
        actions: np.ndarray,
        *,
        maximum_control_steps_remaining: int,
        visual_domain: str,
    ) -> RuntimeStepResultV1:
        raise RuntimeError("synthetic runtime failure")


class FakePoolFactory:
    configuration_digest = digest("candidate-config")

    def build(
        self,
        source: BoundSourcePlanV1,
        boundary: RuntimeBoundaryV1,
        *,
        decision_ordinal: int,
        nominal_plan_index: int,
    ):
        base = source.actions[nominal_plan_index : nominal_plan_index + 16]
        candidates = tuple(
            np.asarray(base + (index + 1) * 0.001, dtype=np.float64, order="C")
            for index in range(8)
        )
        return simple_pool_from_actions(
            source,
            boundary,
            decision_ordinal=decision_ordinal,
            nominal_plan_index=nominal_plan_index,
            candidate_actions=candidates,
            candidate_pool_configuration_digest=self.configuration_digest,
        )


class FakeSelector:
    selector_id = "fake-selector"
    visual = False
    select_count = 0

    def select(self, pool, boundary) -> SelectorOutputV1:
        self.select_count += 1
        scores = {
            candidate_id: float(index) / 10.0
            for index, candidate_id in enumerate(pool.ordered_candidate_ids)
        }
        return SelectorOutputV1(
            selector_id=self.selector_id,
            scores=scores,
            ranking=pool.ordered_candidate_ids,
            checkpoint_ensemble_identity=digest("fake-ensemble"),
            probabilities=True,
        )


def config():
    return load_closed_loop_configuration(
        Path("configs/control/m4c/closed-loop-v1.json")
    )


def test_repeated_decisions_and_last_full_window_tail(tmp_path: Path) -> None:
    world = SharedWorld()
    selector = FakeSelector()
    result = run_closed_loop_episode(
        source_plan(36),
        selector=selector,
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(world),
        store=ClosedLoopEpisodeStore(tmp_path / "episode"),
    )
    assert result.episode.state is EpisodeState.HORIZON_EXHAUSTED
    assert result.episode.executed_control_steps == 36
    assert [item.executed_step_count for item in result.episode.boundaries] == [
        4,
        4,
        4,
        4,
        4,
        16,
    ]
    assert result.episode.tail_fallback_count == 1
    assert selector.select_count == 6


def test_early_success_is_distinct_terminal_outcome(tmp_path: Path) -> None:
    world = SharedWorld(success_at=7)
    result = run_closed_loop_episode(
        source_plan(),
        selector=FakeSelector(),
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(world),
        store=ClosedLoopEpisodeStore(tmp_path / "episode"),
    )
    assert result.episode.state is EpisodeState.SUCCESS
    assert result.episode.executed_control_steps == 7
    assert len(result.episode.boundaries) == 2


@pytest.mark.parametrize("outcome", [EpisodeState.TASK_FAILURE, EpisodeState.UNSAFE])
def test_task_failure_and_unsafe_remain_distinct(
    tmp_path: Path, outcome: EpisodeState
) -> None:
    result = run_closed_loop_episode(
        source_plan(),
        selector=FakeSelector(),
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=TerminalRuntime(SharedWorld(), outcome),
        store=ClosedLoopEpisodeStore(tmp_path / outcome.value),
    )
    assert result.episode.state is outcome


def test_runtime_exception_is_execution_error_not_task_failure(tmp_path: Path) -> None:
    result = run_closed_loop_episode(
        source_plan(),
        selector=FakeSelector(),
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=FailingRuntime(SharedWorld()),
        store=ClosedLoopEpisodeStore(tmp_path / "execution-error"),
    )
    assert result.episode.state is EpisodeState.EXECUTION_ERROR
    assert result.episode.execution_error_type == "RuntimeError"


@pytest.mark.parametrize("phase", ["after_selection", "during_stride"])
def test_resume_reuses_finalized_selection_without_rescoring(
    tmp_path: Path, phase: str
) -> None:
    world = SharedWorld(success_at=5)
    selector = FakeSelector()

    def interrupt(observed: str, ordinal: int) -> None:
        if observed == phase and ordinal == 0:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_closed_loop_episode(
            source_plan(),
            selector=selector,
            visual_domain="not_applicable",
            config=config(),
            candidate_factory=FakePoolFactory(),
            runtime=FakeRuntime(world),
            store=ClosedLoopEpisodeStore(tmp_path / "episode"),
            interruption_hook=interrupt,
        )
    decision_before = ClosedLoopEpisodeStore(tmp_path / "episode").load_decision(0)
    assert selector.select_count == 1
    result = run_closed_loop_episode(
        source_plan(),
        selector=selector,
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(world),
        store=ClosedLoopEpisodeStore(tmp_path / "episode"),
    )
    decision_after = ClosedLoopEpisodeStore(tmp_path / "episode").load_decision(0)
    assert selector.select_count == 2
    assert decision_after.content_digest == decision_before.content_digest
    assert result.episode.recovery_count == 1
    assert world.restore_count == 1


def test_complete_episode_resume_executes_zero_work(tmp_path: Path) -> None:
    world = SharedWorld(success_at=1)
    store = ClosedLoopEpisodeStore(tmp_path / "episode")
    first = run_closed_loop_episode(
        source_plan(),
        selector=FakeSelector(),
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(world),
        store=store,
    )
    restored_before = world.restore_count
    second = run_closed_loop_episode(
        source_plan(),
        selector=FakeSelector(),
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(world),
        store=store,
    )
    assert first.episode.content_digest == second.episode.content_digest
    assert second.zero_work_resume is True
    assert world.restore_count == restored_before


def test_crash_after_stride_restores_completed_boundary_without_duplicate(
    tmp_path: Path,
) -> None:
    world = SharedWorld(success_at=5)
    selector = FakeSelector()
    store = ClosedLoopEpisodeStore(tmp_path / "episode")

    def interrupt(phase: str, ordinal: int) -> None:
        if phase == "after_stride_before_next_decision" and ordinal == 0:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_closed_loop_episode(
            source_plan(),
            selector=selector,
            visual_domain="not_applicable",
            config=config(),
            candidate_factory=FakePoolFactory(),
            runtime=FakeRuntime(world),
            store=store,
            interruption_hook=interrupt,
        )
    first_decision = store.load_decision(0)
    result = run_closed_loop_episode(
        source_plan(),
        selector=selector,
        visual_domain="not_applicable",
        config=config(),
        candidate_factory=FakePoolFactory(),
        runtime=FakeRuntime(world),
        store=store,
    )
    assert store.load_decision(0).content_digest == first_decision.content_digest
    assert result.episode.executed_control_steps == 5
    assert selector.select_count == 2
