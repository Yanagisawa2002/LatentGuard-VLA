"""CPU-only tests for the native PickCube expert gate."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from latentguard.policies.experts import (
    PICKCUBE_EXPERT_PHASES,
    ExpertEpisodeAudit,
    PickCubeExpertPhaseTracker,
    assert_pickcube_expert_has_no_teleport_calls,
    summarize_expert_evaluation,
)
from latentguard.policies.experts.pickcube_expert import PickCubeExpertError


def _phase_counts(value: int = 1) -> MappingProxyType[str, int]:
    return MappingProxyType({phase.value: value for phase in PICKCUBE_EXPERT_PHASES})


def test_expert_phase_tracker_requires_fixed_order_and_real_actions() -> None:
    tracker = PickCubeExpertPhaseTracker()
    tracker.capture_boundary(object(), 0)
    for index, phase in enumerate(PICKCUBE_EXPERT_PHASES, start=1):
        tracker.transition(phase)
        tracker.capture_boundary(object(), index)
    tracker.require_complete()
    assert dict(tracker.counts_by_name()) == {
        phase.value: 1 for phase in PICKCUBE_EXPERT_PHASES
    }


def test_expert_phase_tracker_rejects_unexecuted_phase() -> None:
    tracker = PickCubeExpertPhaseTracker()
    for index, phase in enumerate(PICKCUBE_EXPERT_PHASES[:-1], start=1):
        tracker.transition(phase)
        tracker.capture_boundary(object(), index)
    with pytest.raises(PickCubeExpertError, match="phases executed no actions"):
        tracker.require_complete()


def test_expert_source_audit_rejects_no_current_teleport_calls() -> None:
    digest = assert_pickcube_expert_has_no_teleport_calls()
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_fixed_100_seed_expert_gate_accepts_exactly_95_successes() -> None:
    episodes = tuple(
        ExpertEpisodeAudit(
            seed=600000 + index,
            success=index < 95,
            action_count=24,
            phase_action_counts=_phase_counts(),
            failure_category=None if index < 95 else "motion_planning_failure",
        )
        for index in range(100)
    )
    summary = summarize_expert_evaluation(
        episodes,
        source_audit_digest="sha256:" + "a" * 64,
    )
    assert summary.success_rate == 0.95
    assert summary.authorized_for_demonstration_collection
    assert summary.failure_taxonomy == {"motion_planning_failure": 5}


def test_expert_gate_fails_closed_for_simulator_error() -> None:
    episodes = [
        ExpertEpisodeAudit(
            seed=600000 + index,
            success=True,
            action_count=24,
            phase_action_counts=_phase_counts(),
        )
        for index in range(100)
    ]
    episodes[-1] = ExpertEpisodeAudit(
        seed=600099,
        success=False,
        action_count=0,
        phase_action_counts=_phase_counts(0),
        simulator_error=True,
        failure_category="simulator_error",
    )
    summary = summarize_expert_evaluation(
        episodes,
        source_audit_digest="sha256:" + "b" * 64,
    )
    assert summary.success_rate == 0.99
    assert not summary.authorized_for_demonstration_collection
    assert not summary.gate_checks["simulator_error_zero"]
