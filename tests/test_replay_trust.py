"""CPU-only replay state, result, protocol, and trust-boundary tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from latentguard.evaluation.models import EvaluationStatus
from latentguard.models import LabelSource, LabelStrength
from latentguard.replay.base import ReplayEnvironmentSession, ReplayValidationError
from latentguard.replay.models import (
    ActionExecutionEvidence,
    PairedReplayResult,
    ReplayStateReference,
    ReplayTaskReference,
    ReplayTrustDescriptor,
    ReplayTrustTier,
    StateComparisonSemantic,
    StateMatchKind,
    StateRestorationEvidence,
    TerminalTaskEvidence,
    TerminalTaskStatus,
)
from latentguard.replay.validation import validate_replay_trust_claim

_DIGEST = "sha256:" + "a" * 64


def _restoration(
    *,
    expected: str = _DIGEST,
    observed: str = _DIGEST,
    semantic: StateComparisonSemantic = StateComparisonSemantic.EXACT_DIGEST,
    tolerance: float = 0.0,
    maximum_error: float | None = None,
    match_kind: StateMatchKind = StateMatchKind.EXACT,
    verified: bool = True,
) -> StateRestorationEvidence:
    return StateRestorationEvidence(
        expected_state_digest=expected,
        observed_state_digest=observed,
        comparison_semantic=semantic,
        comparison_tolerance=tolerance,
        compared_component_count=3,
        maximum_absolute_error=maximum_error,
        match_kind=match_kind,
        restoration_verified=verified,
    )


def _execution(*, complete: bool = True) -> ActionExecutionEvidence:
    return ActionExecutionEvidence(
        requested_step_count=2,
        executed_step_count=2 if complete else 1,
        complete=complete,
        termination_reason=None if complete else "controlled partial execution",
    )


def _task(
    *,
    status: TerminalTaskStatus = TerminalTaskStatus.COMPLETE,
    success: bool | None = True,
    progress: float | None = 1.0,
    unsafe: bool | None = False,
) -> TerminalTaskEvidence:
    return TerminalTaskEvidence(
        status=status,
        success=success,
        progress=progress,
        unsafe=unsafe,
        termination_reason=(
            None
            if status is TerminalTaskStatus.COMPLETE
            else "task evidence is intentionally incomplete"
        ),
    )


def _paired_result(
    *,
    status: EvaluationStatus = EvaluationStatus.CONCLUSIVE,
    corrupted_terminal: TerminalTaskEvidence | None = None,
) -> PairedReplayResult:
    terminal = corrupted_terminal or _task(success=False, progress=0.4)
    return PairedReplayResult(
        replay_case_id="rpc-sha256-" + "b" * 64,
        proposal_id="proposal-0",
        status=status,
        termination_reason="paired replay complete",
        baseline_restoration=_restoration(),
        baseline_initial_task=_task(success=False, progress=0.0),
        baseline_execution=_execution(),
        baseline_terminal_task=_task(success=True, progress=1.0),
        corrupted_restoration=_restoration(),
        corrupted_initial_task=_task(success=False, progress=0.0),
        corrupted_execution=_execution(),
        corrupted_terminal_task=terminal,
        replayed_control_steps=4,
    )


def _descriptor(tier: ReplayTrustTier) -> ReplayTrustDescriptor:
    if tier is ReplayTrustTier.EXACT_SIMULATOR:
        return ReplayTrustDescriptor(
            trust_tier=tier,
            label_source=LabelSource.SIMULATOR,
            maximum_label_strength=LabelStrength.STRONG,
            simulator_verification_allowed=True,
            exact_state_verification_required=True,
        )
    return ReplayTrustDescriptor(
        trust_tier=tier,
        label_source=LabelSource.DETERMINISTIC_EVALUATOR,
        maximum_label_strength=LabelStrength.WEAK,
        simulator_verification_allowed=False,
        exact_state_verification_required=True,
    )


def test_exact_and_numeric_restoration_semantics_are_strict() -> None:
    assert _restoration().restoration_verified
    mismatch = _restoration(
        observed="sha256:" + "b" * 64,
        maximum_error=1.0,
        match_kind=StateMatchKind.MISMATCH,
        verified=False,
    )
    assert not mismatch.restoration_verified
    tolerance_match = _restoration(
        observed="sha256:" + "b" * 64,
        semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
        tolerance=0.01,
        maximum_error=0.005,
        match_kind=StateMatchKind.WITHIN_TOLERANCE,
    )
    assert tolerance_match.restoration_verified

    with pytest.raises(ReplayValidationError, match="requires zero"):
        _restoration(tolerance=0.01)
    with pytest.raises(ReplayValidationError, match="exceeds comparison tolerance"):
        _restoration(
            observed="sha256:" + "b" * 64,
            semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
            tolerance=0.01,
            maximum_error=0.02,
            match_kind=StateMatchKind.WITHIN_TOLERANCE,
        )
    with pytest.raises(ReplayValidationError, match="must be true exactly"):
        _restoration(verified=False)


@pytest.mark.parametrize(
    ("status", "success", "progress", "unsafe"),
    [
        (TerminalTaskStatus.COMPLETE, None, 1.0, False),
        (TerminalTaskStatus.COMPLETE, True, None, False),
        (TerminalTaskStatus.COMPLETE, True, 1.0, None),
        (TerminalTaskStatus.INDETERMINATE, True, 1.0, False),
    ],
)
def test_terminal_task_status_never_manufactures_missing_values(
    status: TerminalTaskStatus,
    success: bool | None,
    progress: float | None,
    unsafe: bool | None,
) -> None:
    with pytest.raises(ReplayValidationError):
        _task(status=status, success=success, progress=progress, unsafe=unsafe)


def test_indeterminate_terminal_evidence_preserves_partial_values() -> None:
    evidence = _task(
        status=TerminalTaskStatus.INDETERMINATE,
        success=None,
        progress=0.4,
        unsafe=False,
    )
    assert evidence.success is None
    assert evidence.progress == 0.4


def test_paired_result_requires_baseline_success_and_two_verified_restorations() -> (
    None
):
    conclusive_failure = _paired_result()
    assert conclusive_failure.status is EvaluationStatus.CONCLUSIVE
    assert conclusive_failure.corrupted_terminal_task is not None
    assert conclusive_failure.corrupted_terminal_task.success is False

    with pytest.raises(ReplayValidationError, match="valid baseline"):
        PairedReplayResult(
            replay_case_id=conclusive_failure.replay_case_id,
            proposal_id=conclusive_failure.proposal_id,
            status=EvaluationStatus.CONCLUSIVE,
            termination_reason="invalid baseline",
            baseline_restoration=_restoration(),
            baseline_initial_task=_task(success=False, progress=0.0),
            baseline_execution=_execution(),
            baseline_terminal_task=_task(success=False, progress=0.5),
            replayed_control_steps=2,
        )


def test_baseline_task_failure_is_valid_invalid_context_not_corrupted_failure() -> None:
    result = PairedReplayResult(
        replay_case_id="rpc-sha256-" + "b" * 64,
        proposal_id="proposal-0",
        status=EvaluationStatus.INVALID,
        termination_reason="baseline task did not succeed",
        baseline_restoration=_restoration(),
        baseline_initial_task=_task(success=False, progress=0.0),
        baseline_execution=_execution(),
        baseline_terminal_task=_task(success=False, progress=0.5),
        replayed_control_steps=2,
    )
    assert result.status is EvaluationStatus.INVALID


def test_close_only_failure_can_remain_execution_error_after_complete_phases() -> None:
    complete = _paired_result()
    result = PairedReplayResult(
        replay_case_id=complete.replay_case_id,
        proposal_id=complete.proposal_id,
        status=EvaluationStatus.EXECUTION_ERROR,
        termination_reason="execution_error:corrupted_close:FixtureCloseError",
        baseline_restoration=complete.baseline_restoration,
        baseline_initial_task=complete.baseline_initial_task,
        baseline_execution=complete.baseline_execution,
        baseline_terminal_task=complete.baseline_terminal_task,
        corrupted_restoration=complete.corrupted_restoration,
        corrupted_initial_task=complete.corrupted_initial_task,
        corrupted_execution=complete.corrupted_execution,
        corrupted_terminal_task=complete.corrupted_terminal_task,
        replayed_control_steps=4,
        diagnostics={"replay_error_phase": "corrupted_close"},
    )
    assert result.status is EvaluationStatus.EXECUTION_ERROR


@pytest.mark.parametrize(
    "tier",
    [
        ReplayTrustTier.FIXTURE,
        ReplayTrustTier.DETERMINISTIC_NON_SIMULATOR,
        ReplayTrustTier.EXACT_SIMULATOR,
    ],
)
def test_valid_trust_descriptors_are_explicit(tier: ReplayTrustTier) -> None:
    assert _descriptor(tier).trust_tier is tier


@pytest.mark.parametrize(
    "updates",
    [
        {"label_source": LabelSource.SIMULATOR},
        {"maximum_label_strength": LabelStrength.STRONG},
        {"simulator_verification_allowed": True},
        {"exact_state_verification_required": False},
    ],
)
def test_fixture_descriptor_rejects_forged_authority(
    updates: dict[str, object],
) -> None:
    fields: dict[str, object] = {
        "trust_tier": ReplayTrustTier.FIXTURE,
        "label_source": LabelSource.DETERMINISTIC_EVALUATOR,
        "maximum_label_strength": LabelStrength.WEAK,
        "simulator_verification_allowed": False,
        "exact_state_verification_required": True,
    }
    fields.update(updates)
    with pytest.raises(ReplayValidationError):
        ReplayTrustDescriptor(**fields)  # type: ignore[arg-type]


def test_fixture_claim_is_weak_unverified_and_forged_claims_are_rejected() -> None:
    descriptor = _descriptor(ReplayTrustTier.FIXTURE)
    validate_replay_trust_claim(
        descriptor,
        label_source=LabelSource.DETERMINISTIC_EVALUATOR,
        label_strength=LabelStrength.WEAK,
        simulator_replay_verified=False,
        replay_result=_paired_result(),
    )
    with pytest.raises(ReplayValidationError):
        validate_replay_trust_claim(
            descriptor,
            label_source=LabelSource.SIMULATOR,
            label_strength=LabelStrength.STRONG,
            simulator_replay_verified=True,
            replay_result=_paired_result(),
        )


def test_exact_simulator_trust_still_requires_all_replay_gates() -> None:
    descriptor = _descriptor(ReplayTrustTier.EXACT_SIMULATOR)
    conclusive_task_failure = _paired_result()
    validate_replay_trust_claim(
        descriptor,
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
        replay_result=conclusive_task_failure,
    )
    with pytest.raises(ReplayValidationError, match="paired result"):
        validate_replay_trust_claim(
            descriptor,
            label_source=LabelSource.SIMULATOR,
            label_strength=LabelStrength.STRONG,
            simulator_replay_verified=True,
        )


def test_exact_simulator_trust_rejects_tolerance_state_matches() -> None:
    descriptor = _descriptor(ReplayTrustTier.EXACT_SIMULATOR)
    tolerance_match = _restoration(
        semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
        tolerance=1e-3,
        maximum_error=1e-4,
        match_kind=StateMatchKind.WITHIN_TOLERANCE,
    )
    result = replace(
        _paired_result(),
        baseline_restoration=tolerance_match,
        corrupted_restoration=tolerance_match,
    )

    with pytest.raises(ReplayValidationError, match="exact state restorations"):
        validate_replay_trust_claim(
            descriptor,
            label_source=LabelSource.SIMULATOR,
            label_strength=LabelStrength.STRONG,
            simulator_replay_verified=True,
            replay_result=result,
        )


def test_paired_result_rejects_mixed_restoration_comparison_contracts() -> None:
    numeric_exact = _restoration(
        semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
        tolerance=1e-3,
        maximum_error=0.0,
        match_kind=StateMatchKind.EXACT,
    )

    with pytest.raises(ReplayValidationError, match="comparison contract"):
        replace(_paired_result(), corrupted_restoration=numeric_exact)


@dataclass
class _ProtocolSession:
    def restore_state(
        self, reference: ReplayStateReference
    ) -> StateRestorationEvidence:
        del reference
        return _restoration()

    def evaluate_task(self, task: ReplayTaskReference) -> TerminalTaskEvidence:
        del task
        return _task()

    def step_action(self, action: np.ndarray[tuple[int], np.dtype[np.float32]]) -> None:
        del action

    def close(self) -> None:
        pass


def test_environment_protocol_is_structural_and_simulator_independent() -> None:
    assert isinstance(_ProtocolSession(), ReplayEnvironmentSession)
