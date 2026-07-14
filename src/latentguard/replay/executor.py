"""Generic independent-session exact-state paired replay execution."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.models import (
    CorruptedActionProposal,
    validate_corrupted_action_proposal,
)
from latentguard.evaluation.models import EvaluationStatus, compute_configuration_digest
from latentguard.models import ActionChunk
from latentguard.replay.base import (
    ExactReplayAdapter,
    ReplayEnvironmentSession,
    ReplayExecutionError,
    ReplayInvalidContextError,
)
from latentguard.replay.identity import canonical_json_value
from latentguard.replay.models import (
    ActionExecutionEvidence,
    PairedReplayResult,
    ReplayBundle,
    ReplayCase,
    ReplayExecutionRole,
    ReplayTrustDescriptor,
    StateMatchKind,
    StateRestorationEvidence,
    TerminalTaskEvidence,
    TerminalTaskStatus,
)
from latentguard.replay.validation import (
    validate_replay_bundle,
    validate_replay_case,
    validate_replay_trust_descriptor,
)


class PairedReplayExecutionError(ReplayExecutionError):
    """Raised when paired replay cannot safely produce a structured result."""


@dataclass(frozen=True, slots=True)
class _SessionRecord:
    session: ReplayEnvironmentSession | None = None
    restoration: StateRestorationEvidence | None = None
    initial_task: TerminalTaskEvidence | None = None
    execution: ActionExecutionEvidence | None = None
    terminal_task: TerminalTaskEvidence | None = None


@dataclass(frozen=True, slots=True)
class _AdapterExecutionContract:
    adapter_id: str
    adapter_version: str
    configuration_digest: str
    trust_descriptor: ReplayTrustDescriptor


class _SessionFailure(Exception):
    def __init__(
        self,
        *,
        phase: str,
        error: Exception,
        record: _SessionRecord,
        close_also_failed: bool = False,
    ) -> None:
        self.phase = phase
        self.error_type = _safe_error_type(type(error).__name__)
        self.record = record
        self.close_also_failed = close_also_failed
        super().__init__(f"{phase}:{self.error_type}")


def _safe_error_type(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:128] or "ReplayError"


def _validate_adapter_execution_contract(
    adapter: ExactReplayAdapter,
    replay_case: ReplayCase,
    replay_bundle: ReplayBundle,
    *,
    expected: _AdapterExecutionContract | None = None,
) -> _AdapterExecutionContract:
    """Bind direct executor use to one validated bundle and stable adapter contract."""

    try:
        validate_replay_case(replay_case)
        validate_replay_bundle(replay_bundle)
        matching_cases = tuple(
            candidate
            for candidate in replay_bundle.replay_cases
            if candidate.proposal_id == replay_case.proposal_id
        )
        if len(matching_cases) != 1 or matching_cases[0].case_id != replay_case.case_id:
            raise PairedReplayExecutionError(
                "replay case is not the content-bound case in the supplied bundle"
            )
        for actual, required, field in (
            (adapter.adapter_id, replay_bundle.adapter_id, "adapter_id"),
            (adapter.adapter_version, replay_bundle.adapter_version, "adapter_version"),
            (
                adapter.configuration_digest,
                replay_bundle.adapter_configuration_digest,
                "configuration_digest",
            ),
        ):
            if actual != required:
                raise PairedReplayExecutionError(
                    f"adapter {field} does not match the supplied replay bundle"
                )
        configuration = adapter.resolved_configuration()
        if not isinstance(configuration, Mapping):
            raise PairedReplayExecutionError(
                "adapter resolved configuration must be a mapping"
            )
        canonical_json_value(
            configuration,
            context="ExactReplayAdapter.resolved_configuration",
            reject_runtime_paths=True,
        )
        if (
            compute_configuration_digest(configuration)
            != replay_bundle.adapter_configuration_digest
        ):
            raise PairedReplayExecutionError(
                "adapter resolved configuration does not match the supplied "
                "replay bundle"
            )
        trust = adapter.trust_descriptor()
        validate_replay_trust_descriptor(trust)
        current = _AdapterExecutionContract(
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            configuration_digest=adapter.configuration_digest,
            trust_descriptor=trust,
        )
    except PairedReplayExecutionError:
        raise
    except Exception as exc:
        raise PairedReplayExecutionError(
            "adapter execution contract validation failed: "
            f"{_safe_error_type(type(exc).__name__)}"
        ) from exc
    if expected is not None and current != expected:
        raise PairedReplayExecutionError(
            "adapter execution contract changed during paired replay"
        )
    return current


def _validate_proposal_case_binding(
    proposal: CorruptedActionProposal, replay_case: ReplayCase
) -> None:
    """Reject an adapter that resolves a proposal to different source/action data."""

    try:
        validate_corrupted_action_proposal(proposal)
    except (TypeError, ValueError) as exc:
        raise PairedReplayExecutionError(
            "input corruption proposal failed validation"
        ) from exc
    mismatches = tuple(
        field
        for field, actual, expected in (
            ("proposal_id", replay_case.proposal_id, proposal.proposal_id),
            (
                "source_episode_id",
                replay_case.source_episode_id,
                proposal.source_episode_id,
            ),
            (
                "source_candidate_id",
                replay_case.source_candidate_id,
                proposal.source_candidate_id,
            ),
            ("split_group_id", replay_case.split_group_id, proposal.split_group_id),
            (
                "source_task_id",
                replay_case.task_reference.task_id,
                proposal.source_task_id,
            ),
        )
        if actual != expected
    )
    if mismatches:
        raise PairedReplayExecutionError(
            "resolved replay case does not match input proposal fields: "
            + ", ".join(mismatches)
        )
    if _action_snapshot(replay_case.transformed_action) != _action_snapshot(
        proposal.transformed_action
    ):
        raise PairedReplayExecutionError(
            "resolved replay case transformed action does not match input proposal"
        )


def _action_snapshot(action: ActionChunk) -> tuple[object, ...]:
    return (
        action.actions.dtype.str,
        action.actions.shape,
        action.actions.tobytes(order="C"),
        action.coordinate_frame,
        action.control_period_s,
        action.schema_version,
    )


def _detached_row(row: NDArray[Any]) -> NDArray[Any]:
    detached = np.array(row, copy=True, order="C", subok=False)
    immutable = detached.tobytes(order="C")
    return np.frombuffer(immutable, dtype=detached.dtype).reshape(detached.shape)


def _restoration_matches(evidence: StateRestorationEvidence) -> bool:
    return evidence.restoration_verified and evidence.match_kind in {
        StateMatchKind.EXACT,
        StateMatchKind.WITHIN_TOLERANCE,
    }


def _check_restoration_contract(
    replay_case: ReplayCase, evidence: object
) -> StateRestorationEvidence:
    if not isinstance(evidence, StateRestorationEvidence):
        raise PairedReplayExecutionError(
            "restore_state did not return StateRestorationEvidence"
        )
    if (
        evidence.expected_state_digest
        != replay_case.state_reference.expected_state_digest
    ):
        raise PairedReplayExecutionError(
            "restoration evidence expected digest does not match replay case"
        )
    if (
        evidence.comparison_semantic
        is not replay_case.state_reference.comparison_semantic
    ):
        raise PairedReplayExecutionError(
            "restoration evidence comparison semantic does not match replay case"
        )
    return evidence


def _check_task_evidence(evidence: object) -> TerminalTaskEvidence:
    if not isinstance(evidence, TerminalTaskEvidence):
        raise PairedReplayExecutionError(
            "evaluate_task did not return TerminalTaskEvidence"
        )
    return evidence


def _close_after_error(
    session: ReplayEnvironmentSession | None, primary: BaseException
) -> bool:
    if session is None:
        return False
    try:
        session.close()
    except BaseException as close_error:
        if isinstance(primary, Exception) and not isinstance(close_error, Exception):
            close_error.add_note(
                "Replay operation also failed before close; details were intentionally "
                f"redacted ({_safe_error_type(type(primary).__name__)})."
            )
            raise close_error from primary
        primary.add_note(
            "Replay session close also failed; details were intentionally redacted "
            f"({_safe_error_type(type(close_error).__name__)})."
        )
        return True
    return False


def _run_session(
    adapter: ExactReplayAdapter,
    replay_case: ReplayCase,
    *,
    role: ReplayExecutionRole,
    action: ActionChunk,
    forbidden_session: ReplayEnvironmentSession | None,
) -> _SessionRecord:
    session: ReplayEnvironmentSession | None = None
    record = _SessionRecord()
    phase = f"{role.value}_create_session"
    try:
        created = adapter.create_session(replay_case, execution_role=role)
        if not isinstance(created, ReplayEnvironmentSession):
            raise PairedReplayExecutionError(
                "create_session did not return ReplayEnvironmentSession"
            )
        if forbidden_session is not None and created is forbidden_session:
            raise PairedReplayExecutionError(
                "baseline and corrupted sessions must be distinct objects"
            )
        session = created
        record = replace(record, session=session)

        phase = f"{role.value}_restore_state"
        restoration = _check_restoration_contract(
            replay_case, session.restore_state(replay_case.state_reference)
        )
        record = replace(record, restoration=restoration)
        if _restoration_matches(restoration):
            phase = f"{role.value}_evaluate_initial"
            initial = _check_task_evidence(
                session.evaluate_task(replay_case.task_reference)
            )
            record = replace(record, initial_task=initial)

            requested = int(action.actions.shape[0])
            executed = 0
            phase = f"{role.value}_step_action"
            try:
                for row in action.actions:
                    session.step_action(_detached_row(row))
                    executed += 1
            except Exception:
                record = replace(
                    record,
                    execution=ActionExecutionEvidence(
                        requested_step_count=requested,
                        executed_step_count=executed,
                        complete=False,
                        termination_reason="step_action_exception",
                    ),
                )
                raise
            record = replace(
                record,
                execution=ActionExecutionEvidence(
                    requested_step_count=requested,
                    executed_step_count=executed,
                    complete=True,
                    termination_reason="action_sequence_complete",
                ),
            )

            phase = f"{role.value}_evaluate_terminal"
            terminal = _check_task_evidence(
                session.evaluate_task(replay_case.task_reference)
            )
            record = replace(record, terminal_task=terminal)
    except BaseException as primary:
        close_failed = _close_after_error(session, primary)
        if not isinstance(primary, Exception):
            raise
        raise _SessionFailure(
            phase=phase,
            error=primary,
            record=record,
            close_also_failed=close_failed,
        ) from primary

    phase = f"{role.value}_close"
    try:
        session.close()
    except BaseException as close_error:
        if not isinstance(close_error, Exception):
            raise
        raise _SessionFailure(
            phase=phase,
            error=close_error,
            record=record,
        ) from close_error
    return record


def _result(
    replay_case: ReplayCase,
    *,
    status: EvaluationStatus,
    termination_reason: str,
    baseline: _SessionRecord | None = None,
    corrupted: _SessionRecord | None = None,
    baseline_valid: bool | None = None,
    error_phase: str | None = None,
    error_type: str | None = None,
    close_also_failed: bool = False,
) -> PairedReplayResult:
    baseline = baseline or _SessionRecord()
    corrupted = corrupted or _SessionRecord()
    baseline_steps = (
        0 if baseline.execution is None else baseline.execution.executed_step_count
    )
    corrupted_steps = (
        0 if corrupted.execution is None else corrupted.execution.executed_step_count
    )
    diagnostics: dict[str, str | int | float | bool | None] = {}
    if baseline_valid is not None:
        diagnostics["replay_baseline_valid"] = baseline_valid
    if error_phase is not None:
        diagnostics["replay_error_phase"] = error_phase
    if error_type is not None:
        diagnostics["replay_error_type"] = error_type
    if close_also_failed:
        diagnostics["replay_close_also_failed"] = True
    return PairedReplayResult(
        replay_case_id=replay_case.case_id,
        proposal_id=replay_case.proposal_id,
        status=status,
        termination_reason=termination_reason,
        baseline_restoration=baseline.restoration,
        baseline_initial_task=baseline.initial_task,
        baseline_execution=baseline.execution,
        baseline_terminal_task=baseline.terminal_task,
        corrupted_restoration=corrupted.restoration,
        corrupted_initial_task=corrupted.initial_task,
        corrupted_execution=corrupted.execution,
        corrupted_terminal_task=corrupted.terminal_task,
        replayed_control_steps=baseline_steps + corrupted_steps,
        diagnostics=diagnostics,
    )


def _execution_error_result(
    replay_case: ReplayCase,
    failure: _SessionFailure,
    *,
    baseline: _SessionRecord | None = None,
) -> PairedReplayResult:
    role_is_baseline = failure.phase.startswith("baseline_")
    return _result(
        replay_case,
        status=EvaluationStatus.EXECUTION_ERROR,
        termination_reason=f"execution_error:{failure.phase}:{failure.error_type}",
        baseline=failure.record if role_is_baseline else baseline,
        corrupted=None if role_is_baseline else failure.record,
        baseline_valid=(None if role_is_baseline else True),
        error_phase=failure.phase,
        error_type=failure.error_type,
        close_also_failed=failure.close_also_failed,
    )


def _validate_static_action_contract(replay_case: ReplayCase) -> str | None:
    original = replay_case.original_action
    transformed = replay_case.transformed_action
    if original.actions.shape != transformed.actions.shape:
        return "static_action_shape_mismatch"
    if original.actions.dtype != transformed.actions.dtype:
        return "static_action_dtype_mismatch"
    if original.control_period_s != transformed.control_period_s:
        return "static_action_control_period_mismatch"
    if original.coordinate_frame != transformed.coordinate_frame:
        return "static_action_coordinate_frame_mismatch"
    if original.schema_version != transformed.schema_version:
        return "static_action_schema_mismatch"
    return None


def _execute_replay_case(
    adapter: ExactReplayAdapter,
    replay_case: ReplayCase,
    replay_bundle: ReplayBundle,
) -> PairedReplayResult:
    """Execute one bundle-bound case in independent baseline/corrupted sessions."""
    original_snapshot = _action_snapshot(replay_case.original_action)
    transformed_snapshot = _action_snapshot(replay_case.transformed_action)

    static_reason = _validate_static_action_contract(replay_case)
    if static_reason is not None:
        return _result(
            replay_case,
            status=EvaluationStatus.INVALID,
            termination_reason=static_reason,
            baseline_valid=False,
        )
    adapter_contract = _validate_adapter_execution_contract(
        adapter, replay_case, replay_bundle
    )
    try:
        baseline = _run_session(
            adapter,
            replay_case,
            role=ReplayExecutionRole.BASELINE,
            action=replay_case.original_action,
            forbidden_session=None,
        )
    except _SessionFailure as failure:
        _validate_adapter_execution_contract(
            adapter, replay_case, replay_bundle, expected=adapter_contract
        )
        return _execution_error_result(replay_case, failure)
    _validate_adapter_execution_contract(
        adapter, replay_case, replay_bundle, expected=adapter_contract
    )

    if baseline.restoration is None or not _restoration_matches(baseline.restoration):
        result = _result(
            replay_case,
            status=EvaluationStatus.INVALID,
            termination_reason="baseline_state_restoration_mismatch",
            baseline=baseline,
            baseline_valid=False,
        )
    elif (
        baseline.execution is None
        or not baseline.execution.complete
        or baseline.execution.executed_step_count
        != baseline.execution.requested_step_count
    ):
        result = _result(
            replay_case,
            status=EvaluationStatus.EXECUTION_ERROR,
            termination_reason="execution_error:baseline_action_incomplete",
            baseline=baseline,
            error_phase="baseline_step_action",
            error_type="IncompleteActionExecution",
        )
    elif (
        baseline.terminal_task is None
        or baseline.terminal_task.status is not TerminalTaskStatus.COMPLETE
        or baseline.terminal_task.success is not True
    ):
        result = _result(
            replay_case,
            status=EvaluationStatus.INVALID,
            termination_reason="baseline_task_not_successful",
            baseline=baseline,
            baseline_valid=False,
        )
    else:
        try:
            corrupted = _run_session(
                adapter,
                replay_case,
                role=ReplayExecutionRole.CORRUPTED,
                action=replay_case.transformed_action,
                forbidden_session=baseline.session,
            )
        except _SessionFailure as failure:
            result = _execution_error_result(replay_case, failure, baseline=baseline)
        else:
            if corrupted.restoration is None or not _restoration_matches(
                corrupted.restoration
            ):
                result = _result(
                    replay_case,
                    status=EvaluationStatus.INVALID,
                    termination_reason="corrupted_state_restoration_mismatch",
                    baseline=baseline,
                    corrupted=corrupted,
                    baseline_valid=True,
                )
            elif (
                baseline.restoration.expected_state_digest
                != corrupted.restoration.expected_state_digest
                or baseline.restoration.comparison_semantic
                is not corrupted.restoration.comparison_semantic
                or baseline.restoration.comparison_tolerance
                != corrupted.restoration.comparison_tolerance
            ):
                result = _result(
                    replay_case,
                    status=EvaluationStatus.EXECUTION_ERROR,
                    termination_reason="execution_error:restoration_contract_mismatch",
                    baseline=baseline,
                    corrupted=corrupted,
                    baseline_valid=True,
                    error_phase="corrupted_restore_state",
                    error_type="RestorationContractMismatch",
                )
            elif (
                corrupted.execution is None
                or not corrupted.execution.complete
                or corrupted.execution.executed_step_count
                != corrupted.execution.requested_step_count
            ):
                result = _result(
                    replay_case,
                    status=EvaluationStatus.EXECUTION_ERROR,
                    termination_reason="execution_error:corrupted_action_incomplete",
                    baseline=baseline,
                    corrupted=corrupted,
                    baseline_valid=True,
                    error_phase="corrupted_step_action",
                    error_type="IncompleteActionExecution",
                )
            elif (
                corrupted.terminal_task is None
                or corrupted.terminal_task.status is TerminalTaskStatus.INDETERMINATE
            ):
                result = _result(
                    replay_case,
                    status=EvaluationStatus.INDETERMINATE,
                    termination_reason="corrupted_terminal_task_indeterminate",
                    baseline=baseline,
                    corrupted=corrupted,
                    baseline_valid=True,
                )
            else:
                result = _result(
                    replay_case,
                    status=EvaluationStatus.CONCLUSIVE,
                    termination_reason="paired_replay_complete",
                    baseline=baseline,
                    corrupted=corrupted,
                    baseline_valid=True,
                )

    _validate_adapter_execution_contract(
        adapter, replay_case, replay_bundle, expected=adapter_contract
    )
    if (
        _action_snapshot(replay_case.original_action) != original_snapshot
        or _action_snapshot(replay_case.transformed_action) != transformed_snapshot
    ):
        return _result(
            replay_case,
            status=EvaluationStatus.EXECUTION_ERROR,
            termination_reason="execution_error:action_mutation_detected",
            baseline_valid=None,
            error_phase="action_immutability_check",
            error_type="ActionMutationDetected",
        )
    return result


def execute_paired_replay(
    adapter: ExactReplayAdapter,
    proposal: CorruptedActionProposal,
    replay_bundle: ReplayBundle,
) -> PairedReplayResult:
    """Resolve one proposal and execute it against one content-bound bundle."""
    try:
        replay_case = adapter.resolve_case(proposal)
    except ReplayInvalidContextError:
        raise
    if not isinstance(replay_case, ReplayCase):
        raise PairedReplayExecutionError(
            "resolve_case did not return a validated ReplayCase"
        )
    try:
        validate_replay_case(replay_case)
    except (TypeError, ValueError) as exc:
        raise PairedReplayExecutionError(
            "resolved replay case failed validation"
        ) from exc
    _validate_proposal_case_binding(proposal, replay_case)
    return _execute_replay_case(adapter, replay_case, replay_bundle)


__all__ = [
    "PairedReplayExecutionError",
    "execute_paired_replay",
]
