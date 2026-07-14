"""Ordered, deterministic, resume-safe proposal evaluation runner."""

from __future__ import annotations

import platform as platform_module
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    validate_corruption_dataset,
)
from latentguard.evaluation.base import ApplicabilityDecision, ProposalEvaluator
from latentguard.evaluation.models import (
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EvaluationEvidence,
    EvaluationIdentityError,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
    validate_evaluator_identity,
)
from latentguard.evaluation.reporting import build_evaluation_summary
from latentguard.evaluation.security import (
    REDACTED_OPERATIONAL_TEXT,
    contains_sensitive_operational_content,
    sanitize_launch_command,
    sanitize_operational_text,
)
from latentguard.evaluation.serialization import (
    EvaluationDataset,
    LedgerEntry,
    LedgerState,
    RunEnvironment,
    RunManifest,
    RunState,
    compute_corruption_dataset_content_digest,
    compute_evaluation_seed,
    compute_run_identifier,
    load_evaluation_dataset,
    save_evaluation_dataset,
    update_evaluation_dataset,
)
from latentguard.evaluation.validation import validate_evaluation_evidence
from latentguard.models import JsonScalar

_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class EvaluationRunnerError(RuntimeError):
    """Raised when a run plan, evaluator result, or resume context conflicts."""


class EvaluationRunConflictError(EvaluationRunnerError):
    """Raised when resume inputs do not match the persisted deterministic run."""


class EvaluatorContractError(EvaluationRunnerError):
    """Raised internally when an evaluator returns malformed evidence."""


@dataclass(frozen=True, slots=True)
class PlannedAttempt:
    """One side-effect-free attempt identity shown by dry-run planning."""

    proposal_id: str
    attempt_ordinal: int
    evaluation_seed: int
    evidence_id: str


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Exact deterministic selection and attempt-zero identities for a run."""

    run_id: str
    source_corruption_dataset_digest: str
    source_dataset_id: str
    evaluator_id: str
    evaluator_version: str
    evaluator_configuration_digest: str
    resolved_evaluator_configuration: Mapping[str, object]
    base_seed: int
    max_proposals: int | None
    selected_proposal_ids: tuple[str, ...]
    attempts: tuple[PlannedAttempt, ...]


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Final persisted state and operational counts for one invocation."""

    dataset: EvaluationDataset
    resumed: bool
    evaluated_attempts: int
    recovered_attempts: int
    retried_attempts: int
    stopped_early: bool


def utc_timestamp() -> str:
    """Return one timezone-aware UTC timestamp for operational audit only."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_audit_timestamp(value: object) -> tuple[datetime, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationRunnerError(
            "audit clock must return a non-empty ISO-8601 timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvaluationRunnerError(
            "audit clock must return a valid ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise EvaluationRunnerError("audit clock timestamp must include a timezone")
    return parsed, value


def _next_audit_timestamp(
    clock: Callable[[], str],
    dataset: EvaluationDataset | None = None,
    minimum: str | None = None,
) -> str:
    """Return a valid nondecreasing audit timestamp despite wall-clock rollback."""
    candidate_time, candidate_text = _parse_audit_timestamp(clock())
    if dataset is None and minimum is None:
        return candidate_text
    bounds: list[str] = []
    if dataset is not None:
        bounds.append(dataset.run_manifest.started_at)
        if dataset.run_manifest.finished_at is not None:
            bounds.append(dataset.run_manifest.finished_at)
        for entry in dataset.ledger:
            if entry.started_at is not None:
                bounds.append(entry.started_at)
            if entry.finished_at is not None:
                bounds.append(entry.finished_at)
    if minimum is not None:
        bounds.append(minimum)
    latest_time, latest_text = max(
        (_parse_audit_timestamp(value) for value in bounds),
        key=lambda item: item[0],
    )
    return latest_text if candidate_time < latest_time else candidate_text


def _latest_audit_timestamp(dataset: EvaluationDataset) -> str:
    bounds = [dataset.run_manifest.started_at]
    if dataset.run_manifest.finished_at is not None:
        bounds.append(dataset.run_manifest.finished_at)
    for entry in dataset.ledger:
        if entry.started_at is not None:
            bounds.append(entry.started_at)
        if entry.finished_at is not None:
            bounds.append(entry.finished_at)
    return max(
        bounds,
        key=lambda value: _parse_audit_timestamp(value)[0],
    )


def collect_run_environment(
    launch_command: Sequence[str] = (), cwd: Path | None = None
) -> RunEnvironment:
    """Collect local runtime and optional Git metadata without using the network."""
    working_directory = Path.cwd() if cwd is None else Path(cwd)
    commit = _git_value(("rev-parse", "HEAD"), working_directory)
    if commit is not None and _FULL_SHA_RE.fullmatch(commit) is not None:
        commit = commit.lower()
    else:
        commit = None
    branch = _git_value(("branch", "--show-current"), working_directory)
    if branch is not None:
        branch = sanitize_operational_text(branch)
    return RunEnvironment(
        git_commit_sha=commit,
        git_branch=branch,
        python_version=platform_module.python_version(),
        numpy_version=np.__version__,
        platform=sanitize_operational_text(platform_module.platform()),
        launch_command=sanitize_launch_command(launch_command),
    )


def _git_value(arguments: Sequence[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def derive_evaluation_seed(
    *,
    base_seed: int,
    proposal_id: str,
    evaluator_id: str,
    evaluator_version: str,
    evaluator_configuration_digest: str,
    attempt_ordinal: int,
) -> int:
    """Public runner-level wrapper for deterministic attempt seed derivation."""
    return compute_evaluation_seed(
        base_seed=base_seed,
        proposal_id=proposal_id,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        evaluator_configuration_digest=evaluator_configuration_digest,
        attempt_ordinal=attempt_ordinal,
    )


def plan_evaluation(
    corruption_dataset: CorruptionDataset,
    evaluator: ProposalEvaluator,
    *,
    source_corruption_dataset_digest: str,
    base_seed: int,
    max_proposals: int | None = None,
) -> EvaluationPlan:
    """Validate inputs and compute an exact plan without evaluator calls or writes."""
    validate_corruption_dataset(corruption_dataset)
    actual_corruption_digest = compute_corruption_dataset_content_digest(
        corruption_dataset
    )
    if source_corruption_dataset_digest != actual_corruption_digest:
        raise EvaluationRunnerError(
            "source_corruption_dataset_digest does not match the supplied "
            "corruption dataset contents"
        )
    if max_proposals is not None and (
        type(max_proposals) is not int or max_proposals <= 0
    ):
        raise EvaluationRunnerError("max_proposals must be a positive integer or null")
    evaluator_id = _required_evaluator_text(evaluator.evaluator_id, "evaluator_id")
    evaluator_version = _required_evaluator_text(
        evaluator.evaluator_version, "evaluator_version"
    )
    try:
        validate_evaluator_identity(
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
        )
    except EvaluationIdentityError as exc:
        raise EvaluationRunnerError(str(exc)) from exc
    resolved = evaluator.resolved_configuration()
    if not isinstance(resolved, Mapping):
        raise EvaluationRunnerError(
            "evaluator.resolved_configuration must return a mapping"
        )
    configuration = MappingProxyType(dict(resolved))
    expected_digest = compute_configuration_digest(configuration)
    if evaluator.configuration_digest != expected_digest:
        raise EvaluationRunnerError(
            "evaluator configuration digest does not match resolved configuration"
        )
    proposals = (
        corruption_dataset.proposals
        if max_proposals is None
        else corruption_dataset.proposals[:max_proposals]
    )
    selected = tuple(proposal.proposal_id for proposal in proposals)
    run_id = compute_run_identifier(
        source_corruption_dataset_digest=source_corruption_dataset_digest,
        source_dataset_id=corruption_dataset.source_dataset_id,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        evaluator_configuration_digest=expected_digest,
        base_seed=base_seed,
        max_proposals=max_proposals,
        selected_proposal_ids=selected,
    )
    attempts: list[PlannedAttempt] = []
    for proposal in proposals:
        seed = derive_evaluation_seed(
            base_seed=base_seed,
            proposal_id=proposal.proposal_id,
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
            evaluator_configuration_digest=expected_digest,
            attempt_ordinal=0,
        )
        attempts.append(
            PlannedAttempt(
                proposal_id=proposal.proposal_id,
                attempt_ordinal=0,
                evaluation_seed=seed,
                evidence_id=compute_evidence_identifier(
                    proposal_id=proposal.proposal_id,
                    evaluator_id=evaluator_id,
                    evaluator_version=evaluator_version,
                    evaluator_configuration_digest=expected_digest,
                    evaluation_seed=seed,
                    attempt_ordinal=0,
                ),
            )
        )
    return EvaluationPlan(
        run_id=run_id,
        source_corruption_dataset_digest=source_corruption_dataset_digest,
        source_dataset_id=corruption_dataset.source_dataset_id,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        evaluator_configuration_digest=expected_digest,
        resolved_evaluator_configuration=configuration,
        base_seed=base_seed,
        max_proposals=max_proposals,
        selected_proposal_ids=selected,
        attempts=tuple(attempts),
    )


def run_evaluation(
    corruption_dataset: CorruptionDataset,
    evaluator: ProposalEvaluator,
    *,
    source_corruption_dataset_digest: str,
    output_dir: Path,
    base_seed: int,
    max_proposals: int | None = None,
    resume: bool = False,
    fail_fast: bool = False,
    retry_execution_errors: bool = False,
    run_environment: RunEnvironment | None = None,
    launch_command: Sequence[str] = (),
    clock: Callable[[], str] = utc_timestamp,
) -> EvaluationRunResult:
    """Create or resume an ordered run and persist every state transition."""
    if type(resume) is not bool or type(fail_fast) is not bool:
        raise EvaluationRunnerError("resume and fail_fast must be booleans")
    if type(retry_execution_errors) is not bool:
        raise EvaluationRunnerError("retry_execution_errors must be a boolean")
    plan = plan_evaluation(
        corruption_dataset,
        evaluator,
        source_corruption_dataset_digest=source_corruption_dataset_digest,
        base_seed=base_seed,
        max_proposals=max_proposals,
    )
    proposal_by_id = {
        proposal.proposal_id: proposal
        for proposal in corruption_dataset.proposals[: len(plan.selected_proposal_ids)]
    }
    environment = run_environment or collect_run_environment(launch_command)
    if resume:
        dataset = load_evaluation_dataset(
            output_dir,
            corruption_dataset=corruption_dataset,
            expected_corruption_digest=source_corruption_dataset_digest,
        )
        _assert_resume_compatible(dataset, plan)
    else:
        dataset = _initial_dataset(plan, environment, _next_audit_timestamp(clock))
        save_evaluation_dataset(dataset, output_dir)
    audit_floor = _latest_audit_timestamp(dataset)

    retried_this_invocation = 0
    if resume and retry_execution_errors:
        dataset, retried_this_invocation = _append_explicit_retries(dataset)
        if retried_this_invocation:
            dataset = _with_run_state(dataset, RunState.IN_PROGRESS, None)
            update_evaluation_dataset(dataset, output_dir)

    work_keys = tuple(
        (entry.proposal_id, entry.attempt_ordinal)
        for entry in dataset.ledger
        if entry.state in {LedgerState.PENDING, LedgerState.RUNNING}
    )
    if not work_keys:
        if dataset.run_state is not RunState.COMPLETE:
            dataset = _with_run_state(
                dataset,
                RunState.COMPLETE,
                _next_audit_timestamp(clock, dataset, audit_floor),
            )
            update_evaluation_dataset(dataset, output_dir)
        return EvaluationRunResult(
            dataset=dataset,
            resumed=resume,
            evaluated_attempts=0,
            recovered_attempts=0,
            retried_attempts=retried_this_invocation,
            stopped_early=False,
        )

    evaluated_attempts = 0
    recovered_attempts = 0
    try:
        for key in work_keys:
            current = _ledger_entry(dataset, key)
            if current.state is LedgerState.RUNNING:
                running = current
                recovered_attempts += 1
                if dataset.run_state is not RunState.IN_PROGRESS:
                    dataset = _with_run_state(dataset, RunState.IN_PROGRESS, None)
                    update_evaluation_dataset(dataset, output_dir)
            else:
                started_at = _next_audit_timestamp(clock, dataset, audit_floor)
                audit_floor = started_at
                running = replace(
                    current,
                    evidence_id=None,
                    state=LedgerState.RUNNING,
                    error_type=None,
                    error_message=None,
                    retry_eligible=False,
                    started_at=started_at,
                    finished_at=None,
                )
                dataset = _replace_ledger(dataset, running)
                dataset = _with_run_state(dataset, RunState.IN_PROGRESS, None)
                update_evaluation_dataset(dataset, output_dir)

            proposal = proposal_by_id[running.proposal_id]
            evidence, error_type, error_message = _evaluate_attempt(
                evaluator,
                proposal,
                source_dataset_id=dataset.source_dataset_id,
                evaluation_seed=running.evaluation_seed,
                attempt_ordinal=running.attempt_ordinal,
            )
            evaluated_attempts += 1
            ledger_state = _ledger_state_for(evidence.status)
            finished_at = _next_audit_timestamp(clock, dataset, audit_floor)
            audit_floor = finished_at
            terminal = replace(
                running,
                evidence_id=evidence.evidence_id,
                state=ledger_state,
                error_type=error_type,
                error_message=error_message,
                retry_eligible=ledger_state is LedgerState.EXECUTION_ERROR,
                finished_at=finished_at,
            )
            dataset = _append_evidence_and_replace_ledger(dataset, evidence, terminal)
            unfinished = any(
                entry.state in {LedgerState.PENDING, LedgerState.RUNNING}
                for entry in dataset.ledger
            )
            stopped_early = fail_fast and (ledger_state is LedgerState.EXECUTION_ERROR)
            if stopped_early:
                dataset = _with_run_state(dataset, RunState.INTERRUPTED, finished_at)
            elif not unfinished:
                dataset = _with_run_state(dataset, RunState.COMPLETE, finished_at)
            else:
                dataset = _with_run_state(dataset, RunState.IN_PROGRESS, None)
            update_evaluation_dataset(dataset, output_dir)
            if stopped_early:
                return EvaluationRunResult(
                    dataset=dataset,
                    resumed=resume,
                    evaluated_attempts=evaluated_attempts,
                    recovered_attempts=recovered_attempts,
                    retried_attempts=retried_this_invocation,
                    stopped_early=True,
                )
    except KeyboardInterrupt as interruption:
        try:
            persisted = load_evaluation_dataset(
                output_dir,
                corruption_dataset=corruption_dataset,
                expected_corruption_digest=source_corruption_dataset_digest,
            )
            if persisted.run_state not in {RunState.COMPLETE, RunState.INTERRUPTED}:
                interrupted = _with_run_state(
                    persisted,
                    RunState.INTERRUPTED,
                    _next_audit_timestamp(clock, persisted, audit_floor),
                )
                update_evaluation_dataset(interrupted, output_dir)
        except Exception as persistence_error:
            interruption.add_note(
                "Could not persist interruption state; the last atomic manifest "
                f"remains authoritative ({type(persistence_error).__name__})."
            )
        raise

    reloaded = load_evaluation_dataset(
        output_dir,
        corruption_dataset=corruption_dataset,
        expected_corruption_digest=source_corruption_dataset_digest,
    )
    return EvaluationRunResult(
        dataset=reloaded,
        resumed=resume,
        evaluated_attempts=evaluated_attempts,
        recovered_attempts=recovered_attempts,
        retried_attempts=retried_this_invocation,
        stopped_early=False,
    )


def _initial_dataset(
    plan: EvaluationPlan, environment: RunEnvironment, started_at: str
) -> EvaluationDataset:
    ledger = tuple(
        LedgerEntry(
            proposal_id=attempt.proposal_id,
            evidence_id=None,
            attempt_ordinal=attempt.attempt_ordinal,
            evaluator_id=plan.evaluator_id,
            evaluator_version=plan.evaluator_version,
            evaluator_configuration_digest=plan.evaluator_configuration_digest,
            evaluation_seed=attempt.evaluation_seed,
            state=LedgerState.PENDING,
            error_type=None,
            error_message=None,
            retry_eligible=False,
            started_at=None,
            finished_at=None,
        )
        for attempt in plan.attempts
    )
    state = RunState.IN_PROGRESS if ledger else RunState.COMPLETE
    finished_at = None if ledger else started_at
    manifest = RunManifest(
        environment=environment,
        evaluator_id=plan.evaluator_id,
        evaluator_version=plan.evaluator_version,
        evaluator_configuration_digest=plan.evaluator_configuration_digest,
        source_corruption_dataset_digest=plan.source_corruption_dataset_digest,
        source_dataset_id=plan.source_dataset_id,
        seed=plan.base_seed,
        started_at=started_at,
        finished_at=finished_at,
        final_state=state,
    )
    summary = build_evaluation_summary((), ledger)
    return EvaluationDataset(
        source_corruption_dataset_digest=plan.source_corruption_dataset_digest,
        source_dataset_id=plan.source_dataset_id,
        evaluator_id=plan.evaluator_id,
        evaluator_version=plan.evaluator_version,
        resolved_evaluator_configuration=plan.resolved_evaluator_configuration,
        evaluator_configuration_digest=plan.evaluator_configuration_digest,
        run_id=plan.run_id,
        base_seed=plan.base_seed,
        max_proposals=plan.max_proposals,
        selected_proposal_ids=plan.selected_proposal_ids,
        evidence=(),
        ledger=ledger,
        summary=summary,
        artifact_references=(),
        run_state=state,
        run_manifest=manifest,
    )


def _assert_resume_compatible(dataset: EvaluationDataset, plan: EvaluationPlan) -> None:
    comparisons = (
        (dataset.run_id, plan.run_id, "run_id"),
        (
            dataset.source_corruption_dataset_digest,
            plan.source_corruption_dataset_digest,
            "source_corruption_dataset_digest",
        ),
        (dataset.source_dataset_id, plan.source_dataset_id, "source_dataset_id"),
        (dataset.evaluator_id, plan.evaluator_id, "evaluator_id"),
        (dataset.evaluator_version, plan.evaluator_version, "evaluator_version"),
        (
            dataset.evaluator_configuration_digest,
            plan.evaluator_configuration_digest,
            "evaluator_configuration_digest",
        ),
        (dataset.base_seed, plan.base_seed, "base_seed"),
        (dataset.max_proposals, plan.max_proposals, "max_proposals"),
        (
            dataset.selected_proposal_ids,
            plan.selected_proposal_ids,
            "selected_proposal_ids",
        ),
    )
    mismatches = [
        field for actual, expected, field in comparisons if actual != expected
    ]
    if mismatches:
        raise EvaluationRunConflictError(
            "resume inputs conflict with persisted run: " + ", ".join(mismatches)
        )


def _append_explicit_retries(
    dataset: EvaluationDataset,
) -> tuple[EvaluationDataset, int]:
    entries = list(dataset.ledger)
    additions: list[LedgerEntry] = []
    for proposal_id in dataset.selected_proposal_ids:
        attempts = [entry for entry in entries if entry.proposal_id == proposal_id]
        latest = max(attempts, key=lambda entry: entry.attempt_ordinal)
        if latest.state is not LedgerState.EXECUTION_ERROR:
            continue
        ordinal = latest.attempt_ordinal + 1
        additions.append(
            LedgerEntry(
                proposal_id=proposal_id,
                evidence_id=None,
                attempt_ordinal=ordinal,
                evaluator_id=dataset.evaluator_id,
                evaluator_version=dataset.evaluator_version,
                evaluator_configuration_digest=dataset.evaluator_configuration_digest,
                evaluation_seed=derive_evaluation_seed(
                    base_seed=dataset.base_seed,
                    proposal_id=proposal_id,
                    evaluator_id=dataset.evaluator_id,
                    evaluator_version=dataset.evaluator_version,
                    evaluator_configuration_digest=dataset.evaluator_configuration_digest,
                    attempt_ordinal=ordinal,
                ),
                state=LedgerState.PENDING,
                error_type=None,
                error_message=None,
                retry_eligible=False,
                started_at=None,
                finished_at=None,
            )
        )
    if not additions:
        return dataset, 0
    order = {
        proposal_id: index
        for index, proposal_id in enumerate(dataset.selected_proposal_ids)
    }
    ledger = tuple(
        sorted(
            (*entries, *additions),
            key=lambda entry: (order[entry.proposal_id], entry.attempt_ordinal),
        )
    )
    manifest = replace(
        dataset.run_manifest,
        finished_at=None,
        final_state=RunState.IN_PROGRESS,
    )
    updated = replace(
        dataset,
        ledger=ledger,
        summary=build_evaluation_summary(dataset.evidence, ledger),
        run_state=RunState.IN_PROGRESS,
        run_manifest=manifest,
    )
    return updated, len(additions)


def _evaluate_attempt(
    evaluator: ProposalEvaluator,
    proposal: CorruptedActionProposal,
    *,
    source_dataset_id: str,
    evaluation_seed: int,
    attempt_ordinal: int,
) -> tuple[EvaluationEvidence, str | None, str | None]:
    source_snapshot = proposal.transformed_action.actions.tobytes(order="C")
    try:
        decision = evaluator.check_applicability(
            proposal, source_dataset_id=source_dataset_id
        )
        if not isinstance(decision, ApplicabilityDecision):
            raise EvaluatorContractError(
                "check_applicability did not return ApplicabilityDecision"
            )
        if decision.status is None:
            evidence = evaluator.evaluate(
                proposal,
                source_dataset_id=source_dataset_id,
                evaluation_seed=evaluation_seed,
                attempt_ordinal=attempt_ordinal,
            )
        else:
            evidence = _applicability_evidence(
                evaluator,
                proposal,
                source_dataset_id=source_dataset_id,
                evaluation_seed=evaluation_seed,
                attempt_ordinal=attempt_ordinal,
                decision=decision,
            )
        if proposal.transformed_action.actions.tobytes(order="C") != source_snapshot:
            raise EvaluatorContractError("evaluator mutated the source proposal action")
        _validate_returned_evidence(
            evidence,
            evaluator,
            proposal,
            source_dataset_id=source_dataset_id,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
        )
        if evidence.status is EvaluationStatus.EXECUTION_ERROR:
            evidence = _sanitize_returned_execution_error(evidence)
            return (
                evidence,
                "EvaluatorReportedExecutionError",
                sanitize_operational_text(
                    evidence.notes or evidence.termination_reason
                ),
            )
        return evidence, None, None
    except Exception as caught:
        error: Exception = caught
        if proposal.transformed_action.actions.tobytes(order="C") != source_snapshot:
            error = EvaluatorContractError(
                "evaluator mutated the source proposal action"
            )
        error_type = _sanitize_error_type(type(error).__name__)
        error_message = sanitize_operational_text(error)
        evidence = _execution_error_evidence(
            evaluator,
            proposal,
            source_dataset_id=source_dataset_id,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
            error_type=error_type,
            error_message=error_message,
        )
        return evidence, error_type, error_message


def _sanitize_returned_execution_error(
    evidence: EvaluationEvidence,
) -> EvaluationEvidence:
    sanitized_metrics: dict[str, JsonScalar] = {}
    for index, (key, value) in enumerate(sorted(evidence.metrics.items())):
        sensitive_key = contains_sensitive_operational_content(key)
        sanitized_key = f"redacted_metric_{index}" if sensitive_key else key
        sanitized_metrics[sanitized_key] = (
            REDACTED_OPERATIONAL_TEXT
            if sensitive_key
            else sanitize_operational_text(value)
            if isinstance(value, str)
            else value
        )
    sanitized = replace(
        evidence,
        termination_reason=sanitize_operational_text(evidence.termination_reason),
        metrics=sanitized_metrics,
        notes=(
            None
            if evidence.notes is None
            else sanitize_operational_text(evidence.notes)
        ),
    )
    validate_evaluation_evidence(sanitized)
    return sanitized


def _applicability_evidence(
    evaluator: ProposalEvaluator,
    proposal: CorruptedActionProposal,
    *,
    source_dataset_id: str,
    evaluation_seed: int,
    attempt_ordinal: int,
    decision: ApplicabilityDecision,
) -> EvaluationEvidence:
    if decision.status not in {EvaluationStatus.SKIPPED, EvaluationStatus.INVALID}:
        raise EvaluatorContractError("non-applicable decision has unsupported status")
    reason = sanitize_operational_text(decision.reason)
    evidence = _empty_task_evidence(
        evaluator,
        proposal,
        source_dataset_id=source_dataset_id,
        evaluation_seed=evaluation_seed,
        attempt_ordinal=attempt_ordinal,
        status=decision.status,
        termination_reason=reason,
        notes=reason,
    )
    validate_evaluation_evidence(evidence)
    return evidence


def _execution_error_evidence(
    evaluator: ProposalEvaluator,
    proposal: CorruptedActionProposal,
    *,
    source_dataset_id: str,
    evaluation_seed: int,
    attempt_ordinal: int,
    error_type: str,
    error_message: str,
) -> EvaluationEvidence:
    evidence = _empty_task_evidence(
        evaluator,
        proposal,
        source_dataset_id=source_dataset_id,
        evaluation_seed=evaluation_seed,
        attempt_ordinal=attempt_ordinal,
        status=EvaluationStatus.EXECUTION_ERROR,
        termination_reason=f"execution_error:{error_type}",
        notes=error_message,
    )
    validate_evaluation_evidence(evidence)
    return evidence


def _empty_task_evidence(
    evaluator: ProposalEvaluator,
    proposal: CorruptedActionProposal,
    *,
    source_dataset_id: str,
    evaluation_seed: int,
    attempt_ordinal: int,
    status: EvaluationStatus,
    termination_reason: str,
    notes: str,
) -> EvaluationEvidence:
    return EvaluationEvidence(
        evidence_id=compute_evidence_identifier(
            proposal_id=proposal.proposal_id,
            evaluator_id=evaluator.evaluator_id,
            evaluator_version=evaluator.evaluator_version,
            evaluator_configuration_digest=evaluator.configuration_digest,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
        ),
        proposal_id=proposal.proposal_id,
        source_dataset_id=source_dataset_id,
        source_episode_id=proposal.source_episode_id,
        source_candidate_id=proposal.source_candidate_id,
        split_group_id=proposal.split_group_id,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        evaluator_configuration_digest=evaluator.configuration_digest,
        evaluation_seed=evaluation_seed,
        attempt_ordinal=attempt_ordinal,
        status=status,
        success=None,
        progress_before=None,
        progress_after=None,
        progress_delta=None,
        unsafe=None,
        failure_events=(),
        termination_reason=termination_reason,
        replayed_control_steps=0,
        metrics={},
        artifact_references=(),
        label_source=None,
        label_strength=None,
        simulator_replay_verified=False,
        notes=notes,
        schema_version=EVALUATION_EVIDENCE_SCHEMA_VERSION,
    )


def _validate_returned_evidence(
    evidence: object,
    evaluator: ProposalEvaluator,
    proposal: CorruptedActionProposal,
    *,
    source_dataset_id: str,
    evaluation_seed: int,
    attempt_ordinal: int,
) -> None:
    if not isinstance(evidence, EvaluationEvidence):
        raise EvaluatorContractError("evaluate did not return EvaluationEvidence")
    try:
        validate_evaluation_evidence(evidence)
    except (TypeError, ValueError) as exc:
        raise EvaluatorContractError(
            f"evaluate returned invalid evidence: {sanitize_operational_text(exc)}"
        ) from exc
    expected = {
        "proposal_id": proposal.proposal_id,
        "source_dataset_id": source_dataset_id,
        "source_episode_id": proposal.source_episode_id,
        "source_candidate_id": proposal.source_candidate_id,
        "split_group_id": proposal.split_group_id,
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_version": evaluator.evaluator_version,
        "evaluator_configuration_digest": evaluator.configuration_digest,
        "evaluation_seed": evaluation_seed,
        "attempt_ordinal": attempt_ordinal,
    }
    mismatches = [
        field for field, value in expected.items() if getattr(evidence, field) != value
    ]
    if mismatches:
        raise EvaluatorContractError(
            "evaluator evidence identity mismatch: " + ", ".join(mismatches)
        )


def _append_evidence_and_replace_ledger(
    dataset: EvaluationDataset,
    evidence: EvaluationEvidence,
    entry: LedgerEntry,
) -> EvaluationDataset:
    evidence_items = (*dataset.evidence, evidence)
    order = {
        proposal_id: index
        for index, proposal_id in enumerate(dataset.selected_proposal_ids)
    }
    sorted_evidence = tuple(
        sorted(
            evidence_items,
            key=lambda item: (order[item.proposal_id], item.attempt_ordinal),
        )
    )
    updated = _replace_ledger(dataset, entry, evidence=sorted_evidence)
    return updated


def _replace_ledger(
    dataset: EvaluationDataset,
    replacement: LedgerEntry,
    *,
    evidence: tuple[EvaluationEvidence, ...] | None = None,
) -> EvaluationDataset:
    key = (replacement.proposal_id, replacement.attempt_ordinal)
    found = False
    entries: list[LedgerEntry] = []
    for entry in dataset.ledger:
        if (entry.proposal_id, entry.attempt_ordinal) == key:
            entries.append(replacement)
            found = True
        else:
            entries.append(entry)
    if not found:
        raise EvaluationRunnerError(f"ledger attempt is missing: {key!r}")
    return _replace_dataset(
        dataset,
        ledger=tuple(entries),
        evidence=dataset.evidence if evidence is None else evidence,
    )


def _replace_dataset(
    dataset: EvaluationDataset,
    *,
    ledger: tuple[LedgerEntry, ...] | None = None,
    evidence: tuple[EvaluationEvidence, ...] | None = None,
) -> EvaluationDataset:
    next_ledger = dataset.ledger if ledger is None else ledger
    next_evidence = dataset.evidence if evidence is None else evidence
    artifacts = tuple(
        sorted(
            {
                reference
                for item in next_evidence
                for reference in item.artifact_references
            }
        )
    )
    return replace(
        dataset,
        ledger=next_ledger,
        evidence=next_evidence,
        summary=build_evaluation_summary(next_evidence, next_ledger),
        artifact_references=artifacts,
    )


def _with_run_state(
    dataset: EvaluationDataset, state: RunState, finished_at: str | None
) -> EvaluationDataset:
    manifest = replace(
        dataset.run_manifest,
        finished_at=finished_at,
        final_state=state,
    )
    return replace(dataset, run_state=state, run_manifest=manifest)


def _ledger_entry(dataset: EvaluationDataset, key: tuple[str, int]) -> LedgerEntry:
    for entry in dataset.ledger:
        if (entry.proposal_id, entry.attempt_ordinal) == key:
            return entry
    raise EvaluationRunnerError(f"ledger attempt is missing: {key!r}")


def _ledger_state_for(status: EvaluationStatus) -> LedgerState:
    return {
        EvaluationStatus.CONCLUSIVE: LedgerState.COMPLETED,
        EvaluationStatus.INDETERMINATE: LedgerState.INDETERMINATE,
        EvaluationStatus.INVALID: LedgerState.INVALID,
        EvaluationStatus.SKIPPED: LedgerState.SKIPPED,
        EvaluationStatus.EXECUTION_ERROR: LedgerState.EXECUTION_ERROR,
    }[status]


def _required_evaluator_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvaluationRunnerError(
            f"evaluator.{field} must be a non-empty string without whitespace"
        )
    return value


def _sanitize_error_type(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return sanitized[:128] or "EvaluatorError"


__all__ = [
    "EvaluationPlan",
    "EvaluationRunConflictError",
    "EvaluationRunResult",
    "EvaluationRunnerError",
    "EvaluatorContractError",
    "PlannedAttempt",
    "collect_run_environment",
    "derive_evaluation_seed",
    "plan_evaluation",
    "run_evaluation",
    "sanitize_launch_command",
    "sanitize_operational_text",
    "utc_timestamp",
]
