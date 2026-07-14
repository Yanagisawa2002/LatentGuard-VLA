"""CPU-only integration tests for the ordered resume-safe M2A runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import latentguard.evaluation.runner as runner_module
from latentguard.corruptions.layout import (
    ActionField,
    ActionLayout,
    ActionSemantic,
)
from latentguard.corruptions.models import (
    CorruptedActionProposal,
    compute_proposal_identifier,
)
from latentguard.corruptions.serialization import CorruptionDataset
from latentguard.evaluation.base import ApplicabilityDecision
from latentguard.evaluation.fixture import create_deterministic_fixture_evaluator
from latentguard.evaluation.models import EvaluationEvidence, EvaluationStatus
from latentguard.evaluation.runner import (
    EvaluationRunConflictError,
    EvaluationRunnerError,
    derive_evaluation_seed,
    plan_evaluation,
    run_evaluation,
    sanitize_launch_command,
    sanitize_operational_text,
)
from latentguard.evaluation.serialization import (
    LedgerState,
    RunEnvironment,
    RunState,
    compute_corruption_dataset_content_digest,
    load_evaluation_dataset,
)
from latentguard.models import ActionChunk


def _configuration(**updates: object) -> dict[str, object]:
    configuration: dict[str, object] = {
        "schema_version": "1.0",
        "success_mean_abs_threshold": 10.0,
        "unsafe_max_abs_threshold": 100.0,
        "skip_mean_abs_below": None,
        "indeterminate_temporal_variation_below": None,
        "execution_error_max_abs_above": None,
    }
    configuration.update(updates)
    return configuration


def _corruption_dataset(*, proposal_count: int = 3) -> CorruptionDataset:
    layout = ActionLayout(
        action_dim=2,
        fields=(
            ActionField(
                name="control",
                indices=(0, 1),
                semantic=ActionSemantic.AUXILIARY,
            ),
        ),
    )
    proposals = []
    for ordinal in range(proposal_count):
        parameters = {"target_indices": (0, 1), "bias": (0.1, -0.1)}
        proposal_id = compute_proposal_identifier(
            source_episode_id=f"episode-{ordinal}",
            source_candidate_id=f"candidate-{ordinal}",
            corruption_name="constant_bias",
            resolved_parameters=parameters,
            seed=100 + ordinal,
            generation_ordinal=ordinal,
        )
        proposals.append(
            CorruptedActionProposal(
                proposal_id=proposal_id,
                source_episode_id=f"episode-{ordinal}",
                source_candidate_id=f"candidate-{ordinal}",
                source_policy_id="policy-fixture",
                source_task_id="task-fixture",
                split_group_id=f"episode-{ordinal}",
                transformed_action=ActionChunk(
                    actions=np.array(
                        [[0.1 + ordinal, 0.2], [0.3, 0.4 + ordinal]],
                        dtype=np.float32,
                    ),
                    coordinate_frame="fixture-frame",
                    control_period_s=0.1,
                ),
                corruption_type="constant_bias",
                resolved_parameters=parameters,
                seed=100 + ordinal,
                generation_ordinal=ordinal,
            )
        )
    return CorruptionDataset(
        source_dataset_id="sha256:" + "c" * 64,
        action_layout=layout,
        proposals=tuple(proposals),
    )


def _environment() -> RunEnvironment:
    return RunEnvironment(
        git_commit_sha=None,
        git_branch=None,
        python_version="3.11.14",
        numpy_version="1.26.4",
        platform="test-platform",
        launch_command="latentguard evaluate-data [sanitized]",
    )


class _Clock:
    def __init__(self) -> None:
        self._next = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)

    def __call__(self) -> str:
        value = self._next.isoformat().replace("+00:00", "Z")
        self._next += timedelta(seconds=1)
        return value


class _RecordingEvaluator:
    def __init__(
        self,
        *,
        configuration: Mapping[str, object] | None = None,
        output_dir: Path | None = None,
        fail_attempts: set[tuple[str, int]] | None = None,
        interrupt_once: bool = False,
    ) -> None:
        self.delegate = create_deterministic_fixture_evaluator(
            configuration or _configuration()
        )
        self.output_dir = output_dir
        self.fail_attempts = fail_attempts or set()
        self.interrupt_once = interrupt_once
        self.calls: list[tuple[str, int, int]] = []
        self.persisted_states: list[tuple[LedgerState, ...]] = []

    @property
    def evaluator_id(self) -> str:
        return self.delegate.evaluator_id

    @property
    def evaluator_version(self) -> str:
        return self.delegate.evaluator_version

    @property
    def configuration_digest(self) -> str:
        return self.delegate.configuration_digest

    def resolved_configuration(self) -> Mapping[str, object]:
        return self.delegate.resolved_configuration()

    def check_applicability(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
    ) -> ApplicabilityDecision:
        return self.delegate.check_applicability(
            proposal, source_dataset_id=source_dataset_id
        )

    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        self.calls.append((proposal.proposal_id, attempt_ordinal, evaluation_seed))
        if self.output_dir is not None:
            persisted = load_evaluation_dataset(self.output_dir)
            self.persisted_states.append(
                tuple(entry.state for entry in persisted.ledger)
            )
        if self.interrupt_once:
            self.interrupt_once = False
            raise KeyboardInterrupt
        if (proposal.proposal_id, attempt_ordinal) in self.fail_attempts:
            raise RuntimeError(
                "password=top-secret C:\\Users\\private\\trace.txt\n"
                "controlled evaluator traceback must not persist"
            )
        return self.delegate.evaluate(
            proposal,
            source_dataset_id=source_dataset_id,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
        )


def _run(
    dataset: CorruptionDataset,
    evaluator: _RecordingEvaluator,
    output: Path,
    **options: Any,
):
    return run_evaluation(
        dataset,
        evaluator,
        source_corruption_dataset_digest=compute_corruption_dataset_content_digest(
            dataset
        ),
        output_dir=output,
        base_seed=271828,
        run_environment=_environment(),
        **options,
    )


def test_plan_preserves_generation_order_and_derives_stable_seeds() -> None:
    dataset = _corruption_dataset()
    evaluator = _RecordingEvaluator()

    first = plan_evaluation(
        dataset,
        evaluator,
        source_corruption_dataset_digest=compute_corruption_dataset_content_digest(
            dataset
        ),
        base_seed=271828,
    )
    second = plan_evaluation(
        dataset,
        evaluator,
        source_corruption_dataset_digest=compute_corruption_dataset_content_digest(
            dataset
        ),
        base_seed=271828,
    )

    expected_ids = tuple(item.proposal_id for item in dataset.proposals)
    assert first.selected_proposal_ids == expected_ids
    assert first == second
    assert tuple(item.proposal_id for item in first.attempts) == expected_ids
    assert len({item.evaluation_seed for item in first.attempts}) == 3
    for attempt in first.attempts:
        assert attempt.evaluation_seed == derive_evaluation_seed(
            base_seed=271828,
            proposal_id=attempt.proposal_id,
            evaluator_id=evaluator.evaluator_id,
            evaluator_version=evaluator.evaluator_version,
            evaluator_configuration_digest=evaluator.configuration_digest,
            attempt_ordinal=0,
        )
        retry_seed = derive_evaluation_seed(
            base_seed=271828,
            proposal_id=attempt.proposal_id,
            evaluator_id=evaluator.evaluator_id,
            evaluator_version=evaluator.evaluator_version,
            evaluator_configuration_digest=evaluator.configuration_digest,
            attempt_ordinal=1,
        )
        assert retry_seed != attempt.evaluation_seed


def test_runner_evaluates_in_stable_order_and_completes(tmp_path: Path) -> None:
    dataset = _corruption_dataset()
    evaluator = _RecordingEvaluator()

    result = _run(dataset, evaluator, tmp_path / "ordered")

    expected_ids = [item.proposal_id for item in dataset.proposals]
    assert [call[0] for call in evaluator.calls] == expected_ids
    assert result.dataset.run_state is RunState.COMPLETE
    assert [item.proposal_id for item in result.dataset.evidence] == expected_ids
    assert [entry.state for entry in result.dataset.ledger] == [
        LedgerState.COMPLETED
    ] * 3
    assert result.dataset.summary.conclusive == 3
    assert result.evaluated_attempts == 3


def test_running_state_is_persisted_before_each_evaluator_call(tmp_path: Path) -> None:
    dataset = _corruption_dataset()
    output = tmp_path / "incremental"
    evaluator = _RecordingEvaluator(output_dir=output)

    _run(dataset, evaluator, output)

    assert evaluator.persisted_states == [
        (LedgerState.RUNNING, LedgerState.PENDING, LedgerState.PENDING),
        (LedgerState.COMPLETED, LedgerState.RUNNING, LedgerState.PENDING),
        (LedgerState.COMPLETED, LedgerState.COMPLETED, LedgerState.RUNNING),
    ]


def test_noop_resume_does_not_duplicate_or_rewrite_completed_run(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset()
    output = tmp_path / "resume"
    _run(dataset, _RecordingEvaluator(), output)
    before = (output / "manifest.json").read_bytes()
    evaluator = _RecordingEvaluator()

    result = _run(dataset, evaluator, output, resume=True)

    assert evaluator.calls == []
    assert result.evaluated_attempts == 0
    assert len(result.dataset.evidence) == 3
    assert (output / "manifest.json").read_bytes() == before


def test_interrupted_running_attempt_is_recovered_without_duplication(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset()
    output = tmp_path / "interrupted"
    interrupting = _RecordingEvaluator(interrupt_once=True)

    with pytest.raises(KeyboardInterrupt):
        _run(dataset, interrupting, output)

    interrupted = load_evaluation_dataset(output)
    assert interrupted.run_state is RunState.INTERRUPTED
    assert interrupted.ledger[0].state is LedgerState.RUNNING
    assert interrupted.evidence == ()

    evaluator = _RecordingEvaluator()
    resumed = _run(dataset, evaluator, output, resume=True)

    assert resumed.recovered_attempts == 1
    assert resumed.evaluated_attempts == 3
    assert resumed.dataset.run_state is RunState.COMPLETE
    assert len(resumed.dataset.evidence) == 3
    assert [entry.attempt_ordinal for entry in resumed.dataset.ledger] == [0, 0, 0]


def test_exception_becomes_sanitized_execution_error_not_task_failure(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset()
    first_id = dataset.proposals[0].proposal_id
    evaluator = _RecordingEvaluator(fail_attempts={(first_id, 0)})

    result = _run(dataset, evaluator, tmp_path / "execution-error")

    error_evidence = result.dataset.evidence[0]
    error_entry = result.dataset.ledger[0]
    assert error_evidence.status is EvaluationStatus.EXECUTION_ERROR
    assert error_evidence.success is None
    assert error_evidence.progress_before is None
    assert error_evidence.progress_after is None
    assert error_evidence.progress_delta is None
    assert error_evidence.unsafe is None
    assert error_evidence.failure_events == ()
    assert error_evidence.label_source is None
    assert error_evidence.label_strength is None
    assert error_entry.state is LedgerState.EXECUTION_ERROR
    assert error_entry.retry_eligible is True
    assert error_entry.error_type == "RuntimeError"
    serialized = " ".join(
        filter(
            None,
            [error_entry.error_message, error_evidence.notes],
        )
    )
    assert "top-secret" not in serialized
    assert "C:\\Users" not in serialized
    assert "Traceback (most recent call last)" not in serialized
    assert 'File "' not in serialized
    assert "operational error details redacted" in serialized
    assert result.dataset.summary.execution_error == 1
    assert result.dataset.summary.conclusive == 2


def test_fail_fast_persists_error_and_leaves_later_attempts_pending(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset()
    first_id = dataset.proposals[0].proposal_id
    evaluator = _RecordingEvaluator(fail_attempts={(first_id, 0)})

    result = _run(dataset, evaluator, tmp_path / "fail-fast", fail_fast=True)

    assert result.stopped_early is True
    assert result.dataset.run_state is RunState.INTERRUPTED
    assert [entry.state for entry in result.dataset.ledger] == [
        LedgerState.EXECUTION_ERROR,
        LedgerState.PENDING,
        LedgerState.PENDING,
    ]
    assert len(result.dataset.evidence) == 1


def test_explicit_retry_preserves_failed_attempt_and_appends_success(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    proposal_id = dataset.proposals[0].proposal_id
    output = tmp_path / "retry"
    first = _run(
        dataset,
        _RecordingEvaluator(fail_attempts={(proposal_id, 0)}),
        output,
    )
    failed_evidence_id = first.dataset.evidence[0].evidence_id

    resumed = _run(
        dataset,
        _RecordingEvaluator(),
        output,
        resume=True,
        retry_execution_errors=True,
    )

    assert resumed.retried_attempts == 1
    assert [entry.attempt_ordinal for entry in resumed.dataset.ledger] == [0, 1]
    assert [entry.state for entry in resumed.dataset.ledger] == [
        LedgerState.EXECUTION_ERROR,
        LedgerState.COMPLETED,
    ]
    assert resumed.dataset.evidence[0].evidence_id == failed_evidence_id
    assert resumed.dataset.evidence[1].evidence_id != failed_evidence_id
    assert resumed.dataset.summary.execution_error == 1
    assert resumed.dataset.summary.conclusive == 1
    assert resumed.dataset.summary.retried_attempts == 1


def test_execution_error_is_not_retried_without_explicit_policy(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    proposal_id = dataset.proposals[0].proposal_id
    output = tmp_path / "no-retry"
    _run(
        dataset,
        _RecordingEvaluator(fail_attempts={(proposal_id, 0)}),
        output,
    )
    evaluator = _RecordingEvaluator()

    resumed = _run(dataset, evaluator, output, resume=True)

    assert evaluator.calls == []
    assert len(resumed.dataset.ledger) == 1
    assert resumed.dataset.ledger[0].state is LedgerState.EXECUTION_ERROR


@pytest.mark.parametrize("conflict", ["configuration", "seed", "selection"])
def test_resume_rejects_changed_plan_inputs(tmp_path: Path, conflict: str) -> None:
    dataset = _corruption_dataset()
    output = tmp_path / conflict
    _run(dataset, _RecordingEvaluator(), output, max_proposals=2)
    evaluator = _RecordingEvaluator(
        configuration=(
            _configuration(success_mean_abs_threshold=0.1)
            if conflict == "configuration"
            else _configuration()
        )
    )
    options: dict[str, object] = {"resume": True, "max_proposals": 2}
    if conflict == "seed":
        with pytest.raises(EvaluationRunConflictError, match="conflict"):
            run_evaluation(
                dataset,
                evaluator,
                source_corruption_dataset_digest=(
                    compute_corruption_dataset_content_digest(dataset)
                ),
                output_dir=output,
                base_seed=9,
                max_proposals=2,
                resume=True,
                run_environment=_environment(),
                clock=_Clock(),
            )
        return
    if conflict == "selection":
        options["max_proposals"] = 1

    with pytest.raises(EvaluationRunConflictError, match="conflict"):
        _run(dataset, evaluator, output, **options)


def test_resume_rejects_changed_source_digest(tmp_path: Path) -> None:
    dataset = _corruption_dataset()
    output = tmp_path / "source-conflict"
    _run(dataset, _RecordingEvaluator(), output)

    with pytest.raises(EvaluationRunnerError, match="does not match"):
        run_evaluation(
            dataset,
            _RecordingEvaluator(),
            source_corruption_dataset_digest="sha256:" + "d" * 64,
            output_dir=output,
            base_seed=271828,
            resume=True,
            run_environment=_environment(),
            clock=_Clock(),
        )


def test_runner_preserves_source_proposals_across_errors_and_retries(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    proposal = dataset.proposals[0]
    before = proposal.transformed_action.actions.copy()
    output = tmp_path / "immutable"
    _run(
        dataset,
        _RecordingEvaluator(fail_attempts={(proposal.proposal_id, 0)}),
        output,
    )
    _run(
        dataset,
        _RecordingEvaluator(),
        output,
        resume=True,
        retry_execution_errors=True,
    )

    np.testing.assert_array_equal(proposal.transformed_action.actions, before)
    assert proposal.transformed_action.actions.flags.writeable is False
    assert dataset.proposals[0] is proposal


def test_sanitizers_remove_credentials_paths_and_path_arguments() -> None:
    secret = "token=do-not-store user:password@host C:\\private\\trace.txt"

    text = sanitize_operational_text(secret)
    command = sanitize_launch_command(
        (
            "latentguard",
            "replay-data",
            "--source-dir",
            "C:\\private\\source",
            "--corruption-dir",
            "C:\\private\\corruptions",
            "--output-dir=C:\\private\\evaluated",
            "--token",
            "do-not-store",
        )
    )

    assert "do-not-store" not in text
    assert "password" not in text
    assert "C:\\private" not in text
    assert "do-not-store" not in command
    assert "C:\\private" not in command
    assert "<path>" in command
    assert "[REDACTED]" in command


@pytest.mark.parametrize(
    "option",
    [
        "--credential",
        "--credentials",
        "--private-key",
        "--private_key",
        "--api_key",
        "--ssh-password",
        "--password_file",
    ],
)
def test_launch_sanitizer_redacts_separate_credential_values(option: str) -> None:
    command = sanitize_launch_command(
        ("latentguard", "evaluate-data", option, "abc123")
    )

    assert "abc123" not in command
    assert option not in command
    assert "[REDACTED]" in command


@pytest.mark.parametrize(
    "executable",
    [
        "/usr/bin/ssh",
        r"C:\Windows\System32\OpenSSH\ssh.exe",
        "/usr/bin/scp",
    ],
)
def test_launch_sanitizer_redacts_absolute_ssh_invocations(executable: str) -> None:
    command = sanitize_launch_command((executable, "-p", "22", "root@example.invalid"))

    assert command == "operational command redacted"
    assert "root@example.invalid" not in command


@pytest.mark.parametrize(
    "raw",
    [
        r"failed at \\server\private share\trace.txt",
        r"failed at C:\Users\private folder\trace.txt",
        "environment={'API_TOKEN': 'do-not-store'}",
        "environment={'FOO': 'bar'}",
        "AWS_SECRET_ACCESS_KEY=abc123",
        "PASSWORD_FILE=abc123",
        "Authorization: Bearer abc123",
        "path:/home/private/model.ckpt",
        "file:/home/private/model.ckpt",
        "ssh\t-p 22 root@example.invalid",
        "Traceback (most recent call last): relative_module.py line 10",
        "ssh -p 22 user@example.invalid",
        "connection failed for root@example.invalid:47228,",
        "https://user:do-not-store@example.invalid/log",
    ],
)
def test_operational_sanitizer_redacts_entire_sensitive_messages(raw: str) -> None:
    assert sanitize_operational_text(raw) == "operational error details redacted"


class _ReturnedExecutionErrorEvaluator(_RecordingEvaluator):
    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        conclusive = self.delegate.evaluate(
            proposal,
            source_dataset_id=source_dataset_id,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
        )
        return replace(
            conclusive,
            status=EvaluationStatus.EXECUTION_ERROR,
            success=None,
            progress_before=None,
            progress_after=None,
            progress_delta=None,
            unsafe=None,
            failure_events=(),
            termination_reason=r"failed at C:\Users\private folder\trace.txt",
            replayed_control_steps=0,
            metrics={
                "password": "do-not-store",
                "detail": r"\\server\private share\debug.log",
            },
            label_source=None,
            label_strength=None,
            simulator_replay_verified=False,
            notes="environment={'API_TOKEN': 'do-not-store'}",
        )


def test_returned_execution_error_diagnostics_are_sanitized(tmp_path: Path) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    output = tmp_path / "returned-error"

    result = _run(dataset, _ReturnedExecutionErrorEvaluator(), output)

    evidence = result.dataset.evidence[0]
    ledger = result.dataset.ledger[0]
    assert evidence.status is EvaluationStatus.EXECUTION_ERROR
    assert evidence.termination_reason == "operational error details redacted"
    assert evidence.notes == "operational error details redacted"
    assert evidence.metrics == {
        "detail": "operational error details redacted",
        "redacted_metric_1": "operational error details redacted",
    }
    assert evidence.success is None
    assert evidence.failure_events == ()
    assert evidence.label_source is None
    assert evidence.label_strength is None
    assert evidence.simulator_replay_verified is False
    assert ledger.error_type == "EvaluatorReportedExecutionError"
    assert ledger.error_message == "operational error details redacted"
    assert ledger.retry_eligible is True
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert "do-not-store" not in manifest
    assert "C:\\Users" not in manifest
    assert "private share" not in manifest


def test_wrong_in_memory_digest_fails_before_evaluation_or_output(
    tmp_path: Path,
) -> None:
    supplied = _corruption_dataset(proposal_count=1)
    evaluated = _corruption_dataset(proposal_count=2)
    evaluator = _RecordingEvaluator()
    output = tmp_path / "wrong-digest"

    with pytest.raises(EvaluationRunnerError, match="does not match"):
        run_evaluation(
            evaluated,
            evaluator,
            source_corruption_dataset_digest=(
                compute_corruption_dataset_content_digest(supplied)
            ),
            output_dir=output,
            base_seed=271828,
            run_environment=_environment(),
        )

    assert evaluator.calls == []
    assert not output.exists()


def test_nonmonotonic_clock_is_floored_for_resumable_audit_order(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    output = tmp_path / "backwards-clock"
    timestamps = iter(
        (
            "2026-07-14T12:00:00+00:00",
            "2026-07-14T12:00:02+00:00",
            "2026-07-14T12:00:01+00:00",
        )
    )

    result = _run(
        dataset,
        _RecordingEvaluator(),
        output,
        clock=lambda: next(timestamps),
    )

    assert result.dataset.run_state is RunState.COMPLETE
    assert result.dataset.ledger[0].started_at == "2026-07-14T12:00:02+00:00"
    assert result.dataset.ledger[0].finished_at == "2026-07-14T12:00:02+00:00"


def test_resume_retains_interruption_audit_floor_across_clock_skew(
    tmp_path: Path,
) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    output = tmp_path / "resume-clock-skew"
    first_clock = iter(
        (
            "2026-07-14T12:00:00+00:00",
            "2026-07-14T12:00:01+00:00",
            "2026-07-14T13:00:00+00:00",
        )
    )
    with pytest.raises(KeyboardInterrupt):
        _run(
            dataset,
            _RecordingEvaluator(interrupt_once=True),
            output,
            clock=lambda: next(first_clock),
        )

    resumed = _run(
        dataset,
        _RecordingEvaluator(),
        output,
        resume=True,
        clock=lambda: "2026-07-14T11:00:00+00:00",
    )

    assert resumed.dataset.run_state is RunState.COMPLETE
    assert resumed.dataset.ledger[0].finished_at == "2026-07-14T13:00:00+00:00"


def test_keyboard_interrupt_after_terminal_commit_preserves_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    output = tmp_path / "interrupt-after-commit"
    real_update = runner_module.update_evaluation_dataset
    interrupted = False

    def interrupt_after_commit(dataset_update: Any, output_dir: Path) -> Path:
        nonlocal interrupted
        path = real_update(dataset_update, output_dir)
        if dataset_update.run_state is RunState.COMPLETE and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return path

    monkeypatch.setattr(
        runner_module, "update_evaluation_dataset", interrupt_after_commit
    )

    with pytest.raises(KeyboardInterrupt):
        _run(dataset, _RecordingEvaluator(), output)

    persisted = load_evaluation_dataset(output)
    assert persisted.run_state is RunState.COMPLETE
    assert persisted.ledger[0].state is LedgerState.COMPLETED
    assert len(persisted.evidence) == 1


class _SensitiveArtifactErrorEvaluator(_ReturnedExecutionErrorEvaluator):
    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        evidence = super().evaluate(
            proposal,
            source_dataset_id=source_dataset_id,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
        )
        return replace(
            evidence,
            artifact_references=("artifacts/password=do-not-store.log",),
        )


def test_sensitive_artifact_reference_is_not_persisted(tmp_path: Path) -> None:
    dataset = _corruption_dataset(proposal_count=1)
    output = tmp_path / "sensitive-artifact"

    result = _run(dataset, _SensitiveArtifactErrorEvaluator(), output)

    assert result.dataset.evidence[0].artifact_references == ()
    assert result.dataset.ledger[0].error_type == "EvaluatorContractError"
    assert "do-not-store" not in (output / "manifest.json").read_text(encoding="utf-8")
