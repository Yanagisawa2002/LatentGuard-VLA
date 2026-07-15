"""M2A runner reuse, trust mapping, resume, and replay reporting tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    save_corruption_dataset,
)
from latentguard.evaluation.models import EvaluationStatus
from latentguard.evaluation.runner import (
    EvaluationRunConflictError,
    run_evaluation,
)
from latentguard.evaluation.serialization import (
    RunEnvironment,
    compute_corruption_dataset_content_digest,
    load_evaluation_dataset,
)
from latentguard.evaluation.validation import (
    OutcomeProjectionError,
    evidence_to_outcome_label,
    validate_evaluation_evidence,
)
from latentguard.models import LabelSource, LabelStrength
from latentguard.replay.base import ReplayEnvironmentSession
from latentguard.replay.evaluator import (
    ExactReplayEvaluatorConfigurationError,
    ExactStatePairedReplayEvaluator,
    create_exact_state_paired_replay_evaluator,
)
from latentguard.replay.fixture import (
    DeterministicReplayFixtureAdapter,
    create_deterministic_replay_fixture_adapter,
)
from latentguard.replay.models import (
    ReplayCase,
    ReplayExecutionRole,
    ReplayTrustDescriptor,
    ReplayTrustTier,
    StateComparisonSemantic,
)
from latentguard.replay.reporting import (
    build_replay_summary,
    format_replay_summary,
    replay_audit_lines,
)
from latentguard.replay.source import ReplaySourceBinding
from latentguard.replay.validation import (
    ReplayValidationError,
    validate_replay_trust_claim,
)
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    save_episodes,
)
from latentguard.synthetic import generate_synthetic_episodes

_CORRUPTION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "corruptions" / "m1-smoke.json"
)


def _configuration(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "state_dimension": 7,
        "action_scale": 1.0,
        "progress_scale": 1.0,
        "success_tolerance": 1e-6,
        "unsafe_bound": 10.0,
        "baseline_failure_ordinals": [],
        "restoration_mismatch_ordinals": [],
        "indeterminate_ordinals": [],
        "step_exception_ordinals": [],
        "close_exception_ordinals": [],
    }
    value.update(updates)
    return value


def _stack(
    configuration: Mapping[str, object] | None = None,
) -> tuple[
    CorruptionDataset,
    ReplaySourceBinding,
    DeterministicReplayFixtureAdapter,
    ExactStatePairedReplayEvaluator,
]:
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=8,
        action_dim=7,
        robot_state_dim=10,
        camera_count=0,
    )
    plan = load_corruption_plan(_CORRUPTION_CONFIG)
    generated = generate_corruption_proposals(
        episodes,
        plan.action_layout,
        plan.corruptions,
        base_seed=314159,
        proposal_limit=6,
    )
    dataset = CorruptionDataset(
        source_dataset_id="sha256:" + "c" * 64,
        action_layout=plan.action_layout,
        proposals=generated.proposals,
    )
    binding = ReplaySourceBinding.from_datasets(episodes, dataset)
    adapter = create_deterministic_replay_fixture_adapter(
        configuration or _configuration(), binding
    )
    evaluator = create_exact_state_paired_replay_evaluator(
        adapter, adapter.replay_bundle
    )
    return dataset, binding, adapter, evaluator


def _environment() -> RunEnvironment:
    return RunEnvironment(
        git_commit_sha=None,
        git_branch=None,
        python_version="3.11.14",
        numpy_version="1.26.4",
        platform="test-platform",
        launch_command="latentguard replay-data [sanitized]",
    )


def _run(
    dataset: CorruptionDataset,
    evaluator: ExactStatePairedReplayEvaluator,
    output: Path,
    **options: object,
):
    return run_evaluation(
        dataset,
        evaluator,
        source_corruption_dataset_digest=compute_corruption_dataset_content_digest(
            dataset
        ),
        output_dir=output,
        base_seed=161803,
        run_environment=_environment(),
        **options,  # type: ignore[arg-type]
    )


def test_fixture_evidence_validates_projects_and_never_claims_simulator() -> None:
    dataset, _, _, evaluator = _stack()
    proposal = dataset.proposals[0]

    evidence = evaluator.evaluate(
        proposal,
        source_dataset_id=dataset.source_dataset_id,
        evaluation_seed=123,
        attempt_ordinal=0,
    )

    validate_evaluation_evidence(evidence)
    outcome = evidence_to_outcome_label(evidence)
    assert evidence.status is EvaluationStatus.CONCLUSIVE
    assert evidence.label_source is LabelSource.DETERMINISTIC_EVALUATOR
    assert evidence.label_strength is LabelStrength.WEAK
    assert evidence.simulator_replay_verified is False
    assert outcome.progress == evidence.progress_after
    assert outcome.success is False


def test_indeterminate_and_invalid_replay_evidence_cannot_be_projected() -> None:
    dataset, _, _, evaluator = _stack(
        _configuration(
            baseline_failure_ordinals=[0],
            indeterminate_ordinals=[2],
        )
    )

    invalid = evaluator.evaluate(
        dataset.proposals[0],
        source_dataset_id=dataset.source_dataset_id,
        evaluation_seed=1,
        attempt_ordinal=0,
    )
    indeterminate = evaluator.evaluate(
        dataset.proposals[2],
        source_dataset_id=dataset.source_dataset_id,
        evaluation_seed=2,
        attempt_ordinal=0,
    )

    assert invalid.status is EvaluationStatus.INVALID
    assert invalid.success is invalid.progress_after is invalid.unsafe is None
    assert indeterminate.status is EvaluationStatus.INDETERMINATE
    assert indeterminate.success is None
    for evidence in (invalid, indeterminate):
        with pytest.raises(OutcomeProjectionError):
            evidence_to_outcome_label(evidence)


def test_m2a_runner_persists_all_replay_statuses_and_machine_counts(
    tmp_path: Path,
) -> None:
    dataset, binding, adapter, evaluator = _stack(
        _configuration(
            baseline_failure_ordinals=[0],
            restoration_mismatch_ordinals=[1],
            indeterminate_ordinals=[2],
        )
    )
    output = tmp_path / "evaluated"

    result = _run(dataset, evaluator, output)
    reloaded = load_evaluation_dataset(
        output,
        corruption_dataset=dataset,
        expected_corruption_digest=binding.corruption_dataset_digest,
    )
    summary = build_replay_summary(
        reloaded,
        source_episode_count=binding.source_episode_count,
        source_candidate_count=binding.source_candidate_count,
        adapter_trust_tier=adapter.trust_descriptor().trust_tier.value,
    )

    assert result.evaluated_attempts == 6
    assert reloaded.summary.conclusive == 3
    assert reloaded.summary.indeterminate == 1
    assert reloaded.summary.invalid == 2
    assert reloaded.summary.execution_error == 0
    assert reloaded.summary.projected_outcome_labels == 3
    assert summary.valid_baseline_count == 4
    assert summary.invalid_baseline_count == 2
    assert summary.conclusive_success_count == 1
    assert summary.conclusive_task_failure_count == 2
    assert len(replay_audit_lines(reloaded)) == 6
    formatted = format_replay_summary(summary, resumed=False)
    assert "fixture_warning=non-physical-weak-non-simulator" in formatted


def test_path_backed_source_mutation_becomes_execution_error(
    tmp_path: Path,
) -> None:
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=8,
        action_dim=7,
        robot_state_dim=10,
        camera_count=0,
    )
    source_dir = tmp_path / "source"
    corruption_dir = tmp_path / "corruption"
    save_episodes(episodes, source_dir)
    plan = load_corruption_plan(_CORRUPTION_CONFIG)
    generated = generate_corruption_proposals(
        episodes,
        plan.action_layout,
        plan.corruptions,
        base_seed=314159,
        proposal_limit=1,
    )
    dataset = CorruptionDataset(
        source_dataset_id=compute_episode_bundle_identifier(source_dir),
        action_layout=plan.action_layout,
        proposals=generated.proposals,
    )
    save_corruption_dataset(dataset, corruption_dir)
    binding = ReplaySourceBinding.from_paths(source_dir, corruption_dir)
    adapter = create_deterministic_replay_fixture_adapter(_configuration(), binding)
    evaluator = create_exact_state_paired_replay_evaluator(
        adapter, adapter.replay_bundle
    )
    with (source_dir / "manifest.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    result = _run(dataset, evaluator, tmp_path / "evaluated", max_proposals=1)

    assert result.dataset.summary.execution_error == 1
    evidence = result.dataset.evidence[0]
    assert evidence.status is EvaluationStatus.EXECUTION_ERROR
    assert evidence.termination_reason == "execution_error:ReplaySourceMutationError"


def test_completed_resume_performs_no_work_and_does_not_duplicate_evidence(
    tmp_path: Path,
) -> None:
    dataset, binding, adapter, evaluator = _stack()
    output = tmp_path / "evaluated"
    _run(dataset, evaluator, output)
    before = (output / "manifest.json").read_bytes()

    resumed = _run(dataset, evaluator, output, resume=True)
    after = (output / "manifest.json").read_bytes()
    reloaded = load_evaluation_dataset(output, corruption_dataset=dataset)
    summary = build_replay_summary(
        reloaded,
        source_episode_count=binding.source_episode_count,
        source_candidate_count=binding.source_candidate_count,
        adapter_trust_tier=adapter.trust_descriptor().trust_tier.value,
        resumed_without_rerun_count=len(dataset.proposals),
    )

    assert resumed.evaluated_attempts == 0
    assert len(reloaded.evidence) == len(dataset.proposals)
    assert before == after
    assert summary.resumed_without_rerun_count == len(dataset.proposals)


def test_execution_error_retry_uses_new_attempt_ordinal(
    tmp_path: Path,
) -> None:
    dataset, _, _, evaluator = _stack(_configuration(step_exception_ordinals=[0]))
    output = tmp_path / "evaluated"
    first = _run(dataset, evaluator, output)

    second = _run(
        dataset,
        evaluator,
        output,
        resume=True,
        retry_execution_errors=True,
    )
    reloaded = load_evaluation_dataset(output, corruption_dataset=dataset)
    attempts = [
        entry
        for entry in reloaded.ledger
        if entry.proposal_id == dataset.proposals[0].proposal_id
    ]

    assert first.dataset.summary.execution_error == 1
    assert second.retried_attempts == 1
    assert second.evaluated_attempts == 1
    assert [entry.attempt_ordinal for entry in attempts] == [0, 1]
    assert all(entry.retry_eligible for entry in attempts)
    assert len(reloaded.evidence) == len(dataset.proposals) + 1


def test_changed_adapter_configuration_rejects_resume(tmp_path: Path) -> None:
    dataset, binding, _, evaluator = _stack()
    output = tmp_path / "evaluated"
    _run(dataset, evaluator, output)
    changed_adapter = create_deterministic_replay_fixture_adapter(
        _configuration(success_tolerance=0.01), binding
    )
    changed_evaluator = create_exact_state_paired_replay_evaluator(
        changed_adapter, changed_adapter.replay_bundle
    )

    with pytest.raises(EvaluationRunConflictError):
        _run(dataset, changed_evaluator, output, resume=True)


def test_adapter_configuration_drift_is_rejected_before_applicability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _, _, evaluator = _stack()
    changed = _configuration(success_tolerance=0.25)
    monkeypatch.setattr(
        DeterministicReplayFixtureAdapter,
        "resolved_configuration",
        lambda _self: changed,
    )

    with pytest.raises(
        ExactReplayEvaluatorConfigurationError,
        match="resolved configuration changed",
    ):
        evaluator.check_applicability(
            dataset.proposals[0], source_dataset_id=dataset.source_dataset_id
        )


def test_adapter_trust_drift_is_rejected_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _, _, evaluator = _stack()
    changed = ReplayTrustDescriptor(
        trust_tier=ReplayTrustTier.EXACT_SIMULATOR,
        label_source=LabelSource.SIMULATOR,
        maximum_label_strength=LabelStrength.STRONG,
        simulator_verification_allowed=True,
        exact_state_verification_required=True,
    )
    monkeypatch.setattr(
        DeterministicReplayFixtureAdapter,
        "trust_descriptor",
        lambda _self: changed,
    )

    with pytest.raises(
        ExactReplayEvaluatorConfigurationError,
        match="trust descriptor changed",
    ):
        evaluator.evaluate(
            dataset.proposals[0],
            source_dataset_id=dataset.source_dataset_id,
            evaluation_seed=5,
            attempt_ordinal=0,
        )


class _ExactSimulatorTrustProxy:
    def __init__(
        self,
        delegate: DeterministicReplayFixtureAdapter,
        *,
        state_verification_semantic: StateComparisonSemantic = (
            StateComparisonSemantic.EXACT_DIGEST
        ),
        state_verification_tolerance: float = 0.0,
    ) -> None:
        self.delegate = delegate
        self.state_verification_semantic = state_verification_semantic
        self.state_verification_tolerance = state_verification_tolerance

    @property
    def adapter_id(self) -> str:
        return self.delegate.adapter_id

    @property
    def adapter_version(self) -> str:
        return self.delegate.adapter_version

    @property
    def configuration_digest(self) -> str:
        return self.delegate.configuration_digest

    def trust_descriptor(self) -> ReplayTrustDescriptor:
        return ReplayTrustDescriptor(
            trust_tier=ReplayTrustTier.EXACT_SIMULATOR,
            label_source=LabelSource.SIMULATOR,
            maximum_label_strength=LabelStrength.STRONG,
            simulator_verification_allowed=True,
            exact_state_verification_required=True,
            state_verification_semantic=self.state_verification_semantic,
            state_verification_tolerance=self.state_verification_tolerance,
        )

    def resolved_configuration(self) -> Mapping[str, object]:
        return self.delegate.resolved_configuration()

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        return self.delegate.resolve_case(proposal)

    def create_session(
        self, replay_case: ReplayCase, *, execution_role: ReplayExecutionRole
    ) -> ReplayEnvironmentSession:
        return self.delegate.create_session(replay_case, execution_role=execution_role)


def test_fake_exact_simulator_protocol_allows_strong_conclusive_task_failure() -> None:
    dataset, _, adapter, _ = _stack()
    proxy = _ExactSimulatorTrustProxy(adapter)
    evaluator = create_exact_state_paired_replay_evaluator(proxy, adapter.replay_bundle)

    evidence = evaluator.evaluate(
        dataset.proposals[0],
        source_dataset_id=dataset.source_dataset_id,
        evaluation_seed=7,
        attempt_ordinal=0,
    )

    assert evidence.status is EvaluationStatus.CONCLUSIVE
    assert evidence.success is False
    assert evidence.label_source is LabelSource.SIMULATOR
    assert evidence.label_strength is LabelStrength.STRONG
    assert evidence.simulator_replay_verified is True


def test_trust_comparison_mode_and_tolerance_change_evaluator_identity() -> None:
    _, _, adapter, _ = _stack()
    exact = create_exact_state_paired_replay_evaluator(
        _ExactSimulatorTrustProxy(
            adapter,
            state_verification_semantic=StateComparisonSemantic.EXACT_DIGEST,
        ),
        adapter.replay_bundle,
    )
    numeric_small = create_exact_state_paired_replay_evaluator(
        _ExactSimulatorTrustProxy(
            adapter,
            state_verification_semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
            state_verification_tolerance=1e-7,
        ),
        adapter.replay_bundle,
    )
    numeric_bounded = create_exact_state_paired_replay_evaluator(
        _ExactSimulatorTrustProxy(
            adapter,
            state_verification_semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
            state_verification_tolerance=1e-6,
        ),
        adapter.replay_bundle,
    )

    assert (
        len(
            {
                exact.configuration_digest,
                numeric_small.configuration_digest,
                numeric_bounded.configuration_digest,
            }
        )
        == 3
    )
    assert exact.resolved_configuration()["state_verification_semantic"] == (
        StateComparisonSemantic.EXACT_DIGEST.value
    )
    assert numeric_bounded.resolved_configuration()[
        "state_verification_tolerance"
    ] == pytest.approx(1e-6)


def test_fixture_trust_rejects_forged_strong_or_verified_claims() -> None:
    _, _, adapter, _ = _stack()
    descriptor = adapter.trust_descriptor()
    for strength, verified in (
        (LabelStrength.STRONG, False),
        (LabelStrength.WEAK, True),
    ):
        with pytest.raises(ReplayValidationError):
            validate_replay_trust_claim(
                descriptor,
                label_source=LabelSource.DETERMINISTIC_EVALUATOR,
                label_strength=strength,
                simulator_replay_verified=verified,
            )


def test_evaluator_configuration_binds_all_content_and_trust_identities() -> None:
    _, binding, adapter, evaluator = _stack()
    resolved = evaluator.resolved_configuration()

    assert resolved["source_dataset_digest"] == binding.source_dataset_digest
    assert resolved["corruption_dataset_digest"] == binding.corruption_dataset_digest
    assert resolved["replay_bundle_digest"] == adapter.replay_bundle.bundle_digest
    assert resolved["adapter_configuration_digest"] == adapter.configuration_digest
    assert resolved["trust_tier"] == "fixture"
    assert resolved["state_verification_semantic"] == "exact_digest"
    assert resolved["state_verification_tolerance"] == 0.0
    serialized = repr(dict(resolved)).lower()
    assert "output_dir" not in serialized
    assert "hostname" not in serialized
    assert "timestamp" not in serialized
