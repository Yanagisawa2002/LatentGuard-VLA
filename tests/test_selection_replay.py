"""CPU-only M3C manifest-gated reuse of the M2A replay ledger."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import CorruptionDataset
from latentguard.evaluation.base import ApplicabilityDecision
from latentguard.evaluation.models import (
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.evaluation.runner import (
    EvaluationRunConflictError,
    run_evaluation,
)
from latentguard.evaluation.serialization import (
    EvaluationDataset,
    RunEnvironment,
    compute_corruption_dataset_content_digest,
    compute_evaluation_dataset_content_digest,
)
from latentguard.models import LabelSource, LabelStrength
from latentguard.selection.evaluation import (
    MANIFEST_GATED_REPLAY_EVALUATOR_ID,
    ManifestGatedReplayEvaluator,
    ReplayPhase,
    SelectionReplayError,
    build_manifest_gated_replay_evaluators,
    join_complementary_replay_phases,
    verify_zero_work_resume,
)
from latentguard.synthetic import generate_synthetic_episodes

_CORRUPTION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "corruptions" / "m1-smoke.json"
)


def _sha(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class _Pool:
    content_digest: str
    proposal_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Selection:
    selected_proposal_id: str | None
    abstained: bool = False


@dataclass(frozen=True, slots=True)
class _Manifest:
    content_digest: str
    candidate_pool_digest: str
    selections: tuple[_Selection, ...]
    manifest_envelope_digest: str = _sha("manifest-envelope")


class _MutableManifest:
    def __init__(
        self,
        *,
        content_digest: str,
        candidate_pool_digest: str,
        selections: list[_Selection],
        manifest_envelope_digest: str = _sha("mutable-manifest-envelope"),
    ) -> None:
        self.content_digest = content_digest
        self.candidate_pool_digest = candidate_pool_digest
        self.selections = selections
        self.manifest_envelope_digest = manifest_envelope_digest


class _StrongReplayEvaluator:
    evaluator_id = "strong_m3c_test_replay"
    evaluator_version = "1.0.0"

    def __init__(
        self,
        *,
        component_count: int = 70,
        corrupt_action_count: bool = False,
    ) -> None:
        self.executions: dict[str, int] = {}
        self.applicability_calls: dict[str, int] = {}
        self._component_count = component_count
        self._corrupt_action_count = corrupt_action_count
        self._configuration: Mapping[str, object] = {
            "component_count": component_count,
            "corrupt_action_count": corrupt_action_count,
            "schema_version": "1.0",
        }
        self._configuration_digest = compute_configuration_digest(self._configuration)

    @property
    def configuration_digest(self) -> str:
        return self._configuration_digest

    def resolved_configuration(self) -> Mapping[str, object]:
        return self._configuration

    def check_applicability(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
    ) -> ApplicabilityDecision:
        del source_dataset_id
        self.applicability_calls[proposal.proposal_id] = (
            self.applicability_calls.get(proposal.proposal_id, 0) + 1
        )
        return ApplicabilityDecision.applicable()

    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        proposal_id = proposal.proposal_id
        self.executions[proposal_id] = self.executions.get(proposal_id, 0) + 1
        steps = int(proposal.transformed_action.actions.shape[0])
        corrupted_steps = steps - 1 if self._corrupt_action_count else steps
        metrics: dict[str, int | float | bool | str | None] = {
            "pickcube_candidate_horizon_steps": 16,
            "pickcube_prefix_evaluated_before_continuation": True,
            "replay_baseline_requested_steps": steps,
            "replay_baseline_steps": steps,
            "replay_baseline_valid": True,
            "replay_corrupted_requested_steps": steps,
            "replay_corrupted_steps": corrupted_steps,
        }
        for role in ("baseline", "corrupted"):
            metrics[f"replay_{role}_restoration_complete_state_comparison"] = True
            metrics[f"replay_{role}_restoration_compared_component_count"] = (
                self._component_count
            )
            metrics[f"replay_{role}_restoration_maximum_absolute_error"] = 1.1920929e-7
        return EvaluationEvidence(
            evidence_id=compute_evidence_identifier(
                proposal_id=proposal_id,
                evaluator_id=self.evaluator_id,
                evaluator_version=self.evaluator_version,
                evaluator_configuration_digest=self.configuration_digest,
                evaluation_seed=evaluation_seed,
                attempt_ordinal=attempt_ordinal,
            ),
            proposal_id=proposal_id,
            source_dataset_id=source_dataset_id,
            source_episode_id=proposal.source_episode_id,
            source_candidate_id=proposal.source_candidate_id,
            split_group_id=proposal.split_group_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            evaluator_configuration_digest=self.configuration_digest,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
            status=EvaluationStatus.CONCLUSIVE,
            success=proposal.generation_ordinal % 2 == 0,
            progress_before=0.0,
            progress_after=1.0,
            progress_delta=1.0,
            unsafe=False,
            failure_events=(),
            termination_reason="paired_replay_complete",
            replayed_control_steps=steps * 2,
            metrics=metrics,
            artifact_references=(),
            label_source=LabelSource.SIMULATOR,
            label_strength=LabelStrength.STRONG,
            simulator_replay_verified=True,
            notes="strong CPU fake replay evidence",
            schema_version=EVALUATION_EVIDENCE_SCHEMA_VERSION,
        )


def _dataset() -> CorruptionDataset:
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=24,
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
    return CorruptionDataset(
        source_dataset_id="sha256:" + "c" * 64,
        action_layout=plan.action_layout,
        proposals=generated.proposals,
    )


def _artifacts(
    dataset: CorruptionDataset,
) -> tuple[_Pool, _Manifest]:
    proposal_ids = tuple(item.proposal_id for item in dataset.proposals)
    pool = _Pool(content_digest=_sha("candidate-pool"), proposal_ids=proposal_ids)
    manifest = _Manifest(
        content_digest=_sha("selection-manifest"),
        candidate_pool_digest=pool.content_digest,
        selections=(
            _Selection(proposal_ids[0]),
            _Selection(proposal_ids[0]),
            _Selection(proposal_ids[1]),
            _Selection(None, abstained=True),
        ),
    )
    return pool, manifest


def _environment() -> RunEnvironment:
    return RunEnvironment(
        git_commit_sha=None,
        git_branch=None,
        python_version="3.11.15",
        numpy_version="1.26.4",
        platform="cpu-test",
        launch_command="latentguard replay-selected-candidates [sanitized]",
    )


def _run(
    dataset: CorruptionDataset,
    evaluator: ManifestGatedReplayEvaluator,
    output_dir: Path,
    *,
    resume: bool = False,
):
    return run_evaluation(
        dataset,
        evaluator,
        source_corruption_dataset_digest=compute_corruption_dataset_content_digest(
            dataset
        ),
        output_dir=output_dir,
        base_seed=161803,
        resume=resume,
        run_environment=_environment(),
    )


def _replace_operational_metadata(
    dataset: EvaluationDataset,
    *,
    timestamp: str,
) -> EvaluationDataset:
    environment = replace(
        dataset.run_manifest.environment,
        platform="different-cpu-test",
        launch_command="latentguard replay-selected-candidates different-sanitized-run",
    )
    ledger = tuple(
        replace(
            item,
            started_at=timestamp if item.started_at is not None else None,
            finished_at=timestamp if item.finished_at is not None else None,
        )
        for item in dataset.ledger
    )
    return replace(
        dataset,
        ledger=ledger,
        run_manifest=replace(
            dataset.run_manifest,
            environment=environment,
            started_at=timestamp,
            finished_at=(
                timestamp if dataset.run_manifest.finished_at is not None else None
            ),
        ),
    )


def test_selected_union_deduplicates_physical_execution_and_binds_identity(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    inner = _StrongReplayEvaluator()
    phases = build_manifest_gated_replay_evaluators(
        inner, candidate_pool=pool, selection_manifest=manifest
    )

    result = _run(dataset, phases.selected, tmp_path / "selected")
    proposal_ids = pool.proposal_ids

    assert phases.selected.phase is ReplayPhase.SELECTED
    assert phases.selected.executable_proposal_ids == proposal_ids[:2]
    assert phases.remainder.executable_proposal_ids == proposal_ids[2:]
    assert not (
        set(phases.selected.executable_proposal_ids)
        & set(phases.remainder.executable_proposal_ids)
    )
    assert set(phases.selected.executable_proposal_ids) | set(
        phases.remainder.executable_proposal_ids
    ) == set(proposal_ids)
    assert inner.executions == {proposal_ids[0]: 1, proposal_ids[1]: 1}
    assert result.dataset.summary.conclusive == 2
    assert result.dataset.summary.skipped == len(proposal_ids) - 2
    assert result.dataset.evaluator_id == MANIFEST_GATED_REPLAY_EVALUATOR_ID
    configuration = phases.selected.resolved_configuration()
    assert configuration["phase"] == "selected"
    assert configuration["candidate_pool_digest"] == pool.content_digest
    assert configuration["selection_manifest_digest"] == manifest.content_digest
    assert configuration["executable_proposal_ids"] == proposal_ids[:2]


def test_complementary_phases_execute_every_candidate_once_and_join(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    inner = _StrongReplayEvaluator()
    phases = build_manifest_gated_replay_evaluators(
        inner, candidate_pool=pool, selection_manifest=manifest
    )

    selected = _run(dataset, phases.selected, tmp_path / "selected")
    remainder = _run(dataset, phases.remainder, tmp_path / "remainder")
    joined = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=manifest,
        selected_dataset=selected.dataset,
        remainder_dataset=remainder.dataset,
    )

    assert inner.executions == {proposal_id: 1 for proposal_id in pool.proposal_ids}
    assert tuple(joined.evidence_by_proposal) == pool.proposal_ids
    assert joined.selected_phase_executed_ids == pool.proposal_ids[:2]
    assert joined.remainder_phase_executed_ids == pool.proposal_ids[2:]
    assert joined.content_digest.startswith("sha256:")
    assert all(
        evidence.label_strength is LabelStrength.STRONG
        and evidence.simulator_replay_verified
        for evidence in joined.evidence_by_proposal.values()
    )


def test_join_semantic_digest_binds_task_evidence_and_separates_archive_audit(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    phases = build_manifest_gated_replay_evaluators(
        _StrongReplayEvaluator(), candidate_pool=pool, selection_manifest=manifest
    )
    selected = _run(dataset, phases.selected, tmp_path / "selected")
    remainder = _run(dataset, phases.remainder, tmp_path / "remainder")
    original = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=manifest,
        selected_dataset=selected.dataset,
        remainder_dataset=remainder.dataset,
    )

    first = selected.dataset.evidence[0]
    changed_evidence = replace(
        first,
        metrics={**first.metrics, "content_binding_probe": 1},
    )
    changed_selected = replace(
        selected.dataset,
        evidence=(changed_evidence, *selected.dataset.evidence[1:]),
    )
    changed = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=manifest,
        selected_dataset=changed_selected,
        remainder_dataset=remainder.dataset,
    )

    assert changed.selected_phase_dataset_digest != (
        original.selected_phase_dataset_digest
    )
    assert changed.content_digest != original.content_digest
    assert changed.archive_audit_digest != original.archive_audit_digest

    changed_outcome = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=manifest,
        selected_dataset=replace(
            selected.dataset,
            evidence=(
                replace(first, success=not first.success),
                *selected.dataset.evidence[1:],
            ),
        ),
        remainder_dataset=remainder.dataset,
    )
    assert changed_outcome.content_digest != original.content_digest


def test_join_semantic_digest_excludes_run_environment_and_timestamps(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    phases = build_manifest_gated_replay_evaluators(
        _StrongReplayEvaluator(), candidate_pool=pool, selection_manifest=manifest
    )
    selected = _run(dataset, phases.selected, tmp_path / "selected")
    remainder = _run(dataset, phases.remainder, tmp_path / "remainder")
    original = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=manifest,
        selected_dataset=selected.dataset,
        remainder_dataset=remainder.dataset,
    )
    timestamp = "2040-01-01T00:00:00+00:00"
    changed_selected = _replace_operational_metadata(
        selected.dataset, timestamp=timestamp
    )
    changed_remainder = _replace_operational_metadata(
        remainder.dataset, timestamp=timestamp
    )
    changed = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=manifest,
        selected_dataset=changed_selected,
        remainder_dataset=changed_remainder,
    )

    assert compute_evaluation_dataset_content_digest(changed_selected) != (
        compute_evaluation_dataset_content_digest(selected.dataset)
    )
    assert changed.selected_phase_dataset_digest != (
        original.selected_phase_dataset_digest
    )
    assert changed.remainder_phase_dataset_digest != (
        original.remainder_phase_dataset_digest
    )
    assert changed.content_digest == original.content_digest
    assert changed.archive_audit_digest != original.archive_audit_digest


def test_join_semantic_digest_excludes_manifest_envelope(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    original_phases = build_manifest_gated_replay_evaluators(
        _StrongReplayEvaluator(), candidate_pool=pool, selection_manifest=manifest
    )
    original_selected = _run(
        dataset, original_phases.selected, tmp_path / "original-selected"
    )
    original_remainder = _run(
        dataset, original_phases.remainder, tmp_path / "original-remainder"
    )
    original = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=manifest,
        selected_dataset=original_selected.dataset,
        remainder_dataset=original_remainder.dataset,
    )

    changed_manifest = replace(
        manifest,
        manifest_envelope_digest=_sha("different-manifest-envelope-timestamp"),
    )
    changed_phases = build_manifest_gated_replay_evaluators(
        _StrongReplayEvaluator(),
        candidate_pool=pool,
        selection_manifest=changed_manifest,
    )
    changed_selected = _run(
        dataset, changed_phases.selected, tmp_path / "changed-selected"
    )
    changed_remainder = _run(
        dataset, changed_phases.remainder, tmp_path / "changed-remainder"
    )
    changed = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=changed_manifest,
        selected_dataset=changed_selected.dataset,
        remainder_dataset=changed_remainder.dataset,
    )

    assert changed.selection_manifest_digest == original.selection_manifest_digest
    assert changed.selection_manifest_envelope_digest != (
        original.selection_manifest_envelope_digest
    )
    assert changed.selected_phase_dataset_digest != (
        original.selected_phase_dataset_digest
    )
    assert changed.content_digest == original.content_digest
    assert changed.archive_audit_digest != original.archive_audit_digest


def test_resume_is_zero_work_and_does_not_duplicate_evidence(tmp_path: Path) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    inner = _StrongReplayEvaluator()
    phases = build_manifest_gated_replay_evaluators(
        inner, candidate_pool=pool, selection_manifest=manifest
    )
    output = tmp_path / "selected"
    first = _run(dataset, phases.selected, output)
    executions = dict(inner.executions)

    resumed = _run(dataset, phases.selected, output, resume=True)
    report = verify_zero_work_resume(first.dataset, resumed)

    assert report.zero_duplicate_work is True
    assert report.evidence_count == len(dataset.proposals)
    assert report.evaluated_attempts == 0
    assert inner.executions == executions


def test_changed_manifest_or_phase_rejects_resume(tmp_path: Path) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    first_inner = _StrongReplayEvaluator()
    first = build_manifest_gated_replay_evaluators(
        first_inner, candidate_pool=pool, selection_manifest=manifest
    )
    output = tmp_path / "selected"
    _run(dataset, first.selected, output)
    changed = _Manifest(
        content_digest=_sha("changed-selection-manifest"),
        candidate_pool_digest=pool.content_digest,
        selections=(_Selection(pool.proposal_ids[2]),),
    )
    changed_phases = build_manifest_gated_replay_evaluators(
        _StrongReplayEvaluator(), candidate_pool=pool, selection_manifest=changed
    )

    with pytest.raises(EvaluationRunConflictError):
        _run(dataset, changed_phases.selected, output, resume=True)
    with pytest.raises(EvaluationRunConflictError):
        _run(dataset, first.remainder, output, resume=True)

    changed_envelope = _Manifest(
        content_digest=manifest.content_digest,
        candidate_pool_digest=pool.content_digest,
        selections=manifest.selections,
        manifest_envelope_digest=_sha("changed-manifest-envelope"),
    )
    envelope_phases = build_manifest_gated_replay_evaluators(
        _StrongReplayEvaluator(),
        candidate_pool=pool,
        selection_manifest=changed_envelope,
    )
    with pytest.raises(EvaluationRunConflictError):
        _run(dataset, envelope_phases.selected, output, resume=True)


def test_bound_manifest_mutation_is_rejected_before_inner_execution() -> None:
    dataset = _dataset()
    pool, frozen = _artifacts(dataset)
    manifest = _MutableManifest(
        content_digest=frozen.content_digest,
        candidate_pool_digest=frozen.candidate_pool_digest,
        selections=list(frozen.selections),
    )
    inner = _StrongReplayEvaluator()
    evaluator = ManifestGatedReplayEvaluator(
        inner,
        phase=ReplayPhase.SELECTED,
        candidate_pool=pool,
        selection_manifest=manifest,
    )
    manifest.selections[0] = _Selection(pool.proposal_ids[3])

    with pytest.raises(SelectionReplayError, match="changed after binding"):
        evaluator.check_applicability(
            dataset.proposals[0], source_dataset_id=dataset.source_dataset_id
        )
    assert inner.applicability_calls == {}
    assert inner.executions == {}


def test_unknown_selection_and_unknown_proposal_fail_closed() -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    unknown_manifest = _Manifest(
        content_digest=_sha("unknown-selection"),
        candidate_pool_digest=pool.content_digest,
        selections=(_Selection("cap-sha256-" + "f" * 64),),
    )
    with pytest.raises(SelectionReplayError, match="unknown proposal"):
        build_manifest_gated_replay_evaluators(
            _StrongReplayEvaluator(),
            candidate_pool=pool,
            selection_manifest=unknown_manifest,
        )

    reduced_pool = _Pool(
        content_digest=_sha("reduced-pool"),
        proposal_ids=pool.proposal_ids[:-1],
    )
    reduced_manifest = _Manifest(
        content_digest=_sha("reduced-manifest"),
        candidate_pool_digest=reduced_pool.content_digest,
        selections=(_Selection(reduced_pool.proposal_ids[0]),),
    )
    evaluator = ManifestGatedReplayEvaluator(
        _StrongReplayEvaluator(),
        phase=ReplayPhase.SELECTED,
        candidate_pool=reduced_pool,
        selection_manifest=reduced_manifest,
    )
    decision = evaluator.check_applicability(
        dataset.proposals[-1], source_dataset_id=dataset.source_dataset_id
    )
    assert decision.status is EvaluationStatus.INVALID
    assert decision.reason == "proposal_not_in_candidate_pool"


@pytest.mark.parametrize(
    ("inner", "error"),
    [
        (_StrongReplayEvaluator(component_count=69), "all 70 state components"),
        (_StrongReplayEvaluator(corrupt_action_count=True), "action count"),
    ],
)
def test_strong_replay_gate_rejects_partial_state_or_action_evidence(
    inner: _StrongReplayEvaluator,
    error: str,
) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    evaluator = ManifestGatedReplayEvaluator(
        inner,
        phase=ReplayPhase.SELECTED,
        candidate_pool=pool,
        selection_manifest=manifest,
    )
    proposal = dataset.proposals[0]

    with pytest.raises(SelectionReplayError, match=error):
        evaluator.evaluate(
            proposal,
            source_dataset_id=dataset.source_dataset_id,
            evaluation_seed=17,
            attempt_ordinal=0,
        )


def test_join_rejects_duplicate_or_unknown_phase_evidence(tmp_path: Path) -> None:
    dataset = _dataset()
    pool, manifest = _artifacts(dataset)
    phases = build_manifest_gated_replay_evaluators(
        _StrongReplayEvaluator(), candidate_pool=pool, selection_manifest=manifest
    )
    selected = _run(dataset, phases.selected, tmp_path / "selected")
    remainder = _run(dataset, phases.remainder, tmp_path / "remainder")
    duplicate = selected.dataset.evidence[0]
    object.__setattr__(
        selected.dataset,
        "evidence",
        (*selected.dataset.evidence[:-1], duplicate),
    )

    with pytest.raises(SelectionReplayError, match="unknown or duplicate"):
        join_complementary_replay_phases(
            candidate_pool=pool,
            selection_manifest=manifest,
            selected_dataset=selected.dataset,
            remainder_dataset=remainder.dataset,
        )


def test_selection_manifest_pool_digest_and_abstention_are_strict() -> None:
    dataset = _dataset()
    pool, _ = _artifacts(dataset)
    wrong_pool = _Manifest(
        content_digest=_sha("wrong-pool-manifest"),
        candidate_pool_digest=_sha("another-pool"),
        selections=(_Selection(pool.proposal_ids[0]),),
    )
    with pytest.raises(SelectionReplayError, match="different candidate pool"):
        build_manifest_gated_replay_evaluators(
            _StrongReplayEvaluator(),
            candidate_pool=pool,
            selection_manifest=wrong_pool,
        )

    invalid_abstention = _Manifest(
        content_digest=_sha("invalid-abstention"),
        candidate_pool_digest=pool.content_digest,
        selections=(_Selection(pool.proposal_ids[0], abstained=True),),
    )
    with pytest.raises(SelectionReplayError, match="abstained"):
        build_manifest_gated_replay_evaluators(
            _StrongReplayEvaluator(),
            candidate_pool=pool,
            selection_manifest=invalid_abstention,
        )
