from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from latentguard.action_verifier import (
    CandidateType,
    DatasetSplit,
    load_action_verifier_dataset,
    save_action_verifier_dataset,
)
from latentguard.corruptions import (
    ActionField,
    ActionLayout,
    ActionSemantic,
    ConstantBias,
    CorruptionDataset,
    WindowScopedCorruption,
    generate_corruption_proposals,
)
from latentguard.evaluation.models import (
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.evaluation.reporting import build_evaluation_summary
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
)
from latentguard.integrations.maniskill_pickcube import (
    action_verifier_export as export_module,
)
from latentguard.integrations.maniskill_pickcube.action_verifier_export import (
    ACTION_VERIFIER_SUMMARY_NAME,
    ACTION_VERIFIER_VALIDATION_NAME,
    ActionVerifierExportError,
    collect_action_verifier_evidence_ids,
    export_action_verifier_dataset,
    save_action_verifier_export_reports,
    validate_action_verifier_evidence_bindings,
    validate_serialized_action_verifier_export,
)
from latentguard.integrations.maniskill_pickcube.adapter import (
    MANISKILL_PICKCUBE_ADAPTER_ID,
    MANISKILL_PICKCUBE_ADAPTER_VERSION,
)
from latentguard.integrations.maniskill_pickcube.anchors import PickCubeStateAnchor
from latentguard.integrations.maniskill_pickcube.source_import import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_UNSAFE_SEMANTIC,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    PICKCUBE_STATE_COMPARISON_SEMANTIC,
    PICKCUBE_STATE_COMPARISON_TOLERANCE,
    AnchorBaselineEvidenceV1,
    PickCubeActionControlContractV1,
    PickCubeAnchorBuildResult,
    SourceContinuationIdentityV1,
    build_state_indexed_anchor_sources,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    build_pickcube_verifier_state_v1,
)
from latentguard.models import FailureEvent, LabelSource, LabelStrength
from latentguard.replay.source import compute_episode_content_digest

_COMPATIBILITY = f"sha256:{'a' * 64}"
_SOURCE_DATASET_ID = f"sha256:{hashlib.sha256(b'source-bundle').hexdigest()}"
_STARTED = "2026-07-15T00:00:00+00:00"
_FINISHED = "2026-07-15T00:00:01+00:00"


def _task(index: int, action_count: int) -> PickCubeTaskSnapshotV1:
    terminal = index == action_count
    return PickCubeTaskSnapshotV1(
        success=terminal,
        is_obj_placed=terminal,
        is_robot_static=terminal,
        is_grasped=index >= 4,
        cube_center_z=0.02,
        cube_to_goal_distance=float(action_count - index) * 0.01,
        tcp_to_cube_distance=float(max(4 - index, 0)) * 0.01,
    )


def _state(index: int, action_count: int) -> PickCubeIndexedStateV1:
    terminal = index == action_count
    return PickCubeIndexedStateV1(
        state_index=index,
        source_action_index=index,
        tree={
            "actors": np.array([index, index + 0.25], dtype=np.float32),
            "articulations": {
                "qpos": np.array([index * 0.1, -index * 0.1], dtype=np.float64)
            },
        },
        task_snapshot=_task(index, action_count),
        restored_task_snapshot=_task(index, action_count),
        verifier_state=build_pickcube_verifier_state_v1(
            joint_names=("joint_a", "joint_b"),
            qpos=np.array([index, -index], dtype=np.float32),
            qvel=np.array([0.1, -0.1], dtype=np.float32),
            tcp_position=np.array([0.0, 0.0, 0.1 + index], dtype=np.float32),
            tcp_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            cube_position=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            cube_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            goal_position=np.array([0.4, 0.5, 0.6], dtype=np.float32),
            is_grasped=index >= 4,
            is_obj_placed=terminal,
            is_robot_static=terminal,
        ),
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_trajectory_id="mspc-trajectory-source-a",
    )


def _archive() -> PickCubeStateIndexedArchiveV1:
    action_count = 20
    actions = np.arange(action_count * 2, dtype=np.float32).reshape(action_count, 2)
    actions /= 100.0
    episode = PickCubeStateIndexedEpisodeV1(
        episode_id="mspc-sequence-episode-a",
        source_trajectory_id="mspc-trajectory-source-a",
        source_policy_identity="maniskill/3.0.1/official-pickcube-solver-v1",
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_actions=actions,
        states=tuple(_state(index, action_count) for index in range(action_count + 1)),
    )
    return PickCubeStateIndexedArchiveV1(episodes=(episode,))


def _contract() -> PickCubeActionControlContractV1:
    return PickCubeActionControlContractV1(
        coordinate_frame="joint_position",
        control_period_s=0.05,
        action_dtype="<f4",
        action_dimension=2,
    )


@dataclass
class _BaselineValidator:
    def validate_anchor_baseline(
        self,
        *,
        archive_content_digest: str,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        anchor: PickCubeStateAnchor,
        source_remaining_actions: NDArray[Any],
        continuation: SourceContinuationIdentityV1,
        action_control_contract: PickCubeActionControlContractV1,
    ) -> AnchorBaselineEvidenceV1:
        del archive_content_digest, source_remaining_actions, action_control_contract
        return AnchorBaselineEvidenceV1(
            anchor_id=anchor.anchor_id,
            source_archive_episode_id=source_episode.episode_id,
            source_trajectory_id=source_episode.source_trajectory_id,
            source_seed=source_episode.seed,
            state_index=anchor.state_index,
            source_state_digest=source_state.state_digest,
            compatibility_identity=source_episode.compatibility_identity,
            continuation_identity=continuation.content_digest,
            expected_action_count=anchor.remaining_horizon,
            executed_action_count=anchor.remaining_horizon,
            complete_action_execution=True,
            state_restoration_verified=True,
            complete_state_comparison=True,
            compared_component_count=source_state.numeric_component_count,
            maximum_absolute_error=1.1920929e-7,
            verifier_state_restoration_verified=True,
            verifier_state_compared_component_count=(
                source_state.verifier_state.values.size
            ),
            verifier_state_maximum_absolute_error=1.1920929e-7,
            comparison_semantic=PICKCUBE_STATE_COMPARISON_SEMANTIC,
            comparison_tolerance=PICKCUBE_STATE_COMPARISON_TOLERANCE,
            official_terminal_success=True,
            official_object_placed=True,
            official_robot_static=True,
            progress_before=0.0,
            progress_after_candidate=0.0,
            progress_delta=0.0,
            terminal_progress=1.0,
            progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
            terminal_unsafe=False,
            unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
            label_source=LabelSource.SIMULATOR,
            label_strength=LabelStrength.STRONG,
            simulator_replay_verified=True,
        )


def _anchor_build() -> PickCubeAnchorBuildResult:
    return build_state_indexed_anchor_sources(
        _archive(),
        action_control_contract=_contract(),
        baseline_validator=_BaselineValidator(),
        candidate_horizon=16,
        maximum_anchors_per_trajectory=1,
    )


def _corruptions(build: PickCubeAnchorBuildResult) -> CorruptionDataset:
    layout = ActionLayout(
        action_dim=2,
        fields=(
            ActionField(
                name="arm",
                indices=(0, 1),
                semantic=ActionSemantic.AUXILIARY,
            ),
        ),
    )
    definitions = tuple(
        WindowScopedCorruption(
            inner=ConstantBias(bias=value, target_indices=(0,)),
            window_start=0,
            window_end=16,
            severity_id=severity,
        )
        for value, severity in (
            (0.01, "mild_success"),
            (0.1, "severe_failure"),
            (0.02, "mild_weak"),
            (0.03, "moderate_indeterminate"),
        )
    )
    generated = generate_corruption_proposals(
        build.source_episodes,
        layout,
        definitions,
        base_seed=42,
        strict_applicability=True,
    )
    return CorruptionDataset(
        source_dataset_id=_SOURCE_DATASET_ID,
        action_layout=layout,
        proposals=generated.proposals,
    )


def _configuration(
    build: PickCubeAnchorBuildResult, corruption: CorruptionDataset
) -> Mapping[str, object]:
    adapter_configuration = {
        "adapter_id": MANISKILL_PICKCUBE_ADAPTER_ID,
        "adapter_version": MANISKILL_PICKCUBE_ADAPTER_VERSION,
        "identity": {"compatibility_identity": _COMPATIBILITY},
        "progress_semantic": PICKCUBE_PROGRESS_SEMANTIC,
        "state_verification": {
            "comparison_semantic": "numeric_tolerance",
            "maximum_absolute_tolerance": PICKCUBE_STATE_COMPARISON_TOLERANCE,
            "runtime_semantic": PICKCUBE_STATE_COMPARISON_SEMANTIC,
        },
        "unsafe_semantic": PICKCUBE_UNSAFE_SEMANTIC,
    }
    return {
        "adapter_configuration": adapter_configuration,
        "adapter_configuration_digest": compute_configuration_digest(
            adapter_configuration
        ),
        "adapter_id": MANISKILL_PICKCUBE_ADAPTER_ID,
        "adapter_version": MANISKILL_PICKCUBE_ADAPTER_VERSION,
        "corruption_dataset_digest": compute_corruption_dataset_content_digest(
            corruption
        ),
        "exact_state_verification_required": True,
        "label_source": LabelSource.SIMULATOR.value,
        "maximum_label_strength": LabelStrength.STRONG.value,
        "progress_semantic": PICKCUBE_PROGRESS_SEMANTIC,
        "simulator_verification_allowed": True,
        "source_dataset_digest": compute_episode_content_digest(build.source_episodes),
        "source_dataset_id": _SOURCE_DATASET_ID,
        "state_verification_semantic": "numeric_tolerance",
        "state_verification_tolerance": PICKCUBE_STATE_COMPARISON_TOLERANCE,
        "trust_tier": "exact_simulator",
        "unsafe_semantic": PICKCUBE_UNSAFE_SEMANTIC,
    }


def _metrics(build: PickCubeAnchorBuildResult, *, after: float) -> dict[str, object]:
    remaining = build.source_episodes[0].candidates[0].action.actions.shape[0]
    components = build.manifest.baseline_evidence[0].compared_component_count
    return {
        "pickcube_candidate_horizon_steps": 16,
        "pickcube_candidate_progress_semantic": PICKCUBE_PROGRESS_SEMANTIC,
        "pickcube_progress_before_candidate": 0.0,
        "pickcube_progress_after_candidate": after,
        "pickcube_progress_delta_candidate": after,
        "pickcube_prefix_evaluated_before_continuation": True,
        "replay_baseline_requested_steps": remaining,
        "replay_baseline_steps": remaining,
        "replay_corrupted_requested_steps": remaining,
        "replay_corrupted_steps": remaining,
        "replay_baseline_restoration_compared_component_count": components,
        "replay_baseline_restoration_complete_state_comparison": True,
        "replay_baseline_restoration_maximum_absolute_error": 1.1920929e-7,
        "replay_corrupted_restoration_compared_component_count": components,
        "replay_corrupted_restoration_complete_state_comparison": True,
        "replay_corrupted_restoration_maximum_absolute_error": 1.1920929e-7,
    }


def _evaluation(
    build: PickCubeAnchorBuildResult,
    corruption: CorruptionDataset,
) -> EvaluationDataset:
    configuration = _configuration(build, corruption)
    configuration_digest = compute_configuration_digest(configuration)
    evaluator_id = "exact_state_paired_replay"
    evaluator_version = "1.0.0"
    base_seed = 271828
    selected = tuple(proposal.proposal_id for proposal in corruption.proposals)
    corruption_digest = compute_corruption_dataset_content_digest(corruption)
    run_id = compute_run_identifier(
        source_corruption_dataset_digest=corruption_digest,
        source_dataset_id=_SOURCE_DATASET_ID,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        evaluator_configuration_digest=configuration_digest,
        base_seed=base_seed,
        max_proposals=None,
        selected_proposal_ids=selected,
    )
    evidence: list[EvaluationEvidence] = []
    ledger: list[LedgerEntry] = []
    for index, proposal in enumerate(corruption.proposals):
        seed = compute_evaluation_seed(
            base_seed=base_seed,
            proposal_id=proposal.proposal_id,
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
            evaluator_configuration_digest=configuration_digest,
            attempt_ordinal=0,
        )
        evidence_id = compute_evidence_identifier(
            proposal_id=proposal.proposal_id,
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
            evaluator_configuration_digest=configuration_digest,
            evaluation_seed=seed,
            attempt_ordinal=0,
        )
        if index < 3:
            strong = index < 2
            success = index != 1
            item = EvaluationEvidence(
                evidence_id=evidence_id,
                proposal_id=proposal.proposal_id,
                source_dataset_id=_SOURCE_DATASET_ID,
                source_episode_id=proposal.source_episode_id,
                source_candidate_id=proposal.source_candidate_id,
                split_group_id=proposal.split_group_id,
                evaluator_id=evaluator_id,
                evaluator_version=evaluator_version,
                evaluator_configuration_digest=configuration_digest,
                evaluation_seed=seed,
                attempt_ordinal=0,
                status=EvaluationStatus.CONCLUSIVE,
                success=success,
                progress_before=0.0,
                progress_after=1.0 if success else 0.0,
                progress_delta=1.0 if success else 0.0,
                unsafe=False,
                failure_events=(
                    () if success else (FailureEvent(failure_type="task_failure"),)
                ),
                termination_reason="completed",
                replayed_control_steps=(
                    2 * build.source_episodes[0].candidates[0].action.actions.shape[0]
                ),
                metrics=_metrics(build, after=0.0),
                artifact_references=(),
                label_source=(
                    LabelSource.SIMULATOR
                    if strong
                    else LabelSource.DETERMINISTIC_EVALUATOR
                ),
                label_strength=(LabelStrength.STRONG if strong else LabelStrength.WEAK),
                simulator_replay_verified=strong,
                notes="fake CPU evidence",
            )
            ledger_state = LedgerState.COMPLETED
        else:
            item = EvaluationEvidence(
                evidence_id=evidence_id,
                proposal_id=proposal.proposal_id,
                source_dataset_id=_SOURCE_DATASET_ID,
                source_episode_id=proposal.source_episode_id,
                source_candidate_id=proposal.source_candidate_id,
                split_group_id=proposal.split_group_id,
                evaluator_id=evaluator_id,
                evaluator_version=evaluator_version,
                evaluator_configuration_digest=configuration_digest,
                evaluation_seed=seed,
                attempt_ordinal=0,
                status=EvaluationStatus.INDETERMINATE,
                success=None,
                progress_before=None,
                progress_after=None,
                progress_delta=None,
                unsafe=None,
                failure_events=(),
                termination_reason="inconclusive",
                replayed_control_steps=0,
                metrics={},
                artifact_references=(),
                label_source=None,
                label_strength=None,
                simulator_replay_verified=False,
                notes="fake CPU evidence",
            )
            ledger_state = LedgerState.INDETERMINATE
        evidence.append(item)
        ledger.append(
            LedgerEntry(
                proposal_id=proposal.proposal_id,
                evidence_id=item.evidence_id,
                attempt_ordinal=0,
                evaluator_id=evaluator_id,
                evaluator_version=evaluator_version,
                evaluator_configuration_digest=configuration_digest,
                evaluation_seed=seed,
                state=ledger_state,
                error_type=None,
                error_message=None,
                retry_eligible=False,
                started_at=_STARTED,
                finished_at=_FINISHED,
            )
        )
    environment = RunEnvironment(
        git_commit_sha=None,
        git_branch=None,
        python_version="3.11.14",
        numpy_version="1.26.4",
        platform="test-platform",
        launch_command="latentguard replay-state-indexed-pickcube [sanitized]",
    )
    manifest = RunManifest(
        environment=environment,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        evaluator_configuration_digest=configuration_digest,
        source_corruption_dataset_digest=corruption_digest,
        source_dataset_id=_SOURCE_DATASET_ID,
        seed=base_seed,
        started_at=_STARTED,
        finished_at=_FINISHED,
        final_state=RunState.COMPLETE,
    )
    return EvaluationDataset(
        source_corruption_dataset_digest=corruption_digest,
        source_dataset_id=_SOURCE_DATASET_ID,
        evaluator_id=evaluator_id,
        evaluator_version=evaluator_version,
        resolved_evaluator_configuration=configuration,
        evaluator_configuration_digest=configuration_digest,
        run_id=run_id,
        base_seed=base_seed,
        max_proposals=None,
        selected_proposal_ids=selected,
        evidence=tuple(evidence),
        ledger=tuple(ledger),
        summary=build_evaluation_summary(evidence, ledger),
        artifact_references=(),
        run_state=RunState.COMPLETE,
        run_manifest=manifest,
    )


def _inputs() -> tuple[PickCubeAnchorBuildResult, CorruptionDataset, EvaluationDataset]:
    build = _anchor_build()
    corruption = _corruptions(build)
    return build, corruption, _evaluation(build, corruption)


def _export(
    build: PickCubeAnchorBuildResult,
    corruption: CorruptionDataset,
    evaluation: EvaluationDataset,
):
    return export_action_verifier_dataset(
        anchor_manifest=build.manifest,
        source_episodes=build.source_episodes,
        source_dataset_id=_SOURCE_DATASET_ID,
        corruption_dataset=corruption,
        evaluation_dataset=evaluation,
        split_counts={
            DatasetSplit.TRAIN: 1,
            DatasetSplit.VALIDATION: 0,
            DatasetSplit.TEST: 0,
        },
        split_seed=9,
    )


def test_export_keeps_only_conclusive_strong_simulator_evidence() -> None:
    build, corruption, evaluation = _inputs()
    result = _export(build, corruption, evaluation)
    dataset = result.dataset
    source = tuple(
        sample
        for sample in dataset.samples
        if sample.candidate_type is CandidateType.SOURCE
    )
    corrupted = tuple(
        sample
        for sample in dataset.samples
        if sample.candidate_type is CandidateType.CORRUPTED
    )

    assert len(source) == 1
    assert source[0].state_content_digest == (
        build.manifest.records[0].source_state_digest
    )
    assert source[0].state_content_digest != (
        build.manifest.records[0].source_state_content_digest
    )
    assert source[0].strong_simulator_evidence_id.startswith("mspc-anchor-baseline-")
    assert len(corrupted) == 2
    assert {sample.final_task_success for sample in corrupted} == {True, False}
    assert all(
        sample.candidate_action_chunk.shape == (16, 2) for sample in dataset.samples
    )
    assert all(sample.action_mask.all() for sample in dataset.samples)
    assert result.summary.corrupted_success_count == 1
    assert result.summary.corrupted_failure_count == 1
    assert result.summary.exclusion_reason_counts == {
        "not_strong_simulator_verified": 1,
        "status_indeterminate": 1,
    }
    assert result.summary.split_trajectory_counts == {
        "test": 0,
        "train": 1,
        "validation": 0,
    }
    assert len(result.available_evidence_ids) == 3
    assert result.validation_report.valid
    assert dataset.split_assignments[0].state_digests == tuple(
        sorted(
            set(
                build.manifest.trajectory_state_digests[
                    build.manifest.records[0].anchor.source_trajectory_id
                ]
            )
        )
    )
    assert len(dataset.split_assignments[0].state_digests) > len(
        {sample.state_content_digest for sample in dataset.samples}
    )


def test_evidence_inventory_excludes_weak_and_indeterminate_attempts() -> None:
    build, _, evaluation = _inputs()
    inventory = collect_action_verifier_evidence_ids(build.manifest, evaluation)

    assert inventory == tuple(sorted(inventory))
    assert set(inventory) == {
        build.manifest.baseline_evidence[0].evidence_id,
        evaluation.evidence[0].evidence_id,
        evaluation.evidence[1].evidence_id,
    }
    assert evaluation.evidence[2].evidence_id not in inventory
    assert evaluation.evidence[3].evidence_id not in inventory


def test_anchor_source_state_vector_bytes_are_rebound_to_manifest() -> None:
    build = _anchor_build()
    episode = build.source_episodes[0]
    frame = episode.observations.frames[0]
    changed_state = frame.robot_state.copy()
    changed_state[0] += np.float32(1.0)
    changed_frame = replace(frame, robot_state=changed_state)
    changed_episode = replace(
        episode,
        observations=replace(episode.observations, frames=(changed_frame,)),
    )

    with pytest.raises(ActionVerifierExportError, match="verifier state bytes"):
        export_module._require_source_record_binding(
            build.manifest.records[0], changed_episode
        )


def test_anchor_source_continuation_bytes_are_rebound_to_manifest() -> None:
    build = _anchor_build()
    episode = build.source_episodes[0]
    candidate = episode.candidates[0]
    changed_actions = candidate.action.actions.copy()
    changed_actions[-1, 0] += np.float32(1.0)
    changed_candidate = replace(
        candidate,
        action=replace(candidate.action, actions=changed_actions),
    )
    changed_episode = replace(episode, candidates=(changed_candidate,))

    with pytest.raises(ActionVerifierExportError, match="remaining action bytes"):
        export_module._require_source_record_binding(
            build.manifest.records[0], changed_episode
        )


def test_fixed_adapter_contract_rejects_tolerance_drift() -> None:
    build, corruption, evaluation = _inputs()
    changed_configuration = dict(evaluation.resolved_evaluator_configuration)
    changed_configuration["state_verification_tolerance"] = 1e-5
    object.__setattr__(
        evaluation,
        "resolved_evaluator_configuration",
        changed_configuration,
    )

    with pytest.raises(
        ActionVerifierExportError,
        match="state_verification_tolerance.*fixed M3A scope",
    ):
        export_module._configuration_contract(
            evaluation,
            source_dataset_id=_SOURCE_DATASET_ID,
            source_dataset_digest=compute_episode_content_digest(build.source_episodes),
            corruption_dataset_digest=compute_corruption_dataset_content_digest(
                corruption
            ),
        )


def test_independent_binding_rejects_swapped_per_proposal_evidence() -> None:
    build, corruption, evaluation = _inputs()
    result = _export(build, corruption, evaluation)
    source = next(
        sample
        for sample in result.dataset.samples
        if sample.candidate_type is CandidateType.SOURCE
    )
    first, second = (
        sample
        for sample in result.dataset.samples
        if sample.candidate_type is CandidateType.CORRUPTED
    )
    swapped_first = replace(
        first,
        strong_simulator_evidence_id=second.strong_simulator_evidence_id,
    )
    swapped_second = replace(
        second,
        strong_simulator_evidence_id=first.strong_simulator_evidence_id,
    )
    group = replace(
        result.dataset.candidate_groups[0],
        corrupted_sample_ids=tuple(
            sorted((swapped_first.sample_id, swapped_second.sample_id))
        ),
    )
    tampered = replace(
        result.dataset,
        samples=tuple(
            sorted(
                (source, swapped_first, swapped_second),
                key=lambda sample: sample.sample_id,
            )
        ),
        candidate_groups=(group,),
    )

    with pytest.raises(ActionVerifierExportError, match="foreign keys disagree"):
        validate_action_verifier_evidence_bindings(
            tampered,
            anchor_manifest=build.manifest,
            corruption_dataset=corruption,
            evaluation_dataset=evaluation,
        )


def test_independent_binding_rejects_unbound_trajectory_state_inventory() -> None:
    build, corruption, evaluation = _inputs()
    result = _export(build, corruption, evaluation)
    assignment = result.dataset.split_assignments[0]
    changed_assignment = replace(
        assignment,
        state_digests=tuple(sorted((*assignment.state_digests, f"sha256:{'f' * 64}"))),
    )
    tampered = replace(result.dataset, split_assignments=(changed_assignment,))

    with pytest.raises(
        ActionVerifierExportError,
        match="trajectory state inventory differs from source archive",
    ):
        validate_action_verifier_evidence_bindings(
            tampered,
            anchor_manifest=build.manifest,
            corruption_dataset=corruption,
            evaluation_dataset=evaluation,
        )


def test_export_fails_closed_when_strong_prefix_metrics_are_missing() -> None:
    build, corruption, evaluation = _inputs()
    first = evaluation.evidence[0]
    metrics = dict(first.metrics)
    del metrics["pickcube_progress_after_candidate"]
    changed = replace(first, metrics=metrics)
    tampered = replace(
        evaluation,
        evidence=(changed, *evaluation.evidence[1:]),
    )

    with pytest.raises(ActionVerifierExportError, match="expected a finite number"):
        _export(build, corruption, tampered)


def test_serialized_reload_reports_classes_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    build, corruption, evaluation = _inputs()
    result = _export(build, corruption, evaluation)
    dataset_dir = tmp_path / "dataset"
    save_action_verifier_dataset(result.dataset, dataset_dir)
    report = validate_serialized_action_verifier_export(
        dataset_dir,
        available_evidence_ids=result.available_evidence_ids,
        evidence_dataset_digest=result.evidence_dataset_digest,
        expected_split_counts={
            DatasetSplit.TRAIN: 1,
            DatasetSplit.VALIDATION: 0,
            DatasetSplit.TEST: 0,
        },
    )
    assert report.validation_scope == "serialized_reload"
    assert report.corrupted_success_count == 1
    assert report.corrupted_failure_count == 1

    reloaded = load_action_verifier_dataset(dataset_dir)
    validate_action_verifier_evidence_bindings(
        reloaded,
        anchor_manifest=build.manifest,
        corruption_dataset=corruption,
        evaluation_dataset=evaluation,
    )

    reports_dir = tmp_path / "reports"
    summary_path, validation_path = save_action_verifier_export_reports(
        result,
        reports_dir,
        validation_report=report,
    )
    assert summary_path.name == ACTION_VERIFIER_SUMMARY_NAME
    assert validation_path.name == ACTION_VERIFIER_VALIDATION_NAME

    array_path = dataset_dir / "arrays" / "candidate_action_chunks.npy"
    content = bytearray(array_path.read_bytes())
    content[-1] ^= 1
    array_path.write_bytes(content)
    with pytest.raises(ActionVerifierExportError, match="failed reload"):
        validate_serialized_action_verifier_export(
            dataset_dir,
            available_evidence_ids=result.available_evidence_ids,
            evidence_dataset_digest=result.evidence_dataset_digest,
        )
