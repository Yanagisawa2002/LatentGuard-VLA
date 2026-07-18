from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import latentguard.action_verifier as action_verifier
import latentguard.corruptions.serialization as corruption_serialization
import latentguard.evaluation.serialization as evaluation_serialization
import latentguard.integrations.maniskill_pickcube.state_indexed_build as state_build
import latentguard.selection.blind_protocol as blind_protocol
import latentguard.selection.evaluation as selection_evaluation
import latentguard.selection.manifest as selection_manifest
import latentguard.selection.outcomes as selection_outcomes
import latentguard.selection.serialization as selection_serialization
import latentguard.training.dataset as training_dataset
from latentguard.evaluation.models import (
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.integrations.maniskill_pickcube import (
    state_indexed_archive as state_archive,
)
from latentguard.models import LabelSource, LabelStrength
from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST
from latentguard.vision_data.source_binding import (
    ACCEPTED_M3A_ACCEPTANCE_REPORT_DIGEST,
    ACCEPTED_M3A_ANCHOR_MANIFEST_DIGEST,
    ACCEPTED_M3A_EVIDENCE_DIGEST,
    ACCEPTED_M3A_SIMULATOR_COMPATIBILITY_IDENTITY,
    ACCEPTED_M3A_SOURCE_ARCHIVE_DIGEST,
    ACCEPTED_M3A_SPLIT_DIGEST,
    ACCEPTED_M3C_BLIND_MANIFEST_DIGEST,
    ACCEPTED_M3C_BLIND_MANIFEST_ENVELOPE_DIGEST,
    ACCEPTED_M3C_BOUND_RESULT_DIGEST,
    ACCEPTED_M3C_CANDIDATE_POOL_DIGEST,
    ACCEPTED_M3C_EXECUTION_GIT_SHA,
    ACCEPTED_M3C_FULL_OUTCOME_DIGEST,
    ACCEPTED_M3C_REPLAY_EVIDENCE_DIGEST,
    ACCEPTED_M3C_SIMULATOR_COMPATIBILITY_IDENTITY,
    ACCEPTED_M3C_SOURCE_SET_DIGEST,
    VisualSourceBindingError,
    load_m3a_development_source_binding,
    load_m3c_external_source_binding,
    validate_visual_development_source_binding,
    visual_candidate_array_reference,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _patch_m3a_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dataset_digest: str = ACCEPTED_M3A_DATASET_DIGEST,
    accepted_dataset_digest: str = ACCEPTED_M3A_DATASET_DIGEST,
    source_archive_digest: str = ACCEPTED_M3A_SOURCE_ARCHIVE_DIGEST,
    anchor_manifest_digest: str = ACCEPTED_M3A_ANCHOR_MANIFEST_DIGEST,
    evidence_digest: str = ACCEPTED_M3A_EVIDENCE_DIGEST,
    split_digest: str = ACCEPTED_M3A_SPLIT_DIGEST,
    acceptance_report_digest: str = ACCEPTED_M3A_ACCEPTANCE_REPORT_DIGEST,
    compatibility_identity: str = (ACCEPTED_M3A_SIMULATOR_COMPATIBILITY_IDENTITY),
    manifest_archive_digest: str | None = None,
) -> None:
    sample = SimpleNamespace(
        sample_id="candidate-0",
        evidence_dataset_digest=evidence_digest,
        compatibility_identity=compatibility_identity,
    )
    dataset = SimpleNamespace(
        content_digest=dataset_digest,
        samples=(sample,),
        candidate_groups=(SimpleNamespace(anchor_id="anchor-0"),),
        split_assignments=(SimpleNamespace(source_trajectory_id="trajectory-0"),),
    )
    manifest = SimpleNamespace(
        content_digest=anchor_manifest_digest,
        source_archive_content_digest=(
            source_archive_digest
            if manifest_archive_digest is None
            else manifest_archive_digest
        ),
        records=(
            SimpleNamespace(
                anchor=SimpleNamespace(
                    anchor_id="anchor-0",
                    selection_reason="uniform_anchor",
                )
            ),
        ),
    )
    accepted = SimpleNamespace(
        dataset_digest=accepted_dataset_digest,
        split_digest=split_digest,
        acceptance_report_digest=acceptance_report_digest,
    )
    archive = SimpleNamespace(content_digest=source_archive_digest)
    monkeypatch.setattr(
        action_verifier,
        "load_action_verifier_dataset",
        lambda _: dataset,
    )
    monkeypatch.setattr(state_build, "load_anchor_manifest", lambda _: manifest)
    monkeypatch.setattr(state_archive, "load_state_indexed_archive", lambda _: archive)
    monkeypatch.setattr(
        training_dataset,
        "load_accepted_action_verifier_dataset",
        lambda *args, **kwargs: accepted,
    )


def _strong_evidence(proposal_id: str) -> EvaluationEvidence:
    configuration_digest = compute_configuration_digest(
        {"schema_version": "1.0", "semantic": "m4a-source-binding-test"}
    )
    return EvaluationEvidence(
        evidence_id=compute_evidence_identifier(
            proposal_id=proposal_id,
            evaluator_id="m4a_source_binding_test",
            evaluator_version="1.0.0",
            evaluator_configuration_digest=configuration_digest,
            evaluation_seed=271828,
            attempt_ordinal=0,
        ),
        proposal_id=proposal_id,
        source_dataset_id=ACCEPTED_M3C_SOURCE_SET_DIGEST,
        source_episode_id="trajectory-0",
        source_candidate_id="candidate-source-0",
        split_group_id="split-group-0",
        evaluator_id="m4a_source_binding_test",
        evaluator_version="1.0.0",
        evaluator_configuration_digest=configuration_digest,
        evaluation_seed=271828,
        attempt_ordinal=0,
        status=EvaluationStatus.CONCLUSIVE,
        success=False,
        progress_before=0.2,
        progress_after=0.1,
        progress_delta=-0.1,
        unsafe=True,
        failure_events=(),
        termination_reason="paired_replay_complete",
        replayed_control_steps=32,
        metrics={"complete_candidate_execution": True},
        artifact_references=(),
        label_source=LabelSource.SIMULATOR,
        label_strength=LabelStrength.STRONG,
        simulator_replay_verified=True,
        notes="strong source-binding fixture",
    )


def _patch_m3c_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    git_sha: str = ACCEPTED_M3C_EXECUTION_GIT_SHA,
    source_set_digest: str = ACCEPTED_M3C_SOURCE_SET_DIGEST,
    candidate_pool_digest: str = ACCEPTED_M3C_CANDIDATE_POOL_DIGEST,
    blind_manifest_digest: str = ACCEPTED_M3C_BLIND_MANIFEST_DIGEST,
    blind_envelope_digest: str = ACCEPTED_M3C_BLIND_MANIFEST_ENVELOPE_DIGEST,
    bound_result_digest: str = ACCEPTED_M3C_BOUND_RESULT_DIGEST,
    joined_digest: str = ACCEPTED_M3C_REPLAY_EVIDENCE_DIGEST,
    full_outcome_digest: str = ACCEPTED_M3C_FULL_OUTCOME_DIGEST,
    simulator_compatibility_identity: str = (
        ACCEPTED_M3C_SIMULATOR_COMPATIBILITY_IDENTITY
    ),
    result_replay_digest: str | None = None,
    computed_outcome_digest: str | None = None,
) -> None:
    proposal_id = "proposal-0"
    evidence = _strong_evidence(proposal_id)
    pool = SimpleNamespace(
        source_set_digest=source_set_digest,
        content_digest=candidate_pool_digest,
        proposal_ids=(proposal_id,),
        groups=(
            SimpleNamespace(
                source_trajectory_id="trajectory-0",
                anchor_id="anchor-0",
            ),
        ),
    )
    envelope = SimpleNamespace(
        semantic_digest=blind_manifest_digest,
        envelope_digest=blind_envelope_digest,
    )
    result = SimpleNamespace(
        replay_evidence_digest=(
            joined_digest if result_replay_digest is None else result_replay_digest
        ),
        full_pool_outcome_digest=full_outcome_digest,
        git_sha=git_sha,
        content_digest=bound_result_digest,
        simulator_compatibility_identity=simulator_compatibility_identity,
    )
    joined = SimpleNamespace(
        content_digest=joined_digest,
        evidence_by_proposal={proposal_id: evidence},
    )
    outcome = SimpleNamespace(
        proposal_id=proposal_id,
        evidence_id=evidence.evidence_id,
        success=evidence.success,
        unsafe=evidence.unsafe,
        simulator_replay_verified=evidence.simulator_replay_verified,
    )
    monkeypatch.setattr(selection_serialization, "load_candidate_pool", lambda _: pool)
    monkeypatch.setattr(
        selection_manifest,
        "load_blind_selection_manifest",
        lambda _: envelope,
    )
    monkeypatch.setattr(
        blind_protocol,
        "load_bound_selection_result",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        corruption_serialization,
        "load_corruption_dataset",
        lambda _: object(),
    )
    monkeypatch.setattr(
        evaluation_serialization,
        "compute_corruption_dataset_digest",
        lambda _: _digest("corruptions"),
    )
    monkeypatch.setattr(
        evaluation_serialization,
        "load_evaluation_dataset",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        selection_evaluation,
        "join_complementary_replay_phases",
        lambda **kwargs: joined,
    )
    monkeypatch.setattr(
        selection_outcomes,
        "candidate_outcomes_from_replay_join",
        lambda *args, **kwargs: (outcome,),
    )
    monkeypatch.setattr(
        blind_protocol,
        "compute_full_pool_outcome_digest",
        lambda _: (
            full_outcome_digest
            if computed_outcome_digest is None
            else computed_outcome_digest
        ),
    )


def test_m3a_loader_content_binds_the_accepted_source_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_m3a_sources(monkeypatch)

    binding = load_m3a_development_source_binding(
        tmp_path / "dataset",
        tmp_path / "archive",
        tmp_path / "anchors",
        acceptance_report=tmp_path / "acceptance.json",
    )
    assert binding.dataset_digest == ACCEPTED_M3A_DATASET_DIGEST
    assert binding.source_archive_digest == ACCEPTED_M3A_SOURCE_ARCHIVE_DIGEST
    assert binding.anchor_manifest_digest == ACCEPTED_M3A_ANCHOR_MANIFEST_DIGEST
    assert binding.evidence_digest == ACCEPTED_M3A_EVIDENCE_DIGEST
    assert binding.split_digest == ACCEPTED_M3A_SPLIT_DIGEST
    assert binding.acceptance_report_digest == ACCEPTED_M3A_ACCEPTANCE_REPORT_DIGEST
    assert binding.trajectory_ids == ("trajectory-0",)
    assert binding.anchor_ids == ("anchor-0",)
    assert binding.candidate_sample_ids == ("candidate-0",)
    assert (
        binding.compatibility_identity == ACCEPTED_M3A_SIMULATOR_COMPATIBILITY_IDENTITY
    )


def test_m3a_visual_validation_uses_exported_simulator_state_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentguard.action_verifier.models as action_models
    import latentguard.vision_data.models as visual_models
    from latentguard.action_verifier.models import DatasetSplit
    from latentguard.vision_data.models import SourceCollection, VisualDatasetSplit

    archived_record_digest = _digest("archived-record")
    simulator_state_digest = _digest("simulator-state-bytes")
    verifier_digest = _digest("verifier")
    compatibility = _digest("compatibility")
    dataset_digest = _digest("dataset")
    split_digest = _digest("split")
    evidence_digest = _digest("evidence")
    manifest_digest = _digest("manifest")
    anchor = SimpleNamespace(
        anchor_id="anchor-0",
        source_trajectory_id="trajectory-0",
        source_seed=17,
        split_group_id="split-0",
    )
    group = SimpleNamespace(
        anchor_id=anchor.anchor_id,
        source_trajectory_id=anchor.source_trajectory_id,
        source_seed=anchor.source_seed,
        split_group_id=anchor.split_group_id,
        dataset_split=DatasetSplit.TRAIN,
        state_content_digest=simulator_state_digest,
        state_vector_semantic="verifier-v1",
        source_sample_id="candidate-0",
        corrupted_sample_ids=(),
        group_id="group-0",
    )
    record = SimpleNamespace(
        anchor=anchor,
        source_state_content_digest=archived_record_digest,
        source_state_digest=simulator_state_digest,
        verifier_state_content_digest=verifier_digest,
    )
    packet = SimpleNamespace(
        anchor_id=anchor.anchor_id,
        source_collection=SourceCollection.M3A_DEVELOPMENT,
        source_trajectory_id=anchor.source_trajectory_id,
        split=VisualDatasetSplit.TRAIN,
        split_group_id=anchor.split_group_id,
        state_reference_id=archived_record_digest,
        expected_state_digest=simulator_state_digest,
        verifier_state_semantic=group.state_vector_semantic,
        verifier_state_digest=verifier_digest,
        pickcube_compatibility_identity=compatibility,
    )
    source_dataset = SimpleNamespace(
        content_digest=dataset_digest,
        candidate_groups=(group,),
        split_assignments=(
            SimpleNamespace(
                source_trajectory_id=anchor.source_trajectory_id,
                dataset_split=DatasetSplit.TRAIN,
            ),
        ),
        samples=(),
    )
    anchor_manifest = SimpleNamespace(
        content_digest=manifest_digest,
        records=(record,),
    )
    visual_dataset = SimpleNamespace(
        source_dataset_digest=dataset_digest,
        split_digest=split_digest,
        evidence_digest=evidence_digest,
        source_compatibility_identity=compatibility,
        packets=(packet,),
        candidate_bindings=(),
        samples=(),
    )
    binding = SimpleNamespace(
        dataset_digest=dataset_digest,
        anchor_manifest_digest=manifest_digest,
        split_digest=split_digest,
        evidence_digest=evidence_digest,
        compatibility_identity=compatibility,
    )
    monkeypatch.setattr(action_models, "ActionVerifierDatasetV1", SimpleNamespace)
    monkeypatch.setattr(state_build, "PickCubeAnchorManifestV1", SimpleNamespace)
    monkeypatch.setattr(
        visual_models, "VisualVerifierDevelopmentDatasetV1", SimpleNamespace
    )

    with pytest.raises(
        VisualSourceBindingError, match="candidate inventory or order differs"
    ):
        validate_visual_development_source_binding(
            visual_dataset,
            source_dataset,
            anchor_manifest,
            binding,  # type: ignore[arg-type]
            allow_partial=True,
        )


def test_visual_sample_projection_compares_failure_event_content() -> None:
    import latentguard.vision_data.source_binding as source_binding
    from latentguard.models import FailureEvent
    from latentguard.vision_data.models import SourceCollection

    expected_events = (
        FailureEvent(
            failure_type="task_not_completed",
            description="Official terminal success was false.",
        ),
        FailureEvent(
            failure_type="cube_not_at_goal",
            description="Official object-placed status was false.",
        ),
    )
    packet_ids = ("packet-0", "packet-1", "packet-2")

    def sample(packet_id: str, *, drift: bool = False) -> SimpleNamespace:
        observed_events = tuple(
            FailureEvent(
                failure_type=event.failure_type,
                timestamp_s=event.timestamp_s,
                probability=event.probability,
                description=("drifted" if drift and index == 0 else event.description),
                schema_version=event.schema_version,
            )
            for index, event in enumerate(expected_events)
        )
        return SimpleNamespace(
            packet_id=packet_id,
            candidate_sample_id="candidate-0",
            task_id="maniskill/PickCube-v1",
            canonical_task_text="Pick up the cube and place it at the goal.",
            candidate_action_chunk_reference=(
                visual_candidate_array_reference("candidate-0", "action-chunk")
            ),
            action_mask_reference=visual_candidate_array_reference(
                "candidate-0", "action-mask"
            ),
            final_success=False,
            final_unsafe=False,
            failure_events=observed_events,
            evidence_id="evidence-0",
            source_dataset_digest=_digest("source"),
            candidate_dataset_digest=_digest("candidate"),
            source_collection=SourceCollection.M3A_DEVELOPMENT,
        )

    observed = tuple(sample(packet_id) for packet_id in packet_ids)
    source_binding._require_visual_sample_projection(
        observed,
        candidate_id="candidate-0",
        packet_ids=packet_ids,
        task_id="maniskill/PickCube-v1",
        task_text="Pick up the cube and place it at the goal.",
        final_success=False,
        final_unsafe=False,
        failure_events=expected_events,
        evidence_id="evidence-0",
        source_dataset_digest=_digest("source"),
        candidate_dataset_digest=_digest("candidate"),
        source_collection=SourceCollection.M3A_DEVELOPMENT,
    )

    drifted = (sample(packet_ids[0], drift=True), *observed[1:])
    with pytest.raises(VisualSourceBindingError, match="failure_events"):
        source_binding._require_visual_sample_projection(
            drifted,
            candidate_id="candidate-0",
            packet_ids=packet_ids,
            task_id="maniskill/PickCube-v1",
            task_text="Pick up the cube and place it at the goal.",
            final_success=False,
            final_unsafe=False,
            failure_events=expected_events,
            evidence_id="evidence-0",
            source_dataset_digest=_digest("source"),
            candidate_dataset_digest=_digest("candidate"),
            source_collection=SourceCollection.M3A_DEVELOPMENT,
        )


def test_m3a_loader_rejects_accepted_dataset_identity_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_m3a_sources(
        monkeypatch,
        accepted_dataset_digest=_digest("not-the-accepted-m3a-dataset"),
    )

    with pytest.raises(VisualSourceBindingError, match="dataset_digest"):
        load_m3a_development_source_binding(
            tmp_path / "dataset",
            tmp_path / "archive",
            tmp_path / "anchors",
            acceptance_report=tmp_path / "acceptance.json",
        )


@pytest.mark.parametrize(
    ("updates", "identity_name"),
    (
        (
            {"source_archive_digest": _digest("drifted-source-archive")},
            "source_archive_digest",
        ),
        (
            {"anchor_manifest_digest": _digest("drifted-anchor-manifest")},
            "anchor_manifest_digest",
        ),
        ({"evidence_digest": _digest("drifted-evidence")}, "evidence_digest"),
        ({"split_digest": _digest("drifted-split")}, "split_digest"),
        (
            {"acceptance_report_digest": _digest("drifted-acceptance-report")},
            "acceptance_report_digest",
        ),
        (
            {"compatibility_identity": _digest("drifted-compatibility")},
            "compatibility_identity",
        ),
    ),
)
def test_m3a_loader_rejects_each_accepted_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, str],
    identity_name: str,
) -> None:
    _patch_m3a_sources(monkeypatch, **updates)

    with pytest.raises(VisualSourceBindingError, match=identity_name):
        load_m3a_development_source_binding(
            tmp_path / "dataset",
            tmp_path / "archive",
            tmp_path / "anchors",
            acceptance_report=tmp_path / "acceptance.json",
        )


def test_m3a_loader_rejects_manifest_archive_binding_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_m3a_sources(
        monkeypatch,
        manifest_archive_digest=_digest("different-source-archive"),
    )

    with pytest.raises(VisualSourceBindingError, match="manifest archive digest"):
        load_m3a_development_source_binding(
            tmp_path / "dataset",
            tmp_path / "archive",
            tmp_path / "anchors",
            acceptance_report=tmp_path / "acceptance.json",
        )


def test_m3c_loader_content_binds_evidence_outcomes_and_execution_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_m3c_sources(monkeypatch)

    binding = load_m3c_external_source_binding(
        tmp_path / "pool",
        tmp_path / "blind.json",
        tmp_path / "result.json",
        tmp_path / "corruptions",
        tmp_path / "selected",
        tmp_path / "remainder",
    )
    assert binding.execution_git_sha == ACCEPTED_M3C_EXECUTION_GIT_SHA
    assert binding.trajectory_ids == ("trajectory-0",)
    assert binding.anchor_ids == ("anchor-0",)
    assert binding.candidate_ids == ("proposal-0",)
    assert binding.training_allowed is False
    assert binding.evidence_projection_digest.startswith("sha256:")
    assert binding.evidence_by_candidate["proposal-0"].success is False


@pytest.mark.parametrize(
    ("updates", "identity_name"),
    (
        ({"source_set_digest": _digest("direct-source-set")}, "source_set_digest"),
        (
            {"candidate_pool_digest": _digest("direct-candidate-pool")},
            "candidate_pool_digest",
        ),
        (
            {"blind_manifest_digest": _digest("direct-blind-semantic")},
            "blind_manifest_digest",
        ),
        (
            {"blind_manifest_envelope_digest": _digest("direct-blind-envelope")},
            "blind_manifest_envelope_digest",
        ),
        (
            {"bound_result_digest": _digest("direct-bound-result")},
            "bound_result_digest",
        ),
        (
            {"replay_evidence_digest": _digest("direct-replay")},
            "replay_evidence_digest",
        ),
        (
            {"full_outcome_digest": _digest("direct-outcome")},
            "full_outcome_digest",
        ),
        (
            {
                "simulator_compatibility_identity": _digest(
                    "direct-simulator-compatibility"
                )
            },
            "simulator_compatibility_identity",
        ),
        ({"execution_git_sha": "0" * 40}, "execution Git SHA"),
    ),
)
def test_m3c_binding_rejects_direct_construction_identity_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, str],
    identity_name: str,
) -> None:
    _patch_m3c_sources(monkeypatch)
    binding = load_m3c_external_source_binding(
        tmp_path / "pool",
        tmp_path / "blind.json",
        tmp_path / "result.json",
        tmp_path / "corruptions",
        tmp_path / "selected",
        tmp_path / "remainder",
    )

    with pytest.raises(VisualSourceBindingError, match=identity_name):
        replace(binding, **updates)


def test_m3c_loader_rejects_caller_override_of_accepted_execution_sha(
    tmp_path: Path,
) -> None:
    with pytest.raises(VisualSourceBindingError, match="cannot override"):
        load_m3c_external_source_binding(
            tmp_path / "pool",
            tmp_path / "blind.json",
            tmp_path / "result.json",
            tmp_path / "corruptions",
            tmp_path / "selected",
            tmp_path / "remainder",
            expected_execution_git_sha="0" * 40,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {"git_sha": "0" * 40},
            "execution Git SHA",
        ),
        (
            {"result_replay_digest": _digest("different-replay")},
            "replay evidence",
        ),
        (
            {"computed_outcome_digest": _digest("different-outcome")},
            "full outcome digest",
        ),
    ),
)
def test_m3c_loader_rejects_chain_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, str],
    message: str,
) -> None:
    _patch_m3c_sources(monkeypatch, **updates)

    with pytest.raises(VisualSourceBindingError, match=message):
        load_m3c_external_source_binding(
            tmp_path / "pool",
            tmp_path / "blind.json",
            tmp_path / "result.json",
            tmp_path / "corruptions",
            tmp_path / "selected",
            tmp_path / "remainder",
        )


@pytest.mark.parametrize(
    ("updates", "identity_name"),
    (
        (
            {"source_set_digest": _digest("drifted-source-set")},
            "source_set_digest",
        ),
        (
            {"candidate_pool_digest": _digest("drifted-candidate-pool")},
            "candidate_pool_digest",
        ),
        (
            {"blind_manifest_digest": _digest("drifted-blind-semantic")},
            "blind_manifest_digest",
        ),
        (
            {"blind_envelope_digest": _digest("drifted-blind-envelope")},
            "blind_manifest_envelope_digest",
        ),
        (
            {"bound_result_digest": _digest("drifted-bound-result")},
            "bound_result_digest",
        ),
        (
            {"joined_digest": _digest("drifted-replay")},
            "replay_evidence_digest",
        ),
        (
            {"full_outcome_digest": _digest("drifted-outcome")},
            "full_outcome_digest",
        ),
        (
            {
                "simulator_compatibility_identity": _digest(
                    "drifted-simulator-compatibility"
                )
            },
            "simulator_compatibility_identity",
        ),
    ),
)
def test_m3c_loader_rejects_each_accepted_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, str],
    identity_name: str,
) -> None:
    _patch_m3c_sources(monkeypatch, **updates)

    with pytest.raises(VisualSourceBindingError, match=identity_name):
        load_m3c_external_source_binding(
            tmp_path / "pool",
            tmp_path / "blind.json",
            tmp_path / "result.json",
            tmp_path / "corruptions",
            tmp_path / "selected",
            tmp_path / "remainder",
        )


def test_candidate_array_references_are_path_safe_and_content_derived() -> None:
    reference = visual_candidate_array_reference(
        "candidate/with/runtime/path", "action-chunk"
    )
    assert reference.startswith("source-candidates/")
    assert "candidate/with/runtime/path" not in reference
    assert reference.endswith("/action-chunk")
    assert visual_candidate_array_reference(
        "candidate/with/runtime/path", "action-mask"
    ).endswith("/action-mask")
    with pytest.raises(VisualSourceBindingError, match="unsupported"):
        visual_candidate_array_reference("candidate-0", "pixels")
