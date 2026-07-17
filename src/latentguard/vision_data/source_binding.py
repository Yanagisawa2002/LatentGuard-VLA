"""Strict bindings from accepted M3A/M3C artifacts into M4A source identities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

from latentguard.evaluation.models import EvaluationEvidence, EvaluationStatus
from latentguard.evaluation.validation import validate_evaluation_evidence
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST

ACCEPTED_M3A_SOURCE_ARCHIVE_DIGEST = (
    "sha256:cc48c42f1c8395d348f968be72102b857eb2994702c6ca85bf8e6239f2fb36d5"
)
ACCEPTED_M3A_ANCHOR_MANIFEST_DIGEST = (
    "sha256:c15d4fdaf36994f5cd58304484110407e8f281057983056aa5a7025fad363e9e"
)
ACCEPTED_M3A_EVIDENCE_DIGEST = (
    "sha256:d077dcf31012d32f4b4833a0622b58d55a43d5a96a1f81b4eea045da0f747995"
)
ACCEPTED_M3A_SPLIT_DIGEST = (
    "sha256:173ca40a072a1977cc65dbcccedb4684ae573e97f3984c176e3f91bba09f940a"
)
ACCEPTED_M3A_ACCEPTANCE_REPORT_DIGEST = (
    "sha256:f3d747daffc8dfe3528350de4e193bc2e6d2a2954fe1d4476c3a2797c317c06f"
)
ACCEPTED_M3A_SIMULATOR_COMPATIBILITY_IDENTITY = (
    "sha256:05a560fd89e0989b6d94017c6456efc535de4a0bf87ededaf468f5e2d9ef71f7"
)

ACCEPTED_M3C_EXECUTION_GIT_SHA = "a632a702c709edb1fc21e702c83e30964652ff79"
ACCEPTED_M3C_SOURCE_SET_DIGEST = (
    "sha256:db1f3ece9576850c09a9f95b8c4aee483d11d71ebbbe0772b695db2d60321123"
)
ACCEPTED_M3C_CANDIDATE_POOL_DIGEST = (
    "sha256:e2982acd30297fb81f000ea53081afdb4e2ca9ad5dc2f16c6aa71d199c8ef1b7"
)
ACCEPTED_M3C_BLIND_MANIFEST_DIGEST = (
    "sha256:c5aa3798d0d1797a1714f8de6ad76954f6aa1b31835a3aae8d0abbd7e52dcbcc"
)
ACCEPTED_M3C_BLIND_MANIFEST_ENVELOPE_DIGEST = (
    "sha256:55e077a9818fc13b4a966081c7cbd57d4fb86bdcea447053e8349c62a9f6a37f"
)
ACCEPTED_M3C_BOUND_RESULT_DIGEST = (
    "sha256:e3cee839f5df10dd47578f214d02e0aee089a65bed37dad290f6a554b4aceaae"
)
ACCEPTED_M3C_REPLAY_EVIDENCE_DIGEST = (
    "sha256:c44bb6631243c78b0c8d7db8f573454613ee04f919293c0b9b37d561502cbdd4"
)
ACCEPTED_M3C_FULL_OUTCOME_DIGEST = (
    "sha256:06ac372236586ed7e4dcf3af8fc8d09e05e063f24cdaa687ee6bcdcd8e51d05c"
)
ACCEPTED_M3C_SIMULATOR_COMPATIBILITY_IDENTITY = (
    "sha256:05a560fd89e0989b6d94017c6456efc535de4a0bf87ededaf468f5e2d9ef71f7"
)


class VisualSourceBindingError(ValueError):
    """Raised when an accepted source artifact is missing or identity-drifted."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualSourceBindingError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase sha256 digest")
    return value


@dataclass(frozen=True, slots=True)
class M3ADevelopmentSourceBindingV1:
    """Accepted development-dataset identities needed by visual generation."""

    dataset_digest: str
    source_archive_digest: str
    split_digest: str
    evidence_digest: str
    acceptance_report_digest: str
    anchor_manifest_digest: str
    compatibility_identity: str
    trajectory_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    candidate_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "dataset_digest",
            "source_archive_digest",
            "split_digest",
            "evidence_digest",
            "acceptance_report_digest",
            "anchor_manifest_digest",
            "compatibility_identity",
        ):
            _digest(getattr(self, name), f"M3A binding {name}")
        accepted_identities = {
            "dataset_digest": ACCEPTED_M3A_DATASET_DIGEST,
            "source_archive_digest": ACCEPTED_M3A_SOURCE_ARCHIVE_DIGEST,
            "anchor_manifest_digest": ACCEPTED_M3A_ANCHOR_MANIFEST_DIGEST,
            "evidence_digest": ACCEPTED_M3A_EVIDENCE_DIGEST,
            "split_digest": ACCEPTED_M3A_SPLIT_DIGEST,
            "acceptance_report_digest": ACCEPTED_M3A_ACCEPTANCE_REPORT_DIGEST,
            "compatibility_identity": (ACCEPTED_M3A_SIMULATOR_COMPATIBILITY_IDENTITY),
        }
        drifted = tuple(
            name
            for name, expected in accepted_identities.items()
            if getattr(self, name) != expected
        )
        if drifted:
            _fail(
                "accepted M3A artifact identities",
                "differ in " + ", ".join(drifted),
            )
        _unique_nonempty(self.trajectory_ids, "M3A trajectory IDs")
        _unique_nonempty(self.anchor_ids, "M3A anchor IDs")
        _unique_nonempty(self.candidate_sample_ids, "M3A candidate sample IDs")


@dataclass(frozen=True, slots=True)
class M3CExternalSourceBindingV1:
    """Complete external-evaluation identity and strong outcome projection."""

    source_set_digest: str
    candidate_pool_digest: str
    blind_manifest_digest: str
    blind_manifest_envelope_digest: str
    bound_result_digest: str
    replay_evidence_digest: str
    evidence_projection_digest: str
    full_outcome_digest: str
    simulator_compatibility_identity: str
    execution_git_sha: str
    trajectory_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    outcomes_by_candidate: Mapping[str, object]
    evidence_by_candidate: Mapping[str, EvaluationEvidence]
    training_allowed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_set_digest",
            "candidate_pool_digest",
            "blind_manifest_digest",
            "blind_manifest_envelope_digest",
            "bound_result_digest",
            "replay_evidence_digest",
            "evidence_projection_digest",
            "full_outcome_digest",
            "simulator_compatibility_identity",
        ):
            _digest(getattr(self, name), f"M3C binding {name}")
        if (
            not isinstance(self.execution_git_sha, str)
            or len(self.execution_git_sha) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_git_sha
            )
        ):
            _fail("M3C execution Git SHA", "expected full lowercase 40-hex SHA")
        accepted_identities = {
            "source_set_digest": ACCEPTED_M3C_SOURCE_SET_DIGEST,
            "candidate_pool_digest": ACCEPTED_M3C_CANDIDATE_POOL_DIGEST,
            "blind_manifest_digest": ACCEPTED_M3C_BLIND_MANIFEST_DIGEST,
            "blind_manifest_envelope_digest": (
                ACCEPTED_M3C_BLIND_MANIFEST_ENVELOPE_DIGEST
            ),
            "bound_result_digest": ACCEPTED_M3C_BOUND_RESULT_DIGEST,
            "replay_evidence_digest": ACCEPTED_M3C_REPLAY_EVIDENCE_DIGEST,
            "full_outcome_digest": ACCEPTED_M3C_FULL_OUTCOME_DIGEST,
            "simulator_compatibility_identity": (
                ACCEPTED_M3C_SIMULATOR_COMPATIBILITY_IDENTITY
            ),
        }
        drifted = tuple(
            name
            for name, expected in accepted_identities.items()
            if getattr(self, name) != expected
        )
        if drifted:
            _fail(
                "accepted M3C artifact identities",
                "differ in " + ", ".join(drifted),
            )
        if self.execution_git_sha != ACCEPTED_M3C_EXECUTION_GIT_SHA:
            _fail("M3C execution Git SHA", "differs from accepted execution")
        _unique_nonempty(self.trajectory_ids, "M3C trajectory IDs")
        _unique_nonempty(self.anchor_ids, "M3C anchor IDs")
        candidates = _unique_nonempty(self.candidate_ids, "M3C candidate IDs")
        outcomes = dict(self.outcomes_by_candidate)
        if set(outcomes) != set(candidates):
            _fail("M3C outcomes", "must cover every candidate exactly once")
        evidence = dict(self.evidence_by_candidate)
        if tuple(self.evidence_by_candidate) != candidates or set(evidence) != set(
            candidates
        ):
            _fail("M3C evidence", "must preserve exact candidate coverage and order")
        for candidate in candidates:
            item = evidence[candidate]
            if not isinstance(item, EvaluationEvidence):
                _fail("M3C evidence", "contains an unsupported evidence object")
            validate_evaluation_evidence(item)
            outcome = outcomes[candidate]
            if (
                item.proposal_id != candidate
                or item.status is not EvaluationStatus.CONCLUSIVE
                or item.evidence_id != getattr(outcome, "evidence_id", None)
                or item.success != getattr(outcome, "success", None)
                or item.unsafe != getattr(outcome, "unsafe", None)
                or item.simulator_replay_verified
                != getattr(outcome, "simulator_replay_verified", None)
            ):
                _fail(candidate, "evidence differs from the accepted outcome")
        projected_digest = _evidence_projection_digest(candidates, evidence)
        if projected_digest != self.evidence_projection_digest:
            _fail("M3C evidence", "projected evidence digest differs")
        if self.training_allowed is not False:
            _fail("M3C external binding", "training_allowed must be false")
        object.__setattr__(
            self,
            "outcomes_by_candidate",
            MappingProxyType(
                {candidate: outcomes[candidate] for candidate in candidates}
            ),
        )
        object.__setattr__(
            self,
            "evidence_by_candidate",
            MappingProxyType(
                {candidate: evidence[candidate] for candidate in candidates}
            ),
        )


def _unique_nonempty(values: tuple[str, ...], context: str) -> tuple[str, ...]:
    materialized = tuple(values)
    if (
        not materialized
        or len(materialized) != len(set(materialized))
        or any(not isinstance(item, str) or not item for item in materialized)
    ):
        _fail(context, "expected non-empty unique text inventory")
    return materialized


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return first-seen unique identifiers without changing source order."""

    return tuple(dict.fromkeys(values))


def visual_candidate_array_reference(candidate_id: str, component: str) -> str:
    """Return the path-safe logical reference to an already bound source array."""

    if not isinstance(candidate_id, str) or not candidate_id:
        _fail("visual candidate reference", "candidate ID must be non-empty text")
    if component not in {"action-chunk", "action-mask"}:
        _fail("visual candidate reference", "unsupported array component")
    identifier = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    return f"source-candidates/{identifier}/{component}"


def _ordered_anchor_ids(packets: tuple[object, ...]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for packet in packets:
        anchor_id = cast(str, cast(Any, packet).anchor_id)
        if anchor_id not in seen:
            values.append(anchor_id)
            seen.add(anchor_id)
    return tuple(values)


def _samples_by_candidate(dataset: object) -> dict[str, tuple[object, ...]]:
    grouped: dict[str, list[object]] = {}
    for sample in tuple(getattr(dataset, "samples", ())):
        candidate_id = cast(str, cast(Any, sample).candidate_sample_id)
        grouped.setdefault(candidate_id, []).append(sample)
    return {key: tuple(value) for key, value in grouped.items()}


def _require_packet_source_fields(
    packet: object,
    *,
    source_collection: object,
    source_trajectory_id: str,
    split: object,
    split_group_id: str,
    anchor_id: str,
    state_reference_id: str,
    expected_state_digest: str,
    verifier_state_semantic: str,
    verifier_state_digest: str,
    compatibility_identity: str,
) -> None:
    expected = {
        "source_collection": source_collection,
        "source_trajectory_id": source_trajectory_id,
        "split": split,
        "split_group_id": split_group_id,
        "anchor_id": anchor_id,
        "state_reference_id": state_reference_id,
        "expected_state_digest": expected_state_digest,
        "verifier_state_semantic": verifier_state_semantic,
        "verifier_state_digest": verifier_state_digest,
        "pickcube_compatibility_identity": compatibility_identity,
    }
    changed = [
        name for name, value in expected.items() if getattr(packet, name, None) != value
    ]
    if changed:
        _fail(anchor_id, "packet source fields differ: " + ", ".join(changed))


def _require_visual_sample_projection(
    visual_samples: tuple[object, ...],
    *,
    candidate_id: str,
    packet_ids: tuple[str, str, str],
    task_id: str,
    task_text: str,
    final_success: bool,
    final_unsafe: bool,
    failure_events: tuple[object, ...],
    evidence_id: str,
    source_dataset_digest: str,
    candidate_dataset_digest: str,
    source_collection: object,
) -> None:
    if tuple(getattr(item, "packet_id", None) for item in visual_samples) != packet_ids:
        _fail(candidate_id, "visual sample packet order differs from binding")
    expected = {
        "candidate_sample_id": candidate_id,
        "task_id": task_id,
        "canonical_task_text": task_text,
        "candidate_action_chunk_reference": visual_candidate_array_reference(
            candidate_id, "action-chunk"
        ),
        "action_mask_reference": visual_candidate_array_reference(
            candidate_id, "action-mask"
        ),
        "final_success": final_success,
        "final_unsafe": final_unsafe,
        "failure_events": failure_events,
        "evidence_id": evidence_id,
        "source_dataset_digest": source_dataset_digest,
        "candidate_dataset_digest": candidate_dataset_digest,
        "source_collection": source_collection,
    }
    for sample in visual_samples:
        changed = [
            name
            for name, value in expected.items()
            if getattr(sample, name, None) != value
        ]
        if changed:
            _fail(
                candidate_id,
                "visual label projection differs: " + ", ".join(changed),
            )


def validate_visual_development_source_binding(
    visual_dataset: object,
    source_dataset: object,
    anchor_manifest: object,
    binding: M3ADevelopmentSourceBindingV1,
    *,
    allow_partial: bool = False,
) -> None:
    """Reconcile a visual development dataset with freshly loaded M3A sources."""

    from latentguard.action_verifier.models import ActionVerifierDatasetV1
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        PickCubeAnchorManifestV1,
    )
    from latentguard.vision_data.models import (
        SourceCollection,
        VisualDatasetSplit,
        VisualVerifierDevelopmentDatasetV1,
    )

    if not isinstance(visual_dataset, VisualVerifierDevelopmentDatasetV1):
        _fail("M3A visual source validation", "expected development dataset")
    if not isinstance(source_dataset, ActionVerifierDatasetV1) or not isinstance(
        anchor_manifest, PickCubeAnchorManifestV1
    ):
        _fail("M3A visual source validation", "invalid source model")
    if (
        source_dataset.content_digest != binding.dataset_digest
        or anchor_manifest.content_digest != binding.anchor_manifest_digest
        or visual_dataset.source_dataset_digest != binding.dataset_digest
        or visual_dataset.split_digest != binding.split_digest
        or visual_dataset.evidence_digest != binding.evidence_digest
        or visual_dataset.source_compatibility_identity
        != binding.compatibility_identity
    ):
        _fail("M3A visual source validation", "embedded source identities differ")
    groups = {group.anchor_id: group for group in source_dataset.candidate_groups}
    records = {record.anchor.anchor_id: record for record in anchor_manifest.records}
    assignments = {
        item.source_trajectory_id: item for item in source_dataset.split_assignments
    }
    source_samples = {sample.sample_id: sample for sample in source_dataset.samples}
    visual_anchors = _ordered_anchor_ids(tuple(visual_dataset.packets))
    expected_anchors = tuple(
        group.anchor_id for group in source_dataset.candidate_groups
    )
    if (not allow_partial and visual_anchors != expected_anchors) or (
        allow_partial and any(anchor not in groups for anchor in visual_anchors)
    ):
        _fail("M3A visual source validation", "anchor inventory or order differs")
    packets_by_anchor: dict[str, list[object]] = {}
    for packet in visual_dataset.packets:
        packets_by_anchor.setdefault(packet.anchor_id, []).append(packet)
    for anchor_id in visual_anchors:
        group = groups[anchor_id]
        record = records.get(anchor_id)
        assignment = assignments.get(group.source_trajectory_id)
        if record is None or assignment is None:
            _fail(anchor_id, "source anchor record or split assignment is absent")
        if (
            record.anchor.source_trajectory_id != group.source_trajectory_id
            or record.anchor.source_seed != group.source_seed
            or record.anchor.split_group_id != group.split_group_id
            or record.source_state_content_digest != group.state_content_digest
            or assignment.dataset_split != group.dataset_split
        ):
            _fail(anchor_id, "M3A group, anchor, or split source differs")
        split = VisualDatasetSplit(group.dataset_split.value)
        for source_packet in packets_by_anchor[anchor_id]:
            _require_packet_source_fields(
                source_packet,
                source_collection=SourceCollection.M3A_DEVELOPMENT,
                source_trajectory_id=group.source_trajectory_id,
                split=split,
                split_group_id=group.split_group_id,
                anchor_id=anchor_id,
                state_reference_id=record.source_state_content_digest,
                expected_state_digest=record.source_state_digest,
                verifier_state_semantic=group.state_vector_semantic,
                verifier_state_digest=record.verifier_state_content_digest,
                compatibility_identity=binding.compatibility_identity,
            )
    expected_bindings: list[tuple[str, str, str]] = []
    for anchor_id in visual_anchors:
        group = groups[anchor_id]
        for candidate_id in (group.source_sample_id, *group.corrupted_sample_ids):
            expected_bindings.append((candidate_id, group.group_id, anchor_id))
    observed_bindings = [
        (item.candidate_sample_id, item.candidate_group_id, item.anchor_id)
        for item in visual_dataset.candidate_bindings
    ]
    if observed_bindings != expected_bindings:
        _fail("M3A visual source validation", "candidate inventory or order differs")
    visual_samples = _samples_by_candidate(visual_dataset)
    for binding_item in visual_dataset.candidate_bindings:
        source = source_samples.get(binding_item.candidate_sample_id)
        if source is None or source.anchor_id != binding_item.anchor_id:
            _fail(binding_item.candidate_sample_id, "M3A candidate source is absent")
        _require_visual_sample_projection(
            visual_samples.get(binding_item.candidate_sample_id, ()),
            candidate_id=binding_item.candidate_sample_id,
            packet_ids=binding_item.packet_ids,
            task_id=source.task_id,
            task_text=source.instruction,
            final_success=source.final_task_success,
            final_unsafe=source.final_unsafe,
            failure_events=cast(tuple[object, ...], source.failure_events),
            evidence_id=source.strong_simulator_evidence_id,
            source_dataset_digest=binding.dataset_digest,
            candidate_dataset_digest=binding.dataset_digest,
            source_collection=SourceCollection.M3A_DEVELOPMENT,
        )


def validate_visual_external_source_binding(
    visual_dataset: object,
    candidate_pool: object,
    anchor_manifest: object,
    state_archive: object,
    binding: M3CExternalSourceBindingV1,
    *,
    allow_partial: bool = False,
) -> None:
    """Reconcile an external visual dataset with M3C pool, archive, and evidence."""

    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        PickCubeStateIndexedArchiveV1,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        PickCubeAnchorManifestV1,
    )
    from latentguard.selection.models import CandidatePoolV1
    from latentguard.vision_data.models import (
        PICKCUBE_CANONICAL_TASK_TEXT,
        PICKCUBE_VISUAL_TASK_ID,
        SourceCollection,
        VisualDatasetSplit,
        VisualVerifierExternalDatasetV1,
    )

    if not isinstance(visual_dataset, VisualVerifierExternalDatasetV1):
        _fail("M3C visual source validation", "expected external dataset")
    if (
        not isinstance(candidate_pool, CandidatePoolV1)
        or not isinstance(anchor_manifest, PickCubeAnchorManifestV1)
        or not isinstance(state_archive, PickCubeStateIndexedArchiveV1)
    ):
        _fail("M3C visual source validation", "invalid source model")
    if (
        candidate_pool.content_digest != binding.candidate_pool_digest
        or candidate_pool.source_set_digest != binding.source_set_digest
        or anchor_manifest.source_archive_content_digest != state_archive.content_digest
        or visual_dataset.source_set_identity != binding.source_set_digest
        or visual_dataset.candidate_pool_identity != binding.candidate_pool_digest
        or visual_dataset.blind_manifest_digest != binding.blind_manifest_digest
        or visual_dataset.full_outcome_digest != binding.full_outcome_digest
        or visual_dataset.source_compatibility_identity
        != binding.simulator_compatibility_identity
    ):
        _fail("M3C visual source validation", "embedded source identities differ")
    groups = {group.anchor_id: group for group in candidate_pool.groups}
    records = {record.anchor.anchor_id: record for record in anchor_manifest.records}
    episodes = {episode.episode_id: episode for episode in state_archive.episodes}
    visual_anchors = _ordered_anchor_ids(tuple(visual_dataset.packets))
    expected_anchors = tuple(group.anchor_id for group in candidate_pool.groups)
    if (not allow_partial and visual_anchors != expected_anchors) or (
        allow_partial and any(anchor not in groups for anchor in visual_anchors)
    ):
        _fail("M3C visual source validation", "anchor inventory or order differs")
    packets_by_anchor: dict[str, list[object]] = {}
    for packet in visual_dataset.packets:
        packets_by_anchor.setdefault(packet.anchor_id, []).append(packet)
    for anchor_id in visual_anchors:
        group = groups[anchor_id]
        record = records.get(anchor_id)
        if record is None:
            _fail(anchor_id, "M3C anchor record is absent")
        episode = episodes.get(record.source_archive_episode_id)
        if episode is None or record.anchor.state_index >= len(episode.states):
            _fail(anchor_id, "M3C archived state is absent")
        state = episode.states[record.anchor.state_index]
        if (
            record.anchor.source_trajectory_id != group.source_trajectory_id
            or record.anchor.source_seed != group.trajectory.source_seed
            or record.anchor.split_group_id != group.trajectory.split_group_id
            or record.source_state_content_digest != group.state_content_digest
            or state.content_digest != group.state_content_digest
            or state.verifier_state.content_digest
            != group.verifier_state_content_digest
        ):
            _fail(anchor_id, "M3C pool, anchor, or archived state differs")
        for source_packet in packets_by_anchor[anchor_id]:
            _require_packet_source_fields(
                source_packet,
                source_collection=SourceCollection.M3C_EXTERNAL,
                source_trajectory_id=group.source_trajectory_id,
                split=VisualDatasetSplit.EXTERNAL,
                split_group_id=group.trajectory.split_group_id,
                anchor_id=anchor_id,
                state_reference_id=state.content_digest,
                expected_state_digest=state.state_digest,
                verifier_state_semantic=state.verifier_state.semantic,
                verifier_state_digest=state.verifier_state.content_digest,
                compatibility_identity=binding.simulator_compatibility_identity,
            )
    expected_bindings = [
        (candidate.proposal_id, group.group_id, group.anchor_id)
        for group in candidate_pool.groups
        if group.anchor_id in set(visual_anchors)
        for candidate in group.candidates
    ]
    observed_bindings = [
        (item.candidate_sample_id, item.candidate_group_id, item.anchor_id)
        for item in visual_dataset.candidate_bindings
    ]
    if observed_bindings != expected_bindings:
        _fail("M3C visual source validation", "candidate inventory or order differs")
    visual_samples = _samples_by_candidate(visual_dataset)
    for binding_item in visual_dataset.candidate_bindings:
        candidate_id = binding_item.candidate_sample_id
        evidence = binding.evidence_by_candidate.get(candidate_id)
        outcome = binding.outcomes_by_candidate.get(candidate_id)
        if evidence is None or outcome is None:
            _fail(candidate_id, "M3C evidence or outcome is absent")
        if type(evidence.success) is not bool or type(evidence.unsafe) is not bool:
            _fail(candidate_id, "M3C evidence is not conclusive")
        _require_visual_sample_projection(
            visual_samples.get(candidate_id, ()),
            candidate_id=candidate_id,
            packet_ids=binding_item.packet_ids,
            task_id=PICKCUBE_VISUAL_TASK_ID,
            task_text=PICKCUBE_CANONICAL_TASK_TEXT,
            final_success=evidence.success,
            final_unsafe=evidence.unsafe,
            failure_events=cast(tuple[object, ...], evidence.failure_events),
            evidence_id=evidence.evidence_id,
            source_dataset_digest=binding.source_set_digest,
            candidate_dataset_digest=binding.candidate_pool_digest,
            source_collection=SourceCollection.M3C_EXTERNAL,
        )


def _evidence_projection_digest(
    candidate_ids: tuple[str, ...],
    evidence_by_candidate: Mapping[str, EvaluationEvidence],
) -> str:
    """Bind the exact ordered strong-label and failure-event projection."""

    payload = []
    for candidate_id in candidate_ids:
        evidence = evidence_by_candidate[candidate_id]
        payload.append(
            {
                "candidate_id": candidate_id,
                "evidence_id": evidence.evidence_id,
                "failure_events": [
                    {
                        "description": event.description,
                        "failure_type": event.failure_type,
                        "probability": event.probability,
                        "schema_version": event.schema_version,
                        "timestamp_s": event.timestamp_s,
                    }
                    for event in evidence.failure_events
                ],
                "label_source": (
                    evidence.label_source.value
                    if evidence.label_source is not None
                    else None
                ),
                "label_strength": (
                    evidence.label_strength.value
                    if evidence.label_strength is not None
                    else None
                ),
                "simulator_replay_verified": evidence.simulator_replay_verified,
                "status": evidence.status.value,
                "success": evidence.success,
                "unsafe": evidence.unsafe,
            }
        )
    encoded = canonical_json_bytes(
        {
            "candidate_evidence": payload,
            "schema_version": "m4a_m3c_evidence_projection_v1",
        },
        context="M4AM3CEvidenceProjectionV1",
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def load_m3a_development_source_binding(
    dataset_dir: Path,
    runtime_archive_dir: Path,
    anchor_manifest_dir: Path,
    *,
    acceptance_report: Path,
) -> M3ADevelopmentSourceBindingV1:
    """Independently reload the complete fixed accepted M3A source chain."""

    from latentguard.action_verifier import load_action_verifier_dataset
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.training.dataset import load_accepted_action_verifier_dataset

    dataset = load_action_verifier_dataset(dataset_dir)
    archive = load_state_indexed_archive(runtime_archive_dir)
    manifest = load_anchor_manifest(anchor_manifest_dir)
    if manifest.source_archive_content_digest != archive.content_digest:
        _fail("M3A source archive", "anchor manifest archive digest differs")
    reasons = {
        record.anchor.anchor_id: record.anchor.selection_reason
        for record in manifest.records
    }
    accepted = load_accepted_action_verifier_dataset(
        dataset_dir,
        full_target_report=acceptance_report,
        anchor_selection_reasons=reasons,
        anchor_manifest_content_digest=manifest.content_digest,
        expected_dataset_digest=ACCEPTED_M3A_DATASET_DIGEST,
    )
    evidence = {sample.evidence_dataset_digest for sample in dataset.samples}
    compatibility = {sample.compatibility_identity for sample in dataset.samples}
    if len(evidence) != 1 or len(compatibility) != 1:
        _fail("M3A dataset", "evidence or compatibility identity is not unique")
    if dataset.content_digest != accepted.dataset_digest:
        _fail("M3A dataset", "independent and accepted dataset_digest values differ")
    return M3ADevelopmentSourceBindingV1(
        dataset_digest=accepted.dataset_digest,
        source_archive_digest=archive.content_digest,
        split_digest=accepted.split_digest,
        evidence_digest=next(iter(evidence)),
        acceptance_report_digest=accepted.acceptance_report_digest,
        anchor_manifest_digest=manifest.content_digest,
        compatibility_identity=next(iter(compatibility)),
        trajectory_ids=tuple(
            assignment.source_trajectory_id for assignment in dataset.split_assignments
        ),
        anchor_ids=tuple(group.anchor_id for group in dataset.candidate_groups),
        candidate_sample_ids=tuple(sample.sample_id for sample in dataset.samples),
    )


def load_m3c_external_source_binding(
    candidate_pool_dir: Path,
    blind_manifest_path: Path,
    bound_result_path: Path,
    corruption_dir: Path,
    selected_evaluation_dir: Path,
    remainder_evaluation_dir: Path,
    *,
    expected_execution_git_sha: str = ACCEPTED_M3C_EXECUTION_GIT_SHA,
) -> M3CExternalSourceBindingV1:
    """Reload and cross-bind the complete accepted M3C external outcome chain."""

    if expected_execution_git_sha != ACCEPTED_M3C_EXECUTION_GIT_SHA:
        _fail(
            "M3C execution Git SHA",
            "caller cannot override the fixed accepted execution revision",
        )

    from latentguard.corruptions.serialization import load_corruption_dataset
    from latentguard.evaluation.serialization import (
        compute_corruption_dataset_digest,
        load_evaluation_dataset,
    )
    from latentguard.selection.blind_protocol import (
        compute_full_pool_outcome_digest,
        load_bound_selection_result,
    )
    from latentguard.selection.evaluation import join_complementary_replay_phases
    from latentguard.selection.manifest import load_blind_selection_manifest
    from latentguard.selection.outcomes import candidate_outcomes_from_replay_join
    from latentguard.selection.serialization import load_candidate_pool

    pool = load_candidate_pool(candidate_pool_dir)
    envelope = load_blind_selection_manifest(blind_manifest_path)
    result = load_bound_selection_result(
        bound_result_path,
        selection_envelope=envelope,
        candidate_pool=pool,
    )
    corruptions = load_corruption_dataset(corruption_dir)
    corruption_digest = compute_corruption_dataset_digest(corruption_dir)
    selected = load_evaluation_dataset(
        selected_evaluation_dir,
        corruption_dataset=corruptions,
        expected_corruption_digest=corruption_digest,
    )
    remainder = load_evaluation_dataset(
        remainder_evaluation_dir,
        corruption_dataset=corruptions,
        expected_corruption_digest=corruption_digest,
    )
    joined = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=envelope,
        selected_dataset=selected,
        remainder_dataset=remainder,
    )
    outcomes = candidate_outcomes_from_replay_join(pool, joined)
    if joined.content_digest != result.replay_evidence_digest:
        _fail("M3C replay evidence", "semantic replay digest differs from result")
    if compute_full_pool_outcome_digest(outcomes) != result.full_pool_outcome_digest:
        _fail("M3C outcomes", "full outcome digest differs from result")
    if result.git_sha != expected_execution_git_sha:
        _fail("M3C execution Git SHA", "differs from accepted execution")
    accepted_identities = {
        "source_set_digest": ACCEPTED_M3C_SOURCE_SET_DIGEST,
        "candidate_pool_digest": ACCEPTED_M3C_CANDIDATE_POOL_DIGEST,
        "blind_manifest_digest": ACCEPTED_M3C_BLIND_MANIFEST_DIGEST,
        "blind_manifest_envelope_digest": (ACCEPTED_M3C_BLIND_MANIFEST_ENVELOPE_DIGEST),
        "bound_result_digest": ACCEPTED_M3C_BOUND_RESULT_DIGEST,
        "replay_evidence_digest": ACCEPTED_M3C_REPLAY_EVIDENCE_DIGEST,
        "full_outcome_digest": ACCEPTED_M3C_FULL_OUTCOME_DIGEST,
        "simulator_compatibility_identity": (
            ACCEPTED_M3C_SIMULATOR_COMPATIBILITY_IDENTITY
        ),
    }
    observed_identities = {
        "source_set_digest": pool.source_set_digest,
        "candidate_pool_digest": pool.content_digest,
        "blind_manifest_digest": envelope.semantic_digest,
        "blind_manifest_envelope_digest": envelope.envelope_digest,
        "bound_result_digest": result.content_digest,
        "replay_evidence_digest": joined.content_digest,
        "full_outcome_digest": result.full_pool_outcome_digest,
        "simulator_compatibility_identity": result.simulator_compatibility_identity,
    }
    drifted = tuple(
        name
        for name, expected in accepted_identities.items()
        if observed_identities[name] != expected
    )
    if drifted:
        _fail(
            "accepted M3C artifact identities",
            "differ in " + ", ".join(drifted),
        )
    return M3CExternalSourceBindingV1(
        source_set_digest=pool.source_set_digest,
        candidate_pool_digest=pool.content_digest,
        blind_manifest_digest=envelope.semantic_digest,
        blind_manifest_envelope_digest=envelope.envelope_digest,
        bound_result_digest=result.content_digest,
        replay_evidence_digest=joined.content_digest,
        evidence_projection_digest=_evidence_projection_digest(
            pool.proposal_ids, joined.evidence_by_proposal
        ),
        full_outcome_digest=result.full_pool_outcome_digest,
        simulator_compatibility_identity=result.simulator_compatibility_identity,
        execution_git_sha=result.git_sha,
        trajectory_ids=_ordered_unique(
            tuple(group.source_trajectory_id for group in pool.groups)
        ),
        anchor_ids=tuple(group.anchor_id for group in pool.groups),
        candidate_ids=pool.proposal_ids,
        outcomes_by_candidate={item.proposal_id: item for item in outcomes},
        evidence_by_candidate=joined.evidence_by_proposal,
    )


def require_m3c_anchor_archive_binding(
    binding: M3CExternalSourceBindingV1,
    source_dir: Path,
    runtime_archive_dir: Path,
    anchor_manifest_dir: Path,
) -> None:
    """Require available M3C anchor states to reproduce the accepted source set."""

    from latentguard.selection.candidate_pool import compute_source_set_digest
    from latentguard.selection.source import load_candidate_pool_sources

    sources = load_candidate_pool_sources(
        source_dir,
        runtime_archive_dir,
        anchor_manifest_dir,
    )
    if compute_source_set_digest(sources) != binding.source_set_digest:
        _fail("M3C source archive", "source-set digest differs from accepted result")
    if tuple(item.anchor_id for item in sources) != binding.anchor_ids:
        _fail("M3C source archive", "anchor inventory or ordering differs")


__all__ = [
    "ACCEPTED_M3A_DATASET_DIGEST",
    "ACCEPTED_M3A_ACCEPTANCE_REPORT_DIGEST",
    "ACCEPTED_M3A_ANCHOR_MANIFEST_DIGEST",
    "ACCEPTED_M3A_EVIDENCE_DIGEST",
    "ACCEPTED_M3A_SIMULATOR_COMPATIBILITY_IDENTITY",
    "ACCEPTED_M3A_SOURCE_ARCHIVE_DIGEST",
    "ACCEPTED_M3A_SPLIT_DIGEST",
    "ACCEPTED_M3C_BLIND_MANIFEST_DIGEST",
    "ACCEPTED_M3C_BLIND_MANIFEST_ENVELOPE_DIGEST",
    "ACCEPTED_M3C_BOUND_RESULT_DIGEST",
    "ACCEPTED_M3C_CANDIDATE_POOL_DIGEST",
    "ACCEPTED_M3C_EXECUTION_GIT_SHA",
    "ACCEPTED_M3C_FULL_OUTCOME_DIGEST",
    "ACCEPTED_M3C_REPLAY_EVIDENCE_DIGEST",
    "ACCEPTED_M3C_SIMULATOR_COMPATIBILITY_IDENTITY",
    "ACCEPTED_M3C_SOURCE_SET_DIGEST",
    "M3ADevelopmentSourceBindingV1",
    "M3CExternalSourceBindingV1",
    "VisualSourceBindingError",
    "load_m3a_development_source_binding",
    "load_m3c_external_source_binding",
    "require_m3c_anchor_archive_binding",
    "validate_visual_development_source_binding",
    "validate_visual_external_source_binding",
    "visual_candidate_array_reference",
]
