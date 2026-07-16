"""Manifest-gated reuse of the existing exact replay runner and ledger.

M3C evaluates one immutable candidate pool in two complementary physical replay
phases.  The first phase executes the union of proposals selected blindly by
all selectors.  The second executes only the remaining proposals.  Both phases
still pass the complete, unchanged corruption dataset to the M2A runner; this
preserves proposal ordinals, proposal identities, replay cases, and the durable
evaluation ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.base import ApplicabilityDecision, ProposalEvaluator
from latentguard.evaluation.models import (
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
    validate_evaluator_identity,
)
from latentguard.evaluation.runner import EvaluationRunResult
from latentguard.evaluation.serialization import (
    EvaluationDataset,
    LedgerState,
    compute_evaluation_dataset_content_digest,
)
from latentguard.evaluation.validation import validate_evaluation_evidence
from latentguard.models import LabelSource, LabelStrength

MANIFEST_GATED_REPLAY_EVALUATOR_ID = "m3c_manifest_gated_replay"
MANIFEST_GATED_REPLAY_EVALUATOR_VERSION = "1.0.0"
MANIFEST_GATED_REPLAY_SCHEMA_VERSION = "1.0"
M3C_STATE_COMPONENT_COUNT = 70
M3C_CANDIDATE_HORIZON = 16
M3C_STATE_RESTORATION_TOLERANCE = 1e-6

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReplayPhase(StrEnum):
    """Complementary M3C physical replay phases."""

    SELECTED = "selected"
    REMAINDER = "remainder"


class SelectionReplayError(ValueError):
    """Raised when a blind manifest, phase run, or strong evidence conflicts."""


@runtime_checkable
class CandidatePoolReplayView(Protocol):
    """Minimum validated candidate-pool view consumed by replay gating."""

    @property
    def content_digest(self) -> str:
        """Return the complete path-independent candidate-pool digest."""
        ...

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        """Return every proposal ID in immutable pool order."""
        ...


@runtime_checkable
class BlindSelectionReplayEntryView(Protocol):
    """Minimum immutable selector decision needed for physical replay."""

    @property
    def selected_proposal_id(self) -> str | None:
        """Return the blindly selected proposal, or null for abstention."""
        ...

    @property
    def abstained(self) -> bool:
        """Return whether this selector deliberately abstained."""
        ...


@runtime_checkable
class BlindSelectionManifestReplayView(Protocol):
    """Minimum finalized blind-manifest view consumed by replay gating."""

    @property
    def content_digest(self) -> str:
        """Return the complete immutable selection-manifest digest."""
        ...

    @property
    def manifest_envelope_digest(self) -> str:
        """Return the strict audit digest that also binds the timestamp."""
        ...

    @property
    def candidate_pool_digest(self) -> str:
        """Return the candidate pool bound before any outcome existed."""
        ...

    @property
    def selections(self) -> tuple[BlindSelectionReplayEntryView, ...]:
        """Return all blind selector decisions, including abstentions."""
        ...


@dataclass(frozen=True, slots=True)
class ReplayPhaseInventory:
    """Detached complete, selected-union, and remainder proposal inventories."""

    proposal_ids: tuple[str, ...]
    selected_proposal_ids: tuple[str, ...]
    remainder_proposal_ids: tuple[str, ...]
    candidate_pool_digest: str
    selection_manifest_digest: str
    selection_manifest_envelope_digest: str

    def __post_init__(self) -> None:
        """Require an exact ordered partition of one immutable candidate pool."""
        proposal_ids = _proposal_ids(self.proposal_ids, "candidate pool proposal IDs")
        selected = _proposal_ids(
            self.selected_proposal_ids,
            "selected replay proposal IDs",
            allow_empty=True,
        )
        remainder = _proposal_ids(
            self.remainder_proposal_ids,
            "remainder replay proposal IDs",
            allow_empty=True,
        )
        if set(selected) & set(remainder):
            raise SelectionReplayError(
                "selected and remainder replay proposal inventories overlap"
            )
        if set((*selected, *remainder)) != set(proposal_ids):
            raise SelectionReplayError(
                "selected and remainder replay inventories do not partition the pool"
            )
        expected_selected = tuple(
            item for item in proposal_ids if item in set(selected)
        )
        expected_remainder = tuple(
            item for item in proposal_ids if item not in set(selected)
        )
        if selected != expected_selected or remainder != expected_remainder:
            raise SelectionReplayError(
                "replay phase inventories must preserve candidate-pool order"
            )
        _digest(self.candidate_pool_digest, "candidate pool digest")
        _digest(self.selection_manifest_digest, "selection manifest digest")
        _digest(
            self.selection_manifest_envelope_digest,
            "selection manifest envelope digest",
        )
        object.__setattr__(self, "proposal_ids", proposal_ids)
        object.__setattr__(self, "selected_proposal_ids", selected)
        object.__setattr__(self, "remainder_proposal_ids", remainder)

    def executable_ids(self, phase: ReplayPhase) -> tuple[str, ...]:
        """Return the exact physical execution set for one phase."""
        if phase is ReplayPhase.SELECTED:
            return self.selected_proposal_ids
        if phase is ReplayPhase.REMAINDER:
            return self.remainder_proposal_ids
        raise SelectionReplayError("unsupported replay phase")


@dataclass(frozen=True, slots=True)
class ReplayPhaseEvaluators:
    """The two complementary wrappers derived from one blind manifest."""

    selected: ManifestGatedReplayEvaluator
    remainder: ManifestGatedReplayEvaluator


@dataclass(frozen=True, slots=True)
class CompletePoolReplayJoin:
    """One strong executed result per proposal after complementary phase replay."""

    candidate_pool_digest: str
    selection_manifest_digest: str
    selection_manifest_envelope_digest: str
    selected_phase_dataset_digest: str
    remainder_phase_dataset_digest: str
    proposal_ids: tuple[str, ...]
    selected_phase_executed_ids: tuple[str, ...]
    remainder_phase_executed_ids: tuple[str, ...]
    evidence_by_proposal: Mapping[str, EvaluationEvidence]

    def __post_init__(self) -> None:
        """Detach evidence and recheck the exact one-result-per-proposal invariant."""
        _digest(self.candidate_pool_digest, "candidate pool digest")
        _digest(self.selection_manifest_digest, "selection manifest digest")
        for value, context in (
            (
                self.selection_manifest_envelope_digest,
                "selection manifest envelope digest",
            ),
            (self.selected_phase_dataset_digest, "selected phase dataset digest"),
            (self.remainder_phase_dataset_digest, "remainder phase dataset digest"),
        ):
            _digest(value, context)
        proposal_ids = _proposal_ids(self.proposal_ids, "joined proposal IDs")
        selected = _proposal_ids(
            self.selected_phase_executed_ids,
            "joined selected proposal IDs",
            allow_empty=True,
        )
        remainder = _proposal_ids(
            self.remainder_phase_executed_ids,
            "joined remainder proposal IDs",
            allow_empty=True,
        )
        if set(selected) & set(remainder) or set((*selected, *remainder)) != set(
            proposal_ids
        ):
            raise SelectionReplayError(
                "joined physical replay inventories are not an exact partition"
            )
        evidence = dict(self.evidence_by_proposal)
        if set(evidence) != set(proposal_ids):
            raise SelectionReplayError(
                "joined replay evidence does not cover every candidate exactly once"
            )
        evidence_ids = tuple(item.evidence_id for item in evidence.values())
        if len(evidence_ids) != len(set(evidence_ids)):
            raise SelectionReplayError("joined replay contains duplicate evidence IDs")
        object.__setattr__(self, "proposal_ids", proposal_ids)
        object.__setattr__(self, "selected_phase_executed_ids", selected)
        object.__setattr__(self, "remainder_phase_executed_ids", remainder)
        object.__setattr__(
            self,
            "evidence_by_proposal",
            MappingProxyType({item: evidence[item] for item in proposal_ids}),
        )

    @property
    def content_digest(self) -> str:
        """Return the path-, host-, and time-independent replay identity.

        The complete phase dataset digests and manifest envelope remain exact
        archive-audit fields, but they intentionally do not participate here.
        They contain operational timestamps and environment metadata that must
        never change the semantic identity of an otherwise identical replay.
        """
        return _content_digest(
            {
                "candidate_pool_digest": self.candidate_pool_digest,
                "evidence": [
                    _semantic_evidence_payload(self.evidence_by_proposal[item])
                    for item in self.proposal_ids
                ],
                "proposal_ids": list(self.proposal_ids),
                "remainder_phase_executed_ids": list(self.remainder_phase_executed_ids),
                "schema_version": "m3c_complete_pool_replay_semantic_v1",
                "selected_phase_executed_ids": list(self.selected_phase_executed_ids),
                "selection_manifest_digest": self.selection_manifest_digest,
            }
        )

    @property
    def archive_audit_digest(self) -> str:
        """Bind exact phase archives and the timestamped manifest envelope."""
        return _content_digest(
            {
                "remainder_phase_dataset_digest": self.remainder_phase_dataset_digest,
                "schema_version": "m3c_complete_pool_replay_archive_audit_v1",
                "selected_phase_dataset_digest": self.selected_phase_dataset_digest,
                "selection_manifest_envelope_digest": (
                    self.selection_manifest_envelope_digest
                ),
                "semantic_content_digest": self.content_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class ZeroWorkResumeReport:
    """Proof that one completed phase resume performed no duplicate work."""

    run_id: str
    evidence_count: int
    ledger_count: int
    evaluated_attempts: int
    recovered_attempts: int
    retried_attempts: int
    zero_duplicate_work: bool = True


class ManifestGatedReplayEvaluator:
    """Gate one exact replay evaluator with a finalized blind selection manifest."""

    def __init__(
        self,
        inner: ProposalEvaluator,
        *,
        phase: ReplayPhase,
        candidate_pool: CandidatePoolReplayView,
        selection_manifest: BlindSelectionManifestReplayView,
    ) -> None:
        """Derive the phase execution set without accepting an external allowlist."""
        if not isinstance(inner, ProposalEvaluator):
            raise SelectionReplayError("inner evaluator does not implement protocol")
        if not isinstance(phase, ReplayPhase):
            raise SelectionReplayError("replay phase is invalid")
        validate_evaluator_identity(
            evaluator_id=inner.evaluator_id,
            evaluator_version=inner.evaluator_version,
        )
        inner_configuration = inner.resolved_configuration()
        if not isinstance(inner_configuration, Mapping):
            raise SelectionReplayError(
                "inner evaluator resolved configuration must be a mapping"
            )
        inner_configuration_digest = compute_configuration_digest(inner_configuration)
        if inner.configuration_digest != inner_configuration_digest:
            raise SelectionReplayError(
                "inner evaluator configuration digest does not match its content"
            )
        inventory, pool_snapshot, manifest_snapshot = _derive_phase_inventory(
            candidate_pool, selection_manifest
        )
        executable = inventory.executable_ids(phase)
        configuration: Mapping[str, object] = MappingProxyType(
            {
                "candidate_pool_digest": inventory.candidate_pool_digest,
                "candidate_pool_proposal_count": len(inventory.proposal_ids),
                "candidate_pool_proposal_inventory_digest": _content_digest(
                    {"proposal_ids": list(inventory.proposal_ids)}
                ),
                "executable_proposal_count": len(executable),
                "executable_proposal_ids": executable,
                "executable_proposal_inventory_digest": _content_digest(
                    {"proposal_ids": list(executable)}
                ),
                "inner_evaluator_configuration_digest": inner_configuration_digest,
                "inner_evaluator_id": inner.evaluator_id,
                "inner_evaluator_version": inner.evaluator_version,
                "phase": phase.value,
                "schema_version": MANIFEST_GATED_REPLAY_SCHEMA_VERSION,
                "selection_manifest_digest": inventory.selection_manifest_digest,
                "selection_manifest_envelope_digest": (
                    inventory.selection_manifest_envelope_digest
                ),
            }
        )
        self._inner = inner
        self._phase = phase
        self._candidate_pool = candidate_pool
        self._selection_manifest = selection_manifest
        self._inventory = inventory
        self._pool_snapshot = pool_snapshot
        self._manifest_snapshot = manifest_snapshot
        self._executable = frozenset(executable)
        self._configuration = configuration
        self._configuration_digest = compute_configuration_digest(configuration)
        self._inner_configuration_digest = inner_configuration_digest

    @property
    def evaluator_id(self) -> str:
        """Return the stable M3C phase-gate evaluator identity."""
        return MANIFEST_GATED_REPLAY_EVALUATOR_ID

    @property
    def evaluator_version(self) -> str:
        """Return the phase-gate semantic version."""
        return MANIFEST_GATED_REPLAY_EVALUATOR_VERSION

    @property
    def configuration_digest(self) -> str:
        """Bind phase, pool, manifest, inner evaluator, and execution inventory."""
        return self._configuration_digest

    @property
    def phase(self) -> ReplayPhase:
        """Return the complementary physical replay phase."""
        return self._phase

    @property
    def inventory(self) -> ReplayPhaseInventory:
        """Return the detached partition derived from the blind manifest."""
        return self._inventory

    @property
    def executable_proposal_ids(self) -> tuple[str, ...]:
        """Return executable IDs in candidate-pool order."""
        return self._inventory.executable_ids(self._phase)

    def resolved_configuration(self) -> Mapping[str, object]:
        """Return the complete path-independent phase configuration."""
        return self._configuration

    def check_applicability(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
    ) -> ApplicabilityDecision:
        """Skip the complementary phase without creating a simulator session."""
        self._assert_contract_unchanged()
        if proposal.proposal_id not in set(self._inventory.proposal_ids):
            return ApplicabilityDecision.invalid("proposal_not_in_candidate_pool")
        if proposal.proposal_id not in self._executable:
            return ApplicabilityDecision.skipped("proposal_not_executable_in_phase")
        decision = self._inner.check_applicability(
            proposal, source_dataset_id=source_dataset_id
        )
        if not isinstance(decision, ApplicabilityDecision):
            raise SelectionReplayError(
                "inner evaluator applicability result violates protocol"
            )
        self._assert_contract_unchanged()
        return decision

    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        """Execute one allowed proposal and rebind strong evidence to this phase."""
        self._assert_contract_unchanged()
        if proposal.proposal_id not in self._inventory.proposal_ids:
            raise SelectionReplayError("proposal is unknown to the candidate pool")
        if proposal.proposal_id not in self._executable:
            raise SelectionReplayError(
                "proposal is not executable in this replay phase"
            )
        evidence = self._inner.evaluate(
            proposal,
            source_dataset_id=source_dataset_id,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
        )
        self._assert_contract_unchanged()
        _validate_inner_evidence_binding(
            evidence,
            proposal=proposal,
            source_dataset_id=source_dataset_id,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
            inner=self._inner,
        )
        _validate_m3c_strong_replay_evidence(evidence, proposal)
        rebound = replace(
            evidence,
            evidence_id=compute_evidence_identifier(
                proposal_id=proposal.proposal_id,
                evaluator_id=self.evaluator_id,
                evaluator_version=self.evaluator_version,
                evaluator_configuration_digest=self.configuration_digest,
                evaluation_seed=evaluation_seed,
                attempt_ordinal=attempt_ordinal,
            ),
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            evaluator_configuration_digest=self.configuration_digest,
            schema_version=EVALUATION_EVIDENCE_SCHEMA_VERSION,
        )
        validate_evaluation_evidence(rebound)
        return rebound

    def _assert_contract_unchanged(self) -> None:
        """Reject artifact or delegate drift around every simulator attempt."""
        inventory, pool_snapshot, manifest_snapshot = _derive_phase_inventory(
            self._candidate_pool, self._selection_manifest
        )
        if (
            inventory != self._inventory
            or pool_snapshot != self._pool_snapshot
            or manifest_snapshot != self._manifest_snapshot
        ):
            raise SelectionReplayError(
                "candidate pool or blind selection manifest changed after binding"
            )
        resolved = self._inner.resolved_configuration()
        if (
            not isinstance(resolved, Mapping)
            or self._inner.evaluator_id != self._configuration["inner_evaluator_id"]
            or self._inner.evaluator_version
            != self._configuration["inner_evaluator_version"]
            or self._inner.configuration_digest != self._inner_configuration_digest
            or compute_configuration_digest(resolved)
            != self._inner_configuration_digest
        ):
            raise SelectionReplayError(
                "inner evaluator contract changed after phase binding"
            )


def build_manifest_gated_replay_evaluators(
    inner: ProposalEvaluator,
    *,
    candidate_pool: CandidatePoolReplayView,
    selection_manifest: BlindSelectionManifestReplayView,
) -> ReplayPhaseEvaluators:
    """Build selected-union and exact-complement wrappers from one manifest."""
    selected = ManifestGatedReplayEvaluator(
        inner,
        phase=ReplayPhase.SELECTED,
        candidate_pool=candidate_pool,
        selection_manifest=selection_manifest,
    )
    remainder = ManifestGatedReplayEvaluator(
        inner,
        phase=ReplayPhase.REMAINDER,
        candidate_pool=candidate_pool,
        selection_manifest=selection_manifest,
    )
    if selected.inventory != remainder.inventory:
        raise SelectionReplayError(
            "selected and remainder wrappers derived different pool inventories"
        )
    return ReplayPhaseEvaluators(selected=selected, remainder=remainder)


def join_complementary_replay_phases(
    *,
    candidate_pool: CandidatePoolReplayView,
    selection_manifest: BlindSelectionManifestReplayView,
    selected_dataset: EvaluationDataset,
    remainder_dataset: EvaluationDataset,
) -> CompletePoolReplayJoin:
    """Validate phase outputs and return one strong outcome per pool proposal."""
    inventory, _, _ = _derive_phase_inventory(candidate_pool, selection_manifest)
    selected = _phase_evidence(
        selected_dataset, inventory=inventory, phase=ReplayPhase.SELECTED
    )
    remainder = _phase_evidence(
        remainder_dataset, inventory=inventory, phase=ReplayPhase.REMAINDER
    )
    joined: dict[str, EvaluationEvidence] = {}
    for proposal_id in inventory.proposal_ids:
        selected_evidence = selected[proposal_id]
        remainder_evidence = remainder[proposal_id]
        expected_phase = (
            ReplayPhase.SELECTED
            if proposal_id in set(inventory.selected_proposal_ids)
            else ReplayPhase.REMAINDER
        )
        executed = (
            selected_evidence
            if expected_phase is ReplayPhase.SELECTED
            else remainder_evidence
        )
        skipped = (
            remainder_evidence
            if expected_phase is ReplayPhase.SELECTED
            else selected_evidence
        )
        if executed.status is not EvaluationStatus.CONCLUSIVE:
            raise SelectionReplayError(
                f"proposal {proposal_id!r} lacks conclusive evidence in its phase"
            )
        if (
            skipped.status is not EvaluationStatus.SKIPPED
            or skipped.termination_reason != "proposal_not_executable_in_phase"
        ):
            raise SelectionReplayError(
                f"proposal {proposal_id!r} was not skipped in the complementary phase"
            )
        _validate_joined_strong_evidence(executed)
        joined[proposal_id] = executed
    return CompletePoolReplayJoin(
        candidate_pool_digest=inventory.candidate_pool_digest,
        selection_manifest_digest=inventory.selection_manifest_digest,
        selection_manifest_envelope_digest=(
            inventory.selection_manifest_envelope_digest
        ),
        selected_phase_dataset_digest=compute_evaluation_dataset_content_digest(
            selected_dataset
        ),
        remainder_phase_dataset_digest=compute_evaluation_dataset_content_digest(
            remainder_dataset
        ),
        proposal_ids=inventory.proposal_ids,
        selected_phase_executed_ids=inventory.selected_proposal_ids,
        remainder_phase_executed_ids=inventory.remainder_proposal_ids,
        evidence_by_proposal=joined,
    )


def verify_zero_work_resume(
    before: EvaluationDataset,
    resumed: EvaluationRunResult,
) -> ZeroWorkResumeReport:
    """Require a completed ``--resume`` invocation to preserve all run content."""
    if not isinstance(before, EvaluationDataset):
        raise SelectionReplayError("resume baseline must be EvaluationDataset")
    if not isinstance(resumed, EvaluationRunResult):
        raise SelectionReplayError("resume result must be EvaluationRunResult")
    if (
        not resumed.resumed
        or resumed.evaluated_attempts != 0
        or resumed.recovered_attempts != 0
        or resumed.retried_attempts != 0
        or resumed.stopped_early
        or _dataset_resume_snapshot(resumed.dataset) != _dataset_resume_snapshot(before)
        or resumed.dataset.run_id != before.run_id
        or len(resumed.dataset.evidence) != len(before.evidence)
        or len(resumed.dataset.ledger) != len(before.ledger)
    ):
        raise SelectionReplayError(
            "completed replay resume performed work or changed persisted content"
        )
    return ZeroWorkResumeReport(
        run_id=before.run_id,
        evidence_count=len(before.evidence),
        ledger_count=len(before.ledger),
        evaluated_attempts=resumed.evaluated_attempts,
        recovered_attempts=resumed.recovered_attempts,
        retried_attempts=resumed.retried_attempts,
    )


def _derive_phase_inventory(
    candidate_pool: CandidatePoolReplayView,
    selection_manifest: BlindSelectionManifestReplayView,
) -> tuple[
    ReplayPhaseInventory,
    tuple[object, ...],
    tuple[object, ...],
]:
    try:
        pool_digest = _digest(candidate_pool.content_digest, "candidate pool digest")
        proposal_ids = _proposal_ids(
            candidate_pool.proposal_ids, "candidate pool proposal IDs"
        )
        manifest_digest = _digest(
            selection_manifest.content_digest, "selection manifest digest"
        )
        manifest_envelope_digest = _digest(
            selection_manifest.manifest_envelope_digest,
            "selection manifest envelope digest",
        )
        manifest_pool_digest = _digest(
            selection_manifest.candidate_pool_digest,
            "selection manifest candidate pool digest",
        )
        raw_selections = tuple(selection_manifest.selections)
    except (AttributeError, TypeError) as exc:
        raise SelectionReplayError(
            "candidate pool or selection manifest lacks the replay view contract"
        ) from exc
    if manifest_pool_digest != pool_digest:
        raise SelectionReplayError(
            "selection manifest references a different candidate pool"
        )
    if not raw_selections:
        raise SelectionReplayError(
            "selection manifest contains no blind selector decisions"
        )
    selected_set: set[str] = set()
    selection_snapshot: list[tuple[str | None, bool]] = []
    proposal_set = set(proposal_ids)
    for index, selection in enumerate(raw_selections):
        try:
            selected_id = selection.selected_proposal_id
            abstained = selection.abstained
        except AttributeError as exc:
            raise SelectionReplayError(
                f"selection entry {index} lacks the replay view contract"
            ) from exc
        if type(abstained) is not bool:
            raise SelectionReplayError(
                f"selection entry {index} abstention flag must be boolean"
            )
        if abstained:
            if selected_id is not None:
                raise SelectionReplayError(
                    f"selection entry {index} abstained but retained a proposal ID"
                )
        else:
            _proposal_id(selected_id, f"selection entry {index} proposal ID")
            assert isinstance(selected_id, str)
            if selected_id not in proposal_set:
                raise SelectionReplayError(
                    f"selection entry {index} references an unknown proposal"
                )
            selected_set.add(selected_id)
        selection_snapshot.append((selected_id, abstained))
    selected = tuple(item for item in proposal_ids if item in selected_set)
    remainder = tuple(item for item in proposal_ids if item not in selected_set)
    inventory = ReplayPhaseInventory(
        proposal_ids=proposal_ids,
        selected_proposal_ids=selected,
        remainder_proposal_ids=remainder,
        candidate_pool_digest=pool_digest,
        selection_manifest_digest=manifest_digest,
        selection_manifest_envelope_digest=manifest_envelope_digest,
    )
    pool_snapshot: tuple[object, ...] = (pool_digest, proposal_ids)
    manifest_snapshot: tuple[object, ...] = (
        manifest_digest,
        manifest_envelope_digest,
        manifest_pool_digest,
        tuple(selection_snapshot),
    )
    return inventory, pool_snapshot, manifest_snapshot


def _validate_inner_evidence_binding(
    evidence: object,
    *,
    proposal: CorruptedActionProposal,
    source_dataset_id: str,
    evaluation_seed: int,
    attempt_ordinal: int,
    inner: ProposalEvaluator,
) -> None:
    if not isinstance(evidence, EvaluationEvidence):
        raise SelectionReplayError("inner evaluator did not return EvaluationEvidence")
    validate_evaluation_evidence(evidence)
    expected = {
        "proposal_id": proposal.proposal_id,
        "source_dataset_id": source_dataset_id,
        "source_episode_id": proposal.source_episode_id,
        "source_candidate_id": proposal.source_candidate_id,
        "split_group_id": proposal.split_group_id,
        "evaluator_id": inner.evaluator_id,
        "evaluator_version": inner.evaluator_version,
        "evaluator_configuration_digest": inner.configuration_digest,
        "evaluation_seed": evaluation_seed,
        "attempt_ordinal": attempt_ordinal,
    }
    mismatches = [
        field for field, value in expected.items() if getattr(evidence, field) != value
    ]
    if mismatches:
        raise SelectionReplayError(
            "inner evidence identity mismatch: " + ", ".join(mismatches)
        )


def _validate_m3c_strong_replay_evidence(
    evidence: EvaluationEvidence,
    proposal: CorruptedActionProposal,
) -> None:
    """Require conclusive strong simulator proof and complete M3C replay gates."""
    if (
        evidence.status is not EvaluationStatus.CONCLUSIVE
        or evidence.label_source is not LabelSource.SIMULATOR
        or evidence.label_strength is not LabelStrength.STRONG
        or evidence.simulator_replay_verified is not True
    ):
        raise SelectionReplayError(
            "M3C executable proposal did not produce strong simulator evidence"
        )
    expected_steps = int(proposal.transformed_action.actions.shape[0])
    if expected_steps < M3C_CANDIDATE_HORIZON:
        raise SelectionReplayError(
            "M3C replay action is shorter than the fixed candidate horizon"
        )
    for metric in (
        "replay_baseline_requested_steps",
        "replay_baseline_steps",
        "replay_corrupted_requested_steps",
        "replay_corrupted_steps",
    ):
        if evidence.metrics.get(metric) != expected_steps:
            raise SelectionReplayError(
                f"M3C strong evidence has incomplete action count for {metric}"
            )
    if evidence.replayed_control_steps != expected_steps * 2:
        raise SelectionReplayError(
            "M3C strong evidence replayed-control-step count is incomplete"
        )
    if evidence.metrics.get("replay_baseline_valid") is not True:
        raise SelectionReplayError(
            "M3C strong evidence lacks a successful source baseline"
        )
    if (
        evidence.metrics.get("pickcube_candidate_horizon_steps")
        != M3C_CANDIDATE_HORIZON
        or evidence.metrics.get("pickcube_prefix_evaluated_before_continuation")
        is not True
    ):
        raise SelectionReplayError(
            "M3C strong evidence lacks the fixed candidate/continuation boundary"
        )
    for role in ("baseline", "corrupted"):
        if (
            evidence.metrics.get(f"replay_{role}_restoration_complete_state_comparison")
            is not True
            or evidence.metrics.get(
                f"replay_{role}_restoration_compared_component_count"
            )
            != M3C_STATE_COMPONENT_COUNT
        ):
            raise SelectionReplayError(
                f"M3C {role} restoration did not compare all 70 state components"
            )
        error = evidence.metrics.get(
            f"replay_{role}_restoration_maximum_absolute_error"
        )
        if type(error) not in (int, float):
            raise SelectionReplayError(
                f"M3C {role} restoration exceeds the fixed state tolerance"
            )
        numeric_error = float(cast(int | float, error))
        if (
            not math.isfinite(numeric_error)
            or numeric_error < 0.0
            or numeric_error > M3C_STATE_RESTORATION_TOLERANCE
        ):
            raise SelectionReplayError(
                f"M3C {role} restoration exceeds the fixed state tolerance"
            )


def _validate_joined_strong_evidence(evidence: EvaluationEvidence) -> None:
    validate_evaluation_evidence(evidence)
    if (
        evidence.label_source is not LabelSource.SIMULATOR
        or evidence.label_strength is not LabelStrength.STRONG
        or evidence.simulator_replay_verified is not True
    ):
        raise SelectionReplayError(
            "joined replay outcome is not strong simulator-verified evidence"
        )
    for role in ("baseline", "corrupted"):
        if (
            evidence.metrics.get(f"replay_{role}_restoration_complete_state_comparison")
            is not True
            or evidence.metrics.get(
                f"replay_{role}_restoration_compared_component_count"
            )
            != M3C_STATE_COMPONENT_COUNT
        ):
            raise SelectionReplayError(
                "joined replay evidence lacks complete 70-component restoration"
            )


def _phase_evidence(
    dataset: EvaluationDataset,
    *,
    inventory: ReplayPhaseInventory,
    phase: ReplayPhase,
) -> Mapping[str, EvaluationEvidence]:
    if not isinstance(dataset, EvaluationDataset):
        raise SelectionReplayError("phase output must be EvaluationDataset")
    if tuple(dataset.selected_proposal_ids) != inventory.proposal_ids:
        raise SelectionReplayError(
            "phase ledger does not cover the complete unchanged candidate pool"
        )
    configuration = dataset.resolved_evaluator_configuration
    expected_executable = inventory.executable_ids(phase)
    if (
        dataset.evaluator_id != MANIFEST_GATED_REPLAY_EVALUATOR_ID
        or dataset.evaluator_version != MANIFEST_GATED_REPLAY_EVALUATOR_VERSION
        or configuration.get("phase") != phase.value
        or configuration.get("candidate_pool_digest") != inventory.candidate_pool_digest
        or configuration.get("selection_manifest_digest")
        != inventory.selection_manifest_digest
        or configuration.get("selection_manifest_envelope_digest")
        != inventory.selection_manifest_envelope_digest
        or _proposal_ids(
            configuration.get("executable_proposal_ids", ()),
            "phase executable proposal IDs",
            allow_empty=True,
        )
        != expected_executable
    ):
        raise SelectionReplayError(
            f"{phase.value} replay dataset configuration is not manifest-bound"
        )
    if len(dataset.ledger) != len(inventory.proposal_ids):
        raise SelectionReplayError(
            f"{phase.value} replay contains retries or duplicate ledger entries"
        )
    if len(dataset.evidence) != len(inventory.proposal_ids):
        raise SelectionReplayError(
            f"{phase.value} replay evidence does not cover every pool proposal"
        )
    ledger_by_id = {entry.proposal_id: entry for entry in dataset.ledger}
    evidence_by_id = {item.proposal_id: item for item in dataset.evidence}
    if (
        len(ledger_by_id) != len(dataset.ledger)
        or len(evidence_by_id) != len(dataset.evidence)
        or set(ledger_by_id) != set(inventory.proposal_ids)
        or set(evidence_by_id) != set(inventory.proposal_ids)
    ):
        raise SelectionReplayError(
            f"{phase.value} replay contains unknown or duplicate proposal evidence"
        )
    for proposal_id in inventory.proposal_ids:
        ledger = ledger_by_id[proposal_id]
        evidence = evidence_by_id[proposal_id]
        if (
            ledger.attempt_ordinal != 0
            or ledger.evidence_id != evidence.evidence_id
            or ledger.state
            in {LedgerState.PENDING, LedgerState.RUNNING, LedgerState.EXECUTION_ERROR}
        ):
            raise SelectionReplayError(
                f"{phase.value} replay proposal {proposal_id!r} is incomplete"
            )
    return MappingProxyType(evidence_by_id)


def _proposal_id(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SelectionReplayError(f"{context} must be canonical non-empty text")
    return value


def _proposal_ids(
    values: object,
    context: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise SelectionReplayError(f"{context} must be an ordered sequence")
    result = tuple(
        _proposal_id(item, f"{context}[{index}]") for index, item in enumerate(values)
    )
    if not result and not allow_empty:
        raise SelectionReplayError(f"{context} must not be empty")
    if len(result) != len(set(result)):
        raise SelectionReplayError(f"{context} contains duplicate proposal IDs")
    return result


def _dataset_resume_snapshot(dataset: EvaluationDataset) -> tuple[object, ...]:
    """Return semantic run content without relying on identity-only model equality."""
    evidence = tuple(
        (
            item.evidence_id,
            item.proposal_id,
            item.source_dataset_id,
            item.source_episode_id,
            item.source_candidate_id,
            item.split_group_id,
            item.evaluator_id,
            item.evaluator_version,
            item.evaluator_configuration_digest,
            item.evaluation_seed,
            item.attempt_ordinal,
            item.status,
            item.success,
            item.progress_before,
            item.progress_after,
            item.progress_delta,
            item.unsafe,
            tuple(
                (
                    event.failure_type,
                    event.timestamp_s,
                    event.probability,
                    event.description,
                    event.schema_version,
                )
                for event in item.failure_events
            ),
            item.termination_reason,
            item.replayed_control_steps,
            tuple(sorted(item.metrics.items())),
            item.artifact_references,
            item.label_source,
            item.label_strength,
            item.simulator_replay_verified,
            item.notes,
            item.schema_version,
        )
        for item in dataset.evidence
    )
    return (
        dataset.source_corruption_dataset_digest,
        dataset.source_dataset_id,
        dataset.evaluator_id,
        dataset.evaluator_version,
        compute_configuration_digest(dataset.resolved_evaluator_configuration),
        dataset.evaluator_configuration_digest,
        dataset.run_id,
        dataset.base_seed,
        dataset.max_proposals,
        dataset.selected_proposal_ids,
        evidence,
        dataset.ledger,
        dataset.summary,
        dataset.artifact_references,
        dataset.run_state,
        dataset.run_manifest,
        dataset.schema_version,
    )


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SelectionReplayError(f"{context} must be a lowercase sha256 digest")
    return value


def _semantic_evidence_payload(evidence: EvaluationEvidence) -> Mapping[str, object]:
    """Return complete task evidence without operational archive identity.

    Evidence IDs and evaluator-configuration digests transitively bind the
    timestamped selection envelope.  Evaluation seeds are derived from that
    configuration digest, while attempt ordinals describe ledger mechanics.
    Artifact references and notes are likewise operational diagnostics.  The
    task result, full strong-replay metrics, source identity, and evaluator
    semantic identity remain bound here.
    """
    validate_evaluation_evidence(evidence)
    return {
        "evaluator_id": evidence.evaluator_id,
        "evaluator_version": evidence.evaluator_version,
        "failure_events": [
            {
                "description": item.description,
                "failure_type": item.failure_type,
                "probability": item.probability,
                "schema_version": item.schema_version,
                "timestamp_s": item.timestamp_s,
            }
            for item in evidence.failure_events
        ],
        "label_source": (
            evidence.label_source.value if evidence.label_source is not None else None
        ),
        "label_strength": (
            evidence.label_strength.value
            if evidence.label_strength is not None
            else None
        ),
        "metrics": dict(evidence.metrics),
        "progress_after": evidence.progress_after,
        "progress_before": evidence.progress_before,
        "progress_delta": evidence.progress_delta,
        "proposal_id": evidence.proposal_id,
        "replayed_control_steps": evidence.replayed_control_steps,
        "schema_version": evidence.schema_version,
        "simulator_replay_verified": evidence.simulator_replay_verified,
        "source_candidate_id": evidence.source_candidate_id,
        "source_dataset_id": evidence.source_dataset_id,
        "source_episode_id": evidence.source_episode_id,
        "split_group_id": evidence.split_group_id,
        "status": evidence.status.value,
        "success": evidence.success,
        "termination_reason": evidence.termination_reason,
        "unsafe": evidence.unsafe,
    }


def _content_digest(value: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise SelectionReplayError(
            "replay identity content is not canonical JSON"
        ) from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "MANIFEST_GATED_REPLAY_EVALUATOR_ID",
    "MANIFEST_GATED_REPLAY_EVALUATOR_VERSION",
    "BlindSelectionManifestReplayView",
    "BlindSelectionReplayEntryView",
    "CandidatePoolReplayView",
    "CompletePoolReplayJoin",
    "ManifestGatedReplayEvaluator",
    "ReplayPhase",
    "ReplayPhaseEvaluators",
    "ReplayPhaseInventory",
    "SelectionReplayError",
    "ZeroWorkResumeReport",
    "build_manifest_gated_replay_evaluators",
    "join_complementary_replay_phases",
    "verify_zero_work_resume",
]
