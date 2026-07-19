"""Import-safe protocol for future LangMani adapters."""

from __future__ import annotations

from typing import Protocol

from latentguard.integrations.langmani.models import (
    LangManiActionProposalV1,
    LangManiObservationEnvelopeV1,
    LangManiOutcomeEvidenceV1,
    LangManiPolicyBindingV1,
    LangManiReplaySnapshotV1,
    LangManiTaskContextV1,
)


class LangManiAdapter(Protocol):
    """Optional adapter interface with no import-time LangMani dependency."""

    def task_context(self) -> LangManiTaskContextV1:
        """Return the active task's content-bound identity."""

    def policy_binding(self) -> LangManiPolicyBindingV1:
        """Return the frozen policy and processor identity."""

    def observation(self) -> LangManiObservationEnvelopeV1:
        """Return role-separated inputs at the current boundary."""

    def propose(self) -> LangManiActionProposalV1:
        """Return a raw policy proposal."""

    def executable_proposal(
        self, proposal: LangManiActionProposalV1
    ) -> LangManiActionProposalV1:
        """Return a separately identified executable proposal."""

    def snapshot(self) -> LangManiReplaySnapshotV1:
        """Capture simulator and policy state references."""

    def restore(self, snapshot: LangManiReplaySnapshotV1) -> None:
        """Restore a previously captured boundary."""

    def evaluate_outcome(
        self, proposal: LangManiActionProposalV1
    ) -> LangManiOutcomeEvidenceV1:
        """Execute and classify one proposal without boolean collapse."""
