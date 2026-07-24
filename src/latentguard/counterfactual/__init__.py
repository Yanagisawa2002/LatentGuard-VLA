"""Contracts and analysis helpers for real same-state counterfactual evidence."""

from latentguard.counterfactual.candidates import (
    CandidateDiversityThresholds,
    action_content_sha256,
    classify_candidate_group,
    pairwise_action_metrics,
)
from latentguard.counterfactual.gates import (
    CounterfactualGateEvidence,
    evaluate_counterfactual_gate,
    validate_anchor_registry,
)
from latentguard.counterfactual.models import (
    CandidateDisposition,
    CounterfactualBranch,
    RealPolicyCandidate,
    Recoverability,
    TerminalOutcome,
)

__all__ = [
    "CandidateDisposition",
    "CandidateDiversityThresholds",
    "CounterfactualBranch",
    "CounterfactualGateEvidence",
    "RealPolicyCandidate",
    "Recoverability",
    "TerminalOutcome",
    "action_content_sha256",
    "classify_candidate_group",
    "evaluate_counterfactual_gate",
    "pairwise_action_metrics",
    "validate_anchor_registry",
]
