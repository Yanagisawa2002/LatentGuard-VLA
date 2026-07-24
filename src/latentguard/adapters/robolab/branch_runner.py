"""Gate aggregation for faithful replay and deterministic takeover evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchGateInput:
    """Complete compact counts consumed by the LG-RB1 promotion gate."""

    valid_recorded_episodes: int
    faithful_initial_restore_failures: int
    faithful_per_step_failures: int
    faithful_terminal_mismatches: int
    faithful_success_mismatches: int
    prefix_mismatches: int
    branch_mismatches: int
    isolation_mismatches: int
    semantic_coverage_complete: bool
    official_stack_valid: bool
    source_validation_passed: bool
    remote_audit_passed: bool


def evaluate_lg_rb1_gate(evidence: BranchGateInput) -> dict[str, Any]:
    """Classify Result A/B/C without treating a partial gate as success."""
    faithful_pass = (
        evidence.valid_recorded_episodes >= 10
        and evidence.faithful_initial_restore_failures == 0
        and evidence.faithful_per_step_failures == 0
        and evidence.faithful_terminal_mismatches == 0
        and evidence.faithful_success_mismatches == 0
    )
    takeover_pass = (
        evidence.prefix_mismatches == 0
        and evidence.branch_mismatches == 0
        and evidence.isolation_mismatches == 0
        and evidence.semantic_coverage_complete
    )
    provenance_pass = (
        evidence.official_stack_valid
        and evidence.source_validation_passed
        and evidence.remote_audit_passed
    )
    authorized = faithful_pass and takeover_pass and provenance_pass
    if authorized:
        result = "A"
        definition = (
            "official faithful replay and deterministic intermediate takeover passed"
        )
    elif faithful_pass:
        result = "B"
        definition = (
            "official replay passed, but safe deterministic takeover "
            "was not established"
        )
    else:
        result = "C"
        definition = (
            "official faithful replay was not deterministic under the frozen stack"
        )
    return {
        "schema_version": "lg_rb0_gate_v1",
        "status": "pass",
        "result": result,
        "result_definition": definition,
        "LG_RB1_AUTHORIZED": authorized,
        "faithful_replay_gate": faithful_pass,
        "takeover_gate": takeover_pass,
        "provenance_gate": provenance_pass,
        "no_policy_or_ranker_training": True,
        "no_candidate_selection_or_intervention": True,
        "final_seeds_accessed": False,
    }
