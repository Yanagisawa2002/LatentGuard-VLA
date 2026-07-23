"""Fail-closed LG-R2 data authorization gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LG_R2GateInput:
    """Reviewed LG-R1 evidence required before failure-head work."""

    valid_rollout_episodes: int
    tasks: int
    suites: int
    natural_failed_episodes: int
    failure_windows: int
    successful_windows: int
    stage_annotation_passed: bool
    sarm_evaluation_complete: bool
    episode_split_leakage: int
    processor_identity_completeness: float
    checkpoint_identity_completeness: float


def evaluate_lg_r2_gate(evidence: LG_R2GateInput) -> dict[str, Any]:
    """Return every gate decision and a conservative authorization bit."""

    checks = {
        "valid_rollout_episodes": evidence.valid_rollout_episodes >= 100,
        "tasks": evidence.tasks >= 4,
        "suites": evidence.suites >= 2,
        "natural_failed_episodes": evidence.natural_failed_episodes >= 10,
        "failure_windows": evidence.failure_windows >= 100,
        "successful_windows": evidence.successful_windows >= 100,
        "stage_annotation_gate": evidence.stage_annotation_passed,
        "sarm_evaluation": evidence.sarm_evaluation_complete,
        "episode_split_leakage": evidence.episode_split_leakage == 0,
        "processor_identity": evidence.processor_identity_completeness == 1.0,
        "checkpoint_identity": evidence.checkpoint_identity_completeness == 1.0,
    }
    authorized = all(checks.values())
    return {
        "schema_version": "latentguard.lg_r1.lg_r2_gate.v1",
        "input": asdict(evidence),
        "checks": checks,
        "LG_R2_AUTHORIZED": authorized,
        "status": "pass" if authorized else "FAILURE_DATA_GATE_NOT_MET",
    }
