"""Simulator-independent evidence, evaluators, and resumable execution."""

from latentguard.evaluation.base import (
    ApplicabilityDecision,
    ProposalEvaluator,
)
from latentguard.evaluation.fixture import (
    DETERMINISTIC_FIXTURE_EVALUATOR_ID,
    DeterministicFixtureEvaluator,
    create_deterministic_fixture_evaluator,
)
from latentguard.evaluation.models import (
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_ID_PREFIX,
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.evaluation.registry import (
    EvaluatorRegistry,
    create_evaluator,
    default_evaluator_registry,
    load_evaluator_configuration,
)
from latentguard.evaluation.runner import (
    EvaluationPlan,
    EvaluationRunResult,
    derive_evaluation_seed,
    plan_evaluation,
    run_evaluation,
)
from latentguard.evaluation.serialization import (
    EvaluationDataset,
    EvaluationSummary,
    LedgerEntry,
    LedgerState,
    RunState,
    compute_corruption_dataset_content_digest,
    compute_corruption_dataset_digest,
    load_evaluation_dataset,
    save_evaluation_dataset,
    update_evaluation_dataset,
    validate_evaluation_dataset,
    validate_evaluation_update,
)
from latentguard.evaluation.validation import (
    evidence_to_outcome_label,
    validate_evaluation_evidence,
)

__all__ = [
    "DETERMINISTIC_FIXTURE_EVALUATOR_ID",
    "EVALUATION_EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_ID_PREFIX",
    "ApplicabilityDecision",
    "DeterministicFixtureEvaluator",
    "EvaluationDataset",
    "EvaluationEvidence",
    "EvaluationPlan",
    "EvaluationRunResult",
    "EvaluationStatus",
    "EvaluationSummary",
    "EvaluatorRegistry",
    "LedgerEntry",
    "LedgerState",
    "ProposalEvaluator",
    "RunState",
    "compute_configuration_digest",
    "compute_corruption_dataset_content_digest",
    "compute_corruption_dataset_digest",
    "compute_evidence_identifier",
    "create_deterministic_fixture_evaluator",
    "create_evaluator",
    "default_evaluator_registry",
    "derive_evaluation_seed",
    "evidence_to_outcome_label",
    "load_evaluation_dataset",
    "load_evaluator_configuration",
    "plan_evaluation",
    "run_evaluation",
    "save_evaluation_dataset",
    "update_evaluation_dataset",
    "validate_evaluation_dataset",
    "validate_evaluation_update",
    "validate_evaluation_evidence",
]
