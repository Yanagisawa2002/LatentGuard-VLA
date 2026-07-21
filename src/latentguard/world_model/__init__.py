"""Action-conditioned latent world-model contracts and utilities."""

from .candidates import (
    CandidateContext,
    CandidateOrigin,
    CandidateProvider,
    CandidateRejection,
    CandidateRejectionCode,
    CandidateValidationError,
    PolicyCandidate,
    candidate_set_digest,
    checkpoint_sha256,
    validate_and_deduplicate_candidates,
    verify_checkpoint_hash,
)
from .collector import (
    ActionCandidate,
    CollectionFrame,
    CounterfactualCollector,
    ExactRestoreReport,
    StepResult,
    WorldModelCollectionAdapter,
)
from .d1_collection import (
    D1AnchorJob,
    D1CollectionResult,
    D1CollectionRunner,
    D1RejectionCode,
    EpisodePhase,
    build_d1_reports,
    load_or_build_d1_reports,
)
from .data_schema import (
    CandidateSource,
    DatasetSplit,
    WorldModelSample,
    WorldModelSchemaError,
)
from .manifest import DatasetGate, WorldModelDatasetManifest, build_dataset_manifest

__all__ = [
    "ActionCandidate",
    "CandidateContext",
    "CandidateOrigin",
    "CandidateProvider",
    "CandidateRejection",
    "CandidateRejectionCode",
    "CandidateSource",
    "CandidateValidationError",
    "CollectionFrame",
    "CounterfactualCollector",
    "DatasetGate",
    "DatasetSplit",
    "D1AnchorJob",
    "D1CollectionResult",
    "D1CollectionRunner",
    "D1RejectionCode",
    "EpisodePhase",
    "ExactRestoreReport",
    "StepResult",
    "WorldModelCollectionAdapter",
    "WorldModelDatasetManifest",
    "WorldModelSample",
    "WorldModelSchemaError",
    "build_dataset_manifest",
    "build_d1_reports",
    "candidate_set_digest",
    "checkpoint_sha256",
    "PolicyCandidate",
    "load_or_build_d1_reports",
    "validate_and_deduplicate_candidates",
    "verify_checkpoint_hash",
]
