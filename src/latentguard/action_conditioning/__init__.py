"""Contracts and evaluation controls for numeric action conditioning."""

from latentguard.action_conditioning.contracts import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    DEPLOYABLE_FEATURES,
    ActionContract,
    validate_deployable_features,
)
from latentguard.action_conditioning.gates import evaluate_lg_r2b_gate
from latentguard.action_conditioning.splits import (
    EpisodeRecord,
    assign_grouped_folds,
    validate_grouped_folds,
)

__all__ = [
    "ACTION_DIMENSION",
    "ACTION_HORIZON",
    "DEPLOYABLE_FEATURES",
    "ActionContract",
    "EpisodeRecord",
    "assign_grouped_folds",
    "evaluate_lg_r2b_gate",
    "validate_deployable_features",
    "validate_grouped_folds",
]
