"""RoboLab replay contracts with no import-time RoboLab dependency."""

from latentguard.adapters.robolab.branch_runner import (
    BranchGateInput,
    evaluate_lg_rb1_gate,
)
from latentguard.adapters.robolab.overlay_report import (
    OverlaySkipCategory,
    OverlaySkipReport,
    classify_overlay_skips,
)
from latentguard.adapters.robolab.replay_adapter import (
    ReplayAdapter,
    ReplayContract,
    ReplaySnapshot,
)
from latentguard.adapters.robolab.state_schema import (
    StateCanonicalization,
    StateComparison,
    canonicalize_optional_empty_mappings,
    compare_state_trees,
    state_tree_sha256,
)

__all__ = [
    "BranchGateInput",
    "OverlaySkipCategory",
    "OverlaySkipReport",
    "ReplayAdapter",
    "ReplayContract",
    "ReplaySnapshot",
    "StateCanonicalization",
    "StateComparison",
    "canonicalize_optional_empty_mappings",
    "classify_overlay_skips",
    "compare_state_trees",
    "evaluate_lg_rb1_gate",
    "state_tree_sha256",
]
