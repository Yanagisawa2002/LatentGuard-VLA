"""RoboLab replay contracts with no import-time RoboLab dependency."""

from latentguard.adapters.robolab.branch_runner import (
    BranchGateInput,
    evaluate_lg_rb1_gate,
)
from latentguard.adapters.robolab.replay_adapter import (
    ReplayAdapter,
    ReplayContract,
    ReplaySnapshot,
)
from latentguard.adapters.robolab.state_schema import (
    StateComparison,
    compare_state_trees,
    state_tree_sha256,
)

__all__ = [
    "BranchGateInput",
    "ReplayAdapter",
    "ReplayContract",
    "ReplaySnapshot",
    "StateComparison",
    "compare_state_trees",
    "evaluate_lg_rb1_gate",
    "state_tree_sha256",
]
