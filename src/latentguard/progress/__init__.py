"""Stage-aware progress contracts and evaluation utilities."""

from latentguard.progress.dataset import (
    EpisodeAssignment,
    ProgressWindow,
    assign_episode_split,
    build_progress_windows,
)
from latentguard.progress.gates import LG_R2GateInput, evaluate_lg_r2_gate
from latentguard.progress.metrics import (
    ProgressMetrics,
    StageMetrics,
    evaluate_progress,
    evaluate_stage_predictions,
)
from latentguard.progress.models import StageLabel, compose_progress
from latentguard.progress.rollout import (
    RolloutJob,
    action_step_record,
    build_rollout_jobs,
    completed_episode_ids,
    validate_seed_isolation,
    write_completion_marker,
)
from latentguard.progress.stages.base import TaskStageAdapter
from latentguard.progress.stages.libero_registry import LiberoStageRegistry

__all__ = [
    "EpisodeAssignment",
    "LG_R2GateInput",
    "LiberoStageRegistry",
    "ProgressMetrics",
    "ProgressWindow",
    "RolloutJob",
    "StageLabel",
    "StageMetrics",
    "TaskStageAdapter",
    "action_step_record",
    "assign_episode_split",
    "build_progress_windows",
    "build_rollout_jobs",
    "completed_episode_ids",
    "compose_progress",
    "evaluate_lg_r2_gate",
    "evaluate_progress",
    "evaluate_stage_predictions",
    "validate_seed_isolation",
    "write_completion_marker",
]
