"""Task-stage adapter registry."""

from latentguard.progress.stages.base import TaskStageAdapter
from latentguard.progress.stages.libero_registry import LiberoStageRegistry

__all__ = ["LiberoStageRegistry", "TaskStageAdapter"]
