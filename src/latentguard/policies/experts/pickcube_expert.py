"""Phase-instrumented native PickCube motion-planning expert."""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import textwrap
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn


class PickCubeExpertError(RuntimeError):
    """Raised when the privileged expert or its evidence is invalid."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeExpertError(f"{context}: {reason}")


class PickCubeExpertPhase(StrEnum):
    """Physical stages actually commanded by the native solver."""

    REACH_PREGRASP = "REACH_PREGRASP"
    DESCEND_TO_GRASP = "DESCEND_TO_GRASP"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    TRANSPORT_TO_GOAL = "TRANSPORT_TO_GOAL"


PICKCUBE_EXPERT_PHASES: tuple[PickCubeExpertPhase, ...] = tuple(PickCubeExpertPhase)
_COMPONENT_POSE_WARNING = (
    r"^component\.pose can be ambiguous thus deprecated\. It is equivalent to "
    r"component\.entity_pose, which should be used instead$"
)
_COMPONENT_POSE_WARNING_MODULE = r"^mani_skill\.utils\.geometry\.trimesh_utils$"


@dataclass(slots=True)
class PickCubeExpertPhaseTracker:
    """Record the expert phase attached to every executed control action."""

    current_phase: PickCubeExpertPhase | None = None
    transitions: list[PickCubeExpertPhase] = field(default_factory=list)
    action_counts: dict[PickCubeExpertPhase, int] = field(default_factory=dict)

    def transition(self, phase: PickCubeExpertPhase) -> None:
        """Enter one explicit solver phase before executing its actions."""
        if not isinstance(phase, PickCubeExpertPhase):
            _fail("PickCubeExpertPhaseTracker.transition", "invalid phase")
        if self.current_phase is not phase:
            self.transitions.append(phase)
            self.current_phase = phase

    def capture_boundary(self, environment: object, step_index: int) -> None:
        """Count one real environment action after its boundary is captured."""
        del environment
        if type(step_index) is not int or step_index < 0:
            _fail("PickCubeExpertPhaseTracker.capture_boundary", "invalid step index")
        if step_index == 0:
            return
        if self.current_phase is None:
            _fail(
                "PickCubeExpertPhaseTracker.capture_boundary",
                "executed action has no declared expert phase",
            )
        self.action_counts[self.current_phase] = (
            self.action_counts.get(self.current_phase, 0) + 1
        )

    def require_complete(self) -> None:
        """Require every declared phase to have executed at least one action."""
        missing = [
            phase.value
            for phase in PICKCUBE_EXPERT_PHASES
            if not self.action_counts.get(phase)
        ]
        if missing:
            _fail(
                "PickCubeExpertPhaseTracker", f"phases executed no actions: {missing}"
            )
        expected = tuple(PICKCUBE_EXPERT_PHASES)
        if tuple(self.transitions) != expected:
            _fail(
                "PickCubeExpertPhaseTracker",
                f"phase order mismatch: expected {[p.value for p in expected]}, "
                f"observed {[p.value for p in self.transitions]}",
            )

    def counts_by_name(self) -> Mapping[str, int]:
        """Return all phases in fixed order, including zero counts."""
        return MappingProxyType(
            {
                phase.value: self.action_counts.get(phase, 0)
                for phase in PICKCUBE_EXPERT_PHASES
            }
        )


@dataclass(frozen=True, slots=True)
class PickCubeMotionPlanningExpert:
    """Project-owned instrumentation of ManiSkill's native Panda solver.

    The expert reads privileged cube and goal poses only to generate
    demonstrations. It controls the task solely through environment actions and
    never exposes privileged values to the learned policy input.
    """

    finger_length_m: float = 0.025

    def __post_init__(self) -> None:
        if (
            type(self.finger_length_m) is not float
            or not math.isfinite(self.finger_length_m)
            or self.finger_length_m <= 0.0
        ):
            _fail("PickCubeMotionPlanningExpert", "finger length must be positive")

    def solve(
        self,
        environment: object,
        *,
        seed: int,
        phase_tracker: PickCubeExpertPhaseTracker,
    ) -> object:
        """Execute the native solver sequence using only reset and action steps."""
        if type(seed) is not int or not 0 <= seed < 2**32:
            _fail("PickCubeMotionPlanningExpert.solve", "seed must be uint32")
        if not isinstance(phase_tracker, PickCubeExpertPhaseTracker):
            _fail("PickCubeMotionPlanningExpert.solve", "phase tracker is required")
        try:
            import numpy as np
            import sapien  # type: ignore[import-not-found]
            from mani_skill.examples.motionplanning.base_motionplanner.utils import (  # type: ignore[import-not-found]
                compute_grasp_info_by_obb,
                get_actor_obb,
            )
            from mani_skill.examples.motionplanning.panda.motionplanner import (  # type: ignore[import-not-found]
                PandaArmMotionPlanningSolver,
            )
        except Exception as exc:
            raise PickCubeExpertError(
                "PickCubeMotionPlanningExpert.solve: ManiSkill planner is unavailable"
            ) from exc

        reset = getattr(environment, "reset", None)
        if not callable(reset):
            _fail("PickCubeMotionPlanningExpert.solve", "environment lacks reset")
        reset(seed=seed)
        base = getattr(environment, "unwrapped", environment)
        agent = getattr(base, "agent", None)
        robot = getattr(agent, "robot", None)
        robot_pose = getattr(robot, "pose", None)
        if agent is None or robot is None or robot_pose is None:
            _fail("PickCubeMotionPlanningExpert.solve", "Panda runtime is incomplete")
        planner = PandaArmMotionPlanningSolver(
            environment,
            debug=False,
            vis=False,
            base_pose=robot_pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        try:
            cube = getattr(base, "cube", None)
            goal_site = getattr(base, "goal_site", None)
            tcp = getattr(agent, "tcp", None)
            if cube is None or goal_site is None or tcp is None:
                _fail(
                    "PickCubeMotionPlanningExpert.solve", "task geometry is unavailable"
                )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=_COMPONENT_POSE_WARNING,
                    category=DeprecationWarning,
                    module=_COMPONENT_POSE_WARNING_MODULE,
                )
                obb = get_actor_obb(cube)
            approaching = np.array([0.0, 0.0, -1.0])
            target_closing = tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
            grasp_info = compute_grasp_info_by_obb(
                obb,
                approaching=approaching,
                target_closing=target_closing,
                depth=self.finger_length_m,
            )
            grasp_pose = agent.build_grasp_pose(
                approaching,
                grasp_info["closing"],
                cube.pose.sp.p,
            )

            phase_tracker.transition(PickCubeExpertPhase.REACH_PREGRASP)
            reach_pose = grasp_pose * sapien.Pose([0.0, 0.0, -0.05])
            planner.move_to_pose_with_screw(reach_pose)

            phase_tracker.transition(PickCubeExpertPhase.DESCEND_TO_GRASP)
            planner.move_to_pose_with_screw(grasp_pose)

            phase_tracker.transition(PickCubeExpertPhase.CLOSE_GRIPPER)
            planner.close_gripper()

            phase_tracker.transition(PickCubeExpertPhase.TRANSPORT_TO_GOAL)
            goal_pose = sapien.Pose(goal_site.pose.sp.p, grasp_pose.q)
            return planner.move_to_pose_with_screw(goal_pose)
        finally:
            planner.close()


def assert_pickcube_expert_has_no_teleport_calls() -> str:
    """Statically reject state mutation and task-success fabrication calls."""
    source = textwrap.dedent(inspect.getsource(PickCubeMotionPlanningExpert.solve))
    tree = ast.parse(source)
    prohibited = {
        "set_pose",
        "set_state",
        "set_state_dict",
        "_initialize_episode",
        "set_success",
        "teleport",
    }
    observed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            observed.add(node.func.attr)
    found = sorted(observed & prohibited)
    if found:
        _fail("PickCubeMotionPlanningExpert.source", f"prohibited calls: {found}")
    return f"sha256:{hashlib.sha256(source.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ExpertEpisodeAudit:
    """Compact sanitized result for one independent expert reset seed."""

    seed: int
    success: bool
    action_count: int
    phase_action_counts: Mapping[str, int]
    simulator_error: bool = False
    nonfinite_action_count: int = 0
    action_out_of_bounds_count: int = 0
    workspace_violation_count: int = 0
    failure_category: str | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            _fail("ExpertEpisodeAudit.seed", "expected uint32")
        if type(self.success) is not bool or type(self.simulator_error) is not bool:
            _fail("ExpertEpisodeAudit", "success and simulator_error must be boolean")
        for name in (
            "action_count",
            "nonfinite_action_count",
            "action_out_of_bounds_count",
            "workspace_violation_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                _fail(f"ExpertEpisodeAudit.{name}", "expected non-negative integer")
        expected_phases = {phase.value for phase in PICKCUBE_EXPERT_PHASES}
        if set(self.phase_action_counts) != expected_phases or any(
            type(value) is not int or value < 0
            for value in self.phase_action_counts.values()
        ):
            _fail("ExpertEpisodeAudit.phase_action_counts", "invalid phase inventory")
        if self.success and (
            self.simulator_error
            or self.failure_category is not None
            or self.nonfinite_action_count
            or self.action_out_of_bounds_count
            or self.workspace_violation_count
        ):
            _fail("ExpertEpisodeAudit", "successful episode retains failure evidence")
        if self.failure_category is not None and (
            not self.failure_category
            or any(
                not (char.isalnum() or char in "_.-") for char in self.failure_category
            )
        ):
            _fail("ExpertEpisodeAudit.failure_category", "expected safe category")

    def to_mapping(self) -> dict[str, object]:
        """Return one stack-free JSON-native episode result."""
        return {
            "action_count": self.action_count,
            "action_out_of_bounds_count": self.action_out_of_bounds_count,
            "failure_category": self.failure_category,
            "nonfinite_action_count": self.nonfinite_action_count,
            "phase_action_counts": dict(self.phase_action_counts),
            "seed": self.seed,
            "simulator_error": self.simulator_error,
            "success": self.success,
            "workspace_violation_count": self.workspace_violation_count,
        }


@dataclass(frozen=True, slots=True)
class ExpertEvaluationSummary:
    """Fixed 100-seed quality decision for the PickCube expert."""

    episodes: tuple[ExpertEpisodeAudit, ...]
    source_audit_digest: str
    minimum_episode_count: int = 100
    minimum_success_rate: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", tuple(self.episodes))
        if not self.episodes:
            _fail("ExpertEvaluationSummary", "episodes must not be empty")
        if (
            type(self.minimum_episode_count) is not int
            or self.minimum_episode_count < 100
        ):
            _fail("ExpertEvaluationSummary", "minimum episode count must be >= 100")
        if self.minimum_success_rate != 0.95:
            _fail("ExpertEvaluationSummary", "minimum success rate must equal 0.95")
        if (
            not isinstance(self.source_audit_digest, str)
            or not self.source_audit_digest.startswith("sha256:")
            or len(self.source_audit_digest) != 71
        ):
            _fail("ExpertEvaluationSummary", "invalid source audit digest")

    @property
    def success_count(self) -> int:
        return len([episode for episode in self.episodes if episode.success])

    @property
    def success_rate(self) -> float:
        return self.success_count / len(self.episodes)

    @property
    def failure_taxonomy(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for episode in self.episodes:
            if episode.failure_category is not None:
                counts[episode.failure_category] = (
                    counts.get(episode.failure_category, 0) + 1
                )
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def gate_checks(self) -> Mapping[str, bool]:
        phase_totals = self._phase_totals()
        return MappingProxyType(
            {
                "episode_count_at_least_100": len(self.episodes)
                >= self.minimum_episode_count,
                "every_declared_phase_executed": all(phase_totals.values()),
                "nonfinite_action_zero": sum(
                    item.nonfinite_action_count for item in self.episodes
                )
                == 0,
                "simulator_error_zero": not any(
                    item.simulator_error for item in self.episodes
                ),
                "success_rate_at_least_95_percent": self.success_rate
                >= self.minimum_success_rate,
                "workspace_violation_zero": sum(
                    item.workspace_violation_count for item in self.episodes
                )
                == 0,
            }
        )

    @property
    def authorized_for_demonstration_collection(self) -> bool:
        return all(self.gate_checks.values())

    def to_mapping(self) -> dict[str, object]:
        """Return the complete compact expert quality report."""
        phase_totals = self._phase_totals()
        checks = dict(self.gate_checks)
        return {
            "authorized_for_demonstration_collection": all(checks.values()),
            "episode_count": len(self.episodes),
            "episodes": [episode.to_mapping() for episode in self.episodes],
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "failure_count": len(self.episodes) - self.success_count,
            "failure_taxonomy": dict(self.failure_taxonomy),
            "gate_checks": checks,
            "minimum_episode_count": self.minimum_episode_count,
            "minimum_success_rate": self.minimum_success_rate,
            "phase_action_counts": phase_totals,
            "schema_version": "pickcube-act-expert-evaluation-v1",
            "source_audit_digest": self.source_audit_digest,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
        }

    def _phase_totals(self) -> dict[str, int]:
        totals = {phase.value: 0 for phase in PICKCUBE_EXPERT_PHASES}
        for episode in self.episodes:
            for phase in PICKCUBE_EXPERT_PHASES:
                totals[phase.value] += episode.phase_action_counts[phase.value]
        return totals


def summarize_expert_evaluation(
    episodes: Sequence[ExpertEpisodeAudit], *, source_audit_digest: str
) -> ExpertEvaluationSummary:
    """Build one ordered 100-seed report without hiding failed attempts."""
    values = tuple(episodes)
    if not values:
        _fail("ExpertEvaluationSummary", "episodes must not be empty")
    seeds = tuple(item.seed for item in values)
    if len(set(seeds)) != len(seeds) or seeds != tuple(sorted(seeds)):
        _fail("ExpertEvaluationSummary", "seeds must be unique and sorted")
    if (
        not isinstance(source_audit_digest, str)
        or not source_audit_digest.startswith("sha256:")
        or len(source_audit_digest) != 71
    ):
        _fail("ExpertEvaluationSummary", "invalid source audit digest")
    return ExpertEvaluationSummary(values, source_audit_digest)


def module_file_sha256() -> str:
    """Return the exact file-byte identity used by remote expert runs."""
    return f"sha256:{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"
