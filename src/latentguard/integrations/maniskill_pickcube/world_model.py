"""Explicit real PickCube adapter for the simulator-independent WM-v0 collector."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.control.models import EpisodeState
from latentguard.control.runner import BoundSourcePlanV1, RuntimeBoundaryV1
from latentguard.world_model.collector import (
    CollectionFrame,
    ExactRestoreReport,
    StepResult,
)

from .closed_loop import PickCubeClosedLoopRuntime

PICKCUBE_WM_EVENT_NAMES = (
    "grasped",
    "dropped",
    "wrong_object_contact",
    "collision",
    "object_displacement",
    "workspace_violation",
    "timeout",
    "task_success",
)


@dataclass(frozen=True, slots=True)
class PickCubeWorldModelAnchor:
    """One prepared exact boundary and its owning source plan."""

    anchor_id: str
    source: BoundSourcePlanV1
    state_reference: str
    state_digest: str


class PickCubeWorldModelAdapter:
    """Collect PickCube futures through accepted M4C restore/render mechanics."""

    def __init__(
        self,
        runtime: PickCubeClosedLoopRuntime,
        anchors: tuple[PickCubeWorldModelAnchor, ...],
        *,
        visual_domain: str,
        displacement_threshold: float = 1e-4,
    ) -> None:
        """Bind fixed anchors and one predeclared visual domain."""

        if not anchors:
            raise ValueError("PickCube WM adapter requires at least one anchor")
        self._runtime = runtime
        self._anchors = {anchor.anchor_id: anchor for anchor in anchors}
        if len(self._anchors) != len(anchors):
            raise ValueError("PickCube WM anchors must have unique IDs")
        self._visual_domain = visual_domain
        self._displacement_threshold = displacement_threshold
        self._boundary: RuntimeBoundaryV1 | None = None
        self._task: dict[str, object] | None = None
        self._anchor_goal_distance: float | None = None
        self._previous_grasped: bool | None = None

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Return the accepted fixed M4A/M4C camera order."""

        return ("front_oblique", "overhead", "side_oblique")

    @property
    def event_names(self) -> tuple[str, ...]:
        """Return all required event channels in fixed order."""

        return PICKCUBE_WM_EVENT_NAMES

    def restore(self, anchor_id: str) -> ExactRestoreReport:
        """Restore one content-bound state in a fresh verified session."""

        try:
            anchor = self._anchors[anchor_id]
        except KeyError as exc:
            raise ValueError(f"unknown PickCube WM anchor {anchor_id!r}") from exc
        self._boundary = self._runtime.restore_boundary(
            anchor.source,
            state_reference=anchor.state_reference,
            expected_state_digest=anchor.state_digest,
            visual_domain=self._visual_domain,
        )
        self._boundary, self._task = self._runtime.current_world_model_observation()
        self._anchor_goal_distance = _float_task(self._task, "cube_to_goal_distance")
        self._previous_grasped = _bool_task(self._task, "is_grasped")
        component_count, maximum_error = self._runtime.last_restore_verification
        return ExactRestoreReport(
            anchor_id=anchor_id,
            content_identity=anchor.state_digest,
            verified=True,
            compared_component_count=component_count,
            maximum_absolute_error=maximum_error,
        )

    def observe(self) -> CollectionFrame:
        """Return current verified images, proprioception, progress, and events."""

        if self._boundary is None or self._task is None:
            raise RuntimeError("PickCube WM adapter has no restored boundary")
        if self._boundary.images is None:
            raise RuntimeError("PickCube WM visual capture is unavailable")
        grasped = _bool_task(self._task, "is_grasped")
        goal_distance = _float_task(self._task, "cube_to_goal_distance")
        dropped = bool(self._previous_grasped is True and not grasped)
        displaced = bool(
            self._anchor_goal_distance is not None
            and abs(goal_distance - self._anchor_goal_distance)
            > self._displacement_threshold
        )
        self._previous_grasped = grasped
        return CollectionFrame(
            observations=np.asarray(self._boundary.images, dtype=np.uint8),
            proprio=np.asarray(self._boundary.verifier_state, dtype=np.float32),
            progress=_pickcube_progress(self._task),
            events={
                "grasped": grasped,
                "dropped": dropped,
                "wrong_object_contact": None,
                "collision": None,
                "object_displacement": displaced,
                "workspace_violation": None,
                "timeout": None,
                "task_success": _bool_task(self._task, "success"),
            },
        )

    def step(self, action: NDArray[np.float32]) -> StepResult:
        """Execute one exact row and update the verified observation boundary."""

        values = np.asarray(action, dtype=np.float32)
        if values.shape != (8,) or not bool(np.isfinite(values).all()):
            raise ValueError("PickCube WM action must be finite float32 [8]")
        result = self._runtime.execute(
            np.asarray(values[None, :], dtype=np.dtype("<f8")),
            maximum_control_steps_remaining=1,
            visual_domain=self._visual_domain,
        )
        self._boundary = result.next_boundary
        self._boundary, self._task = self._runtime.current_world_model_observation()
        terminal = result.outcome in {
            EpisodeState.SUCCESS,
            EpisodeState.TASK_FAILURE,
            EpisodeState.UNSAFE,
        }
        return StepResult(
            terminated=terminal,
            truncated=False,
            success=(result.outcome is EpisodeState.SUCCESS if terminal else None),
        )

    def close(self) -> None:
        """Close the owned runtime session."""

        self._runtime.close()
        self._boundary = None
        self._task = None


def _bool_task(task: Mapping[str, object], name: str) -> bool:
    value = task.get(name)
    if type(value) is not bool:
        raise RuntimeError(f"PickCube WM task field {name} is not boolean")
    return value


def _float_task(task: Mapping[str, object], name: str) -> float:
    value = task.get(name)
    if type(value) is not float or not np.isfinite(value):
        raise RuntimeError(f"PickCube WM task field {name} is not finite float")
    return value


def _pickcube_progress(task: Mapping[str, object]) -> float:
    """Return task-specific nonterminal geometric progress, never terminal copying."""

    if _bool_task(task, "is_obj_placed"):
        return 1.0 if _bool_task(task, "is_robot_static") else 0.8
    if _bool_task(task, "is_grasped"):
        distance = _float_task(task, "cube_to_goal_distance")
        return 0.5 + 0.25 * (1.0 - min(distance / 0.5, 1.0))
    distance = _float_task(task, "tcp_to_cube_distance")
    return 0.25 * (1.0 - min(distance / 0.5, 1.0))


def build_pickcube_world_model_adapter(
    config: dict[str, Any],
) -> PickCubeWorldModelAdapter:
    """Build a real adapter from explicit paths in one resolved run config.

    This is the factory for `collect_wm_v0.py --adapter-factory`. Runtime paths
    stay outside semantic identities and private values must not be committed.
    """

    from latentguard.control.source_plans import load_source_plans
    from latentguard.integrations.maniskill_pickcube.closed_loop import (
        PickCubeVisualBoundaryObserver,
    )
    from latentguard.integrations.maniskill_pickcube.closed_loop_state_store import (
        ClosedLoopStateStore,
    )
    from latentguard.integrations.maniskill_pickcube.compatibility import (
        load_compatibility_report,
        validate_compatibility_report,
    )
    from latentguard.integrations.maniskill_pickcube.configuration import (
        load_expected_contract,
        load_maniskill_pickcube_action_layout,
        validate_maniskill_pickcube_action_layout_binding,
    )
    from latentguard.integrations.maniskill_pickcube.serialization import (
        action_contract_from_compatibility,
        environment_settings_from_compatibility,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.task_evidence import (
        PickCubeTaskKeyContract,
    )
    from latentguard.integrations.maniskill_pickcube.visual_rendering import (
        build_pickcube_visual_render_plan,
    )
    from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
    from latentguard.vision_data.configuration import (
        load_camera_rig_configuration,
        load_render_domain_configuration,
    )
    from latentguard.vision_data.domains import RenderDomainConfigurationV1

    runtime_value = config.get("pickcube_runtime")
    if not isinstance(runtime_value, dict):
        raise ValueError("collection config requires pickcube_runtime mapping")
    runtime: dict[str, Any] = runtime_value
    required_paths = (
        "source_archive_dir",
        "source_plan_manifest",
        "expected_contract",
        "compatibility_report",
        "action_layout",
        "camera_rig_config",
        "render_domain_config",
        "state_store",
    )
    if any(not isinstance(runtime.get(name), str) for name in required_paths):
        raise ValueError("PickCube runtime path inventory is incomplete")
    archive = load_state_indexed_archive(Path(runtime["source_archive_dir"]))
    sources = load_source_plans(Path(runtime["source_plan_manifest"]), archive=archive)
    report = load_compatibility_report(Path(runtime["compatibility_report"]))
    expected = load_expected_contract(Path(runtime["expected_contract"]))
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(Path(runtime["action_layout"]))
    validate_maniskill_pickcube_action_layout_binding(
        layout, binding, layout.m1_action_layout
    )
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding, coordinate_frame=layout.coordinate_frame
    )
    task_keys = PickCubeTaskKeyContract.from_compatibility_report(report)
    rig = cast(
        PickCubeMultiViewRigV1,
        load_camera_rig_configuration(Path(runtime["camera_rig_config"])),
    )
    domains = cast(
        RenderDomainConfigurationV1,
        load_render_domain_configuration(Path(runtime["render_domain_config"])),
    )
    visual_domain = str(runtime.get("visual_domain", "canonical"))
    render_seed = int(runtime.get("render_seed", config["seed"]))
    plan = build_pickcube_visual_render_plan(
        rig, domains.domain(visual_domain), domains, render_seed
    )
    state_store = ClosedLoopStateStore(Path(runtime["state_store"]))
    source_by_id = {source.identity.source_trajectory_id: source for source in sources}
    episode_by_id = {
        episode.source_trajectory_id: episode for episode in archive.episodes
    }
    raw_anchors = runtime.get("anchors")
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise ValueError("PickCube runtime requires a non-empty anchors list")
    anchors: list[PickCubeWorldModelAnchor] = []
    for raw_anchor in raw_anchors:
        if not isinstance(raw_anchor, dict):
            raise ValueError("PickCube WM anchor must be a mapping")
        trajectory_id = str(raw_anchor["source_trajectory_id"])
        state_index = int(raw_anchor["state_index"])
        episode = episode_by_id[trajectory_id]
        if not 0 <= state_index < len(episode.states):
            raise ValueError("PickCube WM anchor state index is out of range")
        state = episode.states[state_index]
        reference, digest = state_store.save(state.tree)
        if digest != state.state_digest:
            raise RuntimeError("PickCube WM stored anchor digest differs from archive")
        anchors.append(
            PickCubeWorldModelAnchor(
                anchor_id=str(raw_anchor["anchor_id"]),
                source=source_by_id[trajectory_id],
                state_reference=reference,
                state_digest=digest,
            )
        )
    observer = PickCubeVisualBoundaryObserver(
        plans={visual_domain: plan},
        key_contract=task_keys,
        state_tolerance=settings.state_tolerance,
    )
    collection_runtime = PickCubeClosedLoopRuntime(
        source_archive=archive,
        settings=settings,
        action_contract=action_contract,
        key_contract=task_keys,
        state_store=state_store,
        visual_observer=observer,
    )
    return PickCubeWorldModelAdapter(
        collection_runtime,
        tuple(anchors),
        visual_domain=visual_domain,
    )


__all__ = [
    "PICKCUBE_WM_EVENT_NAMES",
    "PickCubeWorldModelAdapter",
    "PickCubeWorldModelAnchor",
    "build_pickcube_world_model_adapter",
]
