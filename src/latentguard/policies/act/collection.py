"""State-preserving native-expert capture for PickCube ACT demonstrations."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
    PickCubeEpisodeEndedError,
    RecordingEnvironmentProxy,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
    build_pickcube_task_evidence,
)
from latentguard.integrations.maniskill_pickcube.visual_rendering import (
    LazyManiSkillPickCubeVisualRenderer,
    PickCubeVisualRenderHandle,
    PickCubeVisualRenderPlan,
)
from latentguard.policies.act.contracts import PICKCUBE_ACTIVE_JOINT_NAMES
from latentguard.policies.act.data import PickCubeDemoEpisode
from latentguard.policies.experts import (
    PickCubeExpertPhaseTracker,
    PickCubeMotionPlanningExpert,
)
from latentguard.replay.identity import canonical_json_bytes
from latentguard.replay.models import TerminalTaskStatus

_SAPIEN_VULKAN_FALLBACK_WARNING = (
    r"^Failed to find Vulkan ICD file\. This is probably due to an incorrect or "
    r"partial installation of the NVIDIA driver\. SAPIEN will attempt to provide "
    r"an ICD file anyway but it may not work\.$"
)


class PickCubeDemoCollectionError(RuntimeError):
    """Raised when a demonstration attempt cannot produce accepted training data."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeDemoCollectionError(f"{context}: {reason}")


def validate_expert_gate_authorization(value: Mapping[str, object]) -> str:
    """Validate the complete collection gate and return its content digest."""
    if value.get("schema_version") != "pickcube-act-expert-evaluation-v1":
        _fail("expert gate", "schema mismatch")
    episode_count = value.get("episode_count")
    success_count = value.get("success_count")
    success_rate = value.get("success_rate")
    episodes = value.get("episodes")
    checks = value.get("gate_checks")
    if type(episode_count) is not int or episode_count < 100:
        _fail("expert gate", "requires at least 100 independent episodes")
    if type(success_count) is not int or not 0 <= success_count <= episode_count:
        _fail("expert gate", "success count is invalid")
    if type(success_rate) not in (int, float):
        _fail("expert gate", "success rate is not numeric")
    success_rate_value = float(cast(int | float, success_rate))
    if (
        not math.isfinite(success_rate_value)
        or success_rate_value != success_count / episode_count
        or success_rate_value < 0.95
    ):
        _fail("expert gate", "success rate is below the fixed 0.95 threshold")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        _fail("expert gate", "episode evidence is missing")
    if len(episodes) != episode_count:
        _fail("expert gate", "episode inventory count differs")
    seeds: list[int] = []
    for index, raw_episode in enumerate(episodes):
        if not isinstance(raw_episode, Mapping):
            _fail("expert gate", f"episode {index} is not an object")
        seed = raw_episode.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32:
            _fail("expert gate", f"episode {index} seed is invalid")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        _fail("expert gate", "episode reset seeds are not independent")
    required_checks = {
        "episode_count_at_least_100",
        "every_declared_phase_executed",
        "nonfinite_action_zero",
        "simulator_error_zero",
        "success_rate_at_least_95_percent",
        "workspace_violation_zero",
    }
    if (
        not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or any(checks[name] is not True for name in required_checks)
    ):
        _fail("expert gate", "one or more mandatory checks failed")
    if value.get("authorized_for_demonstration_collection") is not True:
        _fail("expert gate", "collection authorization is false")
    digest = hashlib.sha256(
        canonical_json_bytes(value, context="PickCubeExpertGateAuthorization")
    ).hexdigest()
    return f"sha256:{digest}"


@dataclass(slots=True)
class PickCubePreActionCapture:
    """Capture one fixed RGB/state/action sample before every expert action."""

    factory: LazyManiSkillSourceEnvironmentFactory
    renderer: LazyManiSkillPickCubeVisualRenderer
    render_plan: PickCubeVisualRenderPlan
    phase_tracker: PickCubeExpertPhaseTracker
    _handle: PickCubeVisualRenderHandle | None = field(default=None, init=False)
    _rgb: list[NDArray[np.uint8]] = field(default_factory=list, init=False)
    _states: list[NDArray[np.float32]] = field(default_factory=list, init=False)
    _actions: list[NDArray[Any]] = field(default_factory=list, init=False)
    _phases: list[str] = field(default_factory=list, init=False)
    _camera_configuration_digest: str | None = field(default=None, init=False)

    def capture_boundary(self, environment: object, step_index: int) -> None:
        """Prepare cameras after reset and count each executed phase action."""
        self.phase_tracker.capture_boundary(environment, step_index)
        if step_index == 0:
            if self._handle is not None or self._rgb:
                _fail("pre-action capture", "reset boundary was observed twice")
            self._handle = self.renderer.prepare(environment, self.render_plan)

    def capture_pre_action(
        self,
        environment: object,
        step_index: int,
        action: NDArray[Any],
    ) -> None:
        """Render and detach the allowlisted observation before one action."""
        if self._handle is None:
            _fail("pre-action capture", "renderer was not prepared after reset")
        if step_index != len(self._actions):
            _fail("pre-action capture", "action ordinal is discontinuous")
        phase = self.phase_tracker.current_phase
        if phase is None:
            _fail("pre-action capture", "expert phase is unavailable")
        views = self._handle.render_views()
        by_id = {view.camera_id: view for view in views}
        if set(by_id) != {"front_oblique", "overhead", "side_oblique"}:
            _fail("pre-action capture", "rendered camera inventory changed")
        front = by_id["front_oblique"]
        names, state = self.factory.extract_named_robot_state(environment)
        if names != PICKCUBE_ACTIVE_JOINT_NAMES:
            _fail("pre-action capture", "active joint order changed")
        state_array = np.asarray(state)
        if state_array.dtype != np.dtype(np.float32) or state_array.shape != (18,):
            _fail("pre-action capture", "proprioception must be float32[18]")
        action_array = np.asarray(action)
        if (
            action_array.shape != (8,)
            or action_array.dtype.hasobject
            or not np.issubdtype(action_array.dtype, np.floating)
            or not np.all(np.isfinite(action_array))
        ):
            _fail("pre-action capture", "expert action must be finite floating[8]")
        digest = front.camera_configuration_digest
        if self._camera_configuration_digest not in (None, digest):
            _fail("pre-action capture", "camera configuration changed within episode")
        self._camera_configuration_digest = digest
        self._rgb.append(np.array(front.rgb, copy=True, order="C"))
        self._states.append(np.array(state_array, copy=True, order="C"))
        self._actions.append(np.array(action_array, copy=True, order="C"))
        self._phases.append(phase.value)

    def build_episode(
        self,
        *,
        scene_seed: int,
        recorded_actions: Sequence[NDArray[Any]],
        compatibility_identity: str,
        contract_digest: str,
    ) -> PickCubeDemoEpisode:
        """Finalize one aligned, complete, successful demonstration."""
        if not self._actions or self._camera_configuration_digest is None:
            _fail("pre-action capture", "episode contains no captured actions")
        if len(recorded_actions) != len(self._actions):
            _fail("pre-action capture", "recorded and captured action counts differ")
        for index, (captured, recorded) in enumerate(
            zip(self._actions, recorded_actions, strict=True)
        ):
            observed = np.asarray(recorded)
            if (
                captured.dtype != observed.dtype
                or captured.shape != observed.shape
                or captured.tobytes(order="C") != observed.tobytes(order="C")
            ):
                _fail("pre-action capture", f"action {index} content changed")
        return PickCubeDemoEpisode(
            scene_seed=scene_seed,
            rgb=np.stack(self._rgb, axis=0),
            proprioception=np.stack(self._states, axis=0),
            actions=np.stack(self._actions, axis=0),
            phases=tuple(self._phases),
            compatibility_identity=compatibility_identity,
            contract_digest=contract_digest,
            camera_configuration_digest=self._camera_configuration_digest,
        )


def collect_native_demo_episode(
    *,
    seed: int,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
    render_plan: PickCubeVisualRenderPlan,
    compatibility_identity: str,
    contract_digest: str,
    expert: PickCubeMotionPlanningExpert,
    factory: LazyManiSkillSourceEnvironmentFactory | None = None,
    renderer: LazyManiSkillPickCubeVisualRenderer | None = None,
) -> PickCubeDemoEpisode:
    """Execute and capture one independently reset successful native episode."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        _fail("collect_native_demo_episode", "seed must be uint32")
    source_factory = factory or LazyManiSkillSourceEnvironmentFactory()
    visual_renderer = renderer or LazyManiSkillPickCubeVisualRenderer()
    tracker = PickCubeExpertPhaseTracker()
    capture = PickCubePreActionCapture(
        factory=source_factory,
        renderer=visual_renderer,
        render_plan=render_plan,
        phase_tracker=tracker,
    )
    environment: object | None = None
    recorder: RecordingEnvironmentProxy | None = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_SAPIEN_VULKAN_FALLBACK_WARNING,
                category=UserWarning,
                module=r"^sapien\._vulkan_tricks$",
            )
            environment = source_factory.create_environment(
                settings,
                action_contract,
                purpose="official_source_generation",
            )
        recorder = RecordingEnvironmentProxy(
            environment,
            action_contract,
            trajectory_action_limit=50,
            boundary_capture=capture.capture_boundary,
            pre_action_capture=capture.capture_pre_action,
            stop_on_episode_end=True,
        )
        try:
            expert.solve(recorder, seed=seed, phase_tracker=tracker)
        except PickCubeEpisodeEndedError:
            pass
        recorder.verify_interception_complete()
        snapshot = source_factory.capture_task_snapshot(environment, key_contract)
        evidence = build_pickcube_task_evidence(snapshot, key_contract)
        if not (
            evidence.status is TerminalTaskStatus.COMPLETE
            and evidence.success is True
            and evidence.unsafe is False
        ):
            _fail("collect_native_demo_episode", "official terminal success is false")
        tracker.require_complete()
        return capture.build_episode(
            scene_seed=seed,
            recorded_actions=cast(Sequence[NDArray[Any]], recorder.actions),
            compatibility_identity=compatibility_identity,
            contract_digest=contract_digest,
        )
    finally:
        if recorder is not None:
            recorder.close()
        elif environment is not None:
            close = getattr(environment, "close", None)
            if callable(close):
                close()


def expert_gate_summary(value: Mapping[str, object]) -> Mapping[str, object]:
    """Return a compact collection-safe projection of an authorized gate."""
    digest = validate_expert_gate_authorization(value)
    return MappingProxyType(
        {
            "authorized_for_demonstration_collection": True,
            "episode_count": value["episode_count"],
            "expert_gate_digest": digest,
            "failure_taxonomy": value.get("failure_taxonomy", {}),
            "success_count": value["success_count"],
            "success_rate": value["success_rate"],
        }
    )


__all__ = [
    "PickCubeDemoCollectionError",
    "PickCubePreActionCapture",
    "collect_native_demo_episode",
    "expert_gate_summary",
    "validate_expert_gate_authorization",
]
