from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
)
from latentguard.integrations.maniskill_pickcube.state_tree import clone_state_tree
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PickCubeVerifierStateV1,
    build_pickcube_verifier_state_v1,
)
from latentguard.integrations.maniskill_pickcube.visual_probe import (
    ManiSkillVisualProbeError,
    SessionPickCubeVisualProbeRuntime,
    load_visual_compatibility_report,
    probe_maniskill_pickcube_visual,
)
from latentguard.integrations.maniskill_pickcube.visual_rendering import (
    EXTRINSIC_4X4_SEMANTIC,
    GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC,
    UINT8_COLOR_TO_RGB_UINT8_SEMANTIC,
    LazyManiSkillPickCubeVisualRenderer,
    ManiSkillVisualRenderingError,
    PickCubeVisualRenderPlan,
    RenderedVisualView,
    VisualRendererApiObservation,
    build_pickcube_visual_render_plan,
)
from latentguard.integrations.maniskill_pickcube.visual_session import (
    ManiSkillVisualSessionError,
    PickCubeVisualSession,
    build_visual_observation_packet,
)
from latentguard.replay.models import ReplayExecutionRole
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.configuration import (
    load_camera_rig_configuration,
    load_render_domain_configuration,
)
from latentguard.vision_data.domains import RenderDomainConfigurationV1
from latentguard.vision_data.models import SourceCollection, VisualDatasetSplit

_COMPATIBILITY = f"sha256:{'a' * 64}"
_CAMERA_CONFIG = Path("configs/vision/m4a/camera-rig-v1.json")
_DOMAIN_CONFIG = Path("configs/vision/m4a/render-domains-v1.json")
_JOINT_NAMES = tuple(f"panda_joint_{index}" for index in range(9))
_QPOS = np.linspace(-0.4, 0.4, 9, dtype=np.float32)
_QVEL = np.linspace(0.04, -0.04, 9, dtype=np.float32)
_TCP = np.array([0.0, 0.0, 0.1], dtype=np.float32)
_CUBE = np.array([0.1, 0.2, 0.3], dtype=np.float32)
_GOAL = np.array([0.4, 0.5, 0.6], dtype=np.float32)


def _task(*, terminal: bool = False) -> PickCubeTaskSnapshotV1:
    return PickCubeTaskSnapshotV1(
        success=terminal,
        is_obj_placed=terminal,
        is_robot_static=terminal,
        is_grasped=False,
        cube_center_z=float(_CUBE[2]),
        cube_to_goal_distance=float(np.linalg.norm(_CUBE - _GOAL)),
        tcp_to_cube_distance=float(np.linalg.norm(_TCP - _CUBE)),
    )


def _verifier(*, terminal: bool = False) -> PickCubeVerifierStateV1:
    return build_pickcube_verifier_state_v1(
        joint_names=_JOINT_NAMES,
        qpos=_QPOS,
        qvel=_QVEL,
        tcp_position=_TCP,
        tcp_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        cube_position=_CUBE,
        cube_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        goal_position=_GOAL,
        is_grasped=False,
        is_obj_placed=terminal,
        is_robot_static=terminal,
    )


def _source() -> tuple[PickCubeStateIndexedEpisodeV1, PickCubeIndexedStateV1]:
    state = PickCubeIndexedStateV1(
        state_index=0,
        source_action_index=0,
        tree={"complete_state": np.arange(70, dtype=np.float32)},
        task_snapshot=_task(),
        restored_task_snapshot=_task(),
        verifier_state=_verifier(),
        seed=17,
        compatibility_identity=_COMPATIBILITY,
        source_trajectory_id="mspc-visual-trajectory-a",
    )
    terminal = PickCubeIndexedStateV1(
        state_index=1,
        source_action_index=1,
        tree={"complete_state": np.arange(70, dtype=np.float32) + 1.0},
        task_snapshot=_task(terminal=True),
        restored_task_snapshot=_task(terminal=True),
        verifier_state=_verifier(terminal=True),
        seed=17,
        compatibility_identity=_COMPATIBILITY,
        source_trajectory_id="mspc-visual-trajectory-a",
    )
    episode = PickCubeStateIndexedEpisodeV1(
        episode_id="mspc-visual-episode-a",
        source_trajectory_id=state.source_trajectory_id,
        source_policy_identity="maniskill/official-pickcube-solver-v1",
        seed=state.seed,
        compatibility_identity=_COMPATIBILITY,
        source_actions=np.zeros((1, 8), dtype=np.float32),
        states=(state, terminal),
    )
    return episode, state


class _Pose:
    def __init__(self, position: NDArray[np.float32], quaternion: NDArray[np.float32]):
        self.p = np.array(position, copy=True)
        self.q = np.array(quaternion, copy=True)


class _Environment:
    def __init__(self) -> None:
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.unwrapped = self
        self.state: object = {"complete_state": np.zeros(70, dtype=np.float32)}
        self.elapsed_steps = np.array([0], dtype=np.int64)
        self.agent = SimpleNamespace(tcp=SimpleNamespace(pose=_Pose(_TCP, identity)))
        self.cube = SimpleNamespace(pose=_Pose(_CUBE, identity))
        self.goal_site = SimpleNamespace(pose=_Pose(_GOAL, identity))

    def get_state_dict(self) -> object:
        return clone_state_tree(self.state)


@dataclass
class _FakeStateRuntime:
    close_error: bool = False
    step_calls: int = 0
    environment: _Environment | None = None
    created_environments: list[_Environment] = field(default_factory=list)

    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        execution_role: ReplayExecutionRole,
    ) -> object:
        del settings, action_contract, execution_role
        self.environment = _Environment()
        self.created_environments.append(self.environment)
        return self.environment

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        del environment
        return clone_state_tree(state_tree)

    def reset_environment(self, environment: object, *, seed: int) -> None:
        del environment
        assert seed == 17

    def set_state_dict(self, environment: object, state_tree: object) -> None:
        assert isinstance(environment, _Environment)
        environment.state = clone_state_tree(state_tree)

    def get_state_dict(self, environment: object) -> object:
        assert isinstance(environment, _Environment)
        return environment.get_state_dict()

    def step_action(
        self,
        environment: object,
        action: NDArray[Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        del environment, action, action_contract
        self.step_calls += 1
        raise AssertionError("visual rendering must never execute an action")

    def capture_task_snapshot(
        self, environment: object, key_contract: PickCubeTaskKeyContract
    ) -> RawPickCubeTaskSnapshot:
        return _FakeProjectionFactory().capture_task_snapshot(environment, key_contract)

    def close_environment(self, environment: object) -> None:
        del environment
        if self.close_error:
            raise RuntimeError("synthetic close failure")


class _FakeProjectionFactory:
    def create_environment(
        self,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        *,
        purpose: str,
    ) -> object:
        del settings, action_contract, purpose
        return _Environment()

    def prepare_state_tree(self, environment: object, state_tree: object) -> object:
        del environment
        return clone_state_tree(state_tree)

    def capture_task_snapshot(
        self, environment: object, key_contract: PickCubeTaskKeyContract
    ) -> RawPickCubeTaskSnapshot:
        del environment
        return RawPickCubeTaskSnapshot(
            evaluator_values={
                key_contract.success: np.array(False),
                key_contract.object_placed: np.array(False),
                key_contract.robot_static: np.array(False),
                key_contract.grasped: np.array(False),
            },
            cube_center_z=np.array(_CUBE[2]),
            cube_to_goal_distance=np.array(np.linalg.norm(_CUBE - _GOAL)),
        )

    def extract_named_robot_state(
        self, environment: object
    ) -> tuple[tuple[str, ...], NDArray[np.float32]]:
        del environment
        return _JOINT_NAMES, np.concatenate((_QPOS, _QVEL))

    def step_action(
        self,
        environment: object,
        action: NDArray[Any],
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        del environment, action, action_contract
        raise AssertionError("visual projection must never step")


@dataclass
class _FakeRenderHandle:
    environment: _Environment
    plan: PickCubeVisualRenderPlan
    nondeterministic: bool
    mutate_state: bool
    fail: bool
    fixed_drift: bool = False
    calls: int = 0

    @property
    def api_observation(self) -> VisualRendererApiObservation:
        return VisualRendererApiObservation(
            renderer_backend="fake_gpu",
            scene_type="fake.Scene",
            camera_type="fake.Camera",
            shader_configuration=self.plan.shader_configuration,
            camera_configuration_api="fake.add_camera",
            camera_group_initialization_semantic=(
                GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC
            ),
            camera_group_texture_names=("Color",),
            camera_group_count=3,
            underlying_camera_count_per_group=1,
            camera_groups_ready=True,
            world_camera_pose_representation="fake.Pose",
            sensor_update_calls=("fake.take_picture",),
            raw_color_texture_dtype="uint8",
            rgb_conversion_semantic=UINT8_COLOR_TO_RGB_UINT8_SEMANTIC,
            runtime_intrinsics_dtype="float64",
            runtime_extrinsics_dtype="float64",
            raw_extrinsic_matrix_shape="[4,4]",
            runtime_extrinsics_semantic=EXTRINSIC_4X4_SEMANTIC,
        )

    def render_views(self) -> tuple[RenderedVisualView, ...]:
        if self.fail:
            raise ManiSkillVisualRenderingError("synthetic renderer failure")
        self.calls += 1
        if self.mutate_state:
            tree = self.environment.state
            assert isinstance(tree, dict)
            tree["complete_state"][0] += 1.0
        results: list[RenderedVisualView] = []
        for index, camera in enumerate(self.plan.cameras):
            rgb = np.full((224, 224, 3), index + 1, dtype=np.uint8)
            if (
                self.fixed_drift or (self.nondeterministic and self.calls > 1)
            ) and index == 0:
                rgb[4, 5, 1] += 1
            results.append(
                RenderedVisualView(
                    camera_id=camera.camera_id,
                    rgb=rgb,
                    intrinsics=camera.intrinsics,
                    extrinsics=camera.extrinsics,
                    camera_configuration_digest=camera.camera_configuration_digest,
                )
            )
        return tuple(results)


@dataclass
class _FakeRenderer:
    nondeterministic: bool = False
    mutate_state: bool = False
    fail: bool = False

    def prepare(
        self, environment: object, plan: PickCubeVisualRenderPlan
    ) -> _FakeRenderHandle:
        assert isinstance(environment, _Environment)
        return _FakeRenderHandle(
            environment,
            plan,
            self.nondeterministic,
            self.mutate_state,
            self.fail,
        )


@dataclass
class _FreshEnvironmentDriftRenderer(_FakeRenderer):
    drift_fresh_environment_index: int = 1
    prepared_environment_count: int = 0

    def prepare(
        self, environment: object, plan: PickCubeVisualRenderPlan
    ) -> _FakeRenderHandle:
        assert isinstance(environment, _Environment)
        self.prepared_environment_count += 1
        return _FakeRenderHandle(
            environment,
            plan,
            self.nondeterministic,
            self.mutate_state,
            self.fail,
            fixed_drift=(
                self.prepared_environment_count
                == self.drift_fresh_environment_index + 1
            ),
        )


class _ElapsedReader:
    def elapsed_steps(self, environment: object) -> int:
        assert isinstance(environment, _Environment)
        return int(environment.elapsed_steps[0])


def _session(
    *,
    renderer: _FakeRenderer | None = None,
    runtime: _FakeStateRuntime | None = None,
    environment_initialized_observer: Callable[[], None] | None = None,
) -> tuple[PickCubeVisualSession, _FakeStateRuntime]:
    selected_runtime = runtime or _FakeStateRuntime()
    return (
        PickCubeVisualSession(
            settings=ManiSkillPickCubeEnvironmentSettings(
                obs_mode="state_dict", state_tolerance=1e-6
            ),
            action_contract=PickCubeReplayActionContract(
                total_dimension=8,
                environment_numpy_dtype="<f4",
                lower_bounds=np.full(8, -1.0, dtype=np.float32),
                upper_bounds=np.full(8, 1.0, dtype=np.float32),
                environment_shape=(1, 8),
                coordinate_frame="unspecified",
                control_period_s=0.05,
            ),
            task_key_contract=PickCubeTaskKeyContract(
                success="success",
                object_placed="is_obj_placed",
                robot_static="is_robot_static",
                grasped="is_grasped",
            ),
            state_runtime=selected_runtime,
            projection_factory=_FakeProjectionFactory(),
            renderer=renderer or _FakeRenderer(),
            elapsed_step_reader=_ElapsedReader(),
            environment_initialized_observer=environment_initialized_observer,
        ),
        selected_runtime,
    )


def _configuration() -> tuple[PickCubeMultiViewRigV1, RenderDomainConfigurationV1]:
    return (
        cast(
            PickCubeMultiViewRigV1,
            load_camera_rig_configuration(_CAMERA_CONFIG),
        ),
        cast(
            RenderDomainConfigurationV1,
            load_render_domain_configuration(_DOMAIN_CONFIG),
        ),
    )


@dataclass(frozen=True)
class _Operational:
    torch_version: str = "2.7.0"
    cuda_runtime_version: str = "12.8"
    gpu_model: str = "RTX5090"
    gpu_capability: str = "12.0"


@dataclass(frozen=True)
class _CompatibilityReport:
    compatibility_identity: str = _COMPATIBILITY
    mani_skill_version: str = "3.0.1"
    sapien_version: str = "3.0.0"
    operational: _Operational = _Operational()


@dataclass(frozen=True)
class _Binding:
    report: _CompatibilityReport = _CompatibilityReport()

    def require_trusted_replay_ready(self) -> None:
        return None


@dataclass(frozen=True)
class _TrustedVisual:
    visual_compatibility_identity: str
    pickcube_compatibility_identity: str
    camera_rig_digest: str
    render_domain_configuration_digest: str
    renderer_api: VisualRendererApiObservation

    def require_trusted_visual_generation_ready(self) -> None:
        return None


def test_render_plan_is_deterministic_and_lighting_keeps_camera_identity() -> None:
    rig, configuration = _configuration()
    base = tuple(camera.camera_configuration_digest for camera in rig.cameras)
    for domain in configuration.domains:
        first = build_pickcube_visual_render_plan(rig, domain, configuration, 29)
        second = build_pickcube_visual_render_plan(rig, domain, configuration, 29)
        first_digests = tuple(
            camera.camera_configuration_digest for camera in first.cameras
        )
        assert first_digests == tuple(
            camera.camera_configuration_digest for camera in second.cameras
        )
        if "camera" in domain.domain_id:
            assert first_digests != base
        else:
            assert first_digests == base
        assert first.lighting == second.lighting


@dataclass(frozen=True)
class _InstalledShaderConfig:
    shader_pack: str
    texture_names: MappingProxyType[str, object] = field(
        default_factory=lambda: MappingProxyType(
            {"Color": object(), "PositionSegmentation": object()}
        )
    )


class _InstalledPose:
    def __init__(self, position: object, quaternion: object) -> None:
        self.position = position
        self.quaternion = quaternion


class _InstalledCameraGroup:
    def take_picture(self) -> None:
        raise AssertionError("prepare must not capture an image")

    def get_picture_cuda(self, name: str) -> object:
        del name
        raise AssertionError("prepare must not fetch an image")


class _InstalledRenderSystemGroup:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls: list[tuple[list[object], list[str]]] = []
        self.groups: list[_InstalledCameraGroup] = []

    def create_camera_group(
        self, render_cameras: list[object], texture_names: list[str]
    ) -> _InstalledCameraGroup:
        self.calls.append((render_cameras, texture_names))
        if self.fail_at == len(self.calls):
            raise RuntimeError("synthetic camera-group failure")
        group = _InstalledCameraGroup()
        self.groups.append(group)
        return group


class _RejectingCameraGroupRegistry(dict[str, object]):
    def __setitem__(self, key: str, value: object) -> None:
        del key, value
        raise RuntimeError("synthetic camera-group registry failure")


class _InstalledScene:
    backend = "fake_gpu"
    gpu_sim_enabled = True
    parallel_in_single_scene = False
    num_envs = 1

    def __init__(
        self,
        *,
        update_error: bool = False,
        group_fail_at: int | None = None,
        reject_registry_write: bool = False,
    ) -> None:
        self.camera_calls: list[dict[str, object]] = []
        self.runtime_cameras: list[SimpleNamespace] = []
        self.ambient: object = None
        self.directional: object = None
        self.camera_groups: dict[str, object] = (
            _RejectingCameraGroupRegistry() if reject_registry_write else {}
        )
        self.render_system_group: _InstalledRenderSystemGroup | None = None
        self.update_error = update_error
        self.group_fail_at = group_fail_at
        self.events: list[str] = []

    def can_render(self) -> bool:
        return True

    def add_camera(self, **kwargs: object) -> object:
        self.camera_calls.append(kwargs)
        self.events.append(f"add_camera:{kwargs['name']}")
        camera = SimpleNamespace(_render_cameras=[object()], camera_group=None)
        self.runtime_cameras.append(camera)
        return camera

    def update_render(
        self, *, update_sensors: bool, update_human_render_cameras: bool
    ) -> None:
        assert update_sensors is False
        assert update_human_render_cameras is False
        self.events.append("update_render")
        if self.update_error:
            raise RuntimeError("synthetic update-render failure")
        if self.render_system_group is None:
            self.render_system_group = _InstalledRenderSystemGroup(
                fail_at=self.group_fail_at
            )

    def set_ambient_light(self, value: object) -> None:
        if self.render_system_group is not None:
            raise RuntimeError("batched render system forbids scene mutation")
        self.events.append("ambient")
        self.ambient = value

    def add_directional_light(self, **kwargs: object) -> None:
        if self.render_system_group is not None:
            raise RuntimeError("batched render system forbids scene mutation")
        self.events.append("directional")
        self.directional = kwargs


def _installed_renderer_modules(
    registry: object, *, render_system: object = "3.0"
) -> tuple[Callable[[str], object], list[object]]:
    applied: list[object] = []

    def module_importer(name: str) -> object:
        if name == "sapien":
            return SimpleNamespace(Pose=_InstalledPose)
        if name == "mani_skill.render":
            return SimpleNamespace(
                PREBUILT_SHADER_CONFIGS=registry,
                SAPIEN_RENDER_SYSTEM=render_system,
                ShaderConfig=_InstalledShaderConfig,
                set_shader_pack=applied.append,
            )
        raise AssertionError(f"unexpected module import {name!r}")

    return module_importer, applied


def test_installed_renderer_resolves_the_bound_prebuilt_shader_object() -> None:
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 30
    )
    shader_config = _InstalledShaderConfig(shader_pack="minimal")
    module_importer, applied = _installed_renderer_modules({"minimal": shader_config})
    scene = _InstalledScene()
    renderer = LazyManiSkillPickCubeVisualRenderer(module_importer=module_importer)

    handle = renderer.prepare(
        SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)), plan
    )

    assert len(applied) == 1
    assert applied[0] is shader_config
    assert handle.api_observation.shader_configuration == "minimal"
    assert (
        handle.api_observation.camera_group_initialization_semantic
        == GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC
    )
    assert handle.api_observation.camera_group_texture_names == (
        "Color",
        "PositionSegmentation",
    )
    assert handle.api_observation.camera_group_count == 3
    assert handle.api_observation.underlying_camera_count_per_group == 1
    assert handle.api_observation.camera_groups_ready
    assert len(scene.camera_calls) == 3
    assert scene.render_system_group is not None
    assert len(scene.render_system_group.calls) == 3
    for index, (render_cameras, texture_names) in enumerate(
        scene.render_system_group.calls
    ):
        assert render_cameras is scene.runtime_cameras[index]._render_cameras
        assert texture_names == list(shader_config.texture_names)
        assert (
            scene.runtime_cameras[index].camera_group
            is scene.render_system_group.groups[index]
        )
        camera_id = str(scene.camera_calls[index]["name"])
        assert scene.camera_groups[camera_id] is scene.render_system_group.groups[index]
    assert scene.events == [
        "add_camera:front_oblique",
        "add_camera:overhead",
        "add_camera:side_oblique",
        "ambient",
        "directional",
        "update_render",
    ]
    assert scene.ambient == [0.3, 0.3, 0.3]


@pytest.mark.parametrize(
    ("registry", "message"),
    (
        (None, "lacks the prebuilt shader"),
        ({}, "configured prebuilt shader is unavailable"),
        ({"minimal": "minimal"}, "unexpected type"),
    ),
)
def test_installed_renderer_rejects_unbound_prebuilt_shader_registries(
    registry: object, message: str
) -> None:
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 30
    )
    module_importer, _ = _installed_renderer_modules(registry)
    renderer = LazyManiSkillPickCubeVisualRenderer(module_importer=module_importer)
    scene = _InstalledScene()

    with pytest.raises(ManiSkillVisualRenderingError, match=message):
        renderer.prepare(SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)), plan)
    assert scene.ambient is None
    assert scene.directional is None
    assert scene.camera_calls == []


@pytest.mark.parametrize(
    ("scene", "message"),
    (
        (_InstalledScene(update_error=True), "initialize the GPU render system"),
        (_InstalledScene(group_fail_at=2), "create GPU camera group"),
        (_InstalledScene(reject_registry_write=True), "bind GPU camera groups"),
    ),
)
def test_installed_renderer_camera_group_initialization_fails_closed(
    scene: _InstalledScene, message: str
) -> None:
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 30
    )
    shader_config = _InstalledShaderConfig(shader_pack="minimal")
    module_importer, _ = _installed_renderer_modules({"minimal": shader_config})
    renderer = LazyManiSkillPickCubeVisualRenderer(module_importer=module_importer)

    with pytest.raises(ManiSkillVisualRenderingError, match=message):
        renderer.prepare(SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)), plan)
    assert scene.ambient == [0.3, 0.3, 0.3]
    assert scene.directional is not None
    assert scene.camera_groups == {}
    assert all(camera.camera_group is None for camera in scene.runtime_cameras)


@pytest.mark.parametrize(
    ("render_system", "texture_names", "message"),
    (
        ("3.1", ("Color",), "pinned SAPIEN render system 3.0"),
        ("3.0", ("PositionSegmentation",), "invalid ordered texture inventory"),
    ),
)
def test_installed_renderer_rejects_unbound_gpu_group_contracts(
    render_system: object, texture_names: tuple[str, ...], message: str
) -> None:
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 30
    )
    shader_config = _InstalledShaderConfig(
        shader_pack="minimal",
        texture_names=MappingProxyType({name: object() for name in texture_names}),
    )
    module_importer, _ = _installed_renderer_modules(
        {"minimal": shader_config}, render_system=render_system
    )
    scene = _InstalledScene()
    renderer = LazyManiSkillPickCubeVisualRenderer(module_importer=module_importer)

    with pytest.raises(ManiSkillVisualRenderingError, match=message):
        renderer.prepare(SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)), plan)
    assert scene.camera_calls == []
    assert scene.ambient is None


def test_visual_session_has_no_step_path_and_checks_every_repetition() -> None:
    episode, state = _source()
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 31
    )
    session, runtime = _session()
    result = session.render_state(
        source_episode=episode,
        source_state=state,
        render_plan=plan,
        repeat_count=2,
    )
    assert not hasattr(session, "step")
    assert runtime.step_calls == 0
    assert result.integrity.compared_state_component_count == 70
    assert result.integrity.verifier_component_count == 38
    assert result.integrity.verifier_state_maximum_absolute_error == 0.0
    assert result.integrity.verified_render_count == 2
    assert result.integrity.render_mutation_maximum_absolute_error == 0.0
    assert result.integrity.task_projection_exact
    assert tuple(view.camera_id for view in result.views) == (
        "front_oblique",
        "overhead",
        "side_oblique",
    )
    assert all(view.rgb.dtype == np.uint8 for view in result.views)
    assert all(view.rgb.shape == (224, 224, 3) for view in result.views)

    with pytest.raises(ManiSkillVisualSessionError, match="non-negative integer"):
        replace(result.integrity, elapsed_steps_before=True)
    with pytest.raises(ManiSkillVisualSessionError, match="non-negative integer"):
        replace(result.integrity, elapsed_steps_after=-1)


def test_visual_session_records_only_successfully_initialized_environments() -> None:
    episode, state = _source()
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 31
    )
    initialized = 0

    def record() -> None:
        nonlocal initialized
        initialized += 1

    session, _ = _session(environment_initialized_observer=record)
    session.render_state(
        source_episode=episode,
        source_state=state,
        render_plan=plan,
    )
    assert initialized == 1


@pytest.mark.parametrize(
    ("renderer", "runtime", "message"),
    [
        (_FakeRenderer(mutate_state=True), _FakeStateRuntime(), "post-render state"),
        (_FakeRenderer(fail=True), _FakeStateRuntime(), "synthetic renderer failure"),
        (_FakeRenderer(), _FakeStateRuntime(close_error=True), "close failed"),
    ],
)
def test_visual_session_rejects_state_renderer_and_close_failures(
    renderer: _FakeRenderer,
    runtime: _FakeStateRuntime,
    message: str,
) -> None:
    episode, state = _source()
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 37
    )
    session, _ = _session(renderer=renderer, runtime=runtime)
    with pytest.raises(
        (ManiSkillVisualSessionError, ManiSkillVisualRenderingError), match=message
    ):
        session.render_state(
            source_episode=episode,
            source_state=state,
            render_plan=plan,
        )


def test_probe_reports_exact_and_controlled_nondeterminism(tmp_path: Path) -> None:
    episode, state = _source()
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 41
    )
    exact_session, exact_runtime = _session()
    report_path = tmp_path / "visual-report.json"
    archive = PickCubeStateIndexedArchiveV1(episodes=(episode,))
    exact = probe_maniskill_pickcube_visual(
        compatibility_binding=_Binding(),  # type: ignore[arg-type]
        runtime=SessionPickCubeVisualProbeRuntime(exact_session),
        source_archive=archive,
        source_episode=episode,
        source_state=state,
        camera_rig=rig,
        render_domain_configuration=configuration,
        render_plan=plan,
        report_path=report_path,
    )
    assert exact.trusted_visual_generation_ready
    assert exact.probe_source.source_archive_digest == archive.content_digest
    assert exact.probe_source.source_episode_id == episode.episode_id
    assert exact.probe_source.source_reset_seed == episode.seed
    assert exact.probe_source.state_index == state.state_index
    assert exact.probe_source.state_content_digest == state.content_digest
    assert exact.probe_source.expected_state_digest == state.state_digest
    assert exact.repeated_render.changed_pixel_count == 0
    assert exact.fresh_environment_render.changed_pixel_count == 0
    assert exact.same_environment_render_count == 3
    assert exact.fresh_environment_render_count == 2
    assert exact.environment_initialization_count == 3
    assert len(exact.repeated_render.samples) == 2
    assert len(exact.fresh_environment_render.samples) == 2
    assert [sample.comparison_id for sample in exact.repeated_render.samples] == [
        "same_environment_render_2_vs_1",
        "same_environment_render_3_vs_1",
    ]
    assert [
        sample.comparison_id for sample in exact.fresh_environment_render.samples
    ] == [
        "fresh_environment_1_render_1_vs_same_environment_render_1",
        "fresh_environment_2_render_1_vs_same_environment_render_1",
    ]
    assert len(exact_runtime.created_environments) == 3
    assert len({id(item) for item in exact_runtime.created_environments}) == 3
    assert exact.renderer_api.runtime_intrinsics_dtype == "float64"
    assert exact.renderer_api.runtime_extrinsics_dtype == "float64"
    assert load_visual_compatibility_report(report_path) == exact

    changed_dtype = replace(
        exact,
        renderer_api=replace(
            exact.renderer_api,
            runtime_intrinsics_dtype="float32",
            runtime_extrinsics_dtype="float32",
        ),
    )
    assert changed_dtype.visual_compatibility_identity != (
        exact.visual_compatibility_identity
    )
    changed_camera_group_inventory = replace(
        exact,
        renderer_api=replace(
            exact.renderer_api,
            camera_group_texture_names=("PositionSegmentation", "Color"),
        ),
    )
    assert changed_camera_group_inventory.visual_compatibility_identity != (
        exact.visual_compatibility_identity
    )

    unstable_session, _ = _session(renderer=_FakeRenderer(nondeterministic=True))
    unstable = probe_maniskill_pickcube_visual(
        compatibility_binding=_Binding(),  # type: ignore[arg-type]
        runtime=SessionPickCubeVisualProbeRuntime(unstable_session),
        source_archive=archive,
        source_episode=episode,
        source_state=state,
        camera_rig=rig,
        render_domain_configuration=configuration,
        render_plan=plan,
        require_trusted_generation=False,
    )
    assert not unstable.trusted_visual_generation_ready
    assert unstable.repeated_render.changed_pixel_count == 1
    assert unstable.repeated_render.maximum_per_channel_absolute_difference == 1
    assert tuple(
        sample.changed_pixel_count for sample in unstable.repeated_render.samples
    ) == (1, 1)
    assert tuple(
        sample.changed_pixel_count
        for sample in unstable.fresh_environment_render.samples
    ) == (0, 0)
    with pytest.raises(ManiSkillVisualProbeError, match="did not authorize"):
        unstable.require_trusted_visual_generation_ready()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["same_environment_render_count"] = 2
    changed_count = tmp_path / "changed-probe-count.json"
    changed_count.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManiSkillVisualProbeError, match="exactly three"):
        load_visual_compatibility_report(changed_count)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["repeated_render"]["samples"].pop()
    missing_comparison = tmp_path / "missing-pixel-comparison.json"
    missing_comparison.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManiSkillVisualProbeError, match="pixel inventory"):
        load_visual_compatibility_report(missing_comparison)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["fresh_environment_render"]["samples"][0]["comparison_id"] = (
        "unbound-fresh-comparison"
    )
    renamed_comparison = tmp_path / "renamed-pixel-comparison.json"
    renamed_comparison.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManiSkillVisualProbeError, match=r"bounded 3\+2 plan"):
        load_visual_compatibility_report(renamed_comparison)


@pytest.mark.parametrize(
    ("fresh_environment_index", "expected_changed_pixel_counts"),
    [(1, (1, 0)), (2, (0, 1))],
)
def test_probe_trusted_gate_rejects_each_fresh_environment_pixel_drift(
    fresh_environment_index: int,
    expected_changed_pixel_counts: tuple[int, int],
) -> None:
    episode, state = _source()
    archive = PickCubeStateIndexedArchiveV1(episodes=(episode,))
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 42
    )
    renderer = _FreshEnvironmentDriftRenderer(
        drift_fresh_environment_index=fresh_environment_index
    )
    session, _ = _session(renderer=renderer)

    report = probe_maniskill_pickcube_visual(
        compatibility_binding=_Binding(),  # type: ignore[arg-type]
        runtime=SessionPickCubeVisualProbeRuntime(session),
        source_archive=archive,
        source_episode=episode,
        source_state=state,
        camera_rig=rig,
        render_domain_configuration=configuration,
        render_plan=plan,
        require_trusted_generation=False,
    )

    assert renderer.prepared_environment_count == 3
    assert report.repeated_render.exact_match
    assert tuple(
        sample.changed_pixel_count for sample in report.repeated_render.samples
    ) == (0, 0)
    assert not report.fresh_environment_render.exact_match
    assert (
        tuple(
            sample.changed_pixel_count
            for sample in report.fresh_environment_render.samples
        )
        == expected_changed_pixel_counts
    )
    assert not report.trusted_visual_generation_ready
    with pytest.raises(ManiSkillVisualProbeError, match="did not authorize"):
        report.require_trusted_visual_generation_ready()


def test_probe_rejects_unbound_archive_state_and_source_evidence_tampering(
    tmp_path: Path,
) -> None:
    episode, state = _source()
    archive = PickCubeStateIndexedArchiveV1(episodes=(episode,))
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 42
    )
    session, _ = _session()
    with pytest.raises(ManiSkillVisualProbeError, match="episode is absent"):
        probe_maniskill_pickcube_visual(
            compatibility_binding=_Binding(),  # type: ignore[arg-type]
            runtime=SessionPickCubeVisualProbeRuntime(session),
            source_archive=archive,
            source_episode=replace(episode, episode_id="unbound-episode"),
            source_state=state,
            camera_rig=rig,
            render_domain_configuration=configuration,
            render_plan=plan,
        )

    report_path = tmp_path / "visual-source-report.json"
    valid_session, _ = _session()
    probe_maniskill_pickcube_visual(
        compatibility_binding=_Binding(),  # type: ignore[arg-type]
        runtime=SessionPickCubeVisualProbeRuntime(valid_session),
        source_archive=archive,
        source_episode=episode,
        source_state=state,
        camera_rig=rig,
        render_domain_configuration=configuration,
        render_plan=plan,
        report_path=report_path,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["probe_source"]["source_episode_id"] = "tampered-episode"
    tampered_path = tmp_path / "tampered-source-report.json"
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManiSkillVisualProbeError, match="identity.*tampered"):
        load_visual_compatibility_report(tampered_path)


def test_report_loader_rejects_duplicate_and_non_finite_fields(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"1.0","schema_version":"1.0"}')
    with pytest.raises(ManiSkillVisualProbeError, match="duplicate"):
        load_visual_compatibility_report(duplicate)
    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text(json.dumps({"value": 1}).replace("1", "NaN"))
    with pytest.raises(ManiSkillVisualProbeError, match="rejects JSON constant"):
        load_visual_compatibility_report(non_finite)


def test_verified_session_builds_core_packet_and_reference_keyed_npy_mapping() -> None:
    episode, state = _source()
    rig, configuration = _configuration()
    domain = configuration.domain("canonical")
    plan = build_pickcube_visual_render_plan(rig, domain, configuration, 43)
    session, _ = _session()
    result = session.render_state(
        source_episode=episode,
        source_state=state,
        render_plan=plan,
    )
    prepared = build_visual_observation_packet(
        result,
        source_collection=SourceCollection.M3A_DEVELOPMENT,
        anchor_id="anchor-a",
        split=VisualDatasetSplit.TRAIN,
        split_group_id="split-group-a",
        state_reference_id=state.content_digest,
        camera_rig=rig,
        render_domain_configuration=configuration,
        visual_compatibility=_TrustedVisual(
            visual_compatibility_identity=f"sha256:{'b' * 64}",
            pickcube_compatibility_identity=_COMPATIBILITY,
            camera_rig_digest=rig.rig_digest,
            render_domain_configuration_digest=configuration.content_digest,
            renderer_api=result.renderer_api,
        ),
    )
    references = tuple(view.image_reference for view in prepared.packet.views)
    assert tuple(prepared.images) == references
    assert all(reference.endswith(".npy") for reference in references)
    assert prepared.packet.expected_state_digest == state.state_digest
    assert prepared.packet.verifier_state_digest == state.verifier_state.content_digest
    assert prepared.packet.verifier_state_component_count == 38
    assert prepared.packet.verifier_state_maximum_absolute_error == 0.0
    assert (
        prepared.packet.elapsed_simulation_steps_before
        == prepared.packet.elapsed_simulation_steps_after
        == 0
    )
    assert prepared.packet.environment_close_passed is True
    assert tuple(view.intrinsics_dtype for view in prepared.packet.views) == tuple(
        view.intrinsics.dtype.name for view in result.views
    )
    assert tuple(view.extrinsics_dtype for view in prepared.packet.views) == tuple(
        view.extrinsics.dtype.name for view in result.views
    )
    assert prepared.packet.pickcube_compatibility_identity == _COMPATIBILITY
    assert isinstance(prepared.images, MappingProxyType)

    with pytest.raises(ManiSkillVisualSessionError, match="successfully closed"):
        build_visual_observation_packet(
            replace(result, environment_close_passed=False),
            source_collection=SourceCollection.M3A_DEVELOPMENT,
            anchor_id="anchor-a",
            split=VisualDatasetSplit.TRAIN,
            split_group_id="split-group-a",
            state_reference_id=state.content_digest,
            camera_rig=rig,
            render_domain_configuration=configuration,
            visual_compatibility=_TrustedVisual(
                visual_compatibility_identity=f"sha256:{'b' * 64}",
                pickcube_compatibility_identity=_COMPATIBILITY,
                camera_rig_digest=rig.rig_digest,
                render_domain_configuration_digest=configuration.content_digest,
                renderer_api=result.renderer_api,
            ),
        )

    with pytest.raises(ManiSkillVisualSessionError, match="reviewed visual"):
        build_visual_observation_packet(
            result,
            source_collection=SourceCollection.M3A_DEVELOPMENT,
            anchor_id="anchor-a",
            split=VisualDatasetSplit.TRAIN,
            split_group_id="split-group-a",
            state_reference_id=state.content_digest,
            camera_rig=rig,
            render_domain_configuration=configuration,
            visual_compatibility=_TrustedVisual(
                visual_compatibility_identity=f"sha256:{'b' * 64}",
                pickcube_compatibility_identity=_COMPATIBILITY,
                camera_rig_digest=rig.rig_digest,
                render_domain_configuration_digest=configuration.content_digest,
                renderer_api=replace(
                    result.renderer_api, renderer_backend="drifted_backend"
                ),
            ),
        )


def test_float_color_conversion_rejects_out_of_range_values_without_clipping() -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    valid = np.zeros((224, 224, 4), dtype=np.float32)
    valid[..., 3] = 1.0
    converted, _, _ = visual_rendering._convert_color_to_rgb_uint8(valid)
    assert converted.dtype == np.uint8
    assert not np.any(converted)

    below = valid.copy()
    below[0, 0, 0] = -0.001
    with pytest.raises(ManiSkillVisualRenderingError, match=r"authorized \[0,1\]"):
        visual_rendering._convert_color_to_rgb_uint8(below)

    above = valid.copy()
    above[0, 0, 0] = 1.001
    with pytest.raises(ManiSkillVisualRenderingError, match=r"authorized \[0,1\]"):
        visual_rendering._convert_color_to_rgb_uint8(above)


def test_runtime_calibration_must_match_content_bound_plan_exactly_after_cast() -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    planned = np.array([[1.25, 0.0], [0.0, 2.5]], dtype=np.float64)
    observed = planned.astype(np.float32)
    visual_rendering._require_runtime_calibration_matches_plan(
        observed, planned, field_name="test intrinsics"
    )

    changed = observed.copy()
    changed[0, 0] += np.float32(0.001)
    with pytest.raises(
        ManiSkillVisualRenderingError, match="content-bound camera plan"
    ):
        visual_rendering._require_runtime_calibration_matches_plan(
            changed, planned, field_name="test intrinsics"
        )
