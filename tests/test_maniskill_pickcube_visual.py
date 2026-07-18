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
    EXTRINSIC_3X4_TO_4X4_SEMANTIC,
    EXTRINSIC_4X4_SEMANTIC,
    GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC,
    UINT8_COLOR_TO_RGB_UINT8_SEMANTIC,
    LazyManiSkillPickCubeVisualRenderer,
    ManiSkillVisualRenderingError,
    PickCubeVisualRenderPlan,
    RenderedVisualView,
    RuntimeCalibrationEvidenceV1,
    VisualRendererApiObservation,
    build_pickcube_visual_render_plan,
    runtime_calibration_array_digest,
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
            runtime_intrinsics_dtype="float32",
            runtime_extrinsics_dtype="float32",
            raw_extrinsic_matrix_shape="[1,3,4]",
            runtime_extrinsics_semantic=EXTRINSIC_3X4_TO_4X4_SEMANTIC,
            runtime_camera_pose_dtype="float32",
            raw_camera_pose_shape="[1,7]",
            runtime_camera_pose_device_type="cuda",
            runtime_pose_component_count=7,
            runtime_underlying_pose_type="sapien.pysapien.Pose",
            runtime_underlying_position_type="numpy.ndarray",
            runtime_underlying_quaternion_type="numpy.ndarray",
            runtime_underlying_pose_dtype="float32",
            raw_underlying_position_shape="[3]",
            raw_underlying_quaternion_shape="[4]",
            runtime_underlying_pose_component_count=7,
            underlying_camera_pose_pre_post_bitwise=True,
            runtime_extrinsic_component_count=12,
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
                    intrinsics=camera.intrinsics.astype(np.float32),
                    extrinsics=camera.extrinsics.astype(np.float32),
                    runtime_extrinsics=camera.extrinsics.astype(np.float32),
                    expected_runtime_extrinsics_digest=(
                        runtime_calibration_array_digest(
                            camera.extrinsics.astype(np.float32)
                        )
                    ),
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
    runtime_calibration_evidence: tuple[RuntimeCalibrationEvidenceV1, ...]

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


def test_probe_calibration_stability_rejects_one_bit_raw_fresh_drift() -> None:
    from latentguard.integrations.maniskill_pickcube import visual_probe

    episode, state = _source()
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 34
    )
    session, _ = _session()
    repeated = session.render_state(
        source_episode=episode,
        source_state=state,
        render_plan=plan,
        repeat_count=2,
    )
    fresh = session.render_state(
        source_episode=episode,
        source_state=state,
        render_plan=plan,
        repeat_count=1,
    )
    changed_extrinsics = np.array(fresh.views[0].runtime_extrinsics, copy=True)
    changed_extrinsics[0, 0] = np.nextafter(
        changed_extrinsics[0, 0], np.float32(np.inf)
    )
    changed_view = replace(
        fresh.views[0],
        runtime_extrinsics=changed_extrinsics,
        expected_runtime_extrinsics_digest=runtime_calibration_array_digest(
            changed_extrinsics
        ),
    )
    changed_fresh = replace(
        fresh,
        render_repetitions=((changed_view, *fresh.views[1:]),),
    )
    changed_dtype_runtime = fresh.views[0].runtime_extrinsics.astype(np.float64)
    changed_dtype_view = replace(
        fresh.views[0],
        extrinsics=fresh.views[0].extrinsics.astype(np.float64),
        runtime_extrinsics=changed_dtype_runtime,
        expected_runtime_extrinsics_digest=runtime_calibration_array_digest(
            changed_dtype_runtime
        ),
    )
    changed_dtype_fresh = replace(
        fresh,
        render_repetitions=((changed_dtype_view, *fresh.views[1:]),),
    )
    signed_zero_extrinsics = np.array(fresh.views[0].runtime_extrinsics, copy=True)
    assert signed_zero_extrinsics[3, 0] == 0.0
    signed_zero_extrinsics[3, 0] = np.float32(-0.0)
    assert np.array_equal(signed_zero_extrinsics, fresh.views[0].runtime_extrinsics)
    signed_zero_view = replace(
        fresh.views[0],
        runtime_extrinsics=signed_zero_extrinsics,
        expected_runtime_extrinsics_digest=runtime_calibration_array_digest(
            signed_zero_extrinsics
        ),
    )
    signed_zero_fresh = replace(
        fresh,
        render_repetitions=((signed_zero_view, *fresh.views[1:]),),
    )
    assert visual_probe._calibration_stable(repeated, (fresh,))
    assert not visual_probe._calibration_stable(repeated, (changed_fresh,))
    assert not visual_probe._calibration_stable(repeated, (changed_dtype_fresh,))
    assert not visual_probe._calibration_stable(repeated, (signed_zero_fresh,))


def test_session_calibration_bit_comparison_rejects_signed_zero_drift() -> None:
    from latentguard.integrations.maniskill_pickcube import visual_session

    positive = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    negative = positive.copy()
    negative[0, 1] = np.float32(-0.0)

    assert np.array_equal(positive, negative)
    assert not visual_session._array_bits_equal(positive, negative)


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
    assert exact.renderer_api.runtime_intrinsics_dtype == "float32"
    assert exact.renderer_api.runtime_extrinsics_dtype == "float32"
    assert tuple(item.camera_id for item in exact.runtime_calibration_evidence) == (
        "front_oblique",
        "overhead",
        "side_oblique",
    )
    assert all(
        item.runtime_extrinsics_digest == item.expected_runtime_extrinsics_digest
        for item in exact.runtime_calibration_evidence
    )
    assert load_visual_compatibility_report(report_path) == exact

    normalized = replace(
        exact,
        runtime_calibration_evidence=cast(
            Any, (item for item in exact.runtime_calibration_evidence)
        ),
    )
    assert normalized.runtime_calibration_evidence == exact.runtime_calibration_evidence
    assert (
        normalized.visual_compatibility_identity == exact.visual_compatibility_identity
    )

    changed_dtype = replace(
        exact,
        renderer_api=replace(
            exact.renderer_api,
            runtime_intrinsics_dtype="float64",
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
    changed_digest = f"sha256:{'d' * 64}"
    changed_calibration_evidence = replace(
        exact,
        runtime_calibration_evidence=(
            replace(
                exact.runtime_calibration_evidence[0],
                runtime_extrinsics_digest=changed_digest,
                expected_runtime_extrinsics_digest=changed_digest,
            ),
            *exact.runtime_calibration_evidence[1:],
        ),
    )
    assert changed_calibration_evidence.visual_compatibility_identity != (
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

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    malformed_digest = f"sha256:{'g' * 64}"
    payload["runtime_calibration_evidence"][0]["runtime_extrinsics_digest"] = (
        malformed_digest
    )
    payload["runtime_calibration_evidence"][0]["expected_runtime_extrinsics_digest"] = (
        malformed_digest
    )
    malformed_calibration = tmp_path / "malformed-calibration-digest.json"
    malformed_calibration.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManiSkillVisualRenderingError, match="lowercase SHA-256"):
        load_visual_compatibility_report(malformed_calibration)


def test_runtime_calibration_evidence_rejects_non_hex_digest() -> None:
    malformed_digest = f"sha256:{'G' * 64}"
    with pytest.raises(ManiSkillVisualRenderingError, match="lowercase SHA-256"):
        RuntimeCalibrationEvidenceV1(
            camera_id="front_oblique",
            camera_configuration_digest=malformed_digest,
            runtime_extrinsics_digest=malformed_digest,
            expected_runtime_extrinsics_digest=malformed_digest,
        )


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
            runtime_calibration_evidence=tuple(
                view.runtime_calibration_evidence for view in result.views
            ),
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
    assert tuple(
        view.runtime_extrinsics_digest for view in prepared.packet.views
    ) == tuple(view.runtime_extrinsics_digest for view in result.views)
    assert tuple(
        view.expected_runtime_extrinsics_digest for view in prepared.packet.views
    ) == tuple(view.expected_runtime_extrinsics_digest for view in result.views)
    assert all(
        np.array_equal(
            np.asarray(view.extrinsics, dtype=np.float64),
            np.asarray(planned.extrinsics, dtype=result_view.extrinsics.dtype).astype(
                np.float64
            ),
        )
        for view, planned, result_view in zip(
            prepared.packet.views,
            result.render_plan.cameras,
            result.views,
            strict=True,
        )
    )
    assert prepared.packet.pickcube_compatibility_identity == _COMPATIBILITY
    assert isinstance(prepared.images, MappingProxyType)

    drifted_canonical = np.array(result.views[0].extrinsics, copy=True)
    drifted_canonical[0, 0] = np.nextafter(drifted_canonical[0, 0], np.float32(np.inf))
    assert not np.array_equal(
        drifted_canonical,
        np.asarray(result.render_plan.cameras[0].extrinsics, dtype=np.float32),
    )
    drifted_view = replace(result.views[0], extrinsics=drifted_canonical)
    drifted_result = replace(
        result,
        render_repetitions=((drifted_view, *result.views[1:]),),
    )
    with pytest.raises(
        ManiSkillVisualSessionError,
        match="calibration differs from its exact plan or runtime evidence",
    ):
        build_visual_observation_packet(
            drifted_result,
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
                runtime_calibration_evidence=tuple(
                    view.runtime_calibration_evidence for view in result.views
                ),
            ),
        )

    verified_runtime = np.array(result.views[0].runtime_extrinsics, copy=True)
    verified_runtime[0, 0] = np.nextafter(verified_runtime[0, 0], np.float32(np.inf))
    with pytest.raises(
        ManiSkillVisualRenderingError,
        match="runtime extrinsics lack matching verified evidence",
    ):
        replace(result.views[0], runtime_extrinsics=verified_runtime)
    verified_runtime_view = replace(
        result.views[0],
        runtime_extrinsics=verified_runtime,
        expected_runtime_extrinsics_digest=runtime_calibration_array_digest(
            verified_runtime
        ),
    )
    verified_runtime_result = replace(
        result,
        render_repetitions=((verified_runtime_view, *result.views[1:]),),
    )
    with pytest.raises(
        ManiSkillVisualSessionError,
        match="canonical runtime calibration differs from reviewed probe evidence",
    ):
        build_visual_observation_packet(
            verified_runtime_result,
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
                runtime_calibration_evidence=tuple(
                    view.runtime_calibration_evidence for view in result.views
                ),
            ),
        )

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
                runtime_calibration_evidence=tuple(
                    view.runtime_calibration_evidence for view in result.views
                ),
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
                runtime_calibration_evidence=tuple(
                    view.runtime_calibration_evidence for view in result.views
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


def test_world_to_camera_extrinsics_and_runtime_shape_normalization_are_explicit() -> (
    None
):
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    expected = np.asarray(
        (
            (0.0, -1.0, 0.0, 2.0),
            (0.0, 0.0, -1.0, 3.0),
            (1.0, 0.0, 0.0, -1.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    assert np.array_equal(
        visual_rendering._world_to_camera_extrinsics(
            (1.0, 2.0, 3.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
        expected,
    )
    for raw, raw_shape, expected_semantic in (
        (expected[:3], "[3,4]", EXTRINSIC_3X4_TO_4X4_SEMANTIC),
        (
            expected[:3][None, ...],
            "[1,3,4]",
            EXTRINSIC_3X4_TO_4X4_SEMANTIC,
        ),
        (expected, "[4,4]", EXTRINSIC_4X4_SEMANTIC),
        (expected[None, ...], "[1,4,4]", EXTRINSIC_4X4_SEMANTIC),
    ):
        normalized, observed_shape, observed_semantic = (
            visual_rendering._single_extrinsic_matrix(raw)
        )
        assert observed_shape == raw_shape
        assert observed_semantic == expected_semantic
        assert np.array_equal(normalized, expected)


class Tensor:
    """Minimal CUDA-tensor double for strict renderer-calibration tests."""

    __module__ = "torch"

    def __init__(self, value: object) -> None:
        self.array = np.asarray(value)
        self.device = SimpleNamespace(type="cuda", index=0)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def T(self) -> Tensor:
        return Tensor(self.array.T)

    def detach(self) -> Tensor:
        return self

    def cpu(self) -> Tensor:
        return self

    def numpy(self) -> NDArray[Any]:
        return self.array

    def clone(self) -> Tensor:
        return Tensor(self.array.copy())

    def new_tensor(self, value: object) -> Tensor:
        return Tensor(np.asarray(value, dtype=self.array.dtype))

    def __matmul__(self, other: Tensor) -> Tensor:
        assert self.array.shape == (4, 4)
        assert other.array.shape == (1, 4, 4)
        product = np.empty((1, 4, 4), dtype=np.float32)
        for row in range(4):
            for column in range(4):
                total = np.float32(0.0)
                for component in range(4):
                    total = np.float32(
                        total
                        + np.float32(
                            self.array[row, component]
                            * other.array[0, component, column]
                        )
                    )
                product[0, row, column] = total
        return Tensor(product)

    def __getitem__(self, key: object) -> Tensor:
        return Tensor(self.array[key])


class Pose:
    """Minimal ManiSkill Pose double with a deterministic float32 inverse matrix."""

    __module__ = "mani_skill.utils.structs.pose"

    def __init__(self, raw_pose: Tensor) -> None:
        self.raw_pose = raw_pose

    @classmethod
    def create(cls, raw_pose: Tensor, *, device: object = None) -> Pose:
        del device
        return cls(raw_pose)

    def inv(self) -> _InversePose:
        return _InversePose(self.raw_pose)


class _InversePose:
    def __init__(self, raw_pose: Tensor) -> None:
        self.raw_pose = raw_pose

    def to_transformation_matrix(self) -> Tensor:
        position = self.raw_pose.array[0, :3]
        w, x, y, z = self.raw_pose.array[0, 3:]
        squared_norm = np.float32(w * w + x * x + y * y + z * z)
        two_s = np.float32(2.0) / squared_norm
        camera_to_world = np.asarray(
            (
                (
                    np.float32(1.0) - two_s * (y * y + z * z),
                    two_s * (x * y - z * w),
                    two_s * (x * z + y * w),
                ),
                (
                    two_s * (x * y + z * w),
                    np.float32(1.0) - two_s * (x * x + z * z),
                    two_s * (y * z - x * w),
                ),
                (
                    two_s * (x * z - y * w),
                    two_s * (y * z + x * w),
                    np.float32(1.0) - two_s * (x * x + y * y),
                ),
            ),
            dtype=np.float32,
        )
        matrix = np.zeros((1, 4, 4), dtype=np.float32)
        matrix[0, :3, :3] = camera_to_world.T
        for row in range(3):
            total = np.float32(0.0)
            for component in range(3):
                total = np.float32(
                    total
                    + np.float32(camera_to_world[component, row] * position[component])
                )
            matrix[0, row, 3] = -total
        matrix[0, 3, 3] = np.float32(1.0)
        return Tensor(matrix)


class _SapienPose:
    """Minimal native SAPIEN pose double exposing float32 NumPy components."""

    __module__ = "sapien.pysapien"
    __qualname__ = "Pose"

    def __init__(self, raw_pose: NDArray[np.float32]) -> None:
        self.p = np.array(raw_pose[:3], copy=True, order="C")
        self.q = np.array(raw_pose[3:], copy=True, order="C")


class _StrictUnderlyingRenderCamera:
    """Expose the live, uncached native component pose for strict tests."""

    def __init__(self, owner: _StrictRuntimeCamera) -> None:
        self.owner = owner

    def get_local_pose(self) -> _SapienPose:
        return _SapienPose(self.owner.raw_pose[0])


class _StrictRuntimeCamera:
    def __init__(
        self,
        plan: object,
        *,
        pose_drift: bool = False,
        public_drift: bool = False,
        wrong_axis: bool = False,
        mutate_on_take: bool = False,
        mutate_on_get_picture: bool = False,
        mutate_on_second_extrinsic: bool = False,
        cache_wrapper_pose: bool = False,
        public_device_index: int = 0,
        signed_zero_intrinsics: bool = False,
        signed_zero_intrinsics_after_take: bool = False,
        fixed_public_extrinsics: NDArray[np.float32] | None = None,
    ) -> None:
        self.plan = plan
        raw_pose = np.asarray(
            ((*plan.position, *plan.quaternion_wxyz),), dtype=np.float32
        )
        self.raw_pose = raw_pose
        self.wrapper_raw_pose = raw_pose.copy()
        if pose_drift:
            self.wrapper_raw_pose[0, 0] = np.nextafter(
                self.wrapper_raw_pose[0, 0], np.float32(np.inf)
            )
        self.use_wrapper_raw_pose = pose_drift or cache_wrapper_pose
        self.public_device_index = public_device_index
        self.signed_zero_intrinsics = signed_zero_intrinsics
        self.public_drift = public_drift
        self.wrong_axis = wrong_axis
        self.mutate_on_take = mutate_on_take
        self.mutate_on_get_picture = mutate_on_get_picture
        self.mutate_on_second_extrinsic = mutate_on_second_extrinsic
        self.signed_zero_intrinsics_after_take = signed_zero_intrinsics_after_take
        self.fixed_public_extrinsics = fixed_public_extrinsics
        self.captured = False
        self.extrinsic_calls = 0
        self.mount = None
        self._render_cameras = [_StrictUnderlyingRenderCamera(self)]

    def get_global_pose(self) -> Pose:
        raw_pose = self.wrapper_raw_pose if self.use_wrapper_raw_pose else self.raw_pose
        return Pose(Tensor(raw_pose.copy()))

    def get_intrinsic_matrix(self) -> Tensor:
        intrinsics = np.asarray(self.plan.intrinsics, dtype=np.float32)[None, ...]
        if self.signed_zero_intrinsics:
            intrinsics = intrinsics.copy()
            intrinsics[0, 0, 1] = np.float32(-0.0)
        if self.signed_zero_intrinsics_after_take and self.captured:
            intrinsics = intrinsics.copy()
            intrinsics[0, 0, 1] = np.float32(-0.0)
        return Tensor(intrinsics)

    def get_extrinsic_matrix(self) -> Tensor:
        self.extrinsic_calls += 1
        if self.fixed_public_extrinsics is not None:
            tensor = Tensor(self.fixed_public_extrinsics.copy())
            tensor.device.index = self.public_device_index
            return tensor
        inverse = self.get_global_pose().inv().to_transformation_matrix()
        axis = Tensor(
            np.asarray(
                (
                    (0.0, 0.0, 1.0, 0.0),
                    (-1.0, 0.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dtype=np.float32,
            )
        )
        raw = (axis.T @ inverse)[:, :3, :4].array
        if self.wrong_axis:
            raw = raw.copy()
            raw[:, 0, :] = -raw[:, 0, :]
        if self.public_drift:
            raw = raw.copy()
            raw[0, 0, 0] = np.nextafter(raw[0, 0, 0], np.float32(np.inf))
        if self.mutate_on_second_extrinsic and self.extrinsic_calls == 2:
            self.raw_pose[0, 0] = np.nextafter(self.raw_pose[0, 0], np.float32(np.inf))
        tensor = Tensor(raw)
        tensor.device.index = self.public_device_index
        return tensor

    def take_picture(self) -> None:
        self.captured = True
        if self.mutate_on_take:
            self.raw_pose[0, 0] = np.nextafter(self.raw_pose[0, 0], np.float32(np.inf))

    def get_picture(self, names: list[str]) -> list[Tensor]:
        assert names == ["Color"]
        if self.mutate_on_get_picture:
            self.raw_pose[0, 0] = np.nextafter(self.raw_pose[0, 0], np.float32(np.inf))
        return [Tensor(np.zeros((1, 224, 224, 4), dtype=np.uint8))]


class _StrictRuntimeScene:
    def update_render(self, **kwargs: object) -> None:
        assert kwargs == {
            "update_sensors": False,
            "update_human_render_cameras": False,
        }


def _unobserved_renderer_api(
    plan: PickCubeVisualRenderPlan,
) -> VisualRendererApiObservation:
    return VisualRendererApiObservation(
        renderer_backend="fake_gpu",
        scene_type="fake.Scene",
        camera_type="fake.Camera",
        shader_configuration=plan.shader_configuration,
        camera_configuration_api="fake.add_camera",
        camera_group_initialization_semantic=GPU_CAMERA_GROUP_INITIALIZATION_SEMANTIC,
        camera_group_texture_names=("Color",),
        camera_group_count=3,
        underlying_camera_count_per_group=1,
        camera_groups_ready=True,
        world_camera_pose_representation="sapien.Pose(position,quaternion_wxyz)",
        sensor_update_calls=("fake.take_picture",),
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("runtime_pose_component_count", 0.0),
        ("runtime_pose_component_count", False),
        ("runtime_underlying_pose_component_count", 0.0),
        ("runtime_underlying_pose_component_count", False),
        ("runtime_extrinsic_component_count", 0.0),
        ("runtime_extrinsic_component_count", False),
    ),
)
def test_renderer_api_component_counts_require_strict_integers(
    field_name: str, invalid_value: float | bool
) -> None:
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 41
    )
    observation = _unobserved_renderer_api(plan)

    with pytest.raises(ManiSkillVisualRenderingError, match="must be an integer"):
        replace(observation, **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("camera_kwargs", "message"),
    (
        ({"pose_drift": True}, "runtime world pose differs at the bit level"),
        ({"public_drift": True}, "public extrinsics derivation differs"),
        ({"wrong_axis": True}, "public extrinsics derivation differs"),
        ({"public_device_index": 1}, "public extrinsics crossed CUDA devices"),
        (
            {"signed_zero_intrinsics": True},
            "intrinsics differs from the content-bound camera plan",
        ),
    ),
)
def test_runtime_camera_calibration_rejects_pose_public_and_axis_drift(
    camera_kwargs: dict[str, object], message: str
) -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 42
    )
    camera_plan = plan.cameras[0]
    camera = _StrictRuntimeCamera(camera_plan, **camera_kwargs)
    with pytest.raises(ManiSkillVisualRenderingError, match=message):
        visual_rendering._read_verified_runtime_calibration(
            plan=camera_plan,
            camera=camera,
            get_intrinsic=camera.get_intrinsic_matrix,
            get_extrinsic=camera.get_extrinsic_matrix,
        )


def test_runtime_camera_calibration_matches_independent_nontrivial_golden() -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 44
    )
    golden = np.asarray(
        (
            (-0.0, 1.0, -0.0, -2.0),
            (-0.0, -0.0, -1.0, 3.0),
            (-1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    camera_plan = replace(
        plan.cameras[0],
        position=(1.0, 2.0, 3.0),
        quaternion_wxyz=(0.0, 0.0, 0.0, 1.0),
        extrinsics=golden,
    )
    runtime_golden = np.asarray(
        (
            (0.0, 1.0, 0.0, -2.0),
            (0.0, 0.0, -1.0, 3.0),
            (-1.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )[None, ...]
    camera = _StrictRuntimeCamera(
        camera_plan,
        fixed_public_extrinsics=runtime_golden,
    )
    observed = visual_rendering._read_verified_runtime_calibration(
        plan=camera_plan,
        camera=camera,
        get_intrinsic=camera.get_intrinsic_matrix,
        get_extrinsic=camera.get_extrinsic_matrix,
    )
    assert np.array_equal(observed.extrinsics, golden.astype(np.float32))


def test_render_plan_rejects_signed_zero_extrinsic_drift() -> None:
    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 48
    )
    changed = np.array(plan.cameras[0].extrinsics, copy=True)
    assert changed[3, 0] == 0.0
    changed[3, 0] = -0.0

    with pytest.raises(ManiSkillVisualRenderingError, match="exactly derive"):
        replace(plan.cameras[0], extrinsics=changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("pose_type", "underlying pose type is unsupported"),
        ("array_type", "underlying pose arrays are unsupported"),
        ("dtype", "underlying position must have dtype float32"),
        ("shape", "underlying position must have shape"),
        ("contiguity", "underlying pose arrays must be C-contiguous"),
        ("quaternion_drift", "underlying world quaternion differs at the bit level"),
    ),
)
def test_underlying_camera_pose_contract_fails_closed(
    mutation: str, message: str
) -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 47
    )
    camera_plan = plan.cameras[0]
    raw_pose = np.asarray(
        (*camera_plan.position, *camera_plan.quaternion_wxyz), dtype=np.float32
    )
    native_pose: object = _SapienPose(raw_pose)
    if mutation == "pose_type":
        native_pose = SimpleNamespace(p=raw_pose[:3], q=raw_pose[3:])
    else:
        assert isinstance(native_pose, _SapienPose)
        if mutation == "array_type":
            native_pose.p = list(native_pose.p)  # type: ignore[assignment]
        elif mutation == "dtype":
            native_pose.p = native_pose.p.astype(np.float64)
        elif mutation == "shape":
            native_pose.p = native_pose.p.reshape(1, 3)
        elif mutation == "contiguity":
            storage = np.empty((3, 2), dtype=np.float32)
            storage[:, 0] = native_pose.p
            native_pose.p = storage[:, 0]
        else:
            native_pose.q[0] = np.nextafter(native_pose.q[0], np.float32(np.inf))
    render_camera = SimpleNamespace(get_local_pose=lambda: native_pose)

    with pytest.raises(ManiSkillVisualRenderingError, match=message):
        visual_rendering._read_verified_underlying_world_pose(
            plan=camera_plan,
            render_camera=render_camera,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (Tensor(np.zeros((1, 7), dtype=np.float64)), "pinned float32"),
        (Tensor(np.zeros((7,), dtype=np.float32)), "shape"),
    ),
)
def test_pinned_runtime_tensor_rejects_dtype_and_shape(
    value: Tensor, message: str
) -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    with pytest.raises(ManiSkillVisualRenderingError, match=message):
        visual_rendering._require_pinned_runtime_tensor(
            value,
            shape=(1, 7),
            field_name="test pose",
        )

    cpu = Tensor(np.zeros((1, 7), dtype=np.float32))
    cpu.device.type = "cpu"
    with pytest.raises(ManiSkillVisualRenderingError, match="CUDA"):
        visual_rendering._require_pinned_runtime_tensor(
            cpu,
            shape=(1, 7),
            field_name="test pose",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("mount", "unmounted world camera"),
        ("multi_camera", "one underlying camera"),
    ),
)
def test_runtime_camera_calibration_rejects_mount_and_camera_count(
    mutation: str, message: str
) -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 46
    )
    camera = _StrictRuntimeCamera(plan.cameras[0])
    if mutation == "mount":
        camera.mount = object()
    else:
        camera._render_cameras.append(object())
    with pytest.raises(ManiSkillVisualRenderingError, match=message):
        visual_rendering._read_verified_runtime_calibration(
            plan=plan.cameras[0],
            camera=camera,
            get_intrinsic=camera.get_intrinsic_matrix,
            get_extrinsic=camera.get_extrinsic_matrix,
        )


def test_installed_render_handle_checks_calibration_before_and_after_capture() -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 43
    )
    stable_cameras = tuple(
        (camera_plan, _StrictRuntimeCamera(camera_plan)) for camera_plan in plan.cameras
    )
    handle = visual_rendering._InstalledRenderHandle(
        _StrictRuntimeScene(),
        stable_cameras,
        _unobserved_renderer_api(plan),
    )
    views = handle.render_views()
    assert len(views) == 3
    assert handle.api_observation.runtime_camera_pose_dtype == "float32"
    assert handle.api_observation.raw_camera_pose_shape == "[1,7]"
    assert handle.api_observation.raw_extrinsic_matrix_shape == "[1,3,4]"

    changed_camera = _StrictRuntimeCamera(
        plan.cameras[0], mutate_on_take=True, cache_wrapper_pose=True
    )
    changed_handle = visual_rendering._InstalledRenderHandle(
        _StrictRuntimeScene(),
        ((plan.cameras[0], changed_camera),),
        _unobserved_renderer_api(plan),
    )
    with pytest.raises(
        ManiSkillVisualRenderingError,
        match="underlying world position differs at the bit level",
    ):
        changed_handle.render_views()

    picture_changed_camera = _StrictRuntimeCamera(
        plan.cameras[0], mutate_on_get_picture=True, cache_wrapper_pose=True
    )
    picture_changed_handle = visual_rendering._InstalledRenderHandle(
        _StrictRuntimeScene(),
        ((plan.cameras[0], picture_changed_camera),),
        _unobserved_renderer_api(plan),
    )
    with pytest.raises(
        ManiSkillVisualRenderingError,
        match="underlying world position differs at the bit level",
    ):
        picture_changed_handle.render_views()

    getter_changed_camera = _StrictRuntimeCamera(
        plan.cameras[0],
        mutate_on_second_extrinsic=True,
        cache_wrapper_pose=True,
    )
    getter_changed_handle = visual_rendering._InstalledRenderHandle(
        _StrictRuntimeScene(),
        ((plan.cameras[0], getter_changed_camera),),
        _unobserved_renderer_api(plan),
    )
    with pytest.raises(
        ManiSkillVisualRenderingError,
        match="underlying world position differs at the bit level",
    ):
        getter_changed_handle.render_views()

    signed_zero_camera = _StrictRuntimeCamera(
        plan.cameras[0], signed_zero_intrinsics_after_take=True
    )
    signed_zero_handle = visual_rendering._InstalledRenderHandle(
        _StrictRuntimeScene(),
        ((plan.cameras[0], signed_zero_camera),),
        _unobserved_renderer_api(plan),
    )
    with pytest.raises(
        ManiSkillVisualRenderingError,
        match="intrinsics differs from the content-bound camera plan",
    ):
        signed_zero_handle.render_views()


def test_installed_render_handle_compares_pre_post_public_extrinsic_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 45
    )
    camera_plan = plan.cameras[1]
    camera = _StrictRuntimeCamera(camera_plan)
    baseline = visual_rendering._read_verified_runtime_calibration(
        plan=camera_plan,
        camera=camera,
        get_intrinsic=camera.get_intrinsic_matrix,
        get_extrinsic=camera.get_extrinsic_matrix,
    )
    changed_raw = np.array(baseline.raw_public_extrinsics, copy=True)
    assert changed_raw[0, 0, 0] == 0.0
    changed_raw[0, 0, 0] = np.float32(-0.0)
    assert np.array_equal(changed_raw, baseline.raw_public_extrinsics)
    reads = iter(
        (
            baseline,
            replace(baseline, raw_public_extrinsics=changed_raw),
        )
    )
    monkeypatch.setattr(
        visual_rendering,
        "_read_verified_runtime_calibration",
        lambda **_kwargs: next(reads),
    )
    handle = visual_rendering._InstalledRenderHandle(
        _StrictRuntimeScene(),
        ((camera_plan, camera),),
        _unobserved_renderer_api(plan),
    )
    with pytest.raises(
        ManiSkillVisualRenderingError,
        match="pre/post-capture public extrinsics differs at the bit level",
    ):
        handle.render_views()


def test_installed_render_handle_compares_pre_post_intrinsic_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from latentguard.integrations.maniskill_pickcube import visual_rendering

    rig, configuration = _configuration()
    plan = build_pickcube_visual_render_plan(
        rig, configuration.domain("canonical"), configuration, 49
    )
    camera_plan = plan.cameras[0]
    camera = _StrictRuntimeCamera(camera_plan)
    baseline = visual_rendering._read_verified_runtime_calibration(
        plan=camera_plan,
        camera=camera,
        get_intrinsic=camera.get_intrinsic_matrix,
        get_extrinsic=camera.get_extrinsic_matrix,
    )
    changed = np.array(baseline.intrinsics, copy=True)
    assert changed[0, 1] == 0.0
    changed[0, 1] = np.float32(-0.0)
    reads = iter((baseline, replace(baseline, intrinsics=changed)))
    monkeypatch.setattr(
        visual_rendering,
        "_read_verified_runtime_calibration",
        lambda **_kwargs: next(reads),
    )
    handle = visual_rendering._InstalledRenderHandle(
        _StrictRuntimeScene(),
        ((camera_plan, camera),),
        _unobserved_renderer_api(plan),
    )

    with pytest.raises(
        ManiSkillVisualRenderingError,
        match="pre/post-capture intrinsics differs at the bit level",
    ):
        handle.render_views()
