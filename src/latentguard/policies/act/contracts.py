"""Exact PickCube observation, action, and environment contract for ACT."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import NoReturn

from latentguard.integrations.maniskill_pickcube.compatibility import (
    CompatibilityBinding,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    ManiSkillPickCubeActionLayout,
    validate_maniskill_pickcube_action_layout_binding,
)
from latentguard.replay.identity import canonical_json_bytes
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1

PICKCUBE_POLICY_CAMERA_IDS = ("front_oblique",)
PICKCUBE_ACTIVE_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)
PICKCUBE_ACTION_ORDER = (
    "panda_joint1_target_rad",
    "panda_joint2_target_rad",
    "panda_joint3_target_rad",
    "panda_joint4_target_rad",
    "panda_joint5_target_rad",
    "panda_joint6_target_rad",
    "panda_joint7_target_rad",
    "panda_gripper_normalized_target",
)
PICKCUBE_ACTION_UNITS = (
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "normalized_unitless",
)


class PickCubeActContractError(ValueError):
    """Raised when the P0 policy contract differs from the trusted runtime."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActContractError(f"{context}: {reason}")


def _digest(value: object) -> str:
    encoded = canonical_json_bytes(
        value,
        context="PickCubeActContractV1",
        reject_runtime_paths=True,
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _camera_contract(rig: PickCubeMultiViewRigV1) -> list[dict[str, object]]:
    by_id = {camera.camera_id: camera for camera in rig.cameras}
    if set(PICKCUBE_POLICY_CAMERA_IDS) - set(by_id):
        _fail("PickCubeActContract.camera", "policy camera is absent from fixed rig")
    return [
        {
            "camera_configuration_digest": by_id[camera_id].content_digest,
            "camera_id": camera_id,
            "dtype": "uint8",
            "raw_shape_hwc": [
                by_id[camera_id].height,
                by_id[camera_id].width,
                3,
            ],
            "tensor_shape_chw": [
                3,
                by_id[camera_id].height,
                by_id[camera_id].width,
            ],
        }
        for camera_id in PICKCUBE_POLICY_CAMERA_IDS
    ]


def build_pickcube_act_contract(
    binding: CompatibilityBinding,
    action_layout: ManiSkillPickCubeActionLayout,
    camera_rig: PickCubeMultiViewRigV1,
    observed_joint_names: Sequence[str],
) -> Mapping[str, object]:
    """Build the path-independent P0 contract after strict runtime binding."""
    if not isinstance(binding, CompatibilityBinding):
        _fail("PickCubeActContract.binding", "expected CompatibilityBinding")
    binding.require_trusted_replay_ready()
    if not isinstance(action_layout, ManiSkillPickCubeActionLayout):
        _fail("PickCubeActContract.action_layout", "unexpected layout type")
    validate_maniskill_pickcube_action_layout_binding(
        action_layout,
        binding,
        action_layout.m1_action_layout,
    )
    if not isinstance(camera_rig, PickCubeMultiViewRigV1):
        _fail("PickCubeActContract.camera_rig", "unexpected rig type")
    names = tuple(observed_joint_names)
    if names != PICKCUBE_ACTIVE_JOINT_NAMES:
        _fail(
            "PickCubeActContract.proprioception",
            f"active joint order differs from Panda contract: {names!r}",
        )

    report = binding.report
    if report.action_space.action_dimension != len(PICKCUBE_ACTION_ORDER):
        _fail("PickCubeActContract.action", "action width differs from fixed order")
    if tuple(component.name for component in report.controller.components) != (
        "arm",
        "gripper",
    ):
        _fail("PickCubeActContract.action", "controller component order drifted")

    proprio_fields = [
        *[f"qpos.{name}" for name in names],
        *[f"qvel.{name}" for name in names],
    ]
    contract: dict[str, object] = {
        "action": {
            "action_contract_digest": report.action_contract_digest,
            "bounds": {
                "lower": list(report.action_space.lower_bounds),
                "upper": list(report.action_space.upper_bounds),
            },
            "clipping": "prohibited_fail_closed",
            "controller_configuration_identity": (
                report.controller.configuration_identity
            ),
            "dimension": report.action_space.action_dimension,
            "dtype": report.action_space.dtype,
            "gripper_convention": {
                "closed_command": -1.0,
                "controller_physical_target_m": [-0.01, 0.04],
                "open_command": 1.0,
                "semantic": "normalized_mimic_joint_position_target_v1",
            },
            "order": list(PICKCUBE_ACTION_ORDER),
            "units": list(PICKCUBE_ACTION_UNITS),
        },
        "environment": {
            "compatibility_identity": report.compatibility_identity,
            "control_frequency_hz": report.control_frequency_hz,
            "control_mode": report.control_mode,
            "environment_id": report.environment_id,
            "episode_horizon_control_steps": 50,
            "failure_definition": (
                "terminal success false, timeout, unsafe state, or execution error"
            ),
            "mani_skill_version": report.mani_skill_version,
            "mplib_version": report.mplib_version,
            "num_envs": report.num_envs,
            "observation_mode": report.observation_mode,
            "reset_seed_semantic": "gymnasium_uint32_deterministic_scene_reset_v1",
            "robot_uid": report.robot_uid,
            "sapien_version": report.sapien_version,
            "simulation_frequency_hz": report.simulation_frequency_hz,
            "success_definition": "is_obj_placed and is_robot_static",
            "task_implementation_identity": (report.task_implementation.source_sha256),
            "termination_definition": (
                "task success is evaluated explicitly; Gymnasium truncates at "
                "50 control steps"
            ),
        },
        "observation": {
            "camera_rig_digest": camera_rig.rig_digest,
            "cameras": _camera_contract(camera_rig),
            "model_input_allowlist": [
                "observation.images.front_oblique",
                "observation.state",
            ],
            "privileged_input_denylist": [
                "cube_pose",
                "goal_pose",
                "contact_graph",
                "expert_phase",
                "success",
                "outcome",
            ],
            "proprioception": {
                "dimension": len(proprio_fields),
                "dtype": "float32",
                "fields": proprio_fields,
                "semantic": "panda_active_joint_qpos_then_qvel_named_v1",
            },
        },
        "schema_version": "pickcube-act-contract-v1",
    }
    contract["contract_digest"] = _digest(contract)
    return contract


__all__ = [
    "PICKCUBE_ACTION_ORDER",
    "PICKCUBE_ACTION_UNITS",
    "PICKCUBE_ACTIVE_JOINT_NAMES",
    "PICKCUBE_POLICY_CAMERA_IDS",
    "PickCubeActContractError",
    "build_pickcube_act_contract",
]
