"""CPU-only binding tests for the native PickCube ACT contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from latentguard.integrations.maniskill_pickcube.compatibility import (
    load_compatibility_report,
    validate_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
)
from latentguard.policies.act.contracts import (
    PICKCUBE_ACTION_ORDER,
    PICKCUBE_ACTIVE_JOINT_NAMES,
    PickCubeActContractError,
    build_pickcube_act_contract,
)
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.configuration import load_camera_rig_configuration

_ROOT = Path(__file__).resolve().parents[1]
_INTEGRATION_CONFIG = _ROOT / "configs" / "integrations" / "maniskill_pickcube"
_REPORT = (
    _ROOT
    / "reports"
    / "m2c"
    / "20260715T080753Z_m2c-pickcube-trusted_aafe838_seed0"
    / "compatibility-trusted.json"
)


def _inputs() -> tuple[object, object, PickCubeMultiViewRigV1]:
    report = load_compatibility_report(_REPORT)
    expected = load_expected_contract(_INTEGRATION_CONFIG / "expected-contract-v1.json")
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(
        _INTEGRATION_CONFIG / "action-layout-v1.json"
    )
    rig = load_camera_rig_configuration(
        _ROOT / "configs" / "vision" / "m4a" / "camera-rig-v1.json"
    )
    assert isinstance(rig, PickCubeMultiViewRigV1)
    return binding, layout, rig


def test_pickcube_act_contract_binds_action_order_and_policy_inputs() -> None:
    binding, layout, rig = _inputs()
    contract = build_pickcube_act_contract(  # type: ignore[arg-type]
        binding,
        layout,
        rig,
        PICKCUBE_ACTIVE_JOINT_NAMES,
    )
    action = contract["action"]
    observation = contract["observation"]
    assert isinstance(action, dict)
    assert isinstance(observation, dict)
    assert action["dimension"] == 8
    assert action["order"] == list(PICKCUBE_ACTION_ORDER)
    assert action["clipping"] == "prohibited_fail_closed"
    assert observation["model_input_allowlist"] == [
        "observation.images.front_oblique",
        "observation.state",
    ]
    proprioception = observation["proprioception"]
    assert isinstance(proprioception, dict)
    assert proprioception["dimension"] == 18
    assert len(proprioception["fields"]) == 18
    assert str(contract["contract_digest"]).startswith("sha256:")


def test_pickcube_act_contract_rejects_joint_order_drift() -> None:
    binding, layout, rig = _inputs()
    drifted = tuple(reversed(PICKCUBE_ACTIVE_JOINT_NAMES))
    with pytest.raises(PickCubeActContractError, match="active joint order differs"):
        build_pickcube_act_contract(  # type: ignore[arg-type]
            binding,
            layout,
            rig,
            drifted,
        )
