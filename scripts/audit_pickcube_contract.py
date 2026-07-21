"""Audit and persist the native PickCube ACT policy contract."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from latentguard.control.serialization import write_atomic_json
from latentguard.integrations.maniskill_pickcube.compatibility import (
    load_compatibility_report,
    validate_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
)
from latentguard.integrations.maniskill_pickcube.serialization import (
    action_contract_from_compatibility,
    environment_settings_from_compatibility,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
)
from latentguard.policies.act import build_pickcube_act_contract
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.configuration import load_camera_rig_configuration


def _mapping(path: Path, *, context: str) -> Mapping[str, object]:
    source = path.absolute()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{context}: expected regular unlinked configuration")
    value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a JSON object")
    return cast(Mapping[str, object], value)


def _path(item: Mapping[str, object], field: str) -> Path:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"environment config {field}: expected non-empty path")
    return Path(value)


def _compatibility_path(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    value = os.environ.get("LATENTGUARD_PICKCUBE_COMPATIBILITY_REPORT")
    if not value:
        raise ValueError(
            "pass --compatibility-report or set "
            "LATENTGUARD_PICKCUBE_COMPATIBILITY_REPORT"
        )
    return Path(value)


def audit_contract(
    config_path: Path,
    compatibility_path: Path,
) -> Mapping[str, object]:
    """Validate one real reset and return the content-bound policy contract."""
    config = _mapping(config_path, context="environment config")
    expected = load_expected_contract(_path(config, "expected_contract"))
    report = load_compatibility_report(compatibility_path)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(_path(config, "action_layout"))
    rig = load_camera_rig_configuration(_path(config, "camera_rig"))
    if not isinstance(rig, PickCubeMultiViewRigV1):
        raise TypeError("camera rig loader returned an unexpected contract")
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding,
        coordinate_frame=layout.coordinate_frame,
    )
    factory = LazyManiSkillSourceEnvironmentFactory()
    environment = factory.create_environment(
        settings,
        action_contract,
        purpose="official_source_generation",
    )
    try:
        reset = getattr(environment, "reset", None)
        if not callable(reset):
            raise RuntimeError("PickCube environment lacks reset")
        reset(seed=0)
        joint_names, state = factory.extract_named_robot_state(environment)
        if state.shape != (2 * len(joint_names),):
            raise RuntimeError("proprioception shape drifted during runtime audit")
        return build_pickcube_act_contract(binding, layout, rig, joint_names)
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def main() -> int:
    """Run the exact contract audit and write a compact JSON artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = _mapping(args.config, context="environment config")
    output = args.output or _path(config, "output")
    payload = audit_contract(
        args.config,
        _compatibility_path(args.compatibility_report),
    )
    write_atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
