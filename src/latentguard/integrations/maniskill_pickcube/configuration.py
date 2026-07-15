"""Strict, fail-closed configuration for the ManiSkill PickCube integration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn, cast

from latentguard.corruptions.layout import (
    ActionField,
    ActionLayout,
    ActionLayoutError,
    ActionSemantic,
)
from latentguard.models import JsonScalar
from latentguard.replay.identity import canonical_json_bytes

if TYPE_CHECKING:
    from .compatibility import CompatibilityBinding

EXPECTED_CONTRACT_SCHEMA_VERSION = "1.0"
REQUIRED_MANISKILL_VERSION = "3.0.1"
REQUIRED_ENVIRONMENT_ID = "PickCube-v1"
REQUIRED_ROBOT_UID = "panda"
REQUIRED_NUM_ENVS = 1
REQUIRED_OBSERVATION_MODE = "none"
REQUIRED_CONTROL_MODE = "pd_joint_pos"
REQUIRED_SIM_BACKEND_REQUEST = "gpu"
STATE_TREE_SEMANTIC = "maniskill_state_tree_v1"
TASK_CONTRACT_VERSION = "1.0.0"
TASK_CONTRACT_SEMANTIC = "maniskill_pickcube_task_v1"
PROGRESS_SEMANTIC = "pickcube_binary_completion_v0"
UNSAFE_SEMANTIC = "pickcube_cube_center_below_world_zero_v0"
PICKCUBE_STATE_VERIFICATION_SEMANTIC = "tolerance_verified_full_state_v1"
PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_STATE_TOLERANCE = PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE

_CONTRACT_STATUSES = frozenset({"probe_required", "verified"})
_EXPECTED_FIELDS = frozenset(
    {
        "schema_version",
        "contract_status",
        "mani_skill_version",
        "sapien_version",
        "mplib_version",
        "environment_id",
        "robot_uid",
        "num_envs",
        "observation_mode",
        "control_mode",
        "sim_backend_request",
        "resolved_sim_backend",
        "action_dimension",
        "action_dtype",
        "action_contract_digest",
        "control_frequency_hz",
        "simulation_frequency_hz",
        "solver_sha256",
        "task_sha256",
        "controller_configuration_identity",
        "state_tree_structure_digest",
        "compatibility_identity",
        "state_round_trip_tolerance",
    }
)
_DISCOVERED_FIELDS = (
    "sapien_version",
    "mplib_version",
    "observation_mode",
    "resolved_sim_backend",
    "action_dimension",
    "action_dtype",
    "action_contract_digest",
    "control_frequency_hz",
    "simulation_frequency_hz",
    "solver_sha256",
    "task_sha256",
    "controller_configuration_identity",
    "state_tree_structure_digest",
    "compatibility_identity",
)
_ACTION_LAYOUT_FIELDS = frozenset(
    {
        "schema_version",
        "action_dim",
        "fields",
        "description",
        "metadata",
        "coordinate_frame",
        "control_mode",
        "control_period_s",
        "action_contract_digest",
    }
)
_ACTION_FIELD_FIELDS = frozenset(
    {"name", "indices", "semantic", "units", "description", "metadata"}
)


class ManiSkillConfigurationError(ValueError):
    """Raised when a PickCube compatibility contract is malformed or unresolved."""


@dataclass(frozen=True, slots=True)
class ExpectedManiSkillPickCubeContract:
    """Checked-in expectations that bind a remote compatibility observation."""

    contract_status: str
    mani_skill_version: str
    sapien_version: str | None
    mplib_version: str | None
    environment_id: str
    robot_uid: str
    num_envs: int
    observation_mode: str | None
    control_mode: str
    sim_backend_request: str
    resolved_sim_backend: str | None
    action_dimension: int | None
    action_dtype: str | None
    action_contract_digest: str | None
    control_frequency_hz: float | None
    simulation_frequency_hz: float | None
    solver_sha256: str | None
    task_sha256: str | None
    controller_configuration_identity: str | None
    state_tree_structure_digest: str | None
    compatibility_identity: str | None
    state_round_trip_tolerance: float = DEFAULT_STATE_TOLERANCE
    schema_version: str = EXPECTED_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate fixed scope and all populated discovery fields."""
        _validate_expected_contract(self)

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        """Return fields that still require an observed remote value."""
        return tuple(name for name in _DISCOVERED_FIELDS if getattr(self, name) is None)

    @property
    def trusted_replay_ready(self) -> bool:
        """Return whether the checked-in contract can authorize trusted replay."""
        return self.contract_status == "verified" and not self.unresolved_fields

    def require_trusted_replay_ready(self) -> None:
        """Reject an initial probe template at every trusted replay boundary."""
        if self.trusted_replay_ready:
            return
        unresolved = ", ".join(self.unresolved_fields) or "contract_status"
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract: trusted replay is unavailable until the "
            f"locally checked-in contract is verified; unresolved: {unresolved}"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation in declared field order."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class ManiSkillPickCubeActionLayout:
    """Path-independent, probe-bound PickCube action-layout contract."""

    schema_version: str
    action_dim: int
    fields: tuple[ActionField, ...]
    description: str | None
    metadata: Mapping[str, JsonScalar]
    coordinate_frame: str
    control_mode: str
    control_period_s: float
    action_contract_digest: str
    _m1_action_layout: ActionLayout = dataclass_field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate the M1 layout and the simulator action contract together."""
        if self.control_mode != REQUIRED_CONTROL_MODE:
            raise ManiSkillConfigurationError(
                "ManiSkillPickCubeActionLayout.control_mode: expected "
                f"{REQUIRED_CONTROL_MODE!r}, got {self.control_mode!r}"
            )
        if (
            type(self.control_period_s) not in (int, float)
            or not math.isfinite(float(self.control_period_s))
            or float(self.control_period_s) <= 0.0
        ):
            raise ManiSkillConfigurationError(
                "ManiSkillPickCubeActionLayout.control_period_s: expected a "
                "positive finite number"
            )
        object.__setattr__(self, "control_period_s", float(self.control_period_s))
        if not _is_sha256_identity(self.action_contract_digest):
            raise ManiSkillConfigurationError(
                "ManiSkillPickCubeActionLayout.action_contract_digest: expected "
                "sha256:<64 lowercase hex>"
            )
        if (
            not isinstance(self.coordinate_frame, str)
            or not self.coordinate_frame.strip()
        ):
            raise ManiSkillConfigurationError(
                "ManiSkillPickCubeActionLayout.coordinate_frame: expected "
                "non-empty text"
            )
        try:
            m1_layout = ActionLayout(
                schema_version=self.schema_version,
                action_dim=self.action_dim,
                fields=tuple(self.fields),
                description=self.description,
                metadata=self.metadata,
            )
        except (ActionLayoutError, TypeError, ValueError) as exc:
            raise ManiSkillConfigurationError(
                f"ManiSkillPickCubeActionLayout: invalid M1 ActionLayout: {exc}"
            ) from exc
        for action_field in m1_layout.fields:
            if action_field.units is None:
                raise ManiSkillConfigurationError(
                    "ManiSkillPickCubeActionLayout.fields"
                    f"[{action_field.name!r}].units: must be explicit; use "
                    "'unspecified' when the unit cannot be verified"
                )
        covered = {
            index for action_field in m1_layout.fields for index in action_field.indices
        }
        missing = sorted(set(range(m1_layout.action_dim)) - covered)
        if missing:
            raise ManiSkillConfigurationError(
                "ManiSkillPickCubeActionLayout.fields: every action index must be "
                f"covered exactly once; missing {missing}"
            )
        object.__setattr__(self, "fields", m1_layout.fields)
        object.__setattr__(self, "metadata", m1_layout.metadata)
        object.__setattr__(self, "_m1_action_layout", m1_layout)
        try:
            canonical_json_bytes(
                self.to_dict(),
                context="ManiSkillPickCubeActionLayout",
                reject_runtime_paths=True,
            )
        except ValueError as exc:
            raise ManiSkillConfigurationError(
                "ManiSkillPickCubeActionLayout: semantic configuration must be "
                f"finite and path-independent: {exc}"
            ) from exc

    @property
    def m1_action_layout(self) -> ActionLayout:
        """Return the validated M1 layout used by corruption generation."""
        return self._m1_action_layout

    @property
    def action_layout_digest(self) -> str:
        """Return the stable digest of every checked-in layout contract field."""
        return compute_maniskill_pickcube_action_layout_digest(self)

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-native checked-in representation."""
        return {
            "schema_version": self.schema_version,
            "action_dim": self.action_dim,
            "fields": [
                {
                    "name": action_field.name,
                    "indices": list(action_field.indices),
                    "semantic": action_field.semantic.value,
                    "units": action_field.units,
                    "description": action_field.description,
                    "metadata": dict(action_field.metadata),
                }
                for action_field in self.fields
            ],
            "description": self.description,
            "metadata": dict(self.metadata),
            "coordinate_frame": self.coordinate_frame,
            "control_mode": self.control_mode,
            "control_period_s": self.control_period_s,
            "action_contract_digest": self.action_contract_digest,
        }

    def as_mapping(self) -> Mapping[str, object]:
        """Return a deeply immutable semantic mapping without runtime paths."""
        frozen = _freeze_json_value(self.to_dict())
        return cast(Mapping[str, object], frozen)


def _validate_expected_contract(contract: ExpectedManiSkillPickCubeContract) -> None:
    if contract.schema_version != EXPECTED_CONTRACT_SCHEMA_VERSION:
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract.schema_version: unsupported version "
            f"{contract.schema_version!r}"
        )
    if contract.contract_status not in _CONTRACT_STATUSES:
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract.contract_status: expected probe_required "
            "or verified"
        )
    fixed = (
        ("mani_skill_version", contract.mani_skill_version, REQUIRED_MANISKILL_VERSION),
        ("environment_id", contract.environment_id, REQUIRED_ENVIRONMENT_ID),
        ("robot_uid", contract.robot_uid, REQUIRED_ROBOT_UID),
        ("num_envs", contract.num_envs, REQUIRED_NUM_ENVS),
        ("control_mode", contract.control_mode, REQUIRED_CONTROL_MODE),
        (
            "sim_backend_request",
            contract.sim_backend_request,
            REQUIRED_SIM_BACKEND_REQUEST,
        ),
    )
    for name, value, expected in fixed:
        if type(value) is not type(expected) or value != expected:
            raise ManiSkillConfigurationError(
                f"ManiSkillPickCubeContract.{name}: expected {expected!r}, "
                f"got {value!r}"
            )
    if contract.observation_mode not in (None, REQUIRED_OBSERVATION_MODE):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract.observation_mode: M2C supports only 'none'"
        )
    for name in (
        "sapien_version",
        "mplib_version",
        "resolved_sim_backend",
        "action_dtype",
    ):
        value = getattr(contract, name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ManiSkillConfigurationError(
                f"ManiSkillPickCubeContract.{name}: expected non-empty text or null"
            )
    if contract.action_dimension is not None and (
        type(contract.action_dimension) is not int or contract.action_dimension <= 0
    ):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract.action_dimension: expected a positive "
            "integer or null"
        )
    for name in ("control_frequency_hz", "simulation_frequency_hz"):
        value = getattr(contract, name)
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ManiSkillConfigurationError(
                f"ManiSkillPickCubeContract.{name}: expected a positive finite "
                "number or null"
            )
    if (
        type(contract.state_round_trip_tolerance) not in (int, float)
        or not math.isfinite(float(contract.state_round_trip_tolerance))
        or float(contract.state_round_trip_tolerance)
        != PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
    ):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract.state_round_trip_tolerance: must equal the "
            "authorized fixed maximum absolute tolerance "
            f"{PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE!r}"
        )
    for name in (
        "action_contract_digest",
        "solver_sha256",
        "task_sha256",
        "controller_configuration_identity",
        "state_tree_structure_digest",
        "compatibility_identity",
    ):
        value = getattr(contract, name)
        if value is not None and not _is_sha256_identity(value):
            raise ManiSkillConfigurationError(
                f"ManiSkillPickCubeContract.{name}: expected sha256:<64 lowercase "
                "hex> or null"
            )
    if contract.contract_status == "verified" and contract.unresolved_fields:
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract.contract_status: verified contracts cannot "
            "contain unresolved discovery fields"
        )


def load_expected_contract(path: Path) -> ExpectedManiSkillPickCubeContract:
    """Load a duplicate-free, exact-field checked compatibility expectation."""
    raw = _read_strict_json(path)
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract: expected a JSON object"
        )
    item = cast(dict[str, object], raw)
    missing = sorted(_EXPECTED_FIELDS - set(item))
    unexpected = sorted(set(item) - _EXPECTED_FIELDS)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeContract: invalid fields (" + "; ".join(details) + ")"
        )
    try:
        return ExpectedManiSkillPickCubeContract(
            schema_version=_required_string(item, "schema_version"),
            contract_status=_required_string(item, "contract_status"),
            mani_skill_version=_required_string(item, "mani_skill_version"),
            sapien_version=_optional_string(item, "sapien_version"),
            mplib_version=_optional_string(item, "mplib_version"),
            environment_id=_required_string(item, "environment_id"),
            robot_uid=_required_string(item, "robot_uid"),
            num_envs=_required_integer(item, "num_envs"),
            observation_mode=_optional_string(item, "observation_mode"),
            control_mode=_required_string(item, "control_mode"),
            sim_backend_request=_required_string(item, "sim_backend_request"),
            resolved_sim_backend=_optional_string(item, "resolved_sim_backend"),
            action_dimension=_optional_integer(item, "action_dimension"),
            action_dtype=_optional_string(item, "action_dtype"),
            action_contract_digest=_optional_string(item, "action_contract_digest"),
            control_frequency_hz=_optional_number(item, "control_frequency_hz"),
            simulation_frequency_hz=_optional_number(item, "simulation_frequency_hz"),
            solver_sha256=_optional_string(item, "solver_sha256"),
            task_sha256=_optional_string(item, "task_sha256"),
            controller_configuration_identity=_optional_string(
                item, "controller_configuration_identity"
            ),
            state_tree_structure_digest=_optional_string(
                item, "state_tree_structure_digest"
            ),
            compatibility_identity=_optional_string(item, "compatibility_identity"),
            state_round_trip_tolerance=_required_number(
                item, "state_round_trip_tolerance"
            ),
        )
    except ManiSkillConfigurationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ManiSkillConfigurationError(
            f"ManiSkillPickCubeContract: invalid configuration: {exc}"
        ) from exc


def load_maniskill_pickcube_action_layout(
    path: Path,
) -> ManiSkillPickCubeActionLayout:
    """Load an exact-field, duplicate-free PickCube action-layout contract."""
    context = "ManiSkillPickCubeActionLayout.config"
    raw = _read_strict_json(path, context=context)
    item = _required_mapping(raw, context)
    _require_exact_fields(item, _ACTION_LAYOUT_FIELDS, context)
    raw_fields = _required_list(item, "fields", context)
    action_fields = tuple(
        _parse_maniskill_pickcube_action_field(value, index)
        for index, value in enumerate(raw_fields)
    )
    try:
        return ManiSkillPickCubeActionLayout(
            schema_version=_layout_string(item, "schema_version", context),
            action_dim=_layout_integer(item, "action_dim", context),
            fields=action_fields,
            description=_layout_optional_string(item, "description", context),
            metadata=_layout_metadata(item["metadata"], f"{context}.metadata"),
            coordinate_frame=_layout_string(item, "coordinate_frame", context),
            control_mode=_layout_string(item, "control_mode", context),
            control_period_s=_layout_number(item, "control_period_s", context),
            action_contract_digest=_layout_string(
                item,
                "action_contract_digest",
                context,
            ),
        )
    except ManiSkillConfigurationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ManiSkillConfigurationError(f"{context}: invalid layout: {exc}") from exc


def compute_maniskill_pickcube_action_layout_digest(
    layout: ManiSkillPickCubeActionLayout,
) -> str:
    """Compute a canonical SHA-256 identity independent of config file location."""
    if not isinstance(layout, ManiSkillPickCubeActionLayout):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeActionLayout.digest: expected "
            "ManiSkillPickCubeActionLayout"
        )
    encoded = canonical_json_bytes(
        layout.to_dict(),
        context="ManiSkillPickCubeActionLayout.digest",
        reject_runtime_paths=True,
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def validate_maniskill_pickcube_action_layout_binding(
    layout: ManiSkillPickCubeActionLayout,
    compatibility_binding: CompatibilityBinding,
    m1_action_layout: ActionLayout,
) -> str:
    """Bind one layout to a trusted probe report and the exact M1 layout in use."""
    if not isinstance(layout, ManiSkillPickCubeActionLayout):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeActionLayout.binding: expected a checked-in layout"
        )
    if not isinstance(m1_action_layout, ActionLayout):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeActionLayout.binding: expected an M1 ActionLayout"
        )

    # The local import avoids a configuration/compatibility import cycle while
    # still rejecting lookalike or fixture objects at the trust boundary.
    from .compatibility import CompatibilityBinding as RuntimeCompatibilityBinding

    if not isinstance(compatibility_binding, RuntimeCompatibilityBinding):
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeActionLayout.binding: expected CompatibilityBinding"
        )
    try:
        compatibility_binding.require_trusted_replay_ready()
    except ValueError as exc:
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeActionLayout.binding: compatibility probe is not "
            "trusted replay ready"
        ) from exc

    report = compatibility_binding.report
    mismatches: list[str] = []
    if layout.action_dim != report.action_space.action_dimension:
        mismatches.append(
            "action_dimension "
            f"layout={layout.action_dim!r} report="
            f"{report.action_space.action_dimension!r}"
        )
    if layout.action_contract_digest != report.action_contract_digest:
        mismatches.append(
            "action_contract_digest "
            f"layout={layout.action_contract_digest!r} "
            f"report={report.action_contract_digest!r}"
        )
    if layout.control_period_s != report.control_period_s:
        mismatches.append(
            "control_period_s "
            f"layout={layout.control_period_s!r} report={report.control_period_s!r}"
        )
    if layout.control_mode != report.control_mode:
        mismatches.append(
            "control_mode "
            f"layout={layout.control_mode!r} report={report.control_mode!r}"
        )
    reported_components = tuple(
        (component.name, component.indices)
        for component in report.controller.components
    )
    layout_components = tuple(
        (action_field.name, action_field.indices)
        for action_field in layout.m1_action_layout.fields
    )
    if layout_components != reported_components:
        mismatches.append(
            "fields name/indices differ from probe controller.components "
            f"layout={layout_components!r} report={reported_components!r}"
        )
    for action_field in layout.m1_action_layout.fields:
        if action_field.semantic is not ActionSemantic.UNSPECIFIED:
            mismatches.append(
                f"fields[{action_field.name!r}].semantic must be 'unspecified' "
                "because the probe did not attest semantic meaning"
            )
        if action_field.units != "unspecified":
            mismatches.append(
                f"fields[{action_field.name!r}].units must be 'unspecified' "
                "because the probe did not attest physical units"
            )
    if layout.coordinate_frame != "unspecified":
        mismatches.append(
            "coordinate_frame must be 'unspecified' because the probe did not "
            "attest a coordinate frame"
        )
    if _action_layout_mapping(layout.m1_action_layout) != _action_layout_mapping(
        m1_action_layout
    ):
        mismatches.append("M1 ActionLayout differs from checked-in layout")
    if mismatches:
        raise ManiSkillConfigurationError(
            "ManiSkillPickCubeActionLayout.binding: mismatch ("
            + "; ".join(mismatches)
            + ")"
        )
    return layout.action_layout_digest


def _parse_maniskill_pickcube_action_field(
    value: object,
    index: int,
) -> ActionField:
    context = f"ManiSkillPickCubeActionLayout.config.fields[{index}]"
    item = _required_mapping(value, context)
    _require_exact_fields(item, _ACTION_FIELD_FIELDS, context)
    raw_indices = _required_list(item, "indices", context)
    indices = tuple(
        _layout_integer_value(raw_index, f"{context}.indices[{position}]")
        for position, raw_index in enumerate(raw_indices)
    )
    semantic_text = _layout_string(item, "semantic", context)
    try:
        semantic = ActionSemantic(semantic_text)
    except ValueError as exc:
        raise ManiSkillConfigurationError(
            f"{context}.semantic: unsupported semantic {semantic_text!r}"
        ) from exc
    try:
        return ActionField(
            name=_layout_string(item, "name", context),
            indices=indices,
            semantic=semantic,
            units=_layout_string(item, "units", context),
            description=_layout_optional_string(item, "description", context),
            metadata=_layout_metadata(item["metadata"], f"{context}.metadata"),
        )
    except (ActionLayoutError, TypeError, ValueError) as exc:
        raise ManiSkillConfigurationError(f"{context}: invalid field: {exc}") from exc


def _action_layout_mapping(layout: ActionLayout) -> dict[str, object]:
    return {
        "schema_version": layout.schema_version,
        "action_dim": layout.action_dim,
        "fields": [
            {
                "name": action_field.name,
                "indices": list(action_field.indices),
                "semantic": action_field.semantic.value,
                "units": action_field.units,
                "description": action_field.description,
                "metadata": dict(action_field.metadata),
            }
            for action_field in layout.fields
        ],
        "description": layout.description,
        "metadata": dict(layout.metadata),
    }


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _required_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManiSkillConfigurationError(f"{context}: expected a JSON object")
    return cast(dict[str, object], value)


def _require_exact_fields(
    item: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    missing = sorted(expected - set(item))
    unexpected = sorted(set(item) - expected)
    if not missing and not unexpected:
        return
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ManiSkillConfigurationError(
        f"{context}: invalid fields (" + "; ".join(details) + ")"
    )


def _required_list(
    item: Mapping[str, object],
    name: str,
    context: str,
) -> list[object]:
    value = item[name]
    if not isinstance(value, list):
        raise ManiSkillConfigurationError(f"{context}.{name}: expected a JSON array")
    return cast(list[object], value)


def _layout_string(
    item: Mapping[str, object],
    name: str,
    context: str,
) -> str:
    value = item[name]
    if not isinstance(value, str) or not value.strip():
        raise ManiSkillConfigurationError(f"{context}.{name}: expected non-empty text")
    return value


def _layout_optional_string(
    item: Mapping[str, object],
    name: str,
    context: str,
) -> str | None:
    if item[name] is None:
        return None
    return _layout_string(item, name, context)


def _layout_integer(
    item: Mapping[str, object],
    name: str,
    context: str,
) -> int:
    return _layout_integer_value(item[name], f"{context}.{name}")


def _layout_integer_value(value: object, context: str) -> int:
    if type(value) is not int:
        raise ManiSkillConfigurationError(f"{context}: expected an integer")
    return value


def _layout_number(
    item: Mapping[str, object],
    name: str,
    context: str,
) -> float:
    value = item[name]
    if type(value) not in (int, float):
        raise ManiSkillConfigurationError(f"{context}.{name}: expected a finite number")
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)):
        raise ManiSkillConfigurationError(f"{context}.{name}: expected a finite number")
    return float(numeric)


def _layout_metadata(value: object, context: str) -> dict[str, JsonScalar]:
    item = _required_mapping(value, context)
    result: dict[str, JsonScalar] = {}
    for key, raw in item.items():
        if raw is not None and type(raw) not in (str, int, float, bool):
            raise ManiSkillConfigurationError(
                f"{context}.{key}: expected a JSON scalar"
            )
        if isinstance(raw, float) and not math.isfinite(raw):
            raise ManiSkillConfigurationError(f"{context}.{key}: expected finite data")
        result[key] = cast(JsonScalar, raw)
    return result


def _read_strict_json(
    path: Path,
    *,
    context: str = "ManiSkillPickCubeContract.config",
) -> object:
    config_path = Path(path)
    try:
        if config_path.is_symlink() or not config_path.is_file():
            raise ManiSkillConfigurationError(
                f"{context}: missing or unsafe regular file"
            )
        if config_path.stat().st_nlink != 1:
            raise ManiSkillConfigurationError(
                f"{context}: hard-linked files are unsupported"
            )
        return cast(
            object,
            json.loads(
                config_path.read_text(encoding="utf-8"),
                parse_constant=lambda value: _reject_json_constant(
                    value,
                    context=context,
                ),
                object_pairs_hook=lambda pairs: _reject_duplicate_fields(
                    pairs,
                    context=context,
                ),
            ),
        )
    except ManiSkillConfigurationError:
        raise
    except Exception as exc:
        raise ManiSkillConfigurationError(
            f"{context}: could not read safely: {exc}"
        ) from exc


def _required_string(item: Mapping[str, object], name: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value.strip():
        raise ManiSkillConfigurationError(
            f"ManiSkillPickCubeContract.{name}: expected non-empty text"
        )
    return value


def _optional_string(item: Mapping[str, object], name: str) -> str | None:
    value = item[name]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ManiSkillConfigurationError(
            f"ManiSkillPickCubeContract.{name}: expected non-empty text or null"
        )
    return value


def _required_integer(item: Mapping[str, object], name: str) -> int:
    value = item[name]
    if type(value) is not int:
        raise ManiSkillConfigurationError(
            f"ManiSkillPickCubeContract.{name}: expected an integer"
        )
    return value


def _optional_integer(item: Mapping[str, object], name: str) -> int | None:
    value = item[name]
    return None if value is None else _required_integer(item, name)


def _required_number(item: Mapping[str, object], name: str) -> float:
    value = item[name]
    if type(value) not in (int, float):
        raise ManiSkillConfigurationError(
            f"ManiSkillPickCubeContract.{name}: expected a finite number"
        )
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)):
        raise ManiSkillConfigurationError(
            f"ManiSkillPickCubeContract.{name}: expected a finite number"
        )
    return float(numeric)


def _optional_number(item: Mapping[str, object], name: str) -> float | None:
    return None if item[name] is None else _required_number(item, name)


def _is_sha256_identity(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value.removeprefix("sha256:")
    return len(suffix) == 64 and all(
        character in "0123456789abcdef" for character in suffix
    )


def _reject_json_constant(value: str, *, context: str) -> NoReturn:
    raise ManiSkillConfigurationError(f"{context}: {value!r} is unsupported")


def _reject_duplicate_fields(
    pairs: list[tuple[str, object]],
    *,
    context: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManiSkillConfigurationError(f"{context}: duplicate field {key!r}")
        result[key] = value
    return result


__all__ = [
    "DEFAULT_STATE_TOLERANCE",
    "EXPECTED_CONTRACT_SCHEMA_VERSION",
    "ExpectedManiSkillPickCubeContract",
    "ManiSkillPickCubeActionLayout",
    "ManiSkillConfigurationError",
    "PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE",
    "PICKCUBE_STATE_VERIFICATION_SEMANTIC",
    "PROGRESS_SEMANTIC",
    "REQUIRED_CONTROL_MODE",
    "REQUIRED_ENVIRONMENT_ID",
    "REQUIRED_MANISKILL_VERSION",
    "REQUIRED_NUM_ENVS",
    "REQUIRED_OBSERVATION_MODE",
    "REQUIRED_ROBOT_UID",
    "REQUIRED_SIM_BACKEND_REQUEST",
    "STATE_TREE_SEMANTIC",
    "TASK_CONTRACT_VERSION",
    "TASK_CONTRACT_SEMANTIC",
    "UNSAFE_SEMANTIC",
    "compute_maniskill_pickcube_action_layout_digest",
    "load_expected_contract",
    "load_maniskill_pickcube_action_layout",
    "validate_maniskill_pickcube_action_layout_binding",
]
