"""Versioned public-state vector contract for the PickCube verifier.

This module is deliberately simulator-free.  A runtime integration must first
extract the named values through verified public APIs and then pass detached
NumPy arrays to :func:`build_pickcube_verifier_state_v1`.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

PICKCUBE_VERIFIER_STATE_SEMANTIC = "PickCubeVerifierStateV1"
PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION = "1.0"
PICKCUBE_VERIFIER_STATE_DTYPE = np.dtype("<f4").str

_POSE_POSITION_COMPONENTS = ("x", "y", "z")
_POSE_QUATERNION_COMPONENTS = ("w", "x", "y", "z")
_NON_JOINT_COMPONENT_NAMES = (
    *(f"tcp/position/{axis}" for axis in _POSE_POSITION_COMPONENTS),
    *(f"tcp/quaternion/{axis}" for axis in _POSE_QUATERNION_COMPONENTS),
    *(f"cube/position/{axis}" for axis in _POSE_POSITION_COMPONENTS),
    *(f"cube/quaternion/{axis}" for axis in _POSE_QUATERNION_COMPONENTS),
    *(f"goal/position/{axis}" for axis in _POSE_POSITION_COMPONENTS),
    "task/is_grasped",
    "task/is_obj_placed",
    "task/is_robot_static",
)


class PickCubeVerifierStateError(ValueError):
    """Raised when the public PickCube state-vector contract is violated."""


def _validate_joint_names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise PickCubeVerifierStateError("joint names must be an ordered sequence")
    try:
        names = tuple(value)
    except TypeError as exc:
        raise PickCubeVerifierStateError(
            "joint names must be an ordered sequence"
        ) from exc
    if not names:
        raise PickCubeVerifierStateError("joint names must not be empty")
    for index, name in enumerate(names):
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None
        ):
            raise PickCubeVerifierStateError(
                f"joint name at index {index} is not canonical text"
            )
    if len(set(names)) != len(names):
        raise PickCubeVerifierStateError("joint names must be unique")
    return names


def _freeze_float32_vector(value: object, *, context: str) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise PickCubeVerifierStateError(f"{context} must be a numpy.ndarray")
    if value.dtype != np.dtype(PICKCUBE_VERIFIER_STATE_DTYPE):
        raise PickCubeVerifierStateError(
            f"{context} must use dtype {PICKCUBE_VERIFIER_STATE_DTYPE}"
        )
    if value.ndim != 1:
        raise PickCubeVerifierStateError(f"{context} must be rank one")
    if not bool(np.all(np.isfinite(value))):
        raise PickCubeVerifierStateError(f"{context} must contain only finite values")
    detached = np.array(value, copy=True, order="C", subok=False)
    raw = detached.tobytes(order="C")
    return np.frombuffer(raw, dtype=detached.dtype).reshape(detached.shape)


def _public_float_vector(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    context: str,
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise PickCubeVerifierStateError(f"{context} must be a numpy.ndarray")
    if value.dtype.hasobject or not np.issubdtype(value.dtype, np.floating):
        raise PickCubeVerifierStateError(f"{context} must use a floating dtype")
    if value.shape != expected_shape:
        raise PickCubeVerifierStateError(
            f"{context} must have shape {expected_shape}, got {value.shape}"
        )
    if not bool(np.all(np.isfinite(value))):
        raise PickCubeVerifierStateError(f"{context} must contain only finite values")
    return np.array(value, copy=True, order="C", subok=False)


def _strict_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise PickCubeVerifierStateError(f"{context} must be a boolean")
    return bool(value)


@dataclass(frozen=True, slots=True)
class PickCubeVerifierStateSchemaV1:
    """Content-bound component schema derived from verified active-joint names."""

    joint_names: tuple[str, ...]
    semantic: str = PICKCUBE_VERIFIER_STATE_SEMANTIC
    dtype: str = PICKCUBE_VERIFIER_STATE_DTYPE
    schema_version: str = PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze names and reject any semantic, dtype, or version drift."""
        object.__setattr__(self, "joint_names", _validate_joint_names(self.joint_names))
        if self.semantic != PICKCUBE_VERIFIER_STATE_SEMANTIC:
            raise PickCubeVerifierStateError("verifier-state semantic mismatch")
        if self.dtype != PICKCUBE_VERIFIER_STATE_DTYPE:
            raise PickCubeVerifierStateError("verifier-state dtype must be float32")
        if self.schema_version != PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION:
            raise PickCubeVerifierStateError(
                "unsupported verifier-state schema version"
            )

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return the complete fixed component order for this joint inventory."""
        return (
            *(f"panda/joint_position/{name}" for name in self.joint_names),
            *(f"panda/joint_velocity/{name}" for name in self.joint_names),
            *_NON_JOINT_COMPONENT_NAMES,
        )

    @property
    def shape(self) -> tuple[int]:
        """Return the fixed rank-one vector shape."""
        return (len(self.component_names),)

    @property
    def dimension(self) -> int:
        """Return the fixed component count."""
        return self.shape[0]

    def as_mapping(self) -> Mapping[str, object]:
        """Return the canonical schema payload used for content identity."""
        return MappingProxyType(
            {
                "component_names": self.component_names,
                "dtype": self.dtype,
                "joint_names": self.joint_names,
                "schema_version": self.schema_version,
                "semantic": self.semantic,
                "shape": self.shape,
            }
        )

    @property
    def schema_digest(self) -> str:
        """Return the path-independent SHA-256 schema identity."""
        digest = hashlib.sha256(canonical_json_bytes(self.as_mapping())).hexdigest()
        return f"sha256:{digest}"


@dataclass(frozen=True, slots=True, eq=False)
class PickCubeVerifierStateV1:
    """One immutable finite float32 state vector and its explicit schema."""

    schema: PickCubeVerifierStateSchemaV1
    values: NDArray[Any]

    def __post_init__(self) -> None:
        """Detach values and require exact dtype, shape, and schema coverage."""
        if not isinstance(self.schema, PickCubeVerifierStateSchemaV1):
            raise PickCubeVerifierStateError(
                "verifier state requires PickCubeVerifierStateSchemaV1"
            )
        values = _freeze_float32_vector(self.values, context="verifier state values")
        if values.shape != self.schema.shape:
            raise PickCubeVerifierStateError(
                "verifier state values do not match the complete component schema"
            )
        object.__setattr__(self, "values", values)

    @property
    def semantic(self) -> str:
        """Return the versioned state-vector semantic."""
        return self.schema.semantic

    @property
    def schema_digest(self) -> str:
        """Return the bound component-schema identity."""
        return self.schema.schema_digest

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return names aligned one-to-one with vector values."""
        return self.schema.component_names

    @property
    def content_digest(self) -> str:
        """Hash schema, dtype, shape, and exact float32 bytes."""
        payload = {
            "content_sha256": hashlib.sha256(
                self.values.tobytes(order="C")
            ).hexdigest(),
            "dtype": self.values.dtype.str,
            "schema_digest": self.schema_digest,
            "semantic": self.semantic,
            "shape": self.values.shape,
        }
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        return f"sha256:{digest}"


def build_pickcube_verifier_state_v1(
    *,
    joint_names: Sequence[str],
    qpos: NDArray[Any],
    qvel: NDArray[Any],
    tcp_position: NDArray[Any],
    tcp_quaternion: NDArray[Any],
    cube_position: NDArray[Any],
    cube_quaternion: NDArray[Any],
    goal_position: NDArray[Any],
    is_grasped: bool,
    is_obj_placed: bool,
    is_robot_static: bool,
) -> PickCubeVerifierStateV1:
    """Build V1 from verified public values without normalization or defaults.

    Quaternion inputs are interpreted in explicit ``(w, x, y, z)`` order and
    are copied exactly before the documented float32 projection.  They are not
    normalized, sign-canonicalized, or otherwise repaired.
    """
    schema = PickCubeVerifierStateSchemaV1(
        joint_names=_validate_joint_names(joint_names)
    )
    joint_shape = (len(schema.joint_names),)
    components = (
        _public_float_vector(qpos, expected_shape=joint_shape, context="qpos"),
        _public_float_vector(qvel, expected_shape=joint_shape, context="qvel"),
        _public_float_vector(tcp_position, expected_shape=(3,), context="tcp position"),
        _public_float_vector(
            tcp_quaternion, expected_shape=(4,), context="tcp quaternion"
        ),
        _public_float_vector(
            cube_position, expected_shape=(3,), context="cube position"
        ),
        _public_float_vector(
            cube_quaternion, expected_shape=(4,), context="cube quaternion"
        ),
        _public_float_vector(
            goal_position, expected_shape=(3,), context="goal position"
        ),
        np.asarray(
            [
                float(_strict_bool(is_grasped, context="is_grasped")),
                float(_strict_bool(is_obj_placed, context="is_obj_placed")),
                float(_strict_bool(is_robot_static, context="is_robot_static")),
            ],
            dtype=np.float64,
        ),
    )
    joined = np.concatenate(components)
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = joined.astype(np.dtype(PICKCUBE_VERIFIER_STATE_DTYPE), copy=True)
    except FloatingPointError as exc:
        raise PickCubeVerifierStateError(
            "public state values cannot be represented as finite float32"
        ) from exc
    if not all(math.isfinite(float(value)) for value in values):
        raise PickCubeVerifierStateError(
            "public state values cannot be represented as finite float32"
        )
    return PickCubeVerifierStateV1(schema=schema, values=values)


__all__ = [
    "PICKCUBE_VERIFIER_STATE_DTYPE",
    "PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION",
    "PICKCUBE_VERIFIER_STATE_SEMANTIC",
    "PickCubeVerifierStateError",
    "PickCubeVerifierStateSchemaV1",
    "PickCubeVerifierStateV1",
    "build_pickcube_verifier_state_v1",
]
