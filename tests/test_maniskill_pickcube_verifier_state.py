from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np
import pytest

from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_DTYPE,
    PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY,
    PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION,
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
    PickCubeVerifierStateError,
    PickCubeVerifierStateSchemaV1,
    PickCubeVerifierStateV1,
    build_pickcube_verifier_state_v1,
)


def _public_values() -> dict[str, object]:
    return {
        "joint_names": ("joint_a", "joint_b"),
        "qpos": np.array([0.1, 0.2], dtype=np.float64),
        "qvel": np.array([-0.1, -0.2], dtype=np.float64),
        "tcp_position": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "tcp_quaternion": np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "cube_position": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        "cube_quaternion": np.array([1.0, 0.1, 0.2, 0.3], dtype=np.float32),
        "goal_position": np.array([7.0, 8.0, 9.0], dtype=np.float32),
        "is_grasped": True,
        "is_obj_placed": False,
        "is_robot_static": True,
    }


def _build(overrides: Mapping[str, object] | None = None) -> PickCubeVerifierStateV1:
    values = _public_values()
    values.update(dict(overrides or {}))
    return build_pickcube_verifier_state_v1(**values)  # type: ignore[arg-type]


def test_verifier_state_has_explicit_complete_component_order() -> None:
    state = _build()

    assert state.semantic == PICKCUBE_VERIFIER_STATE_SEMANTIC
    assert (
        state.schema.extraction_boundary == PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY
    )
    assert state.schema.schema_version == PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION
    assert state.values.dtype.str == PICKCUBE_VERIFIER_STATE_DTYPE
    assert state.values.shape == (24,)
    assert state.component_names[:4] == (
        "panda/joint_position/joint_a",
        "panda/joint_position/joint_b",
        "panda/joint_velocity/joint_a",
        "panda/joint_velocity/joint_b",
    )
    assert state.component_names[-3:] == (
        "task/is_grasped",
        "task/is_obj_placed",
        "task/is_robot_static",
    )
    assert state.values[-3:].tolist() == [1.0, 0.0, 1.0]


def test_verifier_state_preserves_quaternion_values_without_repair() -> None:
    state = _build()

    assert state.values[7:11].tolist() == [2.0, 0.0, 0.0, 0.0]
    assert state.values[14:18].tolist() == pytest.approx([1.0, 0.1, 0.2, 0.3])


def test_schema_and_vector_digests_are_deterministic_and_content_bound() -> None:
    first = _build()
    second = _build()
    changed_values = _build({"qpos": np.array([0.2, 0.2], dtype=np.float64)})
    reordered = PickCubeVerifierStateSchemaV1(joint_names=("joint_b", "joint_a"))

    assert first.schema_digest == second.schema_digest
    assert first.content_digest == second.content_digest
    assert changed_values.schema_digest == first.schema_digest
    assert changed_values.content_digest != first.content_digest
    assert reordered.schema_digest != first.schema_digest


def test_schema_rejects_extraction_boundary_or_version_drift() -> None:
    with pytest.raises(PickCubeVerifierStateError, match="extraction boundary"):
        PickCubeVerifierStateSchemaV1(
            joint_names=("joint_a", "joint_b"),
            extraction_boundary="source_time_before_archive_v0",
        )
    with pytest.raises(PickCubeVerifierStateError, match="schema version"):
        PickCubeVerifierStateSchemaV1(
            joint_names=("joint_a", "joint_b"), schema_version="1.0"
        )


def test_builder_detaches_every_public_input() -> None:
    values = _public_values()
    state = build_pickcube_verifier_state_v1(**values)  # type: ignore[arg-type]
    qpos = values["qpos"]
    assert isinstance(qpos, np.ndarray)
    qpos[...] = 99.0

    assert state.values[:2].tolist() == pytest.approx([0.1, 0.2])
    with pytest.raises(ValueError):
        state.values[0] = 3.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qpos", np.array([0.0], dtype=np.float32)),
        ("qvel", np.array([0, 1], dtype=np.int64)),
        ("tcp_position", np.array([[0.0, 0.0, 0.0]], dtype=np.float32)),
        ("tcp_quaternion", np.array([1.0, 0.0, np.nan, 0.0], dtype=np.float32)),
        ("cube_position", [0.0, 0.0, 0.0]),
        ("goal_position", np.array([0.0, np.inf, 0.0], dtype=np.float32)),
        ("is_grasped", np.bool_(True)),
    ],
)
def test_builder_fails_closed_on_missing_shape_dtype_finite_or_bool_contract(
    field: str, value: object
) -> None:
    with pytest.raises(PickCubeVerifierStateError):
        _build({field: value})


def test_builder_rejects_duplicate_or_noncanonical_joint_names() -> None:
    with pytest.raises(PickCubeVerifierStateError, match="unique"):
        _build({"joint_names": ("joint_a", "joint_a")})
    with pytest.raises(PickCubeVerifierStateError, match="canonical"):
        _build({"joint_names": (" joint_a", "joint_b")})


def test_builder_rejects_float32_overflow_instead_of_emitting_infinity() -> None:
    with pytest.raises(PickCubeVerifierStateError, match="finite float32"):
        _build({"qpos": np.array([1e100, 0.0], dtype=np.float64)})


def test_vector_model_rejects_wrong_dtype_or_incomplete_shape() -> None:
    schema = PickCubeVerifierStateSchemaV1(joint_names=("joint_a", "joint_b"))
    with pytest.raises(PickCubeVerifierStateError, match="dtype"):
        PickCubeVerifierStateV1(
            schema=schema, values=np.zeros(schema.shape, dtype=np.float64)
        )
    with pytest.raises(PickCubeVerifierStateError, match="component schema"):
        PickCubeVerifierStateV1(
            schema=schema, values=np.zeros((schema.dimension - 1,), dtype=np.float32)
        )


def test_builder_api_cannot_accept_future_success_outcome() -> None:
    signature = inspect.signature(build_pickcube_verifier_state_v1)

    assert "success" not in signature.parameters
    assert set(signature.parameters) == {
        "joint_names",
        "qpos",
        "qvel",
        "tcp_position",
        "tcp_quaternion",
        "cube_position",
        "cube_quaternion",
        "goal_position",
        "is_grasped",
        "is_obj_placed",
        "is_robot_static",
    }


def test_schema_mapping_is_canonical_and_read_only() -> None:
    schema = PickCubeVerifierStateSchemaV1(joint_names=("joint_a", "joint_b"))
    mapping = schema.as_mapping()

    assert mapping["shape"] == (24,)
    assert mapping["extraction_boundary"] == PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY
    with pytest.raises(TypeError):
        mapping["semantic"] = "changed"  # type: ignore[index]
