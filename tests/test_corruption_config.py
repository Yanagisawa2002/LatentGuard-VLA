from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latentguard.corruptions.config import CorruptionPlanError, load_corruption_plan
from latentguard.corruptions.layout import ActionSemantic
from latentguard.models import ActionChunk

_SMOKE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "corruptions" / "m1-smoke.json"
)
_EXPECTED_NAMES = (
    "additive_gaussian_noise",
    "constant_bias",
    "temporal_field_shift",
    "segment_hold",
    "segment_zeroing",
    "local_temporal_permutation",
)


def _valid_config() -> dict[str, Any]:
    return json.loads(_SMOKE_CONFIG.read_text(encoding="utf-8"))


def _write_config(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _action() -> ActionChunk:
    return ActionChunk(
        actions=np.arange(8 * 7, dtype=np.float32).reshape(8, 7),
        coordinate_frame="robot_base",
        control_period_s=0.1,
    )


def test_smoke_config_loads_in_declared_order_and_resolves_completely() -> None:
    plan = load_corruption_plan(_SMOKE_CONFIG)

    assert plan.schema_version == "1.0"
    assert plan.action_layout.action_dim == 7
    assert tuple(field.name for field in plan.action_layout.fields) == (
        "translation",
        "rotation",
        "gripper",
    )
    assert tuple(field.semantic for field in plan.action_layout.fields) == (
        ActionSemantic.TRANSLATION,
        ActionSemantic.ROTATION,
        ActionSemantic.GRIPPER,
    )
    assert tuple(corruption.name for corruption in plan.corruptions) == _EXPECTED_NAMES

    resolved = tuple(
        dict(corruption.resolved_parameters_for(_action(), plan.action_layout))
        for corruption in plan.corruptions
    )
    assert resolved == (
        {
            "target_indices": (0, 1, 2),
            "standard_deviation": 0.05,
            "mean": 0.0,
        },
        {"target_indices": (3, 4, 5), "bias": (0.01, -0.01, 0.02)},
        {
            "target_field": "gripper",
            "target_indices": (6,),
            "shift_steps": 1,
            "fill_policy": "edge",
        },
        {"start_step": 2, "end_step": 4, "target_indices": (3, 4, 5)},
        {"start_step": 4, "end_step": 6, "target_indices": (6,)},
        {"start_step": 1, "end_step": 5, "target_indices": tuple(range(7))},
    )


def test_loader_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    config = _valid_config()
    config["unexpected"] = True

    with pytest.raises(CorruptionPlanError, match="unexpected unexpected"):
        load_corruption_plan(_write_config(tmp_path / "unknown-top.json", config))


def test_loader_rejects_unknown_corruption_parameter(tmp_path: Path) -> None:
    config = _valid_config()
    config["corruptions"][0]["parameters"]["clip"] = True

    with pytest.raises(CorruptionPlanError, match="unknown clip"):
        load_corruption_plan(_write_config(tmp_path / "unknown-parameter.json", config))


def test_loader_rejects_unknown_corruption_type(tmp_path: Path) -> None:
    config = _valid_config()
    config["corruptions"][0]["type"] = "not_registered"

    with pytest.raises(
        CorruptionPlanError, match="unknown corruption 'not_registered'"
    ):
        load_corruption_plan(_write_config(tmp_path / "unknown-type.json", config))


@pytest.mark.parametrize("invalid_indices", [[-1], [7], [1, 1], [True]])
def test_loader_rejects_invalid_explicit_indices(
    tmp_path: Path, invalid_indices: list[object]
) -> None:
    config = _valid_config()
    parameters = config["corruptions"][1]["parameters"]
    parameters.pop("target_indices")
    parameters["target_indices"] = invalid_indices

    with pytest.raises(CorruptionPlanError, match="target_indices"):
        load_corruption_plan(_write_config(tmp_path / "invalid-indices.json", config))


def test_loader_rejects_layout_field_index_outside_action_dim(tmp_path: Path) -> None:
    config = _valid_config()
    config["action_layout"]["fields"][2]["indices"] = [7]

    with pytest.raises(CorruptionPlanError, match=r"outside \[0, 7\)"):
        load_corruption_plan(
            _write_config(tmp_path / "invalid-layout-index.json", config)
        )


@pytest.mark.parametrize(
    ("corruption_index", "updates", "message"),
    [
        (2, {"shift_steps": 0}, "zero shift"),
        (3, {"start_step": 0}, "must be at least 1"),
        (3, {"start_step": 4, "end_step": 4}, "greater than start_step"),
        (4, {"start_step": 6, "end_step": 6}, "greater than start_step"),
        (5, {"start_step": 2, "end_step": 3}, "at least two control steps"),
    ],
)
def test_loader_rejects_invalid_transform_ranges(
    tmp_path: Path,
    corruption_index: int,
    updates: dict[str, object],
    message: str,
) -> None:
    config = _valid_config()
    config["corruptions"][corruption_index]["parameters"].update(updates)

    with pytest.raises(CorruptionPlanError, match=message):
        load_corruption_plan(_write_config(tmp_path / "invalid-range.json", config))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_loader_rejects_nonfinite_json_constants(tmp_path: Path, constant: str) -> None:
    text = _SMOKE_CONFIG.read_text(encoding="utf-8").replace("0.05", constant, 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(CorruptionPlanError, match="unsupported"):
        load_corruption_plan(path)


@pytest.mark.parametrize("selection_key", ["target_fields", "target_indices"])
def test_loader_rejects_empty_required_target_selection(
    tmp_path: Path, selection_key: str
) -> None:
    config = _valid_config()
    parameters = config["corruptions"][0]["parameters"]
    parameters.pop("target_fields")
    parameters[selection_key] = []

    with pytest.raises(CorruptionPlanError, match="must not be empty|non-empty array"):
        load_corruption_plan(_write_config(tmp_path / "empty-selection.json", config))


def test_loader_rejects_ambiguous_field_and_index_selection(tmp_path: Path) -> None:
    config = _valid_config()
    config["corruptions"][0]["parameters"]["target_indices"] = [0]

    with pytest.raises(CorruptionPlanError, match="ambiguous"):
        load_corruption_plan(
            _write_config(tmp_path / "ambiguous-selection.json", config)
        )


def test_loader_rejects_missing_required_selection(tmp_path: Path) -> None:
    config = _valid_config()
    parameters = config["corruptions"][0]["parameters"]
    parameters.pop("target_fields")

    with pytest.raises(CorruptionPlanError, match="provide target_fields"):
        load_corruption_plan(_write_config(tmp_path / "missing-selection.json", config))


def test_loader_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"1.0","schema_version":"1.0",'
        '"action_layout":{},"corruptions":[]}',
        encoding="utf-8",
    )

    with pytest.raises(CorruptionPlanError, match="duplicate field 'schema_version'"):
        load_corruption_plan(path)


def test_loader_does_not_mutate_source_dictionary(tmp_path: Path) -> None:
    config = _valid_config()
    before = copy.deepcopy(config)

    load_corruption_plan(_write_config(tmp_path / "immutable-input.json", config))

    assert config == before


@pytest.mark.parametrize("value", [9007199254740993, 10**400])
def test_loader_rejects_gaussian_integers_that_cannot_be_exact_floats(
    tmp_path: Path, value: int
) -> None:
    config = _valid_config()
    config["corruptions"][0]["parameters"]["standard_deviation"] = value

    with pytest.raises(CorruptionPlanError, match="floating-point range|exactly"):
        load_corruption_plan(
            _write_config(tmp_path / "oversized-gaussian.json", config)
        )


def test_loader_preserves_large_integer_bias_exactly(tmp_path: Path) -> None:
    value = 9007199254740993
    config = _valid_config()
    config["corruptions"] = [
        {
            "type": "constant_bias",
            "parameters": {"target_indices": [0], "bias": value},
        }
    ]

    plan = load_corruption_plan(_write_config(tmp_path / "exact-bias.json", config))
    action = ActionChunk(np.zeros((2, 7), dtype=np.int64), "robot_base", 0.1)
    resolved = plan.corruptions[0].resolved_parameters_for(action, plan.action_layout)
    transformed = plan.corruptions[0].apply(action, plan.action_layout, seed=0)

    assert resolved["bias"] == (value,)
    assert transformed.actions[0, 0] == value
