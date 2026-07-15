"""Backward-compatible window-scoped corruption tests for M3A-Data."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latentguard.corruptions.base import CorruptionApplicabilityError
from latentguard.corruptions.config import CorruptionPlanError, load_corruption_plan
from latentguard.corruptions.generation import (
    CorruptionGenerationError,
    generate_episode_proposals,
)
from latentguard.corruptions.models import (
    ResolvedParameterValue,
    compute_proposal_identifier,
)
from latentguard.corruptions.serialization import (
    MANIFEST_NAME,
    CorruptionDataset,
    CorruptionSerializationError,
    load_corruption_dataset,
    save_corruption_dataset,
)
from latentguard.corruptions.transforms import (
    ConstantBias,
    WindowScopedCorruption,
)
from latentguard.models import ActionChunk, Episode
from latentguard.synthetic import generate_synthetic_episodes

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_M3A_CONFIG = (
    _REPOSITORY_ROOT
    / "configs"
    / "integrations"
    / "maniskill_pickcube"
    / "m3a-corruptions-v1.json"
)
_M1_CONFIG = _REPOSITORY_ROOT / "configs" / "corruptions" / "m1-smoke.json"
_ALL_FAMILIES = {
    "additive_gaussian_noise",
    "constant_bias",
    "temporal_field_shift",
    "segment_hold",
    "segment_zeroing",
    "local_temporal_permutation",
}


def _action(horizon: int = 24) -> ActionChunk:
    values = np.arange(horizon * 8, dtype=np.float32).reshape(horizon, 8) / 10.0
    return ActionChunk(
        actions=values,
        coordinate_frame="robot_base",
        control_period_s=0.1,
    )


def _episode(horizon: int = 24) -> Episode:
    return generate_synthetic_episodes(
        seed=31,
        episode_count=1,
        episode_length=horizon,
        action_dim=8,
        robot_state_dim=4,
        camera_count=0,
        image_height=2,
        image_width=2,
        candidate_count=3,
    )[0]


def _config() -> dict[str, Any]:
    return json.loads(_M3A_CONFIG.read_text(encoding="utf-8"))


def _write_config(path: Path, config: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


class _OutsideWindowMutator:
    """A deliberately invalid protocol implementation for postcondition testing."""

    @property
    def name(self) -> str:
        """Return a stable diagnostic name."""
        return "outside_window_mutator"

    @property
    def resolved_parameters(self) -> Mapping[str, ResolvedParameterValue]:
        """Declare a window that the implementation intentionally violates."""
        return {
            "target_indices": (0,),
            "window_start": 1,
            "window_end": 3,
            "severity_id": "test_invalid",
        }

    def resolved_parameters_for(
        self, action: ActionChunk, layout: object
    ) -> Mapping[str, ResolvedParameterValue]:
        """Return the invalid implementation's fixed resolved contract."""
        return self.resolved_parameters

    def validate_for(self, action: ActionChunk, layout: object) -> None:
        """Accept the synthetic fixture so generation reaches its postcondition."""

    def apply(self, action: ActionChunk, layout: object, *, seed: int) -> ActionChunk:
        """Modify a row outside the declared window."""
        output = np.array(action.actions, copy=True, order="C")
        output[0, 0] += 1.0
        return ActionChunk(
            actions=output,
            coordinate_frame=action.coordinate_frame,
            control_period_s=action.control_period_s,
            schema_version=action.schema_version,
        )


def test_checked_in_m3a_schedule_is_complete_and_explicit() -> None:
    plan = load_corruption_plan(_M3A_CONFIG)
    action = _action()

    assert plan.schema_version == "1.1"
    assert plan.action_layout.schema_version == "1.0"
    assert plan.action_layout.action_dim == 8
    assert len(plan.corruptions) == 8
    assert {corruption.name for corruption in plan.corruptions} == _ALL_FAMILIES
    assert all(
        isinstance(corruption, WindowScopedCorruption)
        for corruption in plan.corruptions
    )

    resolved = [
        corruption.resolved_parameters_for(action, plan.action_layout)
        for corruption in plan.corruptions
    ]
    severity_ids = [str(parameters["severity_id"]) for parameters in resolved]
    assert len(set(severity_ids)) == 8
    assert (
        sum(severity.startswith(("mild_", "moderate_")) for severity in severity_ids)
        == 4
    )
    assert sum(severity.startswith("severe_") for severity in severity_ids) == 4
    for parameters in resolved:
        assert parameters["window_start"] == 0
        assert parameters["window_end"] == 16
        assert "target_indices" in parameters


def test_checked_in_m3a_schedule_preserves_source_continuation_bytes() -> None:
    plan = load_corruption_plan(_M3A_CONFIG)
    source = _action()
    snapshot = source.actions.tobytes(order="C")

    for ordinal, corruption in enumerate(plan.corruptions):
        transformed = corruption.apply(
            source,
            plan.action_layout,
            seed=ordinal + 100,
        )

        assert source.actions.tobytes(order="C") == snapshot
        assert transformed.actions.shape == source.actions.shape
        assert transformed.actions.dtype == source.actions.dtype
        assert not np.shares_memory(transformed.actions, source.actions)
        assert transformed.actions[16:].tobytes(order="C") == source.actions[
            16:
        ].tobytes(order="C")


def test_wrapper_preserves_bytes_on_both_sides_of_nonzero_window() -> None:
    source = _action(horizon=10)
    plan = load_corruption_plan(_M3A_CONFIG)
    corruption = WindowScopedCorruption(
        inner=ConstantBias(bias=0.25, target_indices=(0,)),
        window_start=3,
        window_end=7,
        severity_id="moderate_nonzero_window",
    )

    transformed = corruption.apply(source, plan.action_layout, seed=9)
    resolved = corruption.resolved_parameters_for(source, plan.action_layout)

    assert transformed.actions[:3].tobytes(order="C") == source.actions[:3].tobytes(
        order="C"
    )
    assert transformed.actions[7:].tobytes(order="C") == source.actions[7:].tobytes(
        order="C"
    )
    np.testing.assert_array_equal(
        transformed.actions[3:7, 0], source.actions[3:7, 0] + np.float32(0.25)
    )
    np.testing.assert_array_equal(transformed.actions[:, 1:], source.actions[:, 1:])
    assert dict(resolved) == {
        "target_indices": (0,),
        "bias": (0.25,),
        "window_start": 3,
        "window_end": 7,
        "severity_id": "moderate_nonzero_window",
    }


@pytest.mark.parametrize("missing", ["window_start", "window_end", "severity_id"])
def test_schema_1_1_requires_every_window_field(tmp_path: Path, missing: str) -> None:
    config = _config()
    del config["corruptions"][0][missing]

    with pytest.raises(CorruptionPlanError, match=f"missing {missing}"):
        load_corruption_plan(
            _write_config(tmp_path / f"missing-{missing}.json", config)
        )


def test_schema_1_1_rejects_extra_definition_fields(tmp_path: Path) -> None:
    config = _config()
    config["corruptions"][0]["window_stride"] = 1

    with pytest.raises(CorruptionPlanError, match="unexpected window_stride"):
        load_corruption_plan(_write_config(tmp_path / "extra-field.json", config))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("window_start", -1, "non-negative"),
        ("window_start", True, "expected an integer"),
        ("window_end", 0, "greater than window_start"),
        ("severity_id", "", "non-empty string"),
        ("severity_id", " severe ", "canonical string"),
    ],
)
def test_schema_1_1_rejects_invalid_window_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config = _config()
    config["corruptions"][0][field] = value

    with pytest.raises(CorruptionPlanError, match=message):
        load_corruption_plan(_write_config(tmp_path / f"invalid-{field}.json", config))


def test_schema_1_0_loads_without_window_wrappers_or_parameters() -> None:
    plan = load_corruption_plan(_M1_CONFIG)
    action = ActionChunk(
        actions=np.arange(8 * 7, dtype=np.float32).reshape(8, 7),
        coordinate_frame="robot_base",
        control_period_s=0.1,
    )

    assert plan.schema_version == "1.0"
    assert not any(
        isinstance(corruption, WindowScopedCorruption)
        for corruption in plan.corruptions
    )
    for corruption in plan.corruptions:
        assert not {
            "window_start",
            "window_end",
            "severity_id",
        }.intersection(corruption.resolved_parameters_for(action, plan.action_layout))


def test_out_of_bounds_window_is_rejected_before_application() -> None:
    plan = load_corruption_plan(_M3A_CONFIG)

    with pytest.raises(CorruptionApplicabilityError, match="outside horizon"):
        plan.corruptions[0].validate_for(_action(horizon=15), plan.action_layout)


def test_generation_rejects_any_implementation_that_changes_outside_window() -> None:
    episode = _episode()
    plan = load_corruption_plan(_M3A_CONFIG)

    with pytest.raises(CorruptionGenerationError, match="outside declared window"):
        generate_episode_proposals(
            episode,
            plan.action_layout,
            (_OutsideWindowMutator(),),
            base_seed=7,
            candidate_limit=1,
        )


def test_window_and_severity_participate_in_proposal_identity() -> None:
    episode = _episode()
    plan = load_corruption_plan(_M3A_CONFIG)
    variants = (
        WindowScopedCorruption(
            inner=ConstantBias(bias=0.1, target_indices=(0,)),
            window_start=0,
            window_end=16,
            severity_id="mild_identity",
        ),
        WindowScopedCorruption(
            inner=ConstantBias(bias=0.1, target_indices=(0,)),
            window_start=0,
            window_end=16,
            severity_id="severe_identity",
        ),
        WindowScopedCorruption(
            inner=ConstantBias(bias=0.1, target_indices=(0,)),
            window_start=1,
            window_end=16,
            severity_id="mild_identity",
        ),
    )

    proposal_ids = {
        generate_episode_proposals(
            episode,
            plan.action_layout,
            (corruption,),
            base_seed=17,
            candidate_limit=1,
        )
        .proposals[0]
        .proposal_id
        for corruption in variants
    }

    assert len(proposal_ids) == 3


def test_windowed_proposals_round_trip_with_complete_resolved_contract(
    tmp_path: Path,
) -> None:
    episode = _episode()
    plan = load_corruption_plan(_M3A_CONFIG)
    result = generate_episode_proposals(
        episode,
        plan.action_layout,
        plan.corruptions,
        base_seed=23,
        candidate_limit=1,
    )
    dataset = CorruptionDataset(
        source_dataset_id="sha256:m3a-window-round-trip",
        action_layout=plan.action_layout,
        proposals=result.proposals,
    )
    bundle = tmp_path / "windowed-corruptions"

    save_corruption_dataset(dataset, bundle)
    loaded = load_corruption_dataset(bundle)

    assert len(loaded.proposals) == 8
    for actual, expected in zip(loaded.proposals, dataset.proposals, strict=True):
        assert actual.proposal_id == expected.proposal_id
        assert actual.corruption_type == expected.corruption_type
        assert actual.resolved_parameters == expected.resolved_parameters
        assert actual.resolved_parameters["window_start"] == 0
        assert actual.resolved_parameters["window_end"] == 16
        assert isinstance(actual.resolved_parameters["severity_id"], str)
        np.testing.assert_array_equal(
            actual.transformed_action.actions,
            expected.transformed_action.actions,
        )


def test_reload_rejects_incomplete_window_contract_with_matching_identifier(
    tmp_path: Path,
) -> None:
    episode = _episode()
    plan = load_corruption_plan(_M3A_CONFIG)
    result = generate_episode_proposals(
        episode,
        plan.action_layout,
        (plan.corruptions[0],),
        base_seed=29,
        candidate_limit=1,
    )
    dataset = CorruptionDataset(
        source_dataset_id="sha256:m3a-window-tamper",
        action_layout=plan.action_layout,
        proposals=result.proposals,
    )
    bundle = tmp_path / "incomplete-window"
    save_corruption_dataset(dataset, bundle)
    manifest_path = bundle / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proposal = manifest["proposals"][0]
    parameters = dict(result.proposals[0].resolved_parameters)
    del parameters["severity_id"]
    proposal["resolved_parameters"] = parameters
    proposal["proposal_id"] = compute_proposal_identifier(
        source_episode_id=proposal["source_episode_id"],
        source_candidate_id=proposal["source_candidate_id"],
        corruption_name=proposal["corruption_type"],
        resolved_parameters=parameters,
        seed=proposal["seed"],
        generation_ordinal=proposal["generation_ordinal"],
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CorruptionSerializationError, match="incomplete window"):
        load_corruption_dataset(bundle)
