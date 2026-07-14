"""CPU-only tests for explicit action semantics and proposal models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from latentguard.corruptions import (
    ActionField,
    ActionLayout,
    ActionLayoutError,
    ActionSemantic,
    CorruptedActionProposal,
    CorruptedActionProposalError,
    compute_proposal_identifier,
    validate_action_layout,
    validate_corrupted_action_proposal,
)
from latentguard.models import ActionChunk

_FLOAT32_DTYPE = np.dtype(np.float32)


def _action(dtype: np.dtype[np.generic] = _FLOAT32_DTYPE) -> ActionChunk:
    return ActionChunk(
        actions=np.arange(30).reshape(5, 6).astype(dtype),
        coordinate_frame="declared-frame",
        control_period_s=0.05,
    )


def test_layout_supports_arbitrary_dimensions_and_unused_indices() -> None:
    layout = ActionLayout(
        action_dim=11,
        fields=(
            ActionField(
                "tool_delta",
                (2, 7, 9),
                ActionSemantic.TRANSLATION,
                units="m",
            ),
            ActionField("clamp", (0,), ActionSemantic.GRIPPER),
            ActionField("vendor_mode", (5, 6), ActionSemantic.AUXILIARY),
        ),
        description="Non-seven-dimensional action layout",
    )

    validate_action_layout(layout)
    assert layout.action_dim == 11
    assert layout.indices_for_fields(("clamp", "tool_delta")) == (0, 2, 7, 9)
    assert set(range(layout.action_dim)) - {
        index for field in layout.fields for index in field.indices
    } == {1, 3, 4, 8, 10}


def test_layout_requires_no_inferred_or_mandatory_semantic_field() -> None:
    layout = ActionLayout(
        action_dim=4,
        fields=(ActionField("opaque", (1,), ActionSemantic.UNSPECIFIED),),
    )

    assert layout.field("opaque").semantic is ActionSemantic.UNSPECIFIED
    with pytest.raises(ActionLayoutError, match="unknown field.*translation"):
        layout.field("translation")


def test_layout_and_fields_detach_mutable_inputs() -> None:
    raw_indices = [1, 3]
    raw_metadata = {"controller": "v2"}
    raw_fields = [
        ActionField(
            "rotation",
            raw_indices,  # type: ignore[arg-type]
            ActionSemantic.ROTATION,
            metadata=raw_metadata,
        )
    ]
    layout = ActionLayout(
        5,
        raw_fields,  # type: ignore[arg-type]
        metadata={"source": "declared"},
    )

    raw_indices.append(4)
    raw_metadata["controller"] = "mutated"
    raw_fields.clear()
    assert layout.fields[0].indices == (1, 3)
    assert layout.fields[0].metadata == {"controller": "v2"}
    with pytest.raises(TypeError):
        layout.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        layout.action_dim = 7  # type: ignore[misc]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            (
                ActionField("first", (0, 1), ActionSemantic.AUXILIARY),
                ActionField("second", (1,), ActionSemantic.GRIPPER),
            ),
            "assigned to both",
        ),
        (
            (
                ActionField("same", (0,), ActionSemantic.AUXILIARY),
                ActionField("same", (1,), ActionSemantic.GRIPPER),
            ),
            "duplicate field name",
        ),
        (
            (ActionField("outside", (4,), ActionSemantic.UNSPECIFIED),),
            r"outside \[0, 4\)",
        ),
    ],
)
def test_layout_rejects_duplicate_or_out_of_range_assignments(
    fields: tuple[ActionField, ...], message: str
) -> None:
    with pytest.raises(ActionLayoutError, match=message):
        ActionLayout(4, fields)


@pytest.mark.parametrize(
    "indices",
    [(), (1, 1), (-1,), (True,)],
)
def test_action_field_rejects_empty_duplicate_negative_or_noninteger_indices(
    indices: tuple[int, ...],
) -> None:
    with pytest.raises(ActionLayoutError, match="indices"):
        ActionField("bad", indices, ActionSemantic.AUXILIARY)


@pytest.mark.parametrize("action_dim", [0, -1, True, 1.5])
def test_layout_rejects_nonpositive_or_noninteger_action_dimension(
    action_dim: object,
) -> None:
    with pytest.raises(ActionLayoutError, match="positive integer"):
        ActionLayout(action_dim, ())  # type: ignore[arg-type]


def test_layout_selection_is_explicit_and_unambiguous() -> None:
    layout = ActionLayout(
        6,
        (
            ActionField("move", (1, 4), ActionSemantic.TRANSLATION),
            ActionField("grip", (5,), ActionSemantic.GRIPPER),
        ),
    )

    assert layout.resolve_selection(target_fields=("move",)) == (1, 4)
    assert layout.resolve_selection(target_indices=(3, 0)) == (3, 0)
    assert layout.resolve_selection(allow_all=True) == (0, 1, 2, 3, 4, 5)
    with pytest.raises(ActionLayoutError, match="ambiguous"):
        layout.resolve_selection(target_fields=("move",), target_indices=(1,))
    with pytest.raises(ActionLayoutError, match="provide"):
        layout.resolve_selection()
    with pytest.raises(ActionLayoutError, match="outside"):
        layout.resolve_selection(target_indices=(6,))


def test_layout_rejects_nonfinite_or_mutable_metadata_values() -> None:
    with pytest.raises(ActionLayoutError, match="finite"):
        ActionLayout(2, (), metadata={"score": float("nan")})
    with pytest.raises(ActionLayoutError, match="JSON scalar"):
        ActionField(
            "bad",
            (0,),
            ActionSemantic.UNSPECIFIED,
            metadata={"nested": [1]},  # type: ignore[dict-item]
        )


def test_proposal_is_unlabeled_complete_and_mutation_safe() -> None:
    source = _action()
    parameters = {"target_indices": (1, 3), "bias": 0.25}
    proposal = CorruptedActionProposal(
        proposal_id=compute_proposal_identifier(
            source_episode_id="episode-1",
            source_candidate_id="candidate-2",
            corruption_name="constant_bias",
            resolved_parameters=parameters,
            seed=17,
            generation_ordinal=8,
        ),
        source_episode_id="episode-1",
        source_candidate_id="candidate-2",
        source_policy_id="policy-3",
        source_task_id="task-4",
        split_group_id="split-episode-1",
        transformed_action=source,
        corruption_type="constant_bias",
        resolved_parameters=parameters,
        seed=17,
        generation_ordinal=8,
        notes="unlabeled proposal",
    )

    assert "outcome" not in {field.name for field in fields(proposal)}
    assert proposal.transformed_action is not source
    assert proposal.transformed_action.actions is not source.actions
    assert not proposal.transformed_action.actions.flags.writeable
    parameters["bias"] = 999.0
    assert proposal.resolved_parameters["bias"] == 0.25
    with pytest.raises(TypeError):
        proposal.resolved_parameters["bias"] = 1.0  # type: ignore[index]
    validate_corrupted_action_proposal(proposal)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source_candidate_id": ""}, "source_candidate_id"),
        ({"seed": -1}, "seed"),
        ({"generation_ordinal": -1}, "generation_ordinal"),
        ({"schema_version": "999"}, "unsupported"),
        ({"resolved_parameters": {"value": float("inf")}}, "finite"),
        ({"resolved_parameters": {"value": [1, 2]}}, "JSON scalar"),
    ],
)
def test_proposal_rejects_incomplete_or_non_json_safe_provenance(
    updates: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "proposal_id": "proposal",
        "source_episode_id": "episode",
        "source_candidate_id": "candidate",
        "source_policy_id": "policy",
        "source_task_id": "task",
        "split_group_id": "split",
        "transformed_action": _action(),
        "corruption_type": "constant_bias",
        "resolved_parameters": {"bias": 1.0},
        "seed": 1,
        "generation_ordinal": 0,
    }
    values.update(updates)
    with pytest.raises(CorruptedActionProposalError, match=message):
        CorruptedActionProposal(**values)  # type: ignore[arg-type]
