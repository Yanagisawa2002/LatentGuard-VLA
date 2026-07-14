"""CPU-only tests for canonical replay models and identities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latentguard.evaluation.models import compute_configuration_digest
from latentguard.models import ActionChunk
from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import (
    compute_action_content_digest,
    compute_replay_bundle_digest,
    compute_replay_case_identifier,
)
from latentguard.replay.models import (
    REPLAY_SCHEMA_VERSION,
    ReplayBundle,
    ReplayCase,
    ReplayStateReference,
    ReplayTaskReference,
    StateComparisonSemantic,
)

_SOURCE_DIGEST = "sha256:" + "1" * 64
_CORRUPTION_DIGEST = "sha256:" + "2" * 64
_STATE_DIGEST = "sha256:" + "3" * 64


def _state_reference(
    *, metadata: object | None = None, state_index: object = 0
) -> ReplayStateReference:
    kwargs: dict[str, object] = {}
    if metadata is not None:
        kwargs["metadata"] = metadata
    return ReplayStateReference(
        adapter_id="deterministic_replay_fixture",
        adapter_version="1.0.0",
        source_reference_id="episode-0:initial",
        expected_state_digest=_STATE_DIGEST,
        comparison_semantic=StateComparisonSemantic.EXACT_DIGEST,
        state_index=state_index,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _task_reference(*, metadata: object | None = None) -> ReplayTaskReference:
    kwargs: dict[str, object] = {}
    if metadata is not None:
        kwargs["metadata"] = metadata
    return ReplayTaskReference(
        task_id="fixture-task",
        task_contract_version="1.0.0",
        **kwargs,  # type: ignore[arg-type]
    )


def _actions() -> tuple[ActionChunk, ActionChunk]:
    original = ActionChunk(
        actions=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        coordinate_frame="fixture_frame",
        control_period_s=0.1,
    )
    transformed = ActionChunk(
        actions=np.array([[0.1, 0.25], [0.3, 0.45]], dtype=np.float32),
        coordinate_frame="fixture_frame",
        control_period_s=0.1,
    )
    return original, transformed


def _case(
    *,
    proposal_id: str = "proposal-0",
    state_reference: ReplayStateReference | None = None,
    task_reference: ReplayTaskReference | None = None,
    original_action: ActionChunk | None = None,
    transformed_action: ActionChunk | None = None,
) -> ReplayCase:
    default_original, default_transformed = _actions()
    original = original_action or default_original
    transformed = transformed_action or default_transformed
    state = state_reference or _state_reference()
    task = task_reference or _task_reference()
    fields = {
        "proposal_id": proposal_id,
        "source_dataset_id": "sha256:" + "0" * 64,
        "source_dataset_digest": _SOURCE_DIGEST,
        "corruption_dataset_digest": _CORRUPTION_DIGEST,
        "source_episode_id": "episode-0",
        "source_candidate_id": "candidate-0",
        "split_group_id": "split-episode-0",
        "original_action": original,
        "transformed_action": transformed,
        "state_reference": state,
        "task_reference": task,
        "adapter_id": "deterministic_replay_fixture",
        "adapter_version": "1.0.0",
        "progress_semantic": "normalized_target_distance_after",
        "unsafe_semantic": "synthetic_state_bound",
        "schema_version": REPLAY_SCHEMA_VERSION,
    }
    case_id = compute_replay_case_identifier(**fields)  # type: ignore[arg-type]
    return ReplayCase(case_id=case_id, **fields)  # type: ignore[arg-type]


def _bundle(cases: tuple[ReplayCase, ...]) -> ReplayBundle:
    configuration_digest = compute_configuration_digest(
        {"schema_version": "1.0", "state_dimension": 2}
    )
    fields = {
        "source_dataset_id": "sha256:" + "0" * 64,
        "source_dataset_digest": _SOURCE_DIGEST,
        "corruption_dataset_digest": _CORRUPTION_DIGEST,
        "adapter_id": "deterministic_replay_fixture",
        "adapter_version": "1.0.0",
        "adapter_configuration_digest": configuration_digest,
        "replay_cases": cases,
        "metadata": {"purpose": "infrastructure", "nested": {"count": len(cases)}},
        "schema_version": REPLAY_SCHEMA_VERSION,
    }
    digest = compute_replay_bundle_digest(**fields)  # type: ignore[arg-type]
    return ReplayBundle(bundle_digest=digest, **fields)  # type: ignore[arg-type]


def test_action_digest_binds_dtype_shape_semantics_and_exact_bytes() -> None:
    original, _ = _actions()
    same = ActionChunk(
        actions=np.array(original.actions, copy=True),
        coordinate_frame=original.coordinate_frame,
        control_period_s=original.control_period_s,
    )
    changed_dtype = ActionChunk(
        actions=original.actions.astype(np.float64),
        coordinate_frame=original.coordinate_frame,
        control_period_s=original.control_period_s,
    )
    changed_value = ActionChunk(
        actions=np.array([[0.1, 0.2], [0.3, 0.5]], dtype=np.float32),
        coordinate_frame=original.coordinate_frame,
        control_period_s=original.control_period_s,
    )

    assert compute_action_content_digest(original) == compute_action_content_digest(
        same
    )
    assert compute_action_content_digest(original) != compute_action_content_digest(
        changed_dtype
    )
    assert compute_action_content_digest(original) != compute_action_content_digest(
        changed_value
    )


def test_case_identity_is_deterministic_and_canonical_metadata_order_independent() -> (
    None
):
    first = _case(
        state_reference=_state_reference(metadata={"b": (2, 3), "a": {"x": True}}),
        task_reference=_task_reference(metadata={"z": 1, "a": "stable"}),
    )
    second = _case(
        state_reference=_state_reference(metadata={"a": {"x": True}, "b": [2, 3]}),
        task_reference=_task_reference(metadata={"a": "stable", "z": 1}),
    )

    assert first.case_id == second.case_id
    assert tuple(first.state_reference.metadata["b"]) == (2, 3)  # type: ignore[arg-type]


def test_case_detaches_actions_and_nested_metadata() -> None:
    source = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    metadata = {"nested": {"values": [1, 2]}}
    original = ActionChunk(
        actions=source,
        coordinate_frame="fixture_frame",
        control_period_s=0.1,
    )
    replay_case = _case(
        original_action=original,
        state_reference=_state_reference(metadata=metadata),
    )
    before = replay_case.original_action.actions.tobytes()

    source[0, 0] = 99.0
    metadata["nested"]["values"].append(3)  # type: ignore[index,union-attr]

    assert replay_case.original_action.actions.tobytes() == before
    assert not replay_case.original_action.actions.flags.writeable
    assert tuple(replay_case.state_reference.metadata["nested"]["values"]) == (  # type: ignore[index]
        1,
        2,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"path": "/private/runtime/state.bin"},
        {"path": r"C:\private\state.bin"},
        {"path": Path("relative-state.bin")},
        {"raw_state": np.zeros(2)},
        {"value": float("nan")},
        {"value": float("inf")},
    ],
)
def test_state_reference_rejects_runtime_or_non_json_metadata(metadata: object) -> None:
    with pytest.raises(ReplayValidationError):
        _state_reference(metadata=metadata)


@pytest.mark.parametrize("state_index", [True, -1, 1.5])
def test_state_reference_rejects_invalid_integer_selector(state_index: object) -> None:
    with pytest.raises(ReplayValidationError):
        _state_reference(state_index=state_index)


def test_state_reference_requires_exactly_one_state_selector() -> None:
    with pytest.raises(ReplayValidationError, match="exactly one"):
        ReplayStateReference(
            adapter_id="deterministic_replay_fixture",
            adapter_version="1.0.0",
            source_reference_id="episode-0:initial",
            expected_state_digest=_STATE_DIGEST,
            comparison_semantic=StateComparisonSemantic.EXACT_DIGEST,
        )
    with pytest.raises(ReplayValidationError, match="exactly one"):
        ReplayStateReference(
            adapter_id="deterministic_replay_fixture",
            adapter_version="1.0.0",
            source_reference_id="episode-0:initial",
            expected_state_digest=_STATE_DIGEST,
            comparison_semantic=StateComparisonSemantic.EXACT_DIGEST,
            state_key="initial",
            state_index=0,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"task_id": ""},
        {"task_contract_version": "version-one"},
        {"metadata": {"runtime_path": "/private/task.json"}},
        {"schema_version": "2.0"},
    ],
)
def test_task_reference_rejects_invalid_identity_metadata_and_version(
    updates: dict[str, object],
) -> None:
    fields: dict[str, object] = {
        "task_id": "fixture-task",
        "task_contract_version": "1.0.0",
    }
    fields.update(updates)
    with pytest.raises(ReplayValidationError):
        ReplayTaskReference(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changed",
    [
        ActionChunk(
            actions=np.zeros((3, 2), dtype=np.float32),
            coordinate_frame="fixture_frame",
            control_period_s=0.1,
        ),
        ActionChunk(
            actions=np.zeros((2, 2), dtype=np.float64),
            coordinate_frame="fixture_frame",
            control_period_s=0.1,
        ),
        ActionChunk(
            actions=np.zeros((2, 2), dtype=np.float32),
            coordinate_frame="other_frame",
            control_period_s=0.1,
        ),
        ActionChunk(
            actions=np.zeros((2, 2), dtype=np.float32),
            coordinate_frame="fixture_frame",
            control_period_s=0.2,
        ),
    ],
)
def test_replay_case_rejects_action_contract_mismatch(changed: ActionChunk) -> None:
    with pytest.raises(ReplayValidationError):
        _case(transformed_action=changed)


def test_replay_case_rejects_tampered_identifier() -> None:
    valid = _case()
    with pytest.raises(ReplayValidationError, match="identifier mismatch"):
        ReplayCase(
            case_id="rpc-sha256-" + "f" * 64,
            proposal_id=valid.proposal_id,
            source_dataset_id=valid.source_dataset_id,
            source_dataset_digest=valid.source_dataset_digest,
            corruption_dataset_digest=valid.corruption_dataset_digest,
            source_episode_id=valid.source_episode_id,
            source_candidate_id=valid.source_candidate_id,
            split_group_id=valid.split_group_id,
            original_action=valid.original_action,
            transformed_action=valid.transformed_action,
            state_reference=valid.state_reference,
            task_reference=valid.task_reference,
            adapter_id=valid.adapter_id,
            adapter_version=valid.adapter_version,
            progress_semantic=valid.progress_semantic,
            unsafe_semantic=valid.unsafe_semantic,
        )


def test_bundle_digest_is_ordered_and_rejects_duplicates_and_tampering() -> None:
    first = _case(proposal_id="proposal-0")
    second = _case(proposal_id="proposal-1")
    ordered = _bundle((first, second))
    reversed_bundle = _bundle((second, first))

    assert ordered.bundle_digest != reversed_bundle.bundle_digest
    with pytest.raises(ReplayValidationError, match="duplicate"):
        _bundle((first, first))
    with pytest.raises(ReplayValidationError, match="content digest mismatch"):
        ReplayBundle(
            source_dataset_id=ordered.source_dataset_id,
            source_dataset_digest=ordered.source_dataset_digest,
            corruption_dataset_digest=ordered.corruption_dataset_digest,
            adapter_id=ordered.adapter_id,
            adapter_version=ordered.adapter_version,
            adapter_configuration_digest=ordered.adapter_configuration_digest,
            replay_cases=ordered.replay_cases,
            bundle_digest="rpb-sha256-" + "f" * 64,
            metadata=ordered.metadata,
        )


def test_bundle_rejects_case_with_conflicting_content_binding() -> None:
    replay_case = _case()
    configuration_digest = compute_configuration_digest({"schema_version": "1.0"})
    fields = {
        "source_dataset_id": replay_case.source_dataset_id,
        "source_dataset_digest": "sha256:" + "a" * 64,
        "corruption_dataset_digest": replay_case.corruption_dataset_digest,
        "adapter_id": replay_case.adapter_id,
        "adapter_version": replay_case.adapter_version,
        "adapter_configuration_digest": configuration_digest,
        "replay_cases": (replay_case,),
        "metadata": {},
    }
    digest = compute_replay_bundle_digest(**fields)  # type: ignore[arg-type]
    with pytest.raises(ReplayValidationError, match="must match bundle"):
        ReplayBundle(bundle_digest=digest, **fields)  # type: ignore[arg-type]
