"""CPU-only tests for content-bound M0/M1 replay source resolution."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.models import (
    CorruptedActionProposal,
    compute_proposal_identifier,
)
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    save_corruption_dataset,
)
from latentguard.corruptions.transforms import ConstantBias
from latentguard.evaluation.serialization import (
    compute_corruption_dataset_content_digest,
)
from latentguard.models import ActionChunk, Episode
from latentguard.replay.source import (
    ReplaySourceBinding,
    ReplaySourceBindingError,
    ReplaySourceMutationError,
    compute_episode_content_digest,
)
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    save_episodes,
)
from latentguard.synthetic import generate_synthetic_episodes


def _episodes() -> tuple[Episode, ...]:
    return generate_synthetic_episodes(
        seed=42,
        episode_count=2,
        episode_length=4,
        action_dim=3,
        robot_state_dim=5,
        camera_count=0,
        candidate_count=2,
    )


def _layout(*, action_dim: int = 3) -> ActionLayout:
    return ActionLayout(
        action_dim=action_dim,
        fields=(
            ActionField(
                name="control",
                indices=tuple(range(action_dim)),
                semantic=ActionSemantic.AUXILIARY,
            ),
        ),
    )


def _dataset(
    episodes: tuple[Episode, ...],
    *,
    source_dataset_id: str = "sha256:" + "a" * 64,
) -> CorruptionDataset:
    result = generate_corruption_proposals(
        episodes,
        _layout(),
        (ConstantBias(bias=0.25, target_indices=(0,)),),
        base_seed=314159,
    )
    return CorruptionDataset(
        source_dataset_id=source_dataset_id,
        action_layout=_layout(),
        proposals=result.proposals,
    )


def _with_source_ids(
    proposal: CorruptedActionProposal,
    *,
    episode_id: str | None = None,
    candidate_id: str | None = None,
) -> CorruptedActionProposal:
    source_episode_id = episode_id or proposal.source_episode_id
    source_candidate_id = candidate_id or proposal.source_candidate_id
    identifier = compute_proposal_identifier(
        source_episode_id=source_episode_id,
        source_candidate_id=source_candidate_id,
        corruption_name=proposal.corruption_type,
        resolved_parameters=proposal.resolved_parameters,
        seed=proposal.seed,
        generation_ordinal=proposal.generation_ordinal,
    )
    return replace(
        proposal,
        proposal_id=identifier,
        source_episode_id=source_episode_id,
        source_candidate_id=source_candidate_id,
    )


def _single_proposal_dataset(
    source: CorruptionDataset,
    proposal: CorruptedActionProposal,
    *,
    layout: ActionLayout | None = None,
) -> CorruptionDataset:
    proposal = replace(proposal, generation_ordinal=0)
    proposal = replace(
        proposal,
        proposal_id=compute_proposal_identifier(
            source_episode_id=proposal.source_episode_id,
            source_candidate_id=proposal.source_candidate_id,
            corruption_name=proposal.corruption_type,
            resolved_parameters=proposal.resolved_parameters,
            seed=proposal.seed,
            generation_ordinal=0,
        ),
    )
    return CorruptionDataset(
        source_dataset_id=source.source_dataset_id,
        action_layout=layout or source.action_layout,
        proposals=(proposal,),
    )


def test_valid_resolution_binds_complete_digests_and_detaches_actions() -> None:
    episodes = _episodes()
    dataset = _dataset(episodes)
    binding = ReplaySourceBinding.from_datasets(episodes, dataset)

    pair = binding.resolve(dataset.proposals[0])
    source_candidate = episodes[0].candidates[0]

    assert pair.source_dataset_id == dataset.source_dataset_id
    assert pair.source_dataset_digest == compute_episode_content_digest(episodes)
    assert pair.corruption_dataset_digest == (
        compute_corruption_dataset_content_digest(dataset)
    )
    assert pair.source_episode_id == episodes[0].episode_id
    assert pair.source_candidate_id == source_candidate.candidate_id
    np.testing.assert_array_equal(
        pair.original_action.actions, source_candidate.action.actions
    )
    np.testing.assert_array_equal(
        pair.transformed_action.actions,
        dataset.proposals[0].transformed_action.actions,
    )
    assert not np.shares_memory(
        pair.original_action.actions, source_candidate.action.actions
    )
    assert not np.shares_memory(
        pair.transformed_action.actions,
        dataset.proposals[0].transformed_action.actions,
    )
    assert not pair.original_action.actions.flags.writeable
    assert not pair.transformed_action.actions.flags.writeable


def test_episode_content_digest_covers_all_arrays_and_is_path_independent(
    tmp_path: Path,
) -> None:
    episodes = _episodes()
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_episodes(episodes, first)
    save_episodes(episodes, second)

    assert compute_episode_content_digest(episodes) == compute_episode_content_digest(
        _episodes()
    )
    assert compute_episode_bundle_identifier(
        first
    ) == compute_episode_bundle_identifier(second)

    changed_action = replace(
        episodes[0].candidates[0].action,
        actions=episodes[0].candidates[0].action.actions + np.float32(1.0),
    )
    changed_candidate = replace(episodes[0].candidates[0], action=changed_action)
    changed_episode = replace(
        episodes[0],
        candidates=(changed_candidate, *episodes[0].candidates[1:]),
    )
    assert compute_episode_content_digest((changed_episode, *episodes[1:])) != (
        compute_episode_content_digest(episodes)
    )


def test_from_paths_independently_verifies_historical_m1_source_identifier(
    tmp_path: Path,
) -> None:
    episodes = _episodes()
    source_dir = tmp_path / "source"
    corruption_dir = tmp_path / "corruptions"
    save_episodes(episodes, source_dir)
    source_identifier = compute_episode_bundle_identifier(source_dir)
    dataset = _dataset(episodes, source_dataset_id=source_identifier)
    save_corruption_dataset(dataset, corruption_dir)

    binding = ReplaySourceBinding.from_paths(source_dir, corruption_dir)

    assert binding.source_bundle_identifier == source_identifier
    assert binding.source_dataset_id == source_identifier
    binding.assert_unchanged()


def test_from_paths_rejects_wrong_m1_source_identifier(tmp_path: Path) -> None:
    episodes = _episodes()
    source_dir = tmp_path / "source"
    corruption_dir = tmp_path / "corruptions"
    save_episodes(episodes, source_dir)
    save_corruption_dataset(_dataset(episodes), corruption_dir)

    with pytest.raises(ReplaySourceBindingError, match="source identifier"):
        ReplaySourceBinding.from_paths(source_dir, corruption_dir)


@pytest.mark.parametrize("kind", ["episode", "candidate"])
def test_binding_rejects_missing_source_reference(kind: str) -> None:
    episodes = _episodes()
    source = _dataset(episodes)
    proposal = source.proposals[0]
    if kind == "episode":
        proposal = _with_source_ids(proposal, episode_id="missing-episode")
    else:
        proposal = _with_source_ids(proposal, candidate_id="missing-candidate")
    broken = _single_proposal_dataset(source, proposal)

    with pytest.raises(ReplaySourceBindingError, match=f"source_{kind}_id"):
        ReplaySourceBinding.from_datasets(episodes, broken)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_task_id", "wrong-task", "source_task_id"),
        ("source_policy_id", "wrong-policy", "source_policy_id"),
        ("split_group_id", "wrong-split", "split_group_id"),
    ],
)
def test_binding_rejects_provenance_mismatch(
    field: str, value: str, message: str
) -> None:
    episodes = _episodes()
    source = _dataset(episodes)
    broken_proposal = replace(source.proposals[0], **{field: value})
    broken = _single_proposal_dataset(source, broken_proposal)

    with pytest.raises(ReplaySourceBindingError, match=message):
        ReplaySourceBinding.from_datasets(episodes, broken)


@pytest.mark.parametrize(
    ("kind", "action", "layout", "message"),
    [
        (
            "horizon",
            ActionChunk(
                actions=np.ones((3, 3), dtype=np.float32),
                coordinate_frame="robot_base",
                control_period_s=0.1,
            ),
            _layout(),
            "horizon",
        ),
        (
            "dimension",
            ActionChunk(
                actions=np.ones((4, 2), dtype=np.float32),
                coordinate_frame="robot_base",
                control_period_s=0.1,
            ),
            _layout(action_dim=2),
            "action_dim",
        ),
    ],
)
def test_binding_rejects_shape_contract_mismatch(
    kind: str,
    action: ActionChunk,
    layout: ActionLayout,
    message: str,
) -> None:
    del kind
    episodes = _episodes()
    source = _dataset(episodes)
    original = source.proposals[0]
    action = replace(
        action,
        coordinate_frame=original.transformed_action.coordinate_frame,
        control_period_s=original.transformed_action.control_period_s,
    )
    proposal = replace(original, transformed_action=action)
    broken = _single_proposal_dataset(source, proposal, layout=layout)

    with pytest.raises(ReplaySourceBindingError, match=message):
        ReplaySourceBinding.from_datasets(episodes, broken)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"control_period_s": 9.0}, "control_period_s"),
        ({"coordinate_frame": "wrong-frame"}, "coordinate_frame"),
    ],
)
def test_binding_rejects_action_metadata_mismatch(
    update: dict[str, object], message: str
) -> None:
    episodes = _episodes()
    source = _dataset(episodes)
    original = source.proposals[0]
    transformed = replace(original.transformed_action, **update)
    proposal = replace(original, transformed_action=transformed)

    broken = _single_proposal_dataset(source, proposal)
    with pytest.raises(ReplaySourceBindingError, match=message):
        ReplaySourceBinding.from_datasets(episodes, broken)


def test_binding_rejects_dtype_mismatch() -> None:
    episodes = _episodes()
    source = _dataset(episodes)
    original = source.proposals[0]
    proposal = replace(
        original,
        transformed_action=replace(
            original.transformed_action,
            actions=original.transformed_action.actions.astype(np.float64),
        ),
    )
    broken = _single_proposal_dataset(source, proposal)

    with pytest.raises(ReplaySourceBindingError, match="dtype"):
        ReplaySourceBinding.from_datasets(episodes, broken)


def test_supplied_source_and_corruption_digest_mismatches_are_rejected() -> None:
    episodes = _episodes()
    dataset = _dataset(episodes)

    with pytest.raises(ReplaySourceBindingError, match="source_dataset_digest"):
        ReplaySourceBinding.from_datasets(
            episodes, dataset, source_dataset_digest="sha256:" + "0" * 64
        )
    with pytest.raises(ReplaySourceBindingError, match="corruption_dataset_digest"):
        ReplaySourceBinding.from_datasets(
            episodes, dataset, corruption_dataset_digest="sha256:" + "0" * 64
        )


def test_resolve_compares_complete_incoming_proposal_not_only_identifier() -> None:
    episodes = _episodes()
    dataset = _dataset(episodes)
    binding = ReplaySourceBinding.from_datasets(episodes, dataset)
    proposal = dataset.proposals[0]
    changed = replace(
        proposal,
        transformed_action=replace(
            proposal.transformed_action,
            actions=proposal.transformed_action.actions + np.float32(1.0),
        ),
    )

    with pytest.raises(ReplaySourceBindingError, match="proposal content"):
        binding.resolve(changed)


def test_bound_source_and_proposal_arrays_remain_immutable() -> None:
    episodes = _episodes()
    dataset = _dataset(episodes)
    binding = ReplaySourceBinding.from_datasets(episodes, dataset)
    source_before = episodes[0].candidates[0].action.actions.tobytes(order="C")
    proposal_before = dataset.proposals[0].transformed_action.actions.tobytes(order="C")

    pair = binding.resolve(dataset.proposals[0])
    with pytest.raises(ValueError):
        pair.original_action.actions[0, 0] = 99.0
    with pytest.raises(ValueError):
        pair.transformed_action.actions[0, 0] = 99.0

    assert episodes[0].candidates[0].action.actions.tobytes(order="C") == source_before
    assert (
        dataset.proposals[0].transformed_action.actions.tobytes(order="C")
        == proposal_before
    )
    binding.assert_unchanged()


def test_assert_unchanged_detects_serialized_source_mutation(tmp_path: Path) -> None:
    episodes = _episodes()
    source_dir = tmp_path / "source"
    corruption_dir = tmp_path / "corruptions"
    save_episodes(episodes, source_dir)
    dataset = _dataset(
        episodes, source_dataset_id=compute_episode_bundle_identifier(source_dir)
    )
    save_corruption_dataset(dataset, corruption_dir)
    binding = ReplaySourceBinding.from_paths(source_dir, corruption_dir)

    with (source_dir / "manifest.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(ReplaySourceMutationError):
        binding.assert_unchanged()


@pytest.mark.parametrize("bundle_kind", ["source", "corruption"])
def test_resolve_detects_serialized_bundle_mutation(
    tmp_path: Path, bundle_kind: str
) -> None:
    episodes = _episodes()
    source_dir = tmp_path / "source"
    corruption_dir = tmp_path / "corruptions"
    save_episodes(episodes, source_dir)
    dataset = _dataset(
        episodes, source_dataset_id=compute_episode_bundle_identifier(source_dir)
    )
    save_corruption_dataset(dataset, corruption_dir)
    binding = ReplaySourceBinding.from_paths(source_dir, corruption_dir)
    target = source_dir / "manifest.json"
    if bundle_kind == "source":
        target.write_bytes(target.read_bytes() + b" ")
    else:
        target = corruption_dir / "manifest.json"
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["source_dataset_id"] = "sha256:" + "0" * 64
        target.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ReplaySourceMutationError):
        binding.resolve(binding.corruption_dataset.proposals[0])
