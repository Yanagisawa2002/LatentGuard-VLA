"""Deterministic generation tests for unlabeled corruption proposals."""

from __future__ import annotations

import inspect
import re
from dataclasses import fields

import numpy as np
import pytest

from latentguard.corruptions.generation import (
    CorruptionGenerationError,
    StrictApplicabilityError,
    build_proposal_identifier,
    derive_corruption_seed,
    generate_corruption_proposals,
    generate_episode_proposals,
)
from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.transforms import (
    AdditiveGaussianNoise,
    ConstantBias,
    SegmentHold,
)
from latentguard.models import Episode, OutcomeLabel
from latentguard.synthetic import generate_synthetic_episodes


def _episodes(
    *, action_dim: int = 5, episode_count: int = 2, candidate_count: int = 2
) -> tuple[Episode, ...]:
    return generate_synthetic_episodes(
        seed=17,
        episode_count=episode_count,
        episode_length=6,
        action_dim=action_dim,
        robot_state_dim=4,
        camera_count=0,
        image_height=2,
        image_width=2,
        depth_enabled=False,
        candidate_count=candidate_count,
    )


def _layout() -> ActionLayout:
    return ActionLayout(
        action_dim=5,
        fields=(
            ActionField(
                name="control_pair",
                indices=(1, 3),
                semantic=ActionSemantic.AUXILIARY,
            ),
            ActionField(
                name="switch",
                indices=(0,),
                semantic=ActionSemantic.UNSPECIFIED,
            ),
        ),
    )


def _corruptions() -> tuple[AdditiveGaussianNoise, ConstantBias]:
    return (
        AdditiveGaussianNoise(
            standard_deviation=0.1,
            target_fields=("control_pair",),
        ),
        ConstantBias(bias=(0.25, -0.5), target_indices=(1, 3)),
    )


def test_generation_is_deterministic_ordered_and_preserves_provenance() -> None:
    episodes = _episodes()
    source_snapshots = {
        candidate.candidate_id: candidate.action.actions.copy()
        for episode in episodes
        for candidate in episode.candidates
    }

    first = generate_corruption_proposals(
        episodes, _layout(), _corruptions(), base_seed=314159
    )
    second = generate_corruption_proposals(
        episodes, _layout(), _corruptions(), base_seed=314159
    )

    assert not first.skips
    assert first.source_episode_count == 2
    assert first.source_candidate_count == 4
    assert first.attempted_transformation_count == 8
    assert dict(first.corruption_counts) == {
        "additive_gaussian_noise": 4,
        "constant_bias": 4,
    }
    assert [proposal.generation_ordinal for proposal in first.proposals] == list(
        range(8)
    )
    assert [proposal.corruption_type for proposal in first.proposals] == [
        "additive_gaussian_noise",
        "constant_bias",
    ] * 4
    assert [proposal.proposal_id for proposal in first.proposals] == [
        proposal.proposal_id for proposal in second.proposals
    ]
    for left, right in zip(first.proposals, second.proposals, strict=True):
        assert np.array_equal(
            left.transformed_action.actions, right.transformed_action.actions
        )

    episode_by_id = {episode.episode_id: episode for episode in episodes}
    for proposal in first.proposals:
        source_episode = episode_by_id[proposal.source_episode_id]
        source_candidate = next(
            candidate
            for candidate in source_episode.candidates
            if candidate.candidate_id == proposal.source_candidate_id
        )
        assert proposal.source_policy_id == source_episode.source_policy_id
        assert proposal.source_task_id == source_episode.task_id
        assert proposal.split_group_id == source_episode.split_group_id
        assert not np.shares_memory(
            proposal.transformed_action.actions, source_candidate.action.actions
        )
        assert re.fullmatch(r"cap-sha256-[0-9a-f]{64}", proposal.proposal_id)
        assert proposal.proposal_id == build_proposal_identifier(
            source_episode_id=proposal.source_episode_id,
            source_candidate_id=proposal.source_candidate_id,
            corruption_name=proposal.corruption_type,
            resolved_parameters=proposal.resolved_parameters,
            seed=proposal.seed,
            generation_ordinal=proposal.generation_ordinal,
        )

    for episode in episodes:
        for candidate in episode.candidates:
            assert np.array_equal(
                candidate.action.actions, source_snapshots[candidate.candidate_id]
            )


def test_generation_resolves_exact_indices_without_guessing_semantics() -> None:
    episode = _episodes(episode_count=1, candidate_count=3)[0]

    result = generate_episode_proposals(
        episode,
        _layout(),
        _corruptions(),
        base_seed=9,
    )

    assert tuple(result.proposals[0].resolved_parameters["target_indices"]) == (1, 3)
    assert tuple(result.proposals[1].resolved_parameters["target_indices"]) == (1, 3)
    assert tuple(result.proposals[1].resolved_parameters["bias"]) == (0.25, -0.5)
    assert result.proposals[0].transformed_action.actions.shape[1] == 5


def test_generation_records_non_applicable_skip_and_strict_mode_fails() -> None:
    episode = _episodes(episode_count=1, candidate_count=3)[0]
    outside_horizon = SegmentHold(start_step=5, end_step=7)

    result = generate_episode_proposals(
        episode,
        _layout(),
        (outside_horizon,),
        base_seed=4,
        candidate_limit=1,
    )

    assert not result.proposals
    assert result.attempted_transformation_count == 1
    assert len(result.skips) == 1
    skip = result.skips[0]
    assert skip.corruption_name == "segment_hold"
    assert skip.source_candidate_id == episode.candidates[0].candidate_id
    assert skip.action_shape == (6, 5)
    assert skip.layout_action_dim == 5
    assert "outside horizon" in skip.reason
    assert "candidate=" in str(skip)

    with pytest.raises(StrictApplicabilityError) as error:
        generate_episode_proposals(
            episode,
            _layout(),
            (outside_horizon,),
            base_seed=4,
            candidate_limit=1,
            strict_applicability=True,
        )
    assert error.value.skip == skip


def test_generation_limits_are_stable_and_global() -> None:
    episodes = _episodes(episode_count=2, candidate_count=3)

    candidates_limited = generate_corruption_proposals(
        episodes,
        _layout(),
        _corruptions(),
        base_seed=5,
        candidate_limit=2,
    )
    proposals_limited = generate_corruption_proposals(
        episodes,
        _layout(),
        _corruptions(),
        base_seed=5,
        proposal_limit=3,
    )

    assert candidates_limited.source_candidate_count == 2
    assert len(candidates_limited.proposals) == 4
    assert proposals_limited.source_candidate_count == 2
    assert len(proposals_limited.proposals) == 3
    assert [
        proposal.generation_ordinal for proposal in proposals_limited.proposals
    ] == [
        0,
        1,
        2,
    ]


@pytest.mark.parametrize("base_seed", [-1, 2**64, True])
def test_generation_rejects_invalid_base_seed(base_seed: object) -> None:
    with pytest.raises(CorruptionGenerationError, match="base_seed"):
        generate_corruption_proposals(
            _episodes(episode_count=1, candidate_count=3),
            _layout(),
            _corruptions(),
            base_seed=base_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("limit", [0, -1, True])
def test_generation_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises(CorruptionGenerationError, match="candidate_limit"):
        generate_corruption_proposals(
            _episodes(episode_count=1, candidate_count=3),
            _layout(),
            _corruptions(),
            base_seed=0,
            candidate_limit=limit,  # type: ignore[arg-type]
        )


def test_identifier_seed_uses_sha256_not_python_hash() -> None:
    first = derive_corruption_seed(
        base_seed=1,
        source_episode_id="episode",
        source_candidate_id="candidate",
        corruption_name="constant_bias",
        configuration_ordinal=0,
    )
    second = derive_corruption_seed(
        base_seed=2,
        source_episode_id="episode",
        source_candidate_id="candidate",
        corruption_name="constant_bias",
        configuration_ordinal=0,
    )

    assert first == 1828626419710841788
    assert first != second
    source = inspect.getsource(
        __import__("latentguard.corruptions.generation", fromlist=["generation"])
    )
    assert "hash(" not in source
    assert "hashlib.sha256" in source


def test_proposal_schema_contains_no_outcome_or_verification_label() -> None:
    names = {field.name for field in fields(CorruptedActionProposal)}

    assert "outcome" not in names
    assert "success" not in names
    assert "unsafe" not in names
    assert "label_source" not in names
    assert "label_strength" not in names
    assert "simulator_replay_verified" not in names
    assert OutcomeLabel not in {field.type for field in fields(CorruptedActionProposal)}
