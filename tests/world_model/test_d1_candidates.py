"""WM-v0 D1 policy candidate provenance and deduplication tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from latentguard.world_model.candidates import (
    CandidateContext,
    CandidateOrigin,
    CandidateRejectionCode,
    CandidateValidationError,
    PolicyCandidate,
    candidate_set_digest,
    checkpoint_sha256,
    validate_and_deduplicate_candidates,
    verify_checkpoint_hash,
)
from latentguard.world_model.data_schema import CandidateSource


def _context() -> CandidateContext:
    return CandidateContext(
        episode_id="episode",
        source_episode_id="source",
        anchor_id="anchor",
        task_id="task",
        scene_group="scene",
        episode_phase="transport",
        time_index=4,
        seed=7,
        expected_action_dimension=3,
        minimum_action_horizon=4,
    )


def _candidate(
    candidate_id: str = "candidate-1", *, value: float = 1.0
) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=candidate_id,
        action_chunk=np.full((4, 3), value, dtype=np.float32),
        action_mask=None,
        candidate_origin=CandidateOrigin.SYNTHETIC_CORRUPTION,
        policy_family="synthetic",
        policy_name="frozen-test-transform",
        checkpoint_id="not_applicable",
        checkpoint_hash=None,
        inference_seed=7,
        sampling_config={"transform": "constant_bias", "value": value},
        action_horizon=4,
        generation_latency_ms=0.0,
        corruption_type="constant_bias",
    )


def test_identical_action_content_is_rejected_with_explicit_code() -> None:
    """Changing an ID cannot turn duplicate action bytes into a new sample."""

    accepted, rejected = validate_and_deduplicate_candidates(
        (_candidate(), _candidate("candidate-2")), context=_context()
    )

    assert [item.candidate_id for item in accepted] == ["candidate-1"]
    assert [item.code for item in rejected] == [CandidateRejectionCode.DUPLICATE_ACTION]


def test_policy_generated_candidate_requires_checkpoint_hash() -> None:
    """Synthetic action edits cannot be relabelled as policy inference."""

    with pytest.raises(CandidateValidationError, match="checkpoint hash"):
        replace(
            _candidate(),
            candidate_origin=CandidateOrigin.POLICY_GENERATED,
            corruption_type=None,
        )


def test_verified_policy_candidate_projects_without_synthetic_fallback() -> None:
    """A complete real-policy declaration remains policy-generated downstream."""

    candidate = replace(
        _candidate(),
        candidate_origin=CandidateOrigin.POLICY_GENERATED,
        policy_family="act",
        policy_name="accepted-controller",
        checkpoint_id="controller-v1",
        checkpoint_hash="sha256:" + "1" * 64,
        corruption_type=None,
    )

    assert (
        candidate.to_action_candidate().policy_source
        is CandidateSource.POLICY_GENERATED
    )


def test_checkpoint_hash_is_content_verified(tmp_path) -> None:
    """Checkpoint identity is verified without loading executable model data."""

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"accepted checkpoint bytes")
    digest = checkpoint_sha256(checkpoint)

    assert verify_checkpoint_hash(checkpoint, digest) == digest
    checkpoint.write_bytes(b"drifted")
    with pytest.raises(CandidateValidationError, match="differs"):
        verify_checkpoint_hash(checkpoint, digest)


def test_candidate_content_digest_is_deterministic() -> None:
    """Equivalent deterministic inference outputs retain one content identity."""

    first = _candidate()
    second = _candidate()

    assert first.action_content_digest == second.action_content_digest
    assert candidate_set_digest((first,)) == candidate_set_digest(
        (replace(second, generation_latency_ms=123.0),)
    )
