"""Outcome-blind fixed, random, and frozen-score selector helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.control.models import (
    FIXED_VISUAL_BATCH_SIZE,
    ClosedLoopCandidatePoolV1,
    SelectorOutputV1,
    content_digest,
)

FIXED_PRIMARY_SELECTOR_ID = "fixed_primary_v1"
RANDOM_SELECTOR_ID = "deterministic_random_v1"


def _selector_identity(selector_id: str, semantic: str) -> str:
    return content_digest(
        {"schema_version": "1.0", "selector_id": selector_id, "semantic": semantic},
        context="M4CSelectorIdentityV1",
    )


def fixed_primary_output(
    pool: ClosedLoopCandidatePoolV1, *, primary_definition_ordinal: int = 0
) -> SelectorOutputV1:
    """Select one predeclared candidate definition with no learned verifier."""

    if not isinstance(pool, ClosedLoopCandidatePoolV1):
        raise ValueError("fixed primary requires a closed-loop candidate pool")
    if (
        type(primary_definition_ordinal) is not int
        or not 0 <= primary_definition_ordinal < 8
    ):
        raise ValueError("primary definition ordinal must be in 0..7")
    ids = pool.ordered_candidate_ids
    ranking = (ids[primary_definition_ordinal],) + tuple(
        candidate_id
        for index, candidate_id in enumerate(ids)
        if index != primary_definition_ordinal
    )
    scores = {
        candidate_id: float(index != primary_definition_ordinal)
        for index, candidate_id in enumerate(ids)
    }
    return SelectorOutputV1(
        selector_id=FIXED_PRIMARY_SELECTOR_ID,
        scores=scores,
        ranking=ranking,
        checkpoint_ensemble_identity=_selector_identity(
            FIXED_PRIMARY_SELECTOR_ID,
            f"fixed_definition_{primary_definition_ordinal}_v1",
        ),
        probabilities=False,
    )


def deterministic_random_output(pool: ClosedLoopCandidatePoolV1) -> SelectorOutputV1:
    """Rank candidates by stable SHA-256-derived uniform values."""

    if not isinstance(pool, ClosedLoopCandidatePoolV1):
        raise ValueError("random selector requires a closed-loop candidate pool")
    scores: dict[str, float] = {}
    for candidate_id in pool.ordered_candidate_ids:
        payload = (
            f"{pool.content_digest}\n{candidate_id}\n{RANDOM_SELECTOR_ID}".encode()
        )
        raw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        scores[candidate_id] = raw / float(2**64)
    ranking = tuple(sorted(scores, key=lambda item: (scores[item], item)))
    return SelectorOutputV1(
        selector_id=RANDOM_SELECTOR_ID,
        scores=scores,
        ranking=ranking,
        checkpoint_ensemble_identity=_selector_identity(
            RANDOM_SELECTOR_ID, "sha256_pool_candidate_uniform_ranking_v1"
        ),
        probabilities=False,
    )


def frozen_probability_output(
    pool: ClosedLoopCandidatePoolV1,
    *,
    selector_id: str,
    checkpoint_ensemble_identity: str,
    probability_function: Callable[[ClosedLoopCandidatePoolV1], Mapping[str, float]],
) -> SelectorOutputV1:
    """Apply one frozen inference callable and rank minimum failure probability."""

    probabilities = dict(probability_function(pool))
    if set(probabilities) != set(pool.ordered_candidate_ids):
        raise ValueError("frozen selector returned a different candidate inventory")
    ranking = tuple(sorted(probabilities, key=lambda item: (probabilities[item], item)))
    return SelectorOutputV1(
        selector_id=selector_id,
        scores=probabilities,
        ranking=ranking,
        checkpoint_ensemble_identity=checkpoint_ensemble_identity,
        probabilities=True,
    )


def build_fixed_visual_batch(images: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Repeat three ordered RGB224 images round-robin to exactly batch 128."""

    if (
        not isinstance(images, np.ndarray)
        or images.dtype != np.dtype(np.uint8)
        or images.shape != (3, 224, 224, 3)
    ):
        raise ValueError("visual input must be uint8 [3,224,224,3]")
    indices = np.arange(FIXED_VISUAL_BATCH_SIZE, dtype=np.int64) % 3
    batch = np.asarray(images[indices], dtype=np.uint8, order="C")
    if not np.array_equal(batch[:3], images):
        raise RuntimeError("fixed visual batch changed the three real slots")
    return batch


def consume_fixed_visual_features(features: NDArray[Any]) -> NDArray[np.float32]:
    """Consume only slots 0..2 from an exact float32 [128,512] result."""

    if (
        not isinstance(features, np.ndarray)
        or features.dtype != np.dtype("<f4")
        or features.shape != (FIXED_VISUAL_BATCH_SIZE, 512)
        or not bool(np.all(np.isfinite(features)))
    ):
        raise ValueError("backbone output must be finite float32 [128,512]")
    return np.array(features[:3], copy=True, order="C", dtype=np.float32)


__all__ = [
    "FIXED_PRIMARY_SELECTOR_ID",
    "RANDOM_SELECTOR_ID",
    "build_fixed_visual_batch",
    "consume_fixed_visual_features",
    "deterministic_random_output",
    "fixed_primary_output",
    "frozen_probability_output",
]
