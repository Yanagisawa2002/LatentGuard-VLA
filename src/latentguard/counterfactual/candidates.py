"""Content identity, pair metrics, and pre-registered candidate deduplication."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt

from latentguard.counterfactual.models import (
    CandidateDisposition,
    action_content_sha256,
)


@dataclass(frozen=True, slots=True)
class CandidateDiversityThresholds:
    """Frozen controller-aware thresholds for candidate content relations."""

    near_full_chunk_l2: float
    meaningful_full_chunk_l2: float
    meaningful_endpoint_translation: float
    meaningful_cumulative_rotation: float
    gripper_disagreement_epsilon: float

    def __post_init__(self) -> None:
        """Reject invalid or internally inconsistent thresholds."""
        values = (
            self.near_full_chunk_l2,
            self.meaningful_full_chunk_l2,
            self.meaningful_endpoint_translation,
            self.meaningful_cumulative_rotation,
            self.gripper_disagreement_epsilon,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in values
        ):
            raise ValueError("candidate thresholds must be finite and non-negative")
        if self.near_full_chunk_l2 >= self.meaningful_full_chunk_l2:
            raise ValueError("near threshold must be below meaningful threshold")


def _action(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    action = np.asarray(value, dtype=np.float64)
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 7:
        raise ValueError("candidate action must have shape [horizon, 7]")
    if not np.isfinite(action).all():
        raise ValueError("candidate action contains non-finite values")
    return action


def pairwise_action_metrics(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    *,
    gripper_epsilon: float,
) -> dict[str, float | bool]:
    """Compute the registered same-anchor action-distance diagnostics."""
    first = _action(left)
    second = _action(right)
    if first.shape != second.shape:
        raise ValueError("candidate actions must have identical shapes")
    difference = first - second
    left_flat = first.reshape(-1)
    right_flat = second.reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    cosine_distance = (
        0.0
        if denominator == 0 and np.array_equal(first, second)
        else (
            1.0
            if denominator == 0
            else 1.0 - float(np.dot(left_flat, right_flat) / denominator)
        )
    )
    translation_difference = difference[:, :3]
    rotation_difference = difference[:, 3:6]
    return {
        "full_chunk_normalized_l2": float(np.linalg.norm(difference)),
        "full_chunk_cosine_distance": cosine_distance,
        "first_action_l2": float(np.linalg.norm(difference[0])),
        "chunk_endpoint_displacement": float(
            np.linalg.norm(np.sum(translation_difference, axis=0))
        ),
        "cumulative_translation_difference": float(
            np.sum(np.linalg.norm(translation_difference, axis=1))
        ),
        "cumulative_rotation_difference": float(
            np.sum(np.linalg.norm(rotation_difference, axis=1))
        ),
        "gripper_disagreement": bool(
            np.any(np.abs(difference[:, 6]) > float(gripper_epsilon))
        ),
    }


def _relation(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    thresholds: CandidateDiversityThresholds,
) -> CandidateDisposition:
    if np.array_equal(
        np.ascontiguousarray(np.asarray(left, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(right, dtype=np.float32)),
    ):
        return CandidateDisposition.EXACT_DUPLICATE
    metrics = pairwise_action_metrics(
        left,
        right,
        gripper_epsilon=thresholds.gripper_disagreement_epsilon,
    )
    full_l2 = float(metrics["full_chunk_normalized_l2"])
    meaningful = (
        full_l2 >= thresholds.meaningful_full_chunk_l2
        or float(metrics["chunk_endpoint_displacement"])
        >= thresholds.meaningful_endpoint_translation
        or float(metrics["cumulative_rotation_difference"])
        >= thresholds.meaningful_cumulative_rotation
        or bool(metrics["gripper_disagreement"])
    )
    if meaningful:
        return CandidateDisposition.MEANINGFULLY_DISTINCT
    if full_l2 <= thresholds.near_full_chunk_l2:
        return CandidateDisposition.NEAR_DUPLICATE
    return CandidateDisposition.NEAR_DUPLICATE


def classify_candidate_group(
    actions: list[npt.ArrayLike],
    thresholds: CandidateDiversityThresholds,
) -> tuple[CandidateDisposition, ...]:
    """Classify candidates sequentially against every earlier anchor candidate."""
    dispositions: list[CandidateDisposition] = []
    for index, action in enumerate(actions):
        _action(action)
        if index == 0:
            dispositions.append(CandidateDisposition.MEANINGFULLY_DISTINCT)
            continue
        relations = [
            _relation(action, actions[prior], thresholds) for prior in range(index)
        ]
        if CandidateDisposition.EXACT_DUPLICATE in relations:
            dispositions.append(CandidateDisposition.EXACT_DUPLICATE)
        elif all(
            relation == CandidateDisposition.MEANINGFULLY_DISTINCT
            for relation in relations
        ):
            dispositions.append(CandidateDisposition.MEANINGFULLY_DISTINCT)
        else:
            dispositions.append(CandidateDisposition.NEAR_DUPLICATE)
    return tuple(dispositions)


def candidate_group_metrics(
    candidate_ids: list[str],
    actions: list[npt.ArrayLike],
    thresholds: CandidateDiversityThresholds,
) -> dict[str, Any]:
    """Return complete pairwise metrics and content-based dispositions."""
    if len(candidate_ids) != len(actions) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise ValueError("candidate ids must align and be unique")
    dispositions = classify_candidate_group(actions, thresholds)
    pairs = []
    for left, right in combinations(range(len(actions)), 2):
        pairs.append(
            {
                "left_candidate_id": candidate_ids[left],
                "right_candidate_id": candidate_ids[right],
                **pairwise_action_metrics(
                    actions[left],
                    actions[right],
                    gripper_epsilon=thresholds.gripper_disagreement_epsilon,
                ),
            }
        )
    return {
        "candidate_count": len(actions),
        "dispositions": {
            candidate_id: disposition.value
            for candidate_id, disposition in zip(
                candidate_ids,
                dispositions,
                strict=True,
            )
        },
        "meaningfully_distinct_count": sum(
            disposition == CandidateDisposition.MEANINGFULLY_DISTINCT
            for disposition in dispositions
        ),
        "pairs": pairs,
    }


__all__ = [
    "CandidateDiversityThresholds",
    "action_content_sha256",
    "candidate_group_metrics",
    "classify_candidate_group",
    "pairwise_action_metrics",
]
