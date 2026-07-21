"""Checkpoint action-diversity metrics over content-bound real observations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray


class PickCubeActDiversityError(ValueError):
    """Raised when candidate chunks or anchor inventories are invalid."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActDiversityError(f"{context}: {reason}")


def action_chunk_digest(chunk: NDArray[Any]) -> str:
    """Return an exact dtype/shape/content digest for one candidate chunk."""
    value = np.asarray(chunk)
    if (
        value.ndim != 2
        or value.shape[1:] != (8,)
        or value.dtype.hasobject
        or not np.issubdtype(value.dtype, np.floating)
        or not np.all(np.isfinite(value))
    ):
        _fail("action chunk", "expected finite floating[T,8]")
    digest = hashlib.sha256(b"pickcube-act-diversity-chunk-v1\0")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def compare_action_chunks(
    left: NDArray[Any],
    right: NDArray[Any],
    *,
    chunk_l2_threshold: float,
    first_action_l2_threshold: float,
) -> Mapping[str, object]:
    """Compare two same-contract chunks without normalizing or repairing them."""
    first = np.asarray(left)
    second = np.asarray(right)
    action_chunk_digest(first)
    action_chunk_digest(second)
    if first.shape != second.shape or first.dtype != second.dtype:
        _fail("action chunk comparison", "dtype or shape differs")
    if (
        type(chunk_l2_threshold) not in (int, float)
        or type(first_action_l2_threshold) not in (int, float)
        or not math.isfinite(float(chunk_l2_threshold))
        or not math.isfinite(float(first_action_l2_threshold))
        or float(chunk_l2_threshold) <= 0.0
        or float(first_action_l2_threshold) <= 0.0
    ):
        _fail("action chunk comparison", "thresholds must be positive and finite")
    flat_first = first.astype(np.float64, copy=False).reshape(-1)
    flat_second = second.astype(np.float64, copy=False).reshape(-1)
    difference = flat_first - flat_second
    chunk_l2 = float(np.linalg.norm(difference))
    first_action_l2 = float(
        np.linalg.norm(first[0].astype(np.float64) - second[0].astype(np.float64))
    )
    endpoint_l2 = float(
        np.linalg.norm(first[-1].astype(np.float64) - second[-1].astype(np.float64))
    )
    denominator = float(np.linalg.norm(flat_first) * np.linalg.norm(flat_second))
    cosine_distance = (
        None
        if denominator == 0.0
        else 1.0 - float(np.dot(flat_first, flat_second) / denominator)
    )
    gripper_disagreement = float(
        np.mean(np.signbit(first[:, -1]) != np.signbit(second[:, -1]))
    )
    exact_duplicate = first.tobytes(order="C") == second.tobytes(order="C")
    meaningfully_distinct = (
        chunk_l2 > float(chunk_l2_threshold)
        or first_action_l2 > float(first_action_l2_threshold)
        or gripper_disagreement > 0.0
    )
    return {
        "chunk_l2_distance": chunk_l2,
        "cosine_distance": cosine_distance,
        "exact_duplicate": exact_duplicate,
        "first_action_l2_distance": first_action_l2,
        "gripper_disagreement_ratio": gripper_disagreement,
        "meaningfully_distinct": meaningfully_distinct,
        "chunk_endpoint_l2_distance": endpoint_l2,
    }


def summarize_checkpoint_diversity(
    *,
    chunks_by_role: Mapping[str, Sequence[NDArray[Any]]],
    anchor_ids: Sequence[str],
    chunk_l2_threshold: float,
    first_action_l2_threshold: float,
    minimum_distinct_ratio: float,
) -> Mapping[str, object]:
    """Summarize every pair over the exact same ordered anchor inventory."""
    roles = sorted(chunks_by_role)
    if (
        len(roles) < 2
        or len(anchor_ids) < 20
        or len(set(anchor_ids)) != len(anchor_ids)
    ):
        _fail("diversity summary", "requires two roles and 20 unique anchors")
    if not 0.0 <= minimum_distinct_ratio <= 1.0:
        _fail("diversity summary", "minimum ratio is invalid")
    if any(len(chunks_by_role[role]) != len(anchor_ids) for role in roles):
        _fail("diversity summary", "role/anchor inventories differ")
    pairs: list[dict[str, object]] = []
    for left_role, right_role in combinations(roles, 2):
        comparisons = [
            compare_action_chunks(
                left,
                right,
                chunk_l2_threshold=chunk_l2_threshold,
                first_action_l2_threshold=first_action_l2_threshold,
            )
            for left, right in zip(
                chunks_by_role[left_role],
                chunks_by_role[right_role],
                strict=True,
            )
        ]
        distinct_count = sum(
            item["meaningfully_distinct"] is True for item in comparisons
        )
        pairs.append(
            {
                "anchor_comparisons": [
                    {"anchor_id": anchor_id, **dict(comparison)}
                    for anchor_id, comparison in zip(
                        anchor_ids, comparisons, strict=True
                    )
                ],
                "exact_duplicate_anchor_count": sum(
                    item["exact_duplicate"] is True for item in comparisons
                ),
                "left_role": left_role,
                "meaningfully_distinct_anchor_count": distinct_count,
                "meaningfully_distinct_anchor_ratio": distinct_count / len(anchor_ids),
                "passes_minimum_distinct_ratio": (
                    distinct_count / len(anchor_ids) >= minimum_distinct_ratio
                ),
                "right_role": right_role,
            }
        )
    return {
        "anchor_count": len(anchor_ids),
        "anchor_ids": list(anchor_ids),
        "checkpoint_action_hashes": {
            role: [action_chunk_digest(chunk) for chunk in chunks_by_role[role]]
            for role in roles
        },
        "exact_duplicate_checkpoint_pair_exists": any(
            pair["exact_duplicate_anchor_count"] == len(anchor_ids) for pair in pairs
        ),
        "pair_comparisons": pairs,
        "role_count": len(roles),
        "roles": roles,
        "schema_version": "pickcube-native-act-checkpoint-diversity-v1",
    }


__all__ = [
    "PickCubeActDiversityError",
    "action_chunk_digest",
    "compare_action_chunks",
    "summarize_checkpoint_diversity",
]
