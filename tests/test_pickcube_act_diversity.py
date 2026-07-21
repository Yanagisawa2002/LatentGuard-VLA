"""CPU-only checkpoint diversity metric tests."""

from __future__ import annotations

import numpy as np

from latentguard.policies.act.diversity import (
    compare_action_chunks,
    summarize_checkpoint_diversity,
)


def test_chunk_metrics_detect_exact_and_meaningful_differences() -> None:
    left = np.zeros((16, 8), dtype=np.float64)
    identical = compare_action_chunks(
        left,
        left.copy(),
        chunk_l2_threshold=1e-3,
        first_action_l2_threshold=1e-4,
    )
    assert identical["exact_duplicate"] is True
    assert identical["meaningfully_distinct"] is False
    right = left.copy()
    right[0, 0] = 0.1
    different = compare_action_chunks(
        left,
        right,
        chunk_l2_threshold=1e-3,
        first_action_l2_threshold=1e-4,
    )
    assert different["exact_duplicate"] is False
    assert different["meaningfully_distinct"] is True


def test_summary_requires_shared_twenty_anchor_inventory() -> None:
    anchors = tuple(f"anchor-{index}" for index in range(20))
    base = [np.zeros((16, 8), dtype=np.float64) for _ in anchors]
    changed = [value.copy() for value in base]
    for value in changed:
        value[0, 0] = 0.1
    summary = summarize_checkpoint_diversity(
        chunks_by_role={"early": base, "best": changed},
        anchor_ids=anchors,
        chunk_l2_threshold=1e-3,
        first_action_l2_threshold=1e-4,
        minimum_distinct_ratio=0.7,
    )
    assert summary["exact_duplicate_checkpoint_pair_exists"] is False
    assert summary["pair_comparisons"][0]["passes_minimum_distinct_ratio"] is True
