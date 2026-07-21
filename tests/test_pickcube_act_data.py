"""CPU-only episode, split, alignment, and normalization tests for P0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latentguard.policies.act.data import (
    PickCubeDemoDataError,
    PickCubeDemoEpisode,
    PickCubeDemoSplit,
    action_chunk_at,
    build_demo_dataset_reports,
    collect_episode_references,
    read_demo_episode,
    save_demo_episode,
    split_for_scene_seed,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64


def _episode(seed: int, *, offset: int = 0) -> PickCubeDemoEpisode:
    rgb = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    rgb[0, ..., 0] = np.uint8((seed + offset) % 256)
    rgb[1, ..., 1] = np.uint8((seed + offset + 10) % 256)
    state = np.full((2, 18), seed + offset, dtype=np.float32)
    actions = np.full((2, 8), (seed + offset) / 100.0, dtype=np.float64)
    return PickCubeDemoEpisode(
        scene_seed=seed,
        rgb=rgb,
        proprioception=state,
        actions=actions,
        phases=("REACH_PREGRASP", "DESCEND_TO_GRASP"),
        compatibility_identity=_DIGEST_A,
        contract_digest=_DIGEST_B,
        camera_configuration_digest=_DIGEST_C,
    )


def test_scene_seed_split_is_exact_80_10_10_over_decades() -> None:
    splits = [split_for_scene_seed(seed) for seed in range(100)]
    assert splits.count(PickCubeDemoSplit.TRAIN) == 80
    assert splits.count(PickCubeDemoSplit.VALIDATION) == 10
    assert splits.count(PickCubeDemoSplit.TEST) == 10


def test_episode_round_trip_preserves_float64_executed_actions(tmp_path: Path) -> None:
    source = _episode(2)
    reference = save_demo_episode(tmp_path, source)
    loaded = read_demo_episode(tmp_path, reference)

    assert loaded.episode_id == source.episode_id
    assert loaded.actions.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(loaded.rgb, source.rgb)
    np.testing.assert_array_equal(loaded.proprioception, source.proprioception)
    np.testing.assert_array_equal(loaded.actions, source.actions)
    assert loaded.split is PickCubeDemoSplit.TRAIN


def test_episode_alignment_is_fail_closed() -> None:
    source = _episode(2)
    with pytest.raises(PickCubeDemoDataError, match="alignment differs"):
        PickCubeDemoEpisode(
            scene_seed=source.scene_seed,
            rgb=source.rgb[:1],
            proprioception=source.proprioception,
            actions=source.actions,
            phases=source.phases,
            compatibility_identity=source.compatibility_identity,
            contract_digest=source.contract_digest,
            camera_configuration_digest=source.camera_configuration_digest,
        )


def test_dataset_reports_keep_episode_splits_and_train_only_stats(
    tmp_path: Path,
) -> None:
    for seed in range(20):
        save_demo_episode(tmp_path, _episode(seed))
    manifest, quality, normalization = build_demo_dataset_reports(
        tmp_path,
        contract_digest=_DIGEST_B,
        expert_success_rate=0.97,
    )

    assert manifest["episode_count"] == 20
    assert manifest["frame_count"] == 40
    assert manifest["split_episode_counts"] == {
        "test": 2,
        "train": 16,
        "validation": 2,
    }
    assert quality["split_leakage_count"] == 0
    assert quality["duplicate_episode_count"] == 0
    assert quality["expert_success_rate"] == 0.97
    assert normalization["source_split"] == "train"
    assert str(normalization["normalization_digest"]).startswith("sha256:")
    json.dumps(dict(manifest))
    json.dumps(dict(quality))
    json.dumps(dict(normalization))


def test_duplicate_scene_seed_is_rejected_even_when_content_differs(
    tmp_path: Path,
) -> None:
    save_demo_episode(tmp_path, _episode(2))
    save_demo_episode(tmp_path, _episode(2, offset=1))
    with pytest.raises(PickCubeDemoDataError, match="duplicate scene seed"):
        collect_episode_references(tmp_path)


def test_array_tampering_is_detected_before_reload(tmp_path: Path) -> None:
    reference = save_demo_episode(tmp_path, _episode(2))
    metadata_path = tmp_path / Path(reference.relative_directory) / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frame_count"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PickCubeDemoDataError, match="file inventory changed"):
        read_demo_episode(tmp_path, reference)


def test_action_chunk_padding_preserves_valid_prefix() -> None:
    actions = np.arange(24, dtype=np.float64).reshape(3, 8)
    chunk, padding = action_chunk_at(actions, 1, 4)
    assert chunk.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(chunk[:2], actions[1:].astype(np.float32))
    np.testing.assert_array_equal(chunk[2:], np.zeros((2, 8), dtype=np.float32))
    np.testing.assert_array_equal(
        padding,
        np.array([False, False, True, True], dtype=np.bool_),
    )
