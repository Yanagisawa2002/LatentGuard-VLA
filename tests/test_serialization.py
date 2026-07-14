"""Round-trip and safety tests for versioned local episode bundles."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import latentguard.serialization as serialization_module
from latentguard.models import LabelSource, LabelStrength
from latentguard.serialization import (
    MANIFEST_NAME,
    SerializationError,
    UnsupportedSerializationVersionError,
    load_episodes,
    save_episodes,
)
from latentguard.synthetic import generate_synthetic_episodes
from latentguard.validation import DataValidationError


def _episodes(*, depth_enabled: bool = True, camera_count: int = 2):
    return generate_synthetic_episodes(
        seed=42,
        episode_count=2,
        episode_length=3,
        action_dim=5,
        robot_state_dim=4,
        camera_count=camera_count,
        image_height=4,
        image_width=6,
        depth_enabled=depth_enabled,
        candidate_count=4,
    )


def _manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_complete_round_trip_preserves_arrays_labels_ids_and_provenance(
    tmp_path: Path,
) -> None:
    original = _episodes()
    manifest_path = save_episodes(original, tmp_path)

    loaded = load_episodes(tmp_path)

    assert manifest_path == tmp_path / MANIFEST_NAME
    assert len(loaded) == len(original)
    for before, after in zip(original, loaded, strict=True):
        assert after.episode_id == before.episode_id
        assert after.schema_version == before.schema_version
        assert after.split_group_id == before.split_group_id
        assert after.observations.history_id == before.observations.history_id
        for before_frame, after_frame in zip(
            before.observations.frames, after.observations.frames, strict=True
        ):
            assert after_frame.timestamp_s == before_frame.timestamp_s
            assert after_frame.robot_state.dtype == before_frame.robot_state.dtype
            assert after_frame.robot_state.shape == before_frame.robot_state.shape
            np.testing.assert_array_equal(
                after_frame.robot_state, before_frame.robot_state
            )
            for before_camera, after_camera in zip(
                before_frame.cameras, after_frame.cameras, strict=True
            ):
                assert after_camera.camera_id == before_camera.camera_id
                assert after_camera.rgb.dtype == before_camera.rgb.dtype
                assert after_camera.rgb.shape == before_camera.rgb.shape
                np.testing.assert_array_equal(after_camera.rgb, before_camera.rgb)
                assert after_camera.depth is not None
                assert before_camera.depth is not None
                assert after_camera.depth.dtype == before_camera.depth.dtype
                assert after_camera.depth.shape == before_camera.depth.shape
                assert after_camera.intrinsics is not None
                assert before_camera.intrinsics is not None
                assert after_camera.extrinsics is not None
                assert before_camera.extrinsics is not None
                np.testing.assert_array_equal(
                    after_camera.intrinsics, before_camera.intrinsics
                )
                np.testing.assert_array_equal(
                    after_camera.extrinsics, before_camera.extrinsics
                )
        for before_candidate, after_candidate in zip(
            before.candidates, after.candidates, strict=True
        ):
            assert after_candidate.candidate_id == before_candidate.candidate_id
            assert (
                after_candidate.action.actions.dtype
                == before_candidate.action.actions.dtype
            )
            assert (
                after_candidate.action.actions.shape
                == before_candidate.action.actions.shape
            )
            np.testing.assert_array_equal(
                after_candidate.action.actions, before_candidate.action.actions
            )
            assert after_candidate.outcome.label_source is LabelSource(
                before_candidate.outcome.label_source
            )
            assert after_candidate.outcome.label_strength is LabelStrength(
                before_candidate.outcome.label_strength
            )
            assert (
                after_candidate.outcome.simulator_replay_verified
                == before_candidate.outcome.simulator_replay_verified
            )
            assert dict(after_candidate.provenance.transformation_parameters) == dict(
                before_candidate.provenance.transformation_parameters
            )
            assert after_candidate.provenance.source_episode_id == before.episode_id


@pytest.mark.parametrize(("depth_enabled", "camera_count"), [(False, 1), (False, 0)])
def test_optional_depth_and_zero_camera_state_are_preserved(
    tmp_path: Path, depth_enabled: bool, camera_count: int
) -> None:
    save_episodes(
        _episodes(depth_enabled=depth_enabled, camera_count=camera_count), tmp_path
    )

    loaded = load_episodes(tmp_path)

    cameras = loaded[0].observations.frames[0].cameras
    assert len(cameras) == camera_count
    assert all(camera.depth is None for camera in cameras)


def test_manifest_is_human_readable_and_arrays_are_non_pickle_npy(
    tmp_path: Path,
) -> None:
    save_episodes(_episodes(camera_count=1), tmp_path)

    text = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")
    manifest = json.loads(text)
    array_paths = sorted((tmp_path / "arrays").glob("*.npy"))

    assert text.startswith("{\n")
    assert manifest["format"] == "latentguard-episode-bundle"
    assert manifest["serialization_version"] == 1
    assert manifest["schema_version"] == "1.0"
    assert array_paths
    for path in array_paths:
        loaded = np.load(path, allow_pickle=False)
        assert isinstance(loaded, np.ndarray)
        assert not loaded.dtype.hasobject


def test_same_content_produces_byte_identical_bundle_files(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    episodes = _episodes(camera_count=1)

    save_episodes(episodes, first)
    save_episodes(episodes, second)

    first_files = sorted(
        path.relative_to(first) for path in first.rglob("*") if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    assert first_files == second_files
    assert all(
        (first / path).read_bytes() == (second / path).read_bytes()
        for path in first_files
    )


@pytest.mark.parametrize("field", ["serialization_version", "schema_version"])
def test_unsupported_versions_fail_clearly(tmp_path: Path, field: str) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    manifest = _manifest(tmp_path)
    manifest[field] = 999 if field == "serialization_version" else "999.0"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(UnsupportedSerializationVersionError, match="unsupported"):
        load_episodes(tmp_path)


def test_duplicate_json_fields_are_rejected(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_text(
        '{"format": "first", "format": "second"}',
        encoding="utf-8",
    )

    with pytest.raises(SerializationError, match="duplicate field 'format'"):
        load_episodes(tmp_path)


def test_array_path_traversal_is_rejected(tmp_path: Path) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    manifest = _manifest(tmp_path)
    robot_state = manifest["episodes"][0]["observations"]["frames"][0]["robot_state"]
    robot_state["__array__"]["path"] = "../outside.npy"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(SerializationError, match="unsafe array path"):
        load_episodes(tmp_path)


def test_array_path_with_embedded_nul_is_rejected(tmp_path: Path) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    manifest = _manifest(tmp_path)
    robot_state = manifest["episodes"][0]["observations"]["frames"][0]["robot_state"]
    robot_state["__array__"]["path"] = "arrays/\x00.npy"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(SerializationError, match="unsafe array path"):
        load_episodes(tmp_path)


@pytest.mark.parametrize(
    ("metadata", "message"), [("dtype", "dtype"), ("shape", "shape")]
)
def test_array_metadata_mismatch_is_rejected(
    tmp_path: Path, metadata: str, message: str
) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    manifest = _manifest(tmp_path)
    reference = manifest["episodes"][0]["observations"]["frames"][0]["robot_state"][
        "__array__"
    ]
    reference[metadata] = "<f8" if metadata == "dtype" else [999]
    _write_manifest(tmp_path, manifest)

    with pytest.raises(SerializationError, match=message):
        load_episodes(tmp_path)


def test_loading_revalidates_reconstructed_episode(tmp_path: Path) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    manifest = _manifest(tmp_path)
    manifest["episodes"][0]["candidates"][0]["outcome"]["progress"] = 2.0
    _write_manifest(tmp_path, manifest)

    with pytest.raises(DataValidationError, match="progress"):
        load_episodes(tmp_path)


def test_nonempty_destination_is_refused_without_modifying_valid_bundle(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bundle"
    save_episodes(_episodes(camera_count=1), destination)
    before = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }

    with pytest.raises(SerializationError, match="absent or empty"):
        save_episodes(_episodes(camera_count=0), destination)

    after = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_failed_staged_save_leaves_empty_destination_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "save", fail_save)
    with pytest.raises(SerializationError, match="transactionally"):
        save_episodes(_episodes(camera_count=0), destination)

    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_cleanup_failure_does_not_mask_primary_save_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "bundle"
    destination.mkdir()
    original_rmtree = shutil.rmtree

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected write failure")

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected cleanup failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(np, "save", fail_save)
        scoped.setattr(serialization_module.shutil, "rmtree", fail_cleanup)
        with pytest.raises(SerializationError, match="transactionally") as caught:
            save_episodes(_episodes(camera_count=0), destination)

    assert "injected write failure" in str(caught.value)
    assert any(
        "cleanup also failed" in note.lower()
        for note in getattr(caught.value, "__notes__", ())
    )
    for staging in tmp_path.glob(".bundle.staging-*"):
        original_rmtree(staging)


def test_truncated_array_is_reported_as_serialization_error(tmp_path: Path) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    (tmp_path / "arrays" / "000000.npy").write_bytes(b"")

    with pytest.raises(SerializationError, match="could not load safely"):
        load_episodes(tmp_path)


def test_corrupt_zip_like_array_is_reported_as_serialization_error(
    tmp_path: Path,
) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    (tmp_path / "arrays" / "000000.npy").write_bytes(b"PK\x03\x04broken")

    with pytest.raises(SerializationError, match="could not load safely"):
        load_episodes(tmp_path)


def test_hard_linked_array_file_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    external = tmp_path / "external.npy"
    save_episodes(_episodes(camera_count=0), bundle)
    array_path = bundle / "arrays" / "000000.npy"
    shutil.copy2(array_path, external)
    array_path.unlink()
    os.link(external, array_path)

    with pytest.raises(SerializationError, match="hard-linked"):
        load_episodes(bundle)


def test_hard_linked_manifest_file_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    external = tmp_path / "external.json"
    save_episodes(_episodes(camera_count=0), bundle)
    manifest_path = bundle / MANIFEST_NAME
    shutil.copy2(manifest_path, external)
    manifest_path.unlink()
    os.link(external, manifest_path)

    with pytest.raises(SerializationError, match="hard-linked"):
        load_episodes(bundle)


def test_bundle_root_symbolic_link_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    alias = tmp_path / "alias"
    save_episodes(_episodes(camera_count=0), bundle)
    try:
        alias.symlink_to(bundle, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable")

    with pytest.raises(SerializationError, match="symbolic links and junctions"):
        load_episodes(alias)


@pytest.mark.parametrize("mutation", ["missing_count", "wrong_count", "extra_field"])
def test_manifest_integrity_fields_are_enforced(tmp_path: Path, mutation: str) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    manifest = _manifest(tmp_path)
    if mutation == "missing_count":
        del manifest["array_count"]
    elif mutation == "wrong_count":
        manifest["array_count"] += 1
    else:
        manifest["unexpected"] = "discard-me"
    _write_manifest(tmp_path, manifest)

    with pytest.raises(SerializationError):
        load_episodes(tmp_path)


def test_unreferenced_array_file_is_rejected(tmp_path: Path) -> None:
    save_episodes(_episodes(camera_count=0), tmp_path)
    np.save(tmp_path / "arrays" / "unreferenced.npy", np.zeros(1), allow_pickle=False)

    with pytest.raises(SerializationError, match="exactly match"):
        load_episodes(tmp_path)


def test_numpy_scalar_metadata_round_trips(tmp_path: Path) -> None:
    episode = _episodes(camera_count=0)[0]
    frames = list(episode.observations.frames)
    frames[0] = replace(frames[0], timestamp_s=np.float32(0.0))
    observations = replace(episode.observations, frames=tuple(frames))
    candidates = list(episode.candidates)
    candidates[0] = replace(
        candidates[0],
        provenance=replace(candidates[0].provenance, seed=np.int64(42)),
    )
    episode = replace(
        episode,
        observations=observations,
        candidates=tuple(candidates),
    )

    save_episodes((episode,), tmp_path)
    loaded = load_episodes(tmp_path)[0]

    assert loaded.observations.frames[0].timestamp_s == 0.0
    assert loaded.candidates[0].provenance.seed == 42


def test_oversized_json_integer_is_wrapped_as_serialization_error(
    tmp_path: Path,
) -> None:
    (tmp_path / MANIFEST_NAME).write_text(
        '{"oversized": ' + ("1" * 5000) + "}",
        encoding="utf-8",
    )

    with pytest.raises(SerializationError, match="could not read"):
        load_episodes(tmp_path)


def test_json_parser_recursion_error_is_wrapped_as_serialization_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    def fail_loads(*_args: object, **_kwargs: object) -> object:
        raise RecursionError("injected parser recursion")

    monkeypatch.setattr(serialization_module.json, "loads", fail_loads)

    with pytest.raises(SerializationError, match="could not read"):
        load_episodes(tmp_path)
