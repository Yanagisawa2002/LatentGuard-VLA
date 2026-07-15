from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    STATE_INDEXED_ARCHIVE_FORMAT,
    STATE_INDEXED_ARCHIVE_VERSION,
    STATE_INDEXED_EPISODE_SCHEMA_VERSION,
    STATE_INDEXED_STATE_SCHEMA_VERSION,
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
    StateIndexedArchiveError,
    UnsupportedStateIndexedArchiveVersionError,
    assert_state_indexed_archive_unchanged,
    find_state_indexed_episode,
    load_indexed_state,
    load_state_indexed_archive,
    save_state_indexed_archive,
)
from latentguard.integrations.maniskill_pickcube.state_tree import (
    compute_state_tree_digest,
    compute_state_tree_structure_digest,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY,
    build_pickcube_verifier_state_v1,
)

_COMPATIBILITY = f"sha256:{'a' * 64}"


def _task(index: int, terminal: int = 2) -> PickCubeTaskSnapshotV1:
    return PickCubeTaskSnapshotV1(
        success=index == terminal,
        is_obj_placed=index == terminal,
        is_robot_static=index == terminal,
        is_grasped=index >= 1,
        cube_center_z=0.02,
        cube_to_goal_distance=float(terminal - index) * 0.1,
        tcp_to_cube_distance=float(max(1 - index, 0)) * 0.1,
    )


def _restored_task(index: int, terminal: int = 2) -> PickCubeTaskSnapshotV1:
    return PickCubeTaskSnapshotV1(
        success=index == terminal,
        is_obj_placed=index == terminal,
        is_robot_static=index == terminal,
        is_grasped=False,
        cube_center_z=0.02,
        cube_to_goal_distance=float(terminal - index) * 0.1,
        tcp_to_cube_distance=float(max(1 - index, 0)) * 0.1,
    )


def _vector(index: int, *, joint_names: tuple[str, ...] = ("joint_a", "joint_b")):
    return build_pickcube_verifier_state_v1(
        joint_names=joint_names,
        qpos=np.array([index, -index], dtype=np.float32),
        qvel=np.array([0.1, -0.1], dtype=np.float32),
        tcp_position=np.array([0.0, 0.0, 0.1 + index], dtype=np.float32),
        tcp_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        cube_position=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        cube_quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        goal_position=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        is_grasped=False,
        is_obj_placed=index == 2,
        is_robot_static=index == 2,
    )


def _state(
    index: int,
    *,
    trajectory_id: str = "mspc-trajectory-source-a",
    tree: object | None = None,
    joint_names: tuple[str, ...] = ("joint_a", "joint_b"),
) -> PickCubeIndexedStateV1:
    resolved_tree = tree or {
        "actors": np.array([index, index + 0.25], dtype=np.float32),
        "articulations": {
            "qpos": np.array([index * 0.1, -index * 0.1], dtype=np.float64)
        },
    }
    return PickCubeIndexedStateV1(
        state_index=index,
        source_action_index=index,
        tree=resolved_tree,
        task_snapshot=_task(index),
        restored_task_snapshot=_restored_task(index),
        verifier_state=_vector(index, joint_names=joint_names),
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_trajectory_id=trajectory_id,
    )


def _episode() -> PickCubeStateIndexedEpisodeV1:
    return PickCubeStateIndexedEpisodeV1(
        episode_id="mspc-sequence-episode-a",
        source_trajectory_id="mspc-trajectory-source-a",
        source_policy_identity="maniskill/official-pickcube-solver-v1",
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_actions=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
        states=(_state(0), _state(1), _state(2)),
    )


def _archive() -> PickCubeStateIndexedArchiveV1:
    return PickCubeStateIndexedArchiveV1(episodes=(_episode(),))


def _manifest(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((path / "manifest.json").read_text(encoding="utf-8")),
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_models_enforce_t_plus_one_and_store_complete_state_inventory() -> None:
    episode = _episode()

    assert len(episode.states) == episode.source_actions.shape[0] + 1
    assert tuple(state.state_index for state in episode.states) == (0, 1, 2)
    assert tuple(state.source_action_index for state in episode.states) == (0, 1, 2)
    assert {state.leaf_count for state in episode.states} == {2}
    assert {state.numeric_component_count for state in episode.states} == {4}
    assert all(state.state_digest.startswith("sha256:") for state in episode.states)
    assert all(
        state.structure_digest == episode.states[0].structure_digest
        for state in episode.states
    )
    assert episode.states[-1].task_snapshot.success is True
    assert episode.states[1].task_snapshot.is_grasped is True
    assert episode.states[1].restored_task_snapshot.is_grasped is False
    assert episode.states[1].verifier_state.values[-3] == 0.0


def test_archive_round_trip_is_path_independent_and_selects_intermediate_state(
    tmp_path: Path,
) -> None:
    archive = _archive()
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_state_indexed_archive(archive, first)
    save_state_indexed_archive(archive, second)

    first_loaded = load_state_indexed_archive(first)
    second_loaded = load_state_indexed_archive(second)
    selected = load_indexed_state(first, _episode().episode_id, 1)

    assert first_loaded.content_digest == archive.content_digest
    assert second_loaded.content_digest == archive.content_digest
    assert first_loaded.content_digest == second_loaded.content_digest
    assert selected.archive_content_digest == archive.content_digest
    assert selected.state.state_index == 1
    assert selected.state.state_digest == _episode().states[1].state_digest
    assert selected.state.numeric_component_count == 4
    assert selected.state.verifier_state.schema_digest == (
        _episode().states[1].verifier_state.schema_digest
    )
    assert find_state_indexed_episode(first_loaded, _episode().episode_id).seed == 7


def test_archive_manifest_and_models_record_all_required_per_state_fields(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archive"
    save_state_indexed_archive(_archive(), output)
    manifest = _manifest(output)
    encoded = manifest["episodes"][0]["states"][1]

    assert manifest["format"] == STATE_INDEXED_ARCHIVE_FORMAT
    assert manifest["serialization_version"] == STATE_INDEXED_ARCHIVE_VERSION
    assert encoded["state_index"] == 1
    assert encoded["source_action_index"] == 1
    assert encoded["seed"] == 7
    assert encoded["compatibility_identity"] == _COMPATIBILITY
    assert encoded["source_trajectory_id"] == "mspc-trajectory-source-a"
    assert encoded["leaf_count"] == 2
    assert encoded["numeric_component_count"] == 4
    assert encoded["state_digest"] == compute_state_tree_digest(_state(1).tree)
    assert encoded["structure_digest"] == compute_state_tree_structure_digest(
        _state(1).tree
    )
    assert encoded["task_snapshot"]["is_grasped"] is True
    assert encoded["restored_task_snapshot"]["is_grasped"] is False
    assert (
        encoded["verifier_state"]["extraction_boundary"]
        == PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY
    )
    assert encoded["verifier_state"]["schema_digest"].startswith("sha256:")
    assert encoded["state_content_digest"].startswith("sha256:")


def test_archive_detaches_source_actions_state_trees_and_vectors() -> None:
    actions = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    tree_leaf = np.array([0.0, 0.25], dtype=np.float32)
    state = _state(0, tree={"actors": tree_leaf, "other": np.zeros(2)})
    terminal_one = _state(1, tree={"actors": tree_leaf + 1, "other": np.zeros(2)})
    terminal_two = _state(2, tree={"actors": tree_leaf + 2, "other": np.zeros(2)})
    episode = PickCubeStateIndexedEpisodeV1(
        episode_id="mspc-sequence-episode-a",
        source_trajectory_id="mspc-trajectory-source-a",
        source_policy_identity="maniskill/official-pickcube-solver-v1",
        seed=7,
        compatibility_identity=_COMPATIBILITY,
        source_actions=actions,
        states=(state, terminal_one, terminal_two),
    )
    actions[...] = 9.0
    tree_leaf[...] = 9.0

    assert np.array_equal(
        episode.source_actions,
        np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
    )
    assert compute_state_tree_digest(episode.states[0].tree) == state.state_digest


def test_episode_rejects_missing_state_or_out_of_order_state_index() -> None:
    episode = _episode()
    with pytest.raises(StateIndexedArchiveError, match=r"T\+1"):
        replace(episode, states=episode.states[:-1])

    wrong_first = replace(episode.states[0], state_index=1, source_action_index=1)
    with pytest.raises(StateIndexedArchiveError, match="ordered"):
        replace(episode, states=(wrong_first, *episode.states[1:]))


def test_episode_rejects_metadata_structure_or_vector_schema_drift() -> None:
    episode = _episode()
    wrong_trajectory = replace(
        episode.states[1], source_trajectory_id="mspc-trajectory-other"
    )
    with pytest.raises(StateIndexedArchiveError, match="metadata"):
        replace(
            episode,
            states=(episode.states[0], wrong_trajectory, episode.states[2]),
        )

    changed_tree = _state(
        1,
        tree={
            "actors": np.array([1.0, 1.25], dtype=np.float32),
            "different": np.array([0.1, -0.1], dtype=np.float64),
        },
    )
    with pytest.raises(StateIndexedArchiveError, match="tree inventory"):
        replace(episode, states=(episode.states[0], changed_tree, episode.states[2]))

    changed_vector = _state(1, joint_names=("other_a", "other_b"))
    with pytest.raises(StateIndexedArchiveError, match="component schema"):
        replace(episode, states=(episode.states[0], changed_vector, episode.states[2]))


def test_episode_requires_successful_terminal_source() -> None:
    episode = _episode()
    failed_terminal = replace(
        episode.states[2],
        task_snapshot=replace(episode.states[2].task_snapshot, success=False),
    )
    with pytest.raises(StateIndexedArchiveError, match="end in success"):
        replace(episode, states=(*episode.states[:2], failed_terminal))


def test_task_snapshot_rejects_missing_boolean_type_or_invalid_distance() -> None:
    with pytest.raises(StateIndexedArchiveError, match="boolean"):
        replace(_task(0), is_grasped=np.bool_(False))  # type: ignore[arg-type]
    with pytest.raises(StateIndexedArchiveError, match="non-negative"):
        replace(_task(0), tcp_to_cube_distance=-0.1)


def test_indexed_state_rejects_restored_snapshot_or_vector_flag_mismatch() -> None:
    state = _state(1)
    with pytest.raises(StateIndexedArchiveError, match="restored PickCube"):
        replace(state, restored_task_snapshot=object())  # type: ignore[arg-type]

    changed_restored = replace(state.restored_task_snapshot, is_grasped=True)
    with pytest.raises(StateIndexedArchiveError, match="task flags differ"):
        replace(state, restored_task_snapshot=changed_restored)

    changed_vector = _vector(1)
    values = np.array(changed_vector.values, copy=True)
    values[-2] = 1.0
    with pytest.raises(StateIndexedArchiveError, match="task flags differ"):
        replace(
            state,
            verifier_state=replace(changed_vector, values=values),
        )


def test_restored_task_snapshot_is_content_bound_independently() -> None:
    state = _state(1)
    changed = replace(
        state,
        restored_task_snapshot=replace(
            state.restored_task_snapshot,
            tcp_to_cube_distance=0.25,
        ),
    )

    assert changed.task_snapshot == state.task_snapshot
    assert changed.verifier_state.content_digest == state.verifier_state.content_digest
    assert changed.content_digest != state.content_digest


def test_archive_rejects_tampered_array_bytes(tmp_path: Path) -> None:
    output = tmp_path / "archive"
    save_state_indexed_archive(_archive(), output)
    manifest = _manifest(output)
    relative = manifest["episodes"][0]["source_actions"]["path"]
    path = output / relative
    loaded = np.array(np.load(path, allow_pickle=False), copy=True)
    loaded[0, 0] += 0.5
    with path.open("wb") as stream:
        np.save(stream, loaded, allow_pickle=False)

    with pytest.raises(StateIndexedArchiveError, match="content"):
        load_state_indexed_archive(output)


def test_archive_rejects_tampered_state_digest_or_unknown_manifest_field(
    tmp_path: Path,
) -> None:
    first = tmp_path / "digest"
    save_state_indexed_archive(_archive(), first)
    manifest = _manifest(first)
    manifest["episodes"][0]["states"][0]["state_digest"] = f"sha256:{'0' * 64}"
    _write_manifest(first, manifest)
    with pytest.raises(StateIndexedArchiveError, match="state_digest"):
        load_state_indexed_archive(first)

    second = tmp_path / "unknown"
    save_state_indexed_archive(_archive(), second)
    manifest = _manifest(second)
    manifest["unexpected"] = True
    _write_manifest(second, manifest)
    with pytest.raises(StateIndexedArchiveError, match="unexpected"):
        load_state_indexed_archive(second)


def test_archive_rejects_persisted_restored_projection_and_schema_tampering(
    tmp_path: Path,
) -> None:
    restored = tmp_path / "restored-task"
    save_state_indexed_archive(_archive(), restored)
    manifest = _manifest(restored)
    manifest["episodes"][0]["states"][1]["restored_task_snapshot"][
        "tcp_to_cube_distance"
    ] = 0.25
    _write_manifest(restored, manifest)
    with pytest.raises(StateIndexedArchiveError, match="state_content_digest"):
        load_state_indexed_archive(restored)

    state_version = tmp_path / "state-version"
    save_state_indexed_archive(_archive(), state_version)
    manifest = _manifest(state_version)
    manifest["episodes"][0]["states"][0]["schema_version"] = "1.0"
    _write_manifest(state_version, manifest)
    with pytest.raises(UnsupportedStateIndexedArchiveVersionError):
        load_state_indexed_archive(state_version)

    episode_version = tmp_path / "episode-version"
    save_state_indexed_archive(_archive(), episode_version)
    manifest = _manifest(episode_version)
    manifest["episodes"][0]["schema_version"] = "1.0"
    _write_manifest(episode_version, manifest)
    with pytest.raises(UnsupportedStateIndexedArchiveVersionError):
        load_state_indexed_archive(episode_version)

    extraction_boundary = tmp_path / "extraction-boundary"
    save_state_indexed_archive(_archive(), extraction_boundary)
    manifest = _manifest(extraction_boundary)
    manifest["episodes"][0]["states"][0]["verifier_state"]["extraction_boundary"] = (
        "source_time_before_archive_v0"
    )
    _write_manifest(extraction_boundary, manifest)
    with pytest.raises(StateIndexedArchiveError, match="extraction_boundary"):
        load_state_indexed_archive(extraction_boundary)


def test_archive_rejects_path_traversal_and_unmanifested_array(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal"
    save_state_indexed_archive(_archive(), traversal)
    manifest = _manifest(traversal)
    manifest["episodes"][0]["source_actions"]["path"] = "../outside.npy"
    _write_manifest(traversal, manifest)
    with pytest.raises(StateIndexedArchiveError, match="unsafe array path"):
        load_state_indexed_archive(traversal)

    inventory = tmp_path / "inventory"
    save_state_indexed_archive(_archive(), inventory)
    with (inventory / "arrays" / "unmanifested.npy").open("wb") as stream:
        np.save(stream, np.zeros(1, dtype=np.float32), allow_pickle=False)
    with pytest.raises(StateIndexedArchiveError, match="exactly match"):
        load_state_indexed_archive(inventory)


def test_archive_rejects_unsupported_version_and_nonempty_destination(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archive"
    save_state_indexed_archive(_archive(), output)
    manifest = _manifest(output)
    assert STATE_INDEXED_ARCHIVE_VERSION == 2
    manifest["serialization_version"] = 1
    _write_manifest(output, manifest)
    with pytest.raises(UnsupportedStateIndexedArchiveVersionError):
        load_state_indexed_archive(output)

    with pytest.raises(StateIndexedArchiveError, match="non-empty"):
        save_state_indexed_archive(_archive(), output)


def test_models_reject_old_state_and_episode_schema_versions() -> None:
    assert STATE_INDEXED_STATE_SCHEMA_VERSION == "1.1"
    assert STATE_INDEXED_EPISODE_SCHEMA_VERSION == "1.1"
    with pytest.raises(UnsupportedStateIndexedArchiveVersionError):
        replace(_state(0), schema_version="1.0")
    with pytest.raises(UnsupportedStateIndexedArchiveVersionError):
        replace(_episode(), schema_version="1.0")


def test_archive_binding_and_lookup_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "archive"
    archive = _archive()
    save_state_indexed_archive(archive, output)

    assert_state_indexed_archive_unchanged(output, archive.content_digest)
    with pytest.raises(StateIndexedArchiveError, match="unknown"):
        find_state_indexed_episode(archive, "missing-episode")
    with pytest.raises(StateIndexedArchiveError, match="outside"):
        load_indexed_state(output, _episode().episode_id, 3)
    with pytest.raises(StateIndexedArchiveError, match="expected"):
        assert_state_indexed_archive_unchanged(output, "not-a-digest")


def test_archive_rejects_duplicate_trajectory_or_seed() -> None:
    first = _episode()
    duplicate = replace(first, episode_id="mspc-sequence-episode-b")
    with pytest.raises(StateIndexedArchiveError, match="source trajectory IDs"):
        PickCubeStateIndexedArchiveV1(episodes=(first, duplicate))

    second_trajectory = "mspc-trajectory-source-b"
    states = tuple(
        replace(state, source_trajectory_id=second_trajectory) for state in first.states
    )
    same_seed = replace(
        first,
        episode_id="mspc-sequence-episode-b",
        source_trajectory_id=second_trajectory,
        states=states,
    )
    with pytest.raises(StateIndexedArchiveError, match="source seeds"):
        PickCubeStateIndexedArchiveV1(episodes=(first, same_seed))
