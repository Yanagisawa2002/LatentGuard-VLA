from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latentguard.integrations.maniskill_pickcube.archive import (
    REFERENCE_MANIFEST_NAME,
    ManiSkillReferenceArchive,
    ManiSkillReferenceEpisode,
    ReferenceArchiveError,
    UnsupportedReferenceArchiveVersionError,
    assert_reference_archive_unchanged,
    compute_reference_archive_digest,
    find_reference_episode,
    load_archive_state,
    load_reference_archive,
    save_reference_archive,
)
from latentguard.integrations.maniskill_pickcube.source_import import (
    MANISKILL_PICKCUBE_TASK_ID,
    OFFICIAL_PICKCUBE_SOURCE_POLICY_ID,
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_UNSAFE_SEMANTIC,
    M0SourceImportError,
    build_m0_source_episode,
    build_m0_source_episodes,
    save_m0_source_dataset,
)
from latentguard.integrations.maniskill_pickcube.state_tree import (
    STATE_TREE_SEMANTIC,
    StateTreeError,
    clone_state_tree,
    compare_state_trees,
    compute_state_tree_digest,
    compute_state_tree_structure_digest,
    flatten_state_tree,
    normalize_state_tree,
)
from latentguard.models import LabelSource, LabelStrength
from latentguard.serialization import load_episodes

_COMPATIBILITY_IDENTITY = "sha256:" + "1" * 64
_SOLVER_DIGEST = "sha256:" + "2" * 64


class _FakeTensor:
    def __init__(self, value: np.ndarray[Any]) -> None:
        self._value = np.array(value, copy=True)

    def detach(self) -> _FakeTensor:
        return _FakeTensor(self._value)

    def cpu(self) -> _FakeTensor:
        return _FakeTensor(self._value)

    def numpy(self) -> np.ndarray[Any]:
        return np.array(self._value, copy=True)


def _reference_episode(
    *,
    independent_baseline_success: bool = True,
    cube_center_z: float = 0.04,
) -> ManiSkillReferenceEpisode:
    return ManiSkillReferenceEpisode(
        compatibility_identity=_COMPATIBILITY_IDENTITY,
        source_solver_identity={
            "module_name": (
                "mani_skill.examples.motionplanning.panda.solutions.solvePickCube"
            ),
            "source_sha256": _SOLVER_DIGEST,
        },
        environment_configuration={
            "environment_id": "PickCube-v1",
            "robot_uid": "panda",
            "num_envs": 1,
            "control_mode": "pd_joint_pos",
            "sim_backend": "gpu",
        },
        seed=17,
        source_actions=np.array([[0.1, 0.2, -0.3], [0.2, 0.1, -0.2]], dtype=np.float32),
        initial_state={
            "actors": {
                "cube": np.array([0.1, -0.2, 0.03], dtype=np.float32),
            },
            "agent": [
                np.array([1, 2], dtype=np.int32),
                (np.float64(0.25), np.bool_(True)),
            ],
        },
        terminal_state={
            "actors": {
                "cube": np.array([0.0, 0.0, 0.04], dtype=np.float32),
            },
            "agent": [
                np.array([1, 3], dtype=np.int32),
                (np.float64(0.5), np.bool_(False)),
            ],
        },
        terminal_task_evidence={
            "success": True,
            "is_obj_placed": True,
            "is_robot_static": True,
            "is_grasped": False,
            "cube_center_z": cube_center_z,
            "cube_to_goal_distance": 0.0,
        },
        initial_robot_state=np.array(
            [0.0, 0.1, 0.2, 0.01, 0.02, 0.03], dtype=np.float32
        ),
        robot_state_joint_names=("panda_joint1", "panda_joint2", "panda_joint3"),
        robot_state_semantic="panda_named_joint_position_v1",
        action_contract={
            "contract_version": "remote-probe-bound-v1",
            "semantic_fields": "unspecified",
        },
        action_coordinate_frame="pd_joint_pos",
        control_period_s=0.05,
        source_generation_success=True,
        independent_baseline_success=independent_baseline_success,
    )


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_state_tree_nested_round_trip_preserves_structure_dtype_and_shape() -> None:
    source = {
        "z": [np.array([[1, 2]], dtype=np.int16)],
        "a/key": (np.float32(0.5), {"flag": np.bool_(True)}),
    }

    normalized = normalize_state_tree(source)
    cloned = clone_state_tree(normalized)
    comparison = compare_state_trees(normalized, cloned, atol=0.0)

    assert isinstance(cloned, dict)
    assert isinstance(cloned["z"], list)
    assert isinstance(cloned["a/key"], tuple)
    assert comparison.structure_matches
    assert comparison.exact_digest_match
    assert comparison.within_tolerance
    assert comparison.maximum_absolute_error == 0.0
    leaves = flatten_state_tree(normalized)
    assert [leaf.path for leaf in leaves] == ["$/a~1key/0", "$/a~1key/1/flag", "$/z/0"]
    assert leaves[-1].value.dtype == np.dtype(np.int16)
    assert leaves[-1].value.shape == (1, 2)


def test_state_tree_converts_tensor_like_leaves_without_runtime_imports() -> None:
    tensor = _FakeTensor(np.array([[1.0, 2.0]], dtype=np.float32))

    leaves = flatten_state_tree({"tensor": tensor})

    assert len(leaves) == 1
    assert leaves[0].value.dtype == np.dtype(np.float32)
    assert leaves[0].value.shape == (1, 2)
    assert not leaves[0].value.flags.writeable


def test_state_tree_digest_is_mapping_order_independent_and_value_bound() -> None:
    first = {"b": np.int32(2), "a": np.array([1.0], dtype=np.float64)}
    reordered = {"a": np.array([1.0], dtype=np.float64), "b": np.int32(2)}
    changed = {"a": np.array([1.1], dtype=np.float64), "b": np.int32(2)}

    assert compute_state_tree_digest(first) == compute_state_tree_digest(reordered)
    assert compute_state_tree_digest(first) != compute_state_tree_digest(changed)
    assert compute_state_tree_structure_digest(first) == (
        compute_state_tree_structure_digest(changed)
    )


def test_state_tree_comparison_rejects_dtype_and_shape_mismatch() -> None:
    expected = {"q": np.array([1.0, 2.0], dtype=np.float32)}
    wrong_dtype = {"q": np.array([1.0, 2.0], dtype=np.float64)}
    wrong_shape = {"q": np.array([[1.0, 2.0]], dtype=np.float32)}

    dtype_result = compare_state_trees(expected, wrong_dtype, atol=1e-6)
    shape_result = compare_state_trees(expected, wrong_shape, atol=1e-6)

    assert not dtype_result.structure_matches
    assert not dtype_result.within_tolerance
    assert dtype_result.maximum_absolute_error is None
    assert not shape_result.structure_matches


@pytest.mark.parametrize(
    "value",
    [
        np.array([object()], dtype=object),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float64),
    ],
)
def test_state_tree_rejects_object_and_nonfinite_arrays(value: np.ndarray[Any]) -> None:
    with pytest.raises(StateTreeError):
        normalize_state_tree({"bad": value})


def test_state_tree_rejects_cycles_without_repair() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(StateTreeError, match="cyclic"):
        normalize_state_tree(cyclic)


def test_reference_episode_detaches_all_source_arrays() -> None:
    actions = np.zeros((2, 3), dtype=np.float32)
    initial_leaf = np.array([1.0], dtype=np.float32)
    robot_state = np.array([0.0, 0.1, 0.2, 0.01, 0.02, 0.03], dtype=np.float32)
    reference = _reference_episode()
    reference = replace(
        reference,
        source_actions=actions,
        initial_state={"leaf": initial_leaf},
        initial_robot_state=robot_state,
    )
    before_digest = reference.source_trajectory_id

    actions[:] = 9.0
    initial_leaf[:] = 9.0
    robot_state[:] = 9.0

    assert reference.source_trajectory_id == before_digest
    assert np.all(reference.source_actions == 0.0)
    assert flatten_state_tree(reference.initial_state)[0].value.item() == 1.0
    assert np.allclose(reference.initial_robot_state, [0.0, 0.1, 0.2, 0.01, 0.02, 0.03])


def test_reference_archive_round_trip_is_path_independent(tmp_path: Path) -> None:
    archive = ManiSkillReferenceArchive((_reference_episode(),))
    first = tmp_path / "first"
    second = tmp_path / "second"

    save_reference_archive(archive, first)
    save_reference_archive(archive, second)
    loaded_first = load_reference_archive(first)
    loaded_second = load_reference_archive(second)

    assert loaded_first.content_digest == archive.content_digest
    assert loaded_second.content_digest == archive.content_digest
    assert compute_reference_archive_digest(loaded_first) == (
        compute_reference_archive_digest(loaded_second)
    )
    selected = load_archive_state(
        first, archive.episodes[0].episode_id, "initial_state"
    )
    assert selected.state_digest == archive.episodes[0].initial_state_digest
    assert selected.structure_digest == compute_state_tree_structure_digest(
        archive.episodes[0].initial_state
    )
    assert (
        find_reference_episode(
            loaded_first, archive.episodes[0].episode_id
        ).source_trajectory_id
        == archive.episodes[0].source_trajectory_id
    )


def test_reference_archive_rejects_tampered_numeric_leaf(tmp_path: Path) -> None:
    archive = ManiSkillReferenceArchive((_reference_episode(),))
    output = tmp_path / "archive"
    save_reference_archive(archive, output)
    manifest = json.loads((output / REFERENCE_MANIFEST_NAME).read_text("utf-8"))
    action_reference = manifest["episodes"][0]["source_actions"]
    action_path = output / Path(*action_reference["path"].split("/"))
    with action_path.open("wb") as stream:
        np.save(stream, np.ones((2, 3), dtype=np.float32), allow_pickle=False)

    with pytest.raises(ReferenceArchiveError, match="content_sha256"):
        load_reference_archive(output)


def test_reference_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = ManiSkillReferenceArchive((_reference_episode(),))
    output = tmp_path / "archive"
    save_reference_archive(archive, output)
    manifest_path = output / REFERENCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["episodes"][0]["source_actions"]["path"] = "../escape.npy"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ReferenceArchiveError, match="unsafe array path"):
        load_reference_archive(output)


def test_reference_archive_rejects_unsupported_version(tmp_path: Path) -> None:
    archive = ManiSkillReferenceArchive((_reference_episode(),))
    output = tmp_path / "archive"
    save_reference_archive(archive, output)
    manifest_path = output / REFERENCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["serialization_version"] = 999
    _write_manifest(manifest_path, manifest)

    with pytest.raises(UnsupportedReferenceArchiveVersionError):
        load_reference_archive(output)


def test_reference_archive_is_immutable_after_publication(tmp_path: Path) -> None:
    archive = ManiSkillReferenceArchive((_reference_episode(),))
    output = tmp_path / "archive"
    save_reference_archive(archive, output)

    with pytest.raises(ReferenceArchiveError, match="immutable"):
        save_reference_archive(archive, output)
    assert_reference_archive_unchanged(output, archive.content_digest)


def test_reference_archive_rejects_unmanifested_files(tmp_path: Path) -> None:
    archive = ManiSkillReferenceArchive((_reference_episode(),))
    output = tmp_path / "archive"
    save_reference_archive(archive, output)
    (output / "arrays" / "untracked.npy").write_bytes(b"not an npy")

    with pytest.raises(ReferenceArchiveError, match="inventory|exactly match"):
        load_reference_archive(output)


def test_m0_source_import_has_complete_strong_simulator_provenance() -> None:
    reference = _reference_episode()

    episode = build_m0_source_episode(reference)

    assert episode.task_id == MANISKILL_PICKCUBE_TASK_ID
    assert episode.source_policy_id == OFFICIAL_PICKCUBE_SOURCE_POLICY_ID
    assert episode.split_group_id.startswith("mspc-split-")
    assert len(episode.observations.frames) == 1
    assert episode.observations.frames[0].timestamp_s == 0.0
    assert episode.observations.frames[0].cameras == ()
    candidate = episode.candidates[0]
    assert candidate.action.actions.shape == reference.source_actions.shape
    assert candidate.outcome.success
    assert candidate.outcome.progress == 1.0
    assert not candidate.outcome.unsafe
    assert candidate.outcome.label_source is LabelSource.SIMULATOR
    assert candidate.outcome.label_strength is LabelStrength.STRONG
    assert candidate.outcome.simulator_replay_verified
    provenance = candidate.provenance
    assert provenance.source_episode_id == episode.episode_id
    assert provenance.source_policy_id == OFFICIAL_PICKCUBE_SOURCE_POLICY_ID
    assert provenance.source_task_id == MANISKILL_PICKCUBE_TASK_ID
    assert provenance.split_group_id == episode.split_group_id
    assert provenance.transformation_parameters["state_semantic"] == (
        STATE_TREE_SEMANTIC
    )
    assert provenance.transformation_parameters["progress_semantic"] == (
        PICKCUBE_PROGRESS_SEMANTIC
    )
    assert provenance.transformation_parameters["unsafe_semantic"] == (
        PICKCUBE_UNSAFE_SEMANTIC
    )


def test_m0_source_import_does_not_guess_action_layout() -> None:
    reference = _reference_episode()
    assert reference.action_contract["semantic_fields"] == "unspecified"

    episode = build_m0_source_episode(reference)

    assert episode.candidates[0].action.actions.shape[1] == 3
    assert "gripper" not in episode.candidates[0].provenance.transformation_parameters


def test_m0_source_import_requires_independent_baseline() -> None:
    reference = _reference_episode(independent_baseline_success=False)

    with pytest.raises(M0SourceImportError, match="independent baseline"):
        build_m0_source_episode(reference)


def test_m0_source_import_rejects_narrow_unsafe_source() -> None:
    reference = _reference_episode(cube_center_z=-0.01)

    with pytest.raises(M0SourceImportError, match="below world z=0"):
        build_m0_source_episode(reference)


def test_m0_source_dataset_round_trip(tmp_path: Path) -> None:
    archive = ManiSkillReferenceArchive((_reference_episode(),))
    output = tmp_path / "source"

    save_m0_source_dataset(archive, output)
    loaded = load_episodes(output)

    expected = build_m0_source_episodes(archive)
    assert len(loaded) == len(expected) == 1
    assert loaded[0].episode_id == expected[0].episode_id
    assert loaded[0].candidates[0].provenance.source_episode_id == (
        expected[0].episode_id
    )
