"""Safe T+1 state archives for state-indexed PickCube source trajectories.

The format is independent of the M2C reference-archive v1 format.  It reuses
the same strict state-tree and JSON/NPY primitives while preserving the older
manifest and loader unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

from .archive import (
    ReferenceArchiveError,
    _ArrayReader,
    _ArrayWriter,
    _decode_state_node,
    _encode_state_node,
    _field,
    _integer,
    _mapping,
    _publish_staging_directory,
    _reject_duplicate_json_fields,
    _reject_json_constant,
    _require_empty_destination,
    _require_exact_fields,
    _require_regular_unlinked_file,
    _require_safe_root,
    _string,
    _validate_array_inventory,
)
from .state_tree import (
    STATE_TREE_SEMANTIC,
    NormalizedStateTree,
    StateTreeError,
    compute_state_tree_digest,
    compute_state_tree_structure_digest,
    flatten_state_tree,
    normalize_state_tree,
)
from .verifier_state import (
    PICKCUBE_VERIFIER_STATE_DTYPE,
    PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION,
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
    PickCubeVerifierStateError,
    PickCubeVerifierStateSchemaV1,
    PickCubeVerifierStateV1,
)

STATE_INDEXED_ARCHIVE_FORMAT = (
    "latentguard-maniskill-pickcube-state-indexed-reference-archive"
)
STATE_INDEXED_ARCHIVE_VERSION = 1
STATE_INDEXED_EPISODE_SCHEMA_VERSION = "1.0"
STATE_INDEXED_STATE_SCHEMA_VERSION = "1.0"
STATE_INDEXED_TASK_SNAPSHOT_SCHEMA_VERSION = "1.0"
STATE_INDEXED_MANIFEST_NAME = "manifest.json"

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_FIELDS = frozenset(
    {
        "archive_content_digest",
        "episode_count",
        "episodes",
        "format",
        "serialization_version",
        "state_semantic",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "compatibility_identity",
        "episode_content_digest",
        "episode_id",
        "schema_version",
        "seed",
        "source_action_digest",
        "source_actions",
        "source_policy_identity",
        "source_trajectory_id",
        "state_count",
        "states",
    }
)
_STATE_FIELDS = frozenset(
    {
        "compatibility_identity",
        "leaf_count",
        "numeric_component_count",
        "schema_version",
        "seed",
        "source_action_index",
        "source_trajectory_id",
        "state_content_digest",
        "state_digest",
        "state_index",
        "structure_digest",
        "task_snapshot",
        "tree",
        "verifier_state",
    }
)
_TASK_FIELDS = frozenset(
    {
        "cube_center_z",
        "cube_to_goal_distance",
        "is_grasped",
        "is_obj_placed",
        "is_robot_static",
        "schema_version",
        "success",
        "tcp_to_cube_distance",
    }
)
_VECTOR_FIELDS = frozenset(
    {
        "component_names",
        "content_digest",
        "dtype",
        "joint_names",
        "schema_digest",
        "schema_version",
        "semantic",
        "shape",
        "values",
    }
)


class StateIndexedArchiveError(ValueError):
    """Raised when a T+1 archive is malformed, unsafe, or content-drifted."""


class UnsupportedStateIndexedArchiveVersionError(StateIndexedArchiveError):
    """Raised when a state-indexed manifest or model version is unsupported."""


def _canonical_text(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise StateIndexedArchiveError(f"{context}: expected canonical text")
    return value


def _digest(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise StateIndexedArchiveError(
            f"{context}: expected a lowercase sha256 content digest"
        )
    return value


def _finite_number(value: object, *, context: str, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise StateIndexedArchiveError(f"{context}: expected a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise StateIndexedArchiveError(f"{context}: expected a {qualifier} number")
    return number


def _strict_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise StateIndexedArchiveError(f"{context}: expected a boolean")
    return bool(value)


def _value_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise StateIndexedArchiveError(f"{context}: expected a JSON list")
    return value


def _freeze_actions(value: object) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise StateIndexedArchiveError("source actions must be a numpy.ndarray")
    if value.dtype.hasobject or not np.issubdtype(value.dtype, np.floating):
        raise StateIndexedArchiveError("source actions must use a floating dtype")
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise StateIndexedArchiveError(
            "source actions must have positive [horizon, action_dim] shape"
        )
    if not bool(np.all(np.isfinite(value))):
        raise StateIndexedArchiveError("source actions must contain only finite values")
    detached = np.array(value, copy=True, order="C", subok=False)
    raw = detached.tobytes(order="C")
    return np.frombuffer(raw, dtype=detached.dtype).reshape(detached.shape)


@dataclass(frozen=True, slots=True)
class PickCubeTaskSnapshotV1:
    """Canonical public task facts captured at one archived state boundary."""

    success: bool
    is_obj_placed: bool
    is_robot_static: bool
    is_grasped: bool
    cube_center_z: float
    cube_to_goal_distance: float
    tcp_to_cube_distance: float
    schema_version: str = STATE_INDEXED_TASK_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require complete finite scalar facts without inferred defaults."""
        if self.schema_version != STATE_INDEXED_TASK_SNAPSHOT_SCHEMA_VERSION:
            raise UnsupportedStateIndexedArchiveVersionError(
                "unsupported PickCube task snapshot schema version"
            )
        for field in ("success", "is_obj_placed", "is_robot_static", "is_grasped"):
            object.__setattr__(
                self,
                field,
                _strict_bool(getattr(self, field), context=f"task snapshot {field}"),
            )
        object.__setattr__(
            self,
            "cube_center_z",
            _finite_number(self.cube_center_z, context="task snapshot cube_center_z"),
        )
        for field in ("cube_to_goal_distance", "tcp_to_cube_distance"):
            object.__setattr__(
                self,
                field,
                _finite_number(
                    getattr(self, field),
                    context=f"task snapshot {field}",
                    nonnegative=True,
                ),
            )

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical scalar task facts for hashing and JSON storage."""
        return {
            "cube_center_z": self.cube_center_z,
            "cube_to_goal_distance": self.cube_to_goal_distance,
            "is_grasped": self.is_grasped,
            "is_obj_placed": self.is_obj_placed,
            "is_robot_static": self.is_robot_static,
            "schema_version": self.schema_version,
            "success": self.success,
            "tcp_to_cube_distance": self.tcp_to_cube_distance,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PickCubeIndexedStateV1:
    """One exact state boundary and its public task/vector projections."""

    state_index: int
    source_action_index: int
    tree: NormalizedStateTree | object
    task_snapshot: PickCubeTaskSnapshotV1
    verifier_state: PickCubeVerifierStateV1
    seed: int
    compatibility_identity: str
    source_trajectory_id: str
    schema_version: str = STATE_INDEXED_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize the complete tree and validate all boundary identities."""
        if self.schema_version != STATE_INDEXED_STATE_SCHEMA_VERSION:
            raise UnsupportedStateIndexedArchiveVersionError(
                "unsupported indexed-state schema version"
            )
        if type(self.state_index) is not int or self.state_index < 0:
            raise StateIndexedArchiveError("state index must be non-negative")
        if self.source_action_index != self.state_index:
            raise StateIndexedArchiveError(
                "source action index must equal its pre-action state index"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise StateIndexedArchiveError("indexed-state seed must be uint32")
        _digest(self.compatibility_identity, context="indexed-state compatibility")
        _canonical_text(
            self.source_trajectory_id, context="indexed-state source trajectory ID"
        )
        if not isinstance(self.task_snapshot, PickCubeTaskSnapshotV1):
            raise StateIndexedArchiveError(
                "indexed state requires PickCubeTaskSnapshotV1"
            )
        if not isinstance(self.verifier_state, PickCubeVerifierStateV1):
            raise StateIndexedArchiveError(
                "indexed state requires PickCubeVerifierStateV1"
            )
        try:
            tree = normalize_state_tree(self.tree)
        except StateTreeError as exc:
            raise StateIndexedArchiveError(f"indexed state tree: {exc}") from exc
        if self.numeric_component_count <= 0 or self.leaf_count <= 0:
            raise StateIndexedArchiveError(
                "indexed state tree must contain numeric components"
            )
        object.__setattr__(self, "tree", tree)

    @property
    def state_digest(self) -> str:
        """Return the exact structure, dtype, shape, path, and byte digest."""
        return compute_state_tree_digest(self.tree)

    @property
    def structure_digest(self) -> str:
        """Return the exact state-tree inventory digest without values."""
        return compute_state_tree_structure_digest(self.tree)

    @property
    def leaf_count(self) -> int:
        """Return the complete numeric leaf count."""
        return len(flatten_state_tree(self.tree))

    @property
    def numeric_component_count(self) -> int:
        """Return the complete scalar component count."""
        return sum(int(leaf.value.size) for leaf in flatten_state_tree(self.tree))

    @property
    def content_digest(self) -> str:
        """Return the content identity of this state and its public projections."""
        return compute_state_indexed_state_content_digest(self)


@dataclass(frozen=True, slots=True, eq=False)
class PickCubeStateIndexedEpisodeV1:
    """One successful source action sequence with every state s[0] through s[T]."""

    episode_id: str
    source_trajectory_id: str
    source_policy_identity: str
    seed: int
    compatibility_identity: str
    source_actions: NDArray[Any]
    states: tuple[PickCubeIndexedStateV1, ...]
    schema_version: str = STATE_INDEXED_EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze actions and enforce the complete T+1 sequence contract."""
        if self.schema_version != STATE_INDEXED_EPISODE_SCHEMA_VERSION:
            raise UnsupportedStateIndexedArchiveVersionError(
                "unsupported state-indexed episode schema version"
            )
        _canonical_text(self.episode_id, context="state-indexed episode ID")
        _canonical_text(
            self.source_trajectory_id, context="state-indexed source trajectory ID"
        )
        _canonical_text(
            self.source_policy_identity, context="state-indexed source policy identity"
        )
        _digest(self.compatibility_identity, context="episode compatibility identity")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise StateIndexedArchiveError("state-indexed episode seed must be uint32")
        actions = _freeze_actions(self.source_actions)
        states = tuple(self.states)
        if len(states) != actions.shape[0] + 1:
            raise StateIndexedArchiveError(
                "state-indexed episode must contain exactly T+1 states"
            )
        expected_indices = tuple(range(len(states)))
        if tuple(state.state_index for state in states) != expected_indices:
            raise StateIndexedArchiveError(
                "state-indexed episode states must be ordered s[0] through s[T]"
            )
        for state in states:
            if not isinstance(state, PickCubeIndexedStateV1):
                raise StateIndexedArchiveError(
                    "state-indexed episode contains an invalid state model"
                )
            if (
                state.seed != self.seed
                or state.compatibility_identity != self.compatibility_identity
                or state.source_trajectory_id != self.source_trajectory_id
            ):
                raise StateIndexedArchiveError(
                    "state metadata differs from its source trajectory"
                )
        structure_digests = {state.structure_digest for state in states}
        component_counts = {state.numeric_component_count for state in states}
        verifier_schema_digests = {
            state.verifier_state.schema_digest for state in states
        }
        if len(structure_digests) != 1 or len(component_counts) != 1:
            raise StateIndexedArchiveError(
                "every T+1 state must preserve one complete tree inventory"
            )
        if len(verifier_schema_digests) != 1:
            raise StateIndexedArchiveError(
                "every T+1 public vector must use one component schema"
            )
        if states[-1].task_snapshot.success is not True:
            raise StateIndexedArchiveError(
                "published state-indexed source trajectory must end in success"
            )
        object.__setattr__(self, "source_actions", actions)
        object.__setattr__(self, "states", states)

    @property
    def source_action_digest(self) -> str:
        """Return the exact source-action dtype, shape, and byte digest."""
        return compute_state_indexed_source_action_digest(self)

    @property
    def content_digest(self) -> str:
        """Return the path-independent episode content identity."""
        return compute_state_indexed_episode_content_digest(self)


@dataclass(frozen=True, slots=True)
class PickCubeStateIndexedArchiveV1:
    """Ordered immutable collection of complete state-indexed trajectories."""

    episodes: tuple[PickCubeStateIndexedEpisodeV1, ...]
    serialization_version: int = STATE_INDEXED_ARCHIVE_VERSION
    state_semantic: str = STATE_TREE_SEMANTIC

    def __post_init__(self) -> None:
        """Freeze ordering and reject duplicate trajectory identities or seeds."""
        if self.serialization_version != STATE_INDEXED_ARCHIVE_VERSION:
            raise UnsupportedStateIndexedArchiveVersionError(
                "unsupported state-indexed archive version"
            )
        if self.state_semantic != STATE_TREE_SEMANTIC:
            raise UnsupportedStateIndexedArchiveVersionError(
                "unsupported state-indexed state-tree semantic"
            )
        episodes = tuple(self.episodes)
        if not episodes:
            raise StateIndexedArchiveError(
                "state-indexed archive must contain at least one episode"
            )
        if any(
            not isinstance(episode, PickCubeStateIndexedEpisodeV1)
            for episode in episodes
        ):
            raise StateIndexedArchiveError(
                "state-indexed archive contains an invalid episode model"
            )
        for values, field in (
            ((episode.episode_id for episode in episodes), "episode IDs"),
            (
                (episode.source_trajectory_id for episode in episodes),
                "source trajectory IDs",
            ),
            ((episode.seed for episode in episodes), "source seeds"),
        ):
            materialized = tuple(values)
            if len(set(materialized)) != len(materialized):
                raise StateIndexedArchiveError(
                    f"state-indexed archive has duplicate {field}"
                )
        compatibility_identities = {
            episode.compatibility_identity for episode in episodes
        }
        source_policy_identities = {
            episode.source_policy_identity for episode in episodes
        }
        state_structure_digests = {
            episode.states[0].structure_digest for episode in episodes
        }
        verifier_schema_digests = {
            episode.states[0].verifier_state.schema_digest for episode in episodes
        }
        action_contracts = {
            (episode.source_actions.dtype.str, episode.source_actions.shape[1])
            for episode in episodes
        }
        if len(compatibility_identities) != 1:
            raise StateIndexedArchiveError(
                "state-indexed archive requires one compatibility identity"
            )
        if len(source_policy_identities) != 1:
            raise StateIndexedArchiveError(
                "state-indexed archive requires one source policy identity"
            )
        if len(state_structure_digests) != 1:
            raise StateIndexedArchiveError(
                "state-indexed archive requires one complete tree inventory"
            )
        if len(verifier_schema_digests) != 1:
            raise StateIndexedArchiveError(
                "state-indexed archive requires one verifier component schema"
            )
        if len(action_contracts) != 1:
            raise StateIndexedArchiveError(
                "state-indexed archive requires one action dtype and dimension"
            )
        object.__setattr__(self, "episodes", episodes)

    @property
    def content_digest(self) -> str:
        """Return the path-independent aggregate content identity."""
        return compute_state_indexed_archive_content_digest(self)


@dataclass(frozen=True, slots=True)
class LoadedPickCubeIndexedState:
    """One state selected from a fully validated state-indexed archive."""

    episode_id: str
    source_trajectory_id: str
    source_reset_seed: int
    compatibility_identity: str
    archive_content_digest: str
    state: PickCubeIndexedStateV1


def compute_state_indexed_source_action_digest(
    episode: PickCubeStateIndexedEpisodeV1,
) -> str:
    """Hash exact source action bytes together with dtype and shape."""
    actions = _freeze_actions(episode.source_actions)
    payload = {
        "content_sha256": hashlib.sha256(actions.tobytes(order="C")).hexdigest(),
        "dtype": actions.dtype.str,
        "shape": actions.shape,
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def compute_state_indexed_state_content_digest(
    state: PickCubeIndexedStateV1,
) -> str:
    """Hash one indexed state independently of runtime paths and timestamps."""
    if not isinstance(state, PickCubeIndexedStateV1):
        raise StateIndexedArchiveError("expected PickCubeIndexedStateV1")
    payload = {
        "compatibility_identity": state.compatibility_identity,
        "leaf_count": state.leaf_count,
        "numeric_component_count": state.numeric_component_count,
        "schema_version": state.schema_version,
        "seed": state.seed,
        "source_action_index": state.source_action_index,
        "source_trajectory_id": state.source_trajectory_id,
        "state_digest": state.state_digest,
        "state_index": state.state_index,
        "structure_digest": state.structure_digest,
        "task_snapshot": state.task_snapshot.as_mapping(),
        "verifier_state_content_digest": state.verifier_state.content_digest,
        "verifier_state_schema_digest": state.verifier_state.schema_digest,
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def compute_state_indexed_episode_content_digest(
    episode: PickCubeStateIndexedEpisodeV1,
) -> str:
    """Hash one complete T+1 source independently of archive paths."""
    if not isinstance(episode, PickCubeStateIndexedEpisodeV1):
        raise StateIndexedArchiveError("expected PickCubeStateIndexedEpisodeV1")
    payload = {
        "compatibility_identity": episode.compatibility_identity,
        "episode_id": episode.episode_id,
        "schema_version": episode.schema_version,
        "seed": episode.seed,
        "source_action_digest": compute_state_indexed_source_action_digest(episode),
        "source_policy_identity": episode.source_policy_identity,
        "source_trajectory_id": episode.source_trajectory_id,
        "state_content_digests": [state.content_digest for state in episode.states],
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def compute_state_indexed_archive_content_digest(
    archive: PickCubeStateIndexedArchiveV1,
) -> str:
    """Hash every ordered episode without runtime directory identity."""
    if not isinstance(archive, PickCubeStateIndexedArchiveV1):
        raise StateIndexedArchiveError("expected PickCubeStateIndexedArchiveV1")
    payload = {
        "episode_content_digests": [
            episode.content_digest for episode in archive.episodes
        ],
        "format": STATE_INDEXED_ARCHIVE_FORMAT,
        "serialization_version": archive.serialization_version,
        "state_semantic": archive.state_semantic,
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def find_state_indexed_episode(
    archive: PickCubeStateIndexedArchiveV1, episode_id: str
) -> PickCubeStateIndexedEpisodeV1:
    """Return exactly one validated episode by its stable identifier."""
    _canonical_text(episode_id, context="state-indexed episode selector")
    for episode in archive.episodes:
        if episode.episode_id == episode_id:
            return episode
    raise StateIndexedArchiveError(f"unknown state-indexed episode {episode_id!r}")


def load_indexed_state(
    output_dir: Path, episode_id: str, state_index: int
) -> LoadedPickCubeIndexedState:
    """Load, fully validate, and select one immutable indexed state."""
    if type(state_index) is not int or state_index < 0:
        raise StateIndexedArchiveError("state selector must be non-negative")
    archive = load_state_indexed_archive(output_dir)
    episode = find_state_indexed_episode(archive, episode_id)
    if state_index >= len(episode.states):
        raise StateIndexedArchiveError(
            f"state index {state_index} is outside episode {episode_id!r}"
        )
    return LoadedPickCubeIndexedState(
        episode_id=episode.episode_id,
        source_trajectory_id=episode.source_trajectory_id,
        source_reset_seed=episode.seed,
        compatibility_identity=episode.compatibility_identity,
        archive_content_digest=archive.content_digest,
        state=episode.states[state_index],
    )


def _encode_vector(
    vector: PickCubeVerifierStateV1, writer: _ArrayWriter
) -> dict[str, object]:
    return {
        "component_names": list(vector.component_names),
        "content_digest": vector.content_digest,
        "dtype": vector.values.dtype.str,
        "joint_names": list(vector.schema.joint_names),
        "schema_digest": vector.schema_digest,
        "schema_version": vector.schema.schema_version,
        "semantic": vector.semantic,
        "shape": list(vector.values.shape),
        "values": writer.write(vector.values),
    }


def _encode_state(
    state: PickCubeIndexedStateV1, writer: _ArrayWriter
) -> dict[str, object]:
    tree = cast(NormalizedStateTree, state.tree)
    return {
        "compatibility_identity": state.compatibility_identity,
        "leaf_count": state.leaf_count,
        "numeric_component_count": state.numeric_component_count,
        "schema_version": state.schema_version,
        "seed": state.seed,
        "source_action_index": state.source_action_index,
        "source_trajectory_id": state.source_trajectory_id,
        "state_content_digest": state.content_digest,
        "state_digest": state.state_digest,
        "state_index": state.state_index,
        "structure_digest": state.structure_digest,
        "task_snapshot": dict(state.task_snapshot.as_mapping()),
        "tree": _encode_state_node(tree.root, "$", writer),
        "verifier_state": _encode_vector(state.verifier_state, writer),
    }


def _encode_episode(
    episode: PickCubeStateIndexedEpisodeV1, writer: _ArrayWriter
) -> dict[str, object]:
    return {
        "compatibility_identity": episode.compatibility_identity,
        "episode_content_digest": episode.content_digest,
        "episode_id": episode.episode_id,
        "schema_version": episode.schema_version,
        "seed": episode.seed,
        "source_action_digest": episode.source_action_digest,
        "source_actions": writer.write(episode.source_actions),
        "source_policy_identity": episode.source_policy_identity,
        "source_trajectory_id": episode.source_trajectory_id,
        "state_count": len(episode.states),
        "states": [_encode_state(state, writer) for state in episode.states],
    }


def save_state_indexed_archive(
    archive: PickCubeStateIndexedArchiveV1, output_dir: Path
) -> Path:
    """Transactionally publish a strict JSON/NPY T+1 archive."""
    if not isinstance(archive, PickCubeStateIndexedArchiveV1):
        raise StateIndexedArchiveError("expected PickCubeStateIndexedArchiveV1")
    destination = Path(output_dir).absolute()
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        _require_empty_destination(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'state-indexed'}.staging-",
                dir=destination.parent,
            )
        )
        writer = _ArrayWriter(staging)
        manifest: dict[str, object] = {
            "archive_content_digest": archive.content_digest,
            "episode_count": len(archive.episodes),
            "episodes": [
                _encode_episode(episode, writer) for episode in archive.episodes
            ],
            "format": STATE_INDEXED_ARCHIVE_FORMAT,
            "serialization_version": archive.serialization_version,
            "state_semantic": archive.state_semantic,
        }
        payload = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        with (staging / STATE_INDEXED_MANIFEST_NAME).open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(payload + "\n")
        _publish_staging_directory(staging, destination)
    except StateIndexedArchiveError as exc:
        primary_error = exc
        raise
    except ReferenceArchiveError as exc:
        wrapped = StateIndexedArchiveError(f"state-indexed archive: {exc}")
        primary_error = wrapped
        raise wrapped from exc
    except (OSError, TypeError, ValueError) as exc:
        wrapped = StateIndexedArchiveError(
            f"state-indexed archive could not save transactionally: {exc}"
        )
        primary_error = wrapped
        raise wrapped from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if staging is not None and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        f"state-indexed staging cleanup also failed: {cleanup_error}"
                    )
                else:
                    raise StateIndexedArchiveError(
                        "could not clean state-indexed staging directory"
                    ) from cleanup_error
    return destination / STATE_INDEXED_MANIFEST_NAME


def _decode_task(value: object, *, context: str) -> PickCubeTaskSnapshotV1:
    item = _mapping(value, context)
    _require_exact_fields(item, _TASK_FIELDS, context)
    return PickCubeTaskSnapshotV1(
        success=_strict_bool(
            _field(item, "success", context), context=f"{context}.success"
        ),
        is_obj_placed=_strict_bool(
            _field(item, "is_obj_placed", context),
            context=f"{context}.is_obj_placed",
        ),
        is_robot_static=_strict_bool(
            _field(item, "is_robot_static", context),
            context=f"{context}.is_robot_static",
        ),
        is_grasped=_strict_bool(
            _field(item, "is_grasped", context), context=f"{context}.is_grasped"
        ),
        cube_center_z=_finite_number(
            _field(item, "cube_center_z", context),
            context=f"{context}.cube_center_z",
        ),
        cube_to_goal_distance=_finite_number(
            _field(item, "cube_to_goal_distance", context),
            context=f"{context}.cube_to_goal_distance",
            nonnegative=True,
        ),
        tcp_to_cube_distance=_finite_number(
            _field(item, "tcp_to_cube_distance", context),
            context=f"{context}.tcp_to_cube_distance",
            nonnegative=True,
        ),
        schema_version=_string(item, "schema_version", context),
    )


def _string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    return tuple(
        _canonical_text(item, context=f"{context}[{index}]")
        for index, item in enumerate(_value_list(value, context=context))
    )


def _decode_vector(
    value: object, reader: _ArrayReader, *, context: str
) -> PickCubeVerifierStateV1:
    item = _mapping(value, context)
    _require_exact_fields(item, _VECTOR_FIELDS, context)
    joint_names = _string_tuple(
        _field(item, "joint_names", context), context=f"{context}.joint_names"
    )
    schema = PickCubeVerifierStateSchemaV1(joint_names=joint_names)
    if _string(item, "semantic", context) != PICKCUBE_VERIFIER_STATE_SEMANTIC:
        raise StateIndexedArchiveError(f"{context}.semantic: mismatch")
    if (
        _string(item, "schema_version", context)
        != PICKCUBE_VERIFIER_STATE_SCHEMA_VERSION
    ):
        raise StateIndexedArchiveError(f"{context}.schema_version: mismatch")
    if _string(item, "dtype", context) != PICKCUBE_VERIFIER_STATE_DTYPE:
        raise StateIndexedArchiveError(f"{context}.dtype: mismatch")
    if (
        _string_tuple(
            _field(item, "component_names", context),
            context=f"{context}.component_names",
        )
        != schema.component_names
    ):
        raise StateIndexedArchiveError(f"{context}.component_names: mismatch")
    raw_shape = _value_list(_field(item, "shape", context), context=f"{context}.shape")
    if any(type(dimension) is not int or dimension < 0 for dimension in raw_shape):
        raise StateIndexedArchiveError(f"{context}.shape: invalid dimension")
    if tuple(raw_shape) != schema.shape:
        raise StateIndexedArchiveError(f"{context}.shape: mismatch")
    expected_schema_digest = _digest(
        _string(item, "schema_digest", context), context=f"{context}.schema_digest"
    )
    if expected_schema_digest != schema.schema_digest:
        raise StateIndexedArchiveError(f"{context}.schema_digest: content mismatch")
    vector = PickCubeVerifierStateV1(
        schema=schema,
        values=reader.read(_field(item, "values", context), f"{context}.values"),
    )
    expected_content = _digest(
        _string(item, "content_digest", context), context=f"{context}.content_digest"
    )
    if expected_content != vector.content_digest:
        raise StateIndexedArchiveError(f"{context}.content_digest: content mismatch")
    return vector


def _decode_state(
    value: object, reader: _ArrayReader, *, episode_index: int, state_index: int
) -> PickCubeIndexedStateV1:
    context = f"StateIndexedEpisode[{episode_index}].states[{state_index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _STATE_FIELDS, context)
    state = PickCubeIndexedStateV1(
        state_index=_integer(item, "state_index", context),
        source_action_index=_integer(item, "source_action_index", context),
        tree=NormalizedStateTree(
            _decode_state_node(_field(item, "tree", context), "$", reader, depth=0)
        ),
        task_snapshot=_decode_task(
            _field(item, "task_snapshot", context), context=f"{context}.task_snapshot"
        ),
        verifier_state=_decode_vector(
            _field(item, "verifier_state", context),
            reader,
            context=f"{context}.verifier_state",
        ),
        seed=_integer(item, "seed", context),
        compatibility_identity=_string(item, "compatibility_identity", context),
        source_trajectory_id=_string(item, "source_trajectory_id", context),
        schema_version=_string(item, "schema_version", context),
    )
    checks: tuple[tuple[str, object, object], ...] = (
        (
            "state_digest",
            state.state_digest,
            _string(item, "state_digest", context),
        ),
        (
            "structure_digest",
            state.structure_digest,
            _string(item, "structure_digest", context),
        ),
        ("leaf_count", state.leaf_count, _integer(item, "leaf_count", context)),
        (
            "numeric_component_count",
            state.numeric_component_count,
            _integer(item, "numeric_component_count", context),
        ),
        (
            "state_content_digest",
            state.content_digest,
            _string(item, "state_content_digest", context),
        ),
    )
    for field, observed, expected in checks:
        if observed != expected:
            raise StateIndexedArchiveError(f"{context}.{field}: content mismatch")
    return state


def _decode_episode(
    value: object, reader: _ArrayReader, *, episode_index: int
) -> PickCubeStateIndexedEpisodeV1:
    context = f"StateIndexedEpisode[{episode_index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _EPISODE_FIELDS, context)
    raw_states = _value_list(
        _field(item, "states", context), context=f"{context}.states"
    )
    episode = PickCubeStateIndexedEpisodeV1(
        episode_id=_string(item, "episode_id", context),
        source_trajectory_id=_string(item, "source_trajectory_id", context),
        source_policy_identity=_string(item, "source_policy_identity", context),
        seed=_integer(item, "seed", context),
        compatibility_identity=_string(item, "compatibility_identity", context),
        source_actions=reader.read(
            _field(item, "source_actions", context), f"{context}.source_actions"
        ),
        states=tuple(
            _decode_state(
                raw_state,
                reader,
                episode_index=episode_index,
                state_index=state_index,
            )
            for state_index, raw_state in enumerate(raw_states)
        ),
        schema_version=_string(item, "schema_version", context),
    )
    if _integer(item, "state_count", context) != len(episode.states):
        raise StateIndexedArchiveError(f"{context}.state_count: content mismatch")
    if _string(item, "source_action_digest", context) != episode.source_action_digest:
        raise StateIndexedArchiveError(
            f"{context}.source_action_digest: content mismatch"
        )
    if _string(item, "episode_content_digest", context) != episode.content_digest:
        raise StateIndexedArchiveError(
            f"{context}.episode_content_digest: content mismatch"
        )
    return episode


def load_state_indexed_archive(output_dir: Path) -> PickCubeStateIndexedArchiveV1:
    """Reload and independently validate a safe T+1 JSON/NPY archive."""
    try:
        requested = Path(output_dir).absolute()
        _require_safe_root(requested)
        root = requested.resolve()
        manifest_path = root / STATE_INDEXED_MANIFEST_NAME
        _require_regular_unlinked_file(manifest_path, "StateIndexedArchive.manifest")
        raw = cast(
            object,
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
        manifest = _mapping(raw, "StateIndexedArchive.manifest")
        _require_exact_fields(
            manifest, _MANIFEST_FIELDS, "StateIndexedArchive.manifest"
        )
        if (
            _string(manifest, "format", "StateIndexedArchive.manifest")
            != STATE_INDEXED_ARCHIVE_FORMAT
        ):
            raise StateIndexedArchiveError("unsupported state-indexed archive format")
        version = _integer(
            manifest, "serialization_version", "StateIndexedArchive.manifest"
        )
        if version != STATE_INDEXED_ARCHIVE_VERSION:
            raise UnsupportedStateIndexedArchiveVersionError(
                f"unsupported state-indexed archive version {version!r}"
            )
        if (
            _string(manifest, "state_semantic", "StateIndexedArchive.manifest")
            != STATE_TREE_SEMANTIC
        ):
            raise UnsupportedStateIndexedArchiveVersionError(
                "unsupported state-indexed tree semantic"
            )
        expected_digest = _digest(
            _string(manifest, "archive_content_digest", "StateIndexedArchive.manifest"),
            context="StateIndexedArchive.manifest.archive_content_digest",
        )
        encoded = _value_list(
            _field(manifest, "episodes", "StateIndexedArchive.manifest"),
            context="StateIndexedArchive.manifest.episodes",
        )
        if _integer(manifest, "episode_count", "StateIndexedArchive.manifest") != len(
            encoded
        ):
            raise StateIndexedArchiveError(
                "state-indexed manifest episode count does not match episodes"
            )
        reader = _ArrayReader(root)
        episodes = tuple(
            _decode_episode(value, reader, episode_index=index)
            for index, value in enumerate(encoded)
        )
        _validate_array_inventory(reader)
        archive = PickCubeStateIndexedArchiveV1(episodes=episodes)
        if archive.content_digest != expected_digest:
            raise StateIndexedArchiveError(
                "state-indexed archive content digest mismatch"
            )
        return archive
    except StateIndexedArchiveError:
        raise
    except (ReferenceArchiveError, PickCubeVerifierStateError, StateTreeError) as exc:
        raise StateIndexedArchiveError(f"state-indexed archive: {exc}") from exc
    except Exception as exc:
        raise StateIndexedArchiveError(
            f"state-indexed archive could not be read safely: {exc}"
        ) from exc


def assert_state_indexed_archive_unchanged(
    output_dir: Path, expected_content_digest: str
) -> None:
    """Reload an archive and reject any semantic or byte-level content drift."""
    expected = _digest(
        expected_content_digest, context="expected state-indexed archive digest"
    )
    if load_state_indexed_archive(output_dir).content_digest != expected:
        raise StateIndexedArchiveError("state-indexed archive changed after binding")


__all__ = [
    "STATE_INDEXED_ARCHIVE_FORMAT",
    "STATE_INDEXED_ARCHIVE_VERSION",
    "STATE_INDEXED_EPISODE_SCHEMA_VERSION",
    "STATE_INDEXED_MANIFEST_NAME",
    "STATE_INDEXED_STATE_SCHEMA_VERSION",
    "STATE_INDEXED_TASK_SNAPSHOT_SCHEMA_VERSION",
    "LoadedPickCubeIndexedState",
    "PickCubeIndexedStateV1",
    "PickCubeStateIndexedArchiveV1",
    "PickCubeStateIndexedEpisodeV1",
    "PickCubeTaskSnapshotV1",
    "StateIndexedArchiveError",
    "UnsupportedStateIndexedArchiveVersionError",
    "assert_state_indexed_archive_unchanged",
    "compute_state_indexed_archive_content_digest",
    "compute_state_indexed_episode_content_digest",
    "compute_state_indexed_source_action_digest",
    "compute_state_indexed_state_content_digest",
    "find_state_indexed_episode",
    "load_indexed_state",
    "load_state_indexed_archive",
    "save_state_indexed_archive",
]
