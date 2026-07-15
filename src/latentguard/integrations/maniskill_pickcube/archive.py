"""Safe, content-bound runtime archives for ManiSkill PickCube references."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes, canonical_json_value

from .state_tree import (
    STATE_TREE_SEMANTIC,
    MappingStateNode,
    NormalizedStateTree,
    NumericStateNode,
    SequenceStateNode,
    StateNode,
    StateTreeError,
    compute_state_tree_digest,
    compute_state_tree_structure_digest,
    normalize_state_tree,
)

REFERENCE_ARCHIVE_FORMAT = "latentguard-maniskill-pickcube-reference-archive"
REFERENCE_ARCHIVE_VERSION = 1
REFERENCE_EPISODE_SCHEMA_VERSION = 1
REFERENCE_MANIFEST_NAME = "manifest.json"

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_TERMINAL_EVIDENCE_FIELDS = frozenset(
    {
        "success",
        "is_obj_placed",
        "is_robot_static",
        "is_grasped",
        "cube_center_z",
        "cube_to_goal_distance",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "serialization_version",
        "state_semantic",
        "archive_content_digest",
        "episode_count",
        "episodes",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "source_trajectory_id",
        "compatibility_identity",
        "source_solver_identity",
        "environment_configuration",
        "seed",
        "source_actions",
        "source_action_digest",
        "initial_state",
        "initial_state_digest",
        "terminal_state",
        "terminal_state_digest",
        "terminal_task_evidence",
        "initial_robot_state",
        "robot_state_joint_names",
        "robot_state_semantic",
        "action_contract",
        "action_coordinate_frame",
        "control_period_s",
        "source_generation_success",
        "independent_baseline_success",
    }
)


class ReferenceArchiveError(ValueError):
    """Raised when a reference archive is malformed, unsafe, or conflicting."""


class UnsupportedReferenceArchiveVersionError(ReferenceArchiveError):
    """Raised when an archive or episode serialization version is unsupported."""


@dataclass(frozen=True, slots=True, eq=False)
class ManiSkillReferenceEpisode:
    """One recorded trajectory and its independently replayed simulator context."""

    compatibility_identity: str
    source_solver_identity: Mapping[str, object]
    environment_configuration: Mapping[str, object]
    seed: int
    source_actions: NDArray[Any]
    initial_state: NormalizedStateTree | object
    terminal_state: NormalizedStateTree | object
    terminal_task_evidence: Mapping[str, object]
    initial_robot_state: NDArray[Any]
    robot_state_joint_names: tuple[str, ...]
    robot_state_semantic: str
    action_contract: Mapping[str, object]
    action_coordinate_frame: str
    control_period_s: float
    source_generation_success: bool
    independent_baseline_success: bool
    schema_version: int = REFERENCE_EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Detach all mutable inputs and validate the complete episode contract."""
        if isinstance(self.source_actions, np.ndarray):
            object.__setattr__(
                self, "source_actions", _freeze_numeric_array(self.source_actions)
            )
        if isinstance(self.initial_robot_state, np.ndarray):
            object.__setattr__(
                self,
                "initial_robot_state",
                _freeze_numeric_array(self.initial_robot_state),
            )
        try:
            object.__setattr__(
                self, "initial_state", normalize_state_tree(self.initial_state)
            )
            object.__setattr__(
                self, "terminal_state", normalize_state_tree(self.terminal_state)
            )
        except StateTreeError as exc:
            raise ReferenceArchiveError(
                f"ManiSkillReferenceEpisode.state: {exc}"
            ) from exc
        for field in (
            "source_solver_identity",
            "environment_configuration",
            "terminal_task_evidence",
            "action_contract",
        ):
            value = getattr(self, field)
            if isinstance(value, Mapping):
                object.__setattr__(
                    self,
                    field,
                    _freeze_canonical_mapping(value, f"ReferenceEpisode.{field}"),
                )
        object.__setattr__(
            self, "robot_state_joint_names", tuple(self.robot_state_joint_names)
        )
        validate_reference_episode(self)

    @property
    def source_action_digest(self) -> str:
        """Return the path-independent digest of source actions and their contract."""
        return compute_source_action_digest(self)

    @property
    def initial_state_digest(self) -> str:
        """Return the complete initial-state digest."""
        return compute_state_tree_digest(self.initial_state)

    @property
    def terminal_state_digest(self) -> str:
        """Return the complete terminal-state digest."""
        return compute_state_tree_digest(self.terminal_state)

    @property
    def source_trajectory_id(self) -> str:
        """Return a stable identity bound to all trajectory semantics."""
        digest = compute_reference_episode_content_digest(self)
        return f"mspc-trajectory-{digest.removeprefix('sha256:')}"

    @property
    def episode_id(self) -> str:
        """Return the stable M0/runtime episode foreign key."""
        digest = hashlib.sha256(self.source_trajectory_id.encode("utf-8")).hexdigest()
        return f"mspc-episode-{digest}"


@dataclass(frozen=True, slots=True)
class ManiSkillReferenceArchive:
    """An ordered immutable collection of content-bound reference episodes."""

    episodes: tuple[ManiSkillReferenceEpisode, ...]
    serialization_version: int = REFERENCE_ARCHIVE_VERSION
    state_semantic: str = STATE_TREE_SEMANTIC

    def __post_init__(self) -> None:
        """Freeze episode ordering and validate the archive."""
        object.__setattr__(self, "episodes", tuple(self.episodes))
        validate_reference_archive(self)

    @property
    def content_digest(self) -> str:
        """Return the path-independent digest of every ordered episode."""
        return compute_reference_archive_digest(self)


@dataclass(frozen=True, slots=True)
class LoadedArchiveState:
    """One immutable state selected from a validated runtime archive."""

    episode_id: str
    state_key: Literal["initial_state", "terminal_state"]
    tree: NormalizedStateTree
    state_digest: str
    structure_digest: str


class _ArrayWriter:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.arrays_dir = self.root / "arrays"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write(self, array: NDArray[Any]) -> dict[str, object]:
        frozen = _freeze_numeric_array(array)
        relative = PurePosixPath("arrays") / f"{self.count:06d}.npy"
        destination = _safe_array_path(self.root, relative.as_posix(), writing=True)
        try:
            with destination.open("xb") as stream:
                np.save(stream, frozen, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ReferenceArchiveError(
                f"ReferenceArchive.array: could not write {relative.as_posix()!r}"
            ) from exc
        self.count += 1
        return {
            "path": relative.as_posix(),
            "dtype": frozen.dtype.str,
            "shape": list(frozen.shape),
            "content_sha256": hashlib.sha256(frozen.tobytes(order="C")).hexdigest(),
        }


class _ArrayReader:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.references: list[str] = []

    def read(self, value: object, context: str) -> NDArray[Any]:
        reference = _mapping(value, context)
        _require_exact_fields(
            reference, {"path", "dtype", "shape", "content_sha256"}, context
        )
        relative = _string(reference, "path", context)
        dtype_text = _string(reference, "dtype", context)
        shape_values = _list(reference, "shape", context)
        shape: list[int] = []
        for dimension in shape_values:
            if type(dimension) is not int or dimension < 0:
                raise ReferenceArchiveError(f"{context}.shape: invalid dimension")
            shape.append(dimension)
        try:
            expected_dtype = np.dtype(dtype_text)
        except (TypeError, ValueError) as exc:
            raise ReferenceArchiveError(f"{context}.dtype: invalid dtype") from exc
        expected_content = _string(reference, "content_sha256", context)
        if re.fullmatch(r"[0-9a-f]{64}", expected_content) is None:
            raise ReferenceArchiveError(
                f"{context}.content_sha256: expected 64 lowercase hex characters"
            )
        path = _safe_array_path(self.root, relative, writing=False)
        self.references.append(relative)
        try:
            with path.open("rb") as stream:
                loaded = np.load(stream, allow_pickle=False)
        except Exception as exc:
            raise ReferenceArchiveError(
                f"{context}: could not load safe NPY array"
            ) from exc
        if not isinstance(loaded, np.ndarray):
            if hasattr(loaded, "close"):
                loaded.close()
            raise ReferenceArchiveError(f"{context}: expected one NPY array")
        if loaded.dtype != expected_dtype:
            raise ReferenceArchiveError(
                f"{context}.dtype: expected {expected_dtype}, got {loaded.dtype}"
            )
        if loaded.shape != tuple(shape):
            raise ReferenceArchiveError(
                f"{context}.shape: expected {tuple(shape)}, got {loaded.shape}"
            )
        try:
            frozen = _freeze_numeric_array(loaded)
        except ReferenceArchiveError as exc:
            raise ReferenceArchiveError(f"{context}: {exc}") from exc
        observed_content = hashlib.sha256(frozen.tobytes(order="C")).hexdigest()
        if observed_content != expected_content:
            raise ReferenceArchiveError(f"{context}.content_sha256: content mismatch")
        return frozen


def validate_reference_episode(episode: ManiSkillReferenceEpisode) -> None:
    """Validate a reference episode without importing a simulator runtime."""
    context = "ManiSkillReferenceEpisode"
    if not isinstance(episode, ManiSkillReferenceEpisode):
        raise ReferenceArchiveError(f"{context}: expected reference episode")
    if episode.schema_version != REFERENCE_EPISODE_SCHEMA_VERSION:
        raise UnsupportedReferenceArchiveVersionError(
            f"{context}.schema_version: unsupported version "
            f"{episode.schema_version!r}; supported: {REFERENCE_EPISODE_SCHEMA_VERSION}"
        )
    _require_digest(episode.compatibility_identity, f"{context}.compatibility_identity")
    _validate_solver_identity(episode.source_solver_identity, context)
    if not episode.environment_configuration:
        raise ReferenceArchiveError(
            f"{context}.environment_configuration: must not be empty"
        )
    _freeze_canonical_mapping(
        episode.environment_configuration, f"{context}.environment_configuration"
    )
    if type(episode.seed) is not int or not 0 <= episode.seed < 2**64:
        raise ReferenceArchiveError(
            f"{context}.seed: expected an integer in [0, 2**64)"
        )
    actions = _require_numeric_array(
        episode.source_actions, f"{context}.source_actions"
    )
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ReferenceArchiveError(
            f"{context}.source_actions: expected positive [horizon, action_dim], "
            f"got {actions.shape}"
        )
    if not isinstance(episode.initial_state, NormalizedStateTree) or not isinstance(
        episode.terminal_state, NormalizedStateTree
    ):
        raise ReferenceArchiveError(f"{context}.state: expected normalized state trees")
    _validate_terminal_task_evidence(episode.terminal_task_evidence, context)
    robot_state = _require_numeric_array(
        episode.initial_robot_state, f"{context}.initial_robot_state"
    )
    if robot_state.ndim != 1 or robot_state.size == 0:
        raise ReferenceArchiveError(
            f"{context}.initial_robot_state: expected a non-empty rank-1 array"
        )
    names = episode.robot_state_joint_names
    if robot_state.size != 2 * len(names):
        raise ReferenceArchiveError(
            f"{context}.initial_robot_state: expected named qpos followed by named "
            f"qvel with size {2 * len(names)}, got {robot_state.size}"
        )
    if len(set(names)) != len(names):
        raise ReferenceArchiveError(
            f"{context}.robot_state_joint_names: duplicate joint names are unsupported"
        )
    for index, name in enumerate(names):
        _require_canonical_text(name, f"{context}.robot_state_joint_names[{index}]")
    _require_canonical_text(
        episode.robot_state_semantic, f"{context}.robot_state_semantic"
    )
    if not episode.action_contract:
        raise ReferenceArchiveError(f"{context}.action_contract: must not be empty")
    _freeze_canonical_mapping(episode.action_contract, f"{context}.action_contract")
    _require_canonical_text(
        episode.action_coordinate_frame, f"{context}.action_coordinate_frame"
    )
    if (
        type(episode.control_period_s) not in (int, float)
        or not math.isfinite(float(episode.control_period_s))
        or episode.control_period_s <= 0
    ):
        raise ReferenceArchiveError(
            f"{context}.control_period_s: expected a finite positive number"
        )
    for field in ("source_generation_success", "independent_baseline_success"):
        if type(getattr(episode, field)) is not bool:
            raise ReferenceArchiveError(f"{context}.{field}: expected a boolean")
    if episode.independent_baseline_success and not episode.source_generation_success:
        raise ReferenceArchiveError(
            f"{context}.independent_baseline_success: requires source generation "
            "success"
        )
    if episode.independent_baseline_success:
        evidence = episode.terminal_task_evidence
        if (
            evidence["success"] is not True
            or evidence["is_obj_placed"] is not True
            or evidence["is_robot_static"] is not True
        ):
            raise ReferenceArchiveError(
                f"{context}.terminal_task_evidence: baseline success requires official "
                "success, object placement, and robot-static evidence"
            )


def validate_reference_archive(archive: ManiSkillReferenceArchive) -> None:
    """Validate archive version, state semantic, identities, and episode order."""
    context = "ManiSkillReferenceArchive"
    if not isinstance(archive, ManiSkillReferenceArchive):
        raise ReferenceArchiveError(f"{context}: expected reference archive")
    if archive.serialization_version != REFERENCE_ARCHIVE_VERSION:
        raise UnsupportedReferenceArchiveVersionError(
            f"{context}.serialization_version: unsupported version "
            f"{archive.serialization_version!r}; supported: {REFERENCE_ARCHIVE_VERSION}"
        )
    if archive.state_semantic != STATE_TREE_SEMANTIC:
        raise UnsupportedReferenceArchiveVersionError(
            f"{context}.state_semantic: unsupported value "
            f"{archive.state_semantic!r}; supported: {STATE_TREE_SEMANTIC!r}"
        )
    if not archive.episodes:
        raise ReferenceArchiveError(f"{context}.episodes: must not be empty")
    identifiers: set[str] = set()
    for episode in archive.episodes:
        validate_reference_episode(episode)
        if episode.episode_id in identifiers:
            raise ReferenceArchiveError(
                f"{context}.episodes: duplicate episode ID {episode.episode_id!r}"
            )
        identifiers.add(episode.episode_id)


def compute_source_action_digest(episode: ManiSkillReferenceEpisode) -> str:
    """Hash exact source actions together with their semantic control contract."""
    actions = _require_numeric_array(
        episode.source_actions, "ManiSkillReferenceEpisode.source_actions"
    )
    payload = {
        "content_sha256": hashlib.sha256(actions.tobytes(order="C")).hexdigest(),
        "control_period_s": episode.control_period_s,
        "coordinate_frame": episode.action_coordinate_frame,
        "dtype": actions.dtype.str,
        "shape": list(actions.shape),
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def compute_reference_episode_content_digest(
    episode: ManiSkillReferenceEpisode,
) -> str:
    """Hash one trajectory independently of archive and runtime paths."""
    validate_reference_episode(episode)
    robot_state = episode.initial_robot_state
    payload = {
        "schema_version": episode.schema_version,
        "compatibility_identity": episode.compatibility_identity,
        "source_solver_identity": _thaw_json(episode.source_solver_identity),
        "environment_configuration": _thaw_json(episode.environment_configuration),
        "seed": episode.seed,
        "source_action_digest": compute_source_action_digest(episode),
        "initial_state_digest": compute_state_tree_digest(episode.initial_state),
        "initial_state_structure_digest": compute_state_tree_structure_digest(
            episode.initial_state
        ),
        "terminal_state_digest": compute_state_tree_digest(episode.terminal_state),
        "terminal_state_structure_digest": compute_state_tree_structure_digest(
            episode.terminal_state
        ),
        "terminal_task_evidence": _thaw_json(episode.terminal_task_evidence),
        "initial_robot_state": {
            "content_sha256": hashlib.sha256(
                robot_state.tobytes(order="C")
            ).hexdigest(),
            "dtype": robot_state.dtype.str,
            "shape": list(robot_state.shape),
        },
        "robot_state_joint_names": list(episode.robot_state_joint_names),
        "robot_state_semantic": episode.robot_state_semantic,
        "action_contract": _thaw_json(episode.action_contract),
        "source_generation_success": episode.source_generation_success,
        "independent_baseline_success": episode.independent_baseline_success,
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def compute_reference_archive_digest(archive: ManiSkillReferenceArchive) -> str:
    """Hash all ordered reference episodes without runtime directory paths."""
    validate_reference_archive(archive)
    payload = {
        "format": REFERENCE_ARCHIVE_FORMAT,
        "serialization_version": archive.serialization_version,
        "state_semantic": archive.state_semantic,
        "episode_content_digests": [
            compute_reference_episode_content_digest(episode)
            for episode in archive.episodes
        ],
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def find_reference_episode(
    archive: ManiSkillReferenceArchive, episode_id: str
) -> ManiSkillReferenceEpisode:
    """Return exactly one validated episode by its stable content-bound ID."""
    validate_reference_archive(archive)
    _require_canonical_text(episode_id, "ReferenceArchive.episode_id")
    for episode in archive.episodes:
        if episode.episode_id == episode_id:
            return episode
    raise ReferenceArchiveError(
        f"ReferenceArchive.episode_id: unknown episode {episode_id!r}"
    )


def load_archive_state(
    output_dir: Path,
    episode_id: str,
    state_key: Literal["initial_state", "terminal_state"],
) -> LoadedArchiveState:
    """Load a validated archive and select one immutable state tree."""
    if state_key not in ("initial_state", "terminal_state"):
        raise ReferenceArchiveError(
            f"ReferenceArchive.state_key: unsupported selector {state_key!r}"
        )
    archive = load_reference_archive(output_dir)
    episode = find_reference_episode(archive, episode_id)
    tree = cast(NormalizedStateTree, getattr(episode, state_key))
    return LoadedArchiveState(
        episode_id=episode.episode_id,
        state_key=state_key,
        tree=tree,
        state_digest=compute_state_tree_digest(tree),
        structure_digest=compute_state_tree_structure_digest(tree),
    )


def save_reference_archive(
    archive: ManiSkillReferenceArchive, output_dir: Path
) -> Path:
    """Transactionally publish a new immutable JSON/NPY runtime archive."""
    validate_reference_archive(archive)
    destination = Path(output_dir).absolute()
    _require_empty_destination(destination)
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'archive'}.staging-",
                dir=destination.parent,
            )
        )
        writer = _ArrayWriter(staging)
        encoded_episodes = [
            _encode_episode(episode, writer) for episode in archive.episodes
        ]
        manifest: dict[str, object] = {
            "format": REFERENCE_ARCHIVE_FORMAT,
            "serialization_version": REFERENCE_ARCHIVE_VERSION,
            "state_semantic": STATE_TREE_SEMANTIC,
            "archive_content_digest": compute_reference_archive_digest(archive),
            "episode_count": len(archive.episodes),
            "episodes": encoded_episodes,
        }
        payload = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        with (staging / REFERENCE_MANIFEST_NAME).open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(payload + "\n")
        _publish_staging_directory(staging, destination)
    except ReferenceArchiveError as exc:
        primary_error = exc
        raise
    except (OSError, TypeError, ValueError) as exc:
        wrapped = ReferenceArchiveError(
            f"ReferenceArchive: could not save transactionally: {exc}"
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
                        f"Staging cleanup also failed: {cleanup_error}"
                    )
                else:
                    raise ReferenceArchiveError(
                        "ReferenceArchive: could not clean staging directory"
                    ) from cleanup_error
    return destination / REFERENCE_MANIFEST_NAME


def load_reference_archive(output_dir: Path) -> ManiSkillReferenceArchive:
    """Load and fully validate a safe runtime archive without pickle."""
    requested = Path(output_dir).absolute()
    _require_safe_root(requested)
    root = requested.resolve()
    manifest_path = root / REFERENCE_MANIFEST_NAME
    _require_regular_unlinked_file(manifest_path, "ReferenceArchive.manifest")
    try:
        raw = cast(
            object,
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
    except ReferenceArchiveError:
        raise
    except Exception as exc:
        raise ReferenceArchiveError(
            f"ReferenceArchive.manifest: could not read safely: {exc}"
        ) from exc
    manifest = _mapping(raw, "ReferenceArchive.manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "ReferenceArchive.manifest")
    if (
        _string(manifest, "format", "ReferenceArchive.manifest")
        != REFERENCE_ARCHIVE_FORMAT
    ):
        raise ReferenceArchiveError(
            "ReferenceArchive.manifest.format: unsupported format"
        )
    version = _integer(manifest, "serialization_version", "ReferenceArchive.manifest")
    if version != REFERENCE_ARCHIVE_VERSION:
        raise UnsupportedReferenceArchiveVersionError(
            "ReferenceArchive.manifest.serialization_version: unsupported version "
            f"{version!r}; supported: {REFERENCE_ARCHIVE_VERSION}"
        )
    semantic = _string(manifest, "state_semantic", "ReferenceArchive.manifest")
    if semantic != STATE_TREE_SEMANTIC:
        raise UnsupportedReferenceArchiveVersionError(
            "ReferenceArchive.manifest.state_semantic: unsupported value "
            f"{semantic!r}; supported: {STATE_TREE_SEMANTIC!r}"
        )
    expected_digest = _string(
        manifest, "archive_content_digest", "ReferenceArchive.manifest"
    )
    _require_digest(expected_digest, "ReferenceArchive.manifest.archive_content_digest")
    encoded_episodes = _list(manifest, "episodes", "ReferenceArchive.manifest")
    expected_count = _integer(manifest, "episode_count", "ReferenceArchive.manifest")
    if expected_count != len(encoded_episodes):
        raise ReferenceArchiveError(
            "ReferenceArchive.manifest.episode_count: does not match episodes"
        )
    reader = _ArrayReader(root)
    episodes = tuple(
        _decode_episode(value, reader, index)
        for index, value in enumerate(encoded_episodes)
    )
    _validate_array_inventory(reader)
    archive = ManiSkillReferenceArchive(episodes=episodes)
    observed_digest = compute_reference_archive_digest(archive)
    if observed_digest != expected_digest:
        raise ReferenceArchiveError(
            "ReferenceArchive.manifest.archive_content_digest: content mismatch"
        )
    return archive


def assert_reference_archive_unchanged(
    output_dir: Path, expected_content_digest: str
) -> None:
    """Reload an archive and reject any semantic mutation after binding."""
    _require_digest(expected_content_digest, "ReferenceArchive.expected_content_digest")
    observed = load_reference_archive(output_dir).content_digest
    if observed != expected_content_digest:
        raise ReferenceArchiveError(
            "ReferenceArchive: content changed after it was bound"
        )


def _encode_episode(
    episode: ManiSkillReferenceEpisode, writer: _ArrayWriter
) -> dict[str, object]:
    initial_state = cast(NormalizedStateTree, episode.initial_state)
    terminal_state = cast(NormalizedStateTree, episode.terminal_state)
    return {
        "schema_version": episode.schema_version,
        "episode_id": episode.episode_id,
        "source_trajectory_id": episode.source_trajectory_id,
        "compatibility_identity": episode.compatibility_identity,
        "source_solver_identity": _thaw_json(episode.source_solver_identity),
        "environment_configuration": _thaw_json(episode.environment_configuration),
        "seed": episode.seed,
        "source_actions": writer.write(episode.source_actions),
        "source_action_digest": episode.source_action_digest,
        "initial_state": _encode_state_node(initial_state.root, "$", writer),
        "initial_state_digest": episode.initial_state_digest,
        "terminal_state": _encode_state_node(terminal_state.root, "$", writer),
        "terminal_state_digest": episode.terminal_state_digest,
        "terminal_task_evidence": _thaw_json(episode.terminal_task_evidence),
        "initial_robot_state": writer.write(episode.initial_robot_state),
        "robot_state_joint_names": list(episode.robot_state_joint_names),
        "robot_state_semantic": episode.robot_state_semantic,
        "action_contract": _thaw_json(episode.action_contract),
        "action_coordinate_frame": episode.action_coordinate_frame,
        "control_period_s": episode.control_period_s,
        "source_generation_success": episode.source_generation_success,
        "independent_baseline_success": episode.independent_baseline_success,
    }


def _decode_episode(
    value: object, reader: _ArrayReader, index: int
) -> ManiSkillReferenceEpisode:
    context = f"ReferenceEpisode[index={index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _EPISODE_FIELDS, context)
    version = _integer(item, "schema_version", context)
    if version != REFERENCE_EPISODE_SCHEMA_VERSION:
        raise UnsupportedReferenceArchiveVersionError(
            f"{context}.schema_version: unsupported version {version!r}; "
            f"supported: {REFERENCE_EPISODE_SCHEMA_VERSION}"
        )
    episode = ManiSkillReferenceEpisode(
        compatibility_identity=_string(item, "compatibility_identity", context),
        source_solver_identity=_mapping(
            _field(item, "source_solver_identity", context),
            f"{context}.source_solver_identity",
        ),
        environment_configuration=_mapping(
            _field(item, "environment_configuration", context),
            f"{context}.environment_configuration",
        ),
        seed=_integer(item, "seed", context),
        source_actions=reader.read(
            _field(item, "source_actions", context), f"{context}.source_actions"
        ),
        initial_state=NormalizedStateTree(
            _decode_state_node(
                _field(item, "initial_state", context), "$", reader, depth=0
            )
        ),
        terminal_state=NormalizedStateTree(
            _decode_state_node(
                _field(item, "terminal_state", context), "$", reader, depth=0
            )
        ),
        terminal_task_evidence=_mapping(
            _field(item, "terminal_task_evidence", context),
            f"{context}.terminal_task_evidence",
        ),
        initial_robot_state=reader.read(
            _field(item, "initial_robot_state", context),
            f"{context}.initial_robot_state",
        ),
        robot_state_joint_names=tuple(
            _string_value(name, f"{context}.robot_state_joint_names[{name_index}]")
            for name_index, name in enumerate(
                _list(item, "robot_state_joint_names", context)
            )
        ),
        robot_state_semantic=_string(item, "robot_state_semantic", context),
        action_contract=_mapping(
            _field(item, "action_contract", context), f"{context}.action_contract"
        ),
        action_coordinate_frame=_string(item, "action_coordinate_frame", context),
        control_period_s=_number(item, "control_period_s", context),
        source_generation_success=_boolean(item, "source_generation_success", context),
        independent_baseline_success=_boolean(
            item, "independent_baseline_success", context
        ),
        schema_version=version,
    )
    expected_episode_id = _string(item, "episode_id", context)
    expected_trajectory_id = _string(item, "source_trajectory_id", context)
    if episode.episode_id != expected_episode_id:
        raise ReferenceArchiveError(f"{context}.episode_id: content mismatch")
    if episode.source_trajectory_id != expected_trajectory_id:
        raise ReferenceArchiveError(f"{context}.source_trajectory_id: content mismatch")
    if episode.source_action_digest != _string(item, "source_action_digest", context):
        raise ReferenceArchiveError(f"{context}.source_action_digest: content mismatch")
    if episode.initial_state_digest != _string(item, "initial_state_digest", context):
        raise ReferenceArchiveError(f"{context}.initial_state_digest: content mismatch")
    if episode.terminal_state_digest != _string(item, "terminal_state_digest", context):
        raise ReferenceArchiveError(
            f"{context}.terminal_state_digest: content mismatch"
        )
    return episode


def _encode_state_node(
    node: StateNode, path: str, writer: _ArrayWriter
) -> dict[str, object]:
    if isinstance(node, NumericStateNode):
        return {"kind": "leaf", "path": path, "array": writer.write(node.value)}
    if isinstance(node, MappingStateNode):
        return {
            "kind": "mapping",
            "path": path,
            "items": [
                {
                    "key": key,
                    "node": _encode_state_node(
                        child, _mapping_child_path(path, key), writer
                    ),
                }
                for key, child in node.entries
            ],
        }
    return {
        "kind": "sequence",
        "path": path,
        "sequence_kind": node.sequence_kind,
        "items": [
            _encode_state_node(child, f"{path}/{index}", writer)
            for index, child in enumerate(node.items)
        ],
    }


def _decode_state_node(
    value: object,
    expected_path: str,
    reader: _ArrayReader,
    *,
    depth: int,
) -> StateNode:
    if depth > 128:
        raise ReferenceArchiveError(
            f"ReferenceArchive.state[{expected_path}]: nesting exceeds 128"
        )
    context = f"ReferenceArchive.state[{expected_path}]"
    item = _mapping(value, context)
    kind = _string(item, "kind", context)
    if _string(item, "path", context) != expected_path:
        raise ReferenceArchiveError(f"{context}.path: canonical path mismatch")
    if kind == "leaf":
        _require_exact_fields(item, {"kind", "path", "array"}, context)
        return NumericStateNode(
            reader.read(_field(item, "array", context), f"{context}.array")
        )
    if kind == "mapping":
        _require_exact_fields(item, {"kind", "path", "items"}, context)
        entries: list[tuple[str, StateNode]] = []
        for index, raw_entry in enumerate(_list(item, "items", context)):
            entry_context = f"{context}.items[{index}]"
            entry = _mapping(raw_entry, entry_context)
            _require_exact_fields(entry, {"key", "node"}, entry_context)
            key = _string(entry, "key", entry_context)
            entries.append(
                (
                    key,
                    _decode_state_node(
                        _field(entry, "node", entry_context),
                        _mapping_child_path(expected_path, key),
                        reader,
                        depth=depth + 1,
                    ),
                )
            )
        try:
            return MappingStateNode(tuple(entries))
        except StateTreeError as exc:
            raise ReferenceArchiveError(f"{context}: {exc}") from exc
    if kind == "sequence":
        _require_exact_fields(item, {"kind", "path", "sequence_kind", "items"}, context)
        sequence_kind = _string(item, "sequence_kind", context)
        if sequence_kind not in ("list", "tuple"):
            raise ReferenceArchiveError(
                f"{context}.sequence_kind: unsupported value {sequence_kind!r}"
            )
        items = tuple(
            _decode_state_node(
                child,
                f"{expected_path}/{index}",
                reader,
                depth=depth + 1,
            )
            for index, child in enumerate(_list(item, "items", context))
        )
        return SequenceStateNode(cast(Literal["list", "tuple"], sequence_kind), items)
    raise ReferenceArchiveError(f"{context}.kind: unsupported value {kind!r}")


def _validate_terminal_task_evidence(
    evidence: Mapping[str, object], context: str
) -> None:
    if not isinstance(evidence, Mapping):
        raise ReferenceArchiveError(
            f"{context}.terminal_task_evidence: expected a mapping"
        )
    _require_exact_fields(
        evidence, _TERMINAL_EVIDENCE_FIELDS, f"{context}.terminal_task_evidence"
    )
    for field in ("success", "is_obj_placed", "is_robot_static", "is_grasped"):
        if type(evidence[field]) is not bool:
            raise ReferenceArchiveError(
                f"{context}.terminal_task_evidence.{field}: expected a boolean"
            )
    for field in ("cube_center_z", "cube_to_goal_distance"):
        value = evidence[field]
        if type(value) not in (int, float):
            raise ReferenceArchiveError(
                f"{context}.terminal_task_evidence.{field}: expected a finite number"
            )
        number = float(cast(int | float, value))
        if not math.isfinite(number):
            raise ReferenceArchiveError(
                f"{context}.terminal_task_evidence.{field}: expected a finite number"
            )
    distance = cast(int | float, evidence["cube_to_goal_distance"])
    if float(distance) < 0.0:
        raise ReferenceArchiveError(
            f"{context}.terminal_task_evidence.cube_to_goal_distance: "
            "must be non-negative"
        )
    _freeze_canonical_mapping(evidence, f"{context}.terminal_task_evidence")


def _validate_solver_identity(identity: Mapping[str, object], context: str) -> None:
    if not isinstance(identity, Mapping):
        raise ReferenceArchiveError(
            f"{context}.source_solver_identity: expected a mapping"
        )
    _require_exact_fields(
        identity,
        {"module_name", "source_sha256"},
        f"{context}.source_solver_identity",
    )
    module_name = identity["module_name"]
    source_digest = identity["source_sha256"]
    _require_canonical_text(
        module_name, f"{context}.source_solver_identity.module_name"
    )
    _require_digest(source_digest, f"{context}.source_solver_identity.source_sha256")


def _freeze_numeric_array(value: NDArray[Any]) -> NDArray[Any]:
    array = _require_numeric_array(value, "ReferenceArchive.ndarray")
    detached = np.array(array, copy=True, order="C", subok=False)
    immutable = detached.tobytes(order="C")
    return np.frombuffer(immutable, dtype=detached.dtype).reshape(detached.shape)


def _require_numeric_array(value: object, context: str) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise ReferenceArchiveError(f"{context}: expected a numpy.ndarray")
    if value.dtype.hasobject or value.dtype.kind not in "biufc":
        raise ReferenceArchiveError(
            f"{context}: expected a non-object numeric dtype, got {value.dtype}"
        )
    if not bool(np.all(np.isfinite(value))):
        raise ReferenceArchiveError(f"{context}: contains NaN or infinity")
    return value


def _freeze_canonical_mapping(
    value: Mapping[str, object], context: str
) -> Mapping[str, object]:
    try:
        canonical = canonical_json_value(value, context=context)
    except ValueError as exc:
        raise ReferenceArchiveError(f"{context}: {exc}") from exc
    if not isinstance(canonical, dict):
        raise ReferenceArchiveError(f"{context}: expected a canonical JSON mapping")
    frozen = _freeze_json(canonical)
    return cast(Mapping[str, object], frozen)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _safe_array_path(root: Path, relative: str, *, writing: bool) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or "\x00" in relative
        or "\\" in relative
        or pure.is_absolute()
        or not pure.parts
        or pure.parts[0] != "arrays"
        or any(part in ("", ".", "..") for part in pure.parts)
        or len(pure.parts) != 2
        or pure.suffix != ".npy"
    ):
        raise ReferenceArchiveError(
            f"ReferenceArchive.ndarray.path: unsafe array path {relative!r}"
        )
    resolved_root = root.resolve()
    unresolved = resolved_root / Path(*pure.parts)
    candidate = unresolved.resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ReferenceArchiveError(
            f"ReferenceArchive.ndarray.path: path escapes archive {relative!r}"
        ) from exc
    if writing:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    else:
        _require_regular_unlinked_file(unresolved, "ReferenceArchive.ndarray")
        if candidate != unresolved.absolute():
            raise ReferenceArchiveError(
                f"ReferenceArchive.ndarray.path: links are unsafe {relative!r}"
            )
    return candidate


def _require_safe_root(root: Path) -> None:
    try:
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise ReferenceArchiveError(
                "ReferenceArchive.output_dir: missing or unsafe real directory"
            )
    except ReferenceArchiveError:
        raise
    except OSError as exc:
        raise ReferenceArchiveError(
            f"ReferenceArchive.output_dir: could not inspect root: {exc}"
        ) from exc


def _require_empty_destination(destination: Path) -> None:
    try:
        if destination.is_symlink() or destination.resolve() != destination:
            raise ReferenceArchiveError(
                "ReferenceArchive.output_dir: symbolic links and junctions are "
                "unsupported"
            )
    except ReferenceArchiveError:
        raise
    except OSError as exc:
        raise ReferenceArchiveError(
            f"ReferenceArchive.output_dir: could not inspect destination: {exc}"
        ) from exc
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ReferenceArchiveError("ReferenceArchive.output_dir: must be a directory")
    try:
        next(destination.iterdir())
    except StopIteration:
        return
    except OSError as exc:
        raise ReferenceArchiveError(
            f"ReferenceArchive.output_dir: could not inspect destination: {exc}"
        ) from exc
    raise ReferenceArchiveError(
        "ReferenceArchive.output_dir: existing non-empty archive is immutable"
    )


def _publish_staging_directory(staging: Path, destination: Path) -> None:
    destination_existed = destination.exists()
    if destination_existed:
        _require_empty_destination(destination)
        destination.rmdir()
    try:
        staging.replace(destination)
    except OSError:
        if destination_existed and not destination.exists():
            destination.mkdir()
        raise


def _require_regular_unlinked_file(path: Path, context: str) -> None:
    try:
        if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
            raise ReferenceArchiveError(f"{context}: missing or unsafe regular file")
        if path.stat().st_nlink != 1:
            raise ReferenceArchiveError(f"{context}: hard-linked files are unsafe")
    except ReferenceArchiveError:
        raise
    except OSError as exc:
        raise ReferenceArchiveError(
            f"{context}: could not inspect file: {exc}"
        ) from exc


def _validate_array_inventory(reader: _ArrayReader) -> None:
    if len(set(reader.references)) != len(reader.references):
        raise ReferenceArchiveError(
            "ReferenceArchive.arrays: duplicate references are unsupported"
        )
    arrays_dir = reader.root / "arrays"
    try:
        if (
            arrays_dir.is_symlink()
            or not arrays_dir.is_dir()
            or arrays_dir.resolve() != arrays_dir.absolute()
        ):
            raise ReferenceArchiveError(
                "ReferenceArchive.arrays: missing or unsafe arrays directory"
            )
        actual_files: set[str] = set()
        for path in arrays_dir.rglob("*"):
            if path.is_symlink() or path.resolve() != path.absolute():
                raise ReferenceArchiveError(
                    "ReferenceArchive.arrays: links and junctions are unsafe"
                )
            if path.is_dir():
                raise ReferenceArchiveError(
                    "ReferenceArchive.arrays: nested directories are unsupported"
                )
            if path.is_file():
                _require_regular_unlinked_file(path, "ReferenceArchive.array")
                actual_files.add(path.relative_to(reader.root).as_posix())
    except ReferenceArchiveError:
        raise
    except OSError as exc:
        raise ReferenceArchiveError(
            f"ReferenceArchive.arrays: could not inspect inventory: {exc}"
        ) from exc
    if actual_files != set(reader.references):
        raise ReferenceArchiveError(
            "ReferenceArchive.arrays: files do not exactly match manifest references"
        )


def _mapping_child_path(path: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _require_digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ReferenceArchiveError(
            f"{context}: expected sha256:<64 lowercase hexadecimal characters>"
        )
    return value


def _require_canonical_text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ReferenceArchiveError(f"{context}: expected canonical non-empty text")
    return value


def _require_exact_fields(
    item: Mapping[str, object], expected: Collection[str], context: str
) -> None:
    expected_fields = set(expected)
    actual_fields = set(item)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ReferenceArchiveError(f"{context}: invalid fields ({'; '.join(details)})")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReferenceArchiveError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _field(item: Mapping[str, object], field: str, context: str) -> object:
    if field not in item:
        raise ReferenceArchiveError(f"{context}.{field}: missing required field")
    return item[field]


def _string(item: Mapping[str, object], field: str, context: str) -> str:
    return _string_value(_field(item, field, context), f"{context}.{field}")


def _string_value(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ReferenceArchiveError(f"{context}: expected a string")
    return value


def _integer(item: Mapping[str, object], field: str, context: str) -> int:
    value = _field(item, field, context)
    if type(value) is not int:
        raise ReferenceArchiveError(f"{context}.{field}: expected an integer")
    return value


def _number(item: Mapping[str, object], field: str, context: str) -> float:
    value = _field(item, field, context)
    if type(value) not in (int, float):
        raise ReferenceArchiveError(f"{context}.{field}: expected a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ReferenceArchiveError(f"{context}.{field}: expected a finite number")
    return number


def _boolean(item: Mapping[str, object], field: str, context: str) -> bool:
    value = _field(item, field, context)
    if type(value) is not bool:
        raise ReferenceArchiveError(f"{context}.{field}: expected a boolean")
    return value


def _list(item: Mapping[str, object], field: str, context: str) -> list[object]:
    value = _field(item, field, context)
    if not isinstance(value, list):
        raise ReferenceArchiveError(f"{context}.{field}: expected an array")
    return cast(list[object], value)


def _reject_json_constant(value: str) -> NoReturn:
    raise ReferenceArchiveError(
        f"ReferenceArchive.manifest: non-finite JSON constant {value!r} is unsupported"
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceArchiveError(f"JSON object: duplicate field {key!r}")
        result[key] = value
    return result


__all__ = [
    "REFERENCE_ARCHIVE_FORMAT",
    "REFERENCE_ARCHIVE_VERSION",
    "REFERENCE_EPISODE_SCHEMA_VERSION",
    "REFERENCE_MANIFEST_NAME",
    "LoadedArchiveState",
    "ManiSkillReferenceArchive",
    "ManiSkillReferenceEpisode",
    "ReferenceArchiveError",
    "UnsupportedReferenceArchiveVersionError",
    "assert_reference_archive_unchanged",
    "compute_reference_archive_digest",
    "compute_reference_episode_content_digest",
    "compute_source_action_digest",
    "find_reference_episode",
    "load_archive_state",
    "load_reference_archive",
    "save_reference_archive",
    "validate_reference_archive",
    "validate_reference_episode",
]
