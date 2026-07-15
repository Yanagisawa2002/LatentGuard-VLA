"""Strict, simulator-free normalization for ManiSkill numeric state trees."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

STATE_TREE_SEMANTIC = "maniskill_state_tree_v1"
"""Versioned semantic used by M2C state archives and restoration checks."""

_DIGEST_PREFIX = "sha256:"
_MAX_TREE_DEPTH = 128
_NUMERIC_DTYPE_KINDS = frozenset("biufc")


class StateTreeError(ValueError):
    """Raised when a simulator state tree is malformed or unsupported."""


@dataclass(frozen=True, slots=True, eq=False)
class NumericStateNode:
    """One detached, immutable numeric leaf in a normalized state tree."""

    value: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate and freeze the numeric value without changing its semantics."""
        object.__setattr__(self, "value", _freeze_numeric_array(self.value, "$"))


@dataclass(frozen=True, slots=True)
class MappingStateNode:
    """One mapping node with canonical key order."""

    entries: tuple[tuple[str, StateNode], ...]

    def __post_init__(self) -> None:
        """Validate canonical, unique mapping entries."""
        entries = tuple(self.entries)
        keys = tuple(key for key, _ in entries)
        for key in keys:
            _validate_mapping_key(key, "$")
        if len(set(keys)) != len(keys):
            raise StateTreeError("state tree $: duplicate mapping keys are unsupported")
        if keys != tuple(sorted(keys)):
            raise StateTreeError(
                "state tree $: mapping keys must be canonically sorted"
            )
        for _, node in entries:
            if not isinstance(
                node, (NumericStateNode, MappingStateNode, SequenceStateNode)
            ):
                raise StateTreeError("state tree $: invalid normalized mapping child")
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True, slots=True)
class SequenceStateNode:
    """One list or tuple node whose original sequence kind is preserved."""

    sequence_kind: Literal["list", "tuple"]
    items: tuple[StateNode, ...]

    def __post_init__(self) -> None:
        """Validate the sequence kind and freeze its children."""
        if self.sequence_kind not in ("list", "tuple"):
            raise StateTreeError(
                f"state tree $: unsupported sequence kind {self.sequence_kind!r}"
            )
        items = tuple(self.items)
        for node in items:
            if not isinstance(
                node, (NumericStateNode, MappingStateNode, SequenceStateNode)
            ):
                raise StateTreeError("state tree $: invalid normalized sequence child")
        object.__setattr__(self, "items", items)


StateNode: TypeAlias = NumericStateNode | MappingStateNode | SequenceStateNode


@dataclass(frozen=True, slots=True)
class NormalizedStateTree:
    """A detached state tree containing only strict normalized node types."""

    root: StateNode
    semantic: str = STATE_TREE_SEMANTIC

    def __post_init__(self) -> None:
        """Validate the semantic and root node."""
        if self.semantic != STATE_TREE_SEMANTIC:
            raise StateTreeError(
                "state tree semantic: unsupported version "
                f"{self.semantic!r}; supported: {STATE_TREE_SEMANTIC!r}"
            )
        if not isinstance(
            self.root, (NumericStateNode, MappingStateNode, SequenceStateNode)
        ):
            raise StateTreeError("state tree $: invalid normalized root")


@dataclass(frozen=True, slots=True, eq=False)
class StateLeaf:
    """One flattened numeric state leaf and its canonical JSON-pointer path."""

    path: str
    value: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate the path and detach the numeric value."""
        if not isinstance(self.path, str) or not self.path.startswith("$"):
            raise StateTreeError("state leaf path: expected a canonical path")
        object.__setattr__(self, "value", _freeze_numeric_array(self.value, self.path))


@dataclass(frozen=True, slots=True)
class StateTreeComparison:
    """Strict structural and numeric comparison of two state trees."""

    expected_digest: str
    observed_digest: str
    expected_structure_digest: str
    observed_structure_digest: str
    structure_matches: bool
    exact_digest_match: bool
    within_tolerance: bool
    comparison_tolerance: float
    expected_leaf_count: int
    observed_leaf_count: int
    compared_component_count: int
    maximum_absolute_error: float | None


def normalize_state_tree(value: object) -> NormalizedStateTree:
    """Detach and strictly normalize a nested numeric simulator state tree."""
    if isinstance(value, NormalizedStateTree):
        return value
    active_containers: set[int] = set()
    root = _normalize_node(value, "$", active_containers, depth=0)
    return NormalizedStateTree(root=root)


def clone_state_tree(value: object) -> object:
    """Return a fresh mapping/list/tuple tree with writable detached arrays."""
    tree = normalize_state_tree(value)
    return _materialize_node(tree.root)


def flatten_state_tree(value: object) -> tuple[StateLeaf, ...]:
    """Return numeric leaves in canonical traversal order."""
    tree = normalize_state_tree(value)
    leaves: list[StateLeaf] = []
    _collect_leaves(tree.root, "$", leaves)
    return tuple(leaves)


def compute_state_tree_digest(value: object) -> str:
    """Hash structure, canonical paths, dtype, shape, and raw numeric bytes."""
    tree = normalize_state_tree(value)
    digest = hashlib.sha256()
    _update_record(
        digest,
        {
            "format": "latentguard-maniskill-state-tree",
            "semantic": STATE_TREE_SEMANTIC,
        },
    )
    _update_node_digest(digest, tree.root, "$", include_values=True)
    return f"{_DIGEST_PREFIX}{digest.hexdigest()}"


def compute_state_tree_structure_digest(value: object) -> str:
    """Hash state structure, paths, dtypes, and shapes without leaf values."""
    tree = normalize_state_tree(value)
    digest = hashlib.sha256()
    _update_record(
        digest,
        {
            "format": "latentguard-maniskill-state-tree-structure",
            "semantic": STATE_TREE_SEMANTIC,
        },
    )
    _update_node_digest(digest, tree.root, "$", include_values=False)
    return f"{_DIGEST_PREFIX}{digest.hexdigest()}"


def compare_state_trees(
    expected: object,
    observed: object,
    *,
    atol: float,
) -> StateTreeComparison:
    """Compare two states without coercion, reshaping, or missing-leaf repair."""
    if type(atol) not in (int, float) or not math.isfinite(float(atol)) or atol < 0:
        raise StateTreeError("state comparison tolerance: expected a finite value >= 0")
    tolerance = float(atol)
    expected_tree = normalize_state_tree(expected)
    observed_tree = normalize_state_tree(observed)
    expected_digest = compute_state_tree_digest(expected_tree)
    observed_digest = compute_state_tree_digest(observed_tree)
    expected_structure = compute_state_tree_structure_digest(expected_tree)
    observed_structure = compute_state_tree_structure_digest(observed_tree)
    expected_leaves = flatten_state_tree(expected_tree)
    observed_leaves = flatten_state_tree(observed_tree)
    structure_matches = expected_structure == observed_structure

    compared_components = 0
    maximum_error: float | None = None
    within_tolerance = False
    if structure_matches:
        maximum_error = 0.0
        for expected_leaf, observed_leaf in zip(
            expected_leaves, observed_leaves, strict=True
        ):
            compared_components += int(expected_leaf.value.size)
            leaf_error = _maximum_absolute_error(
                expected_leaf.value, observed_leaf.value
            )
            maximum_error = max(maximum_error, leaf_error)
        within_tolerance = maximum_error <= tolerance

    return StateTreeComparison(
        expected_digest=expected_digest,
        observed_digest=observed_digest,
        expected_structure_digest=expected_structure,
        observed_structure_digest=observed_structure,
        structure_matches=structure_matches,
        exact_digest_match=expected_digest == observed_digest,
        within_tolerance=within_tolerance,
        comparison_tolerance=tolerance,
        expected_leaf_count=len(expected_leaves),
        observed_leaf_count=len(observed_leaves),
        compared_component_count=compared_components,
        maximum_absolute_error=maximum_error,
    )


def _normalize_node(
    value: object,
    path: str,
    active_containers: set[int],
    *,
    depth: int,
) -> StateNode:
    if depth > _MAX_TREE_DEPTH:
        raise StateTreeError(
            f"state tree {path}: nesting exceeds maximum depth {_MAX_TREE_DEPTH}"
        )
    if isinstance(value, Mapping):
        return _normalize_mapping(value, path, active_containers, depth=depth)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview, np.ndarray)
    ):
        return _normalize_sequence(value, path, active_containers, depth=depth)
    return NumericStateNode(_coerce_numeric_leaf(value, path))


def _normalize_mapping(
    value: Mapping[object, object],
    path: str,
    active_containers: set[int],
    *,
    depth: int,
) -> MappingStateNode:
    identity = id(value)
    if identity in active_containers:
        raise StateTreeError(f"state tree {path}: cyclic mappings are unsupported")
    active_containers.add(identity)
    try:
        items: list[tuple[str, StateNode]] = []
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise StateTreeError(
                    f"state tree {path}: mapping keys must be strings, got "
                    f"{type(raw_key).__name__}"
                )
            _validate_mapping_key(raw_key, path)
            child_path = _mapping_child_path(path, raw_key)
            items.append(
                (
                    raw_key,
                    _normalize_node(
                        item,
                        child_path,
                        active_containers,
                        depth=depth + 1,
                    ),
                )
            )
        items.sort(key=lambda pair: pair[0])
        return MappingStateNode(tuple(items))
    finally:
        active_containers.remove(identity)


def _normalize_sequence(
    value: Sequence[object],
    path: str,
    active_containers: set[int],
    *,
    depth: int,
) -> SequenceStateNode:
    identity = id(value)
    if identity in active_containers:
        raise StateTreeError(f"state tree {path}: cyclic sequences are unsupported")
    active_containers.add(identity)
    try:
        kind: Literal["list", "tuple"]
        if isinstance(value, list):
            kind = "list"
        elif isinstance(value, tuple):
            kind = "tuple"
        else:
            raise StateTreeError(
                f"state tree {path}: only list and tuple sequences are supported"
            )
        items = tuple(
            _normalize_node(
                item,
                _sequence_child_path(path, index),
                active_containers,
                depth=depth + 1,
            )
            for index, item in enumerate(value)
        )
        return SequenceStateNode(kind, items)
    finally:
        active_containers.remove(identity)


def _coerce_numeric_leaf(value: object, path: str) -> NDArray[Any]:
    candidate = value
    if not isinstance(candidate, (np.ndarray, np.generic)) and type(candidate) not in (
        bool,
        int,
        float,
        complex,
    ):
        candidate = _tensor_to_numpy(candidate, path)
    try:
        array = np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise StateTreeError(
            f"state tree {path}: could not convert numeric leaf to ndarray"
        ) from exc
    return _freeze_numeric_array(array, path)


def _tensor_to_numpy(value: object, path: str) -> object:
    candidate = value
    for method_name in ("detach", "cpu"):
        method = getattr(candidate, method_name, None)
        if not callable(method):
            raise StateTreeError(
                f"state tree {path}: unsupported leaf type {type(value).__name__}"
            )
        try:
            candidate = method()
        except Exception as exc:
            raise StateTreeError(
                f"state tree {path}: tensor {method_name}() conversion failed"
            ) from exc
    numpy_method = getattr(candidate, "numpy", None)
    if not callable(numpy_method):
        raise StateTreeError(
            f"state tree {path}: tensor-like leaf has no callable numpy()"
        )
    try:
        return numpy_method()
    except Exception as exc:
        raise StateTreeError(
            f"state tree {path}: tensor numpy() conversion failed"
        ) from exc


def _freeze_numeric_array(value: object, path: str) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise StateTreeError(f"state tree {path}: expected a numpy.ndarray leaf")
    if value.dtype.hasobject or value.dtype.kind not in _NUMERIC_DTYPE_KINDS:
        raise StateTreeError(
            f"state tree {path}: unsupported non-numeric dtype {value.dtype}"
        )
    try:
        finite = bool(np.all(np.isfinite(value)))
    except TypeError as exc:
        raise StateTreeError(
            f"state tree {path}: dtype {value.dtype} does not support finite checks"
        ) from exc
    if not finite:
        raise StateTreeError(
            f"state tree {path}: numeric leaf contains NaN or infinity"
        )
    detached = np.array(value, copy=True, order="C", subok=False)
    immutable_bytes = detached.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=detached.dtype).reshape(detached.shape)


def _validate_mapping_key(key: object, path: str) -> None:
    if not isinstance(key, str) or not key:
        raise StateTreeError(
            f"state tree {path}: mapping keys must be non-empty strings"
        )
    if key != key.strip():
        raise StateTreeError(
            f"state tree {path}: mapping keys may not have surrounding whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in key):
        raise StateTreeError(
            f"state tree {path}: mapping keys may not contain control characters"
        )


def _mapping_child_path(path: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _sequence_child_path(path: str, index: int) -> str:
    return f"{path}/{index}"


def _materialize_node(node: StateNode) -> object:
    if isinstance(node, NumericStateNode):
        return np.array(node.value, copy=True, order="C", subok=False)
    if isinstance(node, MappingStateNode):
        return {key: _materialize_node(child) for key, child in node.entries}
    items = tuple(_materialize_node(child) for child in node.items)
    return list(items) if node.sequence_kind == "list" else items


def _collect_leaves(node: StateNode, path: str, output: list[StateLeaf]) -> None:
    if isinstance(node, NumericStateNode):
        output.append(StateLeaf(path, node.value))
        return
    if isinstance(node, MappingStateNode):
        for key, child in node.entries:
            _collect_leaves(child, _mapping_child_path(path, key), output)
        return
    for index, child in enumerate(node.items):
        _collect_leaves(child, _sequence_child_path(path, index), output)


def _node_record(node: StateNode, path: str) -> dict[str, object]:
    if isinstance(node, NumericStateNode):
        return {
            "dtype": node.value.dtype.str,
            "kind": "leaf",
            "path": path,
            "shape": list(node.value.shape),
        }
    if isinstance(node, MappingStateNode):
        return {
            "keys": [key for key, _ in node.entries],
            "kind": "mapping",
            "path": path,
        }
    return {
        "kind": "sequence",
        "length": len(node.items),
        "path": path,
        "sequence_kind": node.sequence_kind,
    }


def _update_node_digest(
    digest: Any,
    node: StateNode,
    path: str,
    *,
    include_values: bool,
) -> None:
    _update_record(digest, _node_record(node, path))
    if isinstance(node, NumericStateNode):
        if include_values:
            raw = node.value.tobytes(order="C")
            digest.update(len(raw).to_bytes(8, byteorder="big"))
            digest.update(raw)
        return
    if isinstance(node, MappingStateNode):
        for key, child in node.entries:
            _update_node_digest(
                digest,
                child,
                _mapping_child_path(path, key),
                include_values=include_values,
            )
        return
    for index, child in enumerate(node.items):
        _update_node_digest(
            digest,
            child,
            _sequence_child_path(path, index),
            include_values=include_values,
        )


def _update_record(digest: Any, record: object) -> None:
    encoded = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big"))
    digest.update(encoded)


def _maximum_absolute_error(expected: NDArray[Any], observed: NDArray[Any]) -> float:
    if expected.size == 0:
        return 0.0
    kind = expected.dtype.kind
    if kind == "b":
        return 0.0 if np.array_equal(expected, observed) else 1.0
    if kind in "iu":
        maximum = 0
        for left, right in zip(
            expected.reshape(-1).tolist(),
            observed.reshape(-1).tolist(),
            strict=True,
        ):
            maximum = max(maximum, abs(int(left) - int(right)))
        try:
            error = float(maximum)
        except OverflowError:
            return math.inf
        return error
    difference = np.abs(
        expected.astype(np.complex128 if kind == "c" else np.float64)
        - observed.astype(np.complex128 if kind == "c" else np.float64)
    )
    return float(cast(NDArray[Any], difference).max())


__all__ = [
    "STATE_TREE_SEMANTIC",
    "MappingStateNode",
    "NormalizedStateTree",
    "NumericStateNode",
    "SequenceStateNode",
    "StateLeaf",
    "StateNode",
    "StateTreeComparison",
    "StateTreeError",
    "clone_state_tree",
    "compare_state_trees",
    "compute_state_tree_digest",
    "compute_state_tree_structure_digest",
    "flatten_state_tree",
    "normalize_state_tree",
]
