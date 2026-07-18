"""Safe content-addressed state snapshots for M4C crash recovery."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np

from .state_tree import (
    MappingStateNode,
    NormalizedStateTree,
    NumericStateNode,
    SequenceStateNode,
    StateNode,
    compute_state_tree_digest,
    normalize_state_tree,
)

STATE_SNAPSHOT_FORMAT = "latentguard-m4c-state-snapshot-v1"


class ClosedLoopStateStoreError(ValueError):
    """Raised when a recovery snapshot is incomplete or content-drifted."""


def _encode_node(node: StateNode, arrays: list[np.ndarray[Any, Any]]) -> object:
    if isinstance(node, NumericStateNode):
        index = len(arrays)
        arrays.append(np.asarray(node.value))
        return {
            "array_index": index,
            "dtype": node.value.dtype.str,
            "kind": "numeric",
            "shape": list(node.value.shape),
        }
    if isinstance(node, MappingStateNode):
        return {
            "entries": [
                [key, _encode_node(value, arrays)] for key, value in node.entries
            ],
            "kind": "mapping",
        }
    if isinstance(node, SequenceStateNode):
        return {
            "items": [_encode_node(value, arrays) for value in node.items],
            "kind": "sequence",
            "sequence_kind": node.sequence_kind,
        }
    raise ClosedLoopStateStoreError("unsupported normalized state node")


def _decode_node(value: object, arrays: tuple[np.ndarray[Any, Any], ...]) -> StateNode:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ClosedLoopStateStoreError("state snapshot node is malformed")
    kind = value["kind"]
    if kind == "numeric":
        if set(value) != {"array_index", "dtype", "kind", "shape"}:
            raise ClosedLoopStateStoreError("numeric state node fields differ")
        index = value["array_index"]
        if type(index) is not int or not 0 <= index < len(arrays):
            raise ClosedLoopStateStoreError("numeric state array index is invalid")
        array = arrays[index]
        if value["dtype"] != array.dtype.str or value["shape"] != list(array.shape):
            raise ClosedLoopStateStoreError("numeric state array contract differs")
        return NumericStateNode(array)
    if kind == "mapping":
        if set(value) != {"entries", "kind"} or not isinstance(value["entries"], list):
            raise ClosedLoopStateStoreError("mapping state node fields differ")
        entries: list[tuple[str, StateNode]] = []
        for entry in value["entries"]:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
            ):
                raise ClosedLoopStateStoreError("mapping state entry is malformed")
            entries.append((entry[0], _decode_node(entry[1], arrays)))
        return MappingStateNode(tuple(entries))
    if kind == "sequence":
        if set(value) != {"items", "kind", "sequence_kind"} or not isinstance(
            value["items"], list
        ):
            raise ClosedLoopStateStoreError("sequence state node fields differ")
        sequence_kind = value["sequence_kind"]
        if sequence_kind not in {"list", "tuple"}:
            raise ClosedLoopStateStoreError("sequence kind is unsupported")
        return SequenceStateNode(
            cast(Any, sequence_kind),
            tuple(_decode_node(item, arrays) for item in value["items"]),
        )
    raise ClosedLoopStateStoreError("unknown state snapshot node kind")


class ClosedLoopStateStore:
    """Persist immutable complete states under content-only references."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state: object) -> tuple[str, str]:
        """Save or validate one state and return reference plus exact digest."""

        tree = normalize_state_tree(state)
        digest = compute_state_tree_digest(tree)
        reference = f"m4c-state:{digest}"
        destination = self.root / digest.removeprefix("sha256:")
        if destination.exists():
            loaded = self.load(reference)
            if compute_state_tree_digest(loaded) != digest:
                raise ClosedLoopStateStoreError("existing state snapshot differs")
            return reference, digest
        temporary = self.root / f".tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            arrays: list[np.ndarray[Any, Any]] = []
            encoded = _encode_node(tree.root, arrays)
            array_inventory: list[dict[str, object]] = []
            for index, array in enumerate(arrays):
                path = temporary / f"{index:04d}.npy"
                with path.open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                array_inventory.append(
                    {
                        "dtype": array.dtype.str,
                        "path": path.name,
                        "shape": list(array.shape),
                    }
                )
            manifest = {
                "arrays": array_inventory,
                "format": STATE_SNAPSHOT_FORMAT,
                "root": encoded,
                "semantic": tree.semantic,
                "state_digest": digest,
            }
            with (temporary / "manifest.json").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return reference, digest

    def load(self, reference: str) -> NormalizedStateTree:
        """Reload and digest-verify one content-addressed snapshot."""

        prefix = "m4c-state:sha256:"
        if not reference.startswith(prefix) or len(reference) != len(prefix) + 64:
            raise ClosedLoopStateStoreError("state reference is invalid")
        digest = reference.removeprefix("m4c-state:")
        root = self.root / digest.removeprefix("sha256:")
        manifest_path = root / "manifest.json"
        if (
            root.is_symlink()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            raise ClosedLoopStateStoreError("state snapshot manifest is unsafe")
        raw = cast(object, json.loads(manifest_path.read_text(encoding="utf-8")))
        if not isinstance(raw, dict) or set(raw) != {
            "arrays",
            "format",
            "root",
            "semantic",
            "state_digest",
        }:
            raise ClosedLoopStateStoreError("state snapshot manifest fields differ")
        if raw["format"] != STATE_SNAPSHOT_FORMAT or raw["state_digest"] != digest:
            raise ClosedLoopStateStoreError("state snapshot identity differs")
        inventory = raw["arrays"]
        if not isinstance(inventory, list):
            raise ClosedLoopStateStoreError("state array inventory is malformed")
        arrays: list[np.ndarray[Any, Any]] = []
        for index, entry in enumerate(inventory):
            expected_path = f"{index:04d}.npy"
            if (
                not isinstance(entry, dict)
                or set(entry) != {"dtype", "path", "shape"}
                or entry["path"] != expected_path
            ):
                raise ClosedLoopStateStoreError("state array inventory differs")
            path = root / expected_path
            if path.is_symlink() or not path.is_file():
                raise ClosedLoopStateStoreError("state array file is unsafe")
            with path.open("rb") as stream:
                array = np.load(stream, allow_pickle=False)
            if entry["dtype"] != array.dtype.str or entry["shape"] != list(array.shape):
                raise ClosedLoopStateStoreError("state array content contract differs")
            arrays.append(array)
        tree = NormalizedStateTree(_decode_node(raw["root"], tuple(arrays)))
        if compute_state_tree_digest(tree) != digest:
            raise ClosedLoopStateStoreError("state snapshot digest mismatch")
        return tree


__all__ = ["ClosedLoopStateStore", "ClosedLoopStateStoreError"]
