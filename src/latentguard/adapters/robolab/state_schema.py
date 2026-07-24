"""Strict, simulator-independent comparison for RoboLab state trees."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

EMPTY_MAPPING_COMPONENT = "<empty-mapping>"


@dataclass(frozen=True)
class LeafComparison:
    """Comparison result for one complete state-tree leaf."""

    path: str
    dtype: str
    shape: tuple[int, ...]
    compared_values: int
    maximum_absolute_error: float
    within_tolerance: bool


@dataclass(frozen=True)
class StateCanonicalization:
    """Audit for exact, allowlisted optional-empty state namespaces."""

    allowed_optional_empty_namespaces: tuple[str, ...]
    removed_empty_namespaces: tuple[str, ...]
    before_paths: tuple[str, ...]
    after_paths: tuple[str, ...]
    before_sha256: str
    after_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonicalization audit."""
        return {
            "allowed_optional_empty_namespaces": list(
                self.allowed_optional_empty_namespaces
            ),
            "removed_empty_namespaces": list(self.removed_empty_namespaces),
            "before_paths": list(self.before_paths),
            "after_paths": list(self.after_paths),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


@dataclass(frozen=True)
class StateComparison:
    """Complete comparison result without missing-leaf repair or coercion."""

    status: str
    tolerance: float
    expected_paths: tuple[str, ...]
    observed_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    unexpected_paths: tuple[str, ...]
    dtype_mismatches: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    non_finite_paths: tuple[str, ...]
    leaves: tuple[LeafComparison, ...]
    maximum_absolute_error: float
    expected_canonicalization: StateCanonicalization
    observed_canonicalization: StateCanonicalization

    @property
    def matches(self) -> bool:
        """Return whether every structural, numeric, and tolerance gate passed."""
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this comparison into compact JSON-compatible evidence."""
        return {
            "status": self.status,
            "tolerance": self.tolerance,
            "expected_paths": list(self.expected_paths),
            "observed_paths": list(self.observed_paths),
            "missing_paths": list(self.missing_paths),
            "unexpected_paths": list(self.unexpected_paths),
            "dtype_mismatches": list(self.dtype_mismatches),
            "shape_mismatches": list(self.shape_mismatches),
            "non_finite_paths": list(self.non_finite_paths),
            "maximum_absolute_error": self.maximum_absolute_error,
            "canonicalization": {
                "expected": self.expected_canonicalization.to_dict(),
                "observed": self.observed_canonicalization.to_dict(),
            },
            "leaves": [
                {
                    "path": leaf.path,
                    "dtype": leaf.dtype,
                    "shape": list(leaf.shape),
                    "compared_values": leaf.compared_values,
                    "maximum_absolute_error": leaf.maximum_absolute_error,
                    "within_tolerance": leaf.within_tolerance,
                }
                for leaf in self.leaves
            ],
        }


def _to_numpy(value: Any) -> NDArray[Any]:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    array = np.asarray(candidate)
    if array.dtype == np.dtype("O"):
        raise TypeError("state leaves must be numeric or boolean arrays")
    return np.ascontiguousarray(array)


def flatten_state_tree(tree: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """Flatten a nested state mapping while rejecting ambiguous empty branches."""
    flattened: dict[str, NDArray[Any]] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            if not value:
                if not prefix:
                    raise ValueError("state root mapping is empty")
                flattened[f"{prefix}/{EMPTY_MAPPING_COMPONENT}"] = np.empty(
                    0,
                    dtype=np.uint8,
                )
                return
            for key in sorted(value):
                if not isinstance(key, str) or not key:
                    raise TypeError("state-tree keys must be non-empty strings")
                if key == EMPTY_MAPPING_COMPONENT:
                    raise ValueError("state-tree key collides with empty-map marker")
                path = f"{prefix}/{key}" if prefix else key
                visit(path, value[key])
            return
        if not prefix:
            raise ValueError("state tree must contain named leaves")
        flattened[prefix] = _to_numpy(value)

    visit("", tree)
    if not flattened:
        raise ValueError("state tree has no leaves")
    return flattened


def state_tree_sha256(tree: Mapping[str, Any]) -> str:
    """Hash every state path, dtype, shape, and exact contiguous byte."""
    digest = hashlib.sha256()
    for path, array in flatten_state_tree(tree).items():
        header = json.dumps(
            {
                "dtype": array.dtype.str,
                "path": path,
                "shape": list(array.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonicalize_optional_empty_mappings(
    tree: Mapping[str, Any],
    *,
    allowed_optional_empty_namespaces: Sequence[str],
) -> tuple[dict[str, Any], StateCanonicalization]:
    """Remove only exact allowlisted empty mappings and preserve all other state."""
    allowed = tuple(sorted(set(allowed_optional_empty_namespaces)))
    for namespace in allowed:
        if (
            not namespace.startswith("/")
            or namespace == "/"
            or namespace.endswith("/")
            or f"/{EMPTY_MAPPING_COMPONENT}" in namespace
        ):
            raise ValueError(
                "optional empty namespaces must be exact absolute mapping paths"
            )
    removed: list[str] = []
    absent = object()

    def visit(value: Any, namespace: str) -> Any:
        if not isinstance(value, Mapping):
            return value
        if namespace in allowed:
            if value:
                raise ValueError(
                    f"allowlisted optional namespace is non-empty: {namespace}"
                )
            removed.append(namespace)
            return absent
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise TypeError("state-tree keys must be non-empty strings")
            child_namespace = f"{namespace}/{key}" if namespace else f"/{key}"
            child = visit(value[key], child_namespace)
            if child is not absent:
                result[key] = child
        return result

    canonical = visit(tree, "")
    if canonical is absent or not isinstance(canonical, dict):
        raise ValueError("state root cannot be removed by canonicalization")
    before_paths = tuple(flatten_state_tree(tree))
    after_paths = tuple(flatten_state_tree(canonical))
    return canonical, StateCanonicalization(
        allowed_optional_empty_namespaces=allowed,
        removed_empty_namespaces=tuple(removed),
        before_paths=before_paths,
        after_paths=after_paths,
        before_sha256=state_tree_sha256(tree),
        after_sha256=state_tree_sha256(canonical),
    )


def compare_state_trees(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    tolerance: float,
    allowed_optional_empty_namespaces: Sequence[str] = (),
) -> StateComparison:
    """Compare complete trees after exact allowlisted empty-map canonicalization."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    expected_tree, expected_canonicalization = canonicalize_optional_empty_mappings(
        expected,
        allowed_optional_empty_namespaces=allowed_optional_empty_namespaces,
    )
    observed_tree, observed_canonicalization = canonicalize_optional_empty_mappings(
        observed,
        allowed_optional_empty_namespaces=allowed_optional_empty_namespaces,
    )
    expected_flat = flatten_state_tree(expected_tree)
    observed_flat = flatten_state_tree(observed_tree)
    expected_paths = tuple(expected_flat)
    observed_paths = tuple(observed_flat)
    missing = tuple(sorted(set(expected_flat) - set(observed_flat)))
    unexpected = tuple(sorted(set(observed_flat) - set(expected_flat)))
    dtype_mismatches: list[str] = []
    shape_mismatches: list[str] = []
    non_finite_paths: list[str] = []
    leaves: list[LeafComparison] = []
    maximum = 0.0
    for path in sorted(set(expected_flat) & set(observed_flat)):
        left = expected_flat[path]
        right = observed_flat[path]
        if left.dtype != right.dtype:
            dtype_mismatches.append(path)
            continue
        if left.shape != right.shape:
            shape_mismatches.append(path)
            continue
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            non_finite_paths.append(path)
            continue
        if left.dtype.kind in {"b", "i", "u"}:
            leaf_maximum = float(np.max(left != right)) if left.size else 0.0
        else:
            difference = np.abs(
                left.astype(np.float64, copy=False)
                - right.astype(np.float64, copy=False)
            )
            leaf_maximum = float(np.max(difference)) if difference.size else 0.0
        maximum = max(maximum, leaf_maximum)
        leaves.append(
            LeafComparison(
                path=path,
                dtype=left.dtype.str,
                shape=tuple(left.shape),
                compared_values=int(left.size),
                maximum_absolute_error=leaf_maximum,
                within_tolerance=leaf_maximum <= tolerance,
            )
        )
    status = "pass"
    if (
        missing
        or unexpected
        or dtype_mismatches
        or shape_mismatches
        or non_finite_paths
        or any(not leaf.within_tolerance for leaf in leaves)
    ):
        status = "fail"
    return StateComparison(
        status=status,
        tolerance=float(tolerance),
        expected_paths=expected_paths,
        observed_paths=observed_paths,
        missing_paths=missing,
        unexpected_paths=unexpected,
        dtype_mismatches=tuple(dtype_mismatches),
        shape_mismatches=tuple(shape_mismatches),
        non_finite_paths=tuple(non_finite_paths),
        leaves=tuple(leaves),
        maximum_absolute_error=maximum,
        expected_canonicalization=expected_canonicalization,
        observed_canonicalization=observed_canonicalization,
    )
