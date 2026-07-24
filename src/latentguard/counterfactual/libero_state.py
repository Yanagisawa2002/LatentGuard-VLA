"""Complete, content-bound LIBERO state capture and restoration helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

LIBERO_STATE_SEMANTIC = "libero_complete_runtime_state_v1"
LIBERO_STATE_COMPARISON = "tolerance_verified_complete_runtime_state_v1"

_SIM_DATA_ARRAYS = (
    "act",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "qacc_warmstart",
    "qfrc_applied",
    "qpos",
    "qvel",
    "userdata",
    "xfrc_applied",
)
_SIM_DATA_RENDER_ARRAYS = (
    "body_xmat",
    "body_xpos",
    "body_xquat",
    "cam_xmat",
    "cam_xpos",
    "geom_xmat",
    "geom_xpos",
    "site_xmat",
    "site_xpos",
    "subtree_com",
    "xanchor",
    "xaxis",
    "ximat",
    "xipos",
    "xmat",
    "xpos",
    "xquat",
)
_SCALAR_RUNTIME_ATTRIBUTES = (
    "_elapsed_steps",
    "_episode_started",
    "_success",
    "done",
    "timestep",
)
_CONTROLLER_SKIP_ATTRIBUTES = frozenset(
    {
        "actuator_range",
        "control_dim",
        "control_freq",
        "control_limits",
        "interpolator_ori",
        "interpolator_pos",
        "joint_dim",
        "lite_physics",
        "model",
        "ndim",
        "policy_freq",
        "ramp_ratio",
        "ref_joint_actuator_indexes",
        "ref_joint_indexes",
        "ref_name",
        "robot_name",
        "sim",
    }
)


class LiberoStateError(ValueError):
    """Raised when a LIBERO state is incomplete, malformed, or mismatched."""


@dataclass(frozen=True, slots=True, eq=False)
class LiberoStateSnapshot:
    """Detached numeric arrays plus JSON-shaped runtime and RNG state."""

    arrays: Mapping[str, npt.NDArray[Any]]
    runtime_state: Mapping[str, Any]
    content_sha256: str
    structure_sha256: str
    semantic: str = LIBERO_STATE_SEMANTIC

    def __post_init__(self) -> None:
        """Freeze snapshot content and verify both registered digests."""
        if self.semantic != LIBERO_STATE_SEMANTIC:
            raise LiberoStateError("unsupported LIBERO state semantic")
        arrays = _freeze_arrays(self.arrays)
        runtime = _freeze_json_mapping(self.runtime_state)
        structure = _structure_digest(arrays, runtime)
        content = _content_digest(arrays, runtime)
        if self.structure_sha256 != structure:
            raise LiberoStateError("snapshot structure digest mismatch")
        if self.content_sha256 != content:
            raise LiberoStateError("snapshot content digest mismatch")
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "runtime_state", MappingProxyType(runtime))


@dataclass(frozen=True, slots=True)
class LiberoStateComparison:
    """Complete structural, scalar, and numeric restoration comparison."""

    expected_sha256: str
    observed_sha256: str
    expected_structure_sha256: str
    observed_structure_sha256: str
    structure_matches: bool
    runtime_state_matches: bool
    within_tolerance: bool
    comparison_tolerance: float
    compared_component_count: int
    maximum_absolute_error: float | None


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise LiberoStateError("runtime state must be finite JSON data") from exc


def _freeze_json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    cloned = _json_clone(dict(value))
    if not isinstance(cloned, dict):
        raise LiberoStateError("runtime state must be a mapping")
    return cloned


def _freeze_arrays(
    arrays: Mapping[str, npt.ArrayLike],
) -> dict[str, npt.NDArray[Any]]:
    if not arrays:
        raise LiberoStateError("snapshot must contain numeric arrays")
    output: dict[str, npt.NDArray[Any]] = {}
    for key in sorted(arrays):
        if not isinstance(key, str) or not key:
            raise LiberoStateError("snapshot array keys must be non-empty strings")
        value = np.asarray(arrays[key])
        if value.dtype.hasobject or value.dtype.kind not in "biufc":
            raise LiberoStateError(f"snapshot array {key} is not numeric")
        if not np.isfinite(value).all():
            raise LiberoStateError(f"snapshot array {key} contains non-finite values")
        detached = np.ascontiguousarray(value)
        immutable = np.frombuffer(
            detached.tobytes(order="C"),
            dtype=detached.dtype,
        ).reshape(detached.shape)
        output[key] = immutable
    return output


def _canonical_runtime_bytes(runtime: Mapping[str, Any]) -> bytes:
    return json.dumps(
        runtime,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _structure_digest(
    arrays: Mapping[str, npt.NDArray[Any]],
    runtime: Mapping[str, Any],
) -> str:
    payload = {
        "arrays": [
            {
                "dtype": arrays[key].dtype.str,
                "key": key,
                "shape": list(arrays[key].shape),
            }
            for key in sorted(arrays)
        ],
        "runtime_keys": sorted(runtime),
        "semantic": LIBERO_STATE_SEMANTIC,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _content_digest(
    arrays: Mapping[str, npt.NDArray[Any]],
    runtime: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    structure = _structure_digest(arrays, runtime).encode("ascii")
    digest.update(len(structure).to_bytes(8, "big"))
    digest.update(structure)
    for key in sorted(arrays):
        key_bytes = key.encode("utf-8")
        raw = arrays[key].tobytes(order="C")
        digest.update(len(key_bytes).to_bytes(8, "big"))
        digest.update(key_bytes)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    runtime_bytes = _canonical_runtime_bytes(runtime)
    digest.update(len(runtime_bytes).to_bytes(8, "big"))
    digest.update(runtime_bytes)
    return digest.hexdigest()


def _numeric_attribute(value: Any) -> npt.NDArray[Any] | None:
    if isinstance(value, np.ndarray):
        candidate = value
    elif isinstance(value, np.generic) or type(value) in (bool, int, float):
        candidate = np.asarray(value)
    elif isinstance(value, (list, tuple)):
        try:
            candidate = np.asarray(value)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if candidate.dtype.hasobject or candidate.dtype.kind not in "biufc":
        return None
    if not np.isfinite(candidate).all():
        raise LiberoStateError("controller state contains non-finite values")
    return candidate


def _controller_arrays(task_env: Any) -> dict[str, npt.NDArray[Any]]:
    output: dict[str, npt.NDArray[Any]] = {}
    robots = getattr(task_env, "robots", ())
    for robot_index, robot in enumerate(robots):
        controller = getattr(robot, "controller", None)
        if controller is None:
            raise LiberoStateError(f"robot {robot_index} has no controller")
        state = getattr(controller, "__dict__", {})
        if not isinstance(state, dict):
            raise LiberoStateError("controller state is not inspectable")
        for name, value in sorted(state.items()):
            if name.startswith("__") or name in _CONTROLLER_SKIP_ATTRIBUTES:
                continue
            candidate = _numeric_attribute(value)
            if candidate is not None:
                output[f"controller/{robot_index}/{name}"] = candidate
    return output


def _numpy_rng_state() -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "algorithm": str(algorithm),
        "cached_gaussian": float(cached),
        "has_gaussian": int(has_gauss),
        "keys": np.asarray(keys, dtype=np.uint32).tolist(),
        "position": int(position),
    }


def _torch_rng_arrays() -> dict[str, npt.NDArray[Any]]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return {}
    output = {
        "rng/torch_cpu": torch.get_rng_state().detach().cpu().numpy(),
    }
    if bool(torch.cuda.is_available()):
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            output[f"rng/torch_cuda/{index}"] = state.detach().cpu().numpy()
    return output


def _environment_rng_state(single_env: Any, control_env: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, owner in (
        ("wrapper", single_env),
        ("control", control_env),
        ("task", getattr(control_env, "env", None)),
    ):
        if owner is None:
            continue
        generator = getattr(owner, "np_random", None)
        bit_generator = getattr(generator, "bit_generator", None)
        state = getattr(bit_generator, "state", None)
        if state is not None:
            output[label] = _json_clone(state)
    return output


def _runtime_scalars(
    single_env: Any,
    control_env: Any,
    task_env: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, owner in (
        ("wrapper", single_env),
        ("control", control_env),
        ("task", task_env),
    ):
        values: dict[str, bool | int | float] = {}
        for name in _SCALAR_RUNTIME_ATTRIBUTES:
            if not hasattr(owner, name):
                continue
            value = getattr(owner, name)
            if type(value) in (bool, int, float) and math.isfinite(float(value)):
                values[name] = value
        output[label] = values
    return output


def capture_libero_state(single_env: Any) -> LiberoStateSnapshot:
    """Capture simulator, controller, wrapper, latch, and RNG state."""
    control_env = getattr(single_env, "_env", None)
    if control_env is None:
        raise LiberoStateError("LIBERO inner control environment is unavailable")
    task_env = getattr(control_env, "env", None)
    sim = getattr(task_env, "sim", None)
    if task_env is None or sim is None:
        raise LiberoStateError("LIBERO task simulator is unavailable")
    flattened = np.asarray(control_env.get_sim_state())
    arrays: dict[str, npt.NDArray[Any]] = {"sim/flattened": flattened}
    sim_data = getattr(sim, "data", None)
    if sim_data is None:
        raise LiberoStateError("MuJoCo simulation data is unavailable")
    for name in (*_SIM_DATA_ARRAYS, *_SIM_DATA_RENDER_ARRAYS):
        if hasattr(sim_data, name):
            arrays[f"sim/data/{name}"] = np.asarray(getattr(sim_data, name))
    arrays.update(_controller_arrays(task_env))
    arrays.update(_torch_rng_arrays())
    runtime = {
        "environment_rng": _environment_rng_state(single_env, control_env),
        "numpy_random": _numpy_rng_state(),
        "python_random": _json_clone(random.getstate()),
        "runtime_scalars": _runtime_scalars(single_env, control_env, task_env),
    }
    frozen_arrays = _freeze_arrays(arrays)
    frozen_runtime = _freeze_json_mapping(runtime)
    return LiberoStateSnapshot(
        arrays=frozen_arrays,
        runtime_state=frozen_runtime,
        content_sha256=_content_digest(frozen_arrays, frozen_runtime),
        structure_sha256=_structure_digest(frozen_arrays, frozen_runtime),
    )


def _restore_array(target: Any, value: npt.NDArray[Any], name: str) -> None:
    observed = np.asarray(target)
    if observed.shape != value.shape or observed.dtype != value.dtype:
        raise LiberoStateError(f"restore target {name} changed dtype or shape")
    np.copyto(observed, value, casting="no")


def _restore_controller_arrays(
    task_env: Any,
    arrays: Mapping[str, npt.NDArray[Any]],
) -> None:
    robots = getattr(task_env, "robots", ())
    for key, value in arrays.items():
        if not key.startswith("controller/"):
            continue
        _, raw_index, name = key.split("/", 2)
        index = int(raw_index)
        if index >= len(robots):
            raise LiberoStateError("controller count changed during restoration")
        controller = getattr(robots[index], "controller", None)
        if controller is None or not hasattr(controller, name):
            raise LiberoStateError(f"controller restore target is missing: {key}")
        current = getattr(controller, name)
        if isinstance(current, np.ndarray):
            _restore_array(current, value, key)
        elif np.asarray(current).shape == ():
            setattr(controller, name, np.asarray(value).item())
        else:
            setattr(controller, name, np.array(value, copy=True))


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_rng(
    single_env: Any,
    control_env: Any,
    task_env: Any,
    snapshot: LiberoStateSnapshot,
) -> None:
    runtime = snapshot.runtime_state
    numpy_state = runtime["numpy_random"]
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    random.setstate(_nested_tuple(runtime["python_random"]))
    environment = runtime["environment_rng"]
    for label, owner in (
        ("wrapper", single_env),
        ("control", control_env),
        ("task", task_env),
    ):
        if label not in environment:
            continue
        generator = getattr(owner, "np_random", None)
        bit_generator = getattr(generator, "bit_generator", None)
        if bit_generator is None:
            raise LiberoStateError(f"{label} RNG restore target is unavailable")
        bit_generator.state = _json_clone(environment[label])
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        if any(key.startswith("rng/torch_") for key in snapshot.arrays):
            raise LiberoStateError(
                "torch RNG was captured but torch is unavailable"
            ) from None
        return
    if "rng/torch_cpu" in snapshot.arrays:
        torch.set_rng_state(
            torch.from_numpy(np.array(snapshot.arrays["rng/torch_cpu"], copy=True))
        )
    cuda_states = [
        snapshot.arrays[key]
        for key in sorted(snapshot.arrays)
        if key.startswith("rng/torch_cuda/")
    ]
    if cuda_states:
        if not bool(torch.cuda.is_available()):
            raise LiberoStateError("CUDA RNG was captured but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [torch.from_numpy(np.array(state, copy=True)) for state in cuda_states]
        )


def _restore_runtime_scalars(
    single_env: Any,
    control_env: Any,
    task_env: Any,
    runtime: Mapping[str, Any],
) -> None:
    values = runtime["runtime_scalars"]
    for label, owner in (
        ("wrapper", single_env),
        ("control", control_env),
        ("task", task_env),
    ):
        for name, value in values[label].items():
            if not hasattr(owner, name):
                raise LiberoStateError(f"{label} runtime latch disappeared: {name}")
            setattr(owner, name, value)


def restore_libero_state(
    single_env: Any,
    snapshot: LiberoStateSnapshot,
    *,
    atol: float = 0.0,
) -> LiberoStateComparison:
    """Restore a complete snapshot and immediately verify every captured component."""
    control_env = getattr(single_env, "_env", None)
    if control_env is None:
        raise LiberoStateError("LIBERO inner control environment is unavailable")
    task_env = getattr(control_env, "env", None)
    sim = getattr(task_env, "sim", None)
    if task_env is None or sim is None:
        raise LiberoStateError("LIBERO task simulator is unavailable")
    setter = getattr(sim, "set_state_from_flattened", None)
    if not callable(setter):
        raise LiberoStateError("MuJoCo flattened-state restore API is unavailable")
    setter(np.array(snapshot.arrays["sim/flattened"], copy=True))
    forward = getattr(sim, "forward", None)
    if not callable(forward):
        raise LiberoStateError("MuJoCo forward API is unavailable")
    forward()
    sim_data = sim.data
    for name in _SIM_DATA_ARRAYS:
        key = f"sim/data/{name}"
        if key in snapshot.arrays:
            if not hasattr(sim_data, name):
                raise LiberoStateError(f"MuJoCo restore target disappeared: {name}")
            _restore_array(getattr(sim_data, name), snapshot.arrays[key], key)
    _restore_controller_arrays(task_env, snapshot.arrays)
    _restore_runtime_scalars(
        single_env,
        control_env,
        task_env,
        snapshot.runtime_state,
    )
    _restore_rng(single_env, control_env, task_env, snapshot)
    for name in _SIM_DATA_RENDER_ARRAYS:
        key = f"sim/data/{name}"
        if key in snapshot.arrays:
            if not hasattr(sim_data, name):
                raise LiberoStateError(
                    f"MuJoCo render-state target disappeared: {name}"
                )
            _restore_array(getattr(sim_data, name), snapshot.arrays[key], key)
    observed = capture_libero_state(single_env)
    return compare_libero_states(snapshot, observed, atol=atol)


def compare_libero_states(
    expected: LiberoStateSnapshot,
    observed: LiberoStateSnapshot,
    *,
    atol: float,
) -> LiberoStateComparison:
    """Compare complete structures and values without repair or coercion."""
    if (
        isinstance(atol, bool)
        or not isinstance(atol, (int, float))
        or not math.isfinite(float(atol))
        or float(atol) < 0
    ):
        raise LiberoStateError("comparison tolerance must be finite and non-negative")
    structure_matches = expected.structure_sha256 == observed.structure_sha256
    runtime_matches = expected.runtime_state == observed.runtime_state
    components = 0
    maximum_error: float | None = None
    within = False
    if structure_matches:
        maximum_error = 0.0
        for key in sorted(expected.arrays):
            left = expected.arrays[key]
            right = observed.arrays[key]
            components += int(left.size)
            if left.size:
                if left.dtype.kind in "biu":
                    error = (
                        0.0
                        if np.array_equal(left, right)
                        else float(
                            max(
                                abs(int(a) - int(b))
                                for a, b in zip(
                                    left.reshape(-1).tolist(),
                                    right.reshape(-1).tolist(),
                                    strict=True,
                                )
                            )
                        )
                    )
                else:
                    error = float(
                        np.max(
                            np.abs(left.astype(np.float64) - right.astype(np.float64))
                        )
                    )
                maximum_error = max(maximum_error, error)
        within = runtime_matches and maximum_error <= float(atol)
    return LiberoStateComparison(
        expected_sha256=expected.content_sha256,
        observed_sha256=observed.content_sha256,
        expected_structure_sha256=expected.structure_sha256,
        observed_structure_sha256=observed.structure_sha256,
        structure_matches=structure_matches,
        runtime_state_matches=runtime_matches,
        within_tolerance=within,
        comparison_tolerance=float(atol),
        compared_component_count=components,
        maximum_absolute_error=maximum_error,
    )


def save_libero_state(path: Path, snapshot: LiberoStateSnapshot) -> None:
    """Write one content-bound snapshot directory outside tracked artifacts."""
    if path.exists():
        raise FileExistsError(f"snapshot path already exists: {path}")
    path.mkdir(parents=True)
    arrays_path = path / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        **{key: value for key, value in snapshot.arrays.items()},
    )
    manifest = {
        "schema_version": "latentguard.lg_r2b0.libero_state_snapshot.v1",
        "semantic": snapshot.semantic,
        "comparison_semantic": LIBERO_STATE_COMPARISON,
        "content_sha256": snapshot.content_sha256,
        "structure_sha256": snapshot.structure_sha256,
        "runtime_state": dict(snapshot.runtime_state),
        "arrays_file": "arrays.npz",
        "arrays_file_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_libero_state(path: Path) -> LiberoStateSnapshot:
    """Reload and verify one state snapshot without allowing pickle content."""
    manifest_path = path / "manifest.json"
    arrays_path = path / "arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise LiberoStateError("snapshot directory is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("arrays_file") != "arrays.npz":
        raise LiberoStateError("snapshot arrays locator is invalid")
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != manifest.get(
        "arrays_file_sha256"
    ):
        raise LiberoStateError("snapshot arrays archive digest mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return LiberoStateSnapshot(
        arrays=arrays,
        runtime_state=manifest["runtime_state"],
        content_sha256=str(manifest["content_sha256"]),
        structure_sha256=str(manifest["structure_sha256"]),
        semantic=str(manifest["semantic"]),
    )


__all__ = [
    "LIBERO_STATE_COMPARISON",
    "LIBERO_STATE_SEMANTIC",
    "LiberoStateComparison",
    "LiberoStateError",
    "LiberoStateSnapshot",
    "capture_libero_state",
    "compare_libero_states",
    "load_libero_state",
    "restore_libero_state",
    "save_libero_state",
]
