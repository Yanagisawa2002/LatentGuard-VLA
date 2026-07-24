"""Shared remote-only runtime helpers for LG-R2b0."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from _lg_r0_runtime import (
    configure_libero_assets,
    load_processors,
    load_stack,
)
from _lg_r1b_common import (
    read_json,
    read_yaml,
    repo_root,
    resolve_repo_path,
    sha256_path,
    write_json,
)
from lg_r1_collect_rollouts import _extract_privileged, _step_without_terminal_reset

from latentguard.adapters.vla_jepa.constants import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    LEROBOT_COMMIT,
    LEROBOT_VERSION,
)
from latentguard.counterfactual.candidates import action_content_sha256
from latentguard.counterfactual.models import RealPolicyCandidate

FINAL_SEEDS = frozenset(range(900_000, 900_100))


def output_root() -> Path:
    """Return the required ignored LG-R2b0 runtime root."""
    configured = os.environ.get("LG_R2B0_OUTPUT_ROOT")
    if not configured:
        raise RuntimeError("LG_R2B0_OUTPUT_ROOT must be explicit")
    root = Path(configured).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def git_output(*arguments: str) -> str:
    """Run a read-only Git query against the local execution checkout."""
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo_root(),
        text=True,
    ).strip()


def runtime_identity() -> dict[str, Any]:
    """Return sanitized execution identity without machine-specific locators."""
    import torch

    return {
        "git_branch": git_output("branch", "--show-current"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }


def validate_execution_checkout(expected_commit: str) -> None:
    """Fail if the remote checkout is dirty or not at the pushed revision."""
    if git_output("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("execution checkout does not match expected commit")
    if git_output("status", "--porcelain"):
        raise RuntimeError("execution checkout contains tracked or untracked changes")


def validate_no_final_seed(values: list[int], label: str) -> None:
    """Reject every access to the sealed final seed range."""
    overlap = sorted(set(values) & FINAL_SEEDS)
    if overlap:
        raise ValueError(f"{label} accesses sealed final seeds: {overlap}")


def canonical_sha256(value: Any) -> str:
    """Hash finite JSON-shaped data with a canonical representation."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: Any) -> str:
    """Hash array dtype, shape, and exact contiguous bytes."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def unbatch_observation(value: Any) -> Any:
    """Remove exactly one vector-environment batch dimension."""
    if isinstance(value, dict):
        return {key: unbatch_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.shape[0] != 1:
            raise ValueError("LG-R2b0 requires exactly one environment")
        return value[0]
    return value


def batch_observation(value: Any) -> Any:
    """Add exactly one vector-environment batch dimension."""
    if isinstance(value, dict):
        return {key: batch_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None]
    return value


def observation_sha256(raw_observation: dict[str, Any], instruction: str) -> str:
    """Bind both camera bytes and the exact instruction at an anchor."""
    digest = hashlib.sha256()
    digest.update(instruction.encode("utf-8"))
    for key in sorted(raw_observation["pixels"]):
        image = np.ascontiguousarray(raw_observation["pixels"][key])
        digest.update(key.encode("utf-8"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def proprioception_sha256(raw_observation: dict[str, Any]) -> str:
    """Bind the complete nested public robot observation."""
    digest = hashlib.sha256()

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(f"{prefix}/{key}", value[key])
            return
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(prefix.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())

    visit("robot_state", raw_observation["robot_state"])
    return digest.hexdigest()


def rendered_observation_sha256(raw_observation: dict[str, Any]) -> str:
    """Hash exact rendered camera arrays at one state boundary."""
    digest = hashlib.sha256()
    for key in sorted(raw_observation["pixels"]):
        image = np.ascontiguousarray(raw_observation["pixels"][key])
        digest.update(key.encode("utf-8"))
        digest.update(image.tobytes())
    return digest.hexdigest()


def compare_rendered_observations(
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, float | int | bool]:
    """Compare complete rendered arrays without resizing or coercion."""
    expected_pixels = expected["pixels"]
    observed_pixels = observed["pixels"]
    if set(expected_pixels) != set(observed_pixels):
        raise ValueError("rendered camera keys changed")
    total_values = 0
    different_values = 0
    absolute_sum = 0.0
    maximum = 0.0
    for key in sorted(expected_pixels):
        left = np.asarray(expected_pixels[key])
        right = np.asarray(observed_pixels[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"rendered camera contract changed: {key}")
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        total_values += int(difference.size)
        different_values += int(np.count_nonzero(difference))
        absolute_sum += float(np.sum(difference))
        if difference.size:
            maximum = max(maximum, float(np.max(difference)))
    return {
        "exact": different_values == 0,
        "compared_values": total_values,
        "different_values": different_values,
        "different_value_ratio": (
            different_values / total_values if total_values else 0.0
        ),
        "mean_absolute_error": (absolute_sum / total_values if total_values else 0.0),
        "maximum_absolute_error": maximum,
    }


def make_libero_environment(
    *,
    suite: str,
    task_id: int,
    episode_horizon: int,
    seed: int,
    stack: Any,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Create and reset the frozen single-environment LIBERO stack."""
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    env_config = LiberoEnv(
        task=suite,
        task_ids=[task_id],
        episode_length=episode_horizon,
        obs_type="pixels_agent_pos",
        observation_height=224,
        observation_width=224,
        init_states=True,
        control_mode="relative",
    )
    vector_env = make_env(env_config, n_envs=1, use_async_envs=False)[suite][task_id]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config,
        policy_cfg=stack.config,
    )
    raw_observation, _ = vector_env.reset(seed=[seed])
    return (
        vector_env,
        vector_env.envs[0],
        env_preprocessor,
        env_postprocessor,
        raw_observation,
    )


def policy_batch(
    raw_observation: dict[str, Any],
    *,
    instruction: str,
    env_preprocessor: Any,
    preprocessor: Any,
) -> dict[str, Any]:
    """Apply the frozen official environment and policy preprocessing chain."""
    from lerobot.envs import preprocess_observation

    observation = preprocess_observation(deepcopy(raw_observation))
    observation["task"] = [instruction]
    return dict(preprocessor(env_preprocessor(observation)))


def _native_chunk(
    normalized: Any,
    *,
    postprocessor: Any,
    env_postprocessor: Any,
) -> np.ndarray:
    import torch
    from lerobot.utils.constants import ACTION

    if normalized.ndim != 3 or normalized.shape[0] != 1 or normalized.shape[2] != 7:
        raise ValueError("native VLA-JEPA output must have shape [1, horizon, 7]")
    rows = []
    for index in range(int(normalized.shape[1])):
        processed = postprocessor(normalized[:, index, :])
        transition = env_postprocessor({ACTION: processed})
        row = transition[ACTION].detach().to("cpu", dtype=torch.float32).numpy()[0]
        rows.append(row)
    native = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(native).all():
        raise ValueError("official postprocessor emitted non-finite actions")
    if bool(np.any(native < -1.0)) or bool(np.any(native > 1.0)):
        raise ValueError("official postprocessor emitted out-of-bounds actions")
    return native


def generate_policy_candidate(
    *,
    stack: Any,
    raw_observation: dict[str, Any],
    instruction: str,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    anchor_id: str,
    candidate_id: str,
    inference_seed: int,
) -> RealPolicyCandidate:
    """Generate one isolated real VLA-JEPA action chunk from an anchor."""
    import torch

    validate_no_final_seed([inference_seed], "candidate inference")
    stack.policy.reset()
    torch.manual_seed(inference_seed)
    torch.cuda.manual_seed_all(inference_seed)
    batch = policy_batch(
        raw_observation,
        instruction=instruction,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        normalized_tensor = stack.policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1_000.0
    normalized = normalized_tensor.detach().to("cpu", dtype=torch.float32).numpy()[0]
    native = _native_chunk(
        normalized_tensor,
        postprocessor=postprocessor,
        env_postprocessor=env_postprocessor,
    )
    sampling = {
        "source": "global_torch_rng_native_flow_initialization",
        "inference_seed": inference_seed,
        "num_inference_timesteps": int(stack.config.num_inference_timesteps),
        "solver": "explicit_euler",
        "generator_argument_supported": False,
        "noise_argument_effective": False,
        "policy_queue_reset": True,
    }
    content = action_content_sha256(normalized, native, None)
    return RealPolicyCandidate(
        candidate_id=candidate_id,
        anchor_id=anchor_id,
        policy_id=CHECKPOINT_ID,
        checkpoint_revision=CHECKPOINT_REVISION,
        processor_revision=CHECKPOINT_REVISION,
        inference_seed=inference_seed,
        sampling_config=sampling,
        normalized_action_chunk=normalized,
        native_action_chunk=native,
        action_mask=None,
        content_sha256=content,
        generation_latency_ms=latency_ms,
    )


def select_primary_action(
    *,
    stack: Any,
    raw_observation: dict[str, Any],
    instruction: str,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and postprocess one frozen primary-policy action."""
    import torch
    from lerobot.utils.constants import ACTION

    batch = policy_batch(
        raw_observation,
        instruction=instruction,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
    )
    with torch.inference_mode():
        selected = stack.policy.select_action(batch)
    normalized = selected.detach().to("cpu", dtype=torch.float32).numpy()[0]
    transition = env_postprocessor({ACTION: postprocessor(selected)})
    native = transition[ACTION].detach().to("cpu", dtype=torch.float32).numpy()[0]
    if not np.isfinite(native).all() or bool(np.any(np.abs(native) > 1.0)):
        raise ValueError("primary policy emitted an invalid native action")
    return normalized, native


def step_native(
    single_env: Any,
    action: np.ndarray,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    """Execute one native action without the LeRobot terminal auto-reset."""
    observation, reward, terminated, _, info = _step_without_terminal_reset(
        single_env,
        np.asarray(action, dtype=np.float32),
    )
    return observation, reward, terminated, info


def current_observation(single_env: Any) -> dict[str, Any]:
    """Regenerate the public observation from the current state without stepping."""
    control_env = getattr(single_env, "_env", None)
    if control_env is None:
        raise RuntimeError("LIBERO inner environment is unavailable")
    updater = getattr(control_env, "_update_observables", None)
    if not callable(updater):
        raise RuntimeError("LIBERO observable refresh API is unavailable")
    updater(force=True)
    task_env = getattr(control_env, "env", None)
    getter = getattr(task_env, "_get_observations", None)
    if not callable(getter):
        raise RuntimeError("LIBERO observation regeneration API is unavailable")
    return single_env._format_raw_obs(getter())


def privileged_state(
    single_env: Any,
    raw_unbatched: dict[str, Any],
    *,
    terminal_success: bool,
) -> dict[str, Any]:
    """Reuse the accepted LG-R1b privileged state projection."""
    return _extract_privileged(
        single_env,
        raw_unbatched,
        terminal_success=terminal_success,
    )


def load_runtime_stack() -> tuple[Any, Any, Any]:
    """Load the frozen policy and official processors after asset binding."""
    configure_libero_assets()
    stack = load_stack()
    preprocessor, postprocessor = load_processors(stack)
    if stack.checkpoint_revision != CHECKPOINT_REVISION:
        raise ValueError("frozen checkpoint identity mismatch")
    return stack, preprocessor, postprocessor


def public_candidate(candidate: RealPolicyCandidate) -> dict[str, Any]:
    """Serialize one compact candidate without runtime machine paths."""
    return {
        "candidate_id": candidate.candidate_id,
        "anchor_id": candidate.anchor_id,
        "policy_id": candidate.policy_id,
        "checkpoint_revision": candidate.checkpoint_revision,
        "processor_revision": candidate.processor_revision,
        "inference_seed": candidate.inference_seed,
        "sampling_config": dict(candidate.sampling_config),
        "normalized_action_chunk": candidate.normalized_action_chunk.tolist(),
        "native_action_chunk": candidate.native_action_chunk.tolist(),
        "action_mask": (
            None if candidate.action_mask is None else candidate.action_mask.tolist()
        ),
        "content_sha256": candidate.content_sha256,
        "generation_latency_ms": candidate.generation_latency_ms,
        "source_kind": candidate.source_kind,
        "disposition": candidate.disposition.value,
    }


def frozen_policy_identity() -> dict[str, Any]:
    """Return the exact immutable policy/processor/action contract identity."""
    return {
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "processor_revision": CHECKPOINT_REVISION,
        "lerobot_version": LEROBOT_VERSION,
        "lerobot_commit": LEROBOT_COMMIT,
        "action_execution_horizon": 7,
        "control_mode": "relative",
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


__all__ = [
    "FINAL_SEEDS",
    "array_sha256",
    "batch_observation",
    "canonical_sha256",
    "compare_rendered_observations",
    "current_observation",
    "frozen_policy_identity",
    "generate_policy_candidate",
    "git_output",
    "load_runtime_stack",
    "make_libero_environment",
    "observation_sha256",
    "output_root",
    "policy_batch",
    "privileged_state",
    "proprioception_sha256",
    "public_candidate",
    "read_json",
    "read_yaml",
    "rendered_observation_sha256",
    "resolve_repo_path",
    "runtime_identity",
    "select_primary_action",
    "sha256_path",
    "step_native",
    "unbatch_observation",
    "validate_execution_checkout",
    "validate_no_final_seed",
    "write_json",
]
