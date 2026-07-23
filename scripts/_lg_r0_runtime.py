"""Shared runtime helpers for LG-R0 executable audits.

This module is imported only inside the isolated LeRobot 0.6 environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any

from latentguard.adapters.vla_jepa.model_loader import (
    LoadedVLAJepa,
    load_local_vla_jepa,
)


def repo_root() -> Path:
    """Return the repository root containing this script."""

    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, reviewable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_path(path: Path) -> str:
    """Hash one file in streaming mode."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_paths() -> tuple[Path, Path, Path]:
    """Resolve exact local snapshots from one explicit environment root."""

    configured = os.environ.get("LG_R0_MODEL_ROOT")
    root = (
        Path(configured)
        if configured
        else Path.home() / ".cache" / "latentguard-lg-r0" / "models"
    )
    return (
        root / "vla-jepa-libero-735d9f6",
        root / "qwen3-vl-2b-8964489",
        root / "vjepa2-vitl-b3c1679",
    )


def libero_asset_path() -> Path:
    """Resolve the explicitly pinned LIBERO asset snapshot."""

    configured = os.environ.get("LG_R0_LIBERO_ASSET_ROOT")
    if not configured:
        raise RuntimeError("LG_R0_LIBERO_ASSET_ROOT must name the pinned asset root")
    path = Path(configured)
    required = (
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "stable_hope_objects",
    )
    missing = [name for name in required if not (path / name).is_dir()]
    if missing:
        raise RuntimeError(f"LIBERO asset snapshot is incomplete: {missing}")
    return path.resolve()


def configure_libero_assets() -> Path:
    """Bind hf-libero to the explicit pinned snapshot without an online fallback."""

    import libero.libero

    path = libero_asset_path()
    libero.libero._assets_path_cache = str(path)
    return path


def output_root() -> Path:
    """Resolve the ignored runtime-output root."""

    configured = os.environ.get("LG_R0_OUTPUT_ROOT")
    return Path(configured) if configured else repo_root() / "outputs" / "lg_r0"


def load_stack(*, device: str = "cuda") -> LoadedVLAJepa:
    """Load the frozen stack locally with strict state-dict checking."""

    checkpoint, qwen, vjepa = model_paths()
    manifest = read_json(
        repo_root() / "artifacts" / "lg_r0" / "base_model_manifest.json"
    )
    return load_local_vla_jepa(
        checkpoint,
        qwen,
        vjepa,
        asset_manifest=manifest,
        device=device,
    )


def load_processors(stack: LoadedVLAJepa) -> tuple[Any, Any]:
    """Load official processor files with only the execution device overridden."""

    from lerobot.policies.factory import make_pre_post_processors

    return make_pre_post_processors(
        policy_cfg=stack.config,
        pretrained_path=str(stack.checkpoint_path),
        preprocessor_overrides={
            "device_processor": {"device": str(stack.config.device)}
        },
    )


def fixture_raw_observation(
    batch_size: int, image_size: int, seed: int
) -> dict[str, Any]:
    """Create a legal deterministic LIBERO-shaped processor fixture."""

    import numpy as np

    generator = np.random.default_rng(seed)
    images = {
        "image": generator.integers(
            0,
            256,
            size=(batch_size, image_size, image_size, 3),
            dtype=np.uint8,
        ),
        "image2": generator.integers(
            0,
            256,
            size=(batch_size, image_size, image_size, 3),
            dtype=np.uint8,
        ),
    }
    return {
        "pixels": images,
        "robot_state": {
            "eef": {
                "pos": np.zeros((batch_size, 3), dtype=np.float64),
                "quat": np.tile(
                    np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
                    (batch_size, 1),
                ),
                "mat": np.tile(
                    np.eye(3, dtype=np.float64)[None, :, :],
                    (batch_size, 1, 1),
                ),
            },
            "gripper": {
                "qpos": np.zeros((batch_size, 2), dtype=np.float64),
                "qvel": np.zeros((batch_size, 2), dtype=np.float64),
            },
            "joints": {
                "pos": np.zeros((batch_size, 7), dtype=np.float64),
                "vel": np.zeros((batch_size, 7), dtype=np.float64),
            },
        },
    }


def prepare_fixture(
    raw_observation: dict[str, Any],
    instruction: str,
    env_preprocessor: Any,
    preprocessor: Any,
) -> dict[str, Any]:
    """Apply the official environment and policy processor chain."""

    from lerobot.envs import preprocess_observation

    batch_size = int(raw_observation["pixels"]["image"].shape[0])
    observation = preprocess_observation(raw_observation)
    observation["task"] = [instruction] * batch_size
    return dict(preprocessor(env_preprocessor(observation)))


@dataclass
class PolicyInstrumentation:
    """Runtime-only counters and hashes for native action chunks."""

    policy_queries: int = 0
    optimizer_steps: int = 0
    backward_calls: int = 0
    latencies_seconds: list[float] = field(default_factory=list)
    chunk_sha256: list[str] = field(default_factory=list)
    first_chunks: list[list[list[float]]] = field(default_factory=list)
    chunks: list[list[list[float]]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return compact aggregate instrumentation."""

        import numpy as np

        values = np.asarray(self.latencies_seconds, dtype=np.float64)
        return {
            "policy_queries": self.policy_queries,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "mean_query_latency_seconds": (
                float(values.mean()) if values.size else None
            ),
            "p95_query_latency_seconds": (
                float(np.quantile(values, 0.95)) if values.size else None
            ),
            "queries_per_second": (
                float(1.0 / values.mean())
                if values.size and values.mean() > 0
                else None
            ),
            "chunk_sha256": list(self.chunk_sha256),
            "first_chunks": list(self.first_chunks),
        }


def instrument_policy(policy: Any) -> PolicyInstrumentation:
    """Wrap native chunk inference without changing its inputs or outputs."""

    instrumentation = PolicyInstrumentation()
    original = policy.predict_action_chunk

    def wrapped(self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        import numpy as np
        import torch

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        started = time.perf_counter()
        output = original(batch, *args, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        instrumentation.latencies_seconds.append(time.perf_counter() - started)
        instrumentation.policy_queries += 1
        array = output.detach().to("cpu", dtype=torch.float32).numpy()
        instrumentation.chunk_sha256.append(
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        )
        instrumentation.chunks.append(array.tolist())
        if len(instrumentation.first_chunks) < 2:
            instrumentation.first_chunks.append(array.tolist())
        return output

    policy.predict_action_chunk = MethodType(wrapped, policy)
    return instrumentation


def cuda_memory() -> dict[str, Any]:
    """Return current and peak CUDA allocation counters."""

    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
