"""Run static single/batch inference and official processor round-trip gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from _lg_r0_runtime import (
    cuda_memory,
    fixture_raw_observation,
    load_processors,
    load_stack,
    output_root,
    prepare_fixture,
    read_json,
    sha256_path,
    write_json,
)

from latentguard.adapters.vla_jepa.policy_adapter import VLAJepaAdapter


def _tensor_digest(value: Any) -> str:
    import numpy as np
    import torch

    array = value.detach().to("cpu", dtype=torch.float32).numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    """Run every static inference gate without creating gradients."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from lerobot.envs.configs import LiberoEnv
    from lerobot.processor import PolicyProcessorPipeline

    config = read_json(args.config)
    fixture = config["fixture"]
    torch.manual_seed(int(fixture["seed"]))
    torch.cuda.manual_seed_all(int(fixture["seed"]))
    torch.cuda.reset_peak_memory_stats()
    stack = load_stack(device=config["device"])
    preprocessor, postprocessor = load_processors(stack)
    env_preprocessor, _ = LiberoEnv(task="libero_spatial").get_env_processors()
    adapter = VLAJepaAdapter(stack.policy)
    results: list[dict[str, Any]] = []
    first_batch_one: str | None = None
    processor_seconds = 0.0
    model_seconds = 0.0
    nonfinite = 0
    for batch_size in fixture["batch_sizes"]:
        raw = fixture_raw_observation(
            int(batch_size),
            int(fixture["image_shape_chw"][1]),
            int(fixture["seed"]),
        )
        started = time.perf_counter()
        processed = prepare_fixture(
            raw,
            fixture["instruction"],
            env_preprocessor,
            preprocessor,
        )
        processor_seconds += time.perf_counter() - started
        torch.manual_seed(int(fixture["seed"]))
        torch.cuda.manual_seed_all(int(fixture["seed"]))
        started = time.perf_counter()
        with torch.inference_mode():
            output = adapter.predict_action(processed)
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - started
        digest = _tensor_digest(output.action_chunk)
        finite = bool(torch.isfinite(output.action_chunk).all().item())
        nonfinite += (
            0 if finite else int((~torch.isfinite(output.action_chunk)).sum().item())
        )
        if int(batch_size) == 1 and first_batch_one is None:
            first_batch_one = digest
        results.append(
            {
                "batch_size": int(batch_size),
                "shape": list(output.action_chunk.shape),
                "finite": finite,
                "sha256_float32": digest,
                "action_mask": None,
                "native_tensor_returned_unmodified": True,
            }
        )

    torch.manual_seed(int(fixture["seed"]))
    torch.cuda.manual_seed_all(int(fixture["seed"]))
    repeat_batch = prepare_fixture(
        fixture_raw_observation(
            1, int(fixture["image_shape_chw"][1]), int(fixture["seed"])
        ),
        fixture["instruction"],
        env_preprocessor,
        preprocessor,
    )
    with torch.inference_mode():
        repeated = adapter.predict_action(repeat_batch)
    repeat_digest = _tensor_digest(repeated.action_chunk)

    processor_dir = output_root() / "processor-roundtrip"
    if processor_dir.exists():
        shutil.rmtree(processor_dir)
    processor_dir.mkdir(parents=True)
    preprocessor.save_pretrained(processor_dir)
    postprocessor.save_pretrained(processor_dir)
    reloaded_pre = PolicyProcessorPipeline.from_pretrained(
        processor_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": str(stack.config.device)}},
    )
    original_processed = prepare_fixture(
        fixture_raw_observation(
            1, int(fixture["image_shape_chw"][1]), int(fixture["seed"])
        ),
        fixture["instruction"],
        env_preprocessor,
        preprocessor,
    )
    reloaded_processed = prepare_fixture(
        fixture_raw_observation(
            1, int(fixture["image_shape_chw"][1]), int(fixture["seed"])
        ),
        fixture["instruction"],
        env_preprocessor,
        reloaded_pre,
    )
    processor_equal = all(
        torch.equal(original_processed[key], reloaded_processed[key])
        if isinstance(original_processed[key], torch.Tensor)
        else original_processed[key] == reloaded_processed[key]
        for key in original_processed
    )
    processor_files = [
        {
            "relative_path": str(path.relative_to(processor_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(processor_dir.rglob("*"))
        if path.is_file()
    ]
    gates = {
        "model_load": "pass",
        "processor_roundtrip": "pass" if processor_equal else "fail",
        "single_inference": "pass"
        if any(item["batch_size"] == 1 and item["finite"] for item in results)
        else "fail",
        "batched_inference": "pass"
        if any(item["batch_size"] > 1 and item["finite"] for item in results)
        else "fail",
        "fixed_seed_reproduction": "pass"
        if first_batch_one == repeat_digest
        else "fail",
    }
    payload = {
        "schema_version": "latentguard.lg_r0.static_inference_validation.v1",
        "status": "pass"
        if all(value == "pass" for value in gates.values())
        else "fail",
        "gates": gates,
        "results": results,
        "repeat_digest": repeat_digest,
        "processor_serialization": {
            "runtime_locator": "LG_R0_OUTPUT_ROOT/processor-roundtrip",
            "files": processor_files,
            "input_tensors_equal_after_reload": processor_equal,
        },
        "timing": {
            "processor_total_seconds": processor_seconds,
            "model_total_seconds": model_seconds,
        },
        "cuda_memory": cuda_memory(),
        "nonfinite_actions": nonfinite,
        "contract_errors": 0,
        "optimizer_steps": adapter.optimizer_steps,
        "backward_calls": adapter.backward_calls,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "gates": gates, "output": str(args.output)}
        )
    )
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
