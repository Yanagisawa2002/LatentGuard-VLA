"""Strictly load and structurally audit the frozen VLA-JEPA checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _lg_r0_runtime import cuda_memory, load_stack, write_json

from latentguard.adapters.vla_jepa.constants import (
    QWEN_ID,
    QWEN_REVISION,
    VJEPA_ID,
    VJEPA_REVISION,
)


def main() -> None:
    """Load the model and record parameter/module structure."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    import torch

    torch.cuda.reset_peak_memory_stats()
    stack = load_stack()
    module_counts: dict[str, int] = defaultdict(int)
    module_trainable: dict[str, int] = defaultdict(int)
    total = 0
    trainable = 0
    for name, parameter in stack.policy.named_parameters():
        root = "other"
        for prefix in (
            "model.qwen",
            "model.action_model",
            "model.video_encoder",
            "model.video_predictor",
        ):
            if name.startswith(prefix):
                root = prefix
                break
        count = parameter.numel()
        module_counts[root] += count
        total += count
        if parameter.requires_grad:
            module_trainable[root] += count
            trainable += count
    payload: dict[str, Any] = {
        "schema_version": "latentguard.lg_r0.model_structure.v1",
        "status": "pass",
        "strict_load": False,
        "fail_closed_load": True,
        "missing_keys": list(stack.missing_keys),
        "unexpected_keys": list(stack.unexpected_noncritical_keys),
        "noncritical_key_explanation": {
            "model.qwen.model.model.language_model.embed_tokens.weight": (
                "checkpoint duplicates the exactly equal lm_head tensor; the "
                "runtime model ties both names to one storage and safetensors "
                "reports the removed duplicate as unexpected"
            )
        },
        "checkpoint_revision": stack.checkpoint_revision,
        "config": {
            "action_dimension": stack.config.action_dim,
            "state_dimension": stack.config.state_dim,
            "action_horizon": stack.config.chunk_size,
            "executed_action_steps": stack.config.n_action_steps,
            "world_model_enabled": stack.config.enable_world_model,
            "video_frames": stack.config.num_video_frames,
            "qwen_binding": {"id": QWEN_ID, "revision": QWEN_REVISION},
            "vjepa_binding": {"id": VJEPA_ID, "revision": VJEPA_REVISION},
            "torch_dtype": stack.config.torch_dtype,
        },
        "parameters": {
            "total": total,
            "trainable_flags_in_checkpoint": trainable,
            "frozen_flags_in_checkpoint": total - trainable,
            "by_top_level_module": dict(sorted(module_counts.items())),
            "trainable_by_top_level_module": dict(sorted(module_trainable.items())),
            "note": (
                "requires_grad flags are architecture metadata; no training occurred"
            ),
        },
        "cuda_memory_after_load": cuda_memory(),
        "optimizer_steps": 0,
        "backward_calls": 0,
        "audit_only": args.audit_only,
    }
    write_json(args.output, payload)
    print(
        json.dumps({"status": "pass", "parameters": total, "output": str(args.output)})
    )


if __name__ == "__main__":
    main()
