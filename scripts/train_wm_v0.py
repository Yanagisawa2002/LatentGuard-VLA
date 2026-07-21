"""Train WM-v0 or its matched outcome-only ablation on frozen features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latentguard.world_model.model import WorldModelConfig
from latentguard.world_model.training import (
    run_config_from_mapping,
    train_world_model,
)


def main() -> int:
    """Run configuration validation, dry run, short smoke, or full gated training."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=("wm_v0", "outcome_only", "direct_verifier"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-insufficient-data-smoke", action="store_true")
    args = parser.parse_args()
    config: dict[str, Any] = json.loads(args.config.read_text("utf-8"))
    manifest: Any = json.loads(args.dataset_manifest.read_text("utf-8"))
    gate = (
        manifest.get("formal_training_gate", {}) if isinstance(manifest, dict) else {}
    )
    authorized = isinstance(gate, dict) and gate.get("authorized") is True
    if not authorized and not (
        args.allow_insufficient_data_smoke and args.max_steps is not None
    ):
        raise RuntimeError(
            "formal data gate failed; use an explicitly bounded insufficient-data smoke"
        )
    if args.model:
        config["model"] = args.model
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output:
        config["output_directory"] = str(args.output)
    model_config = WorldModelConfig(
        latent_dimension=int(config["latent_dimension"]),
        proprio_dimension=int(config["proprio_dimension"]),
        action_dimension=int(config["action_dimension"]),
        action_horizon=int(config["action_horizon"]),
        prediction_horizon=int(config["prediction_horizon"]),
        view_count=int(config["view_count"]),
        event_count=8,
        model_dimension=int(config["model_dimension"]),
        attention_heads=int(config["attention_heads"]),
        transformer_layers=int(config["transformer_layers"]),
        dropout=float(config["dropout"]),
    )
    summary = train_world_model(
        model_config=model_config,
        run_config=run_config_from_mapping(config),
        feature_directory=args.features,
        max_steps=args.max_steps,
        limit_samples=args.limit_samples,
        resume=args.resume,
        dry_run=args.dry_run,
        device_name=args.device,
    )
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
