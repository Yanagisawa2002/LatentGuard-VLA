"""Evaluate one frozen WM-v0 checkpoint on identical held-out candidate groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latentguard.world_model.baselines import (
    accepted_direct_verifier_config,
    build_direct_verifier_success_baseline,
)
from latentguard.world_model.evaluation import (
    CandidatePrediction,
    binary_auprc,
    binary_auroc,
    calibration_error,
    candidate_ranking_metrics,
    prediction_metrics,
)
from latentguard.world_model.features import load_feature_sample
from latentguard.world_model.model import (
    ActionConditionedLatentWorldModel,
    OutcomeOnlyWorldModel,
    WorldModelConfig,
)


def main() -> int:
    """Compute future, event, outcome, and ranking metrics without fitting."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("wm_v0", "outcome_only", "direct_verifier"),
        required=True,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--non-formal-smoke", action="store_true")
    args = parser.parse_args()
    train: dict[str, Any] = json.loads(args.train_config.read_text("utf-8"))
    evaluate: dict[str, Any] = json.loads(args.eval_config.read_text("utf-8"))
    index: Any = json.loads((args.features / "index.json").read_text("utf-8"))
    records = [
        record
        for record in index["records"]
        if isinstance(record, dict) and record.get("split") == args.split
    ]
    if args.limit_samples:
        records = records[: args.limit_samples]
    if not records:
        raise ValueError("evaluation split has no feature samples")
    samples = [
        load_feature_sample(args.features / f"{record['sample_id']}.npz")
        for record in records
    ]
    config = WorldModelConfig(
        latent_dimension=int(train["latent_dimension"]),
        proprio_dimension=int(train["proprio_dimension"]),
        action_dimension=int(train["action_dimension"]),
        action_horizon=int(train["action_horizon"]),
        prediction_horizon=int(train["prediction_horizon"]),
        view_count=int(train["view_count"]),
        event_count=int(samples[0].event_labels.shape[1]),
        model_dimension=int(train["model_dimension"]),
        attention_heads=int(train["attention_heads"]),
        transformer_layers=int(train["transformer_layers"]),
        dropout=float(train["dropout"]),
    )
    device = torch.device(args.device)
    checkpoint: Any = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    sigmoid = torch.sigmoid
    proprio = torch.from_numpy(np.stack([sample.proprio for sample in samples])).to(
        device
    )
    actions = torch.from_numpy(np.stack([sample.actions for sample in samples])).to(
        device
    )
    action_mask = torch.from_numpy(
        np.stack([sample.action_mask for sample in samples])
    ).to(device)
    if args.model == "direct_verifier":
        direct = build_direct_verifier_success_baseline(
            accepted_direct_verifier_config(), seed=int(train["seed"])
        ).to(device)
        direct.load_state_dict(checkpoint["model"])
        direct.eval()
        with torch.inference_mode():
            success = sigmoid(direct(proprio, actions, action_mask)).cpu().numpy()
        detailed_metrics: dict[str, object] | None = None
    else:
        model_type = (
            ActionConditionedLatentWorldModel
            if args.model == "wm_v0"
            else OutcomeOnlyWorldModel
        )
        model = model_type(config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with torch.inference_mode():
            output = model(
                torch.from_numpy(
                    np.stack([sample.current_latents for sample in samples])
                ).to(device),
                proprio,
                actions,
                action_mask,
            )
        success = sigmoid(output.success_logit).cpu().numpy()
        detailed_metrics = prediction_metrics(
            predicted_latents=output.future_latents.cpu().numpy(),
            target_latents=np.stack([sample.future_latents for sample in samples]),
            predicted_progress=output.future_progress.cpu().numpy(),
            target_progress=np.stack([sample.future_progress for sample in samples]),
            progress_mask=np.stack([sample.progress_mask for sample in samples]),
            event_probabilities=sigmoid(output.event_logits).cpu().numpy(),
            event_labels=np.stack([sample.event_labels for sample in samples]),
            event_mask=np.stack([sample.event_mask for sample in samples]),
            event_names=samples[0].event_names,
            success_probabilities=success,
            terminal_success=np.asarray(
                [sample.terminal_success for sample in samples], dtype=np.float64
            ),
            terminal_success_mask=np.asarray(
                [sample.terminal_success_mask for sample in samples], dtype=np.bool_
            ),
            calibration_bins=int(evaluate["calibration_bins"]),
        )
    terminal = np.asarray(
        [sample.terminal_success for sample in samples], dtype=np.float64
    )
    terminal_mask = np.asarray(
        [sample.terminal_success_mask for sample in samples], dtype=np.bool_
    )
    if detailed_metrics is None:
        if bool(terminal_mask.any()):
            terminal_metrics: dict[str, object] = {
                "auprc": binary_auprc(terminal[terminal_mask], success[terminal_mask]),
                "auroc": binary_auroc(terminal[terminal_mask], success[terminal_mask]),
                "brier": float(
                    np.mean((success[terminal_mask] - terminal[terminal_mask]) ** 2)
                ),
                "ece": calibration_error(
                    terminal[terminal_mask],
                    success[terminal_mask],
                    bins=int(evaluate["calibration_bins"]),
                ),
                "support": int(terminal_mask.sum()),
            }
        else:
            terminal_metrics = {
                "auprc": None,
                "auroc": None,
                "brier": None,
                "ece": None,
                "support": 0,
            }
        metrics: dict[str, object] = {
            "future_latent": "not_applicable_for_direct_verifier",
            "terminal_outcome": terminal_metrics,
        }
    else:
        metrics = detailed_metrics
    result: dict[str, object] = {
        "claim_status": "non_formal_smoke"
        if args.non_formal_smoke
        else "formal_gate_required",
        "metrics": metrics,
        "model": args.model,
        "ranking": _ranking(samples, success),
        "sample_count": len(samples),
        "schema_version": "1.0",
        "split": args.split,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _ranking(samples: list[Any], success: np.ndarray) -> dict[str, object]:
    groups: dict[str, list[tuple[Any, float]]] = {}
    for sample, probability in zip(samples, success, strict=True):
        groups.setdefault(sample.anchor_id, []).append((sample, float(probability)))
    complete = [
        group
        for group in groups.values()
        if all(item.terminal_success_mask for item, _ in group)
    ]
    if not complete:
        return {"status": "no_complete_terminal_candidate_groups"}
    return candidate_ranking_metrics(
        CandidatePrediction(
            group_id=item.anchor_id,
            candidate_id=item.candidate_id,
            policy_source=item.policy_source,
            success_probability=probability,
            terminal_success=bool(item.terminal_success),
        )
        for group in complete
        for item, probability in group
    )


if __name__ == "__main__":
    raise SystemExit(main())
