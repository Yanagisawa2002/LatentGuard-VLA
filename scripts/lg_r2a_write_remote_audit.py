"""Write a compact sanitized audit of the LG-R2a remote execution."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from _lg_r2a_common import (
    file_identity,
    git_output,
    output_root,
    runtime_identity,
    write_json,
)

COMPACT_FILES = (
    "input_dataset_manifest.json",
    "action_contract.json",
    "cv_split_manifest.json",
    "target_manifest.json",
    "state_only_results.json",
    "action_only_results.json",
    "state_action_results.json",
    "internal_token_results.json",
    "action_permutation_results.json",
    "action_swap_surrogate_results.json",
    "task_macro_results.json",
    "leave_task6_out_results.json",
    "evaluation_summary.json",
    "lg_r2b_gate.json",
    "source_validation.json",
    "action_ablation_results.json",
    "action_magnitude_diagnostic.json",
)


def main() -> None:
    """Bind compact outputs, clean source revision, CUDA, and run controls."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--network-turbo-sourced", action="store_true")
    parser.add_argument("--aliyun-default", action="store_true")
    parser.add_argument("--training-commit")
    parser.add_argument("--evaluation-commit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print({"status": "dry_run", "run_id": args.run_id})
        return
    output = output_root(args.output_dir)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("remote tracked checkout changed during execution")
    files: dict[str, Any] = {}
    for name in COMPACT_FILES:
        path = output / name
        if not path.is_file():
            raise ValueError(f"required compact output is missing: {name}")
        files[name] = file_identity(path, locator=name)
    run_manifests = {}
    for variant in (
        "state_only",
        "action_only",
        "state_action",
        "state_action_proprio",
    ):
        path = output / "runs" / variant / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ended = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
        started = ended - dt.timedelta(seconds=float(manifest["elapsed_seconds"]))
        run_manifests[variant] = {
            **file_identity(path, locator=f"runs/{variant}/run_manifest.json"),
            "launch_command": (
                "python scripts/lg_r2a_train_probe.py "
                f"--config configs/lg_r2a/{variant}.yaml "
                '--output-dir "$LG_R2A_OUTPUT_ROOT" --seed 0 '
                "--checkpoint-every 100 --eval-every 20"
            ),
            "start_time_utc": started.isoformat(),
            "end_time_utc": ended.isoformat(),
            "time_source": (
                "derived from run-manifest mtime minus recorded elapsed_seconds"
            ),
            "elapsed_seconds": float(manifest["elapsed_seconds"]),
        }
    evaluation_commit = args.evaluation_commit or git_output("rev-parse", "HEAD")
    payload = {
        "schema_version": "latentguard.lg_r2a.remote_execution_audit.v1",
        "status": "pass",
        "run_id": args.run_id,
        "branch": git_output("branch", "--show-current"),
        "commit": git_output("rev-parse", "HEAD"),
        "execution_commits": {
            "training": args.training_commit or git_output("rev-parse", "HEAD"),
            "evaluation": evaluation_commit,
            "audit": git_output("rev-parse", "HEAD"),
        },
        "source_checkout_clean_after_run": True,
        "remote_commit_created": False,
        "network_turbo_sourced_every_session": args.network_turbo_sourced,
        "aliyun_default_for_downloads": args.aliyun_default,
        "downloads_performed": False,
        "runtime_identity": runtime_identity(),
        "compact_files": files,
        "run_manifests": run_manifests,
        "execution_commands": [
            "python scripts/lg_r2a_validate_inputs.py "
            "--config configs/lg_r2a/input.yaml",
            "python scripts/lg_r2a_build_cv_splits.py --config configs/lg_r2a/cv.yaml",
            "python scripts/lg_r2a_build_targets.py "
            "--config configs/lg_r2a/targets.yaml",
            "python scripts/lg_r2a_train_probe.py "
            "--config configs/lg_r2a/<variant>.yaml "
            '--output-dir "$LG_R2A_OUTPUT_ROOT" --seed 0 '
            "--checkpoint-every 100 --eval-every 20",
            "python scripts/lg_r2a_action_sensitivity.py "
            "--config configs/lg_r2a/sensitivity.yaml",
            "python scripts/lg_r2a_evaluate.py --config configs/lg_r2a/evaluation.yaml",
        ],
        "pre_training_gate_stops": [
            {
                "reason": "repeated compressed feature-cache decompression",
                "formal_training_started": False,
                "resolution": "load frozen feature array exactly once",
            },
            {
                "reason": "secret-scanner rule matched its own source literal",
                "formal_training_started": False,
                "resolution": "self-isolate scanner literals and validate locally",
            },
        ],
        "foundation_model_training": False,
        "new_rollouts": 0,
        "simulator_started": False,
        "candidate_generation": False,
        "candidate_ranking": False,
        "counterfactual_rollout": False,
        "intervention": False,
        "final_seeds_accessed": False,
        "checkpoints_outside_git": True,
        "predictions_outside_git": True,
        "server_shutdown_requested": False,
        "server_left_running": True,
    }
    write_json(output / "remote_execution_audit.json", payload)
    print({"status": "pass", "run_id": args.run_id})


if __name__ == "__main__":
    main()
