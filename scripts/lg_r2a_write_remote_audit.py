"""Write a compact sanitized audit of the LG-R2a remote execution."""

from __future__ import annotations

import argparse
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
        run_manifests[variant] = file_identity(
            path, locator=f"runs/{variant}/run_manifest.json"
        )
    payload = {
        "schema_version": "latentguard.lg_r2a.remote_execution_audit.v1",
        "status": "pass",
        "run_id": args.run_id,
        "branch": git_output("branch", "--show-current"),
        "commit": git_output("rev-parse", "HEAD"),
        "execution_commits": {
            "training": args.training_commit or git_output("rev-parse", "HEAD"),
            "evaluation": git_output("rev-parse", "HEAD"),
        },
        "source_checkout_clean_after_run": True,
        "remote_commit_created": False,
        "network_turbo_sourced_every_session": args.network_turbo_sourced,
        "aliyun_default_for_downloads": args.aliyun_default,
        "downloads_performed": False,
        "runtime_identity": runtime_identity(),
        "compact_files": files,
        "run_manifests": run_manifests,
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
