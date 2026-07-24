"""Run LG-RB0 RoboLab phases with one Isaac Sim process per recording."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from latentguard.adapters.robolab.orchestration import (
    merge_branch_results,
    merge_faithful_results,
    merge_isolation_results,
    merge_prefix_results,
    merge_recording_manifests,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def _recording_specs(protocol: Mapping[str, Any]) -> list[tuple[str, int, str]]:
    recording = protocol["recording"]
    specs: list[tuple[str, int, str]] = []
    for query_value in recording["task_queries"]:
        query = str(query_value)
        for seed_value in recording["seeds"]:
            seed = int(seed_value)
            specs.append((query, seed, f"{_safe_id(query)}-seed-{seed}"))
    expected = int(recording["expected_episode_count"])
    if len(specs) != expected:
        raise ValueError("protocol recording grid differs from expected episode count")
    return specs


def _child_protocol(
    protocol: Mapping[str, Any],
    *,
    query: str,
    seed: int,
) -> dict[str, Any]:
    child = copy.deepcopy(dict(protocol))
    child["recording"]["task_queries"] = [query]
    child["recording"]["seeds"] = [seed]
    child["recording"]["expected_episode_count"] = 1
    return child


def _run_child(
    *,
    runner: Path,
    phase: str,
    protocol_path: Path,
    run_root: Path,
    recording_root: Path,
    artifact_dir: Path,
    expected_commit: str,
    device: str,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        str(runner),
        "--phase",
        phase,
        "--protocol",
        str(protocol_path),
        "--run-root",
        str(run_root),
        "--recording-root",
        str(recording_root),
        "--artifact-dir",
        str(artifact_dir),
        "--expected-commit",
        expected_commit,
        "--expected-robolab-base",
        str(
            yaml.safe_load(protocol_path.read_text(encoding="utf-8"))["external_stack"][
                "commit"
            ]
        ),
        "--expected-robolab-tree",
        str(
            yaml.safe_load(protocol_path.read_text(encoding="utf-8"))["upstream_patch"][
                "patched_tree_digest"
            ]
        ),
        "--device",
        device,
        "--headless",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{phase} shard failed with exit {completed.returncode}: {log_path}"
        )


def _prepare_shard(
    *,
    protocol: Mapping[str, Any],
    query: str,
    seed: int,
    recording_id: str,
    run_root: Path,
    phase: str,
) -> tuple[Path, Path, Path]:
    shard_root = run_root / "orchestration"
    protocol_path = shard_root / "protocols" / f"{recording_id}.yaml"
    artifact_dir = shard_root / "artifacts" / phase / recording_id
    log_path = shard_root / "logs" / phase / f"{recording_id}.log"
    if artifact_dir.exists() or log_path.exists():
        raise RuntimeError(f"refusing to overwrite existing shard: {recording_id}")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        yaml.safe_dump(
            _child_protocol(protocol, query=query, seed=seed),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return protocol_path, artifact_dir, log_path


def _record(
    *,
    protocol: Mapping[str, Any],
    runner: Path,
    run_root: Path,
    recording_root: Path,
    artifact_dir: Path,
    expected_commit: str,
    device: str,
) -> None:
    manifests: list[dict[str, Any]] = []
    specs = _recording_specs(protocol)
    for query, seed, recording_id in specs:
        child_protocol, child_artifacts, log_path = _prepare_shard(
            protocol=protocol,
            query=query,
            seed=seed,
            recording_id=recording_id,
            run_root=run_root,
            phase="record",
        )
        _run_child(
            runner=runner,
            phase="record",
            protocol_path=child_protocol,
            run_root=run_root,
            recording_root=recording_root,
            artifact_dir=child_artifacts,
            expected_commit=expected_commit,
            device=device,
            log_path=log_path,
        )
        manifests.append(_read(child_artifacts / "recording_manifest.json"))
    merged = merge_recording_manifests(
        manifests,
        expected_episode_count=len(specs),
    )
    _write(artifact_dir / "recording_manifest.json", merged)
    _write(
        run_root / "recording_index.json",
        {
            "schema_version": "lg_rb0_recording_index_v1",
            "recording_ids": [item["recording_id"] for item in merged["recordings"]],
            "runtime_process_isolation": "one_recording_per_isaac_sim_process",
        },
    )


def _single_recording_manifest(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "lg_rb0_recording_manifest_v1",
        "status": "pass",
        "controller_kind": "deterministic_fixed_mechanics_probe",
        "policy_or_candidate_source": False,
        "episode_count": 1,
        "valid_episode_count": 1,
        "recordings": [dict(item)],
        "raw_recordings_in_git": False,
        "training_performed": False,
        "final_seeds_accessed": False,
    }


def _faithful(
    *,
    protocol: Mapping[str, Any],
    runner: Path,
    run_root: Path,
    recording_root: Path,
    artifact_dir: Path,
    expected_commit: str,
    device: str,
) -> None:
    manifest = _read(artifact_dir / "recording_manifest.json")
    if manifest.get("status") != "pass":
        raise RuntimeError("faithful replay requires a passing recording manifest")
    by_id = {str(item["recording_id"]): item for item in manifest.get("recordings", [])}
    results: list[dict[str, Any]] = []
    for query, seed, recording_id in _recording_specs(protocol):
        item = by_id.get(recording_id)
        if item is None:
            raise RuntimeError(f"recording manifest missing {recording_id}")
        child_protocol, child_artifacts, log_path = _prepare_shard(
            protocol=protocol,
            query=query,
            seed=seed,
            recording_id=recording_id,
            run_root=run_root,
            phase="faithful",
        )
        _write(
            child_artifacts / "recording_manifest.json",
            _single_recording_manifest(item),
        )
        _run_child(
            runner=runner,
            phase="faithful",
            protocol_path=child_protocol,
            run_root=run_root,
            recording_root=recording_root,
            artifact_dir=child_artifacts,
            expected_commit=expected_commit,
            device=device,
            log_path=log_path,
        )
        results.append(_read(child_artifacts / "faithful_replay_validation.json"))
    merged = merge_faithful_results(
        results,
        expected_episode_count=len(by_id),
    )
    _write(artifact_dir / "faithful_replay_validation.json", merged)
    canonicalization_shards = [
        _read(
            run_root
            / "orchestration"
            / "artifacts"
            / "faithful"
            / recording_id
            / "state_schema_canonicalization.json"
        )
        for _, _, recording_id in _recording_specs(protocol)
    ]
    _write(
        artifact_dir / "state_schema_canonicalization.json",
        {
            "schema_version": "lg_rb01_state_schema_canonicalization_v1",
            "status": (
                "pass"
                if all(
                    value.get("status") == "pass" for value in canonicalization_shards
                )
                else "fail"
            ),
            "symmetric": True,
            "numeric_state_changed": False,
            "allowed_optional_empty_namespaces": list(
                protocol["state_schema"]["allowed_optional_empty_namespaces"]
            ),
            "comparison_count": sum(
                int(value.get("comparison_count", 0))
                for value in canonicalization_shards
            ),
            "shards": canonicalization_shards,
        },
    )
    if merged["status"] != "pass":
        _write_not_run_takeover_outputs(
            artifact_dir,
            reason="blocked_by_faithful_replay_gate",
            include_prefix=True,
        )


def _not_run_result(schema_version: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "status": "not_run",
        "reason": reason,
        "mismatch_count": None,
        "semantic_coverage_complete": None,
        "details": [],
    }


def _write_not_run_takeover_outputs(
    artifact_dir: Path,
    *,
    reason: str,
    include_prefix: bool,
) -> None:
    if include_prefix:
        _write(
            artifact_dir / "prefix_replay_validation.json",
            _not_run_result("lg_rb01_prefix_replay_validation_v1", reason),
        )
    _write(
        artifact_dir / "branch_determinism_validation.json",
        _not_run_result("lg_rb01_branch_determinism_validation_v1", reason),
    )
    _write(
        artifact_dir / "branch_isolation_validation.json",
        _not_run_result("lg_rb01_branch_isolation_validation_v1", reason),
    )


def _takeover(
    *,
    protocol: Mapping[str, Any],
    runner: Path,
    run_root: Path,
    recording_root: Path,
    artifact_dir: Path,
    expected_commit: str,
    device: str,
) -> None:
    faithful = _read(artifact_dir / "faithful_replay_validation.json")
    if faithful.get("status") != "pass":
        raise RuntimeError("takeover is prohibited after faithful replay failure")
    manifest = _read(artifact_dir / "recording_manifest.json")
    by_id = {str(item["recording_id"]): item for item in manifest.get("recordings", [])}

    def run_gate(
        *,
        phase: str,
        output_name: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for query, seed, recording_id in _recording_specs(protocol):
            item = by_id.get(recording_id)
            if item is None:
                raise RuntimeError(f"recording manifest missing {recording_id}")
            child_protocol, child_artifacts, log_path = _prepare_shard(
                protocol=protocol,
                query=query,
                seed=seed,
                recording_id=recording_id,
                run_root=run_root,
                phase=phase,
            )
            _write(
                child_artifacts / "recording_manifest.json",
                _single_recording_manifest(item),
            )
            _run_child(
                runner=runner,
                phase=phase,
                protocol_path=child_protocol,
                run_root=run_root,
                recording_root=recording_root,
                artifact_dir=child_artifacts,
                expected_commit=expected_commit,
                device=device,
                log_path=log_path,
            )
            results.append(_read(child_artifacts / f"{output_name}.json"))
        return results

    prefix = merge_prefix_results(
        run_gate(
            phase="takeover-prefix",
            output_name="prefix_replay_validation",
        ),
        expected_episode_count=len(by_id),
    )
    _write(artifact_dir / "prefix_replay_validation.json", prefix)
    if prefix["status"] != "pass":
        _write_not_run_takeover_outputs(
            artifact_dir,
            reason="blocked_by_prefix_replay_gate",
            include_prefix=False,
        )
        return

    branch = merge_branch_results(
        run_gate(
            phase="takeover-branch",
            output_name="branch_determinism_validation",
        ),
        expected_episode_count=len(by_id),
    )
    _write(artifact_dir / "branch_determinism_validation.json", branch)
    if branch["status"] != "pass":
        _write(
            artifact_dir / "branch_isolation_validation.json",
            _not_run_result(
                "lg_rb01_branch_isolation_validation_v1",
                "blocked_by_branch_determinism_gate",
            ),
        )
        return

    isolation = merge_isolation_results(
        run_gate(
            phase="takeover-isolation",
            output_name="branch_isolation_validation",
        ),
        expected_episode_count=len(by_id),
    )
    _write(artifact_dir / "branch_isolation_validation.json", isolation)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["record", "faithful", "takeover"],
        required=True,
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    function = {
        "record": _record,
        "faithful": _faithful,
        "takeover": _takeover,
    }[args.phase]
    function(
        protocol=protocol,
        runner=args.runner,
        run_root=args.run_root,
        recording_root=args.recording_root or args.run_root,
        artifact_dir=args.artifact_dir,
        expected_commit=args.expected_commit,
        device=args.device,
    )
    print(json.dumps({"phase": args.phase, "status": "complete"}, sort_keys=True))


if __name__ == "__main__":
    _main()
