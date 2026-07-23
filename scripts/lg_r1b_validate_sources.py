"""Validate LG-R1b source, artifact, secret, and repository boundaries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from _lg_r1b_common import git_output, repo_root, resolve_repo_path, write_json


def _tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [repo_root() / path for path in output]


def main() -> None:
    """Fail on malformed evidence, credentials, large payloads, or drift."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    tracked = _tracked_files()
    names = [str(path.relative_to(repo_root()).as_posix()) for path in tracked]
    forbidden_paths = [
        name
        for name in names
        if "langmani_v2" in name.lower()
        or name.lower().endswith((".pt", ".pth", ".ckpt", ".mp4", ".parquet", ".npz"))
    ]
    scoped = [
        path
        for path in tracked
        if str(path.relative_to(repo_root()).as_posix()).startswith(
            (
                "configs/lg_r1b/",
                "artifacts/lg_r1b/",
                "docs/lg_r1b",
                "docs/plans/lg-r1b",
                "scripts/lg_r1b_",
                "scripts/_lg_r1b_",
                "src/latentguard/progress/",
            )
        )
    ]
    sensitive_hits: list[str] = []
    patterns = (
        re.compile(r"password\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
        re.compile(r"BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY"),
        re.compile(r"connect\.[a-z0-9.-]+\.com", re.IGNORECASE),
    )
    for path in scoped:
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(text):
                sensitive_hits.append(
                    f"{path.relative_to(repo_root())}:{pattern.pattern}"
                )
    config_errors = []
    for path in sorted((repo_root() / "configs" / "lg_r1b").glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "schema_version" not in payload:
                raise ValueError("mapping with schema_version required")
        except Exception as exc:
            config_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    json_errors = []
    artifact_root = (
        resolve_repo_path(args.artifact_root)
        if args.artifact_root is not None
        else repo_root() / "artifacts" / "lg_r1b"
    )
    if artifact_root.exists():
        for path in artifact_root.rglob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                json_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    import latentguard

    package_path = Path(latentguard.__file__).resolve()
    expected_package = (repo_root() / "src" / "latentguard").resolve()
    commit = git_output("rev-parse", "HEAD")
    checks: dict[str, Any] = {
        "repository_identity": "LatentGuard-VLA"
        in git_output("remote", "get-url", "origin"),
        "langmani_isolation": not forbidden_paths,
        "sensitive_content": not sensitive_hits,
        "config_parse": not config_errors,
        "json_parse": not json_errors,
        "package_import": package_path.is_relative_to(expected_package),
        "expected_commit": (
            args.expected_commit is None or commit == args.expected_commit
        ),
    }
    payload = {
        "schema_version": "latentguard.lg_r1b.source_validation.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "commit": commit,
        "checks": checks,
        "forbidden_paths": forbidden_paths,
        "sensitive_hits": sensitive_hits,
        "config_errors": config_errors,
        "json_errors": json_errors,
        "package_path": package_path.relative_to(repo_root()).as_posix(),
    }
    write_json(resolve_repo_path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "pass":
        raise RuntimeError("LG-R1b source validation failed")


if __name__ == "__main__":
    main()
