"""Validate LG-R1 source, config, artifact, and isolation contracts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from _lg_r1_common import git_output, repo_root, write_json


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
    """Fail closed on malformed config, secrets, or repository isolation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    tracked = _tracked_files()
    tracked_names = [str(path.relative_to(repo_root()).as_posix()) for path in tracked]
    forbidden_paths = [
        path
        for path in tracked_names
        if "langmani_v2" in path.lower()
        or path.lower().endswith((".pt", ".pth", ".ckpt", ".mp4"))
    ]
    sensitive_hits: list[str] = []
    scoped = [
        path
        for path in tracked
        if str(path.relative_to(repo_root()).as_posix()).startswith(
            (
                "configs/lg_r1/",
                "artifacts/lg_r1/",
                "docs/lg_r1",
                "docs/plans/lg-r1",
                "scripts/lg_r1_",
                "scripts/_lg_r1_",
                "src/latentguard/progress/",
            )
        )
    ]
    patterns = (
        re.compile(
            r"password\s*[:=]\s*['\"][^'\"]+['\"]",
            re.IGNORECASE,
        ),
        re.compile(r"BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY"),
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
    config_errors: list[str] = []
    for path in sorted((repo_root() / "configs" / "lg_r1").glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "schema_version" not in payload:
                raise ValueError("mapping with schema_version required")
        except Exception as exc:
            config_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    json_errors: list[str] = []
    artifact_root = (
        args.artifact_root
        if args.artifact_root is not None
        else repo_root() / "artifacts" / "lg_r1"
    )
    if artifact_root.exists():
        for path in artifact_root.rglob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                json_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    import latentguard

    package_path = Path(latentguard.__file__).resolve()
    expected_package_root = (repo_root() / "src" / "latentguard").resolve()
    commit = git_output("rev-parse", "HEAD")
    checks: dict[str, Any] = {
        "repository_identity": "LatentGuard-VLA"
        in git_output("remote", "get-url", "origin"),
        "langmani_isolation": not forbidden_paths,
        "sensitive_content": not sensitive_hits,
        "config_parse": not config_errors,
        "json_parse": not json_errors,
        "package_import": package_path.is_relative_to(expected_package_root),
        "expected_commit": (
            args.expected_commit is None or commit == args.expected_commit
        ),
    }
    payload = {
        "schema_version": "latentguard.lg_r1.source_validation.v1",
        "status": "pass" if all(checks.values()) else "fail",
        "commit": commit,
        "checks": checks,
        "forbidden_paths": forbidden_paths,
        "sensitive_hits": sensitive_hits,
        "config_errors": config_errors,
        "json_errors": json_errors,
        "package_path": str(package_path),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "pass":
        raise RuntimeError("LG-R1 source validation failed")


if __name__ == "__main__":
    main()
