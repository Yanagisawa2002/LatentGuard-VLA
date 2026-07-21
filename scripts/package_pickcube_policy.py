"""Package a final native PickCube ACT checkpoint for strict D2 consumption."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn, cast

from latentguard.policies.act.packaging import build_pickcube_native_policy_registry


class PickCubeActPackageCliError(RuntimeError):
    """Raised when package CLI inputs are malformed or source is unclean."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeActPackageCliError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeActPackageCliError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _builder_source() -> Mapping[str, object]:
    repository = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PickCubeActPackageCliError(
                "builder source: unable to inspect Git identity"
            ) from exc
        return completed.stdout.strip()

    if git("status", "--short", "--untracked-files=no"):
        _fail("builder source", "tracked worktree is not clean")
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    if not branch or len(commit) != 40:
        _fail("builder source", "branch or full commit is unavailable")
    return {"branch": branch, "commit": commit}


def main() -> int:
    """Build a provisional or smoke-authorized final PolicyPackage registry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--smoke-audit", type=Path)
    args = parser.parse_args()

    config = _mapping(args.config, "package config")
    schema = config.get("schema_version")
    if schema not in {
        "pickcube-native-act-package-config-v1",
        "pickcube-act-bounded-package-config-v1",
    }:
        _fail("package config", "schema mismatch")
    policy_id = config.get("policy_id")
    if not isinstance(policy_id, str):
        _fail("package config", "policy_id is missing")
    run = Path(args.training_run).absolute()
    resolved = _mapping(run / "resolved_config.json", "resolved training config")
    if schema == "pickcube-act-bounded-package-config-v1":
        parameterization = resolved.get("action_parameterization")
        if (
            resolved.get("schema_version")
            != "pickcube-native-act-bounded-experiment-v1"
            or not isinstance(parameterization, Mapping)
            or parameterization.get("parameterization_type")
            != config.get("required_action_parameterization")
            or parameterization.get("checkpoint_schema_version")
            != config.get("required_checkpoint_schema")
        ):
            _fail("package config", "bounded action identity differs")
    run_manifest = _mapping(run / "run_manifest.json", "training run manifest")
    identity = run_manifest.get("training_identity")
    source_commit = run_manifest.get("git_commit")
    if not isinstance(identity, Mapping) or not isinstance(source_commit, str):
        _fail("training run manifest", "training identity or source commit is missing")
    smoke = (
        None
        if args.smoke_audit is None
        else _mapping(args.smoke_audit, "package smoke audit")
    )
    registry = build_pickcube_native_policy_registry(
        checkpoint=args.checkpoint,
        expected_training_identity=identity,
        resolved_training_config=resolved,
        contract=_mapping(args.contract, "PickCube ACT contract"),
        final_evaluation=_mapping(args.evaluation, "final evaluation"),
        normalization_stats_path=args.normalization_stats,
        package_root=args.package_root,
        registry_output=args.registry_output,
        policy_id=policy_id,
        source_commit=source_commit,
        builder_source=_builder_source(),
        smoke_audit=smoke,
    )
    print(
        json.dumps(
            {
                "accepted_compatible_policy_count": registry.accepted_count,
                "package_root": args.package_root.as_posix(),
                "registry_digest": registry.registry_digest,
                "registry_output": args.registry_output.as_posix(),
                "result": registry.result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
