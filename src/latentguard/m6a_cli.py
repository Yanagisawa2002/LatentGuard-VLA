"""CLI entry point for the M6A LangMani static contract audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from latentguard.integrations.langmani.audit import (
    LangManiAuditError,
    audit_langmani_contract,
)

M6A_COMMANDS = frozenset({"audit-langmani-contract"})


def add_m6a_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the M6A read-only audit command."""

    audit = subparsers.add_parser(
        "audit-langmani-contract",
        help="audit one exact LangMani checkout without running a simulator",
    )
    audit.add_argument("--langmani-root", type=Path)
    audit.add_argument("--repo-root", type=Path, default=Path.cwd())
    audit.add_argument("--output", type=Path)
    audit.add_argument("--strict", action="store_true")
    audit.add_argument("--json-output", action="store_true")
    audit.add_argument("--allow-import-probe", action="store_true")


def _resolve_root(value: Path | None) -> Path:
    if value is not None:
        return value
    configured = os.environ.get("LATENTGUARD_LANGMANI_ROOT")
    if not configured:
        raise LangManiAuditError(
            "provide --langmani-root or set LATENTGUARD_LANGMANI_ROOT"
        )
    return Path(configured)


def run_m6a_command(args: argparse.Namespace) -> int | None:
    """Dispatch the M6A command or return ``None`` for another milestone."""

    if args.command not in M6A_COMMANDS:
        return None
    try:
        report = audit_langmani_contract(
            _resolve_root(args.langmani_root),
            latentguard_root=args.repo_root,
            strict=args.strict,
            allow_import_probe=args.allow_import_probe,
        )
        serialized = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8", newline="\n")
        if args.json_output:
            print(serialized)
        else:
            counts = report["blocker_counts"]
            identity = report["repository_identity"]
            issues = report["audit_issues"]
            synthetic = report["synthetic_adapter"]
            assert isinstance(counts, Mapping)
            assert isinstance(identity, Mapping)
            assert isinstance(issues, Sequence)
            assert isinstance(synthetic, Mapping)
            print(
                "audit-langmani-contract OK "
                f"sha={identity['head_sha']} "
                f"readiness={report['overall_readiness']} "
                f"blockers={counts['critical']}/{counts['major']}/{counts['minor']} "
                f"issues={len(issues)} "
                f"strict={str(report['strict']).lower()}"
            )
            print(synthetic["warning"])
        return 0 if report["passed"] else 1
    except (LangManiAuditError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"audit-langmani-contract failed: {exc}", file=sys.stderr)
        return 1


__all__ = ["M6A_COMMANDS", "add_m6a_subparsers", "run_m6a_command"]
