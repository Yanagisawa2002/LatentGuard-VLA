"""CLI entry points for M5 release auditing and portfolio smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from latentguard.release import audit_release, run_portfolio_smoke

M5_COMMANDS = frozenset({"audit-release", "portfolio-smoke"})


def add_m5_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the M5 release commands."""

    smoke = subparsers.add_parser(
        "portfolio-smoke", help="run the CPU-only infrastructure portfolio path"
    )
    smoke.add_argument("--repo-root", type=Path, default=Path.cwd())

    audit = subparsers.add_parser(
        "audit-release", help="strictly audit release evidence and public surfaces"
    )
    audit.add_argument("--repo-root", type=Path, default=Path.cwd())
    audit.add_argument("--strict", action="store_true")
    audit.add_argument("--json-output", action="store_true")


def run_m5_command(args: argparse.Namespace) -> int | None:
    """Dispatch an M5 command or return ``None`` for another milestone."""

    if args.command not in M5_COMMANDS:
        return None
    try:
        if args.command == "portfolio-smoke":
            report = run_portfolio_smoke(args.repo_root)
            print(json.dumps(report, sort_keys=True, allow_nan=False))
            print(report["warning"])
            return 0
        report = audit_release(args.repo_root, strict=args.strict)
        if args.json_output:
            print(json.dumps(report, sort_keys=True, allow_nan=False))
        else:
            print(
                "audit-release OK "
                f"milestones={report['milestone_count']} "
                f"claims={report['claim_count']} results={report['result_count']} "
                f"issues={report['issue_count']} strict={str(report['strict']).lower()}"
            )
        return 0 if report["passed"] else 1
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1


__all__ = ["M5_COMMANDS", "add_m5_subparsers", "run_m5_command"]
