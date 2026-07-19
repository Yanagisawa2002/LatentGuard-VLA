"""CLI entry point for the M6A LangMani static contract audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from latentguard.integrations.langmani.audit import (
    LangManiAuditError,
    audit_langmani_contract,
)

M6A_COMMANDS = frozenset(
    {
        "audit-langmani-contract",
        "audit-langmani-m6b-readiness",
        "build-langmani-initial-candidates",
        "freeze-langmani-smoke-contract",
        "score-langmani-initial-candidates",
    }
)


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

    freeze = subparsers.add_parser(
        "freeze-langmani-smoke-contract",
        help="freeze the bounded M6B contract before proposal or outcomes",
    )
    freeze.add_argument("--policy-binding", type=Path, required=True)
    freeze.add_argument("--selector-configuration", type=Path, required=True)
    freeze.add_argument("--expected-langmani-sha", required=True)
    freeze.add_argument("--repo-root", type=Path, default=Path.cwd())
    freeze.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser(
        "build-langmani-initial-candidates",
        help="build four raw outcome-free prefixes from one LangMani proposal",
    )
    build.add_argument("--initial-proposal", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser(
        "score-langmani-initial-candidates",
        help="blind-score projected prefixes with the accepted action-only ensemble",
    )
    score.add_argument("--projected-candidates", type=Path, required=True)
    score.add_argument("--m3b-result-root", type=Path, required=True)
    score.add_argument("--m3b-runtime-root", type=Path, required=True)
    score.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    score.add_argument("--output", type=Path, required=True)

    readiness = subparsers.add_parser(
        "audit-langmani-m6b-readiness",
        help="audit structural or actual bounded M6B readiness",
    )
    readiness.add_argument("--repo-root", type=Path, default=Path.cwd())
    readiness.add_argument("--smoke-contract", type=Path)
    readiness.add_argument("--initial-proposal", type=Path)
    readiness.add_argument("--raw-candidates", type=Path)
    readiness.add_argument("--projected-candidates", type=Path)
    readiness.add_argument("--score-result", type=Path)
    readiness.add_argument("--output", type=Path)
    readiness.add_argument("--strict", action="store_true")
    readiness.add_argument("--json-output", action="store_true")


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
        if args.command != "audit-langmani-contract":
            return _run_m6a1_command(args)
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


def _run_m6a1_command(args: argparse.Namespace) -> int:
    from latentguard.integrations.langmani.m6a1 import (
        LANGMANI_INITIAL_PROPOSAL_SCHEMA,
        LANGMANI_POLICY_BINDING_SCHEMA,
        LANGMANI_PROJECTED_CANDIDATES_SCHEMA,
        RAW_CANDIDATE_REQUEST_SCHEMA,
        READINESS_SCHEMA,
        SCORE_RESULT_SCHEMA,
        SMOKE_CONTRACT_SCHEMA,
        M6A1BridgeError,
        audit_readiness,
        build_raw_candidates,
        freeze_smoke_contract,
        read_envelope,
        score_projected_candidates,
        validate_envelope,
        write_envelope,
    )

    try:
        if args.command == "freeze-langmani-smoke-contract":
            binding = read_envelope(
                args.policy_binding, expected_schema=LANGMANI_POLICY_BINDING_SCHEMA
            )
            report = freeze_smoke_contract(
                binding_envelope=binding,
                selector_configuration=args.selector_configuration,
                repository_root=args.repo_root,
                expected_langmani_sha=args.expected_langmani_sha,
            )
            write_envelope(args.output, report)
            print(
                f"freeze-langmani-smoke-contract OK digest={report['content_digest']}"
            )
            return 0
        if args.command == "build-langmani-initial-candidates":
            proposal = read_envelope(
                args.initial_proposal, expected_schema=LANGMANI_INITIAL_PROPOSAL_SCHEMA
            )
            report = build_raw_candidates(proposal)
            write_envelope(args.output, report)
            payload = validate_envelope(
                report, expected_schema=RAW_CANDIDATE_REQUEST_SCHEMA
            )
            print(
                "build-langmani-initial-candidates OK "
                f"candidates=4 pool={payload['candidate_pool_digest']}"
            )
            return 0
        if args.command == "score-langmani-initial-candidates":
            projected = read_envelope(
                args.projected_candidates,
                expected_schema=LANGMANI_PROJECTED_CANDIDATES_SCHEMA,
            )
            report = score_projected_candidates(
                projected_envelope=projected,
                m3b_result_root=args.m3b_result_root,
                m3b_runtime_root=args.m3b_runtime_root,
                device=args.device,
            )
            write_envelope(args.output, report)
            payload = validate_envelope(report, expected_schema=SCORE_RESULT_SCHEMA)
            print(
                "score-langmani-initial-candidates OK "
                f"selected={payload['selected_candidate_id']} "
                f"mask_invariance=true digest={report['content_digest']}"
            )
            return 0
        optional = (
            args.smoke_contract,
            args.initial_proposal,
            args.raw_candidates,
            args.projected_candidates,
            args.score_result,
        )
        if any(item is not None for item in optional) and not all(
            item is not None for item in optional
        ):
            raise M6A1BridgeError(
                "actual readiness requires all five bridge artifact paths"
            )
        report = audit_readiness(
            repository_root=args.repo_root,
            contract_envelope=(
                None
                if args.smoke_contract is None
                else read_envelope(
                    args.smoke_contract, expected_schema=SMOKE_CONTRACT_SCHEMA
                )
            ),
            proposal_envelope=(
                None
                if args.initial_proposal is None
                else read_envelope(
                    args.initial_proposal,
                    expected_schema=LANGMANI_INITIAL_PROPOSAL_SCHEMA,
                )
            ),
            raw_candidate_envelope=(
                None
                if args.raw_candidates is None
                else read_envelope(
                    args.raw_candidates, expected_schema=RAW_CANDIDATE_REQUEST_SCHEMA
                )
            ),
            projected_envelope=(
                None
                if args.projected_candidates is None
                else read_envelope(
                    args.projected_candidates,
                    expected_schema=LANGMANI_PROJECTED_CANDIDATES_SCHEMA,
                )
            ),
            score_envelope=(
                None
                if args.score_result is None
                else read_envelope(
                    args.score_result, expected_schema=SCORE_RESULT_SCHEMA
                )
            ),
        )
        if args.output is not None:
            write_envelope(args.output, report)
        payload = validate_envelope(report, expected_schema=READINESS_SCHEMA)
        if args.json_output:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        else:
            print(
                "audit-langmani-m6b-readiness OK "
                f"status={payload['readiness_status']} "
                f"digest={report['content_digest']}"
            )
        passed = all(
            cast(bool, value)
            for value in cast(Mapping[str, object], payload["gates"]).values()
        )
        return 0 if passed or not args.strict else 1
    except (M6A1BridgeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1


__all__ = ["M6A_COMMANDS", "add_m6a_subparsers", "run_m6a_command"]
