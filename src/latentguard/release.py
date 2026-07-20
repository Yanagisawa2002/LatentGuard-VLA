"""Strict release registries, evidence audit, and CPU portfolio smoke."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn, cast

import numpy as np

from latentguard.control.models import content_digest
from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    save_corruption_dataset,
)
from latentguard.evaluation.serialization import load_evaluation_dataset
from latentguard.replay.identity import canonical_json_bytes
from latentguard.replay.source import ReplaySourceBinding
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    save_episodes,
)
from latentguard.synthetic import generate_synthetic_episodes

RELEASE_SCHEMA_VERSION = "1.0"
PORTFOLIO_SMOKE_SEMANTIC_VERSION = "1.0.0"
AUDIT_RELEASE_SEMANTIC_VERSION = "1.0.0"
INFRASTRUCTURE_SMOKE_WARNING = (
    "This is a CPU infrastructure smoke, not a simulator or research-result "
    "reproduction."
)

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RESULT_STATUSES = frozenset(
    {"supported", "partially_supported", "unsupported", "not_tested"}
)
_RESEARCH_STATUSES = frozenset(
    {
        "accepted_positive",
        "accepted_partial",
        "accepted_negative",
        "infrastructure_only",
    }
)
_SURFACES = frozenset(
    {"README", "resume", "interview", "technical report", "research discussion"}
)
_MILESTONE_KEYS = frozenset(
    {
        "milestone_id",
        "title",
        "scope",
        "branch",
        "implementation_commits",
        "accepted_result_commit",
        "remote_execution_commit",
        "run_ids",
        "primary_reports",
        "test_results",
        "research_status",
        "training_occurred",
        "simulator_execution_occurred",
        "visual_data_used",
        "external_evaluation_occurred",
        "known_limitations",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "claim_id",
        "wording",
        "support_status",
        "scope",
        "evidence_references",
        "counterevidence_references",
        "limitations",
        "permitted_usage_surfaces",
    }
)
_RESULT_KEYS = frozenset(
    {
        "result_id",
        "milestone_id",
        "category",
        "experiment",
        "metric",
        "value",
        "unit",
        "baseline",
        "interpretation",
        "source",
        "claim_status",
    }
)
_SOURCE_KEYS = frozenset(
    {"report_path", "report_digest", "json_pointer", "execution_commit", "run_id"}
)
_PUBLIC_DOCUMENTS = (
    Path("README.md"),
    Path("docs/release/claims.md"),
    Path("docs/portfolio/technical-report.md"),
    Path("docs/portfolio/case-study.md"),
    Path("docs/portfolio/architecture.md"),
    Path("docs/portfolio/key-results.md"),
    Path("docs/portfolio/demo-script.md"),
    Path("docs/portfolio/presentation-outline.md"),
    Path("docs/career/resume-bullets.md"),
    Path("docs/career/interview-brief.md"),
    Path("docs/career/project-summary-en.md"),
    Path("docs/career/project-summary-zh.md"),
)
_FORBIDDEN_TRACKED_SUFFIXES = frozenset(
    {".ckpt", ".npy", ".npz", ".onnx", ".pt", ".pth", ".safetensors"}
)
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_PRIVATE_PATH = re.compile(
    r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/root/|/home/[^/\s]+/|\\\\[^\s]+)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:password|passwd|private[_-]?key|secret|token|credential)\s*[:=]\s*[^\s`]+"
)


class ReleaseValidationError(ValueError):
    """Raised when a release registry or public surface is inconsistent."""


def _fail(context: str, reason: str) -> NoReturn:
    raise ReleaseValidationError(f"{context}: {reason}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(context, f"field mismatch missing={missing} extra={extra}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _string_list(value: object, context: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        _fail(context, "expected JSON string list")
    result = [_text(item, f"{context}[{index}]") for index, item in enumerate(value)]
    return result


def _digest(value: object, context: str) -> str:
    result = _text(value, context)
    if _DIGEST_PATTERN.fullmatch(result) is None:
        _fail(context, "expected lowercase sha256 digest")
    return result


def _commit(value: object, context: str) -> str:
    result = _text(value, context)
    if _COMMIT_PATTERN.fullmatch(result) is None:
        _fail(context, "expected full lowercase Git commit SHA")
    return result


def _relative_path(value: object, context: str) -> Path:
    text = _text(value, context)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        _fail(context, "expected normalized repository-relative path")
    return path


def _load_json(path: Path, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(
            f"{context}: cannot load {path.as_posix()}"
        ) from exc
    return _mapping(value, context)


def file_digest(path: Path) -> str:
    """Return the byte-exact SHA-256 digest of one file."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReleaseValidationError(f"cannot read {path.as_posix()}") from exc
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _release_source_file_digest(
    repo_root: Path,
    release_commit: str,
    relative_path: str,
    expected_digest: object,
) -> str | None:
    """Hash a document at the frozen release commit.

    Post-release milestones may extend a current public document. The M5 manifest
    remains bound to the exact bytes at ``release_source_git_sha`` instead of
    requiring the working-tree document to remain permanently uneditable.
    """

    history = subprocess.run(
        ["git", "rev-list", "HEAD", "--", relative_path],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if history.returncode != 0:
        return None
    candidates = tuple(dict.fromkeys((release_commit, *history.stdout.splitlines())))
    observed: str | None = None
    for commit in candidates:
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            continue
        observed = f"sha256:{hashlib.sha256(result.stdout).hexdigest()}"
        if observed == expected_digest:
            return observed
    return observed


def registry_digest(value: Mapping[str, object], *, context: str) -> str:
    """Return the semantic digest of a registry excluding its digest envelope."""

    payload = {key: item for key, item in value.items() if key != "content_digest"}
    encoded = canonical_json_bytes(payload, context=context, reject_runtime_paths=False)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_envelope(
    value: Mapping[str, object], *, collection_key: str, context: str
) -> list[object]:
    _exact_keys(
        value,
        frozenset({"schema_version", "content_digest", collection_key}),
        context,
    )
    if value["schema_version"] != RELEASE_SCHEMA_VERSION:
        _fail(context, "unsupported schema version")
    observed = _digest(value["content_digest"], f"{context}.content_digest")
    expected = registry_digest(value, context=context)
    if observed != expected:
        _fail(context, "content digest mismatch")
    collection = value[collection_key]
    if not isinstance(collection, list):
        _fail(context, f"{collection_key} must be a JSON list")
    return cast(list[object], collection)


def load_milestone_registry(path: Path) -> Mapping[str, object]:
    """Load and strictly validate the accepted milestone registry."""

    value = _load_json(path, "milestone registry")
    records = _validate_envelope(
        value, collection_key="milestones", context="MilestoneRegistryV1"
    )
    identifiers: set[str] = set()
    for index, raw in enumerate(records):
        context = f"MilestoneRegistryV1.milestones[{index}]"
        record = _mapping(raw, context)
        _exact_keys(record, _MILESTONE_KEYS, context)
        identifier = _text(record["milestone_id"], f"{context}.milestone_id")
        if identifier in identifiers:
            _fail(context, f"duplicate milestone ID {identifier!r}")
        identifiers.add(identifier)
        for key in ("title", "scope", "branch", "test_results", "research_status"):
            _text(record[key], f"{context}.{key}")
        if record["research_status"] not in _RESEARCH_STATUSES:
            _fail(context, "unknown research status")
        commits = _string_list(
            record["implementation_commits"],
            f"{context}.implementation_commits",
            nonempty=True,
        )
        for commit in commits:
            _commit(commit, f"{context}.implementation_commits")
        _commit(record["accepted_result_commit"], f"{context}.accepted_result_commit")
        remote_commit = record["remote_execution_commit"]
        if remote_commit is not None:
            _commit(remote_commit, f"{context}.remote_execution_commit")
        _string_list(record["run_ids"], f"{context}.run_ids")
        _string_list(
            record["known_limitations"], f"{context}.known_limitations", nonempty=True
        )
        for key in (
            "training_occurred",
            "simulator_execution_occurred",
            "visual_data_used",
            "external_evaluation_occurred",
        ):
            if type(record[key]) is not bool:
                _fail(context, f"{key} must be boolean")
        reports = record["primary_reports"]
        if not isinstance(reports, list) or not reports:
            _fail(context, "primary_reports must be a non-empty list")
        for report_index, raw_report in enumerate(reports):
            report_context = f"{context}.primary_reports[{report_index}]"
            report = _mapping(raw_report, report_context)
            _exact_keys(report, frozenset({"path", "digest"}), report_context)
            _relative_path(report["path"], f"{report_context}.path")
            _digest(report["digest"], f"{report_context}.digest")
    return value


def load_claims_registry(path: Path) -> Mapping[str, object]:
    """Load and strictly validate the authoritative public-claims registry."""

    value = _load_json(path, "claims registry")
    records = _validate_envelope(
        value,
        collection_key="claims",
        context="ClaimsRegistryV1",
    )
    identifiers: set[str] = set()
    for index, raw in enumerate(records):
        context = f"ClaimsRegistryV1.claims[{index}]"
        record = _mapping(raw, context)
        _exact_keys(record, _CLAIM_KEYS, context)
        identifier = _text(record["claim_id"], f"{context}.claim_id")
        if identifier in identifiers:
            _fail(context, f"duplicate claim ID {identifier!r}")
        identifiers.add(identifier)
        for key in ("wording", "support_status", "scope"):
            _text(record[key], f"{context}.{key}")
        status = cast(str, record["support_status"])
        if status not in _RESULT_STATUSES:
            _fail(context, "unknown support status")
        evidence = _string_list(
            record["evidence_references"], f"{context}.evidence_references"
        )
        counterevidence = _string_list(
            record["counterevidence_references"],
            f"{context}.counterevidence_references",
        )
        _string_list(record["limitations"], f"{context}.limitations", nonempty=True)
        surfaces = _string_list(
            record["permitted_usage_surfaces"],
            f"{context}.permitted_usage_surfaces",
            nonempty=True,
        )
        if not set(surfaces) <= _SURFACES:
            _fail(context, "unknown permitted usage surface")
        if status == "supported" and not evidence:
            _fail(context, "supported claim requires evidence")
        if status in {"partially_supported", "unsupported"} and not counterevidence:
            _fail(context, f"{status} claim requires counterevidence")
        if status == "unsupported" and set(surfaces) & {"README", "resume"}:
            _fail(context, "unsupported claim escalated to a public success surface")
    return value


def load_results_registry(path: Path) -> Mapping[str, object]:
    """Load and strictly validate the unified results registry."""

    value = _load_json(path, "results registry")
    records = _validate_envelope(
        value,
        collection_key="results",
        context="ResultsRegistryV1",
    )
    identifiers: set[str] = set()
    for index, raw in enumerate(records):
        context = f"ResultsRegistryV1.results[{index}]"
        record = _mapping(raw, context)
        _exact_keys(record, _RESULT_KEYS, context)
        identifier = _text(record["result_id"], f"{context}.result_id")
        if identifier in identifiers:
            _fail(context, f"duplicate result ID {identifier!r}")
        identifiers.add(identifier)
        for key in (
            "milestone_id",
            "category",
            "experiment",
            "metric",
            "unit",
            "interpretation",
            "claim_status",
        ):
            _text(record[key], f"{context}.{key}")
        if record["claim_status"] not in _RESULT_STATUSES:
            _fail(context, "unknown claim status")
        if record["baseline"] is not None:
            _text(record["baseline"], f"{context}.baseline")
        source = _mapping(record["source"], f"{context}.source")
        _exact_keys(source, _SOURCE_KEYS, f"{context}.source")
        _relative_path(source["report_path"], f"{context}.source.report_path")
        _digest(source["report_digest"], f"{context}.source.report_digest")
        _text(source["json_pointer"], f"{context}.source.json_pointer")
        _commit(source["execution_commit"], f"{context}.source.execution_commit")
        _text(source["run_id"], f"{context}.source.run_id")
    return value


def load_release_manifest(path: Path) -> Mapping[str, object]:
    """Load and strictly validate the content-bound release manifest."""

    value = _load_json(path, "release manifest")
    expected = frozenset(
        {
            "schema_version",
            "content_digest",
            "release_id",
            "release_source_git_sha",
            "package_version",
            "registry_digests",
            "document_digests",
            "portfolio_smoke_semantic_version",
            "audit_release_semantic_version",
            "test_summary",
            "server_artifact_availability",
        }
    )
    _exact_keys(value, expected, "ReleaseManifestV1")
    if value["schema_version"] != RELEASE_SCHEMA_VERSION:
        _fail("ReleaseManifestV1", "unsupported schema version")
    observed = _digest(value["content_digest"], "ReleaseManifestV1.content_digest")
    if observed != registry_digest(value, context="ReleaseManifestV1"):
        _fail("ReleaseManifestV1", "content digest mismatch")
    _text(value["release_id"], "ReleaseManifestV1.release_id")
    _commit(value["release_source_git_sha"], "ReleaseManifestV1.release_source_git_sha")
    _text(value["package_version"], "ReleaseManifestV1.package_version")
    if value["portfolio_smoke_semantic_version"] != PORTFOLIO_SMOKE_SEMANTIC_VERSION:
        _fail("ReleaseManifestV1", "portfolio smoke semantic version mismatch")
    if value["audit_release_semantic_version"] != AUDIT_RELEASE_SEMANTIC_VERSION:
        _fail("ReleaseManifestV1", "release audit semantic version mismatch")
    for field in ("registry_digests", "document_digests"):
        entries = _mapping(value[field], f"ReleaseManifestV1.{field}")
        if not entries:
            _fail("ReleaseManifestV1", f"{field} must not be empty")
        for key, digest_value in entries.items():
            _relative_path(key, f"ReleaseManifestV1.{field}.key")
            _digest(digest_value, f"ReleaseManifestV1.{field}.{key}")
    _mapping(value["test_summary"], "ReleaseManifestV1.test_summary")
    _mapping(
        value["server_artifact_availability"],
        "ReleaseManifestV1.server_artifact_availability",
    )
    return value


def _json_pointer(value: object, pointer: str) -> object:
    if not pointer.startswith("/"):
        _fail("result source", "JSON pointer must begin with '/'")
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            _fail("result source", f"unresolved JSON pointer {pointer!r}")
    return current


def _commit_exists(repo_root: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _tracked_files(repo_root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _fail("release audit", "cannot enumerate tracked files")
    return tuple(
        Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item
    )


def _audit_links(repo_root: Path, path: Path, text: str) -> list[str]:
    issues: list[str] = []
    for target in _MARKDOWN_LINK.findall(text):
        clean = target.strip().split("#", 1)[0]
        if not clean or clean.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (repo_root / path.parent / clean).resolve()
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            issues.append(f"{path.as_posix()}: link escapes repository: {target}")
            continue
        if not resolved.exists():
            issues.append(f"{path.as_posix()}: broken link: {target}")
    return issues


def audit_release(repo_root: Path, *, strict: bool = False) -> Mapping[str, object]:
    """Audit release registries, evidence, public documents, and tracked artifacts."""

    root = repo_root.resolve()
    issues: list[str] = []
    milestones = load_milestone_registry(root / "docs/release/milestones.json")
    claims = load_claims_registry(root / "docs/release/claims.json")
    results = load_results_registry(root / "docs/release/results.json")
    manifest = load_release_manifest(root / "docs/release/release-manifest.json")

    milestone_records = cast(list[Mapping[str, object]], milestones["milestones"])
    milestone_ids = {cast(str, record["milestone_id"]) for record in milestone_records}
    for record in milestone_records:
        commits = cast(list[str], record["implementation_commits"])
        commits.append(cast(str, record["accepted_result_commit"]))
        remote = record["remote_execution_commit"]
        if isinstance(remote, str):
            commits.append(remote)
        for commit in commits:
            if not _commit_exists(root, commit):
                issues.append(f"unresolved Git commit {commit}")
        for report in cast(list[Mapping[str, object]], record["primary_reports"]):
            path = root / cast(str, report["path"])
            if not path.is_file():
                issues.append(f"missing milestone report {report['path']}")
            elif file_digest(path) != report["digest"]:
                issues.append(f"milestone report digest mismatch {report['path']}")

    result_records = cast(list[Mapping[str, object]], results["results"])
    result_ids = {cast(str, record["result_id"]) for record in result_records}
    result_statuses = {
        cast(str, record["result_id"]): cast(str, record["claim_status"])
        for record in result_records
    }
    for record in result_records:
        if record["milestone_id"] not in milestone_ids:
            issues.append(f"result {record['result_id']} has unknown milestone")
        source = cast(Mapping[str, object], record["source"])
        path = root / cast(str, source["report_path"])
        if not path.is_file():
            issues.append(f"missing result report {source['report_path']}")
            continue
        if file_digest(path) != source["report_digest"]:
            issues.append(f"result report digest mismatch {source['report_path']}")
            continue
        try:
            source_value = json.loads(path.read_text(encoding="utf-8"))
            observed_value = _json_pointer(
                source_value, cast(str, source["json_pointer"])
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ReleaseValidationError,
        ) as exc:
            issues.append(f"result {record['result_id']} source failed: {exc}")
            continue
        if observed_value != record["value"]:
            issues.append(f"result mismatch {record['result_id']}")
        if not _commit_exists(root, cast(str, source["execution_commit"])):
            issues.append(
                f"result {record['result_id']} has unresolved execution commit"
            )

    claim_records = cast(list[Mapping[str, object]], claims["claims"])
    for record in claim_records:
        evidence_references = cast(list[str], record["evidence_references"])
        references = evidence_references + cast(
            list[str], record["counterevidence_references"]
        )
        for reference in references:
            if reference not in result_ids:
                issues.append(
                    f"claim {record['claim_id']} has unknown evidence {reference}"
                )
        if record["support_status"] == "supported" and any(
            result_statuses.get(reference) in {"unsupported", "not_tested"}
            for reference in evidence_references
        ):
            issues.append(f"claim {record['claim_id']} escalates unsupported evidence")

    registry_paths = {
        "docs/release/milestones.json": milestones,
        "docs/release/claims.json": claims,
        "docs/release/results.json": results,
    }
    manifest_registry = cast(Mapping[str, object], manifest["registry_digests"])
    for relative, registry in registry_paths.items():
        if manifest_registry.get(relative) != registry["content_digest"]:
            issues.append(f"release manifest registry mismatch {relative}")
    manifest_documents = cast(Mapping[str, object], manifest["document_digests"])
    release_commit = cast(str, manifest["release_source_git_sha"])
    for relative, expected_digest in manifest_documents.items():
        path = root / relative
        observed_digest = _release_source_file_digest(
            root, release_commit, relative, expected_digest
        )
        if observed_digest is None and path.is_file():
            observed_digest = file_digest(path)
        if observed_digest != expected_digest:
            issues.append(f"release manifest document mismatch {relative}")

    for document_relative in _PUBLIC_DOCUMENTS:
        path = root / document_relative
        if not path.is_file():
            issues.append(f"missing portfolio document {document_relative.as_posix()}")
            continue
        text = path.read_text(encoding="utf-8")
        if _PRIVATE_PATH.search(text):
            issues.append(f"private absolute path in {document_relative.as_posix()}")
        if _SECRET_ASSIGNMENT.search(text):
            issues.append(f"secret-like assignment in {document_relative.as_posix()}")
        if any(marker in text for marker in ("TODO", "TBD", "PLACEHOLDER")):
            issues.append(f"placeholder text in {document_relative.as_posix()}")
        issues.extend(_audit_links(root, document_relative, text))

    key_ids = {
        "m2c-strong-replays",
        "m3a-verified-outcomes",
        "m3b-test-auprc",
        "m3c-temporal-success",
        "m4b-external-visual-success",
        "m4c-distilled-vs-fixed",
        "m4d-fault-gated-success",
        "m4d-override-recall",
    }
    for result_document_relative in (
        Path("README.md"),
        Path("docs/portfolio/technical-report.md"),
        Path("docs/portfolio/key-results.md"),
    ):
        text = (root / result_document_relative).read_text(encoding="utf-8")
        for result_id in key_ids:
            if f"LG-RESULT:{result_id}" not in text:
                issues.append(
                    f"{result_document_relative.as_posix()} missing key result "
                    f"{result_id}"
                )

    for tracked in _tracked_files(root):
        if tracked.suffix.lower() in _FORBIDDEN_TRACKED_SUFFIXES:
            issues.append(f"raw artifact tracked in Git: {tracked.as_posix()}")
        path = root / tracked
        is_reviewable_report = tracked.parts[:1] == (
            "reports",
        ) and tracked.suffix.lower() in {".json", ".md"}
        if (
            path.is_file()
            and path.stat().st_size > 5_000_000
            and not is_reviewable_report
        ):
            issues.append(f"large tracked artifact: {tracked.as_posix()}")

    report = {
        "audit_semantic_version": AUDIT_RELEASE_SEMANTIC_VERSION,
        "claim_count": len(claim_records),
        "issue_count": len(issues),
        "issues": issues,
        "milestone_count": len(milestone_records),
        "passed": not issues,
        "result_count": len(result_records),
        "schema_version": RELEASE_SCHEMA_VERSION,
        "strict": strict,
    }
    if strict and issues:
        raise ReleaseValidationError(
            "strict release audit failed: " + "; ".join(issues)
        )
    return report


def run_portfolio_smoke(repo_root: Path) -> Mapping[str, object]:
    """Run the deterministic CPU-only infrastructure portfolio path."""

    root = repo_root.resolve()
    load_milestone_registry(root / "docs/release/milestones.json")
    load_claims_registry(root / "docs/release/claims.json")
    load_results_registry(root / "docs/release/results.json")
    load_release_manifest(root / "docs/release/release-manifest.json")

    from latentguard import cli

    with tempfile.TemporaryDirectory(prefix="latentguard-m5-smoke-") as temporary:
        work = Path(temporary)
        source_dir = work / "source"
        corruption_dir = work / "corruptions"
        replay_dir = work / "replay"
        episodes = generate_synthetic_episodes(
            seed=42,
            episode_count=1,
            episode_length=8,
            action_dim=7,
            robot_state_dim=10,
            camera_count=0,
            candidate_count=3,
        )
        save_episodes(episodes, source_dir)
        plan = load_corruption_plan(root / "configs/corruptions/m1-smoke.json")
        generated = generate_corruption_proposals(
            episodes,
            plan.action_layout,
            plan.corruptions,
            base_seed=314159,
            proposal_limit=8,
        )
        corruption = CorruptionDataset(
            source_dataset_id=compute_episode_bundle_identifier(source_dir),
            action_layout=plan.action_layout,
            proposals=generated.proposals,
        )
        save_corruption_dataset(corruption, corruption_dir)
        arguments = [
            "replay-data",
            "--source-dir",
            str(source_dir),
            "--corruption-dir",
            str(corruption_dir),
            "--output-dir",
            str(replay_dir),
            "--adapter",
            "deterministic_replay_fixture",
            "--config",
            str(root / "configs/replay/m2b-fixture.json"),
            "--seed",
            "161803",
            "--max-proposals",
            "8",
        ]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            first_code = cli.main(arguments)
        before = (replay_dir / "manifest.json").read_bytes()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            resume_code = cli.main([*arguments, "--resume"])
        after = (replay_dir / "manifest.json").read_bytes()
        if first_code != 0 or resume_code != 0 or before != after:
            _fail("portfolio smoke", "fixture replay or zero-work resume failed")

        binding = ReplaySourceBinding.from_paths(source_dir, corruption_dir)
        dataset = load_evaluation_dataset(
            replay_dir,
            corruption_dataset=binding.corruption_dataset,
            expected_corruption_digest=binding.corruption_dataset_digest,
        )
        probabilities = np.asarray(
            [
                0.0 if item.success is True else 1.0 if item.success is False else 0.5
                for item in dataset.evidence
            ],
            dtype=np.float64,
        )
        if probabilities.shape != (8,) or not bool(np.all(np.isfinite(probabilities))):
            _fail("portfolio smoke", "compact verifier fixture output is invalid")
        selected_index = int(np.argmin(probabilities))
        selected = dataset.evidence[selected_index]
        projected_count = sum(item.success is not None for item in dataset.evidence)
        semantic_payload = {
            "corruption_digest": binding.corruption_dataset_digest,
            "evidence_ids": [item.evidence_id for item in dataset.evidence],
            "projected_outcome_count": projected_count,
            "selected_evidence_id": selected.evidence_id,
            "selected_fixture_failure_probability": float(
                probabilities[selected_index]
            ),
            "source_dataset_id": binding.source_dataset_id,
        }
        semantic_digest = content_digest(
            semantic_payload, context="PortfolioSmokeSemanticV1"
        )

    return {
        "deterministic_identity_digest": semantic_digest,
        "evidence_count": 8,
        "fixture_trust": "non_physical_infrastructure_only",
        "passed": True,
        "portfolio_smoke_semantic_version": PORTFOLIO_SMOKE_SEMANTIC_VERSION,
        "projected_outcome_count": projected_count,
        "registry_loading": "verified",
        "resume_zero_work": True,
        "schema_version": RELEASE_SCHEMA_VERSION,
        "warning": INFRASTRUCTURE_SMOKE_WARNING,
    }


__all__ = [
    "AUDIT_RELEASE_SEMANTIC_VERSION",
    "INFRASTRUCTURE_SMOKE_WARNING",
    "PORTFOLIO_SMOKE_SEMANTIC_VERSION",
    "RELEASE_SCHEMA_VERSION",
    "ReleaseValidationError",
    "audit_release",
    "file_digest",
    "load_claims_registry",
    "load_milestone_registry",
    "load_release_manifest",
    "load_results_registry",
    "registry_digest",
    "run_portfolio_smoke",
]
