"""M5 release registries, audit, portfolio smoke, and public-surface tests."""

from __future__ import annotations

import builtins
import json
import shutil
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

from latentguard import release as release_module
from latentguard.release import (
    INFRASTRUCTURE_SMOKE_WARNING,
    ReleaseValidationError,
    audit_release,
    load_claims_registry,
    load_milestone_registry,
    load_release_manifest,
    load_results_registry,
    registry_digest,
    run_portfolio_smoke,
)

_ROOT = Path(__file__).resolve().parents[1]
_RELEASE = _ROOT / "docs/release"
_KEY_RESULT_IDS = {
    "m2c-strong-replays",
    "m3a-verified-outcomes",
    "m3b-test-auprc",
    "m3c-temporal-success",
    "m4b-external-visual-success",
    "m4c-distilled-vs-fixed",
    "m4d-fault-gated-success",
    "m4d-override-recall",
}


def _json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _write_registry(path: Path, value: dict[str, object], context: str) -> None:
    value["content_digest"] = registry_digest(value, context=context)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clone_release(tmp_path: Path) -> Path:
    root = tmp_path / "release-copy"
    shutil.copytree(_ROOT / "docs", root / "docs")
    shutil.copy2(_ROOT / "README.md", root / "README.md")
    milestones = _json(_RELEASE / "milestones.json")
    results = _json(_RELEASE / "results.json")
    report_paths: set[str] = set()
    for raw in cast(list[dict[str, object]], milestones["milestones"]):
        for report in cast(list[dict[str, str]], raw["primary_reports"]):
            report_paths.add(report["path"])
    for raw in cast(list[dict[str, object]], results["results"]):
        source = cast(dict[str, str], raw["source"])
        report_paths.add(source["report_path"])
    for relative in report_paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ROOT / relative, destination)
    return root


def _stub_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release_module, "_commit_exists", lambda *_: True)
    monkeypatch.setattr(release_module, "_tracked_files", lambda _: ())


def _audit_issue(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    expected: str,
) -> None:
    _stub_git(monkeypatch)
    report = audit_release(root)
    assert not report["passed"]
    assert expected in "\n".join(cast(list[str], report["issues"]))
    with pytest.raises(ReleaseValidationError, match=expected):
        audit_release(root, strict=True)


def test_current_registries_and_release_manifest_are_strictly_valid() -> None:
    milestones = load_milestone_registry(_RELEASE / "milestones.json")
    claims = load_claims_registry(_RELEASE / "claims.json")
    results = load_results_registry(_RELEASE / "results.json")
    manifest = load_release_manifest(_RELEASE / "release-manifest.json")

    assert len(cast(list[object], milestones["milestones"])) == 12
    assert len(cast(list[object], claims["claims"])) == 22
    assert len(cast(list[object], results["results"])) == 57
    assert manifest["release_source_git_sha"] == (
        "942c6572914ef9c7fa6e5f8affdd8c9004ed4916"
    )


@pytest.mark.parametrize(
    ("filename", "collection", "context", "loader"),
    [
        (
            "milestones.json",
            "milestones",
            "MilestoneRegistryV1",
            load_milestone_registry,
        ),
        ("claims.json", "claims", "ClaimsRegistryV1", load_claims_registry),
        ("results.json", "results", "ResultsRegistryV1", load_results_registry),
    ],
)
def test_registry_duplicate_ids_are_rejected(
    tmp_path: Path,
    filename: str,
    collection: str,
    context: str,
    loader: Callable[[Path], Mapping[str, object]],
) -> None:
    value = _json(_RELEASE / filename)
    records = cast(list[object], value[collection])
    records.append(records[0])
    path = tmp_path / filename
    _write_registry(path, value, context)

    with pytest.raises(ReleaseValidationError, match="duplicate"):
        loader(path)


def test_registry_digest_tampering_is_rejected(tmp_path: Path) -> None:
    value = _json(_RELEASE / "results.json")
    records = cast(list[dict[str, object]], value["results"])
    records[0]["value"] = -1
    path = tmp_path / "results.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="content digest mismatch"):
        load_results_registry(path)


def test_malformed_report_digest_is_rejected(tmp_path: Path) -> None:
    value = _json(_RELEASE / "results.json")
    records = cast(list[dict[str, object]], value["results"])
    source = cast(dict[str, object], records[0]["source"])
    source["report_digest"] = "sha256:not-a-digest"
    path = tmp_path / "results.json"
    _write_registry(path, value, "ResultsRegistryV1")

    with pytest.raises(ReleaseValidationError, match="lowercase sha256"):
        load_results_registry(path)


def test_unsupported_claim_cannot_use_readme_or_resume(tmp_path: Path) -> None:
    value = _json(_RELEASE / "claims.json")
    records = cast(list[dict[str, object]], value["claims"])
    unsupported = next(
        record for record in records if record["support_status"] == "unsupported"
    )
    unsupported["permitted_usage_surfaces"] = ["README"]
    path = tmp_path / "claims.json"
    _write_registry(path, value, "ClaimsRegistryV1")

    with pytest.raises(ReleaseValidationError, match="escalated"):
        load_claims_registry(path)


def test_release_audit_accepts_the_current_release() -> None:
    report = audit_release(_ROOT, strict=True)
    assert report["passed"] is True
    assert report["issue_count"] == 0


def test_release_audit_rejects_missing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _clone_release(tmp_path)
    milestones = _json(root / "docs/release/milestones.json")
    first = cast(list[dict[str, object]], milestones["milestones"])[0]
    report = cast(list[dict[str, str]], first["primary_reports"])[0]
    (root / report["path"]).unlink()
    _audit_issue(monkeypatch, root, "missing milestone report")


def test_release_audit_rejects_changed_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _clone_release(tmp_path)
    path = root / "docs/release/results.json"
    value = _json(path)
    records = cast(list[dict[str, object]], value["results"])
    changed_id = cast(str, records[0]["result_id"])
    records[0]["value"] = -123
    _write_registry(path, value, "ResultsRegistryV1")
    _audit_issue(monkeypatch, root, f"result mismatch {changed_id}")


def test_release_audit_rejects_missing_evidence_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _clone_release(tmp_path)
    path = root / "docs/release/claims.json"
    value = _json(path)
    first = cast(list[dict[str, object]], value["claims"])[0]
    references = cast(list[str], first["evidence_references"])
    references.append("missing-result-id")
    _write_registry(path, value, "ClaimsRegistryV1")
    _audit_issue(monkeypatch, root, "has unknown evidence missing-result-id")


def test_release_audit_rejects_unsupported_claim_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _clone_release(tmp_path)
    path = root / "docs/release/claims.json"
    value = _json(path)
    records = cast(list[dict[str, object]], value["claims"])
    unsupported = next(
        record for record in records if record["support_status"] == "unsupported"
    )
    unsupported["support_status"] = "supported"
    unsupported["evidence_references"] = unsupported["counterevidence_references"]
    unsupported["counterevidence_references"] = []
    _write_registry(path, value, "ClaimsRegistryV1")
    _audit_issue(monkeypatch, root, "escalates unsupported evidence")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            "Private path: " + "D:" + "\\private\\artifact\n",
            "private absolute path",
        ),
        ("to" + "ken=do-not-publish\n", "secret-like assignment"),
        ("[broken](missing-document.md)\n", "broken link"),
    ],
)
def test_release_audit_rejects_public_surface_hygiene_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    expected: str,
) -> None:
    root = _clone_release(tmp_path)
    path = root / "docs/career/interview-brief.md"
    path.write_text(path.read_text(encoding="utf-8") + payload, encoding="utf-8")
    _audit_issue(monkeypatch, root, expected)


def test_portfolio_smoke_is_deterministic_cpu_offline_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("portfolio smoke attempted network access")

    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.split(".", 1)[0].lower() in {"torch", "maniskill", "mani_skill"}:
            raise AssertionError(f"portfolio smoke imported forbidden module {name}")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(socket, "socket", refuse_socket)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    first = run_portfolio_smoke(_ROOT)
    second = run_portfolio_smoke(_ROOT)

    assert first == second
    assert first["passed"] is True
    assert first["resume_zero_work"] is True
    assert first["warning"] == INFRASTRUCTURE_SMOKE_WARNING
    assert first["fixture_trust"] == "non_physical_infrastructure_only"


def test_career_documents_have_required_content_and_hygiene() -> None:
    paths = {
        "resume": _ROOT / "docs/career/resume-bullets.md",
        "interview": _ROOT / "docs/career/interview-brief.md",
        "english": _ROOT / "docs/career/project-summary-en.md",
        "chinese": _ROOT / "docs/career/project-summary-zh.md",
    }
    for path in paths.values():
        text = path.read_text(encoding="utf-8")
        assert not release_module._PRIVATE_PATH.search(text)
        assert not release_module._SECRET_ASSIGNMENT.search(text)
        assert not any(marker in text for marker in ("TODO", "TBD", "PLACEHOLDER"))
    resume = paths["resume"].read_text(encoding="utf-8")
    interview = paths["interview"].read_text(encoding="utf-8")
    assert all(
        heading in resume
        for heading in (
            "## Concise two-bullet version",
            "## Detailed three-bullet version",
            "## Research-oriented version",
        )
    )
    assert all(
        heading in interview
        for heading in (
            "## 30-second explanation",
            "## 2-minute explanation",
            "## 5-minute technical explanation",
            "## Positive-result story",
            "## Negative-result story",
            "## Exact limitations",
        )
    )
    for token in ("2,880", "0.8986", "98.33%", "73.33%", "4.03%"):
        assert token in resume or token in interview
    for text in (resume, interview):
        assert "general robot safety" not in text.lower()
        assert "real-robot validation" not in text.lower()


def test_document_key_results_and_claim_boundaries_are_consistent() -> None:
    results = load_results_registry(_RELEASE / "results.json")
    result_ids = {
        cast(str, record["result_id"])
        for record in cast(list[dict[str, object]], results["results"])
    }
    assert _KEY_RESULT_IDS <= result_ids
    for relative in (
        "README.md",
        "docs/portfolio/technical-report.md",
        "docs/portfolio/key-results.md",
    ):
        text = (_ROOT / relative).read_text(encoding="utf-8")
        assert all(f"LG-RESULT:{identifier}" in text for identifier in _KEY_RESULT_IDS)
    case_study = (_ROOT / "docs/portfolio/case-study.md").read_text(encoding="utf-8")
    assert "73.33%" in case_study and "100%" in case_study
    assert "4.03%" in case_study and "rather than a safety claim" in case_study
    key_results = (_ROOT / "docs/portfolio/key-results.md").read_text(encoding="utf-8")
    assert not release_module._audit_links(
        _ROOT,
        Path("docs/portfolio/key-results.md"),
        key_results,
    )
