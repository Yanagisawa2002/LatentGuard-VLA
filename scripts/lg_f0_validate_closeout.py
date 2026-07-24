"""Validate the compact LG-F0 public closeout surfaces."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lg_f0_build_summary import build_summary  # noqa: E402

SUMMARY_PATH = ROOT / "artifacts" / "final" / "final_summary.json"
EXPECTED_FIGURES = {
    "phase_funnel.png",
    "sarm_generalization_gap.png",
    "reward_model_failure_metrics.png",
    "action_conditioning_delta.png",
    "robolab_replay_reproducibility.png",
}
PUBLIC_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "docs" / "index.md",
    ROOT / "docs" / "latentguard_final_report.md",
    ROOT / "docs" / "latentguard_result_matrix.md",
    ROOT / "docs" / "plans" / "lg-f0-final-closeout.md",
    ROOT / "docs" / "portfolio" / "project_brief.md",
    ROOT / "docs" / "portfolio" / "resume_bullets.md",
    ROOT / "docs" / "portfolio" / "interview_pitch.md",
    ROOT / "docs" / "portfolio" / "demo_storyboard.md",
    ROOT / "artifacts" / "README.md",
    ROOT / "docs" / "lg_rb01_upstream_issue_draft.md",
    ROOT / "docs" / "lg_rb01_upstream_pr_draft.md",
)
PATCH_PATH = (
    ROOT / "patches" / "robolab" / "0001-fix-recorded-config-callable-overlay.patch"
)
STOP_EN = (
    "The exact same-state counterfactual route is closed. The project will not "
    "move to a third simulator, add another reward model, or continue "
    "replay-infrastructure repair."
)
STOP_ZH = (
    "精确同状态反事实路线已经关闭。本项目不会迁移到第三个模拟器，"
    "不会继续增加奖励模型，也不会继续修复重放基础设施。"
)
CLOSEOUT_LABELS = (
    "Research closeout",
    "Counterfactual VLA verifier not validated",
    "No online intervention claim",
)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise AssertionError(f"invalid JSON pointer: {pointer}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        else:
            current = current[token]
    return current


def _png_text_metadata(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"not a PNG: {path}")
    metadata: dict[str, str] = {}
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"tEXt":
            key, value = payload.split(b"\0", 1)
            metadata[key.decode("latin-1")] = value.decode("latin-1")
        elif kind == b"zTXt":
            key, compressed = payload.split(b"\0", 1)
            metadata[key.decode("latin-1")] = zlib.decompress(compressed[1:]).decode(
                "latin-1"
            )
        if kind == b"IEND":
            break
    return metadata


def _validate_summary() -> dict[str, Any]:
    observed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert observed == build_summary(), "final summary is not deterministic"
    assert len(observed["sources"]) == 19

    for source in observed["sources"]:
        relative = Path(source["path"])
        assert not relative.is_absolute()
        path = ROOT / relative
        assert path.is_file(), f"missing source: {relative}"
        assert source["sha256"] == _sha256(path), f"source drift: {relative}"
        document = json.loads(path.read_text(encoding="utf-8"))
        for pointer in source["json_pointers"]:
            _resolve_pointer(document, pointer)
    return observed


def _validate_json() -> None:
    count = 0
    for path in (ROOT / "artifacts").rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8-sig"))
        count += 1
    assert count > 0


def _validate_figures(summary: dict[str, Any]) -> None:
    figure_dir = ROOT / "docs" / "figures"
    observed = {path.name for path in figure_dir.glob("*.png")}
    assert observed == EXPECTED_FIGURES, (
        f"expected exactly five final PNGs, found {sorted(observed)}"
    )
    digest = _sha256(SUMMARY_PATH)
    for filename in EXPECTED_FIGURES:
        path = figure_dir / filename
        assert path.stat().st_size > 10_000, f"figure too small: {filename}"
        metadata = _png_text_metadata(path)
        assert metadata.get("Source") == "artifacts/final/final_summary.json"
        assert metadata.get("SourceSHA256") == digest
        assert metadata.get("Description")

    conclusion = summary["one_sentence_conclusion"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "docs" / "latentguard_final_report.md").read_text(encoding="utf-8")
    brief = (ROOT / "docs" / "portfolio" / "project_brief.md").read_text(
        encoding="utf-8"
    )
    for text in (readme, report, brief):
        normalized = _normalize(text)
        assert _normalize(conclusion["en"]) in normalized
        assert _normalize(conclusion["zh"]) in normalized


def _validate_markdown_links() -> None:
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    missing: list[str] = []
    for path in PUBLIC_MARKDOWN:
        text = path.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")) or not target:
                continue
            target = target.split("#", 1)[0]
            if target and not (path.parent / target).resolve().exists():
                missing.append(f"{path.relative_to(ROOT)} -> {raw_target}")
    assert not missing, "missing relative links:\n" + "\n".join(missing)


def _validate_public_text(summary: dict[str, Any]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "docs" / "latentguard_final_report.md").read_text(encoding="utf-8")
    matrix = (ROOT / "docs" / "latentguard_result_matrix.md").read_text(
        encoding="utf-8"
    )
    for text in (readme, report):
        for label in CLOSEOUT_LABELS:
            assert label in text
    for text in (readme, report, matrix):
        normalized = _normalize(text)
        assert _normalize(STOP_EN) in normalized
        assert _normalize(STOP_ZH) in normalized

    positioning = summary["positioning"]
    assert _normalize(positioning["en"]) in _normalize(readme)
    assert _normalize(positioning["zh"]) in _normalize(readme)

    absolute_path_pattern = re.compile(
        r"(?i)(?:\b[A-Z]:\\|/root/|/home/[^/\s]+/|/autodl-tmp/)"
    )
    secret_pattern = re.compile(
        r"(?i)(?:BEGIN [A-Z ]*PRIVATE KEY|"
        r"(?:password|passwd|api[_-]?key|access[_-]?token)\s*[:=]\s*\S+|"
        r"connect\.bjb\d*\.seetacloud\.com)"
    )
    for path in (*PUBLIC_MARKDOWN, PATCH_PATH):
        text = path.read_text(encoding="utf-8")
        assert not absolute_path_pattern.search(text), (
            f"machine-specific absolute path in {path.relative_to(ROOT)}"
        )
        assert not secret_pattern.search(text), (
            f"possible secret or private host in {path.relative_to(ROOT)}"
        )

    patch_text = PATCH_PATH.read_text(encoding="utf-8")
    assert "callable" in patch_text
    assert not re.search(r"\beval\s*\(|\bexec\s*\(", patch_text)


def main() -> None:
    """Run all bounded closeout validations."""
    summary = _validate_summary()
    _validate_json()
    _validate_figures(summary)
    _validate_markdown_links()
    _validate_public_text(summary)
    print("LG-F0 closeout validation: pass")


if __name__ == "__main__":
    main()
