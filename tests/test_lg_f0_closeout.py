"""Regression tests for the evidence-bound LG-F0 closeout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_final_summary_matches_frozen_sources() -> None:
    """The committed final summary must deterministically rebuild."""
    _run("lg_f0_build_summary.py", "--check")


def test_public_closeout_surfaces_are_consistent() -> None:
    """The compact public surfaces must remain source-bound and hygienic."""
    _run("lg_f0_validate_closeout.py")
