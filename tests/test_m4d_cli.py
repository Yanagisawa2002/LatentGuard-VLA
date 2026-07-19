from __future__ import annotations

import pytest

from latentguard.cli import _build_parser
from latentguard.m4d_cli import _phase_sources


@pytest.mark.parametrize(
    "command",
    [
        "diagnose-m4c-interventions",
        "prepare-m4d-source-plans",
        "run-fallback-shield-development",
        "select-fallback-gates",
        "run-fallback-shield-benchmark",
        "resume-fallback-shield-benchmark",
        "evaluate-fallback-shield",
        "inspect-fallback-shield-episode",
    ],
)
def test_all_m4d_help_paths(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args([command, "--help"])
    assert caught.value.code == 0
    assert command in capsys.readouterr().out


def test_development_source_limit_applies_to_runtime_assets() -> None:
    """The six-source smoke must not execute the complete 12-source archive."""

    sources = tuple(range(12))
    assert _phase_sources(
        sources, phase="development", development_source_limit=6
    ) == tuple(range(6))
    assert (
        _phase_sources(sources, phase="development", development_source_limit=12)
        == sources
    )
