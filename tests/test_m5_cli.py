"""M5 command-line integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentguard import cli

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", ["portfolio-smoke", "audit-release"])
def test_m5_help_paths(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        cli.main([command, "--help"])
    assert command in capsys.readouterr().out


def test_audit_release_json_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        cli.main(
            [
                "audit-release",
                "--repo-root",
                str(_ROOT),
                "--strict",
                "--json-output",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["issue_count"] == 0


def test_portfolio_smoke_cli_warns_explicitly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["portfolio-smoke", "--repo-root", str(_ROOT)]) == 0
    output = capsys.readouterr().out.splitlines()
    report = json.loads(output[0])
    assert report["passed"] is True
    assert report["warning"] in output[1]
