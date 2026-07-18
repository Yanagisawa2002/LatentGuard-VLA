from __future__ import annotations

import pytest

from latentguard.cli import _build_parser


@pytest.mark.parametrize(
    "command",
    [
        "prepare-closed-loop-source-plans",
        "run-receding-horizon-selector",
        "benchmark-receding-horizon-selectors",
        "resume-receding-horizon-benchmark",
        "evaluate-receding-horizon-benchmark",
        "inspect-closed-loop-episode",
    ],
)
def test_all_m4c_help_paths(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args([command, "--help"])
    assert caught.value.code == 0
    assert command in capsys.readouterr().out
